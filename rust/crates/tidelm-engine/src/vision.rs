//! The vision tower: the checkpoint's inference/vision.py ViT and Aligner, loaded as
//! engine/vision.py loads them. An image's patches (bf16 [n][3 x 14 x 14], the server's
//! preprocessing) go in; one `dim`-wide row per 3x3 cell of the patch grid comes out, the rows
//! an image span's IMAGE positions take in place of their embeddings. bf16 throughout, each op
//! rounding where the PyTorch module does (vision.cu); the GEMMs are ds41's tensor-core kernels
//! with an fp32 epilogue for the bias, attention a flash kernel that never holds n x n scores.
//! The Python engine runs PyTorch's cuBLAS / SDPA here, so the rows agree to rounding, not bits.

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::ds41::ds_rmsnorm;
use tidelm_kernels::vision::*;
use tidelm_model::Checkpoint;

use crate::config::Config;
use crate::forward::{ck, gemm, np, nul, Wt};
use crate::weights::Loader;
use crate::Result;

/// patch_embed's K (3 x 14 x 14 = 588) padded to the tensor-core kernel's multiple of 64
const PATCH_K_PAD: usize = 640;
const EPS: f32 = 1e-6;

struct Block {
    norm1: DeviceBuffer<u16>,
    norm2: DeviceBuffer<u16>,
    wqkv: DeviceBuffer<u16>,
    bqkv: DeviceBuffer<u16>,
    wo: DeviceBuffer<u16>,
    bo: DeviceBuffer<u16>,
    w1: DeviceBuffer<u16>,
    w2: DeviceBuffer<u16>,
}

pub struct Tower {
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<u16>,
    blocks: Vec<Block>,
    norm: DeviceBuffer<u16>,
    a_w1: DeviceBuffer<u16>,
    a_b1: DeviceBuffer<u16>,
    a_w2: DeviceBuffer<u16>,
    a_b2: DeviceBuffer<u16>,
    vd: usize,
    heads: usize,
    inter: usize,
    patch_k: usize,
    r: usize,
    out_dim: usize,
    theta: f32,
}

/// The three learned span delimiters (image_start, image_newline, image_end), [3][dim] bf16
/// in that order: every rank splices them itself.
pub fn load_delimiters(ckpt: &Checkpoint, stream: &Stream, cfg: &Config) -> Result<DeviceBuffer<u16>> {
    let d = cfg.dim;
    let out = DeviceBuffer::<u16>::new(3 * d)?;
    let mut ld = Loader::new(ckpt, stream)?;
    for (i, name) in ["image_start", "image_newline", "image_end"].iter().enumerate() {
        let meta = ckpt.meta(name)?;
        if meta.dtype != "BF16" || meta.shape != [d] {
            return Err(format!("{name}: {} {:?}, expected BF16 [{d}]", meta.dtype, meta.shape).into());
        }
        ld.read_into(name, 0, d * 2, out.ptr_at(i * d))?;
    }
    Ok(out)
}

impl Tower {
    pub fn load(ckpt: &Checkpoint, stream: &Stream, cfg: &Config) -> Result<Self> {
        let (vd, heads, inter, p, r) = (cfg.vision_dim, cfg.vision_n_heads, cfg.vision_inter_dim, cfg.vision_patch_size,
                                        cfg.vision_downsample_ratio);
        if cfg.vision_n_layers == 0 {
            return Err("this checkpoint has no vision tower (vision_n_layers == 0)".into());
        }
        if heads == 0 || vd / heads != 64 || vd % heads != 0 {
            return Err(format!("the vision attention kernel takes head_dim 64, not {vd}/{heads}").into());
        }
        if cfg.vision_max_wh_ratio.is_some() {
            return Err("vision_max_wh_ratio is set: its resize-only path is not ported".into());
        }
        let patch_k = 3 * p * p;
        if patch_k > PATCH_K_PAD || vd % 64 != 0 || inter % 64 != 0 {
            return Err("vision shapes the tensor-core GEMM cannot take".into());
        }
        let mut ld = Loader::new(ckpt, stream)?;
        // patch_embed's weight, K padded with zeros (the patches are padded the same way)
        let (raw, meta) = ld.host_bytes("vision.patch_embed.proj.weight")?;
        if meta.dtype != "BF16" || meta.shape != [vd, patch_k] {
            return Err(format!("vision.patch_embed.proj.weight: {} {:?}", meta.dtype, meta.shape).into());
        }
        let mut padded = vec![0u16; vd * PATCH_K_PAD];
        for n in 0..vd {
            for k in 0..patch_k {
                let i = (n * patch_k + k) * 2;
                padded[n * PATCH_K_PAD + k] = u16::from_le_bytes([raw[i], raw[i + 1]]);
            }
        }
        let patch_w = DeviceBuffer::from_slice(&padded)?;
        let patch_b = ld.bf16("vision.patch_embed.proj.bias", &[vd])?;
        let mut blocks = Vec::with_capacity(cfg.vision_n_layers);
        for b in 0..cfg.vision_n_layers {
            let n = |s: &str| format!("vision.blocks.{b}.{s}");
            blocks.push(Block {
                norm1: ld.bf16(&n("norm1.weight"), &[vd])?,
                norm2: ld.bf16(&n("norm2.weight"), &[vd])?,
                wqkv: ld.bf16(&n("attn.wqkv.weight"), &[3 * vd, vd])?,
                bqkv: ld.bf16(&n("attn.wqkv.bias"), &[3 * vd])?,
                wo: ld.bf16(&n("attn.wo.weight"), &[vd, vd])?,
                bo: ld.bf16(&n("attn.wo.bias"), &[vd])?,
                w1: ld.bf16(&n("mlp.w1.weight"), &[2 * inter, vd])?,
                w2: ld.bf16(&n("mlp.w2.weight"), &[vd, inter])?,
            });
        }
        let d = cfg.dim;
        let fk = vd * r * r;
        Ok(Tower {
            patch_w,
            patch_b,
            blocks,
            norm: ld.bf16("vision.norm.weight", &[vd])?,
            a_w1: ld.bf16("aligner.w1.weight", &[d, fk])?,
            a_b1: ld.bf16("aligner.w1.bias", &[d])?,
            a_w2: ld.bf16("aligner.w2.weight", &[d, d])?,
            a_b2: ld.bf16("aligner.w2.bias", &[d])?,
            vd,
            heads,
            inter,
            patch_k,
            r,
            out_dim: d,
            theta: cfg.vision_rope_theta,
        })
    }

    /// Bytes of device memory the weights hold.
    pub fn bytes(&self) -> usize {
        let b: usize = self.blocks.iter().map(|b| {
            [&b.norm1, &b.norm2, &b.wqkv, &b.bqkv, &b.wo, &b.bo, &b.w1, &b.w2].iter().map(|x| x.len() * 2).sum::<usize>()
        }).sum();
        b + [&self.patch_w, &self.patch_b, &self.norm, &self.a_w1, &self.a_b1, &self.a_w2, &self.a_b2].iter().map(|x| x.len() * 2).sum::<usize>()
    }

    /// Rows out of an nh x nw grid of patches: ceil(nh / r) * ceil(nw / r).
    pub fn rows(&self, nh: usize, nw: usize) -> usize {
        nh.div_ceil(self.r) * nw.div_ceil(self.r)
    }

    /// get_vision_cos_sin: per patch (row, column) the 32 angles row x inv_freq[0..16],
    /// column x inv_freq[0..16], fp32.
    fn rope_tables(&self, nh: usize, nw: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.vd / self.heads / 2;
        let inv: Vec<f32> = (0..half / 2).map(|i| 1.0f32 / self.theta.powf((2 * i) as f32 / half as f32)).collect();
        let (mut c, mut s) = (Vec::with_capacity(nh * nw * half), Vec::with_capacity(nh * nw * half));
        for y in 0..nh {
            for x in 0..nw {
                for j in 0..half {
                    let pos = if j < half / 2 { y } else { x } as f32;
                    let a = pos * inv[j % (half / 2)];
                    c.push(a.cos());
                    s.push(a.sin());
                }
            }
        }
        (c, s)
    }

    /// The aligner rows ([rows(nh, nw)][dim] bf16, row-major over the cell grid) of one image's
    /// patches (bf16 [nh x nw][3 x p x p], ViT.forward's input).
    pub fn forward(&self, stream: &Stream, patches: &[u16], nh: usize, nw: usize) -> Result<DeviceBuffer<u16>> {
        let n = nh * nw;
        if n == 0 || patches.len() != n * self.patch_k {
            return Err(format!("{} patch values for a {nh} x {nw} grid", patches.len()).into());
        }
        let (vd, heads, inter, d) = (self.vd, self.heads, self.inter, self.out_dim);
        let l = self.rows(nh, nw);
        let fk = vd * self.r * self.r;
        let mut xp = vec![0u16; n * PATCH_K_PAD];
        for (i, row) in patches.chunks(self.patch_k).enumerate() {
            xp[i * PATCH_K_PAD..i * PATCH_K_PAD + self.patch_k].copy_from_slice(row);
        }
        let xin = DeviceBuffer::from_slice(&xp)?;
        let (cs, sn) = self.rope_tables(nh, nw);
        let (dcos, dsin) = (DeviceBuffer::from_slice(&cs)?, DeviceBuffer::from_slice(&sn)?);
        let x = DeviceBuffer::<u16>::new(n * vd)?;
        let a = DeviceBuffer::<u16>::new(n * vd)?;
        let qkv = DeviceBuffer::<u16>::new(n * 3 * vd)?;
        let att = DeviceBuffer::<u16>::new(n * vd)?;
        let gu = DeviceBuffer::<u16>::new(n * 2 * inter)?;
        let hmid = DeviceBuffer::<u16>::new(n * inter)?;
        let yf = DeviceBuffer::<f32>::new((n * 3 * vd).max(l * d))?;
        // split-K never pays at these row counts: a one-element partials buffer keeps it off
        let part = DeviceBuffer::<f32>::new(1)?;
        let sr = stream.raw();
        let (ni, vdi) = (n as i32, vd as i32);
        let bf = |w: &DeviceBuffer<u16>, n: usize, k: usize| Wt::Bf16 { w: np(w), n, k };

        // TIDELM_DUMP_DIR: the residual stream after the patch embedding and each block (bf16)
        let dump = std::env::var_os("TIDELM_DUMP_DIR").map(std::path::PathBuf::from);
        let save = |name: &str, buf: &DeviceBuffer<u16>| -> Result<()> {
            if let Some(dir) = &dump {
                stream.synchronize()?;
                let v = buf.download()?;
                std::fs::write(dir.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            Ok(())
        };
        gemm(sr, &part, np(&xin), PATCH_K_PAD, bf(&self.patch_w, vd, PATCH_K_PAD), n, np(&yf), true, vd)?;
        ck(unsafe { ds_vit_epilogue(sr, np(&yf), np(&self.patch_b), nul(), np(&x), ni, vdi, 0) }, "patch embed")?;
        save("vit_embed.bf16", &x)?;
        for (bi, b) in self.blocks.iter().enumerate() {
            ck(unsafe { ds_rmsnorm(sr, np(&x), vd as i64, np(&b.norm1), np(&a), vd as i64, ni, vdi, EPS) }, "vit norm1")?;
            gemm(sr, &part, np(&a), vd, bf(&b.wqkv, 3 * vd, vd), n, np(&yf), true, 3 * vd)?;
            ck(unsafe { ds_vit_epilogue(sr, np(&yf), np(&b.bqkv), nul(), np(&qkv), ni, 3 * vdi, 0) }, "vit qkv")?;
            ck(unsafe { ds_vit_rope(sr, np(&qkv), ni, heads as i32, np(&dcos), np(&dsin)) }, "vit rope")?;
            ck(unsafe { ds_vit_attn(sr, np(&qkv), ni, heads as i32, np(&att)) }, "vit attention")?;
            gemm(sr, &part, np(&att), vd, bf(&b.wo, vd, vd), n, np(&yf), true, vd)?;
            ck(unsafe { ds_vit_epilogue(sr, np(&yf), np(&b.bo), np(&x), np(&x), ni, vdi, 0) }, "vit wo")?;
            ck(unsafe { ds_rmsnorm(sr, np(&x), vd as i64, np(&b.norm2), np(&a), vd as i64, ni, vdi, EPS) }, "vit norm2")?;
            gemm(sr, &part, np(&a), vd, bf(&b.w1, 2 * inter, vd), n, np(&gu), false, 2 * inter)?;
            ck(unsafe { ds_vit_silu_mul(sr, np(&gu), ni, inter as i32, np(&hmid)) }, "vit silu")?;
            gemm(sr, &part, np(&hmid), inter, bf(&b.w2, vd, inter), n, np(&yf), true, vd)?;
            ck(unsafe { ds_vit_epilogue(sr, np(&yf), nul(), np(&x), np(&x), ni, vdi, 0) }, "vit w2")?;
            save(&format!("vit_b{bi:02}.bf16"), &x)?;
            if bi == 0 {
                save("vit_b00_attn.bf16", &att)?;
                save("vit_b00_qkv.bf16", &qkv)?;
            }
        }
        ck(unsafe { ds_rmsnorm(sr, np(&x), vd as i64, np(&self.norm), np(&a), vd as i64, ni, vdi, EPS) }, "vit norm")?;
        // the aligner: 3x3 cells (channel-major), w1, gelu, w2
        let f = DeviceBuffer::<u16>::new(l * fk)?;
        ck(unsafe { ds_vit_unfold(sr, np(&a), nh as i32, nw as i32, vdi, np(&f)) }, "aligner unfold")?;
        let g = DeviceBuffer::<u16>::new(l * d)?;
        let out = DeviceBuffer::<u16>::new(l * d)?;
        gemm(sr, &part, np(&f), fk, bf(&self.a_w1, d, fk), l, np(&yf), true, d)?;
        ck(unsafe { ds_vit_epilogue(sr, np(&yf), np(&self.a_b1), nul(), np(&g), l as i32, d as i32, 1) }, "aligner w1")?;
        gemm(sr, &part, np(&g), d, bf(&self.a_w2, d, d), l, np(&yf), true, d)?;
        ck(unsafe { ds_vit_epilogue(sr, np(&yf), np(&self.a_b2), nul(), np(&out), l as i32, d as i32, 0) }, "aligner w2")?;
        save("vit_rows.bf16", &out)?;
        // the scratch buffers are freed on return: their kernels must be done
        stream.synchronize()?;
        Ok(out)
    }
}

/// One image of a request: its span's first prompt position, its patch grid and patches
/// (bf16 [vit_h x vit_w][3 x p x p], or empty when the prefill never reaches the span - a
/// restored prefix holds it already).
#[derive(Clone, Debug)]
pub struct ImageIn {
    pub start: usize,
    pub vit_h: usize,
    pub vit_w: usize,
    pub patches: Vec<u16>,
}

/// image_processor.num_image_tokens over the aligner grid of a vit_h x vit_w patch grid:
/// IMAGE_START, per grid row its cells and IMAGE_NEW_LINE, IMAGE_END.
pub fn span_len(vit_h: usize, vit_w: usize, r: usize) -> usize {
    vit_h.div_ceil(r) * (vit_w.div_ceil(r) + 1) + 2
}

/// What position `offset` of a span reads: Ok(cell index) for an IMAGE position, Err(0, 1, 2)
/// for IMAGE_START / IMAGE_NEW_LINE / IMAGE_END.
pub fn span_slot(offset: usize, vit_h: usize, vit_w: usize, r: usize) -> std::result::Result<usize, usize> {
    let w = vit_w.div_ceil(r);
    if offset == 0 {
        return Err(0);
    }
    if offset + 1 == span_len(vit_h, vit_w, r) {
        return Err(2);
    }
    let (row, col) = ((offset - 1) / (w + 1), (offset - 1) % (w + 1));
    if col == w { Err(1) } else { Ok(row * w + col) }
}

/// Prefix-cache keys carry this bit at an image's positions: no token id has it.
pub const KEY_BIT: u32 = 0x8000_0000;

/// The prefix cache's key for each position of an image's span: its content digest (of the
/// preprocessed patches and grid, as prefix_media.image_fingerprints hashes them) and the
/// offset, folded into an id with KEY_BIT - two prompts share a span's positions only when
/// they show the same image there.
pub fn span_keys(digest: u64, len: usize) -> Vec<u32> {
    (0..len as u64)
        .map(|o| {
            let mut h = digest ^ o.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            h ^= h >> 31;
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 29;
            KEY_BIT | (h as u32 & !KEY_BIT)
        })
        .collect()
}

/// A prefix key back to model ids and the n-gram hasher's mask (false at image positions,
/// which carry `image_token_id`); None when the key has no image.
pub fn split_key(key: &[u32], image_token_id: u32) -> (Vec<u32>, Option<Vec<bool>>) {
    if !key.iter().any(|&t| t & KEY_BIT != 0) {
        return (key.to_vec(), None);
    }
    let ids = key.iter().map(|&t| if t & KEY_BIT != 0 { image_token_id } else { t }).collect();
    let mask = key.iter().map(|&t| t & KEY_BIT == 0).collect();
    (ids, Some(mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_is_start_rows_with_newlines_and_end() {
        // 77 x 111 patches: a 26 x 37 cell grid, 26 * 38 + 2 = 990 positions
        assert_eq!(span_len(77, 111, 3), 990);
        assert_eq!(span_slot(0, 77, 111, 3), Err(0));
        assert_eq!(span_slot(1, 77, 111, 3), Ok(0));
        assert_eq!(span_slot(37, 77, 111, 3), Ok(36));
        assert_eq!(span_slot(38, 77, 111, 3), Err(1));
        assert_eq!(span_slot(39, 77, 111, 3), Ok(37));
        assert_eq!(span_slot(988, 77, 111, 3), Err(1));
        assert_eq!(span_slot(989, 77, 111, 3), Err(2));
        let cells = (0..990).filter(|&o| span_slot(o, 77, 111, 3).is_ok()).count();
        assert_eq!(cells, 26 * 37);
    }

    #[test]
    fn keys_split_back_to_ids_and_mask() {
        let k = span_keys(42, 3);
        assert!(k.iter().all(|&t| t & KEY_BIT != 0) && k[0] != k[1]);
        assert_ne!(span_keys(43, 3), k);
        let key = [5, k[0], k[1], 7];
        assert_eq!(split_key(&key, 129264), (vec![5, 129264, 129264, 7], Some(vec![true, false, false, true])));
        assert_eq!(split_key(&[1, 2], 129264), (vec![1, 2], None));
    }
}
