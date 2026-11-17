//! Inkling 2's vision tower (transformers' Inkling2VisionModel): an image's patches (bf16
//! [tokens x 4][20 x 20 x 3], token-major, a token's 2 x 2 patches in (in_row, in_col) order, each
//! (y, x, channel)) in, one 4608-wide row an image token out, the rows the image placeholders take
//! in place of their embeddings. One image a call: images are independent sequences.
//!
//! The tower's weights (653 MB, BF16 as the checkpoint has them) and its scratch are carved from
//! whatever device memory it is given (flash_vision's Mem), as GLM-5.3-Flash's tower is; the learned
//! 32 x 32 position table stays on the host (4 MB fp32) and is interpolated there per image.
//!
//! The sequence is [class token; patches]. Embeddings: patch_embedding (a GEMM, K padded 1200 ->
//! 1216) plus the position table resampled bilinearly (half-pixel centres, taps clamped to the
//! border) to the image's patch grid, fp32 then bf16; the class token is class_embedding +
//! cls_position_embedding; ln_pre. Per layer (Inkling2VisionEncoderLayer): LayerNorm, q | k | v with
//! biases (one GEMM, the three weights stacked at load), the axial 2D RoPE over the whole head
//! (theta 1e4; patches at (row + 1, col + 1), the class token at (0, 0)), full attention over the
//! sequence, out_proj with bias, LayerScale and the residual; LayerNorm, fc1 + GELU (erf), fc2,
//! LayerScale and the residual. No final norm. The merger: a token's 4 patch rows are one row of
//! 4096 (the patches are token-major), proj_in, GELU, proj_out to 4608. The GEMMs are ds41.cu's BF16
//! kernels with an fp32 epilogue (vision.cu), so the rows agree with PyTorch's to rounding, not bits.
//!
//! RoPE's pairs. The reference turns interleaved channels (2j, 2j + 1) of a head by angle j (j < 16:
//! the column's, j >= 16: the row's); ds_vit_rope turns (d, d + 32) by angle d. q_proj's and
//! k_proj's output channels are permuted at load, the same way for both (2j -> j, 2j + 1 -> j + 32
//! within a head), which moves each pair to where the kernel turns it and leaves every q . k
//! product's terms the same. q and k are never read in the reference's channel order after that.

use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_linear_bf16, ds_mma_bf16};
use tidelm_kernels::vision::{ds_vit_attn, ds_vit_epilogue, ds_vit_rope, ink_vit_ln, ink_vit_ls_epilogue};
use tidelm_model::Checkpoint;

pub use crate::flash_vision::Mem;
use crate::{ck, Result};

const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const INTER: usize = 4096;
const DEPTH: usize = 23;
/// a patch's values: 20 x 20 x 3
pub const PATCH_K: usize = 1200;
/// padded to the tensor-core GEMM's multiple of 64
const PATCH_KP: usize = 1216;
/// patches a token (2 x 2)
pub const PER_TOKEN: usize = 4;
/// side of the learned position table
const SIDE: usize = 32;
const PROJ: usize = 4096;
/// width of a feature row (the language model's hidden size)
pub const OUT: usize = 4608;
const EPS: f32 = 1e-5;
/// rope_parameters is absent from the checkpoint's vision_config: PreTrainedConfig fills
/// rope_theta with RotaryEmbeddingConfigMixin.default_theta (transformers' modeling_rope_utils.py)
const THETA: f32 = 10000.0;
/// most fp32 a chunked GEMM writes: 16M floats
const F32_SCRATCH: usize = 16 << 20;
/// rows of the MLP's 4096-wide activation held at once
const MLP_ROWS: usize = 2048;
const P: &str = "model.visual.";

struct Layer {
    ln1_w: *mut c_void,
    ln1_b: *mut c_void,
    qkv_w: *mut c_void,
    qkv_b: *mut c_void,
    out_w: *mut c_void,
    out_b: *mut c_void,
    ls1: *mut c_void,
    ln2_w: *mut c_void,
    ln2_b: *mut c_void,
    fc1_w: *mut c_void,
    fc1_b: *mut c_void,
    fc2_w: *mut c_void,
    fc2_b: *mut c_void,
    ls2: *mut c_void,
}

pub struct InkVision {
    patch_w: *mut c_void,
    pre_w: *mut c_void,
    pre_b: *mut c_void,
    layers: Vec<Layer>,
    proj_in: *mut c_void,
    proj_out: *mut c_void,
    /// host: the position table as fp32 [32 x 32][1024], and the class token's row (bf16)
    pos: Vec<f32>,
    cls: Vec<u16>,
    /// scratch for images of up to `max` tokens
    max: usize,
    f32n: usize,
    bytes: usize,
    patches: *mut c_void,
    x: *mut c_void,
    h: *mut c_void,
    qkv: *mut c_void,
    att: *mut c_void,
    cs: *mut c_void,
    sn: *mut c_void,
    f32b: *mut c_void,
    mo: *mut c_void,
    y: *mut c_void,
}

/// What a linear's fp32 output goes through on its way to bf16.
#[derive(Clone, Copy)]
enum Epi {
    /// bf16(y + bias), then the residual when there is one
    Plain,
    /// gelu(bf16(y + bias))
    Gelu,
    /// bf16(residual + bf16(gamma * bf16(y + bias)))
    Scale(*const c_void),
}

const fn r256(b: usize) -> usize {
    b.div_ceil(256) * 256
}

/// fp32 values of the GEMM scratch for images of up to `t` tokens.
fn f32_len(t: usize) -> usize {
    ((PER_TOKEN * t + 1) * INTER).min(F32_SCRATCH)
}

/// The scratch pieces' bytes for images of up to `t` tokens, in the order load takes them.
fn scratch(t: usize) -> [usize; 10] {
    let s = PER_TOKEN * t + 1;
    [
        2 * PER_TOKEN * t * PATCH_KP, // patches
        2 * s * D,                    // x
        2 * s * D,                    // h
        2 * s * 3 * D,                // qkv
        2 * s * D,                    // att
        4 * s * 32,                   // cos
        4 * s * 32,                   // sin
        4 * f32_len(t),               // f32b
        2 * s.min(MLP_ROWS) * INTER,  // mo
        2 * t * PROJ,                 // y
    ]
}

/// The tower's weights on the device, bytes (every piece a multiple of 256).
pub const fn weight_bytes() -> usize {
    let layer = 6 * D + 3 * D * D + 3 * D + D * D + D + INTER * D + INTER + D * INTER + D;
    2 * (D * PATCH_KP + 2 * D + DEPTH * layer + PROJ * PER_TOKEN * D + OUT * PROJ)
}

/// The tower's weights and its scratch for images of up to `max_tokens` tokens: what load takes from
/// its Mem, each piece 256-aligned.
pub fn bytes_for(max_tokens: usize) -> usize {
    weight_bytes() + scratch(max_tokens).iter().map(|&b| r256(b)).sum::<usize>()
}

/// Bytes to give InkVision::load when its Mem is several regions: bytes_for plus, for the room a
/// region's end may leave unused, twice the largest piece (proj_out, or the scratch's largest).
pub fn bytes_for_regions(max_tokens: usize) -> usize {
    bytes_for(max_tokens) + 2 * scratch(max_tokens).into_iter().fold(2 * OUT * PROJ, usize::max)
}

fn bf(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// fp32 -> bf16, round to nearest even (torch's .to(bfloat16))
fn to_bf(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn at(p: *mut c_void, elems: usize) -> *mut c_void {
    unsafe { (p as *mut u16).add(elems) as *mut c_void }
}

fn put<T: Copy>(st: &Stream, dst: *mut c_void, v: &[T]) -> Result<()> {
    unsafe {
        tidelm_cuda::memcpy_async(dst, v.as_ptr() as *const c_void, std::mem::size_of_val(v), tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
    }
    st.synchronize()?;
    Ok(())
}

/// TIDELM_VISION_DUMP=PREFIX (checks only): an intermediate [rows][cols] bf16 to PREFIX.NAME.bin.
fn dump(st: &Stream, name: &str, p: *const c_void, rows: usize, cols: usize) -> Result<()> {
    let Ok(prefix) = std::env::var("TIDELM_VISION_DUMP") else { return Ok(()) };
    st.synchronize()?;
    let mut v = vec![0u8; rows * cols * 2];
    ck(unsafe { tidelm_cuda::ffi::cudaMemcpy(v.as_mut_ptr() as *mut c_void, p, v.len(), tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST) },
       "vision dump")?;
    std::fs::write(format!("{prefix}.{name}.bin"), v)?;
    Ok(())
}

/// A q / k projection's weight rows (`cols` 1024) or bias (`cols` 1) with each head's interleaved
/// pairs (2j, 2j + 1) moved to (j, j + 32): the layout ds_vit_rope turns.
fn split_pairs(v: &[u16], cols: usize) -> Vec<u16> {
    let mut o = vec![0u16; v.len()];
    for h in 0..HEADS {
        for j in 0..HD / 2 {
            for (dst, src) in [(h * HD + j, h * HD + 2 * j), (h * HD + HD / 2 + j, h * HD + 2 * j + 1)] {
                o[dst * cols..(dst + 1) * cols].copy_from_slice(&v[src * cols..(src + 1) * cols]);
            }
        }
    }
    o
}

/// One axis' two taps into the position table and their weights for index `i` of `size`
/// (vision_utils' _interpolation_axis_taps_weights: bilinear, align_corners False, border padding -
/// a tap past the table is clamped to its edge and keeps its weight), in fp32 as the reference.
fn taps(i: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let src = (i as f32 + 0.5) * SIDE as f32 / size as f32 - 0.5;
    let fl = src.floor();
    let tap = |o: i64| (fl as i64 + o).clamp(0, SIDE as i64 - 1) as usize;
    let w = |o: f32| (1.0 - (src - fl - o).abs()).max(0.0);
    ([tap(0), tap(1)], [w(0.0), w(1.0)])
}

/// config.json's vision_config against what this file hard-codes: a tower built with other values
/// would load (the tensors' shapes do not carry eps, theta, the activation or the head count) and
/// emit rows that are wrong with nothing in the log.
pub fn check_config(model_dir: &Path) -> Result<()> {
    let path = model_dir.join("config.json");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)?;
    let vc = &v["vision_config"];
    if !vc.is_object() {
        return Err(format!("{}: no vision_config", path.display()).into());
    }
    let bad = |what: &str, got: &serde_json::Value| -> crate::Error {
        format!("{}: vision_config.{what} is {got}, which inkling_vision.rs does not implement", path.display()).into()
    };
    // (names the value goes by, the value this file is written for); absent = the config class' default, which is that value
    let ints: [(&[&str], usize); 10] = [
        (&["hidden_size"], D),
        (&["num_hidden_layers"], DEPTH),
        (&["num_attention_heads"], HEADS),
        (&["num_channels"], 3),
        (&["patch_size", "native_patch_size"], 20),
        (&["spatial_merge_size", "adapter_merge_size"], 2),
        (&["posemb_grid_size"], SIDE),
        (&["projection_hidden_dim"], PROJ),
        (&["text_hidden_size"], OUT),
        (&["intermediate_size"], INTER),
    ];
    for (names, want) in ints {
        for name in names {
            let got = &vc[*name];
            if !got.is_null() && got.as_u64() != Some(want as u64) {
                return Err(bad(name, got));
            }
        }
    }
    let floats: [(&str, f64); 2] = [("mlp_ratio", 4.0), ("layer_norm_eps", EPS as f64)];
    for (name, want) in floats {
        let got = &vc[name];
        if !got.is_null() && !got.as_f64().is_some_and(|g| (g - want).abs() <= 1e-6 * want) {
            return Err(bad(name, got));
        }
    }
    if !vc["hidden_act"].is_null() && vc["hidden_act"].as_str() != Some("gelu") {
        return Err(bad("hidden_act", &vc["hidden_act"]));
    }
    if vc["projector_bias"].as_bool() == Some(true) {
        return Err(bad("projector_bias", &vc["projector_bias"]));
    }
    let rp = &vc["rope_parameters"];
    if !rp.is_null() {
        let theta = rp["rope_theta"].as_f64().unwrap_or(THETA as f64);
        let kind = rp["rope_type"].as_str().unwrap_or("axial");
        if theta != THETA as f64 || !(kind == "axial" || kind == "default") {
            return Err(bad("rope_parameters", rp));
        }
    }
    if !vc["rope_theta"].is_null() && vc["rope_theta"].as_f64() != Some(THETA as f64) {
        return Err(bad("rope_theta", &vc["rope_theta"]));
    }
    Ok(())
}

impl InkVision {
    /// The tower's tensors from the checkpoint into `mem`, its scratch for images of up to
    /// `max_tokens` tokens after them: bytes_for(max_tokens) in all.
    pub fn load(ck: &Checkpoint, mem: &mut Mem, st: &Stream, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 || max_tokens > (i32::MAX as usize) / (PER_TOKEN * 3 * D) {
            return Err(format!("vision tower: scratch for {max_tokens} tokens").into());
        }
        check_config(ck.model_dir())?;
        let rd = |n: &str, shape: &[usize]| -> Result<Vec<u16>> {
            let (b, m) = ck.read_tensor_vec(&format!("{P}{n}")).map_err(|e| format!("{P}{n}: {e}"))?;
            if m.dtype != "BF16" || m.shape != shape {
                return Err(format!("{P}{n}: {} {:?} (BF16 {shape:?} expected)", m.dtype, m.shape).into());
            }
            Ok(u16s(&b))
        };
        let mut used = 0usize;
        let mut ld = |v: Vec<u16>| -> Result<*mut c_void> {
            let p = mem.take(v.len() * 2)?;
            used += r256(v.len() * 2);
            put(st, p, &v)?;
            Ok(p)
        };
        // patch_embedding [1024][1200], K padded with zeros
        let pw = rd("backbone.embeddings.patch_embedding.weight", &[D, PATCH_K])?;
        let mut pwp = vec![0u16; D * PATCH_KP];
        for o in 0..D {
            pwp[o * PATCH_KP..o * PATCH_KP + PATCH_K].copy_from_slice(&pw[o * PATCH_K..(o + 1) * PATCH_K]);
        }
        let patch_w = ld(pwp)?;
        let pre_w = ld(rd("backbone.ln_pre.weight", &[D])?)?;
        let pre_b = ld(rd("backbone.ln_pre.bias", &[D])?)?;
        let mut layers = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let l = format!("backbone.layers.{i}.");
            let a = format!("{l}self_attn.");
            // q | k | v as one [3072][1024] GEMM; q's and k's channels in ds_vit_rope's pair layout
            let mut w = split_pairs(&rd(&format!("{a}q_proj.weight"), &[D, D])?, D);
            w.extend(split_pairs(&rd(&format!("{a}k_proj.weight"), &[D, D])?, D));
            w.extend(rd(&format!("{a}v_proj.weight"), &[D, D])?);
            let mut b = split_pairs(&rd(&format!("{a}q_proj.bias"), &[D])?, 1);
            b.extend(split_pairs(&rd(&format!("{a}k_proj.bias"), &[D])?, 1));
            b.extend(rd(&format!("{a}v_proj.bias"), &[D])?);
            layers.push(Layer {
                ln1_w: ld(rd(&format!("{l}ln_1.weight"), &[D])?)?,
                ln1_b: ld(rd(&format!("{l}ln_1.bias"), &[D])?)?,
                qkv_w: ld(w)?,
                qkv_b: ld(b)?,
                out_w: ld(rd(&format!("{a}out_proj.weight"), &[D, D])?)?,
                out_b: ld(rd(&format!("{a}out_proj.bias"), &[D])?)?,
                ls1: ld(rd(&format!("{l}ls_1.gamma"), &[D])?)?,
                ln2_w: ld(rd(&format!("{l}ln_2.weight"), &[D])?)?,
                ln2_b: ld(rd(&format!("{l}ln_2.bias"), &[D])?)?,
                fc1_w: ld(rd(&format!("{l}mlp.c_fc.weight"), &[INTER, D])?)?,
                fc1_b: ld(rd(&format!("{l}mlp.c_fc.bias"), &[INTER])?)?,
                fc2_w: ld(rd(&format!("{l}mlp.c_proj.weight"), &[D, INTER])?)?,
                fc2_b: ld(rd(&format!("{l}mlp.c_proj.bias"), &[D])?)?,
                ls2: ld(rd(&format!("{l}ls_2.gamma"), &[D])?)?,
            });
        }
        let proj_in = ld(rd("proj_in.weight", &[PROJ, PER_TOKEN * D])?)?;
        let proj_out = ld(rd("proj_out.weight", &[OUT, PROJ])?)?;
        // host: the position table (nn.Embedding(32 x 32, 1024): a row a grid cell) and the class token
        let pos: Vec<f32> = rd("backbone.embeddings.position_embedding", &[SIDE * SIDE, D])?.iter().map(|&h| bf(h)).collect();
        let ce = rd("backbone.embeddings.class_embedding", &[D])?;
        let cp = rd("backbone.embeddings.cls_position_embedding", &[D])?;
        let cls: Vec<u16> = ce.iter().zip(&cp).map(|(&a, &b)| to_bf(bf(a) + bf(b))).collect();
        let sz = scratch(max_tokens);
        let mut i = 0;
        let mut z = || -> Result<*mut c_void> {
            let p = mem.take(sz[i])?;
            used += r256(sz[i]);
            i += 1;
            Ok(p)
        };
        Ok(InkVision {
            patch_w, pre_w, pre_b, layers, proj_in, proj_out, pos, cls,
            max: max_tokens,
            f32n: f32_len(max_tokens),
            patches: z()?,
            x: z()?,
            h: z()?,
            qkv: z()?,
            att: z()?,
            cs: z()?,
            sn: z()?,
            f32b: z()?,
            mo: z()?,
            y: z()?,
            bytes: used,
        })
    }

    /// The most image tokens (rows x cols) an encode takes.
    pub fn max_tokens(&self) -> usize {
        self.max
    }

    /// Device bytes this tower took from its Mem: weight_bytes() and its scratch.
    pub fn device_bytes(&self) -> usize {
        self.bytes
    }

    /// y = x [m][k] . W^T [n][k] bf16 -> the fp32 scratch (for an epilogue) or bf16 at `out`
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, st: &Stream, x: *const c_void, w: *const c_void, m: usize, n: usize, k: usize, out_bf16: Option<*mut c_void>)
            -> Result<()> {
        let null = std::ptr::null_mut();
        let (yb, yf) = match out_bf16 { Some(p) => (p, null), None => (null, self.f32b) };
        let rc = unsafe {
            if m > 16 {
                ds_mma_bf16(st.raw(), x, k as i64, w, m as i32, n as i32, k as i32, 1, yb, yf, n as i64, null)
            } else {
                ds_linear_bf16(st.raw(), x, k as i64, w, m as i32, n as i32, k as i32, 1, yb, yf, n as i64, null)
            }
        };
        ck(rc, "vision gemm")
    }

    /// out [rows][n] = epi(x W^T, bias (or null), residual (or null)) over row chunks that fit the
    /// fp32 scratch; out may be residual.
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, st: &Stream, x: *mut c_void, w: *const c_void, b: *const c_void, rows: usize, n: usize, k: usize, epi: Epi,
              residual: *mut c_void, out: *mut c_void) -> Result<()> {
        let chunk = (self.f32n / n).max(1);
        for r0 in (0..rows).step_by(chunk) {
            let m = chunk.min(rows - r0);
            self.gemm(st, at(x, r0 * k), w, m, n, k, None)?;
            let res: *const c_void = if residual.is_null() { std::ptr::null() } else { at(residual, r0 * n) };
            let o = at(out, r0 * n);
            let rc = unsafe {
                match epi {
                    Epi::Scale(g) => ink_vit_ls_epilogue(st.raw(), self.f32b, b, g, res, o, m as i32, n as i32),
                    Epi::Plain => ds_vit_epilogue(st.raw(), self.f32b, b, res, o, m as i32, n as i32, 0),
                    Epi::Gelu => ds_vit_epilogue(st.raw(), self.f32b, b, res, o, m as i32, n as i32, 1),
                }
            };
            ck(rc, "vision epilogue")?;
        }
        Ok(())
    }

    /// One image of `rows` x `cols` tokens: `patches` bf16 [rows * cols * 4][1200] (host; token-major,
    /// then (in_row, in_col) within the token's 2 x 2 window, each patch 20 x 20 x 3 as (y, x,
    /// channel), normalized) -> `out` device bf16 [rows * cols][4608], 16-byte aligned. An error, and
    /// nothing written, for an image past max_tokens(). The scratch is the tower's own: one encode at
    /// a time.
    pub fn encode(&self, st: &Stream, patches: &[u16], rows: usize, cols: usize, out: *mut c_void) -> Result<()> {
        let t = rows.checked_mul(cols).unwrap_or(usize::MAX);
        if t == 0 || t > self.max || patches.len() != t * PER_TOKEN * PATCH_K || out.is_null() {
            return Err(format!("vision: an image of {rows} x {cols} tokens (max {}), {} patch values ({} expected)", self.max,
                               patches.len(), t.saturating_mul(PER_TOKEN * PATCH_K)).into());
        }
        // n patches on a gh x gw grid, a sequence of s rows: the class token, then the patches
        let (n, gh, gw) = (PER_TOKEN * t, 2 * rows, 2 * cols);
        let s = n + 1;
        let si = s as i32;
        let raw = st.raw();
        let null = std::ptr::null_mut();
        // patch p's place on the grid (the patches are block-major)
        let place = |p: usize| -> (usize, usize) {
            let (tok, inb) = (p / PER_TOKEN, p % PER_TOKEN);
            (2 * (tok / cols) + inb / 2, 2 * (tok % cols) + inb % 2)
        };
        // the patches padded to 1216 columns
        let mut pp = vec![0u16; n * PATCH_KP];
        for r in 0..n {
            pp[r * PATCH_KP..r * PATCH_KP + PATCH_K].copy_from_slice(&patches[r * PATCH_K..(r + 1) * PATCH_K]);
        }
        put(st, self.patches, &pp)?;
        drop(pp);
        // row 0: the class token; row 1 + p: patch p's position embedding, the table's four taps
        // weighted and summed in fp32, then bf16 (Inkling2VisionEmbeddings)
        let (th, tw): (Vec<_>, Vec<_>) = ((0..gh).map(|i| taps(i, gh)).collect(), (0..gw).map(|i| taps(i, gw)).collect());
        let mut e = vec![0u16; s * D];
        e[..D].copy_from_slice(&self.cls);
        for p in 0..n {
            let (r, c) = place(p);
            let ((hi, hw), (wi, ww)) = (th[r], tw[c]);
            let row = |a: usize, b: usize| &self.pos[(hi[a] * SIDE + wi[b]) * D..][..D];
            let (t00, t01, t10, t11) = (row(0, 0), row(0, 1), row(1, 0), row(1, 1));
            let (w00, w01, w10, w11) = (hw[0] * ww[0], hw[0] * ww[1], hw[1] * ww[0], hw[1] * ww[1]);
            for (d, o) in e[(1 + p) * D..(2 + p) * D].iter_mut().enumerate() {
                *o = to_bf(t00[d] * w00 + t01[d] * w01 + t10[d] * w10 + t11[d] * w11);
            }
        }
        put(st, self.h, &e)?;
        drop(e);
        // the RoPE table [s][32]: pair j < 16 turns by the column's angle j, pair 16 + j by the row's
        // (Inkling2VisionRotaryEmbedding: cat([w, h]), each repeated for an interleaved pair)
        let inv: Vec<f32> = (0..HD / 4).map(|j| 1.0f32 / THETA.powf((2 * j) as f32 / (HD / 2) as f32)).collect();
        let (mut cs, mut sn) = (Vec::with_capacity(s * 32), Vec::with_capacity(s * 32));
        for q in 0..s {
            let (ph, pw) = if q == 0 { (0.0f32, 0.0f32) } else { let (r, c) = place(q - 1); ((r + 1) as f32, (c + 1) as f32) };
            for j in 0..HD / 2 {
                let a = if j < HD / 4 { pw * inv[j] } else { ph * inv[j - HD / 4] };
                cs.push(a.cos());
                sn.push(a.sin());
            }
        }
        put(st, self.cs, &cs)?;
        put(st, self.sn, &sn)?;
        // embeddings: h[1..] = bf16(bf16(patch W^T) + position), then ln_pre over the sequence
        let h1 = at(self.h, D);
        self.linear(st, self.patches, self.patch_w, null, n, D, PATCH_KP, Epi::Plain, h1, h1)?;
        dump(st, "embed", self.h, s, D)?;
        ck(unsafe { ink_vit_ln(raw, self.h, self.pre_w, self.pre_b, self.x, si, D as i32, EPS) }, "ln_pre")?;
        dump(st, "pre", self.x, s, D)?;
        for (li, l) in self.layers.iter().enumerate() {
            ck(unsafe { ink_vit_ln(raw, self.x, l.ln1_w, l.ln1_b, self.h, si, D as i32, EPS) }, "ln_1")?;
            self.linear(st, self.h, l.qkv_w, l.qkv_b, s, 3 * D, D, Epi::Plain, null, self.qkv)?;
            ck(unsafe { ds_vit_rope(raw, self.qkv, si, HEADS as i32, self.cs, self.sn) }, "vision rope")?;
            ck(unsafe { ds_vit_attn(raw, self.qkv, si, HEADS as i32, self.att) }, "vision attention")?;
            self.linear(st, self.att, l.out_w, l.out_b, s, D, D, Epi::Scale(l.ls1), self.x, self.x)?;
            ck(unsafe { ink_vit_ln(raw, self.x, l.ln2_w, l.ln2_b, self.h, si, D as i32, EPS) }, "ln_2")?;
            for r0 in (0..s).step_by(MLP_ROWS) {
                let m = MLP_ROWS.min(s - r0);
                self.linear(st, at(self.h, r0 * D), l.fc1_w, l.fc1_b, m, INTER, D, Epi::Gelu, null, self.mo)?;
                let xr = at(self.x, r0 * D);
                self.linear(st, self.mo, l.fc2_w, l.fc2_b, m, D, INTER, Epi::Scale(l.ls2), xr, xr)?;
            }
            dump(st, &format!("b{li}"), self.x, s, D)?;
        }
        // the merger: past the class token, a token's 4 patch rows are one row of 4096
        self.linear(st, at(self.x, D), self.proj_in, std::ptr::null(), t, PROJ, PER_TOKEN * D, Epi::Gelu, null, self.y)?;
        dump(st, "merge", self.y, t, PROJ)?;
        self.gemm(st, self.y, self.proj_out, t, OUT, PROJ, Some(out))?;
        st.synchronize()?;
        Ok(())
    }
}

/// A tower in a buffer of its own (checks; a server can lend it memory through Mem instead).
pub struct OwnedInkVision {
    pub tower: InkVision,
    _mem: DeviceBuffer<u8>,
}

impl OwnedInkVision {
    pub fn load(ck: &Checkpoint, st: &Stream, max_tokens: usize) -> Result<Self> {
        let buf = DeviceBuffer::<u8>::new(bytes_for(max_tokens))?;
        let mut mem = Mem::new(&[(buf.raw(), buf.len())]);
        let tower = InkVision::load(ck, &mut mem, st, max_tokens)?;
        Ok(OwnedInkVision { tower, _mem: buf })
    }

    /// From the checkpoint directory (config.json and the shards with model.visual.*).
    pub fn open(model_dir: &Path, st: &Stream, max_tokens: usize) -> Result<Self> {
        Self::load(&Checkpoint::open(model_dir)?, st, max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(to_bf(1.0), 0x3f80);
        assert_eq!(bf(to_bf(-2.5)), -2.5);
        // halfway between two bf16: to the even one
        assert_eq!(to_bf(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(to_bf(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn taps_match_half_pixel_bilinear() {
        // the table's own size: every position its own cell
        for i in 0..SIDE {
            let (t, w) = taps(i, SIDE);
            assert_eq!((t[0], w[0], w[1]), (i, 1.0, 0.0));
        }
        // twice the table: src = (i + 0.5) / 2 - 0.5; the first is past the edge, both taps cell 0
        let (t, w) = taps(0, 2 * SIDE);
        assert_eq!((t, w), ([0, 0], [0.25, 0.75]));
        let (t, w) = taps(1, 2 * SIDE);
        assert_eq!((t, w), ([0, 1], [0.75, 0.25]));
        let (t, w) = taps(2 * SIDE - 1, 2 * SIDE);
        assert_eq!((t, w), ([SIDE - 1, SIDE - 1], [0.75, 0.25]));
    }

    #[test]
    fn split_pairs_moves_interleaved_pairs_apart() {
        let v: Vec<u16> = (0..D as u16).collect();
        let o = split_pairs(&v, 1);
        // head 1: channels 64.., pair 3 = (70, 71) -> (64 + 3, 64 + 32 + 3)
        assert_eq!((o[64 + 3], o[64 + 32 + 3]), (70, 71));
        let mut sorted = o.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, v);
    }

    #[test]
    fn sizes_add_up() {
        // the checkpoint's model.visual.* is 655,386,624 bytes; on the device the patch weight is
        // padded (+ 16 x 1024 x 2) and the position table and class rows stay on the host
        assert_eq!(weight_bytes(), 655_386_624 + 2 * 16 * D - 2 * (SIDE * SIDE * D + 2 * D));
        assert!(bytes_for(1) < bytes_for(2) && bytes_for_regions(1) > bytes_for(1));
    }
}
