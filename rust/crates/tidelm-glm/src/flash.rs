//! GLM-5.3-Flash (glm5_next) weights for one rank: the checkpoint's BF16 non-expert tensors served
//! as FP8 with 32 x 32 block scales (GLM_FLASH_DENSE=fp8, the default) or as BF16 (=bf16, the
//! reference), fused where one input feeds several projections:
//!   KDA:  q | k | v | f_a | g_a | b  from the normed input (KdaW::offsets)
//!   DSA:  q_a | kv_a | wk | weights_proj | compress_gate  from the normed input (DsaW::OFFSETS_A),
//!         q_b | wq_b  from q_a's norm
//!   MLPs: gate | up
//! Every fused piece is a multiple of 32 rows, so a block scale never straddles two tensors: the
//! fused FP8 matrix is the pieces' own FP8 matrices stacked. The routed experts come from the pack.
//!
//! TP cuts (world > 1, not served yet): heads by rank for q / k / v / b / conv / A_log / dt_bias /
//! f_b / g_b and q_b / kv_b, o_proj by its input columns (summed), f_a / g_a / proj_a / wq_b whole,
//! the MLPs' gate | up by intermediate and down by hidden (gathered), the head by vocabulary.

use std::ops::Range;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::glm::glm_quant_kvb_bf16;
use tidelm_model::Checkpoint;

use crate::config::GlmConfig;
use crate::lin::{self, Bf16Mat, Lin};
use crate::weights::{Attn, DsaW, Embed, HcW, KdaW, Layer, Loader, Mlp, MtpExtra, Weights};
use crate::{ck, Result};

const P: &str = "model.language_model.";

/// GLM_FLASH_DENSE: the non-expert weights' format (fp8, the default, or bf16). It moves values: in
/// the boot guard.
pub fn dense_fp8() -> Result<bool> {
    match std::env::var("GLM_FLASH_DENSE").as_deref() {
        Err(_) | Ok("") | Ok("fp8") => Ok(true),
        Ok("bf16") => Ok(false),
        Ok(v) => Err(format!("GLM_FLASH_DENSE={v}: fp8 or bf16").into()),
    }
}

/// Listed ids of the built-in ranking (GLM_DRAFT_VOCAB_ROWS' default): docs/glm-flash.md has the
/// miss rates and pass times that chose it.
pub const DRAFT_ROWS: usize = 57344;
/// The tokenizer's ranking, most frequent first (tools/build_draft_vocab.py; GLM-5.3's tokenizer is
/// the same one).
const DRAFT_RANKING_V1: &str = include_str!("../../../../engine/glm_draft_vocab_ranking_v1.txt");

/// The MTP draft passes' vocabulary: GLM_DRAFT_VOCAB (unset: the built-in ranking; `all`: every id,
/// no draft head; otherwise a ranking file) and GLM_DRAFT_VOCAB_ROWS (how many of its ids, a
/// multiple of 32: the FP8 head's scale blocks). None: the whole vocabulary.
///
/// A draft pass is its head (2.71 of 3.66 ms: 634 MB over 154,880 ids), and the draft only proposes:
/// the verify pass scores every id, so a token outside the list is a rejected draft, never a wrong
/// token (sampled: the draft's q is 0 there and the rejection sampling stays exact). The DS engine's
/// tidelm_engine::draft_vocab, one box only here.
pub fn draft_ids(vocab: usize) -> Result<Option<Vec<u32>>> {
    let path = std::env::var("GLM_DRAFT_VOCAB").unwrap_or_default();
    if path.trim() == "all" {
        return Ok(None);
    }
    let rows = match std::env::var("GLM_DRAFT_VOCAB_ROWS") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse::<usize>().map_err(|_| format!("GLM_DRAFT_VOCAB_ROWS={v}: not a number"))?,
        _ => DRAFT_ROWS,
    };
    if rows == 0 || rows % 32 != 0 || rows > vocab {
        return Err(format!("GLM_DRAFT_VOCAB_ROWS must be a multiple of 32 in 32..={vocab}, got {rows}").into());
    }
    let text = if path.trim().is_empty() {
        DRAFT_RANKING_V1.to_string()
    } else {
        std::fs::read_to_string(path.trim()).map_err(|e| format!("GLM_DRAFT_VOCAB {path}: {e}"))?
    };
    let order = tidelm_engine::draft_vocab::ranking(&text, vocab)?;
    Ok(Some(tidelm_engine::draft_vocab::select(&order, rows, vocab)?))
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

struct Ld<'a> {
    ld: Loader<'a>,
    fp8: bool,
}

impl Ld<'_> {
    /// A BF16 tensor's shape as [rows, the rest].
    fn dims(&self, name: &str) -> Result<(usize, usize)> {
        let m = self.ld.ck.meta(name).map_err(|e| format!("{name}: {e}"))?;
        if m.dtype != "BF16" {
            return Err(format!("{name}: {} (BF16 expected)", m.dtype).into());
        }
        Ok((m.shape[0], m.shape[1..].iter().product()))
    }

    /// Rows `r` of a BF16 tensor [n][k] (row-major u16).
    fn rows(&self, name: &str, r: Range<usize>) -> Result<(Vec<u16>, usize)> {
        let (n, k) = self.dims(name)?;
        if r.end > n {
            return Err(format!("{name}: rows {r:?} of {n}").into());
        }
        let b = self.ld.bytes(name)?;
        Ok((u16s(&b[r.start * k * 2..r.end * k * 2]), k))
    }

    /// The whole tensor.
    fn all(&self, name: &str) -> Result<(Vec<u16>, usize, usize)> {
        let (n, k) = self.dims(name)?;
        Ok((self.rows(name, 0..n)?.0, n, k))
    }

    /// Rows `ids` of a BF16 tensor [n][k], in that order.
    fn gather(&self, name: &str, ids: &[u32]) -> Result<(Vec<u16>, usize)> {
        let (n, k) = self.dims(name)?;
        let b = self.ld.bytes(name)?;
        let mut o = Vec::with_capacity(ids.len() * k);
        for &i in ids {
            let i = i as usize;
            if i >= n {
                return Err(format!("{name}: row {i} of {n}").into());
            }
            o.extend(b[i * k * 2..(i + 1) * k * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        Ok((o, k))
    }

    /// Columns `c` of every row of a BF16 [n][k].
    fn cols(&self, name: &str, c: Range<usize>) -> Result<(Vec<u16>, usize)> {
        let (v, n, k) = self.all(name)?;
        if c.end > k {
            return Err(format!("{name}: columns {c:?} of {k}").into());
        }
        let mut o = Vec::with_capacity(n * c.len());
        for i in 0..n {
            o.extend_from_slice(&v[i * k + c.start..i * k + c.end]);
        }
        Ok((o, n))
    }

    fn lin(&self, w: Vec<u16>, n: usize, k: usize, split: usize) -> Result<Lin> {
        if self.fp8 {
            Ok(Lin::Fp8(lin::fp8_from_bf16(self.ld.st, &w, n, k, split)?))
        } else {
            Ok(Lin::Bf16(Bf16Mat { w: DeviceBuffer::from_slice(&w)?, n, k, split }))
        }
    }

    /// Row ranges of several [*][k] tensors stacked into one matrix (None: all its rows).
    fn fused(&self, parts: &[(String, Option<Range<usize>>)], split: usize) -> Result<Lin> {
        let (mut all, mut n, mut k0) = (vec![], 0, 0);
        for (name, r) in parts {
            let r = match r {
                Some(r) => r.clone(),
                None => 0..self.dims(name)?.0,
            };
            if r.len() % 32 != 0 {
                return Err(format!("{name}: {} rows, not whole 32-row blocks", r.len()).into());
            }
            let (v, k) = self.rows(name, r.clone())?;
            if k0 != 0 && k != k0 {
                return Err(format!("{name}: K {k}, the fused matrix's {k0}").into());
            }
            k0 = k;
            n += r.len();
            all.extend_from_slice(&v);
        }
        self.lin(all, n, k0, split)
    }

    fn bf16_dev(&self, name: &str) -> Result<DeviceBuffer<u16>> {
        Ok(DeviceBuffer::from_slice(&self.all(name)?.0)?)
    }

    fn f32_dev(&self, name: &str, r: Option<Range<usize>>) -> Result<DeviceBuffer<f32>> {
        let v = self.ld.f32s(name)?;
        Ok(DeviceBuffer::from_slice(&match r { Some(r) => v[r].to_vec(), None => v })?)
    }

    /// Exact BF16 (the mHC mixes and the small things a layer reads whole).
    fn bf16_lin(&self, name: &str, split: usize) -> Result<Lin> {
        let (w, n, k) = self.all(name)?;
        Ok(Lin::Bf16(Bf16Mat { w: DeviceBuffer::from_slice(&w)?, n, k, split }))
    }
}

pub fn load_weights(model_dir: &Path, cfg: &GlmConfig, rank: usize, world: usize, st: &Stream, log: &dyn Fn(&str)) -> Result<Weights> {
    let ckpt = Checkpoint::open(model_dir)?;
    let fp8 = dense_fp8()?;
    let d = Ld { ld: Loader { ck: &ckpt, rank, world, st }, fp8 };
    let (h, r) = (cfg.hidden, rank);
    let hl = cfg.heads / world;
    let kda = cfg.kda.ok_or("GLM-5.3-Flash without KDA")?;
    let khl = kda.heads / world;
    let dl = khl * kda.dim;
    let vl = cfg.vocab / world;
    let il = cfg.moe_inter / world;
    let dil = cfg.inter / world;
    let hcut = |n: usize| Some(r * n..(r + 1) * n);
    let t0 = std::time::Instant::now();
    let embed = Embed::Bf16(d.bf16_dev(&format!("{P}embed_tokens.weight"))?);
    let head = d.fused(&[("lm_head.weight".into(), hcut(vl))], 1)?;
    let norm = d.bf16_dev(&format!("{P}norm.weight"))?;
    let mut bytes = 2 * cfg.vocab * h + head.bytes();
    // one box, MTP: the draft passes' head over the listed ids
    let draft = match draft_ids(cfg.vocab)?.filter(|_| world == 1 && cfg.mtp) {
        Some(ids) => {
            let (w, k) = d.gather("lm_head.weight", &ids)?;
            let head = d.lin(w, ids.len(), k, 1)?;
            log(&format!("[flash] draft head: {} of {} ids listed ({:.0} MB; GLM_DRAFT_VOCAB=all for none)", ids.len(), cfg.vocab,
                         head.bytes() as f64 / 1e6));
            bytes += head.bytes();
            let mut listed = vec![false; cfg.vocab];
            ids.iter().for_each(|&i| listed[i as usize] = true);
            Some(crate::weights::DraftHead { head, ids, listed })
        }
        None => None,
    };
    let mut layers = Vec::with_capacity(cfg.n_layers());
    for l in 0..cfg.n_layers() {
        let p = format!("{P}layers.{l}.");
        let a = format!("{p}self_attn.");
        let mtp = cfg.mtp && l == cfg.mtp_layer();
        let attn = match cfg.attn[l] {
            crate::config::AttnKind::Kda => {
                let conv = {
                    let mut c = vec![];
                    for x in ["q", "k", "v"] {
                        c.extend(d.rows(&format!("{a}{x}_conv1d.weight"), r * dl..(r + 1) * dl)?.0);
                    }
                    DeviceBuffer::from_slice(&c)?
                };
                let o = {
                    let (w, n) = d.cols(&format!("{a}o_proj.weight"), r * dl..(r + 1) * dl)?;
                    d.lin(w, n, dl, 1)?
                };
                Attn::Kda(KdaW {
                    proj: d.fused(&[(format!("{a}q_proj.weight"), hcut(dl)), (format!("{a}k_proj.weight"), hcut(dl)),
                                    (format!("{a}v_proj.weight"), hcut(dl)), (format!("{a}f_a_proj.weight"), None),
                                    (format!("{a}g_a_proj.weight"), None), (format!("{a}b_proj.weight"), hcut(khl))], 1)?,
                    f_b: d.fused(&[(format!("{a}f_b_proj.weight"), hcut(dl))], 1)?,
                    g_b: d.fused(&[(format!("{a}g_b_proj.weight"), hcut(dl))], 1)?,
                    conv,
                    a_log: d.f32_dev(&format!("{a}A_log"), hcut(khl))?,
                    dt_bias: d.f32_dev(&format!("{a}dt_bias"), hcut(dl))?,
                    o_norm: d.bf16_dev(&format!("{a}o_norm.weight"))?,
                    o,
                })
            }
            crate::config::AttnKind::Dsa => {
                let i = format!("{a}indexer.");
                let rows = cfg.kvb_rows();
                let (kvb, k) = d.rows(&format!("{a}kv_b_proj.weight"), r * hl * rows..(r + 1) * hl * rows)?;
                if k != cfg.kv_lora {
                    return Err(format!("{a}kv_b_proj: K {k}").into());
                }
                let stage = DeviceBuffer::from_slice(&kvb)?;
                let kvb_q = DeviceBuffer::<u8>::new(hl * rows * 512)?;
                let kvb_s = DeviceBuffer::<f32>::new(hl * rows * 4)?;
                ck(unsafe { glm_quant_kvb_bf16(st.raw(), stage.raw(), (hl * rows) as i32, kvb_q.raw(), kvb_s.raw()) }, "quant kv_b")?;
                st.synchronize()?;
                drop(stage);
                let vd = cfg.v_dim;
                let o = {
                    let (w, n) = d.cols(&format!("{a}o_proj.weight"), r * hl * vd..(r + 1) * hl * vd)?;
                    d.lin(w, n, hl * vd, 4)?
                };
                Attn::Dsa(DsaW {
                    proj_a: d.fused(&[(format!("{a}q_a_proj.weight"), None), (format!("{a}kv_a_proj_with_mqa.weight"), None),
                                      (format!("{i}wk.weight"), None), (format!("{i}weights_proj.weight"), None),
                                      (format!("{i}index_kpool_compress_gate"), None)], 1)?,
                    q_norm: d.bf16_dev(&format!("{a}q_a_layernorm.weight"))?,
                    kv_norm: d.bf16_dev(&format!("{a}kv_a_layernorm.weight"))?,
                    proj_b: d.fused(&[(format!("{a}q_b_proj.weight"), hcut(hl * cfg.qk_dim())), (format!("{i}wq_b.weight"), None)], 1)?,
                    kvb_q,
                    kvb_s,
                    o,
                    kn_w: d.bf16_dev(&format!("{i}k_norm.weight"))?,
                    kn_b: d.bf16_dev(&format!("{i}k_norm.bias"))?,
                    ape: d.bf16_dev(&format!("{i}index_kpool_compress_ape"))?,
                })
            }
            crate::config::AttnKind::Mla => return Err("GLM-5.3-Flash with an MLA (RoPE) layer".into()),
            crate::config::AttnKind::Swa | crate::config::AttnKind::Full => return Err("GLM-5.3-Flash with an Inkling 2 layer".into()),
        };
        let hc = if mtp {
            None
        } else {
            Some(HcW {
                attn_fn: d.bf16_lin(&format!("{p}hc_attn_fn"), 16)?,
                attn_base: d.f32_dev(&format!("{p}hc_attn_base"), None)?,
                attn_scale: d.f32_dev(&format!("{p}hc_attn_scale"), None)?,
                ffn_fn: d.bf16_lin(&format!("{p}hc_ffn_fn"), 16)?,
                ffn_base: d.f32_dev(&format!("{p}hc_ffn_base"), None)?,
                ffn_scale: d.f32_dev(&format!("{p}hc_ffn_scale"), None)?,
            })
        };
        let mlp = if cfg.dense[l] {
            let m = format!("{p}mlp.");
            Mlp::Dense {
                gate_up: d.fused(&[(format!("{m}gate_proj.weight"), hcut(dil)), (format!("{m}up_proj.weight"), hcut(dil))], 1)?,
                down: d.fused(&[(format!("{m}down_proj.weight"), hcut(h / world))], 1)?,
            }
        } else {
            let m = format!("{p}mlp.shared_experts.");
            let (router, n, k) = d.all(&format!("{p}mlp.gate.weight"))?;
            if (n, k) != (cfg.experts, h) {
                return Err(format!("{p}mlp.gate: {n} x {k}").into());
            }
            Mlp::Moe {
                router: DeviceBuffer::from_slice(&router)?,
                router_bf16: true,
                bias: d.f32_dev(&format!("{p}mlp.gate.e_score_correction_bias"), None)?,
                sh_gate_up: d.fused(&[(format!("{m}gate_proj.weight"), hcut(il)), (format!("{m}up_proj.weight"), hcut(il))], 1)?,
                sh_down: d.fused(&[(format!("{m}down_proj.weight"), hcut(h / world))], 1)?,
            }
        };
        let mtpx = if mtp {
            Some(MtpExtra {
                enorm: d.bf16_dev(&format!("{p}enorm.weight"))?,
                hnorm: d.bf16_dev(&format!("{p}hnorm.weight"))?,
                sh_norm: d.bf16_dev(&format!("{p}shared_head.norm.weight"))?,
                eh: d.fused(&[(format!("{p}eh_proj.weight"), hcut(h / world))], 2)?,
            })
        } else {
            None
        };
        let layer = Layer {
            ln_in: d.bf16_dev(&format!("{p}input_layernorm.weight"))?,
            ln_post: d.bf16_dev(&format!("{p}post_attention_layernorm.weight"))?,
            attn,
            hc,
            mlp,
            mtp: mtpx,
        };
        bytes += layer_bytes(&layer);
        layers.push(layer);
        if l % 9 == 8 {
            log(&format!("[flash] dense weights ({}): layer {l} ({:.1} GB, {:.0} s)", if fp8 { "fp8" } else { "bf16" },
                         bytes as f64 / 1e9, t0.elapsed().as_secs_f64()));
        }
    }
    Ok(Weights { layers, embed, norm, head, draft, cos: DeviceBuffer::from_slice(&[0u16])?, sin: DeviceBuffer::from_slice(&[0u16])?, bytes })
}

fn layer_bytes(l: &Layer) -> usize {
    let mut b = match &l.attn {
        Attn::Kda(k) => k.proj.bytes() + k.f_b.bytes() + k.g_b.bytes() + k.o.bytes() + 2 * k.conv.len(),
        Attn::Dsa(s) => s.proj_a.bytes() + s.proj_b.bytes() + s.o.bytes() + s.kvb_q.len() + 4 * s.kvb_s.len(),
        Attn::Mla(_) | Attn::Ink(_) => 0,
    };
    if let Some(hc) = &l.hc {
        b += hc.attn_fn.bytes() + hc.ffn_fn.bytes();
    }
    b += match &l.mlp {
        Mlp::Dense { gate_up, down } => gate_up.bytes() + down.bytes(),
        Mlp::Moe { router, sh_gate_up, sh_down, .. } => 2 * router.len() + sh_gate_up.bytes() + sh_down.bytes(),
        Mlp::Routed { router, .. } => 2 * router.len(),
    };
    if let Some(m) = &l.mtp {
        b += m.eh.bytes();
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_ranking_lists_whole_blocks_of_this_vocabulary() {
        // glm5_next's vocabulary (config.json vocab_size): an id outside it fails the parse
        let vocab = 154880;
        let order = tidelm_engine::draft_vocab::ranking(DRAFT_RANKING_V1, vocab).unwrap();
        assert!(order.len() >= DRAFT_ROWS, "{} ranked ids for {DRAFT_ROWS} rows", order.len());
        let ids = tidelm_engine::draft_vocab::select(&order, DRAFT_ROWS, vocab).unwrap();
        assert_eq!(ids.len(), DRAFT_ROWS);
        assert_eq!(DRAFT_ROWS % 32, 0);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        // the template's and the tool calls' tokens are the vocabulary's last ids: all listed
        assert!((154820..154856u32).all(|i| ids.binary_search(&i).is_ok()));
    }
}
