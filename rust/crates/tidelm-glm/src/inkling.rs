//! Inkling 2 (inkling2) weights for one rank: the checkpoint's BF16 non-expert tensors (their
//! native names, `model.llm.*`) served as FP8 with 32 x 32 block scales (GLM_INK_DENSE=fp8, the
//! default) or as BF16 (=bf16, the reference), fused where one input feeds several projections
//! (weights::InkW). The routed experts come from the pack; the vision tower and the DSpark drafter
//! (mtp.safetensors) are not loaded.
//!
//! TP cuts: query heads and key / value heads by rank (hl = heads / world query heads, kl = hl / 6
//! key / value heads: a key head's six query heads stay together), so q / k / v / r / gate / sink
//! and the key / value convolutions' taps are cut by rows, o_proj by its input columns (the ranks'
//! partial outputs are summed), the head by vocabulary. The router, the norms, the block
//! residual's queries, the attention-output and MoE-output convolutions and the embedding are whole
//! on every rank: both compute the same rows from the same input.
//!
//! The embedding is stored normed: Inkling2NormedEmbedding is a row lookup and an RMSNorm of that
//! row, a function of the id alone, so the table holds bf16(w * x * rsqrt(mean(x^2) + eps)).

use std::ops::Range;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_model::Checkpoint;

use crate::config::{AttnKind, GlmConfig};
use crate::lin::{self, Bf16Mat, Lin};
use crate::weights::{bf16_from_f32, Attn, Embed, InkOff, InkW, Layer, Loader, Mlp, Weights};
use crate::Result;

const P: &str = "model.llm.";

/// GLM_INK_DENSE: the non-expert weights' format (fp8, the default, or bf16). It moves values: in
/// the boot guard.
pub fn dense_fp8() -> Result<bool> {
    match std::env::var("GLM_INK_DENSE").as_deref() {
        Err(_) | Ok("") | Ok("fp8") => Ok(true),
        Ok("bf16") => Ok(false),
        Ok(v) => Err(format!("GLM_INK_DENSE={v}: fp8 or bf16").into()),
    }
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f32_of(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
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

    fn all(&self, name: &str) -> Result<(Vec<u16>, usize, usize)> {
        let (n, k) = self.dims(name)?;
        Ok((self.rows(name, 0..n)?.0, n, k))
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

    /// Row ranges of several [*][k] tensors stacked into one matrix, each padded with zero rows to
    /// whole 32-row blocks (an FP8 scale never straddles two tensors); the parts' row offsets.
    fn fused(&self, parts: &[(String, Range<usize>)], split: usize) -> Result<(Lin, Vec<usize>)> {
        let (mut all, mut n, mut k0, mut offs) = (vec![], 0usize, 0usize, vec![]);
        for (name, r) in parts {
            let (v, k) = self.rows(name, r.clone())?;
            if k0 != 0 && k != k0 {
                return Err(format!("{name}: K {k}, the fused matrix's {k0}").into());
            }
            k0 = k;
            offs.push(n);
            all.extend_from_slice(&v);
            let pad = r.len().div_ceil(32) * 32;
            all.resize(all.len() + (pad - r.len()) * k, 0);
            n += pad;
        }
        Ok((self.lin(all, n, k0, split)?, offs))
    }

    fn bf16_dev(&self, name: &str) -> Result<DeviceBuffer<u16>> {
        Ok(DeviceBuffer::from_slice(&self.all(name)?.0)?)
    }

    /// Several [H] BF16 vectors stacked.
    fn stack(&self, names: &[String]) -> Result<DeviceBuffer<u16>> {
        let mut v = vec![];
        for n in names {
            v.extend(self.all(n)?.0);
        }
        Ok(DeviceBuffer::from_slice(&v)?)
    }
}

pub fn load_weights(model_dir: &Path, cfg: &GlmConfig, rank: usize, world: usize, st: &Stream, log: &dyn Fn(&str)) -> Result<Weights> {
    let ckpt = Checkpoint::open(model_dir)?;
    let fp8 = dense_fp8()?;
    let d = Ld { ld: Loader { ck: &ckpt, rank, world, st }, fp8 };
    let ink = cfg.ink.ok_or("Inkling 2 without its config")?;
    let (h, r) = (cfg.hidden, rank);
    if cfg.heads % world != 0 || ink.kv_heads % world != 0 || cfg.vocab % (32 * world) != 0 {
        return Err(format!("Inkling 2 on {world} ranks: {} query heads, {} key / value heads and {} ids do not divide", cfg.heads,
                           ink.kv_heads, cfg.vocab).into());
    }
    let hl = cfg.heads / world;
    let kl = ink.kv_heads / world;
    let (dq, dk, dr) = (hl * ink.head_dim, kl * ink.head_dim, hl * ink.d_rel);
    let vl = cfg.vocab / world;
    let cut = |n: usize| r * n..(r + 1) * n;
    let t0 = std::time::Instant::now();

    // the embedding, normed (module doc): fp32 as the reference's RMSNorm, one rounding
    let embed = {
        let (mut tab, n, k) = d.all(&format!("{P}embed.weight"))?;
        let w: Vec<f32> = d.all(&format!("{P}embed_norm.weight"))?.0.into_iter().map(f32_of).collect();
        if (n, k) != (cfg.vocab, h) || w.len() != h {
            return Err(format!("{P}embed.weight: {n} x {k}").into());
        }
        let mut x = vec![0f32; h];
        for row in tab.chunks_exact_mut(h) {
            let mut ss = 0f32;
            for (xf, &b) in x.iter_mut().zip(row.iter()) {
                *xf = f32_of(b);
                ss += *xf * *xf;
            }
            let rs = 1.0 / (ss / h as f32 + cfg.rms_eps).sqrt();
            for ((o, &xf), &wf) in row.iter_mut().zip(&x).zip(&w) {
                *o = bf16_from_f32(wf * (xf * rs));
            }
        }
        Embed::Bf16(DeviceBuffer::from_slice(&tab)?)
    };
    let (head, _) = d.fused(&[(format!("{P}unembed.weight"), cut(vl))], 1)?;
    let norm = d.bf16_dev(&format!("{P}norm.weight"))?;
    let mut bytes = 2 * cfg.vocab * h + head.bytes();
    let mut layers = Vec::with_capacity(cfg.layers);
    for l in 0..cfg.layers {
        let p = format!("{P}layers.{l}.");
        let a = format!("{p}attn.");
        let full = cfg.attn[l] == AttnKind::Full;
        let (q, k, v, rr, g, s) = (format!("{a}wq_du.weight"), format!("{a}wk_dv.weight"), format!("{a}wv_dv.weight"),
                                   format!("{a}wr_du.weight"), format!("{a}gate_attn.weight"), format!("{a}attn_sink_proj.weight"));
        let (proj, proj_k, proj_v, off, k_conv, v_conv) = if full {
            let (proj, o) = d.fused(&[(q, cut(dq)), (rr, cut(dr)), (g, cut(dq)), (s, cut(hl))], 1)?;
            let (pk, ok) = d.fused(&[(k, cut(dk)), (format!("{a}k_sconv.weight_proj"), cut(kl * ink.conv))], 1)?;
            let (pv, _) = d.fused(&[(v, cut(dk)), (format!("{a}v_sconv.weight_proj"), cut(kl * ink.conv))], 1)?;
            // static_w [taps][all channels]: this rank's channels of each lag
            let stat = |name: String| -> Result<DeviceBuffer<u16>> {
                let (w, n) = d.cols(&name, cut(dk))?;
                if n != ink.conv {
                    return Err(format!("{name}: {n} lags").into());
                }
                Ok(DeviceBuffer::from_slice(&w)?)
            };
            (proj, Some(pk), Some(pv), InkOff { q: o[0], r: o[1], gate: o[2], sink: o[3], k: 0, v: 0, taps: ok[1] },
             stat(format!("{a}k_sconv.static_w"))?, stat(format!("{a}v_sconv.static_w"))?)
        } else {
            let (proj, o) = d.fused(&[(q, cut(dq)), (k, cut(dk)), (v, cut(dk)), (rr, cut(dr)), (g, cut(dq)), (s, cut(hl))], 1)?;
            let taps = |name: String| -> Result<DeviceBuffer<u16>> { Ok(DeviceBuffer::from_slice(&d.rows(&name, cut(dk))?.0)?) };
            (proj, None, None, InkOff { q: o[0], k: o[1], v: o[2], r: o[3], gate: o[4], sink: o[5], taps: 0 },
             taps(format!("{a}k_sconv.weight"))?, taps(format!("{a}v_sconv.weight"))?)
        };
        let o = {
            let (w, n) = d.cols(&format!("{a}wo_ud.weight"), cut(dq))?;
            d.lin(w, n, dq, 1)?
        };
        // rel_logits_proj.proj [d_rel][extent] -> [extent][d_rel]
        let proj_t = {
            let (w, n, e) = d.all(&format!("{a}rel_logits_proj.proj"))?;
            let want = if full { ink.rel_extent } else { ink.window };
            if (n, e) != (ink.d_rel, want) {
                return Err(format!("{a}rel_logits_proj.proj: {n} x {e}").into());
            }
            let mut t = vec![0u16; n * e];
            for c in 0..n {
                for x in 0..e {
                    t[x * n + c] = w[c * e + x];
                }
            }
            DeviceBuffer::from_slice(&t)?
        };
        let q_attn = if l == 0 {
            None
        } else {
            let v = d.ld.f32s(&format!("{P}attn_res.attn_queries.{}", l - 1))?;
            if v.len() != if full { 3 * h } else { h } {
                return Err(format!("{P}attn_res.attn_queries.{}: {} values", l - 1, v.len()).into());
            }
            Some(DeviceBuffer::from_slice(&v)?)
        };
        let (router, n, kk) = d.all(&format!("{p}mlp.gate.weight"))?;
        if (n, kk) != (cfg.experts, h) {
            return Err(format!("{p}mlp.gate: {n} x {kk}").into());
        }
        let ln_in = if full {
            d.stack(&[format!("{p}q_input_norm.weight"), format!("{p}k_input_norm.weight"), format!("{p}v_input_norm.weight")])?
        } else {
            d.bf16_dev(&format!("{p}attn_norm.weight"))?
        };
        let w = InkW {
            proj,
            proj_k,
            proj_v,
            off,
            o,
            q_scale: DeviceBuffer::from_slice(&d.rows(&format!("{a}q_l2_clip.scale"), cut(hl))?.0)?,
            r_scale: DeviceBuffer::from_slice(&d.rows(&format!("{a}rel_l2_clip.scale"), cut(hl))?.0)?,
            proj_t,
            k_conv,
            v_conv,
            attn_conv: d.bf16_dev(&format!("{p}attn_sconv.weight"))?,
            mlp_conv: d.bf16_dev(&format!("{p}mlp_sconv.weight"))?,
            q_attn,
            q_mlp: d.ld.f32_dev(&format!("{P}attn_res.mlp_queries.{l}"))?,
        };
        bytes += w.proj.bytes() + w.proj_k.as_ref().map_or(0, |m| m.bytes()) + w.proj_v.as_ref().map_or(0, |m| m.bytes())
            + w.o.bytes() + 2 * (router.len() + w.k_conv.len() + w.v_conv.len() + w.attn_conv.len() + w.mlp_conv.len() + w.proj_t.len());
        layers.push(Layer {
            ln_in,
            ln_post: d.bf16_dev(&format!("{p}mlp_norm.weight"))?,
            attn: Attn::Ink(w),
            hc: None,
            mlp: Mlp::Routed { router: DeviceBuffer::from_slice(&router)?, bias: d.ld.f32_dev(&format!("{p}mlp.gate.bias"))? },
            mtp: None,
        });
        if l % 12 == 11 {
            log(&format!("[inkling] dense weights ({}): layer {l} ({:.1} GB, {:.0} s)", if fp8 { "fp8" } else { "bf16" },
                         bytes as f64 / 1e9, t0.elapsed().as_secs_f64()));
        }
    }
    Ok(Weights { layers, embed, norm, head, draft: None, cos: DeviceBuffer::from_slice(&[0u16])?, sin: DeviceBuffer::from_slice(&[0u16])?, bytes })
}

/// The final mix's query (the block attention residual before the last norm), fp32 [H]: read by
/// the engine's Inkling state at load (it is not a layer's).
pub fn output_query(model_dir: &Path, st: &Stream) -> Result<DeviceBuffer<f32>> {
    let ckpt = Checkpoint::open(model_dir)?;
    Loader { ck: &ckpt, rank: 0, world: 1, st }.f32_dev(&format!("{P}attn_res.output_query"))
}
