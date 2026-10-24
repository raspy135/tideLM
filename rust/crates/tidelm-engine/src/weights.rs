//! Device weights, loaded from the checkpoint in its stored formats (engine/model.py::
//! Weights, tools/v41_ref.py::LayerWeights). Dense projections stay FP8 with their
//! UE8M0 block scales; norms and the router gate stay BF16; HC projections and the
//! attention sink are FP32 as stored.
//!
//! One deliberate format difference from the Python: the router gate and the ratio-2
//! compressor weights are stored BF16 and the Python widens them to fp32 copies before
//! a fp32 GEMM. Widening bf16 is exact, so the bf16-weight kernel with fp32 accumulation
//! computes the same products from half the bytes (7.9 -> 3.9 MB per layer for the
//! gate, 21 -> 10.5 MB per ratio-2 layer for the compressor). Not a numerics change.

use tidelm_cuda::{memcpy_async, DeviceBuffer, PinnedBuffer, Stream, MEMCPY_HOST_TO_DEVICE};
use tidelm_model::{aligned_len, Checkpoint, TensorMeta};

use crate::config::Config;
use crate::Result;

/// Pinned staging + the stream uploads are ordered on.
pub struct Loader<'a> {
    pub ckpt: &'a Checkpoint,
    pub stream: &'a Stream,
    stage: PinnedBuffer<u8>,
    piece: usize,
}

impl<'a> Loader<'a> {
    pub const PIECE: usize = 32 << 20;

    pub fn new(ckpt: &'a Checkpoint, stream: &'a Stream) -> Result<Self> {
        Ok(Loader { ckpt, stream, stage: PinnedBuffer::uninit(Self::PIECE + 2 * 4096)?, piece: Self::PIECE })
    }

    /// Read tensor bytes `[lo, hi)` into device memory at `dst`, piece by piece.
    pub fn read_into(&mut self, name: &str, lo: usize, hi: usize, dst: *mut std::ffi::c_void) -> Result<()> {
        let mut at = lo;
        while at < hi {
            let end = (at + self.piece).min(hi);
            let off = self.ckpt.read_tensor_range(name, at, end, &mut self.stage)?;
            debug_assert!(off + (end - at) <= self.stage.len());
            let src = unsafe { self.stage.as_ptr().add(off) } as *const std::ffi::c_void;
            let d = unsafe { (dst as *mut u8).add(at - lo) } as *mut std::ffi::c_void;
            // the destination is a device buffer the caller sized for [lo, hi)
            unsafe { memcpy_async(d, src, end - at, MEMCPY_HOST_TO_DEVICE, self.stream)? };
            // the staging buffer is reused by the next piece
            self.stream.synchronize()?;
            at = end;
        }
        Ok(())
    }

    fn checked(&self, name: &str, dtype: &str, shape: Option<&[usize]>) -> Result<TensorMeta> {
        let meta = self.ckpt.meta(name)?;
        if meta.dtype != dtype {
            return Err(format!("{name}: dtype {} where {dtype} was expected", meta.dtype).into());
        }
        if let Some(s) = shape {
            if meta.shape != s {
                return Err(format!("{name}: shape {:?} where {s:?} was expected", meta.shape).into());
            }
        }
        Ok(meta)
    }

    pub fn u8(&mut self, name: &str, dtype: &str, shape: Option<&[usize]>) -> Result<DeviceBuffer<u8>> {
        let meta = self.checked(name, dtype, shape)?;
        let buf = DeviceBuffer::<u8>::new(meta.nbytes())?;
        self.read_into(name, 0, meta.nbytes(), buf.raw())?;
        Ok(buf)
    }

    pub fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<DeviceBuffer<u16>> {
        let meta = self.checked(name, "BF16", Some(shape))?;
        let buf = DeviceBuffer::<u16>::new(meta.numel())?;
        self.read_into(name, 0, meta.nbytes(), buf.raw())?;
        Ok(buf)
    }

    pub fn f32(&mut self, name: &str, shape: &[usize]) -> Result<DeviceBuffer<f32>> {
        let meta = self.checked(name, "F32", Some(shape))?;
        let buf = DeviceBuffer::<f32>::new(meta.numel())?;
        self.read_into(name, 0, meta.nbytes(), buf.raw())?;
        Ok(buf)
    }

    /// A small tensor read to the host (engram q/k weights, confidence head).
    pub fn host_bytes(&mut self, name: &str) -> Result<(Vec<u8>, TensorMeta)> {
        Ok(self.ckpt.read_tensor_vec(name)?)
    }

    /// `<prefix>.weight` (F8_E4M3 [N, K]) + `<prefix>.scale` (F8_E8M0 [ceil(N/32), ceil(K/32)]).
    pub fn fp8(&mut self, prefix: &str, n: usize, k: usize) -> Result<Fp8> {
        let w = self.u8(&format!("{prefix}.weight"), "F8_E4M3", Some(&[n, k]))?;
        let s = self.u8(&format!("{prefix}.scale"), "F8_E8M0", Some(&[n.div_ceil(32), k.div_ceil(32)]))?;
        Ok(Fp8 { w, s, n, k })
    }

    /// Bytes of pinned staging an arbitrary range read needs (for the head/embed streams).
    pub fn stage_capacity(&self) -> usize {
        self.stage.len()
    }

    pub fn aligned(lo: u64, hi: u64) -> usize {
        aligned_len(lo, hi)
    }
}

/// FP8 e4m3 weight with UE8M0 32x32 block scales (tools/fp8_linear.py::FP8Weight).
pub struct Fp8 {
    pub w: DeviceBuffer<u8>,
    pub s: DeviceBuffer<u8>,
    pub n: usize,
    pub k: usize,
}

impl Fp8 {
    pub fn lds(&self) -> usize {
        self.k.div_ceil(32)
    }
}

pub struct Compressor {
    pub norm: DeviceBuffer<u16>,
    pub wkv: DeviceBuffer<u16>,
    /// ratio-2 sources only
    pub wgate: Option<DeviceBuffer<u16>>,
}

pub struct IndexKey {
    pub wk: DeviceBuffer<u16>,
    pub k_norm: DeviceBuffer<u16>,
}

pub struct Indexer {
    pub wq_b: Fp8,
    pub weights_proj: DeviceBuffer<u16>,
    /// kv-source indexers own the index key projection of the compressed latent
    pub key: Option<IndexKey>,
}

pub struct EngramW {
    pub wkv: Fp8,
    /// q_weight * k_weight in fp32 ([hc, dim]); the product of two bf16 values is exact.
    pub weight: DeviceBuffer<f32>,
}

pub struct Layer {
    /// `layers.{l}.` or `mtp.{k}.` — also the routed experts' tensor prefix
    pub prefix: String,
    pub n_experts: usize,
    pub topk: usize,
    pub attn_norm: DeviceBuffer<u16>,
    pub ffn_norm: DeviceBuffer<u16>,
    pub q_norm: DeviceBuffer<u16>,
    pub kv_norm: DeviceBuffer<u16>,
    pub attn_sink: DeviceBuffer<f32>,
    pub wq_a: Fp8,
    pub wq_b: Fp8,
    pub wkv: Fp8,
    pub wo_a: Fp8,
    pub wo_b: Fp8,
    pub hc_attn_fn: DeviceBuffer<f32>,
    pub hc_attn_base: DeviceBuffer<f32>,
    pub hc_attn_scale: DeviceBuffer<f32>,
    pub hc_ffn_fn: DeviceBuffer<f32>,
    pub hc_ffn_base: DeviceBuffer<f32>,
    pub hc_ffn_scale: DeviceBuffer<f32>,
    pub gate_w: DeviceBuffer<u16>,
    pub gate_bias: DeviceBuffer<f32>,
    pub sh_w1: Fp8,
    pub sh_w2: Fp8,
    pub sh_w3: Fp8,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
    pub engram: Option<EngramW>,
}

impl Layer {
    /// A backbone layer (`layers.{l}.`).
    pub fn load(ld: &mut Loader, cfg: &Config, l: usize) -> Result<Self> {
        Self::load_block(ld, cfg, &format!("layers.{l}."), Some(l), cfg.n_routed_experts, cfg.n_activated_experts)
    }

    /// A DSpark block (`mtp.{k}.`, model.py::MTPWeights): a window-only block with its own
    /// 128-expert, top-3 MoE.
    pub fn load_mtp(ld: &mut Loader, cfg: &Config, k: usize) -> Result<Self> {
        Self::load_block(ld, cfg, &format!("mtp.{k}."), None, cfg.dspark_n_routed_experts,
                         cfg.dspark_n_activated_experts)
    }

    fn load_block(ld: &mut Loader, cfg: &Config, p: &str, backbone: Option<usize>, n_experts: usize,
                  topk: usize) -> Result<Self> {
        let p = p.to_string();
        let l = backbone.unwrap_or(usize::MAX);
        let (d, hcw) = (cfg.dim, cfg.hc_mix_width());
        let attn = |n: &str| format!("{p}attn.{n}");
        let compressor = if backbone.is_some() && cfg.kv_source_layers.contains(&l) {
            let ratio = cfg.compress_ratios[l];
            Some(Compressor {
                norm: ld.bf16(&attn("compressor.norm.weight"), &[cfg.head_dim])?,
                wkv: ld.bf16(&attn("compressor.wkv.weight"), &[cfg.head_dim, d])?,
                wgate: if ratio > 1 {
                    Some(ld.bf16(&attn("compressor.wgate.weight"), &[cfg.head_dim, d])?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let indexer = if backbone.is_some() && cfg.index_source_layers.contains(&l) {
            let key = if cfg.kv_source_layers.contains(&l) {
                Some(IndexKey {
                    wk: ld.bf16(&attn("indexer.wk.weight"), &[cfg.index_head_dim, cfg.head_dim])?,
                    k_norm: ld.bf16(&attn("indexer.k_norm.weight"), &[cfg.index_head_dim])?,
                })
            } else {
                None
            };
            Some(Indexer {
                wq_b: ld.fp8(&attn("indexer.wq_b"), cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)?,
                weights_proj: ld.bf16(&attn("indexer.weights_proj.weight"), &[cfg.index_n_heads, d])?,
                key,
            })
        } else {
            None
        };
        let engram = if backbone.is_some() && cfg.engram_layer_ids.contains(&l) {
            let width = (cfg.hc_mult + 1) * d;
            let rows = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim;
            let q = bf16_vec(&ld.host_bytes(&format!("{p}engram.q_weight"))?.0);
            let k = bf16_vec(&ld.host_bytes(&format!("{p}engram.k_weight"))?.0);
            let weight: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
            Some(EngramW {
                wkv: ld.fp8(&format!("{p}engram.wkv"), width, rows)?,
                weight: DeviceBuffer::from_slice(&weight)?,
            })
        } else {
            None
        };
        Ok(Layer {
            n_experts,
            topk,
            attn_norm: ld.bf16(&format!("{p}attn_norm.weight"), &[d])?,
            ffn_norm: ld.bf16(&format!("{p}ffn_norm.weight"), &[d])?,
            q_norm: ld.bf16(&attn("q_norm.weight"), &[cfg.q_lora_rank])?,
            kv_norm: ld.bf16(&attn("kv_norm.weight"), &[cfg.head_dim])?,
            attn_sink: ld.f32(&attn("attn_sink"), &[cfg.n_heads])?,
            wq_a: ld.fp8(&attn("wq_a"), cfg.q_lora_rank, d)?,
            wq_b: ld.fp8(&attn("wq_b"), cfg.n_heads * cfg.head_dim, cfg.q_lora_rank)?,
            wkv: ld.fp8(&attn("wkv"), cfg.head_dim, d)?,
            wo_a: ld.fp8(&attn("wo_a"), cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * cfg.head_dim / cfg.o_groups)?,
            wo_b: ld.fp8(&attn("wo_b"), d, cfg.o_groups * cfg.o_lora_rank)?,
            hc_attn_fn: ld.f32(&format!("{p}hc_attn_fn"), &[hcw, cfg.hc_mult * d])?,
            hc_attn_base: ld.f32(&format!("{p}hc_attn_base"), &[hcw])?,
            hc_attn_scale: ld.f32(&format!("{p}hc_attn_scale"), &[3])?,
            hc_ffn_fn: ld.f32(&format!("{p}hc_ffn_fn"), &[hcw, cfg.hc_mult * d])?,
            hc_ffn_base: ld.f32(&format!("{p}hc_ffn_base"), &[hcw])?,
            hc_ffn_scale: ld.f32(&format!("{p}hc_ffn_scale"), &[3])?,
            gate_w: ld.bf16(&format!("{p}ffn.gate.weight"), &[n_experts, d])?,
            gate_bias: ld.f32(&format!("{p}ffn.gate.bias"), &[n_experts])?,
            sh_w1: ld.fp8(&format!("{p}ffn.shared_experts.w1"), cfg.moe_inter_dim, d)?,
            sh_w2: ld.fp8(&format!("{p}ffn.shared_experts.w2"), d, cfg.moe_inter_dim)?,
            sh_w3: ld.fp8(&format!("{p}ffn.shared_experts.w3"), cfg.moe_inter_dim, d)?,
            compressor,
            indexer,
            engram,
            prefix: p,
        })
    }
}

/// The DSpark drafter's own weights besides its three blocks (model.py::MTPWeights k = 0, 2).
pub struct DsparkW {
    pub main_proj: Fp8,
    pub main_norm: DeviceBuffer<u16>,
    pub norm: DeviceBuffer<u16>,
    pub markov_embed: DeviceBuffer<u16>,
    pub markov_head: DeviceBuffer<u16>,
}

impl DsparkW {
    pub fn load(ld: &mut Loader, cfg: &Config) -> Result<Self> {
        let (d, v, r) = (cfg.dim, cfg.vocab_size, cfg.dspark_markov_rank);
        let last = cfg.n_mtp_layers - 1;
        Ok(DsparkW {
            main_proj: ld.fp8("mtp.0.main_proj", d, d * cfg.dspark_target_layer_ids.len())?,
            main_norm: ld.bf16("mtp.0.main_norm.weight", &[d])?,
            norm: ld.bf16(&format!("mtp.{last}.norm.weight"), &[d])?,
            markov_embed: ld.bf16(&format!("mtp.{last}.markov_head.embed.weight"), &[v, r])?,
            markov_head: ld.bf16(&format!("mtp.{last}.markov_head.head.weight"), &[v, r])?,
        })
    }
}

/// Little-endian bf16 bytes -> f32 (exact).
pub fn bf16_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()
}
