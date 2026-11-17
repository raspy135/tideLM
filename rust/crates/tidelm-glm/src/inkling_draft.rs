//! Inkling 2's speculative-decoding drafter ("DSpark", the checkpoint's mtp.safetensors) on one GPU:
//! five dense transformer layers over a block of 8 rows [anchor, mask x 7] and a 7-step recurrent
//! Markov head, 7 draft tokens a round. The specification this follows, with its sources, is
//! docs/inkling-dspark.md (SGLang's inkling-2-small-support branch); its
//! section 8 is what the kernels (inkling_draft.cu) and tools/inkling_draft_ref.py both transcribe.
//!
//! What the drafter reads of the target, and when:
//!   store_context  after every target pass, from the pass's 8 hidden-state taps a row (the inputs
//!                  of layers 5, 11, .., 41 and the final normed hidden): each drafter layer's
//!                  context keys / values at the rows' positions (mix by softmax(fuse_weight), a
//!                  weighted RMSNorm, [k_proj_ctx | v_proj_ctx], k_norm_ctx, RoPE) and f_rnn (the
//!                  head's mix of the same taps), a row a position.
//!   draft          the anchor's normed embedding row and the context: positions [p - 1024, p) for
//!                  layers 0, 1, 3, 4 and as far back as the global ring holds for layer 2, plus the
//!                  block's own 8 keys; f_rnn of position p - 1; the target's unembed (a full copy
//!                  here: the pair's ranks hold half each) for the base logits and, folded with
//!                  hidden_out at load, for the head's correction.
//!
//! State a lane keeps, all by absolute position (slot = position & (ring - 1)), so a verify pass's
//! rejected rows are never undone - the next pass or block writes over them before they are read:
//! key / value rings of `local_ring` positions for the four local layers and `global_ring` for
//! layer 2 (the reference keeps layer 2's whole context; the ring's size is the caller's choice),
//! rows [8 heads][K 128 | V 128] bf16, and 64 positions of f_rnn. A block's own keys / values go
//! to the rings at p .. p + 7 as the reference writes them to its cache.
//!
//! The spec's open questions, as built: the fused residual add norms the unrounded fp32 sum and
//! stores the residual rounded (Q5, SGLang's forward_native); alpha = softmax(fuse_weight) is
//! rounded to bf16 (Q6, SGLang); query head h reads key / value head h / 6 (Q7). Every matrix is a
//! Lin: FP8 with 32 x 32 block scales (`fp8`) or BF16 as stored - the reference is BF16 only. Where
//! this departs from the reference's order of operations, in fp32 and below bf16's resolution:
//! rnn_proj is applied as its block-constant columns ([h_i | f_rnn], one GEMM for the 7 steps) plus
//! its per-step columns ([state | markov_w1[prev]]), summed with the bias and rounded once.
//!
//! Loading never holds a matrix on the host: tensors are read with O_DIRECT in pieces of
//! CHUNK_BYTES and uploaded (and quantized) a piece at a time, and the fold of hidden_out through
//! the unembed (201,024 x 4,608 x 256 products, fp64) runs on the GPU over the unembed's pieces.
//!
//! One draft round at a time an InkDraft (the block's scratch is its own); store_context and draft
//! are ordered by the stream they are given. Neither reads anything back to the host - unless
//! TIDELM_DRAFT_DUMP=PREFIX is set (checks only: inkling_draft_check), which writes intermediates
//! to PREFIX.NAME.bin and synchronizes for each.

use std::os::raw::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::ds41::ds_quant_fp8_block32;
use tidelm_kernels::glm::glm_swiglu;
use tidelm_kernels::inkling_draft::*;
use tidelm_model::direct_io::{AlignedBuf, ALIGN};
use tidelm_model::safetensors::Shard;
use tidelm_model::{aligned_len, WeightIndex};

use crate::lin::{self, Bf16Mat, Fp8Mat, Lin};
use crate::{ck, Result};

/// hidden size
pub const D: usize = 4608;
/// draft tokens a round
pub const DRAFTS: usize = BLOCK - 1;
/// target hidden states a position, in target_layer_ids order
pub const TAPS: usize = 8;
const HQ: usize = 48;
const HK: usize = 8;
const HD: usize = 128;
const QDIM: usize = HQ * HD;
const KDIM: usize = HK * HD;
const QKV: usize = QDIM + 2 * KDIM;
/// elements of a ring row: [8 heads][K 128 | V 128]
const ROW: usize = 2 * KDIM;
const INTER: usize = 18432;
const LAYERS: usize = 5;
const RANK: usize = 256;
/// committed positions a local layer sees before the block
pub const WINDOW: usize = 1024;
/// the layer that sees the whole (here: the global ring's) context
const GLOBAL_LAYER: usize = 2;
/// a layer's place among the local rings
const LOCAL_OF: [usize; LAYERS] = [0, 1, usize::MAX, 2, 3];
const TARGET_LAYER_IDS: [u64; TAPS] = [5, 11, 17, 23, 29, 35, 41, 48];
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
/// Key splits of the block's attention: a local layer's <= 1,032 keys, the global ring. Each split
/// is one block of 128 threads a key / value head; chosen for the grid's size (256 and 512
/// blocks), not measured (2026-11-16: written without a GPU).
const NS_LOCAL: usize = 32;
const NS_GLOBAL: usize = 64;
/// fp32 split-K partials for store_context's [k_proj_ctx | v_proj_ctx] GEMM past 16 rows: the most
/// lin::mma_splits asks for at N = 2,048 (6 splits x 128 rows)
const PART: usize = 6 * 128 * 2 * KDIM;
/// host bytes of a tensor read and uploaded at a time
const CHUNK_BYTES: usize = 48 << 20;
/// a layer's vectors, one buffer: [input_layernorm | post_attention_layernorm | dflare_norm | q_norm
/// | k_norm | k_norm_ctx]
const N_IN: usize = 0;
const N_POST: usize = D;
const N_DFLARE: usize = 2 * D;
const N_Q: usize = 3 * D;
const N_K: usize = 3 * D + HD;
const N_KCTX: usize = 3 * D + 2 * HD;
const N_LEN: usize = 3 * D + 3 * HD;
/// the model's vectors: [norm | mask_embed | markov_head.dflare_norm]
const M_NORM: usize = 0;
const M_MASK: usize = D;
const M_DFLARE: usize = 2 * D;
const MTP_FILE: &str = "mtp.safetensors";
const UNEMBED: &str = "model.llm.unembed.weight";

/// The built-in ranking of the listed vocabulary (engine/inkling_draft_vocab_ranking_v1.txt: how it
/// was built, and each group's held-out tokens unlisted - at 65,536 ids chat 0.91%, code 0.26%,
/// Japanese 0.29%, Chinese 0.90%, Korean 0.46%, European Wikipedia 1.5-2.6%).
pub const RANKING_V1: &str = include_str!("../../../../engine/inkling_draft_vocab_ranking_v1.txt");
pub const LISTED_ROWS: usize = 65536;

/// The listed ids (ascending) for open_with: GLM_DRAFT_VOCAB / GLM_DRAFT_VOCAB_ROWS as
/// GLM-5.3-Flash reads them (unset: RANKING_V1's first LISTED_ROWS; `all`: None, every id).
pub fn listed_ids(real: usize) -> Result<Option<Vec<u32>>> {
    crate::flash::draft_ids_from(RANKING_V1, LISTED_ROWS, real)
}

#[derive(Clone, Copy, Debug)]
pub struct DraftOpts {
    /// every matrix (the layers', the head's, the unembed copy) as FP8 with 32 x 32 block scales;
    /// false: BF16 as stored, the reference
    pub fp8: bool,
    pub lanes: usize,
    /// positions of a local layer's ring: a power of two >= 1024 + rows of a pass
    pub local_ring: usize,
    /// positions of layer 2's ring, a power of two: its context is the last global_ring - 8
    pub global_ring: usize,
    /// rows of one target pass at most (up to 65,535: a grid's y)
    pub max_rows: usize,
}

/// Device bytes by part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftBytes {
    /// the 5 layers' matrices and vectors
    pub layers: usize,
    /// the copy of the target's unembed
    pub unembed: usize,
    /// fold_w, fold_b, markov_w1, rnn_proj, and the model's few vectors
    pub markov: usize,
    /// every lane's rings: 4 local, the global one, f_rnn
    pub rings: usize,
    /// a pass's and a block's activations
    pub scratch: usize,
    /// the listed vocabulary's head (open_with): its unembed rows, fold, ids
    pub listed: usize,
}

impl DraftBytes {
    pub fn total(&self) -> usize {
        self.layers + self.unembed + self.markov + self.rings + self.scratch + self.listed
    }
}

/// The listed vocabulary's own head (open_with): the unembed's and the fold's rows of its ids,
/// ascending, so the lowest index on a tie is the lowest id as over every id.
struct ListHead {
    n: usize,
    /// the ids, i32 [n]: a pick's index to its id
    ids: DeviceBuffer<i32>,
    unembed: Lin,
    fold_w: Lin,
    fold_b: DeviceBuffer<u16>,
}

fn lin_bytes(n: usize, k: usize, fp8: bool) -> usize {
    if fp8 { n * k + (n / 32) * (k / 32) } else { 2 * n * k }
}

/// The drafter's device bytes for these options and a vocabulary of `vocab` ids (201,024), without
/// TIDELM_DRAFT_DUMP's extra buffer.
pub fn bytes_for(o: &DraftOpts, vocab: usize) -> DraftBytes {
    let lb = |n, k| lin_bytes(n, k, o.fp8);
    let layer = lb(QKV, D) + lb(D, QDIM) + lb(2 * INTER, D) + lb(D, INTER) + lb(2 * KDIM, D) + 2 * N_LEN;
    DraftBytes {
        layers: LAYERS * layer,
        unembed: lb(vocab, D),
        markov: lb(vocab, RANK) + 2 * vocab + 2 * vocab * RANK + lb(3 * RANK, 2 * RANK) + lb(3 * RANK, 2 * D) + 2 * 3 * RANK
            + 2 * (LAYERS + 1) * TAPS + 2 * 3 * D + 4 * (HD / 2),
        rings: o.lanes * 2 * ((LAYERS - 1) * o.local_ring * ROW + o.global_ring * ROW + F_RING * D),
        scratch: Act::bytes(o.max_rows, vocab, false),
        listed: 0,
    }
}

/// Layer l's first context position for a block at `p`: the 1,024 positions before the block
/// (local layers) or what the global ring holds beside the block's 8 rows, never below `lo`.
fn context_lo(l: usize, p: usize, lo: usize, global_ring: usize) -> usize {
    let reach = if l == GLOBAL_LAYER { global_ring - BLOCK } else { WINDOW };
    lo.max(p.saturating_sub(reach))
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

/// alpha = bf16(softmax(fuse_weight, taps)) a row, fp32 inside (_fuse_alpha: SGLang rounds it).
fn fuse_alpha(w: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(w.len());
    for row in w.chunks_exact(TAPS) {
        let x: Vec<f32> = row.iter().map(|&h| bf(h)).collect();
        let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        out.extend(e.iter().map(|&v| to_bf(v / s)));
    }
    out
}

/// RoPE's inverse frequencies as the reference's table has them: 1 / theta^(2i / 128) in fp32.
fn inv_freq() -> Vec<f32> {
    (0..HD / 2).map(|i| 1.0f32 / THETA.powf((2 * i) as f32 / HD as f32)).collect()
}

// ---------------------------------------------------------------------- the checkpoint
/// The drafter's tensors (spec section 7): 15 a layer and 9 more, name and shape.
fn tensor_table() -> Vec<(String, Vec<usize>)> {
    let mut t: Vec<(String, Vec<usize>)> = vec![];
    let mut add = |n: String, s: &[usize]| t.push((n, s.to_vec()));
    add("fuse_weight".into(), &[LAYERS + 1, TAPS]);
    add("mask_embed".into(), &[D]);
    add("norm.weight".into(), &[D]);
    for l in 0..LAYERS {
        let p = format!("layers.{l}.");
        let a = format!("{p}self_attn.");
        add(format!("{p}input_layernorm.weight"), &[D]);
        add(format!("{p}post_attention_layernorm.weight"), &[D]);
        add(format!("{p}dflare_norm.weight"), &[D]);
        add(format!("{a}q_proj.weight"), &[QDIM, D]);
        add(format!("{a}k_proj.weight"), &[KDIM, D]);
        add(format!("{a}v_proj.weight"), &[KDIM, D]);
        add(format!("{a}o_proj.weight"), &[D, QDIM]);
        add(format!("{a}q_norm.weight"), &[HD]);
        add(format!("{a}k_norm.weight"), &[HD]);
        add(format!("{a}k_proj_ctx.weight"), &[KDIM, D]);
        add(format!("{a}v_proj_ctx.weight"), &[KDIM, D]);
        add(format!("{a}k_norm_ctx.weight"), &[HD]);
        add(format!("{p}mlp.gate_proj.weight"), &[INTER, D]);
        add(format!("{p}mlp.up_proj.weight"), &[INTER, D]);
        add(format!("{p}mlp.down_proj.weight"), &[D, INTER]);
    }
    add("markov_head.rnn_proj.weight".into(), &[3 * RANK, 2 * RANK + 2 * D]);
    add("markov_head.rnn_proj.bias".into(), &[3 * RANK]);
    add("markov_head.hidden_out.weight".into(), &[D, RANK]);
    add("markov_head.hidden_out.bias".into(), &[D]);
    add("markov_head.dflare_norm.weight".into(), &[D]);
    t
}

/// What config.json fixes that the tensors' shapes do not carry, against what this file and
/// inkling_draft.cu hard-code: a drafter with other values would load and draft tokens that are
/// wrong with nothing in the log. Returns (padded vocabulary, real vocabulary, logit softcap).
pub fn check_config(model_dir: &Path) -> Result<(usize, usize, f32)> {
    let path = model_dir.join("config.json");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)?;
    let (tc, mc) = (&v["text_config"], &v["mtp_config"]);
    if !tc.is_object() || !mc.is_object() {
        return Err(format!("{}: no text_config / mtp_config (the DSpark drafter is Inkling 2's)", path.display()).into());
    }
    let bad = |what: &str, got: &serde_json::Value| -> crate::Error {
        format!("{}: {what} is {got}, which inkling_draft.rs does not implement", path.display()).into()
    };
    let ints: [(&str, usize); 9] = [
        ("hidden_size", D),
        ("num_hidden_layers", LAYERS),
        ("num_attention_heads", HQ),
        ("num_key_value_heads", HK),
        ("head_dim", HD),
        ("intermediate_size", INTER),
        ("sliding_window_size", WINDOW),
        ("block_size", BLOCK),
        ("markov_rank", RANK),
    ];
    for (name, want) in ints {
        if mc[name].as_u64() != Some(want as u64) {
            return Err(bad(&format!("mtp_config.{name}"), &mc[name]));
        }
    }
    for (name, want) in [("rms_norm_eps", EPS as f64), ("rope_theta", THETA as f64)] {
        if !mc[name].as_f64().is_some_and(|g| (g - want).abs() <= 1e-6 * want) {
            return Err(bad(&format!("mtp_config.{name}"), &mc[name]));
        }
    }
    if mc["markov_head_type"].as_str() != Some("rnn_hidden") {
        return Err(bad("mtp_config.markov_head_type", &mc["markov_head_type"]));
    }
    let ids = |x: &serde_json::Value| -> Option<Vec<u64>> { x.as_array()?.iter().map(|i| i.as_u64()).collect() };
    let local: Vec<u64> = (0..LAYERS).filter(|&l| l != GLOBAL_LAYER).map(|l| l as u64).collect();
    if ids(&mc["local_layer_ids"]) != Some(local) {
        return Err(bad("mtp_config.local_layer_ids", &mc["local_layer_ids"]));
    }
    if ids(&mc["target_layer_ids"]) != Some(TARGET_LAYER_IDS.to_vec()) {
        return Err(bad("mtp_config.target_layer_ids", &mc["target_layer_ids"]));
    }
    for name in ["q_bias", "o_bias"] {
        if mc[name].as_bool() == Some(true) {
            return Err(bad(&format!("mtp_config.{name}"), &mc[name]));
        }
    }
    if tc["hidden_size"].as_u64() != Some(D as u64) {
        return Err(bad("text_config.hidden_size", &tc["hidden_size"]));
    }
    let vocab = tc["vocab_size"].as_u64().ok_or_else(|| bad("text_config.vocab_size", &tc["vocab_size"]))? as usize;
    let real = tc["unpadded_vocab_size"].as_u64().ok_or_else(|| bad("text_config.unpadded_vocab_size", &tc["unpadded_vocab_size"]))?
        as usize;
    let cap = tc["final_logit_softcapping"].as_f64().unwrap_or(0.0) as f32;
    if vocab % 32 != 0 || real == 0 || real > vocab || vocab > i32::MAX as usize {
        return Err(bad("text_config.vocab_size / unpadded_vocab_size", &tc["vocab_size"]));
    }
    if !(cap > 0.0) {
        return Err(bad("text_config.final_logit_softcapping", &tc["final_logit_softcapping"]));
    }
    Ok((vocab, real, cap))
}

/// A BF16 tensor [n][k] of an open safetensors file.
struct Tens<'a> {
    sh: &'a Shard,
    name: String,
    n: usize,
    k: usize,
}

impl<'a> Tens<'a> {
    fn new(sh: &'a Shard, name: &str, shape: &[usize]) -> Result<Self> {
        let m = sh.meta(name).map_err(|e| format!("{name}: {e}"))?;
        if m.dtype != "BF16" || m.shape != shape || m.nbytes() != 2 * shape.iter().product::<usize>() {
            return Err(format!("{}: {name} is {} {:?} (BF16 {shape:?} expected)", sh.path.display(), m.dtype, m.shape).into());
        }
        Ok(Tens { sh, name: name.to_string(), n: shape[0], k: shape[1..].iter().product() })
    }

    /// Bytes [lo, hi) of the tensor (O_DIRECT, so nothing lands in the page cache): the aligned
    /// read and where `lo` is in it.
    fn read(&self, lo: usize, hi: usize) -> Result<(AlignedBuf, usize)> {
        let m = self.sh.meta(&self.name)?;
        let (a, b) = (m.lo + lo as u64, m.lo + hi as u64);
        if lo > hi || b > m.hi {
            return Err(format!("{}: bytes {lo}..{hi} of {}", self.name, m.nbytes()).into());
        }
        let alo = a - a % ALIGN as u64;
        let mut buf = AlignedBuf::new(aligned_len(a, b))?;
        self.sh.file.pread_exact(&mut buf, alo, (b - alo) as usize).map_err(|e| format!("{}: {e}", self.name))?;
        Ok((buf, (a - alo) as usize))
    }

    /// The whole tensor (small ones).
    fn all(&self) -> Result<Vec<u16>> {
        let bytes = 2 * self.n * self.k;
        let (buf, off) = self.read(0, bytes)?;
        Ok(u16s(&buf[off..off + bytes]))
    }

    /// The tensor's rows in pieces of about CHUNK_BYTES, whole 32-row blocks each:
    /// f(piece's bytes, its first row, its rows).
    fn pieces(&self, mut f: impl FnMut(&[u8], usize, usize) -> Result<()>) -> Result<()> {
        let rows = (CHUNK_BYTES / (2 * self.k) / 32 * 32).max(32);
        let mut r0 = 0;
        while r0 < self.n {
            let m = rows.min(self.n - r0);
            let (buf, off) = self.read(r0 * self.k * 2, (r0 + m) * self.k * 2)?;
            f(&buf[off..off + m * self.k * 2], r0, m)?;
            r0 += m;
        }
        Ok(())
    }
}

/// Every tensor of the drafter's file is one this module reads, with its shape, and none is
/// missing (the reference's loader rejects both); the target's unembed is [vocab][4608].
fn check_tensors(mtp: &Shard, unembed: &Shard, vocab: usize) -> Result<()> {
    let mut table = tensor_table();
    table.push(("markov_head.markov_w1.weight".into(), vec![vocab, RANK]));
    for (name, shape) in &table {
        Tens::new(mtp, name, shape)?;
    }
    if let Some(extra) = mtp.names().find(|n| !table.iter().any(|(t, _)| t == n)) {
        return Err(format!("{}: {extra} is not a tensor of the drafter inkling_draft.rs implements", mtp.path.display()).into());
    }
    Tens::new(unembed, UNEMBED, &[vocab, D])?;
    Ok(())
}

/// The shard that holds the target's `tensor`, opened for this load alone.
fn target_shard(model_dir: &Path, tensor: &str) -> Result<Shard> {
    let index = WeightIndex::load(model_dir).map_err(|e| format!("{}: model.safetensors.index.json: {e}", model_dir.display()))?;
    let file = index.weight_map.get(tensor).ok_or_else(|| format!("{tensor} not in the weight index"))?;
    Ok(Shard::open(&model_dir.join(file)).map_err(|e| format!("{file}: {e}"))?)
}

/// The target's normed embedding row of `id` (Inkling2NormedEmbedding: the row, then RMSNorm with
/// embed_norm, fp32, one rounding - inkling.rs's table holds exactly this), for a check that has
/// no engine to take the table from.
pub fn normed_embedding_row(model_dir: &Path, id: usize) -> Result<Vec<u16>> {
    let (vocab, _, _) = check_config(model_dir)?;
    if id >= vocab {
        return Err(format!("token {id} of {vocab}").into());
    }
    let (es, ns) = (target_shard(model_dir, "model.llm.embed.weight")?, target_shard(model_dir, "model.llm.embed_norm.weight")?);
    let e = Tens::new(&es, "model.llm.embed.weight", &[vocab, D])?;
    let (buf, off) = e.read(id * D * 2, (id + 1) * D * 2)?;
    let x: Vec<f32> = u16s(&buf[off..off + 2 * D]).into_iter().map(bf).collect();
    let w = Tens::new(&ns, "model.llm.embed_norm.weight", &[D])?.all()?;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rs = 1.0 / (ss / D as f32 + EPS).sqrt();
    Ok(x.iter().zip(&w).map(|(&xf, &wh)| to_bf(bf(wh) * (xf * rs))).collect())
}

fn put_bytes(st: &Stream, dst: *mut c_void, b: &[u8]) -> Result<()> {
    unsafe {
        tidelm_cuda::memcpy_async(dst, b.as_ptr() as *const c_void, b.len(), tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, st)?;
    }
    st.synchronize()?;
    Ok(())
}

/// A small host matrix as a Lin.
fn lin_from_host(st: &Stream, w: &[u16], n: usize, k: usize, fp8: bool) -> Result<Lin> {
    if fp8 {
        Ok(Lin::Fp8(lin::fp8_from_bf16(st, w, n, k, 1)?))
    } else {
        Ok(Lin::Bf16(Bf16Mat { w: DeviceBuffer::from_slice(w)?, n, k, split: 1 }))
    }
}

/// A device BF16 [n][k] as a Lin: itself, or quantized there (the BF16 freed).
fn lin_from_device(st: &Stream, w: DeviceBuffer<u16>, n: usize, k: usize, fp8: bool) -> Result<Lin> {
    if !fp8 {
        return Ok(Lin::Bf16(Bf16Mat { w, n, k, split: 1 }));
    }
    let codes = DeviceBuffer::<u8>::new(n * k)?;
    let scales = DeviceBuffer::<u8>::new((n / 32) * (k / 32))?;
    ck(unsafe { ds_quant_fp8_block32(st.raw(), w.raw(), n as i32, k as i32, codes.raw(), scales.raw()) }, "quant fp8")?;
    st.synchronize()?;
    Ok(Lin::Fp8(Fp8Mat { w: codes, s: scales, n, k, split: 1 }))
}

/// Tensors [n_i][k] stacked by rows into one matrix on the device - FP8 with 32 x 32 block scales
/// or BF16 - a piece at a time: the host holds one piece, the device (FP8) one more as BF16.
/// Every n_i is a multiple of 32 (an FP8 scale block never straddles two tensors; the drafter's
/// all are, so nothing is padded). `each` sees every piece on the device as BF16 [rows][k] with
/// its first row in the stacked matrix, before it is quantized.
fn stack(st: &Stream, parts: &[Tens], fp8: bool, mut each: Option<&mut dyn FnMut(*const c_void, usize, usize) -> Result<()>>)
         -> Result<Lin> {
    let k = parts[0].k;
    let n: usize = parts.iter().map(|p| p.n).sum();
    if let Some(p) = parts.iter().find(|p| p.k != k || p.n % 32 != 0) {
        return Err(format!("{}: {} x {} in a stacked matrix of K {k} (rows in 32s)", p.name, p.n, p.k).into());
    }
    if k % 64 != 0 {
        return Err(format!("{}: K {k} (the GEMMs take multiples of 64)", parts[0].name).into());
    }
    let piece_rows = (CHUNK_BYTES / (2 * k) / 32 * 32).max(32).min(n);
    let (codes, scales, stage, whole) = if fp8 {
        (Some(DeviceBuffer::<u8>::new(n * k)?), Some(DeviceBuffer::<u8>::new((n / 32) * (k / 32))?),
         Some(DeviceBuffer::<u16>::new(piece_rows * k)?), None)
    } else {
        (None, None, None, Some(DeviceBuffer::<u16>::new(n * k)?))
    };
    let mut base = 0;
    for p in parts {
        p.pieces(|bytes, r0, m| {
            let r = base + r0;
            let dst = match (&stage, &whole) {
                (Some(s), _) => s.raw(),
                (_, Some(w)) => w.ptr_at(r * k),
                _ => unreachable!(),
            };
            put_bytes(st, dst, bytes)?;
            if let Some(f) = each.as_deref_mut() {
                f(dst, r, m)?;
            }
            if let (Some(c), Some(s)) = (&codes, &scales) {
                ck(unsafe { ds_quant_fp8_block32(st.raw(), dst, m as i32, k as i32, c.ptr_at(r * k), s.ptr_at((r / 32) * (k / 32))) },
                   "quant fp8")?;
                st.synchronize()?;
            }
            Ok(())
        })?;
        base += p.n;
    }
    Ok(match (codes, scales, whole) {
        (Some(w), Some(s), _) => Lin::Fp8(Fp8Mat { w, s, n, k, split: 1 }),
        (_, _, Some(w)) => Lin::Bf16(Bf16Mat { w, n, k, split: 1 }),
        _ => unreachable!(),
    })
}

// ---------------------------------------------------------------------- the module
struct Layer {
    /// [q_proj | k_proj | v_proj] [8192][4608]
    qkv: Lin,
    /// o_proj [4608][6144]
    o: Lin,
    /// [gate_proj | up_proj] [36864][4608]
    gu: Lin,
    /// down_proj [4608][18432]
    down: Lin,
    /// [k_proj_ctx | v_proj_ctx] [2048][4608]
    ctx: Lin,
    /// the layer's vectors (N_*)
    norms: DeviceBuffer<u16>,
}

impl Layer {
    fn bytes(&self) -> usize {
        self.qkv.bytes() + self.o.bytes() + self.gu.bytes() + self.down.bytes() + self.ctx.bytes() + 2 * self.norms.len()
    }
}

/// Activations: a pass's (store_context) and a block's (draft).
struct Act {
    /// a layer's mixed features [max_rows][4608] and their [k | v] projection [max_rows][2048]
    c: DeviceBuffer<u16>,
    kv: DeviceBuffer<u16>,
    /// the residual and the normed rows [8][4608]
    res: DeviceBuffer<u16>,
    xn: DeviceBuffer<u16>,
    /// [8][q | k | v], the queries as the attention reads them [8][48][128]
    qkv: DeviceBuffer<u16>,
    qc: DeviceBuffer<u16>,
    /// softmax partials [8][8][NS_GLOBAL][6] (pacc x 128)
    pm: DeviceBuffer<f32>,
    pl: DeviceBuffer<f32>,
    pacc: DeviceBuffer<f32>,
    /// the heads' outputs [8][6144], a sublayer's output [8][4608]
    o: DeviceBuffer<u16>,
    a: DeviceBuffer<u16>,
    /// [8][gate | up], the activation [8][18432]
    gu: DeviceBuffer<u16>,
    m: DeviceBuffer<u16>,
    /// base logits [7][vocab]
    base: DeviceBuffer<u16>,
    /// the head: [7][h_i | f_rnn], its product with rnn_proj's columns [7][768], a step's
    /// [state | w1[prev]] and its product [768], the state and the latent [256], latent . fold_w^T
    /// [vocab], the logits' partials [3][blocks], and (dump) a step's logits [vocab]
    zx: DeviceBuffer<u16>,
    pre: DeviceBuffer<f32>,
    zs: DeviceBuffer<u16>,
    rs: DeviceBuffer<f32>,
    state: DeviceBuffer<u16>,
    latent: DeviceBuffer<u16>,
    lw: DeviceBuffer<f32>,
    sp: DeviceBuffer<f32>,
    ylog: Option<DeviceBuffer<f32>>,
}

impl Act {
    /// (bf16 elements, fp32 elements) of the buffers, in new()'s order.
    fn sizes(max_rows: usize, vocab: usize) -> ([usize; 15], [usize; 7]) {
        let parts = BLOCK * HK * NS_GLOBAL * (HQ / HK);
        // a host function of the kernels' file: how many blocks its logits kernel makes of the ids
        let blocks = unsafe { inkd_step_blocks(vocab as i32) } as usize;
        ([max_rows * D, max_rows * 2 * KDIM, BLOCK * D, BLOCK * D, BLOCK * QKV, BLOCK * QDIM, BLOCK * QDIM, BLOCK * D,
          BLOCK * 2 * INTER, BLOCK * INTER, DRAFTS * vocab, DRAFTS * 2 * D, 2 * RANK, RANK, RANK],
         [parts, parts, parts * HD, DRAFTS * 3 * RANK, 3 * RANK, vocab, 3 * blocks])
    }

    fn bytes(max_rows: usize, vocab: usize, dump: bool) -> usize {
        let (h, f) = Self::sizes(max_rows, vocab);
        2 * h.iter().sum::<usize>() + 4 * (f.iter().sum::<usize>() + PART + if dump { vocab } else { 0 })
    }

    fn new(max_rows: usize, vocab: usize, dump: bool) -> Result<Self> {
        let (h, f) = Self::sizes(max_rows, vocab);
        let z = |n: usize| DeviceBuffer::<u16>::zeroed(n);
        let zf = |n: usize| DeviceBuffer::<f32>::zeroed(n);
        Ok(Act {
            c: z(h[0])?,
            kv: z(h[1])?,
            res: z(h[2])?,
            xn: z(h[3])?,
            qkv: z(h[4])?,
            qc: z(h[5])?,
            o: z(h[6])?,
            a: z(h[7])?,
            gu: z(h[8])?,
            m: z(h[9])?,
            base: z(h[10])?,
            zx: z(h[11])?,
            zs: z(h[12])?,
            state: z(h[13])?,
            latent: z(h[14])?,
            pm: zf(f[0])?,
            pl: zf(f[1])?,
            pacc: zf(f[2])?,
            pre: zf(f[3])?,
            rs: zf(f[4])?,
            lw: zf(f[5])?,
            sp: zf(f[6])?,
            ylog: if dump { Some(zf(vocab)?) } else { None },
        })
    }
}

pub struct InkDraft {
    opts: DraftOpts,
    vocab: usize,
    real: usize,
    cap: f32,
    layers: Vec<Layer>,
    /// bf16(softmax(fuse_weight)) [6][8]: a row a layer, the last the head's
    alpha: DeviceBuffer<u16>,
    /// the model's vectors (M_*)
    misc: DeviceBuffer<u16>,
    /// RoPE's inverse frequencies, fp32 [64]
    inv: DeviceBuffer<f32>,
    /// the target's unembed [vocab][4608]
    unembed: Lin,
    /// bf16(unembed . hidden_out.weight) [vocab][256] and bf16(unembed . hidden_out.bias) [vocab]
    fold_w: Lin,
    fold_b: DeviceBuffer<u16>,
    /// markov_w1 [vocab][256] (a row lookup)
    w1: DeviceBuffer<u16>,
    /// rnn_proj's columns [state | w1[prev]] ([768][512]) and [h_i | f_rnn] ([768][9216]); its bias
    rnn_s: Lin,
    rnn_x: Lin,
    rnn_b: DeviceBuffer<u16>,
    /// the local layers' rings [4][lanes][local_ring][8][256], layer 2's [lanes][global_ring][8][256],
    /// f_rnn [lanes][64][4608]
    lring: DeviceBuffer<u16>,
    gring: DeviceBuffer<u16>,
    frnn: DeviceBuffer<u16>,
    act: Act,
    /// split-K partials for store_context's GEMMs
    part: DeviceBuffer<f32>,
    bytes: DraftBytes,
    /// the listed vocabulary's head, when opened with one (draft's `listed`)
    list: Option<ListHead>,
    /// TIDELM_DRAFT_DUMP's prefix, and the passes store_context has dumped
    dump: Option<String>,
    passes: AtomicUsize,
}

/// Programmatic dependent launch on the drafter's own kernels (inkling_draft.cu), as the engines
/// set it for theirs: off until set.
pub fn set_pdl(on: bool) {
    unsafe { inkd_set_pdl(on as i32) };
}

impl InkDraft {
    /// model_dir holds config.json, the target's non-expert shards (for the unembed) and
    /// mtp.safetensors (read by path: it is not in the weight index).
    pub fn open(model_dir: &Path, st: &Stream, opts: DraftOpts) -> Result<Self> {
        Self::open_with(model_dir, st, opts, None)
    }

    /// open, with a listed vocabulary (ascending ids below the real vocabulary, a multiple of 32):
    /// a second head over only those ids, which draft's `listed` rounds read instead of every id's.
    /// The head's step is its unembed (7 rows) and fold_w (7 times): 0.93 + 0.36 GB over all
    /// 200,058 ids in FP8. A token outside the list cannot be drafted; the target verifies every
    /// token either way.
    pub fn open_with(model_dir: &Path, st: &Stream, opts: DraftOpts, listed: Option<&[u32]>) -> Result<Self> {
        let pow2 = |n: usize| n.is_power_of_two();
        if opts.lanes == 0 || opts.max_rows == 0 || !pow2(opts.local_ring) || !pow2(opts.global_ring)
            || opts.local_ring < WINDOW + opts.max_rows.max(BLOCK) || opts.global_ring < 2 * BLOCK
            || opts.local_ring > 1 << 30 || opts.global_ring > 1 << 30 || opts.max_rows > 65535 {
            return Err(format!("the drafter's rings: local {} (a power of two >= {WINDOW} + the pass's {} rows), global {} (a power \
                                of two >= {}), {} lanes", opts.local_ring, opts.max_rows, opts.global_ring, 2 * BLOCK, opts.lanes).into());
        }
        let (vocab, real, cap) = check_config(model_dir)?;
        let path = model_dir.join(MTP_FILE);
        let mtp = Shard::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ush = target_shard(model_dir, UNEMBED)?;
        check_tensors(&mtp, &ush, vocab)?;
        let dump = std::env::var("TIDELM_DRAFT_DUMP").ok().filter(|p| !p.is_empty());
        let fp8 = opts.fp8;
        let t = |name: &str, shape: &[usize]| Tens::new(&mtp, name, shape);
        let vecs = |names: &[String]| -> Result<DeviceBuffer<u16>> {
            let mut v = Vec::with_capacity(names.len() * D);
            for n in names {
                let m = mtp.meta(n)?;
                v.extend(t(n, &m.shape.clone())?.all()?);
            }
            Ok(DeviceBuffer::from_slice(&v)?)
        };

        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("layers.{l}.");
            let a = format!("{p}self_attn.");
            let w = |n: &str| format!("{a}{n}.weight");
            let norms = vecs(&[format!("{p}input_layernorm.weight"), format!("{p}post_attention_layernorm.weight"),
                               format!("{p}dflare_norm.weight"), w("q_norm"), w("k_norm"), w("k_norm_ctx")])?;
            debug_assert_eq!(norms.len(), N_LEN);
            layers.push(Layer {
                qkv: stack(st, &[t(&w("q_proj"), &[QDIM, D])?, t(&w("k_proj"), &[KDIM, D])?, t(&w("v_proj"), &[KDIM, D])?], fp8, None)?,
                o: stack(st, &[t(&w("o_proj"), &[D, QDIM])?], fp8, None)?,
                gu: stack(st, &[t(&format!("{p}mlp.gate_proj.weight"), &[INTER, D])?, t(&format!("{p}mlp.up_proj.weight"), &[INTER, D])?],
                          fp8, None)?,
                down: stack(st, &[t(&format!("{p}mlp.down_proj.weight"), &[D, INTER])?], fp8, None)?,
                ctx: stack(st, &[t(&w("k_proj_ctx"), &[KDIM, D])?, t(&w("v_proj_ctx"), &[KDIM, D])?], fp8, None)?,
                norms,
            });
        }
        let alpha_host = fuse_alpha(&t("fuse_weight", &[LAYERS + 1, TAPS])?.all()?);
        let alpha = DeviceBuffer::from_slice(&alpha_host)?;
        let misc = vecs(&["norm.weight".to_string(), "mask_embed".to_string(), "markov_head.dflare_norm.weight".to_string()])?;
        let inv = DeviceBuffer::from_slice(&inv_freq())?;

        // the head. rnn_proj [768][state 256 | w1 256 | h 4608 | f_rnn 4608]: its first 512 columns
        // take a step's input, the rest the block's (module doc)
        let (rnn_s, rnn_x) = {
            let (ks, kx) = (2 * RANK, 2 * D);
            let w = t("markov_head.rnn_proj.weight", &[3 * RANK, ks + kx])?.all()?;
            let (mut s, mut x) = (Vec::with_capacity(3 * RANK * ks), Vec::with_capacity(3 * RANK * kx));
            for row in w.chunks_exact(ks + kx) {
                s.extend_from_slice(&row[..ks]);
                x.extend_from_slice(&row[ks..]);
            }
            (lin_from_host(st, &s, 3 * RANK, ks, fp8)?, lin_from_host(st, &x, 3 * RANK, kx, fp8)?)
        };
        let rnn_b = DeviceBuffer::from_slice(&t("markov_head.rnn_proj.bias", &[3 * RANK])?.all()?)?;
        let w1 = DeviceBuffer::<u16>::new(vocab * RANK)?;
        t("markov_head.markov_w1.weight", &[vocab, RANK])?.pieces(|bytes, r0, _| put_bytes(st, w1.ptr_at(r0 * RANK), bytes))?;

        // the unembed, and through its pieces the fold: fold_w = bf16(U hidden_out.W), fold_b =
        // bf16(U hidden_out.b), fp64 sums on the GPU, cast to bf16 as torch casts (through fp32)
        let how = DeviceBuffer::from_slice(&t("markov_head.hidden_out.weight", &[D, RANK])?.all()?)?;
        let hob = DeviceBuffer::from_slice(&t("markov_head.hidden_out.bias", &[D])?.all()?)?;
        let fw = DeviceBuffer::<u16>::new(vocab * RANK)?;
        let fold_b = DeviceBuffer::<u16>::new(vocab)?;
        let mut fold = |u: *const c_void, r: usize, m: usize| -> Result<()> {
            ck(unsafe { inkd_fold(st.raw(), u, m as i32, D as i32, RANK as i32, how.raw(), hob.raw(), fw.ptr_at(r * RANK),
                                  fold_b.ptr_at(r)) }, "fold")?;
            st.synchronize()?;
            Ok(())
        };
        let unembed = stack(st, &[Tens::new(&ush, UNEMBED, &[vocab, D])?], fp8, Some(&mut fold))?;
        if let Some(prefix) = &dump {
            // checks: alpha, fold_b and fold_w's first rows as folded (before any quantization)
            let rows = 4096.min(vocab);
            let mut head = vec![0u16; rows * RANK];
            fw.download_async(0, &mut head, st)?;
            st.synchronize()?;
            let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            std::fs::write(format!("{prefix}.alpha.bin"), le(&alpha_host))?;
            std::fs::write(format!("{prefix}.fold_w.bin"), le(&head))?;
            std::fs::write(format!("{prefix}.fold_b.bin"), le(&fold_b.download()?))?;
        }
        let fold_w = lin_from_device(st, fw, vocab, RANK, fp8)?;
        let list = match listed {
            None => None,
            Some(ids) => {
                let k = ids.len();
                if k == 0 || k % 32 != 0 || ids.windows(2).any(|w| w[0] >= w[1]) || ids.last().is_some_and(|&i| i as usize >= real) {
                    return Err(format!("the drafter's listed vocabulary: {k} ids, ascending, below {real}, a multiple of 32").into());
                }
                let mut host = vec![0u16; k * D];
                let mut j = 0usize;
                Tens::new(&ush, UNEMBED, &[vocab, D])?.pieces(|bytes, r0, m| {
                    while j < k && (ids[j] as usize) < r0 + m {
                        let r = ids[j] as usize - r0;
                        for (c, b) in bytes[r * D * 2..(r + 1) * D * 2].chunks_exact(2).enumerate() {
                            host[j * D + c] = u16::from_le_bytes([b[0], b[1]]);
                        }
                        j += 1;
                    }
                    Ok(())
                })?;
                let rows = DeviceBuffer::from_slice(&host)?;
                drop(host);
                let (fwl, fbl) = (DeviceBuffer::<u16>::new(k * RANK)?, DeviceBuffer::<u16>::new(k)?);
                ck(unsafe { inkd_fold(st.raw(), rows.raw(), k as i32, D as i32, RANK as i32, how.raw(), hob.raw(), fwl.raw(), fbl.raw()) },
                   "listed fold")?;
                st.synchronize()?;
                let ids_dev = DeviceBuffer::from_slice(&ids.iter().map(|&i| i as i32).collect::<Vec<i32>>())?;
                Some(ListHead { n: k, ids: ids_dev, unembed: lin_from_device(st, rows, k, D, fp8)?, fold_w: lin_from_device(st, fwl, k, RANK, fp8)?,
                                fold_b: fbl })
            }
        };

        let lring = DeviceBuffer::<u16>::zeroed((LAYERS - 1) * opts.lanes * opts.local_ring * ROW)?;
        let gring = DeviceBuffer::<u16>::zeroed(opts.lanes * opts.global_ring * ROW)?;
        let frnn = DeviceBuffer::<u16>::zeroed(opts.lanes * F_RING * D)?;
        let act = Act::new(opts.max_rows, vocab, dump.is_some())?;
        let part = DeviceBuffer::<f32>::zeroed(PART)?;
        st.synchronize()?;

        let bytes = DraftBytes {
            layers: layers.iter().map(Layer::bytes).sum(),
            unembed: unembed.bytes(),
            markov: fold_w.bytes() + 2 * (fold_b.len() + w1.len() + rnn_b.len() + alpha.len() + misc.len()) + rnn_s.bytes()
                + rnn_x.bytes() + 4 * inv.len(),
            rings: 2 * (lring.len() + gring.len() + frnn.len()),
            scratch: Act::bytes(opts.max_rows, vocab, dump.is_some()),
            listed: list.as_ref().map_or(0, |l| l.unembed.bytes() + l.fold_w.bytes() + 2 * l.n + 4 * l.n),
        };
        debug_assert_eq!(DraftBytes { scratch: Act::bytes(opts.max_rows, vocab, false), listed: 0, ..bytes }, bytes_for(&opts, vocab));
        Ok(InkDraft { opts, vocab, real, cap, layers, alpha, misc, inv, unembed, fold_w, fold_b, w1, rnn_s, rnn_x, rnn_b, lring, gring,
                      frnn, act, part, bytes, list, dump, passes: AtomicUsize::new(0) })
    }

    /// Device bytes the drafter holds: weights, every lane's rings, scratch.
    pub fn device_bytes(&self) -> usize {
        self.bytes.total()
    }

    /// The same by part.
    pub fn bytes(&self) -> DraftBytes {
        self.bytes
    }

    /// Ids of the padded vocabulary (rows of the embedding the anchor indexes).
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Layer l's rings: (the first lane's first row, positions a lane).
    fn ring(&self, l: usize) -> (*mut c_void, usize) {
        if l == GLOBAL_LAYER {
            (self.gring.raw(), self.opts.global_ring)
        } else {
            (self.lring.ptr_at(LOCAL_OF[l] * self.opts.lanes * self.opts.local_ring * ROW), self.opts.local_ring)
        }
    }

    /// One lane's ring of layer l.
    fn lane_ring(&self, l: usize, lane: usize) -> (*mut c_void, usize) {
        let (base, ring) = self.ring(l);
        (unsafe { (base as *mut u16).add(lane * ring * ROW) } as *mut c_void, ring)
    }

    /// TIDELM_DRAFT_DUMP (checks only): `n` elements of T at `p` to PREFIX.NAME.bin.
    fn dump<T: Copy>(&self, st: &Stream, name: &str, p: *const c_void, n: usize) -> Result<()> {
        let Some(prefix) = &self.dump else { return Ok(()) };
        st.synchronize()?;
        let mut v = vec![0u8; n * std::mem::size_of::<T>()];
        ck(unsafe { tidelm_cuda::ffi::cudaMemcpy(v.as_mut_ptr() as *mut c_void, p, v.len(), tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST) },
           "draft dump")?;
        std::fs::write(format!("{prefix}.{name}.bin"), v)?;
        Ok(())
    }

    /// Checks only (a host readback): layer l's ring rows of `lane` for positions [a, b), bf16
    /// [b - a][8][K 128 | V 128] as bytes.
    pub fn ring_rows(&self, st: &Stream, l: usize, lane: usize, a: usize, b: usize) -> Result<Vec<u8>> {
        if l >= LAYERS || lane >= self.opts.lanes || a > b {
            return Err(format!("ring rows {a}..{b} of layer {l}, lane {lane}").into());
        }
        st.synchronize()?;
        let (base, ring) = self.lane_ring(l, lane);
        let mut v = vec![0u8; (b - a) * ROW * 2];
        for (i, row) in v.chunks_exact_mut(ROW * 2).enumerate() {
            let src = unsafe { (base as *const u16).add(((a + i) & (ring - 1)) * ROW) } as *const c_void;
            ck(unsafe { tidelm_cuda::ffi::cudaMemcpy(row.as_mut_ptr() as *mut c_void, src, ROW * 2, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST) },
               "ring rows")?;
        }
        Ok(v)
    }

    /// After a target pass over `t` rows: its taps -> every drafter layer's context K/V at the rows'
    /// positions, and f_rnn rows. `taps`: device bf16, 8 planes tap-major in target_layer_ids order
    /// [5, 11, 17, 23, 29, 35, 41, 48], plane p at `taps + p * plane` elements, each [t][4608]
    /// row-major. The rows' lanes and positions come from the pass's device tables (`lanes` [nl][4]:
    /// first row, rows, first position, cache-row offset; `meta` [nl][4]: the lane first), so this
    /// captures into a CUDA graph: no host readback, and grids that depend on t alone. A row of no
    /// lane group is skipped; so is one a whole ring before its group's last row.
    #[allow(clippy::too_many_arguments)]
    pub fn store_context(&self, st: &Stream, taps: *const c_void, plane: usize, t: usize, lanes: *const c_void, nl: usize,
                         meta: *const c_void) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        if t > self.opts.max_rows || plane < t * D || nl == 0 || taps.is_null() || lanes.is_null() || meta.is_null() {
            return Err(format!("drafter context: {t} rows (scratch for {}), planes {plane} elements apart, {nl} lane groups",
                               self.opts.max_rows).into());
        }
        let s = st.raw();
        let a = &self.act;
        let (ti, nli, di) = (t as i32, nl as i32, D as i32);
        let pass = if self.dump.is_some() { self.passes.fetch_add(1, Ordering::Relaxed) } else { 0 };
        for (l, ly) in self.layers.iter().enumerate() {
            ck(unsafe { inkd_mix(s, taps, plane as i64, self.alpha.ptr_at(l * TAPS), ly.norms.ptr_at(N_DFLARE), a.c.raw(), 0, lanes, nli,
                                 meta, ti, di, EPS) }, "context mix")?;
            self.dump::<u16>(st, &format!("c{pass}.{l}"), a.c.raw(), t * D)?;
            lin::gemm(st, &self.part, a.c.raw(), D, &ly.ctx, t, a.kv.raw(), false, 2 * KDIM)?;
            let (cache, ring) = self.ring(l);
            ck(unsafe { inkd_ctx_kv(s, a.kv.raw(), (2 * KDIM) as i64, ly.norms.ptr_at(N_KCTX), self.inv.raw(), EPS, lanes, nli, meta,
                                    ring as i32, cache, ti) }, "context keys")?;
        }
        // f_rnn: the head's mix of the same rows, each to its lane's slot of its position
        let (al, w) = (self.alpha.ptr_at(LAYERS * TAPS), self.misc.ptr_at(M_DFLARE));
        ck(unsafe { inkd_mix(s, taps, plane as i64, al, w, self.frnn.raw(), 1, lanes, nli, meta, ti, di, EPS) }, "f_rnn")?;
        if self.dump.is_some() {
            ck(unsafe { inkd_mix(s, taps, plane as i64, al, w, a.c.raw(), 0, lanes, nli, meta, ti, di, EPS) }, "f_rnn (dump)")?;
            self.dump::<u16>(st, &format!("c{pass}.{LAYERS}"), a.c.raw(), t * D)?;
        }
        Ok(())
    }

    /// One draft round for `lane`: the anchor token at absolute position `p`; context keys are
    /// positions [max(lo, p - 1024), p) for the local layers and [max(lo, p - global_ring + 8), p)
    /// for layer 2 (`lo`: the caller's floor of valid context), plus the block's own 8 rows; f_rnn
    /// is position p - 1's (zeros when p == lo: no context at all). `embed`: the target's NORMED
    /// embedding table, device bf16 [vocab][4608] (a row lookup). Writes the 7 draft tokens
    /// (`toks`: device i32 [7]; greedy - the argmax, the lowest id on ties) and each one's
    /// probability under softmax of the corrected, softcapped, cropped logits at temperature 1
    /// (`probs`: device fp32 [7]). The 7 steps chain on the device (a step reads the token the one
    /// before wrote to `toks`); nothing is read back, and no grid depends on anchor, p or lo.
    /// `listed`: the steps over the listed vocabulary's head (open_with; ignored without one).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(&self, st: &Stream, lane: usize, anchor: u32, p: usize, lo: usize, embed: *const c_void, toks: *mut c_void,
                 probs: *mut c_void, listed: bool) -> Result<()> {
        if lane >= self.opts.lanes || anchor as usize >= self.vocab || lo > p || p > (i32::MAX as usize) - BLOCK || embed.is_null()
            || toks.is_null() || probs.is_null() {
            return Err(format!("draft round: lane {lane} of {}, anchor {anchor} of {}, position {p}, context from {lo}",
                               self.opts.lanes, self.vocab).into());
        }
        let s = st.raw();
        let a = &self.act;
        let (bl, di, vi) = (BLOCK as i32, D as i32, self.vocab as i32);
        let null = std::ptr::null();
        unsafe {
            ck(inkd_embed(s, embed, anchor as i32, self.misc.ptr_at(M_MASK), a.res.raw(), di), "draft embed")?;
            ck(inkd_norm(s, a.res.raw(), null, self.layers[0].norms.ptr_at(N_IN), a.xn.raw(), bl, di, EPS), "draft norm")?;
        }
        self.dump::<u16>(st, "x0", a.res.raw(), BLOCK * D)?;
        for (l, ly) in self.layers.iter().enumerate() {
            // attention: one softmax over [the layer's context | the block's own keys]
            lin::gemm(st, &self.part, a.xn.raw(), D, &ly.qkv, BLOCK, a.qkv.raw(), false, QKV)?;
            let (cache, ring) = self.lane_ring(l, lane);
            let klo = context_lo(l, p, lo, self.opts.global_ring);
            let ns = if l == GLOBAL_LAYER { NS_GLOBAL } else { NS_LOCAL } as i32;
            unsafe {
                ck(inkd_heads(s, a.qkv.raw(), QKV as i64, ly.norms.ptr_at(N_Q), ly.norms.ptr_at(N_K), self.inv.raw(), EPS, p as i32,
                              ring as i32, cache, a.qc.raw()), "draft heads")?;
                ck(inkd_attn_part(s, a.qc.raw(), cache, ring as i32, klo as i32, (p + BLOCK) as i32, ns, a.pm.raw(), a.pl.raw(),
                                  a.pacc.raw()), "draft attention")?;
                ck(inkd_attn_combine(s, a.pm.raw(), a.pl.raw(), a.pacc.raw(), ns, a.o.raw()), "draft attention combine")?;
            }
            if self.dump.is_some() {
                self.dump::<u16>(st, &format!("q{l}"), a.qc.raw(), BLOCK * QDIM)?;
                std::fs::write(format!("{}.kv{l}.bin", self.dump.as_ref().unwrap()), self.ring_rows(st, l, lane, klo, p + BLOCK)?)?;
                self.dump::<u16>(st, &format!("o{l}"), a.o.raw(), BLOCK * QDIM)?;
            }
            lin::gemm(st, &self.part, a.o.raw(), QDIM, &ly.o, BLOCK, a.a.raw(), false, D)?;
            ck(unsafe { inkd_norm(s, a.res.raw(), a.a.raw(), ly.norms.ptr_at(N_POST), a.xn.raw(), bl, di, EPS) }, "draft norm")?;
            // the MLP: SwiGLU, no caps
            lin::gemm(st, &self.part, a.xn.raw(), D, &ly.gu, BLOCK, a.gu.raw(), false, 2 * INTER)?;
            ck(unsafe { glm_swiglu(s, a.gu.raw(), (2 * INTER) as i64, a.gu.ptr_at(INTER), (2 * INTER) as i64, a.m.raw(), INTER as i64, bl,
                                   INTER as i32) }, "draft swiglu")?;
            lin::gemm(st, &self.part, a.m.raw(), INTER, &ly.down, BLOCK, a.a.raw(), false, D)?;
            let next = if l + 1 < LAYERS { self.layers[l + 1].norms.ptr_at(N_IN) } else { self.misc.ptr_at(M_NORM) };
            ck(unsafe { inkd_norm(s, a.res.raw(), a.a.raw(), next, a.xn.raw(), bl, di, EPS) }, "draft norm")?;
            self.dump::<u16>(st, &format!("res{l}"), a.res.raw(), BLOCK * D)?;
        }
        self.dump::<u16>(st, "hidden", a.xn.raw(), BLOCK * D)?;
        // the head's vocabulary: every id, or the listed ones (their own unembed and fold rows)
        let lh = self.list.as_ref().filter(|_| listed);
        let (unembed, fold_w, fold_b, n, real, ids) = match lh {
            Some(l) => (&l.unembed, &l.fold_w, &l.fold_b, l.n, l.n, l.ids.raw() as *const c_void),
            None => (&self.unembed, &self.fold_w, &self.fold_b, self.vocab, self.real, null),
        };
        // base logits of rows 1 .. 7 (the anchor's own row predicts nothing)
        lin::gemm(st, &self.part, a.xn.ptr_at(D), D, unembed, DRAFTS, a.base.raw(), false, n)?;
        self.dump::<u16>(st, "base", a.base.raw(), DRAFTS * n)?;
        // the head: rnn_proj's block-constant columns for the 7 steps at once
        let f: *const c_void = if p > lo { self.frnn.ptr_at((lane * F_RING + ((p - 1) & (F_RING - 1))) * D) } else { null };
        ck(unsafe { inkd_rnn_x(s, a.xn.raw(), f, a.zx.raw(), di) }, "draft head input")?;
        self.dump::<u16>(st, "zx", a.zx.raw(), DRAFTS * 2 * D)?;
        lin::gemm(st, &self.part, a.zx.raw(), 2 * D, &self.rnn_x, DRAFTS, a.pre.raw(), true, 3 * RANK)?;
        let ylog = a.ylog.as_ref().map_or(std::ptr::null_mut(), |b| b.raw());
        for i in 0..DRAFTS {
            let step = i as i32;
            ck(unsafe { inkd_rnn_in(s, a.state.raw(), self.w1.raw(), toks, step, anchor as i32, vi, a.zs.raw()) }, "draft step input")?;
            lin::gemm(st, &self.part, a.zs.raw(), 2 * RANK, &self.rnn_s, 1, a.rs.raw(), true, 3 * RANK)?;
            ck(unsafe { inkd_rnn_cell(s, a.pre.ptr_at(i * 3 * RANK), a.rs.raw(), self.rnn_b.raw(), a.state.raw(), a.latent.raw(),
                                      (i == 0) as i32) }, "draft cell")?;
            lin::gemm(st, &self.part, a.latent.raw(), RANK, fold_w, 1, a.lw.raw(), true, n)?;
            unsafe {
                ck(inkd_step_part(s, a.base.ptr_at(i * n), a.lw.raw(), fold_b.raw(), n as i32, real as i32, self.cap, a.sp.raw(), ylog),
                   "draft logits")?;
                ck(inkd_step_pick(s, a.sp.raw(), n as i32, step, toks, probs, ids), "draft pick")?;
            }
            if self.dump.is_some() {
                self.dump::<u16>(st, &format!("state{i}"), a.state.raw(), RANK)?;
                self.dump::<u16>(st, &format!("latent{i}"), a.latent.raw(), RANK)?;
                self.dump::<f32>(st, &format!("y{i}"), ylog, n)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(to_bf(1.0), 0x3f80);
        assert_eq!(bf(to_bf(-2.5)), -2.5);
        assert_eq!(to_bf(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(to_bf(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn alpha_is_a_rounded_softmax() {
        // equal logits: 1 / 8 a tap, exact in bf16
        let a = fuse_alpha(&[to_bf(0.5); 2 * TAPS]);
        assert_eq!(a, vec![to_bf(0.125); 2 * TAPS]);
        // one tap far ahead takes (nearly) everything; the rows are independent
        let mut w = vec![to_bf(0.0); 2 * TAPS];
        w[3] = to_bf(40.0);
        let a = fuse_alpha(&w);
        assert_eq!(bf(a[3]), 1.0);
        assert!(a[..TAPS].iter().enumerate().all(|(i, &v)| i == 3 || bf(v) < 1e-12));
        assert_eq!(&a[TAPS..], &vec![to_bf(0.125); TAPS][..]);
    }

    #[test]
    fn rope_frequencies() {
        let f = inv_freq();
        assert_eq!((f.len(), f[0]), (HD / 2, 1.0));
        // the last: theta^(-126 / 128)
        assert!((f[63] as f64 / 1e6f64.powf(-126.0 / 128.0) - 1.0).abs() < 1e-6);
        assert!(f.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn context_ranges() {
        // local layers: the 1,024 positions before the block, whatever the global ring
        for l in [0, 1, 3, 4] {
            assert_eq!(context_lo(l, 5000, 0, 4096), 5000 - WINDOW);
            assert_eq!(context_lo(l, 40, 0, 4096), 0);
            assert_eq!(context_lo(l, 5000, 4500, 4096), 4500);
        }
        // layer 2: the ring's positions less the block's 8 rows
        assert_eq!(context_lo(2, 5000, 0, 4096), 5000 - 4088);
        assert_eq!(context_lo(2, 5000, 0, 16384), 0);
        assert_eq!(context_lo(2, 5000, 1000, 4096), 1000);
        // keys [lo, p + 8) never pass a ring
        assert_eq!(5000 + BLOCK - context_lo(2, 5000, 0, 4096), 4096);
        assert_eq!(5000 + BLOCK - context_lo(0, 5000, 0, 4096), WINDOW + BLOCK);
    }

    #[test]
    fn tensors_and_sizes() {
        // 84 tensors: 15 a layer and 9 more (markov_w1's shape comes from the config's vocabulary)
        let t = tensor_table();
        assert_eq!(t.len() + 1, 84);
        let vocab = 201024;
        let bytes: usize = t.iter().map(|(_, s)| 2 * s.iter().product::<usize>()).sum::<usize>() + 2 * vocab * RANK;
        // mtp.safetensors' data: 3,423,420,768 bytes
        assert_eq!(bytes, 3_423_420_768);
        // on the device as BF16: the file less hidden_out (folded away) and fuse_weight / alpha's
        // difference of nothing, plus the unembed copy, the fold and RoPE's frequencies
        let o = DraftOpts { fp8: false, lanes: 1, local_ring: 2048, global_ring: 4096, max_rows: 64 };
        let b = bytes_for(&o, vocab);
        assert_eq!(b.layers + b.markov, bytes - 2 * (D * RANK + D) + 2 * vocab * RANK + 2 * vocab + 4 * (HD / 2));
        assert_eq!(b.unembed, 2 * vocab * D);
        assert_eq!(b.rings, 2 * (4 * 2048 * ROW + 4096 * ROW + F_RING * D));
        // FP8: a byte a weight and a scale a 32 x 32 block
        let f = bytes_for(&DraftOpts { fp8: true, ..o }, vocab);
        assert_eq!(f.unembed, vocab * D + vocab * D / 1024);
        assert!(f.layers < b.layers / 2 + b.layers / 1000 && f.rings == b.rings && f.scratch == b.scratch);
    }

    /// TIDELM_INK_MODEL=DIR: the real checkpoint's config and headers against this file (no tensor
    /// is read, no CUDA context).
    #[test]
    fn real_checkpoint_headers() {
        let Ok(dir) = std::env::var("TIDELM_INK_MODEL") else { return };
        let dir = Path::new(&dir);
        let (vocab, real, cap) = check_config(dir).unwrap();
        assert_eq!((vocab, real, cap), (201024, 200058, 30.0));
        let mtp = Shard::open(&dir.join(MTP_FILE)).unwrap();
        let ush = target_shard(dir, UNEMBED).unwrap();
        check_tensors(&mtp, &ush, vocab).unwrap();
        assert_eq!(mtp.names().count(), 84);
    }
}
