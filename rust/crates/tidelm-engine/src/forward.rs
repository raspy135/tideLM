//! The DS v4.1 forward pass (engine/model.py::Model.forward + block + attention + moe,
//! tools/v41_ref.py), the DSpark drafter (Model.dspark_seed / dspark_draft) and cache
//! rollback (Caches.rollback), on the ds41.cu kernels and the native FP4 MoE kernels.
//!
//! Two weight sources share every kernel. *Streaming* reads each layer's weights from NVMe
//! per step, routes, loads only the experts the step's rows chose (through a 16-slot arena)
//! and frees the layer again: ~2 GB of device memory, the correctness vehicle that runs next
//! to a live server. *Resident* keeps all weights and routable experts on the device, routes
//! through a slot LUT, and with `graphs` issues each step as one CUDA graph.
//!
//! Semantics follow the checkpoint's reference forward (inference/model.py), which runs
//! every layer over every token. The Python engine's decoder replay (layers past the
//! candidate source over only the last 128 prompt tokens) is an approximation from the
//! tech report; it agrees exactly for prompts up to 128 tokens and is not used here.
//!
//! Batching: the engine holds `max_lanes` sequences (lanes), each with its own caches. A step
//! is a list of segments, one per lane, each that lane's next tokens; their rows are stacked
//! (at most TMAX = 64) and every kernel reads and writes the caches of its row's lane. A step
//! is padded to a row bucket with filler rows in a scratch lane that route to no expert, so a
//! few graphs serve every batch composition.
//!
//! - the indexer runs only where it can change something: an index-source layer whose
//!   compressed count (the step's largest) exceeds index_topk (1024). Below that, model.py's
//!   top-k selects every visible position, so the all-visible selection is the same set at
//!   no cost (and a short lane batched with a long one gets that same set from the indexer).
//! - every kernel is row-invariant, so a row computes the same bits whatever else shares
//!   its step, as long as the step has at most FMA_ROWS = 16 rows. Above that the GEMMs take
//!   the tensor-core kernel, which is row-invariant within itself but rounds differently from
//!   the FMA kernel. Speculation is therefore exact (a verify block computes each row as
//!   one-row steps would) whenever the steps involved stay at or below 16 rows.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::time::Instant;

use tidelm_core::engram_hash::{EngramLayout, NgramHashState};
use tidelm_cuda::{memcpy_async, DeviceBuffer, GraphExec, PinnedBuffer, Stream, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE};
use tidelm_kernels::ds41::*;
use tidelm_kernels::{engram as eg, fp4};
use tidelm_model::{Checkpoint, ExpertReader, STAGE_BYTES};

use crate::config::Config;
use crate::rope;
use crate::weights::{DsparkW, Fp8, Layer, Loader, Tp};
use crate::Result;

/// Rows per decode step, all lanes together (64 rows x top-6 = 384 routed pairs; the router's
/// wide path takes 512): the graphed steps, the head's rows, the drafter's lanes.
pub const TMAX: usize = 64;
/// Rows a prompt-only step may carry (TIDELM_PREFILL_ROWS; Python's DSV41_PREFILL_CHUNK=2048):
/// a step reads each weight its rows touch once, so a prompt of 8K tokens is 4 steps, not 128.
/// Such steps run eager, their GEMMs M-tiled and their pairs routed by ds_route_large.
pub fn prefill_rows() -> usize {
    std::env::var("TIDELM_PREFILL_ROWS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(2048).max(TMAX)
        .div_ceil(TMAX) * TMAX
}
/// The FMA GEMM's row limit; steps with more rows take the tensor-core kernel.
pub const FMA_ROWS: usize = 16;
/// Window-ring slots per lane and layer: a step writes its rows before attending, so the
/// ring holds a step's rows plus the window behind its first row - the prefill step's.
fn ring_slots(window: usize) -> usize {
    (window + prefill_rows()).div_ceil(64) * 64
}
const ARENA_SLOTS: usize = 16;
/// Indexer scores held at once (bf16, rows x keys): 512 MB. A step's rows are scored in slices
/// of INDEX_SCORES / keys rows, so the buffer does not scale with max_seq x rows (2.15 GB at
/// 524,288 tokens and 2,048 rows); a row's scores and top-k depend on no other row's.
const INDEX_SCORES: usize = 256 << 20;
/// INDEX_SCORES, or TIDELM_INDEX_SCORES (tests: a small budget slices a short context's steps).
fn index_scores_budget() -> usize {
    std::env::var("TIDELM_INDEX_SCORES").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n >= 1 << 16).unwrap_or(INDEX_SCORES)
}
/// A packed compressed-KV row: 512 FP4 codes (256 bytes) + 32 E4M3 scales (ds41.cu kPkRow).
const PACKED_ROW: usize = 288;
const HEAD_ROWS: usize = 8192;
const HC_SPLITS: usize = 16;
/// fp32 split-K partials of the tensor-core GEMM, splits x rows x outputs. The model's
/// largest request at 64 rows is ~0.66M (wo_b, main_proj: 2 x 64 x 5120); a request that
/// does not fit runs with fewer splits.
const MMA_PART: usize = 1 << 21;
/// The longest nucleus ds_nucleus returns whole (its sort's capacity); a target row whose
/// nucleus is longer is recomputed on the host from its logits.
const NUC_CAP: usize = 2048;
/// The drafter's nucleus is cut at this many tokens (fastdecode.py DSV41_DRAFT_NUCLEUS_K).
pub const DRAFT_NUCLEUS_K: usize = 512;

/// Temperature + nucleus parameters of a sequence (v41_engine.sample_probs); temperature
/// <= 0 is greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, top_p: 1.0 };

    pub fn is_greedy(&self) -> bool {
        !(self.temperature > 0.0)
    }
}

/// A row's sampling distribution: its nucleus by probability (descending, ties by id),
/// renormalized - one token at probability 1 for a greedy row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Nucleus {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

impl Nucleus {
    pub fn row(&self) -> tidelm_core::spec::SparseRow<'_> {
        tidelm_core::spec::SparseRow { ids: &self.ids, probs: &self.probs }
    }
}

/// What a step reads back from its head rows.
#[derive(Clone, Copy)]
enum Readback<'a> {
    Argmax,
    Logits,
    /// each segment's rows with its sampling parameters
    Nucleus(&'a [Sampling]),
}

pub(crate) fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        return Err(format!("{what}: CUDA error {rc}").into());
    }
    Ok(())
}

pub(crate) fn np<T: Copy>(b: &DeviceBuffer<T>) -> *mut c_void {
    b.raw()
}
pub(crate) fn nul() -> *mut c_void {
    std::ptr::null_mut()
}
fn off(p: *mut c_void, bytes: usize) -> *mut c_void {
    if p.is_null() {
        p
    } else {
        unsafe { (p as *mut u8).add(bytes) as *mut c_void }
    }
}

fn no_splitk() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TIDELM_NO_SPLITK").as_deref() == Ok("1"))
}

/// TIDELM_EXPERT_TOPP (unset, 0 or off: none) and TIDELM_EXPERT_TOPP_MIN_K (default 3), as
/// engine/expert_topp.py::config. Off by default: on the README protocol (pair, drafter split
/// and listed, same prompts and seed) 0.90 moved the medians by +1.6 / -1.0% on prose and -2.1 /
/// -3.4% on code (window / confidence), and code's acceptance fell (window 3.79 / 3.72 / 3.37
/// -> 3.45 / 3.61 / 3.29 tokens a step): the drafter predicts the unpruned target, so the
/// expert reads it saves come back as rejected drafts.
fn expert_topp_from_env() -> Result<Option<(f32, i32)>> {
    let raw = std::env::var("TIDELM_EXPERT_TOPP").unwrap_or_default().trim().to_lowercase();
    if matches!(raw.as_str(), "" | "0" | "off" | "none") {
        return Ok(None);
    }
    let topp: f32 = raw.parse().map_err(|_| format!("TIDELM_EXPERT_TOPP={raw}: a share in (0, 1), e.g. 0.90"))?;
    if !(topp > 0.0 && topp < 1.0) {
        return Err(format!("TIDELM_EXPERT_TOPP={raw}: a share in (0, 1), e.g. 0.90").into());
    }
    let min_k: i32 = std::env::var("TIDELM_EXPERT_TOPP_MIN_K").ok().map(|v| v.trim().parse()).transpose()
        .map_err(|_| "TIDELM_EXPERT_TOPP_MIN_K must be 1..6")?.unwrap_or(3);
    if !(1..=6).contains(&min_k) {
        return Err("TIDELM_EXPERT_TOPP_MIN_K must be 1..6".into());
    }
    Ok(Some((topp, min_k)))
}

/// A routing's blocks in act.block_*: `bm` pairs each, `nb` of them at most (the grid the MoE
/// kernels launch; blocks past the real ones are slot -1 and exit), and whether the prefill
/// kernels (64-pair grouped GEMM) take them.
#[derive(Clone, Copy)]
struct Blocks {
    bm: i32,
    nb: i32,
    prefill: bool,
}

/// Rows from which a prefill step's routed experts take the grouped-GEMM kernels: below, the
/// decode kernels' 16-pair blocks are ~5% faster (10.3 / 11.1 ms against 10.9 / 11.6 at 128 /
/// 256 rows), from 512 the grouped GEMM is (12.1 against 14.2 ms; 13.9 against 29.4 at 2,048;
/// examples/bench_moe_prefill.rs, a layer's 250 residents, this rank's shard).
const PREFILL_MOE_ROWS: usize = 512;

/// The routing blocks of `p` pairs (act.keys, slot values < `slots`, at most `distinct` of them
/// in one call: a layer's experts and the null slot) into act.block_*. A prefill step's
/// (`prefill`: eager, PREFILL_MOE_ROWS rows or more) go to the grouped-GEMM kernels in 64-pair
/// blocks; otherwise
/// the decode kernel's 16-pair blocks, by route_small up to its 512 pairs and the counting sort
/// past them. The decode kernels stay at 16-pair blocks: 32- and 64-pair ones decode each
/// expert's weights half / a quarter as often but ran slower (examples/bench_moe_prefill.rs,
/// 2,048 rows over 250 experts, this rank's shard: 31.8 / 36.5 / 44.9 ms up + down) - those
/// kernels are bound by their activation fragments' shared-memory reads, one per mma at any
/// block size, not by the decode; the grouped GEMM reuses each fragment over 2-4 mma (13.9 ms).
fn route(sr: *mut c_void, a: &Act, p: usize, k: usize, slots: usize, distinct: usize, prefill: bool) -> Result<Blocks> {
    if p <= 512 && !prefill {
        ck(unsafe {
            fp4::fp4_moe_cuda_route_small(sr, np(&a.keys), np(&a.block_slot), np(&a.block_pair), np(&a.block_route), p as i32, k as i32)
        }, "route")?;
        return Ok(Blocks { bm: 16, nb: p as i32, prefill: false });
    }
    if a.route_work.len() < 3 * slots {
        return Err(format!("route: {slots} slots outgrow the work buffer").into());
    }
    let bm = if prefill { 64 } else { 16 };
    // the buffers hold the bound over every slot; the grid, the bound over the slots one call
    // can reach
    let cap = unsafe { ds_route_blocks(p as i32, slots as i32, bm) } as usize;
    if a.block_slot.len() < cap || a.block_route.len() < cap * bm as usize {
        return Err(format!("route: {p} pairs in {bm}-pair blocks outgrow the block buffers").into());
    }
    ck(unsafe {
        ds_route_large(sr, np(&a.keys), p as i32, k as i32, slots as i32, np(&a.route_work), np(&a.block_slot), np(&a.block_pair),
                       np(&a.block_route), bm)
    }, "route (large)")?;
    Ok(Blocks { bm, nb: unsafe { ds_route_blocks(p as i32, slots.min(distinct) as i32, bm) }, prefill })
}

/// ds_gather_rows_u16 over any number of rows (one launch takes at most 65,535): out[m] =
/// table[ids[m]], rows of d bf16.
fn gather_rows(stream: &Stream, table: *mut c_void, ids: &DeviceBuffer<i32>, out: *mut c_void, n: usize, d: usize) -> Result<()> {
    for m0 in (0..n).step_by(65_535) {
        let m = 65_535.min(n - m0);
        ck(unsafe { ds_gather_rows_u16(stream.raw(), table, ids.ptr_at(m0), off(out, m0 * d * 2), m as i32, d as i32) }, "gather rows")?;
    }
    stream.synchronize()?;
    Ok(())
}

fn dump_bf16(path: &std::path::Path, v: &[u16]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|&h| f32::from_bits((h as u32) << 16).to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

/// The rows a step of `t` live rows is padded to: powers of two up to 16 (the FMA kernel's
/// own row tiles, so padding there costs no GEMM time), multiples of 8 above.
pub fn row_bucket(t: usize) -> usize {
    if t <= FMA_ROWS {
        t.max(1).next_power_of_two()
    } else {
        t.div_ceil(8) * 8
    }
}

/// A GEMM weight operand.
#[derive(Clone, Copy)]
pub(crate) enum Wt<'a> {
    Fp8(&'a Fp8),
    /// bf16 `[n][k]`
    Bf16 { w: *mut c_void, n: usize, k: usize },
}

/// y[m][n] = x[m][..k] W^T, bf16 out (f32 with `f32_out`). Up to FMA_ROWS rows the FMA
/// kernel (row-invariant across m, ~230 GB/s of weights); above, the tensor-core kernel
/// (row-invariant within its tile tier, ~165 GB/s at 32 rows and ~133 at 64, rounding
/// differently from the FMA kernel). Shapes it cannot take (K % 64) run the FMA kernel over
/// 16-row slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(sr: *mut c_void, part: &DeviceBuffer<f32>, x: *mut c_void, ldx: usize, w: Wt, m: usize, y: *mut c_void,
        f32_out: bool, ldy: usize) -> Result<()> {
    let (n, k) = match w {
        Wt::Fp8(f) => (f.n, f.k),
        Wt::Bf16 { n, k, .. } => (n, k),
    };
    let (yb, yf) = if f32_out { (nul(), y) } else { (y, nul()) };
    if m > FMA_ROWS && k % 64 == 0 {
        let splits = mma_splits(m, n, k, part.len());
        let pp = if splits > 1 { np(part) } else { nul() };
        ck(unsafe {
            match w {
                Wt::Fp8(f) => ds_mma_fp8(sr, x, ldx as i64, 0, 0, np(&f.w), np(&f.s), f.lds() as i64, m as i32, n as i32,
                                         k as i32, splits as i32, yb, yf, ldy as i64, pp),
                Wt::Bf16 { w, .. } => ds_mma_bf16(sr, x, ldx as i64, w, m as i32, n as i32, k as i32, splits as i32, yb, yf,
                                                  ldy as i64, pp),
            }
        }, "mma gemm")?;
        if splits > 1 {
            ck(unsafe { ds_splits_reduce(sr, pp, splits as i32, m as i32, n as i32, yb, yf, ldy as i64) }, "mma reduce")?;
        }
        return Ok(());
    }
    // the FMA kernel tiles rows past 16 in one launch, each row with its 16-row slice's bits
    let mi = m as i32;
    ck(unsafe {
        match w {
            Wt::Fp8(f) => ds_linear_fp8(sr, x, ldx as i64, np(&f.w), np(&f.s), f.lds() as i64, mi, n as i32, k as i32, 1, yb, yf,
                                        ldy as i64, nul()),
            Wt::Bf16 { w, .. } => ds_linear_bf16(sr, x, ldx as i64, w, mi, n as i32, k as i32, 1, yb, yf, ldy as i64, nul()),
        }
    }, "gemm")
}

/// The grouped wo_a GEMM (output rows of group g read x + g * gstride), dispatched as `gemm`.
#[allow(clippy::too_many_arguments)]
fn gemm_grouped(sr: *mut c_void, part: &DeviceBuffer<f32>, x: *mut c_void, ldx: usize, gstride: usize, rpg: usize,
                w: &Fp8, m: usize, y: *mut c_void, ldy: usize) -> Result<()> {
    let (n, k) = (w.n, w.k);
    // the tensor-core blocks must not straddle a group: rows per group a multiple of 128
    if m > FMA_ROWS && k % 64 == 0 && rpg % 128 == 0 {
        let splits = mma_splits(m, n, k, part.len());
        let pp = if splits > 1 { np(part) } else { nul() };
        ck(unsafe {
            ds_mma_fp8(sr, x, ldx as i64, gstride as i64, rpg as i32, np(&w.w), np(&w.s), w.lds() as i64, m as i32, n as i32,
                       k as i32, splits as i32, y, nul(), ldy as i64, pp)
        }, "mma grouped")?;
        if splits > 1 {
            ck(unsafe { ds_splits_reduce(sr, pp, splits as i32, m as i32, n as i32, y, nul(), ldy as i64) }, "mma reduce")?;
        }
        return Ok(());
    }
    ck(unsafe {
        ds_linear_fp8_grouped(sr, x, ldx as i64, gstride as i64, rpg as i32, np(&w.w), np(&w.s), w.lds() as i64, m as i32,
                              n as i32, k as i32, y, ldy as i64)
    }, "grouped gemm")
}

/// Split-K for the tensor-core kernel: at least two blocks per SM (48 SMs on GB10), K slices
/// of at least 512 (examples/tune_mma.rs), bounded by the partials buffer.
fn mma_splits(m: usize, n: usize, k: usize, part_len: usize) -> usize {
    // TIDELM_NO_SPLITK=1 (tests): one pass over K at every M, so a row's bits do not depend
    // on how many rows share its step
    if no_splitk() {
        return 1;
    }
    // past 64 rows the grid also tiles M by 128
    let mtiles = if m > 64 { m.div_ceil(128) } else { 1 };
    let blocks = n.div_ceil(unsafe { ds_mma_rows_per_block(m as i32) } as usize) * mtiles;
    96usize.div_ceil(blocks).min(k / 512).min(part_len / (m * n)).max(1)
}

/// How the engine is built. `n_layers` and `keep` exist for tests that compare the two
/// weight sources on a truncated model; serving leaves them unset.
#[derive(Clone, Debug)]
pub struct Options {
    pub max_seq: usize,
    /// sequences held at once; every cache is allocated per lane up front
    pub max_lanes: usize,
    /// load the DSpark drafter
    pub spec: bool,
    /// all weights and routable experts on the device, routing through the slot LUT —
    /// the mode the step graph runs in; otherwise every step streams its weights
    pub resident: bool,
    /// resident only: issue each backbone step as one CUDA graph, captured per StepShape
    pub graphs: bool,
    /// run only the first n backbone layers (tests)
    pub n_layers: Option<usize>,
    /// per backbone layer, the routable experts (model.py's prune mask); None = all
    pub keep: Option<Vec<Vec<bool>>>,
    /// per DSpark block, the routable experts (tests; serving keeps all 3 x 128)
    pub dspark_keep: Option<Vec<Vec<bool>>>,
    /// tensor parallel over the pair (weights::Tp): this rank's shard of every backbone
    /// layer, routed expert and the head. None: one box.
    pub tp: Option<Tp>,
    /// resident only: record each lane's expert demand and misses (adaptive residency:
    /// crate::adapt), read with Engine::lane_demand
    pub record_demand: bool,
    /// resident only: arena slots past the residents for experts read in while serving (the
    /// hot-read ring); 0 for none
    pub hot_ring: usize,
    /// image inputs: the vision tower (rank 1 of a pair, else this box), the span delimiters
    /// and the image router bias (DSV41_VISION=1, DSV41_VISION_MODE=peer)
    pub vision: bool,
    /// the decoder SWA bounded replay (DSV41_SWA_REPLAY=1): prompt rows run the encoder half
    /// (layers 0..=candidate_source_layer) and the decoder half replays a prompt's last window
    /// rows once (engine/model.py::decoder_replay)
    pub swa_replay: bool,
}

impl Options {
    pub fn streaming(max_seq: usize, spec: bool) -> Self {
        Options { max_seq, max_lanes: 1, spec, resident: false, graphs: false, n_layers: None, keep: None, dspark_keep: None,
                  tp: None, record_demand: false, hot_ring: 0, vision: false, swa_replay: false }
    }

    /// The bytes both ranks of a pair must agree on before they start (tidelm_comm::join):
    /// every option, the expert sets (as digests), and the executable itself.
    pub fn boot_config(&self, model_dir: &std::path::Path) -> Result<Vec<u8>> {
        let fnv = |bytes: &[u8]| bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        let mask = |m: &Option<Vec<Vec<bool>>>| m.as_ref().map(|m| fnv(&m.iter().flatten().map(|&b| b as u8).collect::<Vec<_>>())).unwrap_or(0);
        // adaptive residency: rank 1 receives rank 0's resident set during the join (pair.rs)
        let keep_digest = if self.record_demand { 0 } else { mask(&self.keep) };
        let exe = std::fs::read("/proc/self/exe")?;
        let cfg = std::fs::read(model_dir.join("config.json")).or_else(|_| std::fs::read(model_dir.join("inference/config.json")))?;
        // the drafter's listed vocabulary by content: two ranks listing different ids draft
        // differently with nothing in the log
        let vocab = Config::load(model_dir)?.vocab_size;
        let listed = crate::draft_vocab::from_env(vocab)?.map(|ids| crate::draft_vocab::digest(&ids)).unwrap_or(0);
        Ok(format!("tidelm-pair v3 tp | max_seq {} lanes {} spec {} resident {} graphs {} layers {:?} | keep {:x} \
                    dspark_keep {:x} record {} hot_ring {} vision {} replay {} | exe {:x} | config {:x} | draft vocab {listed:x} | env NO_QDQ={:?} ENGRAM_CACHE_MB={:?} \
                    DRAFT_HEAD_FMT={:?} TP_DRAFT={:?} EXPERT_TOPP={:?}/{:?} PREFILL_ROWS={} NO_SPLITK={:?} \
                    HOT={:?}/{:?}/{:?} PACKED_KV={:?}",
                   self.max_seq, self.max_lanes, self.spec, self.resident, self.graphs, self.n_layers, keep_digest,
                   mask(&self.dspark_keep), self.record_demand, self.hot_ring, self.vision, self.swa_replay, fnv(&exe), fnv(&cfg), std::env::var("TIDELM_NO_QDQ").ok(),
                   std::env::var("TIDELM_ENGRAM_CACHE_MB").ok(), std::env::var("TIDELM_DRAFT_HEAD_FMT").ok(),
                   std::env::var("TIDELM_TP_DRAFT").ok(), std::env::var("TIDELM_EXPERT_TOPP").ok(),
                   std::env::var("TIDELM_EXPERT_TOPP_MIN_K").ok(), prefill_rows(), std::env::var("TIDELM_NO_SPLITK").ok(),
                   std::env::var("TIDELM_HOT_MIN").ok(), std::env::var("TIDELM_HOT_MAX").ok(),
                   std::env::var("TIDELM_HOT_PER_LAYER").ok(), std::env::var("TIDELM_PACKED_KV").ok())
                   .into_bytes())
    }
}

const OP_FORWARD: u32 = 1;
const OP_DRAFT: u32 = 2;
const OP_ROLLBACK: u32 = 3;
const OP_SEED: u32 = 4;
const OP_RESET_LANE: u32 = 5;
const OP_GRAPHS: u32 = 6;
const OP_SHUTDOWN: u32 = 7;
const OP_DRAFT_SAMPLED: u32 = 8;
const OP_MOVE_EXPERTS: u32 = 9;
const OP_SNAPSHOT: u32 = 10;
const OP_RESTORE: u32 = 11;
const OP_SNAP_DROP: u32 = 12;
const OP_DISK_SAVE: u32 = 13;
const OP_DISK_LOAD: u32 = 14;
const OP_VISION: u32 = 15;
const OP_REPLAY: u32 = 16;

/// This engine's half of a tensor-parallel pair: the RoCE one-shot gather every sharded
/// output goes through, the NCCL communicator, and the control channel. Rank 0 (the
/// leader) sends every engine call to rank 1 before running it; rank 1 runs `follow` and
/// decides nothing.
pub struct Pair {
    pub comm: tidelm_comm::Comm,
    pub ctl: tidelm_comm::Control,
    pub roce: tidelm_comm::roce::Roce,
}

/// The largest shard a step gathers, in bytes: the head's logits (64 rows x half the
/// vocabulary), or a prefill step's engram projection (its rows x half of (hc + 1) x dim) or
/// MoE intermediates (its rows and their top-6 pairs x half of moe_inter_dim).
pub fn tp_max_gather() -> usize {
    let p = prefill_rows();
    (TMAX * 129_280 / 2).max(p * 5 * 5120 / 2).max(p * 7 * 2304 / 2) * 2
}

/// Which logits a segment of a step returns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Want {
    Nothing,
    /// the last row's `[V]`
    Last,
    /// every row's, `[rows][V]` flattened (speculative verification)
    All,
}

/// One lane's recorded expert demand (Engine::lane_demand): per backbone layer and expert
/// [layer * E + e] the picks of the router's unmasked top-k and their scores, and per phase
/// (0 prompt, 1 decode) [missed, picks, missed score, score], a miss a pick that was not
/// routable.
#[derive(Clone, Debug, Default)]
pub struct LaneDemand {
    pub counts: Vec<f64>,
    pub mass: Vec<f64>,
    pub miss: [f64; 8],
}

/// One lane's rows in a step: tokens `ids` at that lane's next positions. `decode`: the rows
/// are generated tokens (a decode step or a verify block), not prompt - only those take
/// TIDELM_EXPERT_TOPP. `next`: the rows the lane's next step will likely feed after these (a
/// prompt's next chunk), whose engram rows are read from NVMe while this step computes - a
/// hint only: what the next step really feeds is gathered as always, hits or not.
/// `hashes`: the rows' engram n-gram hashes ([row][engram layer][col], NgramHashState::hash's
/// layout) instead of hashing `ids` at the lane's positions - a probe's excerpt rows keep the
/// n-grams of their place in the prompt (probe_prefill.excerpt_hashes).
#[derive(Clone, Copy, Debug)]
pub struct Seg<'a> {
    pub lane: usize,
    pub ids: &'a [u32],
    pub want: Want,
    pub decode: bool,
    pub next: &'a [u32],
    pub hashes: Option<&'a [i64]>,
}

/// Per-call wall-clock split, printed by the CLI: where a streaming step's time goes.
#[derive(Default, Debug, Clone)]
pub struct StepTimes {
    pub weights_s: f64,
    pub experts_s: f64,
    pub head_s: f64,
    pub engram_s: f64,
    pub compute_s: f64,
    pub experts_loaded: usize,
    /// backbone steps: host preparation (incl. engram), and launch to logits on the host
    pub prepare_s: f64,
    pub device_s: f64,
}

/// Gathers at or below this many unique rows read through the page cache (the memmap),
/// above it with O_DIRECT sector reads: the Python serving setting DSV41_ENGRAM_DIRECT=prefill
/// (engine/engram.py: a page-cache hit beats an O_DIRECT round trip at decode sizes, while
/// prefill-sized gathers would flood the page cache and stall the MoE kernels in reclaim).
const ENGRAM_MAPPED_MAX: usize = 384;
/// Native gather workers per table (DSV41_ENGRAM_NATIVE_THREADS)
const ENGRAM_THREADS: i32 = 64;
const ENGRAM_ROW: usize = 264;

/// One engram layer's table on NVMe: a page-cache gather over a read-only mapping of its
/// shard, an O_DIRECT gather, and a packed row cache in front of both.
struct EngramTable {
    mapped: *mut c_void,
    direct: *mut c_void,
    map: *mut c_void,
    map_len: usize,
    cache: RowCache,
}

// the gather contexts are internally synchronised native objects; a table is used by one
// thread at a time (moved into the step's gather thread and back)
unsafe impl Send for EngramTable {}

impl EngramTable {
    /// The rows for unique ids `ids` into `out` ([ids][264] bytes): cache hits copied, misses
    /// gathered and cached.
    fn gather(&mut self, ids: &[i64], out: &mut [u8]) -> i32 {
        let (mapped, direct) = (self.mapped, self.direct);
        self.cache.gather(ids, out, &mut |miss: &[i64], buf: &mut [u8]| {
            let ctx = if miss.len() <= ENGRAM_MAPPED_MAX { mapped } else { direct };
            unsafe { eg::engram_gather_run(ctx, miss.as_ptr(), miss.len(), buf.as_mut_ptr()) }
        })
    }

    /// The rows of unique ids `ids` into the cache only (a next step's prefetch).
    fn warm(&mut self, ids: &[i64]) -> i32 {
        let (mapped, direct) = (self.mapped, self.direct);
        self.cache.warm(ids, &mut |miss: &[i64], buf: &mut [u8]| {
            let ctx = if miss.len() <= ENGRAM_MAPPED_MAX { mapped } else { direct };
            unsafe { eg::engram_gather_run(ctx, miss.as_ptr(), miss.len(), buf.as_mut_ptr()) }
        })
    }
}

impl Drop for EngramTable {
    fn drop(&mut self) {
        unsafe {
            eg::engram_gather_destroy(self.mapped);
            eg::engram_gather_destroy(self.direct);
            libc::munmap(self.map, self.map_len);
        }
    }
}

/// engine/engram_cache.py::PackedRowCache: direct-mapped by id % capacity, one tag and one
/// 264-byte row a slot (272 bytes of budget). The tables are immutable, so it never needs
/// clearing; a collision only costs an extra read.
struct RowCache {
    keys: Vec<i64>,
    data: Vec<u8>,
    /// a call's misses, where they go, and their rows as read: kept, so a step allocates nothing
    /// once these have grown to the largest step's
    miss: Vec<i64>,
    at: Vec<usize>,
    buf: Vec<u8>,
}

impl RowCache {
    fn new(budget: usize) -> Self {
        let cap = (budget / 272).max(1);
        let mut data = vec![0u8; cap * ENGRAM_ROW];
        // calloc'd pages are committed on first write: write one byte a page now, so the cache's
        // memory is taken at boot and not as it fills while serving
        for i in (0..data.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(data.as_mut_ptr().add(i), 0) };
        }
        RowCache { keys: vec![-1; cap], data, miss: Vec::new(), at: Vec::new(), buf: Vec::new() }
    }

    fn gather(&mut self, ids: &[i64], out: &mut [u8], read: &mut dyn FnMut(&[i64], &mut [u8]) -> i32) -> i32 {
        let cap = self.keys.len() as i64;
        let (miss, at, buf) = (&mut self.miss, &mut self.at, &mut self.buf);
        (miss.clear(), at.clear());
        for (i, &id) in ids.iter().enumerate() {
            let slot = id.rem_euclid(cap) as usize;
            if self.keys[slot] == id {
                out[i * ENGRAM_ROW..(i + 1) * ENGRAM_ROW].copy_from_slice(&self.data[slot * ENGRAM_ROW..(slot + 1) * ENGRAM_ROW]);
            } else {
                miss.push(id);
                at.push(i);
            }
        }
        if miss.is_empty() {
            return 0;
        }
        // zeroed like the fresh buffer it replaces: the read may leave a row untouched
        buf.clear();
        buf.resize(miss.len() * ENGRAM_ROW, 0);
        let rc = read(miss, buf);
        if rc != 0 {
            return rc;
        }
        for (j, (&id, &i)) in miss.iter().zip(at.iter()).enumerate() {
            let row = &buf[j * ENGRAM_ROW..(j + 1) * ENGRAM_ROW];
            out[i * ENGRAM_ROW..(i + 1) * ENGRAM_ROW].copy_from_slice(row);
            let slot = id.rem_euclid(cap) as usize;
            self.keys[slot] = id;
            self.data[slot * ENGRAM_ROW..(slot + 1) * ENGRAM_ROW].copy_from_slice(row);
        }
        0
    }

    /// Reads the rows of `ids` (unique) the cache misses into it, nothing out: a prefetch.
    fn warm(&mut self, ids: &[i64], read: &mut dyn FnMut(&[i64], &mut [u8]) -> i32) -> i32 {
        let cap = self.keys.len() as i64;
        let (miss, buf) = (&mut self.miss, &mut self.buf);
        miss.clear();
        miss.extend(ids.iter().copied().filter(|&id| self.keys[id.rem_euclid(cap) as usize] != id));
        if miss.is_empty() {
            return 0;
        }
        // zeroed like the fresh buffer it replaces: the read may leave a row untouched
        buf.clear();
        buf.resize(miss.len() * ENGRAM_ROW, 0);
        let rc = read(miss, buf);
        if rc != 0 {
            return rc;
        }
        for (j, &id) in miss.iter().enumerate() {
            let slot = id.rem_euclid(cap) as usize;
            self.keys[slot] = id;
            self.data[slot * ENGRAM_ROW..(slot + 1) * ENGRAM_ROW].copy_from_slice(&buf[j * ENGRAM_ROW..(j + 1) * ENGRAM_ROW]);
        }
        0
    }
}

/// Keys a block of the decode indexer scores (ds_index_scores_mma_chunked): 64 blocks at 64K.
const INDEX_DECODE_CHUNK: usize = 1024;

/// A raw pointer handed to a step's gather thread (see run_step).
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

/// One routed expert's FP4 tensors (w1/w3 [inter][dim], w2 [dim][inter], packed 2 a byte,
/// UE8M0 per 32): bytes of a weight and its scales per output row.
const INTER: usize = 2304;
const DIM: usize = 5120;
const W13_ROW: usize = DIM / 2;
const S13_ROW: usize = DIM / 32;
const W2_ROW: usize = INTER / 2;
const S2_ROW: usize = INTER / 32;

/// The routed-expert arena for streaming: ARENA_SLOTS slots in the native FP4 layout
/// (tools/fp4_moe.py::ExpertArena at world 1).
struct Arena {
    w1: DeviceBuffer<u8>,
    s1: DeviceBuffer<u8>,
    w2: DeviceBuffer<u8>,
    s2: DeviceBuffer<u8>,
    w3: DeviceBuffer<u8>,
    s3: DeviceBuffer<u8>,
    /// rows a slot holds: the intermediate rows of w1/w3 and the output rows of w2 (all of
    /// them on one box, this rank's half under tensor parallel)
    inter: usize,
    dim: usize,
}

impl Arena {
    fn new(slots: usize, tp: Option<Tp>) -> Result<Self> {
        let w = tp.map(|t| t.world).unwrap_or(1);
        let (inter, dim) = (INTER / w, DIM / w);
        Ok(Arena {
            w1: DeviceBuffer::new(slots * inter * W13_ROW)?,
            s1: DeviceBuffer::new(slots * inter * S13_ROW)?,
            w2: DeviceBuffer::new(slots * dim * W2_ROW)?,
            s2: DeviceBuffer::new(slots * dim * S2_ROW)?,
            w3: DeviceBuffer::new(slots * inter * W13_ROW)?,
            s3: DeviceBuffer::new(slots * inter * S13_ROW)?,
            inter,
            dim,
        })
    }

    /// (destination, source offset in the full tensor, bytes) of slot `slot`'s six tensors,
    /// in read_expert_into's order (w1, s1, w2, s2, w3, s3), for rank `rank`'s rows.
    fn slot_dsts(&self, slot: usize, rank: usize) -> [(*mut c_void, usize, usize); 6] {
        let (i, d) = (self.inter, self.dim);
        [
            (self.w1.ptr_at(slot * i * W13_ROW), rank * i * W13_ROW, i * W13_ROW),
            (self.s1.ptr_at(slot * i * S13_ROW), rank * i * S13_ROW, i * S13_ROW),
            (self.w2.ptr_at(slot * d * W2_ROW), rank * d * W2_ROW, d * W2_ROW),
            (self.s2.ptr_at(slot * d * S2_ROW), rank * d * S2_ROW, d * S2_ROW),
            (self.w3.ptr_at(slot * i * W13_ROW), rank * i * W13_ROW, i * W13_ROW),
            (self.s3.ptr_at(slot * i * S13_ROW), rank * i * S13_ROW, i * S13_ROW),
        ]
    }
}

/// Activations for up to TMAX rows, allocated once.
struct Act {
    h: DeviceBuffer<u16>,
    pre: DeviceBuffer<f32>,
    apre: DeviceBuffer<f32>,
    fpre: DeviceBuffer<f32>,
    apost: DeviceBuffer<f32>,
    acomb: DeviceBuffer<f32>,
    fpost: DeviceBuffer<f32>,
    fcomb: DeviceBuffer<f32>,
    rsq: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    partial: DeviceBuffer<f32>,
    mma_part: DeviceBuffer<f32>,
    y: DeviceBuffer<u16>,
    qa: DeviceBuffer<u16>,
    qr: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    kv0: DeviceBuffer<u16>,
    kv: DeviceBuffer<u16>,
    o: DeviceBuffer<u16>,
    oa: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<u16>,
    // per row: position and lane; per lane slot: the step's segment (first row, rows, lane)
    pos: DeviceBuffer<i32>,
    lane: DeviceBuffer<i32>,
    seg_first: DeviceBuffer<i32>,
    seg_count: DeviceBuffer<i32>,
    seg_lane: DeviceBuffer<i32>,
    /// routed pairs of the step's live rows (filler rows' pairs route nowhere)
    live_pairs: DeviceBuffer<i32>,
    /// per row: a generated token's row (Seg::decode) - the split-key attention's rows and
    /// TIDELM_EXPERT_TOPP's; prompt (and filler) rows take attn_prefill_kernel
    decode_rows: DeviceBuffer<i32>,
    cidx1: DeviceBuffer<i32>,
    cidx2: DeviceBuffer<i32>,
    z: DeviceBuffer<f32>,
    idx: DeviceBuffer<i32>,
    rw: DeviceBuffer<f32>,
    keys: DeviceBuffer<i32>,
    block_slot: DeviceBuffer<i32>,
    block_pair: DeviceBuffer<i32>,
    block_route: DeviceBuffer<i32>,
    /// ds_route_large's counts, cursors and block bases (3 x slots)
    route_work: DeviceBuffer<i32>,
    hm: DeviceBuffer<u16>,
    parts: DeviceBuffer<f32>,
    g: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    hs: DeviceBuffer<u16>,
    sh: DeviceBuffer<u16>,
    moe_out: DeviceBuffer<u16>,
    // tensor parallel: shards and their gathered whole rows (unused on one box)
    oa_full: DeviceBuffer<u16>,
    attn_local: DeviceBuffer<u16>,
    hcat: DeviceBuffer<u16>,
    hcat_full: DeviceBuffer<u16>,
    moe_local: DeviceBuffer<u16>,
    ekv_local: DeviceBuffer<u16>,
    tp_logits: DeviceBuffer<u16>,
    tp_scratch: DeviceBuffer<u8>,
    /// the split-key attention's partials (ds_attn_work_bytes for TMAX rows of the drafter's
    /// whole heads over the window and index_topk keys)
    attn_work: DeviceBuffer<u8>,
    kvl: DeviceBuffer<f32>,
    sc: DeviceBuffer<f32>,
    lat: DeviceBuffer<u16>,
    latn: DeviceBuffer<u16>,
    kidx: DeviceBuffer<u16>,
    kidxn: DeviceBuffer<u16>,
    erows: DeviceBuffer<u16>,
    ekv: DeviceBuffer<u16>,
    emb: DeviceBuffer<u16>,
    row_ids: DeviceBuffer<i32>,
    // the head: the final norm of every row, the wanted rows gathered, their logits
    head_all: DeviceBuffer<u16>,
    head_x: DeviceBuffer<u16>,
    head_rows: DeviceBuffer<i32>,
    head_w: DeviceBuffer<u16>,
    logits: DeviceBuffer<u16>,
    /// each head row's argmax (greedy decoding reads these, not the logits)
    am: DeviceBuffer<i32>,
    // indexer (model.py::_indexer): per-row scores over every compressed position
    iq: DeviceBuffer<u16>,
    iwb: DeviceBuffer<u16>,
    iwf: DeviceBuffer<f32>,
    scores: DeviceBuffer<u16>,
    cand: DeviceBuffer<u8>,
    bmax: DeviceBuffer<u16>,
    blocks: DeviceBuffer<i32>,
    topk: DeviceBuffer<i32>,
    clen1: DeviceBuffer<i32>,
    clen2: DeviceBuffer<i32>,
    // DSpark: the target layers' hc means of the last step, the rows a seed reads, the
    // projected main hidden; the draft rows' window ends, key slots and key lists, and the
    // markov chain ([step][lane] ids, step 0 the lanes' last tokens)
    main_h: DeviceBuffer<u16>,
    main_g: DeviceBuffer<u16>,
    main_x: DeviceBuffer<u16>,
    sel: DeviceBuffer<i32>,
    dwin: DeviceBuffer<i32>,
    dslot: DeviceBuffer<i32>,
    dcidx: DeviceBuffer<i32>,
    markov_e: DeviceBuffer<u16>,
    markov_b: DeviceBuffer<u16>,
    /// a split drafter's own columns of the markov bias, before the gather
    markov_bl: DeviceBuffer<u16>,
    d_idx: DeviceBuffer<i32>,
    /// the confidence head's two halves: per draft row from hc_pre (before the norm), and
    /// per [step][lane] from the markov embedding
    xpre: DeviceBuffer<u16>,
    conf_x: DeviceBuffer<f32>,
    conf_m: DeviceBuffer<f32>,
    /// sampling (ds_nucleus): per head row its temperature and top_p and its nucleus
    /// [row][NUC_CAP]; per draft lane the same, the draws' uniforms per [step][lane], and the
    /// drafter's distributions per [step][lane][DRAFT_NUCLEUS_K]
    s_temp: DeviceBuffer<f32>,
    s_topp: DeviceBuffer<f32>,
    nuc_ids: DeviceBuffer<i32>,
    nuc_p: DeviceBuffer<f32>,
    nuc_cnt: DeviceBuffer<i32>,
    d_temp: DeviceBuffer<f32>,
    d_topp: DeviceBuffer<f32>,
    d_u: DeviceBuffer<f32>,
    dq_ids: DeviceBuffer<i32>,
    dq_p: DeviceBuffer<f32>,
    dq_cnt: DeviceBuffer<i32>,
}

/// Which compressed rows the current layer attends over (model.py's `sh.topk`, which an
/// index-source layer sets for itself and the layers after it).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
    /// every visible position (exactly the indexer's result while n_c <= index_topk)
    AllVisible,
    /// the indexer's top-k in `act.topk`
    Indexed,
}

/// Everything about a backbone step that its launches depend on besides buffer contents:
/// the CUDA-graph key. Rows are bucketed (row_bucket) and widths that grow with the context
/// to powers of two, so a graph serves many steps; the padding is masked by the per-row
/// visible counts, and filler rows run in the scratch lane.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct StepShape {
    /// which layers the step runs (Stage)
    stage: Stage,
    t: usize,
    /// all-visible list widths per ratio (bucketed, <= index_topk)
    c1: usize,
    c2: usize,
    /// the indexer's key range per ratio when it runs (bucketed), None while every lane's
    /// n_c <= index_topk
    n1: Option<usize>,
    n2: Option<usize>,
    /// layer 20's candidate-block mask exists (some lane's n_c > 16,384)
    cand: bool,
    /// head rows (bucketed); 0 for none
    head: usize,
    /// the step has prompt rows / decode rows: which attention kernels it launches
    prompt: bool,
    decode: bool,
    /// some rows are an image span's (spliced, routed with bias_vl): the step runs eager
    vision: bool,
}

fn bucket(n: usize, lo: usize, hi: usize) -> usize {
    n.max(1).next_power_of_two().max(lo).min(hi)
}

/// What a step needs after its device work: the shape it ran, and per segment its first
/// row and its (first, count) of head rows.
struct Plan {
    shape: StepShape,
    firsts: Vec<usize>,
    heads: Vec<(usize, usize)>,
}

/// Which layers a step runs (engine/model.py::decoder_replay, the decoder SWA bounded replay):
/// every layer; the encoder half alone - prompt rows through layers 0..=candidate_source_layer
/// (everything that writes global KV), no head; or a replay - a prompt's last window rows from
/// their state entering that layer (kept by the steps that ran them) through the rest, then
/// the head. The replay re-runs the encoder's last layer for its rows to remake their index
/// top-k and candidate pool (rewriting that layer's KV for them with the same values), where
/// the Python engine carries both across.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Stage {
    Full,
    Encoder,
    Replay,
}

/// How a block's attention sees keys: the backbone's (window ring + compressed cache) or a
/// DSpark block's (its seeded ring up to the last main position + the draft block's keys).
#[derive(Clone, Copy)]
enum Ctx {
    Main { l: usize, c1: usize, c2: usize },
    Draft { k: usize },
}

/// A ratio-2 compressor's state across positions, per lane, all on the device so the step
/// can be a graph: the pending (unpaired) position and what rollback needs to restore it
/// (model.py::Caches.pending / _chunk_inputs). `[lanes][...]`; the chunk memory holds each
/// lane's last segment, `[lanes][TMAX][D]`.
struct Ratio2 {
    pkv: DeviceBuffer<f32>,
    psc: DeviceBuffer<f32>,
    flag: DeviceBuffer<i32>,
    mem_kv: DeviceBuffer<f32>,
    mem_sc: DeviceBuffer<f32>,
    before_kv: DeviceBuffer<f32>,
    before_sc: DeviceBuffer<f32>,
    before_flag: DeviceBuffer<i32>,
    mem_start: DeviceBuffer<i32>,
}

/// Every per-sequence device cache, lane-major: lane L's part of a cache starts L lane
/// strides in. Lane `max_lanes` is the scratch lane filler rows write.
struct Caches {
    /// per layer `[lanes][RING][hd]`
    ring: Vec<DeviceBuffer<u16>>,
    /// per kv source `[lanes][rows][hd]` / `[lanes][rows][index_hd]`
    ckv: BTreeMap<usize, DeviceBuffer<u16>>,
    ik: BTreeMap<usize, DeviceBuffer<u16>>,
    /// per kv source: compressed rows per lane (max_seq / ratio + 1)
    crows: BTreeMap<usize, usize>,
    r2: BTreeMap<usize, Ratio2>,
    /// per DSpark block `[lanes][RING][hd]`
    mtp_ring: Vec<DeviceBuffer<u16>>,
    /// the draft block's own keys, `[lanes][block][hd]` (each DSpark block in turn)
    dkv: DeviceBuffer<u16>,
    /// rollback entries, read by the device: target lengths and their lanes
    n_dev: DeviceBuffer<i32>,
    rb_lane: DeviceBuffer<i32>,
}

/// One sequence's host state.
struct Lane {
    len: usize,
    hash: NgramHashState,
    /// (first position, first row, rows) of this lane's segment in the step it last ran in
    last: (usize, usize, usize),
    /// that step's number (a seed reads main_h, which only the latest step's rows are in)
    step: u64,
}

/// The routable experts of one family of blocks (backbone layers, or DSpark blocks) on the
/// device: an arena holding those experts (and `ring` free slots past them, for experts read
/// in while serving) and the [block][expert] -> slot LUT (model.py's slot_lut; -1 for an
/// expert that is not routable). Slots are global across blocks, so moving residency from
/// one layer to another is a read into the freed slot and two LUT entries.
struct ExpertSet {
    arena: Arena,
    /// per block, expert -> slot, plus a last column (the expert id `n_experts`) at the null
    /// slot: a pick TIDELM_EXPERT_TOPP drops routes there and its output is an exact 0
    lut: DeviceBuffer<i32>,
    /// the LUT's host copy, the one every change is made to before it is uploaded
    lut_host: Vec<i32>,
    /// per slot, the (block, expert) it holds; None for a free slot
    owner: Vec<Option<(usize, usize)>>,
    n_experts: usize,
    /// tensor names: `{base}{block}.ffn.experts.{expert}.`
    base: &'static str,
    tp: Option<Tp>,
    /// one past the arena's slots: the FP4 kernels read no weights for it
    null_slot: i32,
    /// the read workers' streams and pinned staging buffers (io_threads() of them), made with
    /// the set and reused by every read, so an adaptation pass allocates nothing
    workers: std::sync::Mutex<Vec<(Stream, PinnedBuffer<u8>)>>,
}

impl ExpertSet {
    /// `tp`: hold this rank's half of every routable expert (w1/w3 intermediate rows, w2
    /// output rows), so the pair's memory holds twice the experts and each rank does half of
    /// every routed pair. `spare`: free slots past the routable experts.
    #[allow(clippy::too_many_arguments)]
    fn load(ckpt: &Checkpoint, base: &'static str, blocks: usize, n_experts: usize, keep: Option<&[Vec<bool>]>,
            tp: Option<Tp>, spare: usize) -> Result<Self> {
        let routable = |b: usize, x: usize| keep.map(|k| k[b][x]).unwrap_or(true);
        let n_res: usize = (0..blocks).map(|b| (0..n_experts).filter(|&x| routable(b, x)).count()).sum();
        let n_slots = n_res + spare;
        let arena = Arena::new(n_slots.max(1), tp)?;
        let mut set = ExpertSet {
            arena,
            lut: DeviceBuffer::zeroed(1)?,
            lut_host: vec![-1i32; blocks * (n_experts + 1)],
            owner: vec![None; n_slots],
            n_experts,
            base,
            tp,
            null_slot: n_slots as i32,
            workers: std::sync::Mutex::new(
                (0..io_threads()).map(|_| Ok((Stream::new()?, PinnedBuffer::<u8>::uninit(STAGE_BYTES)?))).collect::<Result<_>>()?),
        };
        let mut jobs = Vec::with_capacity(n_res);
        for b in 0..blocks {
            for x in 0..n_experts {
                if routable(b, x) {
                    let slot = jobs.len();
                    jobs.push((slot, b, x));
                    set.lut_host[b * (n_experts + 1) + x] = slot as i32;
                    set.owner[slot] = Some((b, x));
                }
            }
            set.lut_host[b * (n_experts + 1) + n_experts] = n_slots as i32;
        }
        set.read_many(ckpt, &jobs)?;
        set.lut = DeviceBuffer::from_slice(&set.lut_host)?;
        Ok(set)
    }

    /// Experts `(slot, block, expert)` from the checkpoint into their slots on io_threads()
    /// workers, each with its own staging buffer and stream; under tensor parallel a rank reads
    /// only its share of each (ExpertReader::read_expert_share_into). Returns once every copy
    /// is done. The LUT is the caller's. One whole expert at a time, the boot load took 31.8 s
    /// for 10,000 (314 a second); examples/bench_expert_read.rs reads 966 shares a second on
    /// 8 threads (whole experts peak at 542 on 4).
    fn read_many(&self, ckpt: &Checkpoint, jobs: &[(usize, usize, usize)]) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (rank, world) = self.tp.map(|t| (t.rank, t.world)).unwrap_or((0, 1));
        // destinations as plain addresses: the workers never touch the arena itself
        let dsts: Vec<[(usize, usize); 6]> =
            jobs.iter().map(|&(slot, _, _)| self.arena.slot_dsts(slot, rank).map(|(d, _, n)| (d as usize, n))).collect();
        let base = self.base;
        let next = AtomicUsize::new(0);
        let failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        let threads = io_threads().min(jobs.len()).max(1);
        let workers = &self.workers;
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut held = None;
                    let mut work = || -> Result<()> {
                        tidelm_cuda::set_device(0)?;
                        let taken = workers.lock().unwrap().pop();
                        let (stream, stage) = match taken {
                            Some(w) => held.insert(w),
                            None => held.insert((Stream::new()?, PinnedBuffer::<u8>::uninit(STAGE_BYTES)?)),
                        };
                        let reader = ExpertReader::new(ckpt);
                        loop {
                            if failed.lock().unwrap().is_some() {
                                return Ok(());
                            }
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(_, b, x)) = jobs.get(i) else { return Ok(()) };
                            let prefix = format!("{base}{b}.ffn.experts.{x}.");
                            let views = reader.read_expert_share_into(stage, &prefix, rank, world)?;
                            for (&(off, len), &(dst, n)) in views.iter().zip(dsts[i].iter()) {
                                if len != n {
                                    return Err(format!("{prefix}: a share of {len} bytes, the arena expects {n}").into());
                                }
                                let src = unsafe { stage.as_ptr().add(off) } as *const c_void;
                                unsafe { memcpy_async(dst as *mut c_void, src, n, MEMCPY_HOST_TO_DEVICE, stream)? };
                            }
                            // the staging buffer is reused by this worker's next expert
                            stream.synchronize()?;
                        }
                    };
                    if let Err(e) = work() {
                        failed.lock().unwrap().get_or_insert(e.to_string());
                    }
                    if let Some(w) = held {
                        workers.lock().unwrap().push(w);
                    }
                });
            }
        });
        match failed.into_inner().unwrap() {
            Some(e) => Err(e.into()),
            None => Ok(()),
        }
    }
}

/// Expert-read workers (TIDELM_IO_THREADS, default 8): GB10's NVMe gives 966 expert shares a
/// second on 8 against 898 on 4 and 821 on 16 (examples/bench_expert_read.rs).
fn io_threads() -> usize {
    std::env::var("TIDELM_IO_THREADS").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n >= 1).unwrap_or(8)
}

/// One hot read for a reader thread: the expert's tensor-name prefix, where its slot's six
/// tensors go (this rank's share), and where to report.
struct HotJob {
    prefix: String,
    dsts: [(usize, usize); 6],
    done: std::sync::mpsc::Sender<std::result::Result<(), String>>,
}

/// Async hot reads' state (crate::hot): the ring over the backbone arena's spare slots, the
/// candidate rows and table on the device, the table's host copy (the next step plans from
/// it), the marks of the step in flight, and the reader threads.
struct Hot {
    cfg: crate::hot::HotConfig,
    ring: crate::hot::HotRing,
    /// [TMAX][topk][2] the step's (share, expert) per row and pick
    rows: DeviceBuffer<f32>,
    /// [layer][2 x WIDTH]
    table: DeviceBuffer<f32>,
    table_host: PinnedBuffer<f32>,
    marks: DeviceBuffer<i32>,
    saved: DeviceBuffer<i32>,
    n_marks: usize,
    jobs: std::sync::mpsc::Sender<HotJob>,
    waits: Vec<std::sync::mpsc::Receiver<std::result::Result<(), String>>>,
    /// arena slot -> its six destinations (this rank's share)
    dsts: HashMap<usize, [(usize, usize); 6]>,
    stats: crate::hot::HotStats,
}

impl Hot {
    /// The state the next plan is made from - the last decode step's candidate table and the
    /// ring - as one number: both ranks must hold the same (their plans are not sent).
    fn digest(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100000001b3);
        for &x in self.table_host.iter() {
            mix(x.to_bits() as u64);
        }
        for (&(l, x), &slot) in self.ring.live.iter().chain(self.ring.pending.iter()) {
            mix(((l as u64) << 40) | ((x as u64) << 20) | slot as u64);
        }
        h
    }

    /// `cap` reader threads (a step's reads in parallel), each with its own checkpoint
    /// handle, staging buffer and stream.
    fn new(cfg: crate::hot::HotConfig, slots: Vec<usize>, model_dir: &std::path::Path, layers: usize, topk: usize,
           set: &ExpertSet) -> Result<Self> {
        let (rank, world) = set.tp.map(|t| (t.rank, t.world)).unwrap_or((0, 1));
        let dsts = slots.iter().map(|&s| (s, set.arena.slot_dsts(s, rank).map(|(d, _, n)| (d as usize, n)))).collect();
        let (tx, rx) = std::sync::mpsc::channel::<HotJob>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let ckpt = std::sync::Arc::new(Checkpoint::open(model_dir)?);
        ckpt.open_all()?;
        for _ in 0..cfg.cap.max(1) {
            let (rx, ckpt) = (rx.clone(), ckpt.clone());
            std::thread::spawn(move || {
                let setup = || -> Result<(Stream, PinnedBuffer<u8>)> {
                    tidelm_cuda::set_device(0)?;
                    Ok((Stream::new()?, PinnedBuffer::<u8>::uninit(STAGE_BYTES)?))
                };
                let (stream, mut stage) = match setup() {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("hot reads: reader setup failed: {e}");
                        return;
                    }
                };
                let reader = ExpertReader::new(&ckpt);
                loop {
                    let job = match rx.lock().unwrap().recv() {
                        Ok(j) => j,
                        Err(_) => return,
                    };
                    let mut run = || -> Result<()> {
                        let views = reader.read_expert_share_into(&mut stage, &job.prefix, rank, world)?;
                        for (&(off, len), &(dst, n)) in views.iter().zip(job.dsts.iter()) {
                            if len != n {
                                return Err(format!("{}: a share of {len} bytes, the arena expects {n}", job.prefix).into());
                            }
                            let src = unsafe { stage.as_ptr().add(off) } as *const c_void;
                            unsafe { memcpy_async(dst as *mut c_void, src, n, MEMCPY_HOST_TO_DEVICE, &stream)? };
                        }
                        stream.synchronize()?;
                        Ok(())
                    };
                    let _ = job.done.send(run().map_err(|e| e.to_string()));
                }
            });
        }
        let n = slots.len();
        Ok(Hot {
            cfg,
            ring: crate::hot::HotRing::new(slots),
            rows: DeviceBuffer::zeroed(TMAX * topk * 2)?,
            table: DeviceBuffer::zeroed(layers * 2 * crate::hot::WIDTH)?,
            table_host: PinnedBuffer::<f32>::zeroed(layers * 2 * crate::hot::WIDTH)?,
            marks: DeviceBuffer::zeroed(3 * n)?,
            saved: DeviceBuffer::zeroed(2 * n)?,
            n_marks: 0,
            jobs: tx,
            waits: Vec::new(),
            dsts,
            stats: Default::default(),
        })
    }
}

/// A lane's state at length `n` that positions past n overwrite - the RAM prefix cache's entry
/// (v41_engine._snapshot_prefix). The compressed rows and index keys below n stay in the
/// lane's caches, which only ever append, so a snapshot holds the rest: every backbone
/// layer's and DSpark block's window rows of positions n - w .. n (w = min(n, window)), and
/// each ratio-2 compressor's pending (unpaired) input. Restoring it and prefilling on from n
/// continues the lane as if it had stopped there.
struct LaneSnap {
    n: usize,
    w: usize,
    /// [ring][w][hd]: the backbone's rings, then the DSpark blocks', rows in position order
    rows: DeviceBuffer<u16>,
    /// per ratio-2 source (kv-source order): [2][hd] f32 (kv, score) and its flag
    pending: DeviceBuffer<f32>,
    flags: DeviceBuffer<i32>,
    /// the replay's: the lane's kept rows (rep_h / rep_pre, all window slots) and its decoder
    /// layers' window floor
    rep_h: DeviceBuffer<u16>,
    rep_pre: DeviceBuffer<f32>,
    floor: i32,
}

/// Everything the resident mode keeps on the device besides the per-step activations.
struct Resident {
    /// lent out (taken) for the duration of each block
    layers: Vec<Option<Layer>>,
    embed: DeviceBuffer<u16>,
    head: DeviceBuffer<u16>,
    /// the drafter's FP8 copy of the head (DSV41_DRAFT_HEAD_FMT=fp8) over this rank's run of
    /// the draft vocabulary: half the bytes of the draft's largest read, and fewer rows with a
    /// listed vocabulary or a split drafter; verification keeps the bf16 head, so outputs are
    /// unchanged
    draft_head: Option<Fp8>,
    experts: ExpertSet,
    dspark_experts: Option<ExpertSet>,
    ids: DeviceBuffer<i32>,
}

struct Dspark {
    blocks: Vec<Layer>,
    w: DsparkW,
    noise_row: DeviceBuffer<u16>,
}

/// A lane's image spans (Engine::set_images) and the table their positions read: the aligner
/// rows of the images its prefill still feeds, then the three delimiters.
struct LaneVision {
    /// (first position, vit_h, vit_w, the span's first table row; None for a span never fed)
    spans: Vec<(usize, usize, usize, Option<usize>)>,
    table: DeviceBuffer<u16>,
    delims: usize,
}

impl LaneVision {
    /// The table row position `p` reads, or None outside every span.
    fn row_at(&self, p: usize, r: usize) -> Result<Option<i32>> {
        for &(s, vh, vw, base) in &self.spans {
            if p >= s && p < s + crate::vision::span_len(vh, vw, r) {
                return Ok(Some(match crate::vision::span_slot(p - s, vh, vw, r) {
                    Ok(cell) => (base.ok_or("an image span the prefill feeds was not encoded")? + cell) as i32,
                    Err(k) => (self.delims + k) as i32,
                }));
            }
        }
        Ok(None)
    }
}

pub struct Engine {
    pub cfg: Config,
    ckpt: Checkpoint,
    stream: Stream,
    rope_w: DeviceBuffer<f32>,
    rope_c: DeviceBuffer<f32>,
    norm: DeviceBuffer<u16>,
    hash: NgramHashState,
    engram_tables: BTreeMap<usize, EngramTable>,
    arena: Arena,
    expert_stage: PinnedBuffer<u8>,
    /// read-back targets: a fresh pageable Vec took ~40 ms per 12 MB of logits (145 MB/s,
    /// nsys, 48-row steps), more than the step's kernels
    logits_host: PinnedBuffer<u16>,
    am_host: PinnedBuffer<i32>,
    /// nuclei read back (the verify rows' or the drafter's): counts, ids, probabilities
    nuc_cnt_host: PinnedBuffer<i32>,
    nuc_ids_host: PinnedBuffer<i32>,
    nuc_p_host: PinnedBuffer<f32>,
    act: Act,
    caches: Caches,
    /// expert parallel over two boxes (attach_pair); None on one box
    pair: Option<Pair>,
    tp: Option<Tp>,
    /// rank 0 of a pair mirrors its calls to rank 1; rank 1 does not
    leader: bool,
    lanes: Vec<Lane>,
    max_lanes: usize,
    /// steps run so far (Lane::step)
    step_no: u64,
    dspark: Option<Dspark>,
    resident: Option<Resident>,
    /// [layer][expert] routable mask (u8) for the router, both modes; None = all
    keep: Option<DeviceBuffer<u8>>,
    /// Options::record_demand: per lane [counts | mass][layer][expert] f64 (adapt.rs's
    /// request demand) and per (lane, phase) [missed, picks, missed score, score]
    demand: Option<DeviceBuffer<f64>>,
    miss: Option<DeviceBuffer<f64>>,
    /// async hot reads (Options::hot_ring > 0): crate::hot
    hot: Option<Hot>,
    /// rank 1: rank 0's hot-read state digest for the step being replayed
    peer_hot: Option<u64>,
    /// the RAM prefix cache's snapshots, by (lane, id) (decode.rs decides; both ranks hold
    /// the same ones)
    snaps: HashMap<(usize, u64), LaneSnap>,
    /// snapshots' buffers not in use, each sized for a whole window (reserve_snapshots fills it
    /// at boot; a dropped snapshot's go back), and the window's row indices they gather by
    snap_free: Vec<LaneSnap>,
    snap_idx: DeviceBuffer<i32>,
    /// the prefix cache's disk tier (enable_prefix_disk) and its namespace
    disk: Option<crate::prefix_disk::Disk>,
    disk_ns: u64,
    /// vision (Options::vision): the span delimiters [3][dim] on every rank; the tower on the
    /// rank that encodes images (rank 1 of a pair, else this box); per lane its request's image
    /// spans and the rows they read (Engine::set_images)
    vis_delims: Option<DeviceBuffer<u16>>,
    tower: Option<crate::vision::Tower>,
    lane_vis: Vec<Option<LaneVision>>,
    /// a step with image rows: per row the lane table's row it takes (-1: none) and 1 for an
    /// image span's rows; the (first row, rows, lane) runs to splice
    vis_idx: DeviceBuffer<i32>,
    vis_img: DeviceBuffer<i32>,
    vis_splices: Vec<(usize, usize, usize)>,
    /// the decoder SWA bounded replay (Options::swa_replay): per lane the state entering the
    /// replay layer of its last window prompt rows (h [hc][d] bf16, pre [hc] f32 at slot
    /// lane * window + position % window), a step's rows' slots (-1: none kept), and per lane
    /// its decoder layers' window floor - the first position their rings hold KV for - with
    /// its host copy
    swa_replay: bool,
    rep_h: DeviceBuffer<u16>,
    rep_pre: DeviceBuffer<f32>,
    rep_idx: DeviceBuffer<i32>,
    win_floor: DeviceBuffer<i32>,
    floors: Vec<i32>,
    /// the same for the DSpark blocks
    dspark_keep: Option<DeviceBuffer<u8>>,
    /// per (drafting lanes, sampled)
    draft_graphs: HashMap<(usize, bool), GraphExec>,
    /// the draft logits' width: the draft vocabulary's size (all ids, or the listed ones)
    draft_cols: usize,
    /// a listed draft vocabulary: column -> token id, on the device and the host
    draft_ids: Option<(DeviceBuffer<i32>, Vec<u32>)>,
    /// the markov bias head's rows of this rank's run of the draft vocabulary (None: the whole
    /// head)
    markov_rows: Option<DeviceBuffer<u16>>,
    /// the last draft_batch's confidence logits, per lane its `dspark_block_size`
    draft_conf: Vec<Vec<f32>>,
    n_layers: usize,
    /// rows of the compressed-rope positions, per ratio: max(pos - (r - 1), 0)
    cpos1: DeviceBuffer<i32>,
    cpos2: DeviceBuffer<i32>,
    /// the one-hot initial pre_mix of every row, copied in on the device each step
    one_hot: DeviceBuffer<f32>,
    /// per engram layer, this step's gathered table rows and their (row, col) map
    eg_raw: Vec<DeviceBuffer<u8>>,
    eg_inv: Vec<DeviceBuffer<i32>>,
    /// the gather threads' pinned output, uploaded before the layer's graph segment
    eg_host: Vec<PinnedBuffer<u8>>,
    /// this step's pending gathers: (engram layer index, unique row ids)
    eg_jobs: Vec<(usize, Vec<i64>)>,
    /// the rows its segments' `next` hints will read, warmed into the caches after them
    eg_prefetch: Vec<(usize, Vec<i64>)>,
    /// the step being issued (select() reads its indexer ranges)
    shape: Option<StepShape>,
    /// live rows of the step being issued (the rest are filler)
    live_rows: usize,
    use_graphs: bool,
    /// per (shape, segment): segments split the step at the engram layers (run_step)
    /// step graphs by (shape, segment), each with its last use: at most graph_cap of them
    /// (TIDELM_GRAPH_CACHE, default 128: a session and two README runs used 85), the least
    /// recently used replaced past that
    graphs: HashMap<(StepShape, usize), (GraphExec, u64)>,
    graph_tick: u64,
    graph_cap: usize,
    /// steps served by a graph replay / by a capture
    pub graph_replays: usize,
    pub graph_captures: usize,
    pub max_seq: usize,
    pub times: StepTimes,
    /// TIDELM_DUMP_DIR: write the hc stream after every layer and the last logits as raw
    /// little-endian f32 (the layout rust/tools/ref_forward.py writes), for parity checks.
    /// Single-lane steps only.
    pub dump_dir: Option<std::path::PathBuf>,
    no_qdq: bool,
    /// TIDELM_PACKED_KV (on unless =0; the Python engine's DSV41_PACKED_KV=1): the compressed
    /// KV stored as FP4 codes + E4M3 scales, 288 bytes a row instead of 1,024 - the QDQ'd bf16
    /// values bit for bit, read back by the attention kernels' key loads
    packed_kv: bool,
    selection: Selection,
    /// the candidate-source layer's block mask is live for this step's later layers
    cand_live: bool,
    /// index-source layer passes in which the indexer actually ran
    pub indexed_layers: usize,
    /// sampled rows whose nucleus outgrew the device's window (recomputed on the host)
    pub nucleus_fallbacks: usize,
    /// TIDELM_EXPERT_TOPP (Python's DSV41_EXPERT_TOPP) and its MIN_K: decode rows keep their
    /// routed picks up to this share of their weight, at least MIN_K of them
    expert_topp: Option<(f32, i32)>,
    /// window-ring slots per lane and layer (ring_slots)
    ring: usize,
    /// rows a prompt-only step may carry (prefill_rows)
    pub pmax: usize,
    /// a prefill-sized (eager, never captured) step's device work is being issued
    eager_step: bool,
}

impl Resident {
    /// Every backbone layer's weights, the embedding and head tables, and the routable
    /// experts of the backbone (and of the drafter, with `spec`).
    /// `draft_rows`: this rank's run of the draft vocabulary (DraftVocab), `draft_tp` the
    /// drafter's shard (None: whole on every rank).
    fn load(ckpt: &Checkpoint, stream: &Stream, cfg: &Config, n_layers: usize, opts: &Options, draft_rows: &[u32],
            draft_tp: Option<Tp>) -> Result<Self> {
        let (d, v) = (cfg.dim, cfg.vocab_size);
        let t0 = Instant::now();
        // the routed experts (most of the bytes) stream in on their own workers while this
        // thread loads the dense weights: the two waited on each other, 1.7 s of dense first
        std::thread::scope(|sc| -> Result<Self> {
        let experts_h = sc.spawn(|| -> Result<(ExpertSet, f64)> {
            tidelm_cuda::set_device(0)?;
            let set = ExpertSet::load(ckpt, "layers.", n_layers, cfg.n_routed_experts, opts.keep.as_deref(), opts.tp,
                                      opts.hot_ring)?;
            Ok((set, t0.elapsed().as_secs_f64()))
        });
        let mut ld = Loader::new(ckpt, stream)?;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(Some(Layer::load(&mut ld, cfg, l, opts.tp)?));
        }
        let t_layers = t0.elapsed().as_secs_f64();
        let embed = ld.bf16("embed.weight", &[v, d])?;
        let full_head = ld.bf16("head.weight", &[v, d])?;
        let draft_head = if opts.spec && std::env::var("TIDELM_DRAFT_HEAD_FMT").as_deref() != Ok("bf16") {
            // the rows in draft-vocabulary order: a contiguous run is read in place, a listed
            // one gathered first. Quantized in 32-row blocks, so a rank's whole-block run gets
            // the same codes as the same rows of one box's head.
            let n = draft_rows.len();
            let contiguous = draft_rows.windows(2).all(|w| w[1] == w[0] + 1);
            let gathered = if contiguous {
                None
            } else {
                let ids = DeviceBuffer::from_slice(&draft_rows.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
                let rows = DeviceBuffer::<u16>::new(n * d)?;
                gather_rows(stream, np(&full_head), &ids, np(&rows), n, d)?;
                stream.synchronize()?;
                Some(rows)
            };
            let src = match &gathered {
                Some(g) => np(g),
                None => full_head.ptr_at(draft_rows[0] as usize * d),
            };
            let (w, s) = (DeviceBuffer::<u8>::new(n * d)?, DeviceBuffer::<u8>::new(n.div_ceil(32) * d.div_ceil(32))?);
            ck(unsafe { ds_quant_fp8_block32(stream.raw(), src, n as i32, d as i32, np(&w), np(&s)) }, "draft head quantization")?;
            stream.synchronize()?;
            Some(Fp8 { w, s, n, k: d })
        } else {
            None
        };
        // tensor parallel: the verifier's head is vocab-sharded (DSV41_TP_HEAD); the drafter's
        // FP8 copy above was made from the whole head, as the drafter runs whole on each rank
        let head = match opts.tp {
            Some(tp) => {
                let (r0, n) = tp.rows(v);
                let local = DeviceBuffer::<u16>::new(n * d)?;
                local.copy_from(0, &full_head, r0 * d, n * d, stream)?;
                stream.synchronize()?;
                local
            }
            None => full_head,
        };
        let t_head = t0.elapsed().as_secs_f64();
        // the drafter's experts split like the backbone's under a split drafter, whole otherwise
        let dspark_experts = if opts.spec {
            Some(ExpertSet::load(ckpt, "mtp.", cfg.n_mtp_layers, cfg.dspark_n_routed_experts, opts.dspark_keep.as_deref(),
                                 draft_tp, 0)?)
        } else {
            None
        };
        let t_draft = t0.elapsed().as_secs_f64();
        let (experts, t_experts) = experts_h.join().map_err(|_| "expert load thread panicked")??;
        eprintln!("resident load: layers {t_layers:.1} s, embed + head by {t_head:.1} s, drafter experts by {t_draft:.1} s; \
                   {} experts by {t_experts:.1} s (in parallel)", experts.owner.len());
        Ok(Resident { layers, embed, head, draft_head, experts, dspark_experts, ids: DeviceBuffer::zeroed(prefill_rows())? })
        })
    }
}

impl Engine {
    /// `opts.spec` loads the DSpark drafter (three blocks + its heads, ~0.8 GB resident).
    pub fn new(model_dir: &std::path::Path, token_map: Vec<i64>, opts: Options) -> Result<Self> {
        let t_new = Instant::now();
        let (max_seq, spec) = (opts.max_seq, opts.spec);
        let max_lanes = opts.max_lanes.max(1);
        let cfg = Config::load(model_dir)?;
        let ring = ring_slots(cfg.window_size);
        let pmax = prefill_rows();
        let n_layers = opts.n_layers.unwrap_or(cfg.n_layers).min(cfg.n_layers);
        let ckpt = Checkpoint::open(model_dir)?;
        // every shard's header now: adaptation reads reach shards the boot load did not
        ckpt.open_all()?;
        let stream = Stream::new()?;
        let (d, hc, rd) = (cfg.dim, cfg.hc_mult, cfg.rope_head_dim);
        let flat = |t: Vec<[f32; 2]>| -> Vec<f32> { t.into_iter().flatten().collect() };
        let rope_w = DeviceBuffer::from_slice(&flat(rope::table(
            rd, max_seq + 8, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        )))?;
        let rope_c = DeviceBuffer::from_slice(&flat(rope::table(
            rd, max_seq + 8, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast,
            cfg.beta_slow,
        )))?;
        // the drafter's vocabulary (listed, or all ids) and, under tensor parallel, its split
        // (TIDELM_TP_DRAFT=0 keeps it whole on each rank): rank r scores its contiguous run
        let listed = if spec { crate::draft_vocab::from_env(cfg.vocab_size)? } else { None };
        let draft_vocab: Vec<u32> = listed.clone().unwrap_or_else(|| (0..cfg.vocab_size as u32).collect());
        let draft_tp = opts.tp.filter(|_| spec && std::env::var("TIDELM_TP_DRAFT").as_deref() != Ok("0"));
        let draft_run = match draft_tp {
            Some(tp) => {
                let n = draft_vocab.len() / tp.world;
                if draft_vocab.len() % (32 * tp.world) != 0 {
                    return Err(format!("a {}-id draft vocabulary does not split into whole 32-row blocks", draft_vocab.len()).into());
                }
                draft_vocab[tp.rank * n..(tp.rank + 1) * n].to_vec()
            }
            None => draft_vocab.clone(),
        };
        if listed.is_some() && std::env::var("TIDELM_DRAFT_HEAD_FMT").as_deref() == Ok("bf16") {
            return Err("a listed draft vocabulary needs the fp8 draft head".into());
        }
        let (norm, dspark) = {
            let mut ld = Loader::new(&ckpt, &stream)?;
            let norm = ld.bf16("norm.weight", &[d])?;
            let dspark = if spec {
                let mut blocks = Vec::new();
                for k in 0..cfg.n_mtp_layers {
                    blocks.push(Layer::load_mtp(&mut ld, &cfg, k, draft_tp)?);
                }
                let w = DsparkW::load(&mut ld, &cfg)?;
                let noise_row = DeviceBuffer::<u16>::new(d)?;
                let id = cfg.dspark_noise_token_id as usize;
                ld.read_into("embed.weight", id * d * 2, (id + 1) * d * 2, noise_row.raw())?;
                Some(Dspark { blocks, w, noise_row })
            } else {
                None
            };
            (norm, dspark)
        };
        let layout = EngramLayout::new(
            &cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size,
            cfg.engram_compressed_vocab_size,
        );
        let hash = NgramHashState::new(layout, token_map, cfg.engram_pad_id);

        // DSV41_ENGRAM_CACHE_MB: the shipped profile gives each table 256 MB
        let engram_cache = std::env::var("TIDELM_ENGRAM_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256) << 20;
        let mut engram_tables = BTreeMap::new();
        for &l in &cfg.engram_layer_ids {
            let wname = format!("layers.{l}.engram.embed.weight");
            let sname = format!("layers.{l}.engram.embed.scale");
            let (wm, sm) = (ckpt.meta(&wname)?, ckpt.meta(&sname)?);
            let shard = ckpt.shard(&wname)?;
            if !std::ptr::eq(shard, ckpt.shard(&sname)?) {
                return Err(format!("layer {l}: engram weight and scale planes live in different shards").into());
            }
            let direct = unsafe {
                eg::engram_gather_create_direct(shard.file.direct_fd(), wm.lo as i64, sm.lo as i64, wm.shape[0] as i64,
                                                ENGRAM_THREADS)
            };
            // the whole shard, read-only; MADV_RANDOM: every row lands on its own page, so
            // readahead would only evict
            let cpath = std::ffi::CString::new(shard.path.as_os_str().as_encoded_bytes())?;
            let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
            let map_len = std::fs::metadata(&shard.path)?.len() as usize;
            let map = if fd >= 0 {
                unsafe { libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, fd, 0) }
            } else {
                libc::MAP_FAILED
            };
            if fd >= 0 {
                unsafe { libc::close(fd) };
            }
            if direct.is_null() || map == libc::MAP_FAILED {
                return Err(format!("layer {l}: engram gather setup failed").into());
            }
            unsafe { libc::madvise(map, map_len, libc::MADV_RANDOM) };
            let base = map as *const u8;
            let mapped = unsafe {
                eg::engram_gather_create(base.add(wm.lo as usize), base.add(sm.lo as usize), wm.shape[0] as i64, ENGRAM_THREADS)
            };
            if mapped.is_null() {
                return Err(format!("layer {l}: engram mapped gather setup failed").into());
            }
            engram_tables.insert(l, EngramTable { mapped, direct, map, map_len, cache: RowCache::new(engram_cache) });
        }

        if unsafe { ds_nucleus_cap() } as usize != NUC_CAP {
            return Err("ds41.cu's nucleus capacity is not the engine's NUC_CAP".into());
        }
        // row buffers hold a prefill step; the head's rows, the read-backs, the split attention's
        // workspace and the drafter's buffers hold a decode step's TMAX
        let t = pmax;
        let tm = TMAX;
        let lanes = max_lanes + 1; // + the scratch lane
        let p = t * cfg.n_activated_experts.max(cfg.dspark_n_activated_experts);
        let hdq = cfg.n_heads * cfg.head_dim;
        let hd = cfg.head_dim;
        let eng_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
        let n_target = cfg.dspark_target_layer_ids.len();
        let bsz = cfg.dspark_block_size;
        let draft_lanes = TMAX / bsz;
        let a = |n: usize| -> Result<DeviceBuffer<u16>> { Ok(DeviceBuffer::zeroed(n)?) };
        let f = |n: usize| -> Result<DeviceBuffer<f32>> { Ok(DeviceBuffer::zeroed(n)?) };
        let i = |n: usize| -> Result<DeviceBuffer<i32>> { Ok(DeviceBuffer::zeroed(n)?) };
        // the slots an expert set can route to: every routable backbone expert (or the drafter's
        // 3 x 128, or the streaming arena), for ds_route_large's work buffer
        let route_slots = match &opts.keep {
            Some(k) => k.iter().take(n_layers).map(|row| row.iter().filter(|&&b| b).count()).sum::<usize>(),
            None => n_layers * cfg.n_routed_experts,
        }
        // and the hot-read ring's slots past them (the null slot is one past those)
        + opts.hot_ring
        .max(cfg.n_mtp_layers * cfg.dspark_n_routed_experts)
        .max(ARENA_SLOTS);
        let route_entries = (unsafe { ds_route_blocks(p as i32, route_slots as i32 + 1, 64) } as usize * 64).max(p * 16);
        let act = Act {
            h: a(t * hc * d)?,
            pre: f(t * hc)?,
            apre: f(t * hc)?,
            fpre: f(t * hc)?,
            apost: f(t * hc)?,
            acomb: f(t * hc * hc)?,
            fpost: f(t * hc)?,
            fcomb: f(t * hc * hc)?,
            rsq: f(t)?,
            mixes: f(t * cfg.hc_mix_width())?,
            partial: f(HC_SPLITS * t * cfg.hc_mix_width())?,
            mma_part: f(MMA_PART)?,
            y: a(t * d)?,
            qa: a(t * cfg.q_lora_rank)?,
            qr: a(t * cfg.q_lora_rank)?,
            q: a(t * hdq)?,
            kv0: a(t * hd)?,
            kv: a(t * hd)?,
            o: a(t * hdq)?,
            oa: a(t * cfg.o_groups * cfg.o_lora_rank)?,
            attn_out: a(t * d)?,
            pos: i(t)?,
            lane: i(t)?,
            seg_first: i(lanes)?,
            seg_count: i(lanes)?,
            seg_lane: i(lanes)?,
            live_pairs: i(1)?,
            decode_rows: i(t)?,
            cidx1: i(t * cfg.index_topk)?,
            cidx2: i(t * cfg.index_topk)?,
            z: f(t * cfg.n_routed_experts)?,
            idx: i(p)?,
            rw: f(p)?,
            keys: i(p)?,
            block_slot: i(p)?,
            // ds_route_large's 64-pair blocks (prefill) over every slot: up to p of them
            block_pair: i(route_entries)?,
            block_route: i(route_entries)?,
            // every slot of the larger expert set, and the null slot
            route_work: i(3 * (route_slots + 1))?,
            hm: a(p * cfg.moe_inter_dim)?,
            parts: f(p * d)?,
            g: a(t * cfg.moe_inter_dim)?,
            u: a(t * cfg.moe_inter_dim)?,
            hs: a(t * cfg.moe_inter_dim)?,
            sh: a(t * d)?,
            moe_out: a(t * d)?,
            oa_full: a(if opts.tp.is_some() { t * cfg.o_groups * cfg.o_lora_rank } else { 1 })?,
            attn_local: a(if opts.tp.is_some() { t * d } else { 1 })?,
            hcat: a(if opts.tp.is_some() { (t + p) * cfg.moe_inter_dim } else { 1 })?,
            hcat_full: a(if opts.tp.is_some() { (t + p) * cfg.moe_inter_dim } else { 1 })?,
            moe_local: a(if opts.tp.is_some() { t * d } else { 1 })?,
            ekv_local: a(if opts.tp.is_some() { t * (hc + 1) * d } else { 1 })?,
            tp_logits: a(if opts.tp.is_some() { tm * cfg.vocab_size } else { 1 })?,
            tp_scratch: DeviceBuffer::zeroed(if opts.tp.is_some() { 2 * tp_max_gather() } else { 1 })?,
            attn_work: DeviceBuffer::zeroed(unsafe {
                ds_attn_work_bytes(tm as i32, cfg.n_heads as i32, (cfg.window_size + cfg.index_topk.max(bsz)) as i32)
            } as usize)?,
            kvl: f(t * hd)?,
            sc: f(t * hd)?,
            lat: a(t * hd)?,
            latn: a(t * hd)?,
            kidx: a(t * cfg.index_head_dim)?,
            kidxn: a(t * cfg.index_head_dim)?,
            erows: a(t * eng_cols * cfg.engram_head_dim)?,
            ekv: a(t * (hc + 1) * d)?,
            emb: a(t * d)?,
            row_ids: DeviceBuffer::from_slice(&(0..t as i32).collect::<Vec<_>>())?,
            head_all: a(t * d)?,
            head_x: a(tm * d)?,
            head_rows: i(tm)?,
            head_w: a(HEAD_ROWS * d)?,
            logits: a(tm * cfg.vocab_size)?,
            am: i(tm)?,
            iq: a(t * cfg.index_n_heads * cfg.index_head_dim)?,
            iwb: a(t * cfg.index_n_heads)?,
            iwf: f(t * cfg.index_n_heads)?,
            // the indexer's scores and block maxima for at most INDEX_SCORES (rows x keys) at a
            // time: a long context scores its step in row slices (Engine::index)
            scores: a((t * (max_seq + 1)).min(index_scores_budget()))?,
            cand: DeviceBuffer::zeroed(t * (max_seq + 1))?,
            bmax: a((t * ((max_seq + 1).div_ceil(8) + 1)).min(index_scores_budget() / 8 + t))?,
            blocks: i(t * 2048)?,
            topk: i(t * cfg.index_topk)?,
            clen1: i(t)?,
            clen2: i(t)?,
            main_h: a(t * n_target * d)?,
            main_g: a(t * n_target * d)?,
            main_x: a(t * d)?,
            sel: i(t)?,
            dwin: i(t)?,
            dslot: DeviceBuffer::from_slice(&(0..t).map(|r| (r % bsz) as i32).collect::<Vec<_>>())?,
            dcidx: DeviceBuffer::from_slice(&(0..t * bsz).map(|j| (j % bsz) as i32).collect::<Vec<_>>())?,
            markov_e: a(draft_lanes * cfg.dspark_markov_rank)?,
            markov_b: a(draft_lanes * cfg.vocab_size)?,
            markov_bl: a(if opts.tp.is_some() { draft_lanes * cfg.vocab_size } else { 1 })?,
            d_idx: i((bsz + 1) * draft_lanes)?,
            xpre: a(t * d)?,
            conf_x: f(t)?,
            conf_m: f(bsz * draft_lanes)?,
            s_temp: f(tm)?,
            s_topp: f(tm)?,
            nuc_ids: i(tm * NUC_CAP)?,
            nuc_p: f(tm * NUC_CAP)?,
            nuc_cnt: i(tm)?,
            d_temp: f(draft_lanes)?,
            d_topp: f(draft_lanes)?,
            d_u: f(bsz * draft_lanes)?,
            dq_ids: i(bsz * draft_lanes * DRAFT_NUCLEUS_K)?,
            dq_p: f(bsz * draft_lanes * DRAFT_NUCLEUS_K)?,
            dq_cnt: i(bsz * draft_lanes)?,
        };
        let no_qdq = std::env::var("TIDELM_NO_QDQ").as_deref() == Ok("1");
        let packed_kv = std::env::var("TIDELM_PACKED_KV").as_deref() != Ok("0");
        if packed_kv && (no_qdq || cfg.head_dim != 512) {
            return Err("packed KV stores the FP4-QDQ'd rows of 512-dim latents: set TIDELM_PACKED_KV=0 with TIDELM_NO_QDQ=1".into());
        }
        let mut caches = Caches {
            ring: Vec::new(),
            ckv: BTreeMap::new(),
            ik: BTreeMap::new(),
            crows: BTreeMap::new(),
            r2: BTreeMap::new(),
            mtp_ring: Vec::new(),
            dkv: a(lanes * bsz * hd)?,
            n_dev: i(lanes)?,
            rb_lane: i(lanes)?,
        };
        for _ in 0..n_layers {
            caches.ring.push(a(lanes * ring * hd)?);
        }
        if spec {
            for _ in 0..cfg.n_mtp_layers {
                caches.mtp_ring.push(a(lanes * ring * hd)?);
            }
        }
        // only the compressors this engine runs: a state for a layer past a truncation would
        // never see a chunk, and rollback would read its never-written chunk memory
        for &l in cfg.kv_source_layers.iter().filter(|&&l| l < n_layers) {
            let r = cfg.compress_ratios[l];
            let rows = max_seq / r + 1;
            caches.crows.insert(l, rows);
            // packed: kPkRow bytes a row, held in the same u16 buffer type (ckv_ls counts u16s)
            caches.ckv.insert(l, a(lanes * rows * if packed_kv { PACKED_ROW / 2 } else { hd })?);
            caches.ik.insert(l, a(lanes * rows * cfg.index_head_dim)?);
            if r > 1 {
                caches.r2.insert(l, Ratio2 {
                    pkv: f(lanes * hd)?,
                    psc: f(lanes * hd)?,
                    flag: i(lanes)?,
                    mem_kv: f(lanes * tm * hd)?,
                    mem_sc: f(lanes * tm * hd)?,
                    before_kv: f(lanes * hd)?,
                    before_sc: f(lanes * hd)?,
                    before_flag: i(lanes)?,
                    mem_start: i(lanes)?,
                });
            }
        }
        let e = cfg.n_routed_experts;
        let keep = match &opts.keep {
            Some(k) => {
                if k.len() < n_layers || k.iter().any(|row| row.len() != e) {
                    return Err(format!("keep mask needs {n_layers} rows of {e} experts").into());
                }
                let flat: Vec<u8> = k[..n_layers].iter().flat_map(|row| row.iter().map(|&b| b as u8)).collect();
                Some(DeviceBuffer::from_slice(&flat)?)
            }
            None => None,
        };
        if opts.tp.is_some() && !opts.resident {
            return Err("tensor parallel needs resident mode".into());
        }
        let expert_topp = expert_topp_from_env()?;
        if expert_topp.is_some() && !opts.resident {
            return Err("TIDELM_EXPERT_TOPP needs resident mode (its dropped picks route to the arena's null slot)".into());
        }
        let resident = if opts.resident {
            eprintln!("engine: {:.1} s before the resident load", t_new.elapsed().as_secs_f64());
            Some(Resident::load(&ckpt, &stream, &cfg, n_layers, &opts, &draft_run, draft_tp)?)
        } else {
            None
        };
        if draft_tp.is_some() && listed.is_none() && resident.as_ref().is_some_and(|r| r.draft_head.is_none()) {
            return Err("a split drafter needs the fp8 draft head".into());
        }
        // the markov bias head's rows for the same run (None: the whole head, in id order)
        let markov_rows = match (&dspark, listed.is_some() || draft_tp.is_some()) {
            (Some(ds), true) => {
                let r = cfg.dspark_markov_rank;
                let ids = DeviceBuffer::from_slice(&draft_run.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
                let rows = DeviceBuffer::<u16>::new(draft_run.len() * r)?;
                gather_rows(&stream, np(&ds.w.markov_head), &ids, np(&rows), draft_run.len(), r)?;
                stream.synchronize()?;
                Some(rows)
            }
            _ => None,
        };
        let draft_ids = match &listed {
            Some(ids) => Some((DeviceBuffer::from_slice(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>())?, ids.clone())),
            None => None,
        };
        let dspark_keep = match &opts.dspark_keep {
            Some(k) => {
                let e = cfg.dspark_n_routed_experts;
                if k.len() != cfg.n_mtp_layers || k.iter().any(|row| row.len() != e) {
                    return Err(format!("dspark keep mask needs {} rows of {e} experts", cfg.n_mtp_layers).into());
                }
                Some(DeviceBuffer::from_slice(&k.iter().flat_map(|r| r.iter().map(|&b| b as u8)).collect::<Vec<u8>>())?)
            }
            None => None,
        };
        let n_eng = cfg.engram_layer_ids.len();
        let eg_raw = (0..n_eng).map(|_| DeviceBuffer::zeroed(t * eng_cols * 264)).collect::<std::result::Result<_, _>>()?;
        let eg_inv = (0..n_eng).map(|_| DeviceBuffer::zeroed(t * eng_cols)).collect::<std::result::Result<_, _>>()?;
        let eg_host = (0..n_eng).map(|_| PinnedBuffer::zeroed(t * eng_cols * ENGRAM_ROW)).collect::<std::result::Result<_, _>>()?;
        // and the probe lane, the scratch lane's caches (Engine::probe_lane)
        let lanes_v = (0..=max_lanes).map(|_| Lane { len: 0, hash: hash.fresh(), last: (0, 0, 0), step: u64::MAX }).collect();
        // async hot reads: the backbone arena's spare slots are the ring. One ring and one plan
        // serve every lane: a decode step's router records the candidates of all its decode rows,
        // whichever lane they are, and the ring's live experts are routable for the whole step.
        // (The Python engine ran one decode path a lane, hence its DSV41_MAX_CONCURRENCY=1.)
        let hot = match (&resident, opts.hot_ring) {
            (_, 0) => None,
            (Some(res), ring) => {
                if opts.keep.is_none() {
                    return Err("hot reads need pruned residency (a keep mask)".into());
                }
                let hcfg = crate::hot::HotConfig::from_env(ring, cfg.n_activated_experts)?;
                let set = &res.experts;
                let slots: Vec<usize> = (0..set.owner.len()).filter(|&i| set.owner[i].is_none()).collect();
                if slots.len() != ring {
                    return Err(format!("{} spare arena slots for a ring of {ring}", slots.len()).into());
                }
                Some(Hot::new(hcfg, slots, model_dir, n_layers, cfg.n_activated_experts, set)?)
            }
            (None, _) => return Err("hot reads need resident mode".into()),
        };
        // per lane, both halves (counts, mass) of every backbone layer's experts
        let (demand, miss) = if opts.record_demand && opts.resident {
            let n = 2 * n_layers * cfg.n_routed_experts;
            // and the probe lane's (Engine::probe_lane)
            (Some(DeviceBuffer::<f64>::zeroed((max_lanes + 1) * n)?), Some(DeviceBuffer::<f64>::zeroed((max_lanes + 1) * 8)?))
        } else {
            (None, None)
        };
        // vision: the delimiters on every rank, the tower where images are encoded - rank 1 of a
        // pair (the profile's DSV41_VISION_MODE=peer: rank 0 keeps its memory for the server),
        // else this box
        // the replay needs the decoder half: a truncated test model runs every layer
        let swa_replay = opts.swa_replay && cfg.candidate_source_layer >= 0 && (cfg.candidate_source_layer as usize) + 1 < n_layers;
        let cfg_w = cfg.window_size;
        let (vis_delims, tower) = if opts.vision {
            let delims = crate::vision::load_delimiters(&ckpt, &stream, &cfg)?;
            let tower = match opts.tp.map(|tp| tp.rank) {
                Some(0) => None,
                _ => {
                    let tv = Instant::now();
                    let tw = crate::vision::Tower::load(&ckpt, &stream, &cfg)?;
                    eprintln!("vision: tower {:.2} GB in {:.1} s", tw.bytes() as f64 / 1e9, tv.elapsed().as_secs_f64());
                    Some(tw)
                }
            };
            (Some(delims), tower)
        } else {
            (None, None)
        };
        Ok(Engine {
            arena: Arena::new(ARENA_SLOTS + 1, None)?,
            expert_stage: PinnedBuffer::uninit(STAGE_BYTES)?,
            logits_host: PinnedBuffer::zeroed(tm * cfg.vocab_size)?,
            am_host: PinnedBuffer::zeroed(tm)?,
            nuc_cnt_host: PinnedBuffer::zeroed(tm)?,
            nuc_ids_host: PinnedBuffer::zeroed(tm * NUC_CAP)?,
            nuc_p_host: PinnedBuffer::zeroed(tm * NUC_CAP)?,
            cfg,
            ckpt,
            stream,
            rope_w,
            rope_c,
            norm,
            hash,
            engram_tables,
            act,
            caches,
            pair: None,
            tp: opts.tp,
            leader: false,
            lanes: lanes_v,
            max_lanes,
            step_no: 0,
            dspark,
            resident,
            keep,
            demand,
            miss,
            hot,
            peer_hot: None,
            snaps: HashMap::new(),
            snap_free: Vec::new(),
            snap_idx: DeviceBuffer::zeroed(cfg_w.max(1))?,
            disk: None,
            disk_ns: 0,
            vis_delims,
            tower,
            lane_vis: (0..=max_lanes).map(|_| None).collect(),
            vis_idx: DeviceBuffer::zeroed(t)?,
            vis_img: DeviceBuffer::zeroed(t)?,
            vis_splices: Vec::new(),
            swa_replay,
            rep_h: DeviceBuffer::zeroed(if swa_replay { (max_lanes + 1) * cfg_w * hc * d } else { 1 })?,
            rep_pre: DeviceBuffer::zeroed(if swa_replay { (max_lanes + 1) * cfg_w * hc } else { 1 })?,
            rep_idx: DeviceBuffer::zeroed(t)?,
            win_floor: DeviceBuffer::zeroed(max_lanes + 1)?,
            floors: vec![0; max_lanes + 1],
            dspark_keep,
            draft_graphs: HashMap::new(),
            draft_cols: draft_vocab.len(),
            draft_ids,
            markov_rows,
            draft_conf: Vec::new(),
            n_layers,
            cpos1: i(t)?,
            cpos2: i(t)?,
            one_hot: DeviceBuffer::from_slice(&(0..t * hc).map(|j| if j % hc == 0 { 1.0f32 } else { 0.0 }).collect::<Vec<_>>())?,
            eg_raw,
            eg_inv,
            eg_host,
            eg_jobs: Vec::new(),
            eg_prefetch: Vec::new(),
            shape: None,
            live_rows: 0,
            use_graphs: opts.graphs && opts.resident,
            graphs: HashMap::new(),
            graph_tick: 0,
            graph_cap: std::env::var("TIDELM_GRAPH_CACHE").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n >= 1).unwrap_or(128),
            graph_replays: 0,
            graph_captures: 0,
            max_seq,
            times: StepTimes::default(),
            dump_dir: std::env::var_os("TIDELM_DUMP_DIR").map(std::path::PathBuf::from),
            no_qdq,
            packed_kv,
            selection: Selection::AllVisible,
            cand_live: false,
            indexed_layers: 0,
            nucleus_fallbacks: 0,
            expert_topp,
            ring,
            pmax,
            eager_step: false,
        })
    }

    /// Join a pair before the first step (its MoE combines all-reduce over `pair.comm`).
    pub fn attach_pair(&mut self, pair: Pair) -> Result<()> {
        if self.tp.map(|t| t.rank) != Some(pair.comm.rank) {
            return Err("attach_pair: the engine was not built for this rank's tensor-parallel shard".into());
        }
        self.leader = pair.comm.rank == 0;
        self.pair = Some(pair);
        Ok(())
    }

    pub fn is_paired(&self) -> bool {
        self.pair.is_some()
    }

    pub fn rank(&self) -> usize {
        self.pair.as_ref().map(|p| p.comm.rank).unwrap_or(0)
    }

    /// Rank 0: send the call to rank 1 before running it, so both issue the same device
    /// work (and reach the same collectives) in the same order.
    fn mirror(&mut self, msg: impl FnOnce(&mut tidelm_comm::Msg)) -> Result<()> {
        if self.leader {
            let mut m = tidelm_comm::Msg::default();
            msg(&mut m);
            self.pair.as_mut().unwrap().ctl.send(&m.0)?;
        }
        Ok(())
    }

    /// Rank 0: let rank 1's `follow` return.
    pub fn release_follower(&mut self) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_SHUTDOWN);
        })
    }

    /// Rank 1: replay rank 0's calls until it releases us. Nothing here decides anything.
    pub fn follow(&mut self) -> Result<()> {
        loop {
            let msg = self.pair.as_mut().ok_or("follow needs a pair")?.ctl.recv()?;
            let mut r = tidelm_comm::Reader::new(&msg);
            match r.u32()? {
                OP_FORWARD => {
                    let stage = match r.u32()? {
                        0 => Stage::Full,
                        1 => Stage::Encoder,
                        _ => return Err("rank 1: a forward step's stage is Full or Encoder".into()),
                    };
                    let n = r.u32()? as usize;
                    let mut owned: Vec<(usize, Want, bool, Vec<u32>, Vec<u32>, Option<Vec<i64>>)> = Vec::with_capacity(n);
                    for _ in 0..n {
                        let lane = r.u32()? as usize;
                        let want = match r.u32()? {
                            0 => Want::Nothing,
                            1 => Want::Last,
                            _ => Want::All,
                        };
                        let decode = r.u32()? != 0;
                        let ids = r.u32s()?;
                        let next = r.u32s()?;
                        let has = r.u32()? != 0;
                        let nh = r.u32()? as usize;
                        let h: Vec<i64> = (0..nh).map(|_| r.u64().map(|x| x as i64)).collect::<std::result::Result<_, _>>()?;
                        owned.push((lane, want, decode, ids, next, has.then_some(h)));
                    }
                    let segs: Vec<Seg> = owned
                        .iter()
                        .map(|(lane, want, decode, ids, next, h)| Seg {
                            lane: *lane, ids, want: *want, decode: *decode, next, hashes: h.as_deref(),
                        })
                        .collect();
                    self.peer_hot = Some(r.u64()?);
                    // what comes back is rank 0's business: the cheapest read-back
                    self.run_step(&segs, Readback::Argmax, stage)?;
                }
                OP_DRAFT => {
                    let n = r.u32()? as usize;
                    let reqs: Vec<(usize, u32, usize)> =
                        (0..n).map(|_| Ok((r.u32()? as usize, r.u32()?, r.u64()? as usize))).collect::<Result<_>>()?;
                    self.draft_batch(&reqs)?;
                }
                OP_DRAFT_SAMPLED => {
                    let n = r.u32()? as usize;
                    let reqs: Vec<(usize, u32, usize)> =
                        (0..n).map(|_| Ok((r.u32()? as usize, r.u32()?, r.u64()? as usize))).collect::<Result<_>>()?;
                    let sm: Vec<Sampling> = (0..n)
                        .map(|_| Ok(Sampling { temperature: f32::from_bits(r.u32()?), top_p: f32::from_bits(r.u32()?) }))
                        .collect::<Result<_>>()?;
                    let u: Vec<f32> = r.u32s()?.into_iter().map(f32::from_bits).collect();
                    self.draft_inner(&reqs, Some((&sm, &u)))?;
                }
                OP_ROLLBACK => {
                    let n = r.u32()? as usize;
                    let reqs: Vec<(usize, usize)> = (0..n).map(|_| Ok((r.u32()? as usize, r.u64()? as usize))).collect::<Result<_>>()?;
                    self.rollback_batch(&reqs)?;
                }
                OP_SEED => {
                    let n = r.u32()? as usize;
                    let reqs: Vec<(usize, usize)> = (0..n).map(|_| Ok((r.u32()? as usize, r.u64()? as usize))).collect::<Result<_>>()?;
                    self.seed_draft_batch(&reqs)?;
                }
                OP_RESET_LANE => {
                    let lane = r.u32()? as usize;
                    self.reset_lane(lane)?;
                }
                OP_GRAPHS => {
                    let on = r.u32()? != 0;
                    self.set_graphs(on);
                }
                OP_MOVE_EXPERTS => {
                    let n = r.u32()? as usize;
                    let moves: Vec<[usize; 4]> = (0..n)
                        .map(|_| Ok([r.u32()? as usize, r.u32()? as usize, r.u32()? as usize, r.u32()? as usize]))
                        .collect::<Result<_>>()?;
                    self.move_experts(&moves)?;
                }
                OP_SNAPSHOT => {
                    let (lane, n, id) = (r.u32()? as usize, r.u64()? as usize, r.u64()?);
                    self.prefix_snapshot(lane, n, id)?;
                }
                OP_RESTORE => {
                    let (lane, id) = (r.u32()? as usize, r.u64()?);
                    self.prefix_restore(lane, id)?;
                }
                OP_DISK_SAVE => {
                    let lane = r.u32()? as usize;
                    let id = String::from_utf8(r.u32s()?.iter().map(|&b| b as u8).collect()).map_err(|_| "disk save: bad id")?;
                    let tokens = r.u32s()?;
                    let n = r.u32()? as usize;
                    let snaps: Vec<u64> = (0..n).map(|_| r.u64()).collect::<std::result::Result<_, _>>()?;
                    self.prefix_disk_save(lane, &id, &tokens, &snaps)?;
                }
                OP_DISK_LOAD => {
                    let lane = r.u32()? as usize;
                    let id = String::from_utf8(r.u32s()?.iter().map(|&b| b as u8).collect()).map_err(|_| "disk load: bad id")?;
                    let (n, snap) = (r.u64()? as usize, r.u64()?);
                    let ok = match self.disk_load_inner(lane, &id, n, snap) {
                        Ok(_) => 1,
                        Err(e) => {
                            eprintln!("rank 1: prefix disk load {id}: {e}");
                            0
                        }
                    };
                    let mut m = tidelm_comm::Msg::default();
                    m.u32(ok);
                    self.pair.as_mut().ok_or("follow needs a pair")?.ctl.send(&m.0)?;
                }
                OP_SNAP_DROP => {
                    let lane = r.u32()? as usize;
                    let n = r.u32()? as usize;
                    let ids: Vec<u64> = (0..n).map(|_| r.u64()).collect::<std::result::Result<_, _>>()?;
                    self.prefix_drop(lane, &ids)?;
                }
                OP_VISION => {
                    let lane = r.u32()? as usize;
                    let n = r.u32()? as usize;
                    let mut images = Vec::with_capacity(n);
                    for _ in 0..n {
                        let (start, vit_h, vit_w) = (r.u64()? as usize, r.u32()? as usize, r.u32()? as usize);
                        images.push(crate::vision::ImageIn { start, vit_h, vit_w, patches: r.u16s()? });
                    }
                    let need: Vec<bool> = images.iter().map(|im| !im.patches.is_empty()).collect();
                    self.set_images_inner(lane, &images, &need)?;
                }
                OP_REPLAY => {
                    let lanes: Vec<usize> = r.u32s()?.into_iter().map(|l| l as usize).collect();
                    self.run_replay(&lanes, Readback::Argmax)?;
                }
                OP_SHUTDOWN => return Ok(()),
                op => return Err(format!("rank 1: unknown control op {op}").into()),
            }
        }
    }

    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    /// A decode step's start (fastdecode._hot_async_begin): plan from the last decode step's
    /// candidates, take ring slots and start their reads, and mark every live ring expert
    /// routable (LUT -> its slot, keep -> 1) for this step's graphs.
    fn hot_begin(&mut self) -> Result<()> {
        let n_layers = self.n_layers;
        let st = &self.stream;
        let (Some(hot), Some(res), Some(keep)) = (self.hot.as_mut(), self.resident.as_ref(), self.keep.as_ref()) else {
            return Ok(());
        };
        let e = res.experts.n_experts;
        let taken = hot.ring.taken();
        let plan = crate::hot::plan_async(&hot.cfg, &hot.table_host, n_layers, &taken, &mut hot.stats);
        for (l, x) in plan {
            let slot = hot.ring.take((l, x));
            let (tx, rx) = std::sync::mpsc::channel();
            let job = HotJob { prefix: format!("layers.{l}.ffn.experts.{x}."), dsts: hot.dsts[&slot], done: tx };
            hot.jobs.send(job).map_err(|_| "hot reads: the reader threads are gone")?;
            hot.waits.push(rx);
        }
        let marks: Vec<i32> =
            hot.ring.live.iter().flat_map(|(&(l, x), &slot)| [(l * (e + 1) + x) as i32, (l * e + x) as i32, slot as i32]).collect();
        hot.n_marks = marks.len() / 3;
        if hot.n_marks > 0 {
            hot.marks.upload_async(0, &marks, st)?;
            ck(unsafe { ds_hot_marks(st.raw(), np(&res.experts.lut), np(keep), np(&hot.marks), np(&hot.saved), hot.n_marks as i32, 1) }, "hot marks")?;
        }
        Ok(())
    }

    /// A decode step's end (fastdecode._hot_async_end): undo the marks after its graphs, copy
    /// its candidate table to the host (final at the step's read-back), wait for the reads it
    /// started, and make them routable from the next step.
    fn hot_end(&mut self) -> Result<()> {
        let st = &self.stream;
        let (Some(hot), Some(res), Some(keep)) = (self.hot.as_mut(), self.resident.as_ref(), self.keep.as_ref()) else {
            return Ok(());
        };
        if hot.n_marks > 0 {
            ck(unsafe { ds_hot_marks(st.raw(), np(&res.experts.lut), np(keep), np(&hot.marks), np(&hot.saved), hot.n_marks as i32, 0) }, "hot unmarks")?;
            hot.n_marks = 0;
        }
        hot.table.download_async(0, &mut hot.table_host[..], st)?;
        let t0 = Instant::now();
        for rx in hot.waits.drain(..) {
            // one rank without the expert would route the same rows to garbage
            rx.recv().map_err(|_| "hot reads: a reader thread died")?.map_err(|e| format!("hot read failed: {e}"))?;
        }
        hot.stats.wait_s += t0.elapsed().as_secs_f64();
        hot.ring.promote();
        Ok(())
    }

    /// Async hot reads' counters since boot, if on.
    pub fn hot_stats(&self) -> Option<crate::hot::HotStats> {
        self.hot.as_ref().map(|h| h.stats.clone())
    }

    /// Lane `lane`'s recorded demand since it was last taken (Options::record_demand), after
    /// every queued step; `take` zeroes it (a request's demand is folded once). Rank-local:
    /// rank 0 plans, so nothing here is mirrored.
    pub fn lane_demand(&mut self, lane: usize, take: bool) -> Result<LaneDemand> {
        let (Some(dm), Some(ms)) = (&self.demand, &self.miss) else {
            return Err("demand is recorded only with Options::record_demand (resident)".into());
        };
        if lane > self.max_lanes {
            return Err(format!("lane {lane} of {}", self.max_lanes).into());
        }
        let n = self.n_layers * self.cfg.n_routed_experts;
        let mut both = vec![0f64; 2 * n];
        let mut miss = [0f64; 8];
        dm.download_async(lane * 2 * n, &mut both, &self.stream)?;
        ms.download_async(lane * 8, &mut miss, &self.stream)?;
        self.stream.synchronize()?;
        if take {
            dm.upload_async(lane * 2 * n, &vec![0f64; 2 * n], &self.stream)?;
            ms.upload_async(lane * 8, &[0f64; 8], &self.stream)?;
            self.stream.synchronize()?;
        }
        let mass = both.split_off(n);
        Ok(LaneDemand { counts: both, mass, miss })
    }

    /// The engram n-gram hashes of `prompt`'s rows at `positions` ([row][engram layer][col]),
    /// each from its own place in the prompt (a fresh hash state over the whole prompt): a
    /// probe excerpt's rows (Seg::hashes) see no n-gram across the excerpt's joins.
    /// Whether this engine takes image inputs (Options::vision).
    pub fn vision(&self) -> bool {
        self.vis_delims.is_some()
    }

    /// Lane `lane`'s request shows `images`: the rows of their spans for its prefill from
    /// position `from` on (an image wholly before it - a restored prefix - is not encoded). The
    /// tower runs on its rank; in a pair rank 1 sends rank 0 the rows over the control channel,
    /// so both splice the same bits. A failure leaves the lane without images on both ranks
    /// and errors here. A text request (no images) clears the lane's.
    pub fn set_images(&mut self, lane: usize, images: &[crate::vision::ImageIn], from: usize) -> Result<()> {
        if lane >= self.max_lanes {
            return Err(format!("lane {lane} of {}", self.max_lanes).into());
        }
        // both ranks hold the same lane state: nothing to say when there is none to clear
        if images.is_empty() && self.lane_vis[lane].is_none() {
            return Ok(());
        }
        if self.vis_delims.is_none() {
            return Err("this engine was started without vision".into());
        }
        let r = self.cfg.vision_downsample_ratio;
        let need: Vec<bool> = images.iter().map(|im| im.start + crate::vision::span_len(im.vit_h, im.vit_w, r) > from).collect();
        self.mirror(|m| {
            m.u32(OP_VISION).u32(lane as u32).u32(images.len() as u32);
            for (im, &nd) in images.iter().zip(&need) {
                m.u64(im.start as u64).u32(im.vit_h as u32).u32(im.vit_w as u32).u16s(if nd { &im.patches } else { &[] });
            }
        })?;
        self.set_images_inner(lane, images, &need)
    }

    fn set_images_inner(&mut self, lane: usize, images: &[crate::vision::ImageIn], need: &[bool]) -> Result<()> {
        self.lane_vis[lane] = None;
        if images.is_empty() {
            return Ok(());
        }
        let (d, r) = (self.cfg.dim, self.cfg.vision_downsample_ratio);
        let cells: Vec<usize> = images.iter().map(|im| im.vit_h.div_ceil(r) * im.vit_w.div_ceil(r)).collect();
        let total: usize = cells.iter().zip(need).filter(|(_, &nd)| nd).map(|(&c, _)| c).sum();
        let table = DeviceBuffer::<u16>::new((total + 3) * d)?;
        let encoded: Result<()> = match &self.tower {
            Some(tower) => (|| {
                let mut at = 0;
                for (im, &nd) in images.iter().zip(need) {
                    if nd {
                        let rows = tower.forward(&self.stream, &im.patches, im.vit_h, im.vit_w)?;
                        table.copy_from(at * d, &rows, 0, rows.len(), &self.stream)?;
                        at += rows.len() / d;
                    }
                }
                self.stream.synchronize()?;
                Ok(())
            })(),
            None => Ok(()),
        };
        match (self.pair.is_some(), self.leader) {
            // rank 1: rank 0 gets the rows, or why there are none (this rank keeps following)
            (true, false) => {
                let mut m = tidelm_comm::Msg::default();
                match &encoded {
                    Ok(()) => {
                        m.u32(1).u16s(&table.download()?[..total * d]);
                    }
                    Err(e) => {
                        eprintln!("rank 1: vision tower: {e}");
                        m.u32(0).u32s(&e.to_string().bytes().map(|b| b as u32).collect::<Vec<_>>());
                    }
                }
                self.pair.as_mut().unwrap().ctl.send(&m.0)?;
                if encoded.is_err() {
                    return Ok(());
                }
            }
            // rank 0: the rows rank 1 encoded
            (true, true) if self.tower.is_none() => {
                let msg = self.pair.as_mut().unwrap().ctl.recv()?;
                let mut rd = tidelm_comm::Reader::new(&msg);
                if rd.u32()? != 1 {
                    let why = String::from_utf8(rd.u32s()?.iter().map(|&b| b as u8).collect()).unwrap_or_default();
                    return Err(format!("the vision tower (rank 1): {why}").into());
                }
                let rows = rd.u16s()?;
                if rows.len() != total * d {
                    return Err(format!("rank 1 sent {} image row values for {total} rows", rows.len()).into());
                }
                table.upload_async(0, &rows, &self.stream)?;
            }
            _ => {}
        }
        encoded?;
        table.copy_from(total * d, self.vis_delims.as_ref().unwrap(), 0, 3 * d, &self.stream)?;
        self.stream.synchronize()?;
        let mut spans = Vec::with_capacity(images.len());
        let mut at = 0;
        for ((im, &nd), &c) in images.iter().zip(need).zip(&cells) {
            spans.push((im.start, im.vit_h, im.vit_w, nd.then_some(at)));
            if nd {
                at += c;
            }
        }
        self.lane_vis[lane] = Some(LaneVision { spans, table, delims: total });
        Ok(())
    }

    pub fn prompt_hashes(&self, prompt: &[u32], positions: &[usize]) -> Vec<i64> {
        let mut h = self.hash.fresh();
        let all = h.hash(prompt, 0, None);
        let per = all.len() / prompt.len().max(1);
        positions.iter().flat_map(|&p| all[p * per..(p + 1) * per].iter().copied()).collect()
    }

    /// The lane prompts may use to probe before their prefill (decode.rs): the scratch lane's
    /// caches (filler rows sit far past any probe's positions), its own demand slot.
    pub fn probe_lane(&self) -> usize {
        self.max_lanes
    }

    /// Snapshot lane `lane` at length `n` as `id` (the RAM prefix cache): n is the lane's
    /// length, or lies within its last step's segment (the pending state there is rollback's).
    /// Rank 0 mirrors it; both ranks keep the same snapshots.
    pub fn prefix_snapshot(&mut self, lane: usize, n: usize, id: u64) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_SNAPSHOT).u32(lane as u32).u64(n as u64).u64(id);
        })?;
        if lane >= self.max_lanes {
            return Err(format!("snapshot: lane {lane} of {}", self.max_lanes).into());
        }
        let l = &self.lanes[lane];
        if n == 0 || n > l.len || (n < l.len && n < l.last.0) {
            return Err(format!("snapshot({n}) of lane {lane}: its length is {}, its last segment from {}", l.len, l.last.0).into());
        }
        let lane_len = l.len;
        let (hd, w) = (self.cfg.head_dim, n.min(self.cfg.window_size));
        let mut snap = self.snap_take()?;
        let st = &self.stream;
        let sr = st.raw();
        let ring = self.ring;
        let rings: Vec<*mut c_void> = self.caches.ring.iter().chain(self.caches.mtp_ring.iter()).map(np).collect();
        self.snap_idx.upload_async(0, &(n - w..n).map(|p| (lane * ring + p % ring) as i32).collect::<Vec<_>>(), st)?;
        let (rows, pending, flags) = (&snap.rows, &snap.pending, &snap.flags);
        for (k, &rg) in rings.iter().enumerate() {
            ck(unsafe { ds_gather_rows_u16(sr, rg, np(&self.snap_idx), rows.ptr_at(k * w * hd), w as i32, hd as i32) }, "snapshot rows")?;
        }
        for (i, r2) in self.caches.r2.values().enumerate() {
            if n == lane_len {
                pending.copy_from(i * 2 * hd, &r2.pkv, lane * hd, hd, st)?;
                pending.copy_from(i * 2 * hd + hd, &r2.psc, lane * hd, hd, st)?;
                flags.copy_from(i, &r2.flag, lane, 1, st)?;
            } else {
                ck(unsafe {
                    ds_pending_snapshot(sr, n as i32, lane as i32, np(&r2.mem_kv), np(&r2.mem_sc), np(&r2.mem_start), np(&r2.before_kv),
                                        np(&r2.before_sc), np(&r2.before_flag), pending.ptr_at(i * 2 * hd), flags.ptr_at(i),
                                        hd as i32, TMAX as i32)
                }, "snapshot pending")?;
            }
        }
        if self.swa_replay {
            // the lane's kept replay rows
            let (w, hc, d) = (self.cfg.window_size, self.cfg.hc_mult, self.cfg.dim);
            snap.rep_h.copy_from(0, &self.rep_h, lane * w * hc * d, w * hc * d, st)?;
            snap.rep_pre.copy_from(0, &self.rep_pre, lane * w * hc, w * hc, st)?;
        }
        // the index buffer is the next snapshot's: the gathers must be done with it
        self.stream.synchronize()?;
        (snap.n, snap.w, snap.floor) = (n, w, self.floors[lane]);
        self.snap_put((lane, id), snap);
        Ok(())
    }

    /// One snapshot's buffers, sized for a whole window (snapshots of shorter lanes use the
    /// front of each ring's block: row k * w * hd on).
    fn snap_alloc(&self) -> Result<LaneSnap> {
        let (hd, w, hc, d) = (self.cfg.head_dim, self.cfg.window_size, self.cfg.hc_mult, self.cfg.dim);
        let rings = self.caches.ring.len() + self.caches.mtp_ring.len();
        let nr = self.caches.r2.len().max(1);
        let rep = if self.swa_replay { (w * hc * d, w * hc) } else { (1, 1) };
        Ok(LaneSnap { n: 0, w: 0, rows: DeviceBuffer::zeroed((rings * w * hd).max(1))?, pending: DeviceBuffer::zeroed(nr * 2 * hd)?,
                      flags: DeviceBuffer::zeroed(nr)?, rep_h: DeviceBuffer::zeroed(rep.0)?, rep_pre: DeviceBuffer::zeroed(rep.1)?,
                      floor: 0 })
    }

    /// Allocate `n` snapshots' buffers now (at boot: the RAM prefix cache's lanes x (kept
    /// snapshots + the prompt's own + a restore's)), so the cache takes no memory while serving.
    pub fn reserve_snapshots(&mut self, n: usize) -> Result<()> {
        while self.snap_free.len() < n {
            let s = self.snap_alloc()?;
            self.snap_free.push(s);
        }
        Ok(())
    }

    /// A snapshot's buffers from the free list, its pending state zeroed: ds_pending_snapshot
    /// writes only the flag at an even length and nothing for a position outside the chunk
    /// memory, so a reused slot would otherwise restore the last owner's pending row (a fresh
    /// snapshot's was zero; reusing them unzeroed measured code 46.9 -> 33.7 tok/s after a
    /// session, drafts accepted 3.5 -> 1.9 a step).
    fn snap_take(&mut self) -> Result<LaneSnap> {
        match self.snap_free.pop() {
            Some(s) => {
                s.pending.zero_async(&self.stream)?;
                s.flags.zero_async(&self.stream)?;
                Ok(s)
            }
            None => self.snap_alloc(),
        }
    }

    /// Keep `snap` as `key`; a snapshot it replaces goes back to the free list.
    fn snap_put(&mut self, key: (usize, u64), snap: LaneSnap) {
        if let Some(old) = self.snaps.insert(key, snap) {
            self.snap_free.push(old);
        }
    }

    /// Restore lane `lane` to snapshot `id`: its window rows and pending state back, its length
    /// the snapshot's. The compressed rows below it must still be the lane's (the caller's
    /// match: the lane has run nothing but this prefix's continuations since).
    pub fn prefix_restore(&mut self, lane: usize, id: u64) -> Result<usize> {
        self.mirror(|m| {
            m.u32(OP_RESTORE).u32(lane as u32).u64(id);
        })?;
        self.restore_inner(lane, id)
    }

    fn restore_inner(&mut self, lane: usize, id: u64) -> Result<usize> {
        let snap = self.snaps.get(&(lane, id)).ok_or_else(|| format!("restore: lane {lane} has no snapshot {id}"))?;
        let (n, w, hd) = (snap.n, snap.w, self.cfg.head_dim);
        let st = &self.stream;
        let sr = st.raw();
        let ring = self.ring;
        let rings: Vec<*mut c_void> = self.caches.ring.iter().chain(self.caches.mtp_ring.iter()).map(np).collect();
        self.snap_idx.upload_async(0, &(n - w..n).map(|p| (lane * ring + p % ring) as i32).collect::<Vec<_>>(), st)?;
        for (k, &rg) in rings.iter().enumerate() {
            ck(unsafe { ds_scatter_rows_u16(sr, snap.rows.ptr_at(k * w * hd), np(&self.snap_idx), rg, w as i32, hd as i32) }, "restore rows")?;
        }
        for (i, r2) in self.caches.r2.values().enumerate() {
            r2.pkv.copy_from(lane * hd, &snap.pending, i * 2 * hd, hd, st)?;
            r2.psc.copy_from(lane * hd, &snap.pending, i * 2 * hd + hd, hd, st)?;
            r2.flag.copy_from(lane, &snap.flags, i, 1, st)?;
        }
        if self.swa_replay {
            let (w, hc, d) = (self.cfg.window_size, self.cfg.hc_mult, self.cfg.dim);
            self.rep_h.copy_from(lane * w * hc * d, &snap.rep_h, 0, w * hc * d, st)?;
            self.rep_pre.copy_from(lane * w * hc, &snap.rep_pre, 0, w * hc, st)?;
        }
        self.floors[lane] = snap.floor;
        self.win_floor.upload_async(0, &self.floors, st)?;
        st.synchronize()?;
        let l = &mut self.lanes[lane];
        l.len = n;
        l.last = (n, 0, 0);
        l.step = u64::MAX;
        Ok(n)
    }

    /// The prefix cache's disk tier in `cfg.dir/rank-R` (both ranks of a pair, after the join):
    /// its namespace is this executable, the model's config and the settings that change what
    /// a bundle holds.
    pub fn enable_prefix_disk(&mut self, cfg: crate::prefix_disk::DiskConfig) -> Result<()> {
        let fnv = |h: u64, bytes: &[u8]| bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        let exe = std::fs::read("/proc/self/exe")?;
        let mut h = fnv(0xcbf29ce484222325, &exe);
        let cfg_json = std::fs::read(self.ckpt.model_dir().join("inference/config.json")).or_else(|_| std::fs::read(self.ckpt.model_dir().join("config.json")))?;
        h = fnv(h, &cfg_json);
        let settings = format!("layers {} spec {} packed {} qdq {} ring {} window {} rank {} replay {}", self.n_layers, self.dspark.is_some(),
                               self.packed_kv, !self.no_qdq, self.ring, self.cfg.window_size, self.rank(), self.swa_replay);
        h = fnv(h, settings.as_bytes());
        self.disk_ns = h;
        self.disk = Some(crate::prefix_disk::Disk::open(cfg, self.rank())?);
        Ok(())
    }

    /// The disk tier's bundles that restore a prefix of `prompt` of at most `max_n` tokens.
    pub fn prefix_disk_candidates(&self, prompt: &[u32], max_n: usize) -> Vec<(usize, String)> {
        self.disk.as_ref().map(|d| d.candidates(prompt, self.disk_ns, max_n)).unwrap_or_default()
    }

    /// Whether a save is due (DSV41_PREFIX_DISK_MIN_INTERVAL_S since the last one), and a
    /// bundle that already restores all of `prefixes` (dedup; touched, so it stays).
    pub fn prefix_disk_due(&self) -> bool {
        self.disk.as_ref().is_some_and(|d| d.last_save.is_none_or(|t| t.elapsed().as_secs_f64() >= d.cfg.min_interval_s))
    }

    pub fn prefix_disk_duplicate(&self, tokens: &[u32], ns: &[usize]) -> bool {
        let Some(d) = self.disk.as_ref() else { return false };
        if !d.cfg.dedup {
            return false;
        }
        let hs = crate::prefix_disk::prefix_hashes(tokens);
        let want: Vec<(usize, u64)> = ns.iter().map(|&n| (n, hs[n])).collect();
        match d.duplicate(self.disk_ns, &want) {
            Some(id) => {
                d.touch(&id);
                true
            }
            None => false,
        }
    }

    /// Save lane `lane`'s prompt `tokens` with its snapshots `snap_ids` as bundle `id`: the
    /// snapshots and the compressed rows below the longest are copied to the host now (the
    /// lane must not have moved past them), the write happens on the disk tier's thread.
    /// Mirrored: each rank saves its own copy.
    pub fn prefix_disk_save(&mut self, lane: usize, id: &str, tokens: &[u32], snap_ids: &[u64]) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_DISK_SAVE).u32(lane as u32).u32s(&id.bytes().map(|b| b as u32).collect::<Vec<_>>()).u32s(tokens)
                .u32(snap_ids.len() as u32);
            for &s in snap_ids {
                m.u64(s);
            }
        })?;
        if self.disk.is_none() {
            return Ok(());
        }
        // the bundle as `encode` lays it out: header fields on the host, the snapshots' and the
        // compressed rows' bytes where they are on the device (save_parts streams them)
        use crate::prefix_disk::{le_u64, Part};
        let (hd, ihd) = (self.cfg.head_dim, self.cfg.index_head_dim);
        let rings = self.caches.ring.len() + self.caches.mtp_ring.len();
        let mut parts: Vec<Part> = Vec::new();
        let mut head = crate::prefix_disk::MAGIC.to_vec();
        let dev = |parts: &mut Vec<Part>, head: &mut Vec<u8>, ptr: *mut c_void, n: usize, size: usize| {
            le_u64(head, n as u64);
            parts.push(Part::Bytes(std::mem::take(head)));
            parts.push(Part::Device { ptr: ptr as usize, len: n * size });
        };
        le_u64(&mut head, self.disk_ns);
        le_u64(&mut head, id.len() as u64);
        head.extend_from_slice(id.as_bytes());
        le_u64(&mut head, tokens.len() as u64);
        head.extend(tokens.iter().flat_map(|t| t.to_le_bytes()));
        le_u64(&mut head, snap_ids.len() as u64);
        let mut n_max = 0;
        let mut ns = Vec::new();
        for &sid in snap_ids {
            let sn = self.snaps.get(&(lane, sid)).ok_or_else(|| format!("disk save: lane {lane} has no snapshot {sid}"))?;
            n_max = n_max.max(sn.n);
            ns.push(sn.n);
            le_u64(&mut head, sn.n as u64);
            le_u64(&mut head, sn.w as u64);
            dev(&mut parts, &mut head, sn.rows.ptr_at(0), (rings * sn.w * hd).max(1), 2);
            dev(&mut parts, &mut head, sn.pending.ptr_at(0), sn.pending.len(), 4);
            dev(&mut parts, &mut head, sn.flags.ptr_at(0), sn.flags.len(), 4);
            let rep = if self.swa_replay { (sn.rep_h.len(), sn.rep_pre.len()) } else { (0, 0) };
            dev(&mut parts, &mut head, sn.rep_h.ptr_at(0), rep.0, 2);
            dev(&mut parts, &mut head, sn.rep_pre.ptr_at(0), rep.1, 4);
            le_u64(&mut head, sn.floor as i64 as u64);
        }
        let row = if self.packed_kv { PACKED_ROW / 2 } else { hd };
        for (caches, width) in [(&self.caches.ckv, row), (&self.caches.ik, ihd)] {
            le_u64(&mut head, caches.len() as u64);
            for (&l, buf) in caches {
                let (r, rows) = (self.cfg.compress_ratios[l], self.caches.crows[&l]);
                le_u64(&mut head, l as u64);
                // stored as bytes (encode's Vec<u8>): the length is a byte count
                dev(&mut parts, &mut head, buf.ptr_at(lane * rows * width), (n_max / r) * width * 2, 1);
            }
        }
        parts.push(Part::Bytes(head));
        let hs = crate::prefix_disk::prefix_hashes(tokens);
        let prefixes = ns.iter().map(|&n| (n, hs[n])).collect();
        let st = &self.stream;
        let mut copy = |dst: *mut u8, src: usize, len: usize| -> Result<()> {
            unsafe { memcpy_async(dst as *mut c_void, src as *const c_void, len, MEMCPY_DEVICE_TO_HOST, st)? };
            Ok(st.synchronize()?)
        };
        let d = self.disk.as_mut().unwrap();
        d.save_parts(id, self.disk_ns, prefixes, &parts, &mut copy)?;
        d.last_save = Some(Instant::now());
        Ok(())
    }

    /// Restore lane `lane` (reset) to length `n` from bundle `id` as snapshot `snap`: its rows
    /// into the lane's caches, the snapshot restored, the n-gram history re-hashed. Rank 0
    /// checks the bundle's tokens against `prompt` and both ranks must succeed: on any failure
    /// the lane is reset on both and the call returns false (a miss).
    pub fn prefix_disk_load(&mut self, lane: usize, id: &str, n: usize, snap: u64, prompt: &[u32]) -> Result<bool> {
        self.mirror(|m| {
            m.u32(OP_DISK_LOAD).u32(lane as u32).u32s(&id.bytes().map(|b| b as u32).collect::<Vec<_>>()).u64(n as u64).u64(snap);
        })?;
        let mine = self.disk_load_inner(lane, id, n, snap);
        let mine = match mine {
            Ok(tokens) if tokens.len() >= n && prompt.len() >= n && tokens[..n] == prompt[..n] => true,
            Ok(_) => {
                eprintln!("prefix disk: bundle {id}'s tokens are not this prompt's");
                false
            }
            Err(e) => {
                eprintln!("prefix disk: load {id}: {e}");
                false
            }
        };
        let theirs = match self.pair.as_mut() {
            Some(p) if self.leader => {
                let msg = p.ctl.recv()?;
                tidelm_comm::Reader::new(&msg).u32()? == 1
            }
            _ => true,
        };
        if mine && theirs {
            return Ok(true);
        }
        self.reset_lane(lane)?;
        self.prefix_drop(lane, &[snap])?;
        Ok(false)
    }

    fn disk_load_inner(&mut self, lane: usize, id: &str, n: usize, snap: u64) -> Result<Vec<u32>> {
        if lane >= self.max_lanes {
            return Err(format!("bundle {id}: lane {lane} of {}", self.max_lanes).into());
        }
        let mut s = self.snap_take()?;
        match self.disk_stream_in(lane, id, n, &mut s) {
            Ok(tokens) => {
                self.snap_put((lane, snap), s);
                self.disk_load_commit(lane, snap, &tokens, n)?;
                Ok(tokens)
            }
            Err(e) => {
                self.snap_free.push(s);
                Err(e)
            }
        }
    }

    /// Bundle `id` into lane `lane`'s caches (its first n / ratio compressed rows a layer) and
    /// the snapshot at `n` into `s`, through the disk tier's chunk; the file's checksum is
    /// checked before this returns, and nothing else is committed until then.
    fn disk_stream_in(&mut self, lane: usize, id: &str, n: usize, s: &mut LaneSnap) -> Result<Vec<u32>> {
        let ns = self.disk_ns;
        let mut r = self.disk.as_mut().ok_or("no prefix disk tier")?.open_bundle(id)?;
        if r.u64()? != ns || r.vec::<u8>()? != id.as_bytes() {
            return Err(format!("prefix disk: bundle {id} is another build's").into());
        }
        let tokens: Vec<u32> = r.vec()?;
        if n > tokens.len() {
            return Err(format!("bundle {id}: snapshot {n} past its {} tokens", tokens.len()).into());
        }
        let (hd, ihd) = (self.cfg.head_dim, self.cfg.index_head_dim);
        let (w_full, hc, d) = (self.cfg.window_size, self.cfg.hc_mult, self.cfg.dim);
        let rings = self.caches.ring.len() + self.caches.mtp_ring.len();
        let st = &self.stream;
        // `len` bytes of the stream to device address `dst` (the pinned path is the driver's)
        let up = |r: &mut crate::prefix_disk::BundleReader, dst: *mut c_void, len: usize| -> Result<()> {
            r.stream(len, &mut |b, off| {
                unsafe { memcpy_async((dst as *mut u8).add(off) as *mut c_void, b.as_ptr() as *const c_void, b.len(), MEMCPY_HOST_TO_DEVICE, st)? };
                Ok(st.synchronize()?)
            })
        };
        let mut found = false;
        for _ in 0..r.u64()? {
            let (sn, sw) = (r.u64()? as usize, r.u64()? as usize);
            let field = |r: &mut crate::prefix_disk::BundleReader| -> Result<usize> { Ok(r.u64()? as usize) };
            if sn != n || found {
                for size in [2, 4, 4, 2, 4] {
                    let k = field(&mut r)?;
                    r.skip(k * size)?;
                }
                r.u64()?;
                continue;
            }
            let rows = field(&mut r)?;
            if rows != (rings * sw * hd).max(1) || sw > w_full {
                return Err(format!("bundle {id}: snapshot rows for another model shape").into());
            }
            up(&mut r, s.rows.ptr_at(0), rows * 2)?;
            for (ptr, len) in [(s.pending.ptr_at(0), s.pending.len()), (s.flags.ptr_at(0), s.flags.len())] {
                let k = field(&mut r)?;
                if k != 0 && k != len {
                    return Err(format!("bundle {id}: snapshot state for another model shape").into());
                }
                up(&mut r, ptr, k * 4)?;
            }
            for (buf, size, want) in [(s.rep_h.ptr_at(0), 2usize, w_full * hc * d), (s.rep_pre.ptr_at(0), 4, w_full * hc)] {
                let k = field(&mut r)?;
                if self.swa_replay && k != want {
                    return Err(format!("bundle {id}: no replay rows for this engine").into());
                }
                if self.swa_replay {
                    up(&mut r, buf, k * size)?;
                } else {
                    r.skip(k * size)?;
                }
            }
            (s.n, s.w, s.floor) = (sn, sw, r.u64()? as i64 as i32);
            found = true;
        }
        if !found {
            return Err(format!("bundle {id} has no snapshot at {n}").into());
        }
        let row = if self.packed_kv { PACKED_ROW / 2 } else { hd };
        for (caches, width, what) in [(&self.caches.ckv, row, "layer"), (&self.caches.ik, ihd, "layer keys")] {
            let mut seen = 0;
            for _ in 0..r.u64()? {
                let l = r.u64()? as usize;
                let bytes = r.u64()? as usize;
                let Some(buf) = caches.get(&l) else {
                    r.skip(bytes)?;
                    continue;
                };
                let (ratio, rows) = (self.cfg.compress_ratios[l], self.caches.crows[&l]);
                let need = (n / ratio) * width * 2;
                if bytes < need {
                    return Err(format!("bundle {id}: {what} {l} holds fewer rows than {n} tokens need").into());
                }
                up(&mut r, buf.ptr_at(lane * rows * width), need)?;
                r.skip(bytes - need)?;
                seen += 1;
            }
            if seen != caches.len() {
                return Err(format!("bundle {id} lacks a {what}").into());
            }
        }
        r.finish()?;
        Ok(tokens)
    }

    /// A streamed bundle's snapshot restored (disk_load_inner, after its checksum).
    fn disk_load_commit(&mut self, lane: usize, snap: u64, tokens: &[u32], n: usize) -> Result<()> {
        self.restore_inner(lane, snap)?;
        // the n-gram history of the restored prefix, as the hashing would have written it
        // (a bundle's tokens are prefix keys: an image's positions hash as dead)
        let (ids, mask) = crate::vision::split_key(&tokens[..n], self.cfg.image_token_id);
        let l = &mut self.lanes[lane];
        l.hash.hash(&ids, 0, mask.as_deref());
        Ok(())
    }

    /// Forget lane `lane`'s snapshots `ids`.
    pub fn prefix_drop(&mut self, lane: usize, ids: &[u64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.mirror(|m| {
            m.u32(OP_SNAP_DROP).u32(lane as u32).u32(ids.len() as u32);
            for &id in ids {
                m.u64(id);
            }
        })?;
        for id in ids {
            if let Some(s) = self.snaps.remove(&(lane, *id)) {
                self.snap_free.push(s);
            }
        }
        Ok(())
    }

    /// The backbone's resident experts, [layer][expert] (resident mode).
    pub fn resident_mask(&self) -> Option<Vec<Vec<bool>>> {
        let set = &self.resident.as_ref()?.experts;
        let e = set.n_experts;
        Some((0..self.n_layers).map(|l| (0..e).map(|x| set.lut_host[l * (e + 1) + x] >= 0).collect()).collect())
    }

    /// Residency moves between steps: each `[from_layer, from_expert, to_layer, to_expert]`
    /// gives a resident expert's arena slot to an absent one (global_residency's swap pairs:
    /// a layer may hand its slot to another), read from NVMe into it; then the slot LUT and
    /// the router's keep mask change in place, so captured graphs route by the new set from
    /// their next replay. Rank 0 mirrors the moves and rank 1 makes the same ones - its
    /// half of each expert - so the pair's routing stays one box's. Returns the seconds taken.
    pub fn move_experts(&mut self, moves: &[[usize; 4]]) -> Result<f64> {
        if moves.is_empty() {
            return Ok(0.0);
        }
        self.mirror(|m| {
            m.u32(OP_MOVE_EXPERTS).u32(moves.len() as u32);
            for mv in moves {
                for &v in mv {
                    m.u32(v as u32);
                }
            }
        })?;
        let t0 = Instant::now();
        let n_layers = self.n_layers;
        let res = self.resident.as_mut().ok_or("moving experts needs resident mode")?;
        let set = &mut res.experts;
        let e = set.n_experts;
        let at = |l: usize, x: usize| l * (e + 1) + x;
        // the whole plan is checked before any read: a source resident and leaving once, a
        // target absent and arriving once
        let mut jobs = Vec::with_capacity(moves.len());
        let (mut leaving, mut arriving) = (std::collections::HashSet::new(), std::collections::HashSet::new());
        for &[lf, xf, lt, xt] in moves {
            if lf >= n_layers || lt >= n_layers || xf >= e || xt >= e {
                return Err(format!("expert move ({lf}, {xf}) -> ({lt}, {xt}) out of range").into());
            }
            let slot = set.lut_host[at(lf, xf)];
            if slot < 0 || set.lut_host[at(lt, xt)] >= 0 || !leaving.insert((lf, xf)) || !arriving.insert((lt, xt)) {
                return Err(format!("expert move ({lf}, {xf}) -> ({lt}, {xt}): the source must be resident, the target not").into());
            }
            jobs.push((slot as usize, lt, xt));
        }
        // nothing queued may still read a slot about to be overwritten
        self.stream.synchronize()?;
        set.read_many(&self.ckpt, &jobs)?;
        for (&[lf, xf, lt, xt], &(slot, _, _)) in moves.iter().zip(&jobs) {
            set.lut_host[at(lf, xf)] = -1;
            set.lut_host[at(lt, xt)] = slot as i32;
            set.owner[slot] = Some((lt, xt));
        }
        set.lut.upload_async(0, &set.lut_host, &self.stream)?;
        if let Some(keep) = &self.keep {
            let flat: Vec<u8> = (0..n_layers).flat_map(|l| (0..e).map(move |x| (l, x))).map(|(l, x)| (set.lut_host[at(l, x)] >= 0) as u8).collect();
            keep.upload_async(0, &flat, &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Lanes a single draft call can take (TMAX rows of draft blocks).
    pub fn max_draft_lanes(&self) -> usize {
        TMAX / self.cfg.dspark_block_size
    }

    /// Lane 0's cached length (the single-sequence API).
    pub fn len(&self) -> usize {
        self.lanes[0].len
    }

    pub fn is_empty(&self) -> bool {
        self.lanes[0].len == 0
    }

    pub fn lane_len(&self, lane: usize) -> usize {
        self.lanes[lane].len
    }

    /// Switch graph capture/replay on or off (resident only).
    pub fn set_graphs(&mut self, on: bool) {
        let _ = self.mirror(|m| {
            m.u32(OP_GRAPHS).u32(on as u32);
        });
        self.use_graphs = on && self.resident.is_some();
    }

    /// Start a new sequence in every lane.
    pub fn reset(&mut self) {
        for lane in 0..self.max_lanes {
            let _ = self.reset_lane(lane);
        }
    }

    /// Start a new sequence in `lane`. Nothing needs clearing: every cache entry a position
    /// reads is written by this sequence before it is read (rings and compressed rows by
    /// position, the hash cache from position 0), so only the length and the pending flag
    /// reset.
    pub fn reset_lane(&mut self, lane: usize) -> Result<()> {
        if lane > self.max_lanes {
            return Err(format!("lane {lane} of {}", self.max_lanes).into());
        }
        self.mirror(|m| {
            m.u32(OP_RESET_LANE).u32(lane as u32);
        })?;
        // the n-gram history stays: it is by position and rewritten before it is read, and a
        // prefix restored after this reset reads the matched prefix's entries from it
        // (re-hashing the prompt on every hit, as the Python engine does, finds the same ids)
        let l = &mut self.lanes[lane];
        (l.len, l.last, l.step) = (0, (0, 0, 0), u64::MAX);
        for st in self.caches.r2.values() {
            st.flag.upload_async(lane, &[0], &self.stream)?;
        }
        self.floors[lane] = 0;
        self.win_floor.upload_async(0, &self.floors, &self.stream)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Lane 0: discard positions `>= n`.
    pub fn rollback(&mut self, n: usize) -> Result<()> {
        self.rollback_batch(&[(0, n)])
    }

    /// Discard positions `>= n` of each (lane, n) (model.py::Caches.rollback). Only the
    /// ratio-2 compressors carry state across positions, and their chunk memory holds a
    /// lane's last segment: n must lie within it. Rings and compressed rows are append-only
    /// and every slot at or after n is rewritten before anything reads it.
    pub fn rollback_batch(&mut self, reqs: &[(usize, usize)]) -> Result<()> {
        if reqs.is_empty() {
            return Ok(());
        }
        self.mirror(|m| {
            m.u32(OP_ROLLBACK).u32(reqs.len() as u32);
            for &(lane, n) in reqs {
                m.u32(lane as u32).u64(n as u64);
            }
        })?;
        let mut seen = vec![false; self.max_lanes];
        for &(lane, n) in reqs {
            if lane >= self.max_lanes || std::mem::replace(&mut seen[lane], true) {
                return Err(format!("rollback: lane {lane} out of range or repeated").into());
            }
            let l = &self.lanes[lane];
            if n > l.len || n < l.last.0 {
                return Err(format!("rollback({n}) of lane {lane}: outside its last segment {}..={}", l.last.0, l.len).into());
            }
            // the compressors keep a segment's last TMAX rows: a longer (prompt) segment cannot
            // be undone, only kept
            if l.last.2 > TMAX && n != l.len {
                return Err(format!("rollback({n}) of lane {lane}: into a {}-row prompt segment", l.last.2).into());
            }
        }
        let st = &self.stream;
        let ns: Vec<i32> = reqs.iter().map(|r| r.1 as i32).collect();
        let ls: Vec<i32> = reqs.iter().map(|r| r.0 as i32).collect();
        self.caches.n_dev.upload_async(0, &ns, st)?;
        self.caches.rb_lane.upload_async(0, &ls, st)?;
        let (sr, hd) = (st.raw(), self.cfg.head_dim as i32);
        for r2 in self.caches.r2.values() {
            ck(unsafe {
                ds_rollback_pending(sr, np(&self.caches.n_dev), np(&self.caches.rb_lane), reqs.len() as i32, np(&r2.mem_kv),
                                    np(&r2.mem_sc), np(&r2.mem_start), np(&r2.before_kv), np(&r2.before_sc),
                                    np(&r2.before_flag), np(&r2.pkv), np(&r2.psc), np(&r2.flag), hd, TMAX as i32)
            }, "rollback pending")?;
        }
        for &(lane, n) in reqs {
            self.lanes[lane].len = n;
        }
        Ok(())
    }

    /// The host half of the embedding for `ids` (padded rows past `live` are filler):
    /// resident uploads the ids, streaming reads the live rows.
    fn embed_host(&mut self, ids: &[u32], live: usize) -> Result<()> {
        let d = self.cfg.dim;
        if let Some(res) = &self.resident {
            let dev_ids: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
            res.ids.upload_async(0, &dev_ids, &self.stream)?;
        } else {
            let noise = self.cfg.dspark_noise_token_id;
            let mut ld = Loader::new(&self.ckpt, &self.stream)?;
            for (ti, &id) in ids[..live].iter().enumerate() {
                match &self.dspark {
                    Some(ds) if id == noise => self.act.emb.copy_from(ti * d, &ds.noise_row, 0, d, &self.stream)?,
                    _ => ld.read_into("embed.weight", id as usize * d * 2, (id as usize + 1) * d * 2, self.act.emb.ptr_at(ti * d))?,
                }
            }
        }
        Ok(())
    }

    /// The device half: expand into the hc stream, one-hot pre_mix.
    fn embed_dev(&self, t: usize) -> Result<()> {
        let (d, hc) = (self.cfg.dim, self.cfg.hc_mult);
        let st = &self.stream;
        let (ids, table) = match &self.resident {
            Some(res) => (np(&res.ids), np(&res.embed)),
            None => (np(&self.act.row_ids), np(&self.act.emb)),
        };
        ck(unsafe { ds_embed_hc(st.raw(), ids, table, np(&self.act.h), t as i32, d as i32, hc as i32) }, "embed")?;
        self.act.pre.copy_from(0, &self.one_hot, 0, t * hc, st)?;
        Ok(())
    }

    /// Lane 0: prompt tokens `ids` at its next positions; returns the bf16 logits `want` asks
    /// for.
    pub fn forward(&mut self, ids: &[u32], want: Want) -> Result<Vec<u16>> {
        self.forward_next(ids, &[], want)
    }

    /// `forward` with the rows the next call will feed (Seg::next), whose engram rows are
    /// read while this one computes.
    pub fn forward_next(&mut self, ids: &[u32], next: &[u32], want: Want) -> Result<Vec<u16>> {
        Ok(self.forward_batch(&[Seg { lane: 0, ids, want, decode: false, next, hashes: None }])?.remove(0))
    }

    /// One backbone step over every segment's rows (at most TMAX together, each lane at most
    /// once, rows of a lane in position order): returns per segment the bf16 logits its
    /// `want` asks for (the engine's head rounds them).
    pub fn forward_batch(&mut self, segs: &[Seg]) -> Result<Vec<Vec<u16>>> {
        let v = self.cfg.vocab_size;
        let plan = self.run_step(segs, Readback::Logits, Stage::Full)?;
        Ok(plan.heads.iter().map(|&(h0, n)| self.logits_host[h0 * v..(h0 + n) * v].to_vec()).collect())
    }

    /// `forward_batch` for greedy decoding: per segment, the argmax of each row its `want`
    /// asks for (first index on ties, as decode::argmax_bf16), computed on the device so
    /// only the ids come back.
    pub fn forward_batch_argmax(&mut self, segs: &[Seg]) -> Result<Vec<Vec<u32>>> {
        let plan = self.run_step(segs, Readback::Argmax, Stage::Full)?;
        Ok(plan.heads.iter().map(|&(h0, n)| self.am_host[h0..h0 + n].iter().map(|&x| x as u32).collect()).collect())
    }

    /// Prompt rows through the encoder half alone (Stage::Encoder; their segments want
    /// nothing): the decoder half runs once over the prompt's tail in replay_batch_*.
    pub fn encode_batch(&mut self, segs: &[Seg]) -> Result<()> {
        if !self.swa_replay {
            return Err("encode_batch: the engine runs without the bounded replay".into());
        }
        if segs.iter().any(|sg| sg.decode || sg.want != Want::Nothing) {
            return Err("encode_batch: prompt rows only, wanting no logits".into());
        }
        self.run_step(segs, Readback::Argmax, Stage::Encoder)?;
        Ok(())
    }

    /// Whether prompts run as encoder steps and a bounded replay (Options::swa_replay).
    pub fn swa_replay(&self) -> bool {
        self.swa_replay
    }

    /// Rows a replay of `lane` covers: its prompt's last window positions.
    pub fn replay_rows(&self, lane: usize) -> usize {
        self.lanes[lane].len.min(self.cfg.window_size)
    }

    /// model.py::decoder_replay for each lane, after the encoder steps of its whole prompt:
    /// the last window rows from their kept state through the replay layers on, their decoder
    /// windows starting at the first of them; the last row's argmax per lane. The rows are the
    /// lanes' last step segments (DSpark seeds from them), the lanes' lengths unchanged.
    pub fn replay_batch_argmax(&mut self, lanes: &[usize]) -> Result<Vec<u32>> {
        let plan = self.run_replay(lanes, Readback::Argmax)?;
        Ok(plan.heads.iter().map(|&(h0, _)| self.am_host[h0] as u32).collect())
    }

    /// replay_batch_argmax, the last row's nucleus at each lane's sampling.
    pub fn replay_batch_sampled(&mut self, lanes: &[usize], sampling: &[Sampling]) -> Result<Vec<Nucleus>> {
        if sampling.len() != lanes.len() {
            return Err(format!("{} lanes, {} samplings", lanes.len(), sampling.len()).into());
        }
        let plan = self.run_replay(lanes, Readback::Nucleus(sampling))?;
        Ok(self.nuclei(&plan, sampling)?.into_iter().map(|mut r| r.remove(0)).collect())
    }

    fn run_replay(&mut self, lanes: &[usize], readback: Readback) -> Result<Plan> {
        self.mirror(|m| {
            m.u32(OP_REPLAY).u32s(&lanes.iter().map(|&l| l as u32).collect::<Vec<_>>());
        })?;
        if !self.swa_replay {
            return Err("replay: the engine runs without the bounded replay".into());
        }
        let tp = Instant::now();
        let plan = self.prepare_replay(lanes)?;
        let shape = plan.shape;
        self.eager_step = shape.t > TMAX;
        let td = Instant::now();
        self.times.prepare_s += (td - tp).as_secs_f64();
        let res = (|| -> Result<()> {
            for (k, (l0, l1)) in self.segments_for(Stage::Replay).into_iter().enumerate() {
                self.run_segment(shape, k, l0, l1)?;
            }
            Ok(())
        })();
        self.eager_step = false;
        res?;
        self.read_heads(&plan, readback)?;
        // the rows are the lanes' last segments (seed_draft_batch reads them); no new positions
        for (i, &ln) in lanes.iter().enumerate() {
            let rows = self.replay_rows(ln);
            let l = &mut self.lanes[ln];
            l.last = (l.len - rows, plan.firsts[i], rows);
            l.step = self.step_no;
        }
        self.times.device_s += td.elapsed().as_secs_f64();
        Ok(plan)
    }

    /// `forward_batch` for sampling: per segment, the nucleus of each row its `want` asks for
    /// at that segment's `sampling` (v41_engine.sample_probs), computed on the device; a row
    /// whose nucleus outgrows the device's window is recomputed here from its logits.
    pub fn forward_batch_sampled(&mut self, segs: &[Seg], sampling: &[Sampling]) -> Result<Vec<Vec<Nucleus>>> {
        if sampling.len() != segs.len() {
            return Err(format!("{} segments, {} samplings", segs.len(), sampling.len()).into());
        }
        let plan = self.run_step(segs, Readback::Nucleus(sampling), Stage::Full)?;
        self.nuclei(&plan, sampling)
    }

    /// Each head row's nucleus after a Readback::Nucleus step, per segment.
    fn nuclei(&mut self, plan: &Plan, sampling: &[Sampling]) -> Result<Vec<Vec<Nucleus>>> {
        let v = self.cfg.vocab_size;
        let mut out = Vec::with_capacity(plan.heads.len());
        for (i, &(h0, n)) in plan.heads.iter().enumerate() {
            let mut rows = Vec::with_capacity(n);
            for h in h0..h0 + n {
                let c = self.nuc_cnt_host[h];
                if c > 0 {
                    let (ids, probs) = (&self.nuc_ids_host[h * NUC_CAP..], &self.nuc_p_host[h * NUC_CAP..]);
                    rows.push(Nucleus { ids: ids[..c as usize].iter().map(|&x| x as u32).collect(), probs: probs[..c as usize].to_vec() });
                    continue;
                }
                // the whole row, on the host (tidelm_core::sampling::sample_probs)
                self.act.logits.download_async(h * v, &mut self.logits_host[..v], &self.stream)?;
                self.stream.synchronize()?;
                let row: Vec<f32> = self.logits_host[..v].iter().map(|&x| f32::from_bits((x as u32) << 16)).collect();
                let p = tidelm_core::sampling::sample_probs(&row, sampling[i].temperature, sampling[i].top_p);
                let mut kept: Vec<u32> = (0..v as u32).filter(|&j| p[j as usize] > 0.0).collect();
                kept.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
                self.nucleus_fallbacks += 1;
                rows.push(Nucleus { probs: kept.iter().map(|&j| p[j as usize]).collect(), ids: kept });
            }
            out.push(rows);
        }
        Ok(out)
    }

    /// The step's layer ranges, cut at each engram layer: a segment that starts at one needs
    /// that layer's gathered rows on the device before it runs (engine/fastdecode.py's three
    /// replays, [L0] [L1-13] [L14-39 + head]).
    fn segments(&self) -> Vec<(usize, usize)> {
        let mut cuts: Vec<usize> = self.cfg.engram_layer_ids.iter().copied().filter(|&l| l > 0 && l < self.n_layers).collect();
        cuts.sort_unstable();
        let mut out = Vec::new();
        let mut l0 = 0;
        for c in cuts.into_iter().chain(std::iter::once(self.n_layers)) {
            if c > l0 {
                out.push((l0, c));
                l0 = c;
            }
        }
        out
    }

    /// The step's layer ranges at `stage`: the encoder's end at the replay layer included, the
    /// replay's from it.
    fn segments_for(&self, stage: Stage) -> Vec<(usize, usize)> {
        let rl = self.rep_layer();
        match stage {
            Stage::Full => self.segments(),
            Stage::Encoder => self.segments().into_iter().filter(|s| s.0 <= rl).map(|(a, b)| (a, b.min(rl + 1))).collect(),
            Stage::Replay => vec![(rl, self.n_layers)],
        }
    }

    /// The layer a replay starts at: the encoder half's last (candidate_source_layer).
    fn rep_layer(&self) -> usize {
        self.cfg.candidate_source_layer.max(0) as usize
    }

    /// One backbone step: host preparation, then the graph segments, each started once the
    /// engram rows it needs have been gathered. The gathers run on their own threads from the
    /// start of the step, so a layer's NVMe reads overlap every segment before it.
    fn run_step(&mut self, segs: &[Seg], readback: Readback, stage: Stage) -> Result<Plan> {
        let hot_digest = self.hot.as_ref().map(|h| h.digest()).unwrap_or(0);
        // rank 1 replays: its hot-read state must be rank 0's, or the pair's plans diverge
        if let Some(theirs) = self.peer_hot.take() {
            if theirs != hot_digest {
                return Err(format!("hot reads: rank 0's state {theirs:x} is not this rank's {hot_digest:x}").into());
            }
        }
        self.mirror(|m| {
            m.u32(OP_FORWARD).u32(stage as u32).u32(segs.len() as u32);
            for sg in segs {
                let want = match sg.want {
                    Want::Nothing => 0,
                    Want::Last => 1,
                    Want::All => 2,
                };
                m.u32(sg.lane as u32).u32(want).u32(sg.decode as u32).u32s(sg.ids).u32s(sg.next);
                let h = sg.hashes.unwrap_or(&[]);
                m.u32(sg.hashes.is_some() as u32).u32(h.len() as u32);
                for &x in h {
                    m.u64(x as u64);
                }
            }
            m.u64(hot_digest);
        })?;
        let tp = Instant::now();
        let plan = self.prepare(segs, stage)?;
        let shape = plan.shape;
        self.eager_step = shape.t > TMAX;
        let td = Instant::now();
        // a decode step reads last step's heaviest misses and routes to the ring's live ones
        let hot_step = self.hot.is_some() && shape.decode && !shape.prompt;
        if hot_step {
            self.hot_begin()?;
        }
        self.times.prepare_s += (td - tp).as_secs_f64();
        let mut tables = std::mem::take(&mut self.engram_tables);
        let jobs = std::mem::take(&mut self.eg_jobs);
        let prefetch = std::mem::take(&mut self.eg_prefetch);
        let cfg_layers = self.cfg.engram_layer_ids.clone();
        let res = std::thread::scope(|sc| -> Result<()> {
            let mut pending = HashMap::new();
            let mut threads = Vec::new();
            for (l, table) in tables.iter_mut() {
                let Some(li) = cfg_layers.iter().position(|x| x == l) else { continue };
                let Some((_, ids)) = jobs.iter().find(|j| j.0 == li) else { continue };
                let warm = prefetch.iter().find(|j| j.0 == li).map(|j| &j.1);
                // the gather thread is the only user of eg_host[li] until it reports below
                let out = SendPtr(self.eg_host[li].as_mut_ptr());
                let n = ids.len();
                let (tx, rx) = std::sync::mpsc::channel();
                pending.insert(*l, (li, n, rx));
                // this step's rows first, reported as soon as they are in; then the next
                // step's into the cache, behind the GPU's work on this one
                threads.push(sc.spawn(move || -> i32 {
                    let out = out;
                    let buf = unsafe { std::slice::from_raw_parts_mut(out.0, n * ENGRAM_ROW) };
                    let rc = table.gather(ids, buf);
                    let _ = tx.send(rc);
                    match warm {
                        Some(w) if rc == 0 => table.warm(w),
                        _ => 0,
                    }
                }));
            }
            let mut first_err = None;
            for (k, (l0, l1)) in self.segments_for(shape.stage).into_iter().enumerate() {
                if let Some((li, n, rx)) = pending.remove(&l0) {
                    let te = Instant::now();
                    let rc = rx.recv().unwrap_or(-1);
                    self.times.engram_s += te.elapsed().as_secs_f64();
                    if rc != 0 {
                        first_err.get_or_insert(format!("layer {l0}: engram gather failed ({rc})"));
                        break;
                    }
                    self.eg_raw[li].upload_async(0, &self.eg_host[li][..n * ENGRAM_ROW], &self.stream)?;
                }
                self.run_segment(shape, k, l0, l1)?;
            }
            // every thread is joined here (its prefetch done) - also one whose segment never
            // ran (an error above); a failed prefetch only costs the next step its hits
            for h in threads {
                let _ = h.join();
            }
            match first_err {
                Some(e) => Err(e.into()),
                None => Ok(()),
            }
        });
        self.engram_tables = tables;
        self.eager_step = false;
        res?;
        if hot_step {
            self.hot_end()?;
        }
        self.collect(segs, &plan, readback)?;
        self.times.device_s += td.elapsed().as_secs_f64();
        Ok(plan)
    }

    fn run_segment(&mut self, shape: StepShape, k: usize, l0: usize, l1: usize) -> Result<()> {
        // prefill-sized steps run eager: their work dwarfs the launches, and a graph per row
        // count would pin memory for shapes seen once
        // an image step's splice and router bias read per-lane tables: never captured
        if !self.use_graphs || shape.t > TMAX || shape.vision {
            return self.device_step(shape, l0, l1);
        }
        let key = (shape, k);
        self.graph_tick += 1;
        if !self.graphs.contains_key(&key) {
            // an instantiated step graph holds ~8 MB (its device part and the driver's heap), and
            // a long session keeps meeting new shapes: past graph_cap the least recently used
            // goes (a graph destroyed while it runs is freed when it completes), so the graphs
            // hold a fixed budget instead of growing with the session
            if self.graphs.len() >= self.graph_cap {
                if let Some(old) = self.graphs.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| *k) {
                    self.graphs.remove(&old);
                }
            }
            // capture records the segment without running it; the launch below runs it once
            self.stream.begin_capture()?;
            let rec = self.device_step(shape, l0, l1);
            let graph = self.stream.end_capture();
            rec?;
            self.graphs.insert(key, (graph?, self.graph_tick));
            self.graph_captures += 1;
        } else {
            self.graph_replays += 1;
        }
        let tick = self.graph_tick;
        let g = self.graphs.get_mut(&key).unwrap();
        g.1 = tick;
        g.0.launch(&self.stream)?;
        Ok(())
    }

    /// Host work before a step's device work: the rows' positions and lanes, the segments,
    /// the embedding's host half, the n-gram hashes and their engram rows (an NVMe gather),
    /// and the step's shape.
    fn prepare(&mut self, segs: &[Seg], stage: Stage) -> Result<Plan> {
        if segs.is_empty() {
            return Err("a step needs at least one segment".into());
        }
        let mut seen = vec![false; self.max_lanes + 1];
        let mut rows = 0;
        for sg in segs {
            // the probe lane takes prompt rows only (its demand is a probe's, never a request's)
            if sg.lane > self.max_lanes || (sg.lane == self.max_lanes && sg.decode) || std::mem::replace(&mut seen[sg.lane], true) {
                return Err(format!("lane {} out of range or repeated in one step", sg.lane).into());
            }
            if sg.ids.is_empty() {
                return Err(format!("lane {}: empty segment", sg.lane).into());
            }
            let end = self.lanes[sg.lane].len + sg.ids.len();
            if end > self.max_seq {
                return Err(format!("lane {}: context {end} exceeds max_seq {}", sg.lane, self.max_seq).into());
            }
            rows += sg.ids.len();
        }
        // past TMAX only prompt rows: a step that large runs eager, and its head rows (the
        // segments' wanted rows) still fit the decode step's head
        if rows > self.pmax {
            return Err(format!("a step takes at most {} rows, got {rows}", self.pmax).into());
        }
        let wanted: usize = segs.iter().map(|sg| match sg.want {
            Want::Nothing => 0,
            Want::Last => 1,
            Want::All => sg.ids.len(),
        }).sum();
        if wanted > TMAX {
            return Err(format!("a step reads at most {TMAX} head rows, got {wanted}").into());
        }
        let cfg = self.cfg.clone();
        let tw = Instant::now();
        let tb = row_bucket(rows);
        let scratch = self.max_lanes as i32;
        // filler rows sit in the scratch lane at its ring's last slot, clear of a probe's
        // positions (Engine::probe_lane uses the same lane from position 0)
        let filler = (self.ring - 1).min(self.max_seq.saturating_sub(1)) as i32;
        let (mut pos, mut lane, mut ids, mut dec) = (vec![filler; tb], vec![scratch; tb], vec![0u32; tb], vec![0i32; tb]);
        let nseg = self.max_lanes + 1;
        let (mut sf, mut sn, mut sl) = (vec![0i32; nseg], vec![0i32; nseg], vec![scratch; nseg]);
        let (mut head_rows, mut heads, mut firsts) = (Vec::new(), Vec::new(), Vec::new());
        let (mut nc1, mut nc2) = (0usize, 0usize);
        let mut r = 0;
        for (i, sg) in segs.iter().enumerate() {
            let (s, t) = (self.lanes[sg.lane].len, sg.ids.len());
            for j in 0..t {
                pos[r + j] = (s + j) as i32;
                lane[r + j] = sg.lane as i32;
                ids[r + j] = sg.ids[j];
                dec[r + j] = sg.decode as i32;
            }
            (sf[i], sn[i], sl[i]) = (r as i32, t as i32, sg.lane as i32);
            let h0 = head_rows.len();
            match sg.want {
                Want::Nothing => {}
                Want::Last => head_rows.push((r + t - 1) as i32),
                Want::All => head_rows.extend((r..r + t).map(|x| x as i32)),
            }
            heads.push((h0, head_rows.len() - h0));
            firsts.push(r);
            nc1 = nc1.max(s + t);
            nc2 = nc2.max((s + t) / 2);
            r += t;
        }
        let nw = head_rows.len();
        let hb = if nw == 0 { 0 } else { row_bucket(nw) };
        head_rows.resize(hb.max(1), 0);
        {
            let st = &self.stream;
            let a = &self.act;
            a.pos.upload_async(0, &pos, st)?;
            a.lane.upload_async(0, &lane, st)?;
            let cpos2: Vec<i32> = pos.iter().map(|&p| (p - 1).max(0)).collect();
            self.cpos1.upload_async(0, &pos, st)?;
            self.cpos2.upload_async(0, &cpos2, st)?;
            a.seg_first.upload_async(0, &sf, st)?;
            a.seg_count.upload_async(0, &sn, st)?;
            a.seg_lane.upload_async(0, &sl, st)?;
            a.head_rows.upload_async(0, &head_rows, st)?;
            a.live_pairs.upload_async(0, &[(rows * cfg.n_activated_experts) as i32], st)?;
            a.decode_rows.upload_async(0, &dec, st)?;
        }
        // image rows: the table row each takes; the n-gram hasher skips them (and the next
        // rows' read-ahead too)
        let vr = cfg.vision_downsample_ratio;
        let (mut vidx, mut vimg) = (Vec::new(), Vec::new());
        self.vis_splices.clear();
        let mut masks: Vec<(Option<Vec<bool>>, Option<Vec<bool>>)> = Vec::with_capacity(segs.len());
        for (i, sg) in segs.iter().enumerate() {
            let s = self.lanes[sg.lane].len;
            let Some(v) = self.lane_vis[sg.lane].as_ref().filter(|_| sg.hashes.is_none()) else {
                masks.push((None, None));
                continue;
            };
            let at: Vec<Option<i32>> = (0..sg.ids.len()).map(|j| v.row_at(s + j, vr)).collect::<Result<_>>()?;
            let ahead: Vec<bool> = (0..sg.next.len()).map(|j| v.row_at(s + sg.ids.len() + j, vr).map(|x| x.is_none())).collect::<Result<_>>()?;
            let ahead = ahead.iter().any(|&live| !live).then_some(ahead);
            if at.iter().all(|x| x.is_none()) {
                masks.push((None, ahead));
                continue;
            }
            if vidx.is_empty() {
                (vidx, vimg) = (vec![-1i32; tb], vec![0i32; tb]);
            }
            for (j, x) in at.iter().enumerate() {
                if let Some(row) = x {
                    vidx[firsts[i] + j] = *row;
                    vimg[firsts[i] + j] = 1;
                }
            }
            self.vis_splices.push((firsts[i], sg.ids.len(), sg.lane));
            masks.push((Some(at.iter().map(|x| x.is_none()).collect()), ahead));
        }
        let vision = !vidx.is_empty();
        if vision {
            self.vis_idx.upload_async(0, &vidx, &self.stream)?;
            self.vis_img.upload_async(0, &vimg, &self.stream)?;
        }
        // the replay's kept rows: each prompt segment's last window rows, at their slots
        if self.swa_replay {
            let w = self.cfg.window_size;
            let mut ri = vec![-1i32; tb];
            for (i, sg) in segs.iter().enumerate().filter(|(_, sg)| !sg.decode) {
                let (s0, t) = (self.lanes[sg.lane].len, sg.ids.len());
                for j in t.saturating_sub(w)..t {
                    ri[firsts[i] + j] = (sg.lane * w + (s0 + j) % w) as i32;
                }
            }
            self.rep_idx.upload_async(0, &ri, &self.stream)?;
        }
        self.live_rows = rows;
        self.embed_host(&ids, rows)?;
        let n_eng = cfg.engram_layer_ids.len();
        let cols = self.hash.layout.n_hash_cols();
        // [row][n_engram_layers][cols], live rows only
        let mut hashes = Vec::with_capacity(rows * n_eng * cols);
        for (i, sg) in segs.iter().enumerate() {
            let l = &mut self.lanes[sg.lane];
            match sg.hashes {
                Some(h) if h.len() == sg.ids.len() * n_eng * cols => hashes.extend_from_slice(h),
                Some(h) => return Err(format!("lane {}: {} hashes for {} rows", sg.lane, h.len(), sg.ids.len()).into()),
                None => hashes.extend(l.hash.hash(sg.ids, l.len, masks[i].0.as_deref())),
            }
        }
        self.times.compute_s += tw.elapsed().as_secs_f64();
        let st = &self.stream;
        self.eg_jobs.clear();
        for li in 0..n_eng {
            let l = cfg.engram_layer_ids[li];
            if l >= self.n_layers {
                continue;
            }
            let col_ids: Vec<i64> =
                (0..rows).flat_map(|ti| hashes[(ti * n_eng + li) * cols..(ti * n_eng + li + 1) * cols].to_vec()).collect();
            let mut uniq = col_ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            // filler rows read the first gathered row
            let mut inv: Vec<i32> = col_ids.iter().map(|h| uniq.binary_search(h).unwrap() as i32).collect();
            inv.resize(tb * cols, 0);
            self.eg_inv[li].upload_async(0, &inv, st)?;
            self.eg_jobs.push((li, uniq));
        }
        // the segments' next rows: hashing ahead writes the lanes' compressed-id caches at
        // positions their next step rewrites before it hashes (NgramHashState::hash)
        self.eg_prefetch.clear();
        let mut ahead: Vec<i64> = Vec::new();
        for (i, sg) in segs.iter().enumerate().filter(|(_, sg)| !sg.next.is_empty()) {
            let l = &mut self.lanes[sg.lane];
            ahead.extend(l.hash.hash(sg.next, l.len + sg.ids.len(), masks[i].1.as_deref()));
        }
        if !ahead.is_empty() {
            let n_ahead = ahead.len() / (n_eng * cols);
            for li in 0..n_eng {
                if cfg.engram_layer_ids[li] >= self.n_layers {
                    continue;
                }
                let mut uniq: Vec<i64> =
                    (0..n_ahead).flat_map(|ti| ahead[(ti * n_eng + li) * cols..(ti * n_eng + li + 1) * cols].iter().copied()).collect();
                uniq.sort_unstable();
                uniq.dedup();
                self.eg_prefetch.push((li, uniq));
            }
        }

        let topk = cfg.index_topk;
        let indexed = |nc: usize, r: usize| -> Option<usize> { (nc > topk).then(|| bucket(nc, 2048, self.max_seq / r + 1)) };
        let cand_src = cfg.candidate_source_layer;
        let cand = cand_src >= 0 && (cand_src as usize) < self.n_layers && {
            let r = cfg.compress_ratios[cand_src as usize].max(1);
            (if r == 1 { nc1 } else { nc2 }) > 2048 * 8
        };
        let shape = StepShape {
            stage,
            t: tb,
            c1: bucket(nc1.min(topk), 16, topk),
            c2: bucket(nc2.min(topk), 16, topk),
            n1: indexed(nc1, 1),
            n2: indexed(nc2, 2),
            cand,
            head: hb,
            prompt: segs.iter().any(|sg| !sg.decode),
            decode: segs.iter().any(|sg| sg.decode),
            vision,
        };
        Ok(Plan { shape, firsts, heads })
    }

    /// A replay step's host half (prepare's, for rows the lanes already hold): per lane its
    /// last window positions in order, prompt rows, the last one's head row; their kept state's
    /// slots; their decoder windows' floor at the first of them; image rows' router flags.
    fn prepare_replay(&mut self, lanes: &[usize]) -> Result<Plan> {
        if lanes.is_empty() {
            return Err("a replay needs a lane".into());
        }
        let cfg = self.cfg.clone();
        let w = cfg.window_size;
        let mut seen = vec![false; self.max_lanes + 1];
        let mut rows = 0;
        for &ln in lanes {
            if ln >= self.max_lanes || std::mem::replace(&mut seen[ln], true) || self.lanes[ln].len == 0 {
                return Err(format!("replay: lane {ln} out of range, repeated or empty").into());
            }
            rows += self.replay_rows(ln);
        }
        if rows > self.pmax {
            return Err(format!("a replay takes at most {} rows, got {rows}", self.pmax).into());
        }
        let tb = row_bucket(rows);
        let scratch = self.max_lanes;
        let filler = (self.ring - 1).min(self.max_seq.saturating_sub(1)) as i32;
        let (mut pos, mut lane, mut dec, mut ri) = (vec![filler; tb], vec![scratch as i32; tb], vec![0i32; tb], vec![(scratch * w) as i32; tb]);
        let nseg = self.max_lanes + 1;
        let (mut sf, mut sn, mut sl) = (vec![0i32; nseg], vec![0i32; nseg], vec![scratch as i32; nseg]);
        let (mut head_rows, mut heads, mut firsts) = (Vec::new(), Vec::new(), Vec::new());
        let (mut nc1, mut nc2, mut vimg, mut any_img) = (0usize, 0usize, vec![0i32; tb], false);
        let vr = cfg.vision_downsample_ratio;
        let mut r = 0;
        for (i, &ln) in lanes.iter().enumerate() {
            let (len, t) = (self.lanes[ln].len, self.replay_rows(ln));
            let s0 = len - t;
            for j in 0..t {
                let p = s0 + j;
                pos[r + j] = p as i32;
                lane[r + j] = ln as i32;
                ri[r + j] = (ln * w + p % w) as i32;
                if let Some(v) = &self.lane_vis[ln] {
                    if v.row_at(p, vr)?.is_some() {
                        vimg[r + j] = 1;
                        any_img = true;
                    }
                }
            }
            dec[r..r + t].iter_mut().for_each(|x| *x = 0);
            (sf[i], sn[i], sl[i]) = (r as i32, t as i32, ln as i32);
            heads.push((head_rows.len(), 1));
            head_rows.push((r + t - 1) as i32);
            firsts.push(r);
            nc1 = nc1.max(len);
            nc2 = nc2.max(len / 2);
            self.floors[ln] = s0 as i32;
            r += t;
        }
        let nw = head_rows.len();
        let hb = row_bucket(nw);
        head_rows.resize(hb.max(1), 0);
        {
            let st = &self.stream;
            let a = &self.act;
            a.pos.upload_async(0, &pos, st)?;
            a.lane.upload_async(0, &lane, st)?;
            let cpos2: Vec<i32> = pos.iter().map(|&p| (p - 1).max(0)).collect();
            self.cpos1.upload_async(0, &pos, st)?;
            self.cpos2.upload_async(0, &cpos2, st)?;
            a.seg_first.upload_async(0, &sf, st)?;
            a.seg_count.upload_async(0, &sn, st)?;
            a.seg_lane.upload_async(0, &sl, st)?;
            a.head_rows.upload_async(0, &head_rows, st)?;
            a.live_pairs.upload_async(0, &[(rows * cfg.n_activated_experts) as i32], st)?;
            a.decode_rows.upload_async(0, &dec, st)?;
            self.rep_idx.upload_async(0, &ri, st)?;
            self.win_floor.upload_async(0, &self.floors, st)?;
            if any_img {
                self.vis_img.upload_async(0, &vimg, st)?;
            }
        }
        self.live_rows = rows;
        self.vis_splices.clear();
        let topk = cfg.index_topk;
        let indexed = |nc: usize, r: usize| -> Option<usize> { (nc > topk).then(|| bucket(nc, 2048, self.max_seq / r + 1)) };
        let cand_src = cfg.candidate_source_layer;
        let cand = cand_src >= 0 && (cand_src as usize) < self.n_layers && {
            let r = cfg.compress_ratios[cand_src as usize].max(1);
            (if r == 1 { nc1 } else { nc2 }) > 2048 * 8
        };
        let shape = StepShape {
            stage: Stage::Replay,
            t: tb,
            c1: bucket(nc1.min(topk), 16, topk),
            c2: bucket(nc2.min(topk), 16, topk),
            n1: indexed(nc1, 1),
            n2: indexed(nc2, 2),
            cand,
            head: hb,
            prompt: true,
            decode: false,
            vision: any_img,
        };
        Ok(Plan { shape, firsts, heads })
    }

    /// The replay step's start (begin_step's): its rows' kept state into the stream in place of
    /// an embedding, and the all-visible compressed lists from their positions.
    fn begin_replay(&mut self, shape: StepShape) -> Result<()> {
        let (t, d, hc) = (shape.t, self.cfg.dim, self.cfg.hc_mult);
        let sr = self.stream.raw();
        let a = &self.act;
        ck(unsafe { ds_gather_rows_u16(sr, np(&self.rep_h), np(&self.rep_idx), np(&a.h), t as i32, (hc * d) as i32) }, "replay rows")?;
        // f32 rows as pairs of 16-bit words
        ck(unsafe { ds_gather_rows_u16(sr, np(&self.rep_pre), np(&self.rep_idx), np(&a.pre), t as i32, (hc * 2) as i32) }, "replay mixes")?;
        ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 1, shape.c1 as i32, np(&a.cidx1), np(&a.clen1)) }, "visible r1")?;
        ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 2, shape.c2 as i32, np(&a.cidx2), np(&a.clen2)) }, "visible r2")?;
        self.selection = Selection::AllVisible;
        self.cand_live = false;
        Ok(())
    }

    /// Every device operation of layers l0..l1 of a backbone step (the embedding first, the
    /// head last), in stream order, with no host decision that depends on buffer contents:
    /// the body a step graph segment captures.
    fn device_step(&mut self, shape: StepShape, l0: usize, l1: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let (d, hc, t) = (cfg.dim, cfg.hc_mult, shape.t);
        let capturing = self.use_graphs;
        self.shape = Some(shape);
        if l0 == 0 {
            self.begin_step(shape)?;
        }
        let rl = self.rep_layer();
        if shape.stage == Stage::Replay && l0 == rl {
            self.begin_replay(shape)?;
        }
        let cols = self.hash.layout.n_hash_cols();
        for l in l0..l1 {
            // a step's prompt rows leave the replay their state entering the replay layer
            if self.swa_replay && l == rl && shape.prompt && shape.stage != Stage::Replay {
                let (sr, a) = (self.stream.raw(), &self.act);
                ck(unsafe { ds_scatter_rows_u16(sr, np(&a.h), np(&self.rep_idx), np(&self.rep_h), t as i32, (hc * d) as i32) }, "replay keep")?;
                ck(unsafe { ds_scatter_rows_u16(sr, np(&a.pre), np(&self.rep_idx), np(&self.rep_pre), t as i32, (hc * 2) as i32) }, "replay keep mixes")?;
            }
            let t0 = Instant::now();
            // resident: borrow the layer out of the resident set for this block (put back below)
            let w = match self.resident.as_mut() {
                Some(res) => res.layers[l].take().expect("resident layer present"),
                None => {
                    let mut ld = Loader::new(&self.ckpt, &self.stream)?;
                    Layer::load(&mut ld, &cfg, l, None)?
                }
            };
            self.times.weights_s += t0.elapsed().as_secs_f64();
            let tc = Instant::now();
            let res = (|| -> Result<()> {
                if let Some(e) = &w.engram {
                    let sr = self.stream.raw();
                    let li = cfg.engram_layer_ids.iter().position(|&x| x == l).unwrap();
                    let a = &self.act;
                    ck(unsafe { ds_engram_dequant(sr, np(&self.eg_raw[li]), np(&self.eg_inv[li]), t as i32, cols as i32, np(&a.erows)) }, "engram dequant")?;
                    let k = cols * cfg.engram_head_dim;
                    let width = (hc + 1) * d;
                    if e.wkv.n == width {
                        gemm(sr, &a.mma_part, np(&a.erows), k, Wt::Fp8(&e.wkv), t, np(&a.ekv), false, width)?;
                    } else {
                        gemm(sr, &a.mma_part, np(&a.erows), k, Wt::Fp8(&e.wkv), t, np(&a.ekv_local), false, e.wkv.n)?;
                        self.tp_gather(np(&a.ekv_local), t, e.wkv.n, np(&a.ekv))?;
                    }
                    ck(unsafe {
                        ds_engram_gate(sr, np(&a.h), np(&a.ekv), np(&e.weight), t as i32, d as i32, hc as i32, cfg.norm_eps,
                                       (d as f64).powf(-0.5) as f32, if shape.vision { np(&self.vis_img) } else { nul() })
                    }, "engram gate")?;
                }
                // the DSpark main hidden reads the attention INPUT of its target layers
                if let Some(ti) = cfg.dspark_target_layer_ids.iter().position(|&x| x == l) {
                    let ld_main = cfg.dspark_target_layer_ids.len() * d;
                    ck(unsafe {
                        ds_hc_mean(self.stream.raw(), np(&self.act.h), self.act.main_h.ptr_at(ti * d), ld_main as i64,
                                   t as i32, d as i32, hc as i32)
                    }, "main hidden")?;
                }
                self.block(&w, Ctx::Main { l, c1: shape.c1, c2: shape.c2 }, t)
            })();
            match self.resident.as_mut() {
                Some(r) => r.layers[l] = Some(w),
                None => {
                    // streaming frees the layer: its kernels must be done with it first
                    self.stream.synchronize()?;
                    drop(w);
                }
            }
            res?;
            self.times.compute_s += tc.elapsed().as_secs_f64();
            if let (Some(dir), false) = (&self.dump_dir, capturing) {
                let mut hb = vec![0u16; self.live_rows * hc * d];
                self.act.h.download_async(0, &mut hb, &self.stream)?;
                self.stream.synchronize()?;
                dump_bf16(&dir.join(format!("h{l:02}.f32")), &hb)?;
            }
        }
        if l1 == self.n_layers && shape.head > 0 {
            let norm = np(&self.norm);
            self.head_dev(t, Some(shape.head), norm, false)?;
            let v = cfg.vocab_size;
            ck(unsafe { ds_argmax_bf16(self.stream.raw(), np(&self.act.logits), v as i64, shape.head as i32, v as i32, np(&self.act.am)) }, "head argmax")?;
        }
        Ok(())
    }

    /// The step's first device work: the embedding, and the all-visible compressed lists
    /// built from the positions (only read while n_c <= index_topk; past that the indexer's
    /// top-k replaces them).
    fn begin_step(&mut self, shape: StepShape) -> Result<()> {
        let t = shape.t;
        self.embed_dev(t)?;
        if shape.vision {
            let (d, hc) = (self.cfg.dim, self.cfg.hc_mult);
            for &(r0, rows, lane) in &self.vis_splices {
                let tab = &self.lane_vis[lane].as_ref().ok_or("an image step's lane has no images")?.table;
                ck(unsafe {
                    ds_splice_rows(self.stream.raw(), self.act.h.ptr_at(r0 * hc * d), np(tab), self.vis_idx.ptr_at(r0), rows as i32,
                                   d as i32, hc as i32)
                }, "image splice")?;
            }
        }
        let sr = self.stream.raw();
        let a = &self.act;
        ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 1, shape.c1 as i32, np(&a.cidx1), np(&a.clen1)) }, "visible r1")?;
        ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 2, shape.c2 as i32, np(&a.cidx2), np(&a.clen2)) }, "visible r2")?;
        self.selection = Selection::AllVisible;
        self.cand_live = false;
        Ok(())
    }

    /// After the step's device work: read back the logits, their argmaxes or their nuclei
    /// into the pinned buffers, advance the lanes.
    fn collect(&mut self, segs: &[Seg], plan: &Plan, readback: Readback) -> Result<()> {
        self.read_heads(plan, readback)?;
        let logits = matches!(readback, Readback::Logits);
        let nw: usize = plan.heads.iter().map(|h| h.1).sum();
        let v = self.cfg.vocab_size;
        for (i, sg) in segs.iter().enumerate() {
            let l = &mut self.lanes[sg.lane];
            l.last = (l.len, plan.firsts[i], sg.ids.len());
            l.step = self.step_no;
            l.len += sg.ids.len();
        }
        if let (Some(dir), true, true, 1) = (&self.dump_dir, logits, nw > 0, segs.len()) {
            dump_bf16(&dir.join("logits.f32"), &self.logits_host[(nw - 1) * v..nw * v])?;
        }
        Ok(())
    }

    /// The step's head rows read back as `readback` asks; the step is counted.
    fn read_heads(&mut self, plan: &Plan, readback: Readback) -> Result<()> {
        let st = &self.stream;
        let v = self.cfg.vocab_size;
        let nw: usize = plan.heads.iter().map(|h| h.1).sum();
        match readback {
            _ if nw == 0 => {}
            Readback::Logits => self.act.logits.download_async(0, &mut self.logits_host[..nw * v], st)?,
            Readback::Argmax => self.act.am.download_async(0, &mut self.am_host[..nw], st)?,
            Readback::Nucleus(sampling) => {
                let a = &self.act;
                let (mut temp, mut topp) = (Vec::with_capacity(nw), Vec::with_capacity(nw));
                for (&(_, n), sm) in plan.heads.iter().zip(sampling) {
                    temp.extend(std::iter::repeat_n(sm.temperature, n));
                    topp.extend(std::iter::repeat_n(sm.top_p, n));
                }
                a.s_temp.upload_async(0, &temp, st)?;
                a.s_topp.upload_async(0, &topp, st)?;
                ck(unsafe {
                    ds_nucleus(st.raw(), np(&a.logits), v as i64, nul(), 0, nw as i32, v as i32, np(&a.s_temp), np(&a.s_topp),
                               nul(), NUC_CAP as i32, np(&a.nuc_ids), np(&a.nuc_p), NUC_CAP as i64, np(&a.nuc_cnt), nul())
                }, "head nucleus")?;
                a.nuc_cnt.download_async(0, &mut self.nuc_cnt_host[..nw], st)?;
                a.nuc_ids.download_async(0, &mut self.nuc_ids_host[..nw * NUC_CAP], st)?;
                a.nuc_p.download_async(0, &mut self.nuc_p_host[..nw * NUC_CAP], st)?;
            }
        }
        st.synchronize()?;
        self.step_no += 1;
        Ok(())
    }

    /// hc_pre with the final pre_mix + `norm` over the step's `t` rows, then the head over
    /// the rows act.head_rows names (`gather`: how many) or over all `t`, into act.logits
    /// `[rows][V]` bf16 (streamed from NVMe unless resident).
    fn head_dev(&mut self, t: usize, gather: Option<usize>, norm: *mut c_void, draft: bool) -> Result<()> {
        let th = Instant::now();
        let cfg = &self.cfg;
        let (d, hc, v) = (cfg.dim, cfg.hc_mult, cfg.vocab_size);
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        ck(unsafe { ds_hc_pre_norm(sr, np(&a.h), np(&a.pre), norm, np(&a.head_all), t as i32, d as i32, hc as i32, cfg.norm_eps) }, "final norm")?;
        let (x, rows) = match gather {
            Some(n) => {
                ck(unsafe { ds_gather_rows_u16(sr, np(&a.head_all), np(&a.head_rows), np(&a.head_x), n as i32, d as i32) }, "head rows")?;
                (np(&a.head_x), n)
            }
            None => (np(&a.head_all), t),
        };
        if let Some(res) = &self.resident {
            let vl = res.head.len() / d;
            match (&res.draft_head, draft) {
                // the draft vocabulary's columns, [rows][draft_cols]; a split drafter's run, gathered
                (Some(f), true) if f.n == self.draft_cols => gemm(sr, &a.mma_part, x, d, Wt::Fp8(f), rows, np(&a.logits), false, f.n)?,
                (Some(f), true) => {
                    gemm(sr, &a.mma_part, x, d, Wt::Fp8(f), rows, np(&a.tp_logits), false, f.n)?;
                    self.tp_gather(np(&a.tp_logits), rows, f.n, np(&a.logits))?;
                }
                _ if vl == v => gemm(sr, &a.mma_part, x, d, Wt::Bf16 { w: np(&res.head), n: v, k: d }, rows, np(&a.logits), false, v)?,
                // tensor parallel: this rank's vocabulary rows, gathered into whole rows
                _ => {
                    gemm(sr, &a.mma_part, x, d, Wt::Bf16 { w: np(&res.head), n: vl, k: d }, rows, np(&a.tp_logits), false, vl)?;
                    self.tp_gather(np(&a.tp_logits), rows, vl, np(&a.logits))?;
                }
            }
        } else {
            let mut ld = Loader::new(&self.ckpt, st)?;
            let mut n0 = 0;
            while n0 < v {
                let n = HEAD_ROWS.min(v - n0);
                ld.read_into("head.weight", n0 * d * 2, (n0 + n) * d * 2, a.head_w.raw())?;
                gemm(sr, &a.mma_part, x, d, Wt::Bf16 { w: np(&a.head_w), n, k: d }, rows, a.logits.ptr_at(n0), false, v)?;
                n0 += n;
            }
        }
        self.times.head_s += th.elapsed().as_secs_f64();
        Ok(())
    }

    /// Lane 0: seed the drafter from the first `rows` rows of its last step.
    pub fn seed_draft(&mut self, rows: usize) -> Result<()> {
        self.seed_draft_batch(&[(0, rows)])
    }

    /// model.py::dspark_seed per (lane, rows): write the drafter's window keys for the first
    /// `rows` rows of that lane's segment in the last step, from their main hiddens (the
    /// target layers' hc means).
    pub fn seed_draft_batch(&mut self, reqs: &[(usize, usize)]) -> Result<()> {
        if reqs.iter().all(|r| r.1 == 0) {
            return Ok(());
        }
        self.mirror(|m| {
            m.u32(OP_SEED).u32(reqs.len() as u32);
            for &(lane, rows) in reqs {
                m.u32(lane as u32).u64(rows as u64);
            }
        })?;
        let Some(ds) = &self.dspark else { return Err("seed_draft needs the drafter (spec)".into()) };
        let (mut sel, mut pos, mut lane) = (Vec::new(), Vec::new(), Vec::new());
        for &(ln, rows) in reqs {
            if ln >= self.max_lanes {
                return Err(format!("seed_draft: lane {ln} of {}", self.max_lanes).into());
            }
            if rows == 0 {
                continue;
            }
            let l = &self.lanes[ln];
            let (p0, r0, t) = l.last;
            if l.step != self.step_no || rows > t {
                return Err(format!("seed_draft({rows}) of lane {ln}: not within its segment of the last step").into());
            }
            for i in 0..rows {
                sel.push((r0 + i) as i32);
                pos.push((p0 + i) as i32);
                lane.push(ln as i32);
            }
        }
        let m = sel.len();
        if m == 0 {
            return Ok(());
        }
        let cfg = &self.cfg;
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        let mi = m as i32;
        a.sel.upload_async(0, &sel, st)?;
        a.pos.upload_async(0, &pos, st)?;
        a.lane.upload_async(0, &lane, st)?;
        let k_main = cfg.dspark_target_layer_ids.len() * d;
        ck(unsafe { ds_gather_rows_u16(sr, np(&a.main_h), np(&a.sel), np(&a.main_g), mi, k_main as i32) }, "seed rows")?;
        gemm(sr, &a.mma_part, np(&a.main_g), k_main, Wt::Fp8(&ds.w.main_proj), m, np(&a.y), false, d)?;
        ck(unsafe { ds_rmsnorm(sr, np(&a.y), d as i64, np(&ds.w.main_norm), np(&a.main_x), d as i64, mi, d as i32, cfg.norm_eps) }, "main_norm")?;
        for (k, blk) in ds.blocks.iter().enumerate() {
            gemm(sr, &a.mma_part, np(&a.main_x), d, Wt::Fp8(&blk.wkv), m, np(&a.kv0), false, hd)?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kv0), hd as i64, np(&blk.kv_norm), np(&a.kv), hd as i64, mi, hd as i32, cfg.norm_eps) }, "draft seed kv_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kv), hd as i64, mi, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(&self.rope_w), 0) }, "draft seed rope")?;
            ck(unsafe {
                ds_scatter_ring(sr, np(&a.kv), np(&self.caches.mtp_ring[k]), self.ring as i32, np(&a.pos), mi, hd as i32,
                                np(&a.lane), (self.ring * hd) as i64)
            }, "draft seed ring")?;
        }
        st.synchronize()?;
        Ok(())
    }

    /// Lane 0: draft after `tok` at position last_main_pos + 1.
    pub fn draft(&mut self, tok: u32, last_main_pos: usize) -> Result<Vec<u32>> {
        Ok(self.draft_batch(&[(0, tok, last_main_pos)])?.remove(0))
    }

    /// model.py::dspark_draft, greedy, per (lane, tok, last_main_pos): the block [tok,
    /// noise x 4] at positions last_main_pos+1.., through the three DSpark blocks, then the
    /// markov-biased argmax chain. Returns each lane's `dspark_block_size` drafted tokens.
    /// Resident with graphs, the whole draft is one CUDA graph per number of lanes, with a
    /// single read-back.
    pub fn draft_batch(&mut self, reqs: &[(usize, u32, usize)]) -> Result<Vec<Vec<u32>>> {
        Ok(self.draft_inner(reqs, None)?.into_iter().map(|d| d.0).collect())
    }

    /// `draft_batch` with sampled drafts (fastdecode.py's DRAFT_NUCLEUS draft): per lane
    /// its `sampling`, and per lane and draft step a uniform in [0, 1) - `u` is
    /// `[lane][dspark_block_size]`. Each step's token is the inverse-CDF draw from the
    /// markov-biased logits' nucleus at that lane's temperature and top_p, cut at
    /// DRAFT_NUCLEUS_K tokens, and that distribution comes back with it: verification divides
    /// by exactly it. Greedy lanes (temperature <= 0) get their argmax and a one-token q.
    pub fn draft_batch_sampled(&mut self, reqs: &[(usize, u32, usize)], sampling: &[Sampling], u: &[f32])
                               -> Result<Vec<(Vec<u32>, Vec<Nucleus>)>> {
        let b = self.cfg.dspark_block_size;
        if sampling.len() != reqs.len() || u.len() != reqs.len() * b {
            return Err(format!("draft: {} lanes, {} samplings, {} uniforms", reqs.len(), sampling.len(), u.len()).into());
        }
        if sampling.iter().all(|s| s.is_greedy()) {
            let drafts = self.draft_inner(reqs, None)?;
            return Ok(drafts.into_iter().map(|(d, _)| {
                let q = d.iter().map(|&t| Nucleus { ids: vec![t], probs: vec![1.0] }).collect();
                (d, q)
            }).collect());
        }
        self.draft_inner(reqs, Some((sampling, u)))
    }

    fn draft_inner(&mut self, reqs: &[(usize, u32, usize)], sampled: Option<(&[Sampling], &[f32])>)
                   -> Result<Vec<(Vec<u32>, Vec<Nucleus>)>> {
        if self.dspark.is_none() {
            return Err("draft needs the drafter (spec)".into());
        }
        let n = reqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.mirror(|m| {
            m.u32(if sampled.is_some() { OP_DRAFT_SAMPLED } else { OP_DRAFT }).u32(n as u32);
            for &(lane, tok, last) in reqs {
                m.u32(lane as u32).u32(tok).u64(last as u64);
            }
            if let Some((sm, u)) = sampled {
                for x in sm {
                    m.u32(x.temperature.to_bits()).u32(x.top_p.to_bits());
                }
                m.u32s(&u.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            }
        })?;
        if n > self.max_draft_lanes() {
            return Err(format!("draft takes at most {} lanes, got {n}", self.max_draft_lanes()).into());
        }
        let b = self.cfg.dspark_block_size;
        let noise = self.cfg.dspark_noise_token_id;
        let mut seen = vec![false; self.max_lanes];
        let (mut ids, mut pos, mut win, mut lane) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(ln, tok, last) in reqs {
            if ln >= self.max_lanes || std::mem::replace(&mut seen[ln], true) {
                return Err(format!("draft: lane {ln} out of range or repeated").into());
            }
            if last + b >= self.max_seq + 8 {
                return Err(format!("draft past max_seq at position {last}").into());
            }
            for j in 0..b {
                ids.push(if j == 0 { tok } else { noise });
                pos.push((last + 1 + j) as i32);
                win.push(last as i32);
                lane.push(ln as i32);
            }
        }
        {
            let st = &self.stream;
            let a = &self.act;
            a.pos.upload_async(0, &pos, st)?;
            a.dwin.upload_async(0, &win, st)?;
            a.lane.upload_async(0, &lane, st)?;
            a.d_idx.upload_async(0, &reqs.iter().map(|r| r.1 as i32).collect::<Vec<_>>(), st)?;
            if let Some((sm, u)) = sampled {
                a.d_temp.upload_async(0, &sm.iter().map(|x| x.temperature).collect::<Vec<_>>(), st)?;
                a.d_topp.upload_async(0, &sm.iter().map(|x| x.top_p).collect::<Vec<_>>(), st)?;
                // [step][lane], as the chain reads them
                a.d_u.upload_async(0, &(0..b * n).map(|k| u[(k % n) * b + k / n]).collect::<Vec<_>>(), st)?;
            }
        }
        self.live_rows = n * b;
        self.embed_host(&ids, n * b)?;
        let key = (n, sampled.is_some());
        if self.use_graphs {
            if !self.draft_graphs.contains_key(&key) {
                self.stream.begin_capture()?;
                let rec = self.draft_dev(n, key.1);
                let graph = self.stream.end_capture();
                rec?;
                self.draft_graphs.insert(key, graph?);
                self.graph_captures += 1;
            } else {
                self.graph_replays += 1;
            }
            self.draft_graphs[&key].launch(&self.stream)?;
        } else {
            self.draft_dev(n, key.1)?;
        }
        let mut out = vec![0i32; (b + 1) * n];
        let (mut cx, mut cm) = (vec![0f32; n * b], vec![0f32; b * n]);
        self.act.d_idx.download_async(0, &mut out, &self.stream)?;
        self.act.conf_x.download_async(0, &mut cx, &self.stream)?;
        self.act.conf_m.download_async(0, &mut cm, &self.stream)?;
        let kq = DRAFT_NUCLEUS_K;
        if sampled.is_some() {
            let a = &self.act;
            a.dq_cnt.download_async(0, &mut self.nuc_cnt_host[..b * n], &self.stream)?;
            a.dq_ids.download_async(0, &mut self.nuc_ids_host[..b * n * kq], &self.stream)?;
            a.dq_p.download_async(0, &mut self.nuc_p_host[..b * n * kq], &self.stream)?;
        }
        self.stream.synchronize()?;
        self.draft_conf = (0..n).map(|j| (0..b).map(|i| cx[j * b + i] + cm[i * n + j]).collect()).collect();
        Ok((0..n)
            .map(|j| {
                let toks: Vec<u32> = (0..b).map(|i| out[(i + 1) * n + j] as u32).collect();
                // a cut-short draft nucleus is still exactly the distribution drawn from
                let q = if sampled.is_some() {
                    (0..b)
                        .map(|i| {
                            let r = i * n + j;
                            let c = self.nuc_cnt_host[r].unsigned_abs() as usize;
                            let (ids, p) = (&self.nuc_ids_host[r * kq..r * kq + c], &self.nuc_p_host[r * kq..r * kq + c]);
                            let id = |c: i32| self.draft_ids.as_ref().map_or(c as u32, |(_, t)| t[c as usize]);
                            Nucleus { ids: ids.iter().map(|&x| id(x)).collect(), probs: p.to_vec() }
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                (toks, q)
            })
            .collect())
    }

    /// The last draft_batch's DSpark confidence logits (raw, before the sigmoid), per lane
    /// and draft position: the conditional acceptance estimate the depth policy reads.
    pub fn draft_conf(&self) -> &[Vec<f32>] {
        &self.draft_conf
    }

    /// The draft's device work for `n` lanes: embedding, the three blocks, the head, and the
    /// markov chain entirely on the device (each step's embedding rows are gathered by the
    /// indices the previous argmax wrote, so the chain needs no host sync).
    fn draft_dev(&mut self, n: usize, sampled: bool) -> Result<()> {
        let cfg = self.cfg.clone();
        let b = cfg.dspark_block_size;
        let rows = n * b;
        self.embed_dev(rows)?;
        let blocks = std::mem::take(&mut self.dspark.as_mut().unwrap().blocks);
        let res = (|| -> Result<()> {
            for (k, w) in blocks.iter().enumerate() {
                self.block(w, Ctx::Draft { k }, rows)?;
            }
            Ok(())
        })();
        self.dspark.as_mut().unwrap().blocks = blocks;
        res?;
        let norm = np(&self.dspark.as_ref().unwrap().w.norm);
        self.head_dev(rows, None, norm, true)?;
        let ds = self.dspark.as_ref().unwrap();
        let (v, r, d, hc) = (cfg.vocab_size, cfg.dspark_markov_rank, cfg.dim, cfg.hc_mult);
        let sr = self.stream.raw();
        let a = &self.act;
        let ni = n as i32;
        // confidence, hc_pre half: the reference feeds the head the UN-normed hc_pre output
        // (inference/model.py::DSparkBlock.forward_head)
        ck(unsafe { ds_hc_pre_norm(sr, np(&a.h), np(&a.pre), nul(), np(&a.xpre), rows as i32, d as i32, hc as i32, cfg.norm_eps) }, "conf hc_pre")?;
        ck(unsafe { ds_linear_f32w(sr, np(&a.xpre), d as i64, np(&ds.w.conf_w), rows as i32, 1, d as i32, 1, np(&a.conf_x), 1, nul()) }, "conf x")?;
        // logits row j * b + i is lane j's draft step i, over the draft vocabulary's l columns
        let l = self.draft_cols;
        for i in 0..b {
            ck(unsafe { ds_gather_rows_u16(sr, np(&ds.w.markov_embed), a.d_idx.ptr_at(i * n), np(&a.markov_e), ni, r as i32) }, "markov embed")?;
            match &self.markov_rows {
                None => gemm(sr, &a.mma_part, np(&a.markov_e), r, Wt::Bf16 { w: np(&ds.w.markov_head), n: v, k: r }, n, np(&a.markov_b), false, v)?,
                Some(m) if m.len() / r == l => {
                    gemm(sr, &a.mma_part, np(&a.markov_e), r, Wt::Bf16 { w: np(m), n: l, k: r }, n, np(&a.markov_b), false, l)?
                }
                // a split drafter: this rank's columns, gathered
                Some(m) => {
                    let mn = m.len() / r;
                    gemm(sr, &a.mma_part, np(&a.markov_e), r, Wt::Bf16 { w: np(m), n: mn, k: r }, n, np(&a.markov_bl), false, mn)?;
                    self.tp_gather(np(&a.markov_bl), n, mn, np(&a.markov_b))?;
                }
            }
            // confidence, markov half: the embedding of step i's input token
            ck(unsafe {
                ds_linear_f32w(sr, np(&a.markov_e), r as i64, ds.w.conf_w.ptr_at(d), ni, 1, r as i32, 1, a.conf_m.ptr_at(i * n), 1, nul())
            }, "conf markov")?;
            if sampled {
                let kq = DRAFT_NUCLEUS_K;
                ck(unsafe {
                    ds_nucleus(sr, a.logits.ptr_at(i * l), (b * l) as i64, np(&a.markov_b), l as i64, ni, l as i32, np(&a.d_temp),
                               np(&a.d_topp), a.d_u.ptr_at(i * n), kq as i32, a.dq_ids.ptr_at(i * n * kq), a.dq_p.ptr_at(i * n * kq),
                               kq as i64, a.dq_cnt.ptr_at(i * n), a.d_idx.ptr_at((i + 1) * n))
                }, "markov nucleus")?;
            } else {
                ck(unsafe {
                    ds_argmax_sum_bf16(sr, a.logits.ptr_at(i * l), (b * l) as i64, np(&a.markov_b), l as i64, ni, l as i32,
                                       a.d_idx.ptr_at((i + 1) * n))
                }, "markov argmax")?;
            }
            // a listed vocabulary's column -> its token id, before the next step embeds it
            if let Some((ids, _)) = &self.draft_ids {
                ck(unsafe { ds_map_i32(sr, np(ids), a.d_idx.ptr_at((i + 1) * n), ni) }, "draft ids")?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hc_mixes(&self, fn_w: &DeviceBuffer<f32>, base: &DeviceBuffer<f32>, scale: &DeviceBuffer<f32>, t: usize,
                pre: &DeviceBuffer<f32>, post: &DeviceBuffer<f32>, comb: &DeviceBuffer<f32>) -> Result<()> {
        let cfg = &self.cfg;
        let a = &self.act;
        let sr = self.stream.raw();
        let k = cfg.hc_mult * cfg.dim;
        let n = cfg.hc_mix_width();
        ck(unsafe { ds_row_rsqrt(sr, np(&a.h), k as i64, t as i32, k as i32, cfg.norm_eps, np(&a.rsq)) }, "hc rsqrt")?;
        // fp32 weights have no tensor-core path: the FMA kernel, its 16-row tiles in one
        // launch (the 2 MB weight stays in L2 across them; launched slice by slice, an 8K
        // prefill took 44,800 launches here, 0.74 s of rank 0's GPU time)
        ck(unsafe {
            ds_linear_f32w(sr, np(&a.h), k as i64, np(fn_w), t as i32, n as i32, k as i32, HC_SPLITS as i32, nul(), n as i64,
                           np(&a.partial))
        }, "hc mix")?;
        ck(unsafe { ds_splits_reduce(sr, np(&a.partial), HC_SPLITS as i32, t as i32, n as i32, nul(), np(&a.mixes), n as i64) }, "hc reduce")?;
        ck(unsafe {
            ds_hc_sinkhorn(sr, np(&a.mixes), np(&a.rsq), np(scale), np(base), t as i32, cfg.hc_sinkhorn_iters as i32,
                           cfg.hc_eps, np(pre), np(post), np(comb))
        }, "hc sinkhorn")
    }

    fn block(&mut self, w: &Layer, ctx: Ctx, t: usize) -> Result<()> {
        let (d, hc) = (self.cfg.dim, self.cfg.hc_mult);
        let sr = self.stream.raw();
        let eps = self.cfg.norm_eps;
        // attention site
        self.hc_mixes(&w.hc_attn_fn, &w.hc_attn_base, &w.hc_attn_scale, t, &self.act.apre, &self.act.apost, &self.act.acomb)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_pre_norm(sr, np(&a.h), np(&a.pre), np(&w.attn_norm), np(&a.y), t as i32, d as i32, hc as i32, eps) }, "attn hc_pre")?;
        }
        self.attention(w, ctx, t)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_post(sr, np(&a.attn_out), np(&a.h), np(&a.apost), np(&a.acomb), np(&a.h), t as i32, d as i32) }, "attn hc_post")?;
        }
        // ffn site: mixes of the post-attention stream, hc_pre with the attention site's pre
        self.hc_mixes(&w.hc_ffn_fn, &w.hc_ffn_base, &w.hc_ffn_scale, t, &self.act.fpre, &self.act.fpost, &self.act.fcomb)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_pre_norm(sr, np(&a.h), np(&a.apre), np(&w.ffn_norm), np(&a.y), t as i32, d as i32, hc as i32, eps) }, "ffn hc_pre")?;
        }
        self.moe(w, t)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_post(sr, np(&a.moe_out), np(&a.h), np(&a.fpost), np(&a.fcomb), np(&a.h), t as i32, d as i32) }, "ffn hc_post")?;
        }
        // the next layer's attention mixes the stream with this layer's ffn pre. A copy, not a
        // buffer swap: a swap would change which pointer each layer reads from step to step,
        // and a captured graph bakes the pointers in.
        self.act.pre.copy_from(0, &self.act.fpre, 0, t * hc, &self.stream)?;
        Ok(())
    }

    fn attention(&mut self, w: &Layer, ctx: Ctx, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let (ratio, l) = match ctx {
            Ctx::Main { l, .. } => (cfg.compress_ratios[l], l),
            Ctx::Draft { .. } => (0, usize::MAX),
        };
        // independent of q and the window: the step's new compressed positions, which this
        // layer's own attention already reads (engine/model.py::_compressed)
        if let (Ctx::Main { .. }, Some(comp)) = (ctx, &w.compressor) {
            self.compress(w, comp, l, t)?;
        }
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        // this rank's heads (all of them on one box and in the drafter's blocks)
        let h = w.wq_b.n / hd;
        let bsz = cfg.dspark_block_size;
        let ti = t as i32;
        {
            let st = &self.stream;
            let sr = st.raw();
            let table = if ratio > 0 { &self.rope_c } else { &self.rope_w };
            let a = &self.act;
            let part = &a.mma_part;
            gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.wq_a), t, np(&a.qa), false, cfg.q_lora_rank)?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.qa), cfg.q_lora_rank as i64, np(&w.q_norm), np(&a.qr), cfg.q_lora_rank as i64, ti, cfg.q_lora_rank as i32, cfg.norm_eps) }, "q_norm")?;
            gemm(sr, part, np(&a.qr), cfg.q_lora_rank, Wt::Fp8(&w.wq_b), t, np(&a.q), false, h * hd)?;
            ck(unsafe { ds_rope(sr, np(&a.q), (h * hd) as i64, ti, h as i32, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 0) }, "q rope")?;
            gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.wkv), t, np(&a.kv0), false, hd)?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kv0), hd as i64, np(&w.kv_norm), np(&a.kv), hd as i64, ti, hd as i32, cfg.norm_eps) }, "kv_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kv), hd as i64, ti, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 0) }, "kv rope")?;
            match ctx {
                Ctx::Main { l, .. } => ck(unsafe {
                    ds_scatter_ring(sr, np(&a.kv), np(&self.caches.ring[l]), self.ring as i32, np(&a.pos), ti, hd as i32,
                                    np(&a.lane), (self.ring * hd) as i64)
                }, "ring write")?,
                // the draft block's own keys go to its lane's block keys, never to the ring:
                // row r of a lane's block is key slot r % block
                Ctx::Draft { .. } => ck(unsafe {
                    ds_scatter_ring(sr, np(&a.kv), np(&self.caches.dkv), bsz as i32, np(&a.dslot), ti, hd as i32, np(&a.lane),
                                    (bsz * hd) as i64)
                }, "draft keys")?,
            }
        }
        if ratio > 0 && cfg.index_source_layers.contains(&l) {
            self.select(w, l, t)?;
        }
        let st = &self.stream;
        let sr = st.raw();
        let table = if ratio > 0 { &self.rope_c } else { &self.rope_w };
        let a = &self.act;
        // (ring, per-row window end, extra keys, their per-row indices, width, extra keys' lane stride)
        // the backbone's compressed rows may be packed; the draft block's keys never are
        let fp4 = (self.packed_kv && matches!(ctx, Ctx::Main { .. })) as i32;
        let (ring, win_pos, ckv_ptr, cidx_ptr, c, ckv_ls) = match ctx {
            Ctx::Main { l, c1, c2 } => {
                let ring = &self.caches.ring[l];
                if ratio > 0 {
                    let src = cfg.kv_source_of(l).expect("compressed layer has a kv source");
                    let (cidx, c) = match self.selection {
                        Selection::Indexed => (&a.topk, cfg.index_topk),
                        Selection::AllVisible if ratio == 1 => (&a.cidx1, c1),
                        Selection::AllVisible => (&a.cidx2, c2),
                    };
                    let row = if self.packed_kv { PACKED_ROW / 2 } else { hd };
                    (ring, &a.pos, np(&self.caches.ckv[&src]), np(cidx), c, self.caches.crows[&src] * row)
                } else {
                    (ring, &a.pos, nul(), nul(), 0, 0)
                }
            }
            // model.py::attention with mtp_extra: the window is the main stream's last 128
            // positions in this block's seeded ring, for every draft query alike, plus every
            // draft key of the lane's block (no causal mask among them)
            Ctx::Draft { k } => (&self.caches.mtp_ring[k], &a.dwin, np(&self.caches.dkv), np(&a.dcidx), bsz, bsz * hd),
        };
        // decode rows take the split-key kernel, prompt rows the one-pass one (a row's kernel
        // depends on what the row is, never on what shares its step); a decode-only step is
        // the split-key launch alone, a prompt-only one the one-pass launch alone
        let scale = (hd as f64).powf(-0.5) as f32;
        // the drafter's rows are always the split-key kernel's
        let (prompt, decode) = match ctx {
            Ctx::Main { .. } => {
                let shape = self.shape.expect("attention runs inside a step");
                (shape.prompt, shape.decode)
            }
            Ctx::Draft { .. } => (false, true),
        };
        let (split_rows, prompt_rows) = (decode || !prompt, prompt);
        let flags = if split_rows && prompt_rows { np(&a.decode_rows) } else { nul() };
        // the decoder layers' rings hold KV from each lane's replay floor on
        let floor = match ctx {
            Ctx::Main { l, .. } if self.swa_replay && l > self.rep_layer() => np(&self.win_floor),
            _ => nul(),
        };
        if split_rows {
            ck(unsafe {
                ds_attn_decode(sr, np(&a.q), np(ring), self.ring as i32, np(win_pos), cfg.window_size as i32, floor, ckv_ptr, cidx_ptr,
                               c as i32, np(&w.attn_sink), scale, np(&a.o), ti, h as i32, np(&a.lane), (self.ring * hd) as i64,
                               ckv_ls as i64, np(&a.attn_work), a.attn_work.len() as i64, flags, fp4)
            }, "attention")?;
        }
        if prompt_rows {
            // a block takes all 32 of this rank's heads, so each key loads once a row: 6.87 ms
            // against 7.39 at 16 heads (2,048 rows over 128 + 1,024 keys, examples/bench_attn.rs
            // --prefill; Python's tools/prefill_attn.py at 32 heads, 7.35)
            ck(unsafe {
                ds_attn_prefill(sr, np(&a.q), np(ring), self.ring as i32, np(win_pos), cfg.window_size as i32, floor, ckv_ptr, cidx_ptr,
                                c as i32, np(&w.attn_sink), scale, np(&a.o), ti, h as i32, np(&a.lane), (self.ring * hd) as i64,
                                ckv_ls as i64, flags, 32, fp4)
            }, "attention (prompt rows)")?;
        }
        ck(unsafe { ds_rope(sr, np(&a.o), (h * hd) as i64, ti, h as i32, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 1) }, "o rope")?;
        let og = cfg.o_groups;
        let gk = cfg.n_heads * hd / og;
        gemm_grouped(sr, &a.mma_part, np(&a.o), h * hd, gk, cfg.o_lora_rank, &w.wo_a, t, np(&a.oa), w.wo_a.n)?;
        if w.wo_b.n == d {
            return gemm(sr, &a.mma_part, np(&a.oa), og * cfg.o_lora_rank, Wt::Fp8(&w.wo_b), t, np(&a.attn_out), false, d);
        }
        // tensor parallel: this rank's heads made its wo_a groups; gather them whole, then
        // this rank's rows of wo_b, then gather those
        self.tp_gather(np(&a.oa), t, w.wo_a.n, np(&a.oa_full))?;
        gemm(sr, &a.mma_part, np(&a.oa_full), og * cfg.o_lora_rank, Wt::Fp8(&w.wo_b), t, np(&a.attn_local), false, w.wo_b.n)?;
        self.tp_gather(np(&a.attn_local), t, w.wo_b.n, np(&a.attn_out))
    }

    /// model.py::_indexer at an index-source layer: which compressed positions this layer
    /// and the following ones (until the next index source) attend over.
    fn select(&mut self, w: &Layer, l: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let r = cfg.compress_ratios[l];
        let shape = self.shape.expect("select runs inside a backbone step");
        // the key range is the bucketed count; keys past a row's true count are masked by clen
        let Some(n_c) = (if r == 1 { shape.n1 } else { shape.n2 }) else {
            // top-k of every visible position is every visible position
            self.selection = Selection::AllVisible;
            return Ok(());
        };
        let ix = w.indexer.as_ref().expect("index-source layer has indexer weights");
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        let ti = t as i32;
        let (ih, ihd, rd) = (cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim);
        let nmax = self.max_seq + 1;
        gemm(sr, &a.mma_part, np(&a.qr), cfg.q_lora_rank, Wt::Fp8(&ix.wq_b), t, np(&a.iq), false, ih * ihd)?;
        // the indexer's queries always take the compressed-rope table (model.py: freqs_c[pos])
        ck(unsafe { ds_rope(sr, np(&a.iq), (ih * ihd) as i64, ti, ih as i32, ihd as i64, (ihd - rd) as i32, rd as i32, np(&a.pos), np(&self.rope_c), 0) }, "index q rope")?;
        gemm(sr, &a.mma_part, np(&a.y), cfg.dim, Wt::Bf16 { w: np(&ix.weights_proj), n: ih, k: cfg.dim }, t, np(&a.iwb), false, ih)?;
        let scale = ((ihd as f64).powf(-0.5) * (ih as f64).powf(-0.5)) as f32;
        ck(unsafe { ds_bf16_to_f32_scale(sr, np(&a.iwb), (t * ih) as i64, scale, np(&a.iwf)) }, "index weights")?;
        let src = cfg.kv_source_of(l).expect("compressed layer has a kv source");
        let clen = if r == 1 { &a.clen1 } else { &a.clen2 };
        let cand_src = cfg.candidate_source_layer;
        let use_cand = self.cand_live && cand_src >= 0 && (l as i64) > cand_src;
        // scores on the tensor cores (0.69 ms against the FMA kernel's 12.45 at 2,048 rows over
        // 2,240 keys, examples/bench_index.rs; the head sum's order kept, each dot accumulated
        // in the mma's order). A decode step's rows fill one 8-row block, so its keys are split
        // over blocks (examples/bench_index_decode.rs): the FMA kernel decode used before cost
        // a step more with every compressed key the context added, as the Python engine's
        // einsum (tensor cores) did not
        let chunk = if self.eager_step { 0 } else { INDEX_DECODE_CHUNK };
        let block = 8usize;
        let topk_blocks = 2048usize;
        let cand_step = l as i64 == cand_src && shape.cand;
        if cand_step {
            a.cand.zero_async(st)?;
        }
        let (nb, k, kw) = (n_c.div_ceil(block), cfg.index_topk.min(n_c), cfg.index_topk);
        let kb = topk_blocks.min(nb);
        // rows a slice: all of a step unless its scores would outgrow INDEX_SCORES (a whole
        // decode step is always one slice, so its graph is unchanged)
        let rc = (index_scores_budget() / n_c.max(1)).min(t).max(8.min(t));
        for r0 in (0..t).step_by(rc) {
            let m = rc.min(t - r0) as i32;
            let (q, keys, iw) = (a.iq.ptr_at(r0 * ih * ihd), np(&self.caches.ik[&src]), a.iwf.ptr_at(r0 * ih));
            let cand = if use_cand { a.cand.ptr_at(r0 * nmax) } else { nul() };
            let kstride = (self.caches.crows[&src] * ihd) as i64;
            ck(unsafe {
                if chunk == 0 {
                    ds_index_scores_mma(sr, q, keys, iw, clen.ptr_at(r0), cand, nmax as i64, n_c as i32, np(&a.scores), n_c as i64, m,
                                        a.lane.ptr_at(r0), kstride)
                } else {
                    ds_index_scores_mma_chunked(sr, q, keys, iw, clen.ptr_at(r0), cand, nmax as i64, n_c as i32, np(&a.scores),
                                                n_c as i64, m, a.lane.ptr_at(r0), kstride, chunk as i32)
                }
            }, "index scores")?;
            if cand_step {
                // model.py::_select_candidates: below topk_blocks blocks every block is kept,
                // so the mask only exists where it can drop something
                ck(unsafe { ds_candidate_blocks(sr, np(&a.scores), n_c as i64, n_c as i32, block as i32, clen.ptr_at(r0), np(&a.bmax), nb as i64, m) }, "candidate blocks")?;
                ck(unsafe { ds_topk_select(sr, np(&a.bmax), nb as i64, nb as i32, kb as i32, a.blocks.ptr_at(r0 * kb), kb as i32, m) }, "candidate select")?;
                ck(unsafe { ds_candidate_set(sr, a.blocks.ptr_at(r0 * kb), kb as i32, block as i32, n_c as i32, a.cand.ptr_at(r0 * nmax), nmax as i64, m) }, "candidate set")?;
            }
            ck(unsafe { ds_topk_select(sr, np(&a.scores), n_c as i64, n_c as i32, k as i32, a.topk.ptr_at(r0 * kw), kw as i32, m) }, "index topk")?;
        }
        if cand_step {
            self.cand_live = true;
        }
        self.selection = Selection::Indexed;
        self.indexed_layers += 1;
        Ok(())
    }

    /// engine/model.py::_compressed for a kv-source layer: new compressed positions of this
    /// step into each row's lane's cache (and their index keys). Every row is computed and
    /// only the rows whose position completes a group are scattered, so the work never
    /// depends on position values — the same kernels in both modes and inside the step graph.
    fn compress(&mut self, w: &Layer, comp: &crate::weights::Compressor, l: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let st = &self.stream;
        let sr = st.raw();
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        let r = cfg.compress_ratios[l];
        let a = &self.act;
        let part = &a.mma_part;
        let ti = t as i32;
        let cpos = if r == 1 { &self.cpos1 } else { &self.cpos2 };
        let wkv = Wt::Bf16 { w: np(&comp.wkv), n: hd, k: d };
        if r > 1 {
            let wgate = comp.wgate.as_ref().expect("ratio-2 compressor has a gate");
            let r2 = &self.caches.r2[&l];
            gemm(sr, part, np(&a.y), d, wkv, t, np(&a.kvl), true, hd)?;
            gemm(sr, part, np(&a.y), d, Wt::Bf16 { w: np(wgate), n: hd, k: d }, t, np(&a.sc), true, hd)?;
            // pairs read the pending position before the update below replaces it
            ck(unsafe { ds_compress_rows(sr, np(&a.kvl), np(&a.sc), np(&r2.pkv), np(&r2.psc), np(&a.pos), np(&a.lat), ti, hd as i32, np(&a.lane)) }, "compress rows")?;
            // one block per lane slot: a lane absent from the step has an empty segment
            ck(unsafe {
                ds_pending_update(sr, np(&a.kvl), np(&a.sc), np(&a.pos), np(&a.seg_first), np(&a.seg_count), np(&a.seg_lane),
                                  (self.max_lanes + 1) as i32, np(&r2.pkv), np(&r2.psc), np(&r2.flag), np(&r2.mem_kv),
                                  np(&r2.mem_sc), np(&r2.before_kv), np(&r2.before_sc), np(&r2.before_flag),
                                  np(&r2.mem_start), hd as i32, TMAX as i32)
            }, "pending update")?;
        } else {
            gemm(sr, part, np(&a.y), d, wkv, t, np(&a.lat), false, hd)?;
        }
        ck(unsafe { ds_rmsnorm(sr, np(&a.lat), hd as i64, np(&comp.norm), np(&a.latn), hd as i64, ti, hd as i32, cfg.norm_eps) }, "comp norm")?;
        let rows = self.caches.crows[&l];
        if let Some(key) = w.indexer.as_ref().and_then(|ix| ix.key.as_ref()) {
            let ihd = cfg.index_head_dim;
            gemm(sr, part, np(&a.latn), hd, Wt::Bf16 { w: np(&key.wk), n: ihd, k: hd }, t, np(&a.kidx), false, ihd)?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kidx), ihd as i64, np(&key.k_norm), np(&a.kidxn), ihd as i64, ti, ihd as i32, cfg.norm_eps) }, "index k_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kidxn), ihd as i64, ti, 1, ihd as i64, (ihd - rd) as i32, rd as i32, np(cpos), np(&self.rope_c), 0) }, "index k rope")?;
            ck(unsafe {
                ds_scatter_compressed(sr, np(&a.kidxn), ihd as i64, np(&self.caches.ik[&l]), np(&a.pos), r as i32, ti, ihd as i32,
                                      np(&a.lane), (rows * ihd) as i64)
            }, "index k scatter")?;
        }
        ck(unsafe { ds_rope(sr, np(&a.latn), hd as i64, ti, 1, hd as i64, (hd - rd) as i32, rd as i32, np(cpos), np(&self.rope_c), 0) }, "comp rope")?;
        // TIDELM_NO_QDQ=1 is a diagnostic only; serving always quantizes, as the checkpoint's
        // model does. Measured 2026-10-24 against tools/v41_ref.py (rust/tools/ref_forward.py,
        // 6-token prompt, all 40 layers): with QDQ off on both sides every row tracks the
        // reference at bf16 level (5e-3..3e-2 rel L2, logits corr 0.99994, top-10 10/10); with
        // it on, latents that agree to 5e-3 come out 5-13% apart, because a 0.5% wobble in a
        // group's amax flips its E4M3 scale (12.5% steps) and requantizes all 16 values — 50
        // router flips instead of 8. Two correct implementations cannot be token-identical
        // through this cache; judge parity with QDQ off, and quality by NLL.
        let ckv = &self.caches.ckv[&l];
        if self.packed_kv {
            // the same QDQ, stored as its codes and scales (engine/packed_kv.py's write)
            ck(unsafe {
                ds_scatter_compressed_fp4(sr, np(&a.latn), hd as i64, np(ckv), np(&a.pos), r as i32, ti, hd as i32, np(&a.lane),
                                          (rows * PACKED_ROW) as i64)
            }, "ckv scatter (packed)")?;
        } else {
            if !self.no_qdq {
                ck(unsafe { ds_fp4_qdq_e4m3(sr, np(&a.latn), (t * hd) as i64) }, "comp qdq")?;
            }
            ck(unsafe {
                ds_scatter_compressed(sr, np(&a.latn), hd as i64, np(ckv), np(&a.pos), r as i32, ti, hd as i32, np(&a.lane),
                                      (rows * hd) as i64)
            }, "ckv scatter")?;
        }
        if let Some(dir) = &self.dump_dir {
            // lane 0's part (single-lane steps)
            let n = (self.lanes[0].len + self.live_rows) / r;
            let mut v = vec![0u16; n * hd];
            if self.packed_kv {
                let tmp = DeviceBuffer::<u16>::new(n.max(1) * hd)?;
                ck(unsafe { ds_unpack_fp4_rows(sr, np(ckv), np(&tmp), n as i32) }, "ckv unpack")?;
                tmp.download_async(0, &mut v, st)?;
            } else {
                ckv.download_async(0, &mut v, st)?;
            }
            st.synchronize()?;
            dump_bf16(&dir.join(format!("ckv{l:02}.f32")), &v)?;
        }
        Ok(())
    }

    /// engine/model.py::moe on one box: routed experts through the resident LUT, or streamed
    /// through the arena.
    fn moe(&mut self, w: &Layer, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let st = &self.stream;
        let sr = st.raw();
        let (d, e, k) = (cfg.dim, w.n_experts, w.topk);
        let ti = t as i32;
        let p = t * k;
        let index = |pre: &str| w.prefix.strip_prefix(pre).and_then(|r| r.trim_end_matches('.').parse::<usize>().ok());
        let (layer, block) = (index("layers."), index("mtp."));
        {
            let a = &self.act;
            let keep = match (layer, block) {
                (Some(l), _) => self.keep.as_ref().map(|kb| kb.ptr_at(l * e)).unwrap_or(nul()),
                (_, Some(b)) => self.dspark_keep.as_ref().map(|kb| kb.ptr_at(b * e)).unwrap_or(nul()),
                _ => nul(),
            };
            gemm(sr, &a.mma_part, np(&a.y), d, Wt::Bf16 { w: np(&w.gate_w), n: e, k: d }, t, np(&a.z), true, e)?;
            // a backbone layer of a recording engine records its lanes' demand on the way, and
            // a decode step's leaves its hot-read candidates
            let hot = match (&self.hot, self.shape, layer) {
                (Some(h), Some(sh), Some(_)) if sh.decode && !sh.prompt => Some(h),
                _ => None,
            };
            let nl = self.n_layers;
            // a replay's first layer ran for its rows in their encoder steps: their demand there
            // is already recorded
            let replayed = self.shape.is_some_and(|sh| sh.stage == Stage::Replay) && layer == Some(self.rep_layer());
            let (counts, mass, ms) = match (layer, &self.demand, &self.miss) {
                (Some(l), Some(dm), Some(ms)) if !replayed => (dm.ptr_at(l * e), dm.ptr_at((nl + l) * e), np(ms)),
                _ => (nul(), nul(), nul()),
            };
            // an image span's rows select with bias_vl in every backbone layer (the drafter
            // reads text only)
            let (img, bias_vl) = match (layer, self.shape) {
                (Some(_), Some(sh)) if sh.vision => {
                    let b = w.gate_bias_vl.as_ref().ok_or("image rows, and the checkpoint has no ffn.gate.bias_vl")?;
                    (np(&self.vis_img), np(b))
                }
                _ => (nul(), nul()),
            };
            if counts.is_null() && hot.is_none() {
                ck(unsafe {
                    ds_router(sr, np(&a.z), np(&w.gate_bias), keep, ti, e as i32, k as i32, cfg.route_scale, np(&a.idx), np(&a.rw),
                              img, bias_vl)
                }, "router")?;
            } else {
                ck(unsafe {
                    ds_router_rec(sr, np(&a.z), np(&w.gate_bias), keep, ti, e as i32, k as i32, cfg.route_scale, np(&a.idx),
                                  np(&a.rw), counts, mass, ms, (2 * nl * e) as i64, np(&a.lane), self.max_lanes as i32 + 1,
                                  np(&a.decode_rows), hot.map(|h| np(&h.rows)).unwrap_or(nul()), np(&a.live_pairs), img, bias_vl)
                }, "router (recording)")?;
            }
            if let (Some(h), Some(l)) = (hot, layer) {
                let wd = crate::hot::WIDTH;
                ck(unsafe { ds_hot_candidates(sr, np(&h.rows), (t * k) as i32, wd as i32, h.table.ptr_at(l * 2 * wd)) }, "hot candidates")?;
            }
            // the backbone's decode rows only; the drafter's router is untouched
            if let (Some(_), Some((topp, min_k))) = (layer, self.expert_topp) {
                ck(unsafe { ds_expert_topp(sr, np(&a.idx), np(&a.rw), ti, k as i32, topp, min_k, e as i32, np(&a.decode_rows)) }, "expert top-p")?;
            }
        }
        let set = match (&self.resident, layer, block) {
            (Some(res), Some(l), _) => Some((&res.experts, l)),
            (Some(res), _, Some(b)) => res.dspark_experts.as_ref().map(|x| (x, b)),
            _ => None,
        };
        if let Some((set, bi)) = set {
            // resident: experts -> slots through the device LUT, no host round trip; a backbone
            // step's filler rows route nowhere (draft steps have none)
            let a = &self.act;
            let ar = &set.arena;
            let live = if layer.is_some() { np(&a.live_pairs) } else { nul() };
            ck(unsafe { ds_lut_slots(sr, np(&a.idx), set.lut.ptr_at(bi * (e + 1)), p as i32, np(&a.keys), live) }, "lut slots")?;
            let blocks = route(sr, a, p, k, set.null_slot as usize + 1, e + 1, self.eager_step && t >= PREFILL_MOE_ROWS)?;
            if ar.inter < cfg.moe_inter_dim {
                return self.moe_tp(w, ar, t, k, set.null_slot, blocks);
            }
            self.routed_experts(ar, t, k, set.null_slot, blocks)?;
            return self.shared_and_combine(w, t);
        }
        let mut idx = vec![0i32; p];
        self.act.idx.download_async(0, &mut idx, st)?;
        st.synchronize()?;
        let live = self.live_rows * k;
        if let Some(dir) = &self.dump_dir {
            if let Some(l) = layer {
                let bytes: Vec<u8> = idx[..live].iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("idx{l:02}.i32")), bytes)?;
            }
        }
        let mut experts: Vec<i32> = idx[..live].to_vec();
        experts.sort_unstable();
        experts.dedup();

        let reader = ExpertReader::new(&self.ckpt);
        for group in experts.chunks(ARENA_SLOTS) {
            let tl = Instant::now();
            for (slot, &ex) in group.iter().enumerate() {
                let views = reader.read_expert_into(&mut self.expert_stage, &format!("{}ffn.experts.{ex}.", w.prefix))?;
                for (&(off, len), &(dst, _, want)) in views.iter().zip(self.arena.slot_dsts(slot, 0).iter()) {
                    if len != want {
                        return Err(format!("{}expert {ex}: tensor of {len} bytes, arena expects {want}", w.prefix).into());
                    }
                    let src = unsafe { self.expert_stage.as_ptr().add(off) } as *const c_void;
                    unsafe { memcpy_async(dst, src, len, MEMCPY_HOST_TO_DEVICE, st)? };
                }
                st.synchronize()?; // the staging buffer is reused by the next expert
            }
            self.times.experts_s += tl.elapsed().as_secs_f64();
            self.times.experts_loaded += group.len();
            // pairs outside this group (and filler rows' pairs) route to slot -1, which both
            // kernels skip without writing; each live (k, token) row of `parts` is written by
            // exactly one group
            let keys: Vec<i32> = idx
                .iter()
                .enumerate()
                .map(|(i, ex)| if i >= live { -1 } else { group.iter().position(|g| g == ex).map(|s| s as i32).unwrap_or(-1) })
                .collect();
            let a = &self.act;
            a.keys.upload_async(0, &keys, st)?;
            let ar = &self.arena;
            let blocks = route(sr, a, p, k, ARENA_SLOTS, ARENA_SLOTS, self.eager_step && t >= PREFILL_MOE_ROWS)?;
            self.routed_experts(ar, t, k, -1, blocks)?;
        }
        self.shared_and_combine(w, t)
    }

    /// The routed experts' up + down over the routing blocks in act, with arena `ar`: the
    /// grouped-GEMM kernels for a prefill step's 64-pair blocks; else the bm16 kernels up to
    /// FMA_ROWS rows (one-row bits), the tensor-core ones above, where bm16 goes compute-bound
    /// as pairs concentrate on experts. `null_slot`: pairs routed there read no weights and
    /// write exact zeros (TIDELM_EXPERT_TOPP's dropped picks); -1 for none.
    fn routed_experts(&self, ar: &Arena, t: usize, k: usize, null_slot: i32, b: Blocks) -> Result<()> {
        let cfg = &self.cfg;
        let sr = self.stream.raw();
        let a = &self.act;
        let (d, inter) = (cfg.dim, cfg.moe_inter_dim);
        let (ti, p) = (t as i32, (t * k) as i32);
        if b.prefill {
            ck(unsafe {
                ds_moe_prefill_up(sr, np(&a.y), d as i64, np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), np(&a.hm), inter as i64,
                                  np(&a.rw), np(&a.block_slot), np(&a.block_route), b.nb, inter as i32, d as i32,
                                  cfg.swiglu_limit, null_slot)
            }, "moe up (prefill)")?;
            return ck(unsafe {
                ds_moe_prefill_down(sr, np(&a.hm), inter as i64, np(&ar.w2), np(&ar.s2), np(&a.parts), d as i64,
                                    np(&a.block_slot), np(&a.block_route), b.nb, d as i32, inter as i32, k as i32, ti, null_slot,
                                    0, 1)
            }, "moe down (prefill)");
        }
        if t > FMA_ROWS {
            ck(unsafe {
                ds_moe_up_mma(sr, np(&a.y), d as i64, np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), np(&a.hm), inter as i64,
                              np(&a.rw), np(&a.block_slot), np(&a.block_route), b.nb, inter as i32, d as i32, cfg.swiglu_limit,
                              null_slot, b.bm)
            }, "moe up (mma)")?;
            return ck(unsafe {
                ds_moe_down_mma(sr, np(&a.hm), inter as i64, np(&ar.w2), np(&ar.s2), np(&a.parts), d as i64, np(&a.block_slot),
                                np(&a.block_pair), b.nb, d as i32, inter as i32, k as i32, ti, null_slot, 0, 1, b.bm)
            }, "moe down (mma)");
        }
        ck(unsafe {
            fp4::fp4_moe_cuda_up_v2(sr, np(&a.y), np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), np(&a.hm), np(&a.rw),
                                    np(&a.block_slot), np(&a.block_route), d as i64, inter as i64, cfg.swiglu_limit,
                                    k as i32, inter as i32, d as i32, p, null_slot)
        }, "moe up")?;
        ck(unsafe {
            fp4::fp4_moe_cuda_down_v2(sr, np(&a.hm), np(&ar.w2), np(&ar.s2), np(&a.parts), np(&a.block_slot),
                                      np(&a.block_pair), inter as i64, d as i64, k as i32, d as i32, inter as i32, ti, p,
                                      null_slot, 0, 1)
        }, "moe down")
    }

    /// The shared expert and the one-box combine (model.py::moe's tail).
    fn shared_and_combine(&self, w: &Layer, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        let sr = self.stream.raw();
        let (d, inter, k) = (cfg.dim, cfg.moe_inter_dim, w.topk);
        let ti = t as i32;
        let a = &self.act;
        let part = &a.mma_part;
        gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.sh_w1), t, np(&a.g), false, inter)?;
        gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.sh_w3), t, np(&a.u), false, inter)?;
        ck(unsafe { ds_swiglu(sr, np(&a.g), inter as i64, np(&a.u), inter as i64, np(&a.hs), inter as i64, ti, inter as i32, cfg.swiglu_limit) }, "shared swiglu")?;
        gemm(sr, part, np(&a.hs), inter, Wt::Fp8(&w.sh_w2), t, np(&a.sh), false, d)?;
        ck(unsafe { ds_moe_combine(sr, np(&a.parts), k as i32, ti, np(&a.sh), np(&a.moe_out), d as i32) }, "moe combine")
    }

    /// The tensor-parallel MoE (DSV41_TP_EXPERTS with the output layout): this rank computes
    /// its half of every routed pair's and the shared expert's intermediate rows (written
    /// behind each other: shared rows, then the pairs), one gather makes them whole, then its
    /// half of every output row, and a second gather the layer's output. Each element is the
    /// one-box dot product: no partial sums cross the pair.
    fn moe_tp(&self, w: &Layer, ar: &Arena, t: usize, k: usize, null_slot: i32, b: Blocks) -> Result<()> {
        let cfg = &self.cfg;
        let sr = self.stream.raw();
        let a = &self.act;
        let part = &a.mma_part;
        let (d, inter) = (cfg.dim, cfg.moe_inter_dim);
        let (il, dl) = (ar.inter, ar.dim);
        let (ti, p) = (t as i32, t * k);
        let pi = p as i32;
        let hl = a.hcat.ptr_at(t * il);
        if b.prefill {
            ck(unsafe {
                ds_moe_prefill_up(sr, np(&a.y), d as i64, np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), hl, il as i64, np(&a.rw),
                                  np(&a.block_slot), np(&a.block_route), b.nb, il as i32, d as i32, cfg.swiglu_limit, null_slot)
            }, "moe up (prefill, tp)")?;
        } else if t > FMA_ROWS {
            ck(unsafe {
                ds_moe_up_mma(sr, np(&a.y), d as i64, np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), hl, il as i64, np(&a.rw),
                              np(&a.block_slot), np(&a.block_route), b.nb, il as i32, d as i32, cfg.swiglu_limit, null_slot, b.bm)
            }, "moe up (mma, tp)")?;
        } else {
            ck(unsafe {
                fp4::fp4_moe_cuda_up_v2(sr, np(&a.y), np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), hl, np(&a.rw),
                                        np(&a.block_slot), np(&a.block_route), d as i64, il as i64, cfg.swiglu_limit,
                                        k as i32, il as i32, d as i32, pi, null_slot)
            }, "moe up (tp)")?;
        }
        gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.sh_w1), t, np(&a.g), false, il)?;
        gemm(sr, part, np(&a.y), d, Wt::Fp8(&w.sh_w3), t, np(&a.u), false, il)?;
        ck(unsafe { ds_swiglu(sr, np(&a.g), il as i64, np(&a.u), il as i64, np(&a.hcat), il as i64, ti, il as i32, cfg.swiglu_limit) }, "shared swiglu (tp)")?;
        self.tp_gather(np(&a.hcat), t + p, il, np(&a.hcat_full))?;
        let hf = a.hcat_full.ptr_at(t * inter);
        if b.prefill {
            ck(unsafe {
                ds_moe_prefill_down(sr, hf, inter as i64, np(&ar.w2), np(&ar.s2), np(&a.parts), dl as i64, np(&a.block_slot),
                                    np(&a.block_route), b.nb, dl as i32, inter as i32, k as i32, ti, null_slot, 0, 1)
            }, "moe down (prefill, tp)")?;
        } else if t > FMA_ROWS {
            ck(unsafe {
                ds_moe_down_mma(sr, hf, inter as i64, np(&ar.w2), np(&ar.s2), np(&a.parts), dl as i64, np(&a.block_slot),
                                np(&a.block_pair), b.nb, dl as i32, inter as i32, k as i32, ti, null_slot, 0, 1, b.bm)
            }, "moe down (mma, tp)")?;
        } else {
            ck(unsafe {
                fp4::fp4_moe_cuda_down_v2(sr, hf, np(&ar.w2), np(&ar.s2), np(&a.parts), np(&a.block_slot), np(&a.block_pair),
                                          inter as i64, dl as i64, k as i32, dl as i32, inter as i32, ti, pi, null_slot, 0, 1)
            }, "moe down (tp)")?;
        }
        gemm(sr, part, np(&a.hcat_full), inter, Wt::Fp8(&w.sh_w2), t, np(&a.sh), false, dl)?;
        ck(unsafe { ds_moe_combine(sr, np(&a.parts), k as i32, ti, np(&a.sh), np(&a.moe_local), dl as i32) }, "moe combine (tp)")?;
        self.tp_gather(np(&a.moe_local), t, dl, np(&a.moe_out))
    }

    /// All-gather this rank's `[m][n]` bf16 shard of a tensor-parallel output over RoCE and
    /// lay it out as full rows `[m][world * n]` at `full` (stream-ordered, graph-capturable).
    fn tp_gather(&self, local: *mut c_void, m: usize, n: usize, full: *mut c_void) -> Result<()> {
        let pair = self.pair.as_ref().ok_or("tensor parallel needs the pair")?;
        let world = self.tp.map(|t| t.world).unwrap_or(1);
        // a prefill-sized step (eager, both ranks alike) gathers through NCCL: the one-shot
        // RoCE gather is built for decode's latency and moved its 10-50 MB shards at ~5 GB/s,
        // 15% of an 8K prefill's GPU time (nsys, rank 0); Python sends > 256 KB through NCCL
        let nbytes = m * n * 2;
        if self.eager_step && nbytes > (1 << 20) {
            pair.comm.allgather(local, np(&self.act.tp_scratch), nbytes, &self.stream)?;
        } else {
            pair.roce.gather(local, np(&self.act.tp_scratch), nbytes, &self.stream)?;
        }
        ck(unsafe { ds_unshard_u16(self.stream.raw(), np(&self.act.tp_scratch), full, m as i32, n as i32, world as i32) }, "unshard")
    }
}
