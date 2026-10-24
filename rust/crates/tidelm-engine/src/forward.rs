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
use tidelm_cuda::{memcpy_async, DeviceBuffer, GraphExec, PinnedBuffer, Stream, MEMCPY_HOST_TO_DEVICE};
use tidelm_kernels::ds41::*;
use tidelm_kernels::{engram as eg, fp4};
use tidelm_model::{Checkpoint, ExpertReader, STAGE_BYTES};

use crate::config::Config;
use crate::rope;
use crate::weights::{DsparkW, Fp8, Layer, Loader};
use crate::Result;

/// Rows per step, all lanes together (64 rows x top-6 = 384 routed pairs; the router's
/// wide path takes 512).
pub const TMAX: usize = 64;
/// The FMA GEMM's row limit; steps with more rows take the tensor-core kernel.
pub const FMA_ROWS: usize = 16;
/// Window-ring slots per lane and layer: a step writes its rows before attending, so the
/// ring must hold a step's rows plus the window behind its first row (checked at boot).
const RING: usize = 256;
const ARENA_SLOTS: usize = 16;
const HEAD_ROWS: usize = 8192;
const HC_SPLITS: usize = 16;
/// fp32 split-K partials of the tensor-core GEMM, splits x rows x outputs. The model's
/// largest request at 64 rows is ~0.66M (wo_b, main_proj: 2 x 64 x 5120); a request that
/// does not fit runs with fewer splits.
const MMA_PART: usize = 1 << 21;

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        return Err(format!("{what}: CUDA error {rc}").into());
    }
    Ok(())
}

fn np<T: Copy>(b: &DeviceBuffer<T>) -> *mut c_void {
    b.raw()
}
fn nul() -> *mut c_void {
    std::ptr::null_mut()
}
fn off(p: *mut c_void, bytes: usize) -> *mut c_void {
    if p.is_null() {
        p
    } else {
        unsafe { (p as *mut u8).add(bytes) as *mut c_void }
    }
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
enum Wt<'a> {
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
fn gemm(sr: *mut c_void, part: &DeviceBuffer<f32>, x: *mut c_void, ldx: usize, w: Wt, m: usize, y: *mut c_void,
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
    let esz = if f32_out { 4 } else { 2 };
    for r0 in (0..m).step_by(FMA_ROWS) {
        let mt = FMA_ROWS.min(m - r0) as i32;
        let (xr, ybr, yfr) = (off(x, r0 * ldx * 2), off(yb, r0 * ldy * esz), off(yf, r0 * ldy * esz));
        ck(unsafe {
            match w {
                Wt::Fp8(f) => ds_linear_fp8(sr, xr, ldx as i64, np(&f.w), np(&f.s), f.lds() as i64, mt, n as i32, k as i32, 1,
                                            ybr, yfr, ldy as i64, nul()),
                Wt::Bf16 { w, .. } => ds_linear_bf16(sr, xr, ldx as i64, w, mt, n as i32, k as i32, 1, ybr, yfr, ldy as i64, nul()),
            }
        }, "gemm")?;
    }
    Ok(())
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
    for r0 in (0..m).step_by(FMA_ROWS) {
        let mt = FMA_ROWS.min(m - r0) as i32;
        ck(unsafe {
            ds_linear_fp8_grouped(sr, off(x, r0 * ldx * 2), ldx as i64, gstride as i64, rpg as i32, np(&w.w), np(&w.s),
                                  w.lds() as i64, mt, n as i32, k as i32, off(y, r0 * ldy * 2), ldy as i64)
        }, "grouped gemm")?;
    }
    Ok(())
}

/// Split-K for the tensor-core kernel: at least two blocks per SM (48 SMs on GB10), K slices
/// of at least 512 (examples/tune_mma.rs), bounded by the partials buffer.
fn mma_splits(m: usize, n: usize, k: usize, part_len: usize) -> usize {
    let blocks = n.div_ceil(unsafe { ds_mma_rows_per_block(m as i32) } as usize);
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
}

impl Options {
    pub fn streaming(max_seq: usize, spec: bool) -> Self {
        Options { max_seq, max_lanes: 1, spec, resident: false, graphs: false, n_layers: None, keep: None, dspark_keep: None }
    }
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

/// One lane's rows in a step: tokens `ids` at that lane's next positions.
#[derive(Clone, Copy, Debug)]
pub struct Seg<'a> {
    pub lane: usize,
    pub ids: &'a [u32],
    pub want: Want,
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
}

struct EngramTable {
    ctx: *mut c_void,
}

impl Drop for EngramTable {
    fn drop(&mut self) {
        unsafe { eg::engram_gather_destroy(self.ctx) };
    }
}

const W13: usize = 2304 * 2560;
const S13: usize = 2304 * 160;
const W2: usize = 5120 * 1152;
const S2: usize = 5120 * 72;

/// The routed-expert arena for streaming: ARENA_SLOTS slots in the native FP4 layout
/// (tools/fp4_moe.py::ExpertArena at world 1).
struct Arena {
    w1: DeviceBuffer<u8>,
    s1: DeviceBuffer<u8>,
    w2: DeviceBuffer<u8>,
    s2: DeviceBuffer<u8>,
    w3: DeviceBuffer<u8>,
    s3: DeviceBuffer<u8>,
}

impl Arena {
    fn new(slots: usize) -> Result<Self> {
        Ok(Arena {
            w1: DeviceBuffer::new(slots * W13)?,
            s1: DeviceBuffer::new(slots * S13)?,
            w2: DeviceBuffer::new(slots * W2)?,
            s2: DeviceBuffer::new(slots * S2)?,
            w3: DeviceBuffer::new(slots * W13)?,
            s3: DeviceBuffer::new(slots * S13)?,
        })
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
    cidx1: DeviceBuffer<i32>,
    cidx2: DeviceBuffer<i32>,
    z: DeviceBuffer<f32>,
    idx: DeviceBuffer<i32>,
    rw: DeviceBuffer<f32>,
    keys: DeviceBuffer<i32>,
    block_slot: DeviceBuffer<i32>,
    block_pair: DeviceBuffer<i32>,
    block_route: DeviceBuffer<i32>,
    hm: DeviceBuffer<u16>,
    parts: DeviceBuffer<f32>,
    g: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    hs: DeviceBuffer<u16>,
    sh: DeviceBuffer<u16>,
    moe_out: DeviceBuffer<u16>,
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
    d_idx: DeviceBuffer<i32>,
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
/// device: an arena holding exactly those experts and the [block][expert] -> slot LUT
/// (model.py's slot_lut; -1 for an expert that is not routable).
struct ExpertSet {
    arena: Arena,
    lut: DeviceBuffer<i32>,
}

impl ExpertSet {
    fn load(ckpt: &Checkpoint, stream: &Stream, prefix: &dyn Fn(usize) -> String, blocks: usize, n_experts: usize,
            keep: Option<&[Vec<bool>]>) -> Result<Self> {
        let routable = |b: usize, x: usize| keep.map(|k| k[b][x]).unwrap_or(true);
        let n_slots: usize = (0..blocks).map(|b| (0..n_experts).filter(|&x| routable(b, x)).count()).sum();
        let arena = Arena::new(n_slots.max(1))?;
        let mut lut = vec![-1i32; blocks * n_experts];
        let mut stage = PinnedBuffer::uninit(STAGE_BYTES)?;
        let reader = ExpertReader::new(ckpt);
        let mut slot = 0usize;
        for b in 0..blocks {
            for x in 0..n_experts {
                if !routable(b, x) {
                    continue;
                }
                let views = reader.read_expert_into(&mut stage, &format!("{}ffn.experts.{x}.", prefix(b)))?;
                let dsts: [(*mut c_void, usize); 6] = [
                    (arena.w1.ptr_at(slot * W13), W13), (arena.s1.ptr_at(slot * S13), S13),
                    (arena.w2.ptr_at(slot * W2), W2), (arena.s2.ptr_at(slot * S2), S2),
                    (arena.w3.ptr_at(slot * W13), W13), (arena.s3.ptr_at(slot * S13), S13),
                ];
                for (&(off, len), &(dst, want)) in views.iter().zip(dsts.iter()) {
                    if len != want {
                        return Err(format!("{}expert {x}: tensor of {len} bytes, arena expects {want}", prefix(b)).into());
                    }
                    let src = unsafe { stage.as_ptr().add(off) } as *const c_void;
                    unsafe { memcpy_async(dst, src, len, MEMCPY_HOST_TO_DEVICE, stream)? };
                }
                stream.synchronize()?;
                lut[b * n_experts + x] = slot as i32;
                slot += 1;
            }
        }
        Ok(ExpertSet { arena, lut: DeviceBuffer::from_slice(&lut)? })
    }
}

/// Everything the resident mode keeps on the device besides the per-step activations.
struct Resident {
    /// lent out (taken) for the duration of each block
    layers: Vec<Option<Layer>>,
    embed: DeviceBuffer<u16>,
    head: DeviceBuffer<u16>,
    experts: ExpertSet,
    dspark_experts: Option<ExpertSet>,
    ids: DeviceBuffer<i32>,
}

struct Dspark {
    blocks: Vec<Layer>,
    w: DsparkW,
    noise_row: DeviceBuffer<u16>,
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
    act: Act,
    caches: Caches,
    lanes: Vec<Lane>,
    max_lanes: usize,
    /// steps run so far (Lane::step)
    step_no: u64,
    dspark: Option<Dspark>,
    resident: Option<Resident>,
    /// [layer][expert] routable mask (u8) for the router, both modes; None = all
    keep: Option<DeviceBuffer<u8>>,
    /// the same for the DSpark blocks
    dspark_keep: Option<DeviceBuffer<u8>>,
    /// per number of drafting lanes
    draft_graphs: HashMap<usize, GraphExec>,
    n_layers: usize,
    /// rows of the compressed-rope positions, per ratio: max(pos - (r - 1), 0)
    cpos1: DeviceBuffer<i32>,
    cpos2: DeviceBuffer<i32>,
    /// the one-hot initial pre_mix of every row, copied in on the device each step
    one_hot: DeviceBuffer<f32>,
    /// per engram layer, this step's gathered table rows and their (row, col) map
    eg_raw: Vec<DeviceBuffer<u8>>,
    eg_inv: Vec<DeviceBuffer<i32>>,
    /// the step being issued (select() reads its indexer ranges)
    shape: Option<StepShape>,
    /// live rows of the step being issued (the rest are filler)
    live_rows: usize,
    use_graphs: bool,
    graphs: HashMap<StepShape, GraphExec>,
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
    selection: Selection,
    /// the candidate-source layer's block mask is live for this step's later layers
    cand_live: bool,
    /// index-source layer passes in which the indexer actually ran
    pub indexed_layers: usize,
}

impl Resident {
    /// Every backbone layer's weights, the embedding and head tables, and the routable
    /// experts of the backbone (and of the drafter, with `spec`).
    fn load(ckpt: &Checkpoint, stream: &Stream, cfg: &Config, n_layers: usize, opts: &Options) -> Result<Self> {
        let (d, v) = (cfg.dim, cfg.vocab_size);
        let mut ld = Loader::new(ckpt, stream)?;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(Some(Layer::load(&mut ld, cfg, l)?));
        }
        let embed = ld.bf16("embed.weight", &[v, d])?;
        let head = ld.bf16("head.weight", &[v, d])?;
        let experts = ExpertSet::load(ckpt, stream, &|l| format!("layers.{l}."), n_layers, cfg.n_routed_experts,
                                      opts.keep.as_deref())?;
        let dspark_experts = if opts.spec {
            Some(ExpertSet::load(ckpt, stream, &|k| format!("mtp.{k}."), cfg.n_mtp_layers, cfg.dspark_n_routed_experts,
                                 opts.dspark_keep.as_deref())?)
        } else {
            None
        };
        Ok(Resident { layers, embed, head, experts, dspark_experts, ids: DeviceBuffer::zeroed(TMAX)? })
    }
}

impl Engine {
    /// `opts.spec` loads the DSpark drafter (three blocks + its heads, ~0.8 GB resident).
    pub fn new(model_dir: &std::path::Path, token_map: Vec<i64>, opts: Options) -> Result<Self> {
        let (max_seq, spec) = (opts.max_seq, opts.spec);
        let max_lanes = opts.max_lanes.max(1);
        let cfg = Config::load(model_dir)?;
        if cfg.window_size + TMAX - 1 > RING {
            return Err(format!("window {} + {TMAX} rows per step overflow the {RING}-slot ring", cfg.window_size).into());
        }
        let n_layers = opts.n_layers.unwrap_or(cfg.n_layers).min(cfg.n_layers);
        let ckpt = Checkpoint::open(model_dir)?;
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
        let (norm, dspark) = {
            let mut ld = Loader::new(&ckpt, &stream)?;
            let norm = ld.bf16("norm.weight", &[d])?;
            let dspark = if spec {
                let mut blocks = Vec::new();
                for k in 0..cfg.n_mtp_layers {
                    blocks.push(Layer::load_mtp(&mut ld, &cfg, k)?);
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

        let mut engram_tables = BTreeMap::new();
        for &l in &cfg.engram_layer_ids {
            let wname = format!("layers.{l}.engram.embed.weight");
            let sname = format!("layers.{l}.engram.embed.scale");
            let (wm, sm) = (ckpt.meta(&wname)?, ckpt.meta(&sname)?);
            let shard = ckpt.shard(&wname)?;
            if !std::ptr::eq(shard, ckpt.shard(&sname)?) {
                return Err(format!("layer {l}: engram weight and scale planes live in different shards").into());
            }
            let ctx = unsafe {
                eg::engram_gather_create_direct(shard.file.direct_fd(), wm.lo as i64, sm.lo as i64, wm.shape[0] as i64, 16)
            };
            if ctx.is_null() {
                return Err(format!("layer {l}: engram gather setup failed").into());
            }
            engram_tables.insert(l, EngramTable { ctx });
        }

        let t = TMAX;
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
            partial: f(HC_SPLITS * FMA_ROWS * cfg.hc_mix_width())?,
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
            cidx1: i(t * cfg.index_topk)?,
            cidx2: i(t * cfg.index_topk)?,
            z: f(t * cfg.n_routed_experts)?,
            idx: i(p)?,
            rw: f(p)?,
            keys: i(p)?,
            block_slot: i(p)?,
            block_pair: i(p * 16)?,
            block_route: i(p * 16)?,
            hm: a(p * cfg.moe_inter_dim)?,
            parts: f(p * d)?,
            g: a(t * cfg.moe_inter_dim)?,
            u: a(t * cfg.moe_inter_dim)?,
            hs: a(t * cfg.moe_inter_dim)?,
            sh: a(t * d)?,
            moe_out: a(t * d)?,
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
            head_x: a(t * d)?,
            head_rows: i(t)?,
            head_w: a(HEAD_ROWS * d)?,
            logits: a(t * cfg.vocab_size)?,
            iq: a(t * cfg.index_n_heads * cfg.index_head_dim)?,
            iwb: a(t * cfg.index_n_heads)?,
            iwf: f(t * cfg.index_n_heads)?,
            scores: a(t * (max_seq + 1))?,
            cand: DeviceBuffer::zeroed(t * (max_seq + 1))?,
            bmax: a(t * ((max_seq + 1).div_ceil(8) + 1))?,
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
            d_idx: i((bsz + 1) * draft_lanes)?,
        };
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
            caches.ring.push(a(lanes * RING * hd)?);
        }
        if spec {
            for _ in 0..cfg.n_mtp_layers {
                caches.mtp_ring.push(a(lanes * RING * hd)?);
            }
        }
        // only the compressors this engine runs: a state for a layer past a truncation would
        // never see a chunk, and rollback would read its never-written chunk memory
        for &l in cfg.kv_source_layers.iter().filter(|&&l| l < n_layers) {
            let r = cfg.compress_ratios[l];
            let rows = max_seq / r + 1;
            caches.crows.insert(l, rows);
            caches.ckv.insert(l, a(lanes * rows * hd)?);
            caches.ik.insert(l, a(lanes * rows * cfg.index_head_dim)?);
            if r > 1 {
                caches.r2.insert(l, Ratio2 {
                    pkv: f(lanes * hd)?,
                    psc: f(lanes * hd)?,
                    flag: i(lanes)?,
                    mem_kv: f(lanes * t * hd)?,
                    mem_sc: f(lanes * t * hd)?,
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
        let resident = if opts.resident {
            Some(Resident::load(&ckpt, &stream, &cfg, n_layers, &opts)?)
        } else {
            None
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
        let lanes_v = (0..max_lanes).map(|_| Lane { len: 0, hash: hash.fresh(), last: (0, 0, 0), step: u64::MAX }).collect();
        Ok(Engine {
            arena: Arena::new(ARENA_SLOTS + 1)?,
            expert_stage: PinnedBuffer::uninit(STAGE_BYTES)?,
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
            lanes: lanes_v,
            max_lanes,
            step_no: 0,
            dspark,
            resident,
            keep,
            dspark_keep,
            draft_graphs: HashMap::new(),
            n_layers,
            cpos1: i(t)?,
            cpos2: i(t)?,
            one_hot: DeviceBuffer::from_slice(&(0..t * hc).map(|j| if j % hc == 0 { 1.0f32 } else { 0.0 }).collect::<Vec<_>>())?,
            eg_raw,
            eg_inv,
            shape: None,
            live_rows: 0,
            use_graphs: opts.graphs && opts.resident,
            graphs: HashMap::new(),
            graph_replays: 0,
            graph_captures: 0,
            max_seq,
            times: StepTimes::default(),
            dump_dir: std::env::var_os("TIDELM_DUMP_DIR").map(std::path::PathBuf::from),
            no_qdq: std::env::var("TIDELM_NO_QDQ").as_deref() == Ok("1"),
            selection: Selection::AllVisible,
            cand_live: false,
            indexed_layers: 0,
        })
    }

    pub fn max_lanes(&self) -> usize {
        self.max_lanes
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
        if lane >= self.max_lanes {
            return Err(format!("lane {lane} of {}", self.max_lanes).into());
        }
        self.lanes[lane] = Lane { len: 0, hash: self.hash.fresh(), last: (0, 0, 0), step: u64::MAX };
        for st in self.caches.r2.values() {
            st.flag.upload_async(lane, &[0], &self.stream)?;
        }
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
        let mut seen = vec![false; self.max_lanes];
        for &(lane, n) in reqs {
            if lane >= self.max_lanes || std::mem::replace(&mut seen[lane], true) {
                return Err(format!("rollback: lane {lane} out of range or repeated").into());
            }
            let l = &self.lanes[lane];
            if n > l.len || n < l.last.0 {
                return Err(format!("rollback({n}) of lane {lane}: outside its last segment {}..={}", l.last.0, l.len).into());
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

    /// Lane 0: tokens `ids` at its next positions; returns the bf16 logits `want` asks for.
    pub fn forward(&mut self, ids: &[u32], want: Want) -> Result<Vec<u16>> {
        Ok(self.forward_batch(&[Seg { lane: 0, ids, want }])?.remove(0))
    }

    /// One backbone step over every segment's rows (at most TMAX together, each lane at most
    /// once, rows of a lane in position order): returns per segment the bf16 logits its
    /// `want` asks for (the engine's head rounds them).
    pub fn forward_batch(&mut self, segs: &[Seg]) -> Result<Vec<Vec<u16>>> {
        let plan = self.prepare(segs)?;
        let shape = plan.shape;
        if self.use_graphs {
            if !self.graphs.contains_key(&shape) {
                // capture records the step without running it; the launch below runs it once
                self.stream.begin_capture()?;
                let rec = self.device_step(shape);
                let graph = self.stream.end_capture();
                rec?;
                self.graphs.insert(shape, graph?);
                self.graph_captures += 1;
            } else {
                self.graph_replays += 1;
            }
            self.graphs[&shape].launch(&self.stream)?;
        } else {
            self.device_step(shape)?;
        }
        self.collect(segs, &plan)
    }

    /// Host work before a step's device work: the rows' positions and lanes, the segments,
    /// the embedding's host half, the n-gram hashes and their engram rows (an NVMe gather),
    /// and the step's shape.
    fn prepare(&mut self, segs: &[Seg]) -> Result<Plan> {
        if segs.is_empty() {
            return Err("a step needs at least one segment".into());
        }
        let mut seen = vec![false; self.max_lanes];
        let mut rows = 0;
        for sg in segs {
            if sg.lane >= self.max_lanes || std::mem::replace(&mut seen[sg.lane], true) {
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
        if rows > TMAX {
            return Err(format!("a step takes at most {TMAX} rows, got {rows}").into());
        }
        let cfg = self.cfg.clone();
        let tw = Instant::now();
        let tb = row_bucket(rows);
        let scratch = self.max_lanes as i32;
        let (mut pos, mut lane, mut ids) = (vec![0i32; tb], vec![scratch; tb], vec![0u32; tb]);
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
        }
        self.live_rows = rows;
        self.embed_host(&ids, rows)?;
        let n_eng = cfg.engram_layer_ids.len();
        let cols = self.hash.layout.n_hash_cols();
        // [row][n_engram_layers][cols], live rows only
        let mut hashes = Vec::with_capacity(rows * n_eng * cols);
        for sg in segs {
            let l = &mut self.lanes[sg.lane];
            hashes.extend(l.hash.hash(sg.ids, l.len, None));
        }
        self.times.compute_s += tw.elapsed().as_secs_f64();
        let te = Instant::now();
        let st = &self.stream;
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
            let mut raw = vec![0u8; uniq.len() * 264];
            let rc = unsafe { eg::engram_gather_run(self.engram_tables[&l].ctx, uniq.as_ptr(), uniq.len(), raw.as_mut_ptr()) };
            if rc != 0 {
                return Err(format!("layer {l}: engram gather failed ({rc})").into());
            }
            self.eg_raw[li].upload_async(0, &raw, st)?;
            self.eg_inv[li].upload_async(0, &inv, st)?;
        }
        self.times.engram_s += te.elapsed().as_secs_f64();

        let topk = cfg.index_topk;
        let indexed = |nc: usize, r: usize| -> Option<usize> { (nc > topk).then(|| bucket(nc, 2048, self.max_seq / r + 1)) };
        let cand_src = cfg.candidate_source_layer;
        let cand = cand_src >= 0 && (cand_src as usize) < self.n_layers && {
            let r = cfg.compress_ratios[cand_src as usize].max(1);
            (if r == 1 { nc1 } else { nc2 }) > 2048 * 8
        };
        let shape = StepShape {
            t: tb,
            c1: bucket(nc1.min(topk), 16, topk),
            c2: bucket(nc2.min(topk), 16, topk),
            n1: indexed(nc1, 1),
            n2: indexed(nc2, 2),
            cand,
            head: hb,
        };
        Ok(Plan { shape, firsts, heads })
    }

    /// Every device operation of one backbone step, in stream order, with no host decision
    /// that depends on buffer contents: the body the step graph captures.
    fn device_step(&mut self, shape: StepShape) -> Result<()> {
        let cfg = self.cfg.clone();
        let (d, hc, t) = (cfg.dim, cfg.hc_mult, shape.t);
        let capturing = self.use_graphs;
        self.shape = Some(shape);
        self.embed_dev(t)?;
        {
            let sr = self.stream.raw();
            let a = &self.act;
            // the all-visible compressed lists, built from the positions; only read while
            // n_c <= index_topk (past that the indexer's top-k replaces them)
            ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 1, shape.c1 as i32, np(&a.cidx1), np(&a.clen1)) }, "visible r1")?;
            ck(unsafe { ds_visible(sr, np(&a.pos), t as i32, 2, shape.c2 as i32, np(&a.cidx2), np(&a.clen2)) }, "visible r2")?;
        }
        self.selection = Selection::AllVisible;
        self.cand_live = false;
        let cols = self.hash.layout.n_hash_cols();
        for l in 0..self.n_layers {
            let t0 = Instant::now();
            // resident: borrow the layer out of the resident set for this block (put back below)
            let w = match self.resident.as_mut() {
                Some(res) => res.layers[l].take().expect("resident layer present"),
                None => {
                    let mut ld = Loader::new(&self.ckpt, &self.stream)?;
                    Layer::load(&mut ld, &cfg, l)?
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
                    gemm(sr, &a.mma_part, np(&a.erows), k, Wt::Fp8(&e.wkv), t, np(&a.ekv), false, e.wkv.n)?;
                    ck(unsafe {
                        ds_engram_gate(sr, np(&a.h), np(&a.ekv), np(&e.weight), t as i32, d as i32, hc as i32, cfg.norm_eps,
                                       (d as f64).powf(-0.5) as f32)
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
        if shape.head > 0 {
            let norm = np(&self.norm);
            self.head_dev(t, Some(shape.head), norm)?;
        }
        Ok(())
    }

    /// After the step's device work: read back the logits it computed, advance the lanes.
    fn collect(&mut self, segs: &[Seg], plan: &Plan) -> Result<Vec<Vec<u16>>> {
        let st = &self.stream;
        let v = self.cfg.vocab_size;
        let nw: usize = plan.heads.iter().map(|h| h.1).sum();
        let mut logits = vec![0u16; nw * v];
        if nw > 0 {
            self.act.logits.download_async(0, &mut logits, st)?;
        }
        st.synchronize()?;
        self.step_no += 1;
        for (i, sg) in segs.iter().enumerate() {
            let l = &mut self.lanes[sg.lane];
            l.last = (l.len, plan.firsts[i], sg.ids.len());
            l.step = self.step_no;
            l.len += sg.ids.len();
        }
        if let (Some(dir), true, 1) = (&self.dump_dir, nw > 0, segs.len()) {
            dump_bf16(&dir.join("logits.f32"), &logits[(nw - 1) * v..])?;
        }
        Ok(plan.heads.iter().map(|&(h0, n)| logits[h0 * v..(h0 + n) * v].to_vec()).collect())
    }

    /// hc_pre with the final pre_mix + `norm` over the step's `t` rows, then the head over
    /// the rows act.head_rows names (`gather`: how many) or over all `t`, into act.logits
    /// `[rows][V]` bf16 (streamed from NVMe unless resident).
    fn head_dev(&mut self, t: usize, gather: Option<usize>, norm: *mut c_void) -> Result<()> {
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
            gemm(sr, &a.mma_part, x, d, Wt::Bf16 { w: np(&res.head), n: v, k: d }, rows, np(&a.logits), false, v)?;
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
                ds_scatter_ring(sr, np(&a.kv), np(&self.caches.mtp_ring[k]), RING as i32, np(&a.pos), mi, hd as i32, np(&a.lane),
                                (RING * hd) as i64)
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
        if self.dspark.is_none() {
            return Err("draft needs the drafter (spec)".into());
        }
        let n = reqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
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
        }
        self.live_rows = n * b;
        self.embed_host(&ids, n * b)?;
        if self.use_graphs {
            if !self.draft_graphs.contains_key(&n) {
                self.stream.begin_capture()?;
                let rec = self.draft_dev(n);
                let graph = self.stream.end_capture();
                rec?;
                self.draft_graphs.insert(n, graph?);
                self.graph_captures += 1;
            } else {
                self.graph_replays += 1;
            }
            self.draft_graphs[&n].launch(&self.stream)?;
        } else {
            self.draft_dev(n)?;
        }
        let mut out = vec![0i32; (b + 1) * n];
        self.act.d_idx.download_async(0, &mut out, &self.stream)?;
        self.stream.synchronize()?;
        Ok((0..n).map(|j| (0..b).map(|i| out[(i + 1) * n + j] as u32).collect()).collect())
    }

    /// The draft's device work for `n` lanes: embedding, the three blocks, the head, and the
    /// markov chain entirely on the device (each step's embedding rows are gathered by the
    /// indices the previous argmax wrote, so the chain needs no host sync).
    fn draft_dev(&mut self, n: usize) -> Result<()> {
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
        self.head_dev(rows, None, norm)?;
        let ds = self.dspark.as_ref().unwrap();
        let (v, r) = (cfg.vocab_size, cfg.dspark_markov_rank);
        let sr = self.stream.raw();
        let a = &self.act;
        let ni = n as i32;
        // logits row j * b + i is lane j's draft step i
        for i in 0..b {
            ck(unsafe { ds_gather_rows_u16(sr, np(&ds.w.markov_embed), a.d_idx.ptr_at(i * n), np(&a.markov_e), ni, r as i32) }, "markov embed")?;
            gemm(sr, &a.mma_part, np(&a.markov_e), r, Wt::Bf16 { w: np(&ds.w.markov_head), n: v, k: r }, n, np(&a.markov_b), false, v)?;
            ck(unsafe {
                ds_argmax_sum_bf16(sr, a.logits.ptr_at(i * v), (b * v) as i64, np(&a.markov_b), v as i64, ni, v as i32,
                                   a.d_idx.ptr_at((i + 1) * n))
            }, "markov argmax")?;
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
        // fp32 weights have no tensor-core path: 16-row slices of the FMA kernel (the 2 MB
        // weight stays in L2 across slices)
        for r0 in (0..t).step_by(FMA_ROWS) {
            let mt = FMA_ROWS.min(t - r0) as i32;
            ck(unsafe {
                ds_linear_f32w(sr, a.h.ptr_at(r0 * k), k as i64, np(fn_w), mt, n as i32, k as i32, HC_SPLITS as i32, nul(),
                               n as i64, np(&a.partial))
            }, "hc mix")?;
            ck(unsafe { ds_splits_reduce(sr, np(&a.partial), HC_SPLITS as i32, mt, n as i32, nul(), a.mixes.ptr_at(r0 * n), n as i64) }, "hc reduce")?;
        }
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
        let (d, hd, rd, h) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim, cfg.n_heads);
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
                    ds_scatter_ring(sr, np(&a.kv), np(&self.caches.ring[l]), RING as i32, np(&a.pos), ti, hd as i32, np(&a.lane),
                                    (RING * hd) as i64)
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
                    (ring, &a.pos, np(&self.caches.ckv[&src]), np(cidx), c, self.caches.crows[&src] * hd)
                } else {
                    (ring, &a.pos, nul(), nul(), 0, 0)
                }
            }
            // model.py::attention with mtp_extra: the window is the main stream's last 128
            // positions in this block's seeded ring, for every draft query alike, plus every
            // draft key of the lane's block (no causal mask among them)
            Ctx::Draft { k } => (&self.caches.mtp_ring[k], &a.dwin, np(&self.caches.dkv), np(&a.dcidx), bsz, bsz * hd),
        };
        ck(unsafe {
            ds_attn_decode(sr, np(&a.q), np(ring), RING as i32, np(win_pos), cfg.window_size as i32, 0, ckv_ptr, cidx_ptr,
                           c as i32, np(&w.attn_sink), (hd as f64).powf(-0.5) as f32, np(&a.o), ti, h as i32, np(&a.lane),
                           (RING * hd) as i64, ckv_ls as i64)
        }, "attention")?;
        ck(unsafe { ds_rope(sr, np(&a.o), (h * hd) as i64, ti, h as i32, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 1) }, "o rope")?;
        let og = cfg.o_groups;
        let gk = h * hd / og;
        gemm_grouped(sr, &a.mma_part, np(&a.o), h * hd, gk, cfg.o_lora_rank, &w.wo_a, t, np(&a.oa), w.wo_a.n)?;
        gemm(sr, &a.mma_part, np(&a.oa), og * cfg.o_lora_rank, Wt::Fp8(&w.wo_b), t, np(&a.attn_out), false, d)
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
        ck(unsafe {
            ds_index_scores(sr, np(&a.iq), np(&self.caches.ik[&src]), np(&a.iwf), np(clen),
                            if use_cand { np(&a.cand) } else { nul() }, nmax as i64, n_c as i32, np(&a.scores),
                            n_c as i64, ti, np(&a.lane), (self.caches.crows[&src] * ihd) as i64)
        }, "index scores")?;
        let block = 8usize;
        let topk_blocks = 2048usize;
        if l as i64 == cand_src && shape.cand {
            // model.py::_select_candidates: below topk_blocks blocks every block is kept, so
            // the mask only exists where it can drop something
            let nb = n_c.div_ceil(block);
            let kb = topk_blocks.min(nb);
            ck(unsafe { ds_candidate_blocks(sr, np(&a.scores), n_c as i64, n_c as i32, block as i32, np(clen), np(&a.bmax), nb as i64, ti) }, "candidate blocks")?;
            ck(unsafe { ds_topk_select(sr, np(&a.bmax), nb as i64, nb as i32, kb as i32, np(&a.blocks), kb as i32, ti) }, "candidate select")?;
            a.cand.zero_async(st)?;
            ck(unsafe { ds_candidate_set(sr, np(&a.blocks), kb as i32, block as i32, n_c as i32, np(&a.cand), nmax as i64, ti) }, "candidate set")?;
            self.cand_live = true;
        }
        let k = cfg.index_topk.min(n_c);
        ck(unsafe { ds_topk_select(sr, np(&a.scores), n_c as i64, n_c as i32, k as i32, np(&a.topk), cfg.index_topk as i32, ti) }, "index topk")?;
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
        if !self.no_qdq {
            ck(unsafe { ds_fp4_qdq_e4m3(sr, np(&a.latn), (t * hd) as i64) }, "comp qdq")?;
        }
        let ckv = &self.caches.ckv[&l];
        ck(unsafe {
            ds_scatter_compressed(sr, np(&a.latn), hd as i64, np(ckv), np(&a.pos), r as i32, ti, hd as i32, np(&a.lane),
                                  (rows * hd) as i64)
        }, "ckv scatter")?;
        if let Some(dir) = &self.dump_dir {
            // lane 0's part (single-lane steps)
            let n = (self.lanes[0].len + self.live_rows) / r;
            let mut v = vec![0u16; n * hd];
            ckv.download_async(0, &mut v, st)?;
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
        let (d, inter, e, k) = (cfg.dim, cfg.moe_inter_dim, w.n_experts, w.topk);
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
            ck(unsafe { ds_router(sr, np(&a.z), np(&w.gate_bias), keep, ti, e as i32, k as i32, cfg.route_scale, np(&a.idx), np(&a.rw)) }, "router")?;
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
            ck(unsafe { ds_lut_slots(sr, np(&a.idx), set.lut.ptr_at(bi * e), p as i32, np(&a.keys), live) }, "lut slots")?;
            ck(unsafe { fp4::fp4_moe_cuda_route_small(sr, np(&a.keys), np(&a.block_slot), np(&a.block_pair), np(&a.block_route), p as i32, k as i32) }, "route")?;
            ck(unsafe {
                fp4::fp4_moe_cuda_up_v2(sr, np(&a.y), np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), np(&a.hm), np(&a.rw),
                                        np(&a.block_slot), np(&a.block_route), d as i64, inter as i64, cfg.swiglu_limit,
                                        k as i32, inter as i32, d as i32, p as i32, -1)
            }, "moe up")?;
            ck(unsafe {
                fp4::fp4_moe_cuda_down_v2(sr, np(&a.hm), np(&ar.w2), np(&ar.s2), np(&a.parts), np(&a.block_slot),
                                          np(&a.block_pair), inter as i64, d as i64, k as i32, d as i32, inter as i32,
                                          ti, p as i32, -1, 0, 1)
            }, "moe down")?;
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
                let ar = &self.arena;
                let dsts: [(*mut c_void, usize); 6] = [
                    (ar.w1.ptr_at(slot * W13), W13), (ar.s1.ptr_at(slot * S13), S13),
                    (ar.w2.ptr_at(slot * W2), W2), (ar.s2.ptr_at(slot * S2), S2),
                    (ar.w3.ptr_at(slot * W13), W13), (ar.s3.ptr_at(slot * S13), S13),
                ];
                for (&(off, len), &(dst, want)) in views.iter().zip(dsts.iter()) {
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
            ck(unsafe { fp4::fp4_moe_cuda_route_small(sr, np(&a.keys), np(&a.block_slot), np(&a.block_pair), np(&a.block_route), p as i32, k as i32) }, "route")?;
            ck(unsafe {
                fp4::fp4_moe_cuda_up_v2(sr, np(&a.y), np(&ar.w1), np(&ar.s1), np(&ar.w3), np(&ar.s3), np(&a.hm), np(&a.rw),
                                        np(&a.block_slot), np(&a.block_route), d as i64, inter as i64, cfg.swiglu_limit,
                                        k as i32, inter as i32, d as i32, p as i32, -1)
            }, "moe up")?;
            ck(unsafe {
                fp4::fp4_moe_cuda_down_v2(sr, np(&a.hm), np(&ar.w2), np(&ar.s2), np(&a.parts), np(&a.block_slot),
                                          np(&a.block_pair), inter as i64, d as i64, k as i32, d as i32, inter as i32,
                                          ti, p as i32, -1, 0, 1)
            }, "moe down")?;
        }
        self.shared_and_combine(w, t)
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
}
