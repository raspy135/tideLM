//! The DS v4.1 forward pass (engine/model.py::Model.forward + block + attention + moe,
//! tools/v41_ref.py), the DSpark drafter (Model.dspark_seed / dspark_draft) and cache
//! rollback (Caches.rollback), on the ds41.cu kernels and the native FP4 MoE kernels.
//!
//! This engine runs in *streaming* mode: each step reads every layer's weights from NVMe,
//! routes, loads only the experts the step's tokens chose (through a 16-slot arena), and
//! frees the layer again. It needs ~2 GB of device memory, so it runs next to a live
//! server; it is the correctness vehicle for the resident engine, which keeps this step
//! code and swaps the weight source.
//!
//! Semantics follow the checkpoint's reference forward (inference/model.py), which runs
//! every layer over every token. The Python engine's decoder replay (layers past the
//! candidate source over only the last 128 prompt tokens) is an approximation from the
//! tech report; it agrees exactly for prompts up to 128 tokens and is not used here.
//!
//! - at most `TMAX` = 16 rows per call (the decode GEMM and MoE router bound; prefill
//!   loops over 16-token chunks);
//! - the indexer runs only where it can change something: an index-source layer whose
//!   compressed count exceeds index_topk (1024). Below that, model.py's top-k selects every
//!   visible position, so the all-visible selection is the same set at no cost.
//! - every kernel is row-invariant, so a verify block of T rows computes each row exactly
//!   as T one-row steps would: speculation cannot change greedy output.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::time::Instant;

use tidelm_core::engram_hash::{EngramLayout, NgramHashState};
use tidelm_cuda::{memcpy_async, DeviceBuffer, PinnedBuffer, Stream, MEMCPY_HOST_TO_DEVICE};
use tidelm_kernels::ds41::*;
use tidelm_kernels::{engram as eg, fp4};
use tidelm_model::{Checkpoint, ExpertReader, STAGE_BYTES};

use crate::config::Config;
use crate::rope;
use crate::weights::{DsparkW, Fp8, Layer, Loader};
use crate::Result;

pub const TMAX: usize = 16;
const RING: usize = 512;
const ARENA_SLOTS: usize = 16;
const HEAD_ROWS: usize = 8192;
const HC_SPLITS: usize = 16;

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

fn dump_bf16(path: &std::path::Path, v: &[u16]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|&h| f32::from_bits((h as u32) << 16).to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

/// How the engine is built. `n_layers` and `keep` exist for tests that compare the two
/// weight sources on a truncated model; serving leaves them unset.
#[derive(Clone, Debug)]
pub struct Options {
    pub max_seq: usize,
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
        Options { max_seq, spec, resident: false, graphs: false, n_layers: None, keep: None, dspark_keep: None }
    }
}

/// Which logits a forward call returns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Want {
    Nothing,
    /// the last row's `[V]`
    Last,
    /// every row's, `[rows][V]` flattened (speculative verification)
    All,
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
    y: DeviceBuffer<u16>,
    qa: DeviceBuffer<u16>,
    qr: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    kv0: DeviceBuffer<u16>,
    kv: DeviceBuffer<u16>,
    o: DeviceBuffer<u16>,
    oa: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<u16>,
    pos: DeviceBuffer<i32>,
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
    head_x: DeviceBuffer<u16>,
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
    // DSpark: the target layers' hc means of the last forward, the projected main hidden,
    // and the draft block's own keys / positions
    main_h: DeviceBuffer<u16>,
    main_x: DeviceBuffer<u16>,
    dkv: DeviceBuffer<u16>,
    dwin: DeviceBuffer<i32>,
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
/// the CUDA-graph key. Widths that grow with the context are bucketed to powers of two so a
/// graph serves many steps; the padding is masked by the per-row visible counts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct StepShape {
    t: usize,
    /// all-visible list widths per ratio (bucketed, <= index_topk)
    c1: usize,
    c2: usize,
    /// the indexer's key range per ratio when it runs (bucketed), None while n_c <= index_topk
    n1: Option<usize>,
    n2: Option<usize>,
    /// layer 20's candidate-block mask exists (n_c > 16,384)
    cand: bool,
    /// (first row, rows) of the head; rows == 0 for none
    head: (usize, usize),
}

fn bucket(n: usize, lo: usize, hi: usize) -> usize {
    n.max(1).next_power_of_two().max(lo).min(hi)
}

/// How a block's attention sees keys: the backbone's (window ring + compressed cache) or a
/// DSpark block's (its seeded ring up to the last main position + the draft block's keys).
#[derive(Clone, Copy)]
enum Ctx {
    Main { l: usize, s: usize, c1: usize, c2: usize },
    #[allow(dead_code)]
    Draft { k: usize, s: usize, win_end: usize },
}

/// A ratio-2 compressor's state across positions, all on the device so the step can be a
/// graph: the pending (unpaired) position and what rollback needs to restore it
/// (model.py::Caches.pending / _chunk_inputs).
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

struct Caches {
    ring: Vec<DeviceBuffer<u16>>,
    ckv: BTreeMap<usize, DeviceBuffer<u16>>,
    ik: BTreeMap<usize, DeviceBuffer<u16>>,
    r2: BTreeMap<usize, Ratio2>,
    mtp_ring: Vec<DeviceBuffer<u16>>,
    /// rollback's target length, read by the device
    n_dev: DeviceBuffer<i32>,
    len: usize,
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
    dspark: Option<Dspark>,
    resident: Option<Resident>,
    /// [layer][expert] routable mask (u8) for the router, both modes; None = all
    keep: Option<DeviceBuffer<u8>>,
    /// the same for the DSpark blocks
    dspark_keep: Option<DeviceBuffer<u8>>,
    draft_graph: Option<tidelm_cuda::GraphExec>,
    n_layers: usize,
    /// rows of the compressed-rope positions, per ratio: max(pos - (r - 1), 0)
    cpos1: DeviceBuffer<i32>,
    cpos2: DeviceBuffer<i32>,
    /// the one-hot initial pre_mix of every row, copied in on the device each step
    one_hot: DeviceBuffer<f32>,
    /// per engram layer, this step's gathered table rows and their (token, col) map
    eg_raw: Vec<DeviceBuffer<u8>>,
    eg_inv: Vec<DeviceBuffer<i32>>,
    /// the step being issued (select() reads its indexer ranges)
    shape: Option<StepShape>,
    use_graphs: bool,
    graphs: std::collections::HashMap<StepShape, tidelm_cuda::GraphExec>,
    /// steps served by a graph replay / by a capture
    pub graph_replays: usize,
    pub graph_captures: usize,
    /// (start position, rows) of the last forward: what `seed_draft` reads `main_h` for
    last_forward: (usize, usize),
    pub max_seq: usize,
    pub times: StepTimes,
    /// TIDELM_DUMP_DIR: write the hc stream after every layer and the last logits as raw
    /// little-endian f32 (the layout rust/tools/ref_forward.py writes), for parity checks.
    pub dump_dir: Option<std::path::PathBuf>,
    no_qdq: bool,
    selection: Selection,
    /// the candidate-source layer's block mask is live for this forward's later layers
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
        let cfg = Config::load(model_dir)?;
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
        let p = t * cfg.n_activated_experts.max(cfg.dspark_n_activated_experts);
        let hdq = cfg.n_heads * cfg.head_dim;
        let eng_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
        let n_target = cfg.dspark_target_layer_ids.len();
        let bsz = cfg.dspark_block_size;
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
            partial: f(HC_SPLITS * t * cfg.hc_mix_width())?,
            y: a(t * d)?,
            qa: a(t * cfg.q_lora_rank)?,
            qr: a(t * cfg.q_lora_rank)?,
            q: a(t * hdq)?,
            kv0: a(t * cfg.head_dim)?,
            kv: a(t * cfg.head_dim)?,
            o: a(t * hdq)?,
            oa: a(t * cfg.o_groups * cfg.o_lora_rank)?,
            attn_out: a(t * d)?,
            pos: i(t)?,
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
            kvl: f(t * cfg.head_dim)?,
            sc: f(t * cfg.head_dim)?,
            lat: a(t * cfg.head_dim)?,
            latn: a(t * cfg.head_dim)?,
            kidx: a(t * cfg.index_head_dim)?,
            kidxn: a(t * cfg.index_head_dim)?,
            erows: a(t * eng_cols * cfg.engram_head_dim)?,
            ekv: a(t * (hc + 1) * d)?,
            emb: a(t * d)?,
            row_ids: DeviceBuffer::from_slice(&(0..t as i32).collect::<Vec<_>>())?,
            head_x: a(t * d)?,
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
            main_x: a(t * d)?,
            dkv: a(bsz * cfg.head_dim)?,
            dwin: i(bsz)?,
            dcidx: DeviceBuffer::from_slice(&(0..bsz * bsz).map(|j| (j % bsz) as i32).collect::<Vec<_>>())?,
            markov_e: a(cfg.dspark_markov_rank)?,
            markov_b: a(cfg.vocab_size)?,
            d_idx: i(bsz + 1)?,
        };
        let mut caches = Caches {
            ring: Vec::new(),
            ckv: BTreeMap::new(),
            ik: BTreeMap::new(),
            r2: BTreeMap::new(),
            mtp_ring: Vec::new(),
            n_dev: i(1)?,
            len: 0,
        };
        for _ in 0..cfg.n_layers {
            caches.ring.push(a(RING * cfg.head_dim)?);
        }
        if spec {
            for _ in 0..cfg.n_mtp_layers {
                caches.mtp_ring.push(a(RING * cfg.head_dim)?);
            }
        }
        // only the compressors this engine runs: a state for a layer past a truncation would
        // never see a chunk, and rollback would read its never-written chunk memory
        for &l in cfg.kv_source_layers.iter().filter(|&&l| l < n_layers) {
            let r = cfg.compress_ratios[l];
            caches.ckv.insert(l, a((max_seq / r + 1) * cfg.head_dim)?);
            caches.ik.insert(l, a((max_seq / r + 1) * cfg.index_head_dim)?);
            if r > 1 {
                caches.r2.insert(l, Ratio2 {
                    pkv: f(cfg.head_dim)?,
                    psc: f(cfg.head_dim)?,
                    flag: i(1)?,
                    mem_kv: f(t * cfg.head_dim)?,
                    mem_sc: f(t * cfg.head_dim)?,
                    before_kv: f(cfg.head_dim)?,
                    before_sc: f(cfg.head_dim)?,
                    before_flag: i(1)?,
                    mem_start: i(1)?,
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
            dspark,
            resident,
            keep,
            dspark_keep,
            draft_graph: None,
            n_layers,
            cpos1: i(t)?,
            cpos2: i(t)?,
            one_hot: DeviceBuffer::from_slice(&(0..t * hc).map(|j| if j % hc == 0 { 1.0f32 } else { 0.0 }).collect::<Vec<_>>())?,
            eg_raw,
            eg_inv,
            shape: None,
            use_graphs: opts.graphs && opts.resident,
            graphs: std::collections::HashMap::new(),
            graph_replays: 0,
            graph_captures: 0,
            last_forward: (0, 0),
            max_seq,
            times: StepTimes::default(),
            dump_dir: std::env::var_os("TIDELM_DUMP_DIR").map(std::path::PathBuf::from),
            no_qdq: std::env::var("TIDELM_NO_QDQ").as_deref() == Ok("1"),
            selection: Selection::AllVisible,
            cand_live: false,
            indexed_layers: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.caches.len
    }

    /// Switch graph capture/replay on or off (resident only).
    pub fn set_graphs(&mut self, on: bool) {
        self.use_graphs = on && self.resident.is_some();
    }

    pub fn is_empty(&self) -> bool {
        self.caches.len == 0
    }

    /// Start a new sequence. Nothing needs clearing: every cache entry a position reads is
    /// written by this sequence before it is read (rings and compressed rows by position,
    /// the hash cache from position 0), so only the lengths and the pending flags reset.
    pub fn reset(&mut self) {
        self.caches.len = 0;
        for st in self.caches.r2.values() {
            let _ = st.flag.zero_async(&self.stream);
        }
        let _ = self.stream.synchronize();
        self.last_forward = (0, 0);
    }

    /// Discard positions `>= n` (model.py::Caches.rollback). Only the ratio-2 compressors
    /// carry state across positions; rings and compressed rows are append-only and every
    /// slot at or after n is rewritten before anything reads it.
    pub fn rollback(&mut self, n: usize) -> Result<()> {
        if n > self.caches.len {
            return Err(format!("rollback({n}) past the cached length {}", self.caches.len).into());
        }
        let st = &self.stream;
        self.caches.n_dev.upload_async(0, &[n as i32], st)?;
        self.rollback_dev()?;
        self.caches.len = n;
        Ok(())
    }

    /// The device half of rollback: restore every ratio-2 compressor's pending position for
    /// the length in `caches.n_dev` (the step graph writes it from the accepted count).
    fn rollback_dev(&self) -> Result<()> {
        let sr = self.stream.raw();
        let hd = self.cfg.head_dim as i32;
        for st in self.caches.r2.values() {
            ck(unsafe {
                ds_rollback_pending(sr, np(&self.caches.n_dev), np(&st.mem_kv), np(&st.mem_sc), np(&st.mem_start),
                                    np(&st.before_kv), np(&st.before_sc), np(&st.before_flag), np(&st.pkv), np(&st.psc),
                                    np(&st.flag), hd, TMAX as i32)
            }, "rollback pending")?;
        }
        Ok(())
    }

    /// The host half of the embedding: resident uploads the ids, streaming reads the rows.
    fn embed_host(&mut self, ids: &[u32]) -> Result<()> {
        let d = self.cfg.dim;
        if let Some(res) = &self.resident {
            let dev_ids: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
            res.ids.upload_async(0, &dev_ids, &self.stream)?;
        } else {
            let noise = self.cfg.dspark_noise_token_id;
            let mut ld = Loader::new(&self.ckpt, &self.stream)?;
            for (ti, &id) in ids.iter().enumerate() {
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

    /// Run tokens `ids` at positions `len..len + ids.len()` (at most TMAX) through the
    /// backbone; returns the bf16 logits `want` asks for (the engine's head rounds them).
    pub fn forward(&mut self, ids: &[u32], want: Want) -> Result<Vec<u16>> {
        let shape = self.prepare(ids, want)?;
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
        self.collect(shape)
    }

    /// Host work before a step's device work: positions, the embedding's host half, the
    /// n-gram hashes and their engram rows (an NVMe gather), and the step's shape.
    fn prepare(&mut self, ids: &[u32], want: Want) -> Result<StepShape> {
        let t = ids.len();
        let s = self.caches.len;
        if t == 0 || t > TMAX {
            return Err(format!("forward takes 1..={TMAX} tokens, got {t}").into());
        }
        if s + t > self.max_seq {
            return Err(format!("context {} exceeds max_seq {}", s + t, self.max_seq).into());
        }
        let cfg = self.cfg.clone();
        let tw = Instant::now();
        {
            let st = &self.stream;
            let pos: Vec<i32> = (s..s + t).map(|p| p as i32).collect();
            self.act.pos.upload_async(0, &pos, st)?;
            let cpos2: Vec<i32> = pos.iter().map(|&p| (p - 1).max(0)).collect();
            self.cpos1.upload_async(0, &pos, st)?;
            self.cpos2.upload_async(0, &cpos2, st)?;
        }
        self.embed_host(ids)?;
        let st = &self.stream;
        let hashes = self.hash.hash(ids, s, None); // [t][n_engram_layers][cols]
        let cols = self.hash.layout.n_hash_cols();
        let n_eng = cfg.engram_layer_ids.len();
        self.times.compute_s += tw.elapsed().as_secs_f64();
        let te = Instant::now();
        for li in 0..n_eng {
            let l = cfg.engram_layer_ids[li];
            if l >= self.n_layers {
                continue;
            }
            let col_ids: Vec<i64> =
                (0..t).flat_map(|ti| hashes[(ti * n_eng + li) * cols..(ti * n_eng + li + 1) * cols].to_vec()).collect();
            let mut uniq = col_ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let inv: Vec<i32> = col_ids.iter().map(|h| uniq.binary_search(h).unwrap() as i32).collect();
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
        let nc = |r: usize| (s + t) / r;
        let indexed = |r: usize| -> Option<usize> {
            (nc(r) > topk).then(|| bucket(nc(r), 2048, self.max_seq / r + 1))
        };
        let cand_src = cfg.candidate_source_layer;
        let head = match want {
            Want::Nothing => (0, 0),
            Want::Last => (t - 1, 1),
            Want::All => (0, t),
        };
        Ok(StepShape {
            t,
            c1: bucket(nc(1).min(topk), 16, topk),
            c2: bucket(nc(2).min(topk), 16, topk),
            n1: indexed(1),
            n2: indexed(2),
            cand: cand_src >= 0 && (cand_src as usize) < self.n_layers && nc(cfg.compress_ratios[cand_src as usize].max(1)) > 2048 * 8,
            head,
        })
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
                    ck(unsafe {
                        ds_linear_fp8(sr, np(&a.erows), k as i64, np(&e.wkv.w), np(&e.wkv.s), e.wkv.lds() as i64, t as i32,
                                      e.wkv.n as i32, k as i32, 1, np(&a.ekv), nul(), e.wkv.n as i64, nul())
                    }, "engram wkv")?;
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
                self.block(&w, Ctx::Main { l, s: 0, c1: shape.c1, c2: shape.c2 }, t)
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
                let mut hb = vec![0u16; t * hc * d];
                self.act.h.download_async(0, &mut hb, &self.stream)?;
                self.stream.synchronize()?;
                dump_bf16(&dir.join(format!("h{l:02}.f32")), &hb)?;
            }
        }
        if shape.head.1 > 0 {
            let norm = np(&self.norm);
            self.head_dev(shape.head.0, shape.head.1, norm)?;
        }
        Ok(())
    }

    /// After the step's device work: read back the logits it computed, advance the length.
    fn collect(&mut self, shape: StepShape) -> Result<Vec<u16>> {
        let st = &self.stream;
        let v = self.cfg.vocab_size;
        let rows = shape.head.1;
        let mut logits = vec![0u16; rows * v];
        if rows > 0 {
            self.act.logits.download_async(0, &mut logits, st)?;
        }
        st.synchronize()?;
        let s = self.caches.len;
        self.caches.len = s + shape.t;
        self.last_forward = (s, shape.t);
        if let (Some(dir), true) = (&self.dump_dir, rows > 0) {
            dump_bf16(&dir.join("logits.f32"), &logits[(rows - 1) * v..])?;
        }
        Ok(logits)
    }

    /// hc_pre with the final pre_mix + `norm` + the head over rows r0..r0+rows of the hc
    /// stream, into act.logits `[rows][V]` bf16 (streamed from NVMe unless resident).
    fn head_dev(&mut self, r0: usize, rows: usize, norm: *mut c_void) -> Result<()> {
        let th = Instant::now();
        let cfg = &self.cfg;
        let (d, hc, v) = (cfg.dim, cfg.hc_mult, cfg.vocab_size);
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        ck(unsafe {
            ds_hc_pre_norm(sr, a.h.ptr_at(r0 * hc * d), a.pre.ptr_at(r0 * hc), norm, np(&a.head_x), rows as i32, d as i32,
                           hc as i32, cfg.norm_eps)
        }, "final norm")?;
        if let Some(res) = &self.resident {
            ck(unsafe {
                ds_linear_bf16(sr, np(&a.head_x), d as i64, np(&res.head), rows as i32, v as i32, d as i32, 1,
                               np(&a.logits), nul(), v as i64, nul())
            }, "head")?;
        } else {
            let mut ld = Loader::new(&self.ckpt, st)?;
            let mut n0 = 0;
            while n0 < v {
                let n = HEAD_ROWS.min(v - n0);
                ld.read_into("head.weight", n0 * d * 2, (n0 + n) * d * 2, a.head_w.raw())?;
                ck(unsafe {
                    ds_linear_bf16(sr, np(&a.head_x), d as i64, np(&a.head_w), rows as i32, n as i32, d as i32, 1,
                                   a.logits.ptr_at(n0), nul(), v as i64, nul())
                }, "head")?;
                n0 += n;
            }
        }
        self.times.head_s += th.elapsed().as_secs_f64();
        Ok(())
    }

    /// model.py::dspark_seed: write the drafter's window keys for the first `rows` rows of
    /// the last forward, from their main hiddens (the target layers' hc means).
    pub fn seed_draft(&mut self, rows: usize) -> Result<()> {
        let Some(ds) = &self.dspark else { return Err("seed_draft needs the drafter (spec)".into()) };
        let (s, t) = self.last_forward;
        if rows > t {
            return Err(format!("seed_draft({rows}) past the last forward's {t} rows").into());
        }
        if rows == 0 {
            return Ok(());
        }
        let cfg = &self.cfg;
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        let ri = rows as i32;
        let pos: Vec<i32> = (s..s + rows).map(|p| p as i32).collect();
        a.pos.upload_async(0, &pos, st)?;
        let lin = |x: *mut c_void, ldx: usize, w: &Fp8, y: *mut c_void, ldy: usize, what: &str| {
            ck(unsafe {
                ds_linear_fp8(sr, x, ldx as i64, np(&w.w), np(&w.s), w.lds() as i64, ri, w.n as i32, w.k as i32, 1, y,
                              nul(), ldy as i64, nul())
            }, what)
        };
        let k_main = cfg.dspark_target_layer_ids.len() * d;
        lin(np(&a.main_h), k_main, &ds.w.main_proj, np(&a.y), d, "main_proj")?;
        ck(unsafe { ds_rmsnorm(sr, np(&a.y), d as i64, np(&ds.w.main_norm), np(&a.main_x), d as i64, ri, d as i32, cfg.norm_eps) }, "main_norm")?;
        for (k, blk) in ds.blocks.iter().enumerate() {
            lin(np(&a.main_x), d, &blk.wkv, np(&a.kv0), hd, "draft seed wkv")?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kv0), hd as i64, np(&blk.kv_norm), np(&a.kv), hd as i64, ri, hd as i32, cfg.norm_eps) }, "draft seed kv_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kv), hd as i64, ri, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(&self.rope_w), 0) }, "draft seed rope")?;
            for i in 0..rows {
                self.caches.mtp_ring[k].copy_from(((s + i) % RING) * hd, &a.kv, i * hd, hd, st)?;
            }
        }
        st.synchronize()?;
        Ok(())
    }

    /// model.py::dspark_draft, greedy: the block [tok, noise x 4] at positions
    /// last_main_pos+1.., through the three DSpark blocks, then the markov-biased argmax
    /// chain. Returns the `dspark_block_size` drafted tokens. Resident with graphs, the whole
    /// draft is one CUDA graph (its shape never changes) with a single read-back.
    pub fn draft(&mut self, tok: u32, last_main_pos: usize) -> Result<Vec<u32>> {
        if self.dspark.is_none() {
            return Err("draft needs the drafter (spec)".into());
        }
        let cfg = self.cfg.clone();
        let b = cfg.dspark_block_size;
        let s = last_main_pos + 1;
        let mut ids = vec![cfg.dspark_noise_token_id; b];
        ids[0] = tok;
        {
            let st = &self.stream;
            let pos: Vec<i32> = (s..s + b).map(|p| p as i32).collect();
            self.act.pos.upload_async(0, &pos, st)?;
            self.act.dwin.upload_async(0, &vec![last_main_pos as i32; b], st)?;
            self.act.d_idx.upload_async(0, &[tok as i32], st)?;
        }
        self.embed_host(&ids)?;
        if self.use_graphs {
            if self.draft_graph.is_none() {
                self.stream.begin_capture()?;
                let rec = self.draft_dev(s, last_main_pos);
                let graph = self.stream.end_capture();
                rec?;
                self.draft_graph = Some(graph?);
                self.graph_captures += 1;
            } else {
                self.graph_replays += 1;
            }
            self.draft_graph.as_ref().unwrap().launch(&self.stream)?;
        } else {
            self.draft_dev(s, last_main_pos)?;
        }
        let mut out = vec![0i32; b + 1];
        self.act.d_idx.download_async(0, &mut out, &self.stream)?;
        self.stream.synchronize()?;
        Ok(out[1..].iter().map(|&x| x as u32).collect())
    }

    /// The draft's device work: embedding, the three blocks, the head, and the markov
    /// chain entirely on the device (each step's embedding row is gathered by the index
    /// the previous argmax wrote, so the chain needs no host sync).
    fn draft_dev(&mut self, s: usize, last_main_pos: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let b = cfg.dspark_block_size;
        self.embed_dev(b)?;
        let blocks = std::mem::take(&mut self.dspark.as_mut().unwrap().blocks);
        let res = (|| -> Result<()> {
            for (k, w) in blocks.iter().enumerate() {
                self.block(w, Ctx::Draft { k, s, win_end: last_main_pos }, b)?;
            }
            Ok(())
        })();
        self.dspark.as_mut().unwrap().blocks = blocks;
        res?;
        let norm = np(&self.dspark.as_ref().unwrap().w.norm);
        self.head_dev(0, b, norm)?;
        let ds = self.dspark.as_ref().unwrap();
        let (v, r) = (cfg.vocab_size, cfg.dspark_markov_rank);
        let sr = self.stream.raw();
        let a = &self.act;
        for i in 0..b {
            ck(unsafe { ds_gather_rows_u16(sr, np(&ds.w.markov_embed), a.d_idx.ptr_at(i), np(&a.markov_e), 1, r as i32) }, "markov embed")?;
            ck(unsafe {
                ds_linear_bf16(sr, np(&a.markov_e), r as i64, np(&ds.w.markov_head), 1, v as i32, r as i32, 1, np(&a.markov_b),
                               nul(), v as i64, nul())
            }, "markov head")?;
            ck(unsafe {
                ds_argmax_sum_bf16(sr, a.logits.ptr_at(i * v), v as i64, np(&a.markov_b), v as i64, 1, v as i32, a.d_idx.ptr_at(i + 1))
            }, "markov argmax")?;
        }
        Ok(())
    }

    fn hc_mixes(&self, fn_w: &DeviceBuffer<f32>, base: &DeviceBuffer<f32>, scale: &DeviceBuffer<f32>, t: usize,
                pre: &DeviceBuffer<f32>, post: &DeviceBuffer<f32>, comb: &DeviceBuffer<f32>) -> Result<()> {
        let cfg = &self.cfg;
        let a = &self.act;
        let sr = self.stream.raw();
        let k = cfg.hc_mult * cfg.dim;
        let n = cfg.hc_mix_width();
        ck(unsafe { ds_row_rsqrt(sr, np(&a.h), k as i64, t as i32, k as i32, cfg.norm_eps, np(&a.rsq)) }, "hc rsqrt")?;
        ck(unsafe {
            ds_linear_f32w(sr, np(&a.h), k as i64, np(fn_w), t as i32, n as i32, k as i32, HC_SPLITS as i32, nul(),
                           n as i64, np(&a.partial))
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
        let (ratio, l, s) = match ctx {
            Ctx::Main { l, s, .. } => (cfg.compress_ratios[l], l, s),
            Ctx::Draft { s, .. } => (0, usize::MAX, s),
        };
        // independent of q and the window: the chunk's new compressed positions, which this
        // layer's own attention already reads (engine/model.py::_compressed)
        if let (Ctx::Main { .. }, Some(comp)) = (ctx, &w.compressor) {
            self.compress(w, comp, l, s, t)?;
        }
        let (d, hd, rd, h) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim, cfg.n_heads);
        let ti = t as i32;
        {
            let st = &self.stream;
            let sr = st.raw();
            let table = if ratio > 0 { &self.rope_c } else { &self.rope_w };
            let a = &self.act;
            let fp8 = |x: *mut c_void, ldx: usize, wt: &Fp8, y: *mut c_void, ldy: usize, what: &str| {
                ck(unsafe {
                    ds_linear_fp8(sr, x, ldx as i64, np(&wt.w), np(&wt.s), wt.lds() as i64, ti, wt.n as i32, wt.k as i32, 1,
                                  y, nul(), ldy as i64, nul())
                }, what)
            };
            fp8(np(&a.y), d, &w.wq_a, np(&a.qa), cfg.q_lora_rank, "wq_a")?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.qa), cfg.q_lora_rank as i64, np(&w.q_norm), np(&a.qr), cfg.q_lora_rank as i64, ti, cfg.q_lora_rank as i32, cfg.norm_eps) }, "q_norm")?;
            fp8(np(&a.qr), cfg.q_lora_rank, &w.wq_b, np(&a.q), h * hd, "wq_b")?;
            ck(unsafe { ds_rope(sr, np(&a.q), (h * hd) as i64, ti, h as i32, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 0) }, "q rope")?;
            fp8(np(&a.y), d, &w.wkv, np(&a.kv0), hd, "wkv")?;
            // the draft block's own keys live in dkv and are never written to its ring
            let kv_out = if matches!(ctx, Ctx::Draft { .. }) { &a.dkv } else { &a.kv };
            ck(unsafe { ds_rmsnorm(sr, np(&a.kv0), hd as i64, np(&w.kv_norm), np(kv_out), hd as i64, ti, hd as i32, cfg.norm_eps) }, "kv_norm")?;
            ck(unsafe { ds_rope(sr, np(kv_out), hd as i64, ti, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 0) }, "kv rope")?;
            if let Ctx::Main { .. } = ctx {
                let ring = &self.caches.ring[l];
                ck(unsafe { ds_scatter_ring(sr, np(&a.kv), np(ring), RING as i32, np(&a.pos), ti, hd as i32) }, "ring write")?;
            }
        }
        if ratio > 0 && cfg.index_source_layers.contains(&l) {
            self.select(w, l, s, t)?;
        }
        let st = &self.stream;
        let sr = st.raw();
        let table = if ratio > 0 { &self.rope_c } else { &self.rope_w };
        let a = &self.act;
        // (ring, per-row window end, extra keys, their per-row indices, width)
        let (ring, win_pos, ckv_ptr, cidx_ptr, c) = match ctx {
            Ctx::Main { l, c1, c2, .. } => {
                let ring = &self.caches.ring[l];
                if ratio > 0 {
                    let src = cfg.kv_source_of(l).expect("compressed layer has a kv source");
                    let (cidx, c) = match self.selection {
                        Selection::Indexed => (&a.topk, cfg.index_topk),
                        Selection::AllVisible if ratio == 1 => (&a.cidx1, c1),
                        Selection::AllVisible => (&a.cidx2, c2),
                    };
                    (ring, &a.pos, np(&self.caches.ckv[&src]), np(cidx), c)
                } else {
                    (ring, &a.pos, nul(), nul(), 0)
                }
            }
            // model.py::attention with mtp_extra: the window is the main stream's last 128
            // positions in this block's seeded ring, for every draft query alike, plus every
            // draft key of the block (no causal mask among them)
            Ctx::Draft { k, .. } => (&self.caches.mtp_ring[k], &a.dwin, np(&a.dkv), np(&a.dcidx), t),
        };
        ck(unsafe {
            ds_attn_decode(sr, np(&a.q), np(ring), RING as i32, np(win_pos), cfg.window_size as i32, 0, ckv_ptr, cidx_ptr,
                           c as i32, np(&w.attn_sink), (hd as f64).powf(-0.5) as f32, np(&a.o), ti, h as i32)
        }, "attention")?;
        ck(unsafe { ds_rope(sr, np(&a.o), (h * hd) as i64, ti, h as i32, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 1) }, "o rope")?;
        let og = cfg.o_groups;
        let gk = h * hd / og;
        ck(unsafe {
            ds_linear_fp8_grouped(sr, np(&a.o), (h * hd) as i64, gk as i64, cfg.o_lora_rank as i32, np(&w.wo_a.w),
                                  np(&w.wo_a.s), w.wo_a.lds() as i64, ti, w.wo_a.n as i32, gk as i32, np(&a.oa),
                                  w.wo_a.n as i64)
        }, "wo_a")?;
        ck(unsafe {
            ds_linear_fp8(sr, np(&a.oa), (og * cfg.o_lora_rank) as i64, np(&w.wo_b.w), np(&w.wo_b.s), w.wo_b.lds() as i64,
                          ti, w.wo_b.n as i32, w.wo_b.k as i32, 1, np(&a.attn_out), nul(), d as i64, nul())
        }, "wo_b")
    }

    /// model.py::_indexer at an index-source layer: which compressed positions this layer
    /// and the following ones (until the next index source) attend over.
    fn select(&mut self, w: &Layer, l: usize, _s: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let r = cfg.compress_ratios[l];
        let shape = self.shape.expect("select runs inside a backbone step");
        // the key range is the bucketed count; keys past the true count are masked by clen
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
        ck(unsafe {
            ds_linear_fp8(sr, np(&a.qr), cfg.q_lora_rank as i64, np(&ix.wq_b.w), np(&ix.wq_b.s), ix.wq_b.lds() as i64,
                          ti, ix.wq_b.n as i32, ix.wq_b.k as i32, 1, np(&a.iq), nul(), (ih * ihd) as i64, nul())
        }, "index wq_b")?;
        // the indexer's queries always take the compressed-rope table (model.py: freqs_c[pos])
        ck(unsafe { ds_rope(sr, np(&a.iq), (ih * ihd) as i64, ti, ih as i32, ihd as i64, (ihd - rd) as i32, rd as i32, np(&a.pos), np(&self.rope_c), 0) }, "index q rope")?;
        ck(unsafe {
            ds_linear_bf16(sr, np(&a.y), cfg.dim as i64, np(&ix.weights_proj), ti, ih as i32, cfg.dim as i32, 1,
                           np(&a.iwb), nul(), ih as i64, nul())
        }, "index weights_proj")?;
        let scale = ((ihd as f64).powf(-0.5) * (ih as f64).powf(-0.5)) as f32;
        ck(unsafe { ds_bf16_to_f32_scale(sr, np(&a.iwb), (t * ih) as i64, scale, np(&a.iwf)) }, "index weights")?;
        let src = cfg.kv_source_of(l).expect("compressed layer has a kv source");
        let clen = if r == 1 { &a.clen1 } else { &a.clen2 };
        let cand_src = cfg.candidate_source_layer;
        let use_cand = self.cand_live && cand_src >= 0 && (l as i64) > cand_src;
        ck(unsafe {
            ds_index_scores(sr, np(&a.iq), np(&self.caches.ik[&src]), np(&a.iwf), np(clen),
                            if use_cand { np(&a.cand) } else { nul() }, nmax as i64, n_c as i32, np(&a.scores),
                            n_c as i64, ti)
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
    /// chunk into the shared cache (and their index keys). Every row is computed and only the
    /// rows whose position completes a group are scattered, so the work never depends on
    /// position values — the same kernels in both modes and inside the step graph.
    fn compress(&mut self, w: &Layer, comp: &crate::weights::Compressor, l: usize, _s: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let st = &self.stream;
        let sr = st.raw();
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        let r = cfg.compress_ratios[l];
        let a = &self.act;
        let ti = t as i32;
        let cpos = if r == 1 { &self.cpos1 } else { &self.cpos2 };
        if r > 1 {
            let wgate = comp.wgate.as_ref().expect("ratio-2 compressor has a gate");
            let r2 = &self.caches.r2[&l];
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&comp.wkv), ti, hd as i32, d as i32, 1, nul(), np(&a.kvl), hd as i64, nul()) }, "comp wkv")?;
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(wgate), ti, hd as i32, d as i32, 1, nul(), np(&a.sc), hd as i64, nul()) }, "comp wgate")?;
            // pairs read the pending position before the update below replaces it
            ck(unsafe { ds_compress_rows(sr, np(&a.kvl), np(&a.sc), np(&r2.pkv), np(&r2.psc), np(&a.pos), np(&a.lat), ti, hd as i32) }, "compress rows")?;
            ck(unsafe {
                ds_pending_update(sr, np(&a.kvl), np(&a.sc), np(&a.pos), ti, np(&r2.pkv), np(&r2.psc), np(&r2.flag),
                                  np(&r2.mem_kv), np(&r2.mem_sc), np(&r2.before_kv), np(&r2.before_sc), np(&r2.before_flag),
                                  np(&r2.mem_start), hd as i32)
            }, "pending update")?;
        } else {
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&comp.wkv), ti, hd as i32, d as i32, 1, np(&a.lat), nul(), hd as i64, nul()) }, "comp wkv r1")?;
        }
        ck(unsafe { ds_rmsnorm(sr, np(&a.lat), hd as i64, np(&comp.norm), np(&a.latn), hd as i64, ti, hd as i32, cfg.norm_eps) }, "comp norm")?;
        if let Some(key) = w.indexer.as_ref().and_then(|ix| ix.key.as_ref()) {
            let ihd = cfg.index_head_dim;
            ck(unsafe { ds_linear_bf16(sr, np(&a.latn), hd as i64, np(&key.wk), ti, ihd as i32, hd as i32, 1, np(&a.kidx), nul(), ihd as i64, nul()) }, "index wk")?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kidx), ihd as i64, np(&key.k_norm), np(&a.kidxn), ihd as i64, ti, ihd as i32, cfg.norm_eps) }, "index k_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kidxn), ihd as i64, ti, 1, ihd as i64, (ihd - rd) as i32, rd as i32, np(cpos), np(&self.rope_c), 0) }, "index k rope")?;
            ck(unsafe { ds_scatter_compressed(sr, np(&a.kidxn), ihd as i64, np(&self.caches.ik[&l]), np(&a.pos), r as i32, ti, ihd as i32) }, "index k scatter")?;
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
        ck(unsafe { ds_scatter_compressed(sr, np(&a.latn), hd as i64, np(ckv), np(&a.pos), r as i32, ti, hd as i32) }, "ckv scatter")?;
        if let Some(dir) = &self.dump_dir {
            let n = (self.caches.len + t) / r;
            let mut v = vec![0u16; n * hd];
            ckv.download_async(0, &mut v, st)?;
            st.synchronize()?;
            dump_bf16(&dir.join(format!("ckv{l:02}.f32")), &v)?;
        }
        Ok(())
    }

    /// engine/model.py::moe on one box, with the routed experts streamed through the arena.
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
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&w.gate_w), ti, e as i32, d as i32, 1, nul(), np(&a.z), e as i64, nul()) }, "gate")?;
            ck(unsafe { ds_router(sr, np(&a.z), np(&w.gate_bias), keep, ti, e as i32, k as i32, cfg.route_scale, np(&a.idx), np(&a.rw)) }, "router")?;
        }
        let set = match (&self.resident, layer, block) {
            (Some(res), Some(l), _) => Some((&res.experts, l)),
            (Some(res), _, Some(b)) => res.dspark_experts.as_ref().map(|x| (x, b)),
            _ => None,
        };
        if let Some((set, bi)) = set {
            // resident: experts -> slots through the device LUT, no host round trip
            let a = &self.act;
            let ar = &set.arena;
            ck(unsafe { ds_lut_slots(sr, np(&a.idx), set.lut.ptr_at(bi * e), p as i32, np(&a.keys)) }, "lut slots")?;
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
        if let Some(dir) = &self.dump_dir {
            if let Some(l) = w.prefix.strip_prefix("layers.").and_then(|r| r.trim_end_matches('.').parse::<usize>().ok()) {
                let bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("idx{l:02}.i32")), bytes)?;
            }
        }
        let mut experts: Vec<i32> = idx.clone();
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
            // pairs outside this group route to slot -1, which both kernels skip without
            // writing; each (k, token) row of `parts` is written by exactly one group
            let keys: Vec<i32> =
                idx.iter().map(|ex| group.iter().position(|g| g == ex).map(|s| s as i32).unwrap_or(-1)).collect();
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
        let fp8 = |x: *mut c_void, ldx: usize, wt: &Fp8, y: *mut c_void, ldy: usize, what: &str| {
            ck(unsafe {
                ds_linear_fp8(sr, x, ldx as i64, np(&wt.w), np(&wt.s), wt.lds() as i64, ti, wt.n as i32, wt.k as i32, 1,
                              y, nul(), ldy as i64, nul())
            }, what)
        };
        fp8(np(&a.y), d, &w.sh_w1, np(&a.g), inter, "shared w1")?;
        fp8(np(&a.y), d, &w.sh_w3, np(&a.u), inter, "shared w3")?;
        ck(unsafe { ds_swiglu(sr, np(&a.g), inter as i64, np(&a.u), inter as i64, np(&a.hs), inter as i64, ti, inter as i32, cfg.swiglu_limit) }, "shared swiglu")?;
        fp8(np(&a.hs), inter, &w.sh_w2, np(&a.sh), d, "shared w2")?;
        ck(unsafe { ds_moe_combine(sr, np(&a.parts), k as i32, ti, np(&a.sh), np(&a.moe_out), d as i32) }, "moe combine")
    }
}
