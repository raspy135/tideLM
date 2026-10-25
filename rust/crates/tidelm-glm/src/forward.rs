//! engine/glm/tp.py's forward on one rank of the pair.
//!
//! Per layer: pre-norm MLA (q_a / kv_a replicated, q_b / kv_b / o_proj split by heads, absorbed
//! into the 512-wide latent), the DSA indexer on "full" layers (its picks reused by the
//! "shared" ones), then a dense MLP (layers 0-2) or the MoE: sigmoid routing over the resident
//! experts, the shared expert, both halves' intermediates in one gather and the summed hidden
//! columns in another. The decoder layers' KV is split across the pair by position parity
//! (kv_split.py): decode scores and attends this rank's positions for every head and merges the
//! other rank's partials; prefill gathers both ranks' rows into position order and runs the
//! unsplit path on them. The MTP layer keeps its whole cache on both ranks.
//!
//! Rows <= 16 take the decode kernels (EXL3 GEMVs, FP8 absorb, flash-decoding splits), more the
//! prefill ones (unpacked EXL3 through cuBLAS, bf16 absorb, one attention pass) in chunks.
//! Rank 0 sends every engine call to rank 1 before it runs it (`mirror`); rank 1 decides nothing.

use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::blas::{Blas, Mat, R_16BF};
use tidelm_kernels::ds41::{ds_linear_bf16, ds_linear_f32w, ds_route_blocks, ds_route_large, ds_splits_reduce};
use tidelm_kernels::exl3::*;
use tidelm_kernels::glm::*;

use crate::arena::{Arena, Pack};
use crate::config::GlmConfig;
use crate::weights::{Exl3Mat, Exl3Scratch, Members, Mlp, Weights, MAX_ROWS};
use crate::{ck, Result};

/// Decode rows a graph or a verify step carries at most (MTP depth 7 + the anchor).
pub const VERIFY_ROWS: usize = 8;
/// Up to this many rows an EXL3 matrix runs the decode GEMV in 16-row blocks (gate/up then come
/// back interleaved [rows][2N]); more unpack the trellis for cuBLAS ([2][rows][N]).
const GEMV_ROWS: usize = 64;
/// The indexer keeps this many picks a row.
const TOPK: usize = 2048;
/// Prefill row blocks of the indexer's [rows, S] fp32 scores stay within this many values
/// (tp._indexer: (32 << 20) // S rows at a time).
const SCORE_VALUES: usize = 32 << 20;
/// The RoCE slot: the largest decode gather (a verify step's queries and top-k scores).
pub const ROCE_BYTES: usize = 512 * 1024;

const OP_FORWARD: u32 = 1;
const OP_MTP: u32 = 2;
const OP_SET_POS: u32 = 3;
const OP_KEEP_HIDDEN: u32 = 4;
const OP_SAMPLING: u32 = 5;
const OP_ZERO_DEMAND: u32 = 7;
const OP_FULL_ROW: u32 = 6;
const OP_EXIT: u32 = 99;
/// Candidates a rank sends a sampled row: its top logits, with its (max, sum of exp) the
/// host's nucleus over the pair's vocabulary (falls back to the whole row when they do not
/// cover it).
pub const CAND: usize = 1024;
/// a row's floats: max, sum, CAND logits, CAND ids
const CROW: usize = 2 + 2 * CAND;

#[derive(Clone, Debug)]
pub struct Options {
    pub max_seq: usize,
    /// prefill chunk rows (GLM_CHUNK)
    pub chunk: usize,
    /// routed expert slots a rank (GLM_SLOTS)
    pub slots: usize,
    /// decoder KV split by position across the pair (GLM_KV_SPLIT)
    pub kv_split: bool,
    /// prefill o_proj partials summed in BF16 (GLM_PREFILL_REDUCE=bf16)
    pub prefill_reduce_bf16: bool,
    /// MoE prefill in sub-chunks of this many rows (bounds the per-pair scratch)
    pub moe_rows: usize,
    pub pack: std::path::PathBuf,
    /// the serving flags both ranks act on (hot reads split the pass on both, graphs, the disk
    /// store's mirrored calls): compared at boot with the rest
    pub guard_extra: String,
}

impl Options {
    /// Everything that decides numerics or control flow, for the pair's boot comparison.
    pub fn boot_config(&self, model_dir: &Path, pack: &Pack) -> Result<Vec<u8>> {
        use sha2::Digest;
        let cfg = std::fs::read(model_dir.join("config.json"))?;
        let exe = std::fs::read(std::env::current_exe()?)?;
        let s = format!(
            "glm-boot v1\nconfig {:x}\nexe {:x}\nmax_seq {}\nchunk {}\nslots {}\nkv_split {}\nreduce_bf16 {}\nmoe_rows {}\n\
             pack_source {}\nenv {}\nflags {}\n",
            sha2::Sha256::digest(&cfg), sha2::Sha256::digest(&exe), self.max_seq, self.chunk, self.slots, self.kv_split,
            self.prefill_reduce_bf16, self.moe_rows, pack.source_sha256,
            ["TIDELM_GLM_HOT", "NCCL_PROTO"].iter().map(|k| format!("{k}={}", std::env::var(k).unwrap_or_default()))
                .collect::<Vec<_>>().join(" "), self.guard_extra);
        Ok(s.into_bytes())
    }
}

/// This engine's half of the pair: NCCL for prefill-sized collectives, the RoCE one-shot gather
/// for decode's, the control channel rank 0 drives rank 1 by.
pub struct Pair {
    pub comm: tidelm_comm::Comm,
    pub ctl: tidelm_comm::Control,
    pub roce: tidelm_comm::roce::Roce,
}

pub(crate) struct LayerKv {
    /// latent FP8 [rows][512] + fp32 [rows][4]
    pub(crate) cq: DeviceBuffer<u8>,
    pub(crate) cs: DeviceBuffer<f32>,
    /// RoPE key BF16 [rows][64]
    pub(crate) cr: DeviceBuffer<u16>,
    /// indexer key FP8 [rows][128] + fp32 [rows] (full layers)
    pub(crate) iq: Option<DeviceBuffer<u8>>,
    pub(crate) is: Option<DeviceBuffer<f32>>,
    /// split by position across the pair (decoder layers under kv_split)
    pub(crate) split: bool,
}

/// Activations and scratch, sized for a prefill chunk.
struct Act {
    tok: DeviceBuffer<i32>,
    pos0: DeviceBuffer<i32>,
    mpos0: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    rows: DeviceBuffer<i32>,
    bound: DeviceBuffer<i32>,
    mpos: DeviceBuffer<i32>,
    h: DeviceBuffer<u16>,
    xn: DeviceBuffer<u16>,
    a: DeviceBuffer<u16>,
    f32buf: DeviceBuffer<f32>,
    gs: DeviceBuffer<u8>,
    qa: DeviceBuffer<u16>,
    qr: DeviceBuffer<u16>,
    ckv: DeviceBuffer<u16>,
    c: DeviceBuffer<u16>,
    kr: DeviceBuffer<u16>,
    qb: DeviceBuffer<u16>,
    /// absorbed queries [rows][hl][576] (prefill), decode's go to qsend
    qf: DeviceBuffer<u16>,
    /// decode: [rows][hl * 576 bf16 | top-k scores fp32] for the split gather
    qsend: DeviceBuffer<u8>,
    qi: DeviceBuffer<u16>,
    ki: DeviceBuffer<u16>,
    iw: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    idx: DeviceBuffer<i32>,
    olat: DeviceBuffer<u16>,
    o: DeviceBuffer<u16>,
    pm: DeviceBuffer<f32>,
    pl: DeviceBuffer<f32>,
    pacc: DeviceBuffer<f32>,
    own_o: DeviceBuffer<f32>,
    own_l: DeviceBuffer<f32>,
    send2: DeviceBuffer<u8>,
    gcq: DeviceBuffer<u8>,
    gcs: DeviceBuffer<f32>,
    gcr: DeviceBuffer<u16>,
    giq: DeviceBuffer<u8>,
    gis: DeviceBuffer<f32>,
    kvb_bf: DeviceBuffer<u16>,
    split_part: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    eids: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    pick: DeviceBuffer<i32>,
    wts: DeviceBuffer<f32>,
    hot: DeviceBuffer<f32>,
    gu: DeviceBuffer<u16>,
    /// [routed xd fp16 [P][inter/W] | shared act bf16 [rows][inter/W]], one gather
    pair: DeviceBuffer<u8>,
    xdf: DeviceBuffer<u16>,
    hf: DeviceBuffer<u16>,
    xh0: DeviceBuffer<u16>,
    xh1: DeviceBuffer<u16>,
    zup: DeviceBuffer<f32>,
    zd: DeviceBuffer<f32>,
    local: DeviceBuffer<f32>,
    shout: DeviceBuffer<f32>,
    mout: DeviceBuffer<u16>,
    dact: DeviceBuffer<u16>,
    dfull: DeviceBuffer<u16>,
    uids: DeviceBuffer<i32>,
    ucount: DeviceBuffer<i32>,
    members: DeviceBuffer<i32>,
    route_work: DeviceBuffer<i32>,
    block_slot: DeviceBuffer<i32>,
    block_pair: DeviceBuffer<i32>,
    block_route: DeviceBuffer<i32>,
    head: DeviceBuffer<f32>,
    pairs: DeviceBuffer<i32>,
    out_tok: DeviceBuffer<i32>,
    hn: DeviceBuffer<u16>,
    mhn: DeviceBuffer<u16>,
    hlast: DeviceBuffer<u16>,
    ehin: DeviceBuffer<u16>,
    ehout: DeviceBuffer<u16>,
    x3s: Exl3Scratch,
    /// the side stream's EXL3 scratch and split-K partials (decode rows)
    x3s2: Exl3Scratch,
    split_part2: DeviceBuffer<f32>,
}

/// Where the MTP layer's previous hidden comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hidden {
    /// the kept target hidden (keep_hidden)
    Last,
    /// the MTP layer's own output row 0 (a draft chain)
    Mtp,
    /// rows [from, from + n) of the last decoder forward's post-norm hidden
    Target(usize),
}

pub struct Engine {
    pub cfg: GlmConfig,
    pub opt: Options,
    pub rank: usize,
    pub world: usize,
    hl: usize,
    vl: usize,
    /// cache rows a split layer holds (ceil(max_seq / world)); its spare row follows
    s_loc: usize,
    pub(crate) st: Stream,
    /// the side stream of a decode pass's independent pieces, and its fork / join events
    st2: Stream,
    ev_fork: tidelm_cuda::Event,
    ev_join: tidelm_cuda::Event,
    blas: Blas,
    pub w: Weights,
    members: Members,
    pub arena: Arena,
    pub mtp_arena: Option<Arena>,
    /// [layer][expert] -> arena slot, -1 not resident (device and host)
    lut: DeviceBuffer<i32>,
    pub lut_h: Vec<i32>,
    allowed: DeviceBuffer<u8>,
    pub allowed_h: Vec<u8>,
    /// the MTP layer: every expert in its own slot
    mtp_lut: DeviceBuffer<i32>,
    all_ones: DeviceBuffer<u8>,
    /// routed weight per [layer][expert] (adaptive residency's demand), and [picks, missed,
    /// mass, missed mass]
    pub demand: DeviceBuffer<f32>,
    pub miss: DeviceBuffer<f64>,
    pub(crate) kv: Vec<LayerKv>,
    act: Act,
    pub pair: Option<Pair>,
    pub pack: Pack,
    /// the sequence's length and tokens (the KV holds rows for these)
    pub pos: usize,
    pub tokens: Vec<u32>,
    /// the MTP layer's first position in the pass running
    mtp_pos: usize,
    /// rows of the last forward's pass in its post-norm hidden (hn): the last is rows - 1
    pub last_rows: usize,
    /// a decode row's bytes in the split gathers (qrow_bytes, row2_bytes)
    qrow: usize,
    row2: usize,
    /// the decoder's MoE layers (hot-read points of a segmented decode pass)
    moe_layers: Vec<usize>,
    /// decode passes replay CUDA graphs (captured per row count on first use)
    pub graphs: bool,
    graphs_map: std::collections::HashMap<GraphKey, tidelm_cuda::GraphExec>,
    /// synchronous hot reads between a decode pass's segments (None: off)
    pub hot: Option<crate::adapt::Hot>,
    pub res: crate::adapt::Res,
    h_ids: tidelm_cuda::PinnedBuffer<i32>,
    h_pos: tidelm_cuda::PinnedBuffer<i32>,
    h_tok: tidelm_cuda::PinnedBuffer<i32>,
    /// the disk prefix store's directory for this rank (None: off) and its pinned staging
    pub prefix_dir: Option<std::path::PathBuf>,
    pub(crate) stage: Option<tidelm_cuda::PinnedBuffer<u8>>,
    /// passes also return each row's sampling candidates (set_sampling)
    pub sampled: bool,
    temp: DeviceBuffer<f32>,
    h_temp: tidelm_cuda::PinnedBuffer<f32>,
    cand: DeviceBuffer<f32>,
    /// both ranks' candidates of the last sampled pass [world][rows][CROW]
    h_cand: tidelm_cuda::PinnedBuffer<f32>,
    cand_rows: usize,
}

fn np<T: Copy>(b: &DeviceBuffer<T>) -> *mut c_void {
    b.raw()
}
fn at<T: Copy>(b: &DeviceBuffer<T>, elem: usize) -> *mut c_void {
    b.ptr_at(elem)
}

impl Engine {
    pub fn load(model_dir: &Path, opt: Options, rank: usize, world: usize, pair: Option<Pair>,
                residents: &[(usize, usize)], log: &dyn Fn(&str)) -> Result<Self> {
        let cfg = GlmConfig::load(model_dir)?;
        let st = Stream::new()?;
        let blas = Blas::new(st.raw())?;
        let hl = cfg.heads / world;
        let vl = cfg.vocab / world;
        let pack = Pack::open(&opt.pack, cfg.hidden, cfg.moe_inter)?;
        if (pack.rank, pack.world) != (rank, world) {
            return Err(format!("{}: pack of rank {} / {}, engine rank {rank} / {world}", pack.path.display(), pack.rank,
                               pack.world).into());
        }
        let t0 = std::time::Instant::now();
        let w = Weights::load(model_dir, &cfg, rank, world, opt.max_seq, &st, log)?;
        log(&format!("[glm] dense weights {:.2} GB a rank in {:.0} s", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64()));
        let nl = cfg.n_layers();
        let e = cfg.experts;
        // the decoder arena (3-bit records) and the MTP layer's (4-bit, all resident)
        let bits = pack.record(cfg.dense.iter().position(|d| !d).ok_or("no MoE layer")?, 0)?.bits;
        let arena = Arena::new(opt.slots + 1, bits, world, cfg.hidden, cfg.moe_inter)?;
        let mut lut_h = vec![-1i32; nl * e];
        for (slot, &(l, ex)) in residents.iter().enumerate() {
            if slot >= opt.slots {
                return Err("more residents than slots".into());
            }
            lut_h[l * e + ex] = slot as i32;
        }
        let t1 = std::time::Instant::now();
        load_slots(&pack, &arena, &residents.iter().enumerate().map(|(s, &(l, ex))| (s, l, ex)).collect::<Vec<_>>())?;
        log(&format!("[glm] {} routed experts resident ({:.1} GB) in {:.0} s", residents.len(),
                     (residents.len() * arena.slot_bytes()) as f64 / 1e9, t1.elapsed().as_secs_f64()));
        let mtp_arena = if cfg.mtp {
            let l = cfg.mtp_layer();
            let a = Arena::new(e, pack.record(l, 0)?.bits, world, cfg.hidden, cfg.moe_inter)?;
            load_slots(&pack, &a, &(0..e).map(|x| (x, l, x)).collect::<Vec<_>>())?;
            Some(a)
        } else {
            None
        };
        let allowed_h: Vec<u8> = lut_h.iter().map(|&s| (s >= 0) as u8).collect();
        let s_loc = if opt.kv_split && world > 1 { opt.max_seq.div_ceil(world) } else { opt.max_seq };
        let mut kv = Vec::with_capacity(nl);
        let mut kv_bytes = 0usize;
        for l in 0..nl {
            let split = opt.kv_split && world > 1 && l < cfg.layers;
            let rows = if split { s_loc + 1 } else { opt.max_seq };
            let full = cfg.index_full[l];
            let lk = LayerKv {
                cq: DeviceBuffer::zeroed(rows * 512)?,
                cs: DeviceBuffer::zeroed(rows * 4)?,
                cr: DeviceBuffer::zeroed(rows * 64)?,
                iq: if full { Some(DeviceBuffer::zeroed(rows * 128)?) } else { None },
                is: if full { Some(DeviceBuffer::zeroed(rows)?) } else { None },
                split,
            };
            kv_bytes += rows * (512 + 16 + 128) + if full { rows * 132 } else { 0 };
            kv.push(lk);
        }
        log(&format!("[glm] KV cache {:.2} GB a rank ({} positions{})", kv_bytes as f64 / 1e9, opt.max_seq,
                     if opt.kv_split { ", split" } else { "" }));
        let act = Act::new(&cfg, &opt, hl, vl, s_loc, opt.slots + 1)?;
        let mtp_lut: Vec<i32> = (0..e as i32).collect();
        let moe_layers: Vec<usize> = (0..cfg.layers).filter(|&l| !cfg.dense[l]).collect();
        let opt_chunk = opt.chunk;
        let opt_slots = opt.slots;
        let moe_layers_res = moe_layers.clone();
        let engine = Engine {
            lut: DeviceBuffer::from_slice(&lut_h)?,
            allowed: DeviceBuffer::from_slice(&allowed_h)?,
            mtp_lut: DeviceBuffer::from_slice(&mtp_lut)?,
            all_ones: DeviceBuffer::from_slice(&vec![1u8; e])?,
            demand: DeviceBuffer::zeroed(nl * e)?,
            miss: DeviceBuffer::zeroed(4)?,
            members: Members::new()?,
            lut_h,
            allowed_h,
            cfg,
            opt,
            rank,
            world,
            hl,
            vl,
            s_loc,
            st,
            st2: Stream::new()?,
            ev_fork: tidelm_cuda::Event::new()?,
            ev_join: tidelm_cuda::Event::new()?,
            blas,
            w,
            arena,
            mtp_arena,
            kv,
            act,
            pair,
            pack,
            pos: 0,
            tokens: Vec::new(),
            mtp_pos: 0,
            last_rows: 0,
            qrow: qrow_bytes(hl),
            row2: row2_bytes(hl),
            moe_layers,
            graphs: false,
            graphs_map: std::collections::HashMap::new(),
            hot: None,
            res: crate::adapt::Res::new(nl, e, opt_slots, moe_layers_res, residents),
            h_ids: tidelm_cuda::PinnedBuffer::zeroed(opt_chunk.max(MAX_ROWS))?,
            h_pos: tidelm_cuda::PinnedBuffer::zeroed(2)?,
            h_tok: tidelm_cuda::PinnedBuffer::zeroed(MAX_ROWS)?,
            prefix_dir: None,
            stage: None,
            sampled: false,
            temp: DeviceBuffer::from_slice(&[1.0f32])?,
            h_temp: tidelm_cuda::PinnedBuffer::zeroed(1)?,
            cand: DeviceBuffer::new(MAX_ROWS * CROW)?,
            h_cand: tidelm_cuda::PinnedBuffer::zeroed(world * MAX_ROWS * CROW)?,
            cand_rows: 0,
        };
        engine.st.synchronize()?;
        Ok(engine)
    }

    pub fn stream(&self) -> &Stream {
        &self.st
    }

    /// The host lut / allowed mirrors to the device (stream-ordered).
    pub(crate) fn lut_upload(&mut self) -> Result<()> {
        self.lut.upload_async(0, &self.lut_h, &self.st)?;
        self.allowed.upload_async(0, &self.allowed_h, &self.st)?;
        Ok(())
    }

    /// One [layer][expert] entry of lut / allowed to the device.
    pub(crate) fn lut_entry(&mut self, i: usize) -> Result<()> {
        self.lut.upload_async(i, &self.lut_h[i..i + 1], &self.st)?;
        self.allowed.upload_async(i, &self.allowed_h[i..i + 1], &self.st)?;
        Ok(())
    }

    // ------------------------------------------------------------------ collectives
    /// out [world][nbytes] <- every rank's `nbytes` at `local`. `big`: an eager prefill-sized
    /// one through NCCL (the RoCE gather is built for decode's latency).
    fn gather(&self, local: *const c_void, nbytes: usize, out: *mut c_void, big: bool) -> Result<()> {
        match &self.pair {
            None => unsafe {
                tidelm_cuda::memcpy_async(out, local, nbytes, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
                Ok(())
            },
            Some(p) => {
                if big || nbytes > ROCE_BYTES {
                    p.comm.allgather(local, out, nbytes, &self.st)
                } else {
                    p.roce.gather(local, out, nbytes, &self.st)
                }
            }
        }
    }

    /// A rank-sharded [m][n] tensor of `esz`-byte elements gathered and laid out as full rows
    /// [m][world * n] at `full`.
    fn gather_cols(&self, local: *const c_void, m: usize, n: usize, esz: usize, full: *mut c_void, big: bool) -> Result<()> {
        if self.world == 1 {
            unsafe { tidelm_cuda::memcpy_async(full, local, m * n * esz, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)? };
            return Ok(());
        }
        self.gather(local, m * n * esz, np(&self.act.gs), big)?;
        ck(unsafe { glm_unshard(self.st.raw(), np(&self.act.gs), 0, full, m as i32, n as i32, self.world as i32, esz as i32) },
           "unshard")
    }

    /// The pair's sum of fp32 [n] partials at `buf` -> bf16 at `out` (decode: a gather and an
    /// order-fixed add, the same bits on both ranks; prefill: NCCL, in fp32 or in bf16).
    fn allreduce(&self, buf: *mut c_void, n: usize, out: *mut c_void, prefill: bool) -> Result<()> {
        let st = self.st.raw();
        if self.world == 1 {
            return ck(unsafe { glm_f32_bf16(st, buf, out, n as i64) }, "f32 -> bf16");
        }
        let p = self.pair.as_ref().unwrap();
        if prefill {
            if self.opt.prefill_reduce_bf16 {
                ck(unsafe { glm_f32_bf16(st, buf, out, n as i64) }, "f32 -> bf16")?;
                return p.comm.allreduce_bf16(out, n, &self.st);
            }
            p.comm.allreduce_f32(buf, n, &self.st)?;
            return ck(unsafe { glm_f32_bf16(st, buf, out, n as i64) }, "f32 -> bf16");
        }
        self.gather(buf, n * 4, np(&self.act.gs), false)?;
        ck(unsafe { glm_sum2(st, np(&self.act.gs), n as i64, out, std::ptr::null_mut()) }, "sum2")
    }

    // ------------------------------------------------------------------ matrices
    /// x [rows][ldx] bf16 -> out [rows][N] through an EXL3 matrix's decode or prefill path.
    fn exl3(&self, m: &Exl3Mat, x: *const c_void, ldx: usize, rows: usize, out: *mut c_void, out_bf16: bool) -> Result<()> {
        self.exl3_on(&self.st, &self.act.x3s, m, x, ldx, rows, out, out_bf16)
    }

    /// exl3 on a given stream with its own scratch (decode rows only off the main stream).
    #[allow(clippy::too_many_arguments)]
    fn exl3_on(&self, st: &Stream, x3: &Exl3Scratch, m: &Exl3Mat, x: *const c_void, ldx: usize, rows: usize, out: *mut c_void,
               out_bf16: bool) -> Result<()> {
        if rows <= MAX_ROWS {
            m.decode(st, &self.members, x3, x, ldx, rows, out, out_bf16)
        } else if rows <= GEMV_ROWS {
            // a short prompt: the decode GEMV over 16-row blocks reads each trellis rows / 16
            // times, where unpacking it writes and rereads 2 bytes a weight (16 bits against 3-5)
            let esz = if out_bf16 { 2 } else { 4 };
            for r0 in (0..rows).step_by(MAX_ROWS) {
                let n = MAX_ROWS.min(rows - r0);
                let xi = unsafe { (x as *const u16).add(r0 * ldx) } as *const c_void;
                let oi = unsafe { (out as *mut u8).add(r0 * m.g * m.n * esz) } as *mut c_void;
                m.decode(st, &self.members, x3, xi, ldx, n, oi, out_bf16)?;
            }
            Ok(())
        } else {
            for g in 0..m.g {
                let o = unsafe { (out as *mut u8).add(g * rows * m.n * if out_bf16 { 2 } else { 4 }) } as *mut c_void;
                m.prefill(&self.st, &self.blas, &self.act.x3s, g, x, ldx, rows, o, out_bf16)?;
            }
            Ok(())
        }
    }

    /// y fp32 [rows][n] = x bf16 [rows][k] . W^T, W fp32 [n][k] (F.linear(x.float(), w)).
    fn linear_f32w(&self, x: *const c_void, ldx: usize, w: &DeviceBuffer<f32>, rows: usize, n: usize, k: usize,
                   y: *mut c_void, splits: usize) -> Result<()> {
        self.linear_f32w_on(&self.st, np(&self.act.split_part), x, ldx, w, rows, n, k, y, splits)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_f32w_on(&self, st: &Stream, part: *mut c_void, x: *const c_void, ldx: usize, w: &DeviceBuffer<f32>, rows: usize,
                      n: usize, k: usize, y: *mut c_void, splits: usize) -> Result<()> {
        ck(unsafe { ds_linear_f32w(st.raw(), x, ldx as i64, np(w), rows as i32, n as i32, k as i32, splits as i32,
                                   if splits > 1 { part } else { y }, n as i64, part) }, "linear f32w")?;
        if splits > 1 {
            ck(unsafe { ds_splits_reduce(st.raw(), part, splits as i32, rows as i32, n as i32, std::ptr::null_mut(), y,
                                         n as i64) }, "splits reduce")?;
        }
        Ok(())
    }

    /// The side stream starts after the main one's work so far.
    fn fork(&self) -> Result<()> {
        self.ev_fork.record(&self.st)?;
        self.st2.wait_event(&self.ev_fork)?;
        Ok(())
    }

    /// The main stream continues after the side one's work.
    fn join(&self) -> Result<()> {
        self.ev_join.record(&self.st2)?;
        self.st.wait_event(&self.ev_join)?;
        Ok(())
    }

    // ------------------------------------------------------------------ attention
    #[allow(clippy::too_many_lines)]
    fn attention(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        let (cfg, lw, a) = (&self.cfg, &self.w.layers[l], &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let hl = self.hl;
        let mtp = cfg.mtp && l == cfg.mtp_layer();
        let kv = &self.kv[l];
        let split = kv.split;
        // positions, the cache rows they take (a split layer's own, or the spare row), the last
        // row each may attend (the local bound on a split layer, else the position itself)
        let (posb, rowsb, bound) = if mtp { (np(&a.mpos), np(&a.mpos), np(&a.mpos)) } else { (np(&a.pos), np(&a.rows), np(&a.bound)) };
        let p0 = if mtp { self.mtp_pos } else { self.pos };
        let full = cfg.index_full[l];
        let decode = !prefill;
        // decode: the latent / RoPE key / indexer key chain runs on the side stream while this
        // one does the queries (independent until the indexer scores: small GEMVs alone are
        // latency-bound, 55-90 GB/s for q_a / kv_a); prefill keeps one stream (cuBLAS is on it)
        let (sst, x3, part2) = if decode { (&self.st2, &a.x3s2, np(&a.split_part2)) } else { (&self.st, &a.x3s, np(&a.split_part)) };
        let s2 = sst.raw();
        if decode {
            self.fork()?;
        }
        self.exl3_on(sst, x3, &lw.kv_a, np(&a.xn), h, t, np(&a.ckv), true)?;
        let ldk = lw.kv_a.n;
        ck(unsafe { glm_rmsnorm(s2, np(&a.ckv), ldk as i64, std::ptr::null(), 0, np(&lw.kv_norm), np(&a.c), 512, t as i32, 512,
                                1e-6) }, "kv norm")?;
        ck(unsafe { glm_kv_fp8(s2, np(&a.c), 512, rowsb, np(&kv.cq), np(&kv.cs), t as i32, 512) }, "latent cache")?;
        ck(unsafe { glm_rope(s2, at(&a.ckv, 512), ldk as i64, 64, np(&a.kr), 64, 64, t as i32, 1, np(&self.w.cos), np(&self.w.sin),
                             posb) }, "rope key")?;
        ck(unsafe { glm_rows_copy(s2, np(&a.kr), 64, rowsb, np(&kv.cr), t as i32, 64) }, "rope cache")?;
        if full {
            let ix = lw.index.as_ref().unwrap();
            if prefill {
                self.blas.gemm_rm(t, 128, h, Mat { ptr: np(&a.xn), ty: R_16BF, ld: h as i64 }, Mat { ptr: np(&ix.wk), ty: R_16BF, ld: h as i64 },
                                  true, np(&a.ki), R_16BF, 128)?;
            } else {
                // 128 outputs: split K 12 ways (4 blocks of output rows alone left the GPU idle)
                ck(unsafe { ds_linear_bf16(s2, np(&a.xn), h as i64, np(&ix.wk), t as i32, 128, h as i32, 12, part2,
                                           std::ptr::null_mut(), 128, part2) }, "index wk")?;
                ck(unsafe { ds_splits_reduce(s2, part2, 12, t as i32, 128, np(&a.ki), std::ptr::null_mut(), 128) }, "wk reduce")?;
            }
            ck(unsafe { glm_index_key(s2, np(&a.ki), 128, np(&ix.kn_w), np(&ix.kn_b), np(&self.w.cos), np(&self.w.sin), posb,
                                      rowsb, np(kv.iq.as_ref().unwrap()), np(kv.is.as_ref().unwrap()), t as i32) }, "index key")?;
            self.linear_f32w_on(sst, part2, np(&a.xn), h, &ix.wp, t, 32, h, np(&a.iw), if prefill { 1 } else { 24 })?;
        }
        // q_a, its norm, q_b (this rank's heads), RoPE on its last 64 dims, the absorbed query
        self.exl3(&lw.q_a, np(&a.xn), h, t, np(&a.qa), true)?;
        ck(unsafe { glm_rmsnorm(st, np(&a.qa), cfg.q_lora as i64, std::ptr::null(), 0, np(&lw.q_norm), np(&a.qr),
                                cfg.q_lora as i64, t as i32, cfg.q_lora as i32, 1e-6) }, "q norm")?;
        self.exl3(&lw.q_b, np(&a.qr), cfg.q_lora, t, np(&a.qb), true)?;
        let qrow = if decode { self.qrow / 2 } else { hl * 576 };
        let qbase = if decode { np(&a.qsend) } else { np(&a.qf) };
        let qrope = unsafe { (qbase as *mut u16).add(512) } as *mut c_void;
        ck(unsafe { glm_rope(st, at(&a.qb, 192), (hl * 256) as i64, 256, qrope, qrow as i64, 576, t as i32, hl as i32,
                             np(&self.w.cos), np(&self.w.sin), posb) }, "rope q")?;
        if decode {
            ck(unsafe { glm_absorb_q(st, np(&a.qb), (hl * 256) as i64, 256, np(&lw.kvb_q), np(&lw.kvb_s), qbase, qrow as i64,
                                     576, t as i32, hl as i32) }, "absorb q")?;
        } else {
            ck(unsafe { glm_kvb_dequant(st, np(&lw.kvb_q), np(&lw.kvb_s), np(&a.kvb_bf), (hl * 448) as i64) }, "kv_b dequant")?;
            // per head: q_lat [t][512] = q_nope [t][192] . Wuk [192][512]
            self.blas.gemm_rm_batched(t, 512, 192, Mat { ptr: np(&a.qb), ty: R_16BF, ld: (hl * 256) as i64 }, 256,
                                      Mat { ptr: np(&a.kvb_bf), ty: R_16BF, ld: 512 }, 448 * 512, false, qbase, R_16BF,
                                      qrow as i64, 576, hl)?;
        }
        let s_all = p0 + t; // positions [0, S) once this step's rows are written
        let dense_causal = prefill && s_all <= TOPK;
        if full {
            let ix = lw.index.as_ref().unwrap();
            self.exl3(&ix.wq_b, np(&a.qr), cfg.q_lora, t, np(&a.qi), true)?;
            ck(unsafe { glm_rope(st, np(&a.qi), 4096, 128, np(&a.qi), 4096, 128, t as i32, 32, np(&self.w.cos), np(&self.w.sin),
                                 posb) }, "rope index q")?;
        }
        if decode {
            self.join()?;
        }
        let scale = (cfg.qk_dim() as f32).powf(-0.5);
        let inv_d = (cfg.index_dim as f32).powf(-0.5);
        let wscale = (cfg.index_heads as f32).powf(-0.5);
        if decode && split {
            // each rank scores and attends its own positions for every head
            let lim = self.s_loc.min(self.opt.max_seq.div_ceil(self.world));
            if full {
                let ix_q = np(kv.iq.as_ref().unwrap());
                let ix_s = np(kv.is.as_ref().unwrap());
                ck(unsafe { glm_index_scores(st, np(&a.qi), np(&a.iw), wscale, ix_q, ix_s, posb, t as i32, lim as i32,
                                             self.world as i32, self.rank as i32, 0, inv_d, np(&a.scores), lim as i64) },
                   "index scores")?;
                let vals = at(&a.qsend, hl * 576 * 2);
                ck(unsafe { glm_topk(st, np(&a.scores), lim as i64, posb, self.world as i32, self.rank as i32, lim as i32,
                                     TOPK as i32, t as i32, np(&a.idx), TOPK as i64, vals, (self.qrow / 4) as i64) }, "topk")?;
            }
            let nbytes = t * self.qrow;
            self.gather(np(&a.qsend), nbytes, np(&a.gs), false)?;
            if full {
                ck(unsafe { glm_split_share(st, np(&a.gs), nbytes as i64, self.qrow as i64, (hl * 576 * 2) as i64, np(&a.idx),
                                            TOPK as i64, self.rank as i32, TOPK as i32, t as i32) }, "split share")?;
            }
            let hh = cfg.heads;
            let (ns, chunk) = decode_split(TOPK, hh, t);
            ck(unsafe { glm_mla(st, 2, np(&a.gs), (nbytes / 2) as i64, (self.qrow / 2) as i64, 576, hl as i32, np(&kv.cq), np(&kv.cs),
                                np(&kv.cr), 0, np(&a.idx), TOPK as i64, TOPK as i32, 0, bound, t as i32, hh as i32, scale,
                                ns as i32, chunk as i32, np(&a.pm), np(&a.pl), np(&a.pacc), std::ptr::null_mut(), 0, 0) },
               "mla (split)")?;
            ck(unsafe { glm_mla_combine(st, np(&a.pm), np(&a.pl), np(&a.pacc), t as i32, hh as i32, ns as i32,
                                        std::ptr::null_mut(), 0, 0, np(&a.own_o), np(&a.own_l), np(&a.send2), self.row2 as i64,
                                        hl as i32, (self.rank * hl) as i32) }, "mla combine (split)")?;
            let n2 = t * self.row2;
            self.gather(np(&a.send2), n2, np(&a.gs), false)?;
            let recv = at(&a.gs, (1 - self.rank) * n2);
            ck(unsafe { glm_mla_merge(st, np(&a.own_o), np(&a.own_l), recv, self.row2 as i64, t as i32, hl as i32, np(&a.olat)) },
               "mla merge")?;
        } else if decode {
            // the MTP layer (or an unsplit cache): this rank's heads over the whole cache
            if full {
                let ix_q = np(kv.iq.as_ref().unwrap());
                let ix_s = np(kv.is.as_ref().unwrap());
                let lim = self.opt.max_seq;
                ck(unsafe { glm_index_scores(st, np(&a.qi), np(&a.iw), wscale, ix_q, ix_s, posb, t as i32, lim as i32, 1, 0, 0,
                                             inv_d, np(&a.scores), lim as i64) }, "index scores")?;
                ck(unsafe { glm_topk(st, np(&a.scores), lim as i64, posb, 1, 0, lim as i32, TOPK as i32, t as i32, np(&a.idx),
                                     TOPK as i64, std::ptr::null_mut(), 0) }, "topk")?;
            }
            let (ns, chunk) = decode_split(TOPK, hl, t);
            ck(unsafe { glm_mla(st, 2, qbase, 0, qrow as i64, 576, hl as i32, np(&kv.cq), np(&kv.cs), np(&kv.cr), 0,
                                np(&a.idx), TOPK as i64, TOPK as i32, 1, bound, t as i32, hl as i32, scale, ns as i32, chunk as i32,
                                np(&a.pm), np(&a.pl), np(&a.pacc), std::ptr::null_mut(), 0, 0) }, "mla")?;
            ck(unsafe { glm_mla_combine(st, np(&a.pm), np(&a.pl), np(&a.pacc), t as i32, hl as i32, ns as i32, np(&a.olat),
                                        (hl * 512) as i64, 512, std::ptr::null_mut(), std::ptr::null_mut(),
                                        std::ptr::null_mut(), 0, 0, 0) }, "mla combine")?;
        } else {
            // prefill: a split layer's caches for [0, S) gathered into position order first
            let (cq, cs, cr, iq, is, split_n) = if split {
                let n = s_all.div_ceil(self.world);
                let big = true;
                self.gather(np(&kv.cq), n * 512, np(&a.gcq), big)?;
                self.gather(np(&kv.cs), n * 16, np(&a.gcs), big)?;
                self.gather(np(&kv.cr), n * 128, np(&a.gcr), big)?;
                if full {
                    self.gather(np(kv.iq.as_ref().unwrap()), n * 128, np(&a.giq), big)?;
                    self.gather(np(kv.is.as_ref().unwrap()), n * 4, np(&a.gis), big)?;
                }
                (np(&a.gcq), np(&a.gcs), np(&a.gcr), np(&a.giq), np(&a.gis), n)
            } else {
                let (iq, is) = if full { (np(kv.iq.as_ref().unwrap()), np(kv.is.as_ref().unwrap())) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
                (np(&kv.cq), np(&kv.cs), np(&kv.cr), iq, is, 0)
            };
            if full && !dense_causal {
                // rows in blocks that bound the [rows, S] scores
                let step = (SCORE_VALUES / s_all.max(1)).max(1);
                let mut r0 = 0;
                while r0 < t {
                    let n = step.min(t - r0);
                    let pr = unsafe { (posb as *mut i32).add(r0) } as *mut c_void;
                    ck(unsafe { glm_index_scores(st, at(&a.qi, r0 * 4096), at(&a.iw, r0 * 32), wscale, iq, is, pr,
                                                 n as i32, s_all as i32, 1, 0, split_n as i32, inv_d, np(&a.scores),
                                                 s_all as i64) }, "index scores (prefill)")?;
                    ck(unsafe { glm_topk(st, np(&a.scores), s_all as i64, pr, 1, 0, s_all as i32, TOPK as i32,
                                         n as i32, at(&a.idx, r0 * TOPK), TOPK as i64, std::ptr::null_mut(), 0) }, "topk (prefill)")?;
                    r0 += n;
                }
            }
            let (idx, k) = if dense_causal { (std::ptr::null_mut(), s_all) } else { (np(&a.idx), TOPK) };
            let posp = posb;
            let chunk = k.div_ceil(32) * 32;
            ck(unsafe { glm_mla(st, 2, qbase, 0, qrow as i64, 576, hl as i32, cq, cs, cr, split_n as i32, idx, TOPK as i64,
                                k as i32, 1, posp, t as i32, hl as i32, scale, 1, chunk as i32, std::ptr::null_mut(),
                                std::ptr::null_mut(), std::ptr::null_mut(), np(&a.olat), (hl * 512) as i64, 512) }, "mla (prefill)")?;
        }
        // the latent output through kv_b's value half, o_proj (this rank's head columns), summed
        if decode {
            ck(unsafe { glm_absorb_o(st, np(&a.olat), (hl * 512) as i64, 512, np(&lw.kvb_q), np(&lw.kvb_s), np(&a.o),
                                     (hl * 256) as i64, t as i32, hl as i32) }, "absorb o")?;
        } else {
            // per head: o [t][256] = o_lat [t][512] . Wuv^T (Wuv [256][512], rows 192.. of the head's kv_b)
            self.blas.gemm_rm_batched(t, 256, 512, Mat { ptr: np(&a.olat), ty: R_16BF, ld: (hl * 512) as i64 }, 512,
                                      Mat { ptr: at(&a.kvb_bf, 192 * 512), ty: R_16BF, ld: 512 }, 448 * 512, true, np(&a.o),
                                      R_16BF, (hl * 256) as i64, 256, hl)?;
        }
        self.exl3(&lw.o, np(&a.o), hl * 256, t, np(&a.f32buf), false)?;
        self.allreduce(np(&a.f32buf), t * h, np(&a.a), prefill)
    }

    // ------------------------------------------------------------------ MLP
    /// A layer's MLP over `t` rows of xn -> a: dense, or the MoE's routing and experts.
    fn mlp(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        let (cfg, lw, a) = (&self.cfg, &self.w.layers[l], &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let w = self.world;
        match &lw.mlp {
            Mlp::Dense { gate_up, down } => {
                let n = gate_up.n;
                self.exl3(gate_up, np(&a.xn), h, t, np(&a.gu), true)?;
                let (g, u, ld) = if t <= GEMV_ROWS { (np(&a.gu), at(&a.gu, n), 2 * n) } else { (np(&a.gu), at(&a.gu, t * n), n) };
                ck(unsafe { glm_swiglu(st, g, ld as i64, u, ld as i64, np(&a.dact), n as i64, t as i32, n as i32) }, "swiglu")?;
                self.gather_cols(np(&a.dact), t, n, 2, np(&a.dfull), prefill)?;
                self.exl3(down, np(&a.dfull), n * w, t, np(&a.shout), false)?;
                ck(unsafe { glm_moe_out(st, std::ptr::null(), np(&a.shout), np(&a.mout), (t * down.n) as i64) }, "dense out")?;
                self.gather_cols(np(&a.mout), t, down.n, 2, np(&a.a), prefill)
            }
            Mlp::Moe { .. } => {
                // rows in sub-chunks: the routed scratch is sized for moe_rows
                let sub = if prefill { self.opt.moe_rows } else { t };
                let mut r0 = 0;
                while r0 < t {
                    let n = sub.min(t - r0);
                    self.moe_exact(l, r0, n)?;
                    self.moe_rest(l, r0, n, prefill)?;
                    r0 += n;
                }
                Ok(())
            }
        }
    }

    /// The MoE layer's router over rows [r0, r0 + n) of xn and its exact (unmasked) top-k: the
    /// demand, miss counters and hot-read candidates come from these.
    fn moe_exact(&self, l: usize, r0: usize, n: usize) -> Result<()> {
        let (cfg, a) = (&self.cfg, &self.act);
        let Mlp::Moe { router, bias, .. } = &self.w.layers[l].mlp else { return Err("moe_exact on a dense layer".into()) };
        let (e, h) = (cfg.experts, cfg.hidden);
        let allowed = self.allowed_ptr(l);
        self.linear_f32w(at(&a.xn, r0 * h), h, router, n, e, h, np(&a.logits), if n <= MAX_ROWS { 6 } else { 1 })?;
        ck(unsafe { glm_router(self.st.raw(), np(&a.logits), np(bias), std::ptr::null(), std::ptr::null(), n as i32,
                               cfg.topk as i32, cfg.routed_scaling, np(&a.eids), np(&a.ew), allowed, np(&a.hot), std::ptr::null(),
                               std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null()) }, "router (exact)")
    }

    fn allowed_ptr(&self, l: usize) -> *mut c_void {
        if self.cfg.mtp && l == self.cfg.mtp_layer() { np(&self.all_ones) } else { at(&self.allowed, l * self.cfg.experts) }
    }

    /// The MoE layer's routing over the residents (after any hot read was installed) and its
    /// experts over rows [r0, r0 + n) of xn -> the same rows of a.
    fn moe_rest(&self, l: usize, r0: usize, n: usize, prefill: bool) -> Result<()> {
        let (cfg, a) = (&self.cfg, &self.act);
        let Mlp::Moe { bias, sh_gate_up, sh_down, .. } = &self.w.layers[l].mlp else { return Err("moe on a dense layer".into()) };
        let (e, h) = (cfg.experts, cfg.hidden);
        let mtp = cfg.mtp && l == cfg.mtp_layer();
        let (lut, arena) = if mtp { (np(&self.mtp_lut), self.mtp_arena.as_ref().unwrap()) } else { (at(&self.lut, l * e), &self.arena) };
        ck(unsafe { glm_router(self.st.raw(), np(&a.logits), np(bias), self.allowed_ptr(l), lut, n as i32, cfg.topk as i32,
                               cfg.routed_scaling, np(&a.pick), np(&a.wts), std::ptr::null(), std::ptr::null_mut(), np(&a.eids),
                               np(&a.ew), at(&self.demand, l * e), np(&self.miss), std::ptr::null()) }, "router")?;
        self.moe_rows(sh_gate_up, sh_down, arena, at(&a.xn, r0 * h), n, prefill, at(&a.a, r0 * h))
    }

    /// One block of MoE rows (routing done: pick / wts): the shared expert and the routed picks,
    /// their intermediates in one gather, the summed hidden columns in another -> out bf16 [n][H].
    #[allow(clippy::too_many_arguments)]
    fn moe_rows(&self, sh_gate_up: &Exl3Mat, sh_down: &Exl3Mat, arena: &Arena, x: *mut c_void, n: usize, prefill: bool,
                out: *mut c_void) -> Result<()> {
        let (cfg, a) = (&self.cfg, &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let k = cfg.topk;
        let p = n * k;
        let il = arena.inter_l;
        let dn = arena.down_n;
        let w = self.world;
        // the shared expert's gate/up -> its activation, into the pair buffer after the routed rows
        // (decode: on the side stream, next to the routed experts)
        let decode = p <= 64;
        let (sst, x3) = if decode { (&self.st2, &a.x3s2) } else { (&self.st, &a.x3s) };
        if decode {
            self.fork()?;
        }
        self.exl3_on(sst, x3, sh_gate_up, x, h, n, np(&a.gu), true)?;
        let hsh = at(&a.pair, p * il * 2);
        let (g, u, ld) = if n <= GEMV_ROWS { (np(&a.gu), at(&a.gu, il), 2 * il) } else { (np(&a.gu), at(&a.gu, n * il), il) };
        ck(unsafe { glm_swiglu(sst.raw(), g, ld as i64, u, ld as i64, hsh, il as i64, n as i32, il as i32) }, "shared swiglu")?;
        // routed: grouping, rotation, gate/up, the down projection's input rotation
        let slots = arena.slots as i32;
        let (maxm, nexp) = if decode {
            unsafe { exl3m_group(np(&a.pick), np(&a.uids), np(&a.ucount), np(&a.members), n as i32, k as i32, slots, MAX_MEMBERS, st) };
            (MAX_MEMBERS, p as i32)
        } else {
            let nb = unsafe { ds_route_blocks(p as i32, arena.slots as i32, 64) };
            ck(unsafe { ds_route_large(st, np(&a.pick), p as i32, k as i32, arena.slots as i32, np(&a.route_work),
                                       np(&a.block_slot), np(&a.block_pair), np(&a.block_route), 64) }, "route blocks")?;
            ck(unsafe { glm_member_codes(st, np(&a.block_pair), nb * 64, k as i32, np(&a.members)) }, "member codes")?;
            ck(unsafe { glm_count_live(st, np(&a.block_slot), nb, np(&a.ucount)) }, "live blocks")?;
            (64, nb)
        };
        let uids = if decode { np(&a.uids) } else { np(&a.block_slot) };
        let up_tiled = !decode && arena.bits <= 3;
        unsafe {
            exl3m_rot_in(x, h as i32, np(&a.pick), arena.pool(3), arena.pool(4), np(&a.xh0), np(&a.xh1), n as i32, h as i32,
                         k as i32, slots, 1, st);
            if up_tiled {
                exl3m_prefill_smem_mode(np(&a.xh0), np(&a.xh1), np(&arena.t1p), np(&arena.t3p), np(&arena.k2), np(&arena.k2),
                                        uids, np(&a.ucount), np(&a.members), np(&a.zup), h as i32, il as i32, p as i32,
                                        k as i32, nexp, 2, 4, 1, 8, -4, st);
            } else {
                exl3m_grouped(np(&a.xh0), np(&a.xh1), np(&arena.t1p), np(&arena.t3p), np(&arena.k2), np(&arena.k2), uids,
                              np(&a.ucount), np(&a.members), np(&a.zup), h as i32, il as i32, p as i32, 1, maxm, k as i32,
                              nexp, 2, 8, 4, if decode { 1 } else { 2 }, 2, 10, CB_MUL1, st);
            }
            let suh2 = (arena.pool(5) as *mut u16).add(self.rank * il) as *const c_void;
            exl3m_gateup(np(&a.zup), np(&a.pick), arena.pool(6), arena.pool(7), suh2, np(&a.pair), n as i32, p as i32, il as i32,
                         1, k as i32, slots, cfg.moe_inter as i32, f32::INFINITY, 1, st);
        }
        ck(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, "routed up")?;
        if decode {
            self.join()?;
        }
        // one gather: both ranks' [routed xd | shared act]
        let per = p * il * 2 + n * il * 2;
        if w > 1 {
            self.gather(np(&a.pair), per, np(&a.gs), prefill)?;
            ck(unsafe { glm_unshard(st, np(&a.gs), (per / 2) as i64, np(&a.xdf), p as i32, il as i32, w as i32, 2) }, "unshard xd")?;
            ck(unsafe { glm_unshard(st, at(&a.gs, p * il * 2), (per / 2) as i64, np(&a.hf), n as i32, il as i32, w as i32, 2) },
               "unshard shared")?;
        } else {
            unsafe {
                tidelm_cuda::memcpy_async(np(&a.xdf), np(&a.pair), p * il * 2, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
                tidelm_cuda::memcpy_async(np(&a.hf), hsh, n * il * 2, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
            }
        }
        // routed down + the weighted combine over picks; the shared down (side stream); summed, gathered
        let kd = il * w;
        if decode {
            self.fork()?;
        }
        self.exl3_on(sst, x3, sh_down, np(&a.hf), kd, n, np(&a.shout), false)?;
        unsafe {
            if up_tiled {
                exl3m_prefill_smem_mode(np(&a.xdf), np(&a.xdf), np(&arena.t2p), np(&arena.t2p), np(&arena.k2), np(&arena.k2),
                                        uids, np(&a.ucount), np(&a.members), np(&a.zd), kd as i32, dn as i32, p as i32,
                                        k as i32, nexp, 1, 4, 1, 8, -4, st);
            } else {
                exl3m_grouped(np(&a.xdf), np(&a.xdf), np(&arena.t2p), np(&arena.t2p), np(&arena.k2), np(&arena.k2), uids,
                              np(&a.ucount), np(&a.members), np(&a.zd), kd as i32, dn as i32, p as i32, 1, maxm, k as i32,
                              nexp, 1, 8, 4, if decode { 1 } else { 2 }, 2, 10, CB_MUL1, st);
            }
            exl3m_down_combine(np(&a.zd), np(&a.pick), arena.pool(8), np(&a.wts), np(&a.local), n as i32, p as i32, dn as i32, 1,
                               k as i32, slots, st);
        }
        ck(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, "routed down")?;
        if decode {
            self.join()?;
        }
        ck(unsafe { glm_moe_out(st, np(&a.local), np(&a.shout), np(&a.mout), (n * dn) as i64) }, "moe out")?;
        self.gather_cols(np(&a.mout), n, dn, 2, out, prefill)
    }

    // ------------------------------------------------------------------ model
    /// Host -> device of a pass's token ids and first position (pinned staging, before any graph).
    fn stage(&mut self, ids: &[u32], pos0: usize, mtp: bool) -> Result<()> {
        let t = ids.len();
        for (d, &x) in self.h_ids.iter_mut().zip(ids) {
            *d = x as i32;
        }
        let pslot = if mtp { 1 } else { 0 };
        self.h_pos[pslot] = pos0 as i32;
        unsafe {
            tidelm_cuda::memcpy_async(np(&self.act.tok), self.h_ids.raw(), t * 4, tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
            let dst = if mtp { np(&self.act.mpos0) } else { np(&self.act.pos0) };
            tidelm_cuda::memcpy_async(dst, (self.h_pos.raw() as *mut i32).add(pslot) as *const c_void, 4,
                                      tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
        }
        Ok(())
    }

    /// The pass's positions from its staged first one (and a split cache's rows and bounds).
    fn positions(&self, t: usize, mtp: bool) -> Result<()> {
        let a = &self.act;
        if mtp {
            ck(unsafe { glm_positions(self.st.raw(), np(&a.mpos0), t as i32, 0, 1, 0, np(&a.mpos), std::ptr::null_mut(),
                                      std::ptr::null_mut()) }, "mtp positions")
        } else {
            ck(unsafe { glm_positions(self.st.raw(), np(&a.pos0), t as i32, self.rank as i32,
                                      if self.opt.kv_split { self.world as i32 } else { 1 }, self.s_loc as i32, np(&a.pos),
                                      np(&a.rows), np(&a.bound)) }, "positions")
        }
    }

    /// The embedding rows of the staged ids (zeroed at position 0 for the MTP layer) -> bf16
    /// at `out`, summed over the pair's vocabulary halves.
    fn embed(&self, t: usize, mtp: bool, out: *mut c_void) -> Result<()> {
        let a = &self.act;
        let h = self.cfg.hidden;
        ck(unsafe { glm_embed(self.st.raw(), np(&a.tok), t as i32, np(&self.w.embed_q), np(&self.w.embed_s),
                              (self.rank * self.vl) as i32, self.vl as i32, h as i32,
                              if mtp { np(&a.mpos) } else { std::ptr::null_mut() }, np(&a.f32buf)) }, "embed")?;
        self.allreduce(np(&a.f32buf), t * h, out, t > MAX_ROWS)
    }

    fn norm(&self, x: *mut c_void, add: bool, w: &DeviceBuffer<u16>, y: *mut c_void, t: usize) -> Result<()> {
        let h = self.cfg.hidden as i64;
        ck(unsafe { glm_rmsnorm(self.st.raw(), x, h, if add { np(&self.act.a) } else { std::ptr::null() }, h, np(w), y, h,
                                t as i32, h as i32, self.cfg.rms_eps) }, "norm")
    }

    /// Layer l's input norm (adding the previous MLP's output past layer 0), attention, the
    /// residual add and the post-attention norm.
    fn layer_pre(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        let lw = &self.w.layers[l];
        let a = &self.act;
        self.norm(np(&a.h), l > 0, &lw.ln_in, np(&a.xn), t)?;
        self.attention(l, t, prefill)?;
        self.norm(np(&a.h), true, &lw.ln_post, np(&a.xn), t)
    }

    /// The decoder layers over `t` rows (staged) at self.pos: the final norm into hn.
    fn layers(&self, t: usize) -> Result<()> {
        let prefill = t > MAX_ROWS;
        self.positions(t, false)?;
        self.embed(t, false, np(&self.act.h))?;
        for l in 0..self.cfg.layers {
            self.layer_pre(l, t, prefill)?;
            self.mlp(l, t, prefill)?;
        }
        self.norm(np(&self.act.h), true, &self.w.norm, np(&self.act.hn), t)
    }

    /// A decode pass in pieces split where hot reads go: segment 0 runs through the first MoE
    /// layer's exact routing, segment i + 1 from MoE layer i's resident routing through the
    /// next one's exact routing, the last through the head's argmax (device only).
    fn decode_segment(&self, seg: usize, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        let moe = &self.moe_layers;
        let start = if seg == 0 {
            self.positions(t, false)?;
            self.embed(t, false, np(&self.act.h))?;
            0
        } else {
            let l = moe[seg - 1];
            self.moe_rest(l, 0, t, false)?;
            l + 1
        };
        let end = if seg < moe.len() { moe[seg] } else { cfg.layers };
        for l in start..end {
            self.layer_pre(l, t, false)?;
            self.mlp(l, t, false)?;
        }
        if seg < moe.len() {
            self.layer_pre(end, t, false)?;
            self.moe_exact(end, 0, t)?;
            if let Some(h) = self.hot.as_ref() {
                unsafe {
                    tidelm_cuda::memcpy_async(h.host.raw(), np(&self.act.hot), t * self.cfg.topk * 8,
                                              tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST, &self.st)?;
                }
            }
            return Ok(());
        }
        self.norm(np(&self.act.h), true, &self.w.norm, np(&self.act.hn), t)?;
        self.head_dev(np(&self.act.hn), t)
    }

    /// The head over `t` rows of `x` (bf16 [t][H]) and each row's argmax into out_tok.
    fn head_dev(&self, x: *const c_void, t: usize) -> Result<()> {
        let a = &self.act;
        let st = self.st.raw();
        self.exl3(&self.w.head, x, self.cfg.hidden, t, np(&a.head), false)?;
        ck(unsafe { glm_argmax(st, np(&a.head), self.vl as i64, t as i32, self.vl as i32, (self.rank * self.vl) as i32,
                               np(&a.pairs)) }, "argmax")?;
        self.gather(np(&a.pairs), t * 8, at(&a.pairs, 2 * VERIFY_ROWS * 2), false)?;
        ck(unsafe { glm_argmax_combine(st, at(&a.pairs, 2 * VERIFY_ROWS * 2), self.world as i32, t as i32, np(&a.out_tok)) },
           "argmax combine")?;
        if self.sampled {
            let c = np(&self.cand);
            ck(unsafe { glm_row_lse(st, np(&a.head), self.vl as i64, t as i32, self.vl as i32, np(&self.temp), c, CROW as i64) },
               "row lse")?;
            ck(unsafe { glm_topk(st, np(&a.head), self.vl as i64, std::ptr::null(), 1, 0, self.vl as i32, CAND as i32, t as i32,
                                 at(&self.cand, 2 + CAND), CROW as i64, at(&self.cand, 2), CROW as i64) }, "candidates")?;
            let nbytes = t * CROW * 4;
            self.gather(c, nbytes, np(&a.gs), false)?;
            unsafe {
                tidelm_cuda::memcpy_async(self.h_cand.raw(), np(&a.gs), self.world * nbytes, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST,
                                          &self.st)?;
            }
        }
        Ok(())
    }

    /// Row `row` of the last sampled pass: its candidates over the pair's vocabulary.
    pub fn candidates(&self, row: usize) -> RowCands {
        let t = self.cand_rows;
        let mut rc = RowCands::default();
        for w in 0..self.world {
            let base = (w * t + row) * CROW;
            let r = &self.h_cand[base..base + CROW];
            rc.lse.push((r[0], r[1]));
            for i in 0..CAND {
                let id = r[2 + CAND + i].to_bits() as i32;
                if id >= 0 {
                    rc.vals.push(r[2 + i]);
                    rc.ids.push((id as usize + w * self.vl) as u32);
                }
            }
            rc.min.push(r[2..2 + CAND].iter().cloned().fold(f32::INFINITY, f32::min));
        }
        rc
    }

    /// Sampling on or off for the passes that follow, at this temperature (both ranks: the
    /// candidates are a gather).
    pub fn set_sampling(&mut self, temp: Option<f32>) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_SAMPLING).u32(temp.map(|t| t.to_bits()).unwrap_or(0)).u32(temp.is_some() as u32);
        })?;
        self.sampling_local(temp)
    }

    fn sampling_local(&mut self, temp: Option<f32>) -> Result<()> {
        self.sampled = temp.is_some();
        if let Some(t) = temp {
            self.h_temp[0] = t;
            unsafe {
                tidelm_cuda::memcpy_async(np(&self.temp), self.h_temp.raw(), 4, tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
            }
            self.st.synchronize()?;
        }
        Ok(())
    }

    /// Row `row` of the last pass's logits over the whole vocabulary (the nucleus fallback).
    pub fn full_row(&mut self, row: usize) -> Result<Vec<f32>> {
        self.mirror(|m| {
            m.u32(OP_FULL_ROW).u32(row as u32);
        })?;
        self.full_row_local(row)
    }

    fn full_row_local(&mut self, row: usize) -> Result<Vec<f32>> {
        let vl = self.vl;
        self.gather(at(&self.act.head, row * vl), vl * 4, np(&self.act.gs), false)?;
        let mut out = vec![0f32; vl * self.world];
        self.act.gs.download_async(0, unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) },
                                   &self.st)?;
        self.st.synchronize()?;
        Ok(out)
    }

    /// out_tok's first `t` rows to the host (synchronizes the stream).
    fn read_tokens(&mut self, t: usize) -> Result<Vec<u32>> {
        unsafe {
            tidelm_cuda::memcpy_async(self.h_tok.raw(), np(&self.act.out_tok), t * 4, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST,
                                      &self.st)?;
        }
        self.st.synchronize()?;
        self.cand_rows = t;
        Ok(self.h_tok[..t].iter().map(|&x| x as u32).collect())
    }

    /// The MTP layer over `t` staged rows (token i + 1 at position pos0 + i) and the previous
    /// hidden `hsrc` -> its post-norm hidden in mhn and, with `logits`, each row's argmax in
    /// out_tok (device only).
    fn mtp_dev(&self, t: usize, hsrc: Hidden, logits: bool) -> Result<()> {
        let l = self.cfg.mtp_layer();
        let (cfg, a) = (&self.cfg, &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let lw = &self.w.layers[l];
        let mx = lw.mtp.as_ref().ok_or("no MTP layer")?;
        let prefill = t > MAX_ROWS;
        self.positions(t, true)?;
        // e = embed(next ids), zero at position 0; x = eh_proj(cat(enorm(e), hnorm(h_prev)))
        self.embed(t, true, np(&a.xn))?;
        ck(unsafe { glm_rmsnorm(st, np(&a.xn), h as i64, std::ptr::null(), 0, np(&mx.enorm), np(&a.ehin), (2 * h) as i64, t as i32,
                                h as i32, cfg.rms_eps) }, "enorm")?;
        let hp = match hsrc {
            Hidden::Last => np(&a.hlast),
            Hidden::Mtp => np(&a.mhn),
            Hidden::Target(from) => at(&a.hn, from * h),
        };
        ck(unsafe { glm_rmsnorm(st, hp, h as i64, std::ptr::null(), 0, np(&mx.hnorm), at(&a.ehin, h), (2 * h) as i64, t as i32,
                                h as i32, cfg.rms_eps) }, "hnorm")?;
        self.exl3(&mx.eh, np(&a.ehin), 2 * h, t, np(&a.ehout), true)?;
        self.gather_cols(np(&a.ehout), t, mx.eh.n, 2, np(&a.h), prefill)?;
        self.norm(np(&a.h), false, &lw.ln_in, np(&a.xn), t)?;
        self.attention(l, t, prefill)?;
        self.norm(np(&a.h), true, &lw.ln_post, np(&a.xn), t)?;
        self.mlp(l, t, prefill)?;
        self.norm(np(&a.h), true, &mx.sh_norm, np(&a.mhn), t)?;
        if logits {
            self.head_dev(np(&a.mhn), t)?;
        }
        Ok(())
    }

    fn mtp_core(&mut self, ids: &[u32], hsrc: Hidden, pos0: usize, logits: bool) -> Result<Option<Vec<u32>>> {
        let t = ids.len();
        self.mtp_pos = pos0;
        self.stage(ids, pos0, true)?;
        if t <= VERIFY_ROWS && self.graphs {
            let key = GraphKey::Mtp { t, logits, hidden: hidden_code(hsrc), sampled: self.sampled };
            self.replay(key, |e| e.mtp_dev(t, hsrc, logits))?;
        } else {
            self.mtp_dev(t, hsrc, logits)?;
        }
        if !logits {
            return Ok(None);
        }
        Ok(Some(self.read_tokens(t)?))
    }

    /// Replay the graph for `key`, capturing it on first use: that first call runs `work`
    /// eagerly (its results are the pass's), then records it for the next.
    fn replay(&mut self, key: GraphKey, work: impl Fn(&Engine) -> Result<()>) -> Result<()> {
        if let Some(g) = self.graphs_map.get(&key) {
            return g.launch(&self.st).map_err(Into::into);
        }
        work(self)?;
        self.st.begin_capture()?;
        let recorded = work(self);
        let g = self.st.end_capture();
        recorded?;
        self.graphs_map.insert(key, g?);
        Ok(())
    }
}

/// A sampled row's candidates from both ranks: each rank's (max, sum exp((z - max) / T)), its
/// top CAND logits (ids global) and the smallest of them.
#[derive(Clone, Debug, Default)]
pub struct RowCands {
    pub lse: Vec<(f32, f32)>,
    pub vals: Vec<f32>,
    pub ids: Vec<u32>,
    pub min: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum GraphKey {
    /// decoder rows t, the whole pass
    Target { t: usize, sampled: bool },
    /// decoder rows t, one segment between hot-read points
    Segment { t: usize, seg: usize, sampled: bool },
    Mtp { t: usize, logits: bool, hidden: u8, sampled: bool },
}

fn hidden_code(h: Hidden) -> u8 {
    match h {
        Hidden::Last => 0,
        Hidden::Mtp => 1,
        Hidden::Target(f) => 2 + f as u8,
    }
}

/// A decode row's bytes in the split gather: its heads' absorbed queries (bf16 [hl][576]) then
/// its top-2048 index scores (fp32).
fn qrow_bytes(hl: usize) -> usize {
    hl * 576 * 2 + TOPK * 4
}
/// A decode row's partials for the other rank's heads: fp16 [hl][512] then fp32 lse [hl].
fn row2_bytes(hl: usize) -> usize {
    hl * 512 * 2 + hl * 4
}

/// Flash-decoding splits for `heads` heads over `k` picks at `t` rows: (splits, picks a split, a
/// multiple of 32). ~128 programs a launch, where sparse_attn.sparse_mla's ~32 left a verify
/// step's rows walking 32 tiles each one after another (132 us a layer, nsys); at most 16
/// splits, so the fp32 partials stay a few MB.
fn decode_split(k: usize, heads: usize, t: usize) -> (usize, usize) {
    let hb = heads.div_ceil(32);
    let want = 128usize.div_ceil(hb * t).next_power_of_two().clamp(1, 16);
    let chunk = k.div_ceil(want).div_ceil(32) * 32;
    (k.div_ceil(chunk), chunk)
}

impl Act {
    fn new(cfg: &GlmConfig, opt: &Options, hl: usize, vl: usize, s_loc: usize, slots: usize) -> Result<Self> {
        let r = opt.chunk.max(MAX_ROWS);
        let d = MAX_ROWS;
        let h = cfg.hidden;
        let il = cfg.moe_inter / (cfg.heads / hl);
        let world = cfg.heads / hl;
        let mr = opt.moe_rows.min(r);
        let p = mr * cfg.topk;
        let pd = p.max(VERIFY_ROWS * cfg.topk);
        let nb = (unsafe { ds_route_blocks(p as i32, slots as i32, 64) }).max(1) as usize;
        let ms = opt.max_seq;
        let z = |n: usize| -> Result<DeviceBuffer<u16>> { Ok(DeviceBuffer::new(n)?) };
        let pair_bytes = pd * il * 2 + mr * il * 2;
        let gs_bytes = [
            2 * r * h * 2,                  // a gathered bf16 [rows][H] (dense layer intermediates, MLP outputs)
            2 * pair_bytes,                 // the MoE pair
            2 * d * qrow_bytes(hl),         // split decode queries
            2 * d * h * 4,                  // decode all-reduce
            2 * r * h * 2,
        ]
        .into_iter()
        .max()
        .unwrap();
        let max_k = (2 * h).max(cfg.inter).max(cfg.heads / world * 256);
        let max_kn = [cfg.inter * h / world, 2 * h * h / world, hl * 256 * h, h * cfg.heads / world * 256, cfg.q_lora * 4096]
            .into_iter()
            .max()
            .unwrap();
        let max_zn = (r * (hl * 256).max(4096).max(h)).max(8 * d * 2 * 6144);
        Ok(Act {
            tok: DeviceBuffer::zeroed(r)?,
            pos0: DeviceBuffer::zeroed(1)?,
            mpos0: DeviceBuffer::zeroed(1)?,
            pos: DeviceBuffer::zeroed(r)?,
            rows: DeviceBuffer::zeroed(r)?,
            bound: DeviceBuffer::zeroed(r)?,
            mpos: DeviceBuffer::zeroed(r)?,
            h: z(r * h)?,
            xn: z(r * h)?,
            a: z(r * h)?,
            f32buf: DeviceBuffer::new(r * h)?,
            gs: DeviceBuffer::new(gs_bytes)?,
            qa: z(r * cfg.q_lora)?,
            qr: z(r * cfg.q_lora)?,
            ckv: z(r * 640)?,
            c: z(r * 512)?,
            kr: z(r * 64)?,
            qb: z(r * hl * 256)?,
            qf: z(r * hl * 576)?,
            qsend: DeviceBuffer::zeroed(d * qrow_bytes(hl))?,
            qi: z(r * 4096)?,
            ki: z(r * 128)?,
            iw: DeviceBuffer::new(r * 32)?,
            scores: DeviceBuffer::new(SCORE_VALUES.max(d * ms).max(d * s_loc))?,
            idx: DeviceBuffer::new(r * TOPK)?,
            olat: z(r * hl * 512)?,
            o: z(r * hl * 256)?,
            pm: DeviceBuffer::new(d * cfg.heads * 16)?,
            pl: DeviceBuffer::new(d * cfg.heads * 16)?,
            pacc: DeviceBuffer::new(d * cfg.heads * 16 * 512)?,
            own_o: DeviceBuffer::new(d * hl * 512)?,
            own_l: DeviceBuffer::new(d * hl)?,
            send2: DeviceBuffer::new(d * row2_bytes(hl))?,
            gcq: DeviceBuffer::new(world * s_loc * 512 + 512)?,
            gcs: DeviceBuffer::new(world * s_loc * 4 + 4)?,
            gcr: z(world * s_loc * 64 + 64)?,
            giq: DeviceBuffer::new(world * s_loc * 128 + 128)?,
            gis: DeviceBuffer::new(world * s_loc + 1)?,
            kvb_bf: z(hl * 448 * 512)?,
            split_part: DeviceBuffer::new(8 * d * cfg.experts.max(4096))?,
            logits: DeviceBuffer::new(mr.max(d) * cfg.experts)?,
            eids: DeviceBuffer::new(mr.max(d) * cfg.topk)?,
            ew: DeviceBuffer::new(mr.max(d) * cfg.topk)?,
            pick: DeviceBuffer::new(mr.max(d) * cfg.topk)?,
            wts: DeviceBuffer::new(mr.max(d) * cfg.topk)?,
            hot: DeviceBuffer::new(d * cfg.topk * 2)?,
            gu: z(r * 2 * (cfg.inter / world).max(il))?,
            pair: DeviceBuffer::new(pair_bytes)?,
            xdf: z(pd * il * world)?,
            hf: z(mr.max(d) * il * world)?,
            xh0: z(pd * h)?,
            xh1: z(pd * h)?,
            zup: DeviceBuffer::new(2 * pd * il)?,
            zd: DeviceBuffer::new(pd * h / world)?,
            local: DeviceBuffer::new(mr.max(d) * h / world)?,
            shout: DeviceBuffer::new(r * h / world)?,
            mout: z(r * h / world)?,
            dact: z(r * cfg.inter / world)?,
            dfull: z(r * cfg.inter)?,
            uids: DeviceBuffer::new(64)?,
            ucount: DeviceBuffer::new(1)?,
            members: DeviceBuffer::new((nb * 64).max(64 * 16))?,
            route_work: DeviceBuffer::new(3 * slots + 64)?,
            block_slot: DeviceBuffer::new(nb)?,
            block_pair: DeviceBuffer::new(nb * 64)?,
            block_route: DeviceBuffer::new(nb * 64)?,
            head: DeviceBuffer::new(d * vl)?,
            pairs: DeviceBuffer::new(2 * VERIFY_ROWS * 2 * (1 + world))?,
            out_tok: DeviceBuffer::new(d)?,
            hn: z(r * h)?,
            mhn: z(r * h)?,
            hlast: z(h)?,
            ehin: z(r * 2 * h)?,
            ehout: z(r * h / world)?,
            x3s: Exl3Scratch::new(r, max_k, max_kn, max_zn)?,
            x3s2: Exl3Scratch::new(GEMV_ROWS, max_k, 8, 8 * 2 * MAX_ROWS * 6144)?,
            split_part2: DeviceBuffer::new(24 * d * 128)?,
        })
    }
}

/// Load records (slot, layer, expert) into an arena on reader threads, each with its own
/// O_DIRECT buffer and stream.
pub fn load_slots(pack: &Pack, arena: &Arena, items: &[(usize, usize, usize)]) -> Result<()> {
    let threads = 12usize.min(items.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let arena_ptr = arena as *const Arena as usize;
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let run = || -> Result<()> {
                    let arena = unsafe { &*(arena_ptr as *const Arena) };
                    tidelm_cuda::set_device(0)?;
                    let st = Stream::new()?;
                    let mut buf = pack.buffer()?;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= items.len() {
                            return Ok(());
                        }
                        let (slot, l, e) = items[i];
                        let r = pack.read(l, e, &mut buf)?;
                        if r.bits != arena.bits {
                            return Err(format!("expert ({l}, {e}) is {} bits, the arena {}", r.bits, arena.bits).into());
                        }
                        arena.load(slot, &buf[..r.nbytes], &st)?;
                        st.synchronize()?;
                    }
                };
                if let Err(e) = run() {
                    *err.lock().unwrap() = Some(e.to_string());
                }
            });
        }
    });
    match err.into_inner().unwrap() {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

// ------------------------------------------------------------------ the engine's calls
/// What a forward returns: each row's argmax (decode), or the last prompt row's.
impl Engine {
    pub(crate) fn mirror_msg(&mut self, msg: impl FnOnce(&mut tidelm_comm::Msg)) -> Result<()> {
        self.mirror(msg)
    }

    fn mirror(&mut self, msg: impl FnOnce(&mut tidelm_comm::Msg)) -> Result<()> {
        if self.rank == 0 {
            if let Some(p) = self.pair.as_mut() {
                let mut m = tidelm_comm::Msg::default();
                msg(&mut m);
                p.ctl.send(&m.0)?;
            }
        }
        Ok(())
    }

    /// Run `ids` at the sequence's end: <= 16 rows the decode path (each row's argmax back),
    /// more in prefill chunks (the last row's). With `mtp_fill`, the MTP layer's cache follows
    /// every row whose next token is in `ids` (the last row waits for the sampled token).
    pub fn forward(&mut self, ids: &[u32], mtp_fill: bool) -> Result<Vec<u32>> {
        self.mirror(|m| {
            m.u32(OP_FORWARD).u32(mtp_fill as u32).u32s(ids);
        })?;
        self.forward_local(ids, mtp_fill)
    }

    fn forward_local(&mut self, ids: &[u32], mtp_fill: bool) -> Result<Vec<u32>> {
        let t = ids.len();
        if t == 0 {
            return Err("forward of no rows".into());
        }
        if self.pos + t > self.opt.max_seq {
            return Err(format!("context {} exceeds max_seq {}", self.pos + t, self.opt.max_seq).into());
        }
        let h = self.cfg.hidden;
        let mtp = mtp_fill && self.cfg.mtp;
        if t <= MAX_ROWS {
            self.stage(ids, self.pos, false)?;
            if t <= VERIFY_ROWS && self.graphs {
                if self.hot.is_some() {
                    self.hot_begin();
                    let n = self.moe_layers.len();
                    for seg in 0..=n {
                        self.replay(GraphKey::Segment { t, seg, sampled: self.sampled }, |e| e.decode_segment(seg, t))?;
                        if seg < n {
                            self.hot_point(self.moe_layers[seg], t, seg, n)?;
                        }
                    }
                    self.hot_release()?;
                } else {
                    let nseg = self.moe_layers.len() + 1;
                    self.replay(GraphKey::Target { t, sampled: self.sampled }, |e| (0..nseg).try_for_each(|seg| e.decode_segment(seg, t)))?;
                }
            } else {
                self.layers(t)?;
                self.head_dev(np(&self.act.hn), t)?;
            }
            self.last_rows = t;
            let out = self.read_tokens(t)?;
            let p0 = self.pos;
            self.tokens.extend_from_slice(ids);
            self.pos += t;
            if mtp && t > 1 {
                self.mtp_core(&ids[1..], Hidden::Target(0), p0, false)?;
            }
            return Ok(out);
        }
        let mut out = vec![];
        let mut c0 = 0;
        while c0 < t {
            let n = self.opt.chunk.min(t - c0);
            let part = &ids[c0..c0 + n];
            self.stage(part, self.pos, false)?;
            self.layers(n)?;
            self.last_rows = n;
            if c0 + n == t {
                self.head_dev(at(&self.act.hn, (n - 1) * h), 1)?;
                out = self.read_tokens(1)?;
            }
            let p0 = self.pos;
            self.tokens.extend_from_slice(part);
            self.pos += n;
            if mtp {
                let next = &ids[c0 + 1..(c0 + n + 1).min(t)];
                if !next.is_empty() {
                    self.mtp_core(next, Hidden::Target(0), p0, false)?;
                }
            }
            c0 += n;
        }
        self.st.synchronize()?;
        Ok(out)
    }

    /// The MTP layer over `ids` at positions pos0.. with the previous hidden from `h`: each
    /// row's argmax with `logits`, else only its cache rows.
    pub fn mtp(&mut self, ids: &[u32], h: Hidden, pos0: usize, logits: bool) -> Result<Option<Vec<u32>>> {
        self.mirror(|m| {
            let (kind, from) = match h {
                Hidden::Last => (0, 0),
                Hidden::Mtp => (1, 0),
                Hidden::Target(f) => (2, f as u32),
            };
            m.u32(OP_MTP).u32(kind).u32(from).u32(pos0 as u32).u32(logits as u32).u32s(ids);
        })?;
        let r = self.mtp_core(ids, h, pos0, logits)?;
        if !logits {
            self.st.synchronize()?;
        }
        Ok(r)
    }

    /// Keep row `row` of the last forward's post-norm hidden as the next MTP call's input.
    pub fn keep_hidden(&mut self, row: usize) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_KEEP_HIDDEN).u32(row as u32);
        })?;
        let h = self.cfg.hidden;
        self.act.hlast.copy_from(0, &self.act.hn, row * h, h, &self.st)?;
        Ok(())
    }

    /// Truncate the sequence to `n` positions (rejected drafts; a new prompt's reuse).
    pub fn set_pos(&mut self, n: usize) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_SET_POS).u32(n as u32);
        })?;
        if n > self.pos {
            return Err(format!("set_pos {n} past the sequence's {}", self.pos).into());
        }
        self.pos = n;
        self.tokens.truncate(n);
        Ok(())
    }

    /// Forget the demand and miss counters so far (after a warm-up) on both ranks.
    pub fn zero_demand(&mut self) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_ZERO_DEMAND);
        })?;
        self.demand.zero_async(&self.st)?;
        self.miss.zero_async(&self.st)?;
        Ok(())
    }

    /// Rank 0: release rank 1's follow loop.
    pub fn stop_follower(&mut self) -> Result<()> {
        self.mirror(|m| {
            m.u32(OP_EXIT);
        })
    }

    /// Rank 1: run rank 0's calls until it exits.
    pub fn follow(&mut self) -> Result<()> {
        loop {
            let msg = self.pair.as_mut().ok_or("follow needs a pair")?.ctl.recv()?;
            let mut r = tidelm_comm::Reader::new(&msg);
            match r.u32()? {
                OP_FORWARD => {
                    let fill = r.u32()? != 0;
                    let ids = r.u32s()?;
                    self.forward_local(&ids, fill)?;
                }
                OP_MTP => {
                    let (kind, from, pos0, logits) = (r.u32()?, r.u32()? as usize, r.u32()? as usize, r.u32()? != 0);
                    let ids = r.u32s()?;
                    let h = match kind {
                        0 => Hidden::Last,
                        1 => Hidden::Mtp,
                        _ => Hidden::Target(from),
                    };
                    self.mtp_core(&ids, h, pos0, logits)?;
                    self.st.synchronize()?;
                }
                OP_KEEP_HIDDEN => {
                    let row = r.u32()? as usize;
                    let h = self.cfg.hidden;
                    self.act.hlast.copy_from(0, &self.act.hn, row * h, h, &self.st)?;
                }
                OP_SET_POS => {
                    let n = r.u32()? as usize;
                    self.pos = n.min(self.pos);
                    self.tokens.truncate(self.pos);
                }
                OP_SAMPLING => {
                    let (bits, on) = (r.u32()?, r.u32()? != 0);
                    self.sampling_local(if on { Some(f32::from_bits(bits)) } else { None })?;
                }
                OP_FULL_ROW => {
                    let row = r.u32()? as usize;
                    self.full_row_local(row)?;
                }
                crate::adapt::OP_ADAPT => self.follow_adapt(&mut r)?,
                op @ (crate::prefix::OP_DISK_SAVE | crate::prefix::OP_DISK_LOAD) => self.follow_disk(op, &mut r)?,
                crate::adapt::OP_BG => self.follow_bg(&mut r)?,
                OP_ZERO_DEMAND => {
                    self.demand.zero_async(&self.st)?;
                    self.miss.zero_async(&self.st)?;
                }
                OP_EXIT => return Ok(()),
                op => return Err(format!("rank 1: unknown op {op}").into()),
            }
        }
    }
}
