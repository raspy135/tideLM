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
const OP_EXIT: u32 = 99;

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
}

impl Options {
    /// Everything that decides numerics or control flow, for the pair's boot comparison.
    pub fn boot_config(&self, model_dir: &Path, pack: &Pack) -> Result<Vec<u8>> {
        use sha2::Digest;
        let cfg = std::fs::read(model_dir.join("config.json"))?;
        let exe = std::fs::read(std::env::current_exe()?)?;
        let s = format!(
            "glm-boot v1\nconfig {:x}\nexe {:x}\nmax_seq {}\nchunk {}\nslots {}\nkv_split {}\nreduce_bf16 {}\nmoe_rows {}\n\
             pack_source {}\nenv {}\n",
            sha2::Sha256::digest(&cfg), sha2::Sha256::digest(&exe), self.max_seq, self.chunk, self.slots, self.kv_split,
            self.prefill_reduce_bf16, self.moe_rows, pack.source_sha256,
            ["TIDELM_GLM_HOT", "NCCL_PROTO"].iter().map(|k| format!("{k}={}", std::env::var(k).unwrap_or_default()))
                .collect::<Vec<_>>().join(" "));
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

struct LayerKv {
    /// latent FP8 [rows][512] + fp32 [rows][4]
    cq: DeviceBuffer<u8>,
    cs: DeviceBuffer<f32>,
    /// RoPE key BF16 [rows][64]
    cr: DeviceBuffer<u16>,
    /// indexer key FP8 [rows][128] + fp32 [rows] (full layers)
    iq: Option<DeviceBuffer<u8>>,
    is: Option<DeviceBuffer<f32>>,
    /// split by position across the pair (decoder layers under kv_split)
    split: bool,
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
    st: Stream,
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
    kv: Vec<LayerKv>,
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
        };
        engine.st.synchronize()?;
        Ok(engine)
    }

    pub fn stream(&self) -> &Stream {
        &self.st
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
        if rows <= MAX_ROWS {
            m.decode(&self.st, &self.members, &self.act.x3s, x, ldx, rows, out, out_bf16)
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
        let part = np(&self.act.split_part);
        ck(unsafe { ds_linear_f32w(self.st.raw(), x, ldx as i64, np(w), rows as i32, n as i32, k as i32, splits as i32,
                                   if splits > 1 { part } else { y }, n as i64, part) }, "linear f32w")?;
        if splits > 1 {
            ck(unsafe { ds_splits_reduce(self.st.raw(), part, splits as i32, rows as i32, n as i32, std::ptr::null_mut(), y,
                                         n as i64) }, "splits reduce")?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ attention
    #[allow(clippy::too_many_lines)]
    fn attention(&self, l: usize, t: usize, prefill: bool, idx_valid: &mut bool) -> Result<()> {
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
        // q_a, its norm; kv_a: the latent (normed, FP8 cache row) and the RoPE key
        self.exl3(&lw.q_a, np(&a.xn), h, t, np(&a.qa), true)?;
        ck(unsafe { glm_rmsnorm(st, np(&a.qa), cfg.q_lora as i64, std::ptr::null(), 0, np(&lw.q_norm), np(&a.qr),
                                cfg.q_lora as i64, t as i32, cfg.q_lora as i32, 1e-6) }, "q norm")?;
        self.exl3(&lw.kv_a, np(&a.xn), h, t, np(&a.ckv), true)?;
        let ldk = lw.kv_a.n;
        ck(unsafe { glm_rmsnorm(st, np(&a.ckv), ldk as i64, std::ptr::null(), 0, np(&lw.kv_norm), np(&a.c), 512, t as i32, 512,
                                1e-6) }, "kv norm")?;
        ck(unsafe { glm_kv_fp8(st, np(&a.c), 512, rowsb, np(&kv.cq), np(&kv.cs), t as i32, 512) }, "latent cache")?;
        ck(unsafe { glm_rope(st, at(&a.ckv, 512), ldk as i64, 64, np(&a.kr), 64, 64, t as i32, 1, np(&self.w.cos), np(&self.w.sin),
                             posb) }, "rope key")?;
        ck(unsafe { glm_rows_copy(st, np(&a.kr), 64, rowsb, np(&kv.cr), t as i32, 64) }, "rope cache")?;
        // q_b (this rank's heads), RoPE on its last 64 dims, the absorbed latent query
        self.exl3(&lw.q_b, np(&a.qr), cfg.q_lora, t, np(&a.qb), true)?;
        let decode = !prefill;
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
        // the indexer on full layers: queries, this step's keys, head weights; scores and top-k
        let s_all = p0 + t; // positions [0, S) once this step's rows are written
        let dense_causal = prefill && s_all <= TOPK;
        if full {
            let ix = lw.index.as_ref().unwrap();
            self.exl3(&ix.wq_b, np(&a.qr), cfg.q_lora, t, np(&a.qi), true)?;
            ck(unsafe { glm_rope(st, np(&a.qi), 4096, 128, np(&a.qi), 4096, 128, t as i32, 32, np(&self.w.cos), np(&self.w.sin),
                                 posb) }, "rope index q")?;
            if prefill {
                self.blas.gemm_rm(t, 128, h, Mat { ptr: np(&a.xn), ty: R_16BF, ld: h as i64 }, Mat { ptr: np(&ix.wk), ty: R_16BF, ld: h as i64 },
                                  true, np(&a.ki), R_16BF, 128)?;
            } else {
                ck(unsafe { ds_linear_bf16(st, np(&a.xn), h as i64, np(&ix.wk), t as i32, 128, h as i32, 1, np(&a.ki),
                                           std::ptr::null_mut(), 128, std::ptr::null_mut()) }, "index wk")?;
            }
            ck(unsafe { glm_index_key(st, np(&a.ki), 128, np(&ix.kn_w), np(&ix.kn_b), np(&self.w.cos), np(&self.w.sin), posb,
                                      rowsb, np(kv.iq.as_ref().unwrap()), np(kv.is.as_ref().unwrap()), t as i32) }, "index key")?;
            self.linear_f32w(np(&a.xn), h, &ix.wp, t, 32, h, np(&a.iw), if prefill { 1 } else { 4 })?;
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
                *idx_valid = true;
            }
            if !*idx_valid {
                return Err(format!("layer {l}: a shared indexer layer before any full one").into());
            }
            let hh = cfg.heads;
            let (ns, chunk) = decode_split(TOPK, hh, t);
            ck(unsafe { glm_mla(st, 2, np(&a.gs), (nbytes / 2) as i64, (self.qrow / 2) as i64, 576, hl as i32, np(&kv.cq), np(&kv.cs),
                                np(&kv.cr), 0, np(&a.idx), TOPK as i64, TOPK as i32, bound, t as i32, hh as i32, scale,
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
                *idx_valid = true;
            }
            let (ns, chunk) = decode_split(TOPK, hl, t);
            ck(unsafe { glm_mla(st, 2, qbase, 0, qrow as i64, 576, hl as i32, np(&kv.cq), np(&kv.cs), np(&kv.cr), 0,
                                np(&a.idx), TOPK as i64, TOPK as i32, bound, t as i32, hl as i32, scale, ns as i32, chunk as i32,
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
            if full {
                *idx_valid = true;
            }
            let (idx, k) = if dense_causal { (std::ptr::null_mut(), s_all) } else { (np(&a.idx), TOPK) };
            if !dense_causal && !*idx_valid {
                return Err(format!("layer {l}: a shared indexer layer before any full one").into());
            }
            let posp = posb;
            let chunk = k.div_ceil(32) * 32;
            ck(unsafe { glm_mla(st, 2, qbase, 0, qrow as i64, 576, hl as i32, cq, cs, cr, split_n as i32, idx, TOPK as i64,
                                k as i32, posp, t as i32, hl as i32, scale, 1, chunk as i32, std::ptr::null_mut(),
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
    fn mlp(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        let (cfg, lw, a) = (&self.cfg, &self.w.layers[l], &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let w = self.world;
        match &lw.mlp {
            Mlp::Dense { gate_up, down } => {
                let n = gate_up.n;
                self.exl3(gate_up, np(&a.xn), h, t, np(&a.gu), true)?;
                let (g, u, ld) = if t <= MAX_ROWS { (np(&a.gu), at(&a.gu, n), 2 * n) } else { (np(&a.gu), at(&a.gu, t * n), n) };
                ck(unsafe { glm_swiglu(st, g, ld as i64, u, ld as i64, np(&a.dact), n as i64, t as i32, n as i32) }, "swiglu")?;
                self.gather_cols(np(&a.dact), t, n, 2, np(&a.dfull), prefill)?;
                self.exl3(down, np(&a.dfull), n * w, t, np(&a.shout), false)?;
                ck(unsafe { glm_moe_out(st, std::ptr::null(), np(&a.shout), np(&a.mout), (t * down.n) as i64) }, "dense out")?;
                self.gather_cols(np(&a.mout), t, down.n, 2, np(&a.a), prefill)
            }
            Mlp::Moe { router, bias, sh_gate_up, sh_down } => {
                let mtp = cfg.mtp && l == cfg.mtp_layer();
                let e = cfg.experts;
                let k = cfg.topk;
                let (lut, allowed, arena) = if mtp {
                    (np(&self.mtp_lut), np(&self.all_ones), self.mtp_arena.as_ref().unwrap())
                } else {
                    (at(&self.lut, l * e), at(&self.allowed, l * e), &self.arena)
                };
                // rows in sub-chunks: the routed scratch is sized for moe_rows
                let sub = if prefill { self.opt.moe_rows } else { t };
                let mut r0 = 0;
                while r0 < t {
                    let n = sub.min(t - r0);
                    let xr = at(&a.xn, r0 * h);
                    self.linear_f32w(xr, h, router, n, e, h, np(&a.logits), if n <= MAX_ROWS { 6 } else { 1 })?;
                    ck(unsafe { glm_router(st, np(&a.logits), np(bias), std::ptr::null(), std::ptr::null(), n as i32, k as i32,
                                           cfg.routed_scaling, np(&a.eids), np(&a.ew), allowed, np(&a.hot), std::ptr::null(),
                                           std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null()) },
                       "router (exact)")?;
                    ck(unsafe { glm_router(st, np(&a.logits), np(bias), allowed, lut, n as i32, k as i32, cfg.routed_scaling,
                                           np(&a.pick), np(&a.wts), std::ptr::null(), std::ptr::null_mut(), np(&a.eids),
                                           np(&a.ew), at(&self.demand, l * e), np(&self.miss), std::ptr::null()) }, "router")?;
                    self.moe_rows(sh_gate_up, sh_down, arena, xr, n, prefill, at(&a.a, r0 * h))?;
                    r0 += n;
                }
                Ok(())
            }
        }
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
        self.exl3(sh_gate_up, x, h, n, np(&a.gu), true)?;
        let hsh = at(&a.pair, p * il * 2);
        let (g, u, ld) = if n <= MAX_ROWS { (np(&a.gu), at(&a.gu, il), 2 * il) } else { (np(&a.gu), at(&a.gu, n * il), il) };
        ck(unsafe { glm_swiglu(st, g, ld as i64, u, ld as i64, hsh, il as i64, n as i32, il as i32) }, "shared swiglu")?;
        // routed: grouping, rotation, gate/up, the down projection's input rotation
        let decode = p <= 64;
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
        // routed down + the weighted combine over picks; the shared down; summed, gathered
        let kd = il * w;
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
        self.exl3(sh_down, np(&a.hf), kd, n, np(&a.shout), false)?;
        ck(unsafe { glm_moe_out(st, np(&a.local), np(&a.shout), np(&a.mout), (n * dn) as i64) }, "moe out")?;
        self.gather_cols(np(&a.mout), n, dn, 2, out, prefill)
    }

    // ------------------------------------------------------------------ model
    fn upload_ids(&self, ids: &[u32], pos0: usize, mtp: bool) -> Result<()> {
        let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        self.act.tok.upload_async(0, &v, &self.st)?;
        let p = [pos0 as i32];
        if mtp {
            self.act.mpos0.upload_async(0, &p, &self.st)?;
            ck(unsafe { glm_positions(self.st.raw(), np(&self.act.mpos0), ids.len() as i32, 0, 1, 0, np(&self.act.mpos),
                                      std::ptr::null_mut(), std::ptr::null_mut()) }, "mtp positions")
        } else {
            self.act.pos0.upload_async(0, &p, &self.st)?;
            ck(unsafe { glm_positions(self.st.raw(), np(&self.act.pos0), ids.len() as i32, self.rank as i32,
                                      if self.opt.kv_split { self.world as i32 } else { 1 }, self.s_loc as i32, np(&self.act.pos),
                                      np(&self.act.rows), np(&self.act.bound)) }, "positions")
        }
    }

    /// The embedding rows of the uploaded ids (zeroed at position 0 for the MTP layer) -> bf16
    /// at `out`, summed over the pair's vocabulary halves.
    fn embed(&self, t: usize, mtp: bool, out: *mut c_void) -> Result<()> {
        let a = &self.act;
        let h = self.cfg.hidden;
        ck(unsafe { glm_embed(self.st.raw(), np(&a.tok), t as i32, np(&self.w.embed_q), np(&self.w.embed_s),
                              (self.rank * self.vl) as i32, self.vl as i32, h as i32,
                              if mtp { np(&a.mpos) } else { std::ptr::null_mut() }, np(&a.f32buf)) }, "embed")?;
        self.allreduce(np(&a.f32buf), t * h, out, t > MAX_ROWS)
    }

    /// The decoder layers over `t` rows at self.pos (ids uploaded): the final norm into hn.
    fn layers(&self, t: usize) -> Result<()> {
        let (cfg, a) = (&self.cfg, &self.act);
        let st = self.st.raw();
        let h = cfg.hidden as i64;
        let prefill = t > MAX_ROWS;
        self.embed(t, false, np(&a.h))?;
        let mut idx_valid = false;
        for l in 0..cfg.layers {
            let lw = &self.w.layers[l];
            if l == 0 {
                ck(unsafe { glm_rmsnorm(st, np(&a.h), h, std::ptr::null(), 0, np(&lw.ln_in), np(&a.xn), h, t as i32, h as i32,
                                        cfg.rms_eps) }, "norm")?;
            } else {
                ck(unsafe { glm_rmsnorm(st, np(&a.h), h, np(&a.a), h, np(&lw.ln_in), np(&a.xn), h, t as i32, h as i32,
                                        cfg.rms_eps) }, "norm")?;
            }
            self.attention(l, t, prefill, &mut idx_valid)?;
            ck(unsafe { glm_rmsnorm(st, np(&a.h), h, np(&a.a), h, np(&lw.ln_post), np(&a.xn), h, t as i32, h as i32,
                                    cfg.rms_eps) }, "norm")?;
            self.mlp(l, t, prefill)?;
        }
        // h += the last MLP; hn = the final norm (the MTP layer's input)
        ck(unsafe { glm_rmsnorm(st, np(&a.h), h, np(&a.a), h, np(&self.w.norm), np(&a.hn), h, t as i32, h as i32, cfg.rms_eps) },
           "final norm")
    }

    /// The head over `t` rows of `x` (bf16 [t][H]) and each row's argmax -> act.out_tok.
    fn head_argmax(&self, x: *const c_void, t: usize) -> Result<Vec<u32>> {
        let a = &self.act;
        let st = self.st.raw();
        self.exl3(&self.w.head, x, self.cfg.hidden, t, np(&a.head), false)?;
        ck(unsafe { glm_argmax(st, np(&a.head), self.vl as i64, t as i32, self.vl as i32, (self.rank * self.vl) as i32,
                               np(&a.pairs)) }, "argmax")?;
        self.gather(np(&a.pairs), t * 8, at(&a.pairs, 2 * VERIFY_ROWS * 2), false)?;
        ck(unsafe { glm_argmax_combine(st, at(&a.pairs, 2 * VERIFY_ROWS * 2), self.world as i32, t as i32, np(&a.out_tok)) },
           "argmax combine")?;
        let mut out = vec![0i32; t];
        a.out_tok.download_async(0, &mut out, &self.st)?;
        self.st.synchronize()?;
        Ok(out.into_iter().map(|x| x as u32).collect())
    }

    /// The MTP layer over rows `ids` (token i + 1 at position pos0 + i) and the previous hidden
    /// `hsrc` -> its post-norm hidden in mhn and, with `logits`, each row's argmax.
    fn mtp_core(&mut self, ids: &[u32], hsrc: Hidden, pos0: usize, logits: bool) -> Result<Option<Vec<u32>>> {
        let t = ids.len();
        let l = self.cfg.mtp_layer();
        self.mtp_pos = pos0;
        self.upload_ids(ids, pos0, true)?;
        let (cfg, a) = (&self.cfg, &self.act);
        let st = self.st.raw();
        let h = cfg.hidden;
        let lw = &self.w.layers[l];
        let mx = lw.mtp.as_ref().ok_or("no MTP layer")?;
        let prefill = t > MAX_ROWS;
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
        ck(unsafe { glm_rmsnorm(st, np(&a.h), h as i64, std::ptr::null(), 0, np(&lw.ln_in), np(&a.xn), h as i64, t as i32,
                                h as i32, cfg.rms_eps) }, "norm")?;
        let mut idx_valid = false;
        self.attention(l, t, prefill, &mut idx_valid)?;
        ck(unsafe { glm_rmsnorm(st, np(&a.h), h as i64, np(&a.a), h as i64, np(&lw.ln_post), np(&a.xn), h as i64, t as i32,
                                h as i32, cfg.rms_eps) }, "norm")?;
        self.mlp(l, t, prefill)?;
        ck(unsafe { glm_rmsnorm(st, np(&a.h), h as i64, np(&a.a), h as i64, np(&mx.sh_norm), np(&a.mhn), h as i64, t as i32,
                                h as i32, cfg.rms_eps) }, "shared head norm")?;
        if !logits {
            return Ok(None);
        }
        Ok(Some(self.head_argmax(np(&a.mhn), t)?))
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

/// Flash-decoding splits for `heads` heads over `k` picks at `t` rows (sparse_attn.sparse_mla's
/// rule, ~32 programs a launch): (splits, picks a split, a multiple of 32).
fn decode_split(k: usize, heads: usize, t: usize) -> (usize, usize) {
    let hb = heads.div_ceil(32);
    let want = 32usize.div_ceil(hb * t).next_power_of_two().clamp(1, 16);
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
            self.upload_ids(ids, self.pos, false)?;
            self.layers(t)?;
            self.last_rows = t;
            let out = self.head_argmax(np(&self.act.hn), t)?;
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
            self.upload_ids(part, self.pos, false)?;
            self.layers(n)?;
            self.last_rows = n;
            if c0 + n == t {
                out = self.head_argmax(at(&self.act.hn, (n - 1) * h), 1)?;
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
                OP_EXIT => return Ok(()),
                op => return Err(format!("rank 1: unknown op {op}").into()),
            }
        }
    }
}
