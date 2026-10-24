//! The DS v4.1 forward pass (engine/model.py::Model.forward + block + attention + moe,
//! tools/v41_ref.py), on the ds41.cu kernels and the native FP4 MoE kernels.
//!
//! This first engine runs in *streaming* mode: each step reads every layer's weights from
//! NVMe, routes, loads only the experts the step's tokens chose (through a 16-slot arena),
//! and frees the layer again. It needs ~1 GB of device memory, so it runs next to a live
//! server; it is the correctness vehicle for the resident engine, which keeps the same
//! step code and swaps the weight source.
//!
//! Semantics follow the checkpoint's reference forward (inference/model.py), which runs
//! every layer over every token. The Python engine's decoder replay (layers past the
//! candidate source over only the last 128 prompt tokens) is an approximation from the
//! tech report; it agrees exactly for prompts up to 128 tokens and is not used here.
//!
//! Limits of this stage, each refused with an error rather than approximated:
//! - at most `TMAX` = 16 tokens per call (the decode GEMM and MoE router bound; prefill
//!   loops over 16-token chunks);
//! - total context <= index_topk (1024): below that the indexer selects every visible
//!   compressed position, so skipping it is exact. The indexer itself lands next.

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
use crate::weights::{Layer, Loader};
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

/// Per-step wall-clock split, printed by the CLI: where a streaming step's time goes.
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

const W13: usize = 2304 * 2560;
const S13: usize = 2304 * 160;
const W2: usize = 5120 * 1152;
const S2: usize = 5120 * 72;

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
    cpos: DeviceBuffer<i32>,
    raw: DeviceBuffer<u8>,
    inv: DeviceBuffer<i32>,
    erows: DeviceBuffer<u16>,
    ekv: DeviceBuffer<u16>,
    emb: DeviceBuffer<u16>,
    row_ids: DeviceBuffer<i32>,
    head_x: DeviceBuffer<u16>,
    head_w: DeviceBuffer<u16>,
    logits: DeviceBuffer<u16>,
}

struct Caches {
    ring: Vec<DeviceBuffer<u16>>,
    ckv: BTreeMap<usize, DeviceBuffer<u16>>,
    ik: BTreeMap<usize, DeviceBuffer<u16>>,
    pending_kv: BTreeMap<usize, DeviceBuffer<f32>>,
    pending_sc: BTreeMap<usize, DeviceBuffer<f32>>,
    pending: BTreeMap<usize, bool>,
    len: usize,
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
    pub max_seq: usize,
    pub times: StepTimes,
}

impl Engine {
    pub fn new(model_dir: &std::path::Path, token_map: Vec<i64>, max_seq: usize) -> Result<Self> {
        let cfg = Config::load(model_dir)?;
        if max_seq > cfg.index_topk {
            return Err(format!(
                "max_seq {max_seq} > index_topk {}: the indexer is not ported yet, and below that bound \
                 skipping it is exact",
                cfg.index_topk
            )
            .into());
        }
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
        let norm = {
            let mut ld = Loader::new(&ckpt, &stream)?;
            ld.bf16("norm.weight", &[d])?
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
                eg::engram_gather_create_direct(
                    shard.file.direct_fd(), wm.lo as i64, sm.lo as i64, wm.shape[0] as i64, 16,
                )
            };
            if ctx.is_null() {
                return Err(format!("layer {l}: engram gather setup failed").into());
            }
            engram_tables.insert(l, EngramTable { ctx });
        }

        let t = TMAX;
        let p = t * cfg.n_activated_experts;
        let hdq = cfg.n_heads * cfg.head_dim;
        let eng_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
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
            cpos: i(t)?,
            raw: DeviceBuffer::zeroed(t * eng_cols * 264)?,
            inv: i(t * eng_cols)?,
            erows: a(t * eng_cols * cfg.engram_head_dim)?,
            ekv: a(t * (hc + 1) * d)?,
            emb: a(t * d)?,
            row_ids: DeviceBuffer::from_slice(&(0..t as i32).collect::<Vec<_>>())?,
            head_x: a(d)?,
            head_w: a(HEAD_ROWS * d)?,
            logits: a(cfg.vocab_size)?,
        };
        let mut caches = Caches {
            ring: Vec::new(),
            ckv: BTreeMap::new(),
            ik: BTreeMap::new(),
            pending_kv: BTreeMap::new(),
            pending_sc: BTreeMap::new(),
            pending: BTreeMap::new(),
            len: 0,
        };
        for _ in 0..cfg.n_layers {
            caches.ring.push(a(RING * cfg.head_dim)?);
        }
        for &l in &cfg.kv_source_layers {
            let r = cfg.compress_ratios[l];
            caches.ckv.insert(l, a((max_seq / r + 1) * cfg.head_dim)?);
            caches.ik.insert(l, a((max_seq / r + 1) * cfg.index_head_dim)?);
            if r > 1 {
                caches.pending_kv.insert(l, f(cfg.head_dim)?);
                caches.pending_sc.insert(l, f(cfg.head_dim)?);
                caches.pending.insert(l, false);
            }
        }
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
            max_seq,
            times: StepTimes::default(),
        })
    }

    pub fn len(&self) -> usize {
        self.caches.len
    }

    pub fn is_empty(&self) -> bool {
        self.caches.len == 0
    }

    /// Run tokens `ids` at positions `len..len + ids.len()` (at most TMAX); with
    /// `want_logits`, return the last row's bf16 logits (the engine's head rounds them).
    pub fn forward(&mut self, ids: &[u32], want_logits: bool) -> Result<Option<Vec<u16>>> {
        let t = ids.len();
        let s = self.caches.len;
        if t == 0 || t > TMAX {
            return Err(format!("forward takes 1..={TMAX} tokens, got {t}").into());
        }
        if s + t > self.max_seq {
            return Err(format!("context {} exceeds max_seq {}", s + t, self.max_seq).into());
        }
        let cfg = self.cfg.clone();
        let (d, hc) = (cfg.dim, cfg.hc_mult);
        let tw = Instant::now();
        let st = &self.stream;
        let sr = st.raw();

        // positions and the visible compressed rows per ratio (all of them: see module doc)
        let pos: Vec<i32> = (s..s + t).map(|p| p as i32).collect();
        self.act.pos.upload_async(0, &pos, st)?;
        let cidx = |r: usize| -> (Vec<i32>, usize) {
            let c = (s + t) / r;
            let mut v = vec![-1i32; t * c.max(1)];
            for (ti, &p) in pos.iter().enumerate() {
                let vis = (p as usize + 1) / r;
                for j in 0..vis {
                    v[ti * c.max(1) + j] = j as i32;
                }
            }
            (v, c)
        };
        let (c1v, c1) = cidx(1);
        let (c2v, c2) = cidx(2);
        self.act.cidx1.upload_async(0, &c1v, st)?;
        self.act.cidx2.upload_async(0, &c2v, st)?;

        // embedding rows straight from the checkpoint (a 10 KB read each)
        {
            let mut ld = Loader::new(&self.ckpt, st)?;
            let row = d * 2;
            for (ti, &id) in ids.iter().enumerate() {
                ld.read_into("embed.weight", id as usize * row, (id as usize + 1) * row, self.act.emb.ptr_at(ti * d))?;
            }
        }
        ck(unsafe { ds_embed_hc(sr, np(&self.act.row_ids), np(&self.act.emb), np(&self.act.h), t as i32, d as i32, hc as i32) }, "embed")?;
        let mut pre = vec![0f32; t * hc];
        for ti in 0..t {
            pre[ti * hc] = 1.0;
        }
        self.act.pre.upload_async(0, &pre, st)?;

        let hashes = self.hash.hash(ids, s, None); // [t][n_engram_layers][cols]
        let cols = self.hash.layout.n_hash_cols();
        let n_eng = cfg.engram_layer_ids.len();
        self.times.compute_s += tw.elapsed().as_secs_f64();

        for l in 0..cfg.n_layers {
            let t0 = Instant::now();
            let w = {
                let mut ld = Loader::new(&self.ckpt, &self.stream)?;
                Layer::load(&mut ld, &cfg, l)?
            };
            let st = &self.stream;
            let sr = st.raw();
            self.times.weights_s += t0.elapsed().as_secs_f64();

            if let Some(e) = &w.engram {
                let te = Instant::now();
                let li = cfg.engram_layer_ids.iter().position(|&x| x == l).unwrap();
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
                self.act.raw.upload_async(0, &raw, st)?;
                self.act.inv.upload_async(0, &inv, st)?;
                self.times.engram_s += te.elapsed().as_secs_f64();
                let a = &self.act;
                ck(unsafe { ds_engram_dequant(sr, np(&a.raw), np(&a.inv), t as i32, cols as i32, np(&a.erows)) }, "engram dequant")?;
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

            let tc = Instant::now();
            self.block(&w, l, s, t, c1, c2)?;
            self.stream.synchronize()?;
            self.times.compute_s += tc.elapsed().as_secs_f64();
            drop(w);
        }
        self.caches.len = s + t;

        if !want_logits {
            return Ok(None);
        }
        let th = Instant::now();
        let st = &self.stream;
        let sr = st.raw();
        let a = &self.act;
        // the last row only: hc_pre with the final pre_mix, the final norm, then the head
        let last_h = a.h.ptr_at((t - 1) * hc * d);
        let last_pre = a.pre.ptr_at((t - 1) * hc);
        ck(unsafe { ds_hc_pre_norm(sr, last_h, last_pre, np(&self.norm), np(&a.head_x), 1, d as i32, hc as i32, cfg.norm_eps) }, "final norm")?;
        let mut ld = Loader::new(&self.ckpt, st)?;
        let v = cfg.vocab_size;
        let mut r0 = 0;
        while r0 < v {
            let rows = HEAD_ROWS.min(v - r0);
            ld.read_into("head.weight", r0 * d * 2, (r0 + rows) * d * 2, a.head_w.raw())?;
            ck(unsafe {
                ds_linear_bf16(sr, np(&a.head_x), d as i64, np(&a.head_w), 1, rows as i32, d as i32, 1,
                               a.logits.ptr_at(r0), nul(), v as i64, nul())
            }, "head")?;
            r0 += rows;
        }
        let mut logits = vec![0u16; v];
        a.logits.download_async(0, &mut logits, st)?;
        st.synchronize()?;
        self.times.head_s += th.elapsed().as_secs_f64();
        Ok(Some(logits))
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

    fn block(&mut self, w: &Layer, l: usize, s: usize, t: usize, c1: usize, c2: usize) -> Result<()> {
        let (d, hc) = (self.cfg.dim, self.cfg.hc_mult);
        let sr = self.stream.raw();
        let eps = self.cfg.norm_eps;
        // attention site
        self.hc_mixes(&w.hc_attn_fn, &w.hc_attn_base, &w.hc_attn_scale, t, &self.act.apre, &self.act.apost, &self.act.acomb)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_pre_norm(sr, np(&a.h), np(&a.pre), np(&w.attn_norm), np(&a.y), t as i32, d as i32, hc as i32, eps) }, "attn hc_pre")?;
        }
        self.attention(w, l, s, t, c1, c2)?;
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
        self.moe(w, l, t)?;
        {
            let a = &self.act;
            ck(unsafe { ds_hc_post(sr, np(&a.moe_out), np(&a.h), np(&a.fpost), np(&a.fcomb), np(&a.h), t as i32, d as i32) }, "ffn hc_post")?;
        }
        // the next layer's attention mixes the stream with this layer's ffn pre
        std::mem::swap(&mut self.act.pre, &mut self.act.fpre);
        Ok(())
    }

    fn attention(&mut self, w: &Layer, l: usize, s: usize, t: usize, c1: usize, c2: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let ratio = cfg.compress_ratios[l];
        // independent of q and the window: the chunk's new compressed positions, which this
        // layer's own attention already reads (engine/model.py::_compressed)
        if ratio > 0 {
            if let Some(comp) = &w.compressor {
                self.compress(w, comp, l, s, t)?;
            }
        }
        let st = &self.stream;
        let sr = st.raw();
        let (d, hd, rd, h) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim, cfg.n_heads);
        let table = if ratio > 0 { &self.rope_c } else { &self.rope_w };
        let a = &self.act;
        let ti = t as i32;
        let fp8 = |x: *mut c_void, ldx: usize, wt: &crate::weights::Fp8, y: *mut c_void, ldy: usize, what: &str| {
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
        ck(unsafe { ds_rmsnorm(sr, np(&a.kv0), hd as i64, np(&w.kv_norm), np(&a.kv), hd as i64, ti, hd as i32, cfg.norm_eps) }, "kv_norm")?;
        ck(unsafe { ds_rope(sr, np(&a.kv), hd as i64, ti, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.pos), np(table), 0) }, "kv rope")?;
        let ring = &self.caches.ring[l];
        for i in 0..t {
            ring.copy_from(((s + i) % RING) * hd, &a.kv, i * hd, hd, st)?;
        }

        let (ckv_ptr, cidx_ptr, c) = if ratio > 0 {
            let src = cfg.kv_source_of(l).expect("compressed layer has a kv source");
            let (cidx, c) = if ratio == 1 { (&a.cidx1, c1) } else { (&a.cidx2, c2) };
            (np(&self.caches.ckv[&src]), np(cidx), c)
        } else {
            (nul(), nul(), 0)
        };
        ck(unsafe {
            ds_attn_decode(sr, np(&a.q), np(ring), RING as i32, np(&a.pos), cfg.window_size as i32, 0, ckv_ptr, cidx_ptr,
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
        fp8(np(&a.oa), og * cfg.o_lora_rank, &w.wo_b, np(&a.attn_out), d, "wo_b")
    }

    /// engine/model.py::_compressed for a kv-source layer: new compressed positions of this
    /// chunk into the shared cache (and their index keys).
    fn compress(&mut self, w: &Layer, comp: &crate::weights::Compressor, l: usize, s: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let st = &self.stream;
        let sr = st.raw();
        let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
        let r = cfg.compress_ratios[l];
        let a = &self.act;
        let ti = t as i32;
        // (compressed index j, absolute position of its rope) per completed group
        let mut groups: Vec<usize> = Vec::new();
        if r > 1 {
            let wgate = comp.wgate.as_ref().expect("ratio-2 compressor has a gate");
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&comp.wkv), ti, hd as i32, d as i32, 1, nul(), np(&a.kvl), hd as i64, nul()) }, "comp wkv")?;
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(wgate), ti, hd as i32, d as i32, 1, nul(), np(&a.sc), hd as i64, nul()) }, "comp wgate")?;
            let (pkv, psc) = (&self.caches.pending_kv[&l], &self.caches.pending_sc[&l]);
            for p in s..s + t {
                if p % r != r - 1 {
                    continue;
                }
                let gi = groups.len();
                let (ka, sa) = if p > s {
                    (a.kvl.ptr_at((p - 1 - s) * hd), a.sc.ptr_at((p - 1 - s) * hd))
                } else {
                    if !self.caches.pending[&l] {
                        return Err(format!("layer {l}: position {p} pairs with a pending position that is not there").into());
                    }
                    (np(pkv), np(psc))
                };
                ck(unsafe {
                    ds_compress_pair(sr, ka, sa, a.kvl.ptr_at((p - s) * hd), a.sc.ptr_at((p - s) * hd), a.lat.ptr_at(gi * hd), hd as i32)
                }, "compress pair")?;
                groups.push(p / r);
            }
            let last = s + t - 1;
            if (s + t) % r != 0 {
                pkv.copy_from(0, &a.kvl, (last - s) * hd, hd, st)?;
                psc.copy_from(0, &a.sc, (last - s) * hd, hd, st)?;
                self.caches.pending.insert(l, true);
            } else {
                self.caches.pending.insert(l, false);
            }
        } else {
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&comp.wkv), ti, hd as i32, d as i32, 1, np(&a.lat), nul(), hd as i64, nul()) }, "comp wkv r1")?;
            groups.extend(s..s + t);
        }
        if groups.is_empty() {
            return Ok(());
        }
        let g = groups.len();
        let gi = g as i32;
        let cpos: Vec<i32> = groups.iter().map(|&j| (j * r) as i32).collect();
        a.cpos.upload_async(0, &cpos, st)?;
        ck(unsafe { ds_rmsnorm(sr, np(&a.lat), hd as i64, np(&comp.norm), np(&a.latn), hd as i64, gi, hd as i32, cfg.norm_eps) }, "comp norm")?;
        if let Some(key) = w.indexer.as_ref().and_then(|ix| ix.key.as_ref()) {
            let ihd = cfg.index_head_dim;
            ck(unsafe { ds_linear_bf16(sr, np(&a.latn), hd as i64, np(&key.wk), gi, ihd as i32, hd as i32, 1, np(&a.kidx), nul(), ihd as i64, nul()) }, "index wk")?;
            ck(unsafe { ds_rmsnorm(sr, np(&a.kidx), ihd as i64, np(&key.k_norm), np(&a.kidxn), ihd as i64, gi, ihd as i32, cfg.norm_eps) }, "index k_norm")?;
            ck(unsafe { ds_rope(sr, np(&a.kidxn), ihd as i64, gi, 1, ihd as i64, (ihd - rd) as i32, rd as i32, np(&a.cpos), np(&self.rope_c), 0) }, "index k rope")?;
            let ik = &self.caches.ik[&l];
            for (i, &j) in groups.iter().enumerate() {
                ik.copy_from(j * ihd, &a.kidxn, i * ihd, ihd, st)?;
            }
        }
        ck(unsafe { ds_rope(sr, np(&a.latn), hd as i64, gi, 1, hd as i64, (hd - rd) as i32, rd as i32, np(&a.cpos), np(&self.rope_c), 0) }, "comp rope")?;
        ck(unsafe { ds_fp4_qdq_e4m3(sr, np(&a.latn), (g * hd) as i64) }, "comp qdq")?;
        let ckv = &self.caches.ckv[&l];
        for (i, &j) in groups.iter().enumerate() {
            ckv.copy_from(j * hd, &a.latn, i * hd, hd, st)?;
        }
        Ok(())
    }

    /// engine/model.py::moe on one box, with the routed experts streamed through the arena.
    fn moe(&mut self, w: &Layer, l: usize, t: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let st = &self.stream;
        let sr = st.raw();
        let (d, inter, e, k) = (cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts, cfg.n_activated_experts);
        let ti = t as i32;
        let p = t * k;
        {
            let a = &self.act;
            ck(unsafe { ds_linear_bf16(sr, np(&a.y), d as i64, np(&w.gate_w), ti, e as i32, d as i32, 1, nul(), np(&a.z), e as i64, nul()) }, "gate")?;
            ck(unsafe { ds_router(sr, np(&a.z), np(&w.gate_bias), nul(), ti, e as i32, k as i32, cfg.route_scale, np(&a.idx), np(&a.rw)) }, "router")?;
        }
        let mut idx = vec![0i32; p];
        self.act.idx.download_async(0, &mut idx, st)?;
        st.synchronize()?;
        let mut experts: Vec<i32> = idx.clone();
        experts.sort_unstable();
        experts.dedup();

        let reader = ExpertReader::new(&self.ckpt);
        for group in experts.chunks(ARENA_SLOTS) {
            let tl = Instant::now();
            for (slot, &ex) in group.iter().enumerate() {
                let views = reader.read_expert_into(&mut self.expert_stage, &ExpertReader::expert_prefix(l, ex as usize))?;
                let ar = &self.arena;
                let dsts: [(*mut c_void, usize); 6] = [
                    (ar.w1.ptr_at(slot * W13), W13), (ar.s1.ptr_at(slot * S13), S13),
                    (ar.w2.ptr_at(slot * W2), W2), (ar.s2.ptr_at(slot * S2), S2),
                    (ar.w3.ptr_at(slot * W13), W13), (ar.s3.ptr_at(slot * S13), S13),
                ];
                for (&(off, len), &(dst, want)) in views.iter().zip(dsts.iter()) {
                    if len != want {
                        return Err(format!("layer {l} expert {ex}: tensor of {len} bytes, arena expects {want}").into());
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
        let a = &self.act;
        let fp8 = |x: *mut c_void, ldx: usize, wt: &crate::weights::Fp8, y: *mut c_void, ldy: usize, what: &str| {
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
