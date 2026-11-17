//! Inkling 2's layers on the GLM engine (a child module of forward: the engine's own state).
//!
//! A decoder layer (modeling_inkling2.py's Inkling2SlidingDecoderLayer / Inkling2FullDecoderLayer),
//! x the layer's input:
//!   a   = attention(norm(mix_a(blocks, x)))                 layer 0: norm(x), nothing to mix
//!   res = conv_a(a)             (+ x unless l % block == 0: such a layer saves x as a block state
//!                                and restarts the residual from its attention's output)
//!   m   = MoE(norm(mix_m(blocks, res)))
//!   out = res + conv_m(m)
//! mix_* is the block attention residual over the saved block states and the current state; a
//! full-attention layer mixes three inputs (q, k, v), each with its own norm. The decoder ends with
//! one more mix and the final norm. As the GLM engines defer a layer's last add into the next
//! layer's start, `out` is formed (ink_close) at the next layer's start or before the final mix, so
//! a decode pass's segments split where they always did: after a layer's MoE input.
//!
//! State a lane keeps between passes, all by position (inkling.cu's module comment): the four
//! convolutions' raw inputs of every layer (rings of CONV_RING positions), the window layers' keys
//! and values (a ring of `ring` positions), the full-attention layers' keys and values (the paged
//! cache, a row a position). A verify pass's rejected rows are never undone - the position goes
//! back and the next pass writes over them - and a set_pos further back than the rings reach
//! (REWIND positions) restores the lane's snapshot: the window before a prompt's end and the
//! convolutions' last three inputs, taken as GLM-5.3-Flash takes KDA's (Engine::want_snapshot).

use super::*;
use tidelm_kernels::inkling::*;

use crate::weights::InkW;

/// Positions a set_pos can go back without a snapshot: the conv rings hold CONV_RING positions,
/// three of them the taps before the target.
const REWIND: usize = CONV_RING - 4;
/// The drafter's global layer keeps this many positions of context (a ring; the reference keeps the
/// whole context: 4 KB a position, 1.3 GB at 320K). An engine choice that changes drafts, never
/// answers (the target verifies every token); its effect on acceptance is not measured yet.
const DRAFT_GLOBAL_RING: usize = 16384;
/// Rank 0 -> the other ranks: a draft round for a lane (anchor, its position).
pub(crate) const OP_INK_DRAFT: u32 = 41;
/// A draft round's result: 7 tokens (i32) then their probabilities (fp32).
const DRAFT_OUT: usize = 7 * 8;
/// TIDELM_INK_FLASH_DECODE (default on): full layers' verify rows through ink_flash_part.
fn dec_flash() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TIDELM_INK_FLASH_DECODE").as_deref() != Ok("0"))
}

/// TIDELM_INK_PF_FLASH (default on): full layers' prefill attention through ink_pf_flash.
fn pf_flash() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TIDELM_INK_PF_FLASH").as_deref() != Ok("0"))
}

/// Key splits of a decode row's attention: the window layers' 1,024 keys, the full layers' context.
const NS_WINDOW: usize = 4;
const NS_FULL: usize = 64;

pub(crate) struct InkAct {
    /// the fused projection [r][ldp]; a full layer's key and value ones [r][ldk]
    proj: DeviceBuffer<u16>,
    pk: DeviceBuffer<u16>,
    pv: DeviceBuffer<u16>,
    /// a full layer's three mixed inputs [3][r][H]
    xin: DeviceBuffer<u16>,
    /// the key / value convolutions' outputs [r][kl x 128]
    kc: DeviceBuffer<u16>,
    vc: DeviceBuffer<u16>,
    /// queries [r][hl][128] and relative states [r][hl][16], as the attention reads them
    qc: DeviceBuffer<u16>,
    rc: DeviceBuffer<u16>,
    /// the residual in fp32 [r][H], the saved block states [r][blocks][H]
    res: DeviceBuffer<f32>,
    blocks: DeviceBuffer<u16>,
    /// decode rows' softmax partials [HEAD_ROWS][kl][NS_FULL][6] (pacc x 128)
    pm: DeviceBuffer<f32>,
    pl: DeviceBuffer<f32>,
    pacc: DeviceBuffer<f32>,
    /// prefill: a row block's keys and values [window + rows][kl][256] (window layers), its
    /// attention weights (bf16, as many as act.scores holds) and weighted values [hl][rows][128]
    kvs: DeviceBuffer<u16>,
    wp: DeviceBuffer<u16>,
    ov: DeviceBuffer<u16>,
}

pub(crate) struct InkState {
    hl: usize,
    kl: usize,
    lanes: usize,
    /// a window layer's ring positions (a power of two), and layer -> its place among them
    ring: usize,
    swa_of: Vec<Option<usize>>,
    nswa: usize,
    /// [window layers][lanes][ring][kl x 256]
    kv_ring: DeviceBuffer<u16>,
    /// the convolutions' rings: keys and values [layers][lanes][CONV_RING][kl x 128], the
    /// attention's and the MoE's outputs [layers][lanes][CONV_RING][H]
    ck: DeviceBuffer<u16>,
    cv: DeviceBuffer<u16>,
    ca: DeviceBuffer<u16>,
    cm: DeviceBuffer<u16>,
    /// prompt-end snapshots, two a lane (2 lane + which): the window before the position
    /// [.][window layers][window][kl x 256] and the rings' last three positions [.][layers][3][..];
    /// per lane the committed one and its position, the pending one's, and the position its next
    /// forward is to snapshot (FlashState's protocol)
    snap_kv: DeviceBuffer<u16>,
    snap_ck: DeviceBuffer<u16>,
    snap_cv: DeviceBuffer<u16>,
    snap_ca: DeviceBuffer<u16>,
    snap_cm: DeviceBuffer<u16>,
    snap_cur: Vec<usize>,
    snap_pos: Vec<Option<usize>>,
    snap_pend: Vec<Option<usize>>,
    pub(crate) snap_want: Vec<Option<usize>>,
    /// the lowest position a plain set_pos may go back to: 0, or where the lane's rings were last
    /// restored (a snapshot or the disk store puts back the window and three conv inputs before
    /// that position and nothing earlier)
    floor: Vec<usize>,
    /// DSpark drafting (inkling_draft): on at all (every rank), the drafter itself on the rank that
    /// drafts (the pair's last), the pass's taps for it [8][rows][H] (the inputs of layers 5, 11, ..,
    /// 41 and the final normed hidden), a round's result here and every rank's (the gather)
    pub(crate) draft_on: bool,
    draft: Option<crate::inkling_draft::InkDraft>,
    /// GLM_INK_KV=fp8 (the default): the full layers' cache rows are FP8 codes and their scales
    pub(crate) kvf: bool,
    /// TIDELM_INK_FUSE (default on; the bench knob ink_fuse): a decode pass's convolutions write their
    /// rings in the same launch, and ink_mix takes its vectorized kernel (2026-11-16 profile of plain
    /// decode, 39.5 ms a token: ink_mix 3.4 ms, the convolutions 1.9 + 0.9 ms of it; a mix 34.9 -> 8.2
    /// us, a convolution with its ring write 8.2 -> 4.1). In-boot A/B, 5 ABBA rounds at 2.5K context:
    /// a pass of 1 / 2 / 4 / 8 rows 38.05 -> 34.87 / 45.97 -> 42.78 / 55.64 -> 52.56 / 83.38 -> 80.29 ms
    /// (-8.3 / -7.0 / -5.5 / -3.7%).
    pub(crate) fuse: bool,
    /// the drafter's listed vocabulary as a membership table [real vocabulary] (every rank: rank 0's
    /// scheduler decides a lane's rounds by it), None: every id
    pub(crate) draft_listed: Option<Vec<bool>>,
    taps: DeviceBuffer<u16>,
    tap_rows: usize,
    dout: DeviceBuffer<u8>,
    dall: DeviceBuffer<u8>,
    /// the drafter's context: the highest position a pass wrote (+1) and the lowest position whose
    /// entries are still the lane's sequence's (rings by position: a rewind past what they hold, or a
    /// restore from disk, leaves nothing valid below where the lane restarts)
    ctx_hi: Vec<usize>,
    ctx_lo: Vec<usize>,
    /// the pass's [2][MAX_LANES][4]: each lane group's lane first
    meta: DeviceBuffer<i32>,
    h_meta: tidelm_cuda::PinnedBuffer<i32>,
    /// the final mix's query [H]
    out_q: DeviceBuffer<f32>,
    pub(crate) act: InkAct,
}

impl InkState {
    pub(crate) fn new(cfg: &GlmConfig, opt: &Options, world: usize, lanes: usize, model_dir: &Path, st: &Stream) -> Result<Self> {
        let ink = cfg.ink.ok_or("no Inkling 2 config")?;
        let hl = cfg.heads / world;
        let kl = ink.kv_heads / world;
        let (h, d) = (cfg.hidden, ink.head_dim);
        let (dk, rowk) = (kl * d, kl * 2 * d);
        let r = opt.chunk.max(HEAD_ROWS);
        let ring = (opt.chunk + ink.window + HEAD_ROWS).next_power_of_two();
        let mut swa_of = vec![None; cfg.layers];
        let mut nswa = 0;
        for l in 0..cfg.layers {
            if cfg.attn[l] == AttnKind::Swa {
                swa_of[l] = Some(nswa);
                nswa += 1;
            }
        }
        let z = |n: usize| -> Result<DeviceBuffer<u16>> { Ok(DeviceBuffer::zeroed(n.max(1))?) };
        let f = |n: usize| -> Result<DeviceBuffer<f32>> { Ok(DeviceBuffer::new(n.max(1))?) };
        // the widest fused projections: a window layer's on one rank, a full layer's key / value one
        let pad = |n: usize| n.div_ceil(32) * 32;
        let ldp = 2 * hl * d + 2 * dk + hl * ink.d_rel + pad(hl);
        let ldk = dk + pad(kl * ink.conv);
        let nl = cfg.layers;
        let act = InkAct {
            proj: z(r * ldp)?,
            pk: z(r * ldk)?,
            pv: z(r * ldk)?,
            xin: z(3 * r * h)?,
            kc: z(r * dk)?,
            vc: z(r * dk)?,
            qc: z(r * hl * d)?,
            rc: z(r * hl * ink.d_rel)?,
            res: f(r * h)?,
            blocks: z(r * cfg.ink_blocks() * h)?,
            pm: f(HEAD_ROWS * kl * NS_FULL * 6)?,
            pl: f(HEAD_ROWS * kl * NS_FULL * 6)?,
            pacc: f(HEAD_ROWS * kl * NS_FULL * 6 * d)?,
            kvs: z((ink.window + r) * rowk)?,
            wp: z(SCORE_VALUES)?,
            ov: z(hl * r.min(1024) * d)?,
        };
        Ok(InkState {
            hl,
            kl,
            lanes,
            ring,
            swa_of,
            nswa,
            kv_ring: z(nswa * lanes * ring * rowk)?,
            ck: z(nl * lanes * CONV_RING * dk)?,
            cv: z(nl * lanes * CONV_RING * dk)?,
            ca: z(nl * lanes * CONV_RING * h)?,
            cm: z(nl * lanes * CONV_RING * h)?,
            snap_kv: z(2 * lanes * nswa * ink.window * rowk)?,
            snap_ck: z(2 * lanes * nl * 3 * dk)?,
            snap_cv: z(2 * lanes * nl * 3 * dk)?,
            snap_ca: z(2 * lanes * nl * 3 * h)?,
            snap_cm: z(2 * lanes * nl * 3 * h)?,
            snap_cur: vec![0; lanes],
            snap_pos: vec![None; lanes],
            snap_pend: vec![None; lanes],
            snap_want: vec![None; lanes],
            floor: vec![0; lanes],
            kvf: {
                let f = crate::inkling::kv_fp8()?;
                if f && !pf_flash() {
                    return Err("GLM_INK_KV=fp8 needs the fused prefill attention (TIDELM_INK_PF_FLASH=0 reads bf16 rows)".into());
                }
                f
            },
            draft_on: false,
            draft: None,
            fuse: {
                let on = std::env::var("TIDELM_INK_FUSE").as_deref() != Ok("0");
                unsafe { ink_set_mix_v(on as i32) };
                on
            },
            draft_listed: None,
            taps: DeviceBuffer::zeroed(1)?,
            tap_rows: 0,
            dout: DeviceBuffer::zeroed(DRAFT_OUT)?,
            dall: DeviceBuffer::zeroed(world * DRAFT_OUT)?,
            ctx_hi: vec![0; lanes],
            ctx_lo: vec![0; lanes],
            meta: DeviceBuffer::zeroed(2 * MAX_LANES * 4)?,
            h_meta: tidelm_cuda::PinnedBuffer::zeroed(2 * MAX_LANES * 4)?,
            out_q: crate::inkling::output_query(model_dir, st)?,
            act,
        })
    }

    /// Device bytes held (the boot log).
    pub(crate) fn bytes(&self) -> usize {
        let a = &self.act;
        2 * (self.kv_ring.len() + self.ck.len() + self.cv.len() + self.ca.len() + self.cm.len() + self.snap_kv.len()
             + self.snap_ck.len() + self.snap_cv.len() + self.snap_ca.len() + self.snap_cm.len()
             + a.proj.len() + a.pk.len() + a.pv.len() + a.xin.len() + a.kc.len() + a.vc.len() + a.qc.len() + a.rc.len()
             + a.blocks.len() + a.kvs.len() + a.wp.len() + a.ov.len())
            + 4 * (a.res.len() + a.pm.len() + a.pl.len() + a.pacc.len())
    }
}

impl Engine {
    fn ink(&self) -> &InkState {
        self.ink.as_ref().expect("an Inkling 2 path on another model's engine")
    }

    // ------------------------------------------------------------------ the pass's lanes
    /// The pass's lane groups' lanes (stage(), after the lane table).
    pub(crate) fn ink_stage_meta(&mut self, groups: &[(usize, usize, usize)], mtp: bool) -> Result<()> {
        let base = if mtp { MAX_LANES * 4 } else { 0 };
        let s = self.ink.as_mut().unwrap();
        for (z, &(lane, n, p0)) in groups.iter().enumerate() {
            s.h_meta[base + 4 * z..base + 4 * z + 4].copy_from_slice(&[lane as i32, 0, 0, 0]);
            if !mtp {
                s.ctx_hi[lane] = s.ctx_hi[lane].max(p0 + n);
            }
        }
        unsafe {
            tidelm_cuda::memcpy_async(at(&s.meta, base), (s.h_meta.raw() as *mut i32).add(base) as *const c_void, MAX_LANES * 16,
                                      tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ rewinds and snapshots
    pub(crate) fn ink_can_rewind(&self, lane: usize, pos: usize, n: usize) -> bool {
        let s = self.ink();
        n == pos || n == 0 || (n < pos && pos - n <= REWIND && n >= s.floor[lane])
            || (n < pos && (s.snap_pos[lane] == Some(n) || s.snap_pend[lane] == Some(n)))
    }

    /// The furthest position at most `keep` a set_pos can take the lane back to through its
    /// snapshot (0: none).
    pub(crate) fn ink_restore_point(&self, lane: usize, keep: usize) -> usize {
        let s = self.ink();
        [s.snap_pos[lane], s.snap_pend[lane]].into_iter().flatten().filter(|&p| p <= keep).max().unwrap_or(0)
    }

    /// The lane's rings around position `p` into snapshot `snap` (`save`), or back from it: the
    /// window layers' keys and values of the `window` positions before p, every convolution's last
    /// three inputs.
    fn ink_copy_snapshot(&self, lane: usize, snap: usize, p: usize, save: bool) -> Result<()> {
        let s = self.ink();
        let ink = self.cfg.ink.unwrap();
        let st = self.st.raw();
        let (h, dk, rowk, nl) = (self.cfg.hidden, s.kl * ink.head_dim, s.kl * 2 * ink.head_dim, self.cfg.layers);
        let to_ring = (!save) as i32;
        let w0 = p.saturating_sub(ink.window);
        let c0 = p.saturating_sub(3);
        unsafe {
            ck(ink_ring_copy(st, at(&s.kv_ring, lane * s.ring * rowk), (s.lanes * s.ring * rowk) as i64, s.ring as i32, rowk as i32,
                             w0 as i32, (p - w0) as i32, s.nswa as i32, at(&s.snap_kv, snap * s.nswa * ink.window * rowk),
                             (ink.window * rowk) as i64, to_ring), "snapshot window")?;
            for (ring, snapb, c) in [(&s.ck, &s.snap_ck, dk), (&s.cv, &s.snap_cv, dk), (&s.ca, &s.snap_ca, h), (&s.cm, &s.snap_cm, h)] {
                ck(ink_ring_copy(st, at(ring, lane * CONV_RING * c), (s.lanes * CONV_RING * c) as i64, CONV_RING as i32, c as i32,
                                 c0 as i32, (p - c0) as i32, nl as i32, at(snapb, snap * nl * 3 * c), (3 * c) as i64, to_ring),
                   "snapshot conv")?;
            }
        }
        Ok(())
    }

    /// The lane's state at position `p` into its pending snapshot, when the rings still hold it (the
    /// lane is at `pos`: p == pos, or p inside the pass that just ran); its want is done either way.
    pub(crate) fn ink_take_snapshot(&mut self, lane: usize, pos: usize, p: usize) -> Result<()> {
        let s = self.ink.as_mut().unwrap();
        s.snap_want[lane] = None;
        if p > pos || pos - p > REWIND || p < s.floor[lane] {
            return Ok(());
        }
        let which = 1 - s.snap_cur[lane];
        self.ink_copy_snapshot(lane, 2 * lane + which, p, true)?;
        self.ink.as_mut().unwrap().snap_pend[lane] = Some(p);
        Ok(())
    }

    /// The lane has decoded past its prompt: its pending snapshot becomes the committed one.
    pub(crate) fn ink_commit_snapshot(&mut self, lane: usize) {
        if let Some(s) = self.ink.as_mut() {
            if let Some(p) = s.snap_pend[lane].take() {
                s.snap_cur[lane] ^= 1;
                s.snap_pos[lane] = Some(p);
            }
        }
    }

    /// set_pos on an Inkling 2 lane at `pos`: nothing to do within the rings' reach (or to 0: an
    /// empty sequence reads no earlier position), else the snapshot at `n`.
    pub(crate) fn ink_set_pos(&mut self, lane: usize, pos: usize, n: usize) -> Result<()> {
        let s = self.ink.as_mut().unwrap();
        // a set_pos ends a want its forward did not reach
        s.snap_want[lane] = None;
        // the drafter's context below n: still the sequence's while its (smaller) rings hold it
        if n < pos {
            let held = s.ctx_hi[lane].saturating_sub(s.ring);
            if n == 0 || held >= n {
                s.ctx_lo[lane] = n;
                s.ctx_hi[lane] = n;
            } else {
                s.ctx_lo[lane] = s.ctx_lo[lane].max(held);
            }
        }
        if n == 0 {
            s.snap_pos[lane] = None;
            s.snap_pend[lane] = None;
            s.floor[lane] = 0;
            return Ok(());
        }
        if n == pos {
            return Ok(());
        }
        // snapshots past n held positions that are no longer the lane's
        if s.snap_pos[lane].is_some_and(|p| p > n) {
            s.snap_pos[lane] = None;
        }
        if s.snap_pend[lane].is_some_and(|p| p > n) {
            s.snap_pend[lane] = None;
        }
        if pos - n <= REWIND && n >= s.floor[lane] {
            return Ok(());
        }
        let which = if s.snap_pos[lane] == Some(n) {
            s.snap_cur[lane]
        } else if s.snap_pend[lane] == Some(n) {
            1 - s.snap_cur[lane]
        } else {
            return Err(format!("lane {lane}: Inkling 2's rings cannot go back from {pos} to {n} (snapshots {:?} / {:?})",
                               s.snap_pos[lane], s.snap_pend[lane]).into());
        };
        s.floor[lane] = n;
        self.ink_copy_snapshot(lane, 2 * lane + which, n, false)
    }

    // ------------------------------------------------------------------ the disk prefix store
    /// The position of the lane's snapshot a saved prompt can end at: the committed one, else the
    /// pending one (the store saves a prompt's rows up to it, with the snapshot as the segment's tail).
    pub fn ink_snapshot_pos(&self, lane: usize) -> Option<usize> {
        let s = self.ink.as_ref()?;
        s.snap_pos[lane].or(s.snap_pend[lane])
    }

    /// Snapshot slot `slot`'s buffers as (address, bytes): a segment file's tail.
    fn ink_slot_ranges(&self, slot: usize) -> Vec<(*mut u8, usize)> {
        let s = self.ink();
        let ink = self.cfg.ink.unwrap();
        let (h, dk, rowk, nl) = (self.cfg.hidden, s.kl * ink.head_dim, s.kl * 2 * ink.head_dim, self.cfg.layers);
        let r = |b: &DeviceBuffer<u16>, per: usize| (at(b, slot * per) as *mut u8, per * 2);
        vec![r(&s.snap_kv, s.nswa * ink.window * rowk), r(&s.snap_ck, nl * 3 * dk), r(&s.snap_cv, nl * 3 * dk), r(&s.snap_ca, nl * 3 * h),
             r(&s.snap_cm, nl * 3 * h)]
    }

    /// A segment's tail on its way to disk: the selected lane's snapshot at `end` (an error without
    /// one: the rows alone could not be restored).
    pub(crate) fn ink_tail_save(&self, end: usize) -> Result<Vec<(*mut u8, usize)>> {
        let (s, lane) = (self.ink(), self.lane);
        let which = if s.snap_pos[lane] == Some(end) {
            s.snap_cur[lane]
        } else if s.snap_pend[lane] == Some(end) {
            1 - s.snap_cur[lane]
        } else {
            return Err(format!("lane {lane}: no snapshot at {end} to save with the segment (snapshots {:?} / {:?})", s.snap_pos[lane],
                               s.snap_pend[lane]).into());
        };
        Ok(self.ink_slot_ranges(2 * lane + which))
    }

    /// A segment's tail on its way back: into the selected lane's pending snapshot slot.
    pub(crate) fn ink_tail_load(&self) -> Vec<(*mut u8, usize)> {
        let (s, lane) = (self.ink(), self.lane);
        self.ink_slot_ranges(2 * lane + 1 - s.snap_cur[lane])
    }

    /// After a chain of segments was read (the last one's tail is in the pending slot): the lane's
    /// rings as they were at `n`, that slot its pending snapshot there.
    pub(crate) fn ink_after_disk_load(&mut self, n: usize) -> Result<()> {
        let lane = self.lane;
        let s = self.ink.as_mut().unwrap();
        let which = 1 - s.snap_cur[lane];
        s.snap_pend[lane] = Some(n);
        s.snap_want[lane] = None;
        // the committed snapshot was the lane's old sequence's: the load did not go through set_pos,
        // which would have dropped it where the sequences part
        s.snap_pos[lane] = None;
        s.floor[lane] = n;
        // the drafter's context was another sequence's (the store keeps none of it)
        s.ctx_lo[lane] = n;
        s.ctx_hi[lane] = n;
        self.ink_copy_snapshot(lane, 2 * lane + which, n, false)
    }

    /// The bytes a segment's tail adds to its file on this rank (the store's budget).
    pub fn seg_tail_bytes(&self) -> u64 {
        if self.ink.is_none() {
            return 0;
        }
        self.ink_slot_ranges(0).iter().map(|r| r.1 as u64).sum()
    }

    /// TIDELM_INK_DUMP=PREFIX (checks only: eager passes, never under graph capture): each decoder
    /// pass writes PREFIX.p{first position}.NAME.bin - the tokens and the residual after each layer
    /// below TIDELM_INK_DUMP_MAX (default all) - for tools/inkling_ref.py.
    fn ink_dump(&self, name: &str, ptr: *mut c_void, bytes: usize, layer: Option<usize>) -> Result<()> {
        let Ok(prefix) = std::env::var("TIDELM_INK_DUMP") else { return Ok(()) };
        let max = std::env::var("TIDELM_INK_DUMP_MAX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
        if layer.is_some_and(|l| l >= max) {
            return Ok(());
        }
        self.st.synchronize()?;
        let mut v = vec![0u8; bytes];
        ck(unsafe { tidelm_cuda::ffi::cudaMemcpy(v.as_mut_ptr() as *mut c_void, ptr, bytes, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST) },
           "dump")?;
        std::fs::write(format!("{prefix}.p{}.{name}.bin", self.pos), &v)?;
        Ok(())
    }

    // ------------------------------------------------------------------ the layer
    /// A short convolution over `c` channels of x [t][ldx], its ring `ring` ([layers][lanes][64][c],
    /// layer l's part), then the ring's update with the pass's rows.
    #[allow(clippy::too_many_arguments)]
    fn ink_sconv(&self, l: usize, x: *mut c_void, ldx: usize, c: usize, w: &DeviceBuffer<u16>, dynw: Option<*mut c_void>,
                 ring: &DeviceBuffer<u16>, t: usize, omode: i32, out: *mut c_void) -> Result<()> {
        let (a, s) = (&self.act, self.ink());
        let st = self.st.raw();
        let (lanes, nl, meta) = (np(&a.lanes), self.pass_nl as i32, np(&s.meta));
        let rg = at(ring, l * s.lanes * CONV_RING * c);
        let ls = (CONV_RING * c) as i64;
        let taps = self.cfg.ink.unwrap().conv as f32;
        // Inkling2ShortConvolution divides by sqrt(taps + 1), the dynamic one by sqrt(2 taps + 1)
        let (wmode, dw, div) = match dynw {
            Some(d) => (1, d, (2.0 * taps + 1.0).sqrt()),
            None => (0, std::ptr::null_mut(), (taps + 1.0).sqrt()),
        };
        unsafe {
            if s.fuse && t + 4 <= CONV_RING {
                return ck(ink_conv_fused(st, x, ldx as i64, c as i32, np(w), wmode, dw, ldx as i64, rg, ls, lanes, nl, meta, t as i32, div,
                                         omode, out, c as i64, np(&s.act.res), np(&a.h)), "short conv");
            }
            ck(ink_conv(st, x, ldx as i64, c as i32, np(w), wmode, dw, ldx as i64, rg, ls, lanes, nl, meta, t as i32, div, omode, out,
                        c as i64, np(&s.act.res), np(&a.h)), "short conv")?;
            ck(ink_conv_state(st, x, ldx as i64, c as i32, rg, ls, lanes, nl, meta), "short conv state")
        }
    }

    /// Layer l's output into the residual stream: h = bf16(res + conv_m(the MoE's output in act.a)).
    fn ink_close(&self, l: usize, t: usize) -> Result<()> {
        let w = self.w.layers[l].ink().ok_or("an Inkling 2 layer without its weights")?;
        let h = self.cfg.hidden;
        self.ink_sconv(l, np(&self.act.a), h, h, &w.mlp_conv, None, &self.ink().cm, t, 3, std::ptr::null_mut())?;
        self.ink_dump(&format!("x{l}"), np(&self.act.h), t * h * 2, Some(l))
    }

    /// layer_pre on an Inkling 2 layer: the previous layer closed, the attention, the residual, the
    /// MoE's input in xn.
    pub(crate) fn ink_layer_pre(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        let (cfg, a, s) = (&self.cfg, &self.act, self.ink());
        let ia = &s.act;
        let ink = cfg.ink.unwrap();
        let lw = &self.w.layers[l];
        let w = lw.ink().ok_or("an Inkling 2 layer without its weights")?;
        let st = self.st.raw();
        let h = cfg.hidden;
        if l > 0 {
            self.ink_close(l - 1, t)?;
        } else {
            self.ink_dump("tok", np(&a.tok), t * 4, None)?;
        }
        // the drafter's taps: the residual stream entering layers 5, 11, .., 41
        if s.draft.is_some() && l % 6 == 5 && l / 6 < 7 {
            unsafe {
                tidelm_cuda::memcpy_async(at(&s.taps, (l / 6) * s.tap_rows * h), np(&a.h) as *const c_void, t * h * 2,
                                          tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
            }
        }
        let nb = cfg.ink_blocks_before(l);
        let ldb = (cfg.ink_blocks() * h) as i64;
        let full = cfg.attn[l] == AttnKind::Full;
        // the attention's input(s): layer 0 norms its input, the others mix the block states first
        let x = match &w.q_attn {
            None => {
                self.norm(np(&a.h), false, &lw.ln_in, np(&a.xn), t)?;
                np(&a.xn)
            }
            Some(q) => {
                let (nq, out) = if full { (3, np(&ia.xin)) } else { (1, np(&a.xn)) };
                ck(unsafe { ink_mix(st, np(&ia.blocks), ldb, nb as i32, np(&a.h), 0, np(q), nq, np(&lw.ln_in), cfg.rms_eps, out,
                                    (t * h) as i64, t as i32, h as i32) }, "attention mix")?;
                out
            }
        };
        self.ink_attention(l, w, x, t, prefill)?;
        // res = conv_a(attention) (+ the layer's input); a block boundary saves its input instead
        let boundary = l % ink.block == 0;
        self.ink_sconv(l, np(&a.a), h, h, &w.attn_conv, None, &s.ca, t, if boundary { 1 } else { 2 }, std::ptr::null_mut())?;
        if boundary {
            ck(unsafe { tidelm_cuda::ffi::cudaMemcpy2DAsync(at(&ia.blocks, nb * h), ldb as usize * 2, np(&a.h) as *const c_void, h * 2,
                                                           h * 2, t, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, st) }, "block state")?;
        }
        ck(unsafe { ink_mix(st, np(&ia.blocks), ldb, (nb + boundary as usize) as i32, np(&ia.res), 1, np(&w.q_mlp), 1, np(&lw.ln_post),
                            cfg.rms_eps, np(&a.xn), 0, t as i32, h as i32) }, "MoE mix")
    }

    /// The decoder's end: the last layer closed, the final mix and norm into hn.
    pub(crate) fn ink_final(&self, t: usize) -> Result<()> {
        let (cfg, a, s) = (&self.cfg, &self.act, self.ink());
        self.ink_close(cfg.layers - 1, t)?;
        let h = cfg.hidden;
        ck(unsafe { ink_mix(self.st.raw(), np(&s.act.blocks), (cfg.ink_blocks() * h) as i64, cfg.ink_blocks() as i32, np(&a.h), 0,
                            np(&s.out_q), 1, np(&self.w.norm), cfg.rms_eps, np(&a.hn), 0, t as i32, h as i32) }, "final mix")?;
        // the drafter's last tap (the final normed hidden), then every row's context for it
        if let Some(d) = s.draft.as_ref() {
            unsafe {
                tidelm_cuda::memcpy_async(at(&s.taps, 7 * s.tap_rows * h), np(&a.hn) as *const c_void, t * h * 2,
                                          tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
            }
            d.store_context(&self.st, np(&s.taps), s.tap_rows * h, t, np(&a.lanes), self.pass_nl, np(&s.meta))?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ drafting
    /// Whether this engine drafts with the model's DSpark drafter.
    pub fn has_drafter(&self) -> bool {
        self.ink.as_ref().is_some_and(|s| s.draft_on)
    }

    /// Drafting on (every rank calls it): the drafter is loaded on the pair's last rank, which has
    /// the memory rank 0 lacks and holds the same hidden states; its drafts come back by a gather.
    /// Returns the drafter's device bytes on this rank (0 where it is not).
    pub fn enable_drafter(&mut self, model_dir: &Path, fp8: bool) -> Result<usize> {
        let (h, r, last) = (self.cfg.hidden, self.opt.chunk.max(HEAD_ROWS), self.rank + 1 == self.world);
        let real = self.cfg.ink.ok_or("the DSpark drafter is Inkling 2's")?.real_vocab;
        let ids = crate::inkling_draft::listed_ids(real)?;
        let s = self.ink.as_mut().ok_or("the DSpark drafter is Inkling 2's")?;
        s.draft_on = true;
        s.draft_listed = ids.as_ref().map(|ids| {
            let mut m = vec![false; real];
            for &i in ids {
                m[i as usize] = true;
            }
            m
        });
        if !last {
            return Ok(0);
        }
        let d = crate::inkling_draft::InkDraft::open_with(model_dir, &self.st, crate::inkling_draft::DraftOpts {
            fp8, lanes: s.lanes, local_ring: s.ring, global_ring: DRAFT_GLOBAL_RING, max_rows: r }, ids.as_deref())?;
        let bytes = d.device_bytes() + 8 * r * h * 2;
        s.taps = DeviceBuffer::zeroed(8 * r * h)?;
        s.tap_rows = r;
        s.draft = Some(d);
        Ok(bytes)
    }

    /// One draft round for `lane` (rank 0): its anchor at position `p` -> the drafter's 7 tokens and
    /// each one's probability under the drafter (the gate's confidence). `listed`: over the listed
    /// vocabulary's head (the scheduler's choice for the lane; ignored without a list).
    pub fn ink_draft(&mut self, lane: usize, anchor: u32, p: usize, listed: bool) -> Result<(Vec<u32>, Vec<f32>)> {
        self.mirror_msg(|m| {
            m.u32(OP_INK_DRAFT).u32(lane as u32).u32(anchor).u32(p as u32).u32(listed as u32);
        })?;
        self.ink_draft_local(lane, anchor, p, listed)?;
        self.st.synchronize()?;
        let b = self.ink().dout.download()?;
        let w = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        Ok(((0..7).map(w).collect(), (7..14).map(|i| f32::from_bits(w(i))).collect()))
    }

    /// A draft round on this rank: drafted here when this rank has the drafter, the drafting rank's
    /// result through a gather otherwise.
    pub(crate) fn ink_draft_local(&mut self, lane: usize, anchor: u32, p: usize, listed: bool) -> Result<()> {
        let s = self.ink();
        let out = np(&s.dout);
        match s.draft.as_ref() {
            Some(d) => {
                let crate::weights::Embed::Bf16(table) = &self.w.embed else { return Err("the drafter needs the BF16 embedding".into()) };
                let probs = unsafe { (out as *mut u8).add(28) } as *mut c_void;
                d.draft(&self.st, lane, anchor, p, s.ctx_lo[lane], np(table), out, probs, listed)?;
            }
            None => ck(unsafe { tidelm_cuda::ffi::cudaMemsetAsync(out, 0, DRAFT_OUT, self.st.raw()) }, "draft out")?,
        }
        if self.world > 1 {
            self.gather(out, DRAFT_OUT, np(&s.dall), false)?;
            unsafe {
                tidelm_cuda::memcpy_async(out, at(&s.dall, (self.world - 1) * DRAFT_OUT), DRAFT_OUT, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE,
                                          &self.st)?;
            }
        }
        Ok(())
    }

    /// Rank 1's side of OP_INK_DRAFT.
    pub(crate) fn follow_ink_draft(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        let (lane, anchor, p, listed) = (r.u32()? as usize, r.u32()?, r.u32()? as usize, r.u32()? != 0);
        self.ink_draft_local(lane, anchor, p, listed)
    }

    /// The embedding rows of the staged ids (the table holds them normed) -> act.h.
    pub(crate) fn embed_ink(&self, t: usize, out: *mut c_void) -> Result<()> {
        let crate::weights::Embed::Bf16(table) = &self.w.embed else { return Err("embed_ink on an INT8 table".into()) };
        ck(unsafe { tidelm_kernels::flash::flash_embed(self.st.raw(), np(&self.act.tok), t as i32, np(table), self.cfg.hidden as i32, 1,
                                                       std::ptr::null(), out) }, "embed")?;
        // the lane's image rows: the tower's features in place of the placeholder's embedding
        // (Inkling2Model's masked_scatter); only in its own eager prefill passes (forward_local)
        let Some(v) = self.vis.as_ref().map(|v| &v.lanes[self.lane]).filter(|v| self.pass_nl == 1 && !v.spans.is_empty()) else {
            return Ok(());
        };
        let Some(feats) = v.feats.as_ref() else { return Err("image spans without features".into()) };
        let h = self.cfg.hidden;
        let (p0, p1) = (self.pos, self.pos + t);
        for &(s, n, row) in &v.spans {
            let (a0, a1) = (s.max(p0), (s + n).min(p1));
            if a0 >= a1 {
                continue;
            }
            let dst = unsafe { (out as *mut u16).add((a0 - p0) * h) } as *mut c_void;
            unsafe {
                tidelm_cuda::memcpy_async(dst, at(feats, (row + a0 - s) * h), (a1 - a0) * h * 2, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE,
                                          &self.st)?;
            }
        }
        Ok(())
    }

    /// The head's logits [t][vl] as the model defines them: softcapped, the padding ids never.
    pub(crate) fn ink_logits_fix(&self, logits: *mut c_void, ld: usize, t: usize) -> Result<()> {
        let ink = self.cfg.ink.unwrap();
        ck(unsafe { ink_logits(self.st.raw(), logits, ld as i64, self.vl as i32, t as i32, ink.logit_cap, (self.rank * self.vl) as i32,
                               ink.real_vocab as i32) }, "logit softcap")
    }

    // ------------------------------------------------------------------ attention
    /// Layer l's attention over `t` rows of its input(s) at `x` (a window layer's normed input
    /// [t][H]; a full layer's q, k, v inputs [3][t][H]) -> act.a.
    fn ink_attention(&self, l: usize, w: &InkW, x: *mut c_void, t: usize, prefill: bool) -> Result<()> {
        let (cfg, a, s) = (&self.cfg, &self.act, self.ink());
        let ia = &s.act;
        let ink = cfg.ink.unwrap();
        let st = self.st.raw();
        let (h, hl, kl, d) = (cfg.hidden, s.hl, s.kl, ink.head_dim);
        let (dk, rowk) = (kl * d, kl * 2 * d);
        let (lanes, nl, meta) = (np(&a.lanes), self.pass_nl as i32, np(&s.meta));
        let full = cfg.attn[l] == AttnKind::Full;
        let off = w.off;
        // the projections
        self.lin(&w.proj, x, h, t, np(&ia.proj), true)?;
        let ldp = w.proj.n_total();
        if let (Some(pk), Some(pv)) = (&w.proj_k, &w.proj_v) {
            let ldk = pk.n_total();
            let xk = unsafe { (x as *mut u16).add(t * h) } as *mut c_void;
            let xv = unsafe { (x as *mut u16).add(2 * t * h) } as *mut c_void;
            self.lin(pk, xk, h, t, np(&ia.pk), true)?;
            self.lin(pv, xv, h, t, np(&ia.pv), true)?;
            // the dynamic convolutions: static taps by lag plus the projected per-token ones
            self.ink_sconv(l, np(&ia.pk), ldk, dk, &w.k_conv, Some(at(&ia.pk, off.taps)), &s.ck, t, 0, np(&ia.kc))?;
            self.ink_sconv(l, np(&ia.pv), ldk, dk, &w.v_conv, Some(at(&ia.pv, off.taps)), &s.cv, t, 0, np(&ia.vc))?;
        } else {
            self.ink_sconv(l, at(&ia.proj, off.k), ldp, dk, &w.k_conv, None, &s.ck, t, 0, np(&ia.kc))?;
            self.ink_sconv(l, at(&ia.proj, off.v), ldp, dk, &w.v_conv, None, &s.cv, t, 0, np(&ia.vc))?;
        }
        // the cache: a window layer's ring, a full layer's paged rows
        let (cache, ring, lane_rows, window, extent, alpha) = match s.swa_of[l] {
            Some(i) => (at(&s.kv_ring, i * s.lanes * s.ring * rowk), s.ring, s.ring, ink.window, ink.window, 0.0),
            None => {
                let kv = self.kv[l].as_ref().ok_or("a full-attention layer without a cache")?;
                (kv.cq.raw(), 0, 0, 0, ink.rel_extent, ink.log_alpha)
            }
        };
        debug_assert_eq!(full, s.swa_of[l].is_none());
        // a full layer's rows in FP8 (GLM_INK_KV); the window layers' rings are bf16
        let kvf = (full && s.kvf) as i32;
        ck(unsafe { ink_heads(st, np(&ia.proj), ldp as i64, off.q as i32, off.r as i32, np(&w.q_scale), np(&w.r_scale), np(&ia.kc),
                              np(&ia.vc), hl as i32, kl as i32, ink.qk_clip, ink.rel_clip, cfg.rms_eps, alpha, ink.log_floor, lanes,
                              nl, meta, ring as i32, lane_rows as i64, t as i32, np(&ia.qc), np(&ia.rc), cache, kvf) }, "heads")?;
        if !prefill {
            let ns = if full { NS_FULL } else { NS_WINDOW };
            // a full layer with a lane of several rows (a verify pass): the split-key fused kernel,
            // which reads each key once for all of the lane's rows. ink_attn_part reads it once a row:
            // at 140K keys 1.24 ms a row a layer, so 40.5 / 77.4 ms a pass's 8 layers at 4 / 8 rows,
            // against the fused 10.9 / 10.8 (fused_decode_attention_time, 2026-11-16); one row a lane
            // keeps ink_attn_part (1.24 against 1.35 ms). TIDELM_INK_FLASH_DECODE=0: ink_attn_part always.
            let part = if full && t > nl as usize && dec_flash() { ink_flash_part } else { ink_attn_part };
            unsafe {
                ck(part(st, np(&ia.qc), np(&ia.rc), np(&w.proj_t), extent as i32, cache, ring as i32, lane_rows as i64,
                                 window as i32, lanes, nl, meta, hl as i32, kl as i32, ns as i32, t as i32, np(&ia.pm), np(&ia.pl),
                                 np(&ia.pacc), kvf), "attention")?;
                ck(ink_attn_combine(st, np(&ia.pm), np(&ia.pl), np(&ia.pacc), np(&ia.proj), ldp as i64, off.sink as i32,
                                    off.gate as i32, hl as i32, kl as i32, ns as i32, t as i32, np(&a.o)), "attention combine")?;
            }
        } else {
            self.ink_attention_prefill(w, cache, ring, lane_rows, window, extent, ldp, t)?;
        }
        self.lin(&w.o, np(&a.o), hl * d, t, np(&a.f32buf), false)?;
        self.allreduce(np(&a.f32buf), t * h, np(&a.a), prefill)
    }

    /// Prefill rows (one lane, from self.pos): row blocks through cuBLAS - scores = q k^T a query
    /// head, the weights (bias, mask, sink, softmax) in bf16, their product with the values.
    #[allow(clippy::too_many_arguments)]
    fn ink_attention_prefill(&self, w: &InkW, cache: *mut c_void, ring: usize, lane_rows: usize, window: usize, extent: usize,
                             ldp: usize, t: usize) -> Result<()> {
        let (a, s) = (&self.act, self.ink());
        let ia = &s.act;
        let ink = self.cfg.ink.unwrap();
        let st = self.st.raw();
        let (hl, kl, d) = (s.hl, s.kl, ink.head_dim);
        let rowk = kl * 2 * d;
        let p0 = self.pos;
        let off = w.off;
        // a full layer: the fused kernel over the lane's flat cache. The cuBLAS path below sized its
        // row blocks to the score buffer, a few rows deep in a context, and read every key again for
        // each block and head: 2,300 rows at 150K took 40-60 s a turn on the pair (2026-11-16), the
        // fused kernel 1.4-2.4 s for the 8 layers (fused_prefill_attention_time). TIDELM_INK_PF_FLASH=0
        // keeps the cuBLAS path (boot-guarded: the two round differently).
        if window == 0 && ring == 0 && pf_flash() {
            let rowb = crate::inkling::kv_row_bytes(kl, d, s.kvf);
            let kv = unsafe { (cache as *mut u8).add(self.lane * self.dec_lane_rows * rowb) as *const c_void };
            return ck(unsafe { ink_pf_flash(st, np(&ia.qc), np(&ia.rc), np(&w.proj_t), extent as i32, kv, rowb as i64, 0, np(&ia.proj),
                                            ldp as i64, off.sink as i32, off.gate as i32, hl as i32, kl as i32, t as i32, p0 as i32, 0,
                                            np(&a.o), s.kvf as i32) }, "prefill attention");
        }
        let budget = a.scores.len().min(ia.wp.len());
        // rows a block: its scores [hl][rb][keys] fit the buffers (keys: the block's rows and the
        // window before them, or every position up to the block's last)
        let keys = |rb: usize, r0: usize| if window > 0 { (p0 + r0 + rb).min(rb + window - 1) } else { p0 + r0 + rb };
        let mut r0 = 0;
        while r0 < t {
            let mut rb = (t - r0).min(ia.ov.len() / (hl * d));
            while rb > 1 && hl * rb * keys(rb, r0) > budget {
                rb = rb.div_ceil(2);
            }
            let last = p0 + r0 + rb - 1;
            let lo = if window > 0 { (p0 + r0 + 1).saturating_sub(window) } else { 0 };
            let sk = last - lo + 1;
            if hl * rb * sk > budget {
                return Err(format!("Inkling 2 prefill: one row's {sk} keys x {hl} heads past the score buffer").into());
            }
            // the block's keys and values, contiguous: out of the ring, or where the flat cache has them
            let kvs = if ring > 0 {
                ck(unsafe { ink_kv_window(st, cache, ring as i32, lane_rows as i64, np(&a.lanes), np(&s.meta), 0, lo as i32, sk as i32,
                                          kl as i32, np(&ia.kvs)) }, "window keys")?;
                np(&ia.kvs)
            } else {
                unsafe { (cache as *mut u16).add((self.lane * self.dec_lane_rows + lo) * rowk) as *mut c_void }
            };
            let el = |p: *mut c_void, n: usize| unsafe { (p as *mut u16).add(n) as *mut c_void };
            for hq in 0..hl {
                let g = hq / (hl / kl);
                self.blas.gemm_rm(rb, sk, d, Mat { ptr: at(&ia.qc, (r0 * hl + hq) * d), ty: R_16BF, ld: (hl * d) as i64 },
                                  Mat { ptr: el(kvs, g * 2 * d), ty: R_16BF, ld: rowk as i64 }, true, at(&a.scores, hq * rb * sk),
                                  tidelm_kernels::blas::R_32F, sk as i64)?;
            }
            ck(unsafe { ink_pf_softmax(st, np(&a.scores), np(&ia.rc), np(&w.proj_t), extent as i32, np(&ia.proj), ldp as i64,
                                       off.sink as i32, hl as i32, r0 as i32, rb as i32, sk as i32, lo as i32, p0 as i32, window as i32,
                                       np(&ia.wp)) }, "prefill softmax")?;
            for hq in 0..hl {
                let g = hq / (hl / kl);
                self.blas.gemm_rm(rb, d, sk, Mat { ptr: at(&ia.wp, hq * rb * sk), ty: R_16BF, ld: sk as i64 },
                                  Mat { ptr: el(kvs, g * 2 * d + d), ty: R_16BF, ld: rowk as i64 }, false, at(&ia.ov, hq * rb * d),
                                  R_16BF, d as i64)?;
            }
            ck(unsafe { ink_pf_gate(st, np(&ia.ov), np(&ia.proj), ldp as i64, off.gate as i32, hl as i32, r0 as i32, rb as i32, np(&a.o)) },
               "prefill gate")?;
            r0 += rb;
        }
        Ok(())
    }
}
