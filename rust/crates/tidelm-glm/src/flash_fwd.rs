//! GLM-5.3-Flash's layers on the GLM engine (a child module of forward: the engine's own state).
//!
//! A decoder layer is two mHC sites (ds41.cu's kernels, in the reference's order: a site's mixes
//! from the stream before its sublayer, its pre weights the input, its post and comb weights the
//! stream after), around KDA or the sparse attention, then the MLP / MoE that GLM-5.3 runs. The
//! MTP layer keeps the plain residual (it has no hyper-connections). As GLM-5.3 defers a layer's
//! MLP residual add into the next norm, a layer's MLP site is closed (hc_post) at the next layer's
//! start or before the final norm, so the decode pass's segments split where they always did.
//!
//! KDA's state is a lane's two slots (flash.cu): stage() names each lane's read and write slot,
//! a decoder pass flips the lane to the slot it wrote (flash_commit), and set_pos inside the lane's
//! last pass replays the kept rows from the slot it read (flash_set_pos).
//!
//! A lane also keeps a snapshot of its state near its last prompt's end (Engine::want_snapshot,
//! taken at a prefill chunk's edge or by a replay), so a set_pos there - a chat's follow-up, whose
//! template drops the last answer's reasoning and so leaves the lane's sequence where that answer
//! began, or a regenerate - restores it instead of prefilling from 0. Two snapshot slots a lane:
//! the committed one, and the one this request's prefill takes, committed when it decodes (a
//! two-pass prefill goes back to the committed one between its passes).

use super::*;
use tidelm_kernels::flash::*;

use crate::weights::{Attn, DsaW, HcW, KdaW};

/// A token row's attended positions, padded (glm.cu's MLA takes K picks, -1 after them).
const IDX_LD: usize = 2056;
/// Pools the indexer keeps a row.
const POOL_TOPK: usize = 512;

pub(crate) struct FlashAct {
    /// a KDA layer's fused projection [r][q | k | v | f_a | g_a | b], or a sparse layer's proj_a
    proj: DeviceBuffer<u16>,
    ldp: usize,
    /// KDA: the conv's output [r][3 dl], f_b / g_b [r][dl], the core's output [r][dl]
    act: DeviceBuffer<u16>,
    f: DeviceBuffer<u16>,
    gt: DeviceBuffer<u16>,
    y: DeviceBuffer<u16>,
    /// mHC: a site's mixed stream [r][H], mixes [r][24], the stream's rsqrt [r], each site's
    /// pre / post [r][4] and comb [r][4][4] (attention, MLP)
    hy: DeviceBuffer<u16>,
    mixes: DeviceBuffer<f32>,
    rsq: DeviceBuffer<f32>,
    site: [[DeviceBuffer<f32>; 3]; 2],
    /// sparse attention: proj_b [r][q_b | wq_b], the indexer query [r][32 x 128], its keys after the
    /// LayerNorm [r][128], head weights [r][32] fp32, the selected pools [r][512], the tokens [r][IDX_LD]
    pb: DeviceBuffer<u16>,
    qi: DeviceBuffer<u16>,
    kln: DeviceBuffer<u16>,
    iw: DeviceBuffer<f32>,
    pools: DeviceBuffer<i32>,
    idx: DeviceBuffer<i32>,
    /// kv_b dequantized for prefill's absorbs [hl * 512][512]
    kvb_bf: DeviceBuffer<u16>,
}

pub(crate) struct FlashState {
    /// layer -> its place among the KDA layers / the sparse layers (the rings)
    kda_of: Vec<Option<usize>>,
    dsa_of: Vec<Option<usize>>,
    nkda: usize,
    /// KDA heads and dims on this rank
    khl: usize,
    dl: usize,
    /// [2 lanes][KDA layers][khl][128][128] fp32; [2 lanes][KDA layers][3][3 dl] bf16
    state: DeviceBuffer<f32>,
    conv: DeviceBuffer<u16>,
    /// a lane's last pass's rows for its rollback: [lanes][KDA layers][SAVE_ROWS][khl][128] (k, g fp32,
    /// v bf16), [..][khl] beta, [..][3 dl] the raw conv rows
    save_k: DeviceBuffer<f32>,
    save_g: DeviceBuffer<f32>,
    save_v: DeviceBuffer<u16>,
    save_b: DeviceBuffer<f32>,
    save_raw: DeviceBuffer<u16>,
    /// the sparse layers' K | gate rings [sparse layers][lanes][64][256]
    ring: DeviceBuffer<u16>,
    lanes: usize,
    nd: usize,
    /// prompt-end snapshots: slots 2 lanes + 2 lane + 0 / 1 of `state` and `conv`, and their rings
    /// [2 lanes][sparse layers][64][256]; per lane the committed one (0 / 1) and its position, the
    /// pending one's position, and the position its next forward is to snapshot
    snap_ring: DeviceBuffer<u16>,
    snap_cur: Vec<usize>,
    snap_pos: Vec<Option<usize>>,
    snap_pend: Vec<Option<usize>>,
    pub(crate) snap_want: Vec<Option<usize>>,
    /// the pass's [2][MAX_LANES][4] (decoder, MTP): lane, KDA slot read, slot written, pool offset
    meta: DeviceBuffer<i32>,
    h_meta: tidelm_cuda::PinnedBuffer<i32>,
    /// a lane's committed slot (0 / 1) and its last decoder pass (first position, rows, saved)
    parity: Vec<usize>,
    last: Vec<Option<(usize, usize, bool)>>,
    pub(crate) act: FlashAct,
}

/// Rows a pass saves for its rollback (a verify pass, a copy round): HEAD_ROWS.
const SAVE_ROWS: usize = HEAD_ROWS;

impl FlashState {
    pub(crate) fn new(cfg: &GlmConfig, opt: &Options, world: usize, lanes: usize) -> Result<Self> {
        let kda = cfg.kda.ok_or("no KDA config")?;
        let khl = kda.heads / world;
        let dl = khl * kda.dim;
        let c = 3 * dl;
        let mut kda_of = vec![None; cfg.n_layers()];
        let mut dsa_of = vec![None; cfg.n_layers()];
        let (mut nk, mut nd) = (0, 0);
        for l in 0..cfg.n_layers() {
            match cfg.attn[l] {
                AttnKind::Kda => {
                    kda_of[l] = Some(nk);
                    nk += 1;
                }
                _ => {
                    dsa_of[l] = Some(nd);
                    nd += 1;
                }
            }
        }
        let r = opt.chunk.max(HEAD_ROWS);
        let hl = cfg.heads / world;
        let h = cfg.hidden;
        let z = |n: usize| -> Result<DeviceBuffer<u16>> { Ok(DeviceBuffer::new(n)?) };
        let f = |n: usize| -> Result<DeviceBuffer<f32>> { Ok(DeviceBuffer::new(n)?) };
        let ldp = KdaW::offsets(dl, khl)[6].max(DsaW::OFFSETS_A[5]);
        let ldb = hl * cfg.qk_dim() + cfg.index_heads * cfg.index_dim;
        let site = || -> Result<[DeviceBuffer<f32>; 3]> { Ok([f(r * 4)?, f(r * 4)?, f(r * 16)?]) };
        let act = FlashAct {
            proj: z(r * ldp)?,
            ldp,
            act: z(r * c)?,
            f: z(r * dl)?,
            gt: z(r * dl)?,
            y: z(r * dl)?,
            hy: z(r * h)?,
            mixes: f(r * 24)?,
            rsq: f(r)?,
            site: [site()?, site()?],
            pb: z(r * ldb)?,
            qi: z(r * cfg.index_heads * cfg.index_dim)?,
            kln: z(r * cfg.index_dim)?,
            iw: f(r * cfg.index_heads)?,
            pools: DeviceBuffer::new(r * POOL_TOPK)?,
            idx: DeviceBuffer::new(r * IDX_LD)?,
            kvb_bf: z(hl * cfg.kvb_rows() * cfg.kv_lora)?,
        };
        // two working slots a lane, then two snapshot slots a lane
        let slots = 4 * lanes;
        Ok(FlashState {
            kda_of,
            dsa_of,
            nkda: nk,
            khl,
            dl,
            state: DeviceBuffer::zeroed(slots * nk * khl * 128 * 128)?,
            conv: DeviceBuffer::zeroed(slots * nk * 3 * c)?,
            save_k: DeviceBuffer::zeroed(lanes * nk * SAVE_ROWS * dl)?,
            save_g: DeviceBuffer::zeroed(lanes * nk * SAVE_ROWS * dl)?,
            save_v: DeviceBuffer::zeroed(lanes * nk * SAVE_ROWS * dl)?,
            save_b: DeviceBuffer::zeroed(lanes * nk * SAVE_ROWS * khl)?,
            save_raw: DeviceBuffer::zeroed(lanes * nk * SAVE_ROWS * c)?,
            ring: DeviceBuffer::zeroed(nd * lanes * 64 * 256)?,
            lanes,
            nd,
            snap_ring: DeviceBuffer::zeroed(2 * lanes * nd * 64 * 256)?,
            snap_cur: vec![0; lanes],
            snap_pos: vec![None; lanes],
            snap_pend: vec![None; lanes],
            snap_want: vec![None; lanes],
            meta: DeviceBuffer::zeroed(2 * MAX_LANES * 4)?,
            h_meta: tidelm_cuda::PinnedBuffer::zeroed(2 * MAX_LANES * 4)?,
            parity: vec![0; lanes],
            last: vec![None; lanes],
            act,
        })
    }

    /// Device bytes held (the boot log).
    pub(crate) fn bytes(&self) -> usize {
        4 * (self.state.len() + self.save_k.len() + self.save_g.len() + self.save_b.len())
            + 2 * (self.conv.len() + self.save_v.len() + self.save_raw.len() + self.ring.len() + self.snap_ring.len())
    }

    /// Lane `lane`'s snapshot slot `which` (0 / 1) in `state` / `conv`.
    fn snap_slot(&self, lane: usize, which: usize) -> usize {
        2 * self.lanes + 2 * lane + which
    }
}

impl Engine {
    fn fl(&self) -> &FlashState {
        self.fl.as_ref().expect("a GLM-5.3-Flash path on a GLM-5.3 engine")
    }

    // ------------------------------------------------------------------ the pass's lanes
    /// The pass's per-group KDA slots and pool offsets (stage(), after the lane table): group z of
    /// `groups` reads its lane's committed slot and writes the other.
    pub(crate) fn flash_stage_meta(&mut self, groups: &[(usize, usize, usize)], mtp: bool) -> Result<()> {
        let stride = if mtp { self.mtp_lane_rows } else { self.dec_lane_rows };
        let base = if mtp { MAX_LANES * 4 } else { 0 };
        let fl = self.fl.as_mut().unwrap();
        for (z, &(lane, _, _)) in groups.iter().enumerate() {
            let p = fl.parity[lane];
            fl.h_meta[base + 4 * z..base + 4 * z + 4]
                .copy_from_slice(&[lane as i32, (2 * lane + p) as i32, (2 * lane + 1 - p) as i32, (lane * stride) as i32]);
        }
        unsafe {
            tidelm_cuda::memcpy_async(at(&fl.meta, base), (fl.h_meta.raw() as *mut i32).add(base) as *const c_void, MAX_LANES * 16,
                                      tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
        }
        Ok(())
    }

    /// After a decoder pass over `groups`: each lane's state is the slot the pass wrote; its rows
    /// can be rolled back (set_pos) while the pass saved them.
    pub(crate) fn flash_commit(&mut self, groups: &[(usize, usize, usize)]) {
        if let Some(fl) = self.fl.as_mut() {
            for &(lane, n, p0) in groups {
                fl.parity[lane] ^= 1;
                fl.last[lane] = Some((p0, n, n <= SAVE_ROWS));
            }
        }
    }

    pub(crate) fn flash_can_rewind(&self, lane: usize, pos: usize, n: usize) -> bool {
        let fl = self.fl();
        n == pos || n == 0 || fl.last[lane].is_some_and(|(p0, rows, saved)| saved && n >= p0 && n < p0 + rows)
            || (n < pos && (fl.snap_pos[lane] == Some(n) || fl.snap_pend[lane] == Some(n)))
    }

    /// The furthest position at most `keep` a set_pos can take the lane back to through its
    /// snapshot (0: none).
    pub(crate) fn flash_restore_point(&self, lane: usize, keep: usize) -> usize {
        let fl = self.fl();
        [fl.snap_pos[lane], fl.snap_pend[lane]].into_iter().flatten().filter(|&s| s <= keep).max().unwrap_or(0)
    }

    /// Copy slot `from`'s KDA state and conv rows to slot `to`.
    fn flash_copy_slot(&self, from: usize, to: usize) -> Result<()> {
        let fl = self.fl();
        let (slot_f, slot_c) = (fl.nkda * fl.khl * 128 * 128, fl.nkda * 3 * 3 * fl.dl);
        unsafe {
            tidelm_cuda::memcpy_async(at(&fl.state, to * slot_f), at(&fl.state, from * slot_f), slot_f * 4,
                                      tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
            tidelm_cuda::memcpy_async(at(&fl.conv, to * slot_c), at(&fl.conv, from * slot_c), slot_c * 2,
                                      tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, &self.st)?;
        }
        Ok(())
    }

    /// The lane's sparse-layer rings into snapshot ring `snap` (`save`), or back from it.
    fn flash_copy_ring(&self, lane: usize, snap: usize, save: bool) -> Result<()> {
        let fl = self.fl();
        let r = 64 * 256;
        let (lane_at, snap_at) = (at(&fl.ring, lane * r), at(&fl.snap_ring, snap * fl.nd * r));
        let (lane_pitch, snap_pitch) = (fl.lanes * r * 2, r * 2);
        let (dst, dp, src, sp) = if save { (snap_at, snap_pitch, lane_at, lane_pitch) } else { (lane_at, lane_pitch, snap_at, snap_pitch) };
        ck(unsafe { tidelm_cuda::ffi::cudaMemcpy2DAsync(dst, dp, src, sp, r * 2, fl.nd, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE,
                                                       self.st.raw()) }, "kda snapshot ring")
    }

    /// The lane's state at position `p` into its pending snapshot: the committed slot when the
    /// lane is at `p` (`pos`), else a replay of its last pass's kept rows when that pass saved them
    /// and holds `p`; out of reach, no snapshot. Its want is done either way.
    pub(crate) fn flash_take_snapshot(&mut self, lane: usize, pos: usize, p: usize) -> Result<()> {
        let fl = self.fl.as_mut().unwrap();
        fl.snap_want[lane] = None;
        let which = 1 - fl.snap_cur[lane];
        let dst = fl.snap_slot(lane, which);
        let (cur, prev) = (2 * lane + fl.parity[lane], 2 * lane + 1 - fl.parity[lane]);
        if p == pos {
            self.flash_copy_slot(cur, dst)?;
        } else if let Some((p0, _, _)) = fl.last[lane].filter(|&(p0, rows, saved)| saved && p >= p0 && p < p0 + rows) {
            let st = self.st.raw();
            let (nk, khl, c) = (fl.nkda, fl.khl, 3 * fl.dl);
            let (slot_f, slot_c) = (nk * khl * 128 * 128, nk * 3 * c);
            let lk = nk * SAVE_ROWS * fl.dl;
            unsafe {
                ck(flash_kda_replay(st, fl.state.raw(), (khl * 128 * 128) as i64, slot_f as i64, nk as i32, prev as i32, dst as i32,
                                    at(&fl.save_k, lane * lk), at(&fl.save_g, lane * lk), at(&fl.save_v, lane * lk),
                                    at(&fl.save_b, lane * nk * SAVE_ROWS * khl), (SAVE_ROWS * khl * 128) as i64, (p - p0) as i32,
                                    khl as i32), "kda snapshot replay")?;
                ck(flash_kda_conv_replay(st, fl.conv.raw(), (3 * c) as i64, slot_c as i64, c as i32, nk as i32, prev as i32, dst as i32,
                                         at(&fl.save_raw, lane * nk * SAVE_ROWS * c), (SAVE_ROWS * c) as i64, (p - p0) as i32),
                   "kda snapshot conv replay")?;
            }
        } else {
            return Ok(());
        }
        // the rings' rows past p are rewritten by whatever prefills from p before they are read
        self.flash_copy_ring(lane, 2 * lane + which, true)?;
        self.fl.as_mut().unwrap().snap_pend[lane] = Some(p);
        Ok(())
    }

    /// The lane has decoded past its prompt: its pending snapshot becomes the committed one.
    pub(crate) fn flash_commit_snapshot(&mut self, lane: usize) {
        if let Some(fl) = self.fl.as_mut() {
            if let Some(p) = fl.snap_pend[lane].take() {
                fl.snap_cur[lane] ^= 1;
                fl.snap_pos[lane] = Some(p);
            }
        }
    }

    /// set_pos on a GLM-5.3-Flash lane at `pos`: to 0 a fresh state, inside the lane's last pass a
    /// replay of its kept rows, anything else is out of KDA's reach (no saved state).
    pub(crate) fn flash_set_pos(&mut self, lane: usize, pos: usize, n: usize) -> Result<()> {
        let st = self.st.raw();
        let fl = self.fl.as_mut().unwrap();
        let (nk, khl, c) = (fl.nkda, fl.khl, 3 * fl.dl);
        let slot_f = nk * khl * 128 * 128;
        let slot_c = nk * 3 * c;
        // a set_pos ends a want its forward did not reach
        fl.snap_want[lane] = None;
        if n == 0 {
            // always, at 0 too: a lane emptied any other way (kv_shrink) must not keep its old state
            unsafe {
                ck(tidelm_cuda::ffi::cudaMemsetAsync(at(&fl.state, 2 * lane * slot_f), 0, 2 * slot_f * 4, st), "kda state reset")?;
                ck(tidelm_cuda::ffi::cudaMemsetAsync(at(&fl.conv, 2 * lane * slot_c), 0, 2 * slot_c * 2, st), "kda conv reset")?;
            }
            fl.parity[lane] = 0;
            fl.last[lane] = None;
            fl.snap_pos[lane] = None;
            fl.snap_pend[lane] = None;
            return Ok(());
        }
        if n == pos {
            return Ok(());
        }
        // snapshots past n held positions that are no longer the lane's
        if fl.snap_pos[lane].is_some_and(|s| s > n) {
            fl.snap_pos[lane] = None;
        }
        if fl.snap_pend[lane].is_some_and(|s| s > n) {
            fl.snap_pend[lane] = None;
        }
        let in_pass = fl.last[lane].filter(|&(p0, rows, saved)| saved && n >= p0 && n < p0 + rows);
        let Some((p0, _, _)) = in_pass else {
            let which = if fl.snap_pos[lane] == Some(n) {
                fl.snap_cur[lane]
            } else if fl.snap_pend[lane] == Some(n) {
                1 - fl.snap_cur[lane]
            } else {
                let last = fl.last[lane].map_or("none".to_string(), |(p0, rows, _)| format!("{rows} rows from {p0}"));
                return Err(format!("lane {lane}: KDA's state cannot go back from {pos} to {n} (its last pass: {last}; snapshots \
                                    {:?} / {:?})", fl.snap_pos[lane], fl.snap_pend[lane]).into());
            };
            let (from, to) = (fl.snap_slot(lane, which), 2 * lane + fl.parity[lane]);
            fl.last[lane] = None;
            self.flash_copy_slot(from, to)?;
            return self.flash_copy_ring(lane, 2 * lane + which, false);
        };
        let keep = n - p0;
        let cur = 2 * lane + fl.parity[lane];
        let prev = 2 * lane + 1 - fl.parity[lane];
        let lk = nk * SAVE_ROWS * fl.dl;
        unsafe {
            ck(flash_kda_replay(st, fl.state.raw(), (khl * 128 * 128) as i64, slot_f as i64, nk as i32, prev as i32, cur as i32,
                                at(&fl.save_k, lane * lk), at(&fl.save_g, lane * lk), at(&fl.save_v, lane * lk),
                                at(&fl.save_b, lane * nk * SAVE_ROWS * khl), (SAVE_ROWS * khl * 128) as i64, keep as i32,
                                khl as i32), "kda replay")?;
            ck(flash_kda_conv_replay(st, fl.conv.raw(), (3 * c) as i64, slot_c as i64, c as i32, nk as i32, prev as i32, cur as i32,
                                     at(&fl.save_raw, lane * nk * SAVE_ROWS * c), (SAVE_ROWS * c) as i64, keep as i32),
               "kda conv replay")?;
        }
        fl.last[lane] = None;
        Ok(())
    }

    /// TIDELM_FLASH_DUMP=PREFIX (checks only: eager passes, never under graph capture): each decoder
    /// pass writes `bytes` at `ptr` to PREFIX.p{first position}.NAME.bin - the tokens and the mHC
    /// streams after each layer below TIDELM_FLASH_DUMP_MAX (default all) - for tools/flash_ref.py
    /// to compare with transformers' glm5_next.
    fn dump(&self, name: &str, ptr: *mut c_void, bytes: usize, layer: Option<usize>) -> Result<()> {
        let Ok(prefix) = std::env::var("TIDELM_FLASH_DUMP") else { return Ok(()) };
        let max = std::env::var("TIDELM_FLASH_DUMP_MAX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
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

    // ------------------------------------------------------------------ mHC
    /// A site's mixes of the stream (act.h [t][4][H]): pre, post, comb into `site` (0 attention, 1 MLP).
    fn hc_mixes(&self, l: usize, site: usize, t: usize) -> Result<()> {
        let hc = self.cfg.hc.unwrap();
        let hw: &HcW = self.w.layers[l].hc.as_ref().ok_or("an mHC site on a layer without one")?;
        let (fnw, base, scale) = if site == 0 { (&hw.attn_fn, &hw.attn_base, &hw.attn_scale) } else { (&hw.ffn_fn, &hw.ffn_base, &hw.ffn_scale) };
        let (a, fa) = (&self.act, &self.fl().act);
        let k = hc.mult * self.cfg.hidden;
        let st = self.st.raw();
        ck(unsafe { tidelm_kernels::ds41::ds_row_rsqrt(st, np(&a.h), k as i64, t as i32, k as i32, self.cfg.rms_eps, np(&fa.rsq)) },
           "hc rsqrt")?;
        crate::lin::gemm(&self.st, &a.split_part, np(&a.h), k, fnw, t, np(&fa.mixes), true, 24)?;
        let s = &fa.site[site];
        ck(unsafe { tidelm_kernels::ds41::ds_hc_sinkhorn(st, np(&fa.mixes), np(&fa.rsq), np(scale), np(base), t as i32, hc.iters as i32,
                                                         hc.eps, np(&s[0]), np(&s[1]), np(&s[2])) }, "hc sinkhorn")
    }

    /// A site's input: its mixes, the streams weighted by pre (bf16, one rounding), RMSNorm'd by
    /// the layer's norm into xn.
    fn hc_pre(&self, l: usize, site: usize, t: usize) -> Result<()> {
        self.hc_mixes(l, site, t)?;
        let lw = &self.w.layers[l];
        let (a, fa) = (&self.act, &self.fl().act);
        let (h, st) = (self.cfg.hidden, self.st.raw());
        ck(unsafe { tidelm_kernels::ds41::ds_hc_pre_norm(st, np(&a.h), np(&fa.site[site][0]), std::ptr::null(), np(&fa.hy), t as i32,
                                                         h as i32, 4, self.cfg.rms_eps) }, "hc pre")?;
        let w = if site == 0 { &lw.ln_in } else { &lw.ln_post };
        ck(unsafe { glm_rmsnorm(st, np(&fa.hy), h as i64, std::ptr::null(), 0, np(w), np(&a.xn), h as i64, t as i32, h as i32,
                                self.cfg.rms_eps) }, "hc norm")
    }

    /// A site's output: the stream <- post (outer) the sublayer's output (act.a) + comb^T the stream.
    fn hc_post(&self, site: usize, t: usize) -> Result<()> {
        let (a, fa) = (&self.act, &self.fl().act);
        let s = &fa.site[site];
        ck(unsafe { tidelm_kernels::ds41::ds_hc_post(self.st.raw(), np(&a.a), np(&a.h), np(&s[1]), np(&s[2]), np(&a.h), t as i32,
                                                     self.cfg.hidden as i32) }, "hc post")
    }

    /// layer_pre on an mHC layer: the previous layer's MLP site closed, the attention site, the MLP
    /// site's input in xn.
    pub(crate) fn flash_layer_pre(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        if l > 0 {
            self.hc_post(1, t)?;
            self.dump(&format!("x{}", l - 1), np(&self.act.h), t * 4 * self.cfg.hidden * 2, Some(l - 1))?;
        } else {
            self.dump("tok", np(&self.act.tok), t * 4, None)?;
        }
        self.hc_pre(l, 0, t)?;
        self.attention(l, t, prefill)?;
        self.hc_post(0, t)?;
        self.hc_pre(l, 1, t)
    }

    /// The decoder's end on an mHC model: the last MLP site closed, the streams' mean (Glm5Next's
    /// hyper head), the final norm into hn.
    pub(crate) fn flash_final(&self, t: usize) -> Result<()> {
        self.hc_post(1, t)?;
        self.dump(&format!("x{}", self.cfg.layers - 1), np(&self.act.h), t * 4 * self.cfg.hidden * 2, Some(self.cfg.layers - 1))?;
        let (a, fa) = (&self.act, &self.fl().act);
        let h = self.cfg.hidden;
        ck(unsafe { tidelm_kernels::ds41::ds_hc_mean(self.st.raw(), np(&a.h), np(&fa.hy), h as i64, t as i32, h as i32, 4) }, "hc mean")?;
        ck(unsafe { glm_rmsnorm(self.st.raw(), np(&fa.hy), h as i64, std::ptr::null(), 0, np(&self.w.norm), np(&a.hn), h as i64,
                                t as i32, h as i32, self.cfg.rms_eps) }, "final norm")
    }

    /// The embedding (BF16, the whole table on every rank): the decoder's 4 streams of each row, or
    /// the MTP layer's one row (zero at position 0) at `out`.
    pub(crate) fn embed_flash(&self, t: usize, mtp: bool, out: *mut c_void) -> Result<()> {
        let crate::weights::Embed::Bf16(table) = &self.w.embed else { return Err("embed_flash on an INT8 table".into()) };
        let a = &self.act;
        let (copies, pos) = if mtp { (1, np(&a.mpos)) } else { (4, std::ptr::null_mut()) };
        ck(unsafe { flash_embed(self.st.raw(), np(&a.tok), t as i32, np(table), self.cfg.hidden as i32, copies, pos, out) }, "embed")?;
        // the lane's image rows: the tower's features in every stream (Glm5NextModel's masked_scatter
        // before the streams are copied); only in its own eager prefill passes (forward_local)
        let Some(v) = self.vis.as_ref().map(|v| &v.lanes[self.lane]).filter(|v| !mtp && self.pass_nl == 1 && !v.spans.is_empty()) else {
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
            for k in 0..4 {
                let dst = unsafe { (out as *mut u16).add(((a0 - p0) * 4 + k) * h) } as *mut c_void;
                let src = at(feats, (row + a0 - s) * h);
                ck(unsafe { tidelm_cuda::ffi::cudaMemcpy2DAsync(dst, 4 * h * 2, src, h * 2, h * 2, a1 - a0,
                                                               tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, self.st.raw()) },
                   "image rows")?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ attention
    pub(crate) fn attention_flash(&self, l: usize, t: usize, prefill: bool) -> Result<()> {
        match &self.w.layers[l].attn {
            Attn::Kda(k) => self.attention_kda(l, k, t, prefill),
            Attn::Dsa(d) => self.attention_dsa(l, d, t, prefill),
            Attn::Mla(_) => unreachable!(),
        }
    }

    /// A KDA layer over `t` rows of xn -> act.a.
    fn attention_kda(&self, l: usize, k: &KdaW, t: usize, prefill: bool) -> Result<()> {
        let fl = self.fl();
        let fa = &fl.act;
        let a = &self.act;
        let st = self.st.raw();
        let (h, dl, khl) = (self.cfg.hidden, fl.dl, fl.khl);
        let c = 3 * dl;
        let kl = fl.kda_of[l].unwrap();
        let nk = fl.nkda;
        let off = KdaW::offsets(dl, khl);
        let ldp = fa.ldp;
        self.lin(&k.proj, np(&a.xn), h, t, np(&fa.proj), true)?;
        // proj's own row stride is its width; the rest of the code reads it at ldp
        let ldq = k.proj.n_total();
        debug_assert!(ldq <= ldp);
        self.lin(&k.f_b, at(&fa.proj, off[3]), ldq, t, np(&fa.f), true)?;
        self.lin(&k.g_b, at(&fa.proj, off[4]), ldq, t, np(&fa.gt), true)?;
        let save = t <= SAVE_ROWS;
        let lk = nk * SAVE_ROWS;
        let (sr, sk, sg, sv, sb) = if save {
            (at(&fl.save_raw, kl * SAVE_ROWS * c), at(&fl.save_k, kl * SAVE_ROWS * dl), at(&fl.save_g, kl * SAVE_ROWS * dl),
             at(&fl.save_v, kl * SAVE_ROWS * dl), at(&fl.save_b, kl * SAVE_ROWS * khl))
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        let (lanes, nl, meta) = (np(&a.lanes), self.pass_nl as i32, np(&fl.meta));
        let conv = at(&fl.conv, kl * 3 * c);
        let slot_c = (nk * 3 * c) as i64;
        let state = at(&fl.state, kl * khl * 128 * 128);
        let slot_f = (nk * khl * 128 * 128) as i64;
        unsafe {
            ck(flash_kda_conv(st, np(&fa.proj), ldq as i64, np(&k.conv), c as i32, conv, slot_c, lanes, nl, meta, t as i32, np(&fa.act),
                              c as i64, sr, (lk * c) as i64, SAVE_ROWS as i32), "kda conv")?;
            ck(flash_kda_conv_state(st, np(&fa.proj), ldq as i64, c as i32, conv, slot_c, lanes, nl, meta), "kda conv state")?;
            ck(flash_kda_chain(st, np(&fa.act), c as i64, dl as i32, np(&fa.f), dl as i64, at(&fa.proj, off[5]), ldq as i64, np(&fa.gt),
                               dl as i64, np(&k.a_log), np(&k.dt_bias), np(&k.o_norm), self.cfg.kda.unwrap().lower, self.cfg.rms_eps,
                               state, slot_f, lanes, nl, meta, np(&fa.y), dl as i64, sk, sg, sv, sb, (lk * dl) as i64,
                               SAVE_ROWS as i32, khl as i32), "kda chain")?;
        }
        self.lin(&k.o, np(&fa.y), dl, t, np(&a.f32buf), false)?;
        self.allreduce(np(&a.f32buf), t * h, np(&a.a), prefill)
    }

    /// A sparse-attention layer (MLA without RoPE over the indexer's selection) over `t` rows of xn
    /// -> act.a; the MTP layer's on its own positions and cache.
    fn attention_dsa(&self, l: usize, d: &DsaW, t: usize, prefill: bool) -> Result<()> {
        let (cfg, a, fl) = (&self.cfg, &self.act, self.fl());
        let fa = &fl.act;
        let st = self.st.raw();
        let (h, hl) = (cfg.hidden, self.hl);
        let mtp = cfg.mtp && l == cfg.mtp_layer();
        let kv = self.kv[l].as_ref().ok_or("a sparse layer without a cache")?;
        let (posb, rowsb, bound, koff) = if mtp { (np(&a.mpos), np(&a.mrows), np(&a.mbound), np(&a.mkoff)) }
                                         else { (np(&a.pos), np(&a.rows), np(&a.bound), np(&a.koff)) };
        let (lanes, nl, meta) = if mtp { (at(&a.lanes, MAX_LANES * 4), self.mtp_nl as i32, at(&fl.meta, MAX_LANES * 4)) }
                                else { (np(&a.lanes), self.pass_nl as i32, np(&fl.meta)) };
        let p0 = if mtp { self.mtp_pos } else { self.pos };
        let o = DsaW::OFFSETS_A;
        let lda = o[5];
        // proj_a: q_a | kv_a | wk | weights_proj | gate
        self.lin(&d.proj_a, np(&a.xn), h, t, np(&fa.proj), true)?;
        ck(unsafe { glm_rmsnorm(st, np(&fa.proj), lda as i64, std::ptr::null(), 0, np(&d.q_norm), np(&a.qr), cfg.q_lora as i64, t as i32,
                                cfg.q_lora as i32, cfg.rms_eps) }, "q norm")?;
        ck(unsafe { glm_rmsnorm(st, at(&fa.proj, o[1]), lda as i64, std::ptr::null(), 0, np(&d.kv_norm), np(&a.c), 512, t as i32, 512,
                                cfg.rms_eps) }, "kv norm")?;
        ck(unsafe { glm_kv_fp8(st, np(&a.c), 512, rowsb, kv.cq.raw(), kv.cs.raw(), t as i32, 512) }, "latent cache")?;
        // the indexer's keys: LayerNorm, the pools completed in this pass, the ring
        let (iq, is) = (kv.iq.unwrap().raw(), kv.is.unwrap().raw());
        let ring = at(&fl.ring, fl.dsa_of[l].unwrap() * fl.lanes * 64 * 256);
        unsafe {
            ck(flash_ln128(st, at(&fa.proj, o[2]), lda as i64, np(&d.kn_w), np(&d.kn_b), np(&fa.kln), 128, t as i32, 1e-6), "index k norm")?;
            ck(flash_widen(st, at(&fa.proj, o[3]), lda as i64, np(&fa.iw), cfg.index_heads as i64, t as i32, cfg.index_heads as i32),
               "index weights")?;
            ck(flash_pool(st, np(&fa.kln), 128, at(&fa.proj, o[4]), lda as i64, np(&d.ape), ring, 64 * 256, posb, lanes, nl, meta,
                          t as i32, iq, is), "index pools")?;
            ck(flash_ring(st, np(&fa.kln), 128, at(&fa.proj, o[4]), lda as i64, ring, 64 * 256, posb, lanes, nl, meta, t as i32),
               "index ring")?;
        }
        // proj_b: q_b (this rank's heads x 256) | wq_b; the indexer query contiguous for its scores
        self.lin(&d.proj_b, np(&a.qr), cfg.q_lora, t, np(&fa.pb), true)?;
        let ldb = d.proj_b.n_total();
        let iqw = cfg.index_heads * cfg.index_dim;
        unsafe {
            ck(tidelm_cuda::ffi::cudaMemcpy2DAsync(np(&fa.qi), iqw * 2, at(&fa.pb, hl * cfg.qk_dim()) as *const c_void, ldb * 2, iqw * 2,
                                                   t, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, st), "index q")?;
        }
        let qrow = hl * 512;
        let s_all = p0 + t;
        let inv_d = (cfg.index_dim as f32).powf(-0.5);
        let wscale = (cfg.index_heads as f32).powf(-0.5);
        let scale = (cfg.qk_dim() as f32).powf(-0.5);
        let k_att = cfg.attend_max();
        if !prefill {
            ck(unsafe { glm_absorb_q_n256(st, np(&fa.pb), ldb as i64, 256, np(&d.kvb_q), np(&d.kvb_s), np(&a.qf), qrow as i64, 512,
                                          t as i32, hl as i32) }, "absorb q")?;
            let lim = self.opt.max_seq / cfg.index_pool + 1;
            ck(unsafe { glm_index_scores(st, np(&fa.qi), np(&fa.iw), wscale, iq, is, posb, t as i32, lim as i32, cfg.index_pool as i32,
                                         (cfg.index_pool - 1) as i32, 0, inv_d, np(&a.scores), lim as i64, lanes, nl) }, "index scores")?;
            ck(unsafe { glm_topk(st, np(&a.scores), lim as i64, posb, cfg.index_pool as i32, (cfg.index_pool - 1) as i32, lim as i32,
                                 POOL_TOPK as i32, t as i32, np(&fa.pools), POOL_TOPK as i64, std::ptr::null_mut(), 0) }, "topk pools")?;
            ck(unsafe { flash_expand(st, np(&fa.pools), POOL_TOPK as i64, POOL_TOPK as i32, posb, np(&fa.idx), IDX_LD as i64, t as i32) },
               "expand")?;
            let (ns, chunk) = decode_split(k_att, hl, t);
            ck(unsafe { glm_mla_nope(st, MLA_DECODE_MT, np(&a.qf), 0, qrow as i64, 512, hl as i32, kv.cq.raw(), kv.cs.raw(),
                                     std::ptr::null(), 0, np(&fa.idx), IDX_LD as i64, k_att as i32, 1, bound, t as i32, hl as i32, scale,
                                     ns as i32, chunk as i32, np(&a.pm), np(&a.pl), np(&a.pacc), std::ptr::null_mut(), 0, 0, koff) },
               "mla")?;
            ck(unsafe { glm_mla_combine(st, np(&a.pm), np(&a.pl), np(&a.pacc), t as i32, hl as i32, ns as i32, np(&a.olat),
                                        (hl * 512) as i64, 512, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0, 0,
                                        0) }, "mla combine")?;
            ck(unsafe { glm_absorb_o_n256(st, np(&a.olat), (hl * 512) as i64, 512, np(&d.kvb_q), np(&d.kvb_s), np(&a.o), (hl * 256) as i64,
                                          t as i32, hl as i32) }, "absorb o")?;
        } else {
            let rows = cfg.kvb_rows();
            ck(unsafe { glm_kvb_dequant(st, np(&d.kvb_q), np(&d.kvb_s), np(&fa.kvb_bf), (hl * rows) as i64) }, "kv_b dequant")?;
            // per head: q_lat [t][512] = q_nope [t][256] . Wuk [256][512]
            self.blas.gemm_rm_batched(t, 512, 256, Mat { ptr: np(&fa.pb), ty: R_16BF, ld: ldb as i64 }, 256,
                                      Mat { ptr: np(&fa.kvb_bf), ty: R_16BF, ld: 512 }, (rows * 512) as i64, false, np(&a.qf), R_16BF,
                                      qrow as i64, 512, hl)?;
            let b = kv.lane_bases(self.lane);
            let dense_causal = s_all <= k_att;
            if !dense_causal {
                let npool = s_all / cfg.index_pool + 1;
                let step = (SCORE_VALUES / npool.max(1)).max(1);
                let mut r0 = 0;
                while r0 < t {
                    let n = step.min(t - r0);
                    let pr = unsafe { (posb as *mut i32).add(r0) } as *mut c_void;
                    ck(unsafe { glm_index_scores(st, at(&fa.qi, r0 * iqw), at(&fa.iw, r0 * cfg.index_heads), wscale, b[2], b[3], pr,
                                                 n as i32, npool as i32, cfg.index_pool as i32, (cfg.index_pool - 1) as i32, 0, inv_d,
                                                 np(&a.scores), npool as i64, std::ptr::null(), 0) }, "index scores (prefill)")?;
                    ck(unsafe { glm_topk(st, np(&a.scores), npool as i64, pr, cfg.index_pool as i32, (cfg.index_pool - 1) as i32,
                                         npool as i32, POOL_TOPK as i32, n as i32, at(&fa.pools, r0 * POOL_TOPK), POOL_TOPK as i64,
                                         std::ptr::null_mut(), 0) }, "topk pools (prefill)")?;
                    r0 += n;
                }
                ck(unsafe { flash_expand(st, np(&fa.pools), POOL_TOPK as i64, POOL_TOPK as i32, posb, np(&fa.idx), IDX_LD as i64,
                                         t as i32) }, "expand (prefill)")?;
            }
            let (idx, k) = if dense_causal { (std::ptr::null_mut(), s_all) } else { (np(&fa.idx), k_att) };
            let chunk = k.div_ceil(32) * 32;
            ck(unsafe { glm_mla_nope(st, 2, np(&a.qf), 0, qrow as i64, 512, hl as i32, b[0], b[1], std::ptr::null(), 0, idx,
                                     IDX_LD as i64, k as i32, 1, posb, t as i32, hl as i32, scale, 1, chunk as i32, std::ptr::null_mut(),
                                     std::ptr::null_mut(), std::ptr::null_mut(), np(&a.olat), (hl * 512) as i64, 512, std::ptr::null()) },
               "mla (prefill)")?;
            // per head: o [t][256] = o_lat [t][512] . Wuv^T (Wuv [256][512], rows 256.. of the head's kv_b)
            self.blas.gemm_rm_batched(t, 256, 512, Mat { ptr: np(&a.olat), ty: R_16BF, ld: (hl * 512) as i64 }, 512,
                                      Mat { ptr: at(&fa.kvb_bf, 256 * 512), ty: R_16BF, ld: 512 }, (rows * 512) as i64, true, np(&a.o),
                                      R_16BF, (hl * 256) as i64, 256, hl)?;
        }
        let _ = bound;
        self.lin(&d.o, np(&a.o), hl * 256, t, np(&a.f32buf), false)?;
        self.allreduce(np(&a.f32buf), t * h, np(&a.a), prefill)
    }
}
