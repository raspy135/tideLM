//! Adaptive expert residency (engine/glm/tp.py: adapt, bg_step, _replay_split's hot reads,
//! _hot_keep): which routed experts hold the arena's slots, and how that follows demand.
//!
//! - adapt(max_swaps): after a prefill (512) and every 200 decoded tokens (256): score =
//!   0.9 * score + demand; across the swap layers, the non-resident experts with the highest
//!   score replace the residents with the lowest while the newcomer beats the victim by x1.5,
//!   never taking a layer below 2 x top-k residents. Rank 0 plans, both ranks load the same
//!   experts into the same slots.
//! - bg_step, every 2 rounds: rank 0 queues up to 32 victims for hot reads to keep (residents
//!   whose recent mass is below 0.01, not installed within 8 periods, lowest score first).
//! - hot reads, between a decode pass's segments: a MoE layer's heaviest missed pick with a
//!   share >= 0.2 of its row's routed weight is read from the pack into the transient slot and
//!   computed exactly (one a layer, 16 a step paced evenly over the layers); after the layer it
//!   takes the next queued victim's slot or is dropped. Both ranks decide from the same
//!   replicated routing, so they read the same experts without a message.

use std::time::Instant;

use tidelm_cuda::PinnedBuffer;

use crate::arena::align_up;
use crate::forward::Engine;
use crate::{ck, Result};

pub const ADAPT_DECAY: f64 = 0.9;
pub const ADAPT_MARGIN: f64 = 1.5;
const BG_DECAY: f64 = 0.7;
const BG_COLD: f64 = 0.01;
const BG_PROTECT: u64 = 8;

/// The residency bookkeeping both ranks keep identical (rank 0's plans are mirrored).
pub struct Res {
    /// per slot: the (layer, expert) it holds
    pub owner: Vec<Option<(usize, usize)>>,
    pub swap_layers: Vec<usize>,
    /// rank 0's planning state: decayed demand (adapt), recent mass (bg), protection
    pub score: Vec<f64>,
    pub bg_score: Vec<f64>,
    pub bg_last: Vec<f64>,
    pub bg_until: Vec<u64>,
    pub bg_period: u64,
    /// victims for kept hot reads, weakest last: (layer, expert, slot)
    pub hot_q: Vec<(usize, usize, usize)>,
    /// the routing history rank 0 saves (mass [layers][experts], tokens)
    pub db_mass: Option<Vec<f64>>,
    pub db_tokens: u64,
    pub db_path: Option<std::path::PathBuf>,
    pub demand_tokens: u64,
    pub stats: ResStats,
}

#[derive(Default, Clone, Debug)]
pub struct ResStats {
    pub swaps: usize,
    pub adapt_s: f64,
    pub hot_reads: usize,
    pub hot_kept: usize,
    pub hot_unkept: usize,
    pub hot_s: f64,
}

/// Synchronous hot reads' settings and per-step state.
pub struct Hot {
    pub min_share: f32,
    pub per_step: usize,
    /// the transient slot a read goes into
    pub slot: usize,
    /// candidates of the current pass, from the device: [rows * topk][2] (share, expert)
    pub host: PinnedBuffer<f32>,
    pub buf: PinnedBuffer<u8>,
    reads: usize,
    /// the read installed for the layer being computed: (layer, expert)
    live: Option<(usize, usize)>,
}

impl Hot {
    pub fn new(slot: usize, max_record: usize, rows: usize, topk: usize) -> Result<Self> {
        Ok(Hot { min_share: 0.2, per_step: 16, slot, host: PinnedBuffer::zeroed(rows * topk * 2)?,
                 buf: PinnedBuffer::uninit(align_up(max_record))?, reads: 0, live: None })
    }
}

impl Res {
    pub fn new(nl: usize, e: usize, slots: usize, swap_layers: Vec<usize>, residents: &[(usize, usize)]) -> Self {
        let mut owner = vec![None; slots + 1];
        for (s, &r) in residents.iter().enumerate() {
            owner[s] = Some(r);
        }
        Res { owner, swap_layers, score: vec![0.0; nl * e], bg_score: vec![0.0; nl * e], bg_last: vec![0.0; nl * e],
              bg_until: vec![0; nl * e], bg_period: 0, hot_q: vec![], db_mass: None, db_tokens: 0, db_path: None,
              demand_tokens: 0, stats: ResStats::default() }
    }
}

pub(crate) const OP_ADAPT: u32 = 20;
pub(crate) const OP_BG: u32 = 21;

impl Engine {
    fn read_demand(&self) -> Result<Vec<f64>> {
        self.stream().synchronize()?;
        Ok(self.demand.download()?.into_iter().map(|x| x as f64).collect())
    }

    /// Apply a swap plan [(layer, expert, victim layer, victim expert)] on this rank: the lut,
    /// owners, and the newcomers' records read into their victims' slots.
    pub(crate) fn apply_swaps(&mut self, plan: &[(usize, usize, usize, usize)]) -> Result<()> {
        let e = self.cfg.experts;
        let mut items = vec![];
        for &(l, c, vl, v) in plan {
            let slot = self.lut_h[vl * e + v];
            if slot < 0 {
                return Err(format!("adapt: victim ({vl}, {v}) is not resident").into());
            }
            self.lut_h[vl * e + v] = -1;
            self.lut_h[l * e + c] = slot;
            self.res.owner[slot as usize] = Some((l, c));
            items.push((slot as usize, l, c));
        }
        if !items.is_empty() {
            self.stream().synchronize()?;
            crate::forward::load_slots(&self.pack, &self.arena, &items)?;
        }
        self.push_lut()
    }

    pub(crate) fn push_lut(&mut self) -> Result<()> {
        for (a, &s) in self.allowed_h.iter_mut().zip(&self.lut_h) {
            *a = (s >= 0) as u8;
        }
        self.lut_upload()
    }

    /// adapt() (rank 0, mirrored): fold the demand into the score and the history, plan the
    /// swaps toward it; both ranks apply the plan.
    pub fn adapt(&mut self, max_swaps: usize) -> Result<usize> {
        let t0 = Instant::now();
        let demand = self.read_demand()?;
        if let Some(m) = self.res.db_mass.as_mut() {
            for (a, d) in m.iter_mut().zip(&demand) {
                *a += d;
            }
            self.res.db_tokens += self.res.demand_tokens;
        }
        self.res.demand_tokens = 0;
        for (s, d) in self.res.score.iter_mut().zip(&demand) {
            *s = *s * ADAPT_DECAY + d;
        }
        let plan = self.plan_swaps(max_swaps);
        let flat: Vec<u32> = plan.iter().flat_map(|&(a, b, c, d)| [a as u32, b as u32, c as u32, d as u32]).collect();
        self.mirror_msg(|m| {
            m.u32(OP_ADAPT).u32s(&flat);
        })?;
        self.adapt_apply(&plan)?;
        self.save_db()?;
        self.res.stats.adapt_s += t0.elapsed().as_secs_f64();
        Ok(plan.len())
    }

    /// Both ranks: the demand restarts, the plan's experts load.
    pub(crate) fn adapt_apply(&mut self, plan: &[(usize, usize, usize, usize)]) -> Result<()> {
        self.demand.zero_async(self.stream())?;
        self.res.bg_last.iter_mut().for_each(|x| *x = 0.0);
        self.apply_swaps(plan)?;
        self.res.stats.swaps += plan.len();
        Ok(())
    }

    pub(crate) fn follow_adapt(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        let plan: Vec<(usize, usize, usize, usize)> =
            r.u32s()?.chunks(4).map(|c| (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize)).collect();
        self.adapt_apply(&plan)
    }

    pub(crate) fn follow_bg(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        let q: Vec<(usize, usize, usize)> = r.u32s()?.chunks(3).map(|c| (c[0] as usize, c[1] as usize, c[2] as usize)).collect();
        self.res.bg_period += 1;
        self.res.hot_q = q.into_iter().rev().collect();
        Ok(())
    }

    /// Rank 0: tp.adapt's cross-layer plan over the swap layers' scores.
    fn plan_swaps(&self, max_swaps: usize) -> Vec<(usize, usize, usize, usize)> {
        let e = self.cfg.experts;
        let floor = 2 * self.cfg.topk;
        let sc = &self.res.score;
        let layers = &self.res.swap_layers;
        let res = |l: usize, x: usize| self.lut_h[l * e + x] >= 0;
        let mut cand: Vec<(usize, usize)> = layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x))).filter(|&(l, x)| !res(l, x)).collect();
        cand.sort_by(|a, b| sc[b.0 * e + b.1].partial_cmp(&sc[a.0 * e + a.1]).unwrap().then(a.cmp(b)));
        let mut vict: Vec<(usize, usize)> = layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x))).filter(|&(l, x)| res(l, x)).collect();
        vict.sort_by(|a, b| sc[a.0 * e + a.1].partial_cmp(&sc[b.0 * e + b.1]).unwrap().then(a.cmp(b)));
        let mut left: std::collections::HashMap<usize, usize> =
            layers.iter().map(|&l| (l, (0..e).filter(|&x| res(l, x)).count())).collect();
        let (mut plan, mut vi) = (vec![], 0);
        for (cl, ce) in cand {
            if plan.len() >= max_swaps || sc[cl * e + ce] <= 0.0 {
                break;
            }
            let mut found = None;
            while vi < vict.len() {
                let (vl, ve) = vict[vi];
                vi += 1;
                if left[&vl] > floor {
                    found = Some((vl, ve));
                    break;
                }
            }
            let Some((vl, ve)) = found else { break };
            if sc[cl * e + ce] <= sc[vl * e + ve] * ADAPT_MARGIN {
                break;
            }
            *left.get_mut(&vl).unwrap() -= 1;
            *left.get_mut(&cl).unwrap() += 1;
            plan.push((cl, ce, vl, ve));
        }
        plan
    }

    pub fn save_db(&self) -> Result<()> {
        if let (Some(m), Some(p)) = (&self.res.db_mass, &self.res.db_path) {
            crate::residency::write_mass(p, m, self.cfg.n_layers(), self.cfg.experts, self.res.db_tokens)?;
        }
        Ok(())
    }

    /// bg_step (rank 0, mirrored): refresh the recent mass, queue the victims kept hot reads
    /// take.
    pub fn bg_step(&mut self, keep: usize) -> Result<()> {
        let e = self.cfg.experts;
        let demand = self.read_demand()?;
        for i in 0..demand.len() {
            let w = demand[i] - self.res.bg_last[i];
            self.res.bg_last[i] = demand[i];
            self.res.bg_score[i] = self.res.bg_score[i] * BG_DECAY + w;
        }
        let period = self.res.bg_period + 1;
        let floor = 2 * self.cfg.topk;
        let layers = self.res.swap_layers.clone();
        let mut left: std::collections::HashMap<usize, usize> =
            layers.iter().map(|&l| (l, (0..e).filter(|&x| self.lut_h[l * e + x] >= 0).count())).collect();
        let mut cold: Vec<(f64, usize, usize)> = layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x)))
            .filter(|&(l, x)| {
                let i = l * e + x;
                self.lut_h[i] >= 0 && self.res.bg_score[i] < BG_COLD && self.res.bg_until[i] <= period
            })
            .map(|(l, x)| (self.res.score[l * e + x] + demand[l * e + x], l, x)).collect();
        cold.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then((a.1, a.2).cmp(&(b.1, b.2))));
        let mut q = vec![];
        for (_, l, x) in cold {
            if q.len() >= keep {
                break;
            }
            if left[&l] > floor {
                *left.get_mut(&l).unwrap() -= 1;
                q.push((l, x, self.lut_h[l * e + x] as usize));
            }
        }
        let flat: Vec<u32> = q.iter().flat_map(|&(a, b, c)| [a as u32, b as u32, c as u32]).collect();
        self.mirror_msg(|m| {
            m.u32(OP_BG).u32s(&flat);
        })?;
        let mut r_msg = tidelm_comm::Msg::default();
        r_msg.u32s(&flat);
        self.follow_bg(&mut tidelm_comm::Reader::new(&r_msg.0))
    }

    /// The pass's miss counters [picks, missed, mass, missed mass] (rank-local; replicated
    /// routing makes them the same on both ranks).
    pub fn miss_totals(&self) -> Result<[f64; 4]> {
        self.stream().synchronize()?;
        let v = self.miss.download()?;
        Ok([v[0], v[1], v[2], v[3]])
    }

    /// Between a decode pass's segments, after MoE layer l's exact routing: release the previous
    /// layer's hot read (kept over a queued victim, or dropped), then read l's heaviest missed
    /// pick if it qualifies.
    pub(crate) fn hot_point(&mut self, l: usize, t: usize, idx: usize, n_layers: usize) -> Result<()> {
        self.hot_release()?;
        let k = self.cfg.topk;
        let e = self.cfg.experts;
        let (min_share, per_step, slot) = match self.hot.as_ref() {
            Some(h) => (h.min_share, h.per_step, h.slot),
            None => return Ok(()),
        };
        let reads = self.hot.as_ref().unwrap().reads;
        // linear pacing: the i-th of n MoE layers may bring the step's reads to ceil(cap * (i+1)/n)
        let room = per_step.saturating_sub(reads).min((per_step * (idx + 1)).div_ceil(n_layers).saturating_sub(reads));
        if room == 0 {
            return Ok(());
        }
        self.stream().synchronize()?;
        let hot = self.hot.as_ref().unwrap();
        let mut best: Option<(f32, usize)> = None;
        for i in 0..t * k {
            let (v, x) = (hot.host[2 * i], hot.host[2 * i + 1] as usize);
            if v >= min_share && best.is_none_or(|(bv, bx)| v > bv || (v == bv && x < bx)) {
                best = Some((v, x));
            }
        }
        let Some((_, x)) = best else { return Ok(()) };
        if self.lut_h[l * e + x] >= 0 {
            return Ok(());
        }
        let t0 = Instant::now();
        {
            let h = self.hot.as_mut().unwrap();
            let r = self.pack.read(l, x, &mut h.buf)?;
            self.arena.load(slot, &h.buf[..r.nbytes], &self.st)?;
            h.reads += 1;
            h.live = Some((l, x));
        }
        self.lut_h[l * e + x] = slot as i32;
        self.allowed_h[l * e + x] = 1;
        self.lut_entry(l * e + x)?;
        self.res.stats.hot_reads += 1;
        self.res.stats.hot_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// _hot_keep / _pf_release for the read of the layer just computed.
    pub(crate) fn hot_release(&mut self) -> Result<()> {
        let e = self.cfg.experts;
        let Some(h) = self.hot.as_mut() else { return Ok(()) };
        let Some((l, x)) = h.live.take() else { return Ok(()) };
        let src = h.slot;
        let i = l * e + x;
        while let Some((vl, ve, vs)) = self.res.hot_q.pop() {
            if self.res.owner[vs] != Some((vl, ve)) {
                continue;
            }
            self.arena.copy_slot(vs, src, self.stream())?;
            self.lut_h[vl * e + ve] = -1;
            self.allowed_h[vl * e + ve] = 0;
            self.res.owner[vs] = Some((l, x));
            self.lut_h[i] = vs as i32;
            self.allowed_h[i] = 1;
            self.res.bg_until[i] = self.res.bg_period + BG_PROTECT;
            self.lut_entry(vl * e + ve)?;
            self.lut_entry(i)?;
            self.res.stats.hot_kept += 1;
            return Ok(());
        }
        self.lut_h[i] = -1;
        self.allowed_h[i] = 0;
        self.lut_entry(i)?;
        self.res.stats.hot_unkept += 1;
        Ok(())
    }

    /// A new pass: no reads yet.
    pub(crate) fn hot_begin(&mut self) {
        if let Some(h) = self.hot.as_mut() {
            h.reads = 0;
        }
    }

    pub fn enable_hot(&mut self, keep: bool) -> Result<()> {
        let rows = crate::forward::VERIFY_ROWS;
        self.hot = Some(Hot::new(self.opt.slots, self.pack.max_record, rows, self.cfg.topk)?);
        let _ = keep;
        ck(0, "")
    }
}
