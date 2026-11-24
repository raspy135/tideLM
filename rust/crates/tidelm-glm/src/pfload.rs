//! Prefill loads (GLM_PF_LOAD): a prefill chunk's missing experts read in before their layer's
//! MoE runs, as SmallTide stages them (its prefill.rs), instead of every miss computing on a
//! substitute. On by default for GLM-5.3 (tidelm-server's --pf-load 4, 32 scratch slots); off
//! for GLM-5.3-Flash and Inkling 2, where it has not been measured.
//!
//! A prefill sub-chunk of up to `moe_rows` rows routes each MoE layer exactly (moe_exact); the
//! rows' top-k picks come to the host, each missing expert's share (a row's picks normalized to
//! sum to one) is summed over the rows, and the heaviest missing experts whose sum reaches the
//! threshold - in tokens' worth of routing - are read into the scratch slots (the transient
//! class's slots below the hot reads' one) on reader threads, made routable, and the layer's
//! routing over the residents (moe_rest) then routes to them. They stay for the layer's later
//! sub-chunks and leave routing after its last: the slots are the next layer's.
//!
//! Why the prefill and not decode: a read serves every row of the chunk that picked the expert
//! (dozens), against one to three in a decode pass, and the prompt's rows are what a
//! conversation keeps (their KV), substitutes and all. On 2026-11-23's traffic (bit280, 33
//! requests) prompts computed 13-26% of their picks on substitutes; the new input was over the
//! two-pass prefill's 512 tokens in 27 of them.
//!
//! Not the Python engine's GLM_PF (engine/glm/tp.py _pf_predict, docs/glm-prefill-plan.md), which
//! stayed off: it predicted the next layer's routing from this layer's MLP input and read up to 32
//! of its heaviest missing experts a layer, no threshold, overlapped with compute, rank 0 deciding;
//! a 3.3K prompt's miss 33% -> 25%, TTFT 23.7 -> 32.8 s. Here the layer's own exact routing picks,
//! and only experts over the threshold are read (2026-11-23, bit280: ~1.5 s a 3.5K prompt); which
//! difference made the cost is not measured.
//!
//! Both ranks decide alone: the rows' exact routing is replicated (the decode rescue relies on
//! the same), the sums run in row order in f64, and ties break by expert id, so the pair reads
//! the same experts with no message. Each rank reads its own half of each record.

use std::sync::mpsc;
use std::time::Instant;

use crate::adapt::{readers, ReadJob};
use crate::forward::Engine;
use crate::Result;

/// Reader threads for prefill loads (their own: a background adaptation's reads queue on the
/// adaptation's threads, and a prefill load would wait behind them).
const PF_READERS: usize = 8;

/// The coverage figures kept per load point: the missed weight the heaviest N missing experts
/// carry, for these N.
pub const PF_TOPS: [usize; 4] = [8, 16, 32, 64];

pub struct PfLoad {
    /// tokens' worth of summed share a missing expert needs to be read in
    pub th: f32,
    /// bench knob `pfload`: off measures (the coverage counters) without reading
    pub on: bool,
    /// the scratch slots [first, first + slots)
    pub first: usize,
    pub slots: usize,
    /// (slot, layer, expert) read in for the current layer
    live: Vec<(usize, usize, usize)>,
    jobs: Option<mpsc::Sender<ReadJob>>,
    ids: Vec<i32>,
    ws: Vec<f32>,
}

#[derive(Default, Clone, Debug)]
pub struct PfStats {
    /// load points (a MoE layer's sub-chunk) and their rows
    pub points: usize,
    pub rows: usize,
    /// the routed weight that missed, in tokens' worth (a row's picks sum to one), before reads
    pub missed: f64,
    /// distinct missing experts summed over the points
    pub distinct: usize,
    /// the missed weight the heaviest PF_TOPS[i] missing experts of each point carry
    pub top: [f64; 4],
    /// missing experts at or over the threshold, and their weight (whether or not read)
    pub over: usize,
    pub over_w: f64,
    /// experts read, the weight they covered, their bytes and the host seconds the reads took
    pub loads: usize,
    pub loaded_w: f64,
    pub bytes: u64,
    pub read_s: f64,
}

impl PfStats {
    /// The request log's part: this request's loads and what they covered (`self` - `s0`).
    pub fn line(&self, s0: &PfStats) -> String {
        let missed = self.missed - s0.missed;
        let pct = |x: f64| if missed > 0.0 { 100.0 * x / missed } else { 0.0 };
        format!("prefill loads {} ({:.2} GB, {:.2} s; {:.0}% of {:.0} tokens' worth missed, {:.0}% over the threshold)",
                self.loads - s0.loads, (self.bytes - s0.bytes) as f64 / 1e9, self.read_s - s0.read_s,
                pct(self.loaded_w - s0.loaded_w), missed, pct(self.over_w - s0.over_w))
    }

    /// The bench's: the coverage curve too.
    pub fn bench_line(&self, s0: &PfStats) -> String {
        let n = (self.points - s0.points).max(1) as f64;
        let missed = self.missed - s0.missed;
        let pct = |x: f64| if missed > 0.0 { 100.0 * x / missed } else { 0.0 };
        let tops: Vec<String> = PF_TOPS.iter().enumerate()
            .map(|(i, k)| format!("top {k} {:.0}%", pct(self.top[i] - s0.top[i]))).collect();
        format!("{}; {} points of {:.0} rows, {:.1} tokens' worth missed and {:.1} missing experts a point, {:.1} over the \
                 threshold; {}", self.line(s0), self.points - s0.points, (self.rows - s0.rows) as f64 / n, missed / n,
                (self.distinct - s0.distinct) as f64 / n, (self.over - s0.over) as f64 / n, tops.join(", "))
    }
}

impl Engine {
    /// GLM_PF_LOAD on: `th` tokens' worth, the arena's `slots` scratch slots (Options::pf_slots,
    /// made by class_arena below the hot reads' transient slot).
    pub fn enable_pf_load(&mut self, th: f32, slots: usize) -> Result<()> {
        if !(th > 0.0) || slots == 0 {
            return Err(format!("GLM_PF_LOAD={th} with {slots} scratch slots: a threshold > 0 and slots").into());
        }
        let last = self.arena.classes.last().ok_or("arena: no classes")?;
        if last.n != slots + 1 || last.first + last.n != self.arena.slots {
            return Err(format!("GLM_PF_LOAD: the arena's transient class has {} slots, not {} (Options::pf_slots)", last.n,
                               slots + 1).into());
        }
        let n = self.opt.moe_rows.max(1) * self.cfg.topk;
        self.pf = Some(PfLoad { th, on: true, first: self.arena.slots - 1 - slots, slots, live: vec![], jobs: None,
                                ids: vec![0; n], ws: vec![0.0; n] });
        Ok(())
    }

    /// After moe_exact over `n` rows of MoE layer `l` (a prefill sub-chunk): the missing experts'
    /// summed shares, the coverage counters, and the heaviest over the threshold read into free
    /// scratch slots and routable before moe_rest.
    pub(crate) fn pf_load_point(&mut self, l: usize, n: usize) -> Result<()> {
        let (k, e) = (self.cfg.topk, self.cfg.experts);
        let Some(mut pf) = self.pf.take() else { return Ok(()) };
        let r = self.pf_point(&mut pf, l, n, k, e);
        self.pf = Some(pf);
        r
    }

    fn pf_point(&mut self, pf: &mut PfLoad, l: usize, n: usize, k: usize, e: usize) -> Result<()> {
        self.pf_routing(n, &mut pf.ids, &mut pf.ws)?;
        let mut miss = vec![0f64; e];
        for r in 0..n {
            let row = &pf.ws[r * k..(r + 1) * k];
            let s: f64 = row.iter().map(|&w| w as f64).sum::<f64>().max(1e-20);
            for j in 0..k {
                let x = pf.ids[r * k + j];
                if x < 0 || x as usize >= e {
                    continue;
                }
                let i = l * e + x as usize;
                if self.lut_h[i] < 0 && !self.adapt_pending(i) {
                    miss[x as usize] += row[j] as f64 / s;
                }
            }
        }
        let mut cand: Vec<(f64, usize)> = miss.iter().enumerate().filter(|(_, &m)| m > 0.0).map(|(x, &m)| (m, x)).collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let th = pf.th as f64;
        {
            let st = &mut self.res.stats.pf;
            st.points += 1;
            st.rows += n;
            st.missed += cand.iter().map(|c| c.0).sum::<f64>();
            st.distinct += cand.len();
            for (i, &t) in PF_TOPS.iter().enumerate() {
                st.top[i] += cand.iter().take(t).map(|c| c.0).sum::<f64>();
            }
            let over: Vec<&(f64, usize)> = cand.iter().filter(|c| c.0 >= th).collect();
            st.over += over.len();
            st.over_w += over.iter().map(|c| c.0).sum::<f64>();
        }
        let free = pf.slots - pf.live.len();
        if !pf.on || free == 0 {
            return Ok(());
        }
        let pick: Vec<(f64, usize)> = cand.into_iter().filter(|c| c.0 >= th).take(free).collect();
        if pick.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        if pf.jobs.is_none() {
            pf.jobs = Some(readers(&self.pack, PF_READERS)?);
        }
        let mut waits = Vec::with_capacity(pick.len());
        for &(_, x) in &pick {
            let slot = pf.first + pf.live.len() + waits.len();
            let rec = self.pack.record(l, x)?;
            // a scratch slot is as wide as any expert of the decoder's MoE layers: it takes this
            // one's widths, and the record's own tensor sizes go to the slot's tensors' starts
            self.arena.set_widths(slot, rec.bits, &self.st)?;
            let sz = crate::arena::record_sizes(rec.bits, self.world, self.cfg.hidden, self.cfg.moe_inter);
            let d = self.arena.slot_dsts(slot);
            let dsts: [(usize, usize); 9] = std::array::from_fn(|t| (d[t].0, sz[t]));
            let (tx, rx) = mpsc::channel();
            pf.jobs.as_ref().unwrap().send(ReadJob { off: rec.off, nbytes: rec.nbytes, dsts, done: tx })
                .map_err(|_| "prefill load: the reader threads are gone")?;
            waits.push((slot, x, rec.nbytes, rx));
        }
        for (_, x, _, rx) in &waits {
            // a read that failed on one rank would leave the pair routing differently: fatal
            rx.recv().map_err(|_| "prefill load: the reader threads are gone")?
                .map_err(|err| format!("prefill load ({l}, {x}): {err}"))?;
        }
        for (&(w, _), (slot, x, nbytes, _)) in pick.iter().zip(waits) {
            let i = l * e + x;
            self.lut_h[i] = slot as i32;
            self.allowed_h[i] = 1;
            self.lut_entry(i)?;
            pf.live.push((slot, l, x));
            let st = &mut self.res.stats.pf;
            st.loads += 1;
            st.loaded_w += w;
            st.bytes += nbytes as u64;
        }
        self.res.stats.pf.read_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// After a MoE layer's last prefill sub-chunk: its read-in experts leave routing (queued on
    /// the stream after the layer's MoE; the next layer's reads start after its load point waits
    /// for the stream).
    pub(crate) fn pf_release(&mut self) -> Result<()> {
        let e = self.cfg.experts;
        let Some(live) = self.pf.as_mut().map(|p| std::mem::take(&mut p.live)) else { return Ok(()) };
        for (_, l, x) in live {
            let i = l * e + x;
            self.lut_h[i] = -1;
            self.allowed_h[i] = 0;
            self.lut_entry(i)?;
        }
        Ok(())
    }
}
