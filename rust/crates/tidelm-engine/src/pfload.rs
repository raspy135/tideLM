//! Prefill loads (DSV41_PF_LOAD; tidelm-glm's pfload.rs on the DeepSeek engine): in an eager
//! step (more than TMAX rows: a prompt's chunk), each backbone layer routes its rows exactly
//! first, and the missing experts the rows lean on most are made routable before the layer's
//! real routing - read into scratch slots past the hot-read ring, or, when the ring already
//! holds one, routed to its ring slot (a prompt step does not mark the ring: hot.rs) - instead
//! of every row's miss computing on its next-best resident. After the layer they leave routing
//! and the slots are the next layer's. On by default (tidelm-server --pf-load 4), by the author's
//! choice: its NLL A/B found no gain (tidelm-server main.rs has the numbers).
//!
//! A missing expert's weight is its share of each row's top-k (a row's weights normalized to
//! sum to one) summed over the step's live rows, in tokens' worth; one at or over the threshold
//! is a candidate, heaviest first, ties to the lower id. Both ranks plan from the replicated
//! exact routing in row order with f64 sums, so the pair reads the same experts with nothing
//! sent (the hot ring relies on the same); each reads its own half of every expert.

/// The reads and ring routes one load point makes, and what it saw.
#[derive(Debug, Default, PartialEq)]
pub struct Plan {
    /// experts routed to their live ring slot: (expert, slot, weight)
    pub ring: Vec<(usize, usize, f64)>,
    /// experts to read into free scratch slots: (expert, weight)
    pub reads: Vec<(usize, f64)>,
    /// missed weight before the plan, and the part of it at or over the threshold
    pub missed: f64,
    pub over: f64,
}

/// The plan for one layer: `ids` / `ws` the live rows' exact top-`k` ([row][k]), `routable(x)`
/// whether expert x is resident, `ring(x)` its live ring slot, `free` the scratch slots.
pub fn plan(ids: &[i32], ws: &[f32], k: usize, e: usize, routable: impl Fn(usize) -> bool,
            ring: impl Fn(usize) -> Option<usize>, th: f64, free: usize) -> Plan {
    let mut miss = vec![0f64; e];
    for (row_ids, row_ws) in ids.chunks(k).zip(ws.chunks(k)) {
        let s = row_ws.iter().map(|&w| w as f64).sum::<f64>().max(1e-20);
        for (&x, &w) in row_ids.iter().zip(row_ws) {
            if x >= 0 && (x as usize) < e && !routable(x as usize) {
                miss[x as usize] += w as f64 / s;
            }
        }
    }
    let mut cand: Vec<(f64, usize)> = miss.iter().enumerate().filter(|(_, &m)| m > 0.0).map(|(x, &m)| (m, x)).collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut p = Plan { missed: cand.iter().map(|c| c.0).sum(), ..Default::default() };
    for (w, x) in cand.into_iter().take_while(|c| c.0 >= th) {
        p.over += w;
        match ring(x) {
            Some(slot) => p.ring.push((x, slot, w)),
            None if p.reads.len() < free => p.reads.push((x, w)),
            None => {}
        }
    }
    p
}

/// Counters since boot.
#[derive(Clone, Debug, Default)]
pub struct PfStats {
    /// load points (a backbone layer of an eager step) and their live rows
    pub points: u64,
    pub rows: u64,
    /// missed weight in tokens' worth before the plans, and the part at or over the threshold
    pub missed: f64,
    pub over: f64,
    /// experts read in, the weight they covered, this rank's bytes and the host seconds
    pub loads: u64,
    pub loaded_w: f64,
    pub bytes: u64,
    pub read_s: f64,
    /// experts routed to the hot-read ring's slot instead, and their weight
    pub ring: u64,
    pub ring_w: f64,
}

impl PfStats {
    /// The counters since `s0` (an earlier copy).
    pub fn since(&self, s0: &PfStats) -> PfStats {
        PfStats { points: self.points - s0.points, rows: self.rows - s0.rows, missed: self.missed - s0.missed,
                  over: self.over - s0.over, loads: self.loads - s0.loads, loaded_w: self.loaded_w - s0.loaded_w,
                  bytes: self.bytes - s0.bytes, read_s: self.read_s - s0.read_s, ring: self.ring - s0.ring,
                  ring_w: self.ring_w - s0.ring_w }
    }

    pub fn line(&self) -> String {
        let pct = |x: f64| if self.missed > 0.0 { 100.0 * x / self.missed } else { 0.0 };
        format!("prefill loads: {} reads ({:.2} GB, {:.2} s) and {} ring routes over {} layer points since boot; {:.0}% read and \
                 {:.0}% ring of {:.0} tokens' worth missed ({:.1} a point of {:.0} rows), {:.0}% over the threshold",
                self.loads, self.bytes as f64 / 1e9, self.read_s, self.ring, self.points, pct(self.loaded_w), pct(self.ring_w),
                self.missed, self.missed / (self.points.max(1) as f64), self.rows as f64 / (self.points.max(1) as f64),
                pct(self.over))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_sums_shares_and_takes_the_heaviest_over_the_threshold() {
        // 3 rows, top-2; expert 0 resident. Row shares: (0: .5, 1: .5), (1: .75, 2: .25), (1: .5, 3: .5)
        let ids = [0, 1, 1, 2, 1, 3];
        let ws = [1.0, 1.0, 3.0, 1.0, 2.0, 2.0];
        let p = plan(&ids, &ws, 2, 4, |x| x == 0, |_| None, 0.5, 8);
        assert_eq!(p.reads, vec![(1, 1.75), (3, 0.5)]);
        assert!((p.missed - 2.5).abs() < 1e-12 && (p.over - 2.25).abs() < 1e-12);
        // one free slot: the heaviest only; ties go to the lower id
        assert_eq!(plan(&ids, &ws, 2, 4, |x| x == 0, |_| None, 0.25, 1).reads, vec![(1, 1.75)]);
        let tie = plan(&[2, 1], &[1.0, 1.0], 2, 4, |_| false, |_| None, 0.5, 1);
        assert_eq!(tie.reads, vec![(1, 0.5)]);
    }

    #[test]
    fn a_ring_expert_is_routed_to_its_slot_and_takes_no_scratch() {
        let ids = [1, 2, 1, 2];
        let ws = [1.0, 1.0, 1.0, 1.0];
        let p = plan(&ids, &ws, 2, 4, |_| false, |x| (x == 1).then_some(77), 1.0, 1);
        assert_eq!(p.ring, vec![(1, 77, 1.0)]);
        assert_eq!(p.reads, vec![(2, 1.0)]);
        // out-of-range and filler ids are ignored
        assert_eq!(plan(&[-1, 9], &[1.0, 1.0], 2, 4, |_| false, |_| None, 0.0, 4), Plan::default());
    }
}
