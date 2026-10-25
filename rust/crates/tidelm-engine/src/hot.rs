//! Async hot reads (engine/hot_reads.py with DSV41_HOT_ASYNC=1, fastdecode.py's
//! _hot_async_begin / _hot_async_end): the heaviest missing experts the router asked for in
//! one decode step are read from NVMe into a ring of spare arena slots during the next step,
//! and are routable from the step after that until the ring wraps over them.
//!
//! Each decode step's router leaves, per backbone layer, the 12 heaviest (share, expert)
//! entries of its rows' unmasked top-6 that were not routable - share = the pick's part of
//! its row's top-6 score (ds_router_rec's `hot`, ds_hot_candidates). The next step plans from
//! that table (plan_async: the heaviest `cap` across layers, `per_layer` a layer, at least
//! `min` share, none already in the ring), takes ring slots for them (FIFO; the slot's old
//! expert stops being routable at once), starts the reads, and marks every live ring expert
//! routable for its graphs (LUT -> ring slot, keep -> 1), undone after them. Both ranks of a
//! pair route the same rows to the same table, so they plan the same reads with nothing sent.

use std::collections::{BTreeMap, BTreeSet};

/// DSV41_HOT_MIN / HOT_MAX / HOT_PER_LAYER / HOT_RING as the serving profile sets them
/// (0.3, 2, 1, 128): TIDELM_HOT_MIN / TIDELM_HOT_MAX / TIDELM_HOT_PER_LAYER, the ring from
/// Options::hot_ring.
#[derive(Clone, Debug, PartialEq)]
pub struct HotConfig {
    pub min: f32,
    pub cap: usize,
    pub per_layer: usize,
    pub ring: usize,
}

/// Candidate entries a layer leaves (hot_reads.py: width = 2 x top-k).
pub const WIDTH: usize = 12;

impl HotConfig {
    pub fn from_env(ring: usize, topk: usize) -> crate::Result<Self> {
        let get = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.into());
        let cfg = HotConfig {
            min: get("TIDELM_HOT_MIN", "0.3").parse().map_err(|_| "TIDELM_HOT_MIN: a share")?,
            cap: get("TIDELM_HOT_MAX", "2").parse().map_err(|_| "TIDELM_HOT_MAX: reads a step")?,
            per_layer: get("TIDELM_HOT_PER_LAYER", "1").parse().map_err(|_| "TIDELM_HOT_PER_LAYER: reads a layer")?,
            ring,
        };
        if cfg.ring < cfg.cap {
            return Err("the hot-read ring must hold at least one step's reads".into());
        }
        if !(1..=topk).contains(&cfg.per_layer) {
            return Err(format!("TIDELM_HOT_PER_LAYER must be in 1..={topk}").into());
        }
        Ok(cfg)
    }
}

/// The ring of read-in experts (hot_reads.HotRing): FIFO by read time, no LRU touch.
pub struct HotRing {
    slots: Vec<usize>,
    owner: Vec<Option<(usize, usize)>>,
    pos: usize,
    /// routable this step
    pub live: BTreeMap<(usize, usize), usize>,
    /// being read this step
    pub pending: BTreeMap<(usize, usize), usize>,
}

impl HotRing {
    pub fn new(slots: Vec<usize>) -> Self {
        let n = slots.len();
        HotRing { slots, owner: vec![None; n], pos: 0, live: BTreeMap::new(), pending: BTreeMap::new() }
    }

    /// The next slot for `key`; its previous expert stops being routable at once.
    pub fn take(&mut self, key: (usize, usize)) -> usize {
        let i = self.pos % self.slots.len();
        self.pos += 1;
        if let Some(old) = self.owner[i].take() {
            self.live.remove(&old);
            self.pending.remove(&old);
        }
        self.owner[i] = Some(key);
        self.pending.insert(key, self.slots[i]);
        self.slots[i]
    }

    /// This step's reads are in: routable from the next step.
    pub fn promote(&mut self) {
        let p = std::mem::take(&mut self.pending);
        self.live.extend(p);
    }

    pub fn taken(&self) -> BTreeSet<(usize, usize)> {
        self.live.keys().chain(self.pending.keys()).copied().collect()
    }
}

/// Counters since boot (hot_reads.HotStats).
#[derive(Clone, Debug, Default)]
pub struct HotStats {
    pub steps: u64,
    pub reads: u64,
    pub capped: u64,
    /// host seconds waiting for the reads at the end of their step
    pub wait_s: f64,
}

/// The reads one step makes from the last decode step's candidate table `rows`
/// ([layer][2 x WIDTH]: shares descending, then the expert ids): the heaviest first (ties:
/// lower layer, then lower expert), each (layer, expert) once and none in `taken`, at most
/// `cap` in all and `per_layer` a layer (hot_reads.HotReads.plan_async).
pub fn plan_async(cfg: &HotConfig, rows: &[f32], layers: usize, taken: &BTreeSet<(usize, usize)>, stats: &mut HotStats)
                  -> Vec<(usize, usize)> {
    let mut cands: Vec<(f64, usize, usize)> = Vec::new();
    for l in 0..layers {
        let row = &rows[l * 2 * WIDTH..(l + 1) * 2 * WIDTH];
        for i in 0..WIDTH {
            let v = row[i];
            if v < cfg.min {
                break;
            }
            cands.push((-(v as f64), l, row[WIDTH + i] as usize));
        }
    }
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let (mut out, mut per, mut seen) = (Vec::new(), BTreeMap::<usize, usize>::new(), BTreeSet::new());
    for (_, l, e) in cands {
        let key = (l, e);
        if seen.contains(&key) || taken.contains(&key) {
            continue;
        }
        seen.insert(key);
        if out.len() >= cfg.cap || *per.get(&l).unwrap_or(&0) >= cfg.per_layer {
            stats.capped += 1;
            continue;
        }
        *per.entry(l).or_insert(0) += 1;
        out.push(key);
    }
    stats.reads += out.len() as u64;
    stats.steps += 1;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(cap: usize, ring: usize) -> HotConfig {
        HotConfig { min: 0.2, cap, per_layer: 1, ring }
    }

    fn table(rows: &[Vec<(f32, usize)>]) -> Vec<f32> {
        rows.iter()
            .flat_map(|pairs| {
                let mut r = vec![0f32; 2 * WIDTH];
                for (i, &(v, e)) in pairs.iter().enumerate() {
                    r[i] = v;
                    r[WIDTH + i] = e as f32;
                }
                r
            })
            .collect()
    }

    /// engine/test_hot_reads.py::test_async_plan_takes_the_heaviest_across_layers_once
    #[test]
    fn the_plan_takes_the_heaviest_across_layers_once() {
        let rows = table(&[vec![(0.4, 3)], vec![(0.9, 5), (0.8, 6)], vec![(0.1, 2)], vec![(0.5, 7)]]);
        let mut st = HotStats::default();
        // one a layer: layer 1's 0.9, then layer 3's 0.5 (layer 1's 0.8 is its second); cap 2
        assert_eq!(plan_async(&cfg(2, 4), &rows, 4, &BTreeSet::new(), &mut st), vec![(1, 5), (3, 7)]);
        // the ring has (1, 5)
        assert_eq!(plan_async(&cfg(2, 4), &rows, 4, &[(1, 5)].into_iter().collect(), &mut st), vec![(1, 6), (3, 7)]);
    }

    /// engine/test_hot_reads.py::test_ring_is_fifo_and_a_read_is_routable_from_the_next_step
    #[test]
    fn the_ring_is_fifo_and_a_read_is_routable_from_the_next_step() {
        let mut r = HotRing::new(vec![100, 101]);
        assert_eq!(r.take((0, 1)), 100);
        assert!(r.live.is_empty());
        r.promote();
        assert_eq!(r.live, [((0, 1), 100)].into_iter().collect());
        assert_eq!(r.take((0, 2)), 101);
        assert_eq!(r.take((0, 3)), 100); // the oldest slot; (0, 1) stops at once
        assert!(r.live.is_empty());
        assert_eq!(r.taken(), [(0, 2), (0, 3)].into_iter().collect());
        r.promote();
        assert_eq!(r.live, [((0, 2), 101), ((0, 3), 100)].into_iter().collect());
    }
}
