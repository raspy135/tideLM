//! Adaptive expert residency: the Python engine's demand history, its blend with the shipped
//! routing trace, the boot selection and the swap planner (engine/adapt_config.py,
//! v41_engine.py's prune DB / blend_demand / plan_swaps, model.py's flush_request_demand),
//! over tidelm_core::residency's pure functions. The scheduler (decode.rs) folds requests in
//! and asks for plans; Engine::move_experts carries them out on both ranks.
//!
//! Units are the Python engine's with its knobs set (source "knobs", request units, score
//! metric): history `counts` / `mass` [layer][expert] f64, each fold a vote that sums to one
//! per layer, decayed by d = 0.5^(1/halflife) at every fold. A request folds twice - at the
//! prefill -> decode boundary and at its end - so a half-life of 10 folds is ~5 requests.
//!
//! One difference from the Python engine, by construction rather than choice: its prefill
//! runs layers 21-39 only over the prompt's last 128 rows (SWA decoder replay), so their
//! prompt vote comes from those rows; this engine computes every layer for every row, and
//! every row it routes votes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tidelm_core::npz;
use tidelm_core::residency::{blend_demand, global_plan, initial_selection, DemandMetric};

use crate::forward::LaneDemand;
use crate::Result;

/// The resolved settings (adapt_config.resolve with the knobs set).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptConfig {
    /// demand half-life in folds
    pub halflife: f64,
    /// the shipped trace's weight, in folds: w = n / (n + prior) per layer
    pub prior: f64,
    /// the miss rate a prefill (or a decode stretch, or a probe) must reach before a pass
    /// re-fits the residents for it
    pub gate: f64,
    /// new prompt tokens a prompt needs before the boundary pass
    pub min_new: usize,
    /// output tokens between decode passes (0: none)
    pub decode_tokens: usize,
    pub swap_max: usize,
    /// a swap must gain more than this times the mean score
    pub min_gain: f64,
    /// residents a layer keeps at least (the router's top-k)
    pub floor: usize,
}

impl AdaptConfig {
    /// `sensitivity`: off, low, medium, high, max (half-lives 40, 20, 10, 5 folds) or the
    /// newest vote's share in [0, 0.5); `prior` in folds; `decode_tokens` 0 turns decode
    /// passes off. None: adaptation off (the boot ranking stays frozen).
    pub fn new(sensitivity: &str, prior: f64, decode_tokens: usize, floor: usize) -> Result<Option<Self>> {
        let v = sensitivity.trim().to_ascii_lowercase();
        let halflife = match v.as_str() {
            "off" => return Ok(None),
            "low" => 40.0,
            "medium" => 20.0,
            "high" => 10.0,
            "max" => 5.0,
            _ => {
                let s: f64 = v.parse().map_err(|_| format!("adapt sensitivity {sensitivity:?}: off, low, medium, high, max or a number in [0, 0.5)"))?;
                if !(0.0..0.5).contains(&s) {
                    return Err(format!("adapt sensitivity {s} must be in [0, 0.5)").into());
                }
                if s == 0.0 {
                    return Ok(None);
                }
                0.5f64.ln() / (1.0 - s).ln()
            }
        };
        if !(prior >= 0.0 && prior.is_finite()) {
            return Err(format!("adapt prior {prior} must be >= 0").into());
        }
        let s = 1.0 - 0.5f64.powf(1.0 / halflife);
        Ok(Some(AdaptConfig {
            halflife,
            prior,
            // 2% at medium; more sensitive acts on smaller misses, clamped (adapt_config.py)
            gate: (0.02 * (1.0 - 0.5f64.powf(1.0 / 20.0)) / s).clamp(0.005, 0.2),
            min_new: 32,
            decode_tokens,
            swap_max: 512,
            min_gain: 0.005,
            floor,
        }))
    }

    /// The per-fold decay.
    pub fn decay(&self) -> f64 {
        0.5f64.powf(1.0 / self.halflife)
    }
}

/// The trace prior from a routing trace's coverage.json (`per_layer[L].counts_coding` and
/// `counts_general`), ranked "sum": each category normalized per layer, then added
/// (v41_engine.py: favours experts both workloads use).
pub fn load_trace(path: &Path, layers: usize, experts: usize) -> Result<BTreeMap<u32, Vec<f64>>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let per = v.get("per_layer").ok_or_else(|| format!("{}: no per_layer", path.display()))?;
    let mut out = BTreeMap::new();
    for l in 0..layers {
        let row = per.get(l.to_string()).ok_or_else(|| format!("{}: no layer {l}", path.display()))?;
        let counts = |key: &str| -> Result<Vec<f64>> {
            let a = row.get(key).and_then(|x| x.as_array()).ok_or_else(|| format!("{}: layer {l} has no {key}", path.display()))?;
            let c: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
            if c.len() != experts || c.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(format!("{}: layer {l} {key}: {} finite counts expected", path.display(), experts).into());
            }
            Ok(c)
        };
        let (cc, cg) = (counts("counts_coding")?, counts("counts_general")?);
        let (sc, sg) = (cc.iter().sum::<f64>(), cg.iter().sum::<f64>());
        out.insert(l as u32, cc.iter().zip(&cg).map(|(a, b)| a / sc + b / sg).collect());
    }
    Ok(out)
}

/// The demand history and what is planned from it.
pub struct Adapt {
    pub cfg: AdaptConfig,
    layers: usize,
    experts: usize,
    trace: BTreeMap<u32, Vec<f64>>,
    /// the history, [layer * experts + e]
    counts: Vec<f64>,
    mass: Vec<f64>,
    db: Option<PathBuf>,
    pub folds: u64,
    /// plans applied (Python's expert_generation)
    pub generation: u64,
}

impl Adapt {
    /// `db`: the history file (np.savez: counts, mass, unit = 1, score_history_version = 2;
    /// the Python engine's DSV41_PRUNE_DB format, shared with it), read if present and
    /// valid - otherwise the history starts empty, as Python's does - and written after every
    /// request.
    pub fn new(cfg: AdaptConfig, trace: BTreeMap<u32, Vec<f64>>, layers: usize, experts: usize, db: Option<PathBuf>)
               -> Result<Self> {
        if trace.len() != layers || trace.values().any(|t| t.len() != experts) {
            return Err(format!("the trace needs {layers} layers of {experts} experts").into());
        }
        let mut a = Adapt { cfg, layers, experts, trace, counts: vec![0.0; layers * experts], mass: vec![0.0; layers * experts], db,
                            folds: 0, generation: 0 };
        if let Some(path) = a.db.clone() {
            match a.load(&path) {
                Ok(true) => eprintln!("adapt: demand history {} loaded", path.display()),
                Ok(false) => eprintln!("adapt: no demand history at {}; starting from the trace", path.display()),
                Err(e) => eprintln!("adapt: demand history {}: {e}; starting fresh", path.display()),
            }
        }
        Ok(a)
    }

    fn load(&mut self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }
        let m = npz::read(path)?;
        let scalar = |k: &str| m.get(k).map(|a| a.to_i64()).transpose().map(|v| v.and_then(|v| v.first().copied()).unwrap_or(0));
        if scalar("unit")? != 1 {
            return Err("not request-unit demand".into());
        }
        if scalar("score_history_version")? != 2 {
            return Err("legacy unnormalized score history".into());
        }
        let arr = |k: &str| -> Result<Vec<f64>> {
            let a = m.get(k).ok_or_else(|| format!("no {k}"))?;
            if a.shape != [self.layers, self.experts] {
                return Err(format!("{k} is {:?}, expected [{}, {}]", a.shape, self.layers, self.experts).into());
            }
            let v = a.to_f64()?;
            if v.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(format!("{k} has non-finite or negative entries").into());
            }
            Ok(v)
        };
        let (c, mm) = (arr("counts")?, arr("mass")?);
        self.counts = c;
        self.mass = mm;
        Ok(true)
    }

    /// The history to its file (rank 0, after every request). A failure is reported, not
    /// fatal: serving does not depend on it.
    pub fn save(&self) {
        let Some(path) = &self.db else { return };
        let mut m = BTreeMap::new();
        m.insert("counts".to_string(), npz::Array::f64s(vec![self.layers, self.experts], &self.counts));
        m.insert("mass".to_string(), npz::Array::f64s(vec![self.layers, self.experts], &self.mass));
        m.insert("unit".to_string(), npz::Array::i64s(vec![1], &[1]));
        m.insert("score_history_version".to_string(), npz::Array::i64s(vec![1], &[2]));
        if let Err(e) = npz::write(path, &m) {
            eprintln!("adapt: saving the demand history to {}: {e}", path.display());
        }
    }

    /// history * d + one normalized vote of `d` per layer (model.py::flush_request_demand):
    /// mass by its score total, counts by its pick total where the layer had any.
    fn with_vote(&self, d: &LaneDemand) -> (Vec<f64>, Vec<f64>) {
        let decay = self.cfg.decay();
        let (mut c, mut m): (Vec<f64>, Vec<f64>) =
            (self.counts.iter().map(|x| x * decay).collect(), self.mass.iter().map(|x| x * decay).collect());
        let e = self.experts;
        for l in 0..self.layers {
            let r = l * e..(l + 1) * e;
            let sm = d.mass[r.clone()].iter().sum::<f64>().max(1e-20);
            for (h, x) in m[r.clone()].iter_mut().zip(&d.mass[r.clone()]) {
                *h += x / sm;
            }
            let sc: f64 = d.counts[r.clone()].iter().sum();
            if sc > 0.0 {
                for (h, x) in c[r.clone()].iter_mut().zip(&d.counts[r.clone()]) {
                    *h += x / sc;
                }
            }
        }
        (c, m)
    }

    /// Fold one request's demand (or its prompt's, at the boundary) into the history.
    pub fn fold(&mut self, d: &LaneDemand) {
        let (c, m) = self.with_vote(d);
        self.counts = c;
        self.mass = m;
        self.folds += 1;
    }

    /// Per layer, the trace blended with `mass` (blend_demand, score metric, request unit):
    /// each layer's ranking sums to one.
    fn ranked(&self, counts: &[f64], mass: &[f64]) -> BTreeMap<u32, Vec<f64>> {
        let e = self.experts;
        let rows = |v: &[f64]| -> BTreeMap<u32, Vec<f64>> { (0..self.layers).map(|l| (l as u32, v[l * e..(l + 1) * e].to_vec())).collect() };
        blend_demand(&self.trace, &rows(counts), &rows(mass), DemandMetric::Score, true, self.cfg.prior).0
    }

    /// The boot resident set: the top `floor` of every layer, then the globally highest
    /// blended scores up to `budget` (global_residency.initial_selection).
    pub fn boot_keep(&self, budget: usize) -> Result<Vec<Vec<bool>>> {
        let ranked = self.ranked(&self.counts, &self.mass);
        let scores: BTreeMap<u32, Vec<f64>> = ranked
            .into_iter()
            .map(|(l, r)| {
                let s: f64 = r.iter().sum();
                (l, r.iter().map(|x| if s > 0.0 { x / s } else { 0.0 }).collect())
            })
            .collect();
        let chosen = initial_selection(&scores, budget, self.cfg.floor).map_err(|e| e.0)?;
        Ok((0..self.layers)
            .map(|l| {
                let mut k = vec![false; self.experts];
                for &x in &chosen[&(l as u32)] {
                    k[x] = true;
                }
                k
            })
            .collect())
    }

    /// The moves (from layer, expert, to layer, expert) that re-fit `keep` to the history -
    /// previewed with `pending` (a decode stretch's or a probe's demand, as a vote that is
    /// not cast) - at most swap_max, each gaining more than min_gain x the mean score
    /// (v41_engine.plan_swaps -> global_residency.global_plan).
    pub fn plan(&self, keep: &[Vec<bool>], pending: Option<&LaneDemand>) -> Result<Vec<[usize; 4]>> {
        let (c, m) = match pending {
            Some(d) => self.with_vote(d),
            None => (self.counts.clone(), self.mass.clone()),
        };
        if c.iter().sum::<f64>() <= 0.0 {
            return Ok(Vec::new());
        }
        let ranked = self.ranked(&c, &m);
        let n: usize = ranked.values().map(Vec::len).sum();
        let mean = ranked.values().flatten().sum::<f64>() / n.max(1) as f64;
        let keeps: BTreeMap<u32, Vec<bool>> = keep.iter().enumerate().map(|(l, k)| (l as u32, k.clone())).collect();
        let (swaps, _) = global_plan(&keeps, &ranked, self.cfg.floor, Some(self.cfg.swap_max), None, None, self.cfg.min_gain * mean)
            .map_err(|e| e.0)?;
        Ok(swaps.into_iter().map(|(lo, eo, li, ei, _)| [lo as usize, eo, li as usize, ei]).collect())
    }

    /// The probe's gate quantity (predictive_prefill.py): per layer the share of the probe's
    /// picks the residents miss, averaged over layers (not token-weighted).
    pub fn predicted_miss(&self, keep: &[Vec<bool>], d: &LaneDemand) -> f64 {
        let e = self.experts;
        let per: f64 = (0..self.layers)
            .map(|l| {
                let row = &d.counts[l * e..(l + 1) * e];
                let s: f64 = row.iter().sum();
                if s <= 0.0 {
                    return 0.0;
                }
                row.iter().enumerate().filter(|&(x, _)| !keep[l][x]).map(|(_, &v)| v / s).sum::<f64>()
            })
            .sum();
        per / self.layers as f64
    }
}

/// The probe before a prompt's prefill (probe_prefill.SmallProbe, the Python engine's
/// DSV41_PROBE_PREFILL=1 "pre-prefill"): a short excerpt of the prompt (or, below
/// `two_pass_below` tokens, all of it) is routed with the current residents first, and if the
/// residents would miss enough of what it asks for, they are re-fitted before the real
/// prefill. The excerpt runs in the request's own lane at positions 0.. and the lane is reset
/// after it; its demand is a preview, never a vote.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeConfig {
    pub max_tokens: usize,
    pub fraction: f64,
    pub min_tokens: usize,
    pub two_pass_below: usize,
}

impl Default for ProbeConfig {
    /// the serving profile: DSV41_PROBE_TOKENS=64, PROBE_FRACTION=.1, PROBE_MIN_TOKENS=32,
    /// PROBE_TWO_PASS_BELOW=500
    fn default() -> Self {
        ProbeConfig { max_tokens: 64, fraction: 0.1, min_tokens: 32, two_pass_below: 500 }
    }
}

impl ProbeConfig {
    /// The prompt positions a `len`-token prompt's probe feeds and its kind, or None for no
    /// probe (probe_prefill.SmallProbe.predict), with `prefix` tokens restored from the prefix
    /// cache - every threshold is on the `len - prefix` new tokens: below two_pass_below the
    /// new suffix whole (on the restored context); none below `min_new`; else an excerpt of
    /// min(max_tokens, new x fraction) tokens, none under min_tokens. An excerpt's positions
    /// below `prefix` (its head, in the cached part) give context but do not vote.
    pub fn positions(&self, len: usize, prefix: usize, min_new: usize) -> Option<(Vec<usize>, &'static str)> {
        let new = len.saturating_sub(prefix);
        if new > 0 && new < self.two_pass_below {
            return Some(((prefix..len).collect(), "two-pass"));
        }
        if new < min_new {
            return None;
        }
        let limit = self.max_tokens.min((new as f64 * self.fraction) as usize);
        if limit < self.min_tokens {
            return None;
        }
        let spans = select_spans(len, prefix, limit);
        (!spans.is_empty()).then(|| (spans.iter().flat_map(|&(a, b)| a..b).collect(), "excerpt"))
    }
}

/// The probe's excerpt (probe_prefill.select_spans): the instruction head, two windows of the
/// suffix and its tail (half the budget), merged where they touch; never the whole suffix.
pub fn select_spans(length: usize, prefix: usize, limit: usize) -> Vec<(usize, usize)> {
    let limit = limit.min(length.saturating_sub(prefix + 1));
    if limit < 4 {
        return Vec::new();
    }
    let (head, tail) = ((limit / 8).max(1), limit / 2);
    let middle = limit - head - tail;
    let stop = length - tail;
    let mut spans = vec![(0usize, head)];
    for (i, size) in [(0i64, middle / 2), (1, middle - middle / 2)] {
        if size == 0 {
            continue;
        }
        let lo = prefix.max(head) as i64;
        // Python's (stop - lo - size) * i // 2: a floor division
        let start = (lo + ((stop as i64 - lo - size as i64) * i).div_euclid(2)) as usize;
        spans.push((start, start + size));
    }
    spans.push((stop, length));
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// A demand's miss rate over both phases (the boundary gate) and over decode alone.
pub fn miss_rate(d: &LaneDemand) -> (f64, f64) {
    let rate = |missed: f64, picks: f64| if picks > 0.0 { missed / picks } else { 0.0 };
    (rate(d.miss[0] + d.miss[4], d.miss[1] + d.miss[5]), rate(d.miss[4], d.miss[5]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(layers: usize, e: usize) -> BTreeMap<u32, Vec<f64>> {
        (0..layers).map(|l| (l as u32, (0..e).map(|x| ((x * 7 + l) % e) as f64 + 1.0).collect())).collect()
    }

    #[test]
    fn high_sensitivity_resolves_as_the_python_knobs_do() {
        let c = AdaptConfig::new("high", 4.0, 600, 6).unwrap().unwrap();
        assert_eq!(c.halflife, 10.0);
        // adapt_config.py: min(0.2, max(0.005, 0.02 * (1 - 0.5 ** (1/20)) / s)), s = 1 - 0.5 ** (1/10)
        assert!((c.gate - 0.010173).abs() < 1e-6, "{}", c.gate);
        assert!((c.decay() - 0.9330329915).abs() < 1e-9);
        assert!(AdaptConfig::new("off", 4.0, 600, 6).unwrap().is_none());
        assert!((AdaptConfig::new("0.034", 4.0, 600, 6).unwrap().unwrap().halflife - 20.0).abs() < 0.3);
    }

    #[test]
    fn a_fold_is_one_normalized_vote_per_layer() {
        let (layers, e) = (2, 8);
        let mut a = Adapt::new(AdaptConfig::new("high", 4.0, 600, 2).unwrap().unwrap(), trace(layers, e), layers, e, None).unwrap();
        let mut d = LaneDemand { counts: vec![0.0; layers * e], mass: vec![0.0; layers * e], miss: [0.0; 8] };
        d.counts[3] = 30.0;
        d.mass[3] = 12.0;
        d.mass[5] = 4.0;
        a.fold(&d);
        assert_eq!(a.counts[3], 1.0);
        assert_eq!((a.mass[3], a.mass[5]), (0.75, 0.25));
        // layer 1 had no demand: nothing added, nothing broken
        assert!(a.mass[e..].iter().all(|&x| x == 0.0));
        a.fold(&d);
        assert!((a.mass[3] - (0.75 * a.cfg.decay() + 0.75)).abs() < 1e-12);
    }

    #[test]
    fn a_plan_brings_in_what_the_history_wants() {
        let (layers, e) = (2, 8);
        let mut a = Adapt::new(AdaptConfig::new("high", 0.0, 600, 2).unwrap().unwrap(), trace(layers, e), layers, e, None).unwrap();
        let keep: Vec<Vec<bool>> = (0..layers).map(|_| (0..e).map(|x| x < 4).collect()).collect();
        let mut d = LaneDemand { counts: vec![0.0; layers * e], mass: vec![0.0; layers * e], miss: [0.0; 8] };
        for l in 0..layers {
            d.counts[l * e + 6] = 10.0;
            d.mass[l * e + 6] = 10.0;
            d.counts[l * e + 1] = 1.0;
            d.mass[l * e + 1] = 1.0;
        }
        assert!(a.plan(&keep, None).unwrap().is_empty(), "an empty history plans nothing");
        a.fold(&d);
        let plan = a.plan(&keep, None).unwrap();
        assert!(plan.iter().any(|&[_, _, lt, xt]| xt == 6 && lt < layers), "{plan:?}");
        assert!(plan.iter().all(|&[lf, xf, _, _]| keep[lf][xf]));
        // a preview casts no vote
        let before = a.mass.clone();
        let _ = a.plan(&keep, Some(&d)).unwrap();
        assert_eq!(a.mass, before);
    }

    #[test]
    fn probe_spans_are_probe_prefills() {
        // probe_prefill.select_spans's worked examples (64 tokens of a 10,000-token prompt;
        // 50 of a 500-token one)
        assert_eq!(select_spans(10000, 0, 64), vec![(0, 20), (4982, 4994), (9968, 10000)]);
        assert_eq!(select_spans(500, 0, 50), vec![(0, 15), (235, 245), (475, 500)]);
        assert_eq!(select_spans(11774, 10810, 64), vec![(0, 8), (10810, 10822), (11270, 11282), (11742, 11774)]);
        assert_eq!(select_spans(5, 0, 64), vec![(0, 2), (3, 5)]);
        assert!(select_spans(4, 0, 64).is_empty());
        let p = ProbeConfig::default();
        assert_eq!(p.positions(499, 0, 32).unwrap().1, "two-pass");
        assert_eq!(p.positions(499, 0, 32).unwrap().0.len(), 499);
        let (pos, kind) = p.positions(500, 0, 32).unwrap();
        assert_eq!((kind, pos.len()), ("excerpt", 50));
        assert_eq!(p.positions(10000, 0, 32).unwrap().0.len(), 64);
        // a cached prefix: the new suffix decides; a short one is probed whole, from the prefix
        assert_eq!(p.positions(10100, 10000, 32).unwrap(), ((10000..10100).collect(), "two-pass"));
        assert!(p.positions(10000, 10000, 32).is_none());
        let (pos, kind) = p.positions(11774, 10810, 32).unwrap();
        assert_eq!((kind, pos.len(), pos[..8].to_vec()), ("excerpt", 64, (0..8).collect::<Vec<_>>()));
    }

    #[test]
    fn the_history_round_trips_through_the_python_format() {
        let (layers, e) = (2, 8);
        let dir = std::env::temp_dir().join(format!("tidelm-adapt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("db.npz");
        let cfg = AdaptConfig::new("high", 4.0, 600, 2).unwrap().unwrap();
        let mut a = Adapt::new(cfg.clone(), trace(layers, e), layers, e, Some(path.clone())).unwrap();
        let mut d = LaneDemand { counts: vec![1.0; layers * e], mass: vec![2.0; layers * e], miss: [0.0; 8] };
        d.mass[0] = 9.0;
        a.fold(&d);
        a.save();
        let b = Adapt::new(cfg, trace(layers, e), layers, e, Some(path)).unwrap();
        assert_eq!((b.counts.clone(), b.mass.clone()), (a.counts.clone(), a.mass.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
