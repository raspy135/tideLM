//! Speculative depth policies, from `engine/spec_depth.py`: recent acceptance
//! (`DepthPolicy`, DSV41_BLOCK_DYNAMIC) and per-draft confidence
//! (`ConfidenceDepthPolicy`, DSV41_BLOCK_CONFIDENCE).
//!
//! Why: the best draft depth depends on how predictable the text is. Measured on TP2
//! (2026-09-23, docs/decode-dynamic-depth.md): depth 5 makes every verify step ~16%
//! slower (107 -> 124 ms) and pays that back only where the drafter is right — code
//! went 35.8 -> 44.6 tok/s, while prose and stories LOST 9-13%. So the engine
//! re-decides as it goes. The rule compares expected tokens per second, never
//! acceptance alone; see the Python docstring for the full derivation, which this
//! port preserves decision for decision.
//!
//! Rank 0 decides; the engine broadcasts the depth with the per-step control flag, so
//! both ranks always verify the same width. Rank 1 never consults its own copy.

use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, PartialEq, Eq)]
pub struct DepthError(pub String);

impl std::fmt::Display for DepthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DepthError {}

fn median_of(samples: &VecDeque<f64>) -> f64 {
    let mut v: Vec<f64> = samples.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn push_sample(samples: &mut VecDeque<f64>, cap: usize, value: f64) {
    if samples.len() == cap {
        samples.pop_front();
    }
    samples.push_back(value);
}

/// Median once MIN_SAMPLES exist, the minimum before that: a median ignores the
/// occasional slow step, and with too few samples the minimum is the safe side
/// (a slow first step is an overhead, never a speedup).
fn robust_time(samples: &VecDeque<f64>, min_samples: usize) -> f64 {
    if samples.len() >= min_samples {
        median_of(samples)
    } else {
        samples.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Numerically stable sigmoid, including arbitrarily large finite logits.
fn sigmoid(x: f64) -> f64 {
    let z = (-x.abs()).exp();
    if x >= 0.0 {
        1.0 / (1.0 + z)
    } else {
        z / (1.0 + z)
    }
}

/// The evidence behind a depth switch, for the server log (rank 0 pops it once).
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchEvidence {
    pub from: usize,
    pub to: usize,
    pub steps: u64,
    pub tokens_per_step: f64,
    pub rate_lo: f64,
    pub rate_hi: f64,
    /// Which side of the comparison was the estimate (the depth not being run).
    pub estimated: usize,
}

/// Tuning knobs of [`DepthPolicy`]. The Python constructor read the DSV41_* environment
/// for its defaults; in Rust the boot guard resolves the environment and passes values,
/// so per-rank config drift stays visible in one place.
#[derive(Debug, Clone, Copy)]
pub struct DepthOptions {
    pub start: Option<usize>,
    /// A decision happens at most once per this many emitted tokens.
    pub interval: u64,
    /// ...and needs this fractional win.
    pub margin: f64,
    /// hi/lo step-time ratio until both have been measured: 124/107 ms on TP2 at 3 vs 5.
    pub prior_ratio: f64,
    /// Prior mean extra drafts accepted past `lo` on a saturated deep step.
    pub extra_prior: f64,
}

impl Default for DepthOptions {
    fn default() -> Self {
        DepthOptions { start: None, interval: 60, margin: 0.03, prior_ratio: 1.16, extra_prior: 1.5 }
    }
}

/// The DSV41_BLOCK_DYNAMIC two-depth controller.
#[derive(Debug, Clone)]
pub struct DepthPolicy {
    pub lo: usize,
    pub hi: usize,
    pub start: usize,
    pub interval: u64,
    pub margin: f64,
    pub prior_ratio: f64,
    pub extra_prior: f64,
    pub step_s: BTreeMap<usize, Option<f64>>,
    samples: BTreeMap<usize, VecDeque<f64>>,
    /// Tests/benchmarks: force one depth.
    pub pinned: Option<usize>,
    pub depth: usize,
    pub switches: u32,
    pub last_switch: Option<SwitchEvidence>,
    pub steps: BTreeMap<usize, u64>,
    extra_sum: f64,
    extra_n: u64,
    n: u64,
    tok: f64,
    tok_lo_cf: f64,
    saturated: u64,
    emitted: u64,
}

impl DepthPolicy {
    pub const EXTRA_WEIGHT: f64 = 5.0;
    pub const STEP_SAMPLES: usize = 32;
    pub const MIN_SAMPLES: usize = 3;

    pub fn new(depths: (usize, usize), opts: DepthOptions) -> Result<Self, DepthError> {
        let (lo, hi) = (depths.0.min(depths.1), depths.0.max(depths.1));
        let start = opts.start.unwrap_or(lo);
        if start != lo && start != hi {
            return Err(DepthError(format!(
                "DSV41_BLOCK_DYNAMIC_START={start} is not one of {lo}, {hi}"
            )));
        }
        let mut pol = DepthPolicy {
            lo,
            hi,
            start,
            interval: opts.interval,
            margin: opts.margin,
            prior_ratio: opts.prior_ratio,
            extra_prior: opts.extra_prior,
            step_s: [(lo, None), (hi, None)].into(),
            samples: [(lo, VecDeque::new()), (hi, VecDeque::new())].into(),
            pinned: None,
            depth: start,
            switches: 0,
            last_switch: None,
            steps: [(lo, 0), (hi, 0)].into(),
            extra_sum: 0.0,
            extra_n: 0,
            n: 0,
            tok: 0.0,
            tok_lo_cf: 0.0,
            saturated: 0,
            emitted: 0,
        };
        pol.reset_request();
        Ok(pol)
    }

    /// New request: back to the start depth. Step times and `extra` are kept — they
    /// describe this process, not the request — while the counters are per request.
    pub fn reset_request(&mut self) {
        self.depth = self.pinned.unwrap_or(self.start);
        self.switches = 0;
        self.last_switch = None;
        self.steps = [(self.lo, 0), (self.hi, 0)].into();
        self.extra_sum = 0.0;
        self.extra_n = 0;
        self.clear_window();
    }

    /// Mean extra drafts accepted past `lo` on a saturated deep step, learned per
    /// request as a prior-weighted mean: EXTRA_WEIGHT pseudo-observations of the prior,
    /// so a handful of real deep steps outweigh it. (An EWMA at 5%/step was tried first
    /// and oscillated: after correctly dropping to `lo`, the stale estimate pulled the
    /// policy straight back up.)
    pub fn extra(&self) -> f64 {
        (self.extra_prior * Self::EXTRA_WEIGHT + self.extra_sum)
            / (Self::EXTRA_WEIGHT + self.extra_n as f64)
    }

    fn clear_window(&mut self) {
        self.n = 0;
        self.tok = 0.0;
        self.tok_lo_cf = 0.0;
        self.saturated = 0;
        self.emitted = 0;
    }

    /// One verify step at `depth` accepted `accepted` drafts and emitted `emitted`
    /// tokens (fewer at a stop or max_tokens); `step_s` is its wall time, or None if
    /// unknown (steps that captured a CUDA graph are not timed).
    pub fn observe(&mut self, depth: usize, accepted: u64, emitted: u64, step_s: Option<f64>) {
        *self.steps.entry(depth).or_insert(0) += 1;
        if let Some(t) = step_s {
            if t > 0.0 {
                let q = self.samples.get_mut(&depth).expect("depth known");
                push_sample(q, Self::STEP_SAMPLES, t);
                self.step_s.insert(depth, Some(robust_time(q, Self::MIN_SAMPLES)));
            }
        }
        if depth != self.depth {
            return; // a step queued before the last switch; not evidence about the current depth
        }
        self.n += 1;
        self.tok += (accepted + 1) as f64;
        self.emitted += emitted;
        if depth == self.hi {
            self.tok_lo_cf += (accepted.min(self.lo as u64) + 1) as f64;
            if accepted >= self.lo as u64 {
                self.extra_sum += (accepted - self.lo as u64) as f64;
                self.extra_n += 1;
            }
        } else if accepted >= self.lo as u64 {
            self.saturated += 1;
        }
    }

    fn times(&self) -> (f64, f64) {
        match (self.step_s[&self.lo], self.step_s[&self.hi]) {
            (None, None) => (1.0, self.prior_ratio),
            (None, Some(hi)) => (hi / self.prior_ratio, hi),
            (Some(lo), None) => (lo, lo * self.prior_ratio),
            (Some(lo), Some(hi)) => (lo, hi),
        }
    }

    /// Depth for the next verify step (rank 0 decides; the result travels in the
    /// per-step control message).
    pub fn decide(&mut self) -> usize {
        if let Some(p) = self.pinned {
            self.depth = p;
            return self.depth;
        }
        if self.emitted < self.interval || self.n == 0 {
            return self.depth;
        }
        let (t_lo, t_hi) = self.times();
        let n = self.n as f64;
        let (switch, target, rate_lo, rate_hi) = if self.depth == self.hi {
            // At the deep depth the shallow alternative is known exactly: the shallow
            // verifier would have checked the first `lo` drafts of the same block.
            let rate_hi = self.tok / (n * t_hi);
            let rate_lo = self.tok_lo_cf / (n * t_lo);
            (rate_lo > rate_hi * (1.0 + self.margin), self.lo, rate_lo, rate_hi)
        } else {
            // At the shallow depth nothing past `lo` drafts is visible: estimate the
            // deep gain from saturated steps times the learned `extra`.
            let rate_lo = self.tok / (n * t_lo);
            let rate_hi = (self.tok + self.saturated as f64 * self.extra()) / (n * t_hi);
            (rate_hi > rate_lo * (1.0 + self.margin), self.hi, rate_lo, rate_hi)
        };
        if switch {
            self.last_switch = Some(SwitchEvidence {
                from: self.depth,
                to: target,
                steps: self.n,
                tokens_per_step: self.tok / n,
                rate_lo,
                rate_hi,
                estimated: if self.depth == self.hi { self.lo } else { self.hi },
            });
            self.depth = target;
            self.switches += 1;
        }
        self.clear_window();
        self.depth
    }

    /// The last switch's evidence, once; None if the depth did not change since.
    pub fn pop_switch(&mut self) -> Option<SwitchEvidence> {
        self.last_switch.take()
    }
}

/// Per-draft expected tokens / measured step cost (DSV41_BLOCK_CONFIDENCE).
///
/// DSpark's logits estimate conditional acceptance, so survival to position k is the
/// product of the first k probabilities. Include the guaranteed verifier token. Never
/// consult the current step's actual acceptance to choose its width. Greedy uses
/// whole-block lookahead; sampling extends a guaranteed prefix at width boundaries,
/// without looking at proposals it might exclude.
///
/// The 88/106/124 ms priors are historical TP2 measurements. Unseen widths inherit the
/// median measured/prior scale; observed widths use robust local timings. Captures are
/// excluded by the caller. This does not save any drafter passes: DSpark computes all
/// five backbone positions together.
#[derive(Debug, Clone)]
pub struct ConfidenceDepthPolicy {
    pub depths: Vec<usize>,
    priors: BTreeMap<usize, f64>,
    samples: BTreeMap<usize, VecDeque<f64>>,
    pub step_s: BTreeMap<usize, Option<f64>>,
    /// Opt-in periodic remeasure of an unselected width (DSV41_CONF_COST_REFRESH); 0 = off.
    pub refresh_interval: u64,
    cost_age: BTreeMap<usize, u64>,
    refresh_depth: Option<usize>,
    refresh_samples: u32,
    pub cost_refreshes: u32,
    /// Qualification only.
    pub pinned: Option<usize>,
    pub depth: usize,
    pub steps: BTreeMap<usize, u64>,
    pub switches: u32,
    pub invalid_confidence: u32,
    pub selection: Option<&'static str>,
}

impl ConfidenceDepthPolicy {
    /// `odd` is DSV41_VERIFY_ODD: opt-in target widths 1..5 with the trained DSpark
    /// block unchanged. The interpolated 2/4 priors are not claimed measurements.
    pub fn new(odd: bool, refresh_interval: u64) -> Self {
        let (depths, priors): (Vec<usize>, Vec<(usize, f64)>) = if odd {
            ((1..=5).collect(), vec![(1, 0.088), (2, 0.097), (3, 0.106), (4, 0.115), (5, 0.124)])
        } else {
            (vec![1, 3, 5], vec![(1, 0.088), (3, 0.106), (5, 0.124)])
        };
        let mut pol = ConfidenceDepthPolicy {
            samples: depths.iter().map(|&d| (d, VecDeque::new())).collect(),
            step_s: depths.iter().map(|&d| (d, None)).collect(),
            cost_age: depths.iter().map(|&d| (d, 0)).collect(),
            steps: depths.iter().map(|&d| (d, 0)).collect(),
            priors: priors.into_iter().collect(),
            depths,
            refresh_interval,
            refresh_depth: None,
            refresh_samples: 0,
            cost_refreshes: 0,
            pinned: None,
            depth: 3,
            switches: 0,
            invalid_confidence: 0,
            selection: None,
        };
        pol.reset_request();
        pol
    }

    pub fn reset_request(&mut self) {
        self.depth = 3;
        for v in self.steps.values_mut() {
            *v = 0;
        }
        self.switches = 0;
        self.invalid_confidence = 0;
        self.selection = None;
    }

    fn times(&self) -> BTreeMap<usize, f64> {
        let ratios: Vec<f64> = self
            .step_s
            .iter()
            .filter_map(|(d, t)| t.map(|t| t / self.priors[d]))
            .collect();
        let scale = if ratios.is_empty() {
            1.0
        } else {
            let mut v = ratios;
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
        };
        self.depths
            .iter()
            .map(|&d| (d, self.step_s[&d].unwrap_or(self.priors[&d] * scale)))
            .collect()
    }

    /// Provisional width in the loop-control message. Final choice follows the draft.
    pub fn decide(&self) -> usize {
        self.depth
    }

    /// Greedy decode: whole-block lookahead over the five confidences.
    pub fn choose(&mut self, logits: &[f64]) -> Result<usize, DepthError> {
        self.selection = Some("lookahead");
        let target = if let Some(p) = self.pinned {
            if !self.depths.contains(&p) {
                return Err(DepthError(format!("confidence depth must be one of {:?}", self.depths)));
            }
            p
        } else if let Some(refresh) = self.refresh_target() {
            refresh
        } else if logits.len() != 5 || logits.iter().any(|x| !x.is_finite()) {
            self.invalid_confidence += 1;
            3
        } else {
            let times = self.times();
            let mut survival = 1.0f64;
            let mut expected = 1.0f64;
            let mut rates: BTreeMap<usize, f64> = BTreeMap::new();
            for (k, &x) in logits.iter().enumerate() {
                survival *= sigmoid(x);
                expected += survival;
                if let Some(t) = times.get(&(k + 1)) {
                    rates.insert(k + 1, expected / t);
                }
            }
            // first maximum: ties prefer shallower
            let mut target = self.depths[0];
            for &d in &self.depths {
                if rates[&d] > rates[&target] {
                    target = d;
                }
            }
            target
        };
        Ok(self.select(target))
    }

    /// Sampling: extend the configured depths using an already selected proposal prefix.
    ///
    /// Confidence i depends on proposals BEFORE i (the Markov embedding), never
    /// proposal i. At depth d, confidence[d] is therefore safe to use when deciding
    /// whether to include the next proposals. This makes the block length a stopping
    /// time of the proposal prefix: inclusion of proposal i is decided without
    /// consulting proposal i or later proposals, so the existing min(1, p/q) acceptance
    /// and residual correction remain valid. A whole-block argmax, or rounding a
    /// token-dependent cutoff DOWN to a graph width, would violate that condition.
    /// Accuracy of the confidence estimate affects cost, not the target distribution.
    pub fn choose_sampled(&mut self, logits: &[f64]) -> Result<usize, DepthError> {
        self.selection = Some("prefix");
        if let Some(p) = self.pinned {
            if !self.depths.contains(&p) {
                return Err(DepthError(format!("confidence depth must be one of {:?}", self.depths)));
            }
            return Ok(self.select(p));
        }
        if let Some(refresh) = self.refresh_target() {
            return Ok(self.select(refresh));
        }
        if logits.len() != 5 {
            self.invalid_confidence += 1;
            return Ok(self.select(3));
        }
        let times = self.times();
        let mut survival = 1.0f64;
        let mut expected = 1.0f64;
        let mut seen = 0usize;
        for w in self.depths.windows(2) {
            let (depth, deeper) = (w[0], w[1]);
            // All these positions are already included. Never validate future logits
            // as a group: even an invalid-value fallback can leak lookahead.
            for &x in &logits[seen..depth] {
                if !x.is_finite() {
                    self.invalid_confidence += 1;
                    return Ok(self.select(depth));
                }
                survival *= sigmoid(x);
                expected += survival;
            }
            seen = depth;
            let x = logits[depth];
            if !x.is_finite() {
                self.invalid_confidence += 1;
                return Ok(self.select(depth));
            }
            let probability = sigmoid(x);
            let extra: f64 =
                survival * (1..=deeper - depth).map(|k| probability.powi(k as i32)).sum::<f64>();
            if (expected + extra) / times[&deeper] <= expected / times[&depth] {
                return Ok(self.select(depth));
            }
        }
        Ok(self.select(*self.depths.last().unwrap()))
    }

    fn select(&mut self, target: usize) -> usize {
        if target != self.depth {
            self.switches += 1;
        }
        self.depth = target;
        target
    }

    /// Occasionally remeasure an unselected width; never inspect draft tokens.
    ///
    /// Three non-capture steps replace a stale estimate. Without this, an old expensive
    /// depth 3 can block the sampled 1->3->5 policy indefinitely. Opt-in: extra probes
    /// can cost throughput, so qualify before enabling.
    fn refresh_target(&mut self) -> Option<usize> {
        if self.refresh_interval == 0 {
            return None;
        }
        if self.refresh_depth.is_none() {
            // first maximum age: ties prefer the shallowest depth
            let mut depth = self.depths[0];
            for &d in &self.depths {
                if self.cost_age[&d] > self.cost_age[&depth] {
                    depth = d;
                }
            }
            if self.cost_age[&depth] < self.refresh_interval {
                return None;
            }
            self.refresh_depth = Some(depth);
            self.refresh_samples = 0;
            self.samples.get_mut(&depth).unwrap().clear();
            self.step_s.insert(depth, None);
            self.cost_refreshes += 1;
        }
        self.refresh_depth
    }

    pub fn observe(&mut self, depth: usize, _accepted: u64, _emitted: u64, step_s: Option<f64>) {
        *self.steps.entry(depth).or_insert(0) += 1;
        for age in self.cost_age.values_mut() {
            *age += 1;
        }
        if let Some(t) = step_s {
            if t.is_finite() && t > 0.0 {
                self.cost_age.insert(depth, 0);
                if self.refresh_depth == Some(depth) {
                    self.refresh_samples += 1;
                    if self.refresh_samples >= 3 {
                        self.refresh_depth = None;
                    }
                }
                let q = self.samples.get_mut(&depth).expect("depth known");
                push_sample(q, 32, t);
                self.step_s.insert(depth, Some(robust_time(q, 3)));
            }
        }
    }

    /// Per-step switches belong in aggregate stats, not the server log.
    pub fn pop_switch(&mut self) -> Option<SwitchEvidence> {
        None
    }

    /// The blended step-time estimates, for stats emission.
    pub fn estimated_step_s(&self) -> BTreeMap<usize, f64> {
        self.times()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T3: f64 = 0.107;
    const T5: f64 = 0.124;

    /// Drive the policy like the decode loop: decide, then observe one step at that
    /// depth. Returns the depth sequence. From engine/test_spec_depth.py::run.
    fn run(pol: &mut DepthPolicy, mut accept_at: impl FnMut(usize) -> u64, steps: usize) -> Vec<usize> {
        let mut seq = Vec::with_capacity(steps);
        for _ in 0..steps {
            let d = pol.decide();
            let a = accept_at(d);
            pol.observe(d, a, a + 1, Some(if d == 3 { T3 } else { T5 }));
            seq.push(d);
        }
        seq
    }

    fn policy(start: usize, interval: u64) -> DepthPolicy {
        DepthPolicy::new((3, 5), DepthOptions { start: Some(start), interval, ..Default::default() })
            .unwrap()
    }

    #[test]
    fn code_climbs_to_deep() {
        let mut pol = policy(3, 60);
        let seq = run(&mut pol, |d| if d == 3 { 3 } else { 5 }, 60);
        assert_eq!(seq[0], 3);
        assert_eq!(*seq.last().unwrap(), 5);
        assert_eq!(pol.switches, 1); // and stays there
        let sw = pol.pop_switch().unwrap();
        assert_eq!((sw.from, sw.to, sw.estimated), (3, 5, 5));
        assert!(sw.rate_hi > sw.rate_lo);
        assert!(pol.pop_switch().is_none()); // logged once
    }

    #[test]
    fn prose_drops_to_shallow() {
        let mut pol = policy(5, 60);
        let seq = run(&mut pol, |_| 1, 80);
        assert_eq!(seq[0], 5);
        assert_eq!(*seq.last().unwrap(), 3);
        assert_eq!(pol.switches, 1);
    }

    #[test]
    fn prose_stays_shallow() {
        let mut pol = policy(3, 60);
        let seq = run(&mut pol, |_| 1, 100);
        assert!(seq.iter().all(|&d| d == 3));
        assert_eq!(pol.switches, 0);
    }

    #[test]
    fn interval_limits_switching() {
        let mut pol = policy(3, 60);
        let mut flip = 0u64;
        let seq = run(
            &mut pol,
            |d| {
                let odd = flip % 2 == 1;
                flip += 1;
                if odd { d as u64 } else { 0 }
            },
            200,
        );
        let changes = seq.windows(2).filter(|w| w[0] != w[1]).count();
        let budget: usize = seq.iter().map(|&d| d + 1).sum::<usize>() / 60 + 1;
        assert!(changes <= budget);
        assert!(seq[..15].iter().all(|&d| d == 3)); // < 60 tokens emitted -> start depth
    }

    #[test]
    fn counterfactual_is_exact_at_deep() {
        // at depth 5 accepting exactly 3 gives 4 tokens at either depth; 3 is cheaper
        // -> down, and having seen that deep steps never get past 3, it must not climb back
        let mut pol = policy(5, 60);
        let seq = run(&mut pol, |_| 3, 100);
        assert_eq!(*seq.last().unwrap(), 3);
        assert_eq!(pol.switches, 1);
    }

    #[test]
    fn measured_times_override_prior() {
        // if the deep step is measured to cost nothing extra, a saturated shallow window
        // must climb even for a small expected gain (the prior 1.16 ratio alone would not)
        let mut pol = DepthPolicy::new(
            (3, 5),
            DepthOptions { start: Some(3), interval: 10, extra_prior: 0.5, ..Default::default() },
        )
        .unwrap();
        pol.step_s.insert(3, Some(0.1));
        pol.step_s.insert(5, Some(0.1));
        for _ in 0..30 {
            let d = pol.decide();
            pol.observe(d, d as u64, d as u64 + 1, Some(0.1));
        }
        assert_eq!(pol.depth, 5);
        // and if the deep step is 3x the cost, it must not
        let mut pol = DepthPolicy::new(
            (3, 5),
            DepthOptions { start: Some(3), interval: 10, extra_prior: 1.5, ..Default::default() },
        )
        .unwrap();
        pol.step_s.insert(3, Some(0.1));
        pol.step_s.insert(5, Some(0.3));
        for _ in 0..30 {
            let d = pol.decide();
            pol.observe(d, d as u64, d as u64 + 1, Some(if d == 3 { 0.1 } else { 0.3 }));
        }
        assert_eq!(pol.depth, 3);
    }

    #[test]
    fn outlier_step_time_ignored() {
        let mut pol = policy(3, 10);
        pol.observe(3, 1, 2, Some(5.0)); // a slow FIRST step (e.g. an untagged capture)
        assert_eq!(pol.step_s[&3], Some(5.0));
        pol.observe(3, 1, 2, Some(0.1));
        assert_eq!(pol.step_s[&3], Some(0.1)); // min while under MIN_SAMPLES
        for _ in 0..5 {
            pol.observe(3, 1, 2, Some(0.1));
        }
        pol.observe(3, 1, 2, Some(4.0));
        assert_eq!(pol.step_s[&3], Some(0.1)); // median once there are enough
    }

    #[test]
    fn slow_start_does_not_trap_deep() {
        // The 2026-09-23 serving failure: prose, the first shallow steps slow (graph
        // captures, 2 s), then 0.105 s; deep steps 0.124 s. The policy may probe depth 5,
        // but must come back to 3 within a couple of windows and stay there.
        let mut pol = policy(3, 60);
        let mut seq = Vec::new();
        for i in 0..400usize {
            let d = pol.decide();
            let a = if d == 3 {
                [0u64, 3, 1, 3, 0, 3, 2, 3][i % 8]
            } else {
                [0u64, 3, 1, 4, 0, 3, 2, 3][i % 8]
            };
            let t = if d == 3 {
                if i < 3 { 2.0 } else { 0.105 }
            } else {
                0.124
            };
            pol.observe(d, a, a + 1, Some(t));
            seq.push(d);
        }
        assert_eq!(*seq.last().unwrap(), 3);
        assert!(seq[200..].iter().filter(|&&d| d == 5).count() <= 60);
    }

    #[test]
    fn pinned_and_reset() {
        let mut pol = policy(3, 10);
        pol.pinned = Some(5);
        pol.reset_request();
        assert_eq!(run(&mut pol, |_| 0, 30), vec![5; 30]);
        pol.pinned = None;
        pol.reset_request();
        assert_eq!(pol.depth, 3);
        assert_eq!(pol.switches, 0);
        assert!(pol.step_s[&5].is_some()); // step times survive the request
    }

    #[test]
    fn bad_config() {
        assert!(DepthPolicy::new(
            (3, 5),
            DepthOptions { start: Some(4), ..Default::default() }
        )
        .is_err());
    }

    // ---------------------------------------------------- ConfidenceDepthPolicy

    #[test]
    fn selects_all_three_depths() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        assert_eq!(p.choose(&[-1000.0; 5]).unwrap(), 1);
        assert_eq!(p.choose(&[1000.0, 1000.0, 1000.0, -1000.0, -1000.0]).unwrap(), 3);
        assert_eq!(p.choose(&[1000.0; 5]).unwrap(), 5);
    }

    #[test]
    fn cost_not_acceptance_alone() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        // All proposals are predicted certain, but the deeper verifier is very slow.
        p.step_s = [(1, Some(0.1)), (3, Some(1.0)), (5, Some(2.0))].into();
        assert_eq!(p.choose(&[1000.0; 5]).unwrap(), 1);
    }

    #[test]
    fn sampled_selects_all_three_depths() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        assert_eq!(p.choose_sampled(&[-1000.0; 5]).unwrap(), 1);
        assert_eq!(p.choose_sampled(&[1000.0, 1000.0, 1000.0, -1000.0, -1000.0]).unwrap(), 3);
        assert_eq!(p.choose_sampled(&[1000.0; 5]).unwrap(), 5);
        assert_eq!(p.selection, Some("prefix"));
        p.step_s = [(1, Some(0.1)), (3, Some(1.0)), (5, Some(2.0))].into();
        assert_eq!(p.choose_sampled(&[1000.0; 5]).unwrap(), 1);
    }

    #[test]
    fn sampled_never_reads_an_excluded_suffix() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        for a in [-1000.0, 1000.0, f64::NAN] {
            for b in [-1000.0, 1000.0, f64::NAN] {
                for c in [-1000.0, 1000.0, f64::NAN] {
                    assert_eq!(p.choose_sampled(&[1000.0, -1000.0, a, b, c]).unwrap(), 1);
                }
            }
        }
        assert_eq!(p.choose_sampled(&[1000.0, 1000.0, 1000.0, -1000.0, f64::NAN]).unwrap(), 3);
        assert_eq!(p.choose_sampled(&[1000.0, 1000.0, 1000.0, 1000.0, f64::NAN]).unwrap(), 5);
        assert_eq!(p.invalid_confidence, 0);
    }

    #[test]
    fn sampled_invalid_values_stop_at_guaranteed_width() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        assert_eq!(p.choose_sampled(&[1000.0, f64::NAN, 1000.0, 1000.0, 1000.0]).unwrap(), 1);
        assert_eq!(p.choose_sampled(&[1000.0, 1000.0, f64::NAN, 1000.0, 1000.0]).unwrap(), 3);
        assert_eq!(p.invalid_confidence, 2);
        p.pinned = Some(3);
        assert_eq!(p.choose_sampled(&[f64::NAN; 5]).unwrap(), 3);
        p.pinned = Some(2);
        assert!(p.choose_sampled(&[0.0; 5]).is_err());
    }

    #[test]
    fn expected_survival_and_no_outcome_oracle() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        let probs = [0.9, 0.5, 0.8, 0.3, 0.7];
        let logits: Vec<f64> = probs.iter().map(|&x: &f64| (x / (1.0 - x)).ln()).collect();
        let mut expected = 1.0;
        let mut survival = 1.0;
        let mut rates: BTreeMap<usize, f64> = BTreeMap::new();
        for (k, &prob) in probs.iter().enumerate() {
            survival *= prob;
            expected += survival;
            if p.depths.contains(&(k + 1)) {
                rates.insert(k + 1, expected / p.priors[&(k + 1)]);
            }
        }
        let target = *rates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(p.choose(&logits).unwrap(), target);
        for accepted in [0u64, 5] {
            p.observe(5, accepted, accepted + 1, None);
            assert_eq!(p.choose(&logits).unwrap(), target);
        }
    }

    #[test]
    fn invalid_logits_fall_back() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        assert_eq!(p.choose(&[0.0; 4]).unwrap(), 3);
        assert_eq!(p.choose(&[0.0; 6]).unwrap(), 3);
        assert_eq!(p.choose(&[f64::NAN; 5]).unwrap(), 3);
        assert_eq!(p.choose(&[f64::INFINITY; 5]).unwrap(), 3);
        assert_eq!(p.invalid_confidence, 4);
    }

    #[test]
    fn capture_and_invalid_timings_are_excluded() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        for t in [None, Some(0.0), Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
            p.observe(3, 2, 3, t);
        }
        assert_eq!(p.step_s[&3], None);
        p.observe(3, 2, 3, Some(0.212));
        let times = p.estimated_step_s();
        assert!((times[&1] - 0.176).abs() < 1e-12);
        assert!((times[&5] - 0.248).abs() < 1e-12);
        for t in [0.106, 0.106, 0.106, 5.0] {
            p.observe(3, 2, 3, Some(t));
        }
        assert!((p.step_s[&3].unwrap() - 0.106).abs() < 1e-12);
    }

    #[test]
    fn reset_keeps_costs_but_clears_request_stats() {
        let mut p = ConfidenceDepthPolicy::new(false, 0);
        p.choose(&[1000.0; 5]).unwrap();
        p.observe(5, 5, 6, Some(0.124));
        p.reset_request();
        assert_eq!(p.steps, [(1, 0), (3, 0), (5, 0)].into());
        assert_eq!(p.switches, 0);
        assert_eq!(p.step_s[&5], Some(0.124));
        p.pinned = Some(1);
        assert_eq!(p.choose(&[1000.0; 5]).unwrap(), 1);
    }

    #[test]
    fn odd_widths_cover_one_through_five() {
        let p = ConfidenceDepthPolicy::new(true, 0);
        assert_eq!(p.depths, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn cost_refresh_probes_the_stalest_width() {
        // engine/test_conf_cost_refresh.py, condensed: after `interval` steps without a
        // timing for a width, the policy pins it for three timed steps, then releases.
        let mut p = ConfidenceDepthPolicy::new(false, 10);
        for _ in 0..10 {
            let d = p.choose(&[-1000.0; 5]).unwrap();
            assert_eq!(d, 1);
            p.observe(d, 0, 1, Some(0.088));
        }
        // widths 3 and 5 are now 10 steps stale -> refresh targets 3 (shallowest tie)
        // and holds it for three timed steps
        for _ in 0..3 {
            assert_eq!(p.choose(&[-1000.0; 5]).unwrap(), 3);
            p.observe(3, 0, 1, Some(0.106));
        }
        assert_eq!(p.cost_refreshes, 1);
        // width 5 is 13 steps stale by now, so it gets the next probe immediately
        for _ in 0..3 {
            assert_eq!(p.choose(&[-1000.0; 5]).unwrap(), 5);
            p.observe(5, 0, 1, Some(0.124));
        }
        assert_eq!(p.cost_refreshes, 2);
        // every width freshly measured: the lookahead rules again
        assert_eq!(p.choose(&[-1000.0; 5]).unwrap(), 1);
    }
}
