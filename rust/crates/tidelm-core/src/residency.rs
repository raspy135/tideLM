//! Expert residency planning, from `engine/expert_budget.py`, `engine/global_residency.py`
//! and the pure functions of `engine/v41_engine.py` (`build_keep_masks`,
//! `pruned_owned_counts`, `require_pruned_residency`, `blend_demand`).
//!
//! All of this runs at boot or between requests, never per decode step — but it decides
//! routing, so under EP2 both ranks must compute it identically. Keep every function here
//! deterministic: ties are broken by stable IDs, never by hash order.

use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub struct ResidencyError(pub String);

impl std::fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ResidencyError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ResidencyError> {
    Err(ResidencyError(msg.into()))
}

/// `DSV41_RESIDENT_EXPERTS=auto` leaves this many LRU slots unassigned. Dynamic swaps
/// reuse the evicted expert's slot, so nothing in that mode needs them; a margin costs
/// ~0.1% of the arena.
pub const AUTO_MARGIN: usize = 16;

// ---------------------------------------------------------------- expert_budget.py

/// Explicit per-layer budgets for controlled retention experiments, off by default.
/// Empty input -> None. From `expert_budget.parse_layer_counts`.
pub fn parse_layer_counts(
    raw: &str,
    layers: usize,
    experts: u32,
    topk: u32,
    fraction: Option<f64>,
    selection: &str,
) -> Result<Option<Vec<u32>>, ResidencyError> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let frac = match fraction {
        Some(f) if 0.0 < f && f < 1.0 => f,
        _ => return err("DSV41_PRUNE_LAYER_COUNTS requires pruned uniform selection"),
    };
    if selection != "uniform" {
        return err("DSV41_PRUNE_LAYER_COUNTS requires pruned uniform selection");
    }
    let mut counts = Vec::with_capacity(layers);
    for part in raw.split(',') {
        match part.trim().parse::<u32>() {
            Ok(v) => counts.push(v),
            Err(_) => return err("DSV41_PRUNE_LAYER_COUNTS must contain integer counts"),
        }
    }
    if counts.len() != layers || counts.iter().any(|&v| v < topk || v > experts) {
        return err(format!(
            "layer counts need {layers} entries, each between {topk} and {experts}"
        ));
    }
    let expected = (topk as usize).max((frac * experts as f64).ceil() as usize) * layers;
    let total: usize = counts.iter().map(|&v| v as usize).sum();
    if total != expected {
        return err(format!(
            "layer counts must preserve the resident budget: {total} != {expected}"
        ));
    }
    Ok(Some(counts))
}

/// The global resident budget under dynamic allocation. From `expert_budget.resident_budget`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentBudget {
    /// Resolved later from the arena's LRU slot count (`auto_budget`).
    Auto,
    /// An exact pinned count (experiments).
    Count(usize),
    /// Unset on a static profile: `PRUNE_KEEP` decides.
    Unset,
}

pub fn resident_budget(
    raw: &str,
    dynamic: bool,
    layers: usize,
    experts: usize,
    topk: usize,
    fraction: Option<f64>,
) -> Result<ResidentBudget, ResidencyError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(if dynamic { ResidentBudget::Auto } else { ResidentBudget::Unset });
    }
    let fraction_ok = matches!(fraction, Some(f) if 0.0 < f && f < 1.0);
    if !dynamic || !fraction_ok {
        return err("DSV41_RESIDENT_EXPERTS requires dynamic pruned allocation");
    }
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(ResidentBudget::Auto);
    }
    let budget: usize = match raw.parse() {
        Ok(v) => v,
        Err(_) => return err("DSV41_RESIDENT_EXPERTS must be an integer"),
    };
    if !(layers * topk <= budget && budget <= layers * experts) {
        return err("DSV41_RESIDENT_EXPERTS outside routing floor/model capacity");
    }
    Ok(ResidentBudget::Count(budget))
}

/// `DSV41_RESIDENT_EXPERTS=auto`: every LRU slot the arena has, less a small margin,
/// capped at the whole model. From `expert_budget.auto_budget`.
pub fn auto_budget(
    lru_slots: usize,
    layers: usize,
    experts: usize,
    topk: usize,
    margin: usize,
) -> Result<usize, ResidencyError> {
    let budget = lru_slots.saturating_sub(margin).min(layers * experts);
    if budget < layers * topk {
        return err(format!(
            "DSV41_RESIDENT_EXPERTS=auto: {lru_slots} LRU slots are below the routing floor \
             ({} + {margin} spare); raise ARENA_GB",
            layers * topk
        ));
    }
    Ok(budget)
}

// ---------------------------------------------------------------- v41_engine.py

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Select {
    /// The top `ceil(frac * experts)` experts of every layer (the v0.2.0-wip scheme).
    Uniform,
    /// One ranking of all (layer, expert) pairs by per-layer-normalized routing
    /// frequency, cut at the same total count, with at least `min_per_layer` per layer.
    Global,
}

/// Which experts stay routable under a budget of `ceil(frac * experts)` per layer ON
/// AVERAGE. From `v41_engine.build_keep_masks`. Returns per-layer keep masks plus the
/// kept expert ids, hottest first. Ties rank the lower expert id first (NumPy's argsort
/// order on ties is unspecified; a fixed rule keeps the two EP ranks identical).
pub fn build_keep_masks(
    counts: &BTreeMap<u32, Vec<f64>>,
    frac: f64,
    select: Select,
    n_experts: usize,
    topk: usize,
    min_per_layer: usize,
    layer_counts: Option<&[u32]>,
) -> Result<(BTreeMap<u32, Vec<bool>>, BTreeMap<u32, Vec<usize>>), ResidencyError> {
    let n_keep = topk.max((frac * n_experts as f64).ceil() as usize);
    let total = n_keep * counts.len();
    if let Some(lc) = layer_counts {
        if lc.len() != counts.len() {
            return err("explicit layer counts need one entry per layer");
        }
        let contiguous = counts.keys().enumerate().all(|(i, &l)| l as usize == i);
        if !contiguous {
            return err("explicit layer counts require contiguous layer IDs");
        }
    }
    let mut keep: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    match select {
        Select::Uniform => {
            for (&layer, c) in counts {
                if c.len() != n_experts {
                    return err("routing counts must cover every expert");
                }
                let quota = match layer_counts {
                    None => n_keep,
                    Some(lc) => lc[layer as usize] as usize,
                };
                keep.insert(layer, top_indices(c, quota));
            }
        }
        Select::Global => {
            let mut keys: Vec<(f64, u32, usize)> = Vec::new();
            for (&layer, c) in counts {
                if c.len() != n_experts {
                    return err("routing counts must cover every expert");
                }
                let sum: f64 = c.iter().sum();
                let norm: Vec<f64> = c.iter().map(|&x| x / if sum != 0.0 { sum } else { 1.0 }).collect();
                let order = top_indices(&norm, n_experts);
                keep.insert(layer, order[..min_per_layer.min(order.len())].to_vec());
                keys.extend(order[min_per_layer.min(order.len())..].iter().map(|&e| (norm[e], layer, e)));
            }
            // Python sorts the (score, layer, expert) tuples descending; ties on score
            // therefore prefer the HIGHER layer and expert id. Kept as-is.
            keys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let kept: usize = keep.values().map(Vec::len).sum();
            for &(_, layer, e) in keys.iter().take(total.saturating_sub(kept)) {
                keep.get_mut(&layer).unwrap().push(e);
            }
        }
    }
    let mut masks = BTreeMap::new();
    for (&layer, ids) in &keep {
        let mut m = vec![false; n_experts];
        for &e in ids {
            m[e] = true;
        }
        masks.insert(layer, m);
    }
    Ok((masks, keep))
}

/// Indices of the `quota` largest values, descending by value, ascending id on ties.
fn top_indices(values: &[f64], quota: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(quota);
    idx
}

/// Number of routable expert pairs assigned to each EP rank.
/// From `v41_engine.pruned_owned_counts`.
pub fn pruned_owned_counts(ranked: &[(u32, u32)], world: usize) -> Vec<usize> {
    let world = world.max(1);
    (0..world)
        .map(|rank| ranked.iter().filter(|(_, e)| *e as usize % world == rank).count())
        .collect()
}

/// Reject a pruned configuration unless every routable expert is resident.
///
/// The pruned path is explicitly the all-resident serving mode. Letting a small tail
/// stream is not merely a performance degradation: a broad prefill can route to more
/// missing experts in one layer than the transient ring can hold. One EP rank then
/// raises while its peer is already inside an NCCL combine, making the surviving GPU
/// look as if it is looping indefinitely. From `v41_engine.require_pruned_residency`.
pub fn require_pruned_residency(
    ranked: &[(u32, u32)],
    world: usize,
    lru_slots: usize,
    prune_keep: f64,
    transient_slots: usize,
) -> Result<Vec<usize>, ResidencyError> {
    let owned = pruned_owned_counts(ranked, world);
    let overflow: Vec<String> = owned
        .iter()
        .enumerate()
        .filter(|(_, &count)| count > lru_slots)
        .map(|(rank, &count)| format!("rank {rank} needs {count}"))
        .collect();
    if !overflow.is_empty() {
        return err(format!(
            "pruned all-resident capacity exceeded at keep={prune_keep:.3}: {}, but each rank has \
             {lru_slots} LRU slots ({transient_slots} transient slots reserved). Reduce \
             --prune-keep or increase --arena-gb; allowing the tail to stream can exhaust the \
             transient ring during prefill and strand the peer in a collective.",
            overflow.join(", ")
        ));
    }
    Ok(owned)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemandMetric {
    Frequency,
    Score,
}

/// Blend the shipped trace ranking with observed demand, per layer.
///
/// `w = observed / (observed + prior)`, computed PER LAYER — `prior_slots` is in routing
/// slots per layer, NOT totalled across layers. A cold database changes nothing and a
/// well-exercised one dominates, with no threshold to tune. Returns the blended per-layer
/// distributions and the mean observed weight. From `v41_engine.blend_demand`.
pub fn blend_demand(
    trace_counts: &BTreeMap<u32, Vec<f64>>,
    db_counts: &BTreeMap<u32, Vec<f64>>,
    db_mass: &BTreeMap<u32, Vec<f64>>,
    metric: DemandMetric,
    request_unit: bool,
    prior_slots: f64,
) -> (BTreeMap<u32, Vec<f64>>, f64) {
    let mut out = BTreeMap::new();
    let mut ws = Vec::new();
    for (&layer, tr) in trace_counts {
        let obs = match metric {
            DemandMetric::Score => &db_mass[&layer],
            DemandMetric::Frequency => &db_counts[&layer],
        };
        let n: f64 = obs.iter().sum();
        // Request score distributions sum to one, just like counts. Slot-mode score units
        // depend on router magnitude: use counts for confidence, scores only for placement.
        let evidence = if request_unit || metric == DemandMetric::Frequency {
            n
        } else {
            db_counts[&layer].iter().sum()
        };
        let w = if n > 0.0 { evidence / (evidence + prior_slots) } else { 0.0 };
        ws.push(w);
        let tr_sum: f64 = tr.iter().sum();
        let tr_sum = if tr_sum != 0.0 { tr_sum } else { 1.0 };
        let n_div = if n != 0.0 { n } else { 1.0 };
        let blended: Vec<f64> = tr
            .iter()
            .zip(obs)
            .map(|(&t, &o)| (1.0 - w) * (t / tr_sum) + w * (o / n_div))
            .collect();
        out.insert(layer, blended);
    }
    let mean_w = if ws.is_empty() { 0.0 } else { ws.iter().sum::<f64>() / ws.len() as f64 };
    (out, mean_w)
}

// ---------------------------------------------------------------- global_residency.py

/// `DSV41_PRUNE_LAYER_WEIGHTS`: one positive finite value per layer, default all ones.
pub fn layer_weights(raw: &str, layers: usize) -> Result<Vec<f64>, ResidencyError> {
    if raw.trim().is_empty() {
        return Ok(vec![1.0; layers]);
    }
    let values: Result<Vec<f64>, _> = raw.split(',').map(|v| v.trim().parse::<f64>()).collect();
    match values {
        Ok(v) if v.len() == layers && v.iter().all(|x| x.is_finite() && *x > 0.0) => Ok(v),
        _ => err("DSV41_PRUNE_LAYER_WEIGHTS needs one positive finite value per layer"),
    }
}

/// Mandatory top-k coverage, then one global ranking for every remaining slot.
/// From `global_residency.initial_selection`. Ties: lower layer, then lower expert id.
pub fn initial_selection(
    scores: &BTreeMap<u32, Vec<f64>>,
    budget: usize,
    floor: usize,
) -> Result<BTreeMap<u32, Vec<usize>>, ResidencyError> {
    let capacity: usize = scores.values().map(Vec::len).sum();
    if !(floor * scores.len() <= budget && budget <= capacity) {
        return err("global resident budget cannot satisfy routing minimum");
    }
    let mut chosen: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut extras: Vec<(f64, u32, usize)> = Vec::new();
    for (&layer, s) in scores {
        let order = top_indices(s, s.len());
        chosen.insert(layer, order[..floor].to_vec());
        extras.extend(order[floor..].iter().map(|&e| (s[e], layer, e)));
    }
    extras.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    for &(_, layer, e) in extras.iter().take(budget - floor * scores.len()) {
        chosen.get_mut(&layer).unwrap().push(e);
    }
    Ok(chosen)
}

/// One slot move: (from_layer, from_expert, to_layer, to_expert, gain).
pub type Swap = (u32, usize, u32, usize, f64);

/// Highest-value absentee vs lowest-value eligible resident, across layers.
///
/// The floor preserves the router's top-k minimum, not a per-layer quota. A same-layer
/// replacement is possible at the floor; a donor to another layer must have spare
/// residents. Every pair preserves the total arena occupancy. Scores must already share
/// units across layers. Stable IDs break ties. From `global_residency.global_plan`.
#[allow(clippy::too_many_arguments)]
pub fn global_plan(
    keeps: &BTreeMap<u32, Vec<bool>>,
    scores: &BTreeMap<u32, Vec<f64>>,
    floor: usize,
    max_loads: Option<usize>,
    protected: Option<&BTreeMap<u32, Vec<usize>>>,
    fallback: Option<&BTreeMap<u32, Vec<f64>>>,
    min_gain: f64,
) -> Result<(Vec<Swap>, BTreeMap<u32, usize>), ResidencyError> {
    if floor < 1 {
        return err("invalid resident floor");
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    let mut incoming: Vec<(f64, f64, u32, usize)> = Vec::new();
    let mut donors: Vec<(f64, f64, u32, usize)> = Vec::new();
    for (&layer, keep) in keeps {
        let score = scores.get(&layer).ok_or_else(|| ResidencyError("invalid global residency scores".into()))?;
        let tie = fallback.map(|f| &f[&layer]).unwrap_or(score);
        if score.len() != keep.len()
            || tie.len() != keep.len()
            || score.iter().any(|v| !v.is_finite() || *v < 0.0)
            || tie.iter().any(|v| !v.is_finite())
        {
            return err("invalid global residency scores");
        }
        let resident = keep.iter().filter(|&&k| k).count();
        counts.insert(layer, resident);
        if !(floor <= resident && resident <= keep.len()) {
            return err("resident count below the streaming capacity floor");
        }
        let lock: &[usize] = protected.and_then(|p| p.get(&layer)).map(|v| v.as_slice()).unwrap_or(&[]);
        for (e, &k) in keep.iter().enumerate() {
            if k && !lock.contains(&e) {
                donors.push((score[e], tie[e], layer, e));
            } else if !k && score[e] > 0.0 {
                incoming.push((score[e], tie[e], layer, e));
            }
        }
    }
    donors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    incoming.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    let mut changes: Vec<Swap> = Vec::new();
    for &(value, _, li, ei) in &incoming {
        if let Some(limit) = max_loads {
            if changes.len() >= limit {
                break;
            }
        }
        let victim = donors
            .iter()
            .position(|&(_, _, lo, _)| lo == li || counts[&lo] > floor);
        let Some(victim) = victim else { continue };
        let (old, _, lo, eo) = donors[victim];
        let gain = value - old;
        if gain <= min_gain {
            continue;
        }
        donors.remove(victim);
        changes.push((lo, eo, li, ei, gain));
        *counts.get_mut(&lo).unwrap() -= 1;
        *counts.get_mut(&li).unwrap() += 1;
    }
    Ok((changes, counts))
}

/// Contiguous token batches: all their cold experts must fit simultaneously.
/// From `global_residency.stream_spans`.
pub fn stream_spans(
    rows: &[Vec<usize>],
    keep: &[bool],
    capacity: usize,
) -> Result<Vec<(usize, usize)>, ResidencyError> {
    if rows.iter().any(|r| r.len() > capacity) {
        return err("transient ring cannot fit one routing row");
    }
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut cold: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for (i, row) in rows.iter().enumerate() {
        let missing: Vec<usize> = row.iter().copied().filter(|&e| !keep[e]).collect();
        let grown = cold.iter().copied().chain(missing.iter().copied()).collect::<std::collections::BTreeSet<_>>();
        if grown.len() > capacity {
            spans.push((start, i));
            start = i;
            cold.clear();
        }
        cold.extend(missing);
    }
    if !rows.is_empty() {
        spans.push((start, rows.len()));
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranking(layers: u32, n: usize) -> BTreeMap<u32, Vec<f64>> {
        (0..layers).map(|l| (l, (0..n).map(|e| e as f64).collect())).collect()
    }

    // engine/test_expert_budget.py::test_disabled_and_valid_edge_allocation
    #[test]
    fn disabled_and_valid_edge_allocation() {
        assert_eq!(parse_layer_counts("", 40, 384, 6, Some(0.61), "uniform").unwrap(), None);
        let mut quotas: Vec<u32> = Vec::new();
        quotas.extend([330u32; 5]);
        quotas.extend([204u32; 10]);
        quotas.extend([203u32; 20]);
        quotas.extend([330u32; 5]);
        let raw = quotas.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let got = parse_layer_counts(&raw, 40, 384, 6, Some(0.61), "uniform").unwrap().unwrap();
        assert_eq!(got.iter().map(|&v| v as usize).sum::<usize>(), 9400);
        let (masks, kept) =
            build_keep_masks(&ranking(40, 384), 0.61, Select::Uniform, 384, 6, 24, Some(&got)).unwrap();
        let total: usize = masks.values().map(|m| m.iter().filter(|&&b| b).count()).sum();
        assert_eq!(total, 9400);
        for (layer, &quota) in quotas.iter().enumerate() {
            let quota = quota as usize;
            let m = &masks[&(layer as u32)];
            assert_eq!(kept[&(layer as u32)].len(), quota);
            assert!(m[384 - quota..].iter().all(|&b| b));
            assert!(m[..384 - quota].iter().all(|&b| !b));
        }
    }

    // engine/test_expert_budget.py::test_invalid_budgets_fail_before_serving
    #[test]
    fn invalid_budgets_fail_before_serving() {
        let cases: Vec<(String, Option<f64>, &str)> = vec![
            (format!("{}234", "235,".repeat(39)), Some(0.61), "uniform"),
            (format!("{}235", "235,".repeat(39)), Some(1.0), "uniform"),
            (format!("{}235", "235,".repeat(39)), Some(0.61), "global"),
            ("235,235".into(), Some(0.61), "uniform"),
            (format!("{}x", "235,".repeat(39)), Some(0.61), "uniform"),
            (format!("{}465,5", "235,".repeat(38)), Some(0.61), "uniform"),
        ];
        for (raw, fraction, selection) in cases {
            assert!(
                parse_layer_counts(&raw, 40, 384, 6, fraction, selection).is_err(),
                "{raw} should be rejected"
            );
        }
    }

    // engine/test_expert_budget.py::ExactResidentBudgetTest
    #[test]
    fn exact_budget_is_not_rounded_per_layer() {
        let n = resident_budget("9500", true, 40, 384, 6, Some(0.61)).unwrap();
        assert_eq!(n, ResidentBudget::Count(9500));
        let scores: BTreeMap<u32, Vec<f64>> =
            (0..40).map(|l| (l, (0..384).map(|e| (e + l as usize) as f64).collect())).collect();
        let keep = initial_selection(&scores, 9500, 6).unwrap();
        assert_eq!(keep.values().map(Vec::len).sum::<usize>(), 9500);
        assert!(keep.values().map(Vec::len).min().unwrap() >= 6);
    }

    #[test]
    fn invalid_budget_and_fixed_mode_rejected() {
        assert_eq!(resident_budget("", true, 40, 384, 6, Some(0.61)).unwrap(), ResidentBudget::Auto);
        for raw in ["239", "15361", "9500.5"] {
            assert!(resident_budget(raw, true, 40, 384, 6, Some(0.61)).is_err());
        }
        assert!(resident_budget("9500", false, 40, 384, 6, Some(0.61)).is_err());
        assert_eq!(resident_budget("", false, 40, 384, 6, Some(0.61)).unwrap(), ResidentBudget::Unset);
    }

    // engine/test_expert_budget.py::AutoResidentBudgetTest
    #[test]
    fn auto_follows_the_arena() {
        assert_eq!(resident_budget("auto", true, 40, 384, 6, Some(0.61)).unwrap(), ResidentBudget::Auto);
        assert_eq!(resident_budget(" AUTO ", true, 40, 384, 6, Some(0.61)).unwrap(), ResidentBudget::Auto);
        assert_eq!(auto_budget(13514, 40, 384, 6, AUTO_MARGIN).unwrap(), 13514 - AUTO_MARGIN);
        assert_eq!(auto_budget(9586, 40, 384, 6, AUTO_MARGIN).unwrap(), 9586 - AUTO_MARGIN);
        assert_eq!(auto_budget(20000, 40, 384, 6, AUTO_MARGIN).unwrap(), 15360);
        assert!(auto_budget(40 * 6 + AUTO_MARGIN - 1, 40, 384, 6, AUTO_MARGIN).is_err());
        assert!(resident_budget("auto", false, 40, 384, 6, Some(0.61)).is_err());
    }

    #[test]
    fn owned_counts_split_by_parity() {
        let ranked: Vec<(u32, u32)> = (0..10).map(|e| (0, e)).collect();
        assert_eq!(pruned_owned_counts(&ranked, 2), vec![5, 5]);
        assert_eq!(pruned_owned_counts(&ranked, 1), vec![10]);
        assert!(require_pruned_residency(&ranked, 2, 5, 0.61, 100).is_ok());
        assert!(require_pruned_residency(&ranked, 2, 4, 0.61, 100).is_err());
    }

    #[test]
    fn blend_cold_db_changes_nothing() {
        let trace: BTreeMap<u32, Vec<f64>> = [(0, vec![3.0, 1.0])].into();
        let zeros: BTreeMap<u32, Vec<f64>> = [(0, vec![0.0, 0.0])].into();
        let (out, w) = blend_demand(&trace, &zeros, &zeros, DemandMetric::Frequency, false, 2e4);
        assert_eq!(w, 0.0);
        assert_eq!(out[&0], vec![0.75, 0.25]);
    }

    #[test]
    fn blend_observed_demand_outvotes_trace_with_evidence() {
        let trace: BTreeMap<u32, Vec<f64>> = [(0, vec![1.0, 0.0])].into();
        let counts: BTreeMap<u32, Vec<f64>> = [(0, vec![0.0, 2e4])].into();
        let (out, w) = blend_demand(&trace, &counts, &counts, DemandMetric::Frequency, false, 2e4);
        assert!((w - 0.5).abs() < 1e-12);
        assert_eq!(out[&0], vec![0.5, 0.5]);
    }

    #[test]
    fn global_plan_moves_highest_value_absentee_onto_cheapest_donor() {
        // layer 0 holds experts {0, 1}; absent expert 2 scores far above resident 1.
        let keeps: BTreeMap<u32, Vec<bool>> = [(0, vec![true, true, false])].into();
        let scores: BTreeMap<u32, Vec<f64>> = [(0, vec![5.0, 0.1, 9.0])].into();
        let (swaps, counts) = global_plan(&keeps, &scores, 1, None, None, None, 0.0).unwrap();
        assert_eq!(swaps.len(), 1);
        let (lo, eo, li, ei, gain) = swaps[0];
        assert_eq!((lo, eo, li, ei), (0, 1, 0, 2));
        assert!((gain - 8.9).abs() < 1e-9);
        assert_eq!(counts[&0], 2);
    }

    #[test]
    fn global_plan_respects_floor_and_min_gain() {
        // one resident at the floor in another layer cannot donate
        let keeps: BTreeMap<u32, Vec<bool>> = [(0, vec![true, false]), (1, vec![true, false])].into();
        let scores: BTreeMap<u32, Vec<f64>> = [(0, vec![0.1, 0.0]), (1, vec![0.2, 9.0])].into();
        let (swaps, _) = global_plan(&keeps, &scores, 1, None, None, None, 0.0).unwrap();
        // expert (1,1) can only displace (1,0): a same-layer replacement at the floor
        assert_eq!(swaps.len(), 1);
        assert_eq!(&swaps[0], &(1, 0, 1, 1, 8.8));
        // ...and a min_gain above that suppresses it
        let (swaps, _) = global_plan(&keeps, &scores, 1, None, None, None, 10.0).unwrap();
        assert!(swaps.is_empty());
    }

    #[test]
    fn stream_spans_split_when_cold_set_overflows() {
        let keep = vec![false, false, false, true];
        let rows: Vec<Vec<usize>> = vec![vec![0, 3], vec![1, 3], vec![2, 3], vec![3, 3]];
        // capacity 2: rows 0+1 fit (cold {0,1}), row 2 overflows -> new span
        assert_eq!(stream_spans(&rows, &keep, 2).unwrap(), vec![(0, 2), (2, 4)]);
        assert!(stream_spans(&[vec![0, 1, 2]], &keep, 2).is_err());
        assert_eq!(stream_spans(&[], &keep, 2).unwrap(), vec![]);
    }
}
