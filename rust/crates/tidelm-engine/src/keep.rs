//! Resident expert sets: the Python engine's boot selection (global_residency.initial_selection)
//! over a demand history at a given budget, and its split across the pair.

use std::collections::BTreeMap;

use tidelm_core::residency::initial_selection;

use crate::Result;

/// `path`: per-layer expert demand as little-endian f64 `[layers][experts]` (the `mass`
/// array of the engine's prune demand database).
pub fn load_demand(path: &std::path::Path, layers: usize, experts: usize) -> Result<Vec<Vec<f64>>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != layers * experts * 8 {
        return Err(format!("{}: {} bytes, expected [{layers}][{experts}] f64", path.display(), bytes.len()).into());
    }
    let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(v.chunks(experts).map(|r| r.to_vec()).collect())
}

/// The router's top-k floor in every layer, then the globally highest-demand experts up to
/// `budget` in total.
pub fn keep_from_scores(scores: &[Vec<f64>], floor: usize, budget: usize) -> Result<Vec<Vec<bool>>> {
    let map: BTreeMap<u32, Vec<f64>> = scores.iter().enumerate().map(|(l, s)| (l as u32, s.clone())).collect();
    let chosen = initial_selection(&map, budget, floor).map_err(|e| e.0)?;
    Ok(scores
        .iter()
        .enumerate()
        .map(|(l, s)| {
            let mut keep = vec![false; s.len()];
            for &e in &chosen[&(l as u32)] {
                keep[e] = true;
            }
            keep
        })
        .collect())
}

pub fn keep_from_demand(path: &std::path::Path, layers: usize, experts: usize, floor: usize, budget: usize)
                        -> Result<Vec<Vec<bool>>> {
    keep_from_scores(&load_demand(path, layers, experts)?, floor, budget)
}

/// The kept experts this rank holds: per layer, alternate ranks down the demand ranking
/// (ties by expert id; without scores, by id), so both ranks get a like share of the hot
/// experts. Both ranks compute it from the same inputs.
pub fn owned_split(keep: &[Vec<bool>], scores: Option<&[Vec<f64>]>, world: usize, rank: usize) -> Vec<Vec<bool>> {
    keep.iter()
        .enumerate()
        .map(|(l, k)| {
            let mut kept: Vec<usize> = (0..k.len()).filter(|&e| k[e]).collect();
            if let Some(s) = scores {
                kept.sort_by(|&a, &b| s[l][b].partial_cmp(&s[l][a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            }
            let mut owned = vec![false; k.len()];
            for (i, &e) in kept.iter().enumerate() {
                owned[e] = i % world == rank;
            }
            owned
        })
        .collect()
}
