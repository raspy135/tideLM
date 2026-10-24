//! Resident expert sets for single-box runs: the Python engine's boot selection
//! (global_residency.initial_selection) over a demand history, at a given budget.

use std::collections::BTreeMap;

use tidelm_core::residency::initial_selection;

use crate::Result;

/// `path`: per-layer expert demand as little-endian f64 `[layers][experts]` (the `mass`
/// array of the engine's prune demand database). Keeps the router's top-k floor in every
/// layer, then the globally highest-demand experts up to `budget` in total.
pub fn keep_from_demand(path: &std::path::Path, layers: usize, experts: usize, floor: usize, budget: usize)
                        -> Result<Vec<Vec<bool>>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != layers * experts * 8 {
        return Err(format!("{}: {} bytes, expected [{layers}][{experts}] f64", path.display(), bytes.len()).into());
    }
    let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let scores: BTreeMap<u32, Vec<f64>> = (0..layers).map(|l| (l as u32, v[l * experts..(l + 1) * experts].to_vec())).collect();
    let chosen = initial_selection(&scores, budget, floor).map_err(|e| e.0)?;
    Ok((0..layers)
        .map(|l| {
            let mut keep = vec![false; experts];
            for &e in &chosen[&(l as u32)] {
                keep[e] = true;
            }
            keep
        })
        .collect())
}
