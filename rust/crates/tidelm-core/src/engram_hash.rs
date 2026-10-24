//! The engram n-gram hash, from the checkpoint's `inference/engram.py`
//! (`EngramLayout`, `compute_hash_multipliers`, `NgramHashState`).
//!
//! Each position hashes the `max_ngram_size - 1` n-grams ending there (2-gram ..
//! 4-gram) over *compressed* token ids, one bucket range per (n-gram size, head) with
//! a prime modulus. The result indexes the engram tables directly, so every constant
//! here must match the checkpoint's Python bit for bit; `rust/fixtures/engram_truth.json`
//! pins the primes, offsets, multipliers and the hashes of a real prompt.
//!
//! The Python runs this on the GPU every decode step (eager, outside the graphs). Here it
//! is host integer arithmetic over at most `max_ngram_size` cached ids per position —
//! nanoseconds — and the result feeds the engram row gather, which runs on the host.

use crate::numpy_rng::Pcg64;

/// Positions that take no part in an n-gram (image spans) are cached as DEAD.
pub const DEAD: i64 = -1;

/// The static layout: bucket primes, flat offsets, and per-(layer, lookback) multipliers.
#[derive(Debug, Clone)]
pub struct EngramLayout {
    pub layer_ids: Vec<usize>,
    pub max_ngram_size: usize,
    pub n_heads: usize,
    /// `[layer][ngram - 2][head]` prime bucket modulus.
    pub primes: Vec<Vec<Vec<i64>>>,
    /// `[layer][(ngram - 2) * n_heads + head]` start row of that bucket range.
    pub offsets: Vec<Vec<i64>>,
    /// `[layer][lookback]`, odd.
    pub multipliers: Vec<Vec<i64>>,
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    // deterministic Miller-Rabin for n < 3.3e24 with these bases
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    let mulmod = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let powmod = |mut a: u64, mut e: u64| {
        let mut acc = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = mulmod(acc, a);
            }
            a = mulmod(a, a);
            e >>= 1;
        }
        acc
    };
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

impl EngramLayout {
    /// `EngramLayout.from_args` + `compute_hash_multipliers`. `compressed_vocab` is the
    /// size of the compressed token map: every multiplier bound derives from it, so a
    /// mismatch there silently rehashes the whole table.
    pub fn new(
        layer_ids: &[usize],
        max_ngram_size: usize,
        n_heads: usize,
        engram_vocab_size: i64,
        compressed_vocab: i64,
    ) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut primes = Vec::new();
        for _ in layer_ids {
            let mut per_ngram = Vec::new();
            for _ in 0..max_ngram_size - 1 {
                let mut sizes = Vec::new();
                let mut current = engram_vocab_size - 1;
                for _ in 0..n_heads {
                    // find_next_prime: the smallest prime above `current` not handed out yet
                    let mut c = current + 1;
                    while !is_prime(c as u64) || seen.contains(&c) {
                        c += 1;
                    }
                    seen.insert(c);
                    current = c;
                    sizes.push(c);
                }
                per_ngram.push(sizes);
            }
            primes.push(per_ngram);
        }
        let offsets = primes
            .iter()
            .map(|layer: &Vec<Vec<i64>>| {
                let flat: Vec<i64> = layer.iter().flatten().copied().collect();
                let mut acc = 0i64;
                flat.iter()
                    .map(|&p| {
                        let o = acc;
                        acc += p;
                        o
                    })
                    .collect()
            })
            .collect();
        let bound = ((i64::MAX / compressed_vocab) / 2).max(1);
        let multipliers = layer_ids
            .iter()
            .map(|&l| {
                Pcg64::from_seed(10007 * l as u128)
                    .integers_i64(0, bound, max_ngram_size)
                    .into_iter()
                    .map(|v| v * 2 + 1)
                    .collect()
            })
            .collect();
        EngramLayout { layer_ids: layer_ids.to_vec(), max_ngram_size, n_heads, primes, offsets, multipliers }
    }

    /// Hash columns per (position, engram layer): (max_ngram_size - 1) * n_heads.
    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }
}

/// `NgramHashState`: the per-sequence cache of compressed ids plus the hash itself.
#[derive(Debug, Clone)]
pub struct NgramHashState {
    pub layout: EngramLayout,
    /// shared by every sequence's state (`fresh`)
    token_map: std::sync::Arc<Vec<i64>>,
    pad_id: i64,
    /// compressed ids by absolute position (DEAD for image positions)
    cache: Vec<i64>,
}

impl NgramHashState {
    /// `token_map` maps every vocabulary id to its compressed id; `engram_pad_id` is a
    /// vocabulary id (2 for this checkpoint) whose compressed id pads blocked lookbacks.
    pub fn new(layout: EngramLayout, token_map: Vec<i64>, engram_pad_id: usize) -> Self {
        let pad_id = token_map[engram_pad_id];
        NgramHashState { layout, token_map: std::sync::Arc::new(token_map), pad_id, cache: Vec::new() }
    }

    /// A state for another sequence: same layout and token map, empty cache.
    pub fn fresh(&self) -> Self {
        NgramHashState { layout: self.layout.clone(), token_map: self.token_map.clone(), pad_id: self.pad_id, cache: Vec::new() }
    }

    /// Hash ids for tokens at absolute positions `start_pos..start_pos + ids.len()`,
    /// written as `[t][engram layer][col]` into the returned flat vector.
    ///
    /// Like the Python, the cache is written at `start_pos` before hashing, so the
    /// caller may re-hash from any earlier position (rollback) and later tokens are
    /// simply overwritten. `token_mask[t] == false` marks a dead (image) position.
    pub fn hash(&mut self, ids: &[u32], start_pos: usize, token_mask: Option<&[bool]>) -> Vec<i64> {
        let end = start_pos + ids.len();
        if self.cache.len() < end {
            self.cache.resize(end, 0);
        }
        for (t, &id) in ids.iter().enumerate() {
            let live = token_mask.map(|m| m[t]).unwrap_or(true);
            self.cache[start_pos + t] = if live { self.token_map[id as usize] } else { DEAD };
        }
        let l = &self.layout;
        let n_layers = l.layer_ids.len();
        let cols = l.n_hash_cols();
        let mut out = Vec::with_capacity(ids.len() * n_layers * cols);
        let mut tokens = vec![0i64; l.max_ngram_size];
        for pos in start_pos..end {
            let mut blocked = false;
            for (shift, tok) in tokens.iter_mut().enumerate() {
                let source = self.cache[pos.saturating_sub(shift)];
                blocked = blocked || pos < shift || source == DEAD;
                *tok = if blocked { self.pad_id } else { source };
            }
            for layer in 0..n_layers {
                let mult = &l.multipliers[layer];
                let mut rolling = tokens[0].wrapping_mul(mult[0]);
                for i in 1..l.max_ngram_size {
                    rolling ^= tokens[i].wrapping_mul(mult[i]);
                    for head in 0..l.n_heads {
                        let p = l.primes[layer][i - 1][head];
                        let col = (i - 1) * l.n_heads + head;
                        out.push(rolling.rem_euclid(p) + l.offsets[layer][col]);
                    }
                }
            }
        }
        out
    }

    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../fixtures/engram_truth.json")).unwrap()
    }

    fn i64s(v: &serde_json::Value) -> Vec<i64> {
        v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
    }

    fn layout() -> EngramLayout {
        EngramLayout::new(&[1, 14], 4, 8, 16_000_000, 99_092)
    }

    #[test]
    fn layout_matches_the_checkpoint() {
        let t = truth();
        let l = layout();
        let want_primes: Vec<Vec<Vec<i64>>> = t["primes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|layer| layer.as_array().unwrap().iter().map(i64s).collect())
            .collect();
        assert_eq!(l.primes, want_primes);
        let want_offsets: Vec<Vec<i64>> = t["offsets"].as_array().unwrap().iter().map(i64s).collect();
        assert_eq!(l.offsets, want_offsets);
        let want_mult: Vec<Vec<i64>> = t["multipliers"].as_array().unwrap().iter().map(i64s).collect();
        assert_eq!(l.multipliers, want_mult);
    }

    #[test]
    fn primality() {
        let primes: Vec<u64> = (2..60).filter(|&n| is_prime(n)).collect();
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]);
        assert!(is_prime(16_000_057));
        assert!(!is_prime(16_000_059));
    }
}
