//! Temperature + nucleus sampling and penalties, from `engine/v41_engine.py`
//! (`sample_probs`, `Penalties`).
//!
//! These are the host-side reference semantics. On the GPU path the same math runs
//! inside the decode graphs; this code is the arbiter when the two disagree, exactly
//! as the Python version was.

use std::collections::HashMap;

/// First index of the maximum, like `torch.argmax` on ties.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// `[V]` f32 logits -> probability vector (temperature + nucleus).
///
/// `temperature <= 0` is greedy: a one-hot at the argmax. Nucleus keeps a token while
/// the cumulative mass *before* it (descending order) is `< top_p`, then renormalizes —
/// the `cum - sp < top_p` rule from the Python, which always keeps the top token.
pub fn sample_probs(logits: &[f32], temperature: f32, top_p: f32) -> Vec<f32> {
    let v = logits.len();
    if temperature <= 0.0 {
        let mut p = vec![0.0; v];
        p[argmax(logits)] = 1.0;
        return p;
    }
    let mut p = softmax_scaled(logits, temperature);
    if top_p < 1.0 {
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept = vec![0.0f32; v];
        let mut cum = 0.0f32;
        for &i in &idx {
            if cum < top_p {
                kept[i] = p[i];
            }
            cum += p[i];
        }
        let total: f32 = kept.iter().sum();
        if total > 0.0 {
            for x in kept.iter_mut() {
                *x /= total;
            }
        }
        p = kept;
    }
    p
}

/// Numerically-stable softmax of `logits / temperature`.
pub fn softmax_scaled(logits: &[f32], temperature: f32) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&x| ((x - m) / temperature).exp()).collect();
    let s: f32 = out.iter().sum();
    for x in out.iter_mut() {
        *x /= s;
    }
    out
}

/// OpenAI presence/frequency penalties, plus a breaker for exact repetition cycles.
///
/// The penalties are the standard ones (a token already generated loses
/// `presence + frequency * count` from its logit) and default to 0, because a flat
/// penalty hurts code, which is legitimately repetitive.
///
/// The cycle breaker is separate and opt-in; it fires only on an *exact* cycle of
/// period <= `cycle_max_period` repeated `cycle_repeats` times, and bans just the one
/// token that would continue the cycle for that single step. It also bans legitimate
/// repetition (the canonical depth-8 JSON nesting answer trips it at position 13), so
/// it must not silently run on ordinary requests or quality comparisons.
#[derive(Debug, Clone)]
pub struct Penalties {
    pub presence: f32,
    pub frequency: f32,
    pub cycle_repeats: usize,
    pub cycle_max_period: usize,
    pub cycle_enabled: bool,
    pub no_repeat_ngram: usize,
    counts: HashMap<i64, u32>,
    pub hits: u64,
    pub ngram_hits: u64,
}

impl Default for Penalties {
    /// The serving default: everything off. The Python default read `DSV41_CYCLE_BREAK`
    /// from the environment; in Rust the caller resolves the environment and passes the
    /// result, so config stays visible at the boot guard.
    fn default() -> Self {
        Penalties::new(0.0, 0.0, 4, 16, false, 0)
    }
}

impl Penalties {
    pub fn new(
        presence: f32,
        frequency: f32,
        cycle_repeats: usize,
        cycle_max_period: usize,
        cycle_enabled: bool,
        no_repeat_ngram: usize,
    ) -> Self {
        Penalties {
            presence,
            frequency,
            cycle_repeats,
            cycle_max_period,
            cycle_enabled,
            no_repeat_ngram,
            counts: HashMap::new(),
            hits: 0,
            ngram_hits: 0,
        }
    }

    pub fn active(&self) -> bool {
        self.presence != 0.0 || self.frequency != 0.0 || self.cycle_enabled || self.no_repeat_ngram > 0
    }

    pub fn observe(&mut self, tokens: &[i64]) {
        for &t in tokens {
            *self.counts.entry(t).or_insert(0) += 1;
        }
    }

    /// The token that would continue an exact cycle, or None.
    pub fn cycle_token(&self, history: &[i64]) -> Option<i64> {
        if !self.cycle_enabled || history.len() < self.cycle_repeats.max(4) {
            return None;
        }
        let n = history.len();
        let max_p = self.cycle_max_period.min(n / self.cycle_repeats);
        for p in 1..=max_p {
            let block = &history[n - p..];
            let repeated = (1..self.cycle_repeats).all(|i| {
                n >= p * (i + 1) && &history[n - p * (i + 1)..n - p * i] == block
            });
            if repeated {
                return Some(block[0]);
            }
        }
        None
    }

    /// Tokens that would repeat an n-gram of `no_repeat_ngram` already in `history`.
    ///
    /// Same rule as transformers' NoRepeatNGramLogitsProcessor: if the last n-1 tokens
    /// have appeared before, every token that followed them is refused.
    pub fn banned_ngram_tokens(&self, history: &[i64]) -> Vec<i64> {
        let n = self.no_repeat_ngram;
        if n == 0 || history.len() < n {
            return Vec::new();
        }
        let prefix = &history[history.len() - (n - 1)..];
        let mut banned: Vec<i64> = Vec::new();
        for i in 0..=history.len() - n {
            if &history[i..i + n - 1] == prefix {
                let t = history[i + n - 1];
                if !banned.contains(&t) {
                    banned.push(t);
                }
            }
        }
        banned
    }

    /// `logits [V]` f32 -> the same buffer, penalised in place.
    pub fn apply(&mut self, logits: &mut [f32], history: &[i64]) {
        if self.presence != 0.0 || self.frequency != 0.0 {
            for (&t, &c) in &self.counts {
                if let Some(slot) = logits.get_mut(t as usize) {
                    *slot -= self.presence + self.frequency * c as f32;
                }
            }
        }
        let ban = self.banned_ngram_tokens(history);
        if !ban.is_empty() {
            for t in ban {
                if let Some(slot) = logits.get_mut(t as usize) {
                    *slot = f32::NEG_INFINITY;
                }
            }
            self.ngram_hits += 1;
        }
        if let Some(t) = self.cycle_token(history) {
            self.hits += 1;
            if let Some(slot) = logits.get_mut(t as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // engine/test_penalties.py::test_default_preserves_nested_json
    #[test]
    fn default_preserves_nested_json() {
        let mut ids: Vec<i64> = vec![24313, 80, 3362];
        for _ in 0..7 {
            ids.extend([28612, 80, 3362]);
        }
        ids.extend([223, 2170, 91428, 91428]);
        let mut pen = Penalties::default();
        assert!(!pen.active());
        let mut scores: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
        let expected = scores.clone();
        for j in 0..ids.len() {
            pen.apply(&mut scores, &ids[..j]);
        }
        assert_eq!(scores, expected);
    }

    // engine/test_penalties.py::test_explicit_cycle_break_still_available
    #[test]
    fn explicit_cycle_break_still_available() {
        let mut pen = Penalties::new(0.0, 0.0, 4, 16, true, 0);
        let mut history: Vec<i64> = vec![24313, 80, 3362];
        for _ in 0..3 {
            history.extend([28612, 80, 3362]);
        }
        history.push(28612);
        assert_eq!(pen.cycle_token(&history), Some(80));
        let mut scores = vec![0.0f32; 100];
        pen.apply(&mut scores, &history);
        assert!(scores[80].is_infinite() && scores[80] < 0.0);
        assert_eq!(pen.hits, 1);
    }

    // engine/test_penalties.py::test_explicit_penalties_still_apply
    #[test]
    fn explicit_penalties_still_apply() {
        let mut pen = Penalties::new(0.5, 0.25, 4, 16, false, 0);
        pen.observe(&[1, 1, 2]);
        let mut scores = vec![0.0f32; 4];
        pen.apply(&mut scores, &[1, 1, 2]);
        assert_eq!(scores, vec![0.0, -1.0, -0.75, 0.0]);
    }

    #[test]
    fn greedy_is_one_hot_at_first_max() {
        let p = sample_probs(&[0.0, 3.0, 3.0, 1.0], 0.0, 0.95);
        assert_eq!(p, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn nucleus_keeps_while_prior_mass_below_top_p() {
        // probs ~ [0.6439, 0.2369, 0.0871, 0.0321]: with top_p=0.8 the exclusive cumsum
        // rule keeps the first two (0 < 0.8, 0.6439 < 0.8) and drops the rest.
        let p = sample_probs(&[3.0, 2.0, 1.0, 0.0], 1.0, 0.8);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!((p[0] / p[1] - 3.0f32.exp() / 2.0f32.exp()).abs() < 1e-4);
    }

    #[test]
    fn ngram_ban_refuses_continuation() {
        let mut pen = Penalties::new(0.0, 0.0, 4, 16, false, 3);
        // history ends with [5, 6]; [5, 6, 7] appeared before -> 7 banned
        let history: Vec<i64> = vec![5, 6, 7, 1, 5, 6];
        assert_eq!(pen.banned_ngram_tokens(&history), vec![7]);
        let mut scores = vec![0.0f32; 10];
        pen.apply(&mut scores, &history);
        assert!(scores[7].is_infinite());
        assert_eq!(pen.ngram_hits, 1);
    }
}
