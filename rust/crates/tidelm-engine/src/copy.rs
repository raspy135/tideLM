//! The copy drafter (prompt-lookup decoding, engine/lookup_draft.py): when the last tokens of a
//! request's output also occur in its prompt, the tokens that followed them there are proposed
//! as the next step's draft. It runs on the CPU - an index of the prompt's n-grams, a lookup a
//! step - and reads no weights, so it costs the GPU nothing the way a drafter that streams them
//! would. A proposal is a fixed sequence, so verifying it (greedy, or sampled with q a delta at
//! each copied token) keeps the target's distribution exactly.
//!
//! DSpark already accepts ~5 of its 6-token cap a step on text it can predict, so a copy pays only
//! past that, at the cost of a wider verify (the pair: 6 rows 84 ms, 17 rows 153, 33 rows 217).
//! The rule here is the one rust/tools/copy_draft_replay.py chose on logged runs: copy only from
//! the prompt (files, tool results, earlier turns - where an agent's copies come from; a long
//! repeat inside the answer itself mostly diverges part way, and DSpark covers those as well at a
//! sixth of the width), after a 6-token match, at least 6 tokens, the next copy sized from the
//! last one's acceptance (2 x accepted + 2, within 8..32, from 16). Replayed: a file rewrite
//! 2.27x, a re-run tool call 1.56x, quoting code 1.40x, fresh code and prose unchanged.

use std::collections::HashMap;

/// The rule's settings (TIDELM_COPY_* overrides, for A/B runs).
#[derive(Clone, Debug)]
pub struct CopyConfig {
    /// tokens of the output's tail that must occur in the prompt
    pub match_len: usize,
    /// the shortest copy worth a step
    pub min_copy: usize,
    /// the next copy's length bounds, and the first's
    pub min_cap: usize,
    pub max_cap: usize,
    pub start: usize,
}

impl Default for CopyConfig {
    fn default() -> Self {
        CopyConfig { match_len: 6, min_copy: 6, min_cap: 8, max_cap: 32, start: 16 }
    }
}

impl CopyConfig {
    pub fn from_env() -> crate::Result<Self> {
        let mut c = CopyConfig::default();
        for (k, v) in [("TIDELM_COPY_MATCH", &mut c.match_len), ("TIDELM_COPY_MIN", &mut c.min_copy),
                       ("TIDELM_COPY_MIN_CAP", &mut c.min_cap), ("TIDELM_COPY_MAX", &mut c.max_cap),
                       ("TIDELM_COPY_START", &mut c.start)] {
            if let Ok(s) = std::env::var(k) {
                *v = s.trim().parse().map_err(|_| format!("{k}={s}: not a number"))?;
            }
        }
        if c.match_len == 0 || c.min_cap == 0 || c.min_cap > c.max_cap || c.max_cap + 1 > crate::forward::TMAX {
            return Err(format!("copy drafter settings out of range: {c:?}").into());
        }
        Ok(c)
    }
}

/// Earlier occurrences kept per n-gram: the most recent ones
const OCCURRENCES: usize = 8;
/// How far back a candidate's match is extended to rank it
const MAX_MATCH: usize = 64;

/// One request's copy drafter: its prompt indexed, its output appended as it is emitted.
pub struct CopyDraft {
    cfg: CopyConfig,
    /// the prompt, then the output
    toks: Vec<u32>,
    prompt_len: usize,
    /// n-gram hash -> the positions just past its occurrences in the prompt
    index: HashMap<u64, Vec<u32>>,
    /// the next copy's length
    cap: usize,
    /// never copied (an image span's placeholder positions)
    skip: Option<u32>,
}

fn key(ngram: &[u32]) -> u64 {
    ngram.iter().fold(0xcbf29ce484222325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100000001b3))
}

impl CopyDraft {
    pub fn new(prompt: &[u32], cfg: CopyConfig, skip: Option<u32>) -> Self {
        let n = cfg.match_len;
        let mut index: HashMap<u64, Vec<u32>> = HashMap::with_capacity(prompt.len());
        for e in n..=prompt.len() {
            let g = &prompt[e - n..e];
            if skip.is_some_and(|s| g.contains(&s)) {
                continue;
            }
            let v = index.entry(key(g)).or_default();
            if v.len() == OCCURRENCES {
                v.remove(0);
            }
            v.push(e as u32);
        }
        let cap = cfg.start.clamp(cfg.min_cap, cfg.max_cap);
        CopyDraft { cfg, toks: prompt.to_vec(), prompt_len: prompt.len(), index, cap, skip }
    }

    /// Tokens the request emitted.
    pub fn push(&mut self, toks: &[u32]) {
        self.toks.extend_from_slice(toks);
    }

    /// The draft after the context's last token: what followed the prompt occurrence of its tail
    /// that matches furthest back (ties: the latest), up to the current cap; empty when there is
    /// none or it is shorter than min_copy.
    pub fn propose(&self) -> Vec<u32> {
        let (n, len) = (self.cfg.match_len, self.toks.len());
        if len <= self.prompt_len || len < n {
            return Vec::new();
        }
        let tail = &self.toks[len - n..];
        let Some(cands) = self.index.get(&key(tail)) else { return Vec::new() };
        let mut best: Option<(usize, usize)> = None; // (match length, position)
        for &e in cands {
            let e = e as usize;
            if self.toks[e - n..e] != *tail {
                continue; // a hash collision
            }
            let mut m = n;
            while m < MAX_MATCH && e > m && self.toks[e - m - 1] == self.toks[len - m - 1] {
                m += 1;
            }
            if best.is_none_or(|(bm, be)| m > bm || (m == bm && e > be)) {
                best = Some((m, e));
            }
        }
        let Some((_, e)) = best else { return Vec::new() };
        let end = (e + self.cap).min(len);
        let mut out: Vec<u32> = self.toks[e..end].to_vec();
        if let Some(s) = self.skip {
            if let Some(i) = out.iter().position(|&t| t == s) {
                out.truncate(i);
            }
        }
        if out.len() < self.cfg.min_copy {
            out.clear();
        }
        out
    }

    /// A copy step's outcome: the next copy is sized from its acceptance.
    pub fn observe(&mut self, accepted: usize) {
        self.cap = (2 * accepted + 2).clamp(self.cfg.min_cap, self.cfg.max_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CopyConfig {
        CopyConfig { match_len: 3, min_copy: 2, min_cap: 4, max_cap: 8, start: 6 }
    }

    #[test]
    fn copies_what_followed_the_longest_match_in_the_prompt() {
        // "1 2 3" occurs twice; the first is preceded by 7, as the output's tail is
        let prompt = [7, 1, 2, 3, 4, 5, 9, 1, 2, 3, 8, 8];
        let mut c = CopyDraft::new(&prompt, cfg(), None);
        assert!(c.propose().is_empty()); // nothing emitted yet
        c.push(&[7, 1, 2, 3]);
        assert_eq!(c.propose(), vec![4, 5, 9, 1, 2, 3]); // the longer match, not the later one
        c.observe(1); // a short acceptance: the next copy at the floor
        assert_eq!(c.propose(), vec![4, 5, 9, 1]);
        c.observe(8);
        assert_eq!(c.propose(), vec![4, 5, 9, 1, 2, 3, 8, 8]);
        c.push(&[6]);
        assert!(c.propose().is_empty()); // "2 3 6" is nowhere in the prompt
    }

    #[test]
    fn never_copies_from_the_output_or_through_an_image() {
        let prompt = [1, 2, 3, 7, 7, 99, 8];
        let mut c = CopyDraft::new(&prompt, cfg(), Some(99));
        c.push(&[5, 6, 5, 6, 5, 6]); // repeats inside the output only
        assert!(c.propose().is_empty());
        c.push(&[1, 2, 3]);
        // "7 7" then the image token: cut before it, and 2 tokens is the minimum
        assert_eq!(c.propose(), vec![7, 7]);
    }
}
