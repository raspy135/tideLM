//! Greedy decode loops over the engine (v41_engine.py::_decode_loop, greedy paths): plain
//! one-token steps, and DSpark speculative steps (draft, verify the block, accept the
//! longest matching prefix plus the bonus token, roll back the rest).

use crate::forward::{Engine, Want, TMAX};
use crate::Result;

/// First index of the maximum of a bf16 row (torch.argmax over the engine's bf16 logits).
pub fn argmax_bf16(row: &[u16]) -> u32 {
    let f = |h: u16| f32::from_bits((h as u32) << 16);
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if f(v) > f(row[best]) {
            best = i;
        }
    }
    best as u32
}

#[derive(Debug, Default, Clone)]
pub struct Generation {
    pub tokens: Vec<u32>,
    /// backbone forward calls after the prompt
    pub steps: usize,
    /// accepted drafts per speculative step
    pub accepted: Vec<usize>,
}

impl Generation {
    pub fn tokens_per_step(&self) -> f64 {
        self.tokens.len() as f64 / self.steps.max(1) as f64
    }
}

fn prefill(engine: &mut Engine, prompt: &[u32], seed: bool) -> Result<u32> {
    let chunks: Vec<&[u32]> = prompt.chunks(TMAX).collect();
    let mut logits = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        logits = engine.forward(chunk, if last { Want::Last } else { Want::Nothing })?;
        if seed {
            engine.seed_draft(chunk.len())?;
        }
    }
    Ok(argmax_bf16(&logits))
}

/// One token per backbone step.
pub fn generate_plain(engine: &mut Engine, prompt: &[u32], max_tokens: usize, stop: &[u32],
                      mut on_token: impl FnMut(&[u32])) -> Result<Generation> {
    engine.reset();
    let mut tok = prefill(engine, prompt, false)?;
    let mut g = Generation { tokens: vec![tok], ..Default::default() };
    on_token(&[tok]);
    while g.tokens.len() < max_tokens && !stop.contains(&tok) {
        let lg = engine.forward(&[tok], Want::Last)?;
        tok = argmax_bf16(&lg);
        g.tokens.push(tok);
        g.steps += 1;
        on_token(&[tok]);
    }
    Ok(g)
}

/// DSpark speculative greedy decoding at a fixed verify depth (<= dspark_block_size).
pub fn generate_spec(engine: &mut Engine, prompt: &[u32], max_tokens: usize, depth: usize, stop: &[u32],
                     on_token: impl FnMut(&[u32])) -> Result<Generation> {
    generate_spec_with(engine, prompt, max_tokens, depth, stop, on_token, |e, tok, last, _| e.draft(tok, last))
}

/// The speculative loop with the drafts supplied by `draft(engine, tok, last_main_pos,
/// emitted_so_far)` — the DSpark drafter in serving, an oracle in tests (drafts with a known
/// error position make every accepted length, and so every rollback depth, reachable).
pub fn generate_spec_with(engine: &mut Engine, prompt: &[u32], max_tokens: usize, depth: usize, stop: &[u32],
                          mut on_token: impl FnMut(&[u32]),
                          mut draft: impl FnMut(&mut Engine, u32, usize, usize) -> Result<Vec<u32>>) -> Result<Generation> {
    engine.reset();
    let mut tok = prefill(engine, prompt, true)?;
    let mut g = Generation { tokens: vec![tok], ..Default::default() };
    on_token(&[tok]);
    let v = engine.cfg.vocab_size;
    let mut pos = prompt.len(); // position of `tok`, not yet forwarded
    while g.tokens.len() < max_tokens && !stop.contains(&tok) {
        let drafts = draft(engine, tok, pos - 1, g.tokens.len())?;
        let mut block = vec![tok];
        block.extend_from_slice(&drafts[..depth.min(drafts.len())]);
        let logits = engine.forward(&block, Want::All)?;
        let am: Vec<u32> = (0..block.len()).map(|r| argmax_bf16(&logits[r * v..(r + 1) * v])).collect();
        let mut a = 0;
        while a + 1 < block.len() && am[a] == block[a + 1] {
            a += 1;
        }
        let mut emitted: Vec<u32> = block[1..=a].to_vec();
        let mut bonus = Some(am[a]);
        // an accepted stop token ends the block, with no bonus
        if let Some(j) = emitted.iter().position(|t| stop.contains(t)) {
            a = j + 1;
            emitted.truncate(j + 1);
            bonus = None;
        }
        engine.rollback(pos + a + 1)?;
        engine.seed_draft(a + 1)?;
        if let Some(b) = bonus {
            emitted.push(b);
        }
        // the limit is on emitted tokens, not verify steps
        emitted.truncate(max_tokens - g.tokens.len());
        pos += a + 1;
        g.steps += 1;
        g.accepted.push(a);
        if let Some(&last) = emitted.last() {
            tok = last;
        }
        on_token(&emitted);
        g.tokens.extend_from_slice(&emitted);
        if emitted.iter().any(|t| stop.contains(t)) {
            break;
        }
    }
    Ok(g)
}
