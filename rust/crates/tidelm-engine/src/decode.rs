//! Greedy decode loops over the engine (v41_engine.py::_decode_loop, greedy paths): plain
//! one-token steps, and DSpark speculative steps (draft, verify the block, accept the
//! longest matching prefix plus the bonus token, roll back the rest) — for one sequence, and
//! for many at once through continuous batching (`generate_batch`).

use crate::forward::{Engine, Seg, Want, TMAX};
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
    /// a hash of every logits row the sequence consumed, in order: two runs with equal
    /// digests computed bit-identical logits
    pub digest: u64,
}

fn digest(h: u64, logits: &[u16]) -> u64 {
    logits.iter().fold(h, |h, &x| (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Greedy verification of `block` = [tok, drafts..] against its rows' argmaxes `am`: the
/// drafts kept and the tokens to emit (the accepted drafts, then the bonus — unless a stop
/// token was accepted, which ends the block there).
fn accept(block: &[u32], am: &[u32], stop: &[u32]) -> (usize, Vec<u32>) {
    let mut a = 0;
    while a + 1 < block.len() && am[a] == block[a + 1] {
        a += 1;
    }
    let mut emitted: Vec<u32> = block[1..=a].to_vec();
    if let Some(j) = emitted.iter().position(|t| stop.contains(t)) {
        emitted.truncate(j + 1);
        return (j + 1, emitted);
    }
    emitted.push(am[a]);
    (a, emitted)
}

impl Generation {
    pub fn tokens_per_step(&self) -> f64 {
        self.tokens.len() as f64 / self.steps.max(1) as f64
    }
}

fn prefill(engine: &mut Engine, prompt: &[u32], seed: bool, g: &mut Generation) -> Result<u32> {
    let chunks: Vec<&[u32]> = prompt.chunks(TMAX).collect();
    let mut logits = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        logits = engine.forward(chunk, if last { Want::Last } else { Want::Nothing })?;
        if seed {
            engine.seed_draft(chunk.len())?;
        }
    }
    g.digest = digest(g.digest, &logits);
    Ok(argmax_bf16(&logits))
}

/// One token per backbone step.
pub fn generate_plain(engine: &mut Engine, prompt: &[u32], max_tokens: usize, stop: &[u32],
                      mut on_token: impl FnMut(&[u32])) -> Result<Generation> {
    engine.reset();
    let mut g = Generation::default();
    let mut tok = prefill(engine, prompt, false, &mut g)?;
    g.tokens.push(tok);
    on_token(&[tok]);
    while g.tokens.len() < max_tokens && !stop.contains(&tok) {
        let lg = engine.forward(&[tok], Want::Last)?;
        g.digest = digest(g.digest, &lg);
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
    let mut g = Generation::default();
    let mut tok = prefill(engine, prompt, true, &mut g)?;
    g.tokens.push(tok);
    on_token(&[tok]);
    let v = engine.cfg.vocab_size;
    let mut pos = prompt.len(); // position of `tok`, not yet forwarded
    while g.tokens.len() < max_tokens && !stop.contains(&tok) {
        let drafts = draft(engine, tok, pos - 1, g.tokens.len())?;
        let mut block = vec![tok];
        block.extend_from_slice(&drafts[..depth.min(drafts.len())]);
        let logits = engine.forward(&block, Want::All)?;
        g.digest = digest(g.digest, &logits);
        let am: Vec<u32> = (0..block.len()).map(|r| argmax_bf16(&logits[r * v..(r + 1) * v])).collect();
        let (a, mut emitted) = accept(&block, &am, stop);
        engine.rollback(pos + a + 1)?;
        engine.seed_draft(a + 1)?;
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

/// One request of a batched run.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct BatchOptions {
    /// DSpark speculative decoding at this verify depth; None: one token per step
    pub spec_depth: Option<usize>,
    /// rows per backbone step (<= TMAX): decode rows first, prefill chunks fill the rest
    pub step_rows: usize,
    pub stop: Vec<u32>,
    /// read back full logits and hash them into each Generation's digest (checks); off,
    /// only the device argmaxes come back
    pub logits: bool,
}

#[derive(Default, Debug, Clone)]
pub struct BatchStats {
    /// backbone steps, and the live rows they ran
    pub steps: usize,
    pub rows: usize,
    /// draft calls (each drafts for every decoding lane of its step)
    pub drafts: usize,
    pub max_active: usize,
    /// wall time in drafting, in backbone steps (host prep, device, read-back), in the host's
    /// accept / argmax / digest pass, and in rollback + seeding
    pub draft_s: f64,
    pub forward_s: f64,
    pub post_s: f64,
    pub cache_s: f64,
}

struct Slot {
    req: usize,
    /// prompt tokens forwarded so far
    fed: usize,
    /// decoding: the last emitted token, not yet forwarded
    tok: Option<u32>,
    gen: Generation,
}

/// Continuous batching over the engine's lanes: requests are admitted into free lanes as
/// lanes retire, and every backbone step carries every decoding lane's rows (one token, or a
/// verify block) and fills the rest of `step_rows` with prompt chunks (chunked prefill).
/// Drafts, rollbacks and drafter seeding are batched across lanes the same way. Each
/// request's tokens are what `generate_plain` / `generate_spec` would emit for it alone,
/// bit for bit while every step stays at or below FMA_ROWS rows (forward.rs, module doc).
pub fn generate_batch(engine: &mut Engine, reqs: &[Request], opt: &BatchOptions) -> Result<(Vec<Generation>, BatchStats)> {
    let lanes = engine.max_lanes();
    let budget = opt.step_rows.clamp(1, TMAX);
    let v = engine.cfg.vocab_size;
    let depth = opt.spec_depth.unwrap_or(0);
    let spec = opt.spec_depth.is_some();
    if spec && (depth == 0 || depth > engine.cfg.dspark_block_size) {
        return Err(format!("verify depth {depth} outside 1..={}", engine.cfg.dspark_block_size).into());
    }
    if 1 + depth > budget {
        return Err(format!("a {}-row verify block does not fit {budget} rows per step", 1 + depth).into());
    }
    if let Some(i) = reqs.iter().position(|r| r.prompt.is_empty() || r.max_tokens == 0) {
        return Err(format!("request {i}: needs a prompt and max_tokens >= 1").into());
    }
    let max_decode = if spec { engine.max_draft_lanes() } else { usize::MAX };
    let mut out: Vec<Option<Generation>> = vec![None; reqs.len()];
    let mut slots: Vec<Option<Slot>> = (0..lanes).map(|_| None).collect();
    let (mut next, mut rr) = (0usize, 0usize);
    let mut stats = BatchStats::default();
    loop {
        for (lane, slot) in slots.iter_mut().enumerate() {
            if slot.is_none() && next < reqs.len() {
                engine.reset_lane(lane)?;
                *slot = Some(Slot { req: next, fed: 0, tok: None, gen: Generation::default() });
                next += 1;
            }
        }
        let active = slots.iter().filter(|s| s.is_some()).count();
        if active == 0 {
            break;
        }
        stats.max_active = stats.max_active.max(active);
        // decoding lanes first, starting from a rotating lane so none waits forever when they
        // do not all fit; prompt chunks take what is left
        let mut room = budget;
        let mut decode = Vec::new();
        for i in 0..lanes {
            let lane = (rr + i) % lanes;
            if matches!(&slots[lane], Some(s) if s.tok.is_some()) && room > depth && decode.len() < max_decode {
                decode.push(lane);
                room -= 1 + depth;
            }
        }
        rr = (rr + 1) % lanes;
        let mut prefill = Vec::new();
        for (lane, slot) in slots.iter().enumerate() {
            if let Some(s) = slot {
                if s.tok.is_none() && room > 0 {
                    let c = (reqs[s.req].prompt.len() - s.fed).min(room);
                    prefill.push((lane, c));
                    room -= c;
                }
            }
        }
        let td = std::time::Instant::now();
        let drafts = if spec && !decode.is_empty() {
            stats.drafts += 1;
            let dreq: Vec<(usize, u32, usize)> =
                decode.iter().map(|&l| (l, slots[l].as_ref().unwrap().tok.unwrap(), engine.lane_len(l) - 1)).collect();
            engine.draft_batch(&dreq)?
        } else {
            Vec::new()
        };
        let blocks: Vec<Vec<u32>> = decode
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let mut b = vec![slots[l].as_ref().unwrap().tok.unwrap()];
                if spec {
                    b.extend_from_slice(&drafts[i][..depth]);
                }
                b
            })
            .collect();
        let mut segs: Vec<Seg> = decode
            .iter()
            .zip(&blocks)
            .map(|(&lane, b)| Seg { lane, ids: b, want: if spec { Want::All } else { Want::Last } })
            .collect();
        for &(lane, c) in &prefill {
            let s = slots[lane].as_ref().unwrap();
            let p = &reqs[s.req].prompt;
            let want = if s.fed + c == p.len() { Want::Last } else { Want::Nothing };
            segs.push(Seg { lane, ids: &p[s.fed..s.fed + c], want });
        }
        stats.steps += 1;
        stats.rows += segs.iter().map(|s| s.ids.len()).sum::<usize>();
        let tf = std::time::Instant::now();
        stats.draft_s += (tf - td).as_secs_f64();
        // per segment, the argmax of each wanted row (and with opt.logits, the rows' logits)
        let (am, logits): (Vec<Vec<u32>>, Vec<Vec<u16>>) = if opt.logits {
            let lg = engine.forward_batch(&segs)?;
            (lg.iter().map(|l| l.chunks(v).map(argmax_bf16).collect()).collect(), lg)
        } else {
            (engine.forward_batch_argmax(&segs)?, Vec::new())
        };
        let tp = std::time::Instant::now();
        stats.forward_s += (tp - tf).as_secs_f64();
        let (mut rollbacks, mut seeds) = (Vec::new(), Vec::new());
        for (i, &lane) in decode.iter().enumerate() {
            let s = slots[lane].as_mut().unwrap();
            if opt.logits {
                s.gen.digest = digest(s.gen.digest, &logits[i]);
            }
            s.gen.steps += 1;
            if spec {
                let block = &blocks[i];
                let (a, mut emitted) = accept(block, &am[i], &opt.stop);
                let pos = engine.lane_len(lane) - block.len(); // position of the block's first token
                rollbacks.push((lane, pos + a + 1));
                seeds.push((lane, a + 1));
                emitted.truncate(reqs[s.req].max_tokens - s.gen.tokens.len());
                s.gen.accepted.push(a);
                if let Some(&last) = emitted.last() {
                    s.tok = Some(last);
                }
                s.gen.tokens.extend_from_slice(&emitted);
            } else {
                let t = am[i][0];
                s.gen.tokens.push(t);
                s.tok = Some(t);
            }
        }
        for (j, &(lane, c)) in prefill.iter().enumerate() {
            let s = slots[lane].as_mut().unwrap();
            s.fed += c;
            if spec {
                seeds.push((lane, c));
            }
            if s.fed == reqs[s.req].prompt.len() {
                if opt.logits {
                    s.gen.digest = digest(s.gen.digest, &logits[decode.len() + j]);
                }
                let t = am[decode.len() + j][0];
                s.gen.tokens.push(t);
                s.tok = Some(t);
            }
        }
        let tc = std::time::Instant::now();
        stats.post_s += (tc - tp).as_secs_f64();
        engine.rollback_batch(&rollbacks)?;
        engine.seed_draft_batch(&seeds)?;
        stats.cache_s += tc.elapsed().as_secs_f64();
        for slot in slots.iter_mut() {
            let done = matches!(slot, Some(s) if s.tok.is_some()
                && (s.gen.tokens.len() >= reqs[s.req].max_tokens || s.gen.tokens.last().is_some_and(|t| opt.stop.contains(t))));
            if done {
                let s = slot.take().unwrap();
                out[s.req] = Some(s.gen);
            }
        }
    }
    Ok((out.into_iter().map(|g| g.expect("every request retires")).collect(), stats))
}
