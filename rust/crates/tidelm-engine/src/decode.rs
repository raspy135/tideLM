//! Decode loops over the engine (v41_engine.py::_decode_loop): plain one-token steps, and
//! DSpark speculative steps (draft, verify the block, accept a prefix plus the bonus token,
//! roll back the rest), greedy or sampled — for one sequence (`generate`), and for many at
//! once through continuous batching (`generate_batch`).
//!
//! Sampled speculation is the Python engine's rejection sampling (engine/spec_sampling.py
//! semantics, `tidelm_core::spec::verify_sparse`): the drafter samples from its own
//! top-p nucleus (DSV41_DRAFT_NUCLEUS) and each draft stands with probability
//! min(1, p/q), the first rejection drawing from (p - q)+ - exact for any q, so the output
//! is distributed as sampling the target alone.

use tidelm_core::depth::{ConfidenceDepthPolicy, DepthPolicy};
use tidelm_core::numpy_rng::Pcg64;
use tidelm_core::spec::{verify_sparse, SparseRow};

use crate::forward::{Engine, Nucleus, Sampling, Seg, Want, TMAX};
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
    /// verify depth per speculative step
    pub depths: Vec<usize>,
    /// sampled speculation's self-check: over every draft position verification reached,
    /// the acceptance probability its rows imply (sum of min(p, q)) and the acceptances
    /// seen. The two agree in expectation exactly when each draft was drawn from the q it
    /// was verified against.
    pub accept_expected: f64,
    pub accept_seen: u64,
    pub accept_tests: u64,
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

/// How a decode loop picks each step's verify depth.
pub enum Depth<'a> {
    /// no drafter: one token per backbone step
    Plain,
    /// DSpark speculation at a fixed depth (<= dspark_block_size)
    Fixed(usize),
    /// DSpark's confidence head every step, widths 1..=5 over measured step time
    /// (DSV41_BLOCK_CONFIDENCE with DSV41_VERIFY_ODD); sampled steps take its proposal-prefix
    /// rule (choose_sampled), which keeps the verification exact
    Confidence(&'a mut ConfidenceDepthPolicy),
    /// the Python engine's shipped controller (DSV41_BLOCK_DYNAMIC=3,5 with
    /// DSV41_BLOCK_DYNAMIC_TOKENS=8): a depth from recent acceptance, chosen before the draft
    Window(&'a mut DepthPolicy),
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub sampling: Sampling,
    /// the sampling stream's seed (PCG64): a sampled run is reproducible from it
    pub seed: u64,
    /// read back every consumed logits row and hash it into Generation::digest (greedy
    /// checks); off, greedy reads back only the rows' device argmaxes
    pub digest: bool,
}

impl DecodeOptions {
    pub fn greedy(digest: bool) -> Self {
        DecodeOptions { sampling: Sampling::GREEDY, seed: 0, digest }
    }
}

/// A uniform in [0, 1) with 24 random bits (every value exact in f32).
fn unif(rng: &mut Pcg64) -> f32 {
    (rng.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

/// The inverse-CDF draw from `n` at `u`, kept inside the positive mass.
fn draw(n: &Nucleus, u: f32) -> u32 {
    let total: f32 = n.probs.iter().sum();
    let th = u * total;
    let mut acc = 0.0f32;
    for (j, &p) in n.probs.iter().enumerate() {
        acc += p;
        if acc > th {
            return n.ids[j];
        }
    }
    *n.ids.last().expect("a nucleus holds at least one token")
}

/// Per wanted row of one step's segment: the greedy token (with its logits hashed in when
/// `digest`) or the row's nucleus.
enum Rows {
    Greedy(Vec<u32>),
    Sampled(Vec<Nucleus>),
}

fn step_rows(engine: &mut Engine, seg: Seg, opt: &DecodeOptions, g: &mut Generation) -> Result<Rows> {
    let v = engine.cfg.vocab_size;
    if !opt.sampling.is_greedy() {
        return Ok(Rows::Sampled(engine.forward_batch_sampled(&[seg], &[opt.sampling])?.remove(0)));
    }
    if opt.digest {
        let lg = engine.forward_batch(&[seg])?.remove(0);
        g.digest = digest(g.digest, &lg);
        return Ok(Rows::Greedy(lg.chunks(v).map(argmax_bf16).collect()));
    }
    Ok(Rows::Greedy(engine.forward_batch_argmax(&[seg])?.remove(0)))
}

/// The next token from a one-row step.
fn next_token(rows: Rows, rng: &mut Pcg64) -> u32 {
    match rows {
        Rows::Greedy(am) => am[0],
        Rows::Sampled(nuc) => draw(&nuc[0], unif(rng)),
    }
}

/// Prefill `prompt` into lane 0 (seeding the drafter when `seed`) and pick the first token.
fn prefill_first(engine: &mut Engine, prompt: &[u32], seed: bool, opt: &DecodeOptions, rng: &mut Pcg64,
                 g: &mut Generation) -> Result<u32> {
    let chunks: Vec<&[u32]> = prompt.chunks(TMAX).collect();
    let mut first = None;
    for (i, chunk) in chunks.iter().enumerate() {
        if i + 1 == chunks.len() {
            first = Some(next_token(step_rows(engine, Seg { lane: 0, ids: chunk, want: Want::Last, decode: false }, opt, g)?, rng));
        } else {
            engine.forward(chunk, Want::Nothing)?;
        }
        if seed {
            engine.seed_draft(chunk.len())?;
        }
    }
    first.ok_or_else(|| "an empty prompt".into())
}

/// One sequence on lane 0: prefill `prompt`, then up to `max_tokens` tokens (the first
/// from the prompt's last row), ending early at a token of `stop`. Each emitted burst goes
/// to `on_token`. Greedy speculation emits exactly plain greedy's tokens at any depth;
/// sampled speculation emits a sample from the same distribution as plain sampling.
pub fn generate(engine: &mut Engine, prompt: &[u32], max_tokens: usize, stop: &[u32], mut depth: Depth,
                opt: &DecodeOptions, mut on_token: impl FnMut(&[u32])) -> Result<Generation> {
    engine.reset();
    match &mut depth {
        Depth::Confidence(p) => p.reset_request(),
        Depth::Window(p) => p.reset_request(),
        Depth::Fixed(d) if *d == 0 || *d > engine.cfg.dspark_block_size => {
            return Err(format!("verify depth {d} outside 1..={}", engine.cfg.dspark_block_size).into());
        }
        _ => {}
    }
    let spec = !matches!(depth, Depth::Plain);
    let greedy = opt.sampling.is_greedy();
    let b = engine.cfg.dspark_block_size;
    let mut rng = Pcg64::from_seed(opt.seed as u128);
    let mut g = Generation::default();
    let mut tok = prefill_first(engine, prompt, spec, opt, &mut rng, &mut g)?;
    g.tokens.push(tok);
    on_token(&[tok]);
    let mut pos = prompt.len(); // position of `tok`, not yet forwarded
    while g.tokens.len() < max_tokens && !stop.contains(&tok) {
        if !spec {
            tok = next_token(step_rows(engine, Seg { lane: 0, ids: &[tok], want: Want::Last, decode: true }, opt, &mut g)?, &mut rng);
            pos += 1;
            g.tokens.push(tok);
            g.steps += 1;
            on_token(&[tok]);
            continue;
        }
        let t0 = std::time::Instant::now();
        let caps = engine.graph_captures;
        let window = match &mut depth {
            Depth::Window(p) => Some(p.decide()),
            _ => None,
        };
        let (drafts, q) = if greedy {
            (engine.draft(tok, pos - 1)?, Vec::new())
        } else {
            let u: Vec<f32> = (0..b).map(|_| unif(&mut rng)).collect();
            engine.draft_batch_sampled(&[(0, tok, pos - 1)], &[opt.sampling], &u)?.remove(0)
        };
        let d = match &mut depth {
            Depth::Fixed(d) => *d,
            Depth::Window(_) => window.unwrap(),
            Depth::Confidence(p) => {
                let conf: Vec<f64> = engine.draft_conf()[0].iter().map(|&x| x as f64).collect();
                if greedy { p.choose(&conf) } else { p.choose_sampled(&conf) }.map_err(|e| e.0)?
            }
            Depth::Plain => unreachable!(),
        }
        .min(drafts.len());
        let mut block = vec![tok];
        block.extend_from_slice(&drafts[..d]);
        let (a, mut emitted) = match step_rows(engine, Seg { lane: 0, ids: &block, want: Want::All, decode: true }, opt, &mut g)? {
            Rows::Greedy(am) => accept(&block, &am, stop),
            Rows::Sampled(p) => {
                let pr: Vec<SparseRow> = p.iter().map(|n| n.row()).collect();
                let qr: Vec<SparseRow> = q[..d].iter().map(|n| n.row()).collect();
                let u: Vec<f32> = (0..=d).map(|_| unif(&mut rng)).collect();
                let ds: Vec<i64> = drafts[..d].iter().map(|&x| x as i64).collect();
                let st: Vec<i64> = stop.iter().map(|&x| x as i64).collect();
                let v = verify_sparse(&pr, &qr, &ds, &u, &st).map_err(|e| format!("verify: {e:?}"))?;
                if v.bonus.is_some() {
                    // positions 0..=accepted were tested (the last one failed unless all passed)
                    for i in 0..(v.accepted + 1).min(d) {
                        let (pi, qi) = (&p[i], &q[i]);
                        if !qi.ids.contains(&drafts[i]) {
                            return Err(format!("draft {i} ({}) is outside the q it was drawn from", drafts[i]).into());
                        }
                        g.accept_expected += qi.ids.iter().zip(&qi.probs)
                            .map(|(id, &qx)| pi.ids.iter().position(|x| x == id).map_or(0.0, |j| pi.probs[j].min(qx)) as f64)
                            .sum::<f64>();
                        g.accept_seen += (i < v.accepted) as u64;
                        g.accept_tests += 1;
                    }
                }
                let mut em: Vec<u32> = v.tokens.iter().map(|&x| x as u32).collect();
                em.extend(v.bonus.map(|x| x as u32));
                (v.accepted, em)
            }
        };
        engine.rollback(pos + a + 1)?;
        engine.seed_draft(a + 1)?;
        let step_s = (engine.graph_captures == caps).then(|| t0.elapsed().as_secs_f64());
        match &mut depth {
            Depth::Confidence(p) => p.observe(d, a as u64, emitted.len() as u64, step_s),
            Depth::Window(p) => p.observe(d, a as u64, emitted.len() as u64, step_s),
            _ => {}
        }
        // the limit is on emitted tokens, not verify steps
        emitted.truncate(max_tokens - g.tokens.len());
        pos += a + 1;
        g.steps += 1;
        g.accepted.push(a);
        g.depths.push(d);
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

/// One token per backbone step, greedy, every logits row hashed into the digest.
pub fn generate_plain(engine: &mut Engine, prompt: &[u32], max_tokens: usize, stop: &[u32],
                      on_token: impl FnMut(&[u32])) -> Result<Generation> {
    generate(engine, prompt, max_tokens, stop, Depth::Plain, &DecodeOptions::greedy(true), on_token)
}

/// DSpark speculative greedy decoding at a fixed verify depth (<= dspark_block_size), every
/// logits row hashed into the digest.
pub fn generate_spec(engine: &mut Engine, prompt: &[u32], max_tokens: usize, depth: usize, stop: &[u32],
                     on_token: impl FnMut(&[u32])) -> Result<Generation> {
    generate(engine, prompt, max_tokens, stop, Depth::Fixed(depth), &DecodeOptions::greedy(true), on_token)
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
    pub sampling: Sampling,
    /// its sampling stream's seed: a request draws the same uniforms batched as alone
    pub seed: u64,
}

impl Request {
    pub fn greedy(prompt: Vec<u32>, max_tokens: usize) -> Self {
        Request { prompt, max_tokens, sampling: Sampling::GREEDY, seed: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct BatchOptions {
    /// DSpark speculative decoding at this verify depth; None: one token per step
    pub spec_depth: Option<usize>,
    /// rows per backbone step (<= TMAX): decode rows first, prefill chunks fill the rest
    pub step_rows: usize,
    pub stop: Vec<u32>,
    /// read back full logits and hash them into each Generation's digest (greedy checks);
    /// off, only the device argmaxes (or the sampled rows' nuclei) come back
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
    rng: Pcg64,
    /// the draft's distributions, while its block is being verified
    q: Vec<Nucleus>,
}

/// Continuous batching over the engine's lanes: requests are admitted into free lanes as
/// lanes retire, and every backbone step carries every decoding lane's rows (one token, or a
/// verify block) and fills the rest of `step_rows` with prompt chunks (chunked prefill).
/// Drafts, rollbacks and drafter seeding are batched across lanes the same way. Each
/// request's tokens are what `generate` would emit for it alone (greedy or sampled from its
/// own seed: each lane draws its uniforms in the same order as the single-sequence loop),
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
    if opt.logits && reqs.iter().any(|r| !r.sampling.is_greedy()) {
        return Err("logits digests check greedy requests only".into());
    }
    let b = engine.cfg.dspark_block_size;
    let max_decode = if spec { engine.max_draft_lanes() } else { usize::MAX };
    let mut out: Vec<Option<Generation>> = vec![None; reqs.len()];
    let mut slots: Vec<Option<Slot>> = (0..lanes).map(|_| None).collect();
    let (mut next, mut rr) = (0usize, 0usize);
    let mut stats = BatchStats::default();
    loop {
        for (lane, slot) in slots.iter_mut().enumerate() {
            if slot.is_none() && next < reqs.len() {
                engine.reset_lane(lane)?;
                *slot = Some(Slot { req: next, fed: 0, tok: None, gen: Generation::default(),
                                    rng: Pcg64::from_seed(reqs[next].seed as u128), q: Vec::new() });
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
            let sm: Vec<Sampling> = decode.iter().map(|&l| reqs[slots[l].as_ref().unwrap().req].sampling).collect();
            if sm.iter().all(|s| s.is_greedy()) {
                engine.draft_batch(&dreq)?
            } else {
                // a greedy lane draws nothing (its uniforms are placeholders)
                let mut u = Vec::with_capacity(decode.len() * b);
                for (&l, x) in decode.iter().zip(&sm) {
                    let s = slots[l].as_mut().unwrap();
                    u.extend((0..b).map(|_| if x.is_greedy() { 0.0 } else { unif(&mut s.rng) }));
                }
                let out = engine.draft_batch_sampled(&dreq, &sm, &u)?;
                out.into_iter()
                    .zip(&decode)
                    .map(|((d, q), &l)| {
                        slots[l].as_mut().unwrap().q = q;
                        d
                    })
                    .collect()
            }
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
            .map(|(&lane, b)| Seg { lane, ids: b, want: if spec { Want::All } else { Want::Last }, decode: true })
            .collect();
        for &(lane, c) in &prefill {
            let s = slots[lane].as_ref().unwrap();
            let p = &reqs[s.req].prompt;
            let want = if s.fed + c == p.len() { Want::Last } else { Want::Nothing };
            segs.push(Seg { lane, ids: &p[s.fed..s.fed + c], want, decode: false });
        }
        stats.steps += 1;
        stats.rows += segs.iter().map(|s| s.ids.len()).sum::<usize>();
        let tf = std::time::Instant::now();
        stats.draft_s += (tf - td).as_secs_f64();
        // per segment, the argmax of each wanted row (and with opt.logits, the rows' logits);
        // with sampled lanes, every row's nucleus (a greedy row's is its argmax alone)
        let seg_sampling: Vec<Sampling> = segs.iter().map(|sg| reqs[slots[sg.lane].as_ref().unwrap().req].sampling).collect();
        let (am, logits, nuc): (Vec<Vec<u32>>, Vec<Vec<u16>>, Vec<Vec<Nucleus>>) = if opt.logits {
            let lg = engine.forward_batch(&segs)?;
            (lg.iter().map(|l| l.chunks(v).map(argmax_bf16).collect()).collect(), lg, Vec::new())
        } else if seg_sampling.iter().all(|s| s.is_greedy()) {
            (engine.forward_batch_argmax(&segs)?, Vec::new(), Vec::new())
        } else {
            let nuc = engine.forward_batch_sampled(&segs, &seg_sampling)?;
            (nuc.iter().map(|rows| rows.iter().map(|n| n.ids[0]).collect()).collect(), Vec::new(), nuc)
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
            let sampled = !reqs[s.req].sampling.is_greedy();
            if spec {
                let block = &blocks[i];
                let (a, mut emitted) = if sampled {
                    let d = block.len() - 1;
                    let pr: Vec<SparseRow> = nuc[i].iter().map(|n| n.row()).collect();
                    let qr: Vec<SparseRow> = s.q[..d].iter().map(|n| n.row()).collect();
                    let u: Vec<f32> = (0..=d).map(|_| unif(&mut s.rng)).collect();
                    let ds: Vec<i64> = block[1..].iter().map(|&x| x as i64).collect();
                    let st: Vec<i64> = opt.stop.iter().map(|&x| x as i64).collect();
                    let vd = verify_sparse(&pr, &qr, &ds, &u, &st).map_err(|e| format!("verify: {e:?}"))?;
                    let mut em: Vec<u32> = vd.tokens.iter().map(|&x| x as u32).collect();
                    em.extend(vd.bonus.map(|x| x as u32));
                    (vd.accepted, em)
                } else {
                    accept(block, &am[i], &opt.stop)
                };
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
                let t = if sampled { draw(&nuc[i][0], unif(&mut s.rng)) } else { am[i][0] };
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
                let t = if reqs[s.req].sampling.is_greedy() { am[decode.len() + j][0] } else {
                    draw(&nuc[decode.len() + j][0], unif(&mut s.rng))
                };
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
