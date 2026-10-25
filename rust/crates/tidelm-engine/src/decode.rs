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
    /// with adaptive residency, the request's routed picks the residents missed, per phase
    /// (0 prompt, 1 decode): [missed, picks, missed score, score] (forward::LaneDemand::miss)
    pub miss: [f64; 8],
    /// prompt tokens restored from the prefix cache rather than prefilled
    pub cached: usize,
    /// steps that verified a copy of the prompt (copy.rs) instead of a DSpark draft, and the
    /// copied tokens they accepted
    pub copy_steps: usize,
    pub copy_accepted: usize,
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
    let chunks: Vec<&[u32]> = prompt.chunks(engine.pmax).collect();
    let mut first = None;
    for (i, chunk) in chunks.iter().enumerate() {
        if i + 1 == chunks.len() {
            first = Some(next_token(step_rows(engine, Seg { lane: 0, ids: chunk, want: Want::Last, decode: false, next: &[], hashes: None }, opt, g)?, rng));
        } else {
            engine.forward_next(chunk, chunks[i + 1], Want::Nothing)?;
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
            tok = next_token(step_rows(engine, Seg { lane: 0, ids: &[tok], want: Want::Last, decode: true, next: &[], hashes: None }, opt, &mut g)?, &mut rng);
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
        let (a, mut emitted) = match step_rows(engine, Seg { lane: 0, ids: &block, want: Want::All, decode: true, next: &[], hashes: None }, opt, &mut g)? {
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
    let chunks: Vec<&[u32]> = prompt.chunks(engine.pmax).collect();
    let mut logits = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        let next: &[u32] = if last { &[] } else { chunks[i + 1] };
        logits = engine.forward_next(chunk, next, if last { Want::Last } else { Want::Nothing })?;
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
    /// adaptive residency: passes that moved experts, the experts moved, and the seconds
    /// spent (planning + reading them in)
    pub adapt_passes: usize,
    pub adapt_moves: usize,
    pub adapt_s: f64,
    /// probes run before prefills, and the rows they fed
    pub probes: usize,
    pub probe_rows: usize,
    /// prefix cache: requests that restored a prefix, and the tokens they did not prefill
    pub prefix_hits: usize,
    pub prefix_tokens: usize,
    /// the copy drafter: steps it drafted, and the copied tokens they accepted
    pub copy_steps: usize,
    pub copy_accepted: usize,
}

/// How the scheduler picks each decoding lane's verify depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthMode {
    /// no drafter: one token a step
    Plain,
    Fixed(usize),
    /// the Python engine's shipped controller, one per lane (Depth::Window)
    Window,
    /// DSpark's confidence head, one policy per lane (Depth::Confidence)
    Confidence,
}

#[derive(Clone, Debug)]
pub struct SchedOptions {
    pub depth: DepthMode,
    /// rows per backbone step (<= TMAX): decode rows first, prefill chunks fill the rest
    pub step_rows: usize,
    /// rows a step with no decoding lane gives its prompt chunks (<= the engine's pmax); None:
    /// step_rows
    pub prefill_rows: Option<usize>,
    /// read back full logits and hash them into each Generation's digest (greedy checks)
    pub logits: bool,
}

/// Why a request ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finish {
    /// a stop token
    Stop,
    /// max_tokens
    Length,
    Cancelled,
    /// it could not start (its images could not be encoded): why
    Error(String),
}

/// What the scheduler reports to a request's sink, on the engine's thread.
pub enum Event {
    /// the tokens one step emitted for it, in order
    Tokens(Vec<u32>),
    Done(Finish, Generation),
}

/// A request for the scheduler: `stop` ends it at (and including) any of these tokens.
pub struct Job {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub seed: u64,
    pub stop: Vec<u32>,
    pub sink: Box<dyn FnMut(Event) + Send>,
    /// the prompt's images: each span's first position in `prompt` (whose span positions
    /// carry image_token_id), grid and patches
    pub images: Vec<crate::vision::ImageIn>,
    /// the prefix cache's key for `prompt` (vision::span_keys at image positions, so a prompt
    /// shares a span only with one that shows the same image); None: the prompt itself
    pub key: Option<Vec<u32>>,
    /// the copy drafter may draft for it (when the scheduler has one)
    pub copy_draft: bool,
}

struct Slot {
    job: Job,
    /// prompt tokens forwarded so far
    fed: usize,
    /// decoding: the last emitted token, not yet forwarded
    tok: Option<u32>,
    gen: Generation,
    rng: Pcg64,
    /// the draft's distributions, while its block is being verified
    q: Vec<Nucleus>,
    /// adaptive residency's decode window: the output token count and the lane's decode
    /// [missed, picks] at the last decode check (crate::adapt)
    mark: (usize, f64, f64),
    /// the probe this request runs before its prefill (adapt::ProbeConfig)
    probe: Option<ProbeRun>,
    /// prompt tokens restored from the prefix cache
    restored: usize,
    /// the copy drafter's index of the prompt and the output (set_copy_draft)
    copy: Option<crate::copy::CopyDraft>,
}

/// A request's probe: the prompt rows it feeds and the lane they go to - the request's own
/// from its restored length for a two-pass probe, the engine's probe lane from position 0 for
/// an excerpt - their engram hashes from their places in the prompt, how many have been fed,
/// from which row on they vote (an excerpt's rows in the cached prefix do not), and for a
/// two-pass probe the snapshot that puts its lane back afterwards (None: reset to 0).
struct ProbeRun {
    ids: Vec<u32>,
    hashes: Vec<i64>,
    fed: usize,
    kind: &'static str,
    lane: usize,
    vote_from: usize,
    restore: Option<u64>,
}

/// The prompt tokens a restore of `restored` of `len` already computed: all of them for an exact
/// re-send - the prefix cache keeps prompts one short, so their last token runs again for its
/// logits alone - else `restored`. The probe and the boundary pass count new tokens from here,
/// as the Python engine's full hit (new == 0) does.
fn known(restored: usize, len: usize) -> usize {
    if restored > 0 && restored + 1 == len { len } else { restored }
}

/// A lane's part of the RAM prefix cache (v41_engine's _prefix_cache / _prefix_snapshots): the
/// prompt whose rows its caches hold, snapshots at its prefill boundaries, and the one at its
/// length - 1 (the next prompt re-runs at least one token; Rust has no tail replay to recompute
/// the last row's logits from).
#[derive(Default)]
struct LanePrefix {
    tokens: Vec<u32>,
    bounds: std::collections::BTreeMap<usize, u64>,
    full: Option<(usize, u64)>,
}

/// A lane's depth policy, kept across the requests the lane serves (its step-time estimates
/// carry over; reset_request starts each request).
enum LanePolicy {
    None,
    Window(DepthPolicy),
    Confidence(ConfidenceDepthPolicy),
}

/// Continuous batching over the engine's lanes, online: requests are submitted at any time
/// and admitted into free lanes as lanes retire; every backbone step carries every decoding
/// lane's rows (one token, or a verify block at that lane's depth) and fills the rest of
/// `step_rows` with prompt chunks (chunked prefill). Drafts, rollbacks and drafter seeding
/// are batched across lanes. A request's tokens are what `generate` would emit for it alone
/// under the same depth rule (greedy or sampled from its own seed: each lane draws its
/// uniforms in the single-sequence order), bit for bit while every step - drafts included -
/// stays at or below FMA_ROWS rows (forward.rs, module doc).
pub struct Scheduler {
    opt: SchedOptions,
    slots: Vec<Option<Slot>>,
    policies: Vec<LanePolicy>,
    queue: std::collections::VecDeque<Job>,
    cancelled: Vec<u64>,
    rr: usize,
    pub stats: BatchStats,
    /// adaptive residency (rank 0 of a pair, or one box): folds each request's demand into the
    /// history and re-fits the residents at the prefill -> decode boundary, every
    /// decode_tokens output tokens, and at the end of each request (set_adapt)
    adapt: Option<crate::adapt::Adapt>,
    probe: Option<crate::adapt::ProbeConfig>,
    /// the RAM prefix cache: boundary snapshots kept a lane (None: off)
    prefix_snaps: Option<usize>,
    prefix: Vec<LanePrefix>,
    next_snap: u64,
    /// the copy drafter's rule (None: DSpark drafts every step)
    copy: Option<crate::copy::CopyConfig>,
}

impl Scheduler {
    pub fn new(engine: &Engine, opt: SchedOptions) -> Result<Self> {
        let b = engine.cfg.dspark_block_size;
        let budget = opt.step_rows.clamp(1, TMAX);
        let max_depth = match opt.depth {
            DepthMode::Plain => 0,
            DepthMode::Fixed(d) => {
                if d == 0 || d > b {
                    return Err(format!("verify depth {d} outside 1..={b}").into());
                }
                d
            }
            DepthMode::Window | DepthMode::Confidence => b,
        };
        if 1 + max_depth > budget {
            return Err(format!("a {}-row verify block does not fit {budget} rows per step", 1 + max_depth).into());
        }
        let lanes = engine.max_lanes();
        let policies = (0..lanes)
            .map(|_| -> Result<LanePolicy> {
                Ok(match opt.depth {
                    DepthMode::Window => LanePolicy::Window(
                        DepthPolicy::new((3, 5), tidelm_core::depth::DepthOptions { interval: 8, ..Default::default() })
                            .map_err(|e| e.0)?,
                    ),
                    DepthMode::Confidence => LanePolicy::Confidence(ConfidenceDepthPolicy::new(true, 0)),
                    _ => LanePolicy::None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Scheduler {
            opt,
            slots: (0..lanes).map(|_| None).collect(),
            policies,
            queue: Default::default(),
            cancelled: Vec::new(),
            rr: 0,
            stats: BatchStats::default(),
            adapt: None,
            probe: None,
            prefix_snaps: None,
            prefix: (0..lanes).map(|_| LanePrefix::default()).collect(),
            next_snap: 1,
            copy: None,
        })
    }

    /// The copy drafter (copy.rs): a decoding lane whose output's tail occurs in its prompt
    /// verifies what followed it there instead of a DSpark draft. Speculative modes only.
    pub fn set_copy_draft(&mut self, cfg: crate::copy::CopyConfig) {
        if self.opt.depth != DepthMode::Plain {
            self.copy = Some(cfg);
        }
    }

    /// The RAM prefix cache (DSV41_PREFIX_CACHE=1, DSV41_PREFIX_SNAPSHOTS=`snapshots`): a
    /// request admitted to a lane resumes from the longest snapshot of the lane's last prompt
    /// that it shares, at least its last token short of its own end.
    pub fn set_prefix_cache(&mut self, snapshots: usize) {
        self.prefix_snaps = Some(snapshots);
    }

    fn new_snap(&mut self) -> u64 {
        self.next_snap += 1;
        self.next_snap
    }

    /// Restore `lane` (just reset) from the longest snapshot `prompt` shares with the lane's
    /// last prompt: its length and id (0, None: a miss, and every snapshot of the lane goes).
    /// Snapshots past it are of rows this prompt rewrites, so they go too (v41_engine
    /// _restore_prefix); the prompt's own snapshot at its length - 1 becomes a boundary.
    fn prefix_admit(&mut self, engine: &mut Engine, lane: usize, prompt: &[u32]) -> Result<(usize, Option<u64>)> {
        if self.prefix_snaps.is_none() {
            return Ok((0, None));
        }
        let lp = &self.prefix[lane];
        let common = lp.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let limit = common.min(prompt.len().saturating_sub(1));
        let best = lp.full.iter().copied().chain(lp.bounds.iter().map(|(&n, &id)| (n, id))).filter(|&(n, _)| n <= limit).max();
        // the disk tier, when it restores more than the lane's RAM snapshots do (ties: RAM)
        let ram_n = best.map(|b| b.0).unwrap_or(0);
        let disk = engine.prefix_disk_candidates(prompt, prompt.len().saturating_sub(1));
        if let Some((n, id)) = disk.into_iter().find(|&(n, _)| n > ram_n) {
            let snap = self.new_snap();
            let lp = &mut self.prefix[lane];
            let old: Vec<u64> = lp.bounds.values().copied().chain(lp.full.map(|(_, id)| id)).collect();
            lp.bounds.clear();
            lp.full = None;
            lp.tokens = prompt.to_vec();
            // the bundle's rows replace the lane's: its RAM snapshots are of another history
            engine.prefix_drop(lane, &old)?;
            if engine.prefix_disk_load(lane, &id, n, snap, prompt)? {
                self.prefix[lane].bounds.insert(n, snap);
                self.stats.prefix_hits += 1;
                self.stats.prefix_tokens += n;
                eprintln!("prefix disk: restored {n} of {} tokens from bundle {id}", prompt.len());
                return Ok((n, Some(snap)));
            }
            return Ok((0, None));
        }
        let lp = &mut self.prefix[lane];
        let mut stale: Vec<u64> = Vec::new();
        let out = match best {
            Some((n, id)) => {
                engine.prefix_restore(lane, id)?;
                stale.extend(lp.bounds.range(n + 1..).map(|(_, &id)| id));
                lp.bounds.retain(|&k, _| k <= n);
                match lp.full.take() {
                    Some((fnn, fid)) if fnn == n => {
                        lp.bounds.insert(fnn, fid);
                    }
                    Some((_, fid)) => stale.push(fid),
                    None => {}
                }
                (n, Some(id))
            }
            None => {
                stale.extend(lp.bounds.values().copied());
                stale.extend(lp.full.map(|(_, id)| id));
                lp.bounds.clear();
                lp.full = None;
                (0, None)
            }
        };
        lp.tokens = prompt.to_vec();
        engine.prefix_drop(lane, &stale)?;
        if out.0 > 0 {
            self.stats.prefix_hits += 1;
            self.stats.prefix_tokens += out.0;
        }
        Ok(out)
    }

    /// A boundary snapshot of `lane` at its fed length `n` before its next prompt chunk (the
    /// Python engine's span starts), the shortest dropped past the limit.
    fn prefix_boundary(&mut self, engine: &mut Engine, lane: usize, n: usize) -> Result<()> {
        let Some(keep) = self.prefix_snaps else { return Ok(()) };
        if n == 0 || keep == 0 || self.prefix[lane].bounds.contains_key(&n) {
            return Ok(());
        }
        let id = self.new_snap();
        engine.prefix_snapshot(lane, n, id)?;
        let lp = &mut self.prefix[lane];
        lp.bounds.insert(n, id);
        let mut gone = Vec::new();
        while lp.bounds.len() > keep {
            let (&k, &v) = lp.bounds.iter().next().unwrap();
            lp.bounds.remove(&k);
            gone.push(v);
        }
        engine.prefix_drop(lane, &gone)
    }

    /// The prompt's own snapshot, at its length - 1, after its last chunk (before decode
    /// writes past it).
    fn prefix_full(&mut self, engine: &mut Engine, lane: usize, p: usize) -> Result<()> {
        if self.prefix_snaps.is_none() || p < 2 {
            return Ok(());
        }
        let id = self.new_snap();
        engine.prefix_snapshot(lane, p - 1, id)?;
        let old = self.prefix[lane].full.replace((p - 1, id));
        engine.prefix_drop(lane, &old.map(|(_, id)| vec![id]).unwrap_or_default())?;
        // the disk tier (prefix_persistence.save): a prompt that prefilled anything, at most
        // once an interval, nothing written when a bundle already restores all its prefixes
        let restored = self.slots[lane].as_ref().map(|s| s.restored).unwrap_or(0);
        if p > restored + 1 && engine.prefix_disk_due() {
            let lp = &self.prefix[lane];
            let snaps: Vec<(usize, u64)> = lp.bounds.iter().map(|(&n, &id)| (n, id)).chain(lp.full).collect();
            let ns: Vec<usize> = snaps.iter().map(|s| s.0).collect();
            if !engine.prefix_disk_duplicate(&lp.tokens, &ns) {
                let bid = format!("{:016x}{:08x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0),
                                  self.next_snap as u32);
                let tokens = lp.tokens.clone();
                let ids: Vec<u64> = snaps.iter().map(|s| s.1).collect();
                engine.prefix_disk_save(lane, &bid, &tokens, &ids)?;
            }
        }
        Ok(())
    }

    /// Adaptive residency for this scheduler's engine, which must record demand
    /// (Options::record_demand).
    /// `probe`: also probe each prompt before its prefill (the "pre-prefill" pass).
    pub fn set_adapt(&mut self, adapt: crate::adapt::Adapt, probe: Option<crate::adapt::ProbeConfig>) {
        self.adapt = Some(adapt);
        self.probe = probe;
    }

    /// A probe's end (predictive_prefill.begin): its demand is a preview of the prompt's; if
    /// the residents would miss at least the gate of it (per layer, averaged), they are
    /// re-fitted to the history previewed with it. Then the lane starts the real prefill.
    fn probe_finish(&mut self, engine: &mut Engine, lane: usize) -> Result<()> {
        let pr = self.slots[lane].as_mut().unwrap().probe.take().unwrap();
        let d = engine.lane_demand(pr.lane, true)?;
        // a two-pass probe ran in the request's lane: put it back where the prefill starts
        if pr.lane == lane {
            match pr.restore {
                Some(id) => {
                    engine.prefix_restore(lane, id)?;
                }
                None => engine.reset_lane(lane)?,
            }
        }
        let Some(adapt) = self.adapt.as_ref() else { return Ok(()) };
        let keep = engine.resident_mask().ok_or("adaptive residency needs resident mode")?;
        let predicted = adapt.predicted_miss(&keep, &d);
        self.stats.probes += 1;
        self.stats.probe_rows += pr.ids.len();
        if predicted >= adapt.cfg.gate {
            let n = self.slots[lane].as_ref().unwrap().job.prompt.len();
            self.adapt_pass(engine, Some(&d), &format!("probe, {} {} of {n} tokens, predicted miss {predicted:.4}", pr.kind, pr.ids.len()))?;
        }
        Ok(())
    }

    /// Re-fit the residents to the history (previewed with `pending`), moving experts on both
    /// ranks; logs and counts what it did. `why` names the trigger.
    fn adapt_pass(&mut self, engine: &mut Engine, pending: Option<&crate::forward::LaneDemand>, why: &str) -> Result<()> {
        let Some(adapt) = self.adapt.as_mut() else { return Ok(()) };
        let t0 = std::time::Instant::now();
        let keep = engine.resident_mask().ok_or("adaptive residency needs resident mode")?;
        let plan = adapt.plan(&keep, pending)?;
        if plan.is_empty() {
            return Ok(());
        }
        engine.move_experts(&plan)?;
        adapt.generation += 1;
        let across = plan.iter().filter(|m| m[0] != m[2]).count();
        let after = engine.resident_mask().unwrap();
        let per: Vec<usize> = after.iter().map(|k| k.iter().filter(|&&b| b).count()).collect();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("adapt ({why}): moved {} experts ({across} across layers; residents a layer {}-{}, generation {}) in {:.2} s",
                  plan.len(), per.iter().min().unwrap(), per.iter().max().unwrap(), adapt.generation, dt);
        self.stats.adapt_passes += 1;
        self.stats.adapt_moves += plan.len();
        self.stats.adapt_s += dt;
        Ok(())
    }

    /// The prefill -> decode boundary of `lane` (model.py's maintain_inline): its prompt's
    /// demand folds into the history; a prompt of at least min_new tokens that the residents
    /// served worse than the gate re-fits them.
    fn adapt_boundary(&mut self, engine: &mut Engine, lane: usize) -> Result<()> {
        let Some(adapt) = self.adapt.as_mut() else { return Ok(()) };
        let d = engine.lane_demand(lane, true)?;
        adapt.fold(&d);
        let s = self.slots[lane].as_mut().unwrap();
        for (a, b) in s.gen.miss.iter_mut().zip(&d.miss) {
            *a += b;
        }
        s.mark = (s.gen.tokens.len().max(1), 0.0, 0.0);
        let (rate, _) = crate::adapt::miss_rate(&d);
        let (picks, new) = (d.miss[1] + d.miss[5], s.job.prompt.len() - known(s.restored, s.job.prompt.len()));
        if new >= adapt.cfg.min_new && picks > 0.0 && rate >= adapt.cfg.gate {
            self.adapt_pass(engine, None, &format!("prompt of {new} tokens, miss rate {rate:.4}"))?;
        }
        Ok(())
    }

    /// Every decode_tokens output tokens of `lane`: a stretch the residents served worse than
    /// the gate re-fits them to the history previewed with the answer's demand so far (no
    /// vote cast).
    fn adapt_decode(&mut self, engine: &mut Engine, lane: usize) -> Result<()> {
        let Some(adapt) = self.adapt.as_ref() else { return Ok(()) };
        let n = adapt.cfg.decode_tokens;
        let (gate, mark) = (adapt.cfg.gate, self.slots[lane].as_ref().unwrap().mark);
        let n_out = self.slots[lane].as_ref().unwrap().gen.tokens.len();
        if n == 0 || n_out < mark.0 + n {
            return Ok(());
        }
        let d = engine.lane_demand(lane, false)?;
        let (missed, picks) = (d.miss[4] - mark.1, d.miss[5] - mark.2);
        self.slots[lane].as_mut().unwrap().mark = (n_out, d.miss[4], d.miss[5]);
        if picks > 0.0 && missed / picks >= gate {
            self.adapt_pass(engine, Some(&d), &format!("output token {n_out}, miss rate {:.4} over the last {} tokens",
                                                       missed / picks, n_out - mark.0))?;
        }
        Ok(())
    }

    pub fn submit(&mut self, job: Job) -> Result<()> {
        if job.prompt.is_empty() || job.max_tokens == 0 {
            return Err(format!("request {}: needs a prompt and max_tokens >= 1", job.id).into());
        }
        if self.opt.logits && !job.sampling.is_greedy() {
            return Err("logits digests check greedy requests only".into());
        }
        self.queue.push_back(job);
        Ok(())
    }

    /// End request `id` at the next step boundary (queued or running).
    pub fn cancel(&mut self, id: u64) {
        self.cancelled.push(id);
    }

    pub fn idle(&self) -> bool {
        self.queue.is_empty() && self.slots.iter().all(|s| s.is_none())
    }

    pub fn active(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// End `lane`'s request: with adaptive residency its demand folds into the history
    /// (saved), and a plan re-fits the residents before the next step.
    fn finish(&mut self, engine: &mut Engine, lane: usize, why: Finish) -> Result<()> {
        let Some(mut s) = self.slots[lane].take() else { return Ok(()) };
        if let Some(adapt) = self.adapt.as_mut() {
            let d = engine.lane_demand(lane, true)?;
            adapt.fold(&d);
            adapt.save();
            for (a, b) in s.gen.miss.iter_mut().zip(&d.miss) {
                *a += b;
            }
        }
        let gen = std::mem::take(&mut s.gen);
        (s.job.sink)(Event::Done(why, gen));
        if let Some(h) = engine.hot_stats() {
            eprintln!("hot reads: {} reads over {} decode steps since boot ({} capped, {:.2} s waited)", h.reads, h.steps, h.capped,
                      h.wait_s);
        }
        if self.adapt.is_some() {
            self.adapt_pass(engine, None, "end of request")?;
        }
        Ok(())
    }

    /// One backbone step (admitting, drafting, verifying, emitting), or nothing when idle.
    pub fn step(&mut self, engine: &mut Engine) -> Result<()> {
        // cancellations: queued jobs end unrun, running ones at this boundary
        if !self.cancelled.is_empty() {
            let cancelled = std::mem::take(&mut self.cancelled);
            let mut keep = std::collections::VecDeque::new();
            while let Some(mut j) = self.queue.pop_front() {
                if cancelled.contains(&j.id) {
                    (j.sink)(Event::Done(Finish::Cancelled, Generation::default()));
                } else {
                    keep.push_back(j);
                }
            }
            self.queue = keep;
            for lane in 0..self.slots.len() {
                if matches!(&self.slots[lane], Some(s) if cancelled.contains(&s.job.id)) {
                    self.finish(engine, lane, Finish::Cancelled)?;
                }
            }
        }
        let lanes = self.slots.len();
        for lane in 0..lanes {
            if self.slots[lane].is_none() {
                if let Some(mut job) = self.queue.pop_front() {
                    engine.reset_lane(lane)?;
                    match &mut self.policies[lane] {
                        LanePolicy::Window(p) => p.reset_request(),
                        LanePolicy::Confidence(p) => p.reset_request(),
                        LanePolicy::None => {}
                    }
                    let rng = Pcg64::from_seed(job.seed as u128);
                    let (restored, snap) = self.prefix_admit(engine, lane, job.key.as_deref().unwrap_or(&job.prompt))?;
                    // the images its prefill still reaches (or none: the lane's previous go)
                    if let Err(e) = engine.set_images(lane, &job.images, restored) {
                        eprintln!("request {}: images: {e}", job.id);
                        (job.sink)(Event::Done(Finish::Error(e.to_string()), Generation::default()));
                        continue;
                    }
                    // one excerpt at a time in the probe lane
                    let probe_lane = engine.probe_lane();
                    let lane_busy = self.slots.iter().flatten().any(|s| s.probe.as_ref().is_some_and(|p| p.lane == probe_lane));
                    // the probe routes prompt rows as text: an image request skips it (probe_prefill.py)
                    let probe = match (&self.adapt, &self.probe) {
                        (Some(a), Some(pc)) if job.images.is_empty() => pc.positions(job.prompt.len(), known(restored, job.prompt.len()), a.cfg.min_new).and_then(|(pos, kind)| {
                            let excerpt = kind == "excerpt";
                            if excerpt && lane_busy {
                                return None;
                            }
                            Some(ProbeRun {
                                ids: pos.iter().map(|&p| job.prompt[p]).collect(),
                                hashes: engine.prompt_hashes(&job.prompt, &pos),
                                fed: 0,
                                kind,
                                lane: if excerpt { probe_lane } else { lane },
                                vote_from: if excerpt { pos.iter().take_while(|&&p| p < restored).count() } else { 0 },
                                restore: if excerpt { None } else { snap },
                            })
                        }),
                        _ => None,
                    };
                    if probe.as_ref().is_some_and(|p| p.lane == probe_lane) {
                        engine.reset_lane(probe_lane)?;
                    }
                    let gen = Generation { cached: restored, ..Default::default() };
                    // an image span's positions all carry its placeholder id: never copied
                    let skip = (!job.images.is_empty()).then_some(engine.cfg.image_token_id);
                    let copy = self.copy.clone().filter(|_| job.copy_draft).map(|c| crate::copy::CopyDraft::new(&job.prompt, c, skip));
                    self.slots[lane] = Some(Slot {
                        job, fed: restored, tok: None, gen, rng, q: Vec::new(), mark: (1, 0.0, 0.0), probe, restored, copy,
                    });
                }
            }
        }
        let active = self.active();
        if active == 0 {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let caps = engine.graph_captures;
        self.stats.max_active = self.stats.max_active.max(active);
        let v = engine.cfg.vocab_size;
        let b = engine.cfg.dspark_block_size;
        let budget = self.opt.step_rows.clamp(1, TMAX);
        let spec = self.opt.depth != DepthMode::Plain;
        let max_decode = if spec { engine.max_draft_lanes() } else { usize::MAX };
        // decoding lanes first, starting from a rotating lane so none waits forever when they
        // do not all fit, each reserving its block (a confidence lane its widest); prompt
        // chunks take what is left. The window policy decides before the draft, as in
        // `generate`; decide() is idempotent until the next observe, so asking a lane that
        // then waits for room changes nothing.
        let mut room = budget;
        let mut decode: Vec<(usize, usize)> = Vec::new(); // (lane, depth: decided now, or the reserve)
        // per decoding lane, the copy it verifies instead of a DSpark draft (copy.rs)
        let mut copies: Vec<Option<Vec<u32>>> = Vec::new();
        for i in 0..lanes {
            let lane = (self.rr + i) % lanes;
            if !matches!(&self.slots[lane], Some(s) if s.tok.is_some()) {
                continue;
            }
            let d = match (self.opt.depth, &mut self.policies[lane]) {
                (DepthMode::Plain, _) => 0,
                (DepthMode::Fixed(d), _) => d,
                (_, LanePolicy::Window(w)) => w.decide(),
                _ => b,
            };
            let copy = self.slots[lane].as_ref().unwrap().copy.as_ref().map(|c| c.propose()).filter(|p| !p.is_empty() && room > p.len());
            let d = copy.as_ref().map_or(d, |p| p.len());
            if room > d && decode.len() < max_decode {
                decode.push((lane, d));
                copies.push(copy);
                room -= 1 + d;
            }
        }
        self.rr = (self.rr + 1) % lanes;
        // a step no lane decodes in is the prompts': it may be a prefill-sized step
        if decode.is_empty() {
            room = self.opt.prefill_rows.unwrap_or(budget).clamp(1, engine.pmax);
        }
        // (lane, rows, a probe's rows)
        let mut prefill = Vec::new();
        for (lane, slot) in self.slots.iter().enumerate() {
            if let Some(s) = slot {
                if s.tok.is_none() && room > 0 {
                    let c = match &s.probe {
                        // an excerpt's non-voting rows run as a step of their own
                        Some(pr) if pr.fed < pr.vote_from => (pr.vote_from - pr.fed).min(room),
                        Some(pr) => (pr.ids.len() - pr.fed).min(room),
                        None => (s.job.prompt.len() - s.fed).min(room),
                    };
                    prefill.push((lane, c, s.probe.is_some()));
                    room -= c;
                }
            }
        }
        // the prefix cache's boundary snapshots: a prompt chunk past position 0 starts where
        // its lane stands now
        for &(lane, _, probing) in &prefill {
            if !probing {
                let fed = self.slots[lane].as_ref().unwrap().fed;
                self.prefix_boundary(engine, lane, fed)?;
            }
        }
        let td = std::time::Instant::now();
        // DSpark drafts for the lanes that copy nothing; a copying lane's q is a delta at each
        // copied token, so a sampled verification keeps the target's distribution
        let drafting: Vec<(usize, usize)> = decode.iter().zip(&copies).filter(|(_, c)| c.is_none()).map(|(&x, _)| x).collect();
        let dspark: Vec<Vec<u32>> = if spec && !drafting.is_empty() {
            self.stats.drafts += 1;
            let dreq: Vec<(usize, u32, usize)> = drafting
                .iter()
                .map(|&(l, _)| (l, self.slots[l].as_ref().unwrap().tok.unwrap(), engine.lane_len(l) - 1))
                .collect();
            let sm: Vec<Sampling> = drafting.iter().map(|&(l, _)| self.slots[l].as_ref().unwrap().job.sampling).collect();
            if sm.iter().all(|s| s.is_greedy()) {
                engine.draft_batch(&dreq)?
            } else {
                // a greedy lane draws nothing (its uniforms are placeholders)
                let mut u = Vec::with_capacity(drafting.len() * b);
                for (&(l, _), x) in drafting.iter().zip(&sm) {
                    let s = self.slots[l].as_mut().unwrap();
                    u.extend((0..b).map(|_| if x.is_greedy() { 0.0 } else { unif(&mut s.rng) }));
                }
                let out = engine.draft_batch_sampled(&dreq, &sm, &u)?;
                out.into_iter()
                    .zip(&drafting)
                    .map(|((d, q), &(l, _))| {
                        self.slots[l].as_mut().unwrap().q = q;
                        d
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };
        let mut next_dspark = dspark.into_iter();
        let drafts: Vec<Vec<u32>> = decode
            .iter()
            .zip(&copies)
            .map(|(&(l, _), c)| match c {
                Some(p) => {
                    self.slots[l].as_mut().unwrap().q = p.iter().map(|&t| Nucleus { ids: vec![t], probs: vec![1.0] }).collect();
                    p.clone()
                }
                None => next_dspark.next().unwrap_or_default(),
            })
            .collect();
        // the confidence policy picks each DSpark lane's depth from its draft's confidences
        if spec {
            let mut j = 0;
            for (i, (lane, d)) in decode.iter_mut().enumerate() {
                if copies[i].is_some() {
                    continue;
                }
                if let LanePolicy::Confidence(p) = &mut self.policies[*lane] {
                    let conf: Vec<f64> = engine.draft_conf()[j].iter().map(|&x| x as f64).collect();
                    let greedy = self.slots[*lane].as_ref().unwrap().job.sampling.is_greedy();
                    *d = if greedy { p.choose(&conf) } else { p.choose_sampled(&conf) }.map_err(|e| e.0)?;
                }
                *d = (*d).min(drafts[i].len());
                j += 1;
            }
        }
        let blocks: Vec<Vec<u32>> = decode
            .iter()
            .enumerate()
            .map(|(i, &(l, d))| {
                let mut blk = vec![self.slots[l].as_ref().unwrap().tok.unwrap()];
                if spec {
                    blk.extend_from_slice(&drafts[i][..d]);
                }
                blk
            })
            .collect();
        let mut segs: Vec<Seg> = decode
            .iter()
            .zip(&blocks)
            .map(|(&(lane, _), blk)| Seg { lane, ids: blk, want: if spec { Want::All } else { Want::Last }, decode: true, next: &[], hashes: None })
            .collect();
        for &(lane, c, probing) in &prefill {
            let s = self.slots[lane].as_ref().unwrap();
            if probing {
                // routed with the residents, read for its demand alone: no logits, no seeding
                let pr = s.probe.as_ref().unwrap();
                let per = pr.hashes.len() / pr.ids.len();
                segs.push(Seg {
                    lane: pr.lane, ids: &pr.ids[pr.fed..pr.fed + c], want: Want::Nothing, decode: false, next: &[],
                    hashes: Some(&pr.hashes[pr.fed * per..(pr.fed + c) * per]),
                });
                continue;
            }
            let p = &s.job.prompt;
            let want = if s.fed + c == p.len() { Want::Last } else { Want::Nothing };
            // the next chunk, as wide as this one: prefetched engram rows past what the next
            // step takes stay cached for the one after
            let next = &p[s.fed + c..p.len().min(s.fed + 2 * c)];
            segs.push(Seg { lane, ids: &p[s.fed..s.fed + c], want, decode: false, next, hashes: None });
        }
        self.stats.steps += 1;
        self.stats.rows += segs.iter().map(|s| s.ids.len()).sum::<usize>();
        if std::env::var_os("TIDELM_TRACE_STEPS").is_some() {
            let free = tidelm_cuda::mem_get_info().map(|m| m.0 >> 20).unwrap_or(0);
            eprintln!("step {}: segments {:?}, {} MiB free", self.stats.steps,
                      segs.iter().map(|s| (s.lane, s.ids.len(), s.decode)).collect::<Vec<_>>(), free);
        }
        let tf = std::time::Instant::now();
        self.stats.draft_s += (tf - td).as_secs_f64();
        // per segment, the argmax of each wanted row (and with logits, the rows' logits); with
        // sampled lanes, every row's nucleus (a greedy row's is its argmax alone)
        // by the request each segment serves (a probe's segment may be in the probe lane): the
        // decoding lanes', then the prefill entries', in segment order
        let owners = decode.iter().map(|d| d.0).chain(prefill.iter().map(|p| p.0));
        let seg_sampling: Vec<Sampling> = owners.map(|l| self.slots[l].as_ref().unwrap().job.sampling).collect();
        let (am, logits, nuc): (Vec<Vec<u32>>, Vec<Vec<u16>>, Vec<Vec<Nucleus>>) = if self.opt.logits {
            let lg = engine.forward_batch(&segs)?;
            (lg.iter().map(|l| l.chunks(v).map(argmax_bf16).collect()).collect(), lg, Vec::new())
        } else if seg_sampling.iter().all(|s| s.is_greedy()) {
            (engine.forward_batch_argmax(&segs)?, Vec::new(), Vec::new())
        } else {
            let nuc = engine.forward_batch_sampled(&segs, &seg_sampling)?;
            (nuc.iter().map(|rows| rows.iter().map(|n| n.ids[0]).collect()).collect(), Vec::new(), nuc)
        };
        let tp = std::time::Instant::now();
        self.stats.forward_s += (tp - tf).as_secs_f64();
        let (mut rollbacks, mut seeds) = (Vec::new(), Vec::new());
        // (lane, tokens emitted, (depth, accepted) of a speculative step, its prompt's end)
        let mut emits: Vec<(usize, Vec<u32>, Option<(usize, usize)>, bool)> = Vec::new();
        for (i, &(lane, d)) in decode.iter().enumerate() {
            let s = self.slots[lane].as_mut().unwrap();
            if self.opt.logits {
                s.gen.digest = digest(s.gen.digest, &logits[i]);
            }
            s.gen.steps += 1;
            let sampled = !s.job.sampling.is_greedy();
            if spec {
                let block = &blocks[i];
                let (a, emitted) = if sampled {
                    let pr: Vec<SparseRow> = nuc[i].iter().map(|n| n.row()).collect();
                    let qr: Vec<SparseRow> = s.q[..d].iter().map(|n| n.row()).collect();
                    let u: Vec<f32> = (0..=d).map(|_| unif(&mut s.rng)).collect();
                    let ds: Vec<i64> = block[1..].iter().map(|&x| x as i64).collect();
                    let st: Vec<i64> = s.job.stop.iter().map(|&x| x as i64).collect();
                    let vd = verify_sparse(&pr, &qr, &ds, &u, &st).map_err(|e| format!("verify: {e:?}"))?;
                    let mut em: Vec<u32> = vd.tokens.iter().map(|&x| x as u32).collect();
                    em.extend(vd.bonus.map(|x| x as u32));
                    (vd.accepted, em)
                } else {
                    accept(block, &am[i], &s.job.stop)
                };
                let pos = engine.lane_len(lane) - block.len(); // position of the block's first token
                rollbacks.push((lane, pos + a + 1));
                seeds.push((lane, a + 1));
                s.gen.accepted.push(a);
                s.gen.depths.push(d);
                if copies[i].is_some() {
                    // the depth policies model DSpark's drafts: a copy step is not theirs to see
                    s.copy.as_mut().unwrap().observe(a);
                    (s.gen.copy_steps, s.gen.copy_accepted) = (s.gen.copy_steps + 1, s.gen.copy_accepted + a);
                    (self.stats.copy_steps, self.stats.copy_accepted) = (self.stats.copy_steps + 1, self.stats.copy_accepted + a);
                    emits.push((lane, emitted, None, false));
                } else {
                    emits.push((lane, emitted, Some((d, a)), false));
                }
            } else {
                let t = if sampled { draw(&nuc[i][0], unif(&mut s.rng)) } else { am[i][0] };
                emits.push((lane, vec![t], None, false));
            }
        }
        let mut probed = Vec::new();
        for (j, &(lane, c, probing)) in prefill.iter().enumerate() {
            let s = self.slots[lane].as_mut().unwrap();
            if probing {
                let pr = s.probe.as_mut().unwrap();
                pr.fed += c;
                // the cached head's rows gave the excerpt context; their demand is no vote
                if pr.fed == pr.vote_from && pr.vote_from > 0 {
                    engine.lane_demand(pr.lane, true)?;
                }
                if pr.fed == pr.ids.len() {
                    probed.push(lane);
                }
                continue;
            }
            s.fed += c;
            if spec {
                seeds.push((lane, c));
            }
            if s.fed == s.job.prompt.len() {
                if self.opt.logits {
                    s.gen.digest = digest(s.gen.digest, &logits[decode.len() + j]);
                }
                let t = if s.job.sampling.is_greedy() { am[decode.len() + j][0] } else {
                    draw(&nuc[decode.len() + j][0], unif(&mut s.rng))
                };
                emits.push((lane, vec![t], None, true));
            }
        }
        let tc = std::time::Instant::now();
        self.stats.post_s += (tc - tp).as_secs_f64();
        engine.rollback_batch(&rollbacks)?;
        engine.seed_draft_batch(&seeds)?;
        self.stats.cache_s += tc.elapsed().as_secs_f64();
        // a step that captured a graph is not timed for the depth policies
        let step_s = (engine.graph_captures == caps).then(|| t0.elapsed().as_secs_f64());
        for (lane, mut emitted, verify, prompt_end) in emits {
            let s = self.slots[lane].as_mut().unwrap();
            if let Some((d, a)) = verify {
                match &mut self.policies[lane] {
                    LanePolicy::Window(p) => p.observe(d, a as u64, emitted.len() as u64, step_s),
                    LanePolicy::Confidence(p) => p.observe(d, a as u64, emitted.len() as u64, step_s),
                    LanePolicy::None => {}
                }
            }
            // the limit is on emitted tokens, not steps
            emitted.truncate(s.job.max_tokens - s.gen.tokens.len());
            if let Some(&last) = emitted.last() {
                s.tok = Some(last);
            }
            s.gen.tokens.extend_from_slice(&emitted);
            if let Some(c) = s.copy.as_mut() {
                c.push(&emitted);
            }
            let stopped = emitted.iter().any(|t| s.job.stop.contains(t));
            let full = s.gen.tokens.len() >= s.job.max_tokens;
            (s.job.sink)(Event::Tokens(emitted));
            // the first token came from the pre-adaptation step, as model.py's does
            if prompt_end {
                let p = s.job.prompt.len();
                self.prefix_full(engine, lane, p)?;
                self.adapt_boundary(engine, lane)?;
            }
            if stopped || full {
                self.finish(engine, lane, if stopped { Finish::Stop } else { Finish::Length })?;
            } else if !prompt_end {
                self.adapt_decode(engine, lane)?;
            }
        }
        for lane in probed {
            self.probe_finish(engine, lane)?;
        }
        Ok(())
    }
}

/// `reqs` through the scheduler at a fixed depth (or plain), every request submitted at once:
/// each request's Generation, in order.
pub fn generate_batch(engine: &mut Engine, reqs: &[Request], opt: &BatchOptions) -> Result<(Vec<Generation>, BatchStats)> {
    let depth = match opt.spec_depth {
        None => DepthMode::Plain,
        Some(d) => DepthMode::Fixed(d),
    };
    let mut sched = Scheduler::new(engine, SchedOptions { depth, step_rows: opt.step_rows, prefill_rows: None, logits: opt.logits })?;
    let out: std::sync::Arc<std::sync::Mutex<Vec<Option<Generation>>>> = std::sync::Arc::new(std::sync::Mutex::new(vec![None; reqs.len()]));
    for (i, r) in reqs.iter().enumerate() {
        let out = out.clone();
        sched.submit(Job {
            id: i as u64,
            prompt: r.prompt.clone(),
            max_tokens: r.max_tokens,
            sampling: r.sampling,
            seed: r.seed,
            stop: opt.stop.clone(),
            sink: Box::new(move |ev| {
                if let Event::Done(_, g) = ev {
                    out.lock().unwrap()[i] = Some(g);
                }
            }),
            images: Vec::new(),
            key: None,
            copy_draft: true,
        })?;
    }
    while !sched.idle() {
        sched.step(engine)?;
    }
    let gens = std::mem::take(&mut *out.lock().unwrap());
    Ok((gens.into_iter().map(|g| g.expect("every request retires")).collect(), sched.stats))
}
