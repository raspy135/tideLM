//! The serving loop (engine/glm/tp.py::_stream) for one request at a time on rank 0: prefix
//! reuse of the live sequence, the prefill (two-pass for a short miss-heavy suffix), then
//! rounds of MTP drafting under the calibrated gate (or a copy: of the prompt, up to 32 tokens
//! by the DS engine's rule, else of the prompt or answer by GLM's lookup), one verify, exact
//! acceptance (greedy, or rejection sampling against each draft's nucleus), the MTP cache
//! refilled for what was accepted; adaptive residency between them. Every engine call it makes
//! is mirrored to rank 1, which decides nothing.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use tidelm_core::numpy_rng::Pcg64;
use tidelm_engine::copy::{CopyConfig, CopyDraft};
use tidelm_engine::decode::{Event, Finish, Generation, Job};

use crate::forward::{Engine, Hidden, MtpRows, RowCands, HEAD_ROWS};
use crate::weights::MAX_ROWS;
use crate::Result;

/// mtp_cal.py: the calibrated draft gate.
pub struct CalGate {
    pub tau: f64,
    acc: Vec<[u32; 8]>,
    n: Vec<[u32; 8]>,
}

const EDGES: [f64; 9] = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0];
const PRIOR: [[(u32, u32); 8]; 7] = [
    [(22, 62), (21, 31), (24, 46), (14, 29), (17, 35), (34, 46), (28, 36), (248, 260)],
    [(15, 146), (19, 65), (25, 59), (19, 35), (11, 31), (27, 39), (17, 29), (117, 141)],
    [(22, 259), (13, 68), (17, 60), (11, 39), (7, 20), (12, 22), (14, 19), (44, 58)],
    [(20, 336), (6, 63), (9, 41), (10, 29), (7, 19), (6, 16), (6, 9), (20, 31)],
    [(8, 407), (4, 41), (6, 32), (3, 20), (3, 9), (4, 10), (5, 5), (10, 18)],
    [(5, 449), (2, 27), (4, 27), (2, 8), (0, 3), (3, 7), (2, 5), (4, 9)],
    [(1, 467), (1, 26), (4, 13), (2, 4), (0, 0), (0, 5), (1, 5), (2, 4)],
];
const SMOOTH: f64 = 4.0;

fn bucket(conf: f64) -> usize {
    let c = conf.clamp(0.0, 1.0);
    (0..7).find(|&i| c < EDGES[i + 1]).unwrap_or(7)
}

impl CalGate {
    pub fn new(tau: f64, prior: bool) -> Self {
        let (mut acc, mut n) = (vec![], vec![]);
        if prior {
            for row in PRIOR {
                acc.push(row.map(|x| x.0));
                n.push(row.map(|x| x.1));
            }
        }
        CalGate { tau, acc, n }
    }

    fn row(&mut self, k: usize) -> usize {
        while self.acc.len() <= k {
            self.acc.push([0; 8]);
            self.n.push([0; 8]);
        }
        k
    }

    pub fn observe(&mut self, k: usize, conf: f64, accepted: bool) {
        let r = self.row(k);
        let b = bucket(conf);
        self.n[r][b] += 1;
        if accepted {
            self.acc[r][b] += 1;
        }
    }

    pub fn prob(&mut self, k: usize, conf: f64) -> f64 {
        let r = self.row(k);
        let b = bucket(conf);
        let (a, n) = (self.acc[r][b] as f64, self.n[r][b] as f64);
        let (tot, seen) = (self.acc[r].iter().sum::<u32>() as f64, self.n[r].iter().sum::<u32>() as f64);
        let pooled = if seen > 0.0 { tot / seen } else { conf };
        (a + SMOOTH * pooled) / (n + SMOOTH)
    }

    pub fn keep(&mut self, k: usize, conf: f64) -> bool {
        self.prob(k, conf) >= self.tau
    }
}

/// engine/lookup_draft.py's ExactDraftCache (VERSION 2): the request's history (prompt, then
/// every emitted token) indexed by `min_match`-token suffixes; a proposal is the settled
/// continuation of the longest backward match (ties to the newest), exactly `depth` tokens.
pub struct Lookup {
    min_match: usize,
    max_cand: usize,
    max_match: usize,
    history: Vec<u32>,
    index: HashMap<Vec<u32>, Vec<usize>>,
}

impl Lookup {
    pub fn new(history: &[u32], min_match: usize) -> Self {
        let mut l = Lookup { min_match, max_cand: 8, max_match: 64.max(min_match), history: history.to_vec(),
                             index: HashMap::new() };
        for end in min_match..l.history.len() {
            l.remember(end);
        }
        l
    }

    fn remember(&mut self, end: usize) {
        let key = self.history[end - self.min_match..end].to_vec();
        let v = self.index.entry(key).or_default();
        v.push(end);
        if v.len() > self.max_cand {
            v.remove(0);
        }
    }

    pub fn extend(&mut self, toks: &[u32]) {
        for &t in toks {
            let end = self.history.len();
            if end >= self.min_match {
                self.remember(end);
            }
            self.history.push(t);
        }
    }

    pub fn propose(&self, depth: usize) -> Option<Vec<u32>> {
        let now = self.history.len();
        if depth == 0 || now < self.min_match {
            return None;
        }
        let pos = self.index.get(&self.history[now - self.min_match..])?;
        let (mut best_end, mut best) = (None, 0);
        for &end in pos.iter().rev() {
            if end + depth > now {
                continue;
            }
            let mut m = self.min_match;
            let (mut l, mut r) = (end as isize - self.min_match as isize - 1, now as isize - self.min_match as isize - 1);
            while m < self.max_match && l >= 0 && r >= 0 && self.history[l as usize] == self.history[r as usize] {
                m += 1;
                l -= 1;
                r -= 1;
            }
            if best_end.is_none() || m > best {
                best_end = Some(end);
                best = m;
            }
        }
        best_end.map(|e| self.history[e..e + depth].to_vec())
    }
}

/// The copy drafter's sizes on GLM: the DS engine's rule (copy.rs: from the prompt, after a 6-token
/// match, the next copy 2 x the last one's accepted + 2) with the shortest copy 7, the longest
/// MTP draft, so a short copy verifies in a graphed 8-row pass. TIDELM_COPY_* override.
pub fn glm_copy_config() -> Result<CopyConfig> {
    let c = CopyConfig::with_env(CopyConfig { match_len: 6, min_copy: 6, min_cap: 7, max_cap: 32, start: 16 })?;
    if c.max_cap + 1 > crate::forward::HEAD_ROWS {
        return Err(format!("TIDELM_COPY_MAX: at most {}", crate::forward::HEAD_ROWS - 1).into());
    }
    Ok(c)
}

/// A row's sampling distribution: (token, probability) by probability descending.
pub type Dist = Vec<(u32, f32)>;

/// The nucleus (sample_probs' cum - p < top_p rule over softmax(z / T), renormalized) from both
/// ranks' candidates, or None when they cannot prove it: the kept set must stay above every
/// rank's smallest candidate and reach top_p.
pub fn nucleus(rc: &RowCands, temp: f32, top_p: f32) -> Option<Dist> {
    // a NaN logit proves nothing: the full row decides (Server::dist)
    if rc.vals.iter().chain(rc.lse.iter().map(|x| &x.0)).chain(rc.lse.iter().map(|x| &x.1)).any(|v| v.is_nan()) {
        return None;
    }
    let m = rc.lse.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = rc.lse.iter().map(|&(mr, s)| s as f64 * (((mr - m) / temp) as f64).exp()).sum();
    let mut c: Vec<(f32, u32)> = rc.vals.iter().zip(&rc.ids).map(|(&v, &i)| (v, i)).collect();
    c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let floor = rc.min.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let (mut out, mut cum) = (vec![], 0f64);
    for (v, i) in c {
        if cum >= top_p as f64 {
            break;
        }
        if v <= floor {
            return None;
        }
        let p = (((v - m) / temp) as f64).exp() / z;
        out.push((i, p as f32));
        cum += p;
    }
    if cum < top_p as f64 && top_p < 1.0 {
        return None;
    }
    if top_p >= 1.0 {
        return None;
    }
    let tot: f32 = out.iter().map(|x| x.1).sum();
    Some(out.into_iter().map(|(i, p)| (i, p / tot)).collect())
}

fn dist_from_logits(z: &[f32], temp: f32, top_p: f32) -> Dist {
    let p = tidelm_core::sampling::sample_probs(z, temp, top_p);
    let mut d: Dist = p.iter().enumerate().filter(|(_, &x)| x > 0.0).map(|(i, &x)| (i as u32, x)).collect();
    d.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    d
}

fn unif(rng: &mut Pcg64) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn draw(d: &Dist, rng: &mut Pcg64) -> u32 {
    let tot: f64 = d.iter().map(|x| x.1 as f64).sum();
    let th = unif(rng) * tot;
    let mut acc = 0f64;
    for &(i, p) in d {
        acc += p as f64;
        if acc > th {
            return i;
        }
    }
    d.last().map(|x| x.0).unwrap_or(0)
}

fn prob_of(d: &Dist, t: u32) -> f32 {
    d.iter().find(|x| x.0 == t).map(|x| x.1).unwrap_or(0.0)
}

/// _tail_cycle: the tokens end in a 1..8-token period repeated `reps` times.
pub fn tail_cycle(seq: &[u32], reps: usize) -> bool {
    (1..=8).any(|p| {
        let n = p * reps;
        n <= seq.len() && (seq.len() - n + p..seq.len()).all(|i| seq[i] == seq[i - p])
    })
}

#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub depth: usize,
    /// GLM_MTP_CAL: the calibrated gate's threshold (None: the flat gate at `dyn_gate`)
    pub cal: Option<f64>,
    pub dyn_gate: f64,
    /// GLM_LOOKUP: copy rounds from a match of this many tokens (0 off)
    pub lookup: usize,
    /// GLM_COPY: longer copies of the prompt (tidelm_engine::copy's rule, glm_copy_config's
    /// sizes) ahead of the lookup's; a request's x_copy_draft=false leaves only the lookup
    pub copy: Option<CopyConfig>,
    pub adapt: bool,
    pub adapt_every: usize,
    pub adapt_prefill: usize,
    pub adapt_decode: usize,
    pub bg_every: usize,
    pub hot_keep: usize,
    pub loop_break: usize,
    /// GLM_PREFILL_TWO_PASS: a suffix of 4..=this many new tokens runs twice when its first pass
    /// computed >= two_pass_miss of its routed weight on substitutes
    pub two_pass: usize,
    pub two_pass_miss: f64,
    /// GLM_LANE_CHUNK: a prompt admitted while other lanes decode prefills in parts of this many
    /// rows, a round of theirs between parts (a part alone would hold them up seconds a chunk)
    pub lane_chunk: usize,
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub copy_rounds: usize,
    pub copy_accepted: usize,
    /// copy rounds that copied the prompt by the long rule (GLM_COPY)
    pub long_copies: usize,
    pub fallbacks: usize,
    pub loop_breaks: usize,
    pub two_pass: usize,
    /// rounds that verified beside another lane's rows (a request's)
    pub shared_rounds: usize,
}

/// Positions a lane must be able to grow past its prompt before it is admitted beside another
/// (the lanes share one KV pool: tidelm_engine::kv).
const KV_HEADROOM: usize = 4096;

/// A request in its lane: generate's state between rounds.
struct Run {
    job: Job,
    g: Generation,
    lane: usize,
    rng: Pcg64,
    greedy: bool,
    temp: f32,
    top_p: f32,
    depth: usize,
    /// the next token, decided, not yet emitted or in the cache
    anchor: u32,
    /// the anchor's distribution, for a loop break (sampled)
    z_anchor: Option<Dist>,
    /// the anchor row's candidates, for a loop break (greedy): later passes overwrite the engine's
    anchor_cands: Option<RowCands>,
    lookup: Option<Lookup>,
    copier: Option<CopyDraft>,
    copied: usize,
    synced: usize,
    out: Vec<u32>,
    next_adapt: usize,
    rounds: usize,
    cancelled: bool,
    /// prefilling beside decoding lanes: the suffix and the rows of it done
    part: Option<(Vec<u32>, usize)>,
    stats: Stats,
    /// at admission: the engine's residency counters (the request line), and its lane's routing
    /// counts at admission and at the prefill's end (Engine::lane_miss: DS's prune_miss_request)
    res0: crate::adapt::ResStats,
    lane0: [f64; 4],
    lane1: Option<[f64; 4]>,
}

/// A phase's routing counts, DS's Generation::miss order: [missed, picks, missed score, score].
fn phase(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [b[1] - a[1], b[0] - a[0], b[3] - a[3], b[2] - a[2]]
}

/// A lane's block in a round, between its drafting and its acceptance.
struct Block {
    /// index into the round's runs
    i: usize,
    p: usize,
    /// no room to draft: one row, the anchor (plain_step)
    plain: bool,
    copy: bool,
    long: bool,
    dmax: usize,
    drafts: Vec<u32>,
    qs: Vec<Option<Dist>>,
    pending: Vec<(usize, f64)>,
    pks: Vec<f64>,
    conf: f64,
    /// still drafting: its row in the last MTP pass
    row: Option<usize>,
}

/// A request that ended: its id and the server's line for it.
pub struct Finished {
    pub id: u64,
    pub line: String,
}

pub struct Server {
    pub eng: Engine,
    pub opt: ServeOptions,
    pub cal: Option<CalGate>,
    pub stats: Stats,
    /// the disk prefix store's index (rank 0), when on
    pub store: Option<crate::prefix::Store>,
    /// the DS engine's routing history beside GLM's (residency::DsHistory), when on
    pub ds_history: Option<crate::residency::DsHistory>,
    /// each lane's candidate temperature as last set
    sampling: Vec<Option<f32>>,
    pub log: Box<dyn Fn(&str) + Send>,
    /// TIDELM_GLM_ROUND_LOG: a JSON line a request (its prompt and tokens) and a round (drafts,
    /// their probabilities, the target's tokens, accepted) for offline gate studies (rank 0)
    round_log: Option<std::fs::File>,
    runs: Vec<Option<Run>>,
    queue: VecDeque<Job>,
    /// lanes by last use, oldest first: a prompt that matches none better takes the oldest
    lru: Vec<usize>,
    done: Vec<Finished>,
}

impl Server {
    pub fn new(eng: Engine, opt: ServeOptions, log: Box<dyn Fn(&str) + Send>) -> Self {
        let cal = opt.cal.map(|t| CalGate::new(t, true));
        let round_log = std::env::var("TIDELM_GLM_ROUND_LOG").ok()
            .and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok());
        let lanes = eng.lanes();
        Server { eng, opt, cal, stats: Stats::default(), store: None, ds_history: None, sampling: vec![None; lanes], log, round_log,
                 runs: (0..lanes).map(|_| None).collect(), queue: VecDeque::new(), lru: (0..lanes).collect(), done: vec![] }
    }

    fn log_json(&mut self, v: serde_json::Value) {
        if let Some(f) = self.round_log.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, "{v}");
        }
    }

    /// Row `row` of the last pass as a distribution (sampled requests).
    fn dist(&mut self, row: usize, temp: f32, top_p: f32, st: &mut Stats) -> Result<Dist> {
        if let Some(d) = nucleus(&self.eng.candidates(row), temp, top_p) {
            return Ok(d);
        }
        self.stats.fallbacks += 1;
        st.fallbacks += 1;
        let mut z = self.eng.full_row(row)?;
        // NaN logits once panicked the sampler's sort and stopped the pair (2026-10-29, a request
        // past 162K; cause not found): logged, taken as -inf, the row sampled from what is left
        let nan = z.iter().filter(|v| v.is_nan()).count();
        if nan > 0 {
            eprintln!("warning: {nan} NaN logits in row {row} of the last pass (lane {}, position {}); sampled as -inf",
                      self.eng.lane, self.eng.pos);
            z.iter_mut().filter(|v| v.is_nan()).for_each(|v| *v = f32::NEG_INFINITY);
        }
        let d = dist_from_logits(&z, temp, top_p);
        if d.is_empty() {
            return Err(format!("row {row}: no finite logits to sample from").into());
        }
        Ok(d)
    }

    /// Row `row`'s chain probability for a greedy draft: max softmax probability (T = 1).
    fn greedy_p(&self, row: usize) -> f64 {
        let rc = self.eng.candidates(row);
        let m = rc.lse.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = rc.lse.iter().map(|&(mr, s)| s as f64 * ((mr - m) as f64).exp()).sum();
        1.0 / z
    }

    // ------------------------------------------------------------------ the scheduler
    pub fn submit(&mut self, job: Job) {
        self.queue.push_back(job);
    }

    /// A queued request ends now; a running one at its lane's next round.
    pub fn cancel(&mut self, id: u64) {
        if let Some(i) = self.queue.iter().position(|j| j.id == id) {
            let mut j = self.queue.remove(i).unwrap();
            (j.sink)(Event::Done(Finish::Cancelled, Generation::default()));
            return;
        }
        for r in self.runs.iter_mut().flatten() {
            if r.job.id == id {
                r.cancelled = true;
            }
        }
    }

    /// Requests queued or in lanes.
    pub fn busy(&self) -> bool {
        !self.queue.is_empty() || self.runs.iter().any(|r| r.is_some())
    }

    /// One request alone, its tokens to `job.sink` as they are decided (the warm-up, benches).
    /// `cancelled` is polled every round.
    pub fn run(&mut self, job: Job, cancelled: &mut dyn FnMut() -> bool) -> Result<()> {
        let id = job.id;
        self.submit(job);
        while self.busy() {
            if cancelled() {
                self.cancel(id);
            }
            self.step()?;
        }
        self.done.clear();
        Ok(())
    }

    /// One turn: queued requests into free lanes (a prompt beside decoding lanes waits for its
    /// prefill parts), one round of every decoding lane, then one part of a prompt prefilling
    /// beside them. Returns the requests that ended. With one request at a time, the engine
    /// calls are generate's for it alone, in its order.
    pub fn step(&mut self) -> Result<Vec<Finished>> {
        self.admit()?;
        let decoding: Vec<usize> = (0..self.runs.len()).filter(|&l| self.runs[l].as_ref().is_some_and(|r| r.part.is_none())).collect();
        if !decoding.is_empty() {
            self.round(&decoding)?;
        }
        if let Some(l) = (0..self.runs.len()).find(|&l| self.runs[l].as_ref().is_some_and(|r| r.part.is_some())) {
            self.prefill_part(l)?;
        }
        Ok(std::mem::take(&mut self.done))
    }

    fn touch(&mut self, lane: usize) {
        self.lru.retain(|&l| l != lane);
        self.lru.push(lane);
    }

    /// Whether `lane` can grow to `end` positions, idle lanes' chunks given back first, oldest
    /// first (their sequences then start over).
    fn make_room(&mut self, lane: usize, end: usize) -> Result<bool> {
        if self.eng.kv_room(lane, end) {
            return Ok(true);
        }
        for l in self.lru.clone() {
            if l != lane && self.runs[l].is_none() && self.eng.kv_held(l) > 0 {
                self.eng.kv_shrink(l, 0)?;
                if self.eng.kv_room(lane, end) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Queued requests into free lanes while the pool can take them: each to the free lane
    /// whose sequence shares most of its prompt (ties: the lane used longest ago).
    fn admit(&mut self) -> Result<()> {
        while let Some(job) = self.queue.front() {
            let free: Vec<usize> = self.lru.iter().copied().filter(|&l| self.runs[l].is_none()).collect();
            if free.is_empty() {
                return Ok(());
            }
            let prompt = &job.prompt;
            let shared = |l: usize| prompt.iter().zip(self.eng.lane_tokens(l)).take_while(|(a, b)| a == b).count();
            let lane = free.iter().copied().fold(None, |best: Option<(usize, usize)>, l| {
                let c = shared(l);
                if best.is_none_or(|b| c > b.1) { Some((l, c)) } else { best }
            }).unwrap().0;
            let max_seq = self.eng.opt.max_seq;
            let others = self.runs.iter().any(|r| r.is_some());
            let want = if others { (prompt.len() + KV_HEADROOM).min(max_seq) } else { prompt.len().min(max_seq) };
            if !prompt.is_empty() && prompt.len() < max_seq && !self.make_room(lane, want)? {
                if others {
                    return Ok(()); // waits for a lane to end
                }
                return Err(format!("kv pool: {} positions do not fit one lane's pool", want).into());
            }
            let job = self.queue.pop_front().unwrap();
            self.start(lane, job)?;
        }
        Ok(())
    }

    /// generate's start for a request in `lane`: the prompt's reuse of the lane's sequence or the
    /// disk store, then its prefill (whole, or in parts beside decoding lanes), the first anchor.
    fn start(&mut self, lane: usize, mut job: Job) -> Result<()> {
        let g = Generation { admitted: Some(Instant::now()), ..Default::default() };
        self.log_json(serde_json::json!({"prompt": job.prompt}));
        let prompt = job.prompt.clone();
        let max_seq = self.eng.opt.max_seq;
        let err = if prompt.is_empty() {
            Some("empty prompt".to_string())
        } else if prompt.len() >= max_seq {
            Some(format!("prompt of {} tokens > {max_seq}", prompt.len()))
        } else {
            None
        };
        if let Some(e) = err {
            self.log_json(serde_json::json!({"tokens": g.tokens}));
            (job.sink)(Event::Done(Finish::Error(e), g));
            return Ok(());
        }
        self.touch(lane);
        self.eng.select(lane)?;
        let greedy = job.sampling.is_greedy();
        let (temp, top_p) = (job.sampling.temperature, job.sampling.top_p);
        // greedy requests too read candidates: the gate's chain probability at T = 1
        let want = if greedy { 1.0 } else { temp };
        if self.sampling[lane] != Some(want) {
            self.eng.set_sampling(Some(want))?;
            self.sampling[lane] = Some(want);
        }
        let depth = if self.eng.cfg.mtp { self.opt.depth } else { 0 };
        // the lane's sequence's prefix, one token back so the reprefilled row refills its MTP row
        let mut common = prompt.iter().zip(&self.eng.tokens).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        // the disk store's deepest saved prefix, when it gains enough over the lane's
        if let Some(store) = self.store.as_mut() {
            if let Some((chain, end)) = store.plan_restore(&prompt, common) {
                let c: Vec<(u64, usize, usize, String)> = chain.iter().map(|s| (s.id, s.start, s.end, s.hash.clone())).collect();
                let ids: Vec<u64> = chain.iter().map(|s| s.id).collect();
                let bad = chain.last().unwrap().id;
                let t0 = Instant::now();
                if !self.eng.kv_room(lane, end) {
                    // the pool cannot take the rows now: prefill them
                } else if self.eng.disk_load(&c, &prompt, end)? {
                    let store = self.store.as_mut().unwrap();
                    store.touch(&ids);
                    store.save_index()?;
                    (self.log)(&format!("[prefix] restored [{common}, {end}) from {} segment(s) in {:.2} s", ids.len(),
                                        t0.elapsed().as_secs_f64()));
                    common = end;
                } else {
                    // a segment some rank could not read: forget it (and what extends it)
                    let store = self.store.as_mut().unwrap();
                    let doomed: Vec<u64> = store.index.segs.values().filter(|s| store.chain(s.id).iter().any(|x| x.id == bad))
                        .map(|s| s.id).collect();
                    store.drop_segments(&doomed);
                    store.save_index()?;
                    (self.log)("[prefix] restore miss: segments dropped");
                }
            }
        }
        let keep = common.saturating_sub(1);
        self.eng.set_pos(keep)?;
        let mut g = g;
        g.cached = keep;
        let suffix = prompt[keep..].to_vec();
        self.eng.take_lane_demand(lane)?; // the lane's demand restarts with this request
        let mut run = Run {
            job, g, lane, rng: Pcg64::from_seed(0), greedy, temp, top_p, depth, anchor: 0, z_anchor: None, anchor_cands: None,
            lookup: None, copier: None, copied: 0, synced: 0, out: vec![], next_adapt: self.opt.adapt_every, rounds: 0,
            cancelled: false, part: None, stats: Stats::default(), res0: self.eng.res.stats.clone(), lane0: self.eng.lane_miss(lane)?,
            lane1: None,
        };
        run.rng = Pcg64::from_seed(run.job.seed as u128);
        let beside = self.runs.iter().any(|r| r.is_some());
        if beside && suffix.len() > self.opt.lane_chunk.max(self.opt.two_pass) {
            run.part = Some((suffix, 0));
            self.runs[lane] = Some(run);
            return Ok(());
        }
        let opt = self.opt.clone();
        let m0 = self.eng.miss_totals()?;
        let mut first = self.eng.forward(&suffix, depth > 0)?;
        let mut adapted = false;
        if opt.adapt && suffix.len() >= 4 && suffix.len() <= opt.two_pass {
            let m1 = self.eng.miss_totals()?;
            let mass = m1[2] - m0[2];
            let miss = if mass > 0.0 { (m1[3] - m0[3]) / mass } else { 0.0 };
            if miss >= opt.two_pass_miss {
                self.eng.adapt(opt.adapt_prefill)?;
                self.eng.set_pos(keep)?;
                first = self.eng.forward(&suffix, depth > 0)?;
                adapted = true;
                self.stats.two_pass += 1;
                run.stats.two_pass += 1;
            }
        }
        self.eng.res.demand_tokens += suffix.len() as u64;
        if opt.adapt && !adapted {
            self.eng.adapt(opt.adapt_prefill)?;
        }
        self.begin_decode(run, &first)
    }

    /// The next part of a prompt prefilling beside decoding lanes (its last through forward,
    /// which ends its prefill as start's would), when the pool can take it.
    fn prefill_part(&mut self, lane: usize) -> Result<()> {
        if self.runs[lane].as_ref().unwrap().cancelled {
            // its rows so far stay the lane's prefix; a prompt not all there is not saved
            let run = self.runs[lane].take().unwrap();
            return self.finish(run, Finish::Cancelled);
        }
        let (suffix, c0) = self.runs[lane].as_ref().unwrap().part.clone().unwrap();
        let depth = self.runs[lane].as_ref().unwrap().depth;
        let n = self.opt.lane_chunk.max(HEAD_ROWS + 1).min(suffix.len() - c0);
        let end = self.eng.lane_pos(lane) + n;
        if !self.make_room(lane, end)? {
            return Ok(()); // waits for the decoding lanes to give chunks back
        }
        self.eng.select(lane)?;
        if c0 + n < suffix.len() && n > HEAD_ROWS {
            self.eng.forward_part(&suffix[c0..c0 + n], depth > 0, suffix[c0 + n])?;
            self.runs[lane].as_mut().unwrap().part = Some((suffix, c0 + n));
            return Ok(());
        }
        let first = self.eng.forward(&suffix[c0..], depth > 0)?;
        self.eng.res.demand_tokens += suffix.len() as u64;
        if self.opt.adapt {
            self.eng.adapt(self.opt.adapt_prefill)?;
        }
        let mut run = self.runs[lane].take().unwrap();
        run.part = None;
        self.begin_decode(run, &first)
    }

    /// After a prefill (the selected lane's, its argmaxes `first`): the first anchor, the kept
    /// hidden, the drafters; the run decodes from the next round.
    fn begin_decode(&mut self, mut run: Run, first: &[u32]) -> Result<()> {
        let last = self.eng.last_rows - 1;
        // a forward returns every row's argmax up to HEAD_ROWS: the suffix's next token is its
        // last row's (a prefill chunk's head ran on that row alone: row 0)
        let arow = first.len() - 1;
        run.anchor = if run.greedy {
            first[arow]
        } else {
            let d = self.dist(arow, run.temp, run.top_p, &mut run.stats)?;
            draw(&d, &mut run.rng)
        };
        if run.greedy && self.opt.loop_break > 0 {
            run.anchor_cands = Some(self.eng.candidates(arow));
        }
        self.eng.keep_hidden(last)?;
        // the prompt's routing: its counts, and its vote in the DS history
        run.lane1 = Some(self.eng.lane_miss(run.lane)?);
        let d = self.eng.take_lane_demand(run.lane)?;
        if let Some(h) = self.ds_history.as_mut() {
            h.fold(&d)?;
        }
        let prompt = &run.job.prompt;
        run.lookup = (self.opt.lookup > 0 && run.depth > 0).then(|| Lookup::new(prompt, self.opt.lookup));
        run.copier = self.opt.copy.clone().filter(|_| run.job.copy_draft && run.depth > 0).map(|c| CopyDraft::new(prompt, c, None));
        let lane = run.lane;
        self.runs[lane] = Some(run);
        Ok(())
    }

    /// A request's end: its Done event, its prompt's rows to the disk store, its line.
    fn finish(&mut self, mut run: Run, finish: Finish) -> Result<()> {
        self.log_json(serde_json::json!({"tokens": run.g.tokens}));
        let ok = !matches!(finish, Finish::Error(_));
        // the request's own routing by phase (its rows alone, beside another lane too), DS's
        // prune_miss_request: a miss a pick that was not routable; the answer's vote in the history
        let l2 = self.eng.lane_miss(run.lane)?;
        let l1 = run.lane1.unwrap_or(l2);
        let (p, d) = (phase(run.lane0, l1), phase(l1, l2));
        run.g.miss = [p[0], p[1], p[2], p[3], d[0], d[1], d[2], d[3]];
        let dem = self.eng.take_lane_demand(run.lane)?;
        if let Some(h) = self.ds_history.as_mut() {
            h.fold(&dem)?;
        }
        let pct = |a: f64, b: f64| if b > 0.0 { a / b * 100.0 } else { 0.0 };
        let s = &run.stats;
        let rounds = s.rounds.max(1) as f64;
        let r1 = &self.eng.res.stats;
        let r0 = &run.res0;
        let beside = if s.shared_rounds > 0 {
            format!(" ({} of {} rounds beside another lane; hot reads and swaps are the lanes')", s.shared_rounds, s.rounds)
        } else {
            String::new()
        };
        let line = format!("request {}, GLM: {:.2} drafted / {:.2} accepted a round, copy rounds {} ({} long, {} accepted), \
                            miss rate {:.1}% (prompt {:.1}%, decode {:.1}%; by score {:.1}%), hot reads {} (kept {}, {:.2} s waited), \
                            swaps {} ({:.2} s waited), fallbacks {}{beside}",
                           run.job.id, s.drafted as f64 / rounds, s.accepted as f64 / rounds, s.copy_rounds, s.long_copies,
                           s.copy_accepted, pct(p[0] + d[0], p[1] + d[1]), pct(p[0], p[1]), pct(d[0], d[1]),
                           pct(p[2] + d[2], p[3] + d[3]), r1.hot_reads - r0.hot_reads, r1.hot_kept - r0.hot_kept,
                           r1.hot_wait_s - r0.hot_wait_s, r1.swaps - r0.swaps, r1.adapt_wait_s - r0.adapt_wait_s, s.fallbacks);
        let (id, lane, prompt) = (run.job.id, run.lane, run.job.prompt.clone());
        let mut job = run.job;
        (job.sink)(Event::Done(finish, run.g));
        if ok {
            self.eng.select(lane)?;
            self.disk_save(&prompt)?;
        }
        self.touch(lane);
        self.done.push(Finished { id, line });
        Ok(())
    }

    /// The prompt's rows past its deepest saved prefix to the disk store (both ranks, the
    /// selected lane's), when they are new enough to keep and the disks have room.
    fn disk_save(&mut self, prompt: &[u32]) -> Result<()> {
        let Some(store) = self.store.as_mut() else { return Ok(()) };
        if self.eng.tokens.len() < prompt.len() || self.eng.tokens[..prompt.len()] != *prompt {
            return Ok(());
        }
        if crate::prefix::free_bytes(&store.dir) < 10_000_000_000 {
            return Ok(());
        }
        let bpp = self.eng.bytes_per_position();
        let Some((seg, evict)) = store.plan_save(prompt, bpp) else { return Ok(()) };
        let t0 = Instant::now();
        if self.eng.disk_save(seg.id, seg.start, seg.end, &seg.hash, &evict)? {
            store.drop_segments(&evict);
            store.write_tokens(seg.id, &prompt[seg.start..seg.end])?;
            let (s, e) = (seg.start, seg.end);
            store.commit(seg)?;
            (self.log)(&format!("[prefix] saved [{s}, {e}) in {:.2} s", t0.elapsed().as_secs_f64()));
        }
        Ok(())
    }

    /// One round of every decoding lane in `lanes`: generate's loop body, each phase for all of
    /// them before the next - the anchors out, the drafts (batched MTP passes), one verify pass,
    /// acceptance, the MTP fill - so the lanes share each pass.
    #[allow(clippy::too_many_lines)]
    fn round(&mut self, lanes: &[usize]) -> Result<()> {
        let opt = self.opt.clone();
        let max_seq = self.eng.opt.max_seq;
        let mut runs: Vec<Run> = lanes.iter().map(|&l| self.runs[l].take().unwrap()).collect();
        // cancels, loop breaks, adaptation, the round count; the background period once a round
        let mut keep: Vec<Run> = vec![];
        let mut bg = false;
        for mut r in runs.drain(..) {
            if r.cancelled {
                self.finish(r, Finish::Cancelled)?;
                continue;
            }
            if opt.loop_break > 0 && {
                let mut s = r.out.clone();
                s.push(r.anchor);
                tail_cycle(&s, opt.loop_break)
            } {
                if r.greedy {
                    // the anchor row's best other token
                    if let Some(rc) = r.anchor_cands.as_ref() {
                        let best = rc.vals.iter().zip(&rc.ids).filter(|(_, &i)| i != r.anchor)
                            .max_by(|a, b| a.0.total_cmp(b.0).then(b.1.cmp(a.1))).map(|(_, &i)| i);
                        if let Some(b) = best {
                            r.anchor = b;
                            self.stats.loop_breaks += 1;
                            r.stats.loop_breaks += 1;
                        }
                    }
                } else if let Some(d) = r.z_anchor.take() {
                    let d: Dist = d.into_iter().filter(|x| x.0 != r.anchor).collect();
                    if !d.is_empty() {
                        r.anchor = draw(&d, &mut r.rng);
                        self.stats.loop_breaks += 1;
                        r.stats.loop_breaks += 1;
                    }
                }
            }
            if opt.adapt && opt.adapt_every > 0 && r.out.len() >= r.next_adapt {
                // the newcomers load while the pair decodes (in line this stopped the GPU ~166 ms)
                self.eng.adapt_background(opt.adapt_decode)?;
                r.next_adapt += opt.adapt_every;
            }
            r.rounds += 1;
            if opt.adapt && opt.bg_every > 0 && r.rounds % opt.bg_every == 0 && self.eng.hot.is_some() {
                bg = true;
            }
            keep.push(r);
        }
        if bg {
            self.eng.bg_step(opt.hot_keep)?;
        }
        // the anchors out; a lane at a stop token or its length ends, the cache holding its anchor
        // (the next turn's prompt matches through it)
        for mut r in keep.drain(..) {
            (r.job.sink)(Event::Tokens(vec![r.anchor]));
            r.out.push(r.anchor);
            r.g.tokens.push(r.anchor);
            if r.job.stop.contains(&r.anchor) || r.out.len() >= r.job.max_tokens {
                self.eng.select(r.lane)?;
                if self.eng.pos < max_seq && self.make_room(r.lane, self.eng.pos + 1)? {
                    self.eng.forward(&[r.anchor], false)?;
                }
                let f = if r.job.stop.contains(&r.anchor) { Finish::Stop } else { Finish::Length };
                self.finish(r, f)?;
                continue;
            }
            runs.push(r);
        }
        if runs.is_empty() {
            return Ok(());
        }
        // each lane's block: a plain step (no room to draft), a copy (a long copy of the prompt,
        // else the lookup's), or MTP drafts under the gate. Beside other lanes a block keeps to
        // its share of HEAD_ROWS; a lane the pool cannot grow ends at its length.
        let share = if runs.len() > 1 { HEAD_ROWS / runs.len() } else { HEAD_ROWS };
        let mut blocks: Vec<Block> = vec![];
        let mut ended: Vec<usize> = vec![];
        for (i, r) in runs.iter_mut().enumerate() {
            let p = self.eng.lane_pos(r.lane);
            let room = max_seq.saturating_sub(p + 1).min(r.job.max_tokens - r.out.len());
            let dmax = r.depth.min(room);
            let mut b = Block { i, p, plain: dmax == 0, copy: false, long: false, dmax, drafts: vec![], qs: vec![], pending: vec![],
                                pks: vec![], conf: 1.0, row: None };
            if !b.plain {
                if let Some(cd) = r.copier.as_mut() {
                    cd.push(&r.out[r.copied..]);
                    r.copied = r.out.len();
                    let mut c = cd.propose();
                    c.truncate(room.min(share - 1));
                    if !c.is_empty() {
                        b.drafts = c;
                        b.copy = true;
                        b.long = true;
                    }
                }
                if let Some(lk) = r.lookup.as_mut() {
                    lk.extend(&r.out[r.synced..]);
                    r.synced = r.out.len();
                    if !b.copy {
                        if let Some(c) = lk.propose(dmax) {
                            b.drafts = c;
                            b.copy = true;
                        }
                    }
                }
                if b.copy {
                    b.qs = vec![None; b.drafts.len()];
                }
            }
            let reach = p + 1 + if b.copy { b.drafts.len() } else { dmax };
            if !self.make_room(r.lane, reach)? {
                (self.log)(&format!("[kv] lane {}: the pool cannot grow it past {p} positions beside the other lanes; ended", r.lane));
                ended.push(i);
                continue;
            }
            blocks.push(b);
        }
        for &i in ended.iter().rev() {
            let r = runs.remove(i);
            self.finish(r, Finish::Length)?;
            for b in blocks.iter_mut() {
                if b.i > i {
                    b.i -= 1;
                }
            }
        }
        if runs.is_empty() {
            return Ok(());
        }
        // drafting: the first MTP pass carries every non-plain lane's anchor (a copy lane's for its
        // cache row alone), the next ones the lanes whose chain goes on
        let mtp_lanes: Vec<usize> = (0..blocks.len()).filter(|&k| !blocks[k].plain).collect();
        if !mtp_lanes.is_empty() {
            let logits = mtp_lanes.iter().any(|&k| !blocks[k].copy);
            let anchors: Vec<[u32; 1]> = mtp_lanes.iter().map(|&k| [runs[blocks[k].i].anchor]).collect();
            let groups: Vec<MtpRows> = mtp_lanes.iter().zip(&anchors).map(|(&k, a)| {
                MtpRows { lane: runs[blocks[k].i].lane, ids: a, hidden: Hidden::Last, pos0: blocks[k].p - 1 }
            }).collect();
            let mut toks = self.eng.mtp_lanes(&groups, logits)?;
            for (row, &k) in mtp_lanes.iter().enumerate() {
                if !blocks[k].copy {
                    blocks[k].row = Some(row);
                }
            }
            let mut step = 0usize;
            while let Some(t) = toks.take() {
                let mut next: Vec<(usize, u32, usize)> = vec![]; // (block, draft, its row)
                for k in 0..blocks.len() {
                    let Some(row) = blocks[k].row.take() else { continue };
                    let r = &mut runs[blocks[k].i];
                    let (d, q, pk) = if r.greedy {
                        (t[row], None, self.greedy_p(row))
                    } else {
                        let (temp, top_p) = (r.temp, r.top_p);
                        let mut st = std::mem::take(&mut r.stats);
                        let q = self.dist(row, temp, top_p, &mut st);
                        let r = &mut runs[blocks[k].i];
                        r.stats = st;
                        let q = q?;
                        let d = draw(&q, &mut r.rng);
                        let pk = prob_of(&q, d) as f64;
                        (d, Some(q), pk)
                    };
                    let b = &mut blocks[k];
                    b.conf *= pk;
                    let keep_it = match self.cal.as_mut() {
                        Some(c) => c.keep(step, b.conf),
                        None => b.conf >= opt.dyn_gate,
                    };
                    if !keep_it {
                        continue;
                    }
                    b.pending.push((step, b.conf));
                    b.pks.push(pk);
                    b.drafts.push(d);
                    b.qs.push(q);
                    if step + 1 < b.dmax {
                        next.push((k, d, row));
                    }
                }
                if next.is_empty() {
                    break;
                }
                step += 1;
                let ids: Vec<[u32; 1]> = next.iter().map(|x| [x.1]).collect();
                let groups: Vec<MtpRows> = next.iter().zip(&ids).map(|(&(k, _, row), id)| {
                    MtpRows { lane: runs[blocks[k].i].lane, ids: id, hidden: Hidden::Mtp(row), pos0: blocks[k].p + step - 1 }
                }).collect();
                toks = self.eng.mtp_lanes(&groups, true)?;
                for (row, &(k, _, _)) in next.iter().enumerate() {
                    blocks[k].row = Some(row);
                }
            }
        }
        // one verify pass: every lane's anchor and drafts (one lane: forward, whose path by width
        // and context is generate's)
        let verify: Vec<Vec<u32>> = blocks.iter().map(|b| {
            let mut v = vec![runs[b.i].anchor];
            v.extend_from_slice(&b.drafts);
            v
        }).collect();
        let ams: Vec<Vec<u32>> = if blocks.len() == 1 {
            self.eng.select(runs[blocks[0].i].lane)?;
            vec![self.eng.forward(&verify[0], false)?]
        } else {
            let bl: Vec<(usize, &[u32])> = blocks.iter().zip(&verify).map(|(b, v)| (runs[b.i].lane, v.as_slice())).collect();
            self.eng.forward_lanes(&bl)?
        };
        let shared = blocks.len() > 1;
        // acceptance, every lane's before the next pass (it reads the verify's candidates)
        let mut row0 = 0usize;
        let mut acc: Vec<(usize, usize, u32)> = vec![]; // (row0, accepted, next anchor)
        for (k, b) in blocks.iter().enumerate() {
            let am = &ams[k];
            let r = &mut runs[b.i];
            let (greedy, temp, top_p) = (r.greedy, r.temp, r.top_p);
            r.g.steps += 1;
            if shared {
                r.stats.shared_rounds += 1;
            }
            if b.plain {
                let nxt = if greedy {
                    am[0]
                } else {
                    let mut st = std::mem::take(&mut r.stats);
                    let d = self.dist(row0, temp, top_p, &mut st);
                    let r = &mut runs[b.i];
                    r.stats = st;
                    draw(&d?, &mut r.rng)
                };
                acc.push((row0, 0, nxt));
                row0 += 1;
                continue;
            }
            if self.round_log.is_some() {
                let v = serde_json::json!({"pos": b.p, "anchor": r.anchor, "copy": b.copy, "drafts": b.drafts, "probs": b.pks, "target": am});
                self.log_json(v);
            }
            let nd = b.drafts.len();
            let (mut n_acc, mut nxt) = (nd, None);
            for kk in 0..nd {
                if greedy {
                    if am[kk] != b.drafts[kk] {
                        n_acc = kk;
                        nxt = Some(am[kk]);
                        break;
                    }
                } else {
                    let mut st = std::mem::take(&mut runs[b.i].stats);
                    let pk = self.dist(row0 + kk, temp, top_p, &mut st);
                    let r = &mut runs[b.i];
                    r.stats = st;
                    let pk = pk?;
                    let dk = b.drafts[kk];
                    let qd = b.qs[kk].as_ref().map(|q| prob_of(q, dk)).unwrap_or(1.0).max(1e-20);
                    let ratio = prob_of(&pk, dk) / qd;
                    if unif(&mut r.rng) >= ratio.min(1.0) as f64 {
                        // the residual (pk - q)+, or pk without the copied token
                        let resid: Dist = match b.qs[kk].as_ref() {
                            Some(q) => pk.iter().map(|&(i, p)| (i, (p - prob_of(q, i)).max(0.0))).filter(|x| x.1 > 0.0).collect(),
                            None => pk.iter().filter(|x| x.0 != dk).cloned().collect(),
                        };
                        let rs = if resid.iter().map(|x| x.1).sum::<f32>() > 0.0 { resid } else { pk.clone() };
                        n_acc = kk;
                        nxt = Some(draw(&rs, &mut r.rng));
                        r.z_anchor = Some(pk);
                        break;
                    }
                }
            }
            let nxt = match nxt {
                Some(t) => t,
                None if greedy => am[nd],
                None => {
                    let mut st = std::mem::take(&mut runs[b.i].stats);
                    let d = self.dist(row0 + nd, temp, top_p, &mut st);
                    let r = &mut runs[b.i];
                    r.stats = st;
                    let d = d?;
                    let t = draw(&d, &mut r.rng);
                    r.z_anchor = Some(d);
                    t
                }
            };
            let r = &mut runs[b.i];
            if greedy {
                r.z_anchor = None;
            }
            if !b.copy {
                if let Some(c) = self.cal.as_mut() {
                    for &(kk, cf) in &b.pending {
                        c.observe(kk, cf, kk < n_acc);
                    }
                }
            }
            for st in [&mut self.stats, &mut r.stats] {
                st.rounds += 1;
                st.drafted += nd;
                st.accepted += n_acc;
                if b.long {
                    st.long_copies += 1;
                }
                if b.copy {
                    st.copy_rounds += 1;
                    st.copy_accepted += n_acc;
                }
            }
            r.g.accepted.push(n_acc);
            r.g.depths.push(nd);
            if b.long {
                r.copier.as_mut().unwrap().observe(n_acc);
            }
            if b.copy {
                r.g.copy_steps += 1;
                r.g.copy_accepted += n_acc;
            }
            acc.push((row0, n_acc, nxt));
            row0 += 1 + nd;
        }
        // the lanes' positions past their accepted drafts, the MTP cache refilled for them (one
        // pass; past MAX_ROWS rows a lane at a time), the kept hidden
        for (k, b) in blocks.iter().enumerate() {
            if !b.plain {
                self.eng.select(runs[b.i].lane)?;
                self.eng.set_pos(b.p + 1 + acc[k].1)?;
            }
        }
        let fills: Vec<MtpRows> = blocks.iter().enumerate().filter(|(k, b)| !b.plain && acc[*k].1 > 0).map(|(k, b)| {
            MtpRows { lane: runs[b.i].lane, ids: &b.drafts[..acc[k].1], hidden: Hidden::Target(acc[k].0), pos0: b.p }
        }).collect();
        if fills.iter().map(|f| f.ids.len()).sum::<usize>() <= MAX_ROWS || fills.len() == 1 {
            if !fills.is_empty() {
                self.eng.mtp_lanes(&fills, false)?;
            }
        } else {
            for f in &fills {
                self.eng.mtp_lanes(std::slice::from_ref(f), false)?;
            }
        }
        for (k, b) in blocks.iter().enumerate() {
            let r = &runs[b.i];
            self.eng.select(r.lane)?;
            self.eng.keep_hidden(acc[k].0 + acc[k].1)?;
            if !b.plain {
                self.eng.res.demand_tokens += (b.drafts.len() + 1) as u64;
            }
        }
        if opt.loop_break > 0 {
            for (k, b) in blocks.iter().enumerate() {
                if runs[b.i].greedy {
                    runs[b.i].anchor_cands = Some(self.eng.candidates(acc[k].0 + acc[k].1));
                }
            }
        }
        // the accepted drafts out; a stop or the length ends the request
        let mut back: Vec<Run> = vec![];
        let mut fin: Vec<(Run, Finish)> = vec![];
        let mut taken: Vec<Option<Run>> = runs.into_iter().map(Some).collect();
        for k in 0..blocks.len() {
            let b = &blocks[k];
            let mut r = taken[b.i].take().unwrap();
            let n_acc = acc[k].1;
            if n_acc > 0 {
                (r.job.sink)(Event::Tokens(b.drafts[..n_acc].to_vec()));
            }
            let mut end = None;
            for &d in &b.drafts[..n_acc] {
                r.out.push(d);
                r.g.tokens.push(d);
                if r.job.stop.contains(&d) {
                    end = Some(Finish::Stop);
                    break;
                }
                if r.out.len() >= r.job.max_tokens {
                    end = Some(Finish::Length);
                    break;
                }
            }
            r.anchor = acc[k].2;
            match end {
                Some(f) => fin.push((r, f)),
                None => back.push(r),
            }
        }
        for (r, f) in fin {
            self.finish(r, f)?;
        }
        for r in back {
            let l = r.lane;
            self.runs[l] = Some(r);
        }
        Ok(())
    }

    /// tp.warmup: two greedy generations on synthetic prompts (graphs captured, kernels warm)
    /// and a sampled one, with no adaptation, gate learning or DS-history votes; with lanes, two at once as well
    /// (the shared passes' graphs); their demand forgotten after, every lane emptied.
    pub fn warmup(&mut self) -> Result<()> {
        let (opt, cal, dsh) = (self.opt.clone(), self.cal.take(), self.ds_history.take());
        self.opt.adapt = false;
        let stop = self.eng.cfg.eos.clone();
        let prompts: [Vec<u32>; 3] = [(0..192).map(|i| 1000 + i % 12).collect(), (0..64).map(|i| (i * 7919) % 50000 + 1000).collect(),
                                      (0..64).map(|i| (i * 104729) % 60000 + 2000).collect()];
        let job = |p: Vec<u32>, sampled: bool| {
            let sampling = if sampled { tidelm_engine::forward::Sampling { temperature: 0.6, top_p: 0.95 } }
                           else { tidelm_engine::forward::Sampling::GREEDY };
            Job { id: 0, prompt: p, max_tokens: 48, sampling, seed: 1, stop: stop.clone(), sink: Box::new(|_| {}), images: vec![],
                  key: None, copy_draft: false }
        };
        for (n, p) in prompts.iter().enumerate() {
            self.run(job(p.clone(), n == 2), &mut || false)?;
        }
        if self.eng.lanes() > 1 {
            self.submit(job(prompts[1].clone(), false));
            self.submit(job(prompts[2].clone(), true));
            while self.busy() {
                self.step()?;
            }
            self.done.clear();
        }
        self.opt = opt;
        self.cal = cal;
        self.ds_history = dsh;
        self.stats = Stats::default();
        self.eng.res.stats = Default::default();
        self.eng.zero_demand()?;
        for l in 0..self.eng.lanes() {
            self.eng.kv_shrink(l, 0)?;
        }
        self.eng.select(0)
    }

    pub fn timing_line(&self, t0: Instant) -> String {
        let s = &self.stats;
        format!("{:.1} s: {} rounds, {:.2} drafted / {:.2} accepted a round, copies {} ({:.2}), fallbacks {}",
                t0.elapsed().as_secs_f64(), s.rounds, s.drafted as f64 / s.rounds.max(1) as f64,
                s.accepted as f64 / s.rounds.max(1) as f64, s.copy_rounds,
                s.copy_accepted as f64 / s.copy_rounds.max(1) as f64, s.fallbacks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_proposes_the_longest_settled_match() {
        let h = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 6, 7, 8, 1, 2, 3];
        let mut l = Lookup::new(&h, 3);
        // "1 2 3" ends at 3 and 9; the later one (followed by 4 6 7) has the same match length
        assert_eq!(l.propose(3), Some(vec![4, 6, 7]));
        l.extend(&[4]);
        assert_eq!(l.propose(2), Some(vec![6, 7])); // "2 3 4" newest first
    }

    #[test]
    fn tail_cycles() {
        assert!(tail_cycle(&[5, 1, 1, 1], 3));
        assert!(!tail_cycle(&[5, 1, 1], 3));
        assert!(tail_cycle(&[9, 1, 2, 1, 2, 1, 2], 3));
    }

    #[test]
    fn nucleus_from_two_ranks_matches_the_full_softmax() {
        let mut z = vec![0f32; 40];
        for (i, v) in z.iter_mut().enumerate() {
            *v = ((i * 37 % 41) as f32) * 0.25;
        }
        let half = 20;
        let mut rc = RowCands::default();
        for w in 0..2 {
            let part = &z[w * half..(w + 1) * half];
            let m = part.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = part.iter().map(|v| ((v - m) / 0.7).exp()).sum();
            rc.lse.push((m, s));
            let mut idx: Vec<usize> = (0..half).collect();
            idx.sort_by(|&a, &b| part[b].partial_cmp(&part[a]).unwrap());
            idx.truncate(12);
            rc.min.push(idx.iter().map(|&i| part[i]).fold(f32::INFINITY, f32::min));
            for i in idx {
                rc.vals.push(part[i]);
                rc.ids.push((w * half + i) as u32);
            }
        }
        let got = nucleus(&rc, 0.7, 0.9).expect("covered");
        let want = dist_from_logits(&z, 0.7, 0.9);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-5);
        }
        // the candidates' sums are at T = 0.7: a nucleus reaching past them cannot be proven
        assert!(nucleus(&rc, 0.7, 0.99999).is_none());
    }
}
