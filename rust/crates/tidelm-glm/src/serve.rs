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

/// The prompt as the disk store keys it: each image's placeholder rows carry the image's key
/// (flash_image::image_key) instead, as ids no vocabulary reaches (2^31 | 31 bits of the key mixed
/// with the row), so a segment's hash tells one image from another; text rows are the prompt's.
/// Before this, every image shared the one placeholder id, so prompts with images were never saved
/// and a restore stopped at the first image: an agent's lane evicted by two short chats re-prefilled
/// its 110K-token, 20-image prompt (about 3 minutes, the other lane at 5-7 tok/s meanwhile;
/// 2026-11-16).
fn store_tokens(prompt: &[u32], images: &[tidelm_engine::vision::ImageIn], keys: &[u128]) -> Vec<u32> {
    let mut out = prompt.to_vec();
    for (im, &k) in images.iter().zip(keys) {
        let n = crate::flash_image::image_rows(im).min(out.len().saturating_sub(im.start));
        for j in 0..n {
            let h = (k as u64 ^ (k >> 64) as u64).wrapping_add((j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let h = (h ^ (h >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            out[im.start + j] = 0x8000_0000 | (h >> 33) as u32;
        }
    }
    out
}

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
    /// GLM_PRELOAD: session-demand preloading. A request that continues a conversation (a
    /// prefix-cache hit with a fresh suffix) loads its conversation's already-voiced demand
    /// before that suffix prefills: the loads the prefill->decode boundary pass would pay for
    /// after the prefill, paid before it, so the suffix computes on the real experts and the
    /// boundary pass plans less. The cap on experts a request may preload; 0 off.
    pub preload: usize,
    /// GLM_LANE_CHUNK: a prompt admitted while other lanes decode prefills in parts of this many
    /// rows, a round of theirs between parts (a part alone would hold them up seconds a chunk)
    pub lane_chunk: usize,
    /// the share of the engine's time a prefill takes beside decoding lanes (GLM_PREFILL_SHARE,
    /// 0.5: a part, then as long in decode rounds; 1: a round a part, the old turn)
    pub prefill_share: f64,
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
    /// rows of a pass with no finite logit (a NaN hidden state): the request ends as an error
    pub nan_rows: usize,
    pub loop_breaks: usize,
    pub two_pass: usize,
    /// rounds that verified beside another lane's rows (a request's)
    pub shared_rounds: usize,
    /// MTP rounds drafted over the listed vocabulary (of `mtp_rounds`; the rest scored every id)
    pub listed_rounds: usize,
    pub mtp_rounds: usize,
}

/// GLM-5.3-Flash's listed draft vocabulary: a lane drafts over it while the share of its recent
/// tokens outside the list (an average over ~64 tokens, started from its prompt's last 256) is at
/// most this. An unlisted token cannot be drafted and ends its round's chain, so past a few percent
/// the smaller head costs more rounds than it saves milliseconds (break-even ~7% for prose, ~4.5% for
/// code, from the README runs' rounds); text in a language the list leaves out is 20-70% unlisted and
/// goes back to the whole head within a few tokens.
const DRAFT_UNLISTED_MAX: f32 = 0.04;
const DRAFT_UNLISTED_EMA: f32 = 1.0 / 64.0;

/// Positions a lane must be able to grow past its prompt before it is admitted beside another
/// (the lanes share one KV pool: tidelm_engine::kv).
const KV_HEADROOM: usize = 4096;

/// The free lane a prompt goes to, of `(lane, reusable, held)` in least-recently-used order: the one
/// with the largest 3 x reusable - held (ties: the first, the lane used longest ago), with its
/// reusable count.
///
/// A prompt's prefill costs its length less what it reuses, and taking a lane destroys what the lane
/// holds past that: counted at half (the conversation there may not come back), the cost is
/// len - reusable + (held - reusable) / 2. Until 2026-11-13 the lane sharing the most tokens won,
/// whatever it held and whether those tokens could be kept. Every chat prompt shares the template's
/// first tokens with every lane, so an unrelated request (a client's title or follow-up-suggestion
/// request after a chat turn, a second chat) took the lane of a 100K-token conversation rather than
/// the empty lane beside it; on GLM-5.3-Flash, whose KDA state goes back only to a snapshot or to 0,
/// it kept nothing of it, and the conversation's next turn prefilled from 0 (132-155 s at 100K).
pub fn pick_lane(cand: &[(usize, usize, usize)]) -> (usize, usize) {
    let score = |&(_, reuse, held): &(usize, usize, usize)| 3 * reuse as i64 - held as i64;
    let best = cand.iter().fold(None, |best: Option<&(usize, usize, usize)>, c| {
        if best.is_none_or(|b| score(c) > score(b)) { Some(c) } else { best }
    }).expect("pick_lane: no free lane");
    (best.0, best.1)
}

/// Tokens of a queued prompt a running request's must share beyond what a free lane and the
/// disk store hold before the queued one waits for it instead of prefilling them itself
/// (Server::share_running): about 2.5 s of prefill at 150K positions, against the store's save
/// and restore of the rows (0.46 and 0.43 s for 152K of them).
const SHARE_MIN: usize = 1024;

/// Conversations whose cumulative votes GLM_PRELOAD keeps, each an [layers x experts] f64 on
/// the host; the least recently used beyond it are dropped.
const CONV_KEEP: usize = 32;

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
    /// the conversation's key (prefix::conv_key of its opening): its votes accumulate at each
    /// of its requests' ends, its continuations preload from them (GLM_PRELOAD)
    conv: u64,
    /// the prompt's phase vote, stashed for the conversation's accumulation at this request's end
    prompt_vote: Option<Vec<f32>>,
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
    /// the share of the lane's recent tokens the draft head does not list, and how many of `out`
    /// it has seen (DRAFT_UNLISTED_MAX)
    unlisted: f32,
    folded: usize,
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
    /// conversations' cumulative routing votes (GLM_PRELOAD), keyed by prefix::conv_key:
    /// (per-(layer, expert) accumulated shares, last-use sequence for CONV_KEEP's bound)
    conv_votes: HashMap<u64, (Vec<f64>, u64)>,
    conv_seq: u64,
    /// each lane's candidate temperature as last set
    sampling: Vec<Option<f32>>,
    pub log: Box<dyn Fn(&str) + Send>,
    /// TIDELM_GLM_ROUND_LOG: a JSON line a request (its prompt and tokens) and a round (drafts,
    /// their probabilities, the target's tokens, accepted) for offline gate studies (rank 0)
    round_log: Option<std::fs::File>,
    runs: Vec<Option<Run>>,
    queue: VecDeque<Job>,
    /// share_running's wait: the queued request's id and the running one's whose prefill it waits for
    held: Option<(u64, u64)>,
    /// seconds of decode rounds the decoding lanes are owed before the next prefill part (step)
    decode_owed: f64,
    /// share_running's early save, made: the queued request's id and the running one's
    saved_for: Option<(u64, u64)>,
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
        Server { eng, opt, cal, stats: Stats::default(), store: None, ds_history: None, conv_votes: HashMap::new(), conv_seq: 0, sampling: vec![None; lanes], log, round_log,
                 runs: (0..lanes).map(|_| None).collect(), queue: VecDeque::new(), held: None, decode_owed: 0.0, saved_for: None, lru: (0..lanes).collect(), done: vec![] }
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
            // a whole row of NaN (seen 2026-10-29 past 162K and 2026-11-01 four times past
            // 323K, one request in ~10 at that context, cause not found): the request ends with an
            // error (finish), the pair stays up. The row draws a stop token so its lane's round
            // ends; rank 1 follows the mirrored calls as for any token.
            self.stats.nan_rows += 1;
            st.nan_rows += 1;
            let stop = self.eng.cfg.eos.first().copied().unwrap_or(0);
            return Ok(vec![(stop, 1.0)]);
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
            let t0 = Instant::now();
            self.round(&decoding)?;
            self.decode_owed -= t0.elapsed().as_secs_f64();
        } else {
            self.decode_owed = 0.0;
        }
        // decoding lanes and a prefill share the engine by time: after a part, the decoding lanes
        // get as long as it took (GLM_PREFILL_SHARE: the part's share, 0.5 here) before the next.
        // One round a part gave a chat beside an 86K-token prefill 2.4 tok/s (a 512-row part is
        // ~0.85 s at 40-80K; 2026-11-16).
        if decoding.is_empty() || self.decode_owed <= 0.0 {
            if let Some(l) = (0..self.runs.len()).find(|&l| self.runs[l].as_ref().is_some_and(|r| r.part.is_some())) {
                let t0 = Instant::now();
                self.prefill_part(l)?;
                let share = self.opt.prefill_share.clamp(0.05, 1.0);
                self.decode_owed = t0.elapsed().as_secs_f64() * (1.0 - share) / share;
            }
        }
        Ok(std::mem::take(&mut self.done))
    }

    fn touch(&mut self, lane: usize) {
        self.lru.retain(|&l| l != lane);
        self.lru.push(lane);
    }

    /// Whether `lane` can grow to `end` positions with `promised` of the free chunks spoken for,
    /// idle lanes' chunks given back first, oldest first (their sequences then start over).
    fn make_room(&mut self, lane: usize, end: usize, promised: usize) -> Result<bool> {
        let room = |s: &Self| s.eng.kv_need(lane, end) + promised <= s.eng.kv_chunks().0;
        if room(self) {
            return Ok(true);
        }
        for l in self.lru.clone() {
            if l != lane && self.runs[l].is_none() && self.eng.kv_held(l) > 0 {
                self.eng.kv_shrink(l, 0)?;
                if room(self) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Queued requests into free lanes while the pool can take them: each to the free lane
    /// pick_lane chooses (what the prompt can keep of the lane's sequence against what it would
    /// destroy there), unless a running request's prompt shares more of it (share_running).
    fn admit(&mut self) -> Result<()> {
        while let Some(job) = self.queue.front() {
            let free: Vec<usize> = self.lru.iter().copied().filter(|&l| self.runs[l].is_none()).collect();
            if free.is_empty() {
                return Ok(());
            }
            let prompt = &job.prompt;
            let plen = prompt.len();
            // a lane's (reusable, held): the shared prefix as start() will keep it (one row back, and
            // for GLM-5.3-Flash only as far as KDA's state goes back), and the lane's positions
            let cand: Vec<(usize, usize, usize)> = free.iter().map(|&l| {
                let shared = prompt.iter().zip(self.eng.lane_tokens(l)).take_while(|(a, b)| a == b).count();
                let keep = shared.min(plen.saturating_sub(1)).saturating_sub(1);
                (l, self.eng.reusable(l, keep), self.eng.lane_pos(l))
            }).collect();
            let (lane, here) = pick_lane(&cand);
            if !self.share_running(here)? {
                return Ok(()); // waits for a running request's prefill
            }
            let max_seq = self.eng.opt.max_seq;
            let others = self.runs.iter().any(|r| r.is_some());
            let want = if others { (plen + KV_HEADROOM).min(max_seq) } else { plen.min(max_seq) };
            if plen > 0 && plen < max_seq && !self.make_room(lane, want, 0)? {
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

    /// The queue's first request against the running ones, before it is admitted to a free lane
    /// that holds `here` tokens of its prompt: whether it may go now. A running request whose
    /// prompt shares SHARE_MIN more tokens with it than that lane and the disk store hold is not
    /// prefilled a second time beside it. While that prefill runs, this one waits (false), and so
    /// do the requests queued after it; it would otherwise compute the same rows at the same
    /// time and finish no sooner. Once the other decodes, its prompt goes to the disk store now
    /// instead of at its end, and this one restores it. Without the store nothing waits: with
    /// no way to share the rows, waiting would only add the other's prefill to this one's.
    /// 2026-11-09: a client's retry of a 152K-token prompt came while the first was still
    /// decoding its first token, 2 s before that prompt was saved at its end; it prefilled all
    /// 152K tokens again (428 s), and the other lane decoded at 1-2 tok/s meanwhile.
    fn share_running(&mut self, here: usize) -> Result<bool> {
        let Some(store) = self.store.as_ref() else { return Ok(true) };
        let job = self.queue.front().unwrap();
        let (id, prompt) = (job.id, &job.prompt);
        let keys: Vec<u128> = job.images.iter().map(crate::flash_image::image_key).collect();
        let skey = store_tokens(prompt, &job.images, &keys);
        let limit = prompt.len().saturating_sub(1);
        let best = self.runs.iter().flatten()
            .map(|r| (r.lane, prompt.iter().zip(&r.job.prompt).take_while(|(a, b)| a == b).count().min(limit)))
            .max_by_key(|&(_, n)| n);
        let Some((lane, n)) = best.filter(|&(_, n)| n >= here + SHARE_MIN) else { return Ok(true) };
        let run = self.runs[lane].as_ref().unwrap();
        let other = run.job.id;
        if run.part.is_some() && self.held == Some((id, other)) {
            return Ok(false); // still prefilling: decided at the first look
        }
        if self.saved_for == Some((id, other)) {
            return Ok(true); // saved (or the store declined) at an earlier look; the pool was full
        }
        // the store already holding the shared rows (the other restored them) leaves nothing to
        // wait for; a disk short of room leaves this one to prefill them, as without the store
        let stored = store.plan_restore(&skey, here).map_or(here, |(_, end)| end);
        if n < stored + SHARE_MIN || crate::prefix::free_bytes(&store.dir) < 10_000_000_000 {
            return Ok(true);
        }
        if run.part.is_some() {
            self.held = Some((id, other));
            (self.log)(&format!("[prefix] request {id} waits for request {other}'s prefill: their prompts share {n} tokens"));
            return Ok(false);
        }
        let p = run.job.prompt.clone();
        let rkeys: Vec<u128> = run.job.images.iter().map(crate::flash_image::image_key).collect();
        let pk = store_tokens(&p, &run.job.images, &rkeys);
        self.saved_for = Some((id, other));
        (self.log)(&format!("[prefix] request {id} shares {n} tokens with request {other}'s prompt: saving that now, to restore it"));
        self.eng.select(lane)?;
        self.disk_save(&p, &pk)?;
        Ok(true)
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
        // drafting: GLM's MTP layer, or Inkling 2's DSpark drafter (7 drafts a round at most)
        let depth = if self.eng.cfg.mtp { self.opt.depth } else if self.eng.has_drafter() { self.opt.depth.min(7) } else { 0 };
        // the lane's sequence's prefix, one token back so the reprefilled row refills its MTP row
        let mut common = prompt.iter().zip(&self.eng.tokens).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        // images share one placeholder id: no prefix past the first image the prompt and the lane's
        // sequence do not share (position, grid, patches)
        let keys: Vec<u128> = job.images.iter().map(crate::flash_image::image_key).collect();
        common = common.min(self.eng.image_floor(lane, &job.images, &keys));
        // the disk store's deepest saved prefix, when it gains enough over the lane's (not with KDA:
        // the store holds KV rows, not the linear-attention state)
        // Inkling 2: a restore ends at a saved segment's end with the rings as they were there, so the
        // prefill goes on from exactly that position
        let mut restored = false;
        if let Some(store) = self.store.as_mut().filter(|_| !self.eng.cfg.is_flash()) {
            // images count by their keys (store_tokens); a restore never ends inside one (a segment
            // ends where a saved prompt's did, past its images, so this does not happen in practice)
            let skey = store_tokens(&prompt, &job.images, &keys);
            let inside = |end: usize| job.images.iter().any(|im| im.start < end && im.start + crate::flash_image::image_rows(im) > end);
            if let Some((chain, end)) = store.plan_restore(&skey, common).filter(|&(_, end)| !inside(end)) {
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
                    restored = self.eng.cfg.is_inkling();
                    // the lane's sequence is the restored one now, its images those wholly before end
                    let seq = job.images.iter().zip(&keys).filter(|(im, _)| im.start + crate::flash_image::image_rows(im) <= end)
                        .map(|(im, &k)| (im.start, crate::flash_image::image_rows(im), k)).collect();
                    self.eng.set_image_seq(lane, seq);
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
        let keep = if restored { common } else { common.saturating_sub(1) };
        // KDA's state goes back within its last pass, or to the lane's snapshot near its last prompt's
        // end (a chat's follow-up: the template drops the last answer's reasoning, so the prompt leaves
        // the lane's sequence where that answer began; a regenerate); anything else prefills again
        let keep = if self.eng.can_rewind(keep) { keep } else { self.eng.restore_point(lane, keep) };
        self.eng.set_pos(keep)?;
        let mut g = g;
        g.cached = keep;
        let suffix = prompt[keep..].to_vec();
        // the request's images, encoded with the tower in lent expert slots, before the prefill
        if self.eng.vis.is_some() || !job.images.is_empty() {
            match self.eng.encode_images(lane, &job.images, &keys, keep) {
                Ok((n, rows, secs)) if !job.images.is_empty() => {
                    let reused = job.images.len() - n;
                    let r = if reused > 0 { format!(", {reused} in the reused prefix") } else { String::new() };
                    (self.log)(&format!("[vision] request {}: {} image(s){r}, {n} encoded ({rows} rows) in {secs:.2} s", job.id,
                                        job.images.len()));
                }
                Ok(_) => {}
                Err(e) => {
                    // the request's failure, not the engine's (the lent slots are back)
                    (self.log)(&format!("[vision] request {}: {e}", job.id));
                    (job.sink)(Event::Done(Finish::Error(format!("images: {e}")), g));
                    return Ok(());
                }
            }
        }
        self.eng.take_lane_demand(lane)?; // the lane's demand restarts with this request
        // the conversation's key (its opening): its votes accumulate at each of its requests' ends,
        // its continuations preload from them (GLM_PRELOAD)
        let conv = crate::prefix::conv_key(&prompt);
        let mut run = Run {
            job, g, lane, rng: Pcg64::from_seed(0), greedy, temp, top_p, depth, anchor: 0, z_anchor: None, anchor_cands: None,
            lookup: None, copier: None, copied: 0, synced: 0, out: vec![], unlisted: 0.0, folded: 0, next_adapt: self.opt.adapt_every, rounds: 0,
            cancelled: false, part: None, conv, prompt_vote: None, stats: Stats::default(), res0: self.eng.res.stats.clone(),
            lane0: self.eng.lane_miss(lane)?, lane1: None,
        };
        run.rng = Pcg64::from_seed(run.job.seed as u128);
        if self.eng.has_draft_list() {
            let tail = &run.job.prompt[run.job.prompt.len().saturating_sub(256)..];
            run.unlisted = tail.iter().filter(|&&t| !self.eng.draft_lists(t)).count() as f32 / tail.len().max(1) as f32;
        }
        // session-demand preloading (GLM_PRELOAD): a continuation's suffix prefills against the demand its
        // conversation already voiced -- the loads the prefill->decode boundary pass would pay for after the
        // prefill, paid before it, so the suffix computes on the real experts and the boundary pass plans less
        // (the preloaded experts are resident)
        if self.opt.preload > 0 && keep > 0 && suffix.len() >= 4 {
            if let Some((m, _)) = self.conv_votes.get(&conv) {
                if self.eng.preload_wanted(m) {
                    let (mb, s) = (self.eng.res.stats.preload_mb, self.eng.res.stats.preload_s);
                    let n = self.eng.preload(m, self.opt.preload)?;
                    let (mbytes, secs) = (self.eng.res.stats.preload_mb, self.eng.res.stats.preload_s);
                    (self.log)(&format!("{} preload request {}: {} experts of its conversation's demand, \
                                         {} MB read, {:.2} s", self.eng.cfg.arch.tag(), run.job.id, n, (mbytes - mb) / 1_000_000, secs - s));
                }
            }
        }
        // GLM-5.3-Flash: the KDA state two rows before the prompt's end, for the next turn (its prompt
        // shares this one to its end; a regenerate's reprefills its last two rows)
        self.eng.want_snapshot(prompt.len().saturating_sub(2))?;
        // a long prompt prefills in parts even alone, so a request that comes meanwhile is admitted
        // at the next part instead of after the whole prompt: a whole 86K-token prefill (96 s) kept a
        // 28-token chat queued 93.5 s (2026-11-16). Alone, a part is the engine's chunk (one pass, as
        // the whole prefill took it); beside other requests, lane_chunk (prefill_part).
        if suffix.len() > self.opt.lane_chunk.max(self.opt.two_pass) {
            run.part = Some((suffix, 0));
            self.runs[lane] = Some(run);
            return Ok(());
        }
        let opt = self.opt.clone();
        let m0 = self.eng.miss_totals()?;
        let mut first = self.eng.forward(&suffix, depth > 0)?;
        let mut adapted = false;
        if opt.adapt && suffix.len() >= 4 && suffix.len() <= opt.two_pass && self.eng.can_rewind(keep) {
            let m1 = self.eng.miss_totals()?;
            let mass = m1[2] - m0[2];
            let miss = if mass > 0.0 { (m1[3] - m0[3]) / mass } else { 0.0 };
            if miss >= opt.two_pass_miss {
                self.eng.adapt(opt.adapt_prefill)?;
                self.eng.set_pos(keep)?;
                self.eng.want_snapshot(prompt.len().saturating_sub(2))?;
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
        let others = !self.queue.is_empty() || self.runs.iter().enumerate().any(|(l, r)| l != lane && r.is_some());
        let size = if others { self.opt.lane_chunk } else { self.eng.opt.chunk.max(self.opt.lane_chunk) };
        let n = size.max(HEAD_ROWS + 1).min(suffix.len() - c0);
        let end = self.eng.lane_pos(lane) + n;
        if !self.make_room(lane, end, 0)? {
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
        self.eng.clear_image_spans(run.lane);
        self.eng.commit_snapshot(run.lane)?;
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
        // stashed for the conversation's accumulation at this request's end (GLM_PRELOAD)
        run.prompt_vote = Some(d);
        let prompt = &run.job.prompt;
        run.lookup = (self.opt.lookup > 0 && run.depth > 0).then(|| Lookup::new(prompt, self.opt.lookup));
        run.copier = self.opt.copy.clone().filter(|_| run.job.copy_draft && run.depth > 0).map(|c| CopyDraft::new(prompt, c, None));
        let lane = run.lane;
        self.runs[lane] = Some(run);
        Ok(())
    }

    /// A request's end: its Done event, its prompt's rows to the disk store, its line.
    fn finish(&mut self, mut run: Run, finish: Finish) -> Result<()> {
        let finish = if run.stats.nan_rows > 0 && !matches!(finish, Finish::Error(_) | Finish::Cancelled) {
            Finish::Error(format!("{} row(s) of a pass had no finite logits (NaN) in lane {} near position {}", run.stats.nan_rows,
                                  run.lane, self.eng.lane_pos(run.lane)))
        } else {
            finish
        };
        // a request cancelled while it prefilled in parts still holds its image features
        self.eng.clear_image_spans(run.lane);
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
        // the conversation's cumulative votes for its continuations' preloads (GLM_PRELOAD):
        // both of this request's phases, per-layer normalized (adapt::conv_fold_mass), no decay,
        // bounded by last use; the histories above stay the only persistence
        if self.opt.preload > 0 {
            let (nl, e) = (self.eng.cfg.n_layers(), self.eng.cfg.experts);
            self.conv_seq = self.conv_seq.wrapping_add(1);
            let seq = self.conv_seq;
            let acc = self
                .conv_votes
                .entry(run.conv)
                .or_insert_with(|| (vec![0.0; nl * e], seq));
            acc.1 = seq;
            if let Some(pv) = run.prompt_vote.as_deref() {
                crate::adapt::conv_fold_mass(&mut acc.0, pv, nl, e);
            }
            crate::adapt::conv_fold_mass(&mut acc.0, &dem, nl, e);
            if self.conv_votes.len() > CONV_KEEP {
                if let Some(oldest) = self
                    .conv_votes
                    .iter()
                    .min_by_key(|(_, v)| v.1)
                    .map(|(&k, _)| k)
                {
                    self.conv_votes.remove(&oldest);
                }
            }
        }
        let pct = |a: f64, b: f64| if b > 0.0 { a / b * 100.0 } else { 0.0 };
        let s = &run.stats;
        let rounds = s.rounds.max(1) as f64;
        let r1 = &self.eng.res.stats;
        let r0 = &run.res0;
        let nan = if s.nan_rows > 0 { format!(", NaN rows {}", s.nan_rows) } else { String::new() };
        let beside = if s.shared_rounds > 0 {
            format!(" ({} of {} rounds beside another lane; hot reads and swaps are the lanes')", s.shared_rounds, s.rounds)
        } else {
            String::new()
        };
        let listed = if self.eng.has_draft_list() && s.mtp_rounds > 0 {
            format!(", drafts over the listed vocabulary in {} of {} rounds", s.listed_rounds, s.mtp_rounds)
        } else {
            String::new()
        };
        let line = format!("request {}, {}: {:.2} drafted / {:.2} accepted a round, copy rounds {} ({} long, {} accepted), \
                            miss rate {:.1}% (prompt {:.1}%, decode {:.1}%; by score {:.1}%), hot reads {} (kept {}, {} inline, {:.2} s waited), \
                            swaps {} ({:.2} s waited), preload {} ({} MB, {:.2} s), fallbacks {}{nan}{beside}{listed}",
                           run.job.id, self.eng.cfg.arch.name(), s.drafted as f64 / rounds, s.accepted as f64 / rounds, s.copy_rounds, s.long_copies,
                           s.copy_accepted, pct(p[0] + d[0], p[1] + d[1]), pct(p[0], p[1]), pct(d[0], d[1]),
                           pct(p[2] + d[2], p[3] + d[3]), r1.hot_reads - r0.hot_reads, r1.hot_kept - r0.hot_kept,
                           r1.hot_rescue - r0.hot_rescue, r1.hot_wait_s - r0.hot_wait_s, r1.swaps - r0.swaps,
                           r1.adapt_wait_s - r0.adapt_wait_s,
                           r1.preload - r0.preload, (r1.preload_mb - r0.preload_mb) / 1_000_000, r1.preload_s - r0.preload_s,
                           s.fallbacks);
        let (id, lane, prompt) = (run.job.id, run.lane, run.job.prompt.clone());
        let rkeys: Vec<u128> = run.job.images.iter().map(crate::flash_image::image_key).collect();
        let pkey = store_tokens(&prompt, &run.job.images, &rkeys);
        let mut job = run.job;
        (job.sink)(Event::Done(finish, run.g));
        if ok {
            self.eng.select(lane)?;
            self.disk_save(&prompt, &pkey)?;
        }
        self.touch(lane);
        self.done.push(Finished { id, line });
        Ok(())
    }

    /// The prompt's rows past its deepest saved prefix to the disk store (both ranks, the
    /// selected lane's), when they are new enough to keep and the disks have room. `key`: the
    /// prompt as the store keys it (store_tokens: its images by their keys).
    fn disk_save(&mut self, prompt: &[u32], key: &[u32]) -> Result<()> {
        let Some(store) = self.store.as_mut() else { return Ok(()) };
        if self.eng.tokens.len() < prompt.len() || self.eng.tokens[..prompt.len()] != *prompt || key.len() != prompt.len() {
            return Ok(());
        }
        // Inkling 2: a segment ends where the lane's snapshot is (near the prompt's end: its tail is
        // that snapshot); no snapshot inside the prompt, nothing to save
        let prompt = if self.eng.cfg.is_inkling() {
            match self.eng.ink_snapshot_pos(self.eng.lane).filter(|&p| p <= prompt.len()) {
                Some(p) => &key[..p],
                None => return Ok(()),
            }
        } else {
            key
        };
        if crate::prefix::free_bytes(&store.dir) < 10_000_000_000 {
            return Ok(());
        }
        let bpp = self.eng.bytes_per_position();
        let Some((mut seg, evict)) = store.plan_save(prompt, bpp) else { return Ok(()) };
        seg.bytes += self.eng.seg_tail_bytes();
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
            if r.stats.nan_rows > 0 {
                self.finish(r, Finish::Stop)?; // finish reports it as an error
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
                if self.eng.pos < max_seq && self.make_room(r.lane, self.eng.pos + 1, 0)? {
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
        let mut promised = 0usize; // chunks the round's lanes so far will map in its passes
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
            // against the free chunks less what this round's earlier lanes will map in it: each lane
            // alone could pass for the same chunks, and the pass's mapping that then failed ended the
            // process (one box: the pool is one lane's max_seq, shared by the lanes)
            if !self.make_room(r.lane, reach, promised)? {
                (self.log)(&format!("[kv] lane {}: the pool cannot grow it past {p} positions beside the other lanes; ended", r.lane));
                ended.push(i);
                continue;
            }
            promised += self.eng.kv_need(r.lane, reach);
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
        // Inkling 2: one block pass of the DSpark drafter a lane gives its 7 drafts at once (greedy:
        // q is a point mass, as a copy's), kept while the gate holds; no MTP layer, no chain of passes
        let dspark = self.eng.has_drafter();
        if dspark {
            for k in 0..blocks.len() {
                if blocks[k].plain || blocks[k].copy {
                    continue;
                }
                let (lane, anchor) = (runs[blocks[k].i].lane, runs[blocks[k].i].anchor);
                // over the listed vocabulary while this lane's recent text is in it (a lane drafts alone
                // here, so the choice is its own: GLM-5.3-Flash's rule, DRAFT_UNLISTED_MAX)
                let listed = self.eng.has_draft_list() && {
                    let r = &mut runs[blocks[k].i];
                    for &t in &r.out[r.folded..] {
                        let miss = if self.eng.draft_lists(t) { 0.0 } else { 1.0 };
                        r.unlisted += (miss - r.unlisted) * DRAFT_UNLISTED_EMA;
                    }
                    r.folded = r.out.len();
                    r.unlisted <= DRAFT_UNLISTED_MAX
                };
                runs[blocks[k].i].stats.mtp_rounds += 1;
                runs[blocks[k].i].stats.listed_rounds += listed as usize;
                let (toks, probs) = self.eng.ink_draft(lane, anchor, blocks[k].p, listed)?;
                let b = &mut blocks[k];
                for step in 0..toks.len().min(b.dmax) {
                    let pk = probs[step] as f64;
                    b.conf *= pk;
                    let keep_it = match self.cal.as_mut() {
                        Some(c) => c.keep(step, b.conf),
                        None => b.conf >= opt.dyn_gate,
                    };
                    if !keep_it {
                        break;
                    }
                    b.pending.push((step, b.conf));
                    b.pks.push(pk);
                    b.drafts.push(toks[step]);
                    b.qs.push(None);
                }
            }
        }
        let mtp_lanes: Vec<usize> = if dspark { vec![] } else { (0..blocks.len()).filter(|&k| !blocks[k].plain).collect() };
        if !mtp_lanes.is_empty() {
            let logits = mtp_lanes.iter().any(|&k| !blocks[k].copy);
            if self.eng.has_draft_list() {
                // the round's drafts over the listed vocabulary while every drafting lane's recent text
                // is in it (one head a pass)
                let mut listed = true;
                for &k in &mtp_lanes {
                    let r = &mut runs[blocks[k].i];
                    for &t in &r.out[r.folded..] {
                        let miss = if self.eng.draft_lists(t) { 0.0 } else { 1.0 };
                        r.unlisted += (miss - r.unlisted) * DRAFT_UNLISTED_EMA;
                    }
                    r.folded = r.out.len();
                    listed &= blocks[k].copy || r.unlisted <= DRAFT_UNLISTED_MAX;
                }
                self.eng.set_draft_pass(listed);
                for &k in mtp_lanes.iter().filter(|&&k| !blocks[k].copy) {
                    let r = &mut runs[blocks[k].i];
                    r.stats.mtp_rounds += 1;
                    r.stats.listed_rounds += listed as usize;
                }
            }
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
        let fills: Vec<MtpRows> = blocks.iter().enumerate().filter(|(k, b)| !dspark && !b.plain && acc[*k].1 > 0).map(|(k, b)| {
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

    #[test]
    fn store_tokens_tell_images_apart() {
        use tidelm_engine::vision::ImageIn;
        let im = |start: usize, v: u16| ImageIn { start, vit_h: 4, vit_w: 4, patches: vec![v; 64] };
        let prompt: Vec<u32> = (0..20).collect();
        let (a, b) = (vec![im(3, 1)], vec![im(3, 2)]);
        let ka: Vec<u128> = a.iter().map(crate::flash_image::image_key).collect();
        let kb: Vec<u128> = b.iter().map(crate::flash_image::image_key).collect();
        let (ta, tb) = (super::store_tokens(&prompt, &a, &ka), super::store_tokens(&prompt, &b, &kb));
        // text rows as they were, the image's 4 rows past any vocabulary, two images apart
        assert_eq!((&ta[..3], &ta[7..]), (&prompt[..3], &prompt[7..]));
        assert!(ta[3..7].iter().all(|&t| t >= 0x8000_0000));
        assert_ne!(ta[3..7], tb[3..7]);
        assert_eq!(ta, super::store_tokens(&prompt, &a, &ka), "deterministic: a restart finds what was saved");
        // the rows of one image differ from each other (no run of one id)
        assert!(ta[3..7].windows(2).all(|w| w[0] != w[1]));
    }
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
    fn a_prompt_goes_to_the_lane_it_costs_least() {
        // an unrelated request beside a long conversation: the empty lane, not the conversation's
        // (GLM-5.3 would keep the template's 9 tokens there, GLM-5.3-Flash nothing)
        assert_eq!(pick_lane(&[(0, 9, 100_000), (1, 0, 0)]), (1, 0));
        assert_eq!(pick_lane(&[(0, 0, 100_000), (1, 0, 0)]), (1, 0));
        // the conversation's next turn: its lane (the snapshot near its last prompt's end)
        assert_eq!(pick_lane(&[(1, 0, 600), (0, 99_000, 100_500)]), (0, 99_000));
        // a regenerate, and a short lane's follow-up
        assert_eq!(pick_lane(&[(0, 1_998, 2_400), (1, 9, 50_000)]), (0, 1_998));
        // nothing reusable anywhere: the lane that holds least, and between equals the first (the
        // least recently used)
        assert_eq!(pick_lane(&[(0, 0, 80_000), (1, 0, 700)]), (1, 0));
        assert_eq!(pick_lane(&[(1, 0, 0), (0, 0, 0)]), (1, 0));
        // an edit 40% into a conversation: its lane still (60K to prefill either way, 40K saved)
        assert_eq!(pick_lane(&[(0, 40_000, 100_000), (1, 0, 0)]), (0, 40_000));
        // an edit near its start: the empty lane, and the conversation stays whole beside it
        assert_eq!(pick_lane(&[(0, 5_000, 100_000), (1, 0, 0)]), (1, 0));
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
