//! The serving loop (engine/glm/tp.py::_stream) for one request at a time on rank 0: prefix
//! reuse of the live sequence, the prefill (two-pass for a short miss-heavy suffix), then
//! rounds of MTP drafting under the calibrated gate (or a copy: of the prompt, up to 32 tokens
//! by the DS engine's rule, else of the prompt or answer by GLM's lookup), one verify, exact
//! acceptance (greedy, or rejection sampling against each draft's nucleus), the MTP cache
//! refilled for what was accepted; adaptive residency between them. Every engine call it makes
//! is mirrored to rank 1, which decides nothing.

use std::collections::HashMap;
use std::time::Instant;

use tidelm_core::numpy_rng::Pcg64;
use tidelm_engine::copy::{CopyConfig, CopyDraft};
use tidelm_engine::decode::{Event, Finish, Generation, Job};

use crate::forward::{Engine, Hidden, RowCands};
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
    d.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
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
}

pub struct Server {
    pub eng: Engine,
    pub opt: ServeOptions,
    pub cal: Option<CalGate>,
    pub stats: Stats,
    /// the disk prefix store's index (rank 0), when on
    pub store: Option<crate::prefix::Store>,
    sampling: Option<f32>,
    pub log: Box<dyn Fn(&str) + Send>,
}

impl Server {
    pub fn new(eng: Engine, opt: ServeOptions, log: Box<dyn Fn(&str) + Send>) -> Self {
        let cal = opt.cal.map(|t| CalGate::new(t, true));
        Server { eng, opt, cal, stats: Stats::default(), store: None, sampling: None, log }
    }

    /// Row `row` of the last pass as a distribution (sampled requests).
    fn dist(&mut self, row: usize, temp: f32, top_p: f32) -> Result<Dist> {
        if let Some(d) = nucleus(&self.eng.candidates(row), temp, top_p) {
            return Ok(d);
        }
        self.stats.fallbacks += 1;
        let z = self.eng.full_row(row)?;
        Ok(dist_from_logits(&z, temp, top_p))
    }

    /// Row `row`'s chain probability for a greedy draft: max softmax probability (T = 1).
    fn greedy_p(&self, row: usize) -> f64 {
        let rc = self.eng.candidates(row);
        let m = rc.lse.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = rc.lse.iter().map(|&(mr, s)| s as f64 * ((mr - m) as f64).exp()).sum();
        1.0 / z
    }

    /// One request, its tokens to `job.sink` as they are decided. `cancelled` is polled every round.
    pub fn run(&mut self, mut job: Job, cancelled: &mut dyn FnMut() -> bool) -> Result<()> {
        let mut g = Generation { admitted: Some(std::time::Instant::now()), ..Default::default() };
        let finish = self.generate(&mut job, &mut g, cancelled)?;
        let ok = !matches!(finish, Finish::Error(_));
        (job.sink)(Event::Done(finish, g));
        if ok {
            self.disk_save(&job.prompt)?;
        }
        Ok(())
    }

    /// The prompt's rows past its deepest saved prefix to the disk store (both ranks), when
    /// they are new enough to keep and the disks have room.
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

    fn generate(&mut self, job: &mut Job, g: &mut Generation, cancelled: &mut dyn FnMut() -> bool) -> Result<Finish> {
        let opt = self.opt.clone();
        let prompt = job.prompt.clone();
        if prompt.is_empty() {
            return Ok(Finish::Error("empty prompt".into()));
        }
        let max_seq = self.eng.opt.max_seq;
        if prompt.len() >= max_seq {
            return Ok(Finish::Error(format!("prompt of {} tokens > {max_seq}", prompt.len())));
        }
        let greedy = job.sampling.is_greedy();
        let (temp, top_p) = (job.sampling.temperature, job.sampling.top_p);
        // greedy requests too read candidates: the gate's chain probability at T = 1
        let want = if greedy { 1.0 } else { temp };
        if self.sampling != Some(want) {
            self.eng.set_sampling(Some(want))?;
            self.sampling = Some(want);
        }
        let mut rng = Pcg64::from_seed(job.seed as u128);
        let stop = job.stop.clone();
        let depth = if self.eng.cfg.mtp { opt.depth } else { 0 };
        // the live sequence's prefix, one token back so the reprefilled row refills its MTP row
        let mut common = prompt.iter().zip(&self.eng.tokens).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        // the disk store's deepest saved prefix, when it gains enough over the live sequence's
        if let Some(store) = self.store.as_mut() {
            if let Some((chain, end)) = store.plan_restore(&prompt, common) {
                let c: Vec<(u64, usize, usize, String)> = chain.iter().map(|s| (s.id, s.start, s.end, s.hash.clone())).collect();
                let t0 = Instant::now();
                if self.eng.disk_load(&c, &prompt, end)? {
                    store.touch(&chain.iter().map(|s| s.id).collect::<Vec<_>>());
                    store.save_index()?;
                    (self.log)(&format!("[prefix] restored [{common}, {end}) from {} segment(s) in {:.2} s", chain.len(),
                                        t0.elapsed().as_secs_f64()));
                    common = end;
                } else {
                    // a segment some rank could not read: forget it (and what extends it)
                    let bad = chain.last().unwrap().id;
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
        g.cached = keep;
        let suffix = &prompt[keep..];
        let m0 = self.eng.miss_totals()?;
        let mut first = self.eng.forward(suffix, depth > 0)?;
        let mut adapted = false;
        if opt.adapt && suffix.len() >= 4 && suffix.len() <= opt.two_pass {
            let m1 = self.eng.miss_totals()?;
            let mass = m1[2] - m0[2];
            let miss = if mass > 0.0 { (m1[3] - m0[3]) / mass } else { 0.0 };
            if miss >= opt.two_pass_miss {
                self.eng.adapt(opt.adapt_prefill)?;
                self.eng.set_pos(keep)?;
                first = self.eng.forward(suffix, depth > 0)?;
                adapted = true;
                self.stats.two_pass += 1;
            }
        }
        self.eng.res.demand_tokens += suffix.len() as u64;
        if opt.adapt && !adapted {
            self.eng.adapt(opt.adapt_prefill)?;
        }
        let last = self.eng.last_rows - 1;
        // a forward returns every row's argmax up to HEAD_ROWS: the suffix's next token is its
        // last row's (a prefill chunk's head ran on that row alone: row 0)
        let arow = first.len() - 1;
        let mut anchor = if greedy { first[arow] } else { draw(&self.dist(arow, temp, top_p)?, &mut rng) };
        let mut z_anchor: Option<Dist> = None; // the anchor's distribution, for a loop break
        let mut anchor_row = arow; // the anchor's row in the last decoder pass
        self.eng.keep_hidden(last)?;
        let mut lookup = (opt.lookup > 0 && depth > 0).then(|| Lookup::new(&prompt, opt.lookup));
        let mut copier = opt.copy.clone().filter(|_| job.copy_draft && depth > 0).map(|c| CopyDraft::new(&prompt, c, None));
        let mut copied = 0;
        let mut out: Vec<u32> = vec![];
        let mut synced = 0;
        let mut next_adapt = opt.adapt_every;
        let mut rounds = 0usize;
        let max_new = job.max_tokens;
        loop {
            if cancelled() {
                return Ok(Finish::Cancelled);
            }
            if opt.loop_break > 0 && {
                let mut s = out.clone();
                s.push(anchor);
                tail_cycle(&s, opt.loop_break)
            } {
                if greedy {
                    // the anchor row's best other token (its candidates are still the last
                    // decoder pass's: MTP fills read no head)
                    let rc = self.eng.candidates(anchor_row);
                    let best = rc.vals.iter().zip(&rc.ids).filter(|(_, &i)| i != anchor)
                        .max_by(|a, b| a.0.partial_cmp(b.0).unwrap().then(b.1.cmp(a.1))).map(|(_, &i)| i);
                    if let Some(b) = best {
                        anchor = b;
                        self.stats.loop_breaks += 1;
                    }
                } else if let Some(d) = z_anchor.take() {
                    let d: Dist = d.into_iter().filter(|x| x.0 != anchor).collect();
                    if !d.is_empty() {
                        anchor = draw(&d, &mut rng);
                        self.stats.loop_breaks += 1;
                    }
                }
            }
            if opt.adapt && opt.adapt_every > 0 && out.len() >= next_adapt {
                self.eng.adapt(opt.adapt_decode)?;
                next_adapt += opt.adapt_every;
            }
            rounds += 1;
            if opt.adapt && opt.bg_every > 0 && rounds % opt.bg_every == 0 && self.eng.hot.is_some() {
                self.eng.bg_step(opt.hot_keep)?;
            }
            (job.sink)(Event::Tokens(vec![anchor]));
            out.push(anchor);
            g.tokens.push(anchor);
            if stop.contains(&anchor) || out.len() >= max_new {
                // the KV holds the last token, so the next turn's prompt matches through it
                if self.eng.pos < max_seq {
                    self.eng.forward(&[anchor], false)?;
                }
                return Ok(if stop.contains(&anchor) { Finish::Stop } else { Finish::Length });
            }
            let p = self.eng.pos;
            let room = max_seq.saturating_sub(p + 1).min(max_new - out.len());
            let dmax = depth.min(room);
            if dmax == 0 {
                anchor = self.plain_step(anchor, greedy, temp, top_p, &mut rng)?;
                anchor_row = 0;
                g.steps += 1;
                continue;
            }
            // a copy round (a long copy of the prompt, else the lookup's), or MTP drafts under the gate
            let mut copy = None;
            let mut long = false;
            if let Some(cd) = copier.as_mut() {
                cd.push(&out[copied..]);
                copied = out.len();
                let mut c = cd.propose();
                c.truncate(room);
                if !c.is_empty() {
                    copy = Some(c);
                    long = true;
                }
            }
            if let Some(lk) = lookup.as_mut() {
                lk.extend(&out[synced..]);
                synced = out.len();
                if copy.is_none() {
                    copy = lk.propose(dmax);
                }
            }
            let mut drafts: Vec<u32> = vec![];
            let mut qs: Vec<Option<Dist>> = vec![];
            let mut pending: Vec<(usize, f64)> = vec![];
            if let Some(c) = copy.as_ref() {
                drafts = c.clone();
                qs = vec![None; drafts.len()];
                self.eng.mtp(&[anchor], Hidden::Last, p - 1, false)?;
            } else {
                let mut conf = 1.0f64;
                let mut d0 = self.eng.mtp(&[anchor], Hidden::Last, p - 1, true)?.unwrap()[0];
                for k in 0..dmax {
                    let (d, q, pk) = if greedy {
                        (d0, None, self.greedy_p(0))
                    } else {
                        let q = self.dist(0, temp, top_p)?;
                        let d = draw(&q, &mut rng);
                        let pk = prob_of(&q, d) as f64;
                        (d, Some(q), pk)
                    };
                    conf *= pk;
                    let keep_it = match self.cal.as_mut() {
                        Some(c) => c.keep(k, conf),
                        None => conf >= opt.dyn_gate,
                    };
                    if !keep_it {
                        break;
                    }
                    pending.push((k, conf));
                    drafts.push(d);
                    qs.push(q);
                    if k + 1 < dmax {
                        d0 = self.eng.mtp(&[d], Hidden::Mtp, p + k, true)?.unwrap()[0];
                    }
                }
            }
            let nd = drafts.len();
            let mut block = vec![anchor];
            block.extend_from_slice(&drafts);
            let am = self.eng.forward(&block, false)?;
            g.steps += 1;
            let (mut n_acc, mut nxt) = (nd, None);
            for k in 0..nd {
                if greedy {
                    if am[k] != drafts[k] {
                        n_acc = k;
                        nxt = Some(am[k]);
                        break;
                    }
                } else {
                    let pk = self.dist(k, temp, top_p)?;
                    let dk = drafts[k];
                    let qd = qs[k].as_ref().map(|q| prob_of(q, dk)).unwrap_or(1.0).max(1e-20);
                    let ratio = prob_of(&pk, dk) / qd;
                    if unif(&mut rng) >= ratio.min(1.0) as f64 {
                        // the residual (pk - q)+, or pk without the copied token
                        let resid: Dist = match qs[k].as_ref() {
                            Some(q) => pk.iter().map(|&(i, p)| (i, (p - prob_of(q, i)).max(0.0))).filter(|x| x.1 > 0.0).collect(),
                            None => pk.iter().filter(|x| x.0 != dk).cloned().collect(),
                        };
                        let r = if resid.iter().map(|x| x.1).sum::<f32>() > 0.0 { resid } else { pk.clone() };
                        n_acc = k;
                        nxt = Some(draw(&r, &mut rng));
                        z_anchor = Some(pk);
                        break;
                    }
                }
            }
            anchor_row = n_acc;
            let nxt = match nxt {
                Some(t) => t,
                None if greedy => am[nd],
                None => {
                    let d = self.dist(nd, temp, top_p)?;
                    let t = draw(&d, &mut rng);
                    z_anchor = Some(d);
                    t
                }
            };
            if greedy {
                z_anchor = None;
            }
            if copy.is_none() {
                if let Some(c) = self.cal.as_mut() {
                    for &(k, cf) in &pending {
                        c.observe(k, cf, k < n_acc);
                    }
                }
            }
            self.stats.rounds += 1;
            self.stats.drafted += nd;
            self.stats.accepted += n_acc;
            g.accepted.push(n_acc);
            g.depths.push(nd);
            if long {
                copier.as_mut().unwrap().observe(n_acc);
                self.stats.long_copies += 1;
            }
            if copy.is_some() {
                self.stats.copy_rounds += 1;
                self.stats.copy_accepted += n_acc;
                g.copy_steps += 1;
                g.copy_accepted += n_acc;
            }
            self.eng.set_pos(p + 1 + n_acc)?;
            if n_acc > 0 {
                self.eng.mtp(&drafts[..n_acc], Hidden::Target(0), p, false)?;
            }
            self.eng.keep_hidden(n_acc)?;
            self.eng.res.demand_tokens += (nd + 1) as u64;
            if n_acc > 0 {
                (job.sink)(Event::Tokens(drafts[..n_acc].to_vec()));
            }
            for &d in &drafts[..n_acc] {
                out.push(d);
                g.tokens.push(d);
                if stop.contains(&d) {
                    return Ok(Finish::Stop);
                }
                if out.len() >= max_new {
                    return Ok(Finish::Length);
                }
            }
            anchor = nxt;
        }
    }

    /// tp.warmup: two greedy generations on synthetic prompts (graphs captured, kernels warm)
    /// and a sampled one, with no adaptation or gate learning; their demand forgotten after.
    pub fn warmup(&mut self) -> Result<()> {
        let (opt, cal) = (self.opt.clone(), self.cal.take());
        self.opt.adapt = false;
        let stop = self.eng.cfg.eos.clone();
        let prompts: [Vec<u32>; 3] = [(0..192).map(|i| 1000 + i % 12).collect(), (0..64).map(|i| (i * 7919) % 50000 + 1000).collect(),
                                      (0..64).map(|i| (i * 104729) % 60000 + 2000).collect()];
        for (n, p) in prompts.into_iter().enumerate() {
            let sampling = if n == 2 { tidelm_engine::forward::Sampling { temperature: 0.6, top_p: 0.95 } }
                           else { tidelm_engine::forward::Sampling::GREEDY };
            let job = Job { id: 0, prompt: p, max_tokens: 48, sampling, seed: 1, stop: stop.clone(), sink: Box::new(|_| {}),
                            images: vec![], key: None, copy_draft: false };
            self.run(job, &mut || false)?;
        }
        self.opt = opt;
        self.cal = cal;
        self.stats = Stats::default();
        self.eng.res.stats = Default::default();
        self.eng.zero_demand()?;
        self.eng.set_pos(0)
    }

    fn plain_step(&mut self, anchor: u32, greedy: bool, temp: f32, top_p: f32, rng: &mut Pcg64) -> Result<u32> {
        let am = self.eng.forward(&[anchor], false)?;
        self.eng.keep_hidden(0)?;
        if greedy {
            return Ok(am[0]);
        }
        let d = self.dist(0, temp, top_p)?;
        Ok(draw(&d, rng))
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
