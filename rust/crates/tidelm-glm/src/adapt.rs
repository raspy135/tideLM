//! Adaptive expert residency (engine/glm/tp.py: adapt, bg_step, _replay_split's hot reads,
//! _hot_keep): which routed experts hold the arena's slots, and how that follows demand.
//!
//! - adapt(max_swaps): after a prefill (512) and every 200 decoded tokens (256: adapt_background,
//!   below): score =
//!   0.9 * score + demand; across the swap layers, the non-resident experts with the highest
//!   score replace the residents with the lowest while the newcomer beats the victim by x1.5,
//!   never taking a layer below 2 x top-k residents. Rank 0 plans, both ranks load the same
//!   experts into the same slots. With mixed widths (arena.rs) a newcomer only replaces a
//!   resident of its own width class, so every victim, swap or hot read, is chosen in its class.
//! - bg_step, every 2 rounds: rank 0 queues up to 32 victims for hot reads to keep (residents
//!   whose recent mass is below 0.01, not installed within 8 periods, lowest score first).
//! - hot reads, between a decode pass's segments: a MoE layer's heaviest missed pick with a
//!   share >= 0.2 of its row's routed weight is read from the pack into the transient slot and
//!   computed exactly (one a layer, 16 a step paced evenly over the layers); after the layer it
//!   takes the next queued victim's slot or is dropped. Both ranks decide from the same
//!   replicated routing, so they read the same experts without a message.
//! - async hot reads (GLM_HOT_ASYNC, tp.py's _hot_plan / _hot_install), in place of those: after
//!   a decode pass, per MoE layer its heaviest missed pick at or above the same threshold, the
//!   heaviest of those up to the same step cap, each into the next queued victim's slot on the
//!   reader threads while the next round drafts; routable from the next decoder pass. The token
//!   that missed the expert keeps its substitute (KL 0.086 -> 0.137 in Python).
//!   GLM_HOT_PREDICT ranks candidates by missed share accumulated over the pass's rows and
//!   decayed over passes (HotPredict), alone or after the heaviest-miss ones (REACT=1);
//!   GLM_HOT_VICTIM=slow ranks the cold victims by slowly decayed demand (BG_SLOW).
//!   GLM_HOT_RESCUE=TH (with GLM_HOT_ASYNC): a pass run as its segments, and a missed pick with
//!   share >= TH read into the transient slot before its layer's MoE, computed exactly, kept
//!   over a victim as hot reads are - the catastrophic first occurrence covered inline, the
//!   async plan unchanged after the pass (a kept rescue expert is resident, so the plan skips
//!   it). The pass pays the segment structure (a host point a MoE layer) to know a first
//!   occurrence before its MoE runs; that structure is the arm's cost and the A/B measures it.
//! - adapt_background(max_swaps), the decode-time adaptation: the same plan, but the victims
//!   leave routing at once and the newcomers' records load on reader threads while the pair
//!   decodes, routable from the ADAPT_BG_PASSES-th decoder pass after (a pass waits for reads not
//!   yet done). Both ranks count the same passes, so they switch between the same two. In line,
//!   a 256-expert adaptation stopped the GPU for ~166 ms every 200 tokens.

use std::os::raw::c_void;
use std::sync::mpsc;
use std::time::Instant;

use tidelm_cuda::{PinnedBuffer, Stream};
use tidelm_model::direct_io::DirectFile;

use crate::arena::{align_up, Pack};
use crate::forward::Engine;
use crate::{ck, Result};

pub const ADAPT_DECAY: f64 = 0.9;
pub const ADAPT_MARGIN: f64 = 1.5;
const BG_DECAY: f64 = 0.7;
const BG_COLD: f64 = 0.01;
const BG_PROTECT: u64 = 8;
/// GLM_HOT_VICTIM=slow: bg_slow's decay a background period (two passes): ~0.99 a pass, a
/// half-life of ~70 passes, where BG_DECAY's is ~4. On heldout_quick (no adaptation) the engine's
/// victims - cold by bg_score, then the lowest long-run score - were picked again within 50 passes
/// half the time, and 37% of the run's missed picks >= 0.3 were experts it had evicted itself.
const BG_SLOW: f64 = 0.98;
/// decoder passes between a background adaptation's start and its newcomers becoming routable
pub const ADAPT_BG_PASSES: u64 = 6;
/// the background adaptation's reader threads
const ADAPT_READERS: usize = 8;
/// GLM_HOT_RESCUE's reads a pass. On heldout_quick (A.trace, skip 16) missed picks >= 0.4 were
/// 0.06 a pass over 4.5% of passes (>= 0.3: 0.25 over 18.6%), so 2 leaves the cap unfelt; the
/// damage band >= 0.4 is 5.1% of the per-(layer,row) heaviest-miss share^3 (78.2% sits below
/// 0.2) - the rescue is insurance for the blown-window class, not a miss-rate lever.
const RESCUE_STEP: usize = 2;

/// The residency bookkeeping both ranks keep identical (rank 0's plans are mirrored).
pub struct Res {
    /// per slot: the (layer, expert) it holds
    pub owner: Vec<Option<(usize, usize)>>,
    pub swap_layers: Vec<usize>,
    /// rank 0's planning state: decayed demand (adapt), recent mass (bg), protection
    pub score: Vec<f64>,
    pub bg_score: Vec<f64>,
    /// the same demand decayed slowly (BG_SLOW a period): GLM_HOT_VICTIM=slow's ranking
    pub bg_slow: Vec<f64>,
    /// GLM_HOT_VICTIM=slow: cold victims ranked by bg_slow first (then the score), not by the score
    pub victim_slow: bool,
    pub bg_last: Vec<f64>,
    pub bg_until: Vec<u64>,
    pub bg_period: u64,
    /// victims for kept hot reads, weakest last: (layer, expert, slot)
    pub hot_q: Vec<(usize, usize, usize)>,
    /// the routing history rank 0 saves (mass [layers][experts], tokens)
    pub db_mass: Option<Vec<f64>>,
    pub db_tokens: u64,
    pub db_path: Option<std::path::PathBuf>,
    pub demand_tokens: u64,
    pub stats: ResStats,
    /// the recent residency events and their causes, for the expert map (host bookkeeping: read by
    /// nothing that decides anything)
    pub events: tidelm_engine::expert_map::EventLog,
}

#[derive(Default, Clone, Debug)]
pub struct ResStats {
    pub swaps: usize,
    pub adapt_s: f64,
    pub hot_reads: usize,
    pub hot_kept: usize,
    pub hot_unkept: usize,
    pub hot_s: f64,
    /// async: host seconds waiting for a round's reads before the next decoder pass
    pub hot_wait_s: f64,
    /// GLM_HOT_RESCUE: reads the segments made inline (in hot_reads too)
    pub hot_rescue: usize,
    /// background adaptations: host seconds a decoder pass waited for their reads
    pub adapt_wait_s: f64,
    /// session-demand preloading: experts a continuation's conversation had already voiced,
    /// loaded before its suffix prefills, their bytes, and the host seconds their reads took
    pub preload: usize,
    pub preload_mb: u64,
    pub preload_s: f64,
}

/// One conversation's cumulative votes: `vote` (a phase's routed weight per (layer, expert))
/// normalized per layer as the DS history's fold (sum to one a layer) and accumulated into
/// `acc` without decay -- the conversation's own total, read only by its continuations'
/// preloads. The histories keep the only persistence; this never leaves the process.
pub fn conv_fold_mass(acc: &mut [f64], vote: &[f32], layers: usize, experts: usize) {
    for l in 0..layers {
        let (lo, hi) = (l * experts, (l + 1) * experts);
        let sm: f64 = vote[lo..hi]
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            .max(1e-20);
        for (a, &x) in acc[lo..hi].iter_mut().zip(&vote[lo..hi]) {
            *a += x as f64 / sm;
        }
    }
}

/// The preload candidates: a conversation's non-resident demand by accumulated share,
/// descending (ties to the lower (layer, expert)). Pure; `resident(l, x)` marks what is
/// routable now.
pub fn preload_candidates(
    conv: &[f64],
    resident: &dyn Fn(usize, usize) -> bool,
    swap_layers: &[usize],
    experts: usize,
) -> Vec<(usize, usize)> {
    let mut cand: Vec<(usize, usize)> = swap_layers
        .iter()
        .flat_map(|&l| (0..experts).map(move |x| (l, x)))
        .filter(|&(l, x)| !resident(l, x) && conv[l * experts + x] > 0.0)
        .collect();
    cand.sort_by(|a, b| {
        conv[b.0 * experts + b.1]
            .partial_cmp(&conv[a.0 * experts + a.1])
            .unwrap()
            .then(a.cmp(b))
    });
    cand
}

/// Synchronous hot reads' settings and per-step state.
pub struct Hot {
    pub min_share: f32,
    pub per_step: usize,
    /// GLM_HOT_RESCUE's threshold (a missed pick's share to read inline before its layer's MoE;
    /// needs the async plan, whose table and victims it shares)
    pub rescue: Option<f32>,
    /// the boot's GLM_HOT_RESCUE (bench knob `rescue` switches `rescue` between it and off)
    pub rescue_boot: Option<f32>,
    /// the transient slot a read goes into
    pub slot: usize,
    /// candidates of the current pass, from the device: [rows * topk][2] (share, expert)
    pub host: PinnedBuffer<f32>,
    pub buf: PinnedBuffer<u8>,
    reads: usize,
    /// the read installed for the layer being computed: (layer, expert), the missed pick's share,
    /// and whether it was a rescue (for the expert map's events)
    live: Option<(usize, usize)>,
    live_cause: (f32, bool),
    /// GLM_HOT_ASYNC's state (None: the synchronous reads above)
    pub asynch: Option<HotAsync>,
}

impl Hot {
    pub fn new(slot: usize, max_record: usize, rows: usize, topk: usize) -> Result<Self> {
        Ok(Hot { min_share: 0.2, per_step: 16, rescue: None, rescue_boot: None, slot, host: PinnedBuffer::zeroed(rows * topk * 2)?,
                 buf: PinnedBuffer::uninit(align_up(max_record))?, reads: 0, live: None, live_cause: (0.0, false), asynch: None })
    }
}

/// Async hot reads: the last decode pass's candidates, the reads in flight, the reader threads.
pub struct HotAsync {
    /// [MoE layer][MAX_ROWS * topk][2] (share, expert), copied behind the head of every decode
    /// pass: a missed pick's share positive, a resident's negated
    pub table: PinnedBuffer<f32>,
    jobs: mpsc::Sender<ReadJob>,
    /// planned reads: (slot, layer, expert, done)
    pending: Vec<(usize, usize, usize, mpsc::Receiver<std::result::Result<(), String>>)>,
    /// GLM_HOT_PREDICT: the reads chosen by recent missed demand (None: one pass's heaviest pick)
    pub predict: Option<HotPredict>,
    /// TIDELM_GLM_HOT_TRACE (logging only; rank 1 to PATH.r1): every pass's table and reads,
    /// for tools/glm_hot_sim.py
    pub trace: Option<std::fs::File>,
    /// plans made (the trace's pass index of the next one)
    pub plans: usize,
    /// TIDELM_GLM_HOT_ORACLE (debug): an earlier run's next-pass heavy misses, read first
    pub oracle: Option<HotOracle>,
}

/// TIDELM_GLM_HOT_ORACLE=PATH,MIN (debug, --bench-nll only): a perfect predictor of the heavy
/// misses at async speed - the most a synchronous read (or a re-run pass) of the misses >= MIN
/// could buy, measured before building one. PATH is an earlier run's TIDELM_GLM_HOT_TRACE of the
/// same bench (corpus, rows, boot: its passes line up with this run's, checked by their rows);
/// the plan after pass p puts first every pick that run missed at >= MIN in pass p + 1 and that
/// is not resident here, read async and so routable for pass p + 1. Rank r > 0 reads PATH.r{r},
/// its own trace of that run (the same bytes; the boot guard compares the files' sums).
pub struct HotOracle {
    /// [recorded pass]: its rows, and its misses >= MIN as (share, layer, expert), one a (layer, expert)
    rows: Vec<u32>,
    misses: Vec<Vec<(f32, usize, usize)>>,
}

impl HotOracle {
    /// (this rank's file, MIN) from TIDELM_GLM_HOT_ORACLE's value
    pub fn spec(v: &str, rank: usize) -> Result<(String, f32)> {
        let bad = || format!("TIDELM_GLM_HOT_ORACLE={v}: PATH,MIN (an earlier run's TIDELM_GLM_HOT_TRACE, MIN > 0)");
        let (path, min) = v.rsplit_once(',').ok_or_else(bad)?;
        let min: f32 = min.trim().parse().map_err(|_| bad())?;
        if min <= 0.0 || path.is_empty() {
            return Err(bad().into());
        }
        Ok((if rank == 0 { path.to_string() } else { format!("{path}.r{rank}") }, min))
    }

    /// The boot guard's word: the setting and this rank's file's FNV-1a sum ("" when unset)
    pub fn guard(rank: usize) -> Result<String> {
        let v = std::env::var("TIDELM_GLM_HOT_ORACLE").unwrap_or_default();
        if v.is_empty() {
            return Ok(String::new());
        }
        let (path, _) = Self::spec(&v, rank)?;
        let b = std::fs::read(&path).map_err(|err| format!("TIDELM_GLM_HOT_ORACLE: {path}: {err}"))?;
        let sum = b.iter().fold(0xcbf29ce484222325u64, |h, &c| (h ^ c as u64).wrapping_mul(0x100000001b3));
        Ok(format!("{v} {sum:016x}"))
    }

    pub fn load(path: &str, min: f32, moe: &[usize], experts: usize, topk: usize) -> Result<Self> {
        let b = std::fs::read(path).map_err(|err| format!("TIDELM_GLM_HOT_ORACLE: {path}: {err}"))?;
        let short = || format!("TIDELM_GLM_HOT_ORACLE: {path}: not a whole hot trace");
        let u32at = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(short);
        let f32at = |o: usize| b.get(o..o + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap())).ok_or_else(short);
        let ids = match b.get(..4) {
            Some(b"GHT1") => false,
            Some(b"GHT2") => true,
            _ => return Err(short().into()),
        };
        let (n, e, k) = (u32at(4)? as usize, u32at(8)? as usize, u32at(12)? as usize);
        let mut off = 16;
        if (n, e, k) != (moe.len(), experts, topk) || (0..n).any(|i| u32at(off + 4 * i).ok() != Some(moe[i] as u32)) {
            return Err(format!("TIDELM_GLM_HOT_ORACLE: {path}: another model's trace").into());
        }
        off += 4 * n + n * e;
        let (mut rows, mut misses) = (Vec::new(), Vec::new());
        while off < b.len() {
            let t = u32at(off)? as usize;
            off += 4 + if ids { 4 + 4 * t } else { 0 };
            let mut m: Vec<(f32, usize, usize)> = Vec::new();
            for (i, &l) in moe.iter().enumerate() {
                for j in 0..t * k {
                    let o = off + (i * t * k + j) * 8;
                    let (v, x) = (f32at(o)?, f32at(o + 4)? as usize);
                    if v >= min {
                        match m.iter_mut().find(|c| (c.1, c.2) == (l, x)) {
                            Some(c) => c.0 = c.0.max(v),
                            None => m.push((v, l, x)),
                        }
                    }
                }
            }
            off += n * t * k * 8;
            off += 4 + 8 * u32at(off)? as usize;
            rows.push(t as u32);
            misses.push(m);
        }
        if off != b.len() {
            return Err(short().into());
        }
        Ok(HotOracle { rows, misses })
    }

    /// The plan after pass `p` of `t` rows: the recorded run's misses >= MIN in pass p + 1.
    pub fn next(&self, p: usize, t: usize) -> Result<&[(f32, usize, usize)]> {
        if self.rows.get(p).copied() != Some(t as u32) {
            return Err(format!("TIDELM_GLM_HOT_ORACLE: pass {p} has {t} rows, the recorded run's {:?}: not the same bench",
                               self.rows.get(p)).into());
        }
        Ok(self.misses.get(p + 1).map(Vec::as_slice).unwrap_or(&[]))
    }

    pub fn summary(&self) -> (usize, usize) {
        (self.rows.len(), self.misses.iter().map(Vec::len).sum())
    }
}

/// GLM_HOT_PREDICT=DECAY,MIN,PER_LAYER[,REACT]: the async plan's candidates ranked by each
/// (layer, expert)'s missed share accumulated over the pass's rows and decayed over passes
/// (acc <- DECAY * acc + this pass's missed shares). Up to PER_LAYER a layer with acc >= MIN,
/// the same per-step cap and victims; REACT=1 keeps the one heaviest missed pick of the pass
/// (the plan without a predictor) first in every layer and adds these after it.
/// Its guess: an expert missed in several rows, or in several recent passes, is the next
/// pass's demand more surely than one row's single heavy miss. Both ranks update it from the
/// same replicated table in the same order, so they plan the same reads without a message.
pub struct HotPredict {
    pub decay: f32,
    pub min: f32,
    pub per_layer: usize,
    /// REACT=1: the engine's own candidate (each layer's heaviest missed pick >= 0.2) first,
    /// the tally's after it; 0 (the 3-field form): the tally's alone - measured worse (+0.024 +-
    /// 0.005 NLL on heldout_quick at 0.48 reads a token against 0.57): a one-off heavy miss never
    /// reaches MIN, and heavy misses cost more than their share. REACT=1 (0.9,1.0,1,1) measured
    /// worse too: +0.122 NLL, a collapse in one text (its next token 16.5 nats through the prefill
    /// path, 16.6 as decode passes: the residency, not the path), and +0.005 +- 0.003 where nothing
    /// collapses (16-token prefills), at 0.73 reads a token. Its own reads are good predictions (70%
    /// picked the next pass, the rule's 34%) of light picks (0.008 picks >= 0.3 each in 16 passes,
    /// the rule's 0.15); the extra reads cut a rule read's life 1098 -> 438 passes, and the heavy
    /// misses left are first heavy picks (0.5% had one in the 16 passes before):
    /// tools/glm_hot_eval.py
    pub react: bool,
    /// [MoE layer index][expert]
    acc: Vec<f32>,
}

impl HotPredict {
    pub fn parse(v: &str, layers: usize, experts: usize) -> Result<Option<Self>> {
        let v = v.trim();
        if v.is_empty() || v == "0" {
            return Ok(None);
        }
        let f: Vec<&str> = v.split(',').map(str::trim).collect();
        let bad = || format!("GLM_HOT_PREDICT={v}: DECAY,MIN,PER_LAYER[,REACT] (0 <= DECAY < 1, MIN > 0, PER_LAYER >= 1, REACT 0/1)");
        if f.len() != 3 && f.len() != 4 {
            return Err(bad().into());
        }
        let react = match f.get(3) {
            None | Some(&"0") => false,
            Some(&"1") => true,
            _ => return Err(bad().into()),
        };
        let decay: f32 = f[0].parse().map_err(|_| bad())?;
        let min: f32 = f[1].parse().map_err(|_| bad())?;
        let per_layer: usize = f[2].parse().map_err(|_| bad())?;
        if !(0.0..1.0).contains(&decay) || min <= 0.0 || per_layer == 0 {
            return Err(bad().into());
        }
        Ok(Some(HotPredict { decay, min, per_layer, react, acc: vec![0.0; layers * experts] }))
    }

    /// One pass's table for MoE layer index `i` (its `t * topk` (share, expert) pairs) into the
    /// accumulators: decay, then add every missed pick's share.
    pub fn fold(&mut self, i: usize, experts: usize, row: &[f32]) {
        let acc = &mut self.acc[i * experts..(i + 1) * experts];
        for a in acc.iter_mut() {
            *a *= self.decay;
        }
        for p in row.chunks_exact(2) {
            if p[0] > 0.0 {
                acc[p[1] as usize] += p[0];
            }
        }
    }

    /// Layer index `i`'s candidates (acc, expert), at most per_layer of them, heaviest first
    /// (ties to the lower expert), among the experts `ok` allows.
    pub fn candidates(&self, i: usize, experts: usize, ok: impl Fn(usize) -> bool) -> Vec<(f32, usize)> {
        let acc = &self.acc[i * experts..(i + 1) * experts];
        let mut c: Vec<(f32, usize)> = (0..experts).filter(|&x| acc[x] >= self.min && ok(x)).map(|x| (acc[x], x)).collect();
        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        c.truncate(self.per_layer);
        c
    }

    /// A read was planned: its accumulator starts over (resident now, it is no candidate; if it
    /// is evicted later, only its demand from then on counts).
    pub fn taken(&mut self, i: usize, experts: usize, x: usize) {
        self.acc[i * experts + x] = 0.0;
    }
}

/// A background adaptation in flight: its reads (slot, layer, expert, done), the decoder pass
/// from which its newcomers are routable, and its own reader threads.
pub struct AdaptAsync {
    jobs: mpsc::Sender<ReadJob>,
    pending: Vec<(usize, usize, usize, mpsc::Receiver<std::result::Result<(), String>>)>,
    /// layer * experts + expert of every pending newcomer (hot reads leave them alone)
    pub newcomers: std::collections::HashSet<usize>,
    install_at: u64,
}

/// One record for a reader thread: where it is in the pack, where its nine tensors go.
struct ReadJob {
    off: u64,
    nbytes: usize,
    dsts: [(usize, usize); 9],
    done: mpsc::Sender<std::result::Result<(), String>>,
}

impl HotAsync {
    fn new(pack: &Pack, threads: usize, table_len: usize) -> Result<Self> {
        Ok(HotAsync { table: PinnedBuffer::zeroed(table_len)?, jobs: readers(pack, threads)?, pending: vec![],
                      predict: None, trace: None, plans: 0, oracle: None })
    }
}

/// `threads` readers, each with its own O_DIRECT handle, staging buffer and stream (made here, so a
/// reader that cannot start fails its caller, not a read): a job's record read from the pack and
/// copied into its slot's tensors, then `done`.
fn readers(pack: &Pack, threads: usize) -> Result<mpsc::Sender<ReadJob>> {
    let (tx, rx) = mpsc::channel::<ReadJob>();
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    for _ in 0..threads {
        let rx = rx.clone();
        let data0 = pack.data0;
        let file = DirectFile::open(&pack.path)?;
        let st = Stream::new()?;
        let mut buf = PinnedBuffer::<u8>::uninit(align_up(pack.max_record))?;
        std::thread::spawn(move || {
            let dev = tidelm_cuda::set_device(0).map_err(|e| e.to_string());
            loop {
                let Ok(job) = rx.lock().unwrap().recv() else { return };
                let mut run = || -> Result<()> {
                    dev.clone()?;
                    file.pread_exact(&mut buf[..align_up(job.nbytes)], data0 + job.off, job.nbytes)?;
                    let mut off = 0;
                    for &(dst, n) in &job.dsts {
                        unsafe {
                            tidelm_cuda::memcpy_async(dst as *mut c_void, buf[off..].as_ptr() as *const c_void, n,
                                                      tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &st)?;
                        }
                        off += n;
                    }
                    st.synchronize()?;
                    Ok(())
                };
                let _ = job.done.send(run().map_err(|e| e.to_string()));
            }
        });
    }
    Ok(tx)
}

impl Res {
    /// `placed`: (slot, layer, expert) of the boot residents; `slots` every slot of the arena.
    pub fn new(nl: usize, e: usize, slots: usize, swap_layers: Vec<usize>, placed: &[(usize, usize, usize)]) -> Self {
        let mut owner = vec![None; slots];
        for &(s, l, x) in placed {
            owner[s] = Some((l, x));
        }
        Res { owner, swap_layers, score: vec![0.0; nl * e], bg_score: vec![0.0; nl * e], bg_slow: vec![0.0; nl * e],
              victim_slow: false, bg_last: vec![0.0; nl * e],
              bg_until: vec![0; nl * e], bg_period: 0, hot_q: vec![], db_mass: None, db_tokens: 0, db_path: None,
              demand_tokens: 0, stats: ResStats::default(), events: Default::default() }
    }
}

pub(crate) const OP_ADAPT: u32 = 20;
pub(crate) const OP_BG: u32 = 21;
pub(crate) const OP_ADAPT_BG: u32 = 22;

impl Engine {
    fn read_demand(&self) -> Result<Vec<f64>> {
        self.stream().synchronize()?;
        Ok(self.demand.download()?.into_iter().map(|x| x as f64).collect())
    }

    /// Apply a swap plan [(layer, expert, victim layer, victim expert)] on this rank: the lut,
    /// owners, and the newcomers' records read into their victims' slots.
    pub(crate) fn apply_swaps(&mut self, plan: &[(usize, usize, usize, usize)]) -> Result<()> {
        let e = self.cfg.experts;
        let mut items = vec![];
        for &(l, c, vl, v) in plan {
            let slot = self.lut_h[vl * e + v];
            if slot < 0 {
                return Err(format!("adapt: victim ({vl}, {v}) is not resident").into());
            }
            self.lut_h[vl * e + v] = -1;
            self.lut_h[l * e + c] = slot;
            self.res.owner[slot as usize] = Some((l, c));
            items.push((slot as usize, l, c));
        }
        if !items.is_empty() {
            self.stream().synchronize()?;
            crate::forward::load_slots(&self.pack, &self.arena, &items)?;
        }
        self.push_lut()
    }

    /// The residency as the expert map shows it (tidelm_engine::expert_map), on rank 0: the
    /// decoder's MoE layers by the host LUT, with the reads in flight (async hot reads, a background
    /// adaptation's newcomers), and the routing history's mass (the saved one, else the decayed
    /// demand). Host data only: nothing here touches the device or rank 1. The MTP layer, whose
    /// experts are all resident in their own arena, is left out.
    pub fn expert_map(&self, busy: bool) -> tidelm_engine::expert_map::ExpertMap {
        use tidelm_engine::expert_map::{ExpertMap, LOADING, RESIDENT};
        let e = self.cfg.experts;
        let layers: Vec<usize> = self.moe_layers().to_vec();
        let mut state = Vec::with_capacity(layers.len() * e);
        for &l in &layers {
            state.extend(self.lut_h[l * e..(l + 1) * e].iter().map(|&s| if s >= 0 { RESIDENT } else { 0 }));
        }
        let mut mark = |l: usize, x: usize| {
            if let Ok(r) = layers.binary_search(&l) {
                state[r * e + x] = LOADING;
            }
        };
        if let Some(a) = self.hot.as_ref().and_then(|h| h.asynch.as_ref()) {
            for &(_, l, x, _) in &a.pending {
                mark(l, x);
            }
        }
        if let Some(a) = &self.adapt_bg {
            for &(_, l, x, _) in &a.pending {
                mark(l, x);
            }
        }
        let mass = self.res.db_mass.as_deref().unwrap_or(&self.res.score);
        let demand = (mass.len() >= self.cfg.layers * e).then(|| layers.iter().flat_map(|&l| mass[l * e..(l + 1) * e].iter().copied()).collect());
        let s = &self.res.stats;
        let mut counters = vec![("experts moved", s.swaps as f64)];
        if self.hot.is_some() {
            counters.push(("hot reads", s.hot_reads as f64));
            counters.push(("hot reads kept", s.hot_kept as f64));
            counters.push(("inline rescues", s.hot_rescue as f64));
        }
        if s.preload > 0 {
            counters.push(("preloaded", s.preload as f64));
        }
        // the last six seconds' events (the page polls each second and replays them at their own spacing)
        let events = self.res.events.recent(6.0, 4000);
        ExpertMap { layers, experts: e, state, demand, slots: self.res.owner.len(), spare: self.hot.is_some() as usize, world: self.world, busy,
                    counters, events }
    }

    pub(crate) fn push_lut(&mut self) -> Result<()> {
        for (a, &s) in self.allowed_h.iter_mut().zip(&self.lut_h) {
            *a = (s >= 0) as u8;
        }
        self.lut_upload()
    }

    /// adapt() (rank 0, mirrored): fold the demand into the score and the history, plan the
    /// swaps toward it; both ranks apply the plan.
    pub fn adapt(&mut self, max_swaps: usize) -> Result<usize> {
        let t0 = Instant::now();
        self.hot_install()?;
        self.adapt_install(true)?;
        let plan = self.adapt_plan(max_swaps)?;
        let flat: Vec<u32> = plan.iter().flat_map(|&(a, b, c, d)| [a as u32, b as u32, c as u32, d as u32]).collect();
        self.mirror_msg(|m| {
            m.u32(OP_ADAPT).u32s(&flat);
        })?;
        self.adapt_apply(&plan)?;
        self.log_plan(&plan, tidelm_engine::expert_map::EV_ADAPT);
        self.save_db()?;
        self.res.stats.adapt_s += t0.elapsed().as_secs_f64();
        Ok(plan.len())
    }

    fn log_plan(&mut self, plan: &[(usize, usize, usize, usize)], kind: u8) {
        for &(l, c, vl, v) in plan {
            self.res.events.push(l, c, kind, 0.0, Some((vl, v)));
        }
    }

    /// adapt() with the newcomers loaded in the background (rank 0, mirrored): the victims stop
    /// being routable now, the newcomers from the ADAPT_BG_PASSES-th decoder pass on.
    pub fn adapt_background(&mut self, max_swaps: usize) -> Result<usize> {
        let t0 = Instant::now();
        self.hot_install()?;
        self.adapt_install(true)?;
        let plan = self.adapt_plan(max_swaps)?;
        let flat: Vec<u32> = plan.iter().flat_map(|&(a, b, c, d)| [a as u32, b as u32, c as u32, d as u32]).collect();
        self.mirror_msg(|m| {
            m.u32(OP_ADAPT_BG).u32s(&flat);
        })?;
        self.adapt_begin(&plan)?;
        self.log_plan(&plan, tidelm_engine::expert_map::EV_ADAPT);
        self.save_db()?;
        self.res.stats.adapt_s += t0.elapsed().as_secs_f64();
        Ok(plan.len())
    }

    /// Session-demand preloading (rank 0, mirrored through OP_ADAPT as the boundary pass): the
    /// conversation's accumulated votes planned and applied BEFORE its continuation's suffix
    /// prefills, so the suffix computes on the experts it will demand instead of substitutes.
    /// Nothing is folded into any history and no live demand is read here -- the request's own
    /// votes still fold at their usual sites; this only times the loads the prefill->decode
    /// boundary pass would otherwise pay for after the prefill. That pass then plans less:
    /// the preloaded experts are already resident, so the reads the request pays stay about
    /// what they were, reordered ahead of the compute they exist for.
    pub fn preload(&mut self, conv: &[f64], max_swaps: usize) -> Result<usize> {
        let t0 = Instant::now();
        self.hot_install()?;
        self.adapt_install(true)?;
        let plan = self.plan_preload(conv, max_swaps);
        let mb: u64 = plan
            .iter()
            .map(|&(l, x, _, _)| self.pack.record(l, x).map(|r| r.nbytes as u64).unwrap_or(0))
            .sum();
        let flat: Vec<u32> = plan
            .iter()
            .flat_map(|&(a, b, c, d)| [a as u32, b as u32, c as u32, d as u32])
            .collect();
        self.mirror_msg(|m| {
            m.u32(OP_ADAPT).u32s(&flat);
        })?;
        self.adapt_apply(&plan)?;
        self.log_plan(&plan, tidelm_engine::expert_map::EV_PRELOAD);
        self.res.stats.preload += plan.len();
        self.res.stats.preload_mb += mb;
        self.res.stats.preload_s += t0.elapsed().as_secs_f64();
        Ok(plan.len())
    }

    /// Whether the conversation's votes name any expert that is not resident (rank 0; the cheap
    /// gate before preload(): an empty plan would still mirror and reset the demand buffers, and
    /// a conversation whose demand is all resident has nothing to time).
    pub fn preload_wanted(&self, conv: &[f64]) -> bool {
        let e = self.cfg.experts;
        self.res
            .swap_layers
            .iter()
            .any(|&l| (l * e..(l + 1) * e).any(|i| conv[i] > 0.0 && self.lut_h[i] < 0))
    }

    /// Rank 0: the preload plan -- candidates only this conversation's accumulated votes name
    /// (preload_candidates), victims the globally weakest residents it did not itself name, the
    /// per-layer floor and the width classes as plan_swaps' (a newcomer takes a slot of its own
    /// class). No margin check: conv shares are per-layer normalized and res.score is decayed
    /// demand, not comparable scales -- the protections are the budget, the floor, and that
    /// every candidate is demand this conversation voiced.
    fn plan_preload(&self, conv: &[f64], max_swaps: usize) -> Vec<(usize, usize, usize, usize)> {
        let e = self.cfg.experts;
        let floor = 2 * self.cfg.topk;
        let sc = &self.res.score;
        let layers = &self.res.swap_layers;
        let res = |l: usize, x: usize| self.lut_h[l * e + x] >= 0;
        let cand = preload_candidates(conv, &res, layers, e);
        let class = |l: usize, x: usize| -> usize {
            let b = self
                .pack
                .records
                .get(&(l, x))
                .map(|r| r.bits)
                .unwrap_or([0; 3]);
            self.arena
                .classes
                .iter()
                .position(|c| c.bits == b)
                .unwrap_or(usize::MAX)
        };
        // victims weakest first, a list a width class; a conversation's own residents stay
        let mut vict: Vec<Vec<(usize, usize)>> = vec![vec![]; self.arena.classes.len()];
        for (l, x) in layers
            .iter()
            .flat_map(|&l| (0..e).map(move |x| (l, x)))
            .filter(|&(l, x)| res(l, x) && conv[l * e + x] <= 0.0)
        {
            let s = self.lut_h[l * e + x] as usize;
            vict[self.arena.class_of[s]].push((l, x));
        }
        for v in vict.iter_mut() {
            v.sort_by(|a, b| {
                sc[a.0 * e + a.1]
                    .partial_cmp(&sc[b.0 * e + b.1])
                    .unwrap()
                    .then(a.cmp(b))
            });
        }
        let mut left: std::collections::HashMap<usize, usize> = layers
            .iter()
            .map(|&l| (l, (0..e).filter(|&x| res(l, x)).count()))
            .collect();
        let (mut plan, mut vi) = (vec![], vec![0usize; vict.len()]);
        let mut done = vec![false; vict.len()];
        for (cl, ce) in cand {
            if plan.len() >= max_swaps || done.iter().all(|&d| d) {
                break;
            }
            let c = class(cl, ce);
            if c >= vict.len() || done[c] {
                continue; // no slot of its width (or its class is settled)
            }
            let mut found = None;
            while vi[c] < vict[c].len() {
                let (vl, ve) = vict[c][vi[c]];
                vi[c] += 1;
                if left[&vl] > floor {
                    found = Some((vl, ve));
                    break;
                }
            }
            let Some((vl, ve)) = found else {
                done[c] = true;
                continue;
            };
            *left.get_mut(&vl).unwrap() -= 1;
            *left.get_mut(&cl).unwrap() += 1;
            plan.push((cl, ce, vl, ve));
        }
        plan
    }

    /// Rank 0: the demand folded into the score and the history, the swaps planned toward it.
    fn adapt_plan(&mut self, max_swaps: usize) -> Result<Vec<(usize, usize, usize, usize)>> {
        let demand = self.read_demand()?;
        if let Some(m) = self.res.db_mass.as_mut() {
            for (a, d) in m.iter_mut().zip(&demand) {
                *a += d;
            }
            self.res.db_tokens += self.res.demand_tokens;
        }
        self.res.demand_tokens = 0;
        for (s, d) in self.res.score.iter_mut().zip(&demand) {
            *s = *s * ADAPT_DECAY + d;
        }
        Ok(self.plan_swaps(max_swaps))
    }

    /// Both ranks: the demand restarts; each victim leaves routing (its slot is about to be
    /// overwritten: the stream is idle, and the next pass sees its lut entry first) and its slot's
    /// newcomer starts loading on the reader threads.
    pub(crate) fn adapt_begin(&mut self, plan: &[(usize, usize, usize, usize)]) -> Result<()> {
        let e = self.cfg.experts;
        self.demand.zero_async(self.stream())?;
        self.res.bg_last.iter_mut().for_each(|x| *x = 0.0);
        self.stream().synchronize()?;
        if self.adapt_bg.is_none() {
            self.adapt_bg = Some(AdaptAsync { jobs: readers(&self.pack, ADAPT_READERS)?, pending: vec![],
                                              newcomers: Default::default(), install_at: 0 });
        }
        for &(l, c, vl, v) in plan {
            let slot = self.lut_h[vl * e + v];
            if slot < 0 {
                return Err(format!("adapt: victim ({vl}, {v}) is not resident").into());
            }
            let slot = slot as usize;
            let r = self.pack.record(l, c)?;
            if r.bits != self.arena.bits(slot) {
                return Err(format!("adapt: expert ({l}, {c}) is {:?} bits, its slot {slot} {:?}", r.bits, self.arena.bits(slot)).into());
            }
            self.lut_h[vl * e + v] = -1;
            self.allowed_h[vl * e + v] = 0;
            self.lut_entry(vl * e + v)?;
            self.res.owner[slot] = Some((l, c));
            let (tx, rx) = mpsc::channel();
            let job = ReadJob { off: r.off, nbytes: r.nbytes, dsts: self.arena.slot_dsts(slot), done: tx };
            let a = self.adapt_bg.as_mut().unwrap();
            a.jobs.send(job).map_err(|_| "adapt: the reader threads are gone")?;
            a.pending.push((slot, l, c, rx));
            a.newcomers.insert(l * e + c);
        }
        let at = self.passes + ADAPT_BG_PASSES;
        self.adapt_bg.as_mut().unwrap().install_at = at;
        self.res.stats.swaps += plan.len();
        Ok(())
    }

    /// Both ranks, before every decoder pass (`force`: before an adaptation): a background
    /// adaptation's newcomers made routable once its pass has come, waiting for reads not done.
    pub(crate) fn adapt_install(&mut self, force: bool) -> Result<()> {
        let e = self.cfg.experts;
        let passes = self.passes;
        let Some(a) = self.adapt_bg.as_mut() else { return Ok(()) };
        if a.pending.is_empty() || (!force && passes < a.install_at) {
            return Ok(());
        }
        let pending = std::mem::take(&mut a.pending);
        a.newcomers.clear();
        let t0 = Instant::now();
        for (_, l, x, rx) in &pending {
            // a read that failed on one rank would leave the pair routing differently: fatal
            rx.recv().map_err(|_| "adapt: the reader threads are gone")?.map_err(|err| format!("adapt read ({l}, {x}): {err}"))?;
        }
        self.res.stats.adapt_wait_s += t0.elapsed().as_secs_f64();
        for (slot, l, x, _) in pending {
            let i = l * e + x;
            self.lut_h[i] = slot as i32;
            self.allowed_h[i] = 1;
            self.lut_entry(i)?;
        }
        Ok(())
    }

    /// Whether (layer, expert) is a background adaptation's newcomer still loading.
    fn adapt_pending(&self, i: usize) -> bool {
        self.adapt_bg.as_ref().is_some_and(|a| a.newcomers.contains(&i))
    }

    /// Both ranks: the demand restarts, the plan's experts load.
    pub(crate) fn adapt_apply(&mut self, plan: &[(usize, usize, usize, usize)]) -> Result<()> {
        self.demand.zero_async(self.stream())?;
        self.res.bg_last.iter_mut().for_each(|x| *x = 0.0);
        self.apply_swaps(plan)?;
        self.res.stats.swaps += plan.len();
        Ok(())
    }

    pub(crate) fn follow_adapt(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        self.hot_install()?;
        self.adapt_install(true)?;
        let plan: Vec<(usize, usize, usize, usize)> =
            r.u32s()?.chunks(4).map(|c| (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize)).collect();
        self.adapt_apply(&plan)
    }

    pub(crate) fn follow_adapt_bg(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        self.hot_install()?;
        self.adapt_install(true)?;
        let plan: Vec<(usize, usize, usize, usize)> =
            r.u32s()?.chunks(4).map(|c| (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize)).collect();
        self.adapt_begin(&plan)
    }

    pub(crate) fn follow_bg(&mut self, r: &mut tidelm_comm::Reader) -> Result<()> {
        let q: Vec<(usize, usize, usize)> = r.u32s()?.chunks(3).map(|c| (c[0] as usize, c[1] as usize, c[2] as usize)).collect();
        self.res.bg_period += 1;
        self.res.hot_q = q.into_iter().rev().collect();
        Ok(())
    }

    /// Rank 0: tp.adapt's cross-layer plan over the swap layers' scores.
    fn plan_swaps(&self, max_swaps: usize) -> Vec<(usize, usize, usize, usize)> {
        let e = self.cfg.experts;
        let floor = 2 * self.cfg.topk;
        let sc = &self.res.score;
        let layers = &self.res.swap_layers;
        let res = |l: usize, x: usize| self.lut_h[l * e + x] >= 0;
        let mut cand: Vec<(usize, usize)> = layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x))).filter(|&(l, x)| !res(l, x)).collect();
        cand.sort_by(|a, b| sc[b.0 * e + b.1].partial_cmp(&sc[a.0 * e + a.1]).unwrap().then(a.cmp(b)));
        // victims weakest first, a list a width class: a newcomer takes a slot of its own class
        let class = |l: usize, x: usize| -> usize {
            let b = self.pack.records.get(&(l, x)).map(|r| r.bits).unwrap_or([0; 3]);
            self.arena.classes.iter().position(|c| c.bits == b).unwrap_or(usize::MAX)
        };
        let mut vict: Vec<Vec<(usize, usize)>> = vec![vec![]; self.arena.classes.len()];
        for (l, x) in layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x))).filter(|&(l, x)| res(l, x)) {
            let s = self.lut_h[l * e + x] as usize;
            vict[self.arena.class_of[s]].push((l, x));
        }
        for v in vict.iter_mut() {
            v.sort_by(|a, b| sc[a.0 * e + a.1].partial_cmp(&sc[b.0 * e + b.1]).unwrap().then(a.cmp(b)));
        }
        let mut left: std::collections::HashMap<usize, usize> =
            layers.iter().map(|&l| (l, (0..e).filter(|&x| res(l, x)).count())).collect();
        let (mut plan, mut vi) = (vec![], vec![0usize; vict.len()]);
        let mut done = vec![false; vict.len()];
        for (cl, ce) in cand {
            if plan.len() >= max_swaps || sc[cl * e + ce] <= 0.0 || done.iter().all(|&d| d) {
                break;
            }
            let c = class(cl, ce);
            if c >= vict.len() || done[c] {
                continue; // no slot of its width (or its class is settled)
            }
            let mut found = None;
            while vi[c] < vict[c].len() {
                let (vl, ve) = vict[c][vi[c]];
                vi[c] += 1;
                if left[&vl] > floor {
                    found = Some((vl, ve));
                    break;
                }
            }
            let Some((vl, ve)) = found else {
                done[c] = true;
                continue;
            };
            if sc[cl * e + ce] <= sc[vl * e + ve] * ADAPT_MARGIN {
                // candidates only weaken from here: this class is settled
                done[c] = true;
                continue;
            }
            *left.get_mut(&vl).unwrap() -= 1;
            *left.get_mut(&cl).unwrap() += 1;
            plan.push((cl, ce, vl, ve));
        }
        plan
    }

    pub fn save_db(&self) -> Result<()> {
        if let (Some(m), Some(p)) = (&self.res.db_mass, &self.res.db_path) {
            crate::residency::write_mass(p, m, self.cfg.n_layers(), self.cfg.experts, self.res.db_tokens)?;
        }
        Ok(())
    }

    /// bg_step (rank 0, mirrored): refresh the recent mass, queue the victims kept hot reads
    /// take.
    pub fn bg_step(&mut self, keep: usize) -> Result<()> {
        let e = self.cfg.experts;
        let demand = self.read_demand()?;
        for i in 0..demand.len() {
            let w = demand[i] - self.res.bg_last[i];
            self.res.bg_last[i] = demand[i];
            self.res.bg_score[i] = self.res.bg_score[i] * BG_DECAY + w;
            self.res.bg_slow[i] = self.res.bg_slow[i] * BG_SLOW + w;
        }
        let period = self.res.bg_period + 1;
        let floor = 2 * self.cfg.topk;
        let layers = self.res.swap_layers.clone();
        let mut left: std::collections::HashMap<usize, usize> =
            layers.iter().map(|&l| (l, (0..e).filter(|&x| self.lut_h[l * e + x] >= 0).count())).collect();
        let slow = self.res.victim_slow;
        let mut cold: Vec<(f64, f64, usize, usize)> = layers.iter().flat_map(|&l| (0..e).map(move |x| (l, x)))
            .filter(|&(l, x)| {
                let i = l * e + x;
                self.lut_h[i] >= 0 && self.res.bg_score[i] < BG_COLD && self.res.bg_until[i] <= period
            })
            .map(|(l, x)| (if slow { self.res.bg_slow[l * e + x] } else { 0.0 }, self.res.score[l * e + x] + demand[l * e + x], l, x))
            .collect();
        cold.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()).then((a.2, a.3).cmp(&(b.2, b.3))));
        // up to `keep` victims a width class (a hot read keeps its expert only in its own class)
        let mut q = vec![];
        let mut per = vec![0usize; self.arena.classes.len()];
        for (_, _, l, x) in cold {
            let s = self.lut_h[l * e + x] as usize;
            let c = self.arena.class_of[s];
            if per[c] < keep && left[&l] > floor {
                *left.get_mut(&l).unwrap() -= 1;
                per[c] += 1;
                q.push((l, x, s));
            }
        }
        let flat: Vec<u32> = q.iter().flat_map(|&(a, b, c)| [a as u32, b as u32, c as u32]).collect();
        self.mirror_msg(|m| {
            m.u32(OP_BG).u32s(&flat);
        })?;
        let mut r_msg = tidelm_comm::Msg::default();
        r_msg.u32s(&flat);
        self.follow_bg(&mut tidelm_comm::Reader::new(&r_msg.0))
    }

    /// The pass's miss counters [picks, missed, mass, missed mass] (rank-local; replicated
    /// routing makes them the same on both ranks).
    pub fn miss_totals(&self) -> Result<[f64; 4]> {
        self.stream().synchronize()?;
        let v = self.miss.download()?;
        Ok([v[0], v[1], v[2], v[3]])
    }

    /// Between a decode pass's segments, after MoE layer l's exact routing: release the previous
    /// layer's hot read (kept over a queued victim, or dropped), then read l's heaviest missed
    /// pick if it qualifies.
    pub(crate) fn hot_point(&mut self, l: usize, t: usize, idx: usize, n_layers: usize) -> Result<()> {
        self.hot_release()?;
        let k = self.cfg.topk;
        let e = self.cfg.experts;
        let (min_share, per_step, slot, rescue) = match self.hot.as_ref() {
            // GLM_HOT_RESCUE (beside the async plan): the rescue threshold, RESCUE_STEP a pass
            // with no linear pacing - a first occurrence can be any layer - and the heaviest
            // *missed* pick (the sync path below takes the heaviest pick and gives up when it
            // is resident; measured behavior, untouched)
            Some(h) => match h.rescue {
                Some(th) => (th, RESCUE_STEP, h.slot, true),
                None => (h.min_share, h.per_step, h.slot, false),
            },
            None => return Ok(()),
        };
        let reads = self.hot.as_ref().unwrap().reads;
        // linear pacing: the i-th of n MoE layers may bring the step's reads to ceil(cap * (i+1)/n)
        let room = if rescue {
            per_step.saturating_sub(reads)
        } else {
            per_step.saturating_sub(reads).min((per_step * (idx + 1)).div_ceil(n_layers).saturating_sub(reads))
        };
        if room == 0 {
            return Ok(());
        }
        self.stream().synchronize()?;
        let hot = self.hot.as_ref().unwrap();
        let mut best: Option<(f32, usize)> = None;
        for i in 0..t * k {
            let (v, x) = (hot.host[2 * i], hot.host[2 * i + 1] as usize);
            if v >= min_share && (!rescue || (self.lut_h[l * e + x] < 0 && !self.adapt_pending(l * e + x)))
                && best.is_none_or(|(bv, bx)| v > bv || (v == bv && x < bx)) {
                best = Some((v, x));
            }
        }
        let Some((share, x)) = best else { return Ok(()) };
        if self.lut_h[l * e + x] >= 0 || self.adapt_pending(l * e + x) {
            return Ok(());
        }
        let t0 = Instant::now();
        {
            let h = self.hot.as_mut().unwrap();
            let r = self.pack.read(l, x, &mut h.buf)?;
            // the transient slot is as wide as any expert: it takes this one's widths
            self.arena.set_widths(slot, r.bits, &self.st)?;
            let d = self.arena.slot_dsts(slot);
            let mut off = 0;
            let sz = crate::arena::record_sizes(r.bits, self.world, self.cfg.hidden, self.cfg.moe_inter);
            for t in 0..9 {
                unsafe {
                    tidelm_cuda::memcpy_async(d[t].0 as *mut c_void, h.buf[off..].as_ptr() as *const c_void, sz[t],
                                              tidelm_cuda::ffi::MEMCPY_HOST_TO_DEVICE, &self.st)?;
                }
                off += sz[t];
            }
            h.reads += 1;
            h.live = Some((l, x));
            h.live_cause = (share, rescue);
        }
        self.lut_h[l * e + x] = slot as i32;
        self.allowed_h[l * e + x] = 1;
        self.lut_entry(l * e + x)?;
        self.res.stats.hot_reads += 1;
        if rescue {
            self.res.stats.hot_rescue += 1;
        }
        self.res.stats.hot_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// _hot_keep / _pf_release for the read of the layer just computed.
    pub(crate) fn hot_release(&mut self) -> Result<()> {
        let e = self.cfg.experts;
        let Some(h) = self.hot.as_mut() else { return Ok(()) };
        let Some((l, x)) = h.live.take() else { return Ok(()) };
        let (share, rescue) = h.live_cause;
        let kind = if rescue { tidelm_engine::expert_map::EV_RESCUE } else { tidelm_engine::expert_map::EV_HOT };
        let src = h.slot;
        let i = l * e + x;
        let bits = self.pack.record(l, x)?.bits;
        while let Some((vl, ve, vs)) = self.take_victim(bits) {
            self.arena.copy_slot(vs, src, self.stream())?;
            self.lut_h[vl * e + ve] = -1;
            self.allowed_h[vl * e + ve] = 0;
            self.res.owner[vs] = Some((l, x));
            self.lut_h[i] = vs as i32;
            self.allowed_h[i] = 1;
            self.res.bg_until[i] = self.res.bg_period + BG_PROTECT;
            self.lut_entry(vl * e + ve)?;
            self.lut_entry(i)?;
            self.res.stats.hot_kept += 1;
            self.res.events.push(l, x, kind, share, Some((vl, ve)));
            return Ok(());
        }
        self.lut_h[i] = -1;
        self.allowed_h[i] = 0;
        self.lut_entry(i)?;
        self.res.stats.hot_unkept += 1;
        self.res.events.push(l, x, tidelm_engine::expert_map::EV_DROPPED, share, None);
        Ok(())
    }

    /// A new pass: no reads yet.
    pub(crate) fn hot_begin(&mut self) {
        if let Some(h) = self.hot.as_mut() {
            h.reads = 0;
        }
    }

    /// Async (both ranks), after a decode pass of `t` rows: per MoE layer its heaviest missed
    /// pick at or above the threshold (ties to the lower expert), the heaviest of those (ties to
    /// the lower layer) up to the step cap, each taking the next queued victim that still holds
    /// its slot. The victim stops being routable now; the reads start on the reader threads.
    /// With GLM_HOT_PREDICT the candidates are HotPredict's instead (the same cap and victims).
    pub(crate) fn hot_plan(&mut self, ids: &[u32], pos0: usize) -> Result<()> {
        let t = ids.len();
        let (e, k) = (self.cfg.experts, self.cfg.topk);
        let Some(h) = self.hot.as_ref() else { return Ok(()) };
        if h.asynch.is_none() || !self.hot_plan_on {
            return Ok(());
        }
        let w = crate::weights::MAX_ROWS * k * 2;
        let per_step = h.per_step;
        let min_share = h.min_share;
        let moe: Vec<usize> = self.moe_layers().to_vec();
        // (tier, score, layer, expert): the engine's own candidates tier 0, the tally's tier 1
        let mut cands: Vec<(u8, f32, usize, usize)> = Vec::new();
        // the predictor's state is folded with the table first (both ranks, same order)
        let mut predict = self.hot.as_mut().unwrap().asynch.as_mut().unwrap().predict.take();
        {
            let a = self.hot.as_ref().unwrap().asynch.as_ref().unwrap();
            for (i, &l) in moe.iter().enumerate() {
                let row = &a.table[i * w..i * w + t * k * 2];
                let mut best: Option<(f32, usize)> = None;
                if predict.as_ref().is_none_or(|p| p.react) {
                    for j in 0..t * k {
                        let (v, x) = (row[2 * j], row[2 * j + 1] as usize);
                        if v >= min_share && self.lut_h[l * e + x] < 0 && !self.adapt_pending(l * e + x)
                            && best.is_none_or(|(bv, bx)| v > bv || (v == bv && x < bx)) {
                            best = Some((v, x));
                        }
                    }
                }
                if let Some((v, x)) = best {
                    cands.push((0, v, l, x));
                }
                if let Some(p) = predict.as_mut() {
                    p.fold(i, e, row);
                    let ok = |x: usize| self.lut_h[l * e + x] < 0 && !self.adapt_pending(l * e + x)
                        && best.is_none_or(|(_, bx)| bx != x);
                    for (v, x) in p.candidates(i, e, ok) {
                        cands.push((1, v, l, x));
                    }
                }
            }
        }
        // TIDELM_GLM_HOT_ORACLE (debug): the recorded run's heavy misses of the next pass, first
        {
            let a = self.hot.as_ref().unwrap().asynch.as_ref().unwrap();
            if let Some(o) = a.oracle.as_ref() {
                for &(v, l, x) in o.next(a.plans, t)? {
                    if self.lut_h[l * e + x] < 0 && !self.adapt_pending(l * e + x) {
                        cands.retain(|c| (c.2, c.3) != (l, x));
                        cands.push((0, 10.0 + v, l, x));
                    }
                }
            }
        }
        // stable: equal shares keep layer order
        cands.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.partial_cmp(&a.1).unwrap()));
        cands.truncate(per_step);
        let mut traced: Vec<u16> = Vec::new();
        for (tier, share, l, x) in cands {
            let r = self.pack.record(l, x)?;
            let Some((vl, ve, vs)) = self.take_victim(r.bits) else {
                self.res.stats.hot_unkept += 1;
                continue;
            };
            if let Some(p) = predict.as_mut() {
                p.taken(moe.binary_search(&l).unwrap(), e, x);
            }
            traced.extend([l as u16, x as u16, vl as u16, ve as u16]);
            self.lut_h[vl * e + ve] = -1;
            self.allowed_h[vl * e + ve] = 0;
            self.lut_entry(vl * e + ve)?;
            self.res.owner[vs] = Some((l, x));
            let (tx, rx) = mpsc::channel();
            let job = ReadJob { off: r.off, nbytes: r.nbytes, dsts: self.arena.slot_dsts(vs), done: tx };
            let a = self.hot.as_mut().unwrap().asynch.as_mut().unwrap();
            a.jobs.send(job).map_err(|_| "hot read: the reader threads are gone")?;
            a.pending.push((vs, l, x, rx));
            self.res.stats.hot_reads += 1;
            let kind = if tier == 1 { tidelm_engine::expert_map::EV_PREDICT } else { tidelm_engine::expert_map::EV_HOT };
            self.res.events.push(l, x, kind, if tier == 1 { 0.0 } else { share.min(1.0) }, Some((vl, ve)));
        }
        let a = self.hot.as_mut().unwrap().asynch.as_mut().unwrap();
        a.predict = predict;
        a.plans += 1;
        if a.trace.is_some() {
            self.hot_trace_pass(ids, pos0, &traced)?;
        }
        Ok(())
    }

    /// TIDELM_GLM_HOT_TRACE: one pass appended - u32 rows, (GHT2:) u32 first row's position and
    /// each row's u32 input token id, the table's rows of every MoE layer (f32 share, f32 expert pairs), u32 reads, then each
    /// read's u16 (layer, expert, victim layer, victim expert). The file opens with the header
    /// hot_trace_open wrote. The ids and position are for measuring predictors of a pass's routing
    /// from what is known before it runs: its own tokens (an MTP draft's), and the routing of the
    /// tokens before it in its sequence (the KV cache's) - unlike its hidden states.
    fn hot_trace_pass(&mut self, ids: &[u32], pos0: usize, reads: &[u16]) -> Result<()> {
        use std::io::Write;
        let t = ids.len();
        let k = self.cfg.topk;
        let w = crate::weights::MAX_ROWS * k * 2;
        let n = self.moe_layers().len();
        let a = self.hot.as_mut().unwrap().asynch.as_mut().unwrap();
        let mut b: Vec<u8> = Vec::with_capacity(8 + t * 4 + n * t * k * 8 + reads.len() * 2);
        b.extend((t as u32).to_le_bytes());
        b.extend((pos0 as u32).to_le_bytes());
        for &id in ids {
            b.extend(id.to_le_bytes());
        }
        for i in 0..n {
            for v in &a.table[i * w..i * w + t * k * 2] {
                b.extend(v.to_le_bytes());
            }
        }
        b.extend(((reads.len() / 4) as u32).to_le_bytes());
        for v in reads {
            b.extend(v.to_le_bytes());
        }
        a.trace.as_mut().unwrap().write_all(&b)?;
        Ok(())
    }

    /// The trace's header: magic "GHT2" ("GHT1": no token ids), u32 MoE layers, experts, topk, each MoE layer's u32
    /// index, then the residency at the start, u8 a (MoE layer, expert).
    fn hot_trace_open(&mut self, path: &str) -> Result<()> {
        use std::io::Write;
        let e = self.cfg.experts;
        let moe: Vec<usize> = self.moe_layers().to_vec();
        let mut b: Vec<u8> = b"GHT2".to_vec();
        for v in [moe.len(), e, self.cfg.topk] {
            b.extend((v as u32).to_le_bytes());
        }
        for &l in &moe {
            b.extend((l as u32).to_le_bytes());
        }
        for &l in &moe {
            b.extend((0..e).map(|x| (self.lut_h[l * e + x] >= 0) as u8));
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&b)?;
        self.hot.as_mut().unwrap().asynch.as_mut().unwrap().trace = Some(f);
        eprintln!("{} hot trace: {path}", self.cfg.arch.tag());
        Ok(())
    }

    /// TIDELM_GLM_VERIFY_PLAN=N (debug, both ranks): before the decode pass that follows the
    /// N-th async plan, every resident slot checked against the pack byte for byte, every
    /// owner against the LUT, and the device LUT / allowed against the host's. Changes nothing.
    pub fn verify_residency(&mut self) -> Result<()> {
        let want: usize = match std::env::var("TIDELM_GLM_VERIFY_PLAN").ok().and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => return Ok(()),
        };
        if self.hot.as_ref().and_then(|h| h.asynch.as_ref()).is_none_or(|a| a.plans != want) {
            return Ok(());
        }
        let t0 = Instant::now();
        self.stream().synchronize()?;
        let e = self.cfg.experts;
        let (lut, allowed) = (self.lut.download()?, self.allowed.download()?);
        let lut_bad = (0..lut.len()).filter(|&i| lut[i] != self.lut_h[i]).count();
        let al_bad = (0..allowed.len()).filter(|&i| allowed[i] != self.allowed_h[i]).count();
        let mut own_bad = 0;
        for (i, &s) in self.lut_h.iter().enumerate() {
            if s >= 0 && self.res.owner.get(s as usize).copied().flatten() != Some((i / e, i % e)) {
                own_bad += 1;
                if own_bad <= 5 {
                    eprintln!("verify: lut ({}, {}) -> slot {s}, whose owner is {:?}", i / e, i % e, self.res.owner[s as usize]);
                }
            }
        }
        let mut buf = self.pack.buffer()?;
        let (mut checked, mut bad) = (0usize, 0usize);
        for s in 0..self.arena.slots - 1 {
            let Some((l, x)) = self.res.owner[s] else { continue };
            if self.lut_h[l * e + x] != s as i32 {
                continue; // a victim whose read is in flight, or a stale owner: counted above if routed
            }
            let r = self.pack.read(l, x, &mut buf)?;
            if r.bits != self.arena.bits(s) {
                bad += 1;
                eprintln!("verify: slot {s} ({l}, {x}) is of widths {:?}, its record {:?}", self.arena.bits(s), r.bits);
                continue;
            }
            let mut off = 0;
            let mut diff = vec![];
            for (t, (dst, n)) in self.arena.slot_dsts(s).into_iter().enumerate() {
                let mut h = vec![0u8; n];
                unsafe {
                    tidelm_cuda::memcpy_async(h.as_mut_ptr() as *mut c_void, dst as *const c_void, n,
                                              tidelm_cuda::ffi::MEMCPY_DEVICE_TO_HOST, self.stream())?;
                }
                self.stream().synchronize()?;
                if h[..] != buf[off..off + n] {
                    diff.push(t);
                }
                off += n;
            }
            checked += 1;
            if !diff.is_empty() {
                bad += 1;
                if bad <= 20 {
                    eprintln!("verify: slot {s} ({l}, {x}) differs from the pack in tensors {diff:?}");
                }
            }
        }
        eprintln!("verify (plan {want}, rank {}): {checked} resident slots read back, {bad} differ; lut {lut_bad} / allowed {al_bad} \
                   entries differ from the host's; {own_bad} lut entries not owned; {:.1} s", self.rank, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// The weakest queued victim of width `bits` that still holds its slot (bg_step queues them
    /// weakest last); stale entries of that class are dropped on the way.
    fn take_victim(&mut self, bits: crate::arena::Bits) -> Option<(usize, usize, usize)> {
        let q = &mut self.res.hot_q;
        let mut i = q.len();
        while i > 0 {
            i -= 1;
            let (vl, ve, vs) = q[i];
            if self.arena.bits(vs) != bits {
                continue;
            }
            q.remove(i);
            if self.res.owner[vs] == Some((vl, ve)) {
                return Some((vl, ve, vs));
            }
        }
        None
    }

    /// Async (both ranks), before a decoder pass or an adaptation: the planned reads waited for
    /// and made routable, protected from the next background periods as a kept read is.
    pub(crate) fn hot_install(&mut self) -> Result<()> {
        let e = self.cfg.experts;
        let Some(a) = self.hot.as_mut().and_then(|h| h.asynch.as_mut()) else { return Ok(()) };
        if a.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut a.pending);
        let t0 = Instant::now();
        for (_, l, x, rx) in &pending {
            // a read that failed on one rank would leave the pair routing differently: fatal
            rx.recv().map_err(|_| "hot read: the reader threads are gone")?
                .map_err(|err| format!("hot read ({l}, {x}): {err}"))?;
        }
        self.res.stats.hot_wait_s += t0.elapsed().as_secs_f64();
        for (vs, l, x, _) in pending {
            let i = l * e + x;
            self.lut_h[i] = vs as i32;
            self.allowed_h[i] = 1;
            self.lut_entry(i)?;
            self.res.bg_until[i] = self.res.bg_period + BG_PROTECT;
            self.res.stats.hot_kept += 1;
        }
        Ok(())
    }

    /// Hot reads on: synchronous between a decode pass's segments, or `asynch` (both need the
    /// background period's victims to keep a read). `predict`: GLM_HOT_PREDICT's spec;
    /// `rescue`: GLM_HOT_RESCUE's threshold (""/"0": off).
    pub fn enable_hot(&mut self, asynch: bool, predict: &str, rescue: &str) -> Result<()> {
        // the graphed pass runs up to MAX_ROWS rows and the segment path copies every row's
        // candidates into the host buffer: sized by VERIFY_ROWS it overran 512 bytes at 9-16
        // rows and cudaMemcpyAsync failed with invalid argument (the sync path's latent bug,
        // 2026-11-08; the async path never copies per layer, which hid it)
        let rows = crate::weights::MAX_ROWS;
        // the decoder arena's last slot is the transient one (forward.rs class_arena)
        let mut h = Hot::new(self.arena.slots - 1, self.pack.max_record, rows, self.cfg.topk)?;
        let p = HotPredict::parse(predict, self.moe_layers().len(), self.cfg.experts)?;
        if p.is_some() && !asynch {
            return Err("GLM_HOT_PREDICT plans async hot reads: it needs GLM_HOT_ASYNC=1".into());
        }
        let r = rescue.trim();
        let r = if r.is_empty() || r == "0" {
            None
        } else {
            let th: f32 = r.parse().map_err(|_| format!("GLM_HOT_RESCUE={r}: TH (0 < TH <= 1, a missed pick's share read inline)"))?;
            if !(0.0..=1.0).contains(&th) {
                return Err(format!("GLM_HOT_RESCUE={r}: TH (0 < TH <= 1, a missed pick's share read inline)").into());
            }
            if !asynch {
                return Err("GLM_HOT_RESCUE reads inline beside the async plan: it needs GLM_HOT_ASYNC=1".into());
            }
            Some(th)
        };
        h.rescue = r;
        h.rescue_boot = r;
        if asynch {
            let len = self.moe_layers().len() * crate::weights::MAX_ROWS * self.cfg.topk * 2;
            let mut a = HotAsync::new(&self.pack, h.per_step.min(8), len)?;
            a.predict = p;
            h.asynch = Some(a);
        }
        self.hot = Some(h);
        if asynch {
            // rank 1 traces too (PATH.r1, on its own box): replicated routing means the two
            // files are the same bytes, and a pair that plans apart shows where
            if let Ok(path) = std::env::var("TIDELM_GLM_HOT_TRACE") {
                if !path.is_empty() {
                    let path = if self.rank == 0 { path } else { format!("{path}.r{}", self.rank) };
                    self.hot_trace_open(&path)?;
                }
            }
        }
        let v = std::env::var("TIDELM_GLM_HOT_ORACLE").unwrap_or_default();
        if !v.is_empty() {
            if !asynch {
                return Err("TIDELM_GLM_HOT_ORACLE plans async hot reads: it needs GLM_HOT_ASYNC=1".into());
            }
            let (path, min) = HotOracle::spec(&v, self.rank)?;
            let o = HotOracle::load(&path, min, &self.moe_layers().to_vec(), self.cfg.experts, self.cfg.topk)?;
            let (np, nm) = o.summary();
            eprintln!("{} hot oracle: {path}, misses >= {min}: {nm} in {np} passes", self.cfg.arch.tag());
            self.hot.as_mut().unwrap().asynch.as_mut().unwrap().oracle = Some(o);
        }
        ck(0, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_predict_accumulates_missed_shares_over_rows_and_decays_over_passes() {
        let e = 4usize;
        let mut p = HotPredict::parse("0.5,0.3,2", 2, e).unwrap().unwrap();
        // layer index 1, two rows: expert 2 missed in both (0.2 + 0.15), expert 1 resident
        // (negated, ignored), expert 3 missed once (0.25)
        let row = [0.2, 2.0, -0.4, 1.0, 0.15, 2.0, 0.25, 3.0];
        p.fold(1, e, &row);
        assert!((p.acc[e + 2] - 0.35).abs() < 1e-6);
        assert_eq!(p.acc[e + 1], 0.0);
        assert_eq!(p.acc[e + 3], 0.25);
        assert!(p.acc[..e].iter().all(|&x| x == 0.0));
        // one row's 0.25 is below MIN, the two rows' 0.35 is not: the sum, not the max, ranks
        assert_eq!(p.candidates(1, e, |_| true), vec![(p.acc[e + 2], 2)]);
        // a pass with no misses halves both; another miss of 3 at 0.2 lifts it to 0.325
        p.fold(1, e, &[-0.5, 1.0]);
        p.fold(1, e, &[0.2, 3.0]);
        assert!((p.acc[e + 3] - (0.25 * 0.25 + 0.2)).abs() < 1e-6);
        // `ok` excludes (a resident or a pending newcomer); per_layer caps, heaviest first
        p.acc[e] = 0.9;
        assert_eq!(p.candidates(1, e, |_| true).iter().map(|c| c.1).collect::<Vec<_>>(), vec![0]);
        p.acc[e + 3] = 0.5;
        assert_eq!(p.candidates(1, e, |_| true).iter().map(|c| c.1).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(p.candidates(1, e, |x| x != 0).iter().map(|c| c.1).collect::<Vec<_>>(), vec![3]);
        // a planned read starts its expert over
        p.taken(1, e, 0);
        assert_eq!(p.acc[e], 0.0);
    }

    #[test]
    fn hot_oracle_reads_the_next_passes_heavy_misses_of_a_trace() {
        // 2 MoE layers (3, 5) of 4 experts, top-2; pass 0 one row, pass 1 two rows
        let (moe, e, k) = ([3usize, 5], 4usize, 2usize);
        let mut b: Vec<u8> = b"GHT1".to_vec();
        for v in [2u32, 4, 2, 3, 5] {
            b.extend(v.to_le_bytes());
        }
        b.extend([1u8; 8]);
        let pass = |b: &mut Vec<u8>, t: u32, pairs: &[(f32, f32)], reads: u32| {
            b.extend(t.to_le_bytes());
            for &(v, x) in pairs {
                b.extend(v.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            b.extend(reads.to_le_bytes());
            b.extend(vec![0u8; 8 * reads as usize]);
        };
        pass(&mut b, 1, &[(0.5, 1.0), (-0.4, 2.0), (0.1, 0.0), (0.2, 3.0)], 1);
        // layer 3: expert 2 missed at 0.35 and 0.3 (one entry, its max), a resident 0.6 (negated);
        // layer 5: expert 0 at 0.29 (below MIN)
        pass(&mut b, 2, &[(0.35, 2.0), (-0.6, 1.0), (0.3, 2.0), (0.1, 0.0),
                          (0.29, 0.0), (0.2, 1.0), (0.05, 3.0), (0.01, 2.0)], 0);
        let path = std::env::temp_dir().join(format!("hot_oracle_test_{}.trace", std::process::id()));
        std::fs::write(&path, &b).unwrap();
        let o = HotOracle::load(path.to_str().unwrap(), 0.3, &moe, e, k).unwrap();
        assert_eq!(o.summary(), (2, 2));
        assert_eq!(o.next(0, 1).unwrap(), &[(0.35, 3, 2)]);
        assert!(o.next(1, 2).unwrap().is_empty());
        assert!(o.next(0, 2).is_err());
        assert!(HotOracle::load(path.to_str().unwrap(), 0.3, &[3, 6], e, k).is_err());
        std::fs::write(&path, &b[..b.len() - 3]).unwrap();
        assert!(HotOracle::load(path.to_str().unwrap(), 0.3, &moe, e, k).is_err());
        // the same as GHT2: each pass's first position and row ids after its row count
        // (header 32 bytes; pass 0 at 32: rows, 4 pairs, 1 read, to 80; pass 1 at 80: rows, 8 pairs, 0 reads)
        assert_eq!(b.len(), 152);
        let mut b2: Vec<u8> = b"GHT2".to_vec();
        b2.extend(&b[4..32]);
        b2.extend([1u32, 40, 7].iter().flat_map(|v| v.to_le_bytes()));
        b2.extend(&b[36..80]);
        b2.extend([2u32, 41, 8, 9].iter().flat_map(|v| v.to_le_bytes()));
        b2.extend(&b[84..]);
        std::fs::write(&path, &b2).unwrap();
        let o = HotOracle::load(path.to_str().unwrap(), 0.3, &moe, e, k).unwrap();
        assert_eq!(o.summary(), (2, 2));
        assert_eq!(o.next(0, 1).unwrap(), &[(0.35, 3, 2)]);
        std::fs::remove_file(&path).ok();
        assert_eq!(HotOracle::spec("/a/b.trace,0.4", 1).unwrap(), ("/a/b.trace.r1".to_string(), 0.4));
        assert!(HotOracle::spec("/a/b.trace", 0).is_err());
        assert!(HotOracle::spec("/a/b.trace,0", 0).is_err());
    }

    #[test]
    fn hot_predict_settings_parse_or_refuse() {
        assert!(HotPredict::parse("", 2, 4).unwrap().is_none());
        assert!(HotPredict::parse("0", 2, 4).unwrap().is_none());
        assert!(HotPredict::parse("1.0,0.3,1", 2, 4).is_err());
        assert!(HotPredict::parse("0.5,0,1", 2, 4).is_err());
        assert!(HotPredict::parse("0.5,0.3,0", 2, 4).is_err());
        assert!(HotPredict::parse("0.5,0.3", 2, 4).is_err());
        let p = HotPredict::parse(" 0.25, 0.4, 3 ", 2, 4).unwrap().unwrap();
        assert_eq!((p.decay, p.min, p.per_layer, p.react, p.acc.len()), (0.25, 0.4, 3, false, 8));
        assert!(HotPredict::parse("0.9,1.0,1,1", 2, 4).unwrap().unwrap().react);
        assert!(!HotPredict::parse("0.9,1.0,1,0", 2, 4).unwrap().unwrap().react);
        assert!(HotPredict::parse("0.9,1.0,1,2", 2, 4).is_err());
    }

    #[test]
    fn conv_votes_fold_per_layer_normalized_and_accumulate() {
        let (layers, e) = (2usize, 3usize);
        // a vote concentrated on one expert of layer 1, none of layer 0
        let mut vote = vec![0.0f32; layers * e];
        vote[e + 2] = 5.0;
        let mut acc = vec![0.0f64; layers * e];
        conv_fold_mass(&mut acc, &vote, layers, e);
        assert_eq!(acc[e + 2], 1.0);
        assert!(acc[..e].iter().all(|&x| x == 0.0));
        conv_fold_mass(&mut acc, &vote, layers, e);
        assert_eq!(acc[e + 2], 2.0);
    }

    #[test]
    fn preload_candidates_rank_the_conversations_non_resident_demand() {
        // layer 0's heaviest (0, 2) and layer 1's (1, 0) are resident: neither is a candidate,
        // whatever their mass; the rest rank by accumulated share, descending
        let conv = vec![0.1, 0.0, 0.9, 0.2, 0.0, 0.4, 0.0, 0.3];
        let resident = |l: usize, x: usize| (l, x) == (0, 2) || (l, x) == (1, 0);
        assert_eq!(
            preload_candidates(&conv, &resident, &[0, 1], 4),
            vec![(1, 1), (1, 3), (0, 3), (0, 0)]
        );
    }
}
