//! An OpenAI-compatible HTTP front for the Rust DS v4.1 engine: server/app.py's chat routes,
//! no Python anywhere.
//!
//!   tidelm-server [--host 0.0.0.0] [--port 8000] [--served-model-name deepseek]
//!                 [--lanes 1] [--depth window|confidence|plain|N] [--step-rows 64]
//!                 [--thinking] [--effort 75]
//!                 --resident --graphs [--keep-db FILE --budget N] [--max-seq 32768]
//!                 [--world 2 --rank R --addr HOST:PORT] [--layers N --keep-first K] (tests)
//!
//! Routes: GET /health, GET /v1/models, POST /v1/chat/completions (streamed or not, with
//! usage), POST /v1/debug/prompt (the rendered prompt and its ids, no engine work - what
//! bench/bench.py's token_len calls), POST /v1/messages and /v1/messages/count_tokens (the
//! Anthropic API, anthropic.rs), and for GLM POST /v1/completions. Requests run through the
//! engine's continuous-batching scheduler, one lane each; with --world 2, rank 1 follows rank
//! 0's engine calls and serves nothing. Ctx::family picks the chat format and response rules:
//! DeepSeek V4.1 (api.rs, server/app.py's) or GLM-5.3 (glm.rs, server/muse_app.py's).
//!
//! Text messages only: tool definitions, tool calls and images are refused (400) rather than
//! rendered differently from the Python server.

mod anthropic;
mod api;
mod glm_main;
mod glm;
mod http;
mod output;
#[cfg(test)]
mod tests;
mod vision;

pub use api::Family;

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use tidelm_engine::decode::{DepthMode, Job, SchedOptions, Scheduler};
use tidelm_engine::forward::{Engine, Options};
use tidelm_text::Text;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// What connection threads send the engine thread.
pub enum Cmd {
    Submit(Job),
    Cancel(u64),
}

/// Everything a connection thread needs.
pub struct Ctx {
    pub text: Text,
    pub model: String,
    pub max_seq: usize,
    pub default_thinking: bool,
    pub default_effort: u32,
    pub eos: u32,
    pub think_end: Option<u32>,
    /// image inputs (the engine has vision): the preprocessing's settings
    pub vision: Option<vision::VisionCtx>,
    pub engine: mpsc::Sender<Cmd>,
    pub next_id: AtomicU64,
    /// the chat format and response rules (Family::load reads the checkpoint's config.json);
    /// default_thinking, default_effort, eos, think_end and vision are DeepSeek's
    pub family: Family,
    /// /v1/messages: each answer's reasoning, put back when a client sends the turn again
    pub memo: anthropic::ReasoningMemo,
}


/// The process's heap, watched: with TIDELM_MEM_LOG_MB set, an allocation of at least that many
/// MB after launch is logged with the engine lines that made it (tidelm_cuda::mem logs the
/// device and pinned ones) - what the engine still allocates past its boot footprint.
struct Heap;

thread_local! {
    static IN_LOG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The watched allocations still live, by address: their size and where they were made.
static LIVE: std::sync::Mutex<Option<std::collections::HashMap<usize, (usize, String)>>> = std::sync::Mutex::new(None);

fn heap_seen(p: *mut u8, bytes: usize) {
    if p.is_null() || !tidelm_cuda::mem::heap_watch(bytes) {
        return;
    }
    // the backtrace and the map allocate: not again from inside them
    let _ = IN_LOG.try_with(|inside| {
        if inside.replace(true) {
            return;
        }
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        let at: Vec<&str> = bt.lines().map(str::trim)
            .filter(|l| l.starts_with("at ") && l.contains("crates/tidelm-") && !l.contains("tidelm-server/src/main.rs"))
            .map(|l| &l[l.find("crates/").unwrap_or(0)..]).take(3).collect();
        let at = at.join(" < ");
        if bytes >= tidelm_cuda::mem::LATE_MB << 20 {
            eprintln!("memory: {:.1} MB of heap allocated after launch, at {at}", bytes as f64 / 1e6);
        }
        LIVE.lock().unwrap().get_or_insert_with(Default::default).insert(p as usize, (bytes, at));
        inside.set(false);
    });
}

fn heap_gone(p: *mut u8) {
    if !tidelm_cuda::mem::heap_watch(usize::MAX) {
        return;
    }
    let _ = IN_LOG.try_with(|inside| {
        if inside.replace(true) {
            return;
        }
        if let Some(m) = LIVE.lock().unwrap().as_mut() {
            m.remove(&(p as usize));
        }
        inside.set(false);
    });
}

/// With TIDELM_MEM_LOG_MB set: the watched heap still live, by where it was made (the top
/// `n`), for the request log - what keeps growing past the boot footprint.
pub fn heap_live_top(n: usize) -> Option<String> {
    let _guard = IN_LOG.try_with(|inside| inside.replace(true)).ok()?;
    let mut by: std::collections::HashMap<String, (usize, usize)> = Default::default();
    if let Some(m) = LIVE.lock().unwrap().as_ref() {
        for (bytes, at) in m.values() {
            let e = by.entry(at.clone()).or_default();
            (e.0, e.1) = (e.0 + bytes, e.1 + 1);
        }
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let s = v.iter().take(n).map(|(at, (b, c))| format!("{:.1} MB in {c} at {at}", *b as f64 / 1e6)).collect::<Vec<_>>().join("; ");
    let _ = IN_LOG.try_with(|inside| inside.set(false));
    Some(s)
}

unsafe impl std::alloc::GlobalAlloc for Heap {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc(l) };
        heap_seen(p, l.size());
        p
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc_zeroed(l) };
        heap_seen(p, l.size());
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        heap_gone(p);
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        heap_gone(p);
        let q = unsafe { std::alloc::System.realloc(p, l, n) };
        heap_seen(q, n);
        q
    }
}

#[global_allocator]
static HEAP: Heap = Heap;

extern "C" {
    fn mallopt(param: i32, value: i32) -> i32;
}

fn main() -> tidelm_engine::Result<()> {
    // glibc's heap: a fixed mmap threshold (M_MMAP_THRESHOLD, -3) puts every allocation of 1 MiB
    // or more in its own mapping, unmapped when it is freed - the default raises the threshold
    // past each large free and then keeps such buffers in arenas that do not shrink - and at
    // most four arenas (M_ARENA_MAX, -8) hold the small ones of every thread
    unsafe {
        mallopt(-3, 1 << 20);
        mallopt(-8, 4);
    }
    let args: Vec<String> = std::env::args().collect();
    if let Some(dir) = arg(&args, "--model-dir").map(PathBuf::from) {
        if glm_main::is_glm(&dir) {
            return glm_main::main(&args, dir);
        }
    }
    // an unknown flag is an error: a test flag the server ignores (--keep-first once) loaded
    // every expert and ran the box out of memory
    const VALUED: [&str; 32] = ["--swa-replay", "--copy-draft", "--bench-width", "--vision", "--model-dir", "--max-seq", "--lanes", "--depth", "--step-rows", "--keep-db", "--budget",
                                "--keep-first", "--layers", "--served-model-name", "--effort", "--host", "--port", "--world",
                                "--rank", "--adapt-db", "--trace", "--adapt-sensitivity", "--adapt-prior", "--adapt-decode-tokens", "--adapt-probe", "--hot-ring", "--prefix-cache",
                                "--prefix-snapshots", "--prefix-disk", "--prefix-disk-gb", "--prefix-disk-interval",
                                "--prefix-disk-dedup"];
    const FLAGS: [&str; 3] = ["--resident", "--graphs", "--thinking"];
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        if VALUED.contains(&a) || a == "--addr" {
            i += 2;
        } else if FLAGS.contains(&a) {
            i += 1;
        } else {
            return Err(format!("tidelm-server: unknown argument {a}").into());
        }
    }
    let flag = |name: &str| args.iter().any(|a| a == name);
    let model_dir = arg(&args, "--model-dir").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
    });
    let max_seq: usize = arg(&args, "--max-seq").map(|v| v.parse().unwrap()).unwrap_or(32768);
    let lanes: usize = arg(&args, "--lanes").map(|v| v.parse().unwrap()).unwrap_or(1);
    let depth = match arg(&args, "--depth").as_deref().unwrap_or("confidence") {
        "plain" => DepthMode::Plain,
        "window" => DepthMode::Window,
        "confidence" => DepthMode::Confidence,
        d => DepthMode::Fixed(d.parse().map_err(|_| format!("--depth {d}: window, confidence, plain or a depth"))?),
    };
    // a step's row budget: prompt chunks fill what decoding rows leave, so the full TMAX keeps
    // a prompt's chunks as wide as the single-sequence loop's
    let step_rows: usize = arg(&args, "--step-rows").map(|v| v.parse().unwrap()).unwrap_or(tidelm_engine::forward::TMAX);
    let t0 = Instant::now();
    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    eprintln!("boot: tokenizer {:.1} s", t0.elapsed().as_secs_f64());
    tidelm_cuda::set_device(0)?;
    let scores = match arg(&args, "--keep-db") {
        Some(db) => Some(tidelm_engine::keep::load_demand(std::path::Path::new(&db), 40, 384)?),
        None => None,
    };
    // --keep-first K (tests): the first K experts of every layer
    let keep_first: Option<usize> = arg(&args, "--keep-first").map(|v| v.parse().unwrap());
    // adaptive residency (--adapt-db HISTORY.npz --trace coverage.json --budget N): the
    // Python engine's DSV41_PRUNE_DB history and routing trace, its knobs as the serving
    // profile sets them (sensitivity high, prior 4, a decode pass every 600 tokens). Rank 0
    // reads the history and picks the boot residents; rank 1 receives them in the join.
    let pair_args = tidelm_engine::pair::pair_args(&args)?;
    let leader = pair_args.as_ref().map(|pa| pa.rank == 0).unwrap_or(true);
    let adapt = match arg(&args, "--adapt-db") {
        Some(db) => {
            let sens = arg(&args, "--adapt-sensitivity").unwrap_or_else(|| "high".into());
            let prior: f64 = arg(&args, "--adapt-prior").map(|v| v.parse()).transpose()?.unwrap_or(4.0);
            let every: usize = arg(&args, "--adapt-decode-tokens").map(|v| v.parse()).transpose()?.unwrap_or(600);
            let cfg = tidelm_engine::adapt::AdaptConfig::new(&sens, prior, every, 6)?
                .ok_or("--adapt-sensitivity off: drop --adapt-db to freeze the residents")?;
            if leader {
                let trace = arg(&args, "--trace").ok_or("--adapt-db needs --trace (a routing trace's coverage.json)")?;
                let trace = tidelm_engine::adapt::load_trace(std::path::Path::new(&trace), 40, 384)?;
                eprintln!("adapt: {cfg:?}");
                Some(tidelm_engine::adapt::Adapt::new(cfg, trace, 40, 384, Some(PathBuf::from(db)))?)
            } else {
                None
            }
        }
        None => None,
    };
    let record = arg(&args, "--adapt-db").is_some();
    let keep = match (&adapt, &scores, arg(&args, "--budget")) {
        (Some(a), _, Some(b)) => Some(a.boot_keep(b.parse()?)?),
        (Some(_), _, None) => return Err("--adapt-db needs --budget (the resident experts)".into()),
        (None, _, _) if record => None,
        (None, Some(s), Some(b)) => Some(tidelm_engine::keep::keep_from_scores(s, 6, b.parse().unwrap())?),
        _ => keep_first.map(|k| vec![(0..384).map(|e| e < k).collect::<Vec<bool>>(); 40]),
    };
    let mut opts = Options {
        max_seq,
        max_lanes: lanes,
        spec: depth != DepthMode::Plain,
        resident: flag("--resident"),
        graphs: flag("--graphs"),
        n_layers: arg(&args, "--layers").map(|v| v.parse().unwrap()),
        keep,
        dspark_keep: None,
        tp: None,
        record_demand: record,
        // async hot reads (DSV41_HOT_READS=1, HOT_ASYNC=1, HOT_RING=128 in the serving profile):
        // on with adaptation, --hot-ring 0 turns them off
        hot_ring: match arg(&args, "--hot-ring") {
            Some(v) => v.parse()?,
            None if record => 128,
            None => 0,
        },
        // image inputs (DSV41_VISION=1 in the serving profile): on when the checkpoint has a
        // tower, --vision 0 off
        vision: arg(&args, "--vision").as_deref() != Some("0")
            && tidelm_engine::config::Config::load(&model_dir)?.vision_n_layers > 0,
        // the decoder SWA bounded replay (DSV41_SWA_REPLAY=1): --swa-replay 0 runs every layer
        // over every prompt row
        swa_replay: arg(&args, "--swa-replay").as_deref() != Some("0"),
    };
    // rank 1 of an adapting pair needs rank 0's residents before it loads them: it joins first
    let pair = match &pair_args {
        Some(pa) if record && pa.rank == 1 => {
            let p = tidelm_engine::pair::join(&mut opts, &model_dir, pa)?;
            Some(std::thread::spawn(move || Ok(p)))
        }
        Some(pa) => Some(tidelm_engine::pair::join_in_background(&mut opts, &model_dir, pa)?),
        None => None,
    };
    eprintln!("boot: demand, residency, pair {:.1} s", t0.elapsed().as_secs_f64());
    let mut engine = Engine::new(&model_dir, token_map, opts)?;
    eprintln!("boot: engine {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(p) = pair {
        engine.attach_pair(p.join().map_err(|_| "pair join thread panicked")??)?;
    }
    // the prefix cache's disk tier (DSV41_PREFIX_DISK=1, _GB=20, _MIN_INTERVAL_S=300, _DEDUP=1):
    // --prefix-disk DIR, each rank its own DIR/rank-R
    if let Some(dir) = arg(&args, "--prefix-disk") {
        engine.enable_prefix_disk(tidelm_engine::prefix_disk::DiskConfig {
            dir: PathBuf::from(dir),
            budget: (arg(&args, "--prefix-disk-gb").map(|v| v.parse::<f64>()).transpose()?.unwrap_or(20.0) * 1e9) as u64,
            min_interval_s: arg(&args, "--prefix-disk-interval").map(|v| v.parse()).transpose()?.unwrap_or(300.0),
            dedup: arg(&args, "--prefix-disk-dedup").as_deref() != Some("0"),
        })?;
    }
    // the RAM prefix cache's snapshots, on both ranks: a lane keeps --prefix-snapshots of them,
    // its prompt's own and one restored from disk, all allocated now rather than while serving
    if arg(&args, "--prefix-cache").as_deref() != Some("0") {
        let keep: usize = arg(&args, "--prefix-snapshots").map(|v| v.parse()).transpose()?.unwrap_or(16);
        engine.reserve_snapshots(lanes * (keep + 2))?;
    }
    if engine.is_paired() {
        if engine.rank() == 1 {
            eprintln!("rank 1: engine up in {:.1}s, following rank 0", t0.elapsed().as_secs_f64());
            tidelm_cuda::mem::mark_up();
            eprintln!("memory at boot: {}", tidelm_cuda::mem::report());
            engine.follow()?;
            std::process::exit(0);
        }
    }
    // --bench-width RUN.json (rank 0, instead of serving): verify steps of each width timed on a
    // logged run's real continuation - what a longer draft costs - then rank 1 is released
    if let Some(path) = arg(&args, "--bench-width") {
        return bench_width(&mut engine, &path);
    }
    // a step no lane decodes in prefills up to the engine's prefill rows (TIDELM_PREFILL_ROWS)
    let mut sched = Scheduler::new(&engine, SchedOptions { depth, step_rows, prefill_rows: Some(engine.pmax), logits: false })?;
    // the RAM prefix cache (DSV41_PREFIX_CACHE=1, DSV41_PREFIX_SNAPSHOTS=16): --prefix-cache 0 off
    if arg(&args, "--prefix-cache").as_deref() != Some("0") {
        sched.set_prefix_cache(arg(&args, "--prefix-snapshots").map(|v| v.parse()).transpose()?.unwrap_or(16));
    }
    // the copy drafter (copy.rs; TIDELM_COPY_* tune it): --copy-draft 0 drafts with DSpark alone
    if arg(&args, "--copy-draft").as_deref() != Some("0") {
        sched.set_copy_draft(tidelm_engine::copy::CopyConfig::from_env()?);
    }
    // the probe before each prefill (DSV41_PROBE_PREFILL=1 in the serving profile): --adapt-probe 0 skips it
    if let Some(a) = adapt {
        let probe = arg(&args, "--adapt-probe").as_deref() != Some("0");
        sched.set_adapt(a, probe.then(tidelm_engine::adapt::ProbeConfig::default));
    }
    let single = |s: &str| -> Option<u32> {
        match text.encode(s) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    };
    let eos = single(tidelm_text::chat::EOS).ok_or("the tokenizer has no single end-of-sentence token")?;
    let think_end = single(tidelm_text::chat::THINK_END);
    let (tx, rx) = mpsc::channel::<Cmd>();
    let ctx = Arc::new(Ctx {
        text,
        model: arg(&args, "--served-model-name").unwrap_or_else(|| "deepseek".into()),
        max_seq,
        default_thinking: flag("--thinking"),
        default_effort: arg(&args, "--effort").map(|v| v.parse().unwrap()).unwrap_or(75),
        eos,
        think_end,
        vision: engine.vision().then(|| {
            let c = &engine.cfg;
            vision::VisionCtx {
                plan: tidelm_image::Plan { patch: c.vision_patch_size, ratio: c.vision_downsample_ratio,
                                           max_tokens: c.vision_max_n_token, min_pixels: c.vision_min_pixels },
                image_token_id: c.image_token_id,
            }
        }),
        engine: tx,
        next_id: AtomicU64::new(1),
        family: Family::Ds,
        memo: anthropic::ReasoningMemo::default(),
    });
    let addr = format!("{}:{}", arg(&args, "--host").unwrap_or_else(|| "0.0.0.0".into()),
                       arg(&args, "--port").unwrap_or_else(|| "8000".into()));
    let listener = TcpListener::bind(&addr)?;
    eprintln!("engine up in {:.1}s; serving {} on {addr} ({lanes} lane{}, depth {depth:?})", t0.elapsed().as_secs_f64(),
              ctx.model, if lanes == 1 { "" } else { "s" });
    tidelm_cuda::mem::mark_up();
    eprintln!("memory at boot: {}", tidelm_cuda::mem::report());
    {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let ctx = ctx.clone();
                std::thread::spawn(move || http::serve(stream, &ctx));
            }
        });
    }
    engine_loop(engine, sched, rx)
}

/// The verify cost of a draft by width: the prompt and the first 200 tokens of a logged run
/// (x_debug's prompt_ids / token_ids) as context, then decode steps of 1..33 rows of the run's
/// own continuation, each rolled back (20 timed after 5 warm); and the DSpark pass a step
/// drafted by lookup would not run.
fn bench_width(engine: &mut Engine, path: &str) -> tidelm_engine::Result<()> {
    use tidelm_engine::forward::{Seg, Want};
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let ids = |k: &str| -> Vec<u32> { v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default() };
    let (prompt, out) = (ids("prompt_ids"), ids("token_ids"));
    let k0 = 200.min(out.len().saturating_sub(40));
    if prompt.is_empty() || out.len() < k0 + 33 {
        return Err(format!("{path}: needs prompt_ids and at least {} token_ids", k0 + 33).into());
    }
    let lane = 0;
    engine.reset_lane(lane)?;
    let ctx: Vec<u32> = prompt.iter().chain(&out[..k0]).copied().collect();
    for c in ctx.chunks(engine.pmax) {
        engine.forward_batch_argmax(&[Seg { lane, ids: c, want: Want::Last, decode: false, next: &[], hashes: None }])?;
    }
    let base = engine.lane_len(lane);
    let median = |mut ts: Vec<f64>| -> (f64, f64, f64) {
        ts.sort_by(f64::total_cmp);
        (ts[ts.len() / 2], ts[0], ts[ts.len() - 1])
    };
    eprintln!("bench-width: context {base} tokens");
    for w in [1, 2, 4, 6, 8, 12, 16, 17, 24, 32, 33] {
        let block = &out[k0..k0 + w];
        let mut ts = Vec::new();
        for i in 0..25 {
            let t = Instant::now();
            engine.forward_batch_argmax(&[Seg { lane, ids: block, want: Want::All, decode: true, next: &[], hashes: None }])?;
            let dt = t.elapsed().as_secs_f64() * 1e3;
            engine.rollback_batch(&[(lane, base)])?;
            if i >= 5 {
                ts.push(dt);
            }
        }
        let (m, lo, hi) = median(ts);
        eprintln!("bench-width: {w:2} rows: verify {m:.2} ms (min {lo:.2}, max {hi:.2})");
    }
    let mut ts = Vec::new();
    for i in 0..25 {
        let t = Instant::now();
        engine.draft_batch(&[(lane, out[k0], base - 1)])?;
        if i >= 5 {
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let (m, lo, hi) = median(ts);
    eprintln!("bench-width: DSpark draft pass {m:.2} ms (min {lo:.2}, max {hi:.2})");
    engine.release_follower()?;
    Ok(())
}

/// The engine thread: admit and cancel at step boundaries, step while there is work, block on
/// the channel while there is none. An engine error is fatal - a pair that failed one call
/// cannot continue in step.
fn engine_loop(mut engine: Engine, mut sched: Scheduler, rx: mpsc::Receiver<Cmd>) -> tidelm_engine::Result<()> {
    let apply = |sched: &mut Scheduler, cmd: Cmd| match cmd {
        Cmd::Submit(job) => {
            let id = job.id;
            if let Err(e) = sched.submit(job) {
                eprintln!("request {id} refused: {e}");
            }
        }
        Cmd::Cancel(id) => sched.cancel(id),
    };
    loop {
        if sched.idle() {
            match rx.recv() {
                Ok(cmd) => apply(&mut sched, cmd),
                Err(_) => break,
            }
        }
        while let Ok(cmd) = rx.try_recv() {
            apply(&mut sched, cmd);
        }
        if let Err(e) = sched.step(&mut engine) {
            eprintln!("engine step failed: {e}");
            let _ = engine.release_follower();
            std::process::exit(1);
        }
    }
    engine.release_follower()?;
    Ok(())
}
