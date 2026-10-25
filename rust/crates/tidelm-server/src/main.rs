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
//! bench/bench.py's token_len calls). Requests run through the engine's continuous-batching
//! scheduler, one lane each; with --world 2, rank 1 follows rank 0's engine calls and serves
//! nothing.
//!
//! Text messages only: tool definitions, tool calls and images are refused (400) rather than
//! rendered differently from the Python server.

mod api;
mod http;

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
    pub engine: mpsc::Sender<Cmd>,
    pub next_id: AtomicU64,
}

fn main() -> tidelm_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // an unknown flag is an error: a test flag the server ignores (--keep-first once) loaded
    // every expert and ran the box out of memory
    const VALUED: [&str; 28] = ["--model-dir", "--max-seq", "--lanes", "--depth", "--step-rows", "--keep-db", "--budget",
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
    if engine.is_paired() {
        if engine.rank() == 1 {
            eprintln!("rank 1: engine up in {:.1}s, following rank 0", t0.elapsed().as_secs_f64());
            engine.follow()?;
            std::process::exit(0);
        }
    }
    // a step no lane decodes in prefills up to the engine's prefill rows (TIDELM_PREFILL_ROWS)
    let mut sched = Scheduler::new(&engine, SchedOptions { depth, step_rows, prefill_rows: Some(engine.pmax), logits: false })?;
    // the RAM prefix cache (DSV41_PREFIX_CACHE=1, DSV41_PREFIX_SNAPSHOTS=16): --prefix-cache 0 off
    if arg(&args, "--prefix-cache").as_deref() != Some("0") {
        sched.set_prefix_cache(arg(&args, "--prefix-snapshots").map(|v| v.parse()).transpose()?.unwrap_or(16));
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
        engine: tx,
        next_id: AtomicU64::new(1),
    });
    let addr = format!("{}:{}", arg(&args, "--host").unwrap_or_else(|| "0.0.0.0".into()),
                       arg(&args, "--port").unwrap_or_else(|| "8000".into()));
    let listener = TcpListener::bind(&addr)?;
    eprintln!("engine up in {:.1}s; serving {} on {addr} ({lanes} lane{}, depth {depth:?})", t0.elapsed().as_secs_f64(),
              ctx.model, if lanes == 1 { "" } else { "s" });
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
