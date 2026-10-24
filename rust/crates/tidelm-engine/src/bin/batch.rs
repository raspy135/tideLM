//! Continuous batching: many requests through the engine's lanes at once, checked against
//! each request run alone, and timed.
//!
//!   tidelm-batch [--lanes 4] [--requests 8] [--max-tokens 24] [--step-rows 16]
//!                [--spec [--depth 5]] [--resident [--graphs]] [--layers N] [--keep-first K]
//!                [--check] [--warmup] [--max-seq 1024] [--model-dir DIR]
//!
//! --check runs every request again alone (one lane busy, the same rows per step) and
//! requires identical tokens, and identical logits digests when --step-rows <= 16 (every
//! kernel is row-invariant on the FMA path, so batching must not change a single bit). Where
//! a request's chunking matches the single-sequence loops it is also checked against
//! generate_plain / generate_spec. Every run prints aggregate generated tokens per second,
//! batched and one at a time.

use std::path::PathBuf;
use std::time::Instant;

use tidelm_engine::decode::{generate_batch, generate_plain, generate_spec, BatchOptions, Generation, Request};
use tidelm_engine::forward::{Engine, Options, FMA_ROWS, TMAX};
use tidelm_text::Text;

const PROMPTS: [&str; 8] = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain why the sky is blue in one paragraph, for a curious ten-year-old who has just asked you at the beach.",
    "1, 2, 3, 5, 8, 13,",
    "Translate to German: The weather is nice today and we are going for a walk along the river.",
    "A haiku about autumn:",
    "Here is a short story about a lighthouse keeper who finds a message in a bottle. It begins on a stormy night,",
    "SELECT name, COUNT(*) FROM orders",
];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> tidelm_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str, default: usize| arg(&args, name).map(|v| v.parse().unwrap()).unwrap_or(default);
    let model_dir = arg(&args, "--model-dir").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
    });
    let (lanes, n_req, max_tokens) = (num("--lanes", 4), num("--requests", 8), num("--max-tokens", 24));
    let step_rows = num("--step-rows", FMA_ROWS);
    let spec = flag("--spec");
    let depth = num("--depth", 5);
    let n_layers: Option<usize> = arg(&args, "--layers").map(|v| v.parse().unwrap());
    let keep_first: Option<usize> = arg(&args, "--keep-first").map(|v| v.parse().unwrap());

    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    let reqs: Vec<Request> = (0..n_req)
        .map(|i| -> tidelm_engine::Result<Request> {
            let mut prompt = vec![0u32];
            prompt.extend(text.encode(PROMPTS[i % PROMPTS.len()])?);
            Ok(Request { prompt, max_tokens })
        })
        .collect::<tidelm_engine::Result<_>>()?;
    tidelm_cuda::set_device(0)?;
    // --keep-db FILE --budget N: the demand-ranked resident set (tidelm_engine::keep); with
    // --world 2 the budget is the pair's
    let scores = match arg(&args, "--keep-db") {
        Some(db) => Some(tidelm_engine::keep::load_demand(std::path::Path::new(&db), 40, 384)?),
        None => None,
    };
    let keep = match (&scores, arg(&args, "--budget")) {
        (Some(s), Some(b)) => Some(tidelm_engine::keep::keep_from_scores(s, 6, b.parse().unwrap())?),
        _ => keep_first.map(|k| vec![(0..384).map(|e| e < k).collect::<Vec<bool>>(); 40]),
    };
    let dspark_keep = keep_first.map(|k| vec![(0..128).map(|e| e < k.min(128)).collect::<Vec<bool>>(); 3]);
    let opts = Options {
        max_seq: num("--max-seq", 1024),
        max_lanes: lanes,
        spec,
        resident: flag("--resident"),
        graphs: flag("--graphs"),
        n_layers,
        keep,
        dspark_keep,
        owned: None,
    };
    let t0 = Instant::now();
    let pair_args = tidelm_engine::pair::pair_args(&args)?;
    let mut opts = opts;
    let pair = match &pair_args {
        Some(pa) => Some(tidelm_engine::pair::join(&mut opts, scores.as_deref(), &model_dir, pa)?),
        None => None,
    };
    let mut engine = Engine::new(&model_dir, token_map, opts)?;
    if let Some(p) = pair {
        engine.attach_pair(p)?;
        if engine.rank() == 1 {
            eprintln!("rank 1: engine up in {:.1}s, following rank 0", t0.elapsed().as_secs_f64());
            engine.follow()?;
            // NCCL's communicator teardown blocks a finished pair (see tidelm-generate)
            std::process::exit(0);
        }
    }
    eprintln!("engine up in {:.1}s: {lanes} lanes, {step_rows} rows per step, {}", t0.elapsed().as_secs_f64(),
              if spec { format!("speculative depth {depth}") } else { "plain".into() });
    let result = run(&mut engine, &args, &reqs, lanes, n_req, max_tokens, step_rows, spec, depth);
    if pair_args.is_some() {
        engine.release_follower()?;
        result?;
        std::process::exit(0);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run(engine: &mut Engine, args: &[String], reqs: &[Request], lanes: usize, n_req: usize, max_tokens: usize,
       step_rows: usize, spec: bool, depth: usize) -> tidelm_engine::Result<()> {
    let flag = |name: &str| args.iter().any(|a| a == name);
    // --check compares logits digests; timing runs read back only the device argmaxes
    let check = flag("--check");
    let opt = BatchOptions { spec_depth: spec.then_some(depth), step_rows, stop: vec![], logits: check };

    if flag("--warmup") {
        generate_batch(engine, reqs, &opt)?;
        for r in reqs {
            generate_batch(engine, std::slice::from_ref(r), &opt)?;
        }
    }
    engine.times = Default::default();
    let caps = engine.graph_captures;
    let t = Instant::now();
    let (batched, stats) = generate_batch(engine, reqs, &opt)?;
    let dt_b = t.elapsed().as_secs_f64();
    eprintln!("graph captures during the batched run: {}", engine.graph_captures - caps);
    let toks: usize = batched.iter().map(|g| g.tokens.len()).sum();
    eprintln!("batched: {n_req} requests, {toks} tokens in {dt_b:.2}s = {:.1} tok/s; {} steps, {:.1} rows/step, {} draft calls, \
               up to {} lanes busy", toks as f64 / dt_b, stats.steps, stats.rows as f64 / stats.steps as f64, stats.drafts,
              stats.max_active);
    eprintln!("         per step: draft {:.1} ms, forward {:.1} ms, host post {:.1} ms, rollback+seed {:.1} ms; engine {:?}",
              1e3 * stats.draft_s / stats.steps as f64, 1e3 * stats.forward_s / stats.steps as f64,
              1e3 * stats.post_s / stats.steps as f64, 1e3 * stats.cache_s / stats.steps as f64, engine.times);

    if flag("--no-alone") {
        return Ok(());
    }
    let t = Instant::now();
    let mut alone: Vec<Generation> = Vec::new();
    let mut steps_alone = 0;
    for r in reqs {
        let (g, s) = generate_batch(engine, std::slice::from_ref(r), &opt)?;
        steps_alone += s.steps;
        alone.extend(g);
    }
    let dt_a = t.elapsed().as_secs_f64();
    eprintln!("alone:   {n_req} requests, {toks} tokens in {dt_a:.2}s = {:.1} tok/s; {steps_alone} steps -> batched {:.2}x",
              toks as f64 / dt_a, dt_a / dt_b);
    eprintln!("graph captures {}, replays {}", engine.graph_captures, engine.graph_replays);

    if !check {
        return Ok(());
    }
    let bitwise = step_rows <= FMA_ROWS;
    let mut ok = true;
    // the argmax-only read-back must pick the same tokens as the host argmax of the logits
    let (fast, _) = generate_batch(engine, reqs, &BatchOptions { logits: false, ..opt.clone() })?;
    for (i, (b, f)) in batched.iter().zip(&fast).enumerate() {
        if b.tokens != f.tokens {
            ok = false;
            eprintln!("request {i}: device argmax and host argmax DIFFER\n  host   {:?}\n  device {:?}", b.tokens, f.tokens);
        }
    }
    for (i, (b, a)) in batched.iter().zip(&alone).enumerate() {
        let same_tokens = b.tokens == a.tokens;
        let same_bits = b.digest == a.digest;
        if !same_tokens || (bitwise && !same_bits) {
            ok = false;
            eprintln!("request {i}: batched and alone DIFFER (tokens {same_tokens}, logits {same_bits})\n  batched {:?}\n  alone   {:?}",
                      b.tokens, a.tokens);
        } else if !same_bits {
            eprintln!("request {i}: same tokens, logits differ in rounding (steps above {FMA_ROWS} rows take the tensor-core GEMM)");
        }
    }
    // the scheduler alone against the single-sequence loops, where their chunking agrees
    for (i, (r, a)) in reqs.iter().zip(&alone).enumerate() {
        if !(r.prompt.len() <= step_rows || step_rows == TMAX) {
            continue;
        }
        let g = if spec {
            generate_spec(engine, &r.prompt, max_tokens, depth, &[], |_| {})?
        } else {
            generate_plain(engine, &r.prompt, max_tokens, &[], |_| {})?
        };
        if g.tokens != a.tokens || g.digest != a.digest {
            ok = false;
            eprintln!("request {i}: scheduler alone and the single-sequence loop DIFFER\n  loop      {:?}\n  scheduler {:?}",
                      g.tokens, a.tokens);
        }
    }
    if !ok {
        eprintln!("BATCH CHECK FAILED");
        std::process::exit(1);
    }
    eprintln!("BATCH CHECK OK: {n_req} requests over {lanes} lanes, tokens identical to each request alone{}",
              if bitwise { ", logits bit-identical" } else { "" });
    Ok(())
}
