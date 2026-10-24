//! Greedy generation with the Rust engine, no Python anywhere in the pipeline.
//!
//!   tidelm-generate --prompt "The capital of France is" [--max-tokens 16]
//!                   [--prompt-file F] [--model-dir DIR] [--max-seq 1024] [--no-bos]
//!                   [--spec [--depth 5]] [--compare]
//!                   [--resident [--graphs]] [--layers N] [--keep-first K]   (tests: truncate / restrict experts)
//!
//! --spec decodes with the DSpark drafter; --compare runs plain greedy and then speculative
//! greedy on the same engine and checks the two token sequences are identical (the repo's
//! engine/test_spec_lossless.py contract).
//!
//! Streaming mode (see tidelm_engine::forward): every step reads the weights from NVMe,
//! so expect seconds per step. It exists to check the forward pass end to end in ~2 GB.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use tidelm_engine::decode::{generate_plain, generate_spec, generate_spec_auto, Generation};
use tidelm_engine::forward::{Engine, Options};
use tidelm_text::Text;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > 2 * n {
        format!("...{}", chars[chars.len() - n..].iter().collect::<String>())
    } else {
        s.to_string()
    }
}

fn main() -> tidelm_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let model_dir = arg(&args, "--model-dir").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
    });
    let prompt = match arg(&args, "--prompt-file") {
        Some(f) => std::fs::read_to_string(f)?,
        None => arg(&args, "--prompt").unwrap_or_else(|| "The capital of France is".into()),
    };
    let max_tokens: usize = arg(&args, "--max-tokens").map(|v| v.parse().unwrap()).unwrap_or(16);
    let max_seq: usize = arg(&args, "--max-seq").map(|v| v.parse().unwrap()).unwrap_or(1024);
    let depth: usize = arg(&args, "--depth").map(|v| v.parse().unwrap()).unwrap_or(5);
    let (spec, compare) = (flag("--spec"), flag("--compare"));
    let n_layers: Option<usize> = arg(&args, "--layers").map(|v| v.parse().unwrap());
    let keep_first: Option<usize> = arg(&args, "--keep-first").map(|v| v.parse().unwrap());

    let t0 = Instant::now();
    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    let mut ids: Vec<u32> = if flag("--no-bos") { vec![] } else { vec![0] };
    ids.extend(text.encode(&prompt)?);
    eprintln!("prompt: {} tokens", ids.len());

    tidelm_cuda::set_device(0)?;
    // --keep-db FILE --budget N: the demand-ranked resident set (tidelm_engine::keep); with
    // --world 2 the budget is the pair's, split between the ranks
    let scores = match arg(&args, "--keep-db") {
        Some(db) => Some(tidelm_engine::keep::load_demand(std::path::Path::new(&db), 40, 384)?),
        None => None,
    };
    let keep = match (&scores, arg(&args, "--budget")) {
        (Some(s), Some(b)) => Some(tidelm_engine::keep::keep_from_scores(s, 6, b.parse().unwrap())?),
        _ => keep_first.map(|k| vec![(0..384).map(|e| e < k).collect::<Vec<bool>>(); 40]),
    };
    let opts = Options { max_seq, max_lanes: 1, spec: spec || compare || flag("--sweep-depth"), resident: flag("--resident"), graphs: flag("--graphs"), n_layers, keep, dspark_keep: None, owned: None };
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
            // NCCL's communicator teardown blocked both ranks indefinitely after a finished run
            // (both mains in a futex wait); there is nothing to flush, so leave without it
            std::process::exit(0);
        }
    }
    eprintln!("engine up in {:.1}s", t0.elapsed().as_secs_f64());
    let stop = [1u32]; // <｜end▁of▁sentence｜>

    let run = |engine: &mut Engine, use_spec: bool| -> tidelm_engine::Result<(Generation, f64)> {
        let t = Instant::now();
        print!("{}", tail(&prompt, 60));
        let mut show = |toks: &[u32]| {
            print!("{}", text.decode(toks).unwrap_or_default());
            std::io::stdout().flush().ok();
        };
        let g = if use_spec {
            generate_spec(engine, &ids, max_tokens, depth, &stop, &mut show)?
        } else {
            generate_plain(engine, &ids, max_tokens, &stop, &mut show)?
        };
        println!();
        Ok((g, t.elapsed().as_secs_f64()))
    };

    if flag("--sweep-depth") {
        // the same greedy continuation at every depth (lossless), so only the cost differs
        let (plain, tp) = run(&mut engine, false)?;
        eprintln!("depth 0 (plain): {} tokens, {:.1} tok/s, {:.1} ms/step", plain.tokens.len(),
                  plain.tokens.len() as f64 / tp, 1e3 * tp / plain.tokens.len() as f64);
        // the confidence policy, twice: the first run learns this machine's step times
        let mut policy = tidelm_core::depth::ConfidenceDepthPolicy::new(true, 0);
        for pass in ["cold", "warm"] {
            let t = Instant::now();
            let g = generate_spec_auto(&mut engine, &ids, max_tokens, &stop, &mut policy, |_| {})?;
            let dt = t.elapsed().as_secs_f64();
            let per: Vec<usize> = (1..=5).map(|d| g.depths.iter().filter(|&&x| x == d).count()).collect();
            eprintln!("auto ({pass}): {} tokens, {} steps, {:.3} tokens/step, {:.1} tok/s, {:.1} ms/step, steps at depth 1..5 {per:?}{}",
                      g.tokens.len(), g.steps, g.tokens_per_step(), g.tokens.len() as f64 / dt, 1e3 * dt / g.steps.max(1) as f64,
                      if g.tokens == plain.tokens { "" } else { "  DIVERGED from plain" });
        }
        // the Python engine's shipped controller on this engine: 3/5, every 8 tokens
        let mut window = tidelm_core::depth::DepthPolicy::new((3, 5), tidelm_core::depth::DepthOptions {
            interval: 8, ..Default::default()
        }).map_err(|e| e.0)?;
        for pass in ["cold", "warm"] {
            let t = Instant::now();
            let g = tidelm_engine::decode::generate_spec_window(&mut engine, &ids, max_tokens, &stop, &mut window)?;
            let dt = t.elapsed().as_secs_f64();
            let per: Vec<usize> = (1..=5).map(|d| g.depths.iter().filter(|&&x| x == d).count()).collect();
            eprintln!("window 3/5 ({pass}): {} tokens, {} steps, {:.3} tokens/step, {:.1} tok/s, {:.1} ms/step, steps at depth 1..5 {per:?}{}",
                      g.tokens.len(), g.steps, g.tokens_per_step(), g.tokens.len() as f64 / dt, 1e3 * dt / g.steps.max(1) as f64,
                      if g.tokens == plain.tokens { "" } else { "  DIVERGED from plain" });
        }
        eprintln!("auto step-time estimates (ms): {:?}",
                  policy.estimated_step_s().iter().map(|(d, t)| (*d, (t * 1e4).round() / 10.0)).collect::<Vec<_>>());
        for d in 1..=engine.cfg.dspark_block_size {
            let t = Instant::now();
            let g = generate_spec(&mut engine, &ids, max_tokens, d, &stop, |_| {})?;
            let dt = t.elapsed().as_secs_f64();
            eprintln!("depth {d}: {} tokens, {} steps, {:.3} tokens/step, {:.1} tok/s, {:.1} ms/step{}", g.tokens.len(), g.steps,
                      g.tokens_per_step(), g.tokens.len() as f64 / dt, 1e3 * dt / g.steps.max(1) as f64,
                      if g.tokens == plain.tokens { "" } else { "  DIVERGED from plain" });
        }
        engine.release_follower()?;
        if engine.rank() == 0 && pair_args.is_some() {
            std::process::exit(0);
        }
        return Ok(());
    }
    if compare {
        let (plain, tp) = run(&mut engine, false)?;
        eprintln!("plain: {} tokens, {} steps, {:.1}s", plain.tokens.len(), plain.steps, tp);
        let (spec_g, ts) = run(&mut engine, true)?;
        eprintln!("spec:  {} tokens, {} steps ({:.2} tokens/step), {:.1}s, accepted {:?}", spec_g.tokens.len(),
                  spec_g.steps, spec_g.tokens_per_step(), ts, spec_g.accepted);
        let n = plain.tokens.len().min(spec_g.tokens.len());
        match (0..n).find(|&i| plain.tokens[i] != spec_g.tokens[i]) {
            None if plain.tokens.len() == spec_g.tokens.len() => {
                eprintln!("LOSSLESS: {n} tokens identical with and without speculation");
            }
            first => {
                eprintln!("DIVERGED at token {first:?} of {n}:\n  plain {:?}\n  spec  {:?}", plain.tokens, spec_g.tokens);
                std::process::exit(1);
            }
        }
    } else {
        let (g, dt) = run(&mut engine, spec)?;
        eprintln!("{} tokens in {} steps ({:.2} tokens/step), {:.1}s; times {:?}; indexed layer passes {}",
                  g.tokens.len(), g.steps, g.tokens_per_step(), dt, engine.times, engine.indexed_layers);
        eprintln!("generated ids: {:?}", g.tokens);
    }
    engine.release_follower()?;
    if pair_args.is_some() {
        std::process::exit(0);
    }
    Ok(())
}
