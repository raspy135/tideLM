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
//! --sweep-depth [--quick] times plain decoding, the confidence and window depth policies and
//! fixed depths on one prompt; with --temperature T [--top-p P] [--seed S] every arm samples
//! (bench.py's protocol is 0.6 / 0.95), greedy otherwise.
//!
//! Streaming mode (see tidelm_engine::forward): every step reads the weights from NVMe,
//! so expect seconds per step. It exists to check the forward pass end to end in ~2 GB.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use tidelm_engine::decode::{generate, generate_plain, generate_spec, DecodeOptions, Depth, Generation};
use tidelm_engine::forward::{Engine, Options, Sampling};
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
    let opts = Options { max_seq, max_lanes: 1, spec: spec || compare || flag("--sweep-depth") || flag("--dist-check") || flag("--bench"), resident: flag("--resident"), graphs: flag("--graphs"), n_layers, keep, dspark_keep: None, tp: None, record_demand: false, hot_ring: 0 };
    let pair_args = tidelm_engine::pair::pair_args(&args)?;
    let mut opts = opts;
    let pair = match &pair_args {
        Some(pa) => Some(tidelm_engine::pair::join(&mut opts, &model_dir, pa)?),
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
        let sampling = Sampling {
            temperature: arg(&args, "--temperature").map(|v| v.parse().unwrap()).unwrap_or(0.0),
            top_p: arg(&args, "--top-p").map(|v| v.parse().unwrap()).unwrap_or(0.95),
        };
        let seed: u64 = arg(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(0);
        let dopt = DecodeOptions { sampling, seed, digest: false };
        let greedy = sampling.is_greedy();
        eprintln!("sampling: {}", if greedy { "greedy".into() } else { format!("temperature {} top_p {} seed {seed}", sampling.temperature, sampling.top_p) });
        // greedy: the same continuation at every depth (lossless), so only the cost differs
        let mut plain: Option<Vec<u32>> = None;
        let mut report = |name: &str, g: &Generation, dt: f64, engine: &Engine| {
            let per: Vec<usize> = (1..=5).map(|d| g.depths.iter().filter(|&&x| x == d).count()).collect();
            let check = match &plain {
                Some(p) if greedy && *p != g.tokens => "  DIVERGED from plain",
                _ => "",
            };
            let selfcheck = if g.accept_tests > 0 {
                // a binomial's sd around the expected count
                let (e, n) = (g.accept_expected, g.accept_tests as f64);
                format!(", accepts {} of {} tested vs {e:.1} expected (sd {:.1})", g.accept_seen, g.accept_tests,
                        (e * (1.0 - e / n)).max(0.0).sqrt())
            } else {
                String::new()
            };
            eprintln!("{name}: {} tokens, {} steps, {:.3} tokens/step, {:.1} tok/s, {:.1} ms/step{}{check}{selfcheck}{}",
                      g.tokens.len(), g.steps, g.tokens_per_step(), g.tokens.len() as f64 / dt, 1e3 * dt / g.steps.max(1) as f64,
                      if g.depths.is_empty() { String::new() } else { format!(", steps at depth 1..5 {per:?}") },
                      if engine.nucleus_fallbacks > 0 { format!(", {} host nuclei so far", engine.nucleus_fallbacks) } else { String::new() });
            if plain.is_none() {
                plain = Some(g.tokens.clone());
            }
        };
        let t = Instant::now();
        let g = generate(&mut engine, &ids, max_tokens, &stop, Depth::Plain, &dopt, |_| {})?;
        report("depth 0 (plain)", &g, t.elapsed().as_secs_f64(), &engine);
        // the confidence policy, twice: the first run learns this machine's step times
        let mut policy = tidelm_core::depth::ConfidenceDepthPolicy::new(true, 0);
        for pass in ["cold", "warm"] {
            let t = Instant::now();
            let g = generate(&mut engine, &ids, max_tokens, &stop, Depth::Confidence(&mut policy), &dopt, |_| {})?;
            report(&format!("auto ({pass})"), &g, t.elapsed().as_secs_f64(), &engine);
        }
        // the Python engine's shipped controller on this engine: 3/5, every 8 tokens
        let mut window = tidelm_core::depth::DepthPolicy::new((3, 5), tidelm_core::depth::DepthOptions {
            interval: 8, ..Default::default()
        }).map_err(|e| e.0)?;
        for pass in ["cold", "warm"] {
            let t = Instant::now();
            let g = generate(&mut engine, &ids, max_tokens, &stop, Depth::Window(&mut window), &dopt, |_| {})?;
            report(&format!("window 3/5 ({pass})"), &g, t.elapsed().as_secs_f64(), &engine);
        }
        eprintln!("auto step-time estimates (ms): {:?}",
                  policy.estimated_step_s().iter().map(|(d, t)| (*d, (t * 1e4).round() / 10.0)).collect::<Vec<_>>());
        // --quick: depth 3 only (the shipped controller's usual depth) besides the policies
        let depths: Vec<usize> = if flag("--quick") { vec![3] } else { (1..=engine.cfg.dspark_block_size).collect() };
        for d in depths {
            let t = Instant::now();
            let g = generate(&mut engine, &ids, max_tokens, &stop, Depth::Fixed(d), &dopt, |_| {})?;
            report(&format!("depth {d}"), &g, t.elapsed().as_secs_f64(), &engine);
        }
        engine.release_follower()?;
        if engine.rank() == 0 && pair_args.is_some() {
            std::process::exit(0);
        }
        return Ok(());
    }
    if let Some(list) = arg(&args, "--bench") {
        // bench/bench.py's protocol (the README rows): per workload one warmup and three
        // measured runs, each a fresh "[req N] " prompt (N from the README run's label salt),
        // 1,024 tokens with EOS ignored, temperature 0.6 / top_p 0.95, sampling seed 42, decode
        // rate (tokens - 1) / (end - first token), the median reported. Both depth policies
        // run each prompt: the Python engine's window 3/5 and the confidence policy.
        let sampling = Sampling {
            temperature: arg(&args, "--temperature").map(|v| v.parse().unwrap()).unwrap_or(0.6),
            top_p: arg(&args, "--top-p").map(|v| v.parse().unwrap()).unwrap_or(0.95),
        };
        let seed: u64 = arg(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(42);
        let runs: u64 = arg(&args, "--runs").map(|v| v.parse().unwrap()).unwrap_or(3);
        let osl: usize = arg(&args, "--max-tokens").map(|v| v.parse().unwrap()).unwrap_or(1024);
        let mut conf = tidelm_core::depth::ConfidenceDepthPolicy::new(true, 0);
        let mut window = tidelm_core::depth::DepthPolicy::new((3, 5), tidelm_core::depth::DepthOptions {
            interval: 8, ..Default::default()
        }).map_err(|e| e.0)?;
        for wl in list.split(',') {
            // bench.py's WORKLOADS, and 1000 + its label salt for the README run's labels
            let (body, base) = match wl {
                "prose" => ("Write a detailed, flowing essay of about 900 words on how tidal forces shaped the evolution of \
                             coastal ecosystems. Use full paragraphs and continuous prose, no bullet points or headings.", 99902u64),
                "code" => ("Write a complete, production-quality Python module implementing an LRU cache with a TTL per \
                            entry, thread safety, and an eviction callback. Include full docstrings, type hints, and a pytest \
                            suite covering expiry, eviction order, and concurrent access.", 96949),
                w => return Err(format!("--bench {w}: prose or code").into()),
            };
            let mut rates: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for i in 0..=runs {
                let req = base + i;
                let text_prompt = format!("<｜begin▁of▁sentence｜><｜User｜>[req {req}] {body}<｜Assistant｜></think>");
                let pids = text.encode(&text_prompt)?;
                for (k, name) in ["window 3/5", "confidence"].iter().enumerate() {
                    let dopt = DecodeOptions { sampling, seed, digest: false };
                    let depth = if k == 0 { Depth::Window(&mut window) } else { Depth::Confidence(&mut conf) };
                    let t0 = Instant::now();
                    let mut first: Option<Instant> = None;
                    let g = generate(&mut engine, &pids, osl, &[], depth, &dopt, |_| {
                        first.get_or_insert_with(Instant::now);
                    })?;
                    let end = Instant::now();
                    let t1 = first.unwrap();
                    let rate = (g.tokens.len() - 1) as f64 / (end - t1).as_secs_f64();
                    let tag = if i == 0 { "warmup".to_string() } else { format!("run{i}") };
                    eprintln!("{wl} {name:>10} {tag:>6}: isl {} ttft {:.0} ms, {} tokens, {} steps, {:.2} tokens/step, decode {rate:.2} tok/s, \
                               accepts {} of {} vs {:.1} expected",
                              pids.len(), (t1 - t0).as_secs_f64() * 1e3, g.tokens.len(), g.steps, g.tokens_per_step(),
                              g.accept_seen, g.accept_tests, g.accept_expected);
                    if i > 0 {
                        rates[k].push(rate);
                    }
                }
            }
            for (k, name) in ["window 3/5", "confidence"].iter().enumerate() {
                let mut r = rates[k].clone();
                r.sort_by(|a, b| a.total_cmp(b));
                eprintln!("{wl} {name:>10} MEDIAN decode {:.2} tok/s over {:?}", r[r.len() / 2], rates[k]);
            }
        }
        if engine.nucleus_fallbacks > 0 {
            eprintln!("{} sampled rows took the host nucleus", engine.nucleus_fallbacks);
        }
        engine.release_follower()?;
        if engine.rank() == 0 && pair_args.is_some() {
            std::process::exit(0);
        }
        return Ok(());
    }
    if let Some(n) = arg(&args, "--dist-check").map(|v| v.parse::<u64>().unwrap()) {
        // sampled speculation must not move the output distribution: per position, the
        // tokens of n seeds decoded plainly against n other seeds decoded at depth 5 (the most
        // rejections), by total variation distance and a two-sample chi-square
        let sampling = Sampling {
            temperature: arg(&args, "--temperature").map(|v| v.parse().unwrap()).unwrap_or(1.0),
            top_p: arg(&args, "--top-p").map(|v| v.parse().unwrap()).unwrap_or(0.95),
        };
        let mut hist: Vec<[std::collections::BTreeMap<u32, u64>; 2]> = Vec::new();
        let (mut acc, mut steps) = (0usize, 0usize);
        for (arm, depth) in [(0usize, 0usize), (1, 5)] {
            for k in 0..n {
                let dopt = DecodeOptions { sampling, seed: 1_000_000 * arm as u64 + k, digest: false };
                let d = if depth == 0 { Depth::Plain } else { Depth::Fixed(depth) };
                let g = generate(&mut engine, &ids, max_tokens, &[], d, &dopt, |_| {})?;
                if depth > 0 {
                    acc += g.accepted.iter().sum::<usize>();
                    steps += g.steps;
                }
                for (i, &t) in g.tokens.iter().enumerate() {
                    if hist.len() <= i {
                        hist.push(Default::default());
                    }
                    *hist[i][arm].entry(t).or_default() += 1;
                }
            }
        }
        eprintln!("depth 5: {:.2} accepted drafts a step; {} host nuclei", acc as f64 / steps.max(1) as f64, engine.nucleus_fallbacks);
        for (i, h) in hist.iter().enumerate() {
            let keys: std::collections::BTreeSet<u32> = h[0].keys().chain(h[1].keys()).copied().collect();
            let (n0, n1) = (h[0].values().sum::<u64>() as f64, h[1].values().sum::<u64>() as f64);
            let (mut tv, mut chi, mut dof) = (0.0, 0.0, 0usize);
            for k in &keys {
                let (a, b) = (*h[0].get(k).unwrap_or(&0) as f64, *h[1].get(k).unwrap_or(&0) as f64);
                tv += (a / n0 - b / n1).abs() / 2.0;
                if a + b >= 5.0 {
                    chi += (a - b).powi(2) / (a + b);
                    dof += 1;
                }
            }
            eprintln!("position {i}: {} distinct tokens, TV {tv:.3}, chi-square {chi:.1} on {} dof (tokens seen >= 5 times)",
                      keys.len(), dof.saturating_sub(1));
        }
        engine.release_follower()?;
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
        eprintln!("logits digest: {:016x}", g.digest);
    }
    engine.release_follower()?;
    if pair_args.is_some() {
        std::process::exit(0);
    }
    Ok(())
}
