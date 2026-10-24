//! Mode parity: the same prompt and greedy steps through the streaming engine, the resident
//! one, and the resident one with every step a CUDA graph. All three must agree bit for bit
//! — they share every kernel and differ only in where weights come from and how the step's
//! launches are issued. Also reports per-step wall time, resident eager vs graph replay.
//!
//!   tidelm-parity [--layers 2] [--keep-first 16] [--steps 4] [--prompt "..."]
//!
//! Defaults fit next to a live server: two layers, 16 routable experts per layer, ~4 GB.

use std::path::PathBuf;

use tidelm_engine::decode::{argmax_bf16, generate_plain, generate_spec, generate_spec_with};
use tidelm_engine::forward::{Engine, Options, Seg, Want};
use tidelm_text::Text;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn run(engine: &mut Engine, prompt: &[u32], steps: usize) -> tidelm_engine::Result<(Vec<Vec<u16>>, Vec<u32>, f64)> {
    engine.reset();
    let mut logits = Vec::new();
    let mut toks = Vec::new();
    let chunks: Vec<&[u32]> = prompt.chunks(tidelm_engine::forward::TMAX).collect();
    let mut lg = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        lg = engine.forward(c, if i + 1 == chunks.len() { Want::Last } else { Want::Nothing })?;
    }
    let mut decode_s = 0.0;
    for i in 0..steps {
        let tok = argmax_bf16(&lg);
        toks.push(tok);
        logits.push(lg);
        let t = std::time::Instant::now();
        lg = engine.forward(&[tok], Want::Last)?;
        // the first decode step of a graph engine captures; time the replays
        if i > 0 {
            decode_s += t.elapsed().as_secs_f64();
        }
    }
    logits.push(lg);
    Ok((logits, toks, decode_s / (steps.max(2) - 1) as f64))
}

fn main() -> tidelm_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = arg(&args, "--model-dir").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
    });
    let layers: usize = arg(&args, "--layers").map(|v| v.parse().unwrap()).unwrap_or(2);
    let keep_first: usize = arg(&args, "--keep-first").map(|v| v.parse().unwrap()).unwrap_or(16);
    let steps: usize = arg(&args, "--steps").map(|v| v.parse().unwrap()).unwrap_or(4);
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "The quick brown fox jumps over the lazy dog because".into());
    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    let mut ids = vec![0u32];
    ids.extend(text.encode(&prompt)?);
    tidelm_cuda::set_device(0)?;
    let keep = Some(vec![(0..384).map(|e| e < keep_first).collect::<Vec<bool>>(); 40]);
    if args.iter().any(|a| a == "--mma") {
        return mma_rounding(&model_dir, token_map, &text, layers, keep);
    }
    if args.iter().any(|a| a == "--spec") {
        return spec_parity(&model_dir, token_map, &ids, layers, keep, steps);
    }
    let base = Options { max_seq: 1024, max_lanes: 1, spec: false, resident: false, graphs: false, n_layers: Some(layers), keep, dspark_keep: None };

    let (sl, st, _) = {
        let mut e = Engine::new(&model_dir, token_map.clone(), base.clone())?;
        run(&mut e, &ids, steps)?
    };
    eprintln!("streaming:        tokens {st:?}");
    // one resident engine serves both resident runs: eager, then with graphs switched on
    let mut e = Engine::new(&model_dir, token_map, Options { resident: true, ..base })?;
    let (rl, rt, r_s) = run(&mut e, &ids, steps)?;
    eprintln!("resident eager:   tokens {rt:?}  {:.3} ms/step", r_s * 1e3);
    e.set_graphs(true);
    let (gl, gt, g_s) = run(&mut e, &ids, steps)?;
    eprintln!("resident graphs:  tokens {gt:?}  {:.3} ms/step  ({} captures, {} replays)", g_s * 1e3,
              e.graph_captures, e.graph_replays);
    let mut ok = st == rt && rt == gt;
    for (i, ((a, b), c)) in sl.iter().zip(&rl).zip(&gl).enumerate() {
        let d1 = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let d2 = b.iter().zip(c).filter(|(x, y)| x != y).count();
        eprintln!("step {i}: streaming/resident {d1} differ, resident/graph {d2} differ (of {})", a.len());
        ok &= d1 == 0 && d2 == 0;
    }
    if !ok {
        eprintln!("PARITY FAILED");
        std::process::exit(1);
    }
    eprintln!("PARITY OK: {} layers, {} experts/layer, {} steps bit-identical across streaming, resident, graphs",
              layers, keep_first, steps + 1);
    Ok(())
}

/// Speculative greedy through the three modes: each must equal plain greedy in its own mode
/// (lossless), and all three must agree with each other.
fn spec_parity(model_dir: &std::path::Path, token_map: Vec<i64>, ids: &[u32], layers: usize,
               keep: Option<Vec<Vec<bool>>>, steps: usize) -> tidelm_engine::Result<()> {
    let dkeep = Some(vec![(0..128).map(|e| e < 8).collect::<Vec<bool>>(); 3]);
    let base = Options { max_seq: 1024, max_lanes: 1, spec: true, resident: false, graphs: false, n_layers: Some(layers), keep,
                         dspark_keep: dkeep };
    let n = steps + 1;
    let mut results = Vec::new();
    {
        let mut e = Engine::new(model_dir, token_map.clone(), base.clone())?;
        let plain = generate_plain(&mut e, ids, n, &[], |_| {})?;
        let spec = generate_spec(&mut e, ids, n, 5, &[], |_| {})?;
        eprintln!("streaming:       plain {:?}\n                 spec  {:?} accepted {:?}", plain.tokens, spec.tokens, spec.accepted);
        let oracle = oracle_spec(&mut e, ids, n, &plain.tokens)?;
        eprintln!("                 oracle accepted {:?}", oracle.accepted);
        results.push((plain.tokens.clone(), spec.tokens, spec.accepted));
        results.push((plain.tokens, oracle.tokens, oracle.accepted));
    }
    let mut e = Engine::new(model_dir, token_map, Options { resident: true, ..base })?;
    for graphs in [false, true] {
        e.set_graphs(graphs);
        let plain = generate_plain(&mut e, ids, n, &[], |_| {})?;
        let spec = generate_spec(&mut e, ids, n, 5, &[], |_| {})?;
        eprintln!("resident {:6}: plain {:?}\n                 spec  {:?} accepted {:?}", if graphs { "graphs" } else { "eager" },
                  plain.tokens, spec.tokens, spec.accepted);
        let oracle = oracle_spec(&mut e, ids, n, &plain.tokens)?;
        eprintln!("                 oracle accepted {:?}", oracle.accepted);
        results.push((plain.tokens.clone(), spec.tokens, spec.accepted));
        results.push((plain.tokens, oracle.tokens, oracle.accepted));
    }
    eprintln!("graph captures {}, replays {}", e.graph_captures, e.graph_replays);
    let lossless = results.iter().all(|(p, s, _)| p == s);
    // drafter runs and oracle runs alternate; each kind must agree across the three modes
    let agree = (0..2).all(|k| results.iter().skip(k).step_by(2).collect::<Vec<_>>().windows(2).all(|w| w[0] == w[1]));
    eprintln!("lossless in every mode: {lossless}; modes agree (tokens and acceptance): {agree}");
    if !(lossless && agree) {
        eprintln!("SPEC PARITY FAILED");
        std::process::exit(1);
    }
    eprintln!("SPEC PARITY OK");
    Ok(())
}

/// Speculation with oracle drafts: the true greedy continuation with the draft at position
/// `step % 6` replaced by a wrong token, so the verify accepts exactly that many drafts
/// (5 = all) — every rollback depth, every pending parity.
fn oracle_spec(e: &mut Engine, ids: &[u32], n: usize, truth: &[u32]) -> tidelm_engine::Result<tidelm_engine::decode::Generation> {
    let mut step = 0usize;
    generate_spec_with(e, ids, n, 5, &[], |_| {}, |_e, _tok, _last, emitted| {
        let k = step % 6;
        step += 1;
        let mut d: Vec<u32> = (0..5).map(|i| truth.get(emitted + i).copied().unwrap_or(0)).collect();
        if k < 5 {
            d[k] = (d[k] + 1) % 129_280;
        }
        Ok(d)
    })
}

/// How far a prompt's logits move when its prefill shares a step of more than 16 rows (the
/// tensor-core GEMM) instead of running alone (the FMA GEMM): relative L2, top-1 agreement,
/// and the top-1/top-2 margin alone — a kernel bug moves logits by O(1), rounding by ~1e-2.
fn mma_rounding(model_dir: &std::path::Path, token_map: Vec<i64>, text: &Text, layers: usize,
                keep: Option<Vec<Vec<bool>>>) -> tidelm_engine::Result<()> {
    let opts = Options { max_seq: 1024, max_lanes: 2, spec: false, resident: true, graphs: false, n_layers: Some(layers), keep,
                         dspark_keep: None };
    let mut e = Engine::new(model_dir, token_map, opts)?;
    let f = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    let filler: Vec<u32> = (0..24).map(|i| 1000 + i).collect();
    for p in ["The capital of France is", "def fibonacci(n):", "1, 2, 3, 5, 8, 13,", "A haiku about autumn:"] {
        let mut ids = vec![0u32];
        ids.extend(text.encode(p)?);
        e.reset();
        let a = e.forward(&ids, Want::Last)?;
        e.reset();
        let b = e.forward_batch(&[Seg { lane: 0, ids: &ids, want: Want::Last }, Seg { lane: 1, ids: &filler, want: Want::Nothing }])?.remove(0);
        let (mut num, mut den) = (0.0, 0.0);
        for (x, y) in a.iter().zip(&b) {
            num += (f(*x) - f(*y)).powi(2);
            den += f(*x).powi(2);
        }
        let (ta, tb) = (argmax_bf16(&a), argmax_bf16(&b));
        let mut sorted: Vec<f64> = a.iter().map(|&x| f(x)).collect();
        sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
        eprintln!("{p:32} rows {:2}+24: rel L2 {:.2e}, top-1 alone {ta} batched {tb}, alone margin {:.4} (top-1 {:.3})",
                  ids.len(), (num / den).sqrt(), sorted[0] - sorted[1], sorted[0]);
    }
    Ok(())
}
