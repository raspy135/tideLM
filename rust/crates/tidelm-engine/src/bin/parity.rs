//! Mode parity: the same prompt and greedy steps through the streaming engine, the resident
//! one, and the resident one with every step a CUDA graph. All three must agree bit for bit
//! — they share every kernel and differ only in where weights come from and how the step's
//! launches are issued. Also reports per-step wall time, resident eager vs graph replay.
//!
//!   tidelm-parity [--layers 2] [--keep-first 16] [--steps 4] [--prompt "..."]
//!
//! Defaults fit next to a live server: two layers, 16 routable experts per layer, ~4 GB.

use std::path::PathBuf;

use tidelm_engine::decode::argmax_bf16;
use tidelm_engine::forward::{Engine, Options, Want};
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
    let base = Options { max_seq: 1024, spec: false, resident: false, graphs: false, n_layers: Some(layers), keep };

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
