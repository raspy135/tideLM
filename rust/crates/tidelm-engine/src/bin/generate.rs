//! Greedy generation with the Rust engine, no Python anywhere in the pipeline.
//!
//!   tidelm-generate --prompt "The capital of France is" [--max-tokens 16]
//!                   [--prompt-file F] [--model-dir DIR] [--max-seq 1024] [--no-bos]
//!                   [--spec [--depth 5]] [--compare]
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

use tidelm_engine::decode::{generate_plain, generate_spec, Generation};
use tidelm_engine::forward::Engine;
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

    let t0 = Instant::now();
    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    let mut ids: Vec<u32> = if flag("--no-bos") { vec![] } else { vec![0] };
    ids.extend(text.encode(&prompt)?);
    eprintln!("prompt: {} tokens", ids.len());

    tidelm_cuda::set_device(0)?;
    let mut engine = Engine::new(&model_dir, token_map, max_seq, spec || compare)?;
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
    Ok(())
}
