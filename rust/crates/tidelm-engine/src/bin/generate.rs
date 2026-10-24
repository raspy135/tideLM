//! Greedy generation with the Rust engine, no Python anywhere in the pipeline.
//!
//!   tidelm-generate --prompt "The capital of France is" [--max-tokens 16]
//!                   [--model-dir DIR] [--max-seq 1024] [--no-bos]
//!
//! Streaming mode (see tidelm_engine::forward): every step reads the weights from NVMe,
//! so expect seconds per token. It exists to check the forward pass end to end in ~1 GB.

use std::path::PathBuf;
use std::time::Instant;

use tidelm_engine::forward::{Engine, TMAX};
use tidelm_text::Text;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn argmax_bf16(logits: &[u16]) -> u32 {
    let f = |h: u16| f32::from_bits((h as u32) << 16);
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if f(v) > f(logits[best]) {
            best = i;
        }
    }
    best as u32
}

fn main() -> tidelm_engine::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = arg(&args, "--model-dir").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017")
    });
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "The capital of France is".into());
    let max_tokens: usize = arg(&args, "--max-tokens").map(|v| v.parse().unwrap()).unwrap_or(16);
    let max_seq: usize = arg(&args, "--max-seq").map(|v| v.parse().unwrap()).unwrap_or(1024);
    let bos = !args.iter().any(|a| a == "--no-bos");

    let t0 = Instant::now();
    let text = Text::load(&model_dir)?;
    let (token_map, _) = text.compressed_token_map()?;
    let mut ids: Vec<u32> = if bos { vec![0] } else { vec![] };
    ids.extend(text.encode(&prompt)?);
    eprintln!("prompt: {} tokens {:?}", ids.len(), ids);

    tidelm_cuda::set_device(0)?;
    let mut engine = Engine::new(&model_dir, token_map, max_seq)?;
    eprintln!("engine up in {:.1}s", t0.elapsed().as_secs_f64());

    let tp = Instant::now();
    let mut logits = None;
    let chunks: Vec<&[u32]> = ids.chunks(TMAX).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        logits = engine.forward(chunk, i + 1 == chunks.len())?;
    }
    let mut tok = argmax_bf16(logits.as_ref().unwrap());
    eprintln!("prefill {:.1}s  {:?}", tp.elapsed().as_secs_f64(), engine.times);

    let mut out = vec![tok];
    print!("{}{}", prompt, text.decode(&[tok])?);
    use std::io::Write;
    std::io::stdout().flush().ok();
    for _ in 1..max_tokens {
        if tok == 1 {
            break; // <｜end▁of▁sentence｜>
        }
        let ts = Instant::now();
        engine.times = Default::default();
        let lg = engine.forward(&[tok], true)?.unwrap();
        tok = argmax_bf16(&lg);
        out.push(tok);
        print!("{}", text.decode(&[tok])?);
        std::io::stdout().flush().ok();
        eprintln!("\n[step {:.2}s {:?}]", ts.elapsed().as_secs_f64(), engine.times);
    }
    println!();
    eprintln!("generated ids: {out:?}");
    Ok(())
}
