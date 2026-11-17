//! Inkling 2's vision tower on one image's patches, for tools/inkling_vision_ref.py to compare with
//! a PyTorch transcription of transformers' Inkling2VisionModel.
//!
//!   inkling_vision_check --model-dir DIR --out OUT.bin [--rows 3 --cols 5]
//!                        (--patches-out P.bin [--seed 1] | --patches P.bin) [--max-tokens N]
//!
//! P.bin: bf16 [rows * cols * 4][1200] (token-major, a token's 2 x 2 patches in (in_row, in_col)
//! order, each 20 x 20 x 3 as (y, x, channel), normalized to [-1, 1]); OUT.bin: bf16 [rows * cols][4608].
//! With --patches-out the patches are synthetic - a smooth picture plus noise from --seed, a pure
//! function of the arguments - and written there for the reference to read; with --patches they are
//! read (a preprocessor's). TIDELM_VISION_DUMP=PREFIX also writes the sequence after the embeddings,
//! ln_pre, each layer and the merger's first half (PREFIX.{embed,pre,b0..b22,merge}.bin).

use std::path::PathBuf;
use std::time::Instant;

use tidelm_glm::inkling_vision::{self, OwnedInkVision, OUT, PATCH_K, PER_TOKEN};
use tidelm_glm::Result;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn to_bf(f: f32) -> u16 {
    let b = f.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A picture of rows x cols tokens as the tower takes it: per pixel and channel a smooth wave over the
/// whole image plus uniform noise (splitmix64 from `seed`), in [-1, 1], rounded to bf16.
fn synthetic(rows: usize, cols: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    let mut uniform = move || -> f32 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let mut v = Vec::with_capacity(rows * cols * PER_TOKEN * PATCH_K);
    for tr in 0..rows {
        for tc in 0..cols {
            for ir in 0..2 {
                for ic in 0..2 {
                    for y in 0..20 {
                        for x in 0..20 {
                            let (gy, gx) = (((2 * tr + ir) * 20 + y) as f32, ((2 * tc + ic) * 20 + x) as f32);
                            for c in 0..3 {
                                let wave = (0.045 * gy + 0.9 * c as f32).sin() * (0.031 * gx - 0.4 * c as f32).cos();
                                v.push(to_bf((0.6 * wave + 0.4 * uniform()).clamp(-1.0, 1.0)));
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--model-dir").ok_or("--model-dir")?);
    let rows: usize = arg(&args, "--rows").map(|s| s.parse()).transpose()?.unwrap_or(3);
    let cols: usize = arg(&args, "--cols").map(|s| s.parse()).transpose()?.unwrap_or(5);
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let max: usize = arg(&args, "--max-tokens").map(|s| s.parse()).transpose()?.unwrap_or(rows * cols);
    let out_path = arg(&args, "--out").ok_or("--out")?;
    let patches: Vec<u16> = match (arg(&args, "--patches"), arg(&args, "--patches-out")) {
        (Some(p), None) => std::fs::read(p)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        (None, Some(p)) => {
            let v = synthetic(rows, cols, seed);
            std::fs::write(p, bytes(&v))?;
            v
        }
        _ => return Err("one of --patches P.bin (read) or --patches-out P.bin (synthetic, written)".into()),
    };
    tidelm_cuda::set_device(0)?;
    let st = tidelm_cuda::Stream::new()?;
    let t0 = Instant::now();
    let tw = OwnedInkVision::open(&dir, &st, max)?;
    eprintln!("tower loaded in {:.2} s: {:.1} MB on the device ({:.1} MB of weights, scratch for {max} tokens)",
              t0.elapsed().as_secs_f64(), tw.tower.device_bytes() as f64 / 1e6, inkling_vision::weight_bytes() as f64 / 1e6);
    let out = tidelm_cuda::DeviceBuffer::<u16>::zeroed(rows * cols * OUT)?;
    for rep in 0..3 {
        let t1 = Instant::now();
        tw.tower.encode(&st, &patches, rows, cols, out.raw())?;
        eprintln!("pass {rep}: {rows} x {cols} tokens ({} patches) in {:.1} ms", rows * cols * PER_TOKEN, 1e3 * t1.elapsed().as_secs_f64());
    }
    // an image past the scratch is an error, not a write past it
    let over = vec![0u16; (max + 1) * PER_TOKEN * PATCH_K];
    match tw.tower.encode(&st, &over, 1, max + 1, out.raw()) {
        Err(e) => eprintln!("an image of {} tokens: refused ({e})", max + 1),
        Ok(()) => return Err(format!("an image of {} tokens was encoded with scratch for {max}", max + 1).into()),
    }
    let v = out.download()?;
    let (mut nan, mut sum) = (0usize, 0f64);
    for &x in &v {
        let f = f32::from_bits((x as u32) << 16);
        if f.is_finite() { sum += (f as f64) * (f as f64) } else { nan += 1 }
    }
    eprintln!("{} rows x {OUT}: rms {:.4}, {nan} non-finite", rows * cols, (sum / v.len() as f64).sqrt());
    std::fs::write(&out_path, bytes(&v))?;
    if nan > 0 {
        return Err(format!("{nan} non-finite feature values").into());
    }
    Ok(())
}
