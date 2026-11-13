//! The GLM-5.3-Flash vision tower on patches from a file, for tools/flash_vision_ref.py to compare
//! with transformers' Glm5NextVisionModel.
//!
//!   flash_vision_check --model-dir DIR --patches P.bin --grids GH,GW[;GH,GW...] --out OUT.bin [--max 16384]
//!
//! P.bin: bf16 [N][1176] (the processor's pixel_values, rounded to bf16); OUT.bin: bf16 [N / 4][4096].

use std::path::PathBuf;
use std::time::Instant;

use tidelm_glm::flash_vision::OwnedTower;
use tidelm_glm::Result;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--model-dir").ok_or("--model-dir")?);
    let max: usize = arg(&args, "--max").map(|s| s.parse()).transpose()?.unwrap_or(16384);
    let grids: Vec<(usize, usize)> = arg(&args, "--grids").ok_or("--grids")?.split(';')
        .map(|g| { let v: Vec<usize> = g.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    let raw = std::fs::read(arg(&args, "--patches").ok_or("--patches")?)?;
    let patches: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    tidelm_cuda::set_device(0)?;
    let st = tidelm_cuda::Stream::new()?;
    let ck = tidelm_model::Checkpoint::open(&dir)?;
    let t0 = Instant::now();
    let tw = OwnedTower::load(&ck, &st, max)?;
    eprintln!("tower loaded in {:.2} s ({:.2} GB with scratch for {max} patches)", t0.elapsed().as_secs_f64(),
              tidelm_glm::flash_vision::bytes_for(max) as f64 / 1e9);
    let n: usize = grids.iter().map(|&(h, w)| h * w).sum();
    let out = tidelm_cuda::DeviceBuffer::<u16>::zeroed(n / 4 * 4096)?;
    for rep in 0..3 {
        let t1 = Instant::now();
        let rows = tw.tower.forward(&st, &patches, &grids, out.raw())?;
        eprintln!("pass {rep}: {n} patches -> {rows} rows in {:.1} ms", 1e3 * t1.elapsed().as_secs_f64());
    }
    let v = out.download()?;
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(arg(&args, "--out").ok_or("--out")?, bytes)?;
    Ok(())
}
