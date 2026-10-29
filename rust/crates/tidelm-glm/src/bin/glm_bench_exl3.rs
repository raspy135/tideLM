//! Decode-GEMV split-K sweep for one layer's EXL3 matrices (rank 0's cut) on one GPU: the time a
//! T-row call takes at each split that divides K, L2-cold (each timed call reads a matrix the
//! previous ones evicted), so the engine's per-matrix split counts can be chosen from it.
//!
//!   glm_bench_exl3 --model-dir DIR [--layer 40]

use std::path::PathBuf;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_glm::weights::{Cut, Exl3Scratch, Loader, Members};
use tidelm_glm::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1).cloned());
    let dir = PathBuf::from(arg("--model-dir").ok_or("--model-dir")?);
    let layer: usize = arg("--layer").map(|v| v.parse()).transpose()?.unwrap_or(40);
    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    let ck = tidelm_model::Checkpoint::open(&dir)?;
    let ld = Loader { ck: &ck, rank: 0, world: 2, st: &st };
    let a = format!("model.layers.{layer}.self_attn.");
    let m = format!("model.layers.{layer}.mlp.shared_experts.");
    let mats: Vec<(&str, Vec<String>, Cut)> = vec![
        ("q_a", vec![format!("{a}q_a_proj")], Cut::None),
        ("kv_a", vec![format!("{a}kv_a_proj_with_mqa")], Cut::None),
        ("q_b", vec![format!("{a}q_b_proj")], Cut::N),
        ("o_proj", vec![format!("{a}o_proj")], Cut::K),
        ("shared gate+up", vec![format!("{m}gate_proj"), format!("{m}up_proj")], Cut::N),
        ("shared down", vec![format!("{m}down_proj")], Cut::N),
        ("head", vec!["lm_head".into()], Cut::N),
    ];
    let members = Members::new()?;
    let scratch = Exl3Scratch::new(16, 16384, 8, 32 * 16 * 77440)?;
    // The L2 is evicted between calls by a read: the head's decode GEMV, 357 MB in, 0.3 MB out.
    // FLUSH=memset (FLUSH_MB, default 256) is the first version's flusher, kept to reproduce its
    // numbers: the timed call then writes the memset's dirty lines back as it reads, and q_b
    // measured 105-112 us (95 GB/s) against 63-72 us after the read and 58-67 in the decode pass.
    let memset = std::env::var("FLUSH").map(|v| v == "memset").unwrap_or(false);
    let flush_mb: usize = std::env::var("FLUSH_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let flush = DeviceBuffer::<u8>::zeroed(if memset { flush_mb << 20 } else { 1 })?;
    let x = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; 16 * 16384])?;
    let out = DeviceBuffer::<f32>::new(16 * 2 * 77440)?;
    let flusher = if memset { None } else { Some(ld.exl3(&["lm_head"], 1, Cut::N, None)?) };
    let fout = DeviceBuffer::<f32>::new(2 * 77440)?;
    // DIRTY_MB: that many MB written after the flush, before each timed call (lines the call evicts dirty)
    let dirty_mb: usize = std::env::var("DIRTY_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dirty = if dirty_mb > 0 { Some(DeviceBuffer::<u8>::zeroed(dirty_mb << 20)?) } else { None };
    for (name, pre, cut) in mats {
        let refs: Vec<&str> = pre.iter().map(|s| s.as_str()).collect();
        let mut mat = ld.exl3(&refs, 1, cut, None)?;
        let kt = mat.k / 16;
        let bytes = mat.bytes();
        let mut line = format!("{name:>15} K {:>5} N {:>6} x{}: ", mat.k, mat.n, mat.g);
        for &(t, pf, nt) in &[(1usize, 1i32, 8usize), (1, -2, 8), (8, 1, 8), (8, -2, 8)] {
            line += &format!(" T{t}nt{nt}:");
            mat.pf = pf;
            mat.nt = nt;
            for sk in [1usize, 2, 4, 8, 16] {
                if kt % (sk * 4) != 0 {
                    continue;
                }
                mat.split = sk;
                let mut ts = vec![];
                let (e0, e1) = (tidelm_cuda::Event::timing()?, tidelm_cuda::Event::timing()?);
                for i in 0..12 {
                    if std::env::var("WARM").is_ok() {
                    } else if let Some(f) = &flusher {
                        f.decode(&st, &members, &scratch, x.raw(), f.k, 1, fout.raw(), true)?;
                    } else {
                        flush.zero_async(&st)?;
                    }
                    if let Some(d) = &dirty {
                        d.zero_async(&st)?;
                    }
                    e0.record(&st)?;
                    mat.decode(&st, &members, &scratch, x.raw(), mat.k, t, out.raw(), true)?;
                    e1.record(&st)?;
                    e1.synchronize()?;
                    if i >= 2 {
                        ts.push(e1.elapsed_ms(&e0)? as f64 * 1e3);
                    }
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let us = ts[ts.len() / 2];
                line += &format!(" s{sk} {us:.0}us({:.0}GB/s)", bytes as f64 / us / 1e3);
            }
        }
        println!("{line}");
    }
    Ok(())
}
