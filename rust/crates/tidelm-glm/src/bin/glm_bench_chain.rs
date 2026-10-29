//! Scratch: one matrix kind's decode GEMV over every MoE layer back to back (cold weights, distinct
//! per layer), each call timed by events. ONE=1: layer 3's matrix 75 times instead (one allocation);
//! LAYER=l: layer l's loaded 75 times (75 allocations); FLUSH=1: the head's GEMV before each call;
//! SYNC=1: the host waits after each call; DIRTY_MB: that many MB written before each call.
use std::path::PathBuf;
use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_glm::weights::{Cut, Exl3Mat, Exl3Scratch, Loader, Members, SPLIT_IQ, SPLIT_KVA, SPLIT_O, SPLIT_QA, SPLIT_QB,
                          SPLIT_DOWN, SPLIT_UP};
use tidelm_glm::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1).cloned());
    let env = |n: &str| std::env::var(n).ok();
    let dir = PathBuf::from(arg("--model-dir").ok_or("--model-dir")?);
    let what = arg("--mat").unwrap_or_else(|| "o_proj".into());
    let rows: usize = arg("--rows").map(|v| v.parse()).transpose()?.unwrap_or(1);
    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    let ck = tidelm_model::Checkpoint::open(&dir)?;
    let ld = Loader { ck: &ck, rank: 0, world: 2, st: &st };
    let members = Members::new()?;
    let scratch = Exl3Scratch::new(16, 16384, 8, 32 * 16 * 77440)?;
    let x = DeviceBuffer::<u16>::from_slice(&vec![0x3c00u16; 16 * 16384])?;
    let out = DeviceBuffer::<f32>::new(16 * 2 * 77440)?;
    let only: Option<usize> = env("LAYER").and_then(|v| v.parse().ok());
    let one = env("ONE").is_some();
    let mut mats: Vec<Exl3Mat> = vec![];
    for l0 in 3..78 {
        if one && !mats.is_empty() {
            break;
        }
        let l = only.unwrap_or(l0);
        let a = format!("model.layers.{l}.self_attn.");
        let m = format!("model.layers.{l}.mlp.shared_experts.");
        // (prefixes, the engine's split, cut); the indexer's wq_b exists on the 21 full layers only
        let (names, split, cut) = match what.as_str() {
            "o_proj" => (vec![format!("{a}o_proj")], SPLIT_O, Cut::K),
            "q_b" => (vec![format!("{a}q_b_proj")], SPLIT_QB, Cut::N),
            "q_a" => (vec![format!("{a}q_a_proj")], SPLIT_QA, Cut::None),
            "kv_a" => (vec![format!("{a}kv_a_proj_with_mqa")], SPLIT_KVA, Cut::None),
            "wq_b" => (vec![format!("{a}indexer.wq_b")], SPLIT_IQ, Cut::None),
            "sh_gu" => (vec![format!("{m}gate_proj"), format!("{m}up_proj")], SPLIT_UP, Cut::N),
            "sh_down" => (vec![format!("{m}down_proj")], SPLIT_DOWN, Cut::N),
            _ => return Err("--mat o_proj|q_b|q_a|kv_a|wq_b|sh_gu|sh_down".into()),
        };
        if what == "wq_b" && !ck.has(&format!("{a}indexer.wq_b.trellis")) {
            continue;
        }
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        mats.push(ld.exl3(&refs, split, cut, None)?);
    }
    for m in mats.iter_mut() {
        if let Some(pf) = env("PF").and_then(|v| v.parse::<i32>().ok()) {
            m.pf = pf;
        }
        if let Some(nt) = env("NT").and_then(|v| v.parse::<usize>().ok()) {
            m.nt = nt;
        }
        if let Some(sk) = env("SPLIT").and_then(|v| v.parse::<usize>().ok()) {
            m.split = sk;
        }
    }
    // CHECK=1: the staged path (pf -2/-3/-4) against the register path, byte for byte, at 1-16 rows
    if env("CHECK").is_some() {
        let mut seed = 0x1234_5678u64;
        let xr: Vec<u16> = (0..16 * 16384).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // bf16 in [-2, 2)
            let f = ((seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
            (f.to_bits() >> 16) as u16
        }).collect();
        let xr = DeviceBuffer::<u16>::from_slice(&xr)?;
        let n_out = 16 * 2 * 77440;
        let (mut a, mut b) = (vec![0f32; n_out], vec![0f32; n_out]);
        let mut bad = 0;
        for (li, m) in mats.iter_mut().enumerate().take(6) {
            for rows in [1usize, 2, 3, 4, 5, 8, 12, 16] {
                m.pf = 1;
                m.nt = 8;
                out.zero_async(&st)?;
                m.decode(&st, &members, &scratch, xr.raw(), m.k, rows, out.raw(), false)?;
                out.download_async(0, &mut a, &st)?;
                st.synchronize()?;
                for (pf, nt) in [(-2i32, 8usize), (-2, 4), (-3, 8)] {
                    m.pf = pf;
                    m.nt = nt;
                    out.zero_async(&st)?;
                    m.decode(&st, &members, &scratch, xr.raw(), m.k, rows, out.raw(), false)?;
                    out.download_async(0, &mut b, &st)?;
                    st.synchronize()?;
                    let used = rows * m.g * m.n;
                    let diff = (0..used).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
                    let nz = (0..used).filter(|&i| a[i] != 0.0).count();
                    if diff > 0 || nz == 0 {
                        bad += 1;
                        println!("{what} #{li} rows {rows} pf {pf} nt {nt}: {diff} of {used} differ ({nz} nonzero)");
                    }
                }
            }
        }
        println!("check {what}: {} mismatching cases", bad);
        return Ok(());
    }
    let seq: Vec<&Exl3Mat> = (0..75).map(|i| &mats[i % mats.len()]).collect();
    let bytes = mats[0].bytes();
    let dirty_mb: usize = env("DIRTY_MB").and_then(|v| v.parse().ok()).unwrap_or(0);
    let dirty = DeviceBuffer::<u8>::zeroed((dirty_mb << 20).max(1))?;
    let flusher = if env("FLUSH").is_some() { Some(ld.exl3(&["lm_head"], 1, Cut::N, None)?) } else { None };
    let bf16 = env("BF16").is_some();
    let evs: Vec<_> = (0..2 * seq.len()).map(|_| tidelm_cuda::Event::timing().unwrap()).collect();
    for round in 0..3 {
        for (i, m) in seq.iter().enumerate() {
            if let Some(f) = &flusher {
                f.decode(&st, &members, &scratch, x.raw(), f.k, 1, out.raw(), true)?;
            }
            if dirty_mb > 0 {
                dirty.zero_async(&st)?;
            }
            evs[2 * i].record(&st)?;
            m.decode(&st, &members, &scratch, x.raw(), m.k, rows, out.raw(), bf16)?;
            evs[2 * i + 1].record(&st)?;
            if env("SYNC").is_some() {
                evs[2 * i + 1].synchronize()?;
            }
        }
        evs[2 * seq.len() - 1].synchronize()?;
        let mut ts: Vec<f64> = (0..seq.len()).map(|i| evs[2 * i + 1].elapsed_ms(&evs[2 * i]).unwrap() as f64 * 1e3).collect();
        let wall = evs[2 * seq.len() - 1].elapsed_ms(&evs[0])? as f64 * 1e3;
        if round == 2 && env("PER").is_some() {
            println!("{}", ts.iter().map(|t| format!("{t:.0}")).collect::<Vec<_>>().join(" "));
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{what} x{} ({} loaded) rows {rows} round {round}: median {:.1} us ({:.0} GB/s), min {:.1}, max {:.1}; chain {:.0} us",
                 seq.len(), mats.len(), ts[ts.len() / 2], bytes as f64 / ts[ts.len() / 2] / 1e3, ts[0], ts[ts.len() - 1], wall);
    }
    Ok(())
}
