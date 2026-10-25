//! Routed-expert reads from NVMe as the boot load and residency swaps make them: whole experts
//! (18.8 MB) or one tensor-parallel rank's share (the scale run + its half of each weight,
//! ~10 MB), from `threads` workers each with its own staging buffer.
//!
//!   cargo run --release -p tidelm-model --example bench_expert_read [experts] [first_layer]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tidelm_model::direct_io::AlignedBuf;
use tidelm_model::{Checkpoint, ExpertReader, STAGE_BYTES};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(256);
    let first: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(0);
    let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("models/DeepSeek-V4.1-Flash-1017");
    let ckpt = Checkpoint::open(&dir).unwrap();
    let mut round = 0usize;
    for share in [false, true] {
        for threads in [1usize, 2, 4, 8, 10, 16] {
            // a fresh range of experts each run, so no device cache serves a repeat
            let base = first * 384 + round * n;
            round += 1;
            let next = AtomicUsize::new(0);
            let bytes = AtomicUsize::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut stage = AlignedBuf::new(STAGE_BYTES).unwrap();
                        let reader = ExpertReader::new(&ckpt);
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n {
                                break;
                            }
                            let (l, e) = ((base + i) / 384 % 40, (base + i) % 384);
                            let p = ExpertReader::expert_prefix(l, e);
                            let v = if share {
                                reader.read_expert_share_into(&mut stage, &p, 0, 2).unwrap()
                            } else {
                                reader.read_expert_into(&mut stage, &p).unwrap()
                            };
                            bytes.fetch_add(v.iter().map(|x| x.1).sum::<usize>(), Ordering::Relaxed);
                        }
                    });
                }
            });
            let dt = t0.elapsed().as_secs_f64();
            println!("{:5} threads {threads:2}: {:6.1} experts/s  {:5.2} GB/s of tensor bytes", if share { "share" } else { "whole" },
                     n as f64 / dt, bytes.load(Ordering::Relaxed) as f64 / dt / 1e9);
        }
    }
}
