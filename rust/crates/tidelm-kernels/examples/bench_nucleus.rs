//! The sampled head's cost: ds_nucleus against the greedy argmax it replaces, at the decode
//! shapes (the drafter's markov step: 1..4 lanes with a bias row, K = 512; the verify rows:
//! 1..6, K = the cap), microseconds a launch from a graph of REPS launches.
//!
//!   cargo run --release -p tidelm-kernels --example bench_nucleus

use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_argmax_sum_bf16, ds_nucleus, ds_nucleus_cap};

const REPS: usize = 64;
const V: usize = 129_280;

fn time(stream: &Stream, launch: &dyn Fn(&Stream) -> i32) -> f64 {
    assert_eq!(launch(stream), 0);
    stream.synchronize().unwrap();
    let g = stream
        .capture(|s| {
            for _ in 0..REPS {
                check(launch(s), "launch")?;
            }
            Ok(())
        })
        .unwrap();
    g.launch(stream).unwrap();
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..5 {
        g.launch(stream).unwrap();
    }
    stream.synchronize().unwrap();
    t.elapsed().as_secs_f64() / (5 * REPS) as f64 * 1e6
}

fn main() {
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();
    let cap = unsafe { ds_nucleus_cap() } as usize;
    // logits like a head's: normal, scale 3 (a few hundred tokens in a 0.6 / 0.95 nucleus)
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut normal = || {
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32).max(1e-7)
        };
        let (a, b) = (u(), u());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    };
    let rows = 8;
    let bf = |x: f32| ((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16;
    let a = DeviceBuffer::from_slice(&(0..rows * V).map(|_| bf(normal() * 3.0)).collect::<Vec<_>>()).unwrap();
    let b = DeviceBuffer::from_slice(&(0..rows * V).map(|_| bf(normal() * 1.5)).collect::<Vec<_>>()).unwrap();
    let temp = DeviceBuffer::from_slice(&vec![0.6f32; rows]).unwrap();
    let greedy = DeviceBuffer::from_slice(&vec![0f32; rows]).unwrap();
    let top_p = DeviceBuffer::from_slice(&vec![0.95f32; rows]).unwrap();
    let u = DeviceBuffer::from_slice(&vec![0.5f32; rows]).unwrap();
    let ids = DeviceBuffer::<i32>::zeroed(rows * cap).unwrap();
    let probs = DeviceBuffer::<f32>::zeroed(rows * cap).unwrap();
    let cnt = DeviceBuffer::<i32>::zeroed(rows).unwrap();
    let tok = DeviceBuffer::<i32>::zeroed(rows).unwrap();
    println!("{:>12} {:>4} {:>8} {:>10} {:>10}", "shape", "rows", "argmax", "nucleus", "greedy rows");
    for &(what, m, bias, k) in &[("draft", 1usize, true, 512usize), ("draft", 4, true, 512), ("verify", 1, false, cap),
                                 ("verify", 4, false, cap), ("verify", 6, false, cap)] {
        let bp = if bias { b.raw() } else { std::ptr::null_mut() };
        let am = time(&stream, &|s| unsafe { ds_argmax_sum_bf16(s.raw(), a.raw(), V as i64, bp, V as i64, m as i32, V as i32, tok.raw()) });
        let nuc = |t: &DeviceBuffer<f32>| {
            time(&stream, &|s| unsafe {
                ds_nucleus(s.raw(), a.raw(), V as i64, bp, V as i64, m as i32, V as i32, t.raw(), top_p.raw(), u.raw(), k as i32,
                           ids.raw(), probs.raw(), cap as i64, cnt.raw(), tok.raw())
            })
        };
        let (ns, ng) = (nuc(&temp), nuc(&greedy));
        let c = cnt.download().unwrap();
        println!("{what:>12} {m:4} {am:8.1} {ns:10.1} {ng:10.1}   (nucleus sizes {:?})", &c[..m]);
    }
}
