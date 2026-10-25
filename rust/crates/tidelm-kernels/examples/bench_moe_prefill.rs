//! The routed-expert MoE at a prefill step's size: 2,048 rows x top-6 spread over 250 experts
//! (a layer's residents), this rank's tensor-parallel shard (1,152 intermediate rows; down
//! 2,560 output rows), routed by ds_route_large in 16- and 64-pair blocks. Milliseconds per
//! up + down and TFLOP/s.
//!
//!   cargo run --release -p tidelm-kernels --example bench_moe_prefill
use std::time::Instant;

use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_moe_down_mma, ds_moe_prefill_down, ds_moe_prefill_up, ds_moe_up_mma, ds_route_blocks, ds_route_large};

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let (slots, dim, inter, topk) = (250usize, 5120usize, 1152usize, 6usize);
    let dl = 2560usize; // down: this rank's output rows over the whole intermediate
    let mut s = 7u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as usize
    };
    let w1 = DeviceBuffer::<u8>::zeroed(slots * inter * dim / 2).unwrap();
    let w3 = DeviceBuffer::<u8>::zeroed(slots * inter * dim / 2).unwrap();
    let s13 = DeviceBuffer::from_slice(&vec![127u8; slots * inter * dim / 32]).unwrap();
    let w2 = DeviceBuffer::<u8>::zeroed(slots * dl * 2304 / 2).unwrap();
    let s2 = DeviceBuffer::from_slice(&vec![127u8; slots * dl * 2304 / 32]).unwrap();
    // BENCH_SPREAD=n: route over n experts only (n >= topk) - weights that stay in L2, the
    // same FLOPs: the kernel's compute side alone
    let spread: usize = std::env::var("BENCH_SPREAD").ok().and_then(|v| v.parse().ok()).unwrap_or(slots);
    for t in [128usize, 256, 512, 1024, 2048] {
        let p = t * topk;
        let mut keys = Vec::with_capacity(p);
        for _ in 0..t {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                let e = (next() % spread) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            keys.extend(picked);
        }
        let dk = DeviceBuffer::from_slice(&keys).unwrap();
        let x = DeviceBuffer::from_slice(&vec![0x3c00u16; t * dim]).unwrap();
        let rw = DeviceBuffer::from_slice(&vec![0.5f32; p]).unwrap();
        let h = DeviceBuffer::<u16>::zeroed(p * 2304).unwrap();
        let parts = DeviceBuffer::<f32>::zeroed(p * dl).unwrap();
        // block entries for the largest blocks: ds_route_blocks(p, slots, 64) * 64 <= p * 64
        let (bs, bp, br) = (DeviceBuffer::<i32>::zeroed(p).unwrap(), DeviceBuffer::<i32>::zeroed(p * 64).unwrap(),
                            DeviceBuffer::<i32>::zeroed(p * 64).unwrap());
        let work = DeviceBuffer::<i32>::zeroed(3 * slots).unwrap();
        // bm 0: the grouped-GEMM prefill kernels over 64-pair blocks
        for bm in [16i32, 32, 64, 0] {
            let go = || unsafe {
                let rb = if bm == 0 { 64 } else { bm };
                assert_eq!(ds_route_large(st.raw(), dk.raw(), p as i32, topk as i32, slots as i32, work.raw(), bs.raw(), bp.raw(),
                                          br.raw(), rb), 0);
                // the grid the engine launches: the block bound, not one block a pair
                let nb = ds_route_blocks(p as i32, slots as i32, rb);
                // BENCH_ONLY=up|down: one of the prefill kernels alone
                let only = std::env::var("BENCH_ONLY").unwrap_or_default();
                if bm == 0 && only == "down" {
                    assert_eq!(ds_moe_prefill_down(st.raw(), h.raw(), 2304, w2.raw(), s2.raw(), parts.raw(), dl as i64, bs.raw(),
                                                   br.raw(), nb, dl as i32, 2304, topk as i32, t as i32, -1, 0, 1), 0);
                    return;
                }
                if bm == 0 && only == "up" {
                    assert_eq!(ds_moe_prefill_up(st.raw(), x.raw(), dim as i64, w1.raw(), s13.raw(), w3.raw(), s13.raw(), h.raw(),
                                                 2304, rw.raw(), bs.raw(), br.raw(), nb, inter as i32, dim as i32, 10.0, -1), 0);
                    return;
                }
                if bm == 0 {
                    assert_eq!(ds_moe_prefill_up(st.raw(), x.raw(), dim as i64, w1.raw(), s13.raw(), w3.raw(), s13.raw(), h.raw(),
                                                 2304, rw.raw(), bs.raw(), br.raw(), nb, inter as i32, dim as i32, 10.0, -1), 0);
                    assert_eq!(ds_moe_prefill_down(st.raw(), h.raw(), 2304, w2.raw(), s2.raw(), parts.raw(), dl as i64, bs.raw(),
                                                   br.raw(), nb, dl as i32, 2304, topk as i32, t as i32, -1, 0, 1), 0);
                    return;
                }
                assert_eq!(ds_moe_up_mma(st.raw(), x.raw(), dim as i64, w1.raw(), s13.raw(), w3.raw(), s13.raw(), h.raw(), inter as i64,
                                         rw.raw(), bs.raw(), br.raw(), nb, inter as i32, dim as i32, 10.0, -1, bm), 0);
                // down reads the gathered whole intermediate (2,304)
                assert_eq!(ds_moe_down_mma(st.raw(), h.raw(), 2304, w2.raw(), s2.raw(), parts.raw(), dl as i64, bs.raw(), bp.raw(),
                                           nb, dl as i32, 2304, topk as i32, t as i32, -1, 0, 1, bm), 0);
            };
            go();
            st.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..5 {
                go();
            }
            st.synchronize().unwrap();
            let dt = t0.elapsed().as_secs_f64() / 5.0;
            let fl = 2.0 * p as f64 * (2.0 * inter as f64 * dim as f64 + dl as f64 * 2304.0);
            println!("rows {t:5} bm {bm:2}: {:7.2} ms  {:5.1} TFLOP/s", dt * 1e3, fl / dt / 1e12);
        }
    }
}
