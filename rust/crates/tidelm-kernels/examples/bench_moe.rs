//! The routed-expert MoE (up + down) at DS v4.1's shapes: fp4_moe_cuda.cu's bm16 kernels
//! against ds41.cu's tensor-core ones, by pairs per call and how many experts they spread
//! over.
//!
//!   cargo run --release -p tidelm-kernels --example bench_moe
//!
//! The arena holds 192 experts (3.6 GB), more than L2 can hide, and each case routes its
//! tokens' top-6 uniformly over `spread` of them, so a call reads every touched expert cold
//! the way a decode step does. Times are per up + down pair of launches, from a CUDA graph
//! of REPS calls.

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_moe_down_mma, ds_moe_up_mma};
use tidelm_kernels::fp4;

const REPS: usize = 16;
const SLOTS: usize = 192;
const DIM: usize = 5120;
const INTER: usize = 2304;
const TOPK: usize = 6;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn time_graph(stream: &Stream, mut launch: impl FnMut(&Stream) -> i32) -> f64 {
    assert_eq!(launch(stream), 0);
    stream.synchronize().unwrap();
    let g = stream
        .capture(|s| {
            for _ in 0..REPS {
                let rc = launch(s);
                if rc != 0 {
                    return check(rc, "launch");
                }
            }
            Ok(())
        })
        .unwrap();
    g.launch(stream).unwrap();
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..3 {
        g.launch(stream).unwrap();
    }
    stream.synchronize().unwrap();
    t.elapsed().as_secs_f64() / (3 * REPS) as f64
}

fn main() {
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();
    let mut rng = Lcg(5);
    let fill = |n: usize, rng: &mut Lcg, lo: u8, span: u64| -> DeviceBuffer<u8> {
        let v: Vec<u8> = (0..n).map(|_| lo + (rng.next() % span) as u8).collect();
        DeviceBuffer::from_slice(&v).unwrap()
    };
    let w1 = fill(SLOTS * INTER * DIM / 2, &mut rng, 0, 256);
    let w3 = fill(SLOTS * INTER * DIM / 2, &mut rng, 0, 256);
    let w2 = fill(SLOTS * DIM * INTER / 2, &mut rng, 0, 256);
    let s1 = fill(SLOTS * INTER * DIM / 32, &mut rng, 119, 6);
    let s3 = fill(SLOTS * INTER * DIM / 32, &mut rng, 119, 6);
    let s2 = fill(SLOTS * DIM * INTER / 32, &mut rng, 119, 6);
    let max_t = 64;
    let xv: Vec<u16> = (0..max_t * DIM).map(|_| 0x3c00 + (rng.next() % 0x200) as u16).collect();
    let x = DeviceBuffer::from_slice(&xv).unwrap();
    let rw = DeviceBuffer::from_slice(&vec![0.5f32; max_t * TOPK]).unwrap();
    let p_max = max_t * TOPK;
    let (bs, bp, br) = (DeviceBuffer::<i32>::zeroed(p_max).unwrap(), DeviceBuffer::<i32>::zeroed(p_max * 16).unwrap(),
                        DeviceBuffer::<i32>::zeroed(p_max * 16).unwrap());
    let h = DeviceBuffer::<u16>::zeroed(p_max * INTER).unwrap();
    let parts = DeviceBuffer::<f32>::zeroed(p_max * DIM).unwrap();
    let r = |b: &DeviceBuffer<i32>| b.raw() as *mut c_void;
    println!("{:>5} {:>6} {:>8}  {:>10} {:>10}  {:>7}", "rows", "pairs", "experts", "bm16 us", "mma us", "mma GB/s");
    for &(t, spread) in &[(6usize, 192usize), (10, 192), (16, 192), (24, 192), (48, 192), (64, 192), (48, 32), (64, 16)] {
        let p = t * TOPK;
        let mut keys = Vec::with_capacity(p);
        for _ in 0..t {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < TOPK {
                let e = (rng.next() % spread as u64) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            keys.extend(picked);
        }
        let mut uniq = keys.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let dk = DeviceBuffer::from_slice(&keys).unwrap();
        let rc = unsafe { fp4::fp4_moe_cuda_route_small(stream.raw(), dk.raw(), r(&bs), r(&bp), r(&br), p as i32, TOPK as i32) };
        assert_eq!(rc, 0);
        let bm16 = time_graph(&stream, |s| unsafe {
            let rc = fp4::fp4_moe_cuda_up_v2(s.raw(), x.raw(), w1.raw(), s1.raw(), w3.raw(), s3.raw(), h.raw(), rw.raw(),
                                             r(&bs), r(&br), DIM as i64, INTER as i64, 10.0, TOPK as i32, INTER as i32,
                                             DIM as i32, p as i32, -1);
            if rc != 0 {
                return rc;
            }
            fp4::fp4_moe_cuda_down_v2(s.raw(), h.raw(), w2.raw(), s2.raw(), parts.raw(), r(&bs), r(&bp), INTER as i64,
                                      DIM as i64, TOPK as i32, DIM as i32, INTER as i32, t as i32, p as i32, -1, 0, 1)
        });
        let mma = time_graph(&stream, |s| unsafe {
            let rc = ds_moe_up_mma(s.raw(), x.raw(), DIM as i64, w1.raw(), s1.raw(), w3.raw(), s3.raw(), h.raw(), INTER as i64,
                                   rw.raw(), r(&bs), r(&br), p as i32, INTER as i32, DIM as i32, 10.0, -1, 16);
            if rc != 0 {
                return rc;
            }
            ds_moe_down_mma(s.raw(), h.raw(), INTER as i64, w2.raw(), s2.raw(), parts.raw(), DIM as i64, r(&bs), r(&bp),
                            p as i32, DIM as i32, INTER as i32, TOPK as i32, t as i32, -1, 0, 1, 16)
        });
        let bytes = uniq.len() * 3 * INTER * DIM / 2;
        println!("{t:5} {p:6} {:8}  {:10.1} {:10.1}  {:7.0}", uniq.len(), bm16 * 1e6, mma * 1e6, bytes as f64 / mma / 1e9);
    }
}
