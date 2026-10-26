//! The decode MoE (bm16 v2 up + down) at the serving pair's per-rank shapes, with and without the
//! two switches of fp4_moe_cuda_{up,down}_v2x: __ldcs weight loads (variant 1) and the (rows,
//! blocks) grid that walks one expert's rows at a time (variant 2); 3 is both. The serving
//! fp4_moe_cuda_{up,down}_v2 launch 1: in the engine 2 and 3 lost (tools/fp4_moe_cuda.cu::load_w).
//!
//!   cargo run --release -p tidelm-kernels --example bench_moe_bw
//!
//! Per rank (TP2 output layout): up n = 1,152 intermediate rows over k = 5,120, down n = 2,560
//! output rows over k = 2,304; a slot is 9.40 MB with its scales. The arena holds 320 slots
//! (3.0 GB), far past L2, and each case routes its tokens' top-6 uniformly over all of them, so a
//! call reads every touched expert cold as a decode step does. Times are per launch pair, from a
//! CUDA graph of REPS calls; every variant's h and parts must be variant 0's bits.

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::fp4;

const REPS: usize = 16;
const SLOTS: usize = 320;
const DIM: usize = 5120;
const INTER: usize = 2304;
const IL: usize = INTER / 2;
const DL: usize = DIM / 2;
const TOPK: usize = 6;
const SLOT_BYTES: usize = 3 * (IL * DIM / 2 + IL * DIM / 32);

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
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        g.launch(stream).unwrap();
        stream.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64() / REPS as f64);
    }
    best
}

fn main() {
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();
    let mut rng = Lcg(5);
    let bytes = |n: usize, rng: &mut Lcg, lo: u8, span: u64| -> DeviceBuffer<u8> {
        let v: Vec<u8> = (0..n).map(|_| lo + (rng.next() % span) as u8).collect();
        DeviceBuffer::from_slice(&v).unwrap()
    };
    let w1 = bytes(SLOTS * IL * DIM / 2, &mut rng, 0, 256);
    let w3 = bytes(SLOTS * IL * DIM / 2, &mut rng, 0, 256);
    let w2 = bytes(SLOTS * DL * INTER / 2, &mut rng, 0, 256);
    let s1 = bytes(SLOTS * IL * DIM / 32, &mut rng, 119, 6);
    let s3 = bytes(SLOTS * IL * DIM / 32, &mut rng, 119, 6);
    let s2 = bytes(SLOTS * DL * INTER / 32, &mut rng, 119, 6);
    let max_t = 16;
    let p_max = max_t * TOPK;
    let act = |n: usize, rng: &mut Lcg| DeviceBuffer::from_slice(&(0..n).map(|_| 0x3c00 + (rng.next() % 0x200) as u16).collect::<Vec<u16>>()).unwrap();
    let x = act(max_t * DIM, &mut rng);
    let hf = act(p_max * INTER, &mut rng);
    let rw = DeviceBuffer::from_slice(&vec![0.5f32; p_max]).unwrap();
    let (bs, bp, br) = (DeviceBuffer::<i32>::zeroed(p_max).unwrap(), DeviceBuffer::<i32>::zeroed(p_max * 16).unwrap(),
                        DeviceBuffer::<i32>::zeroed(p_max * 16).unwrap());
    let h = DeviceBuffer::<u16>::zeroed(p_max * IL).unwrap();
    let parts = DeviceBuffer::<f32>::zeroed(p_max * DL).unwrap();
    let r = |b: &DeviceBuffer<i32>| b.raw() as *mut c_void;
    let names = ["v2", "ldcs", "rows-first", "both"];
    println!("arena {SLOTS} slots x {:.2} MB = {:.2} GB; GB/s of the touched experts' bytes (up + down)", SLOT_BYTES as f64 / 1e6,
             (SLOTS * SLOT_BYTES) as f64 / 1e9);
    println!("{:>4} {:>5} {:>7}  {:>22} {:>22} {:>22} {:>22}", "rows", "pairs", "experts", names[0], names[1], names[2], names[3]);
    for &t in &[1usize, 2, 4, 6, 10, 16] {
        let p = t * TOPK;
        let mut keys = Vec::with_capacity(p);
        for _ in 0..t {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < TOPK {
                let e = (rng.next() % SLOTS as u64) as i32;
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
        let up = |s: &Stream, v: i32| unsafe {
            fp4::fp4_moe_cuda_up_v2x(s.raw(), x.raw(), w1.raw(), s1.raw(), w3.raw(), s3.raw(), h.raw(), rw.raw(), r(&bs), r(&br),
                                     DIM as i64, IL as i64, 10.0, TOPK as i32, IL as i32, DIM as i32, p as i32, -1, v)
        };
        let down = |s: &Stream, v: i32| unsafe {
            fp4::fp4_moe_cuda_down_v2x(s.raw(), hf.raw(), w2.raw(), s2.raw(), parts.raw(), r(&bs), r(&bp), INTER as i64, DL as i64,
                                       TOPK as i32, DL as i32, INTER as i32, t as i32, p as i32, -1, 0, 1, v)
        };
        // the bits: every variant against plain v2 (variant 0), and so the serving entry points (1)
        let outputs = |v: i32| -> (Vec<u16>, Vec<f32>) {
            h.zero_async(&stream).unwrap();
            parts.zero_async(&stream).unwrap();
            assert_eq!(up(&stream, v), 0);
            assert_eq!(down(&stream, v), 0);
            stream.synchronize().unwrap();
            (h.download().unwrap(), parts.download().unwrap())
        };
        let base = outputs(0);
        unsafe {
            h.zero_async(&stream).unwrap();
            parts.zero_async(&stream).unwrap();
            assert_eq!(fp4::fp4_moe_cuda_up_v2(stream.raw(), x.raw(), w1.raw(), s1.raw(), w3.raw(), s3.raw(), h.raw(), rw.raw(), r(&bs),
                                               r(&br), DIM as i64, IL as i64, 10.0, TOPK as i32, IL as i32, DIM as i32, p as i32, -1), 0);
            assert_eq!(fp4::fp4_moe_cuda_down_v2(stream.raw(), hf.raw(), w2.raw(), s2.raw(), parts.raw(), r(&bs), r(&bp), INTER as i64,
                                                 DL as i64, TOPK as i32, DL as i32, INTER as i32, t as i32, p as i32, -1, 0, 1), 0);
        }
        stream.synchronize().unwrap();
        assert!(h.download().unwrap() == base.0 && parts.download().unwrap().iter().zip(&base.1).all(|(a, b)| a.to_bits() == b.to_bits()),
                "the serving entry points changed the bits");
        let mut cells = Vec::new();
        for v in 0..4 {
            let o = outputs(v);
            assert!(o.0 == base.0 && o.1.iter().zip(&base.1).all(|(a, b)| a.to_bits() == b.to_bits()), "{} changed the bits at {t} rows", names[v as usize]);
            let tu = time_graph(&stream, |s| up(s, v));
            let td = time_graph(&stream, |s| down(s, v));
            let gbs = uniq.len() as f64 * SLOT_BYTES as f64 / (tu + td) / 1e9;
            cells.push(format!("{:6.1}+{:6.1} us {:4.0}", tu * 1e6, td * 1e6, gbs));
        }
        println!("{t:4} {p:5} {:7}  {:>22} {:>22} {:>22} {:>22}", uniq.len(), cells[0], cells[1], cells[2], cells[3]);
    }
}
