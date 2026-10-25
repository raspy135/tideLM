//! The indexer at decode shapes, by context: T rows (a verify block, both lanes) against n
//! compressed keys - the scores (FMA kernel, tensor-core kernel), the top-512 select, and layer
//! 20's candidate blocks - microseconds a call, the median of graph-free launches after a warmup.
//! A decode step runs the scores and the select in 8 layers (2, 8, 14 at n = context / 2;
//! 20-36 at n = context) and the candidate steps once.
//!
//!   cargo run --release -p tidelm-kernels --example bench_index_decode

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_candidate_blocks, ds_candidate_set, ds_index_scores, ds_index_scores_mma, ds_index_scores_mma_chunked,
                           ds_topk_select};

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let (h, dd, topk, block, kb) = (32usize, 128usize, 512usize, 8usize, 2048usize);
    let t: usize = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let p = |b: &DeviceBuffer<u16>| b.raw() as *mut c_void;
    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..3 {
            f();
        }
        st.synchronize().unwrap();
        let mut v: Vec<f64> = (0..9)
            .map(|_| {
                let t0 = Instant::now();
                f();
                st.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[4]
    };
    println!("rows {t}: microseconds a call");
    println!("{:>7}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "keys", "fma", "mma", "mma/1024", "topk512", "cand blk", "cand set");
    for &n in &[8192usize, 32768, 65536, 131072, 262144] {
        let q = DeviceBuffer::from_slice(&(0..t * h * dd).map(|i| 0x3c00u16 + (i % 97) as u16).collect::<Vec<_>>()).unwrap();
        let k = DeviceBuffer::from_slice(&(0..n * dd).map(|i| 0x3c00u16 ^ ((i * 2654435761) >> 22) as u16 & 0x3ff).collect::<Vec<_>>()).unwrap();
        let w = DeviceBuffer::from_slice(&vec![0.5f32; t * h]).unwrap();
        let clen = DeviceBuffer::from_slice(&vec![n as i32; t]).unwrap();
        let scores = DeviceBuffer::<u16>::zeroed(t * n).unwrap();
        let out = DeviceBuffer::<i32>::zeroed(t * topk).unwrap();
        let nb = n.div_ceil(block);
        let bmax = DeviceBuffer::<u16>::zeroed(t * nb).unwrap();
        let blocks = DeviceBuffer::<i32>::zeroed(t * kb).unwrap();
        let cand = DeviceBuffer::<u8>::zeroed(t * n).unwrap();
        let sc = |mma: bool| {
            let f = if mma { ds_index_scores_mma } else { ds_index_scores };
            unsafe {
                f(st.raw(), p(&q), p(&k), w.raw() as *const c_void, clen.raw() as *const c_void, std::ptr::null(), 0, n as i32,
                  p(&scores), n as i64, t as i32, std::ptr::null(), 0)
            }
        };
        let fma = time(&mut || assert_eq!(sc(false), 0));
        let mma = time(&mut || assert_eq!(sc(true), 0));
        assert_eq!(sc(true), 0);
        st.synchronize().unwrap();
        let whole = scores.download().unwrap();
        unsafe {
            ds_index_scores_mma_chunked(st.raw(), p(&q), p(&k), w.raw() as *const c_void, clen.raw() as *const c_void,
                                        std::ptr::null(), 0, n as i32, p(&scores), n as i64, t as i32, std::ptr::null(), 0, 1024);
        }
        st.synchronize().unwrap();
        assert!(scores.download().unwrap() == whole, "the chunked scores are the whole kernel's bits");
        let split = time(&mut || unsafe {
            assert_eq!(ds_index_scores_mma_chunked(st.raw(), p(&q), p(&k), w.raw() as *const c_void, clen.raw() as *const c_void,
                                                   std::ptr::null(), 0, n as i32, p(&scores), n as i64, t as i32, std::ptr::null(), 0, 1024), 0);
        });
        let sel = time(&mut || unsafe {
            assert_eq!(ds_topk_select(st.raw(), p(&scores), n as i64, n as i32, topk as i32, out.raw() as *mut c_void, topk as i32, t as i32), 0);
        });
        let cb = time(&mut || unsafe {
            assert_eq!(ds_candidate_blocks(st.raw(), p(&scores), n as i64, n as i32, block as i32, clen.raw() as *const c_void,
                                           p(&bmax), nb as i64, t as i32), 0);
            assert_eq!(ds_topk_select(st.raw(), p(&bmax), nb as i64, nb as i32, kb.min(nb) as i32, blocks.raw() as *mut c_void,
                                      kb.min(nb) as i32, t as i32), 0);
        });
        let cs = time(&mut || unsafe {
            assert_eq!(ds_candidate_set(st.raw(), blocks.raw() as *const c_void, kb.min(nb) as i32, block as i32, n as i32,
                                        cand.raw() as *mut c_void, n as i64, t as i32), 0);
        });
        println!("{n:>7}  {fma:>9.0} {mma:>9.0} {split:>9.0} {sel:>9.0} {cb:>9.0} {cs:>9.0}");
    }
}
