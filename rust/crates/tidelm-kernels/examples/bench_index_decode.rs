//! The indexer at decode shapes, by context: T rows (a verify block, both lanes) against n
//! compressed keys - the scores (FMA kernel, tensor-core kernel), the top-512 select, and layer
//! 20's candidate blocks - microseconds a call, the median of graph-free launches after a warmup.
//! A decode step runs the scores and the select in 8 layers (2, 8, 14 at n = context / 2;
//! 20-36 at n = context) and the candidate steps once. "topk/clen" is the select reading each
//! row's keys up to a clen of 9/16 of the range (148K tokens in the 262K bucket); "list" is a
//! later layer's candidate path (scores of 2,048 random blocks, their select, the positions)
//! at that clen; "mask 0" is the memset of the 2,048 x 524,289-byte mask the list replaced.
//!
//!   cargo run --release -p tidelm-kernels --example bench_index_decode

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_cand_positions, ds_candidate_blocks, ds_candidate_set, ds_index_cand_scores, ds_index_scores,
                           ds_index_scores_mma, ds_index_scores_mma_chunked, ds_topk_select, ds_topk_select_lim};

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
    let mask = DeviceBuffer::<u8>::zeroed(2048 * 524289).unwrap();
    let mask0 = time(&mut || mask.zero_async(&st).unwrap());
    drop(mask);
    println!("rows {t}: microseconds a call; mask 0 {mask0:.0}");
    println!("{:>7}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "keys", "fma", "mma", "mma/1024", "topk512", "topk/clen", "list",
             "cand blk", "cand set");
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
        let short = DeviceBuffer::from_slice(&vec![(n / 16 * 9) as i32; t]).unwrap();
        let sel_lim = time(&mut || unsafe {
            assert_eq!(ds_topk_select_lim(st.raw(), p(&scores), n as i64, n as i32, topk as i32, out.raw() as *mut c_void, topk as i32,
                                          t as i32, short.raw() as *const c_void), 0);
        });
        // 2,048 distinct blocks under the short clen, ascending (an xorshift's picks)
        let (vis, kl) = (n / 16 * 9 / block, kb.min(nb / 16 * 9));
        let mut x = 88172645463325252u64;
        let mut list = Vec::with_capacity(t * kl);
        for _ in 0..t {
            let mut row: Vec<i32> = Vec::with_capacity(vis);
            let mut seen = vec![false; vis];
            while row.len() < kl {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let b = (x % vis as u64) as usize;
                if !std::mem::replace(&mut seen[b], true) {
                    row.push(b as i32);
                }
            }
            row.sort_unstable();
            list.extend(row);
        }
        let dlist = DeviceBuffer::from_slice(&list).unwrap();
        let cap = kl * block;
        let lst = time(&mut || unsafe {
            assert_eq!(ds_index_cand_scores(st.raw(), p(&q), p(&k), w.raw() as *const c_void, short.raw() as *const c_void,
                                            dlist.raw() as *const c_void, kl as i32, n as i32, p(&scores), cap as i64, t as i32,
                                            std::ptr::null(), 0, 64), 0);
            assert_eq!(ds_topk_select(st.raw(), p(&scores), cap as i64, cap as i32, topk as i32, out.raw() as *mut c_void, topk as i32, t as i32), 0);
            assert_eq!(ds_cand_positions(st.raw(), dlist.raw() as *const c_void, kl as i32, out.raw() as *mut c_void, topk as i32, t as i32), 0);
        });
        println!("{n:>7}  {fma:>9.0} {mma:>9.0} {split:>9.0} {sel:>9.0} {sel_lim:>9.0} {lst:>9.0} {cb:>9.0} {cs:>9.0}");
    }
}
