//! The indexer's scores at a prefill step's shape: index_scores_kernel (FMA) against the
//! tensor-core kernel, 2,048 query rows over the compressed keys of an 8K prompt's last step.
//!
//!   cargo run --release -p tidelm-kernels --example bench_index

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_index_scores, ds_index_scores_mma};

fn main() {
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();
    let (h, dd) = (32usize, 128usize);
    let p = |b: &DeviceBuffer<u16>| b.raw() as *mut c_void;
    println!("{:>5} {:>5}  {:>9} {:>9}", "rows", "keys", "fma ms", "mma ms");
    for &(t, n, start) in &[(2048usize, 2240usize, 6911usize), (2048, 1536, 4096), (512, 2240, 8447)] {
        let q = DeviceBuffer::from_slice(&(0..t * h * dd).map(|i| 0x3c00u16 + (i % 97) as u16).collect::<Vec<_>>()).unwrap();
        let k = DeviceBuffer::from_slice(&(0..n * dd).map(|i| 0x3c00u16 + (i % 89) as u16).collect::<Vec<_>>()).unwrap();
        let w = DeviceBuffer::from_slice(&vec![0.5f32; t * h]).unwrap();
        // causal over a step whose rows sit at positions start.., 4 tokens a compressed key
        let clen: Vec<i32> = (0..t).map(|r| (((start + r + 1) / 4) as i32).min(n as i32)).collect();
        let dc = DeviceBuffer::from_slice(&clen).unwrap();
        let out = DeviceBuffer::<u16>::zeroed(t * n).unwrap();
        let time = |mma: bool| -> f64 {
            let run = || unsafe {
                let f = if mma { ds_index_scores_mma } else { ds_index_scores };
                f(stream.raw(), p(&q), p(&k), w.raw() as *const c_void, dc.raw() as *const c_void, std::ptr::null(), 0,
                  n as i32, p(&out), n as i64, t as i32, std::ptr::null(), 0)
            };
            assert_eq!(run(), 0);
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..10 {
                run();
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / 10.0
        };
        println!("{t:5} {n:5}  {:9.2} {:9.2}", time(false) * 1e3, time(true) * 1e3);
    }
}
