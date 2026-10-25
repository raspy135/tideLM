//! Prefill-sized dense GEMMs: the tensor-core FP8 GEMM at M = 512 and 2048 rows, issued as
//! today's prefill does it (one launch a 64-row step) and as one M-tiled launch, against the
//! decode step's 64. TFLOP/s and weight GB/s at the TP shard shapes of a backbone layer.
//!
//!   cargo run --release -p tidelm-kernels --example bench_prefill_gemm
use std::time::Instant;
use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::ds_mma_fp8;

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    // this rank's shards under tensor parallel (output layout)
    let shapes = [("wq_a", 1024usize, 5120usize), ("wq_b", 16384, 1024), ("wkv", 512, 5120), ("wo_b", 2560, 8192),
                  ("sh_w1", 1152, 5120), ("sh_w2", 2560, 2304)];
    let mmax = 2048usize;
    println!("{:8} {:>5}  {:>22}  {:>22}", "", "M", "64-row launches", "one M-tiled launch");
    for (name, n, k) in shapes {
        let w = DeviceBuffer::<u8>::zeroed(n * k).unwrap();
        let s = DeviceBuffer::<u8>::zeroed(n.div_ceil(32) * k / 32).unwrap();
        let x = DeviceBuffer::from_slice(&vec![0x3c00u16; mmax * k]).unwrap();
        let y = DeviceBuffer::<u16>::zeroed(mmax * n).unwrap();
        for m in [64usize, 512, 2048] {
            let run = |tiled: bool| -> f64 {
                let go = || unsafe {
                    if tiled {
                        assert_eq!(ds_mma_fp8(st.raw(), x.raw(), k as i64, 0, 0, w.raw(), s.raw(), (k / 32) as i64, m as i32, n as i32,
                                              k as i32, 1, y.raw(), std::ptr::null_mut(), n as i64, std::ptr::null_mut()), 0);
                    } else {
                        for m0 in (0..m).step_by(64) {
                            assert_eq!(ds_mma_fp8(st.raw(), x.ptr_at(m0 * k), k as i64, 0, 0, w.raw(), s.raw(), (k / 32) as i64, 64,
                                                  n as i32, k as i32, 1, y.ptr_at(m0 * n), std::ptr::null_mut(), n as i64,
                                                  std::ptr::null_mut()), 0);
                        }
                    }
                };
                go();
                st.synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..5 {
                    go();
                }
                st.synchronize().unwrap();
                t.elapsed().as_secs_f64() / 5.0
            };
            let (a, b) = (run(false), run(true));
            let fl = 2.0 * (m * n * k) as f64;
            println!("{name:8} {m:5}  {:8.3} ms {:6.1} TF/s  {:8.3} ms {:6.1} TF/s", a * 1e3, fl / a / 1e12, b * 1e3, fl / b / 1e12);
        }
    }
}
