//! The M-tiled tensor-core FP8 GEMM's tile configurations at prefill sizes (ds_mma_fp8_big),
//! TFLOP/s, on this rank's shard shapes.
//!   cargo run --release -p tidelm-kernels --example tune_prefill_gemm
use std::time::Instant;
use tidelm_cuda::{set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::ds_mma_fp8_big;

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let shapes = [("wq_a", 1024usize, 5120usize), ("wq_b", 16384, 1024), ("wkv", 512, 5120), ("wo_b", 2560, 8192),
                  ("sh_w1", 1152, 5120), ("sh_w2", 2560, 2304)];
    let cfgs = ["64/256/1/4", "128/128/1/2", "128/128/2/1", "256/64/1/1", "128/128/2/2", "256/64/2/1"];
    for m in [512usize, 2048] {
        println!("M={m}: TFLOP/s per (MT/KC/NG/DEPTH)");
        println!("{:8}{}", "", cfgs.iter().map(|c| format!("{c:>13}")).collect::<String>());
        for (name, n, k) in shapes {
            let w = DeviceBuffer::<u8>::zeroed(n * k).unwrap();
            let s = DeviceBuffer::<u8>::zeroed(n.div_ceil(32) * k / 32).unwrap();
            let x = DeviceBuffer::from_slice(&vec![0x3c00u16; m * k]).unwrap();
            let y = DeviceBuffer::<u16>::zeroed(m * n).unwrap();
            let mut row = format!("{name:8}");
            for c in 0..cfgs.len() {
                let go = || unsafe {
                    ds_mma_fp8_big(st.raw(), x.raw(), k as i64, w.raw(), s.raw(), (k / 32) as i64, m as i32, n as i32, k as i32,
                                   y.raw(), c as i32)
                };
                if go() != 0 {
                    row += &format!("{:>13}", "err");
                    continue;
                }
                st.synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..10 {
                    go();
                }
                st.synchronize().unwrap();
                let dt = t.elapsed().as_secs_f64() / 10.0;
                row += &format!("{:>13.1}", 2.0 * (m * n * k) as f64 / dt / 1e12);
            }
            println!("{row}");
        }
    }
}
