//! Bandwidth of the decode GEMMs on the real DS v4.1 shapes.
//!
//!   cargo run --release -p tidelm-kernels --example bench_skinny
//!
//! Decode is weight streaming, so the number that matters is weight bytes per second.
//! Each measurement captures REPS launches in one CUDA graph (no launch overhead in the
//! number) and rotates through enough weight copies to exceed L2, so every read is cold —
//! the regime tools/fp8_linear.py's 211-230 GB/s TP2 decode numbers were taken in.

use std::os::raw::c_void;
use std::time::Instant;

use tidelm_cuda::{check, ffi, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_linear_bf16, ds_linear_fp8, ds_mma_fp8, ds_mma_rows_per_block, ds_splits_reduce};

const REPS: usize = 64;
const L2_BYTES: usize = 64 << 20; // comfortably above GB10's L2

fn time_graph(stream: &Stream, mut launch: impl FnMut(&Stream, usize) -> i32) -> f64 {
    for i in 0..4 {
        assert_eq!(launch(stream, i), 0);
    }
    stream.synchronize().unwrap();
    let g = stream
        .capture(|s| {
            for i in 0..REPS {
                let rc = launch(s, i);
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

    // copy ceiling: D2D moves 2 bytes of traffic per byte copied
    let n = 512usize << 20;
    let (a, b) = (DeviceBuffer::<u8>::zeroed(n).unwrap(), DeviceBuffer::<u8>::zeroed(n).unwrap());
    let dt = time_graph(&stream, |s, _| unsafe {
        ffi::cudaMemcpyAsync(b.raw(), a.raw(), n, ffi::MEMCPY_DEVICE_TO_DEVICE, s.raw())
    });
    println!("D2D copy                      {:7.1} GB/s (read+write)", 2.0 * n as f64 / dt / 1e9);
    drop((a, b));

    let shapes: [(&str, usize, usize); 7] = [
        ("wq_a   [1280, 5120]", 1280, 5120),
        ("wq_b   [32768, 1280]", 32768, 1280),
        ("wkv    [512, 5120]", 512, 5120),
        ("wo_b   [5120, 8192]", 5120, 8192),
        ("sh_w1  [2304, 5120]", 2304, 5120),
        ("sh_w2  [5120, 2304]", 5120, 2304),
        ("e.wkv  [25600, 6144]", 25600, 6144),
    ];
    let ms = [1usize, 6, 16, 32, 64];
    println!("GB/s of weight bytes; FMA kernel (M <= 16) | tensor-core kernel");
    println!("{:22} {}", "fp8 weight", ms.iter().map(|m| format!("{:>13}", format!("M={m}"))).collect::<String>());
    for (name, nn, k) in shapes {
        let wbytes = nn * k + nn.div_ceil(32) * k.div_ceil(32);
        let copies = (L2_BYTES / wbytes).max(1) + 1;
        let ws: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)> = (0..copies)
            .map(|_| {
                (
                    DeviceBuffer::<u8>::zeroed(nn * k).unwrap(),
                    DeviceBuffer::<u8>::zeroed(nn.div_ceil(32) * k.div_ceil(32)).unwrap(),
                )
            })
            .collect();
        let x = DeviceBuffer::<u16>::zeroed(64 * k).unwrap();
        let y = DeviceBuffer::<u16>::zeroed(64 * nn).unwrap();
        let mut row = format!("{name:22}");
        for m in ms {
            let fma = if m <= 16 {
                let dt = time_graph(&stream, |s, i| {
                    let (w, sc) = &ws[i % copies];
                    unsafe {
                        ds_linear_fp8(s.raw(), x.raw(), k as i64, w.raw(), sc.raw(), k.div_ceil(32) as i64, m as i32,
                                      nn as i32, k as i32, 1, y.raw(), std::ptr::null_mut::<c_void>(), nn as i64,
                                      std::ptr::null_mut())
                    }
                });
                format!("{:5.0}", wbytes as f64 / dt / 1e9)
            } else {
                "    -".to_string()
            };
            // split-K to at least two blocks per SM (48 on GB10), K slices of >= 512
            let blocks = nn.div_ceil(unsafe { ds_mma_rows_per_block(m as i32) } as usize);
            let splits = (96usize.div_ceil(blocks)).min(k / 512).max(1);
            let part = DeviceBuffer::<f32>::zeroed(splits * m * nn).unwrap();
            let dt = time_graph(&stream, |s, i| {
                let (w, sc) = &ws[i % copies];
                unsafe {
                    let rc = ds_mma_fp8(s.raw(), x.raw(), k as i64, 0, 0, w.raw(), sc.raw(), k.div_ceil(32) as i64, m as i32,
                                        nn as i32, k as i32, splits as i32, y.raw(), std::ptr::null_mut::<c_void>(), nn as i64,
                                        part.raw());
                    if rc != 0 || splits == 1 {
                        return rc;
                    }
                    ds_splits_reduce(s.raw(), part.raw(), splits as i32, m as i32, nn as i32, y.raw(),
                                     std::ptr::null_mut(), nn as i64)
                }
            });
            row += &format!("  {fma} |{:5.0}", wbytes as f64 / dt / 1e9);
        }
        println!("{row}");
    }

    // the head: bf16 [129280, 5120], read in full every decode step
    let (nn, k) = (129_280usize, 5120usize);
    let w = DeviceBuffer::<u16>::zeroed(nn * k).unwrap();
    let x = DeviceBuffer::<u16>::zeroed(16 * k).unwrap();
    let y = DeviceBuffer::<u16>::zeroed(16 * nn).unwrap();
    let mut row = format!("{:24}", "head bf16 [129280, 5120]");
    for m in [1usize, 6] {
        let dt = time_graph(&stream, |s, _| unsafe {
            ds_linear_bf16(s.raw(), x.raw(), k as i64, w.raw(), m as i32, nn as i32, k as i32, 1, y.raw(),
                           std::ptr::null_mut(), nn as i64, std::ptr::null_mut())
        });
        row += &format!(" {:10.1} {:10.1}", dt * 1e6, (nn * k * 2) as f64 / dt / 1e9);
    }
    println!("{row}");
}
