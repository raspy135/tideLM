//! Sweep the tensor-core GEMM's (rows per warp, K blocks in flight) at M = 32 and 64.
//!   cargo run --release -p tidelm-kernels --example tune_mma
use std::time::Instant;
use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_mma_fp8_cfg, ds_splits_reduce};

fn time(stream: &Stream, mut f: impl FnMut(&Stream, usize) -> i32) -> f64 {
    for i in 0..3 { assert_eq!(f(stream, i), 0); }
    stream.synchronize().unwrap();
    let g = stream.capture(|s| { for i in 0..48 { check(f(s, i), "launch")?; } Ok(()) }).unwrap();
    g.launch(stream).unwrap();
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..3 { g.launch(stream).unwrap(); }
    stream.synchronize().unwrap();
    t.elapsed().as_secs_f64() / 144.0
}

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let shapes = [("wq_a", 1280usize, 5120usize), ("wq_b", 32768, 1280), ("wo_b", 5120, 8192), ("sh_w1", 2304, 5120),
                  ("sh_w2", 5120, 2304), ("e.wkv", 25600, 6144)];
    let cfgs: [(i32, i32); 5] = [(1, 4), (2, 2), (2, 4), (4, 1), (4, 2)];
    for m in [32usize, 64] {
        println!("M={m}: GB/s per (rows-per-warp groups, depth)");
        println!("{:8}{}", "", cfgs.iter().map(|(g, d)| format!("{:>9}", format!("{g}x{d}"))).collect::<String>());
        for (name, n, k) in shapes {
            let wb = n * k;
            let copies = ((64usize << 20) / wb).max(1) + 1;
            let ws: Vec<_> = (0..copies).map(|_| (DeviceBuffer::<u8>::zeroed(wb).unwrap(),
                DeviceBuffer::<u8>::zeroed(n.div_ceil(32) * k / 32).unwrap())).collect();
            let x = DeviceBuffer::<u16>::zeroed(64 * k).unwrap();
            let y = DeviceBuffer::<u16>::zeroed(64 * n).unwrap();
            let mut row = format!("{name:8}");
            for (g, d) in cfgs {
                if m == 32 && (g, d) == (4, 2) { row += &format!("{:>9}", "-"); continue; }
                let blocks = n.div_ceil(64 * g as usize);
                let splits = 96usize.div_ceil(blocks).min(k / 512).max(1);
                let part = DeviceBuffer::<f32>::zeroed(splits * m * n).unwrap();
                let dt = time(&st, |s, i| unsafe {
                    let (w, sc) = &ws[i % copies];
                    let rc = ds_mma_fp8_cfg(s.raw(), x.raw(), k as i64, w.raw(), sc.raw(), (k / 32) as i64, m as i32, n as i32,
                                            k as i32, splits as i32, y.raw(), part.raw(), g, d);
                    if rc != 0 || splits == 1 { return rc; }
                    ds_splits_reduce(s.raw(), part.raw(), splits as i32, m as i32, n as i32, y.raw(), std::ptr::null_mut(), n as i64)
                });
                row += &format!("{:>9.0}", wb as f64 / dt / 1e9);
            }
            println!("{row}");
        }
    }
}
