//! Decode attention (ds_attn_decode) at fixed shapes: rows x heads, a full 128-key window and
//! C compressed keys, microseconds a launch from a graph of REPS launches.
//!
//!   cargo run --release -p tidelm-kernels --example bench_attn

use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_attn_decode, ds_attn_prefill, ds_attn_work_bytes};

const REPS: usize = 8;

fn main() {
    set_device(0).unwrap();
    let stream = Stream::new().unwrap();
    let (d, ring, w) = (512usize, 256usize, 128usize);
    let ringb = DeviceBuffer::from_slice(&vec![0x3c00u16; ring * d]).unwrap();
    let ckv = DeviceBuffer::from_slice(&vec![0x3c00u16; 4096 * d]).unwrap();
    let sink = DeviceBuffer::from_slice(&vec![0f32; 64]).unwrap();
    println!("{:>4} {:>5} {:>6}  {:>8}", "rows", "heads", "ckeys", "us");
    for &(m, h, c) in &[(1usize, 32usize, 16usize), (1, 32, 256), (1, 32, 1024), (1, 64, 256), (6, 32, 256), (6, 32, 1024),
                        (6, 64, 256), (24, 32, 256), (64, 32, 1024), (64, 64, 1024), (512, 32, 1024), (2048, 32, 1024), (2048, 32, 256)] {
        let q = DeviceBuffer::from_slice(&vec![0x3c00u16; m * h * d]).unwrap();
        let out = DeviceBuffer::<u16>::zeroed(m * h * d).unwrap();
        let pos = DeviceBuffer::from_slice(&vec![1000i32; m]).unwrap();
        let cidx: Vec<i32> = (0..m * c).map(|i| (i % c) as i32).collect();
        let cidx = DeviceBuffer::from_slice(&cidx).unwrap();
        let wb = unsafe { ds_attn_work_bytes(m as i32, h as i32, (w + c) as i32) };
        let work = DeviceBuffer::<u8>::zeroed(wb as usize).unwrap();
        let split = std::env::args().any(|a| a == "--per-head");
        // --prefill 16|32: the one-pass prompt-row kernel at that many heads a block
        let prefill: Option<i32> = std::env::args().skip_while(|a| a != "--prefill").nth(1).map(|v| v.parse().unwrap());
        let launch = |s: &Stream| unsafe {
            if let Some(hb) = prefill {
                return ds_attn_prefill(s.raw(), q.raw(), ringb.raw(), ring as i32, pos.raw(), w as i32, 0, ckv.raw(), cidx.raw(),
                                       c as i32, sink.raw(), 0.04, out.raw(), m as i32, h as i32, std::ptr::null_mut(), 0, 0,
                                       std::ptr::null_mut(), hb);
            }
            ds_attn_decode(s.raw(), q.raw(), ringb.raw(), ring as i32, pos.raw(), w as i32, 0, ckv.raw(), cidx.raw(), c as i32,
                           sink.raw(), 0.04, out.raw(), m as i32, h as i32, std::ptr::null_mut(), 0, 0,
                           if split { std::ptr::null_mut() } else { work.raw() }, if split { 0 } else { wb }, std::ptr::null_mut())
        };
        assert_eq!(launch(&stream), 0);
        stream.synchronize().unwrap();
        let g = stream
            .capture(|s| {
                for _ in 0..REPS {
                    check(launch(s), "attn")?;
                }
                Ok(())
            })
            .unwrap();
        g.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..5 {
            g.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        println!("{m:4} {h:5} {c:6}  {:8.1}", t.elapsed().as_secs_f64() / (5 * REPS) as f64 * 1e6);
    }
}
