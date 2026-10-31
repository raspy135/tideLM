//! GLM decode attention (glm_mla + glm_mla_combine) at the engine's shapes: T rows x 64 heads over
//! K = 2048 picks of an FP8 latent cache, flash-decoding splits as forward::decode_split picks
//! them, microseconds a (mla + combine) pair with L2 flushed by a 48 MB copy between pairs (the
//! copy's own time measured apart and subtracted).
//!
//!   cargo run --release -p tidelm-kernels --example bench_mla [--half 0|1] [--rows 1,4,8] [--splits 4,8,16]

use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::glm::{glm_mla, glm_mla_combine, glm_set_mla_half, glm_set_mla_probe};

const REPS: usize = 6;
const FLUSH: usize = 48 << 20;

fn arg(name: &str, d: &str) -> String {
    std::env::args().skip_while(|a| a != name).nth(1).unwrap_or_else(|| d.into())
}

fn list(name: &str, d: &str) -> Vec<usize> {
    arg(name, d).split(',').map(|v| v.trim().parse().unwrap()).collect()
}

/// forward::decode_split
fn decode_split(k: usize, heads: usize, t: usize) -> (usize, usize) {
    let hb = heads.div_ceil(32);
    let want = 128usize.div_ceil(hb * t).next_power_of_two().clamp(1, 16);
    let chunk = k.div_ceil(want).div_ceil(32) * 32;
    (k.div_ceil(chunk), chunk)
}

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let (cache, heads, k) = (65536usize, 64usize, 2048usize);
    let halves: Vec<i32> = list("--half", "0,1").iter().map(|&v| v as i32).collect();
    let rows = list("--rows", "1,4,8");
    let splits = list("--splits", "0"); // 0: decode_split's
    let probes: Vec<i32> = list("--probe", "0").iter().map(|&v| v as i32).collect();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let cq: Vec<u8> = (0..cache * 512).map(|_| (rnd() & 0x7f) as u8).collect();
    let cs: Vec<f32> = (0..cache * 4).map(|_| 0.01 + (rnd() % 100) as f32 * 1e-4).collect();
    let cr: Vec<u16> = (0..cache * 64).map(|_| 0x3c00 | (rnd() & 0x7f) as u16).collect();
    let (d_cq, d_cs, d_cr) = (DeviceBuffer::from_slice(&cq).unwrap(), DeviceBuffer::from_slice(&cs).unwrap(),
                              DeviceBuffer::from_slice(&cr).unwrap());
    let flush_a = DeviceBuffer::<u8>::zeroed(FLUSH).unwrap();
    let flush_b = DeviceBuffer::<u8>::zeroed(FLUSH).unwrap();
    println!("{:>4} {:>6} {:>5}  {:>9} {:>9} {:>9}", "rows", "splits", "half", "mla us", "comb us", "both us");
    for &t in &rows {
        let q: Vec<u16> = (0..t * heads * 576).map(|_| 0x3c00 | (rnd() & 0xff) as u16).collect();
        let d_q = DeviceBuffer::from_slice(&q).unwrap();
        let mut idx = vec![0i32; t * k];
        for r in 0..t {
            let mut v: Vec<i32> = (0..k).map(|_| (rnd() % cache as u64) as i32).collect();
            v.sort();
            idx[r * k..(r + 1) * k].copy_from_slice(&v);
        }
        let d_idx = DeviceBuffer::from_slice(&idx).unwrap();
        let d_b = DeviceBuffer::from_slice(&vec![cache as i32 - 1; t]).unwrap();
        let pm = DeviceBuffer::<f32>::zeroed(t * heads * 64).unwrap();
        let pl = DeviceBuffer::<f32>::zeroed(t * heads * 64).unwrap();
        let pacc = DeviceBuffer::<f32>::zeroed(t * heads * 64 * 512).unwrap();
        let out = DeviceBuffer::<u16>::zeroed(t * heads * 512).unwrap();
        for &sp in &splits {
            let (ns, chunk) = if sp == 0 { decode_split(k, heads, t) } else { (sp, k.div_ceil(sp).div_ceil(32) * 32) };
            for &half in &halves {
            for &probe in &probes {
                unsafe { glm_set_mla_half(half); glm_set_mla_probe(probe); }
                let mla = |s: &Stream| unsafe {
                    glm_mla(s.raw(), 2, d_q.raw(), 0, (heads * 576) as i64, 576, heads as i32, d_cq.raw(), d_cs.raw(), d_cr.raw(), 0,
                            d_idx.raw(), k as i64, k as i32, 1, d_b.raw(), t as i32, heads as i32, 0.0417, ns as i32, chunk as i32,
                            pm.raw(), pl.raw(), pacc.raw(), std::ptr::null_mut(), 0, 0, std::ptr::null())
                };
                let comb = |s: &Stream| unsafe {
                    glm_mla_combine(s.raw(), pm.raw(), pl.raw(), pacc.raw(), t as i32, heads as i32, ns as i32, out.raw(),
                                    (heads * 512) as i64, 512, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0, 0, 0)
                };
                let flush = |s: &Stream| unsafe {
                    tidelm_cuda::memcpy_async(flush_b.raw(), flush_a.raw(), FLUSH, tidelm_cuda::ffi::MEMCPY_DEVICE_TO_DEVICE, s)
                };
                let time = |parts: &[u8]| -> f64 {
                    let g = st.capture(|s| {
                        for _ in 0..REPS {
                            flush(s)?;
                            if parts.contains(&1) {
                                check(mla(s), "mla")?;
                            }
                            if parts.contains(&2) {
                                check(comb(s), "combine")?;
                            }
                        }
                        Ok(())
                    }).unwrap();
                    g.launch(&st).unwrap();
                    st.synchronize().unwrap();
                    let mut best = f64::INFINITY;
                    for _ in 0..4 {
                        let t0 = Instant::now();
                        g.launch(&st).unwrap();
                        st.synchronize().unwrap();
                        best = best.min(t0.elapsed().as_secs_f64() / REPS as f64 * 1e6);
                    }
                    best
                };
                let f = time(&[]);
                let a = time(&[1]) - f;
                let c = time(&[2]) - f;
                let b = time(&[1, 2]) - f;
                println!("{t:4} {ns:6} {half:5} probe {probe:2}  {a:9.1} {c:9.1} {b:9.1}");
            }
            }
        }
    }
}
