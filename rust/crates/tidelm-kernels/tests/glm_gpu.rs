//! GPU tests for cuda/glm.cu and the EXL3 kernels as the GLM engine drives them: each kernel
//! against a CPU transcription of the Python it replaces (engine/glm/*.py), and the EXL3 unpack
//! against golden values computed from the checkpoint with tools/exl3_format.py's numpy oracle.
//! Small allocations only; the checkpoint tests skip when it is absent.

use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{gpu_available, set_device, DeviceBuffer, Stream};
use tidelm_kernels::blas::{Blas, Mat, R_16F, R_32F};
use tidelm_kernels::exl3::*;
use tidelm_kernels::glm::*;

const GLM: &str = "/home/ryan/models/GLM-5.3-EXL3-3.0bpw";

fn bf16_from_f32(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn f32_from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
fn bf(f: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(f))
}
fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0 { s * m * 2f32.powi(-9) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}
/// f32 -> f16 bits, round to nearest (normal range and zero; test inputs only).
fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if e <= 0 {
        return sign;
    }
    let m = b & 0x7f_ffff;
    let r = (((e as u32) << 10) | (m >> 13)) + ((m >> 12) & 1);
    sign | r.min(0x7bff) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 { s * m * 2f32.powi(-24) } else if e == 31 { f32::NAN } else { s * (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unif().max(1e-7), self.unif());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16_from_f32(self.normal() * scale)).collect()
    }
}

fn dev<T: Copy>(v: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_slice(v).unwrap()
}
fn p<T: Copy>(b: &DeviceBuffer<T>) -> *mut c_void {
    b.raw()
}
fn null() -> *mut c_void {
    std::ptr::null_mut()
}
fn setup() -> Option<Stream> {
    if !gpu_available() {
        eprintln!("no CUDA device; skipping");
        return None;
    }
    set_device(0).unwrap();
    Some(Stream::new().unwrap())
}
/// A launch's return code, then the device drained: the test stream is non-blocking, so a
/// download (a legacy-stream copy) does not wait for it.
#[track_caller]
fn ok(rc: i32) {
    assert_eq!(rc, 0, "kernel launch failed with CUDA error {rc}");
    last_ok();
}
fn last_ok() {
    let rc = unsafe { tidelm_cuda::ffi::cudaDeviceSynchronize() };
    assert_eq!(rc, 0, "kernel failed with CUDA error {rc}");
}
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).powi(2)).sum();
    let den: f64 = want.iter().map(|w| w * w).sum();
    (num / den.max(1e-300)).sqrt()
}

// ------------------------------------------------------------------ EXL3
fn read_tensor(name: &str) -> Option<Vec<u8>> {
    if !Path::new(GLM).exists() {
        eprintln!("no GLM checkpoint; skipping");
        return None;
    }
    let ck = tidelm_model::Checkpoint::open(Path::new(GLM)).unwrap();
    Some(ck.read_tensor_vec(name).unwrap().0)
}

fn hadamard128(v: &mut [f64]) {
    let s = 1.0 / 128f64.sqrt();
    for blk in v.chunks_mut(128) {
        let mut h = 1;
        while h < 128 {
            for i in (0..128).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (blk[j], blk[j + h]);
                    blk[j] = a + b;
                    blk[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for x in blk.iter_mut() {
            *x *= s;
        }
    }
}

#[test]
fn exl3_unpack_matches_the_oracle_and_both_paths_match_the_reference() {
    let Some(st) = setup() else { return };
    let pre = "model.layers.3.mlp.experts.0.gate_proj";
    let Some(tr) = read_tensor(&format!("{pre}.trellis")) else { return };
    let suh: Vec<u16> = read_tensor(&format!("{pre}.suh")).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let svh: Vec<u16> = read_tensor(&format!("{pre}.svh")).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let (k, n, bits) = (6144usize, 2048usize, 3);
    assert_eq!(tr.len(), k * n * bits / 8);
    let u16s: Vec<u16> = tr[..12].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    assert_eq!(u16s, [25263, 63879, 49327, 47112, 473, 39093]);
    let d_tr = dev(&tr);
    let d_w = DeviceBuffer::<u16>::new(k * n).unwrap();
    ok(unsafe { glm_exl3_unpack(st.raw(), p(&d_tr), k as i32, n as i32, bits as i32, p(&d_w)) });
    st.synchronize().unwrap();
    let w: Vec<f32> = d_w.download().unwrap().iter().map(|&h| f16_to_f32(h)).collect();
    let want0 = [-0.61083984375, 0.33642578125, -0.61767578125, -0.0288848876953125, 0.7021484375, -0.59033203125,
                 0.09967041015625, -0.482177734375];
    assert_eq!(&w[..8], &want0.map(|v| v as f32));
    assert_eq!(&w[n..n + 4], &[-0.99658203125f32, 0.24853515625, 0.2890625, -2.681640625]);

    // y = H_N(H_K(x * suh) @ W_q) * svh: the oracle's forward of x_i = sin(0.37 i)
    let x: Vec<f64> = (0..k).map(|i| (0.37 * i as f64).sin()).collect();
    let mut xr: Vec<f64> = x.iter().zip(&suh).map(|(&a, &s)| a * f16_to_f32(s) as f64).collect();
    hadamard128(&mut xr);
    let mut z = vec![0f64; n];
    for (kk, &xv) in xr.iter().enumerate() {
        for (j, zj) in z.iter_mut().enumerate() {
            *zj += xv * w[kk * n + j] as f64;
        }
    }
    hadamard128(&mut z);
    let y: Vec<f64> = z.iter().zip(&svh).map(|(&a, &s)| a * f16_to_f32(s) as f64).collect();
    for (g, want) in y.iter().zip([0.007758, -0.221701, 0.020088, 0.038314]) {
        assert!((g - want).abs() < 2e-6, "oracle forward {g} vs {want}");
    }
    let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 25.196571).abs() < 1e-4, "|y| {norm}");

    // the kernels: the decode path (rotate, grouped split-K GEMV, combine) and the prefill path
    // (rotate, unpack, cuBLAS, combine) over 3 rows of bf16 inputs
    let t = 3usize;
    let mut rng = Rng(7);
    let xb = rng.bf16_vec(t * k, 1.0);
    let d_x = dev(&xb);
    let d_suh = dev(&suh);
    let d_svh = dev(&svh);
    let d_xh = DeviceBuffer::<u16>::new(t * k).unwrap();
    unsafe { exl3m_dense_rot(p(&d_x), k as i32, p(&d_suh), p(&d_xh), t as i32, k as i32, 1, 1, st.raw()) };
    last_ok();
    let sk = 8usize;
    let d_z = DeviceBuffer::<f32>::new(sk * t * n).unwrap();
    let d_ptr = dev(&[d_tr.raw() as i64]);
    let d_k2 = dev(&[2 * bits as i32]);
    let d_uids = dev(&[0i32]);
    let d_cnt = dev(&[1i32]);
    let mut members = vec![-1i32; 16];
    for (r, m) in members.iter_mut().enumerate().take(t) {
        *m = r as i32 * 32;
    }
    let d_mem = dev(&members);
    unsafe {
        exl3m_grouped(p(&d_xh), p(&d_xh), p(&d_ptr), p(&d_ptr), p(&d_k2), p(&d_k2), p(&d_uids), p(&d_cnt), p(&d_mem),
                      p(&d_z), k as i32, n as i32, t as i32, sk as i32, 16, 1, 1, 1, 8, 4, 1, 2, 10, CB_MUL1, st.raw())
    };
    let d_yd = DeviceBuffer::<f32>::new(t * n).unwrap();
    unsafe { exl3m_dense_out(p(&d_z), p(&d_svh), p(&d_yd), t as i32, n as i32, sk as i32, 1, 0, st.raw()) };
    let blas = Blas::new(st.raw()).unwrap();
    let d_zp = DeviceBuffer::<f32>::new(t * n).unwrap();
    blas.gemm_rm(t, n, k, Mat { ptr: d_xh.raw(), ty: R_16F, ld: k as i64 }, Mat { ptr: d_w.raw(), ty: R_16F, ld: n as i64 },
                 false, d_zp.raw(), R_32F, n as i64).unwrap();
    let d_yp = DeviceBuffer::<f32>::new(t * n).unwrap();
    unsafe { exl3m_dense_out(p(&d_zp), p(&d_svh), p(&d_yp), t as i32, n as i32, 1, 1, 0, st.raw()) };
    last_ok();
    let (yd, yp) = (d_yd.download().unwrap(), d_yp.download().unwrap());
    let mut want = vec![0f64; t * n];
    for r in 0..t {
        let mut xr: Vec<f64> = (0..k).map(|i| f32_from_bf16(xb[r * k + i]) as f64 * f16_to_f32(suh[i]) as f64).collect();
        hadamard128(&mut xr);
        let mut z = vec![0f64; n];
        for (kk, &xv) in xr.iter().enumerate() {
            for (j, zj) in z.iter_mut().enumerate() {
                *zj += xv * w[kk * n + j] as f64;
            }
        }
        hadamard128(&mut z);
        for j in 0..n {
            want[r * n + j] = z[j] * f16_to_f32(svh[j]) as f64;
        }
    }
    let (ed, ep) = (rel_err(&yd, &want), rel_err(&yp, &want));
    eprintln!("exl3 dense: decode rel {ed:.2e}, prefill rel {ep:.2e}");
    assert!(ed < 2e-3 && ep < 2e-3, "decode {ed}, prefill {ep}");
}

/// The prefill expert GEMM for a pack of mixed widths (exl3m_prefill_smem_cb) against the grouped
/// GEMV the MTP layer prefills with, over the same routed blocks: gate+up of six experts of widths
/// 2/2/2 .. 4/4/4 at the rank shapes, random trellis words, both codebooks; and, at one width and
/// mul1, against the shipped 2-3-bit tile (exl3m_prefill_smem_mode), which it must equal bit for bit.
#[test]
fn mixed_width_prefill_matches_the_grouped_gemv() {
    use tidelm_kernels::ds41::{ds_route_blocks, ds_route_large};
    let Some(st) = setup() else { return };
    let (k, n, topk, rows) = (6144usize, 1024usize, 2usize, 160usize);
    let p_ = rows * topk;
    let run = |widths: &[[usize; 2]], cb: i32, seed: u64| -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
        let e = widths.len();
        let mut rng = Rng(seed);
        let tr: Vec<[DeviceBuffer<u16>; 2]> = widths.iter().map(|w| {
            [0, 1].map(|m| dev(&(0..(k / 16) * (n / 16) * 16 * w[m]).map(|_| rng.next() as u16).collect::<Vec<_>>()))
        }).collect();
        let ptr = |m: usize| dev(&tr.iter().map(|t| t[m].raw() as i64).collect::<Vec<_>>());
        let (t1p, t3p) = (ptr(0), ptr(1));
        let k2 = |m: usize| dev(&widths.iter().map(|w| 2 * w[m] as i32).collect::<Vec<_>>());
        let (k2g, k2u) = (k2(0), k2(1));
        // each row's two picks: experts r % e and (r + 1 + r / e) % e, never the same
        let picks: Vec<i32> = (0..rows).flat_map(|r| [(r % e) as i32, ((r + 1 + r / e % (e - 1)) % e) as i32]).collect();
        let d_pick = dev(&picks);
        let nb = unsafe { ds_route_blocks(p_ as i32, e as i32, 64) } as usize;
        let work = DeviceBuffer::<i32>::new(3 * e + 64).unwrap();
        let (bs, bp, br) = (DeviceBuffer::<i32>::new(nb).unwrap(), DeviceBuffer::<i32>::new(nb * 64).unwrap(),
                            DeviceBuffer::<i32>::new(nb * 64).unwrap());
        let mem = DeviceBuffer::<i32>::new(nb * 64).unwrap();
        let cnt = DeviceBuffer::<i32>::new(1).unwrap();
        ok(unsafe { ds_route_large(st.raw(), p(&d_pick), p_ as i32, topk as i32, e as i32, p(&work), p(&bs), p(&bp), p(&br), 64) });
        ok(unsafe { glm_member_codes(st.raw(), p(&bp), (nb * 64) as i32, topk as i32, p(&mem)) });
        ok(unsafe { glm_count_live(st.raw(), p(&bs), nb as i32, p(&cnt)) });
        let x0 = dev(&(0..p_ * k).map(|_| f32_to_f16(rng.normal() * 0.5)).collect::<Vec<_>>());
        let x1 = dev(&(0..p_ * k).map(|_| f32_to_f16(rng.normal() * 0.5)).collect::<Vec<_>>());
        let z = || DeviceBuffer::<f32>::zeroed(2 * p_ * n).unwrap();
        let (zg, zs) = (z(), z());
        unsafe {
            exl3m_grouped(p(&x0), p(&x1), p(&t1p), p(&t3p), p(&k2g), p(&k2u), p(&bs), p(&cnt), p(&mem), p(&zg), k as i32,
                          n as i32, p_ as i32, 1, 64, topk as i32, nb as i32, 2, 8, 4, 2, 2, 10, cb, st.raw());
            exl3m_prefill_smem_cb(p(&x0), p(&x1), p(&t1p), p(&t3p), p(&k2g), p(&k2u), p(&bs), p(&cnt), p(&mem), p(&zs),
                                  k as i32, n as i32, p_ as i32, topk as i32, nb as i32, 2, cb, 8, st.raw());
        }
        let old = (cb == CB_MUL1 && widths.iter().all(|w| w[0] <= 3 && w[1] <= 3)).then(|| {
            let zo = z();
            unsafe {
                exl3m_prefill_smem_mode(p(&x0), p(&x1), p(&t1p), p(&t3p), p(&k2g), p(&k2u), p(&bs), p(&cnt), p(&mem),
                                        p(&zo), k as i32, n as i32, p_ as i32, topk as i32, nb as i32, 2, 4, 1, 8, -4, st.raw());
            }
            last_ok();
            zo.download().unwrap()
        });
        last_ok();
        (zg.download().unwrap(), zs.download().unwrap(), old)
    };
    let mixed = [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [3, 3]];
    for cb in [CB_MCG, CB_MUL1] {
        let (g, s, _) = run(&mixed, cb, 11 + cb as u64);
        let want: Vec<f64> = g.iter().map(|&v| v as f64).collect();
        let e = rel_err(&s, &want);
        eprintln!("mixed widths, codebook {cb}: smem vs grouped rel {e:.2e}");
        assert!(e < 1e-5, "mixed widths, codebook {cb}: smem vs grouped rel {e}");
    }
    let (g, s, old) = run(&[[3, 3]; 6], CB_MUL1, 5);
    assert_eq!(s, old.unwrap(), "3-bit mul1: the mixed-width tile must equal the shipped one");
    let want: Vec<f64> = g.iter().map(|&v| v as f64).collect();
    assert!(rel_err(&s, &want) < 1e-5);
}

// ------------------------------------------------------------------ elementwise
#[test]
fn rmsnorm_rope_kv_fp8_and_index_key() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(11);
    let (t, n) = (3usize, 6144usize);
    let h = rng.bf16_vec(t * n, 1.0);
    let a = rng.bf16_vec(t * n, 0.5);
    let w = rng.bf16_vec(n, 1.0);
    let d_h = dev(&h);
    let d_a = dev(&a);
    let d_w = dev(&w);
    let d_y = DeviceBuffer::<u16>::new(t * n).unwrap();
    ok(unsafe { glm_rmsnorm(st.raw(), p(&d_h), n as i64, p(&d_a), n as i64, p(&d_w), p(&d_y), n as i64, t as i32, n as i32, 1e-5) });
    let (h2, y) = (d_h.download().unwrap(), d_y.download().unwrap());
    for r in 0..t {
        let hs: Vec<f32> = (0..n).map(|c| bf(f32_from_bf16(h[r * n + c]) + f32_from_bf16(a[r * n + c]))).collect();
        let ss: f64 = hs.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rr = (1.0 / ((ss / n as f64) + 1e-5).sqrt()) as f32;
        for c in 0..n {
            assert_eq!(f32_from_bf16(h2[r * n + c]), hs[c]);
            let want = bf(f32_from_bf16(w[c]) * bf(hs[c] * rr));
            let got = f32_from_bf16(y[r * n + c]);
            assert!((got - want).abs() <= want.abs() * 0.008 + 1e-6, "rms [{r},{c}] {got} vs {want}");
        }
    }

    // RoPE: interleaved in, halves out, at positions 5, 9
    let (hh, d) = (3usize, 64usize);
    let x = rng.bf16_vec(2 * hh * d, 1.0);
    let rows = 16usize;
    let mut cs = vec![0u16; rows * 32];
    let mut sn = vec![0u16; rows * 32];
    for pp in 0..rows {
        for i in 0..32 {
            let f = pp as f64 / 8e6f64.powf(2.0 * i as f64 / 64.0);
            cs[pp * 32 + i] = bf16_from_f32(f.cos() as f32);
            sn[pp * 32 + i] = bf16_from_f32(f.sin() as f32);
        }
    }
    let pos = [5i32, 9];
    let d_x = dev(&x);
    let d_cs = dev(&cs);
    let d_sn = dev(&sn);
    let d_pos = dev(&pos);
    ok(unsafe { glm_rope(st.raw(), p(&d_x), (hh * d) as i64, d as i64, p(&d_x), (hh * d) as i64, d as i64, 2, hh as i32, p(&d_cs), p(&d_sn), p(&d_pos)) });
    let got = d_x.download().unwrap();
    for r in 0..2 {
        for head in 0..hh {
            let b = (r * hh + head) * d;
            for i in 0..32 {
                let (x1, x2) = (f32_from_bf16(x[b + 2 * i]), f32_from_bf16(x[b + 2 * i + 1]));
                let (c, s) = (f32_from_bf16(cs[pos[r] as usize * 32 + i]), f32_from_bf16(sn[pos[r] as usize * 32 + i]));
                assert_eq!(f32_from_bf16(got[b + i]), bf(bf(x1 * c) - bf(x2 * s)));
                assert_eq!(f32_from_bf16(got[b + i + 32]), bf(bf(x2 * c) + bf(x1 * s)));
            }
        }
    }

    // FP8 cache rows: s = max(amax / 448, 1e-12) per 128, q = fp8(x / s), at rows 2 and 0
    let v = rng.bf16_vec(2 * 512, 2.0);
    let d_v = dev(&v);
    let d_rows = dev(&[2i32, 0]);
    let d_q = DeviceBuffer::<u8>::zeroed(3 * 512).unwrap();
    let d_s = DeviceBuffer::<f32>::zeroed(3 * 4).unwrap();
    ok(unsafe { glm_kv_fp8(st.raw(), p(&d_v), 512, p(&d_rows), p(&d_q), p(&d_s), 2, 512) });
    let (q, s) = (d_q.download().unwrap(), d_s.download().unwrap());
    for (r, row) in [2usize, 0].iter().enumerate() {
        for b in 0..4 {
            let vals: Vec<f32> = (0..128).map(|i| f32_from_bf16(v[r * 512 + b * 128 + i])).collect();
            let am = vals.iter().fold(0f32, |m, x| m.max(x.abs()));
            let sc = (am / 448.0).max(1e-12);
            assert_eq!(s[row * 4 + b], sc);
            for i in 0..128 {
                let back = e4m3(q[row * 512 + b * 128 + i]) * sc;
                assert!((back - vals[i]).abs() <= vals[i].abs() * 0.07 + sc * 0.01, "fp8 {back} vs {}", vals[i]);
            }
        }
    }

    // indexer key: layer_norm, RoPE on the first 64, an FP8 row with one scale
    let kr = rng.bf16_vec(2 * 128, 3.0);
    let lw: Vec<f32> = (0..128).map(|_| 1.0 + 0.1 * rng.normal()).collect();
    let lb: Vec<f32> = (0..128).map(|_| 0.1 * rng.normal()).collect();
    let d_k = dev(&kr);
    let d_lw = dev(&lw);
    let d_lb = dev(&lb);
    let d_iq = DeviceBuffer::<u8>::zeroed(3 * 128).unwrap();
    let d_is = DeviceBuffer::<f32>::zeroed(3).unwrap();
    ok(unsafe { glm_index_key(st.raw(), p(&d_k), 128, p(&d_lw), p(&d_lb), p(&d_cs), p(&d_sn), p(&d_pos), p(&d_rows), p(&d_iq), p(&d_is), 2) });
    let (iq, is) = (d_iq.download().unwrap(), d_is.download().unwrap());
    for (r, &row) in [2usize, 0].iter().enumerate() {
        let xs: Vec<f32> = (0..128).map(|i| f32_from_bf16(kr[r * 128 + i])).collect();
        let mean = xs.iter().sum::<f32>() / 128.0;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 128.0;
        let y: Vec<f32> = (0..128).map(|i| bf((xs[i] - mean) / (var + 1e-6).sqrt() * lw[i] + lb[i])).collect();
        let mut z = y.clone();
        for i in 0..32 {
            let (c, s) = (f32_from_bf16(cs[pos[r] as usize * 32 + i]), f32_from_bf16(sn[pos[r] as usize * 32 + i]));
            z[i] = bf(bf(y[2 * i] * c) - bf(y[2 * i + 1] * s));
            z[i + 32] = bf(bf(y[2 * i + 1] * c) + bf(y[2 * i] * s));
        }
        let am = z.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((is[row] - am / 448.0).abs() <= am / 448.0 * 0.01, "index key scale");
        for i in 0..128 {
            let back = e4m3(iq[row * 128 + i]) * is[row];
            assert!((back - z[i]).abs() <= z[i].abs() * 0.07 + is[row] * 0.02, "index key [{r},{i}] {back} vs {}", z[i]);
        }
    }
}

// ------------------------------------------------------------------ indexer
fn fp8_row(rng: &mut Rng, d: usize) -> (Vec<u8>, f32) {
    let s = 0.01 + 0.02 * rng.unif();
    let q = (0..d).map(|_| loop {
        let b = (rng.next() >> 32) as u8;
        if b & 0x7f != 0x7f && (b >> 3) & 0xf < 12 {
            break b;
        }
    }).collect();
    (q, s)
}

#[test]
fn index_scores_and_topk_match_the_reference() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(3);
    let (t, rows, k) = (3usize, 3000usize, 2048usize);
    let (stride, off) = (2i32, 1i32);
    let q = rng.bf16_vec(t * 32 * 128, 0.5);
    let w: Vec<f32> = (0..t * 32).map(|_| rng.normal()).collect();
    let mut kq = vec![0u8; rows * 128];
    let mut ks = vec![0f32; rows];
    for r in 0..rows {
        let (row, s) = fp8_row(&mut rng, 128);
        kq[r * 128..(r + 1) * 128].copy_from_slice(&row);
        ks[r] = s;
    }
    // positions: row 0 sees 5 local rows, row 1 sees 2100 (past top-k), row 2 all 3000
    let pos = [10i32, 4200, 7000];
    let lim = |p: i32| ((p - off) / stride + 1).min(rows as i32) as usize;
    let (d_q, d_w, d_kq, d_ks, d_pos) = (dev(&q), dev(&w), dev(&kq), dev(&ks), dev(&pos));
    let ld = rows;
    let d_out = DeviceBuffer::<f32>::zeroed(t * ld).unwrap();
    let wscale = 32f32.powf(-0.5);
    let inv_d = 128f32.powf(-0.5);
    ok(unsafe { glm_index_scores(st.raw(), p(&d_q), p(&d_w), wscale, p(&d_kq), p(&d_ks), p(&d_pos), t as i32, rows as i32, stride, off, 0, inv_d, p(&d_out), ld as i64, null(), 0) });
    let got = d_out.download().unwrap();
    let mut want = vec![vec![]; t];
    for r in 0..t {
        for j in 0..lim(pos[r]) {
            let mut sc = 0f64;
            for h in 0..32 {
                let mut dot = 0f64;
                for dd in 0..128 {
                    dot += f32_from_bf16(q[(r * 32 + h) * 128 + dd]) as f64 * e4m3(kq[j * 128 + dd]) as f64;
                }
                sc += (w[r * 32 + h] * wscale) as f64 * (dot * ks[j] as f64 * inv_d as f64).max(0.0);
            }
            want[r].push(sc);
            let g = got[r * ld + j] as f64;
            assert!((g - sc).abs() <= 1e-4 * (1.0 + sc.abs()), "score [{r},{j}] {g} vs {sc}");
        }
    }
    // top-k: the largest, ascending, -1 padded; values alongside
    let d_idx = DeviceBuffer::<i32>::zeroed(t * k).unwrap();
    let d_val = DeviceBuffer::<f32>::zeroed(t * k).unwrap();
    ok(unsafe { glm_topk(st.raw(), p(&d_out), ld as i64, p(&d_pos), stride, off, rows as i32, k as i32, t as i32, p(&d_idx), k as i64, p(&d_val), k as i64) });
    let (idx, val) = (d_idx.download().unwrap(), d_val.download().unwrap());
    for r in 0..t {
        let n = lim(pos[r]);
        let row = &idx[r * k..(r + 1) * k];
        let take = n.min(k);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| got[r * ld + b].partial_cmp(&got[r * ld + a]).unwrap().then(a.cmp(&b)));
        let mut sel: Vec<i32> = order[..take].iter().map(|&j| j as i32).collect();
        sel.sort();
        assert_eq!(&row[..take], &sel[..], "topk row {r}");
        assert!(row[take..].iter().all(|&j| j == -1));
        for (i, &j) in row[..take].iter().enumerate() {
            assert_eq!(val[r * k + i], got[r * ld + j as usize]);
        }
    }

    // split share: two ranks' top-K lists (values), the global top-K's members of each
    let kk = 8usize;
    let t2 = 2usize;
    let mut vals = vec![0f32; 2 * t2 * kk];
    for v in vals.iter_mut() {
        *v = (rng.normal() * 4.0).round() / 4.0; // ties
    }
    vals[kk + 3] = f32::NEG_INFINITY;
    let row_bytes = kk * 4 + 16;
    let mut g = vec![0u8; 2 * t2 * row_bytes];
    for wr in 0..2 {
        for r in 0..t2 {
            for i in 0..kk {
                let at = (wr * t2 + r) * row_bytes + 16 + 4 * i;
                g[at..at + 4].copy_from_slice(&vals[(wr * t2 + r) * kk + i].to_le_bytes());
            }
        }
    }
    let d_g = dev(&g);
    for rank in 0..2 {
        let idx0: Vec<i32> = (0..t2 * kk).map(|i| i as i32).collect();
        let d_i = dev(&idx0);
        ok(unsafe { glm_split_share(st.raw(), p(&d_g), (t2 * row_bytes) as i64, row_bytes as i64, 16, p(&d_i), kk as i64, rank, kk as i32, t2 as i32) });
        let kept = d_i.download().unwrap();
        for r in 0..t2 {
            let mut all: Vec<(f32, usize)> = (0..2 * kk).map(|e| (vals[((e / kk) * t2 + r) * kk + e % kk], e)).collect();
            all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let top: std::collections::HashSet<usize> = all[..kk].iter().map(|x| x.1).collect();
            for i in 0..kk {
                let e = rank as usize * kk + i;
                let want = if top.contains(&e) { (r * kk + i) as i32 } else { -1 };
                assert_eq!(kept[r * kk + i], want, "share rank {rank} row {r} pick {i}");
            }
        }
    }
}

// ------------------------------------------------------------------ attention
struct Cache {
    cq: Vec<u8>,
    cs: Vec<f32>,
    cr: Vec<u16>,
}

fn latent(c: &Cache, row: usize, d: usize) -> f32 {
    bf(e4m3(c.cq[row * 512 + d]) * bf(c.cs[row * 4 + d / 128]))
}

/// The attention of q [H][576] over keys `keys` (cache rows, all valid), fp64 scores, p rounded
/// to bf16 for PV as the kernel does (relative to the max: the same up to the per-tile max).
fn ref_attn(c: &Cache, q: &[u16], hh: usize, keys: &[usize], scale: f64) -> Vec<f64> {
    let mut out = vec![0f64; hh * 512];
    for h in 0..hh {
        let qr = &q[h * 576..(h + 1) * 576];
        let sc: Vec<f64> = keys.iter().map(|&k| {
            let mut s = 0f64;
            for d in 0..512 {
                s += f32_from_bf16(qr[d]) as f64 * latent(c, k, d) as f64;
            }
            for d in 0..64 {
                s += f32_from_bf16(qr[512 + d]) as f64 * f32_from_bf16(c.cr[k * 64 + d]) as f64;
            }
            s * scale
        }).collect();
        let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for (j, &k) in keys.iter().enumerate() {
            for d in 0..512 {
                out[h * 512 + d] += e[j] / l * latent(c, k, d) as f64;
            }
        }
    }
    out
}

#[test]
fn mla_attention_direct_split_and_cross_rank_merge() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(5);
    let rows = 700usize;
    let mut c = Cache { cq: vec![0; rows * 512], cs: vec![0.0; rows * 4], cr: rng.bf16_vec(rows * 64, 1.0) };
    for r in 0..rows {
        for b in 0..4 {
            let (q, s) = fp8_row(&mut rng, 128);
            c.cq[r * 512 + b * 128..r * 512 + (b + 1) * 128].copy_from_slice(&q);
            c.cs[r * 4 + b] = s * 20.0;
        }
    }
    let (t, hh, kk) = (2usize, 64usize, 300usize);
    let q = rng.bf16_vec(t * hh * 576, 0.3);
    let bound = [650i32, 400];
    let mut idx = vec![0i32; t * kk];
    for (i, v) in idx.iter_mut().enumerate() {
        *v = (rng.next() % rows as u64) as i32;
        if i % 37 == 0 {
            *v = -1;
        }
    }
    let scale = 1.0 / 16.0;
    let (d_q, d_cq, d_cs, d_cr, d_idx, d_b) = (dev(&q), dev(&c.cq), dev(&c.cs), dev(&c.cr), dev(&idx), dev(&bound));
    let want: Vec<Vec<f64>> = (0..t).map(|r| {
        let keys: Vec<usize> = idx[r * kk..(r + 1) * kk].iter().filter(|&&k| k >= 0 && k <= bound[r]).map(|&k| k as usize).collect();
        ref_attn(&c, &q[r * hh * 576..(r + 1) * hh * 576], hh, &keys, scale)
    }).collect();
    let flat: Vec<f64> = want.concat();
    // direct, both head tilings
    for mt in [1, 2] {
        let d_o = DeviceBuffer::<u16>::zeroed(t * hh * 512).unwrap();
        ok(unsafe {
            glm_mla(st.raw(), mt, p(&d_q), 0, (hh * 576) as i64, 576, hh as i32, p(&d_cq), p(&d_cs), p(&d_cr), 0, p(&d_idx),
                    kk as i64, kk as i32, 0, p(&d_b), t as i32, hh as i32, scale as f32, 1, 320, null(), null(), null(), p(&d_o),
                    (hh * 512) as i64, 512, null())
        });
        last_ok();
        let o: Vec<f32> = d_o.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
        let e = rel_err(&o, &flat);
        eprintln!("mla direct mt {mt}: rel {e:.2e}");
        assert!(e < 1e-2, "mla direct mt {mt}: {e}");
    }
    // split into 4 partials, combined to bf16
    let ns = 4usize;
    let chunk = 96;
    let d_pm = DeviceBuffer::<f32>::zeroed(t * hh * ns).unwrap();
    let d_pl = DeviceBuffer::<f32>::zeroed(t * hh * ns).unwrap();
    let d_pa = DeviceBuffer::<f32>::zeroed(t * hh * ns * 512).unwrap();
    ok(unsafe {
        glm_mla(st.raw(), 2, p(&d_q), 0, (hh * 576) as i64, 576, hh as i32, p(&d_cq), p(&d_cs), p(&d_cr), 0, p(&d_idx), kk as i64,
                kk as i32, 0, p(&d_b), t as i32, hh as i32, scale as f32, ns as i32, chunk, p(&d_pm), p(&d_pl), p(&d_pa), null(), 0, 0,
                null())
    });
    let d_o = DeviceBuffer::<u16>::zeroed(t * hh * 512).unwrap();
    ok(unsafe { glm_mla_combine(st.raw(), p(&d_pm), p(&d_pl), p(&d_pa), t as i32, hh as i32, ns as i32, p(&d_o), (hh * 512) as i64, 512, null(), null(), null(), 0, 0, 0) });
    last_ok();
    let o: Vec<f32> = d_o.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
    let e = rel_err(&o, &flat);
    eprintln!("mla split: rel {e:.2e}");
    assert!(e < 1e-2, "mla split {e}");

    // split KV: picks divided between two "ranks" by key parity; each attends all heads over its
    // share, sends the other's heads, and the merge gives the whole attention for its own heads
    let hl = hh / 2;
    let row_bytes = (hl * 512 * 2 + hl * 4) as i64;
    let mut owns = vec![];
    let mut sends = vec![];
    for rank in 0..2 {
        let mine: Vec<i32> = idx.iter().map(|&k| if k >= 0 && k % 2 == rank { k } else { -1 }).collect();
        let d_i = dev(&mine);
        ok(unsafe {
            glm_mla(st.raw(), 2, p(&d_q), 0, (hh * 576) as i64, 576, hh as i32, p(&d_cq), p(&d_cs), p(&d_cr), 0, p(&d_i),
                    kk as i64, kk as i32, 0, p(&d_b), t as i32, hh as i32, scale as f32, ns as i32, chunk, p(&d_pm), p(&d_pl),
                    p(&d_pa), null(), 0, 0, null())
        });
        let d_oo = DeviceBuffer::<f32>::zeroed(t * hl * 512).unwrap();
        let d_ol = DeviceBuffer::<f32>::zeroed(t * hl).unwrap();
        let d_send = DeviceBuffer::<u8>::zeroed(t * row_bytes as usize).unwrap();
        ok(unsafe {
            glm_mla_combine(st.raw(), p(&d_pm), p(&d_pl), p(&d_pa), t as i32, hh as i32, ns as i32, null(), 0, 0, p(&d_oo),
                            p(&d_ol), p(&d_send), row_bytes, hl as i32, rank * hl as i32)
        });
        last_ok();
        owns.push((d_oo, d_ol));
        sends.push(d_send);
    }
    for rank in 0..2usize {
        let d_m = DeviceBuffer::<u16>::zeroed(t * hl * 512).unwrap();
        ok(unsafe { glm_mla_merge(st.raw(), p(&owns[rank].0), p(&owns[rank].1), p(&sends[1 - rank]), row_bytes, t as i32, hl as i32, p(&d_m)) });
        last_ok();
        let o: Vec<f32> = d_m.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
        let w: Vec<f64> = (0..t).flat_map(|r| want[r][rank * hl * 512..(rank + 1) * hl * 512].to_vec()).collect();
        let e = rel_err(&o, &w);
        eprintln!("mla split-KV rank {rank}: rel {e:.2e}");
        assert!(e < 1e-2, "split kv rank {rank}: {e}");
    }

    // dense causal (no picks): keys 0..=bound
    let d_o = DeviceBuffer::<u16>::zeroed(t * hh * 512).unwrap();
    let bound2 = [40i32, 7];
    let d_b2 = dev(&bound2);
    ok(unsafe {
        glm_mla(st.raw(), 2, p(&d_q), 0, (hh * 576) as i64, 576, hh as i32, p(&d_cq), p(&d_cs), p(&d_cr), 0, null(), 0, 4096, 1,
                p(&d_b2), t as i32, hh as i32, scale as f32, 1, 4096, null(), null(), null(), p(&d_o), (hh * 512) as i64, 512,
                null())
    });
    last_ok();
    let o: Vec<f32> = d_o.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
    let w: Vec<f64> = (0..t).flat_map(|r| {
        let keys: Vec<usize> = (0..=bound2[r] as usize).collect();
        ref_attn(&c, &q[r * hh * 576..(r + 1) * hh * 576], hh, &keys, scale)
    }).collect();
    let e = rel_err(&o, &w);
    eprintln!("mla dense causal: rel {e:.2e}");
    assert!(e < 1e-2, "dense causal {e}");
}

// ------------------------------------------------------------------ lanes
fn off_by<T: Copy>(b: &DeviceBuffer<T>, n: usize) -> *mut c_void {
    unsafe { (p(b) as *mut T).add(n) as *mut c_void }
}

// Two sequences in one pass, each in its own rows of one cache buffer (lane 1's koff rows on):
// positions, index scores and MLA give each lane's rows bit for bit what a pass of that lane
// alone gives over its own rows (the single-lane path, lanes null).
#[test]
fn two_lanes_in_one_pass_match_each_lane_alone() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(11);
    let (world, rank) = (2i32, 1i32);
    let slot = 1100usize; // a lane's cache rows: s_loc of its own and the spare
    let s_loc = slot - 1;
    let rows = 2 * slot;
    // lane 0: 3 rows from position 1500, lane 1: 2 rows from 731
    let (n0, n1, p0, p1) = (3usize, 2usize, 1500i32, 731i32);
    let t = n0 + n1;
    let lanes = [0i32, n0 as i32, p0, 0, n0 as i32, n1 as i32, p1, slot as i32];
    let d_lanes = dev(&lanes);
    let d_pos = DeviceBuffer::<i32>::zeroed(t).unwrap();
    let d_rows = DeviceBuffer::<i32>::zeroed(t).unwrap();
    let d_bound = DeviceBuffer::<i32>::zeroed(t).unwrap();
    let d_koff = DeviceBuffer::<i32>::zeroed(t).unwrap();
    ok(unsafe {
        glm_positions(st.raw(), null(), t as i32, rank, world, s_loc as i32, p(&d_pos), p(&d_rows), p(&d_bound), p(&d_lanes), 2,
                      p(&d_koff))
    });
    let (pos, rws, bound, koff) =
        (d_pos.download().unwrap(), d_rows.download().unwrap(), d_bound.download().unwrap(), d_koff.download().unwrap());
    for r in 0..t {
        let (p, o) = if r < n0 { (p0 + r as i32, 0) } else { (p1 + (r - n0) as i32, slot as i32) };
        assert_eq!(pos[r], p, "pos {r}");
        assert_eq!(koff[r], o, "koff {r}");
        assert_eq!(rws[r], o + if p % world == rank { p / world } else { s_loc as i32 }, "rows {r}");
        assert_eq!(bound[r], (p - rank) / world, "bound {r}");
    }
    // ... and one lane alone, from its device pos0, as today
    let d_p0 = dev(&[p1]);
    let d_pos1 = DeviceBuffer::<i32>::zeroed(n1).unwrap();
    let d_rows1 = DeviceBuffer::<i32>::zeroed(n1).unwrap();
    let d_bound1 = DeviceBuffer::<i32>::zeroed(n1).unwrap();
    ok(unsafe {
        glm_positions(st.raw(), p(&d_p0), n1 as i32, rank, world, s_loc as i32, p(&d_pos1), p(&d_rows1), p(&d_bound1), null(), 0,
                      null())
    });
    assert_eq!(d_pos1.download().unwrap(), &pos[n0..]);
    assert_eq!(d_bound1.download().unwrap(), &bound[n0..]);
    assert_eq!(d_rows1.download().unwrap().iter().map(|&r| r + slot as i32).collect::<Vec<_>>(), &rws[n0..]);

    // index scores over the lane's own keys
    let q = rng.bf16_vec(t * 32 * 128, 0.5);
    let w: Vec<f32> = (0..t * 32).map(|_| rng.normal()).collect();
    let mut kq = vec![0u8; rows * 128];
    let mut ks = vec![0f32; rows];
    for r in 0..rows {
        let (row, s) = fp8_row(&mut rng, 128);
        kq[r * 128..(r + 1) * 128].copy_from_slice(&row);
        ks[r] = s;
    }
    let (d_q, d_w, d_kq, d_ks) = (dev(&q), dev(&w), dev(&kq), dev(&ks));
    let (wscale, inv_d) = (32f32.powf(-0.5), 128f32.powf(-0.5));
    let ld = s_loc;
    let both = DeviceBuffer::<f32>::zeroed(t * ld).unwrap();
    ok(unsafe {
        glm_index_scores(st.raw(), p(&d_q), p(&d_w), wscale, p(&d_kq), p(&d_ks), p(&d_pos), t as i32, s_loc as i32, world, rank,
                         0, inv_d, p(&both), ld as i64, p(&d_lanes), 2)
    });
    let both = both.download().unwrap();
    for (r0, n, o) in [(0, n0, 0), (n0, n1, slot)] {
        let one = DeviceBuffer::<f32>::zeroed(n * ld).unwrap();
        ok(unsafe {
            glm_index_scores(st.raw(), off_by(&d_q, r0 * 32 * 128), off_by(&d_w, r0 * 32), wscale, off_by(&d_kq, o * 128),
                             off_by(&d_ks, o), off_by(&d_pos, r0), n as i32, s_loc as i32, world, rank, 0, inv_d, p(&one),
                             ld as i64, null(), 0)
        });
        let one = one.download().unwrap();
        assert!(one == both[r0 * ld..(r0 + n) * ld], "index scores, the lane from row {r0}");
        assert!(one[..ld].iter().any(|&v| v > 0.0), "index scores: a row scored");
    }

    // MLA over each row's picks in its lane's rows, direct and split + combine
    let mut cq = vec![0u8; rows * 512];
    let mut cs = vec![0f32; rows * 4];
    for r in 0..rows {
        for b in 0..4 {
            let (qq, sc) = fp8_row(&mut rng, 128);
            cq[r * 512 + b * 128..r * 512 + (b + 1) * 128].copy_from_slice(&qq);
            cs[r * 4 + b] = sc * 20.0;
        }
    }
    let cr = rng.bf16_vec(rows * 64, 1.0);
    let (hh, kk) = (64usize, 300usize);
    let qa = rng.bf16_vec(t * hh * 576, 0.3);
    let mut idx = vec![0i32; t * kk];
    for (i, v) in idx.iter_mut().enumerate() {
        let b = bound[i / kk] as u64 + 1;
        *v = if i % 37 == 0 { -1 } else { (rng.next() % b) as i32 };
    }
    let (d_qa, d_cq, d_cs, d_cr, d_idx) = (dev(&qa), dev(&cq), dev(&cs), dev(&cr), dev(&idx));
    let scale = 1.0f32 / 16.0;
    let ns = 4usize;
    let run = |r0: usize, n: usize, o: usize, lanes: bool, split: bool| -> Vec<u16> {
        let d_o = DeviceBuffer::<u16>::zeroed(n * hh * 512).unwrap();
        let (kc, kv) = if lanes { (p(&d_koff), 0) } else { (null(), o) };
        let (cq, cs, cr) = (off_by(&d_cq, kv * 512), off_by(&d_cs, kv * 4), off_by(&d_cr, kv * 64));
        let (qp, ip, bp) = (off_by(&d_qa, r0 * hh * 576), off_by(&d_idx, r0 * kk), off_by(&d_bound, r0));
        if split {
            let d_pm = DeviceBuffer::<f32>::zeroed(n * hh * ns).unwrap();
            let d_pl = DeviceBuffer::<f32>::zeroed(n * hh * ns).unwrap();
            let d_pa = DeviceBuffer::<f32>::zeroed(n * hh * ns * 512).unwrap();
            ok(unsafe {
                glm_mla(st.raw(), 2, qp, 0, (hh * 576) as i64, 576, hh as i32, cq, cs, cr, 0, ip, kk as i64, kk as i32, 0, bp,
                        n as i32, hh as i32, scale, ns as i32, 96, p(&d_pm), p(&d_pl), p(&d_pa), null(), 0, 0, kc)
            });
            ok(unsafe {
                glm_mla_combine(st.raw(), p(&d_pm), p(&d_pl), p(&d_pa), n as i32, hh as i32, ns as i32, p(&d_o),
                                (hh * 512) as i64, 512, null(), null(), null(), 0, 0, 0)
            });
        } else {
            ok(unsafe {
                glm_mla(st.raw(), 2, qp, 0, (hh * 576) as i64, 576, hh as i32, cq, cs, cr, 0, ip, kk as i64, kk as i32, 0, bp,
                        n as i32, hh as i32, scale, 1, 320, null(), null(), null(), p(&d_o), (hh * 512) as i64, 512, kc)
            });
        }
        last_ok();
        d_o.download().unwrap()
    };
    for split in [false, true] {
        let both = run(0, t, 0, true, split);
        for (r0, n, o) in [(0, n0, 0), (n0, n1, slot)] {
            let one = run(r0, n, o, false, split);
            assert!(one == both[r0 * hh * 512..(r0 + n) * hh * 512], "mla (split {split}), the lane from row {r0}");
        }
        // lane 1's rows read its own cache rows, not lane 0's
        let wrong = run(n0, n1, 0, false, split);
        assert!(wrong != both[n0 * hh * 512..], "mla (split {split}): lane 1 read lane 0's rows");
    }
}

// ------------------------------------------------------------------ absorb, router
#[test]
fn absorb_and_router() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(9);
    let hl = 4usize;
    let rows = hl * 448;
    let w: Vec<u8> = (0..rows * 512).map(|_| loop {
        let b = (rng.next() >> 32) as u8;
        if b & 0x7f != 0x7f {
            break b;
        }
    }).collect();
    let ws: Vec<f32> = (0..rows * 4).map(|_| 1e-3 * (1.0 + rng.unif())).collect();
    let wf = |r: usize, c: usize| bf(e4m3(w[r * 512 + c]) * ws[r * 4 + c / 128]) as f64;
    // 33 rows: past the 8 / 16 rows a launch (a copy round's verify)
    for t in [3usize, 33] {
        let q = rng.bf16_vec(t * hl * 256, 1.0);
        let (d_w, d_ws, d_q) = (dev(&w), dev(&ws), dev(&q));
        let d_out = DeviceBuffer::<u16>::zeroed(t * hl * 576).unwrap();
        ok(unsafe { glm_absorb_q(st.raw(), p(&d_q), (hl * 256) as i64, 256, p(&d_w), p(&d_ws), p(&d_out), (hl * 576) as i64, 576, t as i32, hl as i32) });
        let got: Vec<f32> = d_out.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
        let mut gq = vec![];
        let mut wq = vec![];
        for r in 0..t {
            for h in 0..hl {
                for c in 0..512 {
                    let mut a = 0f64;
                    for n in 0..192 {
                        a += f32_from_bf16(q[(r * hl + h) * 256 + n]) as f64 * wf(h * 448 + n, c);
                    }
                    wq.push(a);
                    gq.push(got[(r * hl + h) * 576 + c]);
                }
            }
    }
    let e = rel_err(&gq, &wq);
    assert!(e < 5e-3, "absorb_q {e} at {t} rows");
    let o = rng.bf16_vec(t * hl * 512, 1.0);
    let d_o = dev(&o);
    let d_y = DeviceBuffer::<u16>::zeroed(t * hl * 256).unwrap();
    ok(unsafe { glm_absorb_o(st.raw(), p(&d_o), (hl * 512) as i64, 512, p(&d_w), p(&d_ws), p(&d_y), (hl * 256) as i64, t as i32, hl as i32) });
    let got: Vec<f32> = d_y.download().unwrap().iter().map(|&v| f32_from_bf16(v)).collect();
    let mut wo = vec![];
    for r in 0..t {
        for h in 0..hl {
            for v in 0..256 {
                let mut a = 0f64;
                for c in 0..512 {
                    a += f32_from_bf16(o[(r * hl + h) * 512 + c]) as f64 * wf(h * 448 + 192 + v, c);
                }
                wo.push(a);
            }
        }
    }
    let e = rel_err(&got, &wo);
    assert!(e < 5e-3, "absorb_o {e} at {t} rows");
    }

    // router: sigmoid + bias, top-8 over allowed (descending, ties low), renormalized x 2.5
    let tt = 3usize;
    let logits: Vec<f32> = (0..tt * 256).map(|_| rng.normal() * 2.0).collect();
    let bias: Vec<f32> = (0..256).map(|_| rng.normal() * 0.1).collect();
    let allowed: Vec<u8> = (0..256).map(|e| (e % 3 != 0) as u8).collect();
    let lut: Vec<i32> = (0..256).map(|e| if e % 3 != 0 { 1000 + e } else { -1 }).collect();
    let (d_l, d_bias, d_al, d_lut) = (dev(&logits), dev(&bias), dev(&allowed), dev(&lut));
    let d_eids = DeviceBuffer::<i32>::zeroed(tt * 8).unwrap();
    let d_ew = DeviceBuffer::<f32>::zeroed(tt * 8).unwrap();
    let d_hot = DeviceBuffer::<f32>::zeroed(tt * 16).unwrap();
    ok(unsafe { glm_router(st.raw(), p(&d_l), p(&d_bias), null(), null(), tt as i32, 8, 2.5, p(&d_eids), p(&d_ew), p(&d_al), p(&d_hot), null(), null(), null(), null(), null(), null(), null(), 0, null()) });
    let d_ids = DeviceBuffer::<i32>::zeroed(tt * 8).unwrap();
    let d_rw = DeviceBuffer::<f32>::zeroed(tt * 8).unwrap();
    let d_dem = DeviceBuffer::<f32>::zeroed(256).unwrap();
    let d_acc = DeviceBuffer::<f64>::zeroed(4).unwrap();
    // the rows' lanes: row 0 lane 0, rows 1-2 lane 1 (each lane's counts and demand apart)
    let rl = [0i32, 1, 1];
    let d_rl = dev(&rl);
    let d_lacc = DeviceBuffer::<f64>::zeroed(8).unwrap();
    let d_ldem = DeviceBuffer::<f32>::zeroed(2 * 300).unwrap();
    ok(unsafe { glm_router(st.raw(), p(&d_l), p(&d_bias), p(&d_al), p(&d_lut), tt as i32, 8, 2.5, p(&d_ids), p(&d_rw), null(), null(), p(&d_eids), p(&d_ew), p(&d_dem), p(&d_acc), null(), p(&d_lacc), p(&d_ldem), 300, p(&d_rl)) });
    last_ok();
    let (eids, ew, hot, ids, rw, dem, acc) = (d_eids.download().unwrap(), d_ew.download().unwrap(), d_hot.download().unwrap(),
                                              d_ids.download().unwrap(), d_rw.download().unwrap(), d_dem.download().unwrap(),
                                              d_acc.download().unwrap());
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut want_dem = vec![0f64; 256];
    let mut missed = 0.0;
    let (mut lane_dem, mut lane_acc) = (vec![0f64; 600], [0f64; 8]);
    for r in 0..tt {
        let pick = |mask: &dyn Fn(usize) -> bool| {
            let mut c: Vec<(f32, usize)> = (0..256).filter(|&e| mask(e)).map(|e| (sig(logits[r * 256 + e]) + bias[e], e)).collect();
            c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let ch: Vec<usize> = c[..8].iter().map(|x| x.1).collect();
            let tot: f32 = ch.iter().map(|&e| sig(logits[r * 256 + e])).sum();
            (ch.clone(), ch.iter().map(|&e| sig(logits[r * 256 + e]) / (tot + 1e-20) * 2.5).collect::<Vec<f32>>())
        };
        let (ech, ewt) = pick(&|_| true);
        let (rch, rwt) = pick(&|e| e % 3 != 0);
        for k in 0..8 {
            assert_eq!(eids[r * 8 + k], ech[k] as i32);
            assert!((ew[r * 8 + k] - ewt[k]).abs() < 1e-6);
            assert_eq!(ids[r * 8 + k], 1000 + rch[k] as i32);
            assert!((rw[r * 8 + k] - rwt[k]).abs() < 1e-6);
            let miss = ech[k] % 3 == 0;
            assert_eq!(hot[(r * 8 + k) * 2], if miss { ewt[k] / 2.5 } else { 0.0 });
            assert_eq!(hot[(r * 8 + k) * 2 + 1], ech[k] as f32);
            want_dem[ech[k]] += ewt[k] as f64;
            missed += miss as u32 as f64;
            let z = rl[r] as usize;
            lane_dem[z * 300 + ech[k]] += ewt[k] as f64;
            lane_acc[4 * z] += 1.0;
            lane_acc[4 * z + 1] += miss as u32 as f64;
            lane_acc[4 * z + 2] += ewt[k] as f64;
            lane_acc[4 * z + 3] += if miss { ewt[k] as f64 } else { 0.0 };
        }
    }
    for e in 0..256 {
        assert!((dem[e] as f64 - want_dem[e]).abs() < 1e-5);
    }
    assert_eq!(acc[0], (tt * 8) as f64);
    assert_eq!(acc[1], missed);
    let (lacc, ldem) = (d_lacc.download().unwrap(), d_ldem.download().unwrap());
    for i in 0..8 {
        assert!((lacc[i] - lane_acc[i]).abs() < 1e-5, "lane counter {i}: {} vs {}", lacc[i], lane_acc[i]);
    }
    for i in 0..600 {
        assert!((ldem[i] as f64 - lane_dem[i]).abs() < 1e-5, "lane demand {i}");
    }
}
