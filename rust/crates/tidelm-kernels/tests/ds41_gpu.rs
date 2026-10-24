//! GPU tests for cuda/ds41.cu: every kernel against a CPU transcription of the Python it
//! reproduces. GEMMs are checked against an f64 reference (within one bf16 ulp after
//! rounding, or a tight relative bound for fp32 outputs); elementwise kernels whose math
//! is exact (dequant, embed, QDQ) must match bit for bit. Small allocations only.

use std::os::raw::c_void;

use tidelm_cuda::{gpu_available, set_device, DeviceBuffer, Stream};
use tidelm_kernels::ds41::*;

// ------------------------------------------------------------------ helpers
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
    if e == 15 && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        s * m * 2f32.powi(-9)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}
fn e8m0_lin(b: u8) -> f32 {
    f32::from_bits((b as u32) << 23)
}
/// Round-to-nearest-even into e4m3 (positive finite inputs <= 448), then back to f32.
fn round_e4m3(x: f32) -> f32 {
    let mut best = 0f32;
    let mut best_b = 0u8;
    for b in 0u8..=0x7e {
        let v = e4m3(b);
        let d = (v - x).abs();
        let bd = (best - x).abs();
        if d < bd || (d == bd && (b & 1) == 0 && (best_b & 1) == 1) {
            best = v;
            best_b = b;
        }
    }
    best
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
    fn fp8_vec(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| loop {
                let b = (self.next() >> 32) as u8;
                if b & 0x7f != 0x7f {
                    break b;
                }
            })
            .collect()
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

fn ok(rc: i32) {
    assert_eq!(rc, 0, "kernel launch failed with CUDA error {rc}");
}

/// |got - want| <= 1 bf16 ulp at want's magnitude (bf16 rounding of an f64-exact sum may
/// land either side when the fp32 accumulation differs in its last bits).
fn assert_bf16_close(got: &[u16], want: &[f64], what: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = f32_from_bf16(g) as f64;
        let ulp = (w.abs().max(1e-30)).log2().floor();
        let tol = 2f64.powf(ulp - 7.0) * 1.01;
        assert!((g - w).abs() <= tol + 1e-30, "{what}[{i}]: got {g}, want {w} (tol {tol})");
    }
}

fn assert_f32_close(got: &[f32], want: &[f64], rel: f64, what: &str) {
    let scale = want.iter().fold(0f64, |a, &b| a.max(b.abs())).max(1e-30);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            ((g as f64) - w).abs() <= rel * scale,
            "{what}[{i}]: got {g}, want {w} (scale {scale})"
        );
    }
}

// ------------------------------------------------------------------ GEMMs
/// (sum, sum of |terms|) per output: the second bounds fp32 accumulation error, which with
/// cancelling random terms can exceed one bf16 ulp of a small result.
fn ref_linear_mag(x: &[u16], w: impl Fn(usize, usize) -> f64, m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut out, mut mag) = (vec![0f64; m * n], vec![0f64; m * n]);
    for i in 0..m {
        for j in 0..n {
            for kk in 0..k {
                let t = f32_from_bf16(x[i * k + kk]) as f64 * w(j, kk);
                out[i * n + j] += t;
                mag[i * n + j] += t.abs();
            }
        }
    }
    (out, mag)
}

/// Half a bf16 ulp of rounding plus fp32 accumulation error (a few 2^-24 x sum|terms|).
fn assert_bf16_accum(got: &[u16], want: &[f64], mag: &[f64], what: &str) {
    for (i, ((&g, &w), &a)) in got.iter().zip(want).zip(mag).enumerate() {
        let g = f32_from_bf16(g) as f64;
        let ulp = 2f64.powf((w.abs().max(1e-30)).log2().floor() - 7.0);
        let tol = 0.51 * ulp + 2f64.powi(-24) * 8.0 * a + 1e-30;
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g}, want {w} (tol {tol})");
    }
}

fn ref_linear(x: &[u16], w: impl Fn(usize, usize) -> f64, m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += f32_from_bf16(x[i * k + kk]) as f64 * w(j, kk);
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn linear_fp8_matches_reference_and_is_row_invariant() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(7);
    let (n, k) = (96usize, 1280usize); // N not a multiple of 32 rows per block, K not of 512
    let w = rng.fp8_vec(n * k);
    let lds = k.div_ceil(32);
    let s: Vec<u8> = (0..n.div_ceil(32) * lds).map(|_| 118 + (rng.next() % 12) as u8).collect();
    let wf = |j: usize, kk: usize| e4m3(w[j * k + kk]) as f64 * e8m0_lin(s[(j / 32) * lds + kk / 32]) as f64;
    let (dw, ds) = (dev(&w), dev(&s));
    let mut row0: Option<Vec<u16>> = None;
    for m in [1usize, 3, 6, 16] {
        let x = rng.bf16_vec(m * k, 1.0);
        let want = ref_linear(&x, wf, m, n, k);
        let (dx, dy) = (dev(&x), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
        ok(unsafe {
            ds_linear_fp8(st.raw(), p(&dx), k as i64, p(&dw), p(&ds), lds as i64, m as i32, n as i32,
                          k as i32, 1, p(&dy), null(), n as i64, null())
        });
        st.synchronize().unwrap();
        let got = dy.download().unwrap();
        assert_bf16_close(&got, &want, &format!("fp8 M={m}"));
        // row invariance: the same input row gives the same bits whatever M is
        let x0 = &x[..k];
        let (dx1, dy1) = (dev(x0), DeviceBuffer::<u16>::zeroed(n).unwrap());
        ok(unsafe {
            ds_linear_fp8(st.raw(), p(&dx1), k as i64, p(&dw), p(&ds), lds as i64, 1, n as i32, k as i32,
                          1, p(&dy1), null(), n as i64, null())
        });
        st.synchronize().unwrap();
        assert_eq!(dy1.download().unwrap(), got[..n].to_vec(), "row 0 differs between M=1 and M={m}");
        row0.get_or_insert(got[..n].to_vec());
    }
    // split-K into fp32 partials, reduced in split order
    let m = 4;
    let x = rng.bf16_vec(m * k, 1.0);
    let want = ref_linear(&x, wf, m, n, k);
    let splits = 3;
    let (dx, dpart, dy) = (
        dev(&x),
        DeviceBuffer::<f32>::zeroed(splits * m * n).unwrap(),
        DeviceBuffer::<f32>::zeroed(m * n).unwrap(),
    );
    ok(unsafe {
        ds_linear_fp8(st.raw(), p(&dx), k as i64, p(&dw), p(&ds), lds as i64, m as i32, n as i32, k as i32,
                      splits as i32, null(), null(), n as i64, p(&dpart))
    });
    ok(unsafe { ds_splits_reduce(st.raw(), p(&dpart), splits as i32, m as i32, n as i32, null(), p(&dy), n as i64) });
    st.synchronize().unwrap();
    assert_f32_close(&dy.download().unwrap(), &want, 1e-5, "fp8 split-K");
}

#[test]
fn linear_fp8_grouped_reads_each_groups_slice() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(11);
    let (g, r, k, m) = (4usize, 64usize, 512usize, 3usize);
    let n = g * r;
    let w = rng.fp8_vec(n * k);
    let lds = k / 32;
    let s: Vec<u8> = (0..(n / 32) * lds).map(|_| 120 + (rng.next() % 8) as u8).collect();
    let x = rng.bf16_vec(m * g * k, 1.0); // [M, G, K]
    let mut want = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let grp = j / r;
            let mut acc = 0f64;
            for kk in 0..k {
                acc += f32_from_bf16(x[(i * g + grp) * k + kk]) as f64
                    * e4m3(w[j * k + kk]) as f64
                    * e8m0_lin(s[(j / 32) * lds + kk / 32]) as f64;
            }
            want[i * n + j] = acc;
        }
    }
    let (dx, dw, ds, dy) = (dev(&x), dev(&w), dev(&s), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
    ok(unsafe {
        ds_linear_fp8_grouped(st.raw(), p(&dx), (g * k) as i64, k as i64, r as i32, p(&dw), p(&ds),
                              lds as i64, m as i32, n as i32, k as i32, p(&dy), n as i64)
    });
    st.synchronize().unwrap();
    assert_bf16_close(&dy.download().unwrap(), &want, "grouped");
}

#[test]
fn linear_bf16_and_f32_weights() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(13);
    let (m, n, k) = (5usize, 70usize, 1536usize);
    let x = rng.bf16_vec(m * k, 1.0);
    let wb = rng.bf16_vec(n * k, 0.05);
    let want = ref_linear(&x, |j, kk| f32_from_bf16(wb[j * k + kk]) as f64, m, n, k);
    let (dx, dwb, dy) = (dev(&x), dev(&wb), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
    ok(unsafe {
        ds_linear_bf16(st.raw(), p(&dx), k as i64, p(&dwb), m as i32, n as i32, k as i32, 1, p(&dy), null(),
                       n as i64, null())
    });
    st.synchronize().unwrap();
    assert_bf16_close(&dy.download().unwrap(), &want, "bf16 weights");

    let wf: Vec<f32> = (0..n * k).map(|_| rng.normal() * 0.05).collect();
    let want = ref_linear(&x, |j, kk| wf[j * k + kk] as f64, m, n, k);
    let (dwf, dyf) = (dev(&wf), DeviceBuffer::<f32>::zeroed(m * n).unwrap());
    ok(unsafe {
        ds_linear_f32w(st.raw(), p(&dx), k as i64, p(&dwf), m as i32, n as i32, k as i32, 1, p(&dyf),
                       n as i64, null())
    });
    st.synchronize().unwrap();
    assert_f32_close(&dyf.download().unwrap(), &want, 1e-5, "f32 weights");
}

// ------------------------------------------------------------------ norms / HC
fn ref_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rs = 1.0 / (ss * (1.0 / d as f32) + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| bf(g * (v * rs))).collect()
}

#[test]
fn rmsnorm_and_hc_pre_norm() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(17);
    let (m, d, hc) = (3usize, 5120usize, 4usize);
    let eps = 1e-20f32;
    let x = rng.bf16_vec(m * d, 2.0);
    let w = rng.bf16_vec(d, 1.0);
    let wf: Vec<f32> = w.iter().map(|&b| f32_from_bf16(b)).collect();
    let (dx, dw, dy) = (dev(&x), dev(&w), DeviceBuffer::<u16>::zeroed(m * d).unwrap());
    ok(unsafe { ds_rmsnorm(st.raw(), p(&dx), d as i64, p(&dw), p(&dy), d as i64, m as i32, d as i32, eps) });
    st.synchronize().unwrap();
    let got = dy.download().unwrap();
    for i in 0..m {
        let row: Vec<f32> = x[i * d..(i + 1) * d].iter().map(|&b| f32_from_bf16(b)).collect();
        let want = ref_rmsnorm(&row, &wf, eps);
        let want64: Vec<f64> = want.iter().map(|&v| v as f64).collect();
        assert_bf16_close(&got[i * d..(i + 1) * d], &want64, "rmsnorm");
    }

    let h = rng.bf16_vec(m * hc * d, 1.0);
    let pre: Vec<f32> = (0..m * hc).map(|_| rng.unif()).collect();
    let (dh, dpre, dy2) = (dev(&h), dev(&pre), DeviceBuffer::<u16>::zeroed(m * d).unwrap());
    ok(unsafe { ds_hc_pre_norm(st.raw(), p(&dh), p(&dpre), p(&dw), p(&dy2), m as i32, d as i32, hc as i32, eps) });
    st.synchronize().unwrap();
    let got = dy2.download().unwrap();
    for i in 0..m {
        let v: Vec<f32> = (0..d)
            .map(|j| {
                let mut acc = pre[i * hc] * f32_from_bf16(h[(i * hc) * d + j]);
                for c in 1..hc {
                    acc += pre[i * hc + c] * f32_from_bf16(h[(i * hc + c) * d + j]);
                }
                bf(acc)
            })
            .collect();
        let want: Vec<f64> = ref_rmsnorm(&v, &wf, eps).iter().map(|&v| v as f64).collect();
        assert_bf16_close(&got[i * d..(i + 1) * d], &want, "hc_pre_norm");
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn hc_sinkhorn_matches_the_reference_transcription() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(19);
    let (m, iters, eps) = (5usize, 20, 1e-6f32);
    let mixes: Vec<f32> = (0..m * 24).map(|_| rng.normal()).collect();
    let rsq: Vec<f32> = (0..m).map(|_| 0.5 + rng.unif()).collect();
    let scale = [0.7f32, 1.3, 0.9];
    let base: Vec<f32> = (0..24).map(|_| rng.normal() * 0.3).collect();
    let (dmx, drs, dsc, dba) = (dev(&mixes), dev(&rsq), dev(&scale), dev(&base));
    let (dpre, dpost, dcomb) = (
        DeviceBuffer::<f32>::zeroed(m * 4).unwrap(),
        DeviceBuffer::<f32>::zeroed(m * 4).unwrap(),
        DeviceBuffer::<f32>::zeroed(m * 16).unwrap(),
    );
    ok(unsafe {
        ds_hc_sinkhorn(st.raw(), p(&dmx), p(&drs), p(&dsc), p(&dba), m as i32, iters, eps, p(&dpre), p(&dpost),
                       p(&dcomb))
    });
    st.synchronize().unwrap();
    let (gpre, gpost, gcomb) = (dpre.download().unwrap(), dpost.download().unwrap(), dcomb.download().unwrap());
    for r in 0..m {
        let mx: Vec<f32> = (0..24).map(|i| mixes[r * 24 + i] * rsq[r]).collect();
        for i in 0..4 {
            let pre = sigmoid(mx[i] * scale[0] + base[i]) + eps;
            let post = 2.0 * sigmoid(mx[4 + i] * scale[1] + base[4 + i]);
            assert!((gpre[r * 4 + i] - pre).abs() < 1e-6);
            assert!((gpost[r * 4 + i] - post).abs() < 1e-6);
        }
        let mut c = [[0f32; 4]; 4];
        for i in 0..4 {
            let row: Vec<f32> = (0..4).map(|j| mx[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]).collect();
            let mxr = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - mxr).exp()).collect();
            let s: f32 = e.iter().sum();
            for j in 0..4 {
                c[i][j] = e[j] / s + eps;
            }
        }
        let col = |c: &mut [[f32; 4]; 4]| {
            for j in 0..4 {
                let s: f32 = (0..4).map(|i| c[i][j]).sum();
                for row in c.iter_mut() {
                    row[j] /= s + eps;
                }
            }
        };
        col(&mut c);
        for _ in 0..iters - 1 {
            for row in c.iter_mut() {
                let s: f32 = row.iter().sum();
                for v in row.iter_mut() {
                    *v /= s + eps;
                }
            }
            col(&mut c);
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!((gcomb[r * 16 + i * 4 + j] - c[i][j]).abs() < 2e-6, "comb[{r}][{i}][{j}]");
            }
        }
    }
}

#[test]
fn hc_post_mixes_over_the_first_index() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(23);
    let (m, d) = (2usize, 1000usize);
    let y = rng.bf16_vec(m * d, 1.0);
    let res = rng.bf16_vec(m * 4 * d, 1.0);
    let post: Vec<f32> = (0..m * 4).map(|_| rng.unif() * 2.0).collect();
    let comb: Vec<f32> = (0..m * 16).map(|_| rng.unif()).collect();
    let (dy, dres, dpost, dcomb) = (dev(&y), dev(&res), dev(&post), dev(&comb));
    // in place on the residual, as the engine runs it
    ok(unsafe { ds_hc_post(st.raw(), p(&dy), p(&dres), p(&dpost), p(&dcomb), p(&dres), m as i32, d as i32) });
    st.synchronize().unwrap();
    let got = dres.download().unwrap();
    for r in 0..m {
        for j in 0..4 {
            for c in 0..d {
                let mut mixed = comb[r * 16 + j] * f32_from_bf16(res[(r * 4) * d + c]);
                for i in 1..4 {
                    mixed = comb[r * 16 + i * 4 + j].mul_add(f32_from_bf16(res[(r * 4 + i) * d + c]), mixed);
                }
                let want = bf(post[r * 4 + j] * f32_from_bf16(y[r * d + c]) + mixed);
                assert_eq!(f32_from_bf16(got[(r * 4 + j) * d + c]), want, "hc_post[{r}][{j}][{c}]");
            }
        }
    }
}

// ------------------------------------------------------------------ rope / router / ffn
#[test]
fn rope_rotates_adjacent_pairs_and_inverts() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(29);
    let (rows, heads, hd, rd) = (3usize, 4usize, 512usize, 64usize);
    let x = rng.bf16_vec(rows * heads * hd, 1.0);
    let positions = [0i32, 5, 700];
    let table: Vec<[f32; 2]> = (0..701)
        .flat_map(|p| {
            (0..rd / 2).map(move |i| {
                let f = 1.0f32 / 10000f32.powf((2 * i) as f32 / rd as f32);
                let a = p as f32 * f;
                [a.cos(), a.sin()]
            })
        })
        .collect();
    let flat: Vec<f32> = table.iter().flatten().copied().collect();
    let (dx, dpos, dtab) = (dev(&x), dev(&positions), dev(&flat));
    ok(unsafe {
        ds_rope(st.raw(), p(&dx), (heads * hd) as i64, rows as i32, heads as i32, hd as i64, (hd - rd) as i32,
                rd as i32, p(&dpos), p(&dtab), 0)
    });
    st.synchronize().unwrap();
    let got = dx.download().unwrap();
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            assert_eq!(&got[base..base + hd - rd], &x[base..base + hd - rd], "non-rope dims touched");
            for i in 0..rd / 2 {
                let [c, s] = table[positions[r] as usize * (rd / 2) + i];
                let a = f32_from_bf16(x[base + hd - rd + 2 * i]);
                let b = f32_from_bf16(x[base + hd - rd + 2 * i + 1]);
                assert_eq!(f32_from_bf16(got[base + hd - rd + 2 * i]), bf(a * c - b * s));
                assert_eq!(f32_from_bf16(got[base + hd - rd + 2 * i + 1]), bf(a * s + b * c));
            }
        }
    }
}

#[test]
fn router_selects_topk_on_biased_scores_and_weights_by_raw_scores() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(31);
    let (m, e, k) = (3usize, 384usize, 6usize);
    let z: Vec<f32> = (0..m * e).map(|_| rng.normal() * 3.0).collect();
    let bias: Vec<f32> = (0..e).map(|_| rng.normal() * 0.1).collect();
    let mut keep = vec![1u8; e];
    for i in (0..e).step_by(5) {
        keep[i] = 0;
    }
    let (dz, db, dk) = (dev(&z), dev(&bias), dev(&keep));
    let (di, dw) = (DeviceBuffer::<i32>::zeroed(m * k).unwrap(), DeviceBuffer::<f32>::zeroed(m * k).unwrap());
    ok(unsafe { ds_router(st.raw(), p(&dz), p(&db), p(&dk), m as i32, e as i32, k as i32, 1.5, p(&di), p(&dw)) });
    st.synchronize().unwrap();
    let (gi, gw) = (di.download().unwrap(), dw.download().unwrap());
    for r in 0..m {
        let score: Vec<f32> = (0..e)
            .map(|j| {
                let x = z[r * e + j];
                (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt()
            })
            .collect();
        let mut order: Vec<usize> = (0..e).filter(|&j| keep[j] == 1).collect();
        order.sort_by(|&a, &b| (score[b] + bias[b]).partial_cmp(&(score[a] + bias[a])).unwrap().then(a.cmp(&b)));
        let want: Vec<i32> = order[..k].iter().map(|&j| j as i32).collect();
        assert_eq!(&gi[r * k..(r + 1) * k], &want[..], "router indices row {r}");
        let sum: f32 = order[..k].iter().map(|&j| score[j]).sum();
        for (t, &j) in order[..k].iter().enumerate() {
            let w = score[j] / (sum + 1e-20) * 1.5;
            assert!((gw[r * k + t] - w).abs() < 1e-6, "router weight");
        }
    }
}

#[test]
fn swiglu_and_moe_combine() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(37);
    let (m, n) = (3usize, 2304usize);
    let g = rng.bf16_vec(m * n, 6.0);
    let u = rng.bf16_vec(m * n, 6.0);
    let (dg, du, dh) = (dev(&g), dev(&u), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
    ok(unsafe { ds_swiglu(st.raw(), p(&dg), n as i64, p(&du), n as i64, p(&dh), n as i64, m as i32, n as i32, 10.0) });
    st.synchronize().unwrap();
    let got = dh.download().unwrap();
    for i in 0..m * n {
        let gate = f32_from_bf16(g[i]).min(10.0);
        let up = f32_from_bf16(u[i]).clamp(-10.0, 10.0);
        let want = bf(gate / (1.0 + (-gate).exp()) * up);
        let gv = f32_from_bf16(got[i]);
        assert!((gv - want).abs() <= want.abs() * 2f32.powi(-7) + 1e-30, "swiglu[{i}] {gv} vs {want}");
    }

    let (k, d) = (6usize, 5120usize);
    let parts: Vec<f32> = (0..k * m * d).map(|_| bf(rng.normal())).collect();
    let shared = rng.bf16_vec(m * d, 1.0);
    let (dp, ds, dout) = (dev(&parts), dev(&shared), DeviceBuffer::<u16>::zeroed(m * d).unwrap());
    ok(unsafe { ds_moe_combine(st.raw(), p(&dp), k as i32, m as i32, p(&ds), p(&dout), d as i32) });
    st.synchronize().unwrap();
    let got = dout.download().unwrap();
    for r in 0..m {
        for c in 0..d {
            let mut acc = parts[r * d + c];
            for kk in 1..k {
                acc += parts[(kk * m + r) * d + c];
            }
            assert_eq!(f32_from_bf16(got[r * d + c]), bf(acc + f32_from_bf16(shared[r * d + c])));
        }
    }
}

// ------------------------------------------------------------------ attention
#[test]
fn attention_decode_matches_the_sinked_softmax_reference() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(41);
    let (m, h, d, ring_size, w, c) = (2usize, 8usize, 512usize, 256usize, 128usize, 40usize);
    let q = rng.bf16_vec(m * h * d, 0.2);
    let ring = rng.bf16_vec(ring_size * d, 1.0);
    let n_ckv = 64usize;
    let ckv = rng.bf16_vec(n_ckv * d, 1.0);
    let positions = [60i32, 300]; // the first sees a partial window (masked below 0)
    let mut cidx = vec![-1i32; m * c];
    for (r, row) in cidx.chunks_mut(c).enumerate() {
        for (j, v) in row.iter_mut().enumerate().take(if r == 0 { 10 } else { 33 }) {
            *v = j as i32 * 2 % n_ckv as i32;
        }
    }
    let sink: Vec<f32> = (0..h).map(|_| rng.normal()).collect();
    let scale = (d as f64).powf(-0.5) as f32;
    let (dq, dring, dpos, dckv, dcidx, dsink) = (dev(&q), dev(&ring), dev(&positions), dev(&ckv), dev(&cidx), dev(&sink));
    let dout = DeviceBuffer::<u16>::zeroed(m * h * d).unwrap();
    ok(unsafe {
        ds_attn_decode(st.raw(), p(&dq), p(&dring), ring_size as i32, p(&dpos), w as i32, 0, p(&dckv), p(&dcidx),
                       c as i32, p(&dsink), scale, p(&dout), m as i32, h as i32, null(), 0, 0)
    });
    st.synchronize().unwrap();
    let got = dout.download().unwrap();
    for r in 0..m {
        let mut keys: Vec<&[u16]> = Vec::new();
        for j in 0..w {
            let kp = positions[r] - (w as i32 - 1 - j as i32);
            if kp >= 0 {
                let slot = kp as usize % ring_size;
                keys.push(&ring[slot * d..(slot + 1) * d]);
            }
        }
        for &j in &cidx[r * c..(r + 1) * c] {
            if j >= 0 {
                keys.push(&ckv[j as usize * d..(j as usize + 1) * d]);
            }
        }
        for hh in 0..h {
            let qr = &q[(r * h + hh) * d..(r * h + hh + 1) * d];
            let s: Vec<f64> = keys
                .iter()
                .map(|k| {
                    qr.iter().zip(k.iter()).map(|(&a, &b)| f32_from_bf16(a) as f64 * f32_from_bf16(b) as f64).sum::<f64>()
                        * scale as f64
                })
                .collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(-1e30);
            let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
            let denom = e.iter().sum::<f64>() + (sink[hh] as f64 - mx).exp();
            let want: Vec<f64> = (0..d)
                .map(|dd| keys.iter().zip(&e).map(|(k, &p)| p / denom * f32_from_bf16(k[dd]) as f64).sum())
                .collect();
            assert_bf16_close(&got[(r * h + hh) * d..(r * h + hh + 1) * d], &want, &format!("attn r{r} h{hh}"));
        }
    }
}

// ------------------------------------------------------------------ compressor / engram / misc
#[test]
fn compress_pair_and_fp4_qdq() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(43);
    let d = 512usize;
    let v: Vec<Vec<f32>> = (0..4).map(|_| (0..d).map(|_| rng.normal()).collect()).collect();
    let (dka, dsa, dkb, dsb) = (dev(&v[0]), dev(&v[1]), dev(&v[2]), dev(&v[3]));
    let dout = DeviceBuffer::<u16>::zeroed(d).unwrap();
    ok(unsafe { ds_compress_pair(st.raw(), p(&dka), p(&dsa), p(&dkb), p(&dsb), p(&dout), d as i32) });
    st.synchronize().unwrap();
    let got = dout.download().unwrap();
    for i in 0..d {
        let (a, b) = (v[1][i], v[3][i]);
        let mx = a.max(b);
        let (ea, eb) = ((a - mx).exp(), (b - mx).exp());
        let den = ea + eb;
        let want = bf(v[0][i] * (ea / den) + v[2][i] * (eb / den));
        let g = f32_from_bf16(got[i]);
        assert!((g - want).abs() <= want.abs() * 2f32.powi(-7) + 1e-30, "pair[{i}] {g} vs {want}");
    }

    let x = rng.bf16_vec(4 * d, 1.5);
    let dx = dev(&x);
    ok(unsafe { ds_fp4_qdq_e4m3(st.raw(), p(&dx), (4 * d) as i64) });
    st.synchronize().unwrap();
    let got = dx.download().unwrap();
    for (gi, grp) in x.chunks(16).enumerate() {
        let vals: Vec<f32> = grp.iter().map(|&b| f32_from_bf16(b)).collect();
        let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs())).max(6.0 * 2f32.powi(-9));
        let s = round_e4m3(amax / 6.0);
        for (i, &vv) in vals.iter().enumerate() {
            let t = (vv / s).clamp(-6.0, 6.0);
            let a = t.abs();
            let q = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
                .iter()
                .position(|&mid| a <= mid)
                .map(|i| [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0][i])
                .unwrap_or(6.0);
            let want = bf(q * t.signum() * if t == 0.0 { 0.0 } else { 1.0 } * s);
            assert_eq!(f32_from_bf16(got[gi * 16 + i]), want, "qdq group {gi} value {i}");
        }
    }
}

#[test]
fn engram_dequant_gate_embed_argmax() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(47);
    // dequant: exact
    let n_uniq = 5usize;
    let raw: Vec<u8> = (0..n_uniq)
        .flat_map(|_| {
            let mut row = rng.fp8_vec(256);
            row.extend((0..8).map(|_| 110 + (rng.next() % 20) as u8));
            row
        })
        .collect();
    let (tokens, cols) = (2usize, 24usize);
    let inv: Vec<i32> = (0..tokens * cols).map(|i| (i * 7 % n_uniq) as i32).collect();
    let (draw, dinv) = (dev(&raw), dev(&inv));
    let dout = DeviceBuffer::<u16>::zeroed(tokens * cols * 256).unwrap();
    ok(unsafe { ds_engram_dequant(st.raw(), p(&draw), p(&dinv), tokens as i32, cols as i32, p(&dout)) });
    st.synchronize().unwrap();
    let got = dout.download().unwrap();
    for (tc, &u) in inv.iter().enumerate() {
        let row = &raw[u as usize * 264..(u as usize + 1) * 264];
        for v in 0..256 {
            let want = bf(e4m3(row[v]) * 2f32.powi(row[256 + v / 32] as i32 - 127));
            assert_eq!(f32_from_bf16(got[tc * 256 + v]), want);
        }
    }

    // gate
    let (m, d, hc) = (2usize, 5120usize, 4usize);
    let h = rng.bf16_vec(m * hc * d, 1.0);
    let kv = rng.bf16_vec(m * (hc + 1) * d, 1.0);
    let weight: Vec<f32> = (0..hc * d).map(|_| rng.normal()).collect();
    let (dh, dkv, dwt) = (dev(&h), dev(&kv), dev(&weight));
    let eps = 1e-20f32;
    let dim_scale = (d as f64).powf(-0.5) as f32;
    ok(unsafe { ds_engram_gate(st.raw(), p(&dh), p(&dkv), p(&dwt), m as i32, d as i32, hc as i32, eps, dim_scale) });
    st.synchronize().unwrap();
    let got = dh.download().unwrap();
    for r in 0..m {
        for i in 0..hc {
            let hr = &h[(r * hc + i) * d..(r * hc + i + 1) * d];
            let kr = &kv[(r * (hc + 1) + i) * d..(r * (hc + 1) + i + 1) * d];
            let vr = &kv[(r * (hc + 1) + hc) * d..(r * (hc + 1) + hc + 1) * d];
            let (mut sh, mut sk, mut dot) = (0f64, 0f64, 0f64);
            for c in 0..d {
                let (a, b) = (f32_from_bf16(hr[c]) as f64, f32_from_bf16(kr[c]) as f64);
                sh += a * a;
                sk += b * b;
                dot += a * weight[i * d + c] as f64 * b;
            }
            let rstd = (sh / d as f64 + eps as f64).powf(-0.5) * (sk / d as f64 + eps as f64).powf(-0.5);
            let x = dot * rstd * dim_scale as f64;
            let g = 1.0 / (1.0 + (-(x.abs().max(1e-6).sqrt().copysign(x))).exp());
            let want: Vec<f64> =
                (0..d).map(|c| f32_from_bf16(hr[c]) as f64 + g * f32_from_bf16(vr[c]) as f64).collect();
            assert_bf16_close(&got[(r * hc + i) * d..(r * hc + i + 1) * d], &want, "engram gate");
        }
    }

    // embed: exact copy into every hc stream
    let vocab = 50usize;
    let table = rng.bf16_vec(vocab * d, 1.0);
    let ids = [3i32, 49];
    let (dt, di, dh2) = (dev(&table), dev(&ids), DeviceBuffer::<u16>::zeroed(2 * hc * d).unwrap());
    ok(unsafe { ds_embed_hc(st.raw(), p(&di), p(&dt), p(&dh2), 2, d as i32, hc as i32) });
    st.synchronize().unwrap();
    let got = dh2.download().unwrap();
    for (r, &id) in ids.iter().enumerate() {
        for c in 0..hc {
            assert_eq!(&got[(r * hc + c) * d..(r * hc + c + 1) * d], &table[id as usize * d..(id as usize + 1) * d]);
        }
    }

    // argmax: first index of the maximum
    let n = 129_280usize;
    let mut logits = rng.bf16_vec(2 * n, 1.0);
    logits[777] = bf16_from_f32(50.0);
    logits[90_000] = bf16_from_f32(50.0); // a tie: the first index wins
    logits[n + 128_000] = bf16_from_f32(60.0);
    let (dl, da) = (dev(&logits), DeviceBuffer::<i32>::zeroed(2).unwrap());
    ok(unsafe { ds_argmax_bf16(st.raw(), p(&dl), n as i64, 2, n as i32, p(&da)) });
    st.synchronize().unwrap();
    assert_eq!(da.download().unwrap(), vec![777, 128_000]);
}

// ------------------------------------------------------------------ indexer
#[test]
fn index_scores_match_the_bf16_einsum_reference() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(53);
    let (t, n, h, dd) = (3usize, 300usize, 32usize, 128usize);
    let q = rng.bf16_vec(t * h * dd, 0.3);
    let k = rng.bf16_vec(n * dd, 0.3);
    let w: Vec<f32> = (0..t * h).map(|_| rng.unif()).collect();
    let clen = [100i32, 250, 300];
    let cand: Vec<u8> = (0..t * n).map(|i| ((i / 8) % 3 != 1) as u8).collect();
    let (dq, dk, dw, dc, dcand) = (dev(&q), dev(&k), dev(&w), dev(&clen), dev(&cand));
    for masked in [false, true] {
        let dout = DeviceBuffer::<u16>::zeroed(t * n).unwrap();
        ok(unsafe {
            ds_index_scores(st.raw(), p(&dq), p(&dk), p(&dw), p(&dc), if masked { p(&dcand) } else { null() },
                            n as i64, n as i32, p(&dout), n as i64, t as i32, null(), 0)
        });
        st.synchronize().unwrap();
        let got = dout.download().unwrap();
        for r in 0..t {
            for j in 0..n {
                let g = f32_from_bf16(got[r * n + j]);
                if j as i32 >= clen[r] || (masked && cand[r * n + j] == 0) {
                    assert!(g.is_infinite() && g < 0.0, "row {r} key {j} should be masked, got {g}");
                    continue;
                }
                let (mut sum, mut mag) = (0f64, 0f64);
                for hh in 0..h {
                    let dot: f64 = (0..dd)
                        .map(|c| f32_from_bf16(q[(r * h + hh) * dd + c]) as f64 * f32_from_bf16(k[j * dd + c]) as f64)
                        .sum();
                    let term = (bf(dot as f32).max(0.0) * w[r * h + hh]) as f64;
                    sum += term;
                    mag += term.abs() + 2f64.powi(-8) * (dot.abs() * w[r * h + hh] as f64);
                }
                assert!((g as f64 - sum).abs() <= mag * 2f64.powi(-7) + 1e-6, "score[{r}][{j}] {g} vs {sum}");
            }
        }
    }
}

#[test]
fn topk_select_keeps_the_largest_lowest_index_ties_ascending() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(59);
    let (t, n, width) = (4usize, 5000usize, 1100usize);
    // coarse values force many exact ties at the threshold; some entries masked to -inf
    let s: Vec<u16> = (0..t * n)
        .map(|i| {
            if i % 7 == 3 {
                0xff80
            } else {
                bf16_from_f32(((rng.next() % 64) as f32 - 32.0) / 4.0)
            }
        })
        .collect();
    for (k, rows_n) in [(1024usize, n), (1024, 900), (37, n)] {
        let (ds, dout) = (dev(&s), DeviceBuffer::<i32>::zeroed(t * width).unwrap());
        ok(unsafe { ds_topk_select(st.raw(), p(&ds), n as i64, rows_n as i32, k as i32, p(&dout), width as i32, t as i32) });
        st.synchronize().unwrap();
        let got = dout.download().unwrap();
        for r in 0..t {
            let row = &s[r * n..r * n + rows_n];
            let mut order: Vec<usize> = (0..rows_n).filter(|&i| row[i] != 0xff80).collect();
            order.sort_by(|&a, &b| f32_from_bf16(row[b]).partial_cmp(&f32_from_bf16(row[a])).unwrap().then(a.cmp(&b)));
            let mut want: Vec<i32> = order.iter().take(k).map(|&i| i as i32).collect();
            want.sort_unstable();
            want.resize(width, -1);
            assert_eq!(&got[r * width..(r + 1) * width], &want[..], "row {r} k {k} n {rows_n}");
        }
    }
}

#[test]
fn candidate_blocks_force_the_last_block_and_set_whole_blocks() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(61);
    let (t, n, block, kb) = (2usize, 400usize, 8usize, 10usize);
    let s: Vec<u16> = (0..t * n).map(|_| bf16_from_f32(rng.normal())).collect();
    let clen = [400i32, 123];
    let nb = n / block;
    let (ds, dc) = (dev(&s), dev(&clen));
    let dbmax = DeviceBuffer::<u16>::zeroed(t * nb).unwrap();
    ok(unsafe { ds_candidate_blocks(st.raw(), p(&ds), n as i64, n as i32, block as i32, p(&dc), p(&dbmax), nb as i64, t as i32) });
    let dblocks = DeviceBuffer::<i32>::zeroed(t * kb).unwrap();
    ok(unsafe { ds_topk_select(st.raw(), p(&dbmax), nb as i64, nb as i32, kb as i32, p(&dblocks), kb as i32, t as i32) });
    let dcand = DeviceBuffer::<u8>::zeroed(t * n).unwrap();
    ok(unsafe { ds_candidate_set(st.raw(), p(&dblocks), kb as i32, block as i32, n as i32, p(&dcand), n as i64, t as i32) });
    st.synchronize().unwrap();
    let (bmax, blocks, cand) = (dbmax.download().unwrap(), dblocks.download().unwrap(), dcand.download().unwrap());
    for r in 0..t {
        let last = (clen[r] as usize - 1) / block;
        for b in 0..nb {
            let m = (b * block..(b + 1) * block).map(|i| f32_from_bf16(s[r * n + i])).fold(f32::NEG_INFINITY, f32::max);
            let want = if b == last { f32::INFINITY } else { m };
            assert_eq!(f32_from_bf16(bmax[r * nb + b]), bf(want), "bmax[{r}][{b}]");
        }
        let chosen = &blocks[r * kb..(r + 1) * kb];
        assert!(chosen.contains(&(last as i32)), "the block holding clen-1 is always kept");
        for j in 0..n {
            assert_eq!(cand[r * n + j] == 1, chosen.contains(&((j / block) as i32)), "cand[{r}][{j}]");
        }
    }
}

// ------------------------------------------------------------------ graph-safe step plumbing
#[test]
fn ring_scatter_visible_lists_and_lut() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(67);
    let (t, d, ring) = (3usize, 64usize, 8usize);
    let src = rng.bf16_vec(t * d, 1.0);
    let pos = [6i32, 7, 8];
    let (dsrc, dpos, dring) = (dev(&src), dev(&pos), DeviceBuffer::<u16>::zeroed(ring * d).unwrap());
    ok(unsafe { ds_scatter_ring(st.raw(), p(&dsrc), p(&dring), ring as i32, p(&dpos), t as i32, d as i32, null(), 0) });
    let c = 6usize;
    let (dc, dl) = (DeviceBuffer::<i32>::zeroed(t * c).unwrap(), DeviceBuffer::<i32>::zeroed(t).unwrap());
    ok(unsafe { ds_visible(st.raw(), p(&dpos), t as i32, 2, c as i32, p(&dc), p(&dl)) });
    let lut: Vec<i32> = (0..10).map(|e| if e % 3 == 0 { -1 } else { 100 + e }).collect();
    let idx = [1i32, 2, 4, 8];
    let (dlut, didx, dslots) = (dev(&lut), dev(&idx), DeviceBuffer::<i32>::zeroed(4).unwrap());
    ok(unsafe { ds_lut_slots(st.raw(), p(&didx), p(&dlut), 4, p(&dslots), null()) });
    let (dlive, dslots2) = (dev(&[2i32]), DeviceBuffer::<i32>::zeroed(4).unwrap());
    ok(unsafe { ds_lut_slots(st.raw(), p(&didx), p(&dlut), 4, p(&dslots2), p(&dlive)) });
    st.synchronize().unwrap();
    assert_eq!(dslots2.download().unwrap(), vec![101, 102, -1, -1], "pairs past `live` are filler");
    let r = dring.download().unwrap();
    for (i, &ps) in pos.iter().enumerate() {
        let slot = ps as usize % ring;
        assert_eq!(&r[slot * d..(slot + 1) * d], &src[i * d..(i + 1) * d]);
    }
    assert_eq!(dl.download().unwrap(), vec![3, 4, 4]);
    let cidx = dc.download().unwrap();
    assert_eq!(&cidx[..c], &[0, 1, 2, -1, -1, -1]);
    assert_eq!(&cidx[2 * c..], &[0, 1, 2, 3, -1, -1]);
    assert_eq!(dslots.download().unwrap(), vec![101, 102, 104, 108]);
}

#[test]
fn device_compressor_pairs_pending_and_rollback_match_the_host_logic() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(71);
    let d = 32usize;
    let row = |rng: &mut Rng| -> Vec<f32> { (0..d).map(|_| rng.normal()).collect() };
    // device state: pending (kv, sc, flag) and the rollback memory
    let (pkv, psc, flag) = (DeviceBuffer::<f32>::zeroed(d).unwrap(), DeviceBuffer::<f32>::zeroed(d).unwrap(), DeviceBuffer::<i32>::zeroed(1).unwrap());
    let tmax = 6;
    let (mkv, msc) = (DeviceBuffer::<f32>::zeroed(tmax * d).unwrap(), DeviceBuffer::<f32>::zeroed(tmax * d).unwrap());
    let (bkv, bsc, bflag, mstart) = (DeviceBuffer::<f32>::zeroed(d).unwrap(), DeviceBuffer::<f32>::zeroed(d).unwrap(),
                                     DeviceBuffer::<i32>::zeroed(1).unwrap(), DeviceBuffer::<i32>::zeroed(1).unwrap());
    let pair = |ka: &[f32], sa: &[f32], kb: &[f32], sb: &[f32]| -> Vec<f32> {
        (0..d).map(|i| {
            let mx = sa[i].max(sb[i]);
            let (ea, eb) = ((sa[i] - mx).exp(), (sb[i] - mx).exp());
            bf(ka[i] * (ea / (ea + eb)) + kb[i] * (eb / (ea + eb)))
        }).collect()
    };
    // host mirror
    let mut h_pending: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut all_kv: Vec<Vec<f32>> = Vec::new();
    let mut all_sc: Vec<Vec<f32>> = Vec::new();
    let mut s = 0usize;
    for t in [3usize, 4] {
        let kv: Vec<Vec<f32>> = (0..t).map(|_| row(&mut rng)).collect();
        let sc: Vec<Vec<f32>> = (0..t).map(|_| row(&mut rng)).collect();
        let pos: Vec<i32> = (s..s + t).map(|x| x as i32).collect();
        let (dkv, dsc, dpos) = (dev(&kv.concat()), dev(&sc.concat()), dev(&pos));
        let out = DeviceBuffer::<u16>::zeroed(t * d).unwrap();
        ok(unsafe { ds_compress_rows(st.raw(), p(&dkv), p(&dsc), p(&pkv), p(&psc), p(&dpos), p(&out), t as i32, d as i32, null()) });
        let (sf, sn, sl) = (dev(&[0i32]), dev(&[t as i32]), dev(&[0i32]));
        ok(unsafe {
            ds_pending_update(st.raw(), p(&dkv), p(&dsc), p(&dpos), p(&sf), p(&sn), p(&sl), 1, p(&pkv), p(&psc), p(&flag),
                              p(&mkv), p(&msc), p(&bkv), p(&bsc), p(&bflag), p(&mstart), d as i32, tmax as i32)
        });
        st.synchronize().unwrap();
        let got = out.download().unwrap();
        for ti in 0..t {
            let ps = s + ti;
            if ps % 2 == 1 {
                let (ka, sa) = if ti > 0 { (kv[ti - 1].clone(), sc[ti - 1].clone()) } else { h_pending.clone().unwrap() };
                let want = pair(&ka, &sa, &kv[ti], &sc[ti]);
                let g: Vec<f32> = got[ti * d..(ti + 1) * d].iter().map(|&b| f32_from_bf16(b)).collect();
                for (x, y) in g.iter().zip(&want) {
                    assert!((x - y).abs() <= y.abs() * 2f32.powi(-7) + 1e-30, "pair at position {ps}: {x} vs {y}");
                }
            }
        }
        h_pending = if (s + t) % 2 == 1 { Some((kv[t - 1].clone(), sc[t - 1].clone())) } else { None };
        assert_eq!(flag.download().unwrap()[0], h_pending.is_some() as i32);
        all_kv.extend(kv);
        all_sc.extend(sc);
        s += t;
    }
    // positions 0..7 done; the last chunk was 3..7 (start 3). Roll back to n = 5: position 4
    // is unpaired, and its inputs are row 1 of the last chunk.
    for (n, want_valid, src_pos) in [(5usize, 1, Some(4usize)), (4, 0, None), (3, 1, Some(2))] {
        let dn = dev(&[n as i32]);
        ok(unsafe {
            ds_rollback_pending(st.raw(), p(&dn), null(), 1, p(&mkv), p(&msc), p(&mstart), p(&bkv), p(&bsc), p(&bflag),
                                p(&pkv), p(&psc), p(&flag), d as i32, tmax as i32)
        });
        st.synchronize().unwrap();
        assert_eq!(flag.download().unwrap()[0], want_valid, "rollback({n}) flag");
        if let Some(ps) = src_pos {
            assert_eq!(pkv.download().unwrap(), all_kv[ps], "rollback({n}) pending kv");
            assert_eq!(psc.download().unwrap(), all_sc[ps], "rollback({n}) pending sc");
        }
    }
}

#[test]
fn greedy_accept_and_flag_wait() {
    let Some(st) = setup() else { return };
    let am = [5i32, 7, 9, 4, 4, 4];
    let block = [1i32, 5, 7, 8, 9, 9];
    let pos = [100i32, 101, 102, 103, 104, 105];
    let (dam, db, dp, dout) = (dev(&am), dev(&block), dev(&pos), DeviceBuffer::<i32>::zeroed(4).unwrap());
    ok(unsafe { ds_greedy_accept(st.raw(), p(&dam), p(&db), 6, p(&dp), p(&dout)) });
    st.synchronize().unwrap();
    // drafts 5, 7 accepted (rows 0, 1 predicted them), row 2 predicts 9 != 8 -> bonus 9
    assert_eq!(dout.download().unwrap(), vec![2, 9, 103, 102]);

    let mut flag = tidelm_cuda::PinnedBuffer::<i32>::zeroed(1).unwrap();
    let err = DeviceBuffer::<i32>::zeroed(1).unwrap();
    flag[0] = 7;
    ok(unsafe { ds_wait_flag(st.raw(), flag.raw(), 7, 1_000_000_000, p(&err)) });
    st.synchronize().unwrap();
    assert_eq!(err.download().unwrap()[0], 0, "a set flag passes at once");
    ok(unsafe { ds_wait_flag(st.raw(), flag.raw(), 8, 2_000_000, p(&err)) });
    st.synchronize().unwrap();
    assert_eq!(err.download().unwrap()[0], 1, "an unset flag times out and says so");
}

// ------------------------------------------------------------------ tensor-core skinny GEMM
#[test]
fn mma_gemm_matches_reference_and_is_row_invariant_across_m() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(73);
    let (n, k) = (200usize, 1280usize); // N not a multiple of 64 rows per block
    let w = rng.fp8_vec(n * k);
    let lds = k / 32;
    let s: Vec<u8> = (0..n.div_ceil(32) * lds).map(|_| 118 + (rng.next() % 12) as u8).collect();
    let wb = rng.bf16_vec(n * k, 0.05);
    let (dw, ds, dwb) = (dev(&w), dev(&s), dev(&wb));
    let x = rng.bf16_vec(64 * k, 1.0);
    let dx = dev(&x);
    let mut first_rows: Option<(Vec<u16>, Vec<u16>)> = None;
    for m in [1usize, 6, 17, 24, 33, 64] {
        let wf8 = |j: usize, kk: usize| e4m3(w[j * k + kk]) as f64 * e8m0_lin(s[(j / 32) * lds + kk / 32]) as f64;
        let wf16 = |j: usize, kk: usize| f32_from_bf16(wb[j * k + kk]) as f64;
        let (want8, mag8) = ref_linear_mag(&x[..m * k], wf8, m, n, k);
        let (want16, mag16) = ref_linear_mag(&x[..m * k], wf16, m, n, k);
        let (y8, y16) = (DeviceBuffer::<u16>::zeroed(m * n).unwrap(), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
        ok(unsafe {
            ds_mma_fp8(st.raw(), p(&dx), k as i64, 0, 0, p(&dw), p(&ds), lds as i64, m as i32, n as i32, k as i32, 1,
                       p(&y8), null(), n as i64, null())
        });
        ok(unsafe { ds_mma_bf16(st.raw(), p(&dx), k as i64, p(&dwb), m as i32, n as i32, k as i32, 1, p(&y16), null(), n as i64, null()) });
        st.synchronize().unwrap();
        let (g8, g16) = (y8.download().unwrap(), y16.download().unwrap());
        assert_bf16_accum(&g8, &want8, &mag8, &format!("mma fp8 M={m}"));
        assert_bf16_accum(&g16, &want16, &mag16, &format!("mma bf16 M={m}"));
        // row 0 must not depend on how many rows share the call
        match &first_rows {
            None => first_rows = Some((g8[..n].to_vec(), g16[..n].to_vec())),
            Some((a, b)) => {
                assert_eq!(&g8[..n], &a[..], "fp8 row 0 differs at M={m}");
                assert_eq!(&g16[..n], &b[..], "bf16 row 0 differs at M={m}");
            }
        }
    }
    // split-K into fp32 partials
    let (m, splits) = (24usize, 4usize);
    let want = ref_linear(&x[..m * k], |j, kk| e4m3(w[j * k + kk]) as f64 * e8m0_lin(s[(j / 32) * lds + kk / 32]) as f64, m, n, k);
    let (dpart, dy) = (DeviceBuffer::<f32>::zeroed(splits * m * n).unwrap(), DeviceBuffer::<f32>::zeroed(m * n).unwrap());
    ok(unsafe {
        ds_mma_fp8(st.raw(), p(&dx), k as i64, 0, 0, p(&dw), p(&ds), lds as i64, m as i32, n as i32, k as i32, splits as i32,
                   null(), null(), n as i64, p(&dpart))
    });
    ok(unsafe { ds_splits_reduce(st.raw(), p(&dpart), splits as i32, m as i32, n as i32, null(), p(&dy), n as i64) });
    st.synchronize().unwrap();
    assert_f32_close(&dy.download().unwrap(), &want, 1e-5, "mma split-K");
}

#[test]
fn mma_grouped_reads_each_groups_slice() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(79);
    // rows per group must be a multiple of the block's rows (the model's wo_a groups are 1024)
    let (g, r, k, m) = (4usize, 256usize, 512usize, 20usize);
    let n = g * r;
    let w = rng.fp8_vec(n * k);
    let lds = k / 32;
    let s: Vec<u8> = (0..(n / 32) * lds).map(|_| 120 + (rng.next() % 8) as u8).collect();
    let x = rng.bf16_vec(m * g * k, 1.0);
    let (mut want, mut mag) = (vec![0f64; m * n], vec![0f64; m * n]);
    for i in 0..m {
        for j in 0..n {
            let grp = j / r;
            for kk in 0..k {
                let t = f32_from_bf16(x[(i * g + grp) * k + kk]) as f64 * e4m3(w[j * k + kk]) as f64
                    * e8m0_lin(s[(j / 32) * lds + kk / 32]) as f64;
                want[i * n + j] += t;
                mag[i * n + j] += t.abs();
            }
        }
    }
    let (dx, dw, ds, dy) = (dev(&x), dev(&w), dev(&s), DeviceBuffer::<u16>::zeroed(m * n).unwrap());
    ok(unsafe {
        ds_mma_fp8(st.raw(), p(&dx), (g * k) as i64, k as i64, r as i32, p(&dw), p(&ds), lds as i64, m as i32, n as i32,
                   k as i32, 1, p(&dy), null(), n as i64, null())
    });
    st.synchronize().unwrap();
    assert_bf16_accum(&dy.download().unwrap(), &want, &mag, "mma grouped");
}


// ------------------------------------------------------------------ lanes
/// Two lanes' rows in one call must compute exactly what each lane computes alone, and
/// write only its own lane's caches.
#[test]
fn lane_rows_read_and_write_only_their_own_caches() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(83);
    let (h, d, ring, w, n_ckv) = (8usize, 512usize, 256usize, 128usize, 32usize);
    let lanes = 2usize;
    let rings = rng.bf16_vec(lanes * ring * d, 1.0);
    let ckvs = rng.bf16_vec(lanes * n_ckv * d, 1.0);
    let sink: Vec<f32> = (0..h).map(|_| rng.normal()).collect();
    let scale = (d as f64).powf(-0.5) as f32;
    // rows: lane 1 at position 40, lane 0 at 7 and 8 (lane order need not follow row order)
    let row_lane = [1i32, 0, 0];
    let pos = [40i32, 7, 8];
    let c = 4usize;
    let cidx: Vec<i32> = vec![0, 1, 2, 3, 0, 1, -1, -1, 0, 1, 2, -1];
    let q = rng.bf16_vec(3 * h * d, 0.2);
    let (dq, drings, dckvs, dsink, dl, dp, dc) = (dev(&q), dev(&rings), dev(&ckvs), dev(&sink), dev(&row_lane), dev(&pos), dev(&cidx));
    let both = DeviceBuffer::<u16>::zeroed(3 * h * d).unwrap();
    ok(unsafe {
        ds_attn_decode(st.raw(), p(&dq), p(&drings), ring as i32, p(&dp), w as i32, 0, p(&dckvs), p(&dc), c as i32, p(&dsink),
                       scale, p(&both), 3, h as i32, p(&dl), (ring * d) as i64, (n_ckv * d) as i64)
    });
    st.synchronize().unwrap();
    let both = both.download().unwrap();
    for (row, &ln) in row_lane.iter().enumerate() {
        let ln = ln as usize;
        let (dq1, dr1, dk1, dp1, dc1) = (dev(&q[row * h * d..(row + 1) * h * d]), dev(&rings[ln * ring * d..(ln + 1) * ring * d]),
            dev(&ckvs[ln * n_ckv * d..(ln + 1) * n_ckv * d]), dev(&pos[row..row + 1]), dev(&cidx[row * c..(row + 1) * c]));
        let one = DeviceBuffer::<u16>::zeroed(h * d).unwrap();
        ok(unsafe {
            ds_attn_decode(st.raw(), p(&dq1), p(&dr1), ring as i32, p(&dp1), w as i32, 0, p(&dk1), p(&dc1), c as i32, p(&dsink),
                           scale, p(&one), 1, h as i32, null(), 0, 0)
        });
        st.synchronize().unwrap();
        assert_eq!(&both[row * h * d..(row + 1) * h * d], &one.download().unwrap()[..], "attention row {row} (lane {ln})");
    }

    // ring scatter: each row lands in its own lane's ring only
    let src = rng.bf16_vec(3 * d, 1.0);
    let (dsrc, dring2) = (dev(&src), DeviceBuffer::<u16>::zeroed(lanes * ring * d).unwrap());
    ok(unsafe { ds_scatter_ring(st.raw(), p(&dsrc), p(&dring2), ring as i32, p(&dp), 3, d as i32, p(&dl), (ring * d) as i64) });
    st.synchronize().unwrap();
    let r2 = dring2.download().unwrap();
    for (row, (&ln, &ps)) in row_lane.iter().zip(&pos).enumerate() {
        let at = ln as usize * ring * d + (ps as usize % ring) * d;
        assert_eq!(&r2[at..at + d], &src[row * d..(row + 1) * d]);
    }
    let written: usize = r2.chunks(d).filter(|r| r.iter().any(|&x| x != 0)).count();
    assert_eq!(written, 3, "nothing else in either ring is touched");

    // compressor: a lane's first row pairs with that lane's pending, never the other lane's
    let dd = 32usize;
    let kv: Vec<f32> = (0..3 * dd).map(|_| rng.normal()).collect();
    let sc: Vec<f32> = (0..3 * dd).map(|_| rng.normal()).collect();
    let pkv: Vec<f32> = (0..lanes * dd).map(|_| rng.normal()).collect();
    let psc: Vec<f32> = (0..lanes * dd).map(|_| rng.normal()).collect();
    // lane 1 at position 41 (odd: pairs with its pending), lane 0 at 7 (odd: pending) and 8
    let pos_c = [41i32, 7, 8];
    let (dkv, dsc, dpkv, dpsc, dpc) = (dev(&kv), dev(&sc), dev(&pkv), dev(&psc), dev(&pos_c));
    let out = DeviceBuffer::<u16>::zeroed(3 * dd).unwrap();
    ok(unsafe { ds_compress_rows(st.raw(), p(&dkv), p(&dsc), p(&dpkv), p(&dpsc), p(&dpc), p(&out), 3, dd as i32, p(&dl)) });
    st.synchronize().unwrap();
    let got = out.download().unwrap();
    let pair = |ka: &[f32], sa: &[f32], kb: &[f32], sb: &[f32]| -> Vec<u16> {
        (0..dd).map(|i| {
            let mx = sa[i].max(sb[i]);
            let (ea, eb) = ((sa[i] - mx).exp(), (sb[i] - mx).exp());
            bf16_from_f32(ka[i] * (ea / (ea + eb)) + kb[i] * (eb / (ea + eb)))
        }).collect()
    };
    let close = |a: &[u16], b: &[u16]| a.iter().zip(b).all(|(&x, &y)| (f32_from_bf16(x) - f32_from_bf16(y)).abs() <= f32_from_bf16(y).abs() * 2f32.powi(-7) + 1e-30);
    assert!(close(&got[..dd], &pair(&pkv[dd..], &psc[dd..], &kv[..dd], &sc[..dd])), "lane 1 pairs with its own pending");
    assert!(close(&got[dd..2 * dd], &pair(&pkv[..dd], &psc[..dd], &kv[dd..2 * dd], &sc[dd..2 * dd])), "lane 0 pairs with its own pending");
}

/// Per-lane compressor state: segments of two lanes in one launch (plus an empty slot) leave
/// each lane's pending, rollback memory and flag exactly as two separate launches would, and a
/// per-entry rollback restores each lane from its own memory.
#[test]
fn pending_update_and_rollback_per_lane_segment() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(89);
    let (d, lanes, mem_rows) = (32usize, 3usize, 8usize);
    let z = |n: usize| DeviceBuffer::<f32>::zeroed(n).unwrap();
    let zi = |n: usize| DeviceBuffer::<i32>::zeroed(n).unwrap();
    struct St { pkv: DeviceBuffer<f32>, psc: DeviceBuffer<f32>, flag: DeviceBuffer<i32>, mkv: DeviceBuffer<f32>,
                msc: DeviceBuffer<f32>, bkv: DeviceBuffer<f32>, bsc: DeviceBuffer<f32>, bflag: DeviceBuffer<i32>,
                mstart: DeviceBuffer<i32> }
    let mk = || St { pkv: z(lanes * d), psc: z(lanes * d), flag: zi(lanes), mkv: z(lanes * mem_rows * d),
                     msc: z(lanes * mem_rows * d), bkv: z(lanes * d), bsc: z(lanes * d), bflag: zi(lanes), mstart: zi(lanes) };
    let (both, apart) = (mk(), mk());
    // rows 0..3: lane 2 at positions 10..13 (ends unpaired); rows 3..5: lane 0 at 4..6 (paired)
    let kv: Vec<f32> = (0..5 * d).map(|_| rng.normal()).collect();
    let sc: Vec<f32> = (0..5 * d).map(|_| rng.normal()).collect();
    let pos = [10i32, 11, 12, 4, 5];
    let (dkv, dsc, dpos) = (dev(&kv), dev(&sc), dev(&pos));
    let update = |s: &St, first: &[i32], count: &[i32], lane: &[i32]| {
        let (f, c, l) = (dev(first), dev(count), dev(lane));
        ok(unsafe {
            ds_pending_update(st.raw(), p(&dkv), p(&dsc), p(&dpos), p(&f), p(&c), p(&l), first.len() as i32, p(&s.pkv),
                              p(&s.psc), p(&s.flag), p(&s.mkv), p(&s.msc), p(&s.bkv), p(&s.bsc), p(&s.bflag), p(&s.mstart),
                              d as i32, mem_rows as i32)
        });
        st.synchronize().unwrap();
    };
    update(&both, &[0, 3, 0], &[3, 2, 0], &[2, 0, 1]);
    update(&apart, &[0], &[3], &[2]);
    update(&apart, &[3], &[2], &[0]);
    let dl = |b: &DeviceBuffer<f32>| b.download().unwrap();
    let dli = |b: &DeviceBuffer<i32>| b.download().unwrap();
    assert_eq!(dli(&both.flag), vec![0, 0, 1], "lane 2 ends unpaired, lane 0 paired, lane 1 untouched");
    assert_eq!(dli(&both.mstart), vec![4, 0, 10]);
    assert_eq!(&dl(&both.pkv)[2 * d..], &kv[2 * d..3 * d], "lane 2's pending is its last row");
    for (a, b) in [(&both.pkv, &apart.pkv), (&both.psc, &apart.psc), (&both.mkv, &apart.mkv), (&both.msc, &apart.msc),
                   (&both.bkv, &apart.bkv), (&both.bsc, &apart.bsc)] {
        assert_eq!(dl(a), dl(b));
    }
    assert_eq!(dli(&both.flag), dli(&apart.flag));
    assert_eq!(dli(&both.bflag), dli(&apart.bflag));
    // roll lane 2 back to 12 (even: no pending) and lane 0 back to 5 (odd: position 4 = its row 0)
    let (dn, dlanes) = (dev(&[12i32, 5]), dev(&[2i32, 0]));
    ok(unsafe {
        ds_rollback_pending(st.raw(), p(&dn), p(&dlanes), 2, p(&both.mkv), p(&both.msc), p(&both.mstart), p(&both.bkv),
                            p(&both.bsc), p(&both.bflag), p(&both.pkv), p(&both.psc), p(&both.flag), d as i32, mem_rows as i32)
    });
    st.synchronize().unwrap();
    assert_eq!(dli(&both.flag), vec![1, 0, 0]);
    assert_eq!(&dl(&both.pkv)[..d], &kv[3 * d..4 * d], "lane 0's pending restored from its own memory");
    assert_eq!(&dl(&both.psc)[..d], &sc[3 * d..4 * d]);
}
