//! GPU tests for cuda/inkling.cu (Inkling 2's short convolutions, block attention residual and
//! grouped-query attention): each kernel against a CPU transcription of transformers'
//! modeling_inkling2.py, a sequence run in two passes against one pass (the rings), and the prefill
//! kernels against the decode ones on the same rows.

use std::os::raw::c_void;

use tidelm_cuda::{gpu_available, set_device, DeviceBuffer, Stream};
use tidelm_kernels::inkling::*;

const D: usize = 128;
const G: usize = 6;
const REL: usize = 16;

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
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
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
#[track_caller]
fn ok(rc: i32) {
    assert_eq!(rc, 0, "kernel launch failed with CUDA error {rc}");
    let rc = unsafe { tidelm_cuda::ffi::cudaDeviceSynchronize() };
    assert_eq!(rc, 0, "kernel failed with CUDA error {rc}");
}
fn fl(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&h| f32_from_bf16(h)).collect()
}
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let (mut d, mut n) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        d += ((g - w) as f64).powi(2);
        n += (w as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt() as f32
}

/// A pass's tables: one lane group (`lane`, rows 0.., first position p0, cache-row offset koff).
fn tables(lane: usize, rows: usize, p0: usize, koff: usize) -> (DeviceBuffer<i32>, DeviceBuffer<i32>) {
    (dev(&[0i32, rows as i32, p0 as i32, koff as i32]), dev(&[lane as i32, 0, 0, 0]))
}

// ------------------------------------------------------------------ convolutions
/// y over the whole sequence x [T][c]: (sum_lag w(lag) x(t - lag) + x(t)) / div, oldest tap first.
fn conv_ref(x: &[f32], t: usize, c: usize, w: &dyn Fn(usize, usize, usize) -> f32, div: f32) -> Vec<f32> {
    let mut y = vec![0f32; t * c];
    for i in 0..t {
        for ch in 0..c {
            let mut acc = 0f32;
            for lag in (0..4).rev() {
                if i >= lag {
                    acc = w(i, ch, lag).mul_add(x[(i - lag) * c + ch], acc);
                }
            }
            y[i * c + ch] = bf((acc + x[i * c + ch]) / div);
        }
    }
    y
}

#[test]
fn convolutions_in_two_passes_are_the_sequence_in_one() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0x1ce_c0de);
    let (t, c, lanes, lane) = (41usize, 2 * D, 2usize, 1usize);
    let x = rng.bf16_vec(t * c, 1.0);
    let xf = fl(&x);
    // static taps [c][4] in conv1d's order; dynamic: static [4][c] by lag and per-token taps [t][heads x 4]
    let ws = rng.bf16_vec(c * 4, 0.5);
    let wl = rng.bf16_vec(4 * c, 0.5);
    let wd = rng.bf16_vec(t * (c / D) * 4, 0.5);
    let (wsf, wlf, wdf) = (fl(&ws), fl(&wl), fl(&wd));
    let want_s = conv_ref(&xf, t, c, &|_, ch, lag| wsf[ch * 4 + (3 - lag)], 5f32.sqrt());
    let want_d = conv_ref(&xf, t, c, &|i, ch, lag| wlf[lag * c + ch] + wdf[i * (c / D) * 4 + (ch / D) * 4 + lag], 3.0);
    for (dynamic, want) in [(false, &want_s), (true, &want_d)] {
        for split in [t, 17, 1] {
            // rows 0 .. split, then split .. t (a pass of everything when split == t)
            let ring = DeviceBuffer::<u16>::zeroed(lanes * CONV_RING * c).unwrap();
            let out = DeviceBuffer::<u16>::zeroed(t * c).unwrap();
            let mut r0 = 0;
            for n in [split, t - split] {
                if n == 0 {
                    continue;
                }
                let (lt, meta) = tables(lane, n, r0, 0);
                let xd = dev(&x[r0 * c..(r0 + n) * c]);
                let (w, wmode, dw, ldd, div) = if dynamic {
                    (dev(&wl), 1, dev(&wd[r0 * (c / D) * 4..(r0 + n) * (c / D) * 4]), (c / D) * 4, 3.0)
                } else {
                    (dev(&ws), 0, dev(&[0u16]), 0, 5f32.sqrt())
                };
                unsafe {
                    ok(ink_conv(st.raw(), p(&xd), c as i64, c as i32, p(&w), wmode, p(&dw), ldd as i64, p(&ring), (CONV_RING * c) as i64,
                                p(&lt), 1, p(&meta), n as i32, div, 0, out.ptr_at(r0 * c), c as i64, null(), null()));
                    ok(ink_conv_state(st.raw(), p(&xd), c as i64, c as i32, p(&ring), (CONV_RING * c) as i64, p(&lt), 1, p(&meta)));
                }
                r0 += n;
            }
            let got = fl(&out.download().unwrap());
            let e = rel_err(&got, want);
            eprintln!("conv dynamic {dynamic} split {split}: rel err {e:.2e}");
            assert!(e < 1e-6, "conv dynamic {dynamic} split {split}: rel err {e}");
            // the other lane's ring is untouched
            assert!(ring.download().unwrap()[..CONV_RING * c].iter().all(|&v| v == 0));
        }
    }
}

#[test]
fn conv_output_modes_form_the_residual() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xfeed_5eed);
    let (t, c) = (5usize, 512usize);
    let x = rng.bf16_vec(t * c, 1.0);
    let w = rng.bf16_vec(c * 4, 0.5);
    let h = rng.bf16_vec(t * c, 1.0);
    let (xf, wf, hf) = (fl(&x), fl(&w), fl(&h));
    let y = conv_ref(&xf, t, c, &|_, ch, lag| wf[ch * 4 + (3 - lag)], 5f32.sqrt());
    let (lt, meta) = tables(0, t, 0, 0);
    let ring = DeviceBuffer::<u16>::zeroed(CONV_RING * c).unwrap();
    let run = |omode: i32, res: &DeviceBuffer<f32>, hd: &DeviceBuffer<u16>| unsafe {
        ok(ink_conv(st.raw(), p(&dev(&x)), c as i64, c as i32, p(&dev(&w)), 0, null(), 0, p(&ring), (CONV_RING * c) as i64, p(&lt), 1,
                    p(&meta), t as i32, 5f32.sqrt(), omode, null(), 0, p(res), p(hd)));
    };
    let res = DeviceBuffer::<f32>::zeroed(t * c).unwrap();
    let hd = dev(&h);
    run(1, &res, &hd);
    assert_eq!(res.download().unwrap(), y, "omode 1: res = y");
    run(2, &res, &hd);
    let want2: Vec<f32> = hf.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(res.download().unwrap(), want2, "omode 2: res = h + y");
    run(3, &res, &hd);
    let want3: Vec<f32> = want2.iter().zip(&y).map(|(a, b)| bf(a + b)).collect();
    assert_eq!(fl(&hd.download().unwrap()), want3, "omode 3: h = bf16(res + y)");
}

// ------------------------------------------------------------------ the block attention residual
#[test]
fn mix_matches_the_reference() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xb10c_5);
    let (t, h, nbmax) = (3usize, 4608usize, 8usize);
    for (nb, nq, cur_f32) in [(0usize, 1usize, true), (3, 3, false), (8, 1, false), (2, 1, true)] {
        let blocks = rng.bf16_vec(t * nbmax * h, 2.0);
        let cur_b = rng.bf16_vec(t * h, 3.0);
        let cur_f = rng.f32_vec(t * h, 3.0);
        let q = rng.f32_vec(nq * h, 0.5);
        let nw = rng.bf16_vec(nq * h, 1.0);
        let out = DeviceBuffer::<u16>::zeroed(nq * t * h).unwrap();
        let (cb, cf) = (dev(&cur_b), dev(&cur_f));
        let cur_ptr = if cur_f32 { p(&cf) } else { p(&cb) };
        unsafe {
            ok(ink_mix(st.raw(), p(&dev(&blocks)), (nbmax * h) as i64, nb as i32, cur_ptr, cur_f32 as i32, p(&dev(&q)), nq as i32,
                       p(&dev(&nw)), 1e-6, p(&out), (t * h) as i64, t as i32, h as i32));
        }
        let got = fl(&out.download().unwrap());
        let mut want = vec![0f32; nq * t * h];
        for k in 0..nq {
            for r in 0..t {
                // the states, each to unit RMS
                let mut s: Vec<Vec<f32>> = (0..nb).map(|i| fl(&blocks[(r * nbmax + i) * h..(r * nbmax + i + 1) * h])).collect();
                s.push(if cur_f32 { cur_f[r * h..(r + 1) * h].to_vec() } else { fl(&cur_b[r * h..(r + 1) * h]) });
                for v in s.iter_mut() {
                    let ms = v.iter().map(|x| (x * x) as f64).sum::<f64>() / h as f64;
                    let rs = 1.0 / (ms as f32 + 1e-6).sqrt();
                    v.iter_mut().for_each(|x| *x *= rs);
                }
                let lg: Vec<f32> = s.iter().map(|v| v.iter().zip(&q[k * h..]).map(|(a, b)| (a * b) as f64).sum::<f64>() as f32
                                                  / (h as f32).sqrt()).collect();
                let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = lg.iter().map(|l| (l - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                let mix: Vec<f32> = (0..h).map(|c| s.iter().zip(&e).map(|(v, w)| v[c] * w / sum).sum()).collect();
                let ms = mix.iter().map(|x| (x * x) as f64).sum::<f64>() / h as f64;
                let rn = 1.0 / (ms as f32 + 1e-6).sqrt();
                for c in 0..h {
                    want[(k * t + r) * h + c] = bf(f32_from_bf16(nw[k * h + c]) * (mix[c] * rn));
                }
            }
        }
        let e = rel_err(&got, &want);
        eprintln!("mix nb {nb} nq {nq} fp32 {cur_f32}: rel err {e:.2e}");
        assert!(e < 2e-3, "mix nb {nb} nq {nq}: rel err {e}");
    }
}

// ------------------------------------------------------------------ attention
struct Attn {
    hl: usize,
    t: usize,
    p0: usize,
    ldp: usize,
    /// proj rows: q at 0, r at hl x 128, gate after, sink last
    proj: Vec<u16>,
    proj_t: Vec<u16>,
    extent: usize,
}

impl Attn {
    fn offs(&self) -> (usize, usize, usize, usize) {
        let (q, r) = (0, self.hl * D);
        let g = r + self.hl * REL;
        (q, r, g, g + self.hl * D)
    }

    fn new(rng: &mut Rng, kl: usize, t: usize, p0: usize, extent: usize) -> Self {
        let hl = kl * G;
        let ldp = 2 * hl * D + hl * REL + hl;
        Attn { hl, t, p0, ldp, proj: rng.bf16_vec(t * ldp, 1.0), proj_t: rng.bf16_vec(extent * REL, 0.3), extent }
    }

    /// The reference's output [t][hl][128] from queries qc, relative states rc and the cache rows
    /// `kv(position)` -> [kl][256].
    fn reference(&self, qc: &[f32], rc: &[f32], kv: &dyn Fn(usize) -> Vec<f32>, window: usize) -> Vec<f32> {
        let (_, _, goff, soff) = self.offs();
        let pf = fl(&self.proj);
        let pt = fl(&self.proj_t);
        let mut out = vec![0f32; self.t * self.hl * D];
        for i in 0..self.t {
            let pos = self.p0 + i;
            let lo = if window > 0 { (pos + 1).saturating_sub(window) } else { 0 };
            let rows: Vec<Vec<f32>> = (lo..=pos).map(kv).collect();
            for hq in 0..self.hl {
                let g = hq / G;
                let q = &qc[(i * self.hl + hq) * D..][..D];
                let r = &rc[(i * self.hl + hq) * REL..][..REL];
                let mut lg: Vec<f32> = rows.iter().enumerate().map(|(j, row)| {
                    let d = pos - (lo + j);
                    let dot: f32 = q.iter().zip(&row[g * 2 * D..][..D]).map(|(a, b)| a * b).sum();
                    let bias: f32 = if d < self.extent { r.iter().zip(&pt[d * REL..][..REL]).map(|(a, b)| a * b).sum() } else { 0.0 };
                    dot / (D as f32).sqrt() + bias
                }).collect();
                lg.push(pf[i * self.ldp + soff + hq]);
                let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = lg.iter().map(|l| (l - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..D {
                    let o: f32 = rows.iter().zip(&e).map(|(row, w)| row[g * 2 * D + D + d] * w / sum).sum();
                    out[(i * self.hl + hq) * D + d] = bf(bf(o) * sigmoid(pf[i * self.ldp + goff + hq * D + d]));
                }
            }
        }
        out
    }
}

#[test]
fn heads_scale_clip_and_fill_the_cache() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0x4ead_5);
    let (kl, t, p0, ring, lanes, lane) = (2usize, 4usize, 127_998usize, 2048usize, 2usize, 1usize);
    let a = Attn::new(&mut rng, kl, t, p0, 1024);
    let hl = a.hl;
    let (qoff, roff, _, _) = a.offs();
    let qs = rng.bf16_vec(hl * D, 1.0);
    let rs = rng.bf16_vec(hl * REL, 1.0);
    // large rows so some vectors clip and some do not
    let kx: Vec<u16> = (0..t * kl * D).map(|i| bf16_from_f32(rng.normal() * if i / D % 2 == 0 { 3.0 } else { 0.5 })).collect();
    let vx = rng.bf16_vec(t * kl * D, 2.0);
    let (qclip, rclip, eps) = (22.627_417f32, 8.0f32, 1e-6f32);
    for (alpha, flat) in [(0.0f32, false), (0.1, true)] {
        let rowk = kl * 2 * D;
        let (ringp, lane_rows, koff, cache_len) = if flat { (0, 0, 700usize, (700 + p0 + t) * rowk) } else { (ring, ring, 0, lanes * ring * rowk) };
        let cache = DeviceBuffer::<u16>::zeroed(cache_len).unwrap();
        let (lt, meta) = tables(lane, t, p0, koff);
        let qc = DeviceBuffer::<u16>::zeroed(t * hl * D).unwrap();
        let rc = DeviceBuffer::<u16>::zeroed(t * hl * REL).unwrap();
        unsafe {
            ok(ink_heads(st.raw(), p(&dev(&a.proj)), a.ldp as i64, qoff as i32, roff as i32, p(&dev(&qs)), p(&dev(&rs)), p(&dev(&kx)),
                         p(&dev(&vx)), hl as i32, kl as i32, qclip, rclip, eps, alpha, 128_000.0, p(&lt), 1, p(&meta), ringp as i32,
                         lane_rows as i64, t as i32, p(&qc), p(&rc), p(&cache)));
        }
        let (pf, qsf, rsf, kf, vf) = (fl(&a.proj), fl(&qs), fl(&rs), fl(&kx), fl(&vx));
        let clip = |v: &[f32], thr: f32| -> Vec<f32> {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x * (thr / (n + 1e-6)).min(1.0)).collect()
        };
        let (mut wq, mut wr, mut wk, mut wv) = (vec![], vec![], vec![], vec![]);
        for i in 0..t {
            let pos = p0 + i;
            let tau = if alpha > 0.0 { 1.0 + alpha * ((pos + 1) as f32 / 128_000.0).max(1.0).ln() } else { 1.0 };
            for hq in 0..hl {
                let q: Vec<f32> = (0..D).map(|d| pf[i * a.ldp + qoff + hq * D + d] * qsf[hq * D + d]).collect();
                wq.extend(clip(&q, qclip).into_iter().map(|x| if alpha > 0.0 { bf(bf(x) * tau) } else { bf(x) }));
                let r: Vec<f32> = (0..REL).map(|d| pf[i * a.ldp + roff + hq * REL + d] * rsf[hq * REL + d]).collect();
                wr.extend(clip(&r, rclip).into_iter().map(|x| if alpha > 0.0 { bf(bf(x) * tau) } else { bf(x) }));
            }
            for g in 0..kl {
                wk.extend(clip(&kf[(i * kl + g) * D..][..D], qclip).into_iter().map(bf));
                let v = &vf[(i * kl + g) * D..][..D];
                let rsq = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / D as f32 + eps).sqrt();
                wv.extend(v.iter().map(|x| bf(x * rsq)));
            }
        }
        let (eq, er) = (rel_err(&fl(&qc.download().unwrap()), &wq), rel_err(&fl(&rc.download().unwrap()), &wr));
        let c = fl(&cache.download().unwrap());
        let (mut gk, mut gv) = (vec![], vec![]);
        for i in 0..t {
            let row = if flat { koff + p0 + i } else { lane * ring + ((p0 + i) & (ring - 1)) };
            for g in 0..kl {
                gk.extend_from_slice(&c[row * rowk + g * 2 * D..][..D]);
                gv.extend_from_slice(&c[row * rowk + g * 2 * D + D..][..D]);
            }
        }
        let (ek, ev) = (rel_err(&gk, &wk), rel_err(&gv, &wv));
        eprintln!("heads alpha {alpha} flat {flat}: q {eq:.2e} r {er:.2e} k {ek:.2e} v {ev:.2e}");
        assert!(eq < 1e-5 && er < 1e-5 && ek < 1e-5 && ev < 1e-5, "heads: {eq} {er} {ek} {ev}");
    }
}

#[test]
fn decode_attention_matches_the_reference_and_prefill_matches_decode() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xa77e_17);
    let kl = 2usize;
    let rowk = kl * 2 * D;
    // (window, ring, first position, rows, splits): a window layer before and past its window (the
    // ring wrapping), a full layer on a context past the bias's extent
    for (window, ring, p0, t, ns) in [(1024usize, 2048usize, 300usize, 3usize, 4usize), (1024, 2048, 5000, 5, 4), (0, 0, 2500, 3, 64)] {
        let a = Attn::new(&mut rng, kl, t, p0, 1024);
        let hl = a.hl;
        let (_, _, goff, soff) = a.offs();
        let (lanes, lane) = (2usize, 1usize);
        let n_pos = p0 + t;
        // the cache: every position's row (a window layer keeps the last `ring` of them)
        let all = rng.bf16_vec(n_pos * rowk, 1.0);
        let koff = 64usize;
        let cache_h: Vec<u16> = if ring > 0 {
            let mut c = vec![0u16; lanes * ring * rowk];
            for pos in n_pos.saturating_sub(ring)..n_pos {
                let row = lane * ring + (pos & (ring - 1));
                c[row * rowk..(row + 1) * rowk].copy_from_slice(&all[pos * rowk..(pos + 1) * rowk]);
            }
            c
        } else {
            let mut c = vec![0u16; (koff + n_pos) * rowk];
            c[koff * rowk..].copy_from_slice(&all);
            c
        };
        let cache = dev(&cache_h);
        let qc = rng.bf16_vec(t * hl * D, 2.0);
        let rc = rng.bf16_vec(t * hl * REL, 1.0);
        let (lt, meta) = tables(lane, t, p0, if ring > 0 { 0 } else { koff });
        let (proj, proj_t, qcd, rcd) = (dev(&a.proj), dev(&a.proj_t), dev(&qc), dev(&rc));
        let pm = DeviceBuffer::<f32>::zeroed(t * kl * ns * G).unwrap();
        let pl = DeviceBuffer::<f32>::zeroed(t * kl * ns * G).unwrap();
        let pacc = DeviceBuffer::<f32>::zeroed(t * kl * ns * G * D).unwrap();
        let o = DeviceBuffer::<u16>::zeroed(t * hl * D).unwrap();
        unsafe {
            ok(ink_attn_part(st.raw(), p(&qcd), p(&rcd), p(&proj_t), 1024, p(&cache), ring as i32, ring as i64, window as i32, p(&lt), 1,
                             p(&meta), hl as i32, kl as i32, ns as i32, t as i32, p(&pm), p(&pl), p(&pacc)));
            ok(ink_attn_combine(st.raw(), p(&pm), p(&pl), p(&pacc), p(&proj), a.ldp as i64, soff as i32, goff as i32, hl as i32,
                                kl as i32, ns as i32, t as i32, p(&o)));
        }
        let allf = fl(&all);
        let want = a.reference(&fl(&qc), &fl(&rc), &|pos| allf[pos * rowk..(pos + 1) * rowk].to_vec(), window);
        let got = fl(&o.download().unwrap());
        let e = rel_err(&got, &want);
        eprintln!("decode attention window {window} p0 {p0}: rel err {e:.2e}");
        assert!(e < 3e-3, "decode attention window {window} p0 {p0}: rel err {e}");

        // prefill: the same rows through the window copy, host scores (q . k), the softmax kernel,
        // host weights x values, the gate kernel
        let lo = if window > 0 { (p0 + 1).saturating_sub(window) } else { 0 };
        let sk = p0 + t - lo;
        let kvs = DeviceBuffer::<u16>::zeroed(sk * rowk).unwrap();
        unsafe {
            ok(ink_kv_window(st.raw(), p(&cache), ring as i32, ring as i64, p(&lt), p(&meta), 0, lo as i32, sk as i32, kl as i32, p(&kvs)));
        }
        let kv = kvs.download().unwrap();
        assert_eq!(&kv[..], &all[lo * rowk..(lo + sk) * rowk], "the window copy");
        let (qf, kvf) = (fl(&qc), fl(&kv));
        let mut sc = vec![0f32; hl * t * sk];
        for hq in 0..hl {
            for i in 0..t {
                for j in 0..sk {
                    sc[(hq * t + i) * sk + j] = qf[(i * hl + hq) * D..][..D].iter().zip(&kvf[j * rowk + (hq / G) * 2 * D..][..D])
                        .map(|(x, y)| x * y).sum();
                }
            }
        }
        let w = DeviceBuffer::<u16>::zeroed(hl * t * sk).unwrap();
        unsafe {
            ok(ink_pf_softmax(st.raw(), p(&dev(&sc)), p(&rcd), p(&proj_t), 1024, p(&proj), a.ldp as i64, soff as i32, hl as i32, 0,
                              t as i32, sk as i32, lo as i32, p0 as i32, window as i32, p(&w)));
        }
        let wf = fl(&w.download().unwrap());
        let mut ov = vec![0u16; hl * t * D];
        for hq in 0..hl {
            for i in 0..t {
                for d in 0..D {
                    let s: f32 = (0..sk).map(|j| wf[(hq * t + i) * sk + j] * kvf[j * rowk + (hq / G) * 2 * D + D + d]).sum();
                    ov[(hq * t + i) * D + d] = bf16_from_f32(s);
                }
            }
        }
        let o2 = DeviceBuffer::<u16>::zeroed(t * hl * D).unwrap();
        unsafe {
            ok(ink_pf_gate(st.raw(), p(&dev(&ov)), p(&proj), a.ldp as i64, goff as i32, hl as i32, 0, t as i32, p(&o2)));
        }
        let e2 = rel_err(&fl(&o2.download().unwrap()), &want);
        eprintln!("prefill attention window {window} p0 {p0}: rel err {e2:.2e}");
        assert!(e2 < 1e-2, "prefill attention window {window} p0 {p0}: rel err {e2}");
    }
}

#[test]
fn fused_prefill_attention_matches_the_reference() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xf1a5_4);
    let kl = 2usize;
    let rowk = kl * 2 * D;
    // (window, first position, rows): a full layer from the sequence's start (every key in the bias's
    // band, rows not a multiple of 16), one past the band, a window layer's rows over keys from lo
    for (window, p0, t) in [(0usize, 0usize, 37usize), (0, 2500, 21), (0, 1100, 64), (1024, 5000, 40)] {
        let a = Attn::new(&mut rng, kl, t, p0, 1024);
        let hl = a.hl;
        let (_, _, goff, soff) = a.offs();
        let n_pos = p0 + t;
        let all = rng.bf16_vec(n_pos * rowk, 1.0);
        let lo_key = if window > 0 { (p0 + 1).saturating_sub(window) } else { 0 };
        let kv = dev(&all[lo_key * rowk..]);
        let qc = rng.bf16_vec(t * hl * D, 2.0);
        let rc = rng.bf16_vec(t * hl * REL, 1.0);
        let (proj, proj_t, qcd, rcd) = (dev(&a.proj), dev(&a.proj_t), dev(&qc), dev(&rc));
        let o = DeviceBuffer::<u16>::zeroed(t * hl * D).unwrap();
        unsafe {
            ok(ink_pf_flash(st.raw(), p(&qcd), p(&rcd), p(&proj_t), 1024, p(&kv), rowk as i64, lo_key as i32, p(&proj), a.ldp as i64,
                            soff as i32, goff as i32, hl as i32, kl as i32, t as i32, p0 as i32, window as i32, p(&o)));
        }
        let allf = fl(&all);
        let want = a.reference(&fl(&qc), &fl(&rc), &|pos| allf[pos * rowk..(pos + 1) * rowk].to_vec(), window);
        let got = fl(&o.download().unwrap());
        let e = rel_err(&got, &want);
        let worst = (0..t * hl).map(|i| rel_err(&got[i * D..][..D], &want[i * D..][..D])).fold(0f32, f32::max);
        eprintln!("fused prefill attention window {window} p0 {p0} rows {t}: rel err {e:.2e}, worst (row, head) {worst:.2e}");
        assert!(e < 1e-2 && worst < 3e-2, "fused prefill attention window {window} p0 {p0}: rel err {e}, worst {worst}");
    }
}

/// TIDELM_INK_FLASH_BENCH=1: the fused prefill attention's time for a 2048-row chunk of one rank's
/// 24 heads (kl 4) deep in a context, the case that took the cuBLAS path tens of seconds a turn.
#[test]
fn fused_prefill_attention_time() {
    if std::env::var("TIDELM_INK_FLASH_BENCH").as_deref() != Ok("1") {
        return;
    }
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xbe_7c);
    let kl = 4usize;
    let rowk = kl * 2 * D;
    for (p0, t) in [(16384usize, 2048usize), (65536, 2048), (150000, 2048), (150000, 2284)] {
        let a = Attn::new(&mut rng, kl, t, p0, 1024);
        let hl = a.hl;
        let (_, _, goff, soff) = a.offs();
        let kv = dev(&rng.bf16_vec((p0 + t) * rowk, 1.0));
        let (qcd, rcd) = (dev(&rng.bf16_vec(t * hl * D, 2.0)), dev(&rng.bf16_vec(t * hl * REL, 1.0)));
        let (proj, proj_t) = (dev(&a.proj), dev(&a.proj_t));
        let o = DeviceBuffer::<u16>::zeroed(t * hl * D).unwrap();
        let run = || unsafe {
            ok(ink_pf_flash(st.raw(), p(&qcd), p(&rcd), p(&proj_t), 1024, p(&kv), rowk as i64, 0, p(&proj), a.ldp as i64, soff as i32,
                            goff as i32, hl as i32, kl as i32, t as i32, p0 as i32, 0, p(&o)));
        };
        run();
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 3.0;
        let flops = 4.0 * t as f64 * (p0 as f64 + t as f64 / 2.0) * D as f64 * hl as f64;
        eprintln!("fused prefill attention: {t} rows at {p0}: {ms:.1} ms a layer ({:.1} TFLOP/s; 8 full layers {:.2} s)", flops / ms / 1e9,
                  8.0 * ms / 1e3);
    }
}

// ------------------------------------------------------------------ snapshots, logits
#[test]
fn ring_copy_round_trips_across_the_wrap() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0x5a9_5407);
    let (layers, lanes, lane, r, row, p0, n) = (3usize, 2usize, 1usize, 64usize, 300usize, 60usize, 9usize);
    let ring_h = rng.bf16_vec(layers * lanes * r * row, 1.0);
    let ring = dev(&ring_h);
    let lin = DeviceBuffer::<u16>::zeroed(layers * n * row).unwrap();
    unsafe {
        ok(ink_ring_copy(st.raw(), ring.ptr_at(lane * r * row), (lanes * r * row) as i64, r as i32, row as i32, p0 as i32, n as i32,
                         layers as i32, p(&lin), (n * row) as i64, 0));
    }
    let got = lin.download().unwrap();
    for l in 0..layers {
        for j in 0..n {
            let src = ((l * lanes + lane) * r + ((p0 + j) & (r - 1))) * row;
            assert_eq!(&got[(l * n + j) * row..][..row], &ring_h[src..][..row], "layer {l} position {}", p0 + j);
        }
    }
    // back into a zeroed ring: only those positions of that lane
    let ring2 = DeviceBuffer::<u16>::zeroed(layers * lanes * r * row).unwrap();
    unsafe {
        ok(ink_ring_copy(st.raw(), ring2.ptr_at(lane * r * row), (lanes * r * row) as i64, r as i32, row as i32, p0 as i32, n as i32,
                         layers as i32, p(&lin), (n * row) as i64, 1));
    }
    let back = ring2.download().unwrap();
    for l in 0..layers {
        for ln in 0..lanes {
            for s in 0..r {
                let at = ((l * lanes + ln) * r + s) * row;
                let held = ln == lane && (0..n).any(|j| (p0 + j) & (r - 1) == s);
                if held {
                    assert_eq!(&back[at..][..row], &ring_h[at..][..row]);
                } else {
                    assert!(back[at..][..row].iter().all(|&v| v == 0));
                }
            }
        }
    }
}

#[test]
fn logits_are_softcapped_and_padding_ids_are_never() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0x10_917);
    let (t, n, ld, id0, real) = (3usize, 1000usize, 1024usize, 5000usize, 5900usize);
    let x = rng.f32_vec(t * ld, 40.0);
    let d = dev(&x);
    unsafe { ok(ink_logits(st.raw(), p(&d), ld as i64, n as i32, t as i32, 30.0, id0 as i32, real as i32)) };
    let got = d.download().unwrap();
    for r in 0..t {
        for c in 0..ld {
            let (g, v) = (got[r * ld + c], x[r * ld + c]);
            if c >= n {
                assert_eq!(g, v, "past the row's columns");
            } else if id0 + c >= real {
                assert_eq!(g, f32::NEG_INFINITY);
            } else {
                assert!((g - 30.0 * (v / 30.0).tanh()).abs() < 1e-4, "{g} for {v}");
            }
        }
    }
}

// ------------------------------------------------------------------ the prefill GEMMs
/// The prefill attention's two cuBLAS calls with the strides it uses: scores fp32 = q k^T from bf16
/// operands (keys every `rowk` elements), and bf16 = weights x values.
#[test]
fn prefill_gemms_take_bf16_operands_at_cache_strides() {
    use tidelm_kernels::blas::{Blas, Mat, R_16BF, R_32F};
    let Some(st) = setup() else { return };
    let mut rng = Rng(0xb1a5_0);
    let (rb, sk, kl, hl) = (5usize, 37usize, 2usize, 12usize);
    let rowk = kl * 2 * D;
    let q = rng.bf16_vec(rb * hl * D, 1.0);
    let kv = rng.bf16_vec(sk * rowk, 1.0);
    let w = rng.bf16_vec(rb * sk, 0.1);
    let (qd, kvd, wd) = (dev(&q), dev(&kv), dev(&w));
    let blas = Blas::new(st.raw()).unwrap();
    let (hq, g) = (7usize, 1usize);
    let sc = DeviceBuffer::<f32>::zeroed(rb * sk).unwrap();
    blas.gemm_rm(rb, sk, D, Mat { ptr: qd.ptr_at(hq * D), ty: R_16BF, ld: (hl * D) as i64 },
                 Mat { ptr: kvd.ptr_at(g * 2 * D), ty: R_16BF, ld: rowk as i64 }, true, p(&sc), R_32F, sk as i64).unwrap();
    let ov = DeviceBuffer::<u16>::zeroed(rb * D).unwrap();
    blas.gemm_rm(rb, D, sk, Mat { ptr: p(&wd), ty: R_16BF, ld: sk as i64 },
                 Mat { ptr: kvd.ptr_at(g * 2 * D + D), ty: R_16BF, ld: rowk as i64 }, false, p(&ov), R_16BF, D as i64).unwrap();
    st.synchronize().unwrap();
    let (qf, kf, wf) = (fl(&q), fl(&kv), fl(&w));
    let mut want_s = vec![0f32; rb * sk];
    let mut want_o = vec![0f32; rb * D];
    for i in 0..rb {
        for j in 0..sk {
            want_s[i * sk + j] = (0..D).map(|d| qf[(i * hl + hq) * D + d] * kf[j * rowk + g * 2 * D + d]).sum();
        }
        for d in 0..D {
            want_o[i * D + d] = bf((0..sk).map(|j| wf[i * sk + j] * kf[j * rowk + g * 2 * D + D + d]).sum());
        }
    }
    let (es, eo) = (rel_err(&sc.download().unwrap(), &want_s), rel_err(&fl(&ov.download().unwrap()), &want_o));
    eprintln!("prefill GEMMs: scores rel err {es:.2e}, weighted values {eo:.2e}");
    assert!(es < 1e-5 && eo < 3e-3, "prefill GEMMs: {es} {eo}");
}
