//! GPU tests for cuda/flash.cu (GLM-5.3-Flash's KDA layers and indexer key pools): each kernel
//! against a CPU transcription of transformers' modeling_glm5_next.py, the rollback replay against
//! running only the kept rows (bit for bit), and `kda_bench` (ignored: `cargo test --release -p
//! tidelm-kernels --test flash_gpu -- --ignored --nocapture`) for the decode and prefill row counts.

use std::os::raw::c_void;

use tidelm_cuda::{gpu_available, set_device, DeviceBuffer, Event, Stream};
use tidelm_kernels::flash::*;

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
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|&w| (w as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

const KD: usize = 128;
/// flash_set_kda_split is process-wide: the tests that set it run one at a time
static SPLIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One KDA layer's inputs for `t` rows at `hl` heads, and the CPU reference
/// (recurrent_kimi_delta_attention + Glm5NextTextRMSNormGated, the conv as causal_conv1d_fn).
struct Kda {
    hl: usize,
    t: usize,
    raw: Vec<u16>,  // [t][3 dl] q | k | v projections
    w: Vec<u16>,    // [3 dl][4]
    f: Vec<u16>,    // [t][dl] f_b output
    b: Vec<u16>,    // [t][hl]
    gt: Vec<u16>,   // [t][dl]
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    onorm: Vec<u16>,
    conv0: Vec<u16>,  // [3][3 dl] the conv state the pass reads
    s0: Vec<f32>,     // [hl][128 k][128 v]
}

impl Kda {
    fn new(rng: &mut Rng, hl: usize, t: usize) -> Self {
        let dl = hl * KD;
        Kda {
            hl,
            t,
            raw: rng.bf16_vec(t * 3 * dl, 1.0),
            w: rng.bf16_vec(3 * dl * 4, 0.5),
            f: rng.bf16_vec(t * dl, 1.0),
            b: rng.bf16_vec(t * hl, 1.0),
            gt: rng.bf16_vec(t * dl, 1.0),
            a_log: (0..hl).map(|_| rng.normal() * 0.5).collect(),
            dt_bias: (0..dl).map(|_| rng.normal() * 0.5).collect(),
            onorm: rng.bf16_vec(KD, 1.0),
            conv0: rng.bf16_vec(3 * 3 * dl, 1.0),
            s0: (0..hl * KD * KD).map(|_| rng.normal() * 0.1).collect(),
        }
    }

    /// (y [t][dl], final state [hl][k][v], conv state [3][3 dl]) for rows 0..n
    fn reference(&self, n: usize) -> (Vec<f32>, Vec<f32>, Vec<u16>) {
        let (hl, dl) = (self.hl, self.hl * KD);
        let c = 3 * dl;
        let x = |r: isize, ch: usize| -> f32 {
            if r >= 0 { f32_from_bf16(self.raw[r as usize * c + ch]) } else { f32_from_bf16(self.conv0[(3 + r) as usize * c + ch]) }
        };
        let mut act = vec![0f32; n * c];
        for t in 0..n {
            for ch in 0..c {
                let mut acc = 0f32;
                for tau in 0..4 {
                    acc = f32_from_bf16(self.w[ch * 4 + tau]).mul_add(x(t as isize - 3 + tau as isize, ch), acc);
                }
                act[t * c + ch] = bf(acc / (1.0 + (-acc).exp()));
            }
        }
        let mut s = self.s0.clone();
        let mut y = vec![0f32; n * dl];
        for t in 0..n {
            for h in 0..hl {
                let q: Vec<f32> = (0..KD).map(|i| act[t * c + h * KD + i]).collect();
                let k: Vec<f32> = (0..KD).map(|i| act[t * c + dl + h * KD + i]).collect();
                let v: Vec<f32> = (0..KD).map(|i| act[t * c + 2 * dl + h * KD + i]).collect();
                let nk = (k.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                let nq = (q.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                let kn: Vec<f32> = k.iter().map(|a| a / nk).collect();
                let qn: Vec<f32> = q.iter().map(|a| a / nq * 0.088388346).collect();
                let rate = self.a_log[h].exp();
                let g: Vec<f32> = (0..KD).map(|i| {
                    -5.0 * sigmoid(rate * (f32_from_bf16(self.f[t * dl + h * KD + i]) + self.dt_bias[h * KD + i]))
                }).collect();
                let beta = bf(sigmoid(f32_from_bf16(self.b[t * hl + h])));
                let st = &mut s[h * KD * KD..(h + 1) * KD * KD];
                for kk in 0..KD {
                    let e = g[kk].exp();
                    for vv in 0..KD {
                        st[kk * KD + vv] *= e;
                    }
                }
                let mut o = [0f32; KD];
                for vv in 0..KD {
                    let kv: f32 = (0..KD).map(|kk| st[kk * KD + vv] * kn[kk]).sum();
                    let delta = (v[vv] - kv) * beta;
                    for kk in 0..KD {
                        st[kk * KD + vv] += kn[kk] * delta;
                    }
                    o[vv] = bf((0..KD).map(|kk| st[kk * KD + vv] * qn[kk]).sum());
                }
                let rs = 1.0 / (o.iter().map(|a| a * a).sum::<f32>() / KD as f32 + 1e-5).sqrt();
                for vv in 0..KD {
                    let gate = sigmoid(f32_from_bf16(self.gt[t * dl + h * KD + vv]));
                    y[t * dl + h * KD + vv] = bf(f32_from_bf16(self.onorm[vv]) * (o[vv] * rs) * gate);
                }
            }
        }
        let mut conv = vec![0u16; 3 * c];
        for j in 0..3 {
            let r = n as isize - 3 + j as isize;
            for ch in 0..c {
                conv[j * c + ch] = if r >= 0 { self.raw[r as usize * c + ch] } else { self.conv0[(3 + r) as usize * c + ch] };
            }
        }
        (y, s, conv)
    }
}

/// The device side of one lane's KDA layer: state slots 0 (read) and 1 (written), conv the same.
struct KdaDev {
    raw: DeviceBuffer<u16>,
    w: DeviceBuffer<u16>,
    f: DeviceBuffer<u16>,
    b: DeviceBuffer<u16>,
    gt: DeviceBuffer<u16>,
    a_log: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    onorm: DeviceBuffer<u16>,
    conv: DeviceBuffer<u16>,
    state: DeviceBuffer<f32>,
    act: DeviceBuffer<u16>,
    y: DeviceBuffer<u16>,
    lanes: DeviceBuffer<i32>,
    meta: DeviceBuffer<i32>,
}

impl KdaDev {
    fn new(k: &Kda, slots: usize) -> Self {
        let (dl, c) = (k.hl * KD, 3 * k.hl * KD);
        let mut conv = k.conv0.clone();
        conv.resize(slots * 3 * c, 0);
        let mut state = k.s0.clone();
        state.resize(slots * k.hl * KD * KD, 0.0);
        KdaDev {
            raw: dev(&k.raw),
            w: dev(&k.w),
            f: dev(&k.f),
            b: dev(&k.b),
            gt: dev(&k.gt),
            a_log: dev(&k.a_log),
            dt_bias: dev(&k.dt_bias),
            onorm: dev(&k.onorm),
            conv: dev(&conv),
            state: dev(&state),
            act: DeviceBuffer::zeroed(k.t * c).unwrap(),
            y: DeviceBuffer::zeroed(k.t * dl).unwrap(),
            lanes: dev(&[0, k.t as i32, 0, 0]),
            meta: dev(&[0, 0, 1, 0]),
        }
    }

    /// conv, conv state and the chain over rows 0..n, slot `sin` -> `sout`; `save` the rows (R = n)
    #[allow(clippy::too_many_arguments)]
    fn run(&self, st: &Stream, k: &Kda, n: usize, sin: i32, sout: i32, save: Option<&Save>) {
        let (hl, dl, c) = (k.hl, k.hl * KD, 3 * k.hl * KD);
        self.lanes.upload(&[0, n as i32, 0, 0]).unwrap();
        self.meta.upload(&[0, sin, sout, 0]).unwrap();
        let cs = (3 * c) as i64;
        let ss = (hl * KD * KD) as i64;
        let (sr, sk, sg, sv, sb, r) = match save {
            Some(s) => (p(&s.raw), p(&s.k), p(&s.g), p(&s.v), p(&s.b), s.r),
            None => (null(), null(), null(), null(), null(), 0),
        };
        unsafe {
            ok(flash_kda_conv(st.raw(), p(&self.raw), c as i64, p(&self.w), c as i32, p(&self.conv), cs, p(&self.lanes), 1,
                              p(&self.meta), n as i32, p(&self.act), c as i64, sr, (r * c) as i64, r as i32));
            ok(flash_kda_conv_state(st.raw(), p(&self.raw), c as i64, c as i32, p(&self.conv), cs, p(&self.lanes), 1,
                                    p(&self.meta)));
            ok(flash_kda_chain(st.raw(), p(&self.act), c as i64, dl as i32, p(&self.f), dl as i64, p(&self.b), hl as i64,
                               p(&self.gt), dl as i64, p(&self.a_log), p(&self.dt_bias), p(&self.onorm), -5.0, 1e-5,
                               p(&self.state), ss, p(&self.lanes), 1, p(&self.meta), p(&self.y), dl as i64, sk, sg, sv, sb,
                               (r * hl * KD) as i64, r as i32, hl as i32));
        }
    }
}

struct Save {
    r: usize,
    raw: DeviceBuffer<u16>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    v: DeviceBuffer<u16>,
    b: DeviceBuffer<f32>,
}
impl Save {
    fn new(hl: usize, r: usize) -> Self {
        Save {
            r,
            raw: DeviceBuffer::zeroed(r * 3 * hl * KD).unwrap(),
            k: DeviceBuffer::zeroed(r * hl * KD).unwrap(),
            g: DeviceBuffer::zeroed(r * hl * KD).unwrap(),
            v: DeviceBuffer::zeroed(r * hl * KD).unwrap(),
            b: DeviceBuffer::zeroed(r * hl).unwrap(),
        }
    }
}

#[test]
fn kda_chain_matches_the_reference_for_both_splits() {
    let Some(st) = setup() else { return };
    let _g = SPLIT.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0x5eed_f1a5);
    let (hl, t) = (4, 7);
    let k = Kda::new(&mut rng, hl, t);
    let (want_y, want_s, want_conv) = k.reference(t);
    for sp in [2, 4] {
        assert_eq!(unsafe { flash_set_kda_split(sp) }, 0);
        let d = KdaDev::new(&k, 2);
        d.run(&st, &k, t, 0, 1, None);
        let y: Vec<f32> = d.y.download().unwrap().iter().map(|&h| f32_from_bf16(h)).collect();
        let s = d.state.download().unwrap();
        let conv = d.conv.download().unwrap();
        let ey = rel_err(&y, &want_y);
        let es = rel_err(&s[hl * KD * KD..], &want_s);
        eprintln!("split {sp}: y rel err {ey:.2e}, state rel err {es:.2e}");
        assert!(ey < 4e-3, "y rel err {ey}");
        assert!(es < 1e-5, "state rel err {es}");
        assert_eq!(&conv[3 * 3 * hl * KD..], &want_conv[..], "conv state");
        // the slot it read is untouched
        assert_eq!(&s[..hl * KD * KD], &k.s0[..]);
    }
    unsafe { flash_set_kda_split(2) };
}

#[test]
fn kda_replay_is_running_only_the_kept_rows() {
    let Some(st) = setup() else { return };
    let _g = SPLIT.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0xbead_cafe);
    let (hl, t) = (4, 6);
    let k = Kda::new(&mut rng, hl, t);
    let (c, ss) = (3 * hl * KD, hl * KD * KD);
    for sp in [2, 4] {
        unsafe { flash_set_kda_split(sp) };
        for keep in [0usize, 1, 3, 5, 6] {
            // a verify pass of t rows (saved), then a rollback to `keep`
            let a = KdaDev::new(&k, 2);
            let save = Save::new(hl, t);
            a.run(&st, &k, t, 0, 1, Some(&save));
            unsafe {
                ok(flash_kda_replay(st.raw(), p(&a.state), 0, ss as i64, 1, 0, 1, p(&save.k), p(&save.g), p(&save.v),
                                    p(&save.b), 0, keep as i32, hl as i32));
                ok(flash_kda_conv_replay(st.raw(), p(&a.conv), 0, (3 * c) as i64, c as i32, 1, 0, 1, p(&save.raw), 0,
                                         keep as i32));
            }
            // only `keep` rows
            let b = KdaDev::new(&k, 2);
            if keep > 0 {
                b.run(&st, &k, keep, 0, 1, None);
            } else {
                b.state.copy_from(ss, &b.state, 0, ss, &st).unwrap();
                b.conv.copy_from(3 * c, &b.conv, 0, 3 * c, &st).unwrap();
                st.synchronize().unwrap();
            }
            let (sa, sb) = (a.state.download().unwrap(), b.state.download().unwrap());
            assert!(sa[ss..].iter().zip(&sb[ss..]).all(|(x, y)| x.to_bits() == y.to_bits()), "split {sp} keep {keep}: state");
            assert_eq!(a.conv.download().unwrap()[3 * c..], b.conv.download().unwrap()[3 * c..], "split {sp} keep {keep}: conv");
        }
    }
    unsafe { flash_set_kda_split(2) };
}

#[test]
fn key_pools_across_passes_and_the_expansion() {
    let Some(st) = setup() else { return };
    let mut rng = Rng(0x9001);
    // two lanes in one pass: lane 0 rows 0..4 at positions 6..10 (pool 1 = 4..7 needs ring rows
    // 4, 5; pool 2 = 8..11 not complete), lane 1 rows 4..7 at positions 0..3 (pool 0)
    let n_tok = 11usize;
    let kall: Vec<Vec<u16>> = (0..2).map(|_| rng.bf16_vec(n_tok * 128, 1.0)).collect();
    let gall: Vec<Vec<u16>> = (0..2).map(|_| rng.bf16_vec(n_tok * 128, 1.0)).collect();
    let ape = rng.bf16_vec(4 * 128, 0.5);
    let rows: Vec<(usize, usize)> = (6..10).map(|x| (0, x)).chain((0..4).map(|x| (1, x))).collect();
    let (mut kb, mut gb) = (vec![], vec![]);
    for &(l, x) in &rows {
        kb.extend_from_slice(&kall[l][x * 128..(x + 1) * 128]);
        gb.extend_from_slice(&gall[l][x * 128..(x + 1) * 128]);
    }
    // lane 0's ring holds its positions 0..5 from an earlier pass
    let mut ring = vec![0u16; 2 * 64 * 256];
    for x in 0..6 {
        ring[x * 256..x * 256 + 128].copy_from_slice(&kall[0][x * 128..(x + 1) * 128]);
        ring[x * 256 + 128..x * 256 + 256].copy_from_slice(&gall[0][x * 128..(x + 1) * 128]);
    }
    let pos: Vec<i32> = rows.iter().map(|&(_, x)| x as i32).collect();
    let lanes = dev(&[0, 4, 6, 0, 4, 4, 0, 0]);
    let pool_rows = 8usize;
    let meta = dev(&[0, 0, 0, 0, 1, 0, 0, pool_rows as i32]);
    let (kd, gd, aped, ringd, posd) = (dev(&kb), dev(&gb), dev(&ape), dev(&ring), dev(&pos));
    let pq = DeviceBuffer::<u8>::zeroed(2 * pool_rows * 128).unwrap();
    let ps = DeviceBuffer::<f32>::zeroed(2 * pool_rows).unwrap();
    unsafe {
        ok(flash_pool(st.raw(), p(&kd), 128, p(&gd), 128, p(&aped), p(&ringd), 64 * 256, p(&posd), p(&lanes), 2, p(&meta), 8,
                      p(&pq), p(&ps)));
        ok(flash_ring(st.raw(), p(&kd), 128, p(&gd), 128, p(&ringd), 64 * 256, p(&posd), p(&lanes), 2, p(&meta), 8));
    }
    let (q, s) = (pq.download().unwrap(), ps.download().unwrap());
    let pool_ref = |l: usize, j: usize| -> Vec<f32> {
        (0..128).map(|d| {
            let lam: Vec<f32> = (0..4).map(|jj| f32_from_bf16(gall[l][(4 * j + jj) * 128 + d]) + f32_from_bf16(ape[jj * 128 + d])).collect();
            let m = lam.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = lam.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            bf((0..4).map(|jj| bf(bf(e[jj] / sum) * f32_from_bf16(kall[l][(4 * j + jj) * 128 + d]))).sum())
        }).collect()
    };
    for (l, j) in [(0usize, 1usize), (1, 0)] {
        let row = l * pool_rows + j;
        let got: Vec<f32> = (0..128).map(|d| e4m3(q[row * 128 + d]) * s[row]).collect();
        let e = rel_err(&got, &pool_ref(l, j));
        eprintln!("lane {l} pool {j}: rel err {e:.2e} (FP8)");
        assert!(e < 0.04, "lane {l} pool {j}: {e}");
    }
    assert!(s[2] == 0.0 && s[0] == 0.0, "only completed pools are written");
    let rb = ringd.download().unwrap();
    for &(l, x) in &rows {
        assert_eq!(&rb[(l * 64 + x) * 256..(l * 64 + x) * 256 + 128], &kall[l][x * 128..(x + 1) * 128], "ring K {l} {x}");
    }
    // expansion: a row at position 13 (npool 3, tail 12..13) with pools {0, 2} selected, and one at 2
    let pools = dev(&[0, 2, -1, -1, -1, -1, -1, -1]);
    let posx = dev(&[13, 2]);
    let out = DeviceBuffer::<i32>::zeroed(2 * 20).unwrap();
    unsafe { ok(flash_expand(st.raw(), p(&pools), 4, 4, p(&posx), p(&out), 20, 2)) };
    let o = out.download().unwrap();
    assert_eq!(&o[..12], &[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, -1, -1]);
    assert_eq!(&o[20..24], &[0, 1, 2, -1]);
}

#[test]
#[ignore]
fn kda_bench() {
    // a decode pass's 34 KDA layers back to back, each with its own state (4 MB read and 4 MB written
    // a layer at 64 heads: what a pass streams), conv then chain, timed over the whole sequence
    let Some(st) = setup() else { return };
    let _g = SPLIT.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(7);
    let (hl, layers) = (64usize, 34usize);
    let (dl, c) = (hl * KD, 3 * hl * KD);
    let ss = hl * KD * KD;
    let states: Vec<DeviceBuffer<f32>> = (0..layers).map(|_| DeviceBuffer::zeroed(2 * ss).unwrap()).collect();
    for &t in &[1usize, 2, 4, 8, 33, 512] {
        let k = Kda::new(&mut rng, hl, t);
        let d = KdaDev::new(&k, 2);
        d.lanes.upload(&[0, t as i32, 0, 0]).unwrap();
        d.meta.upload(&[0, 0, 1, 0]).unwrap();
        for sp in [2, 4] {
            unsafe { flash_set_kda_split(sp) };
            let (e0, e1, e2) = (Event::timing().unwrap(), Event::timing().unwrap(), Event::timing().unwrap());
            let reps = if t >= 512 { 2 } else { 6 };
            let (mut conv_ms, mut chain_ms) = (0.0f64, 0.0f64);
            for rep in 0..reps + 1 {
                e0.record(&st).unwrap();
                for _ in 0..layers {
                    unsafe {
                        flash_kda_conv(st.raw(), p(&d.raw), c as i64, p(&d.w), c as i32, p(&d.conv), (3 * c) as i64, p(&d.lanes),
                                       1, p(&d.meta), t as i32, p(&d.act), c as i64, null(), 0, 0);
                    }
                }
                e1.record(&st).unwrap();
                for s in &states {
                    unsafe {
                        flash_kda_chain(st.raw(), p(&d.act), c as i64, dl as i32, p(&d.f), dl as i64, p(&d.b), hl as i64,
                                        p(&d.gt), dl as i64, p(&d.a_log), p(&d.dt_bias), p(&d.onorm), -5.0, 1e-5, p(s), ss as i64,
                                        p(&d.lanes), 1, p(&d.meta), p(&d.y), dl as i64, null(), null(), null(), null(), 0, 0,
                                        hl as i32);
                    }
                }
                e2.record(&st).unwrap();
                e2.synchronize().unwrap();
                if rep > 0 {
                    conv_ms += e1.elapsed_ms(&e0).unwrap() as f64;
                    chain_ms += e2.elapsed_ms(&e1).unwrap() as f64;
                }
            }
            let (cv, ch) = (1e3 * conv_ms / reps as f64 / layers as f64, 1e3 * chain_ms / reps as f64 / layers as f64);
            eprintln!("rows {t:4} split {sp}: a layer conv {cv:7.1} us, chain {ch:7.1} us ({:.2} us a row); a pass's 34: {:.2} ms",
                      ch / t as f64, (cv + ch) * layers as f64 / 1e3);
        }
    }
    unsafe { flash_set_kda_split(2) };
}
