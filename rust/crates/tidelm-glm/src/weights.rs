//! The dense weights one rank holds (engine/glm/exl3_model.py::load_dense): EXL3 matrices from
//! the checkpoint cut at 128 boundaries (each rank's matrix an exact slice of the full one), the
//! raw norms, routers and indexer weights, kv_b quantized to FP8 at load, the embedding in INT8.

use std::os::raw::c_void;
use std::path::Path;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::blas::{Blas, Mat, R_16F, R_32F};
use tidelm_kernels::exl3::*;
use tidelm_kernels::glm::*;
use tidelm_model::Checkpoint;

use crate::config::GlmConfig;
use crate::lin::Lin;
use crate::{ck, Result};

/// Decode calls take at most this many rows (exl3_dense.MAX_ROWS); more go the prefill path.
pub const MAX_ROWS: usize = 16;

/// The grouped kernel's trellis loads in decode calls, dense and routed: two stages of 16-byte
/// cp.async through shared memory. The register path (pf 1: 4-byte loads a lane, one step ahead)
/// read o_proj at 171 GB/s across the layers (184.5 us), q_b 153, q_a 142; staged 209 / 179 / 167
/// (150.6 / 58.6 / 47.1 us), 3 and 4 stages no better; the outputs byte for byte the same
/// (glm_bench_chain).
pub const DECODE_PF: i32 = -2;

/// The routed experts' decode calls: three stages. At split 1 (glm_bench_moe, cold random 3-bit
/// experts, gate+up / down) 1 row 173 / 100 -> 169 / 90 us, 2 rows 299 / 134 -> 267 / 128, 4 rows
/// 479 / 226 -> 444 / 218, 8 rows 868 / 418 -> 848 / 416 against two; the dense matrices were
/// 1-2 us slower with three (glm_bench_chain). The same bytes either way.
pub const DECODE_PF_ROUTED: i32 = -3;

/// How a rank cuts a matrix of the pair: output columns (N tiles, svh) or input rows (K tiles,
/// suh; the outputs are partial sums), or not at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cut {
    None,
    N,
    K,
}

/// Static grouping tables for dense decode calls (exl3_dense._Members): for G groups over T rows,
/// group g's members are rows t * 32 + g, -1 padded to 16; uids 0..G - 1, count G.
pub struct Members {
    table: DeviceBuffer<i32>,
    uids: DeviceBuffer<i32>,
    counts: DeviceBuffer<i32>,
}

impl Members {
    pub fn new() -> Result<Self> {
        let mut t = vec![-1i32; 2 * MAX_ROWS * 32];
        for g in 1..=2usize {
            for rows in 1..=MAX_ROWS {
                let base = ((g - 1) * MAX_ROWS + rows - 1) * 32;
                for gg in 0..g {
                    for r in 0..rows {
                        t[base + gg * 16 + r] = (r * 32 + gg) as i32;
                    }
                }
            }
        }
        Ok(Members { table: DeviceBuffer::from_slice(&t)?, uids: DeviceBuffer::from_slice(&[0, 1])?,
                     counts: DeviceBuffer::from_slice(&[1, 2])? })
    }

    fn get(&self, g: usize, rows: usize) -> (*mut c_void, *mut c_void, *mut c_void) {
        (self.uids.raw(), self.counts.ptr_at(g - 1), self.table.ptr_at(((g - 1) * MAX_ROWS + rows - 1) * 32))
    }
}

/// Scratch an EXL3 call needs, shared by every matrix (sized for the largest).
pub struct Exl3Scratch {
    /// rotated input fp16 [rows][K]
    pub xh: DeviceBuffer<u16>,
    /// G-duplicated input rows (bf16)
    pub xg: DeviceBuffer<u16>,
    /// split-K partials / the prefill GEMM's fp32 output
    pub z: DeviceBuffer<f32>,
    /// prefill: the unpacked W_q fp16 [K][N]
    pub wq: DeviceBuffer<u16>,
}

impl Exl3Scratch {
    pub fn new(rows: usize, max_k: usize, max_kn: usize, max_zn: usize) -> Result<Self> {
        Ok(Exl3Scratch {
            xh: DeviceBuffer::new(rows * max_k)?,
            xg: DeviceBuffer::new(2 * MAX_ROWS * max_k)?,
            z: DeviceBuffer::new(max_zn)?,
            wq: DeviceBuffer::new(max_kn)?,
        })
    }
}

/// One rank's EXL3 matrix, or G same-shape groups over one input (gate and up): y = x @ W,
/// W [K, N] from the trellis (exl3_attn.Exl3Lin / exl3_dense.Exl3Matrix).
pub struct Exl3Mat {
    pub k: usize,
    pub n: usize,
    /// a padded N's used columns (kv_a: 576 of 640)
    pub n_out: usize,
    pub g: usize,
    pub bits: Vec<usize>,
    /// split-K of decode calls: it fixes their summation order
    pub split: usize,
    /// rows above which decode calls take `split_wide` (0: always `split`)
    pub wide_rows: usize,
    /// the grouped kernel's trellis loads in decode calls: PF steps ahead in registers (1 or 2), or
    /// -S, S stages of 16-byte cp.async through shared memory (DECODE_PF); and n-tiles a warp (8, or
    /// 4 with pf 2: half the reduction buffer, more blocks)
    pub pf: i32,
    pub nt: usize,
    pub split_wide: usize,
    tr: DeviceBuffer<u8>,
    tr_bytes: usize,
    suh: DeviceBuffer<u16>,
    svh: DeviceBuffer<u16>,
    ptr: DeviceBuffer<i64>,
    k2: DeviceBuffer<i32>,
    hi: i32,
    /// the trellis codebook: exl3::CB_MUL1 (the EXL3 checkpoint's) or CB_MCG_E4M3 (an exlr8 matrix
    /// rewritten in EXL3's layout, tools/glm_exlr8_dense.py)
    pub cb: i32,
}

impl Exl3Mat {
    pub fn bytes(&self) -> usize {
        self.tr.len() + 2 * (self.suh.len() + self.svh.len())
    }

    /// Decode (rows <= 16): rotate in, the grouped split-K GEMV, combine + rotate out.
    /// x bf16 rows (ldx elements apart) -> out [rows][G * N] contiguous, bf16 or fp32.
    pub fn decode(&self, st: &Stream, m: &Members, s: &Exl3Scratch, x: *const c_void, ldx: usize, rows: usize,
                  out: *mut c_void, out_bf16: bool) -> Result<()> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!("EXL3 decode call of {rows} rows").into());
        }
        let (k, n, g) = (self.k, self.n, self.g);
        let r = rows * g;
        let split = if self.wide_rows > 0 && rows > self.wide_rows { self.split_wide } else { self.split };
        let (xin, xs) = if g > 1 {
            ck(unsafe { glm_dup_rows(st.raw(), x, ldx as i64, s.xg.raw(), rows as i32, k as i32, g as i32) }, "dup rows")?;
            (s.xg.raw() as *const c_void, k)
        } else {
            (x, ldx)
        };
        unsafe {
            exl3m_dense_rot(xin, xs as i32, self.suh.raw(), s.xh.raw(), r as i32, k as i32, g as i32, 1, st.raw());
            let (uids, count, members) = m.get(g, rows);
            exl3m_grouped(s.xh.raw(), s.xh.raw(), self.ptr.raw(), self.ptr.raw(), self.k2.raw(), self.k2.raw(), uids,
                          count, members, s.z.raw(), k as i32, n as i32, r as i32, split as i32, MAX_MEMBERS,
                          g as i32, g as i32, 1, self.nt as i32, 4, self.pf, 2, self.hi, self.cb, st.raw());
            exl3m_dense_out(s.z.raw(), self.svh.raw(), out, r as i32, n as i32, split as i32, g as i32,
                            out_bf16 as i32, st.raw());
        }
        ck(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, "exl3 decode")
    }

    /// Where a decode call's warps read: (the trellis, bytes between the starts of consecutive
    /// warps' k ranges, ranges). A warp owns K / 16 / (split x 4) k tiles of all N tiles' rows of
    /// its n block, and the trellis is [k tile][n tile], so warp w's tiles start at w x the stride
    /// (G groups: each group's matrix is its own run of ranges).
    pub fn warp_runs(&self, rows: usize) -> (u64, u64, usize) {
        let split = if self.wide_rows > 0 && rows > self.wide_rows { self.split_wide } else { self.split };
        let count = split * 4 * self.g;
        (self.tr.raw() as u64, (self.tr.len() / count) as u64, count)
    }

    /// Prefill (any rows): group `grp`'s trellis unpacked to fp16, the rotated rows times it on
    /// cuBLAS (fp32 accumulation), combine + rotate out (Exl3Lin._prefill). out [rows][N].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(&self, st: &Stream, blas: &Blas, s: &Exl3Scratch, grp: usize, x: *const c_void, ldx: usize,
                   rows: usize, out: *mut c_void, out_bf16: bool) -> Result<()> {
        let (k, n) = (self.k, self.n);
        if rows * k > s.xh.len() || k * n > s.wq.len() || rows * n > s.z.len() {
            return Err(format!("EXL3 prefill of {rows} x {k} x {n}: scratch too small").into());
        }
        unsafe {
            exl3m_dense_rot(x, ldx as i32, self.suh.ptr_at(grp * k), s.xh.raw(), rows as i32, k as i32, 1, 1, st.raw());
        }
        let tr = unsafe { (self.tr.raw() as *const u8).add(grp * self.tr_bytes) } as *const c_void;
        ck(unsafe { glm_exl3_unpack_cb(st.raw(), tr, k as i32, n as i32, self.bits[grp] as i32, self.cb, s.wq.raw()) }, "exl3 unpack")?;
        blas.gemm_rm(rows, n, k, Mat { ptr: s.xh.raw(), ty: R_16F, ld: k as i64 }, Mat { ptr: s.wq.raw(), ty: R_16F, ld: n as i64 },
                     false, s.z.raw(), R_32F, n as i64)?;
        unsafe { exl3m_dense_out(s.z.raw(), self.svh.ptr_at(grp * n), out, rows as i32, n as i32, 1, 1, out_bf16 as i32, st.raw()) };
        ck(unsafe { tidelm_cuda::ffi::cudaGetLastError() }, "exl3 prefill")
    }
}

/// The checkpoint, read for one rank.
pub struct Loader<'a> {
    pub ck: &'a Checkpoint,
    pub rank: usize,
    pub world: usize,
    pub st: &'a Stream,
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        s * m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 { s * f32::INFINITY } else { f32::NAN }
    } else {
        s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

pub fn bf16_from_f32(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

impl<'a> Loader<'a> {
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.ck.read_tensor_vec(name).map_err(|e| format!("{name}: {e}"))?.0)
    }

    pub fn dtype(&self, name: &str) -> Result<String> {
        Ok(self.ck.meta(name).map_err(|e| format!("{name}: {e}"))?.dtype.clone())
    }

    /// A BF16 tensor as it is (norms).
    pub fn bf16(&self, name: &str) -> Result<DeviceBuffer<u16>> {
        if self.dtype(name)? != "BF16" {
            return Err(format!("{name}: not BF16").into());
        }
        Ok(DeviceBuffer::from_slice(&u16s(&self.bytes(name)?))?)
    }

    /// An F16 / F32 / BF16 tensor widened to fp32 on the host.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let b = self.bytes(name)?;
        Ok(match self.dtype(name)?.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => u16s(&b).into_iter().map(f16_to_f32).collect(),
            "BF16" => u16s(&b).into_iter().map(|h| f32::from_bits((h as u32) << 16)).collect(),
            d => return Err(format!("{name}: dtype {d}").into()),
        })
    }

    pub fn f32_dev(&self, name: &str) -> Result<DeviceBuffer<f32>> {
        Ok(DeviceBuffer::from_slice(&self.f32s(name)?)?)
    }

    /// An F16 tensor as it is (the routers: ds_linear_f16w widens it in registers).
    pub fn f16_dev(&self, name: &str) -> Result<DeviceBuffer<u16>> {
        if self.dtype(name)? != "F16" {
            return Err(format!("{name}: not F16").into());
        }
        Ok(DeviceBuffer::from_slice(&u16s(&self.bytes(name)?))?)
    }

    /// An F16 weight as BF16 (the indexer's wk: Lin(ck.get(wk)) runs it as a BF16 weight).
    pub fn bf16_from_f16(&self, name: &str) -> Result<DeviceBuffer<u16>> {
        let v: Vec<u16> = self.f32s(name)?.into_iter().map(bf16_from_f32).collect();
        Ok(DeviceBuffer::from_slice(&v)?)
    }

    /// An EXL3 linear `prefix` (trellis / suh / svh / mul1), cut for this rank. `groups`: other
    /// prefixes of the same shape stacked after it (gate then up), read as one G-group matrix.
    pub fn exl3(&self, prefixes: &[&str], split: usize, cut: Cut, n_out: Option<usize>) -> Result<Exl3Mat> {
        let (r, w) = (self.rank, self.world);
        let mut tr_all = vec![];
        let (mut suh_all, mut svh_all, mut bits) = (vec![], vec![], vec![]);
        let (mut k, mut n, mut tr_bytes) = (0, 0, 0);
        let mut cb = 0;
        for (gi, pre) in prefixes.iter().enumerate() {
            // the codebook's marker tensor: `.mul1` (EXL3), or `.mcg_e4m3` / `.mul1_e4m3` (an exlr8 matrix in
            // EXL3's layout: mcg or mul1, every value rounded to FP8 E4M3; tools/glm_exlr8_dense.py - the
            // release's dense matrices are mcg, its shared experts mul1 since 0.6)
            let (marker, want, c) = if self.ck.meta(&format!("{pre}.mcg_e4m3")).is_ok() {
                ("mcg_e4m3", 0xCBAC_1FEDu32, CB_MCG_E4M3)
            } else if self.ck.meta(&format!("{pre}.mul1_e4m3")).is_ok() {
                ("mul1_e4m3", 0x83DC_D12D, tidelm_kernels::exl3::CB_MUL1_E4M3)
            } else {
                ("mul1", 0x83DC_D12D, CB_MUL1)
            };
            let m = self.bytes(&format!("{pre}.{marker}"))?;
            if m.len() != 4 || u32::from_le_bytes([m[0], m[1], m[2], m[3]]) != want {
                return Err(format!("{pre}: not the {marker} codebook").into());
            }
            if gi > 0 && c != cb {
                return Err(format!("{pre}: groups differ in codebook").into());
            }
            cb = c;
            let meta = self.ck.meta(&format!("{pre}.trellis")).map_err(|e| format!("{pre}: {e}"))?;
            if meta.dtype != "I16" || meta.shape.len() != 3 || meta.shape[2] % 16 != 0 {
                return Err(format!("{pre}.trellis: {:?} {:?}", meta.dtype, meta.shape).into());
            }
            let (kt, nt, w16) = (meta.shape[0], meta.shape[1], meta.shape[2]);
            let b = w16 / 16;
            let tr = self.bytes(&format!("{pre}.trellis"))?;
            let mut suh = u16s(&self.bytes(&format!("{pre}.suh"))?);
            let mut svh = u16s(&self.bytes(&format!("{pre}.svh"))?);
            if suh.len() != kt * 16 || svh.len() != nt * 16 {
                return Err(format!("{pre}: suh/svh lengths {} / {} for trellis {kt} x {nt}", suh.len(), svh.len()).into());
            }
            let tile = w16 * 2;
            let tr = match cut {
                Cut::None => tr,
                Cut::N => {
                    let nl = nt / w;
                    let mut o = Vec::with_capacity(kt * nl * tile);
                    for ki in 0..kt {
                        let row = ki * nt * tile;
                        o.extend_from_slice(&tr[row + r * nl * tile..row + (r + 1) * nl * tile]);
                    }
                    svh = svh[r * nl * 16..(r + 1) * nl * 16].to_vec();
                    o
                }
                Cut::K => {
                    let kl = kt / w;
                    suh = suh[r * kl * 16..(r + 1) * kl * 16].to_vec();
                    tr[r * kl * nt * tile..(r + 1) * kl * nt * tile].to_vec()
                }
            };
            let (gk, gn) = (suh.len(), svh.len());
            if gi == 0 {
                (k, n, tr_bytes) = (gk, gn, tr.len());
            } else if (gk, gn, tr.len()) != (k, n, tr_bytes) {
                return Err(format!("{pre}: groups differ in shape").into());
            }
            if k % 128 != 0 || n % 128 != 0 || (k / 16) % (split * 4) != 0 {
                return Err(format!("{pre}: K {k} N {n} does not take split {split}").into());
            }
            tr_all.extend_from_slice(&tr);
            suh_all.extend_from_slice(&suh);
            svh_all.extend_from_slice(&svh);
            bits.push(b);
        }
        let g = prefixes.len();
        if !(1..=2).contains(&g) || bits.iter().any(|&b| !(2..=8).contains(&b)) {
            return Err(format!("{}: {g} groups of bits {bits:?}", prefixes[0]).into());
        }
        let tr = DeviceBuffer::from_slice(&tr_all)?;
        let base = tr.raw() as i64;
        let ptr = DeviceBuffer::from_slice(&(0..g).map(|i| base + (i * tr_bytes) as i64).collect::<Vec<_>>())?;
        let k2 = DeviceBuffer::from_slice(&bits.iter().map(|&b| 2 * b as i32).collect::<Vec<_>>())?;
        let hi = if bits.iter().all(|&b| b <= 5) { 10 } else { 16 };
        Ok(Exl3Mat { k, n, n_out: n_out.unwrap_or(n), g, bits, split, wide_rows: 0, split_wide: split, pf: DECODE_PF, nt: 8, tr, tr_bytes,
                     suh: DeviceBuffer::from_slice(&suh_all)?,
                     svh: DeviceBuffer::from_slice(&svh_all)?, ptr, k2, hi, cb })
    }

    /// kv_b (F16 [heads * 448, 512]) for this rank's heads, quantized to FP8 128x128 blocks with
    /// the scales expanded per row (GLM_KVB_FP8=1): (fp8 [rows][512], fp32 [rows][4]).
    pub fn kvb_fp8(&self, name: &str, hl: usize) -> Result<(DeviceBuffer<u8>, DeviceBuffer<f32>)> {
        let rows = hl * 448;
        if self.dtype(name)? != "F16" {
            return Err(format!("{name}: kv_b is not F16").into());
        }
        let b = self.bytes(name)?;
        let row_bytes = 512 * 2;
        let lo = self.rank * rows * row_bytes;
        let w = DeviceBuffer::<u8>::from_slice(&b[lo..lo + rows * row_bytes])?;
        let q = DeviceBuffer::<u8>::new(rows * 512)?;
        let s = DeviceBuffer::<f32>::new(rows * 4)?;
        ck(unsafe { glm_quant_kvb(self.st.raw(), w.raw(), rows as i32, q.raw(), s.raw()) }, "quant kv_b")?;
        self.st.synchronize()?;
        Ok((q, s))
    }

    /// This rank's vocabulary rows of the embedding as INT8 with an fp32 scale per 128 values.
    pub fn embed_int8(&self, name: &str, vl: usize, h: usize) -> Result<(DeviceBuffer<i8>, DeviceBuffer<f32>)> {
        let q = DeviceBuffer::<i8>::new(vl * h)?;
        let s = DeviceBuffer::<f32>::new(vl * h / 128)?;
        let rows = 8192usize;
        let stage = DeviceBuffer::<u16>::new(rows * h)?;
        let row_bytes = h * 2;
        let meta = self.ck.meta(name).map_err(|e| format!("{name}: {e}"))?;
        let mut buf = tidelm_model::direct_io::AlignedBuf::new(rows * row_bytes + 2 * tidelm_model::direct_io::ALIGN)?;
        for a in (0..vl).step_by(rows) {
            let n = rows.min(vl - a);
            let lo = (self.rank * vl + a) * row_bytes;
            let need = tidelm_model::aligned_len(meta.lo + lo as u64, meta.lo + (lo + n * row_bytes) as u64);
            let off = self.ck.read_tensor_range(name, lo, lo + n * row_bytes, &mut buf[..need])
                .map_err(|e| format!("{name}: {e}"))?;
            stage.upload(&u16s(&buf[off..off + n * row_bytes]))?;
            ck(unsafe { glm_quant_int8_rows(self.st.raw(), stage.raw(), n as i32, h as i32, q.ptr_at(a * h), s.ptr_at(a * h / 128)) },
               "quant embed")?;
            self.st.synchronize()?;
        }
        Ok((q, s))
    }
}

/// A layer's indexer ("full" layers and the MTP layer).
pub struct Indexer {
    pub wq_b: Exl3Mat,
    /// wk as BF16 [128][6144]
    pub wk: DeviceBuffer<u16>,
    pub kn_w: DeviceBuffer<f32>,
    pub kn_b: DeviceBuffer<f32>,
    /// weights_proj fp32 [32][6144]
    pub wp: DeviceBuffer<f32>,
}

pub enum Mlp {
    /// layers 0-2: gate + up (this rank's intermediate columns), down over the whole
    /// intermediate to this rank's hidden columns
    Dense { gate_up: Lin, down: Lin },
    /// the router (GLM-5.3's fp16, computed as the fp32 weight it widens to; GLM-5.3-Flash's
    /// bf16), its correction bias, the shared expert as Dense's halves
    Moe { router: DeviceBuffer<u16>, router_bf16: bool, bias: DeviceBuffer<f32>, sh_gate_up: Lin, sh_down: Lin },
    /// Inkling 2's MoE: the router (bf16) and its selection bias, no shared expert
    Routed { router: DeviceBuffer<u16>, bias: DeviceBuffer<f32> },
}

/// A MoE layer's router and, where the model has one, its shared expert.
pub struct MoeRef<'a> {
    pub router: &'a DeviceBuffer<u16>,
    pub router_bf16: bool,
    pub bias: &'a DeviceBuffer<f32>,
    pub shared: Option<(&'a Lin, &'a Lin)>,
}

impl Mlp {
    pub fn moe(&self) -> Option<MoeRef<'_>> {
        match self {
            Mlp::Dense { .. } => None,
            Mlp::Moe { router, router_bf16, bias, sh_gate_up, sh_down } => {
                Some(MoeRef { router, router_bf16: *router_bf16, bias, shared: Some((sh_gate_up, sh_down)) })
            }
            Mlp::Routed { router, bias } => Some(MoeRef { router, router_bf16: true, bias, shared: None }),
        }
    }
}

pub struct MtpExtra {
    pub enorm: DeviceBuffer<u16>,
    pub hnorm: DeviceBuffer<u16>,
    pub sh_norm: DeviceBuffer<u16>,
    pub eh: Lin,
}

/// GLM-5.3's attention: MLA with RoPE, every projection EXL3.
pub struct MlaW {
    pub q_a: Exl3Mat,
    pub q_norm: DeviceBuffer<u16>,
    pub kv_a: Exl3Mat,
    pub kv_norm: DeviceBuffer<u16>,
    pub q_b: Exl3Mat,
    /// kv_b of this rank's heads: FP8 [hl * 448][512], scales [hl * 448][4]
    pub kvb_q: DeviceBuffer<u8>,
    pub kvb_s: DeviceBuffer<f32>,
    pub o: Exl3Mat,
    pub index: Option<Indexer>,
}

/// GLM-5.3-Flash's KDA layer. `proj` rows: this rank's q | k | v (dl = heads x 128 each), f_a and
/// g_a (128 each, whole), this rank's b (a row a head): KDA_PROJ_OFFSETS.
pub struct KdaW {
    pub proj: Lin,
    /// f_b, g_b [dl][128]: the forget gate's and the output gate's second halves
    pub f_b: Lin,
    pub g_b: Lin,
    /// the short conv's taps, [3 dl][4]: this rank's q channels, then k, then v
    pub conv: DeviceBuffer<u16>,
    pub a_log: DeviceBuffer<f32>,
    pub dt_bias: DeviceBuffer<f32>,
    pub o_norm: DeviceBuffer<u16>,
    /// o_proj over this rank's dl columns (summed across ranks)
    pub o: Lin,
}

impl KdaW {
    /// proj's output columns: (q, k, v, f_a, g_a, b, the row's width) for `dl` dims and `hl` heads
    pub fn offsets(dl: usize, hl: usize) -> [usize; 7] {
        [0, dl, 2 * dl, 3 * dl, 3 * dl + 128, 3 * dl + 256, 3 * dl + 256 + hl]
    }
}

/// GLM-5.3-Flash's sparse-attention layer: MLA without RoPE and the pooled indexer. `proj_a` rows
/// (from the normed input, whole on every rank): q_a 1536 | kv_a 512 | wk 128 | weights_proj 32 |
/// the pools' compress gate 128; `proj_b` rows (from q_a's norm): this rank's q_b heads x 256 |
/// the indexer's wq_b 32 x 128.
pub struct DsaW {
    pub proj_a: Lin,
    pub q_norm: DeviceBuffer<u16>,
    pub kv_norm: DeviceBuffer<u16>,
    pub proj_b: Lin,
    /// kv_b of this rank's heads: FP8 [hl * 512][512], scales [hl * 512][4]
    pub kvb_q: DeviceBuffer<u8>,
    pub kvb_s: DeviceBuffer<f32>,
    pub o: Lin,
    pub kn_w: DeviceBuffer<u16>,
    pub kn_b: DeviceBuffer<u16>,
    /// the pools' absolute position embedding [4][128]
    pub ape: DeviceBuffer<u16>,
}

impl DsaW {
    /// proj_a's output columns: (q_a, kv_a, wk, weights_proj, gate, the row's width)
    pub const OFFSETS_A: [usize; 6] = [0, 1536, 2048, 2176, 2208, 2336];
}

/// Inkling 2's attention (inkling.rs loads it, forward's inkling_fwd runs it). `hl` query heads and
/// `kl` = hl / 6 key / value heads on this rank, 128 dims each, a relative state of 16 a query head.
///
/// A window layer projects everything from one normed input: `proj` rows q | k | v | r | gate |
/// sink. A full-attention layer has an input each for q, k and v: `proj` rows q | r | gate | sink
/// from the first, `proj_k` rows k | the key convolution's per-token taps (kl x 4) from the second,
/// `proj_v` the same for the values from the third. The sink's hl rows and the taps' kl x 4 are
/// padded with zero rows to whole 32-row blocks (the FP8 scales').
pub struct InkW {
    pub proj: Lin,
    pub proj_k: Option<Lin>,
    pub proj_v: Option<Lin>,
    pub off: InkOff,
    /// o_proj over this rank's hl x 128 columns (summed across ranks)
    pub o: Lin,
    /// q_l2_clip.scale [hl][128], rel_l2_clip.scale [hl][16]
    pub q_scale: DeviceBuffer<u16>,
    pub r_scale: DeviceBuffer<u16>,
    /// rel_logits_proj.proj transposed: [rel_extent][16], a distance's 16 values together
    pub proj_t: DeviceBuffer<u16>,
    /// the key / value convolutions' taps over this rank's kl x 128 channels: [c][4] in conv1d's
    /// order (window layers), [4][c] by lag (full layers: static_w)
    pub k_conv: DeviceBuffer<u16>,
    pub v_conv: DeviceBuffer<u16>,
    /// the attention output's and the MoE output's convolutions [H][4]
    pub attn_conv: DeviceBuffer<u16>,
    pub mlp_conv: DeviceBuffer<u16>,
    /// the block attention residual's queries (fp32): the attention input's [1 or 3][H] (none on
    /// layer 0, which mixes nothing), the MoE input's [H]
    pub q_attn: Option<DeviceBuffer<f32>>,
    pub q_mlp: DeviceBuffer<f32>,
}

/// Row offsets in InkW's projections (elements of a row of their outputs).
#[derive(Clone, Copy, Debug, Default)]
pub struct InkOff {
    pub q: usize,
    pub r: usize,
    pub gate: usize,
    pub sink: usize,
    /// window layers: k and v in `proj`; full layers: 0 in proj_k / proj_v
    pub k: usize,
    pub v: usize,
    /// full layers: the taps in proj_k / proj_v
    pub taps: usize,
}

pub enum Attn {
    Mla(MlaW),
    Kda(KdaW),
    Dsa(DsaW),
    Ink(InkW),
}

/// A layer's two mHC sites (attention, then MLP): each its mixing weight [24][4 x hidden] (bf16,
/// the checkpoint's), base [24] and scale [3].
pub struct HcW {
    pub attn_fn: Lin,
    pub attn_base: DeviceBuffer<f32>,
    pub attn_scale: DeviceBuffer<f32>,
    pub ffn_fn: Lin,
    pub ffn_base: DeviceBuffer<f32>,
    pub ffn_scale: DeviceBuffer<f32>,
}

pub struct Layer {
    pub ln_in: DeviceBuffer<u16>,
    pub ln_post: DeviceBuffer<u16>,
    pub attn: Attn,
    pub hc: Option<HcW>,
    pub mlp: Mlp,
    pub mtp: Option<MtpExtra>,
}

impl Layer {
    pub fn mla(&self) -> Option<&MlaW> {
        if let Attn::Mla(m) = &self.attn { Some(m) } else { None }
    }

    pub fn ink(&self) -> Option<&InkW> {
        if let Attn::Ink(m) = &self.attn { Some(m) } else { None }
    }
}

/// The embedding table: GLM-5.3's INT8 rows with an fp32 scale per 128 (this rank's vocabulary),
/// GLM-5.3-Flash's BF16 rows as they are (the whole table on every rank).
pub enum Embed {
    Int8 { q: DeviceBuffer<i8>, s: DeviceBuffer<f32> },
    Bf16(DeviceBuffer<u16>),
}

/// The draft head: the rows of lm_head for the listed token ids (ascending), in the dense weights'
/// format; column c scores token ids[c].
pub struct DraftHead {
    pub head: Lin,
    pub ids: Vec<u32>,
    /// by token id: whether it is listed
    pub listed: Vec<bool>,
}

pub struct Weights {
    pub layers: Vec<Layer>,
    pub embed: Embed,
    pub norm: DeviceBuffer<u16>,
    pub head: Lin,
    /// GLM-5.3-Flash on one box: the MTP draft passes' head over a listed vocabulary (flash.rs)
    pub draft: Option<DraftHead>,
    /// RoPE tables [max_seq][32] BF16 (cos, sin of p * theta^(-2i/64)); empty without RoPE
    pub cos: DeviceBuffer<u16>,
    pub sin: DeviceBuffer<u16>,
    pub bytes: usize,
}

impl Weights {
    /// Every EXL3-layout matrix this rank holds.
    pub fn for_each_mat(&mut self, f: &mut dyn FnMut(&mut Exl3Mat)) {
        let lin = |m: &mut Lin, f: &mut dyn FnMut(&mut Exl3Mat)| {
            if let Some(x) = m.exl3_mut() {
                f(x);
            }
        };
        for l in self.layers.iter_mut() {
            if let Attn::Mla(a) = &mut l.attn {
                for m in [&mut a.q_a, &mut a.kv_a, &mut a.q_b, &mut a.o] {
                    f(m);
                }
                if let Some(ix) = a.index.as_mut() {
                    f(&mut ix.wq_b);
                }
            }
            match &mut l.mlp {
                Mlp::Dense { gate_up, down } => {
                    lin(gate_up, f);
                    lin(down, f);
                }
                Mlp::Moe { sh_gate_up, sh_down, .. } => {
                    lin(sh_gate_up, f);
                    lin(sh_down, f);
                }
                Mlp::Routed { .. } => {}
            }
            if let Some(x) = l.mtp.as_mut() {
                lin(&mut x.eh, f);
            }
        }
        lin(&mut self.head, f);
    }
}

/// The decode split-K counts per projection (exl3_model.py; other splits move the outputs'
/// bits).
pub const SPLIT_QA: usize = 8;
pub const SPLIT_KVA: usize = 8;
pub const SPLIT_QB: usize = 1;
pub const SPLIT_O: usize = 2;
/// o_proj in exlr8's codebook (4-bit, mcg rounded to E4M3: tools/glm_exlr8_dense.py): its arithmetic,
/// not its bytes, bounds the call at split 2 - in a pass 12 MB of the matrix are already in L2. In-boot
/// A/B on the pair (`--bench-ab osplit`, two boots, 2,500 tokens of context): 1 row 61.72 -> 60.36 and
/// 61.80 -> 60.40 ms, 4 rows 112.85 -> 111.28 and 112.84 -> 110.91. q_b at 4 instead of 1: +0.4 / -0.6%,
/// left. (The 5-bit mul1 o_proj reads the same at 2 and 4: 149.5 / 150.5 us a layer, glm_bench_chain.)
pub const SPLIT_O_E4M3: usize = 8;
pub const SPLIT_IQ: usize = 1;
pub const SPLIT_EH: usize = 4;
pub const SPLIT_HEAD: usize = 1;
pub const SPLIT_UP: usize = 4;
pub const SPLIT_DOWN: usize = 2;

pub fn rope_tables(cfg: &GlmConfig, max_seq: usize) -> (Vec<u16>, Vec<u16>) {
    // tp/exl3_model: inv = 1 / theta^(arange(0, 64, 2) / 64) in fp32, f = p * inv (fp32), cos/sin -> bf16
    let half = cfg.rope / 2;
    if half == 0 {
        return (vec![0], vec![0]);
    }
    let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / (cfg.rope_theta as f32).powf((2 * i) as f32 / cfg.rope as f32)).collect();
    let mut cs = vec![0u16; max_seq * half];
    let mut sn = vec![0u16; max_seq * half];
    for p in 0..max_seq {
        for i in 0..half {
            let f = p as f32 * inv[i];
            cs[p * half + i] = bf16_from_f32((f as f64).cos() as f32);
            sn[p * half + i] = bf16_from_f32((f as f64).sin() as f32);
        }
    }
    (cs, sn)
}

impl Weights {
    pub fn load(model_dir: &Path, cfg: &GlmConfig, rank: usize, world: usize, max_seq: usize, st: &Stream,
                log: &dyn Fn(&str)) -> Result<Self> {
        if cfg.is_flash() {
            return crate::flash::load_weights(model_dir, cfg, rank, world, st, log);
        }
        if cfg.is_inkling() {
            return crate::inkling::load_weights(model_dir, cfg, rank, world, st, log);
        }
        let ckpt = Checkpoint::open(model_dir)?;
        let ld = Loader { ck: &ckpt, rank, world, st };
        let hl = cfg.heads / world;
        let vl = cfg.vocab / world;
        let t0 = std::time::Instant::now();
        let (embed_q, embed_s) = ld.embed_int8("model.embed_tokens.weight", vl, cfg.hidden)?;
        let head = ld.exl3(&["lm_head"], SPLIT_HEAD, Cut::N, None)?;
        let norm = ld.bf16("model.norm.weight")?;
        let mut bytes = embed_q.len() + 4 * embed_s.len() + head.bytes();
        let mut layers = Vec::with_capacity(cfg.n_layers());
        for l in 0..cfg.n_layers() {
            let p = format!("model.layers.{l}.");
            let a = format!("{p}self_attn.");
            let (kvb_q, kvb_s) = ld.kvb_fp8(&format!("{a}kv_b_proj.weight"), hl)?;
            let index = if cfg.index_full[l] {
                let i = format!("{a}indexer.");
                Some(Indexer {
                    wq_b: ld.exl3(&[&format!("{i}wq_b")], SPLIT_IQ, Cut::None, None)?,
                    wk: ld.bf16_from_f16(&format!("{i}wk.weight"))?,
                    kn_w: ld.f32_dev(&format!("{i}k_norm.weight"))?,
                    kn_b: ld.f32_dev(&format!("{i}k_norm.bias"))?,
                    wp: ld.f32_dev(&format!("{i}weights_proj.weight"))?,
                })
            } else {
                None
            };
            let mlp = if cfg.dense[l] {
                let m = format!("{p}mlp.");
                Mlp::Dense {
                    gate_up: Lin::Exl3(ld.exl3(&[&format!("{m}gate_proj"), &format!("{m}up_proj")], SPLIT_UP, Cut::N, None)?),
                    down: Lin::Exl3(ld.exl3(&[&format!("{m}down_proj")], SPLIT_DOWN, Cut::N, None)?),
                }
            } else {
                let m = format!("{p}mlp.shared_experts.");
                Mlp::Moe {
                    router: ld.f16_dev(&format!("{p}mlp.gate.weight"))?,
                    router_bf16: false,
                    bias: ld.f32_dev(&format!("{p}mlp.gate.e_score_correction_bias"))?,
                    sh_gate_up: Lin::Exl3(ld.exl3(&[&format!("{m}gate_proj"), &format!("{m}up_proj")], SPLIT_UP, Cut::N, None)?),
                    sh_down: Lin::Exl3(ld.exl3(&[&format!("{m}down_proj")], SPLIT_DOWN, Cut::N, None)?),
                }
            };
            let mtp = if cfg.mtp && l == cfg.mtp_layer() {
                Some(MtpExtra {
                    enorm: ld.bf16(&format!("{p}enorm.weight"))?,
                    hnorm: ld.bf16(&format!("{p}hnorm.weight"))?,
                    sh_norm: ld.bf16(&format!("{p}shared_head.norm.weight"))?,
                    eh: Lin::Exl3(ld.exl3(&[&format!("{p}eh_proj")], SPLIT_EH, Cut::N, None)?),
                })
            } else {
                None
            };
            let mla = MlaW {
                q_a: ld.exl3(&[&format!("{a}q_a_proj")], SPLIT_QA, Cut::None, None)?,
                q_norm: ld.bf16(&format!("{a}q_a_layernorm.weight"))?,
                kv_a: ld.exl3(&[&format!("{a}kv_a_proj_with_mqa")], SPLIT_KVA, Cut::None, Some(cfg.kv_lora + cfg.rope))?,
                kv_norm: ld.bf16(&format!("{a}kv_a_layernorm.weight"))?,
                q_b: ld.exl3(&[&format!("{a}q_b_proj")], SPLIT_QB, Cut::N, None)?,
                kvb_q,
                kvb_s,
                o: {
                    let mut o = ld.exl3(&[&format!("{a}o_proj")], SPLIT_O, Cut::K, None)?;
                    if is_exlr8(o.cb) {
                        o.split = SPLIT_O_E4M3;
                        o.split_wide = SPLIT_O_E4M3;
                    }
                    o
                },
                index,
            };
            bytes += mla.q_a.bytes() + mla.kv_a.bytes() + mla.q_b.bytes() + mla.o.bytes() + mla.kvb_q.len() + 4 * mla.kvb_s.len();
            if let Some(ix) = &mla.index {
                bytes += ix.wq_b.bytes() + 2 * ix.wk.len() + 4 * ix.wp.len();
            }
            let layer = Layer {
                ln_in: ld.bf16(&format!("{p}input_layernorm.weight"))?,
                ln_post: ld.bf16(&format!("{p}post_attention_layernorm.weight"))?,
                attn: Attn::Mla(mla),
                hc: None,
                mlp,
                mtp,
            };
            bytes += match &layer.mlp {
                Mlp::Dense { gate_up, down } => gate_up.bytes() + down.bytes(),
                Mlp::Moe { router, sh_gate_up, sh_down, .. } => 2 * router.len() + sh_gate_up.bytes() + sh_down.bytes(),
                Mlp::Routed { router, .. } => 2 * router.len(),
            };
            if let Some(m) = &layer.mtp {
                bytes += m.eh.bytes();
            }
            layers.push(layer);
            if l % 13 == 12 {
                log(&format!("[glm] dense weights: layer {l} ({:.1} GB, {:.0} s)", bytes as f64 / 1e9, t0.elapsed().as_secs_f64()));
            }
        }
        let (cs, sn) = rope_tables(cfg, max_seq);
        Ok(Weights { layers, embed: Embed::Int8 { q: embed_q, s: embed_s }, norm, head: Lin::Exl3(head), draft: None,
                     cos: DeviceBuffer::from_slice(&cs)?, sin: DeviceBuffer::from_slice(&sn)?, bytes })
    }
}
