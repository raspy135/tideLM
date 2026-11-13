//! A linear layer in whichever format a checkpoint's matrix is served in: EXL3 (GLM-5.3's dense
//! weights, run by weights::Exl3Mat), FP8 E4M3 with 32 x 32 UE8M0 block scales, or BF16 (the
//! latter two through ds41.cu's GEMVs and tensor-core GEMMs, the DS engine's kernels: its gemm()'s
//! dispatch, here for any engine). GLM-5.3-Flash serves its BF16 non-expert weights as FP8
//! (GLM_FLASH_DENSE=fp8, the default) or as they are (bf16, the reference).
//!
//! A gate | up matrix's output: an EXL3 matrix of two groups gives [rows][2N] (decode) or
//! [2][rows][N] (prefill, past 64 rows); FP8 and BF16 matrices are one [2N][K] matrix, so always
//! [rows][2N]. `gu_layout` says which.

use std::os::raw::c_void;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::ds41::{ds_linear_bf16, ds_linear_fp8, ds_mma_bf16, ds_mma_fp8, ds_mma_rows_per_block, ds_quant_fp8_block32,
                           ds_splits_reduce};

use crate::weights::Exl3Mat;
use crate::{ck, Result};

/// Rows up to which the FMA GEMV runs (ds41.cu's skinny kernel); more take the tensor-core GEMM.
pub const FMA_ROWS: usize = 16;

/// FP8 E4M3 [n][k] with a UE8M0 scale a 32 x 32 block ([n / 32][k / 32]).
pub struct Fp8Mat {
    pub w: DeviceBuffer<u8>,
    pub s: DeviceBuffer<u8>,
    pub n: usize,
    pub k: usize,
    /// split-K of decode calls (flash_gpu fp8_gemv_bench: 1 for the large matrices, more for
    /// the narrow long-K ones); it fixes their summation order
    pub split: usize,
}

/// BF16 [n][k].
pub struct Bf16Mat {
    pub w: DeviceBuffer<u16>,
    pub n: usize,
    pub k: usize,
    pub split: usize,
}

pub enum Lin {
    Exl3(Exl3Mat),
    Fp8(Fp8Mat),
    Bf16(Bf16Mat),
}

impl Lin {
    /// Output columns a row.
    pub fn n_total(&self) -> usize {
        match self {
            Lin::Exl3(m) => m.g * m.n,
            Lin::Fp8(m) => m.n,
            Lin::Bf16(m) => m.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Lin::Exl3(m) => m.k,
            Lin::Fp8(m) => m.k,
            Lin::Bf16(m) => m.k,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Lin::Exl3(m) => m.bytes(),
            Lin::Fp8(m) => m.w.len() + m.s.len(),
            Lin::Bf16(m) => 2 * m.w.len(),
        }
    }

    pub fn exl3(&self) -> Option<&Exl3Mat> {
        if let Lin::Exl3(m) = self { Some(m) } else { None }
    }

    pub fn exl3_mut(&mut self) -> Option<&mut Exl3Mat> {
        if let Lin::Exl3(m) = self { Some(m) } else { None }
    }

    /// A gate | up matrix's output over `t` rows: (columns a half, the up half's offset in
    /// elements, the row stride).
    pub fn gu_layout(&self, t: usize, gemv_rows: usize) -> (usize, usize, usize) {
        match self {
            Lin::Exl3(m) if t > gemv_rows => (m.n, t * m.n, m.n),
            Lin::Exl3(m) => (m.n, m.n, 2 * m.n),
            _ => {
                let h = self.n_total() / 2;
                (h, h, 2 * h)
            }
        }
    }

    /// An L2 prefetch of the matrix's first bytes: (base, bytes a run, run stride, runs), the runs
    /// where a decode call's warps start reading.
    pub fn runs(&self, rows: usize) -> (u64, u64, usize) {
        match self {
            Lin::Exl3(m) => m.warp_runs(rows),
            Lin::Fp8(m) => (m.w.raw() as u64, m.w.len() as u64, 1),
            Lin::Bf16(m) => (m.w.raw() as u64, 2 * m.w.len() as u64, 1),
        }
    }
}

/// The tensor-core GEMM's split-K (tidelm_engine::forward::mma_splits): blocks to cover ~96
/// slots, K in pieces of at least 512, within the partials buffer.
pub fn mma_splits(m: usize, n: usize, k: usize, part_len: usize) -> usize {
    let mtiles = if m > 64 { m.div_ceil(128) } else { 1 };
    let blocks = n.div_ceil(unsafe { ds_mma_rows_per_block(m as i32) } as usize) * mtiles;
    96usize.div_ceil(blocks).min(k / 512).min(part_len / (m * n)).max(1)
}

/// y [m][ldy] = x [m][..k] W^T for an FP8 or BF16 matrix, bf16 out (fp32 with `f32_out`): up to
/// FMA_ROWS rows the FMA GEMV at the matrix's split, more the tensor-core GEMM (K % 64 == 0).
/// `part`: split-K partials (fp32), at least split x m x n.
#[allow(clippy::too_many_arguments)]
pub fn gemm(st: &Stream, part: &DeviceBuffer<f32>, x: *const c_void, ldx: usize, w: &Lin, m: usize, y: *mut c_void,
            f32_out: bool, ldy: usize) -> Result<()> {
    let sr = st.raw();
    let null = std::ptr::null_mut();
    let (yb, yf) = if f32_out { (null, y) } else { (y, null) };
    let (n, k) = (w.n_total(), w.k());
    let tc = m > FMA_ROWS && k % 64 == 0;
    let splits = if tc { mma_splits(m, n, k, part.len()) } else {
        let s = match w { Lin::Fp8(f) => f.split, Lin::Bf16(b) => b.split, Lin::Exl3(_) => 1 };
        if s > 1 && s * m * n <= part.len() { s } else { 1 }
    };
    let pp = if splits > 1 { part.raw() } else { null };
    let (ob, of) = if splits > 1 { (pp, null) } else { (yb, yf) };
    ck(unsafe {
        match (w, tc) {
            (Lin::Fp8(f), true) => ds_mma_fp8(sr, x, ldx as i64, 0, 0, f.w.raw(), f.s.raw(), k.div_ceil(32) as i64, m as i32, n as i32,
                                              k as i32, splits as i32, ob, of, ldy as i64, pp),
            (Lin::Bf16(b), true) => ds_mma_bf16(sr, x, ldx as i64, b.w.raw(), m as i32, n as i32, k as i32, splits as i32, ob, of,
                                                ldy as i64, pp),
            (Lin::Fp8(f), false) => ds_linear_fp8(sr, x, ldx as i64, f.w.raw(), f.s.raw(), k.div_ceil(32) as i64, m as i32, n as i32,
                                                  k as i32, splits as i32, ob, of, ldy as i64, pp),
            (Lin::Bf16(b), false) => ds_linear_bf16(sr, x, ldx as i64, b.w.raw(), m as i32, n as i32, k as i32, splits as i32, ob, of,
                                                    ldy as i64, pp),
            (Lin::Exl3(_), _) => return Err("lin::gemm on an EXL3 matrix (Engine::lin_on runs those)".into()),
        }
    }, "lin gemm")?;
    if splits > 1 {
        ck(unsafe { ds_splits_reduce(sr, pp, splits as i32, m as i32, n as i32, yb, yf, ldy as i64) }, "lin reduce")?;
    }
    Ok(())
}

/// A BF16 [n][k] matrix (host bytes, little-endian) as FP8 with 32 x 32 block scales: uploaded to a
/// staging buffer, quantized on the GPU (ds_quant_fp8_block32), the staging freed.
pub fn fp8_from_bf16(st: &Stream, w: &[u16], n: usize, k: usize, split: usize) -> Result<Fp8Mat> {
    if w.len() != n * k || n % 32 != 0 || k % 32 != 0 {
        return Err(format!("fp8_from_bf16: {} values for {n} x {k} (32-aligned)", w.len()).into());
    }
    let stage = DeviceBuffer::from_slice(w)?;
    let codes = DeviceBuffer::<u8>::new(n * k)?;
    let scales = DeviceBuffer::<u8>::new((n / 32) * (k / 32))?;
    ck(unsafe { ds_quant_fp8_block32(st.raw(), stage.raw(), n as i32, k as i32, codes.raw(), scales.raw()) }, "quant fp8")?;
    st.synchronize()?;
    Ok(Fp8Mat { w: codes, s: scales, n, k, split })
}
