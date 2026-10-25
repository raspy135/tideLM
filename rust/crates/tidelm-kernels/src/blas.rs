//! The few cuBLAS calls the GLM engine's prefill makes: an EXL3 matrix unpacked to fp16 times the
//! rotated activations (exl3_attn.Exl3Lin._prefill's torch.mm), the absorbed MLA products per head
//! (torch.einsum / bmm), and fp32 GEMMs (the router, the indexer's head weights). Row-major
//! callers: `gemm_rm` computes y[M][N] = x[M][K] . W with W given row-major [K][N] (or [N][K]
//! transposed), as cuBLAS's column-major y^T = W^T x^T.

use crate::CudaStream;
use std::os::raw::{c_int, c_longlong, c_void};

pub type Handle = *mut c_void;

pub const R_32F: c_int = 0;
pub const R_16F: c_int = 2;
pub const R_16BF: c_int = 14;
pub const COMPUTE_32F: c_int = 68;
pub const GEMM_DEFAULT: c_int = -1;
pub const OP_N: c_int = 0;
pub const OP_T: c_int = 1;

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut Handle) -> c_int;
    pub fn cublasDestroy_v2(handle: Handle) -> c_int;
    pub fn cublasSetStream_v2(handle: Handle, stream: CudaStream) -> c_int;
    pub fn cublasSetWorkspace_v2(handle: Handle, workspace: *mut c_void, bytes: usize) -> c_int;
    pub fn cublasGemmEx(handle: Handle, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void,
                        a: *const c_void, atype: c_int, lda: c_int, b: *const c_void, btype: c_int, ldb: c_int,
                        beta: *const c_void, c: *mut c_void, ctype: c_int, ldc: c_int, compute: c_int,
                        algo: c_int) -> c_int;
    pub fn cublasGemmStridedBatchedEx(handle: Handle, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int,
                                      alpha: *const c_void, a: *const c_void, atype: c_int, lda: c_int,
                                      stride_a: c_longlong, b: *const c_void, btype: c_int, ldb: c_int,
                                      stride_b: c_longlong, beta: *const c_void, c: *mut c_void, ctype: c_int,
                                      ldc: c_int, stride_c: c_longlong, batch: c_int, compute: c_int,
                                      algo: c_int) -> c_int;
}

/// A cuBLAS handle bound to one stream.
pub struct Blas {
    h: Handle,
}

unsafe impl Send for Blas {}

fn ck(rc: c_int, what: &str) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(format!("cuBLAS {what}: status {rc}")) }
}

/// One operand: a device pointer, its cuBLAS type and leading dimension (row-major row stride).
#[derive(Clone, Copy)]
pub struct Mat {
    pub ptr: *const c_void,
    pub ty: c_int,
    pub ld: i64,
}

impl Blas {
    pub fn new(stream: CudaStream) -> Result<Self, String> {
        let mut h: Handle = std::ptr::null_mut();
        ck(unsafe { cublasCreate_v2(&mut h) }, "create")?;
        ck(unsafe { cublasSetStream_v2(h, stream) }, "set stream")?;
        Ok(Blas { h })
    }

    /// y[M][N] (row-major, ldy) = x[M][K] (row-major) . W, fp32 accumulation; W row-major
    /// [K][N] (`w_t` false) or [N][K] (`w_t` true: x . W^T, F.linear's weight).
    pub fn gemm_rm(&self, m: usize, n: usize, k: usize, x: Mat, w: Mat, w_t: bool, y: *mut c_void, y_ty: c_int,
                   ldy: i64) -> Result<(), String> {
        let (alpha, beta) = (1f32, 0f32);
        // column-major: y^T [N][M] = op(W)^T-ish: A = W (as col-major [N][K] when row-major [K][N]), B = x^T
        let transa = if w_t { OP_T } else { OP_N };
        ck(unsafe {
            cublasGemmEx(self.h, transa, OP_N, n as c_int, m as c_int, k as c_int, &alpha as *const f32 as *const c_void,
                         w.ptr, w.ty, w.ld as c_int, x.ptr, x.ty, x.ld as c_int, &beta as *const f32 as *const c_void, y,
                         y_ty, ldy as c_int, COMPUTE_32F, GEMM_DEFAULT)
        }, "gemm")
    }

    /// `gemm_rm` over `batch` operands at fixed strides (elements): the absorbed MLA per head.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_rm_batched(&self, m: usize, n: usize, k: usize, x: Mat, sx: i64, w: Mat, sw: i64, w_t: bool,
                           y: *mut c_void, y_ty: c_int, ldy: i64, sy: i64, batch: usize) -> Result<(), String> {
        let (alpha, beta) = (1f32, 0f32);
        let transa = if w_t { OP_T } else { OP_N };
        ck(unsafe {
            cublasGemmStridedBatchedEx(self.h, transa, OP_N, n as c_int, m as c_int, k as c_int,
                                       &alpha as *const f32 as *const c_void, w.ptr, w.ty, w.ld as c_int, sw, x.ptr,
                                       x.ty, x.ld as c_int, sx, &beta as *const f32 as *const c_void, y, y_ty,
                                       ldy as c_int, sy, batch as c_int, COMPUTE_32F, GEMM_DEFAULT)
        }, "batched gemm")
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.h) };
    }
}
