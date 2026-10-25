//! FFI to `cuda/vision.cu`, the vision tower's kernels (the checkpoint's inference/vision.py
//! around ds41's GEMMs). Pointers are raw device pointers, bf16 unless noted.

use std::os::raw::{c_int, c_void};

use crate::CudaStream;

extern "C" {
    /// Attention over one image: qkv [n][3][h][64] (wqkv's output, RoPE applied), out
    /// [n][h][64]; softmax(q k^T / 8) v per head, no mask.
    pub fn ds_vit_attn(stream: CudaStream, qkv: *const c_void, n: c_int, h: c_int, out: *mut c_void) -> c_int;

    /// 2D RoPE on q and k of qkv in place; cos / sin f32 [n][32] (row angles, then column).
    pub fn ds_vit_rope(stream: CudaStream, qkv: *mut c_void, n: c_int, h: c_int, cos: *const c_void, sin: *const c_void)
                       -> c_int;

    /// out = [bf16(residual + )] [gelu] bf16(y + bias): y f32 [m][n]; bias (or null) [n];
    /// residual (or null) [m][n], may be out.
    pub fn ds_vit_epilogue(stream: CudaStream, y: *const c_void, bias: *const c_void, residual: *const c_void,
                           out: *mut c_void, m: c_int, n: c_int, gelu: c_int) -> c_int;

    /// out [m][i] = bf16(bf16(silu(gu[m][..i])) * gu[m][i..2i])
    pub fn ds_vit_silu_mul(stream: CudaStream, gu: *const c_void, m: c_int, i: c_int, out: *mut c_void) -> c_int;

    /// The aligner's 3x3 unfold of x [nh * nw][c]: out [ceil(nh/3) * ceil(nw/3)][c * 9]
    /// (channel-major, zero past the grid).
    pub fn ds_vit_unfold(stream: CudaStream, x: *const c_void, nh: c_int, nw: c_int, c: c_int, out: *mut c_void) -> c_int;
}
