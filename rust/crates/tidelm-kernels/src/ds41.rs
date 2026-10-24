//! FFI to `cuda/ds41.cu`, the Rust engine's own decode-path kernels. Each declaration
//! mirrors the C ABI one for one; the semantics (and the Python each one reproduces) are
//! documented at the kernel. Pointers are raw device pointers; ownership lives above.

use std::os::raw::{c_int, c_void};

use crate::CudaStream;

extern "C" {
    /// y[m, n] = bf16 or f32 of sum_k x[m, k] * W[n, k]; W fp8 e4m3 with UE8M0 32x32
    /// block scales (`lds` = scale row stride = ceil(K/32)). `splits` > 1 writes fp32
    /// partials `[splits][M][N]` to `partial` (then call `ds_splits_reduce`). M <= 16.
    pub fn ds_linear_fp8(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, s: *const c_void,
        lds: i64, m: c_int, n: c_int, k: c_int, splits: c_int, y_bf16: *mut c_void,
        y_f32: *mut c_void, ldy: i64, partial: *mut c_void,
    ) -> c_int;

    /// The grouped wo_a GEMM: output rows of group g = n / rows_per_group read
    /// x + g * x_group_stride.
    pub fn ds_linear_fp8_grouped(
        stream: CudaStream, x: *const c_void, ldx: i64, x_group_stride: i64, rows_per_group: c_int,
        w: *const c_void, s: *const c_void, lds: i64, m: c_int, n: c_int, k: c_int,
        y_bf16: *mut c_void, ldy: i64,
    ) -> c_int;

    pub fn ds_linear_bf16(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, m: c_int, n: c_int,
        k: c_int, splits: c_int, y_bf16: *mut c_void, y_f32: *mut c_void, ldy: i64,
        partial: *mut c_void,
    ) -> c_int;

    /// fp32 weights, bf16 activations (the reference's `F.linear(x.float(), w_f32)`).
    pub fn ds_linear_f32w(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, m: c_int, n: c_int,
        k: c_int, splits: c_int, y_f32: *mut c_void, ldy: i64, partial: *mut c_void,
    ) -> c_int;

    pub fn ds_splits_reduce(
        stream: CudaStream, partial: *const c_void, splits: c_int, m: c_int, n: c_int,
        y_bf16: *mut c_void, y_f32: *mut c_void, ldy: i64,
    ) -> c_int;

    pub fn ds_rmsnorm(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, y: *mut c_void,
        ldy: i64, m: c_int, d: c_int, eps: f32,
    ) -> c_int;

    /// hc_pre (+ rmsnorm when `w` is non-null): h [M, hc, D], pre [M, hc] -> y [M, D].
    pub fn ds_hc_pre_norm(
        stream: CudaStream, h: *const c_void, pre: *const c_void, w: *const c_void,
        y: *mut c_void, m: c_int, d: c_int, hc: c_int, eps: f32,
    ) -> c_int;

    pub fn ds_row_rsqrt(
        stream: CudaStream, x: *const c_void, ldx: i64, m: c_int, d: c_int, eps: f32,
        out: *mut c_void,
    ) -> c_int;

    pub fn ds_hc_sinkhorn(
        stream: CudaStream, mixes: *const c_void, rsq: *const c_void, scale: *const c_void,
        base: *const c_void, m: c_int, iters: c_int, eps: f32, pre: *mut c_void,
        post: *mut c_void, comb: *mut c_void,
    ) -> c_int;

    /// out may alias residual (in place on the hc stream).
    pub fn ds_hc_post(
        stream: CudaStream, y: *const c_void, residual: *const c_void, post: *const c_void,
        comb: *const c_void, out: *mut c_void, m: c_int, d: c_int,
    ) -> c_int;

    /// RoPE in place on dims [offset, offset + rd) of each of `heads` heads per row;
    /// `table` is float2 [positions][rd/2] (cos, sin).
    pub fn ds_rope(
        stream: CudaStream, x: *mut c_void, ld_row: i64, rows: c_int, heads: c_int,
        ld_head: i64, offset: c_int, rd: c_int, pos: *const c_void, table: *const c_void,
        inverse: c_int,
    ) -> c_int;

    /// z f32 [M, E] gate logits -> idx int32 [M, K], weights f32 [M, K]. `keep` (u8 [E]
    /// or null) masks non-routable experts.
    pub fn ds_router(
        stream: CudaStream, z: *const c_void, bias: *const c_void, keep: *const c_void,
        m: c_int, e: c_int, k: c_int, route_scale: f32, idx: *mut c_void, weights: *mut c_void,
    ) -> c_int;

    pub fn ds_swiglu(
        stream: CudaStream, gate: *const c_void, ldg: i64, up: *const c_void, ldu: i64,
        h: *mut c_void, ldh: i64, m: c_int, n: c_int, limit: f32,
    ) -> c_int;

    pub fn ds_moe_combine(
        stream: CudaStream, parts: *const c_void, k: c_int, m: c_int, shared: *const c_void,
        out: *mut c_void, d: c_int,
    ) -> c_int;

    pub fn ds_attn_decode(
        stream: CudaStream, q: *const c_void, ring: *const c_void, ring_size: c_int,
        pos: *const c_void, w: c_int, win_lo: c_int, ckv: *const c_void, cidx: *const c_void,
        c: c_int, sink: *const c_void, scale: f32, out: *mut c_void, m: c_int, h: c_int,
    ) -> c_int;

    pub fn ds_compress_pair(
        stream: CudaStream, kv_a: *const c_void, sc_a: *const c_void, kv_b: *const c_void,
        sc_b: *const c_void, out: *mut c_void, d: c_int,
    ) -> c_int;

    pub fn ds_fp4_qdq_e4m3(stream: CudaStream, x: *mut c_void, n_values: i64) -> c_int;

    pub fn ds_engram_dequant(
        stream: CudaStream, raw: *const c_void, inv: *const c_void, tokens: c_int, cols: c_int,
        out: *mut c_void,
    ) -> c_int;

    pub fn ds_engram_gate(
        stream: CudaStream, h: *mut c_void, kv: *const c_void, weight: *const c_void, m: c_int,
        d: c_int, hc: c_int, eps: f32, dim_scale: f32,
    ) -> c_int;

    pub fn ds_embed_hc(
        stream: CudaStream, ids: *const c_void, table: *const c_void, h: *mut c_void, m: c_int,
        d: c_int, hc: c_int,
    ) -> c_int;

    pub fn ds_argmax_bf16(
        stream: CudaStream, x: *const c_void, ldx: i64, m: c_int, n: c_int, out: *mut c_void,
    ) -> c_int;
}
