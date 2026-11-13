//! FFI to cuda/flash.cu: GLM-5.3-Flash's KDA (Kimi Delta Attention) layers and its indexer's 4:1
//! key pools. Every launcher takes the stream first and returns `cudaGetLastError()`. `lanes` is
//! glm.cu's pass table ([L][4]: first row, rows, first position, cache-row offset); `meta` is this
//! file's ([L][4]: lane, KDA state slot read, slot written, pool caches' row offset).

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

extern "C" {
    pub fn flash_set_pdl(on: c_int) -> c_int;
    /// threads a KDA value column, 2 (default) or 4: the chain and the replay take it alike
    pub fn flash_set_kda_split(sp: c_int) -> c_int;
    pub fn flash_kda_conv(stream: CudaStream, raw: *const c_void, ldr: i64, w: *const c_void, c: c_int, conv: *const c_void,
                          slot_stride: i64, lanes: *const c_void, nl: c_int, meta: *const c_void, t: c_int, out: *mut c_void,
                          ldo: i64, save_raw: *mut c_void, save_lane: i64, r: c_int) -> c_int;
    pub fn flash_kda_conv_state(stream: CudaStream, raw: *const c_void, ldr: i64, c: c_int, conv: *mut c_void, slot_stride: i64,
                                lanes: *const c_void, nl: c_int, meta: *const c_void) -> c_int;
    pub fn flash_kda_conv_replay(stream: CudaStream, conv: *mut c_void, layer_stride: i64, slot_stride: i64, c: c_int,
                                 layers: c_int, sin: c_int, sout: c_int, save_raw: *const c_void, save_layer: i64,
                                 keep: c_int) -> c_int;
    pub fn flash_kda_chain(stream: CudaStream, qkv: *const c_void, ldq: i64, dl: c_int, f: *const c_void, ldf: i64,
                           b: *const c_void, ldb: i64, gt: *const c_void, ldg: i64, a_log: *const c_void,
                           dt_bias: *const c_void, onorm: *const c_void, lower: f32, eps: f32, state: *mut c_void,
                           slot_stride: i64, lanes: *const c_void, nl: c_int, meta: *const c_void, y: *mut c_void, ldy: i64,
                           save_k: *mut c_void, save_g: *mut c_void, save_v: *mut c_void, save_b: *mut c_void,
                           save_lane: i64, r: c_int, hl: c_int) -> c_int;
    pub fn flash_kda_replay(stream: CudaStream, state: *mut c_void, layer_stride: i64, slot_stride: i64, layers: c_int,
                            sin: c_int, sout: c_int, save_k: *const c_void, save_g: *const c_void, save_v: *const c_void,
                            save_b: *const c_void, save_layer: i64, keep: c_int, hl: c_int) -> c_int;
    pub fn flash_ln128(stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, b: *const c_void, y: *mut c_void,
                       ldy: i64, t: c_int, eps: f32) -> c_int;
    pub fn flash_pool(stream: CudaStream, k: *const c_void, ldk: i64, g: *const c_void, ldg: i64, ape: *const c_void,
                      ring: *const c_void, ring_lane: i64, pos: *const c_void, lanes: *const c_void, nl: c_int,
                      meta: *const c_void, t: c_int, pq: *mut c_void, ps: *mut c_void) -> c_int;
    pub fn flash_ring(stream: CudaStream, k: *const c_void, ldk: i64, g: *const c_void, ldg: i64, ring: *mut c_void,
                      ring_lane: i64, pos: *const c_void, lanes: *const c_void, nl: c_int, meta: *const c_void,
                      t: c_int) -> c_int;
    /// y = bf16(bf16(silu(min(g, lim))) * clamp(u, -lim, lim)); g, u rows `ld` apart
    pub fn flash_swiglu(stream: CudaStream, g: *const c_void, u: *const c_void, ld: i64, y: *mut c_void, ldy: i64, t: c_int, n: c_int,
                        lim: f32) -> c_int;
    /// BF16 embedding rows into `copies` copies a row, zero where pos (non-null) is 0
    pub fn flash_embed(stream: CudaStream, ids: *const c_void, t: c_int, table: *const c_void, h: c_int, copies: c_int,
                       pos: *const c_void, out: *mut c_void) -> c_int;
    pub fn flash_widen(stream: CudaStream, x: *const c_void, ldx: i64, y: *mut c_void, ldy: i64, t: c_int, n: c_int) -> c_int;
    /// the vision tower's per-head q / k RMSNorm of qkv [n][3][h][64] in place (two roundings)
    pub fn flash_vit_qknorm(stream: CudaStream, qkv: *mut c_void, n: c_int, h: c_int, wq: *const c_void, wk: *const c_void,
                            eps: f32) -> c_int;
    /// out [m][n] = gelu(bf16(LayerNorm(x; w, b, eps))), exact gelu, n % 256 == 0
    pub fn flash_vit_ln_gelu(stream: CudaStream, x: *const c_void, w: *const c_void, b: *const c_void, out: *mut c_void, m: c_int,
                             n: c_int, eps: f32) -> c_int;
    pub fn flash_expand(stream: CudaStream, pools: *const c_void, ldp: i64, kp: c_int, pos: *const c_void, out: *mut c_void,
                        ldo: i64, t: c_int) -> c_int;
}
