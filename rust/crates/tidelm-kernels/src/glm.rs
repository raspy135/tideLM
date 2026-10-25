//! FFI to cuda/glm.cu, the GLM-5.3 engine's own kernels (engine/glm's Triton and PyTorch ops).
//! Every launcher takes the stream first and returns `cudaGetLastError()`.

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

extern "C" {
    pub fn glm_embed(stream: CudaStream, ids: *const c_void, t: c_int, q: *const c_void, sc: *const c_void, lo: c_int,
                     vl: c_int, h: c_int, pos: *const c_void, out: *mut c_void) -> c_int;
    pub fn glm_sum2(stream: CudaStream, g: *const c_void, n: i64, out_bf16: *mut c_void, out_f32: *mut c_void) -> c_int;
    pub fn glm_f32_bf16(stream: CudaStream, x: *const c_void, y: *mut c_void, n: i64) -> c_int;
    pub fn glm_unshard(stream: CudaStream, src: *const c_void, rank_stride: i64, dst: *mut c_void, m: c_int, n: c_int,
                       world: c_int, esz: c_int) -> c_int;
    pub fn glm_rmsnorm(stream: CudaStream, h: *mut c_void, ldh: i64, a: *const c_void, lda: i64, w: *const c_void,
                       y: *mut c_void, ldy: i64, t: c_int, n: c_int, eps: f32) -> c_int;
    pub fn glm_rope(stream: CudaStream, x: *const c_void, xt: i64, xh: i64, y: *mut c_void, yt: i64, yh: i64, t: c_int,
                    h: c_int, cos: *const c_void, sin: *const c_void, pos: *const c_void) -> c_int;
    pub fn glm_kv_fp8(stream: CudaStream, x: *const c_void, ldx: i64, rows: *const c_void, q: *mut c_void, s: *mut c_void,
                      t: c_int, d: c_int) -> c_int;
    pub fn glm_rows_copy(stream: CudaStream, x: *const c_void, ldx: i64, rows: *const c_void, out: *mut c_void, t: c_int,
                         d: c_int) -> c_int;
    pub fn glm_positions(stream: CudaStream, pos0: *const c_void, t: c_int, rank: c_int, world: c_int, spare: c_int,
                         pos: *mut c_void, rows: *mut c_void, bound: *mut c_void) -> c_int;
    pub fn glm_index_key(stream: CudaStream, k: *const c_void, ldk: i64, w: *const c_void, b: *const c_void,
                         cos: *const c_void, sin: *const c_void, pos: *const c_void, rows: *const c_void, q: *mut c_void,
                         s: *mut c_void, t: c_int) -> c_int;
    pub fn glm_index_scores(stream: CudaStream, q: *const c_void, w: *const c_void, wscale: f32, kq: *const c_void,
                            ks: *const c_void, pos: *const c_void, t: c_int, nmax: c_int, stride: c_int, off: c_int,
                            split_n: c_int, inv_d: f32, out: *mut c_void, ldo: i64) -> c_int;
    pub fn glm_topk(stream: CudaStream, s: *const c_void, lds: i64, pos: *const c_void, stride: c_int, off: c_int,
                    nmax: c_int, k: c_int, t: c_int, idx: *mut c_void, ldi: i64, val: *mut c_void, ldv: i64) -> c_int;
    pub fn glm_split_share(stream: CudaStream, g: *const c_void, rank_stride: i64, row_stride: i64, val_off: i64,
                           idx: *mut c_void, ldi: i64, rank: c_int, k: c_int, t: c_int) -> c_int;
    pub fn glm_absorb_q(stream: CudaStream, q: *const c_void, ldq_t: i64, ldq_h: i64, w: *const c_void, ws: *const c_void,
                        out: *mut c_void, ldo_t: i64, ldo_h: i64, t: c_int, hl: c_int) -> c_int;
    pub fn glm_absorb_o(stream: CudaStream, o: *const c_void, ldo_t: i64, ldo_h: i64, w: *const c_void, ws: *const c_void,
                        out: *mut c_void, ld_out: i64, t: c_int, hl: c_int) -> c_int;
    pub fn glm_kvb_dequant(stream: CudaStream, w: *const c_void, ws: *const c_void, out: *mut c_void, rows: i64) -> c_int;
    pub fn glm_mla_smem(mt: c_int) -> c_int;
    pub fn glm_mla(stream: CudaStream, mt: c_int, q: *const c_void, q_ldg: i64, q_ldt: i64, q_ldh: i64, hg: c_int,
                   cq: *const c_void, cs: *const c_void, cr: *const c_void, split_n: c_int, idx: *const c_void,
                   ld_idx: i64, k: c_int, bound: *const c_void, t: c_int, h: c_int, scale: f32, nsplit: c_int,
                   chunk: c_int, part_m: *mut c_void, part_l: *mut c_void, part_acc: *mut c_void, out: *mut c_void,
                   o_ldt: i64, o_ldh: i64) -> c_int;
    pub fn glm_mla_combine(stream: CudaStream, pm: *const c_void, pl: *const c_void, pacc: *const c_void, t: c_int,
                           h: c_int, ns: c_int, out: *mut c_void, o_ldt: i64, o_ldh: i64, own_o: *mut c_void,
                           own_l: *mut c_void, send: *mut c_void, row_bytes: i64, hl: c_int, lo: c_int) -> c_int;
    pub fn glm_mla_merge(stream: CudaStream, own_o: *const c_void, own_l: *const c_void, recv: *const c_void,
                         row_bytes: i64, t: c_int, hl: c_int, out: *mut c_void) -> c_int;
    pub fn glm_router(stream: CudaStream, logits: *const c_void, bias: *const c_void, allowed: *const c_void,
                      lut: *const c_void, t: c_int, k: c_int, scale: f32, ids: *mut c_void, w: *mut c_void,
                      resident: *const c_void, hot: *mut c_void, ex_ids: *const c_void, ex_w: *const c_void,
                      demand: *mut c_void, acc: *mut c_void, live: *const c_void) -> c_int;
    pub fn glm_swiglu(stream: CudaStream, g: *const c_void, ldg: i64, u: *const c_void, ldu: i64, y: *mut c_void, ldy: i64,
                      t: c_int, n: c_int) -> c_int;
    pub fn glm_moe_out(stream: CudaStream, routed: *const c_void, shared: *const c_void, y: *mut c_void, n: i64) -> c_int;
    pub fn glm_argmax(stream: CudaStream, x: *const c_void, ldx: i64, m: c_int, n: c_int, offset: c_int,
                      pairs: *mut c_void) -> c_int;
    pub fn glm_argmax_combine(stream: CudaStream, pairs: *const c_void, world: c_int, m: c_int, out: *mut c_void) -> c_int;
    pub fn glm_exl3_unpack(stream: CudaStream, trellis: *const c_void, k: c_int, n: c_int, bits: c_int,
                           w: *mut c_void) -> c_int;
    pub fn glm_member_codes(stream: CudaStream, pair: *const c_void, n: c_int, k: c_int, out: *mut c_void) -> c_int;
    pub fn glm_count_live(stream: CudaStream, block_slot: *const c_void, nb: c_int, out: *mut c_void) -> c_int;
    pub fn glm_quant_kvb(stream: CudaStream, w: *const c_void, rows: c_int, q: *mut c_void, s_rows: *mut c_void) -> c_int;
    pub fn glm_quant_int8_rows(stream: CudaStream, x: *const c_void, rows: c_int, h: c_int, q: *mut c_void,
                               s: *mut c_void) -> c_int;
    pub fn glm_dup_rows(stream: CudaStream, x: *const c_void, ldx: i64, out: *mut c_void, t: c_int, k: c_int,
                        g: c_int) -> c_int;
}
