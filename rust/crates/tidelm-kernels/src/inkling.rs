//! FFI to cuda/inkling.cu: Inkling 2's short convolutions, block attention residual and
//! grouped-query attention (a relative-position bias, a sink logit, an output gate; a 1,024-key
//! window or the whole context). Every launcher takes the stream first and returns
//! `cudaGetLastError()` (1: arguments the kernels are not built for). `lanes` is glm.cu's pass table
//! ([L][4]: first row, rows, first position, cache-row offset); `meta` is this file's ([L][4]: the
//! lane first). A key / value cache is bf16 [rows][kl][K 128 | V 128]: a window layer's ring (`ring`
//! positions a lane, a power of two; `lane_rows` = ring) or a flat cache by position (`ring` 0, the
//! lane's part from its table's cache-row offset).

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

/// Positions a lane's conv ring holds (inkling.cu kConvRing).
pub const CONV_RING: usize = 64;
/// Block states a mix sees at most, the current one included (inkling.cu kMaxBlocks).
pub const MAX_BLOCKS: usize = 9;

extern "C" {
    pub fn ink_set_pdl(on: c_int) -> c_int;
    /// ink_mix through its vectorized kernel where the shape allows (on by default).
    pub fn ink_set_mix_v(on: c_int) -> c_int;
    /// y = bf16((sum_lag w(lag) x(t - lag) + x(t)) / div) over `c` channels of x [t][ldx] bf16; rows
    /// before the pass from the lane's ring [lanes][CONV_RING][c]. wmode 0: w [c][4] (conv1d's
    /// order); 1: w [4][c] by lag plus dyn [t][c / 128 x 4] (`ldd` apart). omode 0: out bf16
    /// [t][ldo] = y; 1: res fp32 [t][c] = y; 2: res = h + y (h bf16 [t][c]); 3: h = bf16(res + y).
    pub fn ink_conv(stream: CudaStream, x: *const c_void, ldx: i64, c: c_int, w: *const c_void, wmode: c_int, dyn_w: *const c_void,
                    ldd: i64, ring: *const c_void, lane_stride: i64, lanes: *const c_void, nl: c_int, meta: *const c_void,
                    t: c_int, div: f32, omode: c_int, out: *mut c_void, ldo: i64, res: *mut c_void, h: *mut c_void) -> c_int;
    /// ink_conv with ink_conv_state's ring write in the same launch, for a pass of at most 60 rows
    /// (returns 1 past that: the two launches it needs).
    pub fn ink_conv_fused(stream: CudaStream, x: *const c_void, ldx: i64, c: c_int, w: *const c_void, wmode: c_int, dyn_w: *const c_void,
                    ldd: i64, ring: *mut c_void, lane_stride: i64, lanes: *const c_void, nl: c_int, meta: *const c_void,
                    t: c_int, div: f32, omode: c_int, out: *mut c_void, ldo: i64, res: *mut c_void, h: *mut c_void) -> c_int;
    /// The ring after the pass: each lane's last rows of x at their positions.
    pub fn ink_conv_state(stream: CudaStream, x: *const c_void, ldx: i64, c: c_int, ring: *mut c_void, lane_stride: i64,
                          lanes: *const c_void, nl: c_int, meta: *const c_void) -> c_int;
    /// The block attention residual and the RMSNorm after it: blocks bf16 [t][nb][h] (`ldb` a row)
    /// and cur (bf16, or fp32 when cur_f32) [t][h] mixed by each of `nq` queries (fp32 [nq][h]),
    /// normed with nw (bf16 [nq][h]) -> out bf16 [nq][t][h] (`plane` elements a query).
    pub fn ink_mix(stream: CudaStream, blocks: *const c_void, ldb: i64, nb: c_int, cur: *const c_void, cur_f32: c_int,
                   q: *const c_void, nq: c_int, nw: *const c_void, neps: f32, out: *mut c_void, plane: i64, t: c_int,
                   h: c_int) -> c_int;
    /// proj's query heads (at qoff) and relative states (at roff) scaled, clipped and (alpha > 0)
    /// log-scaled -> qc [t][hl][128], rc [t][hl][16]; the keys kx clipped and the values vx scaled
    /// to unit RMS -> the rows' cache rows.
    pub fn ink_heads(stream: CudaStream, proj: *const c_void, ldp: i64, qoff: c_int, roff: c_int, qscale: *const c_void,
                     rscale: *const c_void, kx: *const c_void, vx: *const c_void, hl: c_int, kl: c_int, qclip: f32, rclip: f32,
                     eps: f32, alpha: f32, floor_n: f32, lanes: *const c_void, nl: c_int, meta: *const c_void, ring: c_int,
                     lane_rows: i64, t: c_int, qc: *mut c_void, rc: *mut c_void, cache: *mut c_void) -> c_int;
    /// Decode rows: each (row, key / value head)'s keys in `ns` splits -> softmax partials pm, pl
    /// fp32 [t][kl][ns][6] and pacc [t][kl][ns][6][128]. window > 0: the last `window` keys.
    pub fn ink_attn_part(stream: CudaStream, qc: *const c_void, rc: *const c_void, proj_t: *const c_void, rel_extent: c_int,
                         cache: *const c_void, ring: c_int, lane_rows: i64, window: c_int, lanes: *const c_void, nl: c_int,
                         meta: *const c_void, hl: c_int, kl: c_int, ns: c_int, t: c_int, pm: *mut c_void, pl: *mut c_void,
                         pacc: *mut c_void) -> c_int;
    /// The splits and the sink logit (proj at soff) into o [t][hl][128], gated (proj at goff).
    pub fn ink_attn_combine(stream: CudaStream, pm: *const c_void, pl: *const c_void, pacc: *const c_void, proj: *const c_void,
                            ldp: i64, soff: c_int, goff: c_int, hl: c_int, kl: c_int, ns: c_int, t: c_int,
                            o: *mut c_void) -> c_int;
    /// Positions lo .. lo + s of lane group z's cache, contiguous at out [s][kl][256].
    pub fn ink_kv_window(stream: CudaStream, cache: *const c_void, ring: c_int, lane_rows: i64, lanes: *const c_void,
                         meta: *const c_void, z: c_int, lo: c_int, s: c_int, kl: c_int, out: *mut c_void) -> c_int;
    /// Prefill rows r0 .. r0 + rb (first position p0 + r0): scores fp32 [hl][rb][s] (q . k, key j at
    /// position lo + j) -> the attention weights bf16 [hl][rb][s] (bias, causal / window mask, sink).
    pub fn ink_pf_softmax(stream: CudaStream, sc: *const c_void, rc: *const c_void, proj_t: *const c_void, rel_extent: c_int,
                          proj: *const c_void, ldp: i64, soff: c_int, hl: c_int, r0: c_int, rb: c_int, s: c_int, lo: c_int,
                          p0: c_int, window: c_int, w: *mut c_void) -> c_int;
    /// o [t][hl][128] rows r0 .. = bf16(ov [hl][rb][128] * sigmoid(gate)).
    pub fn ink_pf_gate(stream: CudaStream, ov: *const c_void, proj: *const c_void, ldp: i64, goff: c_int, hl: c_int, r0: c_int,
                       rb: c_int, o: *mut c_void) -> c_int;
    /// ink_attn_part's partials (same arguments, same layout, for ink_attn_combine) from a fused
    /// split-key kernel: a lane group's rows (16 a block) share each key tile instead of reading
    /// every key once a row.
    pub fn ink_flash_part(stream: CudaStream, qc: *const c_void, rc: *const c_void, proj_t: *const c_void, rel_extent: c_int,
                          cache: *const c_void, ring: c_int, lane_rows: i64, window: c_int, lanes: *const c_void, nl: c_int,
                          meta: *const c_void, hl: c_int, kl: c_int, ns: c_int, t: c_int, pm: *mut c_void, pl: *mut c_void,
                          pacc: *mut c_void) -> c_int;
    /// The fused prefill attention (flash, tensor cores) of rows 0 .. t of one lane at positions p0 ..:
    /// keys lo_key .. each row's position (window > 0: its last `window`), kv the row of key lo_key
    /// in [keys][kl][K | V] rows rowk elements apart; bias, sink and gate as ink_attn_combine's, o
    /// [t][hl][128]. What ink_pf_softmax + the two GEMMs + ink_pf_gate compute, without the scores.
    pub fn ink_pf_flash(stream: CudaStream, qc: *const c_void, rc: *const c_void, proj_t: *const c_void, rel_extent: c_int,
                        kv: *const c_void, rowk: i64, lo_key: c_int, proj: *const c_void, ldp: i64, soff: c_int, goff: c_int,
                        hl: c_int, kl: c_int, t: c_int, p0: c_int, window: c_int, o: *mut c_void) -> c_int;
    /// Positions p0 .. p0 + n of a lane's rings (`ring`: the lane's part of the first layer's ring
    /// [r][row] bf16, `ring_ls` elements a layer) into lin [layers][n][row] (`lin_ls` a layer), or
    /// back (to_ring): a snapshot and its restore.
    pub fn ink_ring_copy(stream: CudaStream, ring: *mut c_void, ring_ls: i64, r: c_int, row: c_int, p0: c_int, n: c_int,
                         layers: c_int, lin: *mut c_void, lin_ls: i64, to_ring: c_int) -> c_int;
    /// logits fp32 [t][n] (`ld` a row) in place: cap * tanh(x / cap), -inf where id0 + column >= real.
    pub fn ink_logits(stream: CudaStream, x: *mut c_void, ld: i64, n: c_int, t: c_int, cap: f32, id0: c_int, real: c_int) -> c_int;
}
