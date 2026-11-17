//! FFI to cuda/inkling_draft.cu: Inkling 2's drafter (mtp.safetensors, "DSpark") - the target
//! features' mix, the context and block head vectors, the block's attention, its norms, the Markov
//! head's steps and the load-time fold. Every launcher takes the stream first and returns
//! `cudaGetLastError()` (1: arguments the kernels are not built for). `lanes` / `meta` are the
//! target pass's tables (inkling.rs). A key / value ring is bf16 [positions][8 heads][K 128 | V 128],
//! slot = position & (ring - 1), `ring` a power of two.

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

/// Rows of a draft block: the anchor and 7 masks (inkling_draft.cu kRows).
pub const BLOCK: usize = 8;
/// Positions a lane's f_rnn ring holds (inkling_draft.cu kFRing).
pub const F_RING: usize = 64;

extern "C" {
    /// Programmatic dependent launch on every launch of inkling_draft.cu: off until set.
    pub fn inkd_set_pdl(on: c_int) -> c_int;
    /// out = bf16(m * rsqrt(sum(m^2) / h + eps) * w), m = sum_tap taps[tap][row] * alpha[tap] (fp32):
    /// taps 8 planes `plane` elements apart, each bf16 [t][h]; alpha bf16 [8]. mode 0: out [t][h];
    /// mode 1: out the lanes' f_rnn rings [lanes][F_RING][h], a row at its position's slot.
    pub fn inkd_mix(stream: CudaStream, taps: *const c_void, plane: i64, alpha: *const c_void, w: *const c_void, out: *mut c_void,
                    mode: c_int, lanes: *const c_void, nl: c_int, meta: *const c_void, t: c_int, h: c_int, eps: f32) -> c_int;
    /// kv bf16 [t][k 8 x 128 | v 8 x 128] (`ldkv` a row) -> the rows' ring rows of cache
    /// [lanes][ring][8][256]: keys normed (knorm [128]) and rotated at the row's position (inv fp32
    /// [64]), values as they are.
    pub fn inkd_ctx_kv(stream: CudaStream, kv: *const c_void, ldkv: i64, knorm: *const c_void, inv: *const c_void, eps: f32,
                       lanes: *const c_void, nl: c_int, meta: *const c_void, ring: c_int, cache: *mut c_void, t: c_int) -> c_int;
    /// The block's qkv bf16 [8][q 48 x 128 | k 8 x 128 | v 8 x 128] -> qc [8][48][128] and one
    /// lane's ring rows (cache: that lane's [ring][8][256]) of positions p0 .. p0 + 7.
    pub fn inkd_heads(stream: CudaStream, qkv: *const c_void, ld: i64, qnorm: *const c_void, knorm: *const c_void,
                      inv: *const c_void, eps: f32, p0: c_int, ring: c_int, cache: *mut c_void, qc: *mut c_void) -> c_int;
    /// The block's 8 rows over the keys of positions [lo, hi) of one lane's ring, in `ns` splits ->
    /// softmax partials pm, pl fp32 [8][8][ns][6] and pacc [8][8][ns][6][128].
    pub fn inkd_attn_part(stream: CudaStream, qc: *const c_void, cache: *const c_void, ring: c_int, lo: c_int, hi: c_int,
                          ns: c_int, pm: *mut c_void, pl: *mut c_void, pacc: *mut c_void) -> c_int;
    /// The splits into o bf16 [8][48][128].
    pub fn inkd_attn_combine(stream: CudaStream, pm: *const c_void, pl: *const c_void, pacc: *const c_void, ns: c_int,
                             o: *mut c_void) -> c_int;
    /// x bf16 [8][h]: row `anchor` of embed, then mask 7 times.
    pub fn inkd_embed(stream: CudaStream, embed: *const c_void, anchor: c_int, mask: *const c_void, x: *mut c_void,
                      h: c_int) -> c_int;
    /// s = hbuf (+ a, then hbuf = bf16(s)); y = bf16(s * rsqrt(mean(s^2) + eps) * w): the norm of the
    /// unrounded sum, one rounding. a null: no add. All bf16 [t][h].
    pub fn inkd_norm(stream: CudaStream, hbuf: *mut c_void, a: *const c_void, w: *const c_void, y: *mut c_void, t: c_int,
                     h: c_int, eps: f32) -> c_int;
    /// zx bf16 [7][hidden row 1 + i | f] (f bf16 [h], or null for zeros).
    pub fn inkd_rnn_x(stream: CudaStream, hidden: *const c_void, f: *const c_void, zx: *mut c_void, h: c_int) -> c_int;
    /// zs bf16 [state 256 | w1[prev] 256]: prev = anchor at step 0 (and a zero state), toks[step - 1]
    /// (device i32) after.
    pub fn inkd_rnn_in(stream: CudaStream, state: *const c_void, w1: *const c_void, toks: *const c_void, step: c_int,
                       anchor: c_int, vocab: c_int, zs: *mut c_void) -> c_int;
    /// The cell: raw = bf16(xs + ss + bias) (fp32 [768] each, bias bf16) -> state (bf16 [256], in
    /// place; zero before step 0 when `first`) and latent bf16 [256].
    pub fn inkd_rnn_cell(stream: CudaStream, xs: *const c_void, ss: *const c_void, bias: *const c_void, state: *mut c_void,
                         latent: *mut c_void, first: c_int) -> c_int;
    /// Blocks of inkd_step_part over n ids: `part` holds 3 times as many fp32.
    pub fn inkd_step_blocks(n: c_int) -> c_int;
    /// A step's logits y = bf16(cap * tanh(bf16(base + bf16(fb + lw)) / cap)), -inf from id `real`,
    /// reduced a block of ids (maximum, its lowest id, sum exp(y - max)) -> part. base, fb bf16 [n],
    /// lw fp32 [n]; ylog fp32 [n] or null.
    pub fn inkd_step_part(stream: CudaStream, base: *const c_void, lw: *const c_void, fb: *const c_void, n: c_int, real: c_int,
                          cap: f32, part: *mut c_void, ylog: *mut c_void) -> c_int;
    /// part -> toks[step] (i32: the argmax, the lowest id on ties) and probs[step] (fp32: its
    /// softmax probability).
    /// `ids` (or null): a listed head's index -> its token id, i32 [n].
    pub fn inkd_step_pick(stream: CudaStream, part: *const c_void, n: c_int, step: c_int, toks: *mut c_void,
                          probs: *mut c_void, ids: *const c_void) -> c_int;
    /// fw bf16 [n][r] = bf16(u w), fb bf16 [n] = bf16(u b) in fp64: u bf16 [n][h] (rows of the
    /// unembed), w bf16 [h][r], b bf16 [h]; h 4608, r 256.
    pub fn inkd_fold(stream: CudaStream, u: *const c_void, n: c_int, h: c_int, r: c_int, w: *const c_void, b: *const c_void,
                     fw: *mut c_void, fb: *mut c_void) -> c_int;
}
