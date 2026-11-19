//! FFI to `tools/exl3_moe_cuda.cu`, the EXL3 trellis kernels the Python GLM engine JITs
//! (tools/exl3_moe_cuda.py::_library), compiled here from the same source with the same flags.
//!
//! The launchers return `void`: check `cudaGetLastError` after a launch. Bad shapes hit the
//! source's `EXL3_CHECK` (`fprintf` + `exit(2)` in this process) and a K2 the kernels were not
//! instantiated for traps on the GPU, so callers validate shapes before they launch (the GLM
//! engine's weights loader does, once). ABI naming kept from the source: `slots` is the per-row
//! pick count (top-k, or G for a dense call) and `e` (the arena size) is unused. Member codes
//! are `row * 32 + k`. Trellis pointers are per-slot device addresses (int64 tables); the
//! scale pools are dense `[slots, len]` fp16 planes indexed `base + e * len`.

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

/// The mul1 codebook (`exl3_moe_cuda.py` CB_MUL1): bertholomus's GLM-5.3 checkpoint.
pub const CB_MUL1: c_int = 2;
/// The mcg codebook: Mia's AI Lab's mixed-width GLM-5.3 checkpoint.
pub const CB_MCG: c_int = 1;
/// exlr8's codebook: mcg with every value rounded to FP8 E4M3 (kindlingai/glm-5.3-exlr8).
pub const CB_MCG_E4M3: c_int = 3;
/// exlr8's codebook with the rounding in two instructions (ties away from zero, no grid under 2^-6):
/// 1,078 of the 65,536 states decode to a neighbouring E4M3 value. Not the format's; the engine takes
/// it only when asked (GLM_EXLR8_ROUND=fast).
pub const CB_MCG_E4M3_FAST: c_int = 4;
/// exlr8's codebook since the release's 0.6 (2026-11-18, every routed and shared expert re-encoded): mul1
/// with every value rounded to FP8 E4M3. The release's dense trellis matrices stay mcg (CB_MCG_E4M3).
pub const CB_MUL1_E4M3: c_int = 5;
/// CB_MUL1_E4M3 with CB_MCG_E4M3_FAST's two-instruction rounding.
pub const CB_MUL1_E4M3_FAST: c_int = 6;
/// An exlr8 codebook (either hash, either rounding): the pack's sides and the engine's exlr8 settings go with it.
pub fn is_exlr8(cb: c_int) -> bool {
    matches!(cb, CB_MCG_E4M3 | CB_MCG_E4M3_FAST | CB_MUL1_E4M3 | CB_MUL1_E4M3_FAST)
}
/// An exlr8 codebook `cb` at the format's rounding (`exact`) or the two-instruction one, its hash kept.
pub fn exlr8_round(cb: c_int, exact: bool) -> c_int {
    match (cb, exact) {
        (CB_MCG_E4M3 | CB_MCG_E4M3_FAST, true) => CB_MCG_E4M3,
        (CB_MCG_E4M3 | CB_MCG_E4M3_FAST, false) => CB_MCG_E4M3_FAST,
        (CB_MUL1_E4M3 | CB_MUL1_E4M3_FAST, true) => CB_MUL1_E4M3,
        (CB_MUL1_E4M3 | CB_MUL1_E4M3_FAST, false) => CB_MUL1_E4M3_FAST,
        _ => cb,
    }
}
/// Decode grouping: at most this many members a slot (`exl3m_group`'s maxm).
pub const MAX_MEMBERS: c_int = 16;

extern "C" {
    pub fn exl3m_set_pdl(on: c_int) -> c_int;
    pub fn exl3m_set_evict_first(on: c_int) -> c_int;
    pub fn exl3m_group(pick: *const c_void, uids: *mut c_void, ucount: *mut c_void, members: *mut c_void, r: c_int,
                       slots: c_int, e: c_int, maxm: c_int, s: CudaStream);
    pub fn exl3m_rot_in(x: *const c_void, x_stride: c_int, pick: *const c_void, suh0: *const c_void,
                        suh1: *const c_void, out0: *mut c_void, out1: *mut c_void, rows: c_int, k: c_int,
                        slots: c_int, e: c_int, x_bf16: c_int, s: CudaStream);
    /// `exl3m_rot_in` for an exlr8 pack: one rotation a row, each pick's block scales applied after it
    pub fn exl3m_rot_in_shared(x: *const c_void, x_stride: c_int, pick: *const c_void, suh0: *const c_void,
                               suh1: *const c_void, out0: *mut c_void, out1: *mut c_void, rows: c_int, k: c_int,
                               slots: c_int, e: c_int, x_bf16: c_int, s: CudaStream);
    /// `exl3m_down_combine` for an exlr8 pack: the picks combined before one rotation a row
    pub fn exl3m_down_combine_shared(z: *const c_void, pick: *const c_void, svh_d: *const c_void, wts: *const c_void,
                                     out: *mut c_void, rows: c_int, p: c_int, d: c_int, sk: c_int, slots: c_int,
                                     e: c_int, s: CudaStream);
    pub fn exl3m_grouped(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                         k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                         members: *const c_void, z: *mut c_void, k: c_int, n: c_int, p: c_int, sk: c_int,
                         maxm: c_int, slots: c_int, nexp_max: c_int, mats: c_int, nt: c_int, warps: c_int,
                         pf: c_int, lo: c_int, hi: c_int, cb: c_int, s: CudaStream);
    pub fn exl3m_prefill(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                         k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                         members: *const c_void, z: *mut c_void, k: c_int, n: c_int, p: c_int, slots: c_int,
                         nblocks: c_int, mats: c_int, bm: c_int, nt: c_int, warps: c_int, pf: c_int, lo: c_int,
                         hi: c_int, s: CudaStream);
    pub fn exl3m_prefill_smem(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                              k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                              members: *const c_void, z: *mut c_void, k: c_int, n: c_int, p: c_int, slots: c_int,
                              nblocks: c_int, mats: c_int, warps: c_int, mtw: c_int, ntb: c_int, pf: c_int,
                              lo: c_int, hi: c_int, s: CudaStream);
    pub fn exl3m_prefill_smem_mode(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                                   k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void,
                                   ucount: *const c_void, members: *const c_void, z: *mut c_void, k: c_int, n: c_int,
                                   p: c_int, slots: c_int, nblocks: c_int, mats: c_int, warps: c_int, mtw: c_int,
                                   ntb: c_int, mode: c_int, s: CudaStream);
    /// exl3m_prefill_smem_mode's serving tile (4x1x8, mode -4) for a mixed-width arena: codebook `cb`
    /// (CB_MUL1 / CB_MCG), widths up to `hi` (2 x bits, at most 8).
    pub fn exl3m_prefill_smem_cb(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                                 k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                                 members: *const c_void, z: *mut c_void, k: c_int, n: c_int, p: c_int, slots: c_int,
                                 nblocks: c_int, mats: c_int, cb: c_int, hi: c_int, s: CudaStream);
    /// The fat tile (prefill_fat_kernel): mats 2 gate / up over x0 / x1, mats 1 one matrix over x0; `z`
    /// as exl3m_prefill_smem's, bit for bit; codebook `cb`, widths up to `hi`.
    pub fn exl3m_prefill_fat(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                             k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                             members: *const c_void, z: *mut c_void, k: c_int, n: c_int, p: c_int, slots: c_int,
                             nblocks: c_int, mats: c_int, cb: c_int, hi: c_int, s: CudaStream);
    /// exl3m_prefill_fat (mats 2) with exl3m_gateup's epilogue fused: xd [P][N] fp16, bit for bit.
    pub fn exl3m_prefill_fat_gateup(x0: *const c_void, x1: *const c_void, tp0: *const c_void, tp1: *const c_void,
                                    k2_0: *const c_void, k2_1: *const c_void, uids: *const c_void, ucount: *const c_void,
                                    members: *const c_void, k: c_int, n: c_int, p: c_int, slots: c_int, nblocks: c_int,
                                    cb: c_int, hi: c_int, svh_g: *const c_void, svh_u: *const c_void, suh_d: *const c_void,
                                    suh_d_stride: c_int, limit: f32, xd: *mut c_void, s: CudaStream);
    pub fn exl3m_gateup(z: *const c_void, pick: *const c_void, svh_g: *const c_void, svh_u: *const c_void,
                        suh_d: *const c_void, xd: *mut c_void, rows: c_int, p: c_int, n: c_int, sk: c_int,
                        slots: c_int, e: c_int, suh_d_stride: c_int, limit: f32, act_mode: c_int, s: CudaStream);
    pub fn exl3m_dense_rot(x: *const c_void, x_stride: c_int, suh: *const c_void, out: *mut c_void, rows: c_int,
                           k: c_int, g: c_int, x_bf16: c_int, s: CudaStream);
    pub fn exl3m_dense_out(z: *const c_void, svh: *const c_void, out: *mut c_void, rows: c_int, n: c_int, sk: c_int,
                           g: c_int, out_bf16: c_int, s: CudaStream);
    pub fn exl3m_down_combine(z: *const c_void, pick: *const c_void, svh_d: *const c_void, wts: *const c_void,
                              out: *mut c_void, rows: c_int, p: c_int, d: c_int, sk: c_int, slots: c_int, e: c_int,
                              s: CudaStream);
}
