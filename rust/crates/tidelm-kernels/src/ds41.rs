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
        lane: *const c_void, ring_lstride: i64, ckv_lstride: i64, work: *mut c_void, work_bytes: i64,
    ) -> c_int;

    /// Workspace ds_attn_decode's split-key kernel needs for m rows, h heads, at most nk keys
    /// (null / smaller: the per-head kernel).
    pub fn ds_attn_work_bytes(m: c_int, h: c_int, nk: c_int) -> i64;

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

    /// Indexer scores [T, n] bf16 (ld_out): q [T, 32, 128], k [n, 128], w [T, 32] f32,
    /// clen [T] visible counts, cand [T, ld_cand] u8 or null.
    pub fn ds_index_scores(
        stream: CudaStream, q: *const c_void, k: *const c_void, w: *const c_void, clen: *const c_void,
        cand: *const c_void, ld_cand: i64, n: c_int, out: *mut c_void, ld_out: i64, t: c_int,
        lane: *const c_void, k_lstride: i64,
    ) -> c_int;

    /// Per row: the K largest selectable bf16 values' indices, ascending, -1 padded to
    /// `width`.
    pub fn ds_topk_select(
        stream: CudaStream, s: *const c_void, ld_s: i64, n: c_int, k: c_int, out: *mut c_void,
        width: c_int, t: c_int,
    ) -> c_int;

    pub fn ds_candidate_blocks(
        stream: CudaStream, s: *const c_void, ld_s: i64, n: c_int, block: c_int, clen: *const c_void,
        bmax: *mut c_void, ld_b: i64, t: c_int,
    ) -> c_int;

    /// Sets cand[t, j] = 1 for the positions of the selected blocks; zero `cand` first.
    pub fn ds_candidate_set(
        stream: CudaStream, blocks: *const c_void, kb: c_int, block: c_int, n: c_int,
        cand: *mut c_void, ld_cand: i64, t: c_int,
    ) -> c_int;

    pub fn ds_bf16_to_f32_scale(stream: CudaStream, x: *const c_void, n: i64, scale: f32, out: *mut c_void) -> c_int;

    /// bf16 mean over the hc copies: h [M, hc, D] -> out [M, ld_out].
    pub fn ds_hc_mean(stream: CudaStream, h: *const c_void, out: *mut c_void, ld_out: i64, m: c_int, d: c_int, hc: c_int) -> c_int;

    /// argmax over float(a) + float(b) per row (b may be null), first index on ties.
    pub fn ds_argmax_sum_bf16(
        stream: CudaStream, a: *const c_void, lda: i64, b: *const c_void, ldb: i64, m: c_int, n: c_int,
        out: *mut c_void,
    ) -> c_int;

    pub fn ds_gather_rows_u16(stream: CudaStream, table: *const c_void, idx: *const c_void, out: *mut c_void, m: c_int, d: c_int) -> c_int;

    // graph-safe step plumbing (see the kernels' comments)
    pub fn ds_scatter_ring(stream: CudaStream, src: *const c_void, ring: *mut c_void, ring_size: c_int, pos: *const c_void, t: c_int, d: c_int, lane: *const c_void, lstride: i64) -> c_int;
    pub fn ds_visible(stream: CudaStream, pos: *const c_void, t: c_int, r: c_int, c: c_int, cidx: *mut c_void, clen: *mut c_void) -> c_int;
    pub fn ds_compress_rows(stream: CudaStream, kv: *const c_void, sc: *const c_void, pkv: *const c_void, psc: *const c_void, pos: *const c_void, out: *mut c_void, t: c_int, d: c_int, lane: *const c_void) -> c_int;
    pub fn ds_scatter_compressed(stream: CudaStream, src: *const c_void, ld_src: i64, dst: *mut c_void, pos: *const c_void, r: c_int, t: c_int, d: c_int, lane: *const c_void, lstride: i64) -> c_int;
    /// One block per lane segment of the step: rows seg_first[b] .. + seg_count[b] of lane seg_lane[b]
    /// (a segment with no rows is skipped).
    pub fn ds_pending_update(stream: CudaStream, kv: *const c_void, sc: *const c_void, pos: *const c_void, seg_first: *const c_void, seg_count: *const c_void, seg_lane: *const c_void, segments: c_int, pkv: *mut c_void, psc: *mut c_void, flag: *mut c_void, mem_kv: *mut c_void, mem_sc: *mut c_void, before_kv: *mut c_void, before_sc: *mut c_void, before_flag: *mut c_void, mem_start: *mut c_void, d: c_int, mem_rows: c_int) -> c_int;
    /// One block per entry: lane lanes[b] (null: lane 0) back to length n_dev[b].
    pub fn ds_rollback_pending(stream: CudaStream, n_dev: *const c_void, lanes: *const c_void, entries: c_int, mem_kv: *const c_void, mem_sc: *const c_void, mem_start: *const c_void, before_kv: *const c_void, before_sc: *const c_void, before_flag: *const c_void, pkv: *mut c_void, psc: *mut c_void, flag: *mut c_void, d: c_int, mem_rows: c_int) -> c_int;
    /// `live` (int32 on the device, or null): pairs at or past it get slot -1.
    pub fn ds_lut_slots(stream: CudaStream, idx: *const c_void, lut: *const c_void, p: c_int, slots: *mut c_void, live: *const c_void) -> c_int;
    /// DSV41_EXPERT_TOPP over the router's idx/rw [T][K] on rows with rows[t] != 0: dropped
    /// picks get weight 0 and expert `sentinel`.
    pub fn ds_expert_topp(stream: CudaStream, idx: *mut c_void, rw: *mut c_void, t: c_int, k: c_int, topp: f32, min_k: c_int,
                          sentinel: c_int, rows: *const c_void) -> c_int;
    pub fn ds_greedy_accept(stream: CudaStream, am: *const c_void, block: *const c_void, t: c_int, pos: *const c_void, out: *mut c_void) -> c_int;
    pub fn ds_wait_flag(stream: CudaStream, flag: *const c_void, expect: c_int, timeout_ns: i64, err: *mut c_void) -> c_int;

    /// Tensor-core skinny GEMM over fp8 block-scaled weights, M <= 64 (rows_per_group > 0
    /// for the grouped wo_a form, as ds_linear_fp8_grouped). K % 64 == 0.
    pub fn ds_mma_fp8(
        stream: CudaStream, x: *const c_void, ldx: i64, x_group_stride: i64, rows_per_group: c_int, w: *const c_void,
        s: *const c_void, lds: i64, m: c_int, n: c_int, k: c_int, splits: c_int, y_bf16: *mut c_void, y_f32: *mut c_void,
        ldy: i64, partial: *mut c_void,
    ) -> c_int;

    pub fn ds_mma_bf16(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, m: c_int, n: c_int, k: c_int, splits: c_int,
        y_bf16: *mut c_void, y_f32: *mut c_void, ldy: i64, partial: *mut c_void,
    ) -> c_int;

    /// Output rows per block of the tensor-core GEMM at m rows (for choosing split-K).
    pub fn ds_mma_rows_per_block(m: c_int) -> c_int;

    /// Per row of x [M][N]: the first-index argmax as (offset + index, value bits) pairs [M][2].
    pub fn ds_argmax_bf16_pairs(stream: CudaStream, x: *const c_void, ldx: i64, m: c_int, n: c_int, offset: c_int,
                                out: *mut c_void) -> c_int;
    /// The larger of the ranks' pairs [world][M][2] per row, the earlier rank on a tie.
    pub fn ds_argmax_combine(stream: CudaStream, pairs: *const c_void, world: c_int, m: c_int, out: *mut c_void) -> c_int;
    /// Temperature + nucleus per row of a + b (bf16 [M][N], b may be null), parameters per row
    /// from the device: temp [M] (<= 0: greedy, the first-index argmax), top_p [M], u [M] or
    /// null. Writes the nucleus by `cum - p < top_p` over (value desc, id asc), at most K
    /// <= ds_nucleus_cap() tokens, renormalized: ids/probs [M][ldo], cnt [M] = n, or -n when
    /// it was cut short (by K or by the candidate window) - still exactly the distribution
    /// written, but not the row's whole nucleus. With u, tok [M]: the inverse-CDF draw.
    pub fn ds_nucleus(
        stream: CudaStream, a: *const c_void, lda: i64, b: *const c_void, ldb: i64, m: c_int, n: c_int, temp: *const c_void,
        top_p: *const c_void, u: *const c_void, k: c_int, ids: *mut c_void, probs: *mut c_void, ldo: i64, cnt: *mut c_void,
        tok: *mut c_void,
    ) -> c_int;
    pub fn ds_nucleus_cap() -> c_int;
    /// x[i] = table[x[i]] (int32, on the device).
    pub fn ds_map_i32(stream: CudaStream, table: *const c_void, x: *mut c_void, n: c_int) -> c_int;
    /// [world][M][n] -> [M][world * n].
    pub fn ds_unshard_u16(stream: CudaStream, src: *const c_void, dst: *mut c_void, m: c_int, n: c_int, world: c_int) -> c_int;

    /// bf16 [N][K] -> e4m3 codes [N][K] + UE8M0 scales [ceil(N/32)][ceil(K/32)]
    /// (tools/fp8_linear.py::quantize_to_fp8).
    pub fn ds_quant_fp8_block32(stream: CudaStream, w: *const c_void, n: c_int, k: c_int, codes: *mut c_void,
                                scales: *mut c_void) -> c_int;

    /// Expert parallel: this rank's k-sum of `parts` [K][M][D] in fp32 (ds_moe_combine
    /// without the shared expert), and after the pair's all-reduce, + shared rounded to bf16.
    pub fn ds_moe_partial(stream: CudaStream, parts: *const c_void, k: c_int, m: c_int, out: *mut c_void, d: c_int) -> c_int;
    pub fn ds_moe_finish(stream: CudaStream, routed: *const c_void, shared: *const c_void, out: *mut c_void, m: c_int,
                         d: c_int) -> c_int;

    /// The FP4 MoE up GEMM + SwiGLU on the tensor cores: fp4_moe_cuda_up_v2's contract
    /// (route_small blocks, `block_route` = pair | token << 16), for steps past 16 rows.
    pub fn ds_moe_up_mma(
        stream: CudaStream, x: *const c_void, ldx: i64, w1: *const c_void, s1: *const c_void, w3: *const c_void,
        s3: *const c_void, h: *mut c_void, ldh: i64, route_weight: *const c_void, block_slot: *const c_void,
        block_route: *const c_void, nb: c_int, n: c_int, k: c_int, limit: f32, null_slot: c_int,
    ) -> c_int;

    /// The FP4 MoE down GEMM on the tensor cores: fp4_moe_cuda_down_v2's contract.
    pub fn ds_moe_down_mma(
        stream: CudaStream, h: *const c_void, ldh: i64, w2: *const c_void, s2: *const c_void, y: *mut c_void, ldy: i64,
        block_slot: *const c_void, block_pair: *const c_void, nb: c_int, n: c_int, k: c_int, topk: c_int, ntok: c_int,
        null_slot: c_int, partial: c_int, parts_world: c_int,
    ) -> c_int;

    /// Benchmark-only: the tensor-core fp8 GEMM at an explicit (ng, depth).
    pub fn ds_mma_fp8_cfg(
        stream: CudaStream, x: *const c_void, ldx: i64, w: *const c_void, s: *const c_void, lds: i64, m: c_int, n: c_int,
        k: c_int, splits: c_int, y_bf16: *mut c_void, partial: *mut c_void, ng: c_int, depth: c_int,
    ) -> c_int;
}
