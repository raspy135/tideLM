// Decode-sized native CUDA kernels for fp4_moe.py.
//
// The serving Triton kernel deliberately uses a BM=16 tensor-core tile even when an expert
// receives only one routed pair.  That is the right general grouped-GEMM shape, but decode usually
// has 18-36 pairs spread over almost as many experts.  These kernels instead assign one warp to an
// output row, stream a coalesced 64-byte FP4 weight tile, and do work only for the real pairs in the
// block. Pair-outer loops keep one pair's state live; activation loads are vectorized. The default
// FP4_RELAXED_REDUCE removes more shuffles by changing summation order; set it to 0 to retain the
// original CUDA arithmetic order. BF16 output boundaries remain intact in both modes. See
// docs/gotchas.md for the measured tradeoffs and remaining whole-engine validation scope.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kBlockM = 16;
#ifndef FP4_WARPS_SHIFT
#define FP4_WARPS_SHIFT 3
#endif
constexpr int kWarpsShift = FP4_WARPS_SHIFT;
constexpr int kWarps = 1 << kWarpsShift;
constexpr int kThreads = kWarps << 5;
#ifndef FP4_ROWS_PER_WARP
#define FP4_ROWS_PER_WARP 1
#endif
constexpr int kRowsPerWarp = FP4_ROWS_PER_WARP;
constexpr int kRowsPerBlock = kRowsPerWarp << kWarpsShift;

constexpr int log2_exact(unsigned value) {
    int shift = 0;
    while (value > 1) { value >>= 1; ++shift; }
    return shift;
}

static_assert(kRowsPerBlock > 0 && (kRowsPerBlock & (kRowsPerBlock - 1)) == 0,
              "rows per block must remain a power of two");
constexpr int kRowsPerBlockShift = log2_exact(kRowsPerBlock);
#ifndef FP4_RELAXED_REDUCE
#define FP4_RELAXED_REDUCE 1
#endif

__device__ __forceinline__ void decode_e2m1x2(uint8_t packed, float &even, float &odd) {
    uint32_t bits;
    uint32_t src = packed;
    // sm_121a requires the cvt source to be a real .b8 register.  The low output half is the
    // low-nibble (even-K) value and the high output half is the high-nibble (odd-K) value.
    asm volatile(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "}"
        : "=r"(bits) : "r"(src));
    __half_raw lo{static_cast<uint16_t>(bits)};
    __half_raw hi{static_cast<uint16_t>(bits >> 16)};
    even = __half2float(static_cast<__half>(lo));
    odd = __half2float(static_cast<__half>(hi));
}

__device__ __forceinline__ float ue8m0(uint8_t scale) {
    return __uint_as_float(static_cast<uint32_t>(scale) << 23);
}

// Random-access byte reconstruction; all floating-point arithmetic stays unchanged.
template <bool PACKED>
__device__ __forceinline__ uint8_t scale_byte(const uint8_t *scales, int slot, int index,
                                              int plane_size, int stride) {
    if constexpr (!PACKED) return scales[static_cast<int64_t>(slot) * plane_size + index];
    const int blocks = (plane_size + 63) >> 6;
    const int code0 = (blocks + 7) & ~7;
    const int escape0 = code0 + blocks * 8;
    const int offsets0 = escape0 + blocks * 8;
    const int payload0 = (offsets0 + blocks * 4 + 7) & ~7;
    const uint8_t *record = scales + static_cast<int64_t>(slot) * stride;
    const int block = index >> 6, bit = index & 63;
    const uint64_t escaped = reinterpret_cast<const uint64_t *>(record + escape0)[block];
    if ((escaped >> bit) & 1) {
        const int offset = reinterpret_cast<const int32_t *>(record + offsets0)[block];
        const int ordinal = __popcll(escaped & ((1ULL << bit) - 1));
        return record[payload0 + offset + ordinal];
    }
    const uint64_t codes = reinterpret_cast<const uint64_t *>(record + code0)[block];
    return record[block] + ((codes >> bit) & 1);
}

__device__ __forceinline__ void load_activation4(
    const __nv_bfloat16 *p, float &x0, float &x1, float &x2, float &x3) {
    // Model strides and lane offsets are multiples of four BF16 elements (8-byte aligned).
    const uint2 bits = *reinterpret_cast<const uint2 *>(p);
    x0 = __half2float(__float2half_rn(__bfloat162float(__ushort_as_bfloat16(bits.x & 0xffff))));
    x1 = __half2float(__float2half_rn(__bfloat162float(__ushort_as_bfloat16(bits.x >> 16))));
    x2 = __half2float(__float2half_rn(__bfloat162float(__ushort_as_bfloat16(bits.y & 0xffff))));
    x3 = __half2float(__float2half_rn(__bfloat162float(__ushort_as_bfloat16(bits.y >> 16))));
}

__device__ __forceinline__ float subgroup8_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 4, 8);
    v += __shfl_down_sync(0xffffffffu, v, 2, 8);
    v += __shfl_down_sync(0xffffffffu, v, 1, 8);
    return v;
}

__device__ __forceinline__ float ordered_quad_sum(float v) {
    // The four subgroup leaders hold the per-32-K partials.  Add them in K order, matching
    // _quad_dot rather than allowing the compiler to tree-reduce across scale boundaries.
    // Only lane 0 consumes the result, so its first partial needs no broadcast.
    float out = v;
    out += __shfl_sync(0xffffffffu, v, 8);
    out += __shfl_sync(0xffffffffu, v, 16);
    out += __shfl_sync(0xffffffffu, v, 24);
    return out;
}

__device__ __forceinline__ float silu_like_triton(float x) {
    float e, inv;
    const float exponent = -x * 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
    const float denom = 1.0f + e;
    asm("div.full.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(denom));
    return x * inv;
}

__global__ void moe_up_bm16(
    const __nv_bfloat16 *__restrict__ x,
    const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1,
    const uint8_t *__restrict__ w3, const uint8_t *__restrict__ s3,
    __nv_bfloat16 *__restrict__ h, const float *__restrict__ route_weight,
    const int32_t *__restrict__ block_slot, const int32_t *__restrict__ block_pair,
    int64_t stride_x, int64_t stride_h, float limit,
    int topk, int n, int k, int null_slot) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slot_i = block_slot[blockIdx.x];
    if (slot_i < 0 || slot_i == null_slot) return;

    // k is a positive model dimension and a multiple of 128. Keep these as explicit shifts: these
    // address calculations sit in the hot kernel and must not depend on strength reduction.
    const int kb = k >> 1;
    const int sg = k >> 5;
    const int q_count = k >> 7;
    const int subgroup = lane >> 3;
    const int subgroup_lane = lane & 7;
    for (int row_in_warp = 0; row_in_warp < kRowsPerWarp; ++row_in_warp) {
        const int out_n = (static_cast<int>(blockIdx.y) << kRowsPerBlockShift)
                        + (row_in_warp << kWarpsShift) + warp;
        if (out_n >= n) continue;
        const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
        const uint8_t *w1_row = w1 + row * kb;
        const uint8_t *w3_row = w3 + row * kb;
        const uint8_t *s1_row = s1 + row * sg;
        const uint8_t *s3_row = s3 + row * sg;

        // Routing entries are contiguous, then padded with -1. Keep only one pair's accumulators
        // live across K, instead of sixteen pairs and sixteen validity branches per K tile.
        for (int m = 0; m < kBlockM; ++m) {
            const int route = block_pair[(static_cast<int64_t>(blockIdx.x) << 4) + m];
            if (route < 0) break;
            const int pair = route & 0xffff;
            const __nv_bfloat16 *x_row = x + static_cast<int64_t>(route >> 16) * stride_x;
            float gate = 0.0f, up = 0.0f;
            // One warp load is 32 x uint16 = 64 contiguous bytes = 128 logical K values.
            for (int q = 0; q < q_count; ++q) {
                const int byte_k = (q << 6) + (lane << 1);
                const uint16_t p1 = *reinterpret_cast<const uint16_t *>(w1_row + byte_k);
                const uint16_t p3 = *reinterpret_cast<const uint16_t *>(w3_row + byte_k);
                float w1e0, w1o0, w1e1, w1o1, w3e0, w3o0, w3e1, w3o1;
                decode_e2m1x2(static_cast<uint8_t>(p1), w1e0, w1o0);
                decode_e2m1x2(static_cast<uint8_t>(p1 >> 8), w1e1, w1o1);
                decode_e2m1x2(static_cast<uint8_t>(p3), w3e0, w3o0);
                decode_e2m1x2(static_cast<uint8_t>(p3 >> 8), w3e1, w3o1);
                const float scale1 = ue8m0(s1_row[(q << 2) + subgroup]);
                const float scale3 = ue8m0(s3_row[(q << 2) + subgroup]);
                const int xk = (q << 7) + (lane << 2);

                float pge = 0.0f, pgo = 0.0f, pue = 0.0f, puo = 0.0f;
                const __nv_bfloat16 *xp = x_row + xk;
                float x0, x1, x2, x3;
                load_activation4(xp, x0, x1, x2, x3);
                pge = fmaf(x0, w1e0, pge); pge = fmaf(x2, w1e1, pge);
                pgo = fmaf(x1, w1o0, pgo); pgo = fmaf(x3, w1o1, pgo);
                pue = fmaf(x0, w3e0, pue); pue = fmaf(x2, w3e1, pue);
                puo = fmaf(x1, w3o0, puo); puo = fmaf(x3, w3o1, puo);
                float pg, pu;
                if constexpr (FP4_RELAXED_REDUCE) {
                    pg = subgroup8_sum(pge + pgo);
                    pu = subgroup8_sum(pue + puo);
                } else {
                    pg = subgroup8_sum(pge) + subgroup8_sum(pgo);
                    pu = subgroup8_sum(pue) + subgroup8_sum(puo);
                }
                if (subgroup_lane == 0) { pg *= scale1; pu *= scale3; }
                if constexpr (FP4_RELAXED_REDUCE) {
                    // Keep each scale subgroup's sum across K; combine the four only once at the end.
                    gate += pg; up += pu;
                } else {
                    pg = ordered_quad_sum(pg);
                    pu = ordered_quad_sum(pu);
                    if (lane == 0) { gate += pg; up += pu; }
                }
            }

            if constexpr (FP4_RELAXED_REDUCE) {
                gate = ordered_quad_sum(gate);
                up = ordered_quad_sum(up);
            }
            if (lane == 0) {
                // The reference quantized linear rounds to BF16 before the FP32 epilogue.
                float g = __bfloat162float(__float2bfloat16_rn(gate));
                float u = __bfloat162float(__float2bfloat16_rn(up));
                g = fminf(g, limit);
                u = fminf(fmaxf(u, -limit), limit);
                const float silu = silu_like_triton(g);
                const float value = silu * u * route_weight[pair];
                h[static_cast<int64_t>(pair) * stride_h + out_n] = __float2bfloat16_rn(value);
            }
        }
    }
}

__global__ void moe_down_bm16(
    const __nv_bfloat16 *__restrict__ h,
    const uint8_t *__restrict__ w2, const uint8_t *__restrict__ s2,
    float *__restrict__ y, const int32_t *__restrict__ block_slot,
    const int32_t *__restrict__ block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int null_slot, int partial, int parts_world) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slot_i = block_slot[blockIdx.x];
    if (slot_i < 0) return;

    const int kb = k >> 1;
    const int sg = k >> 5;
    const int q_count = k >> 7;
    const int subgroup = lane >> 3;
    const int subgroup_lane = lane & 7;
    for (int row_in_warp = 0; row_in_warp < kRowsPerWarp; ++row_in_warp) {
        const int out_n = (static_cast<int>(blockIdx.y) << kRowsPerBlockShift)
                        + (row_in_warp << kWarpsShift) + warp;
        if (out_n >= n) continue;
        const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
        for (int m = 0; m < kBlockM; ++m) {
            const int pair = block_pair[(static_cast<int64_t>(blockIdx.x) << 4) + m];
            if (pair < 0) break;
            float acc = 0.0f;
            if (slot_i != null_slot) {
                const uint8_t *w2_row = w2 + row * kb;
                const uint8_t *s2_row = s2 + row * sg;
                const __nv_bfloat16 *h_row = h + static_cast<int64_t>(pair) * stride_h;
                for (int q = 0; q < q_count; ++q) {
                    const int byte_k = (q << 6) + (lane << 1);
                    const uint16_t packed = *reinterpret_cast<const uint16_t *>(w2_row + byte_k);
                    float we0, wo0, we1, wo1;
                    decode_e2m1x2(static_cast<uint8_t>(packed), we0, wo0);
                    decode_e2m1x2(static_cast<uint8_t>(packed >> 8), we1, wo1);
                    const float scale = ue8m0(s2_row[(q << 2) + subgroup]);
                    const int hk = (q << 7) + (lane << 2);
                    float pe = 0.0f, po = 0.0f;
                    const __nv_bfloat16 *hp = h_row + hk;
                    float x0, x1, x2, x3;
                    load_activation4(hp, x0, x1, x2, x3);
                    pe = fmaf(x0, we0, pe); pe = fmaf(x2, we1, pe);
                    po = fmaf(x1, wo0, po); po = fmaf(x3, wo1, po);
                    float p;
                    if constexpr (FP4_RELAXED_REDUCE) p = subgroup8_sum(pe + po);
                    else p = subgroup8_sum(pe) + subgroup8_sum(po);
                    if (subgroup_lane == 0) p *= scale;
                    if constexpr (FP4_RELAXED_REDUCE) acc += p;
                    else {
                        p = ordered_quad_sum(p);
                        if (lane == 0) acc += p;
                    }
                }
            }

            if constexpr (FP4_RELAXED_REDUCE) acc = ordered_quad_sum(acc);
            if (lane == 0) {
                const int local_n = n / parts_world;
                const int token = pair / topk;
                const int expert_k = pair % topk;
                const int64_t out_row = static_cast<int64_t>(expert_k) * ntok + token;
                int64_t offset;
                if (parts_world > 1) {
                    offset = ((out_n / local_n) * static_cast<int64_t>(topk * ntok) + out_row)
                             * local_n + out_n % local_n;
                } else {
                    offset = out_row * stride_y + out_n;
                }
                float value = slot_i == null_slot ? 0.0f : acc;
                if (!partial) value = __bfloat162float(__float2bfloat16_rn(value));
                y[offset] = value;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// v2: the same arithmetic as moe_up_bm16 / moe_down_bm16 with FP4_RELAXED_REDUCE=1, re-laid out for
// memory-level parallelism. v1 gives each lane 2 bytes of a 64-byte warp tile per K step and walks
// the row once per routed pair; v2 gives each lane whole 32-value scale groups (one 16-byte load per
// matrix), keeps the row in registers across the pairs, and replays v1's reduction exactly:
//   per group:   8 "virtual lane" partials s_l = fma(x2,w2,fma(x0,w0,0)) + fma(x3,w3,fma(x1,w1,0)),
//                tree ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)) as subgroup8_sum left it, times the scale;
//   per row:     subgroup i's running sum over K steps q = 0..: G_i += P[4q + i], in q order,
//                then ((G0 + G1) + G2) + G3 as ordered_quad_sum.
// The running sums need their q order, so the group partials go through shared memory and four
// lanes per matrix replay the chains. FP4_V2_FMA_SCALE selects how v1's `p *= scale; acc += p`
// compiled (plain or contracted into one fma); the unit test finds which one v1 is.
#ifndef FP4_V2_FMA_SCALE
#define FP4_V2_FMA_SCALE 0
#endif
constexpr int kV2MaxGroups = 192;  // k <= 6144 (GLM-5.3 hidden; was 160 = 5120 for V4.1)

// Benchmark-only compile switch. 1 retains the deployed v2 kernels; 2 stages two routed
// pairs' partials before replaying their independent ordered chains in parallel.
// No loader/environment switch promotes this experimental spelling into serving.
// Cold actual-weight TP-output screen (2026-10-07, rows 1/4/6): 84 intermediate/final
// bit checks passed. At rows6/U6 up+down was 573.7 -> 518.1 us, but rows4/U14 was
// 609.3 -> 620.0 us and rows6/U20 866.4 -> 884.3 us. Higher shared-memory occupancy
// and workload mix prevent a general win; keep default 1. Even favorable cells imply
// only a few milliseconds over 40 layers. tools/bench_fp4_pairbatch.py reproduces the
// screen; results/fp4-pairbatch-20261007 preserves measurements and source snapshot.
#ifndef FP4_V2_PAIR_BATCH
#define FP4_V2_PAIR_BATCH 1
#endif
static_assert(FP4_V2_PAIR_BATCH == 1 || FP4_V2_PAIR_BATCH == 2,
              "FP4_V2_PAIR_BATCH must be 1 or 2");

__device__ __forceinline__ void load_activation32(const __nv_bfloat16 *p, float *xv) {
    // 32 consecutive BF16 activations with v1's load_activation4 conversion (BF16 -> FP16 -> FP32).
    const uint4 *q = reinterpret_cast<const uint4 *>(p);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 v = q[c];
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            xv[c * 8 + w * 2] = __half2float(__float2half_rn(
                __bfloat162float(__ushort_as_bfloat16(words[w] & 0xffff))));
            xv[c * 8 + w * 2 + 1] = __half2float(__float2half_rn(
                __bfloat162float(__ushort_as_bfloat16(words[w] >> 16))));
        }
    }
}

__device__ __forceinline__ float group_partial(const uint4 packed, const float *xv) {
    // v1's per-K-step lane work for the 8 lanes of one scale subgroup, and its subgroup8_sum tree.
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float s[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t two = (words[l >> 1] >> ((l & 1) * 16)) & 0xffff;
        float we0, wo0, we1, wo1;
        decode_e2m1x2(static_cast<uint8_t>(two), we0, wo0);
        decode_e2m1x2(static_cast<uint8_t>(two >> 8), we1, wo1);
        float pe = 0.0f, po = 0.0f;
        pe = fmaf(xv[4 * l + 0], we0, pe); pe = fmaf(xv[4 * l + 2], we1, pe);
        po = fmaf(xv[4 * l + 1], wo0, po); po = fmaf(xv[4 * l + 3], wo1, po);
        s[l] = pe + po;
    }
    return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}

__device__ __forceinline__ float ordered_chain(const float *part, const float *scale, int groups, int i) {
    // Subgroup i's running sum of v1: acc starts at 0 and adds group 4q + i for q = 0, 1, ...
    float acc = 0.0f;
    for (int g = i; g < groups; g += 4) {
#if FP4_V2_FMA_SCALE
        acc = fmaf(part[g], scale[g], acc);
#else
        acc += part[g] * scale[g];
#endif
    }
    return acc;
}

// STREAM loads the weights with __ldcs (evict-first: they are read once a step, so they leave L2 to
// the activations every warp re-reads per routed pair), and ROWS_FIRST swaps the grid to (rows,
// blocks) so consecutive blocks walk one expert's rows instead of every expert's first rows at once.
// Neither changes what a block computes. rust/crates/tidelm-kernels/examples/bench_moe_bw.rs, the
// serving pair's per-rank shapes (up 1,152 x 5,120, down 2,560 x 2,304), a cold 3 GB arena, routing
// spread uniformly, three runs (2026-10-25), GB/s of the touched experts' bytes, up + down:
//   rows / experts   plain v2    STREAM    ROWS_FIRST   both
//   4 / 23           225-229     245-250   200-205      257-258
//   6 / 35           228-231     251       200-203      256-260
//   10 / 53          228-231     246-247   192-193      247-248
//   16 / 88          227-230     233-238   190-192      235-238
// In the engine the screen did not carry over. bench/bench.py on the pair (prose / code, seed 42,
// --ignore-eos, a fresh seed history per arm, TIDELM_STEP_SPLIT forward ms a step at the rows shown):
//   plain v2     prose 53.7-55.0 at 2.5-2.6 rows   code 67.1-70.4 at 4.2-4.4 rows
//   STREAM       prose 53.2-54.4 at 2.5-2.6        code 65.9-66.9 at 4.2-4.4    (~2 ms, ~3%)
//   both         prose 53.8-56.2 at 2.4-2.6        code 65.7-66.7 at 3.7-3.9    (slower at equal rows;
//                the depth policy, which reads step times, then drafted shallower: 3.2 tokens a
//                step on code against 3.5)
// So the serving fp4_moe_cuda_{up,down}_v2 take STREAM alone, over the (blocks, rows) grid; ROWS_FIRST
// stays a benchmark switch. The scale pack entries (PACKED, the Python engine's) keep plain loads.
template <bool STREAM>
__device__ __forceinline__ uint4 load_w(const uint8_t *row, int g) {
    if constexpr (STREAM) return __ldcs(reinterpret_cast<const uint4 *>(row) + g);
    else return reinterpret_cast<const uint4 *>(row)[g];
}

template <bool PACKED = false, bool STREAM = false, bool ROWS_FIRST = false>
__global__ void moe_up_bm16_v2(
    const __nv_bfloat16 *__restrict__ x,
    const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1,
    const uint8_t *__restrict__ w3, const uint8_t *__restrict__ s3,
    __nv_bfloat16 *__restrict__ h, const float *__restrict__ route_weight,
    const int32_t *__restrict__ block_slot, const int32_t *__restrict__ block_pair,
    int64_t stride_x, int64_t stride_h, float limit,
    int topk, int n, int k, int null_slot, int scale_stride = 0) {
    constexpr int kPerLane = kV2MaxGroups / 32;
    __shared__ float part_s[kWarps][2][kV2MaxGroups];
    __shared__ float scale_s[kWarps][2][kV2MaxGroups];
    __shared__ float chain_s[kWarps][8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int bx = ROWS_FIRST ? blockIdx.y : blockIdx.x, by = ROWS_FIRST ? blockIdx.x : blockIdx.y;
    const int slot_i = block_slot[bx];
    if (slot_i < 0 || slot_i == null_slot) return;
    const int out_n = (by << kWarpsShift) + warp;
    if (out_n >= n) return;
    const int groups = k >> 5;
    const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
    const uint8_t *w1_row = w1 + row * (k >> 1);
    const uint8_t *w3_row = w3 + row * (k >> 1);
    // The whole row, loaded once: this lane's groups g = lane + 32 r.
    uint4 p1[kPerLane], p3[kPerLane];
#pragma unroll
    for (int r = 0; r < kPerLane; ++r) {
        const int g = lane + 32 * r;
        if (g < groups) {
            p1[r] = load_w<STREAM>(w1_row, g);
            p3[r] = load_w<STREAM>(w3_row, g);
            scale_s[warp][0][g] = ue8m0(scale_byte<PACKED>(s1, slot_i, out_n * groups + g, n * groups, scale_stride));
            scale_s[warp][1][g] = ue8m0(scale_byte<PACKED>(s3, slot_i, out_n * groups + g, n * groups, scale_stride));
        }
    }
    for (int m = 0; m < kBlockM; ++m) {
        const int route = block_pair[(static_cast<int64_t>(bx) << 4) + m];
        if (route < 0) break;
        const int pair = route & 0xffff;
        const __nv_bfloat16 *x_row = x + static_cast<int64_t>(route >> 16) * stride_x;
#pragma unroll
        for (int r = 0; r < kPerLane; ++r) {
            const int g = lane + 32 * r;
            if (g < groups) {
                float xv[32];
                load_activation32(x_row + g * 32, xv);
                part_s[warp][0][g] = group_partial(p1[r], xv);
                part_s[warp][1][g] = group_partial(p3[r], xv);
            }
        }
        __syncwarp();
        if (lane < 8) chain_s[warp][lane] = ordered_chain(part_s[warp][lane >> 2], scale_s[warp][lane >> 2],
                                                          groups, lane & 3);
        __syncwarp();
        if (lane == 0) {
            const float *c = chain_s[warp];
            const float gate = ((c[0] + c[1]) + c[2]) + c[3];
            const float up = ((c[4] + c[5]) + c[6]) + c[7];
            float g = __bfloat162float(__float2bfloat16_rn(gate));
            float u = __bfloat162float(__float2bfloat16_rn(up));
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
            const float silu = silu_like_triton(g);
            const float value = silu * u * route_weight[pair];
            h[static_cast<int64_t>(pair) * stride_h + out_n] = __float2bfloat16_rn(value);
        }
        __syncwarp();
    }
}

template <bool PACKED = false, bool STREAM = false, bool ROWS_FIRST = false>
__global__ void moe_down_bm16_v2(
    const __nv_bfloat16 *__restrict__ h,
    const uint8_t *__restrict__ w2, const uint8_t *__restrict__ s2,
    float *__restrict__ y, const int32_t *__restrict__ block_slot,
    const int32_t *__restrict__ block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int null_slot, int partial, int parts_world, int scale_stride = 0) {
    constexpr int kPerLane = kV2MaxGroups / 32;
    __shared__ float part_s[kWarps][kV2MaxGroups];
    __shared__ float scale_s[kWarps][kV2MaxGroups];
    __shared__ float chain_s[kWarps][4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int bx = ROWS_FIRST ? blockIdx.y : blockIdx.x, by = ROWS_FIRST ? blockIdx.x : blockIdx.y;
    const int slot_i = block_slot[bx];
    if (slot_i < 0) return;
    const int out_n = (by << kWarpsShift) + warp;
    if (out_n >= n) return;
    const int groups = k >> 5;
    const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
    const bool real = slot_i != null_slot;
    uint4 pw[kPerLane];
    if (real) {
        const uint8_t *w2_row = w2 + row * (k >> 1);
#pragma unroll
        for (int r = 0; r < kPerLane; ++r) {
            const int g = lane + 32 * r;
            if (g < groups) {
                pw[r] = load_w<STREAM>(w2_row, g);
                scale_s[warp][g] = ue8m0(scale_byte<PACKED>(s2, slot_i, out_n * groups + g, n * groups, scale_stride));
            }
        }
    }
    for (int m = 0; m < kBlockM; ++m) {
        const int pair = block_pair[(static_cast<int64_t>(bx) << 4) + m];
        if (pair < 0) break;
        float acc = 0.0f;
        if (real) {
            const __nv_bfloat16 *h_row = h + static_cast<int64_t>(pair) * stride_h;
#pragma unroll
            for (int r = 0; r < kPerLane; ++r) {
                const int g = lane + 32 * r;
                if (g < groups) {
                    float xv[32];
                    load_activation32(h_row + g * 32, xv);
                    part_s[warp][g] = group_partial(pw[r], xv);
                }
            }
            __syncwarp();
            if (lane < 4) chain_s[warp][lane] = ordered_chain(part_s[warp], scale_s[warp], groups, lane);
            __syncwarp();
            const float *c = chain_s[warp];
            acc = ((c[0] + c[1]) + c[2]) + c[3];
        }
        if (lane == 0) {
            const int local_n = n / parts_world;
            const int token = pair / topk;
            const int expert_k = pair % topk;
            const int64_t out_row = static_cast<int64_t>(expert_k) * ntok + token;
            int64_t offset;
            if (parts_world > 1) {
                offset = ((out_n / local_n) * static_cast<int64_t>(topk * ntok) + out_row)
                         * local_n + out_n % local_n;
            } else {
                offset = out_row * stride_y + out_n;
            }
            float value = slot_i == null_slot ? 0.0f : acc;
            if (!partial) value = __bfloat162float(__float2bfloat16_rn(value));
            y[offset] = value;
        }
        __syncwarp();
    }
}

// v3 down (DSV41_FP4_CUDA_DOWN_ROWS = 2 or 4): R output rows a warp instead of one. A TP2 down row
// is only 72 scale groups (k = 2304), so a v2 warp loads 2-3 uint4 a lane and then pays the whole
// per-pair loop -- activation loads, two syncs, an 18-step chain -- for 1.15 KB of weights. The
// 2026-10 microbenchmark shows the result: v2 down took ~300 us at U = 14 and at U = 20 alike
// (147 -> 207 GB/s), a latency floor rather than bytes. Here each lane keeps the same groups
// g = lane + 32 j for all R rows, so one activation load feeds R partials and the per-pair cost is
// paid once for R rows. Each row's arithmetic is v2's exactly (group_partial, ordered_chain over
// the same groups in the same order, the same four-chain sum), so the output bits are v2's.
constexpr int kDownMaxGroups = 96;  // k <= 3072 (V4.1 down k = 2304); larger k stays on v2

template <int R, bool PACKED = false>
__global__ void moe_down_bm16_v3(
    const __nv_bfloat16 *__restrict__ h,
    const uint8_t *__restrict__ w2, const uint8_t *__restrict__ s2,
    float *__restrict__ y, const int32_t *__restrict__ block_slot,
    const int32_t *__restrict__ block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int null_slot, int partial, int parts_world, int scale_stride = 0) {
    constexpr int kPerLane = kDownMaxGroups / 32;
    __shared__ float part_s[kWarps][R][kDownMaxGroups];
    __shared__ float scale_s[kWarps][R][kDownMaxGroups];
    __shared__ float chain_s[kWarps][4 * R];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slot_i = block_slot[blockIdx.x];
    if (slot_i < 0) return;
    const int n0 = ((static_cast<int>(blockIdx.y) << kWarpsShift) + warp) * R;
    if (n0 >= n) return;
    const int rows = min(R, n - n0);
    const int groups = k >> 5;
    const bool real = slot_i != null_slot;
    uint4 pw[R][kPerLane];
    if (real) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (rr < rows) {
                const int64_t row = static_cast<int64_t>(slot_i) * n + n0 + rr;
                const uint8_t *w2_row = w2 + row * (k >> 1);
#pragma unroll
                for (int r = 0; r < kPerLane; ++r) {
                    const int g = lane + 32 * r;
                    if (g < groups) {
                        pw[rr][r] = reinterpret_cast<const uint4 *>(w2_row)[g];
                        scale_s[warp][rr][g] = ue8m0(scale_byte<PACKED>(s2, slot_i, (n0 + rr) * groups + g, n * groups, scale_stride));
                    }
                }
            }
        }
    }
    for (int m = 0; m < kBlockM; ++m) {
        const int pair = block_pair[(static_cast<int64_t>(blockIdx.x) << 4) + m];
        if (pair < 0) break;
        if (real) {
            const __nv_bfloat16 *h_row = h + static_cast<int64_t>(pair) * stride_h;
#pragma unroll
            for (int r = 0; r < kPerLane; ++r) {
                const int g = lane + 32 * r;
                if (g < groups) {
                    float xv[32];
                    load_activation32(h_row + g * 32, xv);
#pragma unroll
                    for (int rr = 0; rr < R; ++rr)
                        if (rr < rows) part_s[warp][rr][g] = group_partial(pw[rr][r], xv);
                }
            }
            __syncwarp();
            if (lane < 4 * rows)
                chain_s[warp][lane] = ordered_chain(part_s[warp][lane >> 2], scale_s[warp][lane >> 2],
                                                    groups, lane & 3);
            __syncwarp();
        }
        if (lane < rows) {
            float acc = 0.0f;
            if (real) {
                const float *c = chain_s[warp] + 4 * lane;
                acc = ((c[0] + c[1]) + c[2]) + c[3];
            }
            const int out_n = n0 + lane;
            const int local_n = n / parts_world;
            const int token = pair / topk;
            const int expert_k = pair % topk;
            const int64_t out_row = static_cast<int64_t>(expert_k) * ntok + token;
            int64_t offset;
            if (parts_world > 1) {
                offset = ((out_n / local_n) * static_cast<int64_t>(topk * ntok) + out_row)
                         * local_n + out_n % local_n;
            } else {
                offset = out_row * stride_y + out_n;
            }
            float value = real ? acc : 0.0f;
            if (!partial) value = __bfloat162float(__float2bfloat16_rn(value));
            y[offset] = value;
        }
        __syncwarp();
    }
}

#if FP4_V2_PAIR_BATCH == 2
// Keep group_partial and ordered_chain unchanged. The extra pair axis changes scheduling and
// synchronization only: each pair keeps exactly v2's products, reduction tree, K-chain order,
// and the same BF16 boundaries. Weights/scales are still loaded once per expert/output row.
__global__ void moe_up_bm16_v2_pairbatch(
    const __nv_bfloat16 *__restrict__ x,
    const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1,
    const uint8_t *__restrict__ w3, const uint8_t *__restrict__ s3,
    __nv_bfloat16 *__restrict__ h, const float *__restrict__ route_weight,
    const int32_t *__restrict__ block_slot, const int32_t *__restrict__ block_pair,
    int64_t stride_x, int64_t stride_h, float limit,
    int topk, int n, int k, int null_slot) {
    constexpr int PB = 2, kPerLane = kV2MaxGroups / 32;
    __shared__ float part_s[kWarps][PB][2][kV2MaxGroups];
    __shared__ float scale_s[kWarps][2][kV2MaxGroups];
    __shared__ float chain_s[kWarps][PB][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int slot_i = block_slot[blockIdx.x];
    if (slot_i < 0 || slot_i == null_slot) return;
    const int out_n = (static_cast<int>(blockIdx.y) << kWarpsShift) + warp;
    if (out_n >= n) return;
    const int groups = k >> 5;
    const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
    const uint8_t *w1_row = w1 + row * (k >> 1), *w3_row = w3 + row * (k >> 1);
    uint4 p1[kPerLane], p3[kPerLane];
#pragma unroll
    for (int r = 0; r < kPerLane; ++r) {
        const int g = lane + 32 * r;
        if (g < groups) {
            p1[r] = reinterpret_cast<const uint4 *>(w1_row)[g];
            p3[r] = reinterpret_cast<const uint4 *>(w3_row)[g];
            scale_s[warp][0][g] = ue8m0(s1[row * groups + g]);
            scale_s[warp][1][g] = ue8m0(s3[row * groups + g]);
        }
    }
    for (int m = 0; m < kBlockM; m += PB) {
        int routes[PB];
#pragma unroll
        for (int b = 0; b < PB; ++b)
            routes[b] = block_pair[(static_cast<int64_t>(blockIdx.x) << 4) + m + b];
        if (routes[0] < 0) break;
#pragma unroll
        for (int b = 0; b < PB; ++b) {
            if (routes[b] >= 0) {
                const __nv_bfloat16 *xr = x + static_cast<int64_t>(routes[b] >> 16) * stride_x;
#pragma unroll
                for (int r = 0; r < kPerLane; ++r) {
                    const int g = lane + 32 * r;
                    if (g < groups) {
                        float xv[32];
                        load_activation32(xr + g * 32, xv);
                        part_s[warp][b][0][g] = group_partial(p1[r], xv);
                        part_s[warp][b][1][g] = group_partial(p3[r], xv);
                    }
                }
            }
        }
        __syncwarp();
        const int b = lane >> 3, c = lane & 7;
        if (b < PB && routes[b] >= 0)
            chain_s[warp][b][c] = ordered_chain(part_s[warp][b][c >> 2],
                                               scale_s[warp][c >> 2], groups, c & 3);
        __syncwarp();
        if (b < PB && c == 0 && routes[b] >= 0) {
            const float *v = chain_s[warp][b];
            const float gate = ((v[0] + v[1]) + v[2]) + v[3];
            const float up = ((v[4] + v[5]) + v[6]) + v[7];
            float g = __bfloat162float(__float2bfloat16_rn(gate));
            float u = __bfloat162float(__float2bfloat16_rn(up));
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
            const int pair = routes[b] & 0xffff;
            const float value = silu_like_triton(g) * u * route_weight[pair];
            h[static_cast<int64_t>(pair) * stride_h + out_n] = __float2bfloat16_rn(value);
        }
        __syncwarp();
    }
}

__global__ void moe_down_bm16_v2_pairbatch(
    const __nv_bfloat16 *__restrict__ h,
    const uint8_t *__restrict__ w2, const uint8_t *__restrict__ s2,
    float *__restrict__ y, const int32_t *__restrict__ block_slot,
    const int32_t *__restrict__ block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int null_slot, int partial, int parts_world) {
    constexpr int PB = 2, kPerLane = kV2MaxGroups / 32;
    __shared__ float part_s[kWarps][PB][kV2MaxGroups];
    __shared__ float scale_s[kWarps][kV2MaxGroups];
    __shared__ float chain_s[kWarps][PB][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int slot_i = block_slot[blockIdx.x];
    if (slot_i < 0) return;
    const int out_n = (static_cast<int>(blockIdx.y) << kWarpsShift) + warp;
    if (out_n >= n) return;
    const int groups = k >> 5;
    const int64_t row = static_cast<int64_t>(slot_i) * n + out_n;
    const bool real = slot_i != null_slot;
    uint4 pw[kPerLane];
    if (real) {
        const uint8_t *wr = w2 + row * (k >> 1);
#pragma unroll
        for (int r = 0; r < kPerLane; ++r) {
            const int g = lane + 32 * r;
            if (g < groups) {
                pw[r] = reinterpret_cast<const uint4 *>(wr)[g];
                scale_s[warp][g] = ue8m0(s2[row * groups + g]);
            }
        }
    }
    for (int m = 0; m < kBlockM; m += PB) {
        int pairs[PB];
#pragma unroll
        for (int b = 0; b < PB; ++b)
            pairs[b] = block_pair[(static_cast<int64_t>(blockIdx.x) << 4) + m + b];
        if (pairs[0] < 0) break;
        if (real) {
#pragma unroll
            for (int b = 0; b < PB; ++b) {
                if (pairs[b] >= 0) {
                    const __nv_bfloat16 *hr = h + static_cast<int64_t>(pairs[b]) * stride_h;
#pragma unroll
                    for (int r = 0; r < kPerLane; ++r) {
                        const int g = lane + 32 * r;
                        if (g < groups) {
                            float xv[32];
                            load_activation32(hr + g * 32, xv);
                            part_s[warp][b][g] = group_partial(pw[r], xv);
                        }
                    }
                }
            }
            __syncwarp();
            const int b = lane >> 2, c = lane & 3;
            if (b < PB && pairs[b] >= 0)
                chain_s[warp][b][c] = ordered_chain(part_s[warp][b], scale_s[warp], groups, c);
            __syncwarp();
        }
        const int b = lane >> 2;
        if (b < PB && (lane & 3) == 0 && pairs[b] >= 0) {
            float value = 0.0f;
            if (real) {
                const float *v = chain_s[warp][b];
                value = ((v[0] + v[1]) + v[2]) + v[3];
            }
            const int local_n = n / parts_world;
            const int64_t out_row = static_cast<int64_t>(pairs[b] % topk) * ntok + pairs[b] / topk;
            const int64_t offset = parts_world > 1
                ? ((out_n / local_n) * static_cast<int64_t>(topk * ntok) + out_row) * local_n + out_n % local_n
                : out_row * stride_y + out_n;
            if (!partial) value = __bfloat162float(__float2bfloat16_rn(value));
            y[offset] = value;
        }
        __syncwarp();
    }
}
#endif

__global__ void route_small_bm16(const int32_t *slots, int32_t *block_slot,
                                 int32_t *block_pair, int32_t *block_route,
                                 int pairs, int topk) {
    const int p = threadIdx.x;
    for (int i = p; i < pairs; i += blockDim.x) block_slot[i] = -1;
    for (int i = p; i < (pairs << 4); i += blockDim.x) {
        block_pair[i] = -1;
        block_route[i] = -1;
    }
    __shared__ int32_t key[64];
    __shared__ int32_t sorted_slot[64];
    __shared__ int32_t sorted_pair[64];
    __shared__ int32_t run_offset[64];
    __shared__ uint32_t starts[2];
    if (p < pairs) key[p] = slots[p];
    __syncthreads();
    // A stable sort by slot, the torch builder's argsort contract: pair p lands after every pair
    // with a smaller slot and every earlier pair with the same slot. Counting both in one O(P)
    // pass per thread replaced a one-thread insertion sort, O(P^2) serial shared-memory steps
    // that took 19.6 us a call on the verify block (2026-10-22, 43 calls a step).
    if (p < pairs) {
        const int32_t mine = key[p];
        int below = 0, before = 0;
        for (int i = 0; i < pairs; ++i) {
            const int32_t other = key[i];
            below += other < mine;
            before += other == mine && i < p;
        }
        sorted_slot[below + before] = mine;
        sorted_pair[below + before] = p;
        run_offset[below + before] = before;
    }
    __syncthreads();
    // The block cut, in parallel over sorted positions: a block starts at each new slot and every
    // 16 pairs into a run (long runs are split, not truncated, when small top-k permits T > 16),
    // so a position's block is the number of starts at or before it, minus one.
    const bool valid = p < pairs;
    const bool start = valid && run_offset[p] % kBlockM == 0;
    const uint32_t ballot = __ballot_sync(0xffffffffu, start);
    if ((p & 31) == 0) starts[p >> 5] = ballot;
    __syncthreads();
    if (!valid) return;
    const int lane = p & 31;
    const int block = __popc(starts[p >> 5] & (0xffffffffu >> (31 - lane)))
                      + ((p >> 5) ? __popc(starts[0]) : 0) - 1;
    const int32_t pair = sorted_pair[p];
    const int index = (block << 4) + run_offset[p] % kBlockM;
    if (start) block_slot[block] = sorted_slot[p];
    block_pair[index] = pair;
    block_route[index] = pair | ((pair / topk) << 16);
}

// Batched multi-request decode stacks several verify blocks: 4 lanes x 6 rows x top-6 is 144
// pairs, past route_small_bm16's 64-entry shared arrays. Same contract, same output: a stable
// sort of the pairs by slot (unique, so it is exactly the insertion sort's order) and then the same
// sequential block cut. The rank is computed in parallel -- each thread counts the pairs that sort
// ahead of its own, O(P) per thread -- and only the O(P) cut runs on one thread.
constexpr int kRouteWideMax = 512;

__global__ void route_wide_bm16(const int32_t *slots, int32_t *block_slot,
                                int32_t *block_pair, int32_t *block_route,
                                int pairs, int topk) {
    const int p = threadIdx.x;
    for (int i = p; i < pairs; i += blockDim.x) block_slot[i] = -1;
    for (int i = p; i < (pairs << 4); i += blockDim.x) {
        block_pair[i] = -1;
        block_route[i] = -1;
    }
    __shared__ int32_t key[kRouteWideMax];
    __shared__ int32_t sorted_slot[kRouteWideMax];
    __shared__ int32_t sorted_pair[kRouteWideMax];
    if (p < pairs) key[p] = slots[p];
    __syncthreads();
    if (p < pairs) {
        const int32_t mine = key[p];
        int rank = 0;
        for (int i = 0; i < pairs; ++i) {
            const int32_t other = key[i];
            rank += (other < mine) || (other == mine && i < p);
        }
        sorted_slot[rank] = mine;
        sorted_pair[rank] = p;
    }
    __syncthreads();
    if (p != 0) return;
    int block = -1, rank = 0;
    int32_t previous = INT32_MIN;
    for (int i = 0; i < pairs; ++i) {
        const int32_t value = sorted_slot[i];
        if (!i || value != previous || rank == kBlockM) {
            ++block; rank = 0; previous = value; block_slot[block] = value;
        }
        const int index = (block << 4) + rank;
        block_pair[index] = sorted_pair[i];
        block_route[index] = sorted_pair[i] | ((sorted_pair[i] / topk) << 16);
        ++rank;
    }
}

__global__ void round_reduce(const float *parts, void *out, int td, int topk, int out_bf16) {
    // This internal kernel is always launched with 256 threads below.
    const int i = (static_cast<int>(blockIdx.x) << 8) + threadIdx.x;
    if (i >= td) return;
    float total = 0.0f;
    for (int k = 0; k < topk; ++k)
        total += __bfloat162float(__float2bfloat16_rn(parts[static_cast<int64_t>(k) * td + i]));
    if (out_bf16)
        static_cast<__nv_bfloat16 *>(out)[i] = __float2bfloat16_rn(total);
    else
        static_cast<float *>(out)[i] = total;
}

// Dreaming FP8 tier (engine/dream_overlay.py): a few trained experts are served from e4m3 codes
// with one UE8M0 scale per 32 weights instead of MXFP4. Their pairs still run through the FP4
// kernels above (the expert's MXFP4 copy is resident, so routing and blocks are unchanged); these
// two kernels then OVERWRITE those pairs' rows -- the activated intermediate after `up`, the
// per-pair output after `down` -- in the same layouts, so everything downstream (the TP gathers,
// the k-sum, the BF16 rounding points) is untouched. The grid is static (pairs x row blocks) and a
// block whose pair is not an FP8 expert exits at once, so the launches capture into decode graphs.
// One warp a row; each lane takes whole 32-weight groups and the lanes' sums meet in a fixed
// butterfly, so the bits depend only on the inputs: each rank computes its own rows, deterministically.
__device__ __forceinline__ float fp8_group_dot(const uint8_t *w, const __nv_bfloat16 *x) {
    // 32 e4m3 weights x 32 BF16 activations, accumulated in index order.
    const uint4 wq[2] = {reinterpret_cast<const uint4 *>(w)[0], reinterpret_cast<const uint4 *>(w)[1]};
    const uint4 *xq = reinterpret_cast<const uint4 *>(x);
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 xv = xq[c];
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
        const uint4 wv = wq[c >> 1];
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // weights 8c + 2j, 8c + 2j + 1 sit in byte pair (c & 1) * 4 + j of the 16-byte word
            const int b = (c & 1) * 8 + 2 * j;
            const uint16_t pair = static_cast<uint16_t>((ww[b >> 2] >> ((b & 3) * 8)) & 0xffff);
            const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
            const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&hr));
            acc = fmaf(__bfloat162float(__ushort_as_bfloat16(xw[j] & 0xffff)), wf.x, acc);
            acc = fmaf(__bfloat162float(__ushort_as_bfloat16(xw[j] >> 16)), wf.y, acc);
        }
    }
    return acc;
}

__device__ __forceinline__ float fp8_row_dot(const uint8_t *w_row, const uint8_t *s_row,
                                             const __nv_bfloat16 *x_row, int groups, int lane) {
    float acc = 0.0f;
    for (int g = lane; g < groups; g += 32)
        acc = fmaf(fp8_group_dot(w_row + g * 32, x_row + g * 32), ue8m0(s_row[g]), acc);
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

__global__ void dream_fp8_up(
    const __nv_bfloat16 *__restrict__ x, int64_t stride_x, const int64_t *__restrict__ idx,
    const int32_t *__restrict__ map, int n_experts, const uint8_t *__restrict__ w13,
    const uint8_t *__restrict__ s13, __nv_bfloat16 *__restrict__ h, int64_t stride_h,
    const float *__restrict__ route_weight, float limit, int topk, int n, int k) {
    const int pair = blockIdx.x;
    const int64_t e = idx[pair];
    if (e < 0 || e >= n_experts) return;
    const int slot = map[e];
    if (slot < 0) return;
    const int lane = threadIdx.x & 31;
    const int row = (static_cast<int>(blockIdx.y) << kWarpsShift) + (threadIdx.x >> 5);
    if (row >= n) return;
    const int groups = k >> 5;
    const __nv_bfloat16 *x_row = x + static_cast<int64_t>(pair / topk) * stride_x;
    const int64_t g_row = static_cast<int64_t>(slot) * 2 * n + row, u_row = g_row + n;
    const float gate = fp8_row_dot(w13 + g_row * k, s13 + g_row * groups, x_row, groups, lane);
    const float up = fp8_row_dot(w13 + u_row * k, s13 + u_row * groups, x_row, groups, lane);
    if (lane == 0) {
        // the FP4 kernels' epilogue: BF16 projections, clamp, SiLU, routing weight, BF16 out
        float g = __bfloat162float(__float2bfloat16_rn(gate));
        float u = __bfloat162float(__float2bfloat16_rn(up));
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        h[static_cast<int64_t>(pair) * stride_h + row] =
            __float2bfloat16_rn(silu_like_triton(g) * u * route_weight[pair]);
    }
}

__global__ void dream_fp8_down(
    const __nv_bfloat16 *__restrict__ h, int64_t stride_h, const int64_t *__restrict__ idx,
    const int32_t *__restrict__ map, int n_experts, const uint8_t *__restrict__ w2,
    const uint8_t *__restrict__ s2, float *__restrict__ y, int64_t stride_y,
    int topk, int n, int k, int ntok, int partial) {
    const int pair = blockIdx.x;
    const int64_t e = idx[pair];
    if (e < 0 || e >= n_experts) return;
    const int slot = map[e];
    if (slot < 0) return;
    const int lane = threadIdx.x & 31;
    const int row = (static_cast<int>(blockIdx.y) << kWarpsShift) + (threadIdx.x >> 5);
    if (row >= n) return;
    const int groups = k >> 5;
    const int64_t w_row = static_cast<int64_t>(slot) * n + row;
    const float acc = fp8_row_dot(w2 + w_row * k, s2 + w_row * groups,
                                  h + static_cast<int64_t>(pair) * stride_h, groups, lane);
    if (lane == 0) {
        // v2 down's parts_world == 1 layout: row (expert_k * ntok + token)
        const int64_t out_row = static_cast<int64_t>(pair % topk) * ntok + pair / topk;
        y[out_row * stride_y + row] = partial ? acc : __bfloat162float(__float2bfloat16_rn(acc));
    }
}

} // namespace

extern "C" int fp4_moe_cuda_up(
    void *stream, const void *x, const void *w1, const void *s1, const void *w3, const void *s3,
    void *h, const void *route_weight, const void *block_slot, const void *block_pair,
    int64_t stride_x, int64_t stride_h, float limit, int topk, int n, int k, int nb,
    int null_slot) {
    dim3 grid(nb, (n + kRowsPerBlock - 1) >> kRowsPerBlockShift);
    moe_up_bm16<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(x), static_cast<const uint8_t *>(w1),
        static_cast<const uint8_t *>(s1), static_cast<const uint8_t *>(w3),
        static_cast<const uint8_t *>(s3), static_cast<__nv_bfloat16 *>(h),
        static_cast<const float *>(route_weight), static_cast<const int32_t *>(block_slot),
        static_cast<const int32_t *>(block_pair), stride_x, stride_h, limit, topk, n, k,
        null_slot);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_down(
    void *stream, const void *h, const void *w2, const void *s2, void *y,
    const void *block_slot, const void *block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int nb, int null_slot, int partial, int parts_world) {
    dim3 grid(nb, (n + kRowsPerBlock - 1) >> kRowsPerBlockShift);
    moe_down_bm16<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(h), static_cast<const uint8_t *>(w2),
        static_cast<const uint8_t *>(s2), static_cast<float *>(y),
        static_cast<const int32_t *>(block_slot), static_cast<const int32_t *>(block_pair),
        stride_h, stride_y, topk, n, k, ntok, null_slot, partial, parts_world);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_up_v2(
    void *stream, const void *x, const void *w1, const void *s1, const void *w3, const void *s3,
    void *h, const void *route_weight, const void *block_slot, const void *block_pair,
    int64_t stride_x, int64_t stride_h, float limit, int topk, int n, int k, int nb,
    int null_slot) {
    if (k > (kV2MaxGroups << 5) || (k & 31)) return static_cast<int>(cudaErrorInvalidValue);
#if FP4_V2_PAIR_BATCH == 2
    dim3 grid(nb, (n + kWarps - 1) >> kWarpsShift);
    moe_up_bm16_v2_pairbatch<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
#else
    // __ldcs weights (load_w): v2's bits, ~2 ms a 4-row decode step on the pair
    dim3 grid(nb, (n + kWarps - 1) >> kWarpsShift);
    moe_up_bm16_v2<false, true, false><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
#endif
        static_cast<const __nv_bfloat16 *>(x), static_cast<const uint8_t *>(w1),
        static_cast<const uint8_t *>(s1), static_cast<const uint8_t *>(w3),
        static_cast<const uint8_t *>(s3), static_cast<__nv_bfloat16 *>(h),
        static_cast<const float *>(route_weight), static_cast<const int32_t *>(block_slot),
        static_cast<const int32_t *>(block_pair), stride_x, stride_h, limit, topk, n, k,
        null_slot);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_down_v2(
    void *stream, const void *h, const void *w2, const void *s2, void *y,
    const void *block_slot, const void *block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int nb, int null_slot, int partial, int parts_world) {
    if (k > (kV2MaxGroups << 5) || (k & 31)) return static_cast<int>(cudaErrorInvalidValue);
#if FP4_V2_PAIR_BATCH == 2
    dim3 grid(nb, (n + kWarps - 1) >> kWarpsShift);
    moe_down_bm16_v2_pairbatch<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
#else
    dim3 grid(nb, (n + kWarps - 1) >> kWarpsShift);
    moe_down_bm16_v2<false, true, false><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
#endif
        static_cast<const __nv_bfloat16 *>(h), static_cast<const uint8_t *>(w2),
        static_cast<const uint8_t *>(s2), static_cast<float *>(y),
        static_cast<const int32_t *>(block_slot), static_cast<const int32_t *>(block_pair),
        stride_h, stride_y, topk, n, k, ntok, null_slot, partial, parts_world);
    return static_cast<int>(cudaGetLastError());
}

// fp4_moe_cuda_{up,down}_v2 with either switch (load_w): variant bit 1 = STREAM (__ldcs weights),
// bit 2 = ROWS_FIRST (grid (rows, blocks)); 1 is what the serving entries launch. Benchmark only.
extern "C" int fp4_moe_cuda_up_v2x(
    void *stream, const void *x, const void *w1, const void *s1, const void *w3, const void *s3,
    void *h, const void *route_weight, const void *block_slot, const void *block_pair,
    int64_t stride_x, int64_t stride_h, float limit, int topk, int n, int k, int nb,
    int null_slot, int variant) {
    if (k > (kV2MaxGroups << 5) || (k & 31) || variant < 0 || variant > 3) return static_cast<int>(cudaErrorInvalidValue);
    const int rows = (n + kWarps - 1) >> kWarpsShift;
    const dim3 grid = variant & 2 ? dim3(rows, nb) : dim3(nb, rows);
#define FP4_UP_V2X(S, R)                                                                         \
    moe_up_bm16_v2<false, S, R><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(       \
        static_cast<const __nv_bfloat16 *>(x), static_cast<const uint8_t *>(w1),                 \
        static_cast<const uint8_t *>(s1), static_cast<const uint8_t *>(w3),                      \
        static_cast<const uint8_t *>(s3), static_cast<__nv_bfloat16 *>(h),                       \
        static_cast<const float *>(route_weight), static_cast<const int32_t *>(block_slot),     \
        static_cast<const int32_t *>(block_pair), stride_x, stride_h, limit, topk, n, k, null_slot)
    switch (variant) {
    case 0: FP4_UP_V2X(false, false); break;
    case 1: FP4_UP_V2X(true, false); break;
    case 2: FP4_UP_V2X(false, true); break;
    default: FP4_UP_V2X(true, true); break;
    }
#undef FP4_UP_V2X
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_down_v2x(
    void *stream, const void *h, const void *w2, const void *s2, void *y,
    const void *block_slot, const void *block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int nb, int null_slot, int partial, int parts_world, int variant) {
    if (k > (kV2MaxGroups << 5) || (k & 31) || variant < 0 || variant > 3) return static_cast<int>(cudaErrorInvalidValue);
    const int rows = (n + kWarps - 1) >> kWarpsShift;
    const dim3 grid = variant & 2 ? dim3(rows, nb) : dim3(nb, rows);
#define FP4_DOWN_V2X(S, R)                                                                       \
    moe_down_bm16_v2<false, S, R><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(     \
        static_cast<const __nv_bfloat16 *>(h), static_cast<const uint8_t *>(w2),                 \
        static_cast<const uint8_t *>(s2), static_cast<float *>(y),                               \
        static_cast<const int32_t *>(block_slot), static_cast<const int32_t *>(block_pair),      \
        stride_h, stride_y, topk, n, k, ntok, null_slot, partial, parts_world)
    switch (variant) {
    case 0: FP4_DOWN_V2X(false, false); break;
    case 1: FP4_DOWN_V2X(true, false); break;
    case 2: FP4_DOWN_V2X(false, true); break;
    default: FP4_DOWN_V2X(true, true); break;
    }
#undef FP4_DOWN_V2X
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_down_v3(
    void *stream, const void *h, const void *w2, const void *s2, void *y,
    const void *block_slot, const void *block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int nb, int null_slot, int partial, int parts_world,
    int rows_per_warp) {
    if (k > (kDownMaxGroups << 5) || (k & 31)) return static_cast<int>(cudaErrorInvalidValue);
#define FP4_DOWN_V3(R)                                                                          \
    moe_down_bm16_v3<R><<<dim3(nb, (n + (kWarps * R) - 1) / (kWarps * R)), kThreads, 0,          \
                          static_cast<cudaStream_t>(stream)>>>(                                 \
        static_cast<const __nv_bfloat16 *>(h), static_cast<const uint8_t *>(w2),                 \
        static_cast<const uint8_t *>(s2), static_cast<float *>(y),                               \
        static_cast<const int32_t *>(block_slot), static_cast<const int32_t *>(block_pair),      \
        stride_h, stride_y, topk, n, k, ntok, null_slot, partial, parts_world)
    switch (rows_per_warp) {
    case 2: FP4_DOWN_V3(2); break;
    case 4: FP4_DOWN_V3(4); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
    }
#undef FP4_DOWN_V3
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_route_small(void *stream, const void *slots, void *block_slot,
                                          void *block_pair, void *block_route,
                                          int pairs, int topk) {
    if (pairs > 64) {
        if (pairs > kRouteWideMax) return static_cast<int>(cudaErrorInvalidValue);
        route_wide_bm16<<<1, kRouteWideMax, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const int32_t *>(slots), static_cast<int32_t *>(block_slot),
            static_cast<int32_t *>(block_pair), static_cast<int32_t *>(block_route), pairs, topk);
        return static_cast<int>(cudaGetLastError());
    }
    route_small_bm16<<<1, 64, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int32_t *>(slots), static_cast<int32_t *>(block_slot),
        static_cast<int32_t *>(block_pair), static_cast<int32_t *>(block_route), pairs, topk);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_round_reduce(void *stream, const void *parts, void *out,
                                           int td, int topk, int out_bf16) {
    round_reduce<<<(td + 255) >> 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(parts), out, td, topk, out_bf16);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int dream_fp8_cuda_up(
    void *stream, const void *x, int64_t stride_x, const void *idx, const void *map, int n_experts,
    const void *w13, const void *s13, void *h, int64_t stride_h, const void *route_weight,
    float limit, int topk, int pairs, int n, int k) {
    if (k & 31) return static_cast<int>(cudaErrorInvalidValue);
    dream_fp8_up<<<dim3(pairs, (n + kWarps - 1) / kWarps), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(x), stride_x, static_cast<const int64_t *>(idx),
        static_cast<const int32_t *>(map), n_experts, static_cast<const uint8_t *>(w13),
        static_cast<const uint8_t *>(s13), static_cast<__nv_bfloat16 *>(h), stride_h,
        static_cast<const float *>(route_weight), limit, topk, n, k);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int dream_fp8_cuda_down(
    void *stream, const void *h, int64_t stride_h, const void *idx, const void *map, int n_experts,
    const void *w2, const void *s2, void *y, int64_t stride_y, int topk, int pairs, int n, int k,
    int ntok, int partial) {
    if (k & 31) return static_cast<int>(cudaErrorInvalidValue);
    dream_fp8_down<<<dim3(pairs, (n + kWarps - 1) / kWarps), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(h), stride_h, static_cast<const int64_t *>(idx),
        static_cast<const int32_t *>(map), n_experts, static_cast<const uint8_t *>(w2),
        static_cast<const uint8_t *>(s2), static_cast<float *>(y), stride_y, topk, n, k, ntok, partial);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_up_scales(
    void *stream, const void *x, const void *w1, const void *s1, const void *w3, const void *s3,
    void *h, const void *route_weight, const void *block_slot, const void *block_pair,
    int64_t stride_x, int64_t stride_h, float limit, int topk, int n, int k, int nb,
    int null_slot, int scale_stride) {
    if (k > (kV2MaxGroups << 5) || (k & 31)) return static_cast<int>(cudaErrorInvalidValue);
    moe_up_bm16_v2<true><<<dim3(nb, (n + kWarps - 1) >> kWarpsShift), kThreads, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(x), static_cast<const uint8_t *>(w1),
        static_cast<const uint8_t *>(s1), static_cast<const uint8_t *>(w3),
        static_cast<const uint8_t *>(s3), static_cast<__nv_bfloat16 *>(h),
        static_cast<const float *>(route_weight), static_cast<const int32_t *>(block_slot),
        static_cast<const int32_t *>(block_pair), stride_x, stride_h, limit, topk, n, k,
        null_slot, scale_stride);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fp4_moe_cuda_down_scales(
    void *stream, const void *h, const void *w2, const void *s2, void *y,
    const void *block_slot, const void *block_pair, int64_t stride_h, int64_t stride_y,
    int topk, int n, int k, int ntok, int nb, int null_slot, int partial, int parts_world,
    int rows_per_warp, int scale_stride) {
#define SCALE_DOWN_ARGS static_cast<const __nv_bfloat16 *>(h), static_cast<const uint8_t *>(w2), \
    static_cast<const uint8_t *>(s2), static_cast<float *>(y), \
    static_cast<const int32_t *>(block_slot), static_cast<const int32_t *>(block_pair), \
    stride_h, stride_y, topk, n, k, ntok, null_slot, partial, parts_world, scale_stride
    if (k > (kV2MaxGroups << 5) || (k & 31)) return static_cast<int>(cudaErrorInvalidValue);
    if (rows_per_warp == 2 && k <= 3072)
        moe_down_bm16_v3<2, true><<<dim3(nb, (n + kWarps * 2 - 1) / (kWarps * 2)), kThreads, 0,
                                   static_cast<cudaStream_t>(stream)>>>(SCALE_DOWN_ARGS);
    else if (rows_per_warp == 4 && k <= 3072)
        moe_down_bm16_v3<4, true><<<dim3(nb, (n + kWarps * 4 - 1) / (kWarps * 4)), kThreads, 0,
                                   static_cast<cudaStream_t>(stream)>>>(SCALE_DOWN_ARGS);
    else
        moe_down_bm16_v2<true><<<dim3(nb, (n + kWarps - 1) >> kWarpsShift), kThreads, 0,
                                static_cast<cudaStream_t>(stream)>>>(SCALE_DOWN_ARGS);
#undef SCALE_DOWN_ARGS
    return static_cast<int>(cudaGetLastError());
}

__global__ void scale_bytes_kernel(const uint8_t *packed, uint8_t *out, int size, int stride, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    if (i < size && slot < slots) out[static_cast<int64_t>(slot) * size + i] = scale_byte<true>(packed, slot, i, size, stride);
}
extern "C" int fp4_moe_cuda_scale_expand(void *stream, const void *packed, void *out,
                                          int size, int stride, int slots) {
    scale_bytes_kernel<<<dim3((size + 255) / 256, slots), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(packed), static_cast<uint8_t *>(out), size, stride, slots);
    return static_cast<int>(cudaGetLastError());
}
