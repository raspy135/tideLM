// DS v4.1 decode-path kernels for the Rust engine.
//
// Every kernel reproduces the arithmetic of the Python reference path it replaces
// (tools/v41_ref.py, engine/model.py, the checkpoint's inference/model.py): the same
// fp32 accumulation, the same bf16 rounding points, the same epsilons. They differ
// from the Python only in summation ORDER inside a reduction, never in what is rounded
// where. Each kernel names its Python counterpart; the Rust GPU tests compare every one
// against a CPU transcription of that counterpart.
//
// Row invariance: a row's result never depends on how many rows are in the call (the
// property engine/model.py pads every decode GEMM to 16 rows to get from cuBLAS). The
// skinny GEMM below accumulates each row with the same instruction sequence whatever M
// is, so 1-token decode and a 6-token verify block agree by construction.
//
// C ABI: stream first, cudaGetLastError() returned as int, no allocation, graph-capturable.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <algorithm>

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChunk = 1024;  // K values of x staged in shared memory per pass

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

// UE8M0 as the FP8 linear kernel reads it (tools/fp8_linear.py: `(s << 7)` as bf16 bits):
// 2^(b-127) for b >= 1, and 0 for b == 0.
__device__ __forceinline__ float e8m0_lin(uint32_t b) { return __uint_as_float(b << 23); }
// UE8M0 as torch.exp2(b - 127) computes it (engram rows): b == 0 is 2^-127, a subnormal.
__device__ __forceinline__ float e8m0_exp2(uint32_t b) {
    return b ? __uint_as_float(b << 23) : __uint_as_float(0x00400000u);
}

__device__ __forceinline__ void bf16x8_to_f32(uint4 v, float* out) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = __uint_as_float(w[i] << 16);
        out[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

__device__ __forceinline__ void fp8x4_to_f32(uint32_t word, float* out) {
    __half2_raw a = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(word & 0xffffu), __NV_E4M3);
    __half2_raw b = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(word >> 16), __NV_E4M3);
    const float2 fa = __half22float2(*reinterpret_cast<__half2*>(&a));
    const float2 fb = __half22float2(*reinterpret_cast<__half2*>(&b));
    out[0] = fa.x; out[1] = fa.y; out[2] = fb.x; out[3] = fb.y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Deterministic block reductions (fixed tree): every thread gets the result.
__device__ float block_sum(float v, float* scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float t = lane < nw ? scratch[lane] : 0.f;
    t = warp_sum(t);
    return t;
}

__device__ float block_max(float v, float* scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float t = lane < nw ? scratch[lane] : -INFINITY;
    return warp_max(t);
}

// ------------------------------------------------------------------ skinny GEMM
// y[m, n] = sum_k x[m, k] * W[n, k], x bf16 row-major, fp32 accumulate.
//   KIND 0: W fp8 e4m3 [N, K] + UE8M0 scales [ceil(N/32), ceil(K/32)] (tools/fp8_linear.py)
//   KIND 1: W bf16 [N, K]                                          (F.linear, bf16 weight)
//   KIND 2: W fp32 [N, K]                                          (F.linear on x.float())
// One warp owns ROWS consecutive output rows; lanes split K in 16-byte weight loads; x is
// staged in shared memory. `rows_per_group` > 0 makes it the grouped wo_a GEMM: rows of
// group g read x at x + g * x_group_stride (tools/fp8_linear.py::fp8_grouped_linear).
// gridDim.y > 1 splits K; partial sums go to `partial` [splits][M][N] and
// ds_splits_reduce adds them in split order (deterministic). gridDim.z tiles M in MT-row
// blocks: a row's products and their order do not depend on MT or on its tile, so one launch
// over M rows gives each row the bits of a launch of its 16-row slice alone.

template <int KIND> struct Lane;
template <> struct Lane<0> { static constexpr int V = 16, ESZ = 1; };
template <> struct Lane<1> { static constexpr int V = 8, ESZ = 2; };
template <> struct Lane<2> { static constexpr int V = 4, ESZ = 4; };

template <int KIND>
__device__ __forceinline__ void decode_w(uint4 raw, float* out) {
    if (KIND == 0) {
        fp8x4_to_f32(raw.x, out); fp8x4_to_f32(raw.y, out + 4);
        fp8x4_to_f32(raw.z, out + 8); fp8x4_to_f32(raw.w, out + 12);
    } else if (KIND == 1) {
        bf16x8_to_f32(raw, out);
    } else {
        out[0] = __uint_as_float(raw.x); out[1] = __uint_as_float(raw.y);
        out[2] = __uint_as_float(raw.z); out[3] = __uint_as_float(raw.w);
    }
}

template <int V>
__device__ __forceinline__ void load_x(const __nv_bfloat16* p, float* out) {
    if (V == 16) {
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(p), out);
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(p + 8), out + 8);
    } else if (V == 8) {
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(p), out);
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        out[0] = __uint_as_float(v.x << 16); out[1] = __uint_as_float(v.x & 0xffff0000u);
        out[2] = __uint_as_float(v.y << 16); out[3] = __uint_as_float(v.y & 0xffff0000u);
    }
}

template <int KIND, int ROWS, int MT>
__global__ void __launch_bounds__(kThreads) skinny_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, int64_t x_group_stride, int rows_per_group,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int64_t lds,
    int M, int N, int K, int ksplit,
    __nv_bfloat16* __restrict__ ybf, float* __restrict__ yf, int64_t ldy, float* __restrict__ partial) {
    constexpr int V = Lane<KIND>::V, ESZ = Lane<KIND>::ESZ, VW = V * 32;
    __shared__ __align__(16) __nv_bfloat16 xs[MT][kChunk];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_block = blockIdx.x * (kWarps * ROWS);
    const int n0 = n_block + warp * ROWS;
    const int k_begin = blockIdx.y * ksplit;
    const int k_end = min(K, k_begin + ksplit);
    const int m_base = blockIdx.z * MT, Mt = min(MT, M - m_base);
    const __nv_bfloat16* xg =
        x + (rows_per_group > 0 ? (int64_t)(n_block / rows_per_group) * x_group_stride : 0) + (int64_t)m_base * ldx;

    float acc[ROWS][MT];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

    for (int k0 = k_begin; k0 < k_end; k0 += kChunk) {
        const int kc = min(kChunk, k_end - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < MT * (kChunk / 8); i += kThreads) {
            const int m = i / (kChunk / 8), j = (i % (kChunk / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m < Mt && j < kc) v = *reinterpret_cast<const uint4*>(xg + (int64_t)m * ldx + k0 + j);
            *reinterpret_cast<uint4*>(&xs[m][j]) = v;
        }
        __syncthreads();
        for (int kk = 0; kk < kc; kk += VW) {
            const int kl = kk + lane * V;
            if (kl >= kc) continue;
            const int k = k0 + kl;
            uint4 raw[ROWS];
            float scale[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int n = n0 + r;
                raw[r] = n < N ? *reinterpret_cast<const uint4*>(w + ((int64_t)n * K + k) * ESZ)
                               : make_uint4(0, 0, 0, 0);
                scale[r] = 1.f;
                if (KIND == 0 && n < N) scale[r] = e8m0_lin(s[(int64_t)(n >> 5) * lds + (k >> 5)]);
            }
            float wv[ROWS][V];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) decode_w<KIND>(raw[r], wv[r]);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float xv[V];
                load_x<V>(&xs[m][kl], xv);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float p = 0.f;
#pragma unroll
                    for (int v = 0; v < V; ++v) p = fmaf(wv[r][v], xv[v], p);
                    // a power-of-two scale is exact, so scaling the 16-term partial equals
                    // scaling each weight (the Triton kernel's dequant-then-dot)
                    acc[r][m] = KIND == 0 ? fmaf(p, scale[r], acc[r][m]) : acc[r][m] + p;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = warp_sum(acc[r][m]);
    if (lane != 0) return;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int n = n0 + r;
        if (n >= N) continue;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= Mt) continue;
            const int64_t row = m_base + m;
            if (gridDim.y > 1) {
                partial[((int64_t)blockIdx.y * M + row) * N + n] = acc[r][m];
            } else if (ybf) {
                ybf[row * ldy + n] = to_bf(acc[r][m]);
            } else {
                yf[row * ldy + n] = acc[r][m];
            }
        }
    }
}

template <int KIND>
int launch_skinny(cudaStream_t st, const void* x, int64_t ldx, int64_t gstride, int rpg,
                  const void* w, const void* s, int64_t lds, int M, int N, int K, int splits,
                  void* ybf, void* yf, int64_t ldy, void* partial) {
    constexpr int ROWS = 4;
    if (M < 1 || splits < 1 || K % 32 || (splits > 1 && !partial)) return cudaErrorInvalidValue;
    // slice length: a multiple of 32 so slices never split a scale block
    const int ksplit = ((K + splits - 1) / splits + 31) / 32 * 32;
    // past 16 rows, 16-row tiles over gridDim.z (the slices a caller used to launch one by one)
    dim3 grid((N + kWarps * ROWS - 1) / (kWarps * ROWS), splits, (M + 15) / 16);
    auto xb = static_cast<const __nv_bfloat16*>(x);
    auto wb = static_cast<const uint8_t*>(w);
    auto sb = static_cast<const uint8_t*>(s);
    auto yb = static_cast<__nv_bfloat16*>(ybf);
    auto yff = static_cast<float*>(yf);
    auto pp = static_cast<float*>(partial);
#define DS_SKINNY(MT) \
    skinny_kernel<KIND, ROWS, MT><<<grid, kThreads, 0, st>>>(xb, ldx, gstride, rpg, wb, sb, lds, M, N, K, ksplit, yb, yff, ldy, pp)
    if (M <= 1) DS_SKINNY(1);
    else if (M <= 2) DS_SKINNY(2);
    else if (M <= 4) DS_SKINNY(4);
    else if (M <= 8) DS_SKINNY(8);
    else DS_SKINNY(16);
#undef DS_SKINNY
    return static_cast<int>(cudaGetLastError());
}

__global__ void splits_reduce_kernel(const float* __restrict__ partial, int splits, int M, int N,
                                     __nv_bfloat16* __restrict__ ybf, float* __restrict__ yf, int64_t ldy) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += partial[((int64_t)s * M + m) * N + n];
    if (ybf) ybf[(int64_t)m * ldy + n] = to_bf(acc);
    else yf[(int64_t)m * ldy + n] = acc;
}

// ------------------------------------------------------------------ tensor-core skinny GEMM
// The same products as skinny_kernel on mma.sync.m16n8k16 (bf16 in, fp32 accumulate), for M
// up to 64 rows per launch: the batched-decode shape (lanes x verify rows) that the FMA kernel
// cannot hold in registers. One warp owns 8 output rows; per 64-wide K block each lane loads 16
// contiguous weight values of one row, and the K order inside the block is permuted so those
// land exactly in the lane's B-fragment positions:
//   logical k L of mma step s  <->  physical k = (L % 8 / 2) * 16 + s * 4 + (L / 8) * 2 + L % 2
// The activation fragments are read with the same permutation, so it is the same dot product
// in a fixed order per row — row-invariant across M like skinny_kernel (not bit-identical to it).
// FP8 weights take their power-of-two block scale before the bf16 conversion: exact, as in
// tools/fp8_linear.py's dequant-then-dot.

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 16 weights of one row at (n, k..k+16) as 8 packed bf16 pairs (scale folded in for fp8).
template <int KIND>
__device__ __forceinline__ void load_w16(const uint8_t* w, const uint8_t* s, int64_t lds, int n, int K, int k,
                                         bool valid, uint32_t* out) {
    float v[16];
    if (!valid) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = 0;
        return;
    }
    if (KIND == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(w + (int64_t)n * K + k);
        decode_w<0>(raw, v);
        const float sc = e8m0_lin(s[(int64_t)(n >> 5) * lds + (k >> 5)]);
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] *= sc;
    } else {
        const uint4* p = reinterpret_cast<const uint4*>(w + ((int64_t)n * K + k) * 2);
        bf16x8_to_f32(p[0], v);
        bf16x8_to_f32(p[1], v + 8);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = pack_bf16(v[2 * i], v[2 * i + 1]);
}

// KC: K values of x staged per pass; MT * (KC + 2) bf16 must stay under the 48 KB static limit.
// NG: 8-row groups per warp. A block re-stages x for every pass, so x traffic per weight byte
// is 2 * M / rows_per_block: NG grows with M to keep that below ~0.5 (NG = 1 at M=64 measured
// x staging at 2x the weight bytes). DEPTH: K blocks whose weight loads are issued together.
template <int KIND, int MT, int KC, int NG, int DEPTH>
__global__ void __launch_bounds__(256) mma_skinny_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, int64_t x_group_stride, int rows_per_group,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int64_t lds,
    int M, int N, int K, int ksplit,
    __nv_bfloat16* __restrict__ ybf, float* __restrict__ yf, int64_t ldy, float* __restrict__ partial) {
    constexpr int MTILES = MT / 16;
    constexpr int XS = KC + 2;  // row stride in bf16: an odd number of words, so the 8 rows x 4
                                // lanes of a fragment read hit 32 distinct banks
    __shared__ __align__(16) __nv_bfloat16 xs[MT * XS];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = lane >> 2, q = lane & 3;
    const int n_block = blockIdx.x * (kWarps * 8 * NG);
    const int n0 = n_block + warp * 8 * NG;
    const int k_begin = blockIdx.y * ksplit;
    const int k_end = min(K, k_begin + ksplit);
    // gridDim.z tiles M by MT rows (prefill): each tile is the same kernel on its own rows
    const int m_base = blockIdx.z * MT;
    const __nv_bfloat16* xg =
        x + (rows_per_group > 0 ? (int64_t)(n_block / rows_per_group) * x_group_stride : 0) + (int64_t)m_base * ldx;
    const int Mt = min(MT, M - m_base);

    float acc[NG][MTILES][4];
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int mt = 0; mt < MTILES; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[g][mt][i] = 0.f;

    for (int k0 = k_begin; k0 < k_end; k0 += KC) {
        const int kc = min(KC, k_end - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < MT * (KC / 2); i += blockDim.x) {
            const int m = i / (KC / 2), j = (i % (KC / 2)) * 2;
            uint32_t v = 0;
            if (m < Mt && j < kc) v = *reinterpret_cast<const uint32_t*>(xg + (int64_t)m * ldx + k0 + j);
            *reinterpret_cast<uint32_t*>(&xs[m * XS + j]) = v;
        }
        __syncthreads();
        for (int kb0 = 0; kb0 < kc; kb0 += 64 * DEPTH) {
            uint32_t b[DEPTH][NG][8];
#pragma unroll
            for (int u = 0; u < DEPTH; ++u)
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int kb = kb0 + u * 64, n = n0 + g * 8 + r;
                    load_w16<KIND>(w, s, lds, n, K, k0 + kb + q * 16, n < N && kb < kc, b[u][g]);
                }
#pragma unroll
            for (int u = 0; u < DEPTH; ++u) {
                const int kb = kb0 + u * 64;
                if (kb >= kc) break;
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const int kx = kb + q * 16 + st * 4;
#pragma unroll
                    for (int mt = 0; mt < MTILES; ++mt) {
                        const __nv_bfloat16* r0 = &xs[(mt * 16 + r) * XS + kx];
                        const __nv_bfloat16* r1 = &xs[(mt * 16 + r + 8) * XS + kx];
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0);
                        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(r1);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 2);
                        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(r1 + 2);
#pragma unroll
                        for (int g = 0; g < NG; ++g)
                            mma_bf16(acc[g][mt], a0, a1, a2, a3, b[u][g][2 * st], b[u][g][2 * st + 1]);
                    }
                }
            }
        }
    }
    // D fragment: rows (mt*16 + r, +8), columns n0 + g*8 + q*2 + {0, 1}
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int mt = 0; mt < MTILES; ++mt)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int m = mt * 16 + r + half * 8;
                if (m >= Mt) continue;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int n = n0 + g * 8 + q * 2 + j;
                    if (n >= N) continue;
                    const float v = acc[g][mt][half * 2 + j];
                    const int mg = m_base + m;
                    if (gridDim.y > 1) partial[((int64_t)blockIdx.y * M + mg) * N + n] = v;
                    else if (ybf) ybf[(int64_t)mg * ldy + n] = to_bf(v);
                    else yf[(int64_t)mg * ldy + n] = v;
                }
            }
}

template <int KIND>
int launch_mma(cudaStream_t st, const void* x, int64_t ldx, int64_t gstride, int rpg, const void* w, const void* s,
               int64_t lds, int M, int N, int K, int splits, void* ybf, void* yf, int64_t ldy, void* partial) {
    // past 64 rows (prefill) the grid's z dimension tiles M by 64: each row the same dot products
    if (M < 1 || splits < 1 || K % 64 || (splits > 1 && !partial)) return cudaErrorInvalidValue;
    const int ksplit = ((K + splits - 1) / splits + 63) / 64 * 64;
    auto xb = static_cast<const __nv_bfloat16*>(x);
    auto wb = static_cast<const uint8_t*>(w);
    auto sb = static_cast<const uint8_t*>(s);
    auto yb = static_cast<__nv_bfloat16*>(ybf);
    auto yff = static_cast<float*>(yf);
    auto pp = static_cast<float*>(partial);
#define DS_MMA(MT, KC, NG, DEPTH)                                                                          \
    {                                                                                                      \
        if (rpg > 0 && rpg % (kWarps * 8 * NG)) return cudaErrorInvalidValue;                             \
        dim3 grid((N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG), splits, (M + MT - 1) / MT);              \
        mma_skinny_kernel<KIND, MT, KC, NG, DEPTH><<<grid, kThreads, 0, st>>>(xb, ldx, gstride, rpg, wb, sb, lds, M, N, \
                                                                              K, ksplit, yb, yff, ldy, pp); \
    }
    // (NG, DEPTH) per tile, from examples/tune_mma.rs on the decode shapes (2026-10-24, GB/s
    // of weight bytes, split-K to >= 2 blocks/SM): M=32 2x2 avg ~165 (1x4 157, 4x1 154);
    // M=64 1x4 avg ~133 (2x4 126, 4x1 121, 4x2 106). Neither reaches the ~230 GB/s the FMA
    // kernel streams at M <= 16 - this kernel exists for M > 16, where the FMA one cannot
    // hold its accumulators, and a deeper pipeline (cp.async, ldmatrix) is the next step.
    // Prefill (M > 64): 128-row tiles, two 8-row groups a warp - the FP8 decode amortizes over
    // twice the rows of the 64-row tile (examples/tune_prefill_gemm.rs, M = 2048, TFLOP/s on the
    // shard shapes: 31-48 against 28-34; 256-row tiles with 64-wide K stages 16-31).
    if (M <= 16) DS_MMA(16, 512, 1, 4)
    else if (M <= 32) DS_MMA(32, 512, 2, 2)
    else if (M <= 64) DS_MMA(64, 256, 1, 4)
    else DS_MMA(128, 128, 2, 1)
#undef DS_MMA
    return static_cast<int>(cudaGetLastError());
}

/// Benchmark entry: fp8 weights with one M-tiled configuration (cfg 0: MT 64 / KC 256 / NG 1 /
/// DEPTH 4, today's; 1: 128/128/1/2; 2: 128/128/2/1; 3: 256/64/1/1; 4: 128/128/2/2;
/// 5: 256/64/2/1).
extern "C" int ds_mma_fp8_big(void* stream, const void* x, int64_t ldx, const void* w, const void* s, int64_t lds, int M,
                              int N, int K, void* y_bf16, int cfg) {
    if (M < 1 || K % 64) return cudaErrorInvalidValue;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto xb = static_cast<const __nv_bfloat16*>(x);
    auto wb = static_cast<const uint8_t*>(w);
    auto sb = static_cast<const uint8_t*>(s);
    auto yb = static_cast<__nv_bfloat16*>(y_bf16);
#define DS_BIG(I, MT, KC, NG, DEPTH)                                                                                   \
    if (cfg == I) {                                                                                                  \
        dim3 grid((N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG), 1, (M + MT - 1) / MT);                                \
        mma_skinny_kernel<0, MT, KC, NG, DEPTH><<<grid, kThreads, 0, st>>>(xb, ldx, 0, 0, wb, sb, lds, M, N, K, K, yb,  \
                                                                           nullptr, N, nullptr);                       \
        return static_cast<int>(cudaGetLastError());                                                                  \
    }
    DS_BIG(0, 64, 256, 1, 4) DS_BIG(1, 128, 128, 1, 2) DS_BIG(2, 128, 128, 2, 1) DS_BIG(3, 256, 64, 1, 1)
    DS_BIG(4, 128, 128, 2, 2) DS_BIG(5, 256, 64, 2, 1)
#undef DS_BIG
    return cudaErrorInvalidValue;
}

/// Rows per block of the tensor-core GEMM at M rows (callers size split-K with it).
extern "C" int ds_mma_rows_per_block(int M) { return kWarps * 8 * (M <= 16 ? 1 : M <= 32 ? 2 : M <= 64 ? 1 : 2); }

/// Tuning entry: fp8 weights at an explicit (NG, DEPTH) for the tile M needs (benchmarks only).
extern "C" int ds_mma_fp8_cfg(void* stream, const void* x, int64_t ldx, const void* w, const void* s, int64_t lds, int M,
                              int N, int K, int splits, void* y_bf16, void* partial, int ng, int depth) {
    if (M < 1 || M > 64 || K % 64) return cudaErrorInvalidValue;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const int ksplit = ((K + splits - 1) / splits + 63) / 64 * 64;
    auto xb = static_cast<const __nv_bfloat16*>(x);
    auto wb = static_cast<const uint8_t*>(w);
    auto sb = static_cast<const uint8_t*>(s);
    auto yb = static_cast<__nv_bfloat16*>(y_bf16);
    auto pp = static_cast<float*>(partial);
#define DS_CFG(MT, KC, NG, DEPTH)                                                                                    \
    if (ng == NG && depth == DEPTH && (M <= MT) && (MT == 16 || M > MT / 2)) {                                        \
        dim3 grid((N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG), splits);                                            \
        mma_skinny_kernel<0, MT, KC, NG, DEPTH><<<grid, kThreads, 0, st>>>(xb, ldx, 0, 0, wb, sb, lds, M, N, K, ksplit, \
                                                                           yb, nullptr, N, pp);                       \
        return static_cast<int>(cudaGetLastError());                                                                  \
    }
    DS_CFG(32, 512, 1, 4) DS_CFG(32, 512, 2, 2) DS_CFG(32, 512, 2, 4) DS_CFG(32, 512, 4, 1)
    DS_CFG(64, 256, 1, 4) DS_CFG(64, 256, 2, 2) DS_CFG(64, 256, 2, 4) DS_CFG(64, 256, 4, 1) DS_CFG(64, 256, 4, 2)
#undef DS_CFG
    return cudaErrorInvalidValue;
}

// ------------------------------------------------------------------ tensor-core FP4 MoE
// The routed-expert GEMMs of tools/fp4_moe_cuda.cu (moe_up_bm16_v2 / moe_down_bm16_v2) on
// mma.sync, for steps of more than 16 rows. The bm16 kernels run every routed pair's dot
// products on the CUDA cores: streaming-bound while an expert has a pair or two, compute-
// bound once pairs concentrate (a 64-row step over 16 experts: 13.8 ms against 1.5 here), and
// the Python engine leaves them for Triton tensor-core kernels above 64 pairs too. Same inputs
// - the route_small blocks of up to 16 pairs of one slot - and the same outputs and rounding
// points: gate and up rounded to bf16, the clamped
// SwiGLU times the routing weight rounded to bf16; each down output rounded to bf16 into its
// (k, token) row of `parts`. The dot products differ only in summation order (inside fp16's normal
// range), the same relation ds_mma_* has to the FMA GEMM. The tensor cores take f16: the
// hardware e2m1 -> f16x2 conversion is the weight decode, and the activations are bm16's own
// fp16 rounding (mma_fp4_chunk has the scale handling). Row-invariant: a pair's output never
// depends on which pairs share its block or its step.

constexpr int kMoeNB = 4, kMoeNG = 1, kMoeKC = 512;  // tile shape: see mma_fp4_chunk

__device__ __forceinline__ float silu_triton(float x) {
    // fp4_moe_cuda.cu::silu_like_triton, instruction for instruction
    float e, inv;
    const float exponent = -x * 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
    const float denom = 1.0f + e;
    asm("div.full.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(denom));
    return x * inv;
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 8 FP4 weights (4 bytes, low nibble = even k) as 4 f16x2 words: the B operands (b0, b1) of
// two consecutive mma steps.
__device__ __forceinline__ void e2m1x8_f16(uint32_t w, uint32_t* b) {
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(w));
}

// Stage rows of the block's pairs (row index per m, -1 for none) for K values k0..k0+kc as
// f16 - bm16's activation rounding (bf16 -> fp16), exact inside fp16's normal range.
template <int KC, int MT = 1>
__device__ __forceinline__ void stage_rows_f16(__half* xs, const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                               const int* rows, int k0, int kc) {
    constexpr int XS = KC + 8;
    for (int i = threadIdx.x; i < 16 * MT * (KC / 2); i += blockDim.x) {
        const int m = i / (KC / 2), j = (i % (KC / 2)) * 2;
        __half2 v = __floats2half2_rn(0.f, 0.f);
        if (rows[m] >= 0 && j < kc) {
            const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(x + (int64_t)rows[m] * ldx + k0 + j);
            v = __floats2half2_rn(__low2float(b), __high2float(b));
        }
        *reinterpret_cast<__half2*>(&xs[m * XS + j]) = v;
    }
}

// 2^-d as an f16x2 pair (d >= 0; subnormal past 14, zero past 24).
__device__ __forceinline__ uint32_t pow2_neg_f16x2(int d) {
    const uint32_t h = d <= 14 ? (uint32_t)(15 - d) << 10 : d <= 24 ? 1u << (24 - d) : 0u;
    return h | (h << 16);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// acc[w][g] += xs (16 rows) . W_w rows n0 + g*8 .. over the staged chunk, for NW weight
// matrices sharing the activations (up: w1 and w3). Per 128 K (four scale groups) lane
// (r, q) loads group q of row n0 + g*8 + r whole - 16 contiguous bytes, a quad covers 64,
// NB blocks of a pass 64 * NB: the access pattern that streams. (4-byte loads that kept each
// lane inside one group ran at 125 GB/s against 190 for this pattern at NB=1, bench_moe; a
// bf16 decode-and-repack of the weights instead of the f16 conversion, 119 GB/s.) A lane's 32 values feed its fragment
// positions of the block's eight mma steps, so one step mixes four groups: each lane
// pre-scales its group by 2^(its scale - the row's largest of the four) - exact in f16
// unless a group sits 2^-23 below its neighbours - and the largest scale multiplies the
// block's partial in fp32.
// Tile shape, examples/bench_moe.rs (2026-10-24, up + down, cold 192-expert arena; GB/s of
// weights at 6 / 16 / 64 rows spread over 32 / 72 / 162 experts, then 64 rows over 16):
//   NB=4 NG=1 KC=512   204 / 204 / 201, 1.52 ms   <- shipped
//   NB=2 NG=2 KC=512   195 / 196 / 193, 1.64 ms   (NG 1 or 4, KC 1024: within 2%)
//   NB=8 NG=1 KC=1024  210 / 209 / 208, 2.06 ms   (concentrated case loses a third)
//   ld.global.nc.L1::no_allocate on the weights: no gain spread, 1.80 ms concentrated.
// bm16 streams 214 / 213 / 167 and takes 13.8 ms concentrated: at <= 16 rows it is within 6%
// and keeps those steps bit-identical to one-row decode, so it keeps them.
// MT: m16 tiles of pairs a block (prefill's 64-pair blocks: MT = 4) - each weight decoded
// once for all of them; a pair's products and their order are the same at any MT.
template <int NW, int NG, int KC, int NB, int MT = 1>
__device__ __forceinline__ void mma_fp4_chunk(float (*acc)[MT][NG][4], const __half* xs, const uint8_t* const* w,
                                              const uint8_t* const* s, int N, int K, int n0, int k0, int kc) {
    constexpr int XS = KC + 8;
    const int lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    for (int kb0 = 0; kb0 < kc; kb0 += 128 * NB) {
        // NB blocks of 128 K per pass, all loads issued first
        uint4 wv[NB][NW][NG];
        uint32_t pre[NB][NW][NG], sref[NB][NW][NG];
#pragma unroll
        for (int bi = 0; bi < NB; ++bi)
#pragma unroll
            for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int n = n0 + g * 8 + r, kb = kb0 + bi * 128;
                    const bool ok = n < N && kb < kc;
                    const uint4* src = reinterpret_cast<const uint4*>(w[wi] + (int64_t)n * (K >> 1) + ((k0 + kb) >> 1) + q * 16);
                    wv[bi][wi][g] = ok ? *src : make_uint4(0, 0, 0, 0);
                    // the row's four scales (K/32 is a multiple of 4: K % 128 == 0)
                    const uint32_t sw = ok ? *reinterpret_cast<const uint32_t*>(s[wi] + (int64_t)n * (K >> 5) + ((k0 + kb) >> 5)) : 0u;
                    uint32_t m = __vmaxu4(sw, sw >> 16);
                    m = __vmaxu4(m, m >> 8) & 0xff;
                    sref[bi][wi][g] = m;
                    pre[bi][wi][g] = pow2_neg_f16x2((int)m - (int)((sw >> (8 * q)) & 0xff));
                }
#pragma unroll
        for (int bi = 0; bi < NB; ++bi) {
            const int kb = kb0 + bi * 128;
            if (kb >= kc) break;
            float c[NW][MT][NG][4];
#pragma unroll
            for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                    for (int g = 0; g < NG; ++g)
#pragma unroll
                        for (int i = 0; i < 4; ++i) c[wi][mt][g][i] = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // word j: this lane's values 8j .. 8j+8 of its group = steps 2j and 2j+1
                uint32_t a[MT][2][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                    for (int st = 0; st < 2; ++st) {
                        const int kx = kb + q * 32 + j * 8 + st * 4;
                        const uint2 r0 = *reinterpret_cast<const uint2*>(&xs[(mt * 16 + r) * XS + kx]);
                        const uint2 r1 = *reinterpret_cast<const uint2*>(&xs[(mt * 16 + r + 8) * XS + kx]);
                        a[mt][st][0] = r0.x; a[mt][st][1] = r1.x; a[mt][st][2] = r0.y; a[mt][st][3] = r1.y;
                    }
#pragma unroll
                for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                    for (int g = 0; g < NG; ++g) {
                        const uint4 v = wv[bi][wi][g];
                        const uint32_t word = j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
                        uint32_t b[4];
                        e2m1x8_f16(word, b);
#pragma unroll
                        for (int i = 0; i < 4; ++i) b[i] = hmul2(b[i], pre[bi][wi][g]);
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            mma_f16(c[wi][mt][g], a[mt][0][0], a[mt][0][1], a[mt][0][2], a[mt][0][3], b[0], b[1]);
                            mma_f16(c[wi][mt][g], a[mt][1][0], a[mt][1][1], a[mt][1][2], a[mt][1][3], b[2], b[3]);
                        }
                    }
            }
            // the C fragment's columns are rows 2q and 2q+1 of each 8-row group
#pragma unroll
            for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const float fa = e8m0_lin(__shfl_sync(0xffffffffu, sref[bi][wi][g], (2 * q) * 4));
                    const float fb = e8m0_lin(__shfl_sync(0xffffffffu, sref[bi][wi][g], (2 * q + 1) * 4));
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[wi][mt][g][0] += c[wi][mt][g][0] * fa;
                        acc[wi][mt][g][1] += c[wi][mt][g][1] * fb;
                        acc[wi][mt][g][2] += c[wi][mt][g][2] * fa;
                        acc[wi][mt][g][3] += c[wi][mt][g][3] * fb;
                    }
                }
        }
    }
}

// MT m16 tiles of pairs a block (bm = 16, 32, 64; KC and NB keep the staged rows in static
// shared memory). Measured slower past one tile at every prefill size (bench_moe_prefill.rs,
// 2,048 rows: 31.8 / 36.5 / 44.9 ms at 16 / 32 / 64 pairs) though each weight is decoded once a
// block: a warp's mma each re-read a 16-row activation fragment from shared memory against its
// one 8-row weight group, so the shared-memory traffic per mma is the same at any MT. The engine
// uses 16; the way past is more weight groups a warp (NG), which reuses each fragment.
template <int NG, int MT = 1, int KC = kMoeKC, int NB = kMoeNB>
__global__ void __launch_bounds__(256) moe_up_mma_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ s1,
    const uint8_t* __restrict__ w3, const uint8_t* __restrict__ s3, __nv_bfloat16* __restrict__ h, int64_t ldh,
    const float* __restrict__ route_weight, const int* __restrict__ block_slot, const int* __restrict__ block_route,
    int N, int K, float limit, int null_slot) {
    constexpr int BM = 16 * MT;
    __shared__ __align__(16) __half xs[BM * (KC + 8)];
    __shared__ int tok_s[BM], pair_s[BM];
    const int slot = block_slot[blockIdx.x];
    if (slot < 0 || slot == null_slot) return;
    if (threadIdx.x < BM) {
        const int route = block_route[(int64_t)blockIdx.x * BM + threadIdx.x];
        tok_s[threadIdx.x] = route < 0 ? -1 : route >> 16;
        pair_s[threadIdx.x] = route < 0 ? -1 : route & 0xffff;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int n0 = blockIdx.y * (kWarps * 8 * NG) + warp * 8 * NG;
    const uint8_t* w[2] = {w1 + (int64_t)slot * N * (K >> 1), w3 + (int64_t)slot * N * (K >> 1)};
    const uint8_t* sc[2] = {s1 + (int64_t)slot * N * (K >> 5), s3 + (int64_t)slot * N * (K >> 5)};
    float acc[2][MT][NG][4];
#pragma unroll
    for (int wi = 0; wi < 2; ++wi)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int g = 0; g < NG; ++g)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[wi][mt][g][i] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kc = min(KC, K - k0);
        __syncthreads();
        stage_rows_f16<KC, MT>(xs, x, ldx, tok_s, k0, kc);
        __syncthreads();
        mma_fp4_chunk<2, NG, KC, NB, MT>(acc, xs, w, sc, N, K, n0, k0, kc);
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int pair = pair_s[mt * 16 + r + half * 8];
            if (pair < 0) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = n0 + g * 8 + q * 2 + j;
                if (n >= N) continue;
                float gt = bf(to_bf(acc[0][mt][g][half * 2 + j]));
                float up = bf(to_bf(acc[1][mt][g][half * 2 + j]));
                gt = fminf(gt, limit);
                up = fminf(fmaxf(up, -limit), limit);
                h[(int64_t)pair * ldh + n] = to_bf(silu_triton(gt) * up * route_weight[pair]);
            }
        }
}

template <int NG, int MT = 1, int KC = kMoeKC, int NB = kMoeNB>
__global__ void __launch_bounds__(256) moe_down_mma_kernel(
    const __nv_bfloat16* __restrict__ h, int64_t ldh, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    float* __restrict__ y, int64_t ldy, const int* __restrict__ block_slot, const int* __restrict__ block_pair,
    int N, int K, int topk, int ntok, int null_slot, int partial, int parts_world) {
    constexpr int BM = 16 * MT;
    __shared__ __align__(16) __half xs[BM * (KC + 8)];
    __shared__ int pair_s[BM];
    const int slot = block_slot[blockIdx.x];
    if (slot < 0) return;
    const bool real = slot != null_slot;
    if (threadIdx.x < BM) pair_s[threadIdx.x] = block_pair[(int64_t)blockIdx.x * BM + threadIdx.x];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int n0 = blockIdx.y * (kWarps * 8 * NG) + warp * 8 * NG;
    const uint8_t* w[1] = {w2 + (int64_t)slot * N * (K >> 1)};
    const uint8_t* sc[1] = {s2 + (int64_t)slot * N * (K >> 5)};
    float acc[1][MT][NG][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[0][mt][g][i] = 0.f;
    if (real) {
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = min(KC, K - k0);
            __syncthreads();
            stage_rows_f16<KC, MT>(xs, h, ldh, pair_s, k0, kc);
            __syncthreads();
            mma_fp4_chunk<1, NG, KC, NB, MT>(acc, xs, w, sc, N, K, n0, k0, kc);
        }
    } else {
        __syncthreads();
    }
    const int local_n = N / parts_world;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int pair = pair_s[mt * 16 + r + half * 8];
            if (pair < 0) continue;
            const int64_t out_row = (int64_t)(pair % topk) * ntok + pair / topk;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = n0 + g * 8 + q * 2 + j;
                if (n >= N) continue;
                float v = real ? acc[0][mt][g][half * 2 + j] : 0.f;
                if (!partial) v = bf(to_bf(v));
                const int64_t off = parts_world > 1
                    ? ((int64_t)(n / local_n) * topk * ntok + out_row) * local_n + n % local_n
                    : out_row * ldy + n;
                y[off] = v;
            }
        }
}

// ------------------------------------------------------------------ prefill MoE (grouped GEMM)
// The routed experts for prompt-sized steps (thousands of pairs, ~50 an expert a layer), as a
// grouped GEMM: a block takes up to 64 pairs of one slot (ds_route_large's 64-pair blocks) and
// 64 weight rows, stages both operands in shared memory a 64-wide K step at a time - the
// activations as they are (bf16), the FP4 weights decoded exactly to bf16 (e2m1 x 2^scale is a
// bf16 value) - and each of 8 warps computes a 32 x 16 tile from ldmatrix fragments, so every
// fragment feeds two or four mma (the decode kernels re-read a 16-row fragment per mma and stay
// at ~14 TFLOP/s here). fp32 accumulation over the whole K; gate and up rounded to bf16, the
// clamped SwiGLU times the routing weight rounded to bf16 (up), each output rounded to bf16 into
// its (k, token) row of `parts` (down) - the decode kernels' rounding points. The dot products
// themselves are bf16 x bf16 rather than f16 x f16 (no activation rounding to f16), so this is
// not bit-identical to the decode kernels, only to itself at any step size: a pair's output
// never depends on the pairs sharing its block.
// At 2,048 rows over a layer's 250 residents (examples/bench_moe_prefill.rs) it reads ~2.4 GB
// of FP4 weights for 6.7 ms of tensor-core work (its time with the weights in L2), so it is
// bound by the weight stream: one K step's 32 bytes a row a fetch took 19.4 ms (124 GB/s, the
// rate of 32-byte runs on this bus); two steps' 64 bytes a fetch 16.2 ms at 174 registers (one
// block an SM), 13.9 ms held to 128 (two blocks an SM, 72 bytes of spills).
constexpr int kPmBM = 64, kPmBN = 128, kPmBK = 64, kPmLD = kPmBK + 8, kPmWN = kPmBN / 4 / 8;  // n8 tiles a warp

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// 8 e2m1 codes (one 4-byte word, low nibble first) times 2^(scale - 127) as 8 bf16 (exact).
__device__ __forceinline__ uint4 e2m1x8_bf16(uint32_t word, float scale) {
    uint32_t h[4];
    e2m1x8_f16(word, h);
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 v = *reinterpret_cast<const __half2*>(&h[i]);
        const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(v) * scale, __high2float(v) * scale);
        o[i] = *reinterpret_cast<const uint32_t*>(&b);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// NW = 2: up (w1, w3 -> h, SwiGLU epilogue); NW = 1: down (w2 -> parts)
template <int NW>
__global__ void __launch_bounds__(256, 2) moe_prefill_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ wa, const uint8_t* __restrict__ sa,
    const uint8_t* __restrict__ wb, const uint8_t* __restrict__ sb, void* __restrict__ out, int64_t ldo,
    const float* __restrict__ route_weight, const int* __restrict__ block_slot, const int* __restrict__ block_route,
    int N, int K, float limit, int null_slot, int topk, int ntok, int partial, int parts_world) {
    __shared__ __align__(16) __nv_bfloat16 xs[kPmBM * kPmLD];
    __shared__ __align__(16) __nv_bfloat16 ws[NW][kPmBN * kPmLD];
    __shared__ int row_s[kPmBM], pair_s[kPmBM];
    // grid (N tiles, pair blocks): a pair block's N tiles run back to back, so its activation
    // rows stay in L2 across them - with the pair blocks fastest instead, every N tile re-read
    // the step's activations from DRAM (h alone is 57 MB at 2,048 rows)
    const int pb = blockIdx.y;
    const int slot = block_slot[pb];
    if (slot < 0) return;
    const bool real = slot != null_slot;
    if (NW == 2 && !real) return;  // up: a null pair's h is never read
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < kPmBM) {
        const int route = block_route[(int64_t)pb * kPmBM + tid];
        pair_s[tid] = route < 0 ? -1 : route & 0xffff;
        // up reads the token's row of x, down the pair's row of h
        row_s[tid] = route < 0 ? -1 : (NW == 2 ? route >> 16 : route & 0xffff);
    }
    const int n_block = blockIdx.x * kPmBN;
    const uint8_t* w[2] = {wa + (int64_t)slot * N * (K >> 1), NW == 2 ? wb + (int64_t)slot * N * (K >> 1) : nullptr};
    const uint8_t* sc[2] = {sa + (int64_t)slot * N * (K >> 5), NW == 2 ? sb + (int64_t)slot * N * (K >> 5) : nullptr};
    // warp tile: rows wm*32 .. +32 (two m16), columns wn * kPmWN * 8 .. (kPmWN n8 tiles)
    const int wm = warp >> 2, wn = warp & 3;
    float acc[NW][2][kPmWN][4];
#pragma unroll
    for (int a = 0; a < NW; ++a)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < kPmWN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[a][i][j][e] = 0.f;
    __syncthreads();
    if (real) {
        // software pipeline: the next K step's activations and the next K step pair's weights
        // are in flight (in registers) while the tensor cores work on this step's shared-
        // memory tiles. Weights come two K steps a fetch, a row's 64 contiguous bytes from 4
        // adjacent lanes: one step's 32 bytes a row streamed at 124 GB/s, the bus's 32-byte
        // run rate, against 215 for 64-byte runs (a read-only walk of this layout). A thread
        // holds pieces (row tid / 4, chunk tid % 4) and (64 + tid / 4, chunk tid % 4 ^ 2) of
        // the pair's 4 x 32 codes a row, so each step stashes exactly one piece a matrix.
        constexpr int XP = kPmBM * (kPmBK / 8) / 256;
        static_assert(kPmBN == 128, "two 64-row weight pieces a thread");
        uint4 xr[XP];
        uint4 wc[2][NW], wnx[2][NW];
        float sc_c[2][NW], sc_n[2][NW];
        auto fetch_x = [&](int k0) {
#pragma unroll
            for (int u = 0; u < XP; ++u) {
                const int i = tid + u * 256, m = i / (kPmBK / 8), c = (i % (kPmBK / 8)) * 8;
                const int row = row_s[m];
                xr[u] = row >= 0 ? *reinterpret_cast<const uint4*>(x + (int64_t)row * ldx + k0 + c) : make_uint4(0, 0, 0, 0);
            }
        };
        // the pair of K steps at kp: chunk c covers codes kp + 32 c .. + 32 of its row
        auto fetch_w = [&](int kp, uint4 (&wv)[2][NW], float (&sv)[2][NW]) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int n = n_block + u * 64 + (tid >> 2), kk = kp + ((tid & 3) ^ (u * 2)) * 32;
                const bool in = n < N && kk < K;
#pragma unroll
                for (int a = 0; a < NW; ++a) {
                    wv[u][a] = in ? *reinterpret_cast<const uint4*>(w[a] + (int64_t)n * (K >> 1) + (kk >> 1)) : make_uint4(0, 0, 0, 0);
                    sv[u][a] = in ? e8m0_lin(sc[a][(int64_t)n * (K >> 5) + (kk >> 5)]) : 0.f;
                }
            }
        };
        // step `p` (0, 1) of the pair in wc: this thread's piece whose chunk falls in it
        auto stash = [&](int p) {
#pragma unroll
            for (int u = 0; u < XP; ++u) {
                const int i = tid + u * 256, m = i / (kPmBK / 8), c = (i % (kPmBK / 8)) * 8;
                *reinterpret_cast<uint4*>(&xs[m * kPmLD + c]) = xr[u];
            }
            const bool hi = ((tid & 3) >> 1) != p;  // piece u = 1 holds this step's chunk
            const int nr = (hi ? 64 : 0) + (tid >> 2), c = (((tid & 3) ^ (hi ? 2 : 0)) & 1) * 32;
#pragma unroll
            for (int a = 0; a < NW; ++a) {
                const uint4 v = hi ? wc[1][a] : wc[0][a];
                const float sv = hi ? sc_c[1][a] : sc_c[0][a];
                __nv_bfloat16* dst = &ws[a][nr * kPmLD + c];
                *reinterpret_cast<uint4*>(dst) = e2m1x8_bf16(v.x, sv);
                *reinterpret_cast<uint4*>(dst + 8) = e2m1x8_bf16(v.y, sv);
                *reinterpret_cast<uint4*>(dst + 16) = e2m1x8_bf16(v.z, sv);
                *reinterpret_cast<uint4*>(dst + 24) = e2m1x8_bf16(v.w, sv);
            }
        };
        fetch_x(0);
        fetch_w(0, wc, sc_c);
        for (int k0 = 0; k0 < K; k0 += kPmBK) {
            const int p = (k0 / kPmBK) & 1;
            stash(p);
            __syncthreads();
            if (k0 + kPmBK < K) fetch_x(k0 + kPmBK);
            if (p == 0 && k0 + 2 * kPmBK < K) fetch_w(k0 + 2 * kPmBK, wnx, sc_n);
#pragma unroll
            for (int ks = 0; ks < kPmBK / 16; ++ks) {
                const int kc = ks * 16;
                uint32_t af[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    ldsm_x4(af[i], &xs[(wm * 32 + i * 16 + (lane & 15)) * kPmLD + kc + (lane >> 4) * 8]);
#pragma unroll
                for (int a = 0; a < NW; ++a)
#pragma unroll
                    for (int jp = 0; jp < kPmWN / 2; ++jp) {
                        // two n8 tiles x (k lo, k hi): lanes 0-7 / 8-15 tile 0, 16-23 / 24-31 tile 1
                        uint32_t bfr[4];
                        ldsm_x4(bfr, &ws[a][(wn * kPmWN * 8 + jp * 16 + (lane >> 4) * 8 + (lane & 7)) * kPmLD + kc + ((lane >> 3) & 1) * 8]);
#pragma unroll
                        for (int i = 0; i < 2; ++i)
#pragma unroll
                            for (int j = 0; j < 2; ++j)
                                mma_bf16(acc[a][i][jp * 2 + j], af[i][0], af[i][1], af[i][2], af[i][3], bfr[2 * j], bfr[2 * j + 1]);
                    }
            }
            __syncthreads();
            if (p == 1) {
#pragma unroll
                for (int u = 0; u < 2; ++u)
#pragma unroll
                    for (int a = 0; a < NW; ++a) {
                        wc[u][a] = wnx[u][a];
                        sc_c[u][a] = sc_n[u][a];
                    }
            }
        }
    }
    // C fragment: rows (lane >> 2) and +8 of each m16, columns 2 * (lane & 3) + {0, 1} of each n8
    const int r = lane >> 2, q = lane & 3;
    const int local_n = N / parts_world;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int m = wm * 32 + i * 16 + r + half * 8;
            const int pair = pair_s[m];
            if (pair < 0) continue;
#pragma unroll
            for (int j = 0; j < kPmWN; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int n = n_block + wn * kPmWN * 8 + j * 8 + 2 * q + e;
                    if (n >= N) continue;
                    if (NW == 2) {
                        float gt = bf(to_bf(acc[0][i][j][half * 2 + e]));
                        float up = bf(to_bf(acc[NW - 1][i][j][half * 2 + e]));
                        gt = fminf(gt, limit);
                        up = fminf(fmaxf(up, -limit), limit);
                        static_cast<__nv_bfloat16*>(out)[(int64_t)pair * ldo + n] = to_bf(silu_triton(gt) * up * route_weight[pair]);
                    } else {
                        float v = real ? acc[0][i][j][half * 2 + e] : 0.f;
                        if (!partial) v = bf(to_bf(v));
                        const int64_t out_row = (int64_t)(pair % topk) * ntok + pair / topk;
                        const int64_t at = parts_world > 1 ? ((int64_t)(n / local_n) * topk * ntok + out_row) * local_n + n % local_n
                                                           : out_row * ldo + n;
                        static_cast<float*>(out)[at] = v;
                    }
                }
        }
}

// ------------------------------------------------------------------ norms
// v41_ref.rmsnorm: y = bf16(w * (x * rsqrt(mean(x^2) + eps))), fp32 math. One block per row.
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                               const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                               int64_t ldy, int D, float eps) {
    __shared__ float scratch[32];
    const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * ldx;
    float ss = 0.f;
    // unrolled so a thread's loads overlap (its FMA order, and so the bits, do not change)
#pragma unroll 8
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = bf(xr[i]);
        ss = fmaf(v, v, ss);
    }
    const float rs = rsqrtf(block_sum(ss, scratch) * (1.f / (float)D) + eps);
    __nv_bfloat16* yr = y + (int64_t)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = to_bf(bf(w[i]) * (bf(xr[i]) * rs));
}

// v41_ref.hc_pre then rmsnorm: v = bf16(sum_i pre[i] * h[i]) over the hc copies, then the
// rmsnorm above on v. `w` may be null for the bare hc_pre (no norm).
__global__ void hc_pre_norm_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ pre,
                                   const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                   int D, int hc, float eps) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __shared__ float scratch[32];
    const int m = blockIdx.x;
    const __nv_bfloat16* hr = h + (int64_t)m * hc * D;
    const float* pm = pre + (int64_t)m * hc;
    float ss = 0.f;
#pragma unroll 4
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = pm[0] * bf(hr[i]);
        for (int c = 1; c < hc; ++c) acc += pm[c] * bf(hr[(int64_t)c * D + i]);
        const __nv_bfloat16 b = to_bf(acc);
        v[i] = b;
        const float f = bf(b);
        ss = fmaf(f, f, ss);
    }
    __nv_bfloat16* yr = y + (int64_t)m * D;
    if (!w) {
        __syncthreads();
        for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = v[i];
        return;
    }
    const float rs = rsqrtf(block_sum(ss, scratch) * (1.f / (float)D) + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = to_bf(bf(w[i]) * (bf(v[i]) * rs));
}

// rsqrt(mean(x^2) + eps) per row of bf16 x (v41_ref.rms_rsqrt on x.float()).
__global__ void row_rsqrt_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int D, float eps,
                                 float* __restrict__ out) {
    __shared__ float scratch[32];
    const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * ldx;
    float ss = 0.f;
    // unrolled so a thread's loads overlap (its FMA order, and so the bits, do not change):
    // 80 dependent loads a thread made this 13.7 us at one row
#pragma unroll 8
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = bf(xr[i]);
        ss = fmaf(v, v, ss);
    }
    const float t = block_sum(ss, scratch);
    if (threadIdx.x == 0) out[blockIdx.x] = rsqrtf(t * (1.f / (float)D) + eps);
}

// The same two kernels over 8-wide (16-byte) loads, a thread per 8 columns: one load round
// a row instead of D / 256 dependent ones (13.5 and 20.6 us at one row before, nsys). Sums
// of squares go in another order than the scalar kernels'; each element's own arithmetic
// (the hc mix, the norm) is unchanged. D % 8 == 0.
__global__ void row_rsqrt_v8_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int D, float eps,
                                    float* __restrict__ out) {
    __shared__ float scratch[32];
    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)blockIdx.x * ldx);
    float ss = 0.f;
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) {
        float v[8];
        bf16x8_to_f32(xr[i], v);
#pragma unroll
        for (int j = 0; j < 8; ++j) ss = fmaf(v[j], v[j], ss);
    }
    const float t = block_sum(ss, scratch);
    if (threadIdx.x == 0) out[blockIdx.x] = rsqrtf(t * (1.f / (float)D) + eps);
}

__global__ void hc_pre_norm_v8_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ pre,
                                      const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int D,
                                      int hc, float eps) {
    __shared__ float scratch[32];
    const int m = blockIdx.x;
    const __nv_bfloat16* hr = h + (int64_t)m * hc * D;
    const float* pm = pre + (int64_t)m * hc;
    const int i = threadIdx.x;  // this thread's 8 columns
    __nv_bfloat16 vb[8];
    float ss = 0.f;
    if (i < D / 8) {
        float acc[8], t[8];
        bf16x8_to_f32(reinterpret_cast<const uint4*>(hr)[i], t);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = pm[0] * t[j];
        for (int c = 1; c < hc; ++c) {
            bf16x8_to_f32(reinterpret_cast<const uint4*>(hr + (int64_t)c * D)[i], t);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += pm[c] * t[j];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            vb[j] = to_bf(acc[j]);
            const float f = bf(vb[j]);
            ss = fmaf(f, f, ss);
        }
    }
    __nv_bfloat16* yr = y + (int64_t)m * D;
    if (!w) {
        if (i < D / 8)
            for (int j = 0; j < 8; ++j) yr[i * 8 + j] = vb[j];
        return;
    }
    const float rs = rsqrtf(block_sum(ss, scratch) * (1.f / (float)D) + eps);
    if (i < D / 8)
        for (int j = 0; j < 8; ++j) yr[i * 8 + j] = to_bf(bf(w[i * 8 + j]) * (bf(vb[j]) * rs));
}

// ------------------------------------------------------------------ hyper-connections
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// v41_ref.hc_split_sinkhorn on mixes * rsqrt (v41_ref.hc_mixes); hc = 4. One warp per row:
// lane i*4+j holds comb[i][j] and every row or column sum is gathered by shuffles and added
// in the serial order ((c0 + c1) + c2) + c3, so the bits are a single thread's. One thread
// per row took ~20 us at decode sizes (the 20 iterations run serially).
__global__ void hc_sinkhorn_kernel(const float* __restrict__ mixes, const float* __restrict__ rsq,
                                   const float* __restrict__ scale, const float* __restrict__ base,
                                   int M, int iters, float eps, float* __restrict__ pre,
                                   float* __restrict__ post, float* __restrict__ comb) {
    constexpr int HC = 4;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;
    const int m = warp;
    const float r = rsq[m];
    const float* mr = mixes + (int64_t)m * (2 + HC) * HC;
    if (lane < HC) {
        pre[m * HC + lane] = sigmoidf_(mr[lane] * r * scale[0] + base[lane]) + eps;
    } else if (lane < 2 * HC) {
        const int i = lane - HC;
        post[m * HC + i] = 2.f * sigmoidf_(mr[HC + i] * r * scale[1] + base[HC + i]);
    }
    // lanes 0..15 hold the matrix; the others shadow lane & 15 so every shuffle is full-warp
    const int li = (lane & 15) >> 2, lj = lane & 3;
    float c = mr[2 * HC + li * HC + lj] * r * scale[2] + base[2 * HC + li * HC + lj];
    auto row = [&](float v, int k) { return __shfl_sync(0xffffffffu, v, li * 4 + k); };
    auto col = [&](float v, int k) { return __shfl_sync(0xffffffffu, v, k * 4 + lj); };
    {
        float mxr = -INFINITY;
        for (int k = 0; k < HC; ++k) mxr = fmaxf(mxr, row(c, k));
        const float e = expf(c - mxr);
        float sum = 0.f;
        for (int k = 0; k < HC; ++k) sum += row(e, k);
        c = e / sum + eps;
    }
    // column normalize (sum over the first index), then iters-1 x (row, column)
    {
        float s = 0.f;
        for (int k = 0; k < HC; ++k) s += col(c, k);
        c = c / (s + eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
        float s = 0.f;
        for (int k = 0; k < HC; ++k) s += row(c, k);
        c = c / (s + eps);
        s = 0.f;
        for (int k = 0; k < HC; ++k) s += col(c, k);
        c = c / (s + eps);
    }
    if (lane < HC * HC) comb[(m * HC + li) * HC + lj] = c;
}

// v41_ref.hc_post: out[j] = bf16(post[j] * y + sum_i comb[i, j] * residual[i]). In place on
// `h` is safe: each thread reads all hc copies of its column before writing any.
__global__ void hc_post_kernel(const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* residual,
                               const float* __restrict__ post, const float* __restrict__ comb,
                               __nv_bfloat16* out, int D) {
    constexpr int HC = 4;
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float yv = bf(y[(int64_t)m * D + d]);
    float r[HC];
    for (int i = 0; i < HC; ++i) r[i] = bf(residual[((int64_t)m * HC + i) * D + d]);
    const float* cm = comb + m * HC * HC;
    float o[HC];
    for (int j = 0; j < HC; ++j) {
        float mixed = cm[0 * HC + j] * r[0];
        for (int i = 1; i < HC; ++i) mixed = fmaf(cm[i * HC + j], r[i], mixed);
        o[j] = post[m * HC + j] * yv + mixed;
    }
    for (int j = 0; j < HC; ++j) out[((int64_t)m * HC + j) * D + d] = to_bf(o[j]);
}

// ------------------------------------------------------------------ rope
// v41_ref.apply_rotary on the last `rd` dims of each head: adjacent pairs as complex numbers
// times freqs_cis[pos] (cos, sin) in fp32, rounded back to bf16; `inverse` uses the conjugate.
__global__ void rope_kernel(__nv_bfloat16* x, int64_t ld_row, int heads, int64_t ld_head, int offset,
                            int rd, const int* __restrict__ pos, const float2* __restrict__ table,
                            int inverse) {
    const int row = blockIdx.x;
    const int pairs = rd / 2;
    const float2* f = table + (int64_t)pos[row] * pairs;
    for (int t = threadIdx.x; t < heads * pairs; t += blockDim.x) {
        const int h = t / pairs, i = t % pairs;
        __nv_bfloat16* p = x + (int64_t)row * ld_row + (int64_t)h * ld_head + offset + 2 * i;
        const float a = bf(p[0]), b = bf(p[1]);
        const float c = f[i].x, s = inverse ? -f[i].y : f[i].y;
        p[0] = to_bf(a * c - b * s);
        p[1] = to_bf(a * s + b * c);
    }
}

// ------------------------------------------------------------------ router
// model.py::moe for text tokens: scores = sqrt(softplus(z)); selection on scores + bias
// (optionally masked to resident experts); weights from the raw scores, normalized over the
// top-k and scaled. torch.topk order: descending; ties take the lower expert id here.
// Demand recording for adaptive residency (model.py::_record_prune_miss): each row's top-K
// of the UNMASKED selection logits (score + bias; ties to the lower index), per pick
// counts[lane][e] += 1 and mass[lane][e] += score[e] (the pre-bias score), and per (lane,
// phase) [missed, picks, missed score, score] with a miss a pick `keep` masks. Rows whose lane
// is >= `lanes` (filler) record nothing. `hot`: per row and unmasked pick its share of the
// row's top-K score, zero for a routable pick, and the expert id (hot reads' candidates).
struct RouterRec {
    double* counts;
    double* mass;
    double* miss;
    int64_t lane_stride;
    const int* lane;
    int lanes;
    const int* phase;
    float* hot;
    // the step's live pairs (rows x K) or null: rows past them are filler and ask for nothing
    const int* live;
    // image rows (model.py moe's image_mask: a whole image span) select with bias_vl, or null
    const int* img;
    const float* bias_vl;
};

__global__ void router_kernel(const float* __restrict__ z, const float* __restrict__ bias,
                              const uint8_t* __restrict__ keep, int E, int K, float route_scale,
                              int* __restrict__ idx_out, float* __restrict__ w_out, RouterRec rec) {
    extern __shared__ float sh[];
    float* score = sh;      // [E]
    float* sel = sh + E;    // [E]
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int m = blockIdx.x;
    if (rec.img && rec.img[m]) bias = rec.bias_vl;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float x = z[(int64_t)m * E + e];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        score[e] = sqrtf(sp);
        sel[e] = (keep && !keep[e]) ? -INFINITY : score[e] + bias[e];
    }
    __syncthreads();
    int chosen[8];
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int besti = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > best || (v == best && e < besti)) { best = v; besti = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
            if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (lane == 0) { bv[warp] = best; bi[warp] = besti; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int bidx = bi[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
                if (bv[i] > b || (bv[i] == b && bi[i] < bidx)) { b = bv[i]; bidx = bi[i]; }
            bi[0] = bidx;
            sel[bidx] = -INFINITY;
        }
        __syncthreads();
        chosen[k] = bi[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += score[chosen[k]];
        for (int k = 0; k < K; ++k) {
            idx_out[m * K + k] = chosen[k];
            w_out[m * K + k] = score[chosen[k]] / (sum + 1e-20f) * route_scale;
        }
    }
    const int ln = rec.lane ? rec.lane[m] : 0;
    const bool filler = ln >= rec.lanes || (rec.live && m * K >= *rec.live);
    const bool record = rec.counts && !filler;
    if (!record && !rec.hot) return;
    // the unmasked choices: the same selection over every expert
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) sel[e] = score[e] + bias[e];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int besti = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > best || (v == best && e < besti)) { best = v; besti = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
            if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (lane == 0) { bv[warp] = best; bi[warp] = besti; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int bidx = bi[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
                if (bv[i] > b || (bv[i] == b && bi[i] < bidx)) { b = bv[i]; bidx = bi[i]; }
            bi[0] = bidx;
            sel[bidx] = -INFINITY;
        }
        __syncthreads();
        chosen[k] = bi[0];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    float ssum = 0.f;
    for (int k = 0; k < K; ++k) ssum += score[chosen[k]];
    double missed = 0.0, missed_mass = 0.0, mass = 0.0;
    for (int k = 0; k < K; ++k) {
        const int e = chosen[k];
        const bool miss = keep && !keep[e];
        if (record) {
            atomicAdd(&rec.counts[(int64_t)ln * rec.lane_stride + e], 1.0);
            atomicAdd(&rec.mass[(int64_t)ln * rec.lane_stride + e], (double)score[e]);
            missed += miss;
            missed_mass += miss ? (double)score[e] : 0.0;
            mass += score[e];
        }
        if (rec.hot) {
            // a filler row asks for nothing
            rec.hot[((int64_t)m * K + k) * 2] = miss && !filler ? score[e] / (ssum + 1e-20f) : 0.f;
            rec.hot[((int64_t)m * K + k) * 2 + 1] = (float)e;
        }
    }
    if (record) {
        double* ms = rec.miss + (int64_t)ln * 8 + (rec.phase && rec.phase[m] ? 4 : 0);
        atomicAdd(&ms[0], missed);
        atomicAdd(&ms[1], (double)K);
        atomicAdd(&ms[2], missed_mass);
        atomicAdd(&ms[3], mass);
    }
}

// Hot reads' marks (fastdecode._hot_async_begin / _end): `marks` [n][3] = (LUT index, keep
// index, ring slot). Apply saves each LUT entry and keep byte into `saved` [n][2] and routes
// the expert to its ring slot; undo puts the saved values back.
__global__ void hot_marks_kernel(int* __restrict__ lut, uint8_t* __restrict__ keep, const int* __restrict__ marks,
                                 int* __restrict__ saved, int n, int apply) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int li = marks[3 * i], ki = marks[3 * i + 1];
    if (apply) {
        saved[2 * i] = lut[li];
        saved[2 * i + 1] = keep[ki];
        lut[li] = marks[3 * i + 2];
        keep[ki] = 1;
    } else {
        lut[li] = saved[2 * i];
        keep[ki] = (uint8_t)saved[2 * i + 1];
    }
}

// A hot layer's candidates (hot_reads.HotReads.candidates): of the step's M x K (share, id)
// entries the W largest shares, descending (ties: the lower row, then the lower pick), as
// out[0..n) shares and out[W..W+n) ids (as f32), zeros past n. One block.
__global__ void __launch_bounds__(256) hot_candidates_kernel(const float* __restrict__ hot, int n, int W,
                                                             float* __restrict__ out) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int taken[32];
    const int tid = threadIdx.x;
    for (int w = 0; w < W; ++w) {
        float best = -1.f;
        int besti = 0x7fffffff;
        for (int i = tid; i < n; i += blockDim.x) {
            bool used = false;
            for (int j = 0; j < w; ++j) used |= taken[j] == i;
            const float v = hot[2 * i];
            if (!used && (v > best || (v == best && i < besti))) { best = v; besti = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
            if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
        }
        if ((tid & 31) == 0) { bv[tid >> 5] = best; bi[tid >> 5] = besti; }
        __syncthreads();
        if (tid == 0) {
            float b = bv[0];
            int bidx = bi[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
                if (bv[i] > b || (bv[i] == b && bi[i] < bidx)) { b = bv[i]; bidx = bi[i]; }
            taken[w] = bidx;
            const bool ok = bidx < n;
            out[w] = ok ? hot[2 * bidx] : 0.f;
            out[W + w] = ok ? hot[2 * bidx + 1] : 0.f;
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------ FFN pieces
// v41_ref.expert_ffn's activation for the shared expert: clamp, silu(gate) * up, bf16.
// gate/up are the bf16 outputs of w1/w3 (or the two halves of a merged w13 output).
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ gate, int64_t ldg,
                              const __nv_bfloat16* __restrict__ up, int64_t ldu,
                              __nv_bfloat16* __restrict__ h, int64_t ldh, int N, float limit) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float g = bf(gate[(int64_t)m * ldg + n]);
    float u = bf(up[(int64_t)m * ldu + n]);
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    const float silu = g / (1.f + expf(-g));
    h[(int64_t)m * ldh + n] = to_bf(silu * u);
}

// model.py::moe tail on one box: out = bf16(sum_k parts[k*M + m] + shared[m]); parts are the
// fp32 rows the FP4 down kernel writes (each expert already rounded to bf16).
__global__ void moe_combine_kernel(const float* __restrict__ parts, int K, int M,
                                   const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ out,
                                   int D) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = parts[(int64_t)m * D + d];
    for (int k = 1; k < K; ++k) acc += parts[((int64_t)k * M + m) * D + d];
    out[(int64_t)m * D + d] = to_bf(acc + bf(shared[(int64_t)m * D + d]));
}

// tools/fp8_linear.py::quantize_to_fp8 for a bf16 [N][K] weight (the drafter's copy of the
// LM head, DSV41_DRAFT_HEAD_FMT=fp8): per 32x32 block, scale 2^ceil(log2(amax/448)), raised
// once more if that would still clip, then the round-to-nearest e4m3 conversion. One block a tile.
__global__ void quant_fp8_block32_kernel(const __nv_bfloat16* __restrict__ w, int N, int K, uint8_t* __restrict__ codes,
                                         uint8_t* __restrict__ scales, int nbk) {
    __shared__ float red[256];
    const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
    float amax = 0.f;
    for (int i = threadIdx.x; i < 1024; i += 256) {
        const int r = r0 + i / 32, c = c0 + i % 32;
        if (r < N && c < K) amax = fmaxf(amax, fabsf(bf(w[(int64_t)r * K + c])));
    }
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    amax = fmaxf(red[0], 448.f * exp2f(-126.f));
    float e = ceilf(log2f(amax / 448.f));
    if (amax > 448.f * exp2f(e)) e += 1.f;
    const float b = fminf(fmaxf(e + 127.f, 0.f), 255.f);
    if (threadIdx.x == 0) scales[(int64_t)blockIdx.y * nbk + blockIdx.x] = (uint8_t)b;
    const float inv = exp2f(127.f - b);
    for (int i = threadIdx.x; i < 1024; i += 256) {
        const int r = r0 + i / 32, c = c0 + i % 32;
        if (r < N && c < K)
            codes[(int64_t)r * K + c] = (uint8_t)__nv_cvt_float_to_fp8(bf(w[(int64_t)r * K + c]) * inv, __NV_SATFINITE, __NV_E4M3);
    }
}

// Expert parallel: moe_combine split at the all-reduce. Each rank sums its k parts (zeros
// for the peer's experts) in k order into fp32, the pair sums those (one a + b, the same
// bits on both ranks), and the shared expert joins at the single-box rounding point.
__global__ void moe_partial_kernel(const float* __restrict__ parts, int K, int M, float* __restrict__ out, int D) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = parts[(int64_t)m * D + d];
    for (int k = 1; k < K; ++k) acc += parts[((int64_t)k * M + m) * D + d];
    out[(int64_t)m * D + d] = acc;
}

__global__ void moe_finish_kernel(const float* __restrict__ routed, const __nv_bfloat16* __restrict__ shared,
                                  __nv_bfloat16* __restrict__ out, int D) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    out[(int64_t)m * D + d] = to_bf(routed[(int64_t)m * D + d] + bf(shared[(int64_t)m * D + d]));
}

// ------------------------------------------------------------------ attention (decode, split keys)
// Flash-decoding of the same sinked softmax (tools/decode_attn.py's kernel, in CUDA): a block
// takes 16 heads of one row (one mma m16 tile of queries; MQA, so every head reads the same
// keys) and a run of 64-key tiles; each tile is staged in shared memory once for all 16 heads.
// Scores q.k on mma.sync bf16 -> fp32, scaled; an online softmax per head (max clamped to
// >= -1e30 like the reference, so an all-masked run stays finite); PV on the tensor cores with
// the probabilities split into a bf16 high part and a bf16 remainder (~16 mantissa bits, the
// Python kernel's PV_SPLIT). Each block writes an unnormalized partial (o, m, l); the combine
// kernel adds the sink and normalizes. The key axis is split so a decode row fills the GPU
// (the per-head kernel above walked keys serially: 328 us a layer at 1,024 indexed keys).
// v41_ref.fp4_qdq(x, 16, "e4m3") for one 16-value group: the E4M3 scale byte and the 16
// E2M1 codes (bit 3 the sign: t < 0, so a tiny negative that rounds to 0 keeps its sign and
// -0.0 does not), element j in bits 4j..4j+3 of the 64-bit `codes`. The QDQ kernel and the
// packed KV writer both quantize through this, so a packed row decodes to the QDQ'd bf16 row.
__device__ __forceinline__ void fp4_quant16(const float* v, uint8_t& s8, uint64_t& codes) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, 6.f * 0.001953125f);  // clamp_min(6 * 2^-9)
    s8 = __nv_cvt_float_to_fp8(amax / 6.f, __NV_NOSAT, __NV_E4M3);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(s8, __NV_E4M3);
    const float s = __half2float(*reinterpret_cast<const __half*>(&hr));
    codes = 0;
    for (int i = 0; i < 16; ++i) {
        const float t = fminf(fmaxf(v[i] / s, -6.f), 6.f);
        const float a = fabsf(t);
        // torch.bucketize(|t|, midpoints) with right=False: a value equal to a midpoint
        // takes the lower grid point
        const uint32_t m = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.f);
        codes |= (uint64_t)(m | (t < 0.f ? 8u : 0u)) << (4 * i);
    }
}

// An E2M1 code times its group's E4M3 scale as bf16 bits: the product has at most 6
// significant bits, so it is exact; the sign goes on after, keeping -0.
__device__ __forceinline__ uint16_t fp4_value(uint32_t code, float s) {
    const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const __nv_bfloat16 b = to_bf(g[code & 7] * s);
    return *reinterpret_cast<const uint16_t*>(&b) | (uint16_t)((code & 8) << 12);
}

__device__ __forceinline__ float e4m3_float(uint8_t s8) {
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(s8, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
}

// Packed KV rows (engine/packed_kv.py, DSV41_PACKED_KV=1): a D-value row is D/2 code bytes
// (element e in byte e/2, low nibble when e is even) then D/16 E4M3 scale bytes (group g in
// byte D/2 + g) - 288 bytes for D = 512. Values 8c..8c+7 of a packed 512-row as bf16x8.
constexpr int kPkRow = 288;
__device__ __forceinline__ uint4 fp4_unpack8(const uint8_t* row, int c) {
    const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 4 * c);
    const float s = e4m3_float(row[256 + c / 2]);
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        o[i] = (uint32_t)fp4_value((w >> (8 * i)) & 15, s) | ((uint32_t)fp4_value((w >> (8 * i + 4)) & 15, s) << 16);
    return make_uint4(o[0], o[1], o[2], o[3]);
}

constexpr int kFdH = 16, kFdN = 64, kFdD = 512, kFdKS = kFdD + 8, kFdPS = kFdN + 4;
// Keys a split covers. Fixed in key space - never chosen from the row count - so a row's
// partials, and its bits, do not depend on what else shares its step: a verify block's rows
// equal one-row decode, as speculation requires.
constexpr int kFdSplitTiles = 2;

__device__ __forceinline__ void ldsm_x2_trans(uint32_t& b0, uint32_t& b1, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(a));
}

__global__ void __launch_bounds__(256) attn_fd_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ring_base, int ring_size,
    const int* __restrict__ pos, int W, const int* __restrict__ win_lo, const __nv_bfloat16* __restrict__ ckv_base,
    const int* __restrict__ cidx, int C, float scale, int H, const int* __restrict__ row_lane,
    int64_t ring_lstride, int64_t ckv_lstride, int tiles_per_split, int S, float* __restrict__ part_o,
    float* __restrict__ part_ml, const int* __restrict__ decode_rows, int ckv_fp4) {
    // a prompt row of a mixed step is attn_prefill_kernel's
    if (decode_rows && !decode_rows[blockIdx.z]) return;
    extern __shared__ __align__(16) unsigned char fd_smem[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(fd_smem);            // [16][KS]
    __nv_bfloat16* Ks = Qs + kFdH * kFdKS;                                     // [64][KS]
    float* Ps = reinterpret_cast<float*>(Ks + kFdN * kFdKS);                    // [16][PS]
    float* m_run = Ps + kFdH * kFdPS;
    float* l_run = m_run + kFdH;
    float* alpha = l_run + kFdH;
    int* valid = reinterpret_cast<int*>(alpha + kFdH);
    int* idx_s = valid + kFdN;                                                  // [split's keys]
    const int m = blockIdx.z, split = blockIdx.y, h0 = blockIdx.x * kFdH;
    const int nh = min(kFdH, H - h0);
    const int NK = W + C;
    const int p = pos[m];
    const int ln = row_lane ? row_lane[m] : 0;
    // the lane's window floor: the first position whose window KV this layer's ring holds
    // (null: 0) - a decoder layer's after a prompt's bounded replay
    const int lo = win_lo ? win_lo[ln] : 0;
    const __nv_bfloat16* ring = ring_base + (int64_t)ln * ring_lstride;
    const __nv_bfloat16* ckv = ckv_base ? ckv_base + (int64_t)ln * ckv_lstride : nullptr;
    // packed KV: the same lane offset (ckv_lstride counts bf16 units), rows of kPkRow bytes
    const uint8_t* ckp = ckv_fp4 && ckv ? reinterpret_cast<const uint8_t*>(ckv) : nullptr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, qd = lane & 3;
    const int n_tiles = (NK + kFdN - 1) / kFdN;
    const int t0 = split * tiles_per_split, t1 = min(n_tiles, t0 + tiles_per_split);

    for (int i = tid; i < kFdH * (kFdD / 8); i += blockDim.x) {
        const int hh = i / (kFdD / 8), c = i % (kFdD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (hh < nh) v = reinterpret_cast<const uint4*>(q + ((int64_t)m * H + h0 + hh) * kFdD)[c];
        *reinterpret_cast<uint4*>(Qs + hh * kFdKS + c * 8) = v;
    }
    if (tid < kFdH) { m_run[tid] = -1e30f; l_run[tid] = 0.f; }
    // the split's compressed picks in shared memory, and the next tile's keys in registers
    // while the tensor cores work on this one (attn_prefill_kernel's pipeline; the math and
    // its order are unchanged)
    const int s_lo = t0 * kFdN, s_hi = min(NK, t1 * kFdN);
    for (int n = s_lo + tid; n < s_hi; n += blockDim.x)
        if (n >= W) idx_s[n - s_lo] = cidx[(int64_t)m * C + (n - W)];
    constexpr int KP = kFdN * (kFdD / 8) / 256;
    uint4 kreg[KP];
    bool kval[KP];
    auto fetch = [&](int n0) {
#pragma unroll
        for (int u = 0; u < KP; ++u) {
            const int n = n0 + tid / 64 + 4 * u, c = tid % 64;
            const __nv_bfloat16* kr = nullptr;
            if (n < W) {
                const int kp = p - (W - 1 - n);
                if (kp >= 0 && kp >= lo) kr = ring + (int64_t)(kp % ring_size) * kFdD;
            } else if (n < NK) {
                const int jj = idx_s[n - s_lo];
                if (jj >= 0) {
                    if (ckp) {
                        kval[u] = true;
                        kreg[u] = fp4_unpack8(ckp + (int64_t)jj * kPkRow, c);
                        continue;
                    }
                    kr = ckv + (int64_t)jj * kFdD;
                }
            }
            kval[u] = kr != nullptr;
            kreg[u] = kr ? reinterpret_cast<const uint4*>(kr)[c] : make_uint4(0, 0, 0, 0);
        }
    };
    __syncthreads();  // idx_s
    if (t0 < t1) fetch(t0 * kFdN);
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;

    for (int tile = t0; tile < t1; ++tile) {
        __syncthreads();  // the previous tile's PV is done with Ks / Ps
#pragma unroll
        for (int u = 0; u < KP; ++u) {
            const int j = tid / 64 + 4 * u, c = tid % 64;
            *reinterpret_cast<uint4*>(Ks + j * kFdKS + c * 8) = kreg[u];
            if (c == 0) valid[j] = kval[u];
        }
        __syncthreads();
        if (tile + 1 < t1) fetch((tile + 1) * kFdN);
        // scores: warp w takes keys w*8 .. w*8+8 for all 16 heads
        {
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            const __nv_bfloat16* kn = Ks + (warp * 8 + r) * kFdKS;
#pragma unroll 8
            for (int ks = 0; ks < kFdD / 16; ++ks) {
                const int d0 = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(Qs + r * kFdKS + d0 + 2 * qd);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(Qs + (r + 8) * kFdKS + d0 + 2 * qd);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(Qs + r * kFdKS + d0 + 8 + 2 * qd);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(Qs + (r + 8) * kFdKS + d0 + 8 + 2 * qd);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kn + d0 + 2 * qd);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kn + d0 + 8 + 2 * qd);
                mma_bf16(c4, a0, a1, a2, a3, b0, b1);
            }
            const int k0 = warp * 8 + 2 * qd;
            Ps[r * kFdPS + k0] = valid[k0] ? c4[0] * scale : -INFINITY;
            Ps[r * kFdPS + k0 + 1] = valid[k0 + 1] ? c4[1] * scale : -INFINITY;
            Ps[(r + 8) * kFdPS + k0] = valid[k0] ? c4[2] * scale : -INFINITY;
            Ps[(r + 8) * kFdPS + k0 + 1] = valid[k0 + 1] ? c4[3] * scale : -INFINITY;
        }
        __syncthreads();
        // online softmax: warp w takes heads 2w and 2w+1, a lane keys lane and lane + 32
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int hh = warp * 2 + u;
            const float s0 = Ps[hh * kFdPS + lane], s1 = Ps[hh * kFdPS + lane + 32];
            const float mo = m_run[hh];
            const float mn = fmaxf(mo, warp_max(fmaxf(s0, s1)));
            const float e0 = expf(s0 - mn), e1 = expf(s1 - mn);
            const float ls = warp_sum(e0 + e1);
            Ps[hh * kFdPS + lane] = e0;
            Ps[hh * kFdPS + lane + 32] = e1;
            __syncwarp();
            if (lane == 0) {
                const float al = expf(mo - mn);
                alpha[hh] = al;
                l_run[hh] = l_run[hh] * al + ls;
                m_run[hh] = mn;
            }
        }
        __syncthreads();
        const float al0 = alpha[r], al1 = alpha[r + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            acc[nt][0] *= al0; acc[nt][1] *= al0;
            acc[nt][2] *= al1; acc[nt][3] *= al1;
        }
        // PV: warp w takes dims w*64 .. w*64+64; P split into bf16 hi + lo
#pragma unroll
        for (int ks = 0; ks < kFdN / 16; ++ks) {
            const int k0 = ks * 16;
            uint32_t ah[4], alo[4];
            const float* pr0 = Ps + r * kFdPS + k0 + 2 * qd;
            const float* pr1 = Ps + (r + 8) * kFdPS + k0 + 2 * qd;
            const float pv[4][2] = {{pr0[0], pr0[1]}, {pr1[0], pr1[1]}, {pr0[8], pr0[9]}, {pr1[8], pr1[9]}};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __nv_bfloat162 hi = __floats2bfloat162_rn(pv[i][0], pv[i][1]);
                const __nv_bfloat162 lo = __floats2bfloat162_rn(pv[i][0] - __low2float(hi), pv[i][1] - __high2float(hi));
                ah[i] = *reinterpret_cast<const uint32_t*>(&hi);
                alo[i] = *reinterpret_cast<const uint32_t*>(&lo);
            }
            const int key = k0 + (lane & 15);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                uint32_t b0, b1;
                ldsm_x2_trans(b0, b1, Ks + key * kFdKS + warp * 64 + nt * 8);
                mma_bf16(acc[nt], ah[0], ah[1], ah[2], ah[3], b0, b1);
                mma_bf16(acc[nt], alo[0], alo[1], alo[2], alo[3], b0, b1);
            }
        }
    }
    __syncthreads();
    const int64_t base = ((int64_t)m * S + split) * H + h0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int d = warp * 64 + nt * 8 + 2 * qd;
        if (r < nh) {
            part_o[(base + r) * kFdD + d] = acc[nt][0];
            part_o[(base + r) * kFdD + d + 1] = acc[nt][1];
        }
        if (r + 8 < nh) {
            part_o[(base + r + 8) * kFdD + d] = acc[nt][2];
            part_o[(base + r + 8) * kFdD + d + 1] = acc[nt][3];
        }
    }
    if (tid < nh) {
        part_ml[(base + tid) * 2] = m_run[tid];
        part_ml[(base + tid) * 2 + 1] = l_run[tid];
    }
}

// ------------------------------------------------------------------ attention (prompt rows)
// The split kernel's tiles and arithmetic in one pass over every key (tools/prefill_attn.py's
// shape: a block takes a row's MT x 16 heads and walks all its key tiles), with the sink and
// the normalization fused: no partials, no combine. A prompt row has keys enough to keep the
// GPU busy without a split, and the split's partial traffic (S x [heads][512] fp32 a row) was
// most of a prefill step's attention. `rows`: rows with rows[m] != 0 are a decode step's and
// left to the split kernel (null: every row is this kernel's).
// NT: keys a tile (64; 32 for 32 heads a block, whose 64-key tiles outgrow shared memory).
// At 2,048 rows over 128 + 1,024 keys, 32 heads (examples/bench_attn.rs --prefill): 16 heads
// with each tile loaded between barriers 14.7 ms, 32 heads 10.5, and with the next tile's keys
// in registers during this one's math 7.39 / 6.87 ms.
template <int MT, int NT>
__global__ void __launch_bounds__(256) attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ring_base, int ring_size,
    const int* __restrict__ pos, int W, const int* __restrict__ win_lo, const __nv_bfloat16* __restrict__ ckv_base,
    const int* __restrict__ cidx, int C, float scale, int H, const int* __restrict__ row_lane,
    int64_t ring_lstride, int64_t ckv_lstride, const float* __restrict__ sink, const int* __restrict__ rows,
    __nv_bfloat16* __restrict__ out, int ckv_fp4) {
    constexpr int HB = 16 * MT, PS = NT + 4;
    static_assert(NT == 32 || NT == 64, "a softmax lane takes one or two keys");
    extern __shared__ __align__(16) unsigned char fp_smem[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(fp_smem);            // [HB][KS]
    __nv_bfloat16* Ks = Qs + HB * kFdKS;                                       // [NT][KS]
    float* Ps = reinterpret_cast<float*>(Ks + NT * kFdKS);                      // [HB][PS]
    float* m_run = Ps + HB * PS;
    float* l_run = m_run + HB;
    float* alpha = l_run + HB;
    int* valid = reinterpret_cast<int*>(alpha + HB);
    int* idx_s = valid + NT;                                                    // [C]: the row's picks
    const int m = blockIdx.y, h0 = blockIdx.x * HB;
    if (rows && rows[m]) return;
    const int nh = min(HB, H - h0);
    const int NK = W + C;
    const int p = pos[m];
    const int ln = row_lane ? row_lane[m] : 0;
    // the lane's window floor: the first position whose window KV this layer's ring holds
    // (null: 0) - a decoder layer's after a prompt's bounded replay
    const int lo = win_lo ? win_lo[ln] : 0;
    const __nv_bfloat16* ring = ring_base + (int64_t)ln * ring_lstride;
    const __nv_bfloat16* ckv = ckv_base ? ckv_base + (int64_t)ln * ckv_lstride : nullptr;
    // packed KV: the same lane offset (ckv_lstride counts bf16 units), rows of kPkRow bytes
    const uint8_t* ckp = ckv_fp4 && ckv ? reinterpret_cast<const uint8_t*>(ckv) : nullptr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, qd = lane & 3;
    const int n_tiles = (NK + NT - 1) / NT;

    for (int i = tid; i < HB * (kFdD / 8); i += blockDim.x) {
        const int hh = i / (kFdD / 8), c = i % (kFdD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (hh < nh) v = reinterpret_cast<const uint4*>(q + ((int64_t)m * H + h0 + hh) * kFdD)[c];
        *reinterpret_cast<uint4*>(Qs + hh * kFdKS + c * 8) = v;
    }
    if (tid < HB) { m_run[tid] = -1e30f; l_run[tid] = 0.f; }
    for (int i = tid; i < C; i += blockDim.x) idx_s[i] = cidx[(int64_t)m * C + i];
    // software pipeline: the next tile's keys are in flight (in registers) while the tensor
    // cores work on this one - loaded after a barrier and consumed before the next, a tile's
    // two dependent L2 round trips (pick, then key) stalled the block, one an SM at this
    // shared memory. A thread's pieces: 16-byte chunk tid % 64 of keys tid / 64 + 4 u.
    constexpr int KP = NT * (kFdD / 8) / 256;
    uint4 kreg[KP];
    bool kval[KP];
    auto fetch = [&](int n0) {
#pragma unroll
        for (int u = 0; u < KP; ++u) {
            const int n = n0 + tid / 64 + 4 * u, c = tid % 64;
            const __nv_bfloat16* kr = nullptr;
            if (n < W) {
                const int kp = p - (W - 1 - n);
                if (kp >= 0 && kp >= lo) kr = ring + (int64_t)(kp % ring_size) * kFdD;
            } else if (n < NK) {
                const int jj = idx_s[n - W];
                if (jj >= 0) {
                    if (ckp) {
                        kval[u] = true;
                        kreg[u] = fp4_unpack8(ckp + (int64_t)jj * kPkRow, c);
                        continue;
                    }
                    kr = ckv + (int64_t)jj * kFdD;
                }
            }
            kval[u] = kr != nullptr;
            kreg[u] = kr ? reinterpret_cast<const uint4*>(kr)[c] : make_uint4(0, 0, 0, 0);
        }
    };
    __syncthreads();  // idx_s
    fetch(0);
    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    for (int tile = 0; tile < n_tiles; ++tile) {
        __syncthreads();  // the previous tile's PV is done with Ks / Ps
#pragma unroll
        for (int u = 0; u < KP; ++u) {
            const int j = tid / 64 + 4 * u, c = tid % 64;
            *reinterpret_cast<uint4*>(Ks + j * kFdKS + c * 8) = kreg[u];
            if (c == 0) valid[j] = kval[u];
        }
        __syncthreads();
        if (tile + 1 < n_tiles) fetch((tile + 1) * NT);
        // scores: (m tile, 8 keys) jobs over the warps - at 64-key tiles warp w takes keys
        // w*8 .. w*8+8 for every head of the block
#pragma unroll
        for (int job = warp; job < MT * (NT / 8); job += 8) {
            const int mt = job / (NT / 8), kb = job % (NT / 8);
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            const __nv_bfloat16* kn = Ks + (kb * 8 + r) * kFdKS;
            const __nv_bfloat16* qa = Qs + (mt * 16) * kFdKS;
#pragma unroll 8
            for (int ks = 0; ks < kFdD / 16; ++ks) {
                const int d0 = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(qa + r * kFdKS + d0 + 2 * qd);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(qa + (r + 8) * kFdKS + d0 + 2 * qd);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(qa + r * kFdKS + d0 + 8 + 2 * qd);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(qa + (r + 8) * kFdKS + d0 + 8 + 2 * qd);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kn + d0 + 2 * qd);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kn + d0 + 8 + 2 * qd);
                mma_bf16(c4, a0, a1, a2, a3, b0, b1);
            }
            const int k0 = kb * 8 + 2 * qd;
            float* pr = Ps + (mt * 16) * PS;
            pr[r * PS + k0] = valid[k0] ? c4[0] * scale : -INFINITY;
            pr[r * PS + k0 + 1] = valid[k0 + 1] ? c4[1] * scale : -INFINITY;
            pr[(r + 8) * PS + k0] = valid[k0] ? c4[2] * scale : -INFINITY;
            pr[(r + 8) * PS + k0 + 1] = valid[k0 + 1] ? c4[3] * scale : -INFINITY;
        }
        __syncthreads();
        // online softmax: warp w takes heads w*2*MT .. (a lane keys lane, and lane + 32 at 64)
#pragma unroll
        for (int u = 0; u < 2 * MT; ++u) {
            const int hh = warp * 2 * MT + u;
            const float mo = m_run[hh];
            float mn, ls;
            if (NT == 64) {
                const float s0 = Ps[hh * PS + lane], s1 = Ps[hh * PS + lane + 32];
                mn = fmaxf(mo, warp_max(fmaxf(s0, s1)));
                const float e0 = expf(s0 - mn), e1 = expf(s1 - mn);
                ls = warp_sum(e0 + e1);
                Ps[hh * PS + lane] = e0;
                Ps[hh * PS + lane + 32] = e1;
            } else {
                const float s0 = Ps[hh * PS + lane];
                mn = fmaxf(mo, warp_max(s0));
                const float e0 = expf(s0 - mn);
                ls = warp_sum(e0);
                Ps[hh * PS + lane] = e0;
            }
            __syncwarp();
            if (lane == 0) {
                const float al = expf(mo - mn);
                alpha[hh] = al;
                l_run[hh] = l_run[hh] * al + ls;
                m_run[hh] = mn;
            }
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float al0 = alpha[mt * 16 + r], al1 = alpha[mt * 16 + r + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mt][nt][0] *= al0; acc[mt][nt][1] *= al0;
                acc[mt][nt][2] *= al1; acc[mt][nt][3] *= al1;
            }
        }
        // PV: warp w takes dims w*64 .. w*64+64; P split into bf16 hi + lo (PV_SPLIT)
#pragma unroll
        for (int ks = 0; ks < NT / 16; ++ks) {
            const int k0 = ks * 16;
            uint32_t ah[MT][4], alo[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float* pr0 = Ps + (mt * 16 + r) * PS + k0 + 2 * qd;
                const float* pr1 = Ps + (mt * 16 + r + 8) * PS + k0 + 2 * qd;
                const float pv[4][2] = {{pr0[0], pr0[1]}, {pr1[0], pr1[1]}, {pr0[8], pr0[9]}, {pr1[8], pr1[9]}};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat162 hi = __floats2bfloat162_rn(pv[i][0], pv[i][1]);
                    const __nv_bfloat162 lo = __floats2bfloat162_rn(pv[i][0] - __low2float(hi), pv[i][1] - __high2float(hi));
                    ah[mt][i] = *reinterpret_cast<const uint32_t*>(&hi);
                    alo[mt][i] = *reinterpret_cast<const uint32_t*>(&lo);
                }
            }
            const int key = k0 + (lane & 15);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                uint32_t b0, b1;
                ldsm_x2_trans(b0, b1, Ks + key * kFdKS + warp * 64 + nt * 8);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma_bf16(acc[mt][nt], ah[mt][0], ah[mt][1], ah[mt][2], ah[mt][3], b0, b1);
                    mma_bf16(acc[mt][nt], alo[mt][0], alo[mt][1], alo[mt][2], alo[mt][3], b0, b1);
                }
            }
        }
    }
    __syncthreads();
    // the combine's formula for one partial: m* = max(m, sink), o = e^(m - m*) acc /
    // (e^(m - m*) l + e^(sink - m*))
    if (tid < nh) {
        const float sk = sink[h0 + tid], mx = fmaxf(m_run[tid], sk), e = expf(m_run[tid] - mx);
        alpha[tid] = e / (e * l_run[tid] + expf(sk - mx));
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int d = warp * 64 + nt * 8 + 2 * qd;
            const int ha = mt * 16 + r, hb = ha + 8;
            if (ha < nh) {
                __nv_bfloat16* o = out + ((int64_t)m * H + h0 + ha) * kFdD + d;
                o[0] = to_bf(acc[mt][nt][0] * alpha[ha]);
                o[1] = to_bf(acc[mt][nt][1] * alpha[ha]);
            }
            if (hb < nh) {
                __nv_bfloat16* o = out + ((int64_t)m * H + h0 + hb) * kFdD + d;
                o[0] = to_bf(acc[mt][nt][2] * alpha[hb]);
                o[1] = to_bf(acc[mt][nt][3] * alpha[hb]);
            }
        }
}

// The partials' merge with the sink: m* = max(m_s, sink), out = sum_s e^(m_s - m*) o_s /
// (sum_s e^(m_s - m*) l_s + e^(sink - m*)), rounded to bf16. One block per (head, row).
__global__ void attn_fd_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int S, int H,
                                       const float* __restrict__ sink, __nv_bfloat16* __restrict__ out,
                                       const int* __restrict__ decode_rows) {
    const int h = blockIdx.x, m = blockIdx.y;
    if (decode_rows && !decode_rows[m]) return;
    float mx = sink[h];
    for (int s = 0; s < S; ++s) mx = fmaxf(mx, part_ml[(((int64_t)m * S + s) * H + h) * 2]);
    float den = expf(sink[h] - mx);
    for (int s = 0; s < S; ++s) {
        const float* ml = part_ml + (((int64_t)m * S + s) * H + h) * 2;
        den += expf(ml[0] - mx) * ml[1];
    }
    for (int d = threadIdx.x; d < kFdD; d += blockDim.x) {
        float o = 0.f;
        for (int s = 0; s < S; ++s) {
            const int64_t at = ((int64_t)m * S + s) * H + h;
            o += expf(part_ml[at * 2] - mx) * part_o[at * kFdD + d];
        }
        out[((int64_t)m * H + h) * kFdD + d] = to_bf(o / den);
    }
}

// ------------------------------------------------------------------ attention (decode)
// model.py::attention + _softmax_attn for decode rows: keys are the layer's window ring
// (positions pos-W+1 .. pos, oldest first, masked below `win_lo`) followed by `C` selected
// compressed rows (cidx, -1 = none). K and V are the same vectors. Scores and PV in fp32:
//   s = (q . k) * scale; mx = max(max s, -1e30); p = exp(s - mx);
//   o = sum_n (p_n / (sum p + exp(sink - mx))) * k_n
// One block per (row, group of HG heads).
constexpr int kAttnHG = 4;

// Batched rows: row m reads the caches of lane[m] (lane == nullptr: lane 0), each lane's
// ring / compressed rows `*_lstride` elements after the previous lane's. HG heads a block:
// every head's arithmetic is its own (scores, softmax, PV in key order), so the grouping is
// a launch choice, not a numerics one - few rows take HG = 1 to fill the GPU (one row of a
// tensor-parallel rank's 32 heads made 8 blocks at HG = 4).
template <int HG>
__global__ void __launch_bounds__(256) attn_decode_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ring_base, int ring_size,
    const int* __restrict__ pos, int W, const int* __restrict__ win_lo, const __nv_bfloat16* __restrict__ ckv_base,
    const int* __restrict__ cidx, int C, const float* __restrict__ sink, float scale,
    __nv_bfloat16* __restrict__ out, int H, const int* __restrict__ row_lane, int64_t ring_lstride,
    int64_t ckv_lstride) {
    constexpr int D = 512;
    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                    // [HG][D]
    float* sc = smem + HG * D;      // [HG][W + C]
    __shared__ float scratch[32];
    const int m = blockIdx.y;
    const int h0 = blockIdx.x * HG;
    const int NK = W + C;
    const int p = pos[m];
    const int ln = row_lane ? row_lane[m] : 0;
    const int lo = win_lo ? win_lo[ln] : 0;
    const __nv_bfloat16* ring = ring_base + (int64_t)ln * ring_lstride;
    const __nv_bfloat16* ckv = ckv_base ? ckv_base + (int64_t)ln * ckv_lstride : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    for (int i = threadIdx.x; i < HG * D; i += blockDim.x)
        qs[i] = bf(q[((int64_t)m * H + h0 + i / D) * D + i % D]);
    __syncthreads();

    auto key_row = [&](int n) -> const __nv_bfloat16* {
        if (n < W) {
            const int kp = p - (W - 1 - n);
            if (kp < 0 || kp < lo) return nullptr;
            return ring + (int64_t)(kp % ring_size) * D;
        }
        const int j = cidx[(int64_t)m * C + (n - W)];
        return j < 0 ? nullptr : ckv + (int64_t)j * D;
    };

    // two keys a warp iteration, so their loads and reductions overlap (each key's dot is
    // the same lane split and the same warp_sum tree as one at a time)
    for (int n = warp; n < NK; n += 2 * nw) {
        const __nv_bfloat16* kr[2] = {key_row(n), n + nw < NK ? key_row(n + nw) : nullptr};
        float kv[2][16];
#pragma unroll
        for (int u = 0; u < 2; ++u)
            if (kr[u]) {
                bf16x8_to_f32(*reinterpret_cast<const uint4*>(kr[u] + lane * 16), kv[u]);
                bf16x8_to_f32(*reinterpret_cast<const uint4*>(kr[u] + lane * 16 + 8), kv[u] + 8);
            }
        float dots[2][HG];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int g = 0; g < HG; ++g) {
                float acc = 0.f;
#pragma unroll
                for (int v = 0; v < 16; ++v) acc = fmaf(qs[g * D + lane * 16 + v], kv[u][v], acc);
                dots[u][g] = kr[u] ? warp_sum(acc) : 0.f;
            }
        if (lane == 0)
            for (int u = 0; u < 2; ++u)
                if (n + u * nw < NK)
                    for (int g = 0; g < HG; ++g) sc[g * NK + n + u * nw] = kr[u] ? dots[u][g] * scale : -INFINITY;
    }
    __syncthreads();

    for (int g = 0; g < HG; ++g) {
        float mx = -INFINITY;
        for (int n = threadIdx.x; n < NK; n += blockDim.x) mx = fmaxf(mx, sc[g * NK + n]);
        mx = fmaxf(block_max(mx, scratch), -1e30f);
        float sum = 0.f;
        for (int n = threadIdx.x; n < NK; n += blockDim.x) {
            const float e = expf(sc[g * NK + n] - mx);
            sc[g * NK + n] = e;
            sum += e;
        }
        sum = block_sum(sum, scratch);
        const float denom = sum + expf(sink[h0 + g] - mx);
        __syncthreads();
        for (int n = threadIdx.x; n < NK; n += blockDim.x) sc[g * NK + n] = sc[g * NK + n] / denom;
        __syncthreads();
    }

    // PV: thread owns dims 2t, 2t+1
    float acc[HG][2];
    for (int g = 0; g < HG; ++g) acc[g][0] = acc[g][1] = 0.f;
    const int d = threadIdx.x * 2;
    // 8 keys' loads issued ahead of their FMAs, which stay in key order (the same bits)
    for (int n0 = 0; n0 < NK; n0 += 8) {
        __nv_bfloat162 kv2[8];
        bool ok[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const __nv_bfloat16* kr = n0 + u < NK ? key_row(n0 + u) : nullptr;
            ok[u] = kr != nullptr && d < D;
            if (ok[u]) kv2[u] = *reinterpret_cast<const __nv_bfloat162*>(kr + d);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (!ok[u]) continue;
            const float k0 = __low2float(kv2[u]), k1 = __high2float(kv2[u]);
            for (int g = 0; g < HG; ++g) {
                const float pn = sc[g * NK + n0 + u];
                acc[g][0] = fmaf(pn, k0, acc[g][0]);
                acc[g][1] = fmaf(pn, k1, acc[g][1]);
            }
        }
    }
    if (d < D)
        for (int g = 0; g < HG; ++g) {
            __nv_bfloat16* o = out + ((int64_t)m * H + h0 + g) * D + d;
            o[0] = to_bf(acc[g][0]);
            o[1] = to_bf(acc[g][1]);
        }
}

// ------------------------------------------------------------------ compressor
// model.py::_compressed, ratio 2: the softmax-gated pair sum of two positions' (kv, score),
// rounded to bf16 (before the rmsnorm the caller runs next).
__global__ void compress_pair_kernel(const float* __restrict__ kv_a, const float* __restrict__ sc_a,
                                     const float* __restrict__ kv_b, const float* __restrict__ sc_b,
                                     __nv_bfloat16* __restrict__ out, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float a = sc_a[d], b = sc_b[d];
    const float mx = fmaxf(a, b);
    const float ea = expf(a - mx), eb = expf(b - mx);
    const float den = ea + eb;
    out[d] = to_bf(kv_a[d] * (ea / den) + kv_b[d] * (eb / den));
}

// v41_ref.fp4_qdq(x, 16, "e4m3"): FP4 values with an E4M3 group scale, dequantized back,
// in place on bf16 rows. One thread per 16-value group.
__global__ void fp4_qdq_e4m3_kernel(__nv_bfloat16* x, int64_t n_groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    __nv_bfloat16* p = x + g * 16;
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = bf(p[i]);
    uint8_t s8;
    uint64_t codes;
    fp4_quant16(v, s8, codes);
    const float s = e4m3_float(s8);
    for (int i = 0; i < 16; ++i) {
        const uint16_t b = fp4_value((uint32_t)(codes >> (4 * i)) & 15, s);
        p[i] = *reinterpret_cast<const __nv_bfloat16*>(&b);
    }
}

// ds_scatter_compressed into packed rows: the compressed rows of a step (rows whose
// (pos + 1) % r == 0), quantized as fp4_qdq_e4m3 would and stored packed at their lane's
// row (pos + 1) / r - 1. One block a row, one thread a 16-value group. `lstride` in bytes.
__global__ void scatter_compressed_fp4_kernel(const __nv_bfloat16* __restrict__ src, int64_t ld_src, uint8_t* __restrict__ dst,
                                              const int* __restrict__ pos, int r, int D, const int* __restrict__ lane,
                                              int64_t lstride) {
    const int t = blockIdx.x, g = threadIdx.x;
    if (g >= D / 16 || (pos[t] + 1) % r != 0) return;
    float v[16];
    const __nv_bfloat16* x = src + (int64_t)t * ld_src + 16 * g;
    for (int i = 0; i < 16; ++i) v[i] = bf(x[i]);
    uint8_t s8;
    uint64_t codes;
    fp4_quant16(v, s8, codes);
    uint8_t* row = dst + (int64_t)(lane ? lane[t] : 0) * lstride + (int64_t)((pos[t] + 1) / r - 1) * (D / 2 + D / 16);
    *reinterpret_cast<uint2*>(row + 8 * g) = make_uint2((uint32_t)codes, (uint32_t)(codes >> 32));
    row[D / 2 + g] = s8;
}

// Packed rows back to bf16 rows (tests, dumps): rows x D values.
__global__ void unpack_fp4_rows_kernel(const uint8_t* __restrict__ src, __nv_bfloat16* __restrict__ dst, int rows) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * 64) return;
    const int64_t r = i / 64;
    const int c = (int)(i % 64);
    *reinterpret_cast<uint4*>(dst + r * 512 + 8 * c) = fp4_unpack8(src + r * kPkRow, c);
}

// ------------------------------------------------------------------ engram
// engine/engram.py::rows -> bf16: raw rows are 256 e4m3 values + 8 UE8M0 scales (one per 32);
// value = e4m3 * exp2(scale - 127), exact, then bf16 (exact too: e4m3 fits in bf16).
// inv maps (token, col) to a unique row of `raw`.
__global__ void engram_dequant_kernel(const uint8_t* __restrict__ raw, const int* __restrict__ inv,
                                      int cols, __nv_bfloat16* __restrict__ out) {
    const int tc = blockIdx.x;  // token * cols + col
    const uint8_t* row = raw + (int64_t)inv[tc] * 264;
    const int v = threadIdx.x;  // 256 threads
    __nv_fp8_storage_t b = row[v];
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    const float f = __half2float(*reinterpret_cast<const __half*>(&hr)) * e8m0_exp2(row[256 + v / 32]);
    out[(int64_t)tc * 256 + v] = to_bf(f);
    (void)cols;
}

// v41_ref.engram_combine: per (token, hc copy i): gate = sigmoid(signed_sqrt(
//   sum(h * weight * key) * rstd(h) * rstd(key) * dim^-0.5)), out = bf16(h + gate * value).
// kv rows are [hc * D keys | D values]. One block per (token, i); in place on h.
__global__ void engram_gate_kernel(__nv_bfloat16* h, const __nv_bfloat16* __restrict__ kv,
                                   const float* __restrict__ weight, int D, int hc, float eps, float dim_scale,
                                   const int* __restrict__ skip) {
    __shared__ float scratch[32];
    const int m = blockIdx.x / hc, i = blockIdx.x % hc;
    // an image row: engram_combine's gate is zero where engram_mask is False
    if (skip && skip[m]) return;
    __nv_bfloat16* hr = h + ((int64_t)m * hc + i) * D;
    const __nv_bfloat16* kr = kv + (int64_t)m * (hc + 1) * D + (int64_t)i * D;
    const __nv_bfloat16* vr = kv + (int64_t)m * (hc + 1) * D + (int64_t)hc * D;
    const float* wr = weight + (int64_t)i * D;
    float sh = 0.f, sk = 0.f, dot = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float a = bf(hr[d]), b = bf(kr[d]);
        sh = fmaf(a, a, sh);
        sk = fmaf(b, b, sk);
        dot = fmaf(a * wr[d], b, dot);
    }
    sh = block_sum(sh, scratch);
    sk = block_sum(sk, scratch);
    dot = block_sum(dot, scratch);
    const float inv_d = 1.f / (float)D;
    const float rstd = rsqrtf(sh * inv_d + eps) * rsqrtf(sk * inv_d + eps);
    const float x = dot * rstd * dim_scale;
    const float g = sigmoidf_(copysignf(sqrtf(fmaxf(fabsf(x), 1e-6f)), x));
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) hr[d] = to_bf(bf(hr[d]) + g * bf(vr[d]));
}

// ------------------------------------------------------------------ embedding / sampling
__global__ void embed_hc_kernel(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                __nv_bfloat16* __restrict__ h, int D, int hc) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const __nv_bfloat16 v = table[(int64_t)ids[m] * D + d];
    for (int c = 0; c < hc; ++c) h[((int64_t)m * hc + c) * D + d] = v;
}

// Row kernels: one 1024-thread block a vocabulary row.
constexpr int kRowThreads = 1024;

// f(x[n], n) over the row, 16-byte loads when `vec`
template <class F>
__device__ __forceinline__ void nuc_scan(const __nv_bfloat16* ar, const __nv_bfloat16* br, int N, bool vec, F&& f) {
    if (vec) {
        const int nc = N >> 3;
        for (int c = threadIdx.x; c < nc; c += kRowThreads) {
            const uint4 ra = reinterpret_cast<const uint4*>(ar)[c];
            const uint4 rb = br ? reinterpret_cast<const uint4*>(br)[c] : make_uint4(0, 0, 0, 0);
            const __nv_bfloat16* xa = reinterpret_cast<const __nv_bfloat16*>(&ra);
            const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&rb);
#pragma unroll
            for (int k = 0; k < 8; ++k) f(bf(xa[k]) + (br ? bf(xb[k]) : 0.f), 8 * c + k);
        }
    } else {
        for (int n = threadIdx.x; n < N; n += kRowThreads) f(bf(ar[n]) + (br ? bf(br[n]) : 0.f), n);
    }
}

// First index of the maximum (torch.argmax), over bf16 logits.
// `pair`: also the maximum, the index written as offset + n, as (index, value bits) pairs -
// a vocab-sharded head's per-rank candidate for argmax_combine_kernel.
// 16-byte loads when `vec` (26 -> 7 us a 129,280-wide row, bench_nucleus).
__global__ void __launch_bounds__(kRowThreads) argmax_bf16_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int N,
                                                                   bool vec, int* __restrict__ out, int pair, int offset) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * ldx;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    nuc_scan(xr, nullptr, N, vec, [&](float v, int n) {
        if (v > best || (v == best && n < besti)) { best = v; besti = n; }
    });
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { bv[warp] = best; bi[warp] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
        best = bv[0];
        besti = bi[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
            if (bv[i] > best || (bv[i] == best && bi[i] < besti)) { best = bv[i]; besti = bi[i]; }
        if (pair) {
            out[2 * blockIdx.x] = besti + offset;
            out[2 * blockIdx.x + 1] = __float_as_int(best);
        } else {
            out[blockIdx.x] = besti;
        }
    }
}

// The vocab-sharded head's argmax: per row the larger of the ranks' (index, value) pairs
// [world][M][2], the earlier rank on a tie - its indices are the lower ones, so this is the
// one-box first-index argmax.
__global__ void argmax_combine_kernel(const int* __restrict__ pairs, int world, int M, int* __restrict__ out) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int best = pairs[2 * m];
    float bv = __int_as_float(pairs[2 * m + 1]);
    for (int r = 1; r < world; ++r) {
        const float v = __int_as_float(pairs[2 * ((int64_t)r * M + m) + 1]);
        if (v > bv) { bv = v; best = pairs[2 * ((int64_t)r * M + m)]; }
    }
    out[m] = best;
}

// A tensor-parallel gather's rank-major result [world][M][n] as full rows [M][world * n].
__global__ void unshard_u16_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int M, int n, int world) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)world * M * n;
    if (i >= total) return;
    const int64_t r = i / ((int64_t)M * n), rem = i % ((int64_t)M * n);
    const int64_t m = rem / n, j = rem % n;
    dst[m * (int64_t)world * n + r * n + j] = src[i];
}

// ------------------------------------------------------------------ indexer
// model.py::_indexer for decode rows. The score buffer is a RANKING quantity there (masked,
// block-max'd, top-k'd, never fed to arithmetic), stored bf16, with each per-head dot product
// rounded to bf16 by the bf16 einsum before the relu:
//   s[t, j] = bf16( sum_h relu(bf16(q[t, h] . k[j])) * w[t, h] ),   -inf where j >= clen[t]
//   (or where the candidate mask `cand` [T, ld_cand] is 0).
constexpr int kIdxH = 32, kIdxD = 128, kIdxTile = 64;

__global__ void __launch_bounds__(256) index_scores_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_base, const float* __restrict__ w,
    const int* __restrict__ clen, const uint8_t* __restrict__ cand, int64_t ld_cand, int n,
    __nv_bfloat16* __restrict__ out, int64_t ld_out, const int* __restrict__ lane, int64_t k_lstride) {
    __shared__ float qs[kIdxH][kIdxD + 1];
    __shared__ __align__(16) __nv_bfloat16 ks[kIdxTile][kIdxD];
    const int t = blockIdx.y;
    const int j0 = blockIdx.x * kIdxTile;
    const int h = threadIdx.x & 31, jj = threadIdx.x >> 5;
    const __nv_bfloat16* k = k_base + (int64_t)(lane ? lane[t] : 0) * k_lstride;
    for (int i = threadIdx.x; i < kIdxH * kIdxD; i += blockDim.x)
        qs[i / kIdxD][i % kIdxD] = bf(q[(int64_t)t * kIdxH * kIdxD + i]);
    for (int i = threadIdx.x; i < kIdxTile * kIdxD / 8; i += blockDim.x) {
        const int r = i / (kIdxD / 8), c = (i % (kIdxD / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j0 + r < n) v = *reinterpret_cast<const uint4*>(k + (int64_t)(j0 + r) * kIdxD + c);
        *reinterpret_cast<uint4*>(&ks[r][c]) = v;
    }
    __syncthreads();
    const float wh = w[t * kIdxH + h];
    const int lim = clen[t];
    for (int i = 0; i < kIdxTile / 8; ++i) {
        const int jl = jj + 8 * i, j = j0 + jl;
        float dot = 0.f;
#pragma unroll 16
        for (int d = 0; d < kIdxD; ++d) dot = fmaf(qs[h][d], bf(ks[jl][d]), dot);
        float v = fmaxf(bf(to_bf(dot)), 0.f) * wh;
        v = warp_sum(v);
        if (h == 0 && j < n) {
            const bool live = j < lim && (!cand || cand[(int64_t)t * ld_cand + j]);
            out[(int64_t)t * ld_out + j] = live ? to_bf(v) : to_bf(-INFINITY);
        }
    }
}

// The same scores for prefill-sized steps on the tensor cores: a warp per query row holds its
// q (32 heads x 128) as mma A fragments, and its block's 8 rows share each 64-key tile in
// shared memory. The mma's C layout gives a thread heads g, g + 8, g + 16, g + 24, so the
// head sum reproduces warp_sum's tree exactly (xor 16 and 8 in registers, then 4, 2, 1 as
// shuffles over lanes 16, 8, 4): the one difference from index_scores_kernel is each
// 128-term dot's fp32 accumulation order inside the mma (the bf16 einsum's, not a sequential
// chain), which moves a dot across a bf16 rounding boundary now and then. Keys no row of the
// block sees (j >= clen) are written -inf without being computed. A block takes `chunk` keys
// (blockIdx.y's run of them): a prefill step's 8-row blocks fill the GPU with chunk = n, a
// decode step's one or two need the keys split; every score is still one block's, so the
// bits do not depend on the split.
constexpr int kIsRows = 8, kIsLD = kIdxD + 8;

__global__ void __launch_bounds__(256) index_scores_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_base, const float* __restrict__ w,
    const int* __restrict__ clen, const uint8_t* __restrict__ cand, int64_t ld_cand, int n,
    __nv_bfloat16* __restrict__ out, int64_t ld_out, const int* __restrict__ lane, int64_t k_lstride, int T, int chunk) {
    __shared__ __align__(16) __nv_bfloat16 ks[kIdxTile * kIsLD];
    __shared__ int lane_s[kIsRows], lim_s[kIsRows];
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, g = l >> 2, c = l & 3;
    const int t0 = blockIdx.x * kIsRows, t = t0 + warp;
    const bool row = t < T;
    if (threadIdx.x < kIsRows) {
        const int tt = t0 + threadIdx.x;
        lane_s[threadIdx.x] = tt < T ? (lane ? lane[tt] : 0) : -1;
        lim_s[threadIdx.x] = tt < T ? min(clen[tt], n) : 0;
    }
    __syncthreads();
    int max_lim = 0;
#pragma unroll
    for (int r = 0; r < kIsRows; ++r) max_lim = max(max_lim, lim_s[r]);
    const int my_lane = row ? lane_s[warp] : -1, lim = row ? lim_s[warp] : 0;
    uint32_t af[2][kIdxD / 16][4];
    float wv[4] = {0.f, 0.f, 0.f, 0.f};
    if (row) {
        const uint32_t* qr = reinterpret_cast<const uint32_t*>(q + (int64_t)t * kIdxH * kIdxD);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int kk = 0; kk < kIdxD / 16; ++kk) {
                const int r0 = (mt * 16 + g) * kIdxD + kk * 16 + 2 * c, r1 = r0 + 8 * kIdxD;
                af[mt][kk][0] = qr[r0 / 2];
                af[mt][kk][1] = qr[r1 / 2];
                af[mt][kk][2] = qr[(r0 + 8) / 2];
                af[mt][kk][3] = qr[(r1 + 8) / 2];
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = w[(int64_t)t * kIdxH + g + 8 * i];
    }
    __nv_bfloat16* orow = out + (int64_t)t * ld_out;
    const __nv_bfloat16 ninf = to_bf(-INFINITY);
    const int jb = blockIdx.y * chunk, je = min(n, jb + chunk);
    for (int j0 = jb; j0 < je; j0 += kIdxTile) {
        const int j1 = min(je, j0 + kIdxTile);
        if (j0 >= max_lim) {
            if (row)
                for (int j = j0 + l; j < j1; j += 32) orow[j] = ninf;
            continue;
        }
        // one pass a distinct lane among the block's rows (a prefill step's rows share one)
        for (int r = 0; r < kIsRows; ++r) {
            const int ln = lane_s[r];
            bool first = ln >= 0;
            for (int r2 = 0; r2 < r; ++r2) first = first && lane_s[r2] != ln;
            if (!first) continue;
            const __nv_bfloat16* kb = k_base + (int64_t)ln * k_lstride;
            __syncthreads();
            for (int i = threadIdx.x; i < kIdxTile * kIdxD / 8; i += blockDim.x) {
                const int rr = i / (kIdxD / 8), cc = (i % (kIdxD / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (j0 + rr < n) v = *reinterpret_cast<const uint4*>(kb + (int64_t)(j0 + rr) * kIdxD + cc);
                *reinterpret_cast<uint4*>(&ks[rr * kIsLD + cc]) = v;
            }
            __syncthreads();
            if (my_lane != ln) continue;
            if (j0 >= lim) {
                for (int j = j0 + l; j < j1; j += 32) orow[j] = ninf;
                continue;
            }
#pragma unroll 2
            for (int nt = 0; nt < kIdxTile / 8; ++nt) {
                float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
                for (int k2 = 0; k2 < kIdxD / 32; ++k2) {
                    uint32_t b[4];
                    ldsm_x4(b, &ks[(nt * 8 + (l & 7)) * kIsLD + k2 * 32 + (l >> 3) * 8]);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const uint32_t* a0 = af[mt][2 * k2];
                        const uint32_t* a1 = af[mt][2 * k2 + 1];
                        mma_bf16(acc[mt], a0[0], a0[1], a0[2], a0[3], b[0], b[1]);
                        mma_bf16(acc[mt], a1[0], a1[1], a1[2], a1[3], b[2], b[3]);
                    }
                }
                // acc[mt][e]: head mt * 16 + g, key 2c + e; acc[mt][2 + e]: head mt * 16 + g + 8
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float h0 = fmaxf(bf(to_bf(acc[0][e])), 0.f) * wv[0];
                    const float h8 = fmaxf(bf(to_bf(acc[0][2 + e])), 0.f) * wv[1];
                    const float h16 = fmaxf(bf(to_bf(acc[1][e])), 0.f) * wv[2];
                    const float h24 = fmaxf(bf(to_bf(acc[1][2 + e])), 0.f) * wv[3];
                    float sum = (h0 + h16) + (h8 + h24);
                    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
                    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
                    sum += __shfl_xor_sync(0xffffffffu, sum, 4);
                    const int j = j0 + nt * 8 + 2 * c + e;
                    if (g == 0 && j < n) {
                        const bool live = j < lim && (!cand || cand[(int64_t)t * ld_cand + j]);
                        orow[j] = live ? to_bf(sum) : ninf;
                    }
                }
            }
        }
    }
}

// Orderable key of a bf16 value: larger value -> larger key. -inf and NaN are not selectable.
__device__ __forceinline__ bool bf16_key(uint16_t b, uint32_t& key) {
    if (b == 0xff80u || (b & 0x7f80u) == 0x7f80u && (b & 0x7fu)) return false;  // -inf, NaN
    key = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
    return true;
}

// Per row: the indices of the K largest selectable bf16 values, ascending, then -1 up to
// `width` columns (model.py: topk(k).indices.sort(), positions past clen -> -1, _pad_topk).
// Ties at the threshold keep the lowest indices (torch.topk leaves tie order unspecified).
// One block of 1024 threads per row: an 8+8-bit radix select for the threshold, then a stable
// compaction that emits the selected indices already in ascending order. The count and the
// histograms read the row coalesced (thread t takes t, t + 1024, ...) into a histogram a warp
// - most bf16 scores share their exponent byte, and one shared histogram serialized on it -
// and only the ordered compaction walks a thread's contiguous run.
constexpr int kSelThreads = 1024;

__global__ void __launch_bounds__(kSelThreads) topk_select_kernel(
    const __nv_bfloat16* __restrict__ s, int64_t ld_s, int n, int K, int* __restrict__ out, int width) {
    __shared__ int hist[256];
    __shared__ int whist[kSelThreads / 32][256];
    __shared__ int scan[kSelThreads];
    __shared__ uint32_t sh_thresh;
    __shared__ int sh_gt, sh_need, sh_all;
    const uint16_t* row = reinterpret_cast<const uint16_t*>(s + (int64_t)blockIdx.x * ld_s);
    int* orow = out + (int64_t)blockIdx.x * width;
    const int tid = threadIdx.x;
    const int chunk = (n + kSelThreads - 1) / kSelThreads;
    const int lo = min(n, tid * chunk), hi = min(n, lo + chunk);

    // pass 0: selectable count; pass 1/2: the K-th largest key, one byte at a time
    int finite = 0;
    for (int i = tid; i < n; i += kSelThreads) { uint32_t key; finite += bf16_key(row[i], key); }
    scan[tid] = finite;
    __syncthreads();
    for (int o = kSelThreads / 2; o > 0; o >>= 1) {
        if (tid < o) scan[tid] += scan[tid + o];
        __syncthreads();
    }
    if (tid == 0) { sh_all = scan[0] <= K; sh_thresh = 0; sh_gt = 0; sh_need = 0; }
    __syncthreads();
    const bool take_all = sh_all;
    if (!take_all) {
        uint32_t prefix = 0;
        int remaining = K, above = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const int shift = pass == 0 ? 8 : 0;
            for (int b = tid; b < (kSelThreads / 32) * 256; b += kSelThreads) (&whist[0][0])[b] = 0;
            __syncthreads();
            int* mine = whist[tid >> 5];
            for (int i = tid; i < n; i += kSelThreads) {
                uint32_t key;
                if (!bf16_key(row[i], key)) continue;
                if (pass == 1 && (key >> 8) != (prefix >> 8)) continue;
                atomicAdd(&mine[(key >> shift) & 0xff], 1);
            }
            __syncthreads();
            for (int b = tid; b < 256; b += kSelThreads) {
                int sum = 0;
                for (int wv = 0; wv < kSelThreads / 32; ++wv) sum += whist[wv][b];
                hist[b] = sum;
            }
            __syncthreads();
            if (tid == 0) {
                int acc = 0, b = 255;
                for (; b > 0; --b) {
                    if (acc + hist[b] >= remaining) break;
                    acc += hist[b];
                }
                prefix |= (uint32_t)b << shift;
                above += acc;
                remaining -= acc;
                sh_thresh = prefix;
                sh_gt = above;
                sh_need = remaining;
            }
            __syncthreads();
            prefix = sh_thresh;
            above = sh_gt;
            remaining = sh_need;
        }
    }
    const uint32_t thresh = sh_thresh;
    const int need_eq = sh_need;

    // equal-to-threshold ranks in index order, then the selected count per thread
    int eq = 0;
    if (!take_all)
        for (int i = lo; i < hi; ++i) { uint32_t key; if (bf16_key(row[i], key) && key == thresh) ++eq; }
    __syncthreads();
    scan[tid] = eq;
    __syncthreads();
    for (int o = 1; o < kSelThreads; o <<= 1) {  // inclusive Hillis-Steele scan
        const int v = tid >= o ? scan[tid - o] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int eq_rank = scan[tid] - eq;
    int sel = 0;
    for (int i = lo, r = eq_rank; i < hi; ++i) {
        uint32_t key;
        if (!bf16_key(row[i], key)) continue;
        if (take_all || key > thresh) ++sel;
        else if (key == thresh && r++ < need_eq) ++sel;
    }
    __syncthreads();
    scan[tid] = sel;
    __syncthreads();
    for (int o = 1; o < kSelThreads; o <<= 1) {
        const int v = tid >= o ? scan[tid - o] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    const int total = scan[kSelThreads - 1];
    int at = scan[tid] - sel;
    for (int i = lo, r = eq_rank; i < hi; ++i) {
        uint32_t key;
        if (!bf16_key(row[i], key)) continue;
        bool take = take_all || key > thresh;
        if (!take && key == thresh && r++ < need_eq) take = true;
        if (take && at < width) orow[at++] = i;
    }
    for (int i = total + tid; i < width; i += kSelThreads) orow[i] = -1;
}

// model.py::_select_candidates, the scoring half: per row, the max of each `block` scores,
// with the block holding position clen-1 forced to +inf (it is always kept).
__global__ void candidate_blocks_kernel(const __nv_bfloat16* __restrict__ s, int64_t ld_s, int n, int block,
                                        const int* __restrict__ clen, __nv_bfloat16* __restrict__ bmax,
                                        int64_t ld_b) {
    const int t = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = (n + block - 1) / block;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int i = b * block; i < min(n, (b + 1) * block); ++i) m = fmaxf(m, bf(s[(int64_t)t * ld_s + i]));
    if (clen[t] > 0 && b == (clen[t] - 1) / block) m = INFINITY;
    bmax[(int64_t)t * ld_b + b] = to_bf(m);
}

// The mask half: cand[t, j] = 1 iff j's block was selected (block ids from topk_select,
// -1 padded). The caller zeroes `cand` first; this kernel only sets.
__global__ void candidate_set_kernel(const int* __restrict__ blocks, int kb, int block, int n,
                                     uint8_t* __restrict__ cand, int64_t ld_cand) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // over kb * block
    if (i >= kb * block) return;
    const int b = blocks[(int64_t)t * kb + i / block];
    if (b < 0) return;
    const int j = b * block + i % block;
    if (j < n) cand[(int64_t)t * ld_cand + j] = 1;
}

// bf16 -> f32 times a scale (model.py::_indexer's `mm(x, weights_proj).float() * c`).
__global__ void bf16_to_f32_scale_kernel(const __nv_bfloat16* __restrict__ x, int64_t n, float scale,
                                         float* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf(x[i]) * scale;
}

// model.py: `h.float().mean(dim=1)` over the hc copies (the DSpark main hidden of a target
// layer), rounded to bf16 where the seed hands it to main_proj.
__global__ void hc_mean_kernel(const __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ out,
                               int64_t ld_out, int D, int hc) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = bf(h[((int64_t)m * hc) * D + d]);
    for (int c = 1; c < hc; ++c) acc += bf(h[((int64_t)m * hc + c) * D + d]);
    out[(int64_t)m * ld_out + d] = to_bf(acc * (1.f / (float)hc));
}

// First index of the maximum of a + b over bf16 rows, in fp32 (the DSpark markov step:
// logits[i] + bias, both bf16-valued). `b` may be null.
__global__ void __launch_bounds__(kRowThreads) argmax_sum_bf16_kernel(const __nv_bfloat16* __restrict__ a, int64_t lda,
                                                                       const __nv_bfloat16* __restrict__ b, int64_t ldb,
                                                                       int N, bool vec, int* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* ar = a + (int64_t)blockIdx.x * lda;
    const __nv_bfloat16* br = b ? b + (int64_t)blockIdx.x * ldb : nullptr;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    nuc_scan(ar, br, N, vec, [&](float v, int n) {
        if (v > best || (v == best && n < besti)) { best = v; besti = n; }
    });
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { bv[warp] = best; bi[warp] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
        best = bv[0];
        besti = bi[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
            if (bv[i] > best || (bv[i] == best && bi[i] < besti)) { best = bv[i]; besti = bi[i]; }
        out[blockIdx.x] = besti;
    }
}

// ---- temperature + nucleus: v41_engine.sample_probs, and fastdecode.py's DRAFT_NUCLEUS draft ----
//
// One 1024-thread block per row x = a + b (bf16 rows, b may be null, summed in fp32 as
// argmax_sum_bf16_kernel does). temp[r] <= 0: greedy, the row's first-index argmax with n = 1.
// Otherwise p = softmax(x / T) over the whole row and its nucleus by the `cum - p < top_p` rule
// over (value desc, id asc), at most K tokens (DRAFT_NUCLEUS_K; the target passes kNucCap),
// renormalized: ids[r][0..n), probs[r][0..n), cnt[r] = n. cnt[r] = -n when the nucleus was cut
// short (by K, or by the candidate window below): still a distribution to draw from - the
// drafter does, and verification divides by exactly it - but not the target's full nucleus,
// which the host then recomputes from the row. With u: tok[r] is the inverse-CDF draw at u[r].
//
// Candidates: a histogram of each token's log-probability gap below the max (kNucBinsPerNat
// bins a nat over kNucGap nats - a token further down holds under e^-64 of the mass and is
// never a candidate), in counts and in fixed-point mass. The window is every bin up to the one
// that reaches K tokens or, one bin past it, the one that reaches top_p of the mass, whichever
// comes first: whole bins, so the window is the top of the row by value, ties included. When
// that is more than kNucCap tokens, the K-th token's bin is split once more into kNucSub
// sub-bins (one more pass, only then), and the window ends at the sub-bin reaching K. The
// fixed-point mass rounds down, which can only widen the window; the cut itself is taken on
// the sorted window's float sums and checked there. Only a crowd of more than kNucCap tokens
// within 1/256 nat of each other (exact ties, in practice) still cuts the window short.
constexpr int kNucThreads = kRowThreads, kNucBins = 128, kNucSub = 128, kNucCap = 2048;
constexpr float kNucBinsPerNat = 2.f, kNucGap = kNucBins / kNucBinsPerNat;
// 2^18: a warp's at most 4,096 tokens cannot overflow a 32-bit bin
constexpr float kNucMassScale = 262144.f;

__device__ __forceinline__ uint32_t nuc_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float nuc_val(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__global__ void __launch_bounds__(kNucThreads) nucleus_kernel(
    const __nv_bfloat16* __restrict__ a, int64_t lda, const __nv_bfloat16* __restrict__ b, int64_t ldb, int N, bool vec,
    const float* __restrict__ temp, const float* __restrict__ top_p, const float* __restrict__ u, int K,
    int* __restrict__ ids, float* __restrict__ probs, int64_t ldo, int* __restrict__ cnt, int* __restrict__ tok) {
    __shared__ union {
        struct { unsigned c[32][kNucBins]; unsigned m[32][kNucBins]; } h;
        struct { unsigned long long e[kNucCap]; float incl[kNucCap]; } s;
    } sm;
    __shared__ unsigned tot_c[kNucBins], sub_c[kNucSub];
    __shared__ unsigned long long tot_m[kNucBins];
    __shared__ float rf[32];
    __shared__ int ri[32];
    __shared__ float s_max, s_z;
    __shared__ int s_arg, s_sel, s_sub, s_lo, s_nc, s_n, s_draw;
    const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* ar = a + (int64_t)r * lda;
    const __nv_bfloat16* br = b ? b + (int64_t)r * ldb : nullptr;
    int* ir = ids + (int64_t)r * ldo;
    float* pr = probs + (int64_t)r * ldo;
    // 1. the max and its first index (argmax_sum_bf16_kernel's result)
    float best = -INFINITY;
    int besti = 0x7fffffff;
    nuc_scan(ar, br, N, vec, [&](float v, int n) {
        if (v > best || (v == best && n < besti)) { best = v; besti = n; }
    });
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if (lane == 0) { rf[warp] = best; ri[warp] = besti; }
    __syncthreads();
    if (tid == 0) {
        best = rf[0];
        besti = ri[0];
        for (int i = 1; i < 32; ++i)
            if (rf[i] > best || (rf[i] == best && ri[i] < besti)) { best = rf[i]; besti = ri[i]; }
        s_max = best;
        s_arg = besti;
    }
    __syncthreads();
    const float m = s_max;
    const float T = temp[r];
    if (!(T > 0.f)) {
        if (tid == 0) {
            ir[0] = s_arg;
            pr[0] = 1.f;
            cnt[r] = 1;
            if (tok) tok[r] = s_arg;
        }
        return;
    }
    const float inv_t = 1.f / T;
    // 2. the normalizer and the per-warp gap histograms (no cross-warp atomics)
    for (int i = tid; i < 32 * kNucBins; i += kNucThreads) {
        (&sm.h.c[0][0])[i] = 0u;
        (&sm.h.m[0][0])[i] = 0u;
    }
    __syncthreads();
    float z = 0.f;
    nuc_scan(ar, br, N, vec, [&](float v, int) {
        const float g = (m - v) * inv_t;
        const float e = expf(-g);
        z += e;
        if (g < kNucGap) {
            const int bin = (int)(g * kNucBinsPerNat);
            atomicAdd(&sm.h.c[warp][bin], 1u);
            atomicAdd(&sm.h.m[warp][bin], (unsigned)(e * kNucMassScale));
        }
    });
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0) rf[warp] = z;
    __syncthreads();
    if (tid < kNucBins) {
        unsigned c = 0;
        unsigned long long w = 0;
        for (int i = 0; i < 32; ++i) { c += sm.h.c[i][tid]; w += sm.h.m[i][tid]; }
        tot_c[tid] = c;
        tot_m[tid] = w;
    }
    if (tid == 0) {
        float zs = 0.f;
        for (int i = 0; i < 32; ++i) zs += rf[i];
        s_z = zs;
    }
    __syncthreads();
    const float tp = top_p[r];
    // 3. the window, from warp 0's scan over the bins (kNucBins / 32 a lane)
    if (warp == 0) {
        constexpr int B4 = kNucBins / 32;
        unsigned c[B4];
        unsigned long long w[B4];
        unsigned cs = 0;
        unsigned long long ws = 0;
        for (int j = 0; j < B4; ++j) {
            c[j] = tot_c[lane * B4 + j];
            w[j] = tot_m[lane * B4 + j];
            cs += c[j];
            ws += w[j];
        }
        unsigned ci = cs;
        unsigned long long wi = ws;
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned co = __shfl_up_sync(0xffffffffu, ci, o);
            const unsigned long long wo = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) { ci += co; wi += wo; }
        }
        const double thr = (double)tp * (double)s_z * (double)kNucMassScale;
        // the first bin reaching K tokens (the last bin if none does), the first reaching top_p
        // of the mass (kNucBins if none)
        int bc = kNucBins - 1, bm = kNucBins;
        const unsigned lc = __ballot_sync(0xffffffffu, ci >= (unsigned)K);
        if (lc) {
            const int L = __ffs(lc) - 1;
            int x = 0;
            if (lane == L) {
                unsigned acc = ci - cs;
                for (int j = 0; j < B4; ++j) {
                    acc += c[j];
                    if (acc >= (unsigned)K) { x = lane * B4 + j; break; }
                }
            }
            bc = __shfl_sync(0xffffffffu, x, L);
        }
        const unsigned lm = __ballot_sync(0xffffffffu, (double)wi >= thr);
        if (lm) {
            const int L = __ffs(lm) - 1;
            int x = 0;
            if (lane == L) {
                unsigned long long acc = wi - ws;
                for (int j = 0; j < B4; ++j) {
                    acc += w[j];
                    if ((double)acc >= thr) { x = lane * B4 + j; break; }
                }
            }
            bm = __shfl_sync(0xffffffffu, x, L);
        }
        int sel = min(bc, bm + 1);
        const int L = sel / B4;
        unsigned through = 0;
        if (lane == L) {
            through = ci - cs;
            for (int j = 0; j <= sel % B4; ++j) through += c[j];
        }
        through = __shfl_sync(0xffffffffu, through, L);
        if (lane == 0) {
            // over the cap only when sel is bc: its earlier bins hold fewer than K <= kNucCap
            s_sel = sel;
            s_sub = through > (unsigned)kNucCap ? -1 : kNucSub - 1;
            s_lo = (int)(through - tot_c[sel]);
            s_nc = 0;
            s_n = 0x7fffffff;
            s_draw = 0;
        }
    }
    for (int i = tid; i < kNucSub; i += kNucThreads) sub_c[i] = 0u;
    __syncthreads();
    const int sel = s_sel;
    const auto sub_of = [&](float g) { return min(kNucSub - 1, (int)((g * kNucBinsPerNat - (float)sel) * (float)kNucSub)); };
    if (s_sub < 0) {
        // 3b. the K-th token's bin is crowded: split it
        nuc_scan(ar, br, N, vec, [&](float v, int) {
            const float g = (m - v) * inv_t;
            if (g < kNucGap && (int)(g * kNucBinsPerNat) == sel) atomicAdd(&sub_c[sub_of(g)], 1u);
        });
        __syncthreads();
        if (tid == 0) {
            unsigned acc = (unsigned)s_lo;
            int j = 0;
            for (; j < kNucSub; ++j) {
                if (acc + sub_c[j] > (unsigned)kNucCap) break;
                acc += sub_c[j];
                if (acc >= (unsigned)K) { ++j; break; }
            }
            s_sub = j - 1;  // the last sub-bin in the window; -1: none of this bin
        }
        __syncthreads();
    }
    // 4. the window's (value, id) keys - the histograms are spent, the union is the sort's now
    const int ssub = s_sub;
    nuc_scan(ar, br, N, vec, [&](float v, int n) {
        const float g = (m - v) * inv_t;
        if (g < kNucGap) {
            const int bin = (int)(g * kNucBinsPerNat);
            if (bin < sel || (bin == sel && (ssub == kNucSub - 1 || sub_of(g) <= ssub)))
                sm.s.e[atomicAdd(&s_nc, 1)] = ((unsigned long long)nuc_key(v) << 32) | (0xffffffffu - (unsigned)n);
        }
    });
    __syncthreads();
    if (tid == 0 && s_nc == 0) {
        // the top bin alone overflows the window: the argmax by itself
        sm.s.e[0] = ((unsigned long long)nuc_key(m) << 32) | (0xffffffffu - (unsigned)s_arg);
        s_nc = 1;
    }
    __syncthreads();
    const int nc = s_nc;
    int P = 1;
    while (P < nc) P <<= 1;
    for (int i = nc + tid; i < P; i += kNucThreads) sm.s.e[i] = 0ull;
    __syncthreads();
    // 5. bitonic sort, descending: value desc, then id asc
    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < P; i += kNucThreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned long long x = sm.s.e[i], y = sm.s.e[ixj];
                    if (((i & k) == 0) ? x < y : x > y) { sm.s.e[i] = y; sm.s.e[ixj] = x; }
                }
            }
            __syncthreads();
        }
    }
    // 6. probabilities over the whole row, their running sums, the cut
    const int Kc = min(nc, K);
    const float zt = s_z;
    const int j0 = 2 * tid, j1 = j0 + 1;
    const float p0 = j0 < Kc ? expf(-((m - nuc_val((uint32_t)(sm.s.e[j0] >> 32))) * inv_t)) / zt : 0.f;
    const float p1 = j1 < Kc ? expf(-((m - nuc_val((uint32_t)(sm.s.e[j1] >> 32))) * inv_t)) / zt : 0.f;
    float x = p0 + p1;
    for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    float ex = __shfl_up_sync(0xffffffffu, x, 1);
    if (lane == 0) ex = 0.f;
    __syncthreads();
    if (lane == 31) rf[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const float t = rf[lane];
        float y = t;
        for (int o = 1; o < 32; o <<= 1) {
            const float q = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o) y += q;
        }
        float e0 = __shfl_up_sync(0xffffffffu, y, 1);
        if (lane == 0) e0 = 0.f;
        rf[lane] = e0;
    }
    __syncthreads();
    const float incl0 = rf[warp] + ex + p0;
    const float incl1 = incl0 + p1;
    if (j0 < Kc) sm.s.incl[j0] = incl0;
    if (j1 < Kc) sm.s.incl[j1] = incl1;
    __syncthreads();
    // kept while the mass before the token is < top_p
    if (j0 < Kc && (j0 ? sm.s.incl[j0 - 1] : 0.f) >= tp) atomicMin(&s_n, j0);
    if (j1 < Kc && sm.s.incl[j1 - 1] >= tp) atomicMin(&s_n, j1);
    __syncthreads();
    const int n = max(1, min(s_n, Kc));
    const float S = sm.s.incl[n - 1];
    if (j0 < n) { ir[j0] = (int)(0xffffffffu - (uint32_t)sm.s.e[j0]); pr[j0] = p0 / S; }
    if (j1 < n) { ir[j1] = (int)(0xffffffffu - (uint32_t)sm.s.e[j1]); pr[j1] = p1 / S; }
    if (tid == 0) cnt[r] = (n < Kc || S >= tp) ? n : -n;
    if (u) {
        // first j whose running mass passes u * S
        const float th = u[r] * S;
        const int c = (j0 < n && sm.s.incl[j0] <= th) + (j1 < n && sm.s.incl[j1] <= th);
        if (c) atomicAdd(&s_draw, c);
        __syncthreads();
        if (tid == 0) tok[r] = (int)(0xffffffffu - (uint32_t)sm.s.e[min(s_draw, n - 1)]);
    }
}

// ---- pair routing for prefill-sized steps ----
// fp4_moe_cuda_route_small's contract - blocks of up to 16 pairs of one slot (block_slot,
// block_pair, block_route = pair | token << 16), pairs at slot -1 dropped - past route_wide's 512
// pairs, by a counting sort over the slots: count, scan (block bases, block_slot), place. Blocks
// come in slot order; within a slot the pairs' order is the atomics', which no output depends
// on - every MoE kernel computes a pair from its own row alone, whatever block it sits in.
// blocks: ds_route_blocks(P, S, bm), entries blocks * bm
__global__ void route_init_kernel(int blocks, int bm, int S, int* block_slot, int* block_pair, int* block_route,
                                  int* counts, int* cursor) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)blocks * bm) { block_pair[i] = -1; block_route[i] = -1; }
    if (i < blocks) block_slot[i] = -1;
    if (i < S) { counts[i] = 0; cursor[i] = 0; }
}

__global__ void route_count_kernel(const int* __restrict__ slots, int P, int* counts) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P && slots[p] >= 0) atomicAdd(&counts[slots[p]], 1);
}

// one 1024-thread block: thread t owns slots [t * per, (t + 1) * per)
__global__ void __launch_bounds__(1024) route_scan_kernel(const int* __restrict__ counts, int S, int bm, int* block_base,
                                                          int* block_slot) {
    __shared__ int warp_tot[32];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int per = (S + 1023) / 1024, s0 = min(S, t * per), s1 = min(S, s0 + per);
    int mine = 0;
    for (int s = s0; s < s1; ++s) mine += (counts[s] + bm - 1) / bm;
    int x = mine;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_tot[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = warp_tot[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        warp_tot[lane] = w;
    }
    __syncthreads();
    int base = (warp ? warp_tot[warp - 1] : 0) + x - mine;
    for (int s = s0; s < s1; ++s) {
        block_base[s] = base;
        const int nb = (counts[s] + bm - 1) / bm;
        for (int b = 0; b < nb; ++b) block_slot[base + b] = s;
        base += nb;
    }
}

__global__ void route_place_kernel(const int* __restrict__ slots, int P, int topk, int bm, const int* __restrict__ block_base,
                                   int* cursor, int* block_pair, int* block_route) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int s = slots[p];
    if (s < 0) return;
    const int i = atomicAdd(&cursor[s], 1);
    const int at = (block_base[s] + i / bm) * bm + i % bm;
    block_pair[at] = p;
    block_route[at] = p | ((p / topk) << 16);
}

// x[i] = table[x[i]]: a listed draft vocabulary's columns to token ids (graph-safe).
__global__ void map_i32_kernel(const int* __restrict__ table, int* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = table[x[i]];
}

// out[m] = table[idx[m]] for 16-bit rows of D values; idx lives on the device (graph-safe).
__global__ void gather_rows_u16_kernel(const uint16_t* __restrict__ table, const int* __restrict__ idx,
                                       uint16_t* __restrict__ out, int D) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) out[(int64_t)m * D + d] = table[(int64_t)idx[m] * D + d];
}

// ------------------------------------------------------------------ graph-safe step plumbing
// The resident step is one CUDA graph, so every decision the streaming engine takes on the
// host from positions (ring slots, compressor pairing, visible lists, rollback) is taken here
// on the device from position buffers instead. Each kernel works on every row and masks, so
// a graph's shape depends only on (rows, position parity, index bucket), never on a value.

// dst[pos[t] % ring_size] = src[t]  (the window ring write, model.py: ring[pos % RING] = kv)
__global__ void scatter_ring_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ ring, int ring_size,
                                    const int* __restrict__ pos, int D, const int* __restrict__ lane, int64_t lstride) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t base = (int64_t)(lane ? lane[t] : 0) * lstride;
    if (d < D) ring[base + (int64_t)(pos[t] % ring_size) * D + d] = src[(int64_t)t * D + d];
}

// out[t, j] = j for j < (pos[t] + 1) / r, else -1; clen[t] = (pos[t] + 1) / r.
__global__ void visible_kernel(const int* __restrict__ pos, int r, int C, int* __restrict__ cidx,
                               int* __restrict__ clen) {
    const int t = blockIdx.y;
    const int vis = (pos[t] + 1) / r;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < C; j += gridDim.x * blockDim.x)
        cidx[(int64_t)t * C + j] = j < vis ? j : -1;
    if (blockIdx.x == 0 && threadIdx.x == 0 && clen) clen[t] = vis;
}

// model.py::_compressed for ratio 2, every row: row t whose position completes a group
// ((pos+1) % 2 == 0) gets the gated pair of (row t-1 or the pending position, row t) in
// `out`; other rows are left as they are (they are never scattered). `pending_valid` is the
// device flag for the pending position.
__global__ void compress_rows_kernel(const float* __restrict__ kv, const float* __restrict__ sc,
                                     const float* __restrict__ pkv, const float* __restrict__ psc,
                                     const int* __restrict__ pos, __nv_bfloat16* __restrict__ out, int D,
                                     const int* __restrict__ lane) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || (pos[t] + 1) % 2 != 0) return;
    const int ln = lane ? lane[t] : 0;
    // rows of one lane are contiguous and in position order; the first row of a lane's run
    // pairs with that lane's pending position
    const bool prev_same = t > 0 && (lane ? lane[t - 1] == ln : true);
    const float* ka = prev_same ? kv + (int64_t)(t - 1) * D : pkv + (int64_t)ln * D;
    const float* sa = prev_same ? sc + (int64_t)(t - 1) * D : psc + (int64_t)ln * D;
    const float a = sa[d], b = sc[(int64_t)t * D + d];
    const float mx = fmaxf(a, b);
    const float ea = expf(a - mx), eb = expf(b - mx);
    const float den = ea + eb;
    out[(int64_t)t * D + d] = to_bf(ka[d] * (ea / den) + kv[(int64_t)t * D + d] * (eb / den));
}

// dst[(pos[t] + 1) / r - 1] = src[t] for rows whose position completes a group of r.
__global__ void scatter_compressed_kernel(const uint16_t* __restrict__ src, int64_t ld_src, uint16_t* __restrict__ dst,
                                          const int* __restrict__ pos, int r, int D, const int* __restrict__ lane,
                                          int64_t lstride) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || (pos[t] + 1) % r != 0) return;
    const int64_t base = (int64_t)(lane ? lane[t] : 0) * lstride;
    dst[base + (int64_t)((pos[t] + 1) / r - 1) * D + d] = src[(int64_t)t * ld_src + d];
}

// After a ratio-2 chunk, per lane segment of the step (block b: rows seg_first[b] ..
// + seg_count[b] of lane seg_lane[b]): the segment's last position is unpaired iff
// (pos[last] + 1) is odd; save the segment's inputs and the pending state before it (for
// rollback) and set the new pending state. Per-lane state is `[lanes][...]`.
__global__ void pending_update_kernel(const float* __restrict__ kv, const float* __restrict__ sc, const int* __restrict__ pos,
                                      const int* __restrict__ seg_first, const int* __restrict__ seg_count,
                                      const int* __restrict__ seg_lane, float* pkv, float* psc, int* flag, float* mem_kv,
                                      float* mem_sc, float* before_kv, float* before_sc, int* before_flag, int* mem_start,
                                      int D, int mem_rows) {
    const int f = seg_first[blockIdx.x], T = seg_count[blockIdx.x], L = seg_lane[blockIdx.x];
    // the step graph launches one block per lane slot; a lane absent from this step has T == 0
    if (T == 0) return;
    float* pk = pkv + (int64_t)L * D;
    float* ps = psc + (int64_t)L * D;
    // the segment's last mem_rows rows: a rollback reaches only into a verify block, and a
    // prefill segment longer than the memory is never rolled back (the engine refuses it)
    const int kept = min(T, mem_rows), f0 = f + T - kept;
    for (int i = threadIdx.x; i < kept * D; i += blockDim.x) {
        mem_kv[(int64_t)L * mem_rows * D + i] = kv[(int64_t)f0 * D + i];
        mem_sc[(int64_t)L * mem_rows * D + i] = sc[(int64_t)f0 * D + i];
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        before_kv[(int64_t)L * D + d] = pk[d];
        before_sc[(int64_t)L * D + d] = ps[d];
    }
    __syncthreads();
    const bool unpaired = (pos[f + T - 1] + 1) % 2 != 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        if (unpaired) { pk[d] = kv[(int64_t)(f + T - 1) * D + d]; ps[d] = sc[(int64_t)(f + T - 1) * D + d]; }
    if (threadIdx.x == 0) {
        before_flag[L] = flag[L];
        flag[L] = unpaired ? 1 : 0;
        mem_start[L] = pos[f0];
    }
}

// A ratio-2 source's pending state at length n of lane L (model.py::Caches.rollback): even n
// has none; odd n's is position n - 1's input, from the last chunk's memory, or the state
// before that chunk if n - 1 precedes it. Written to (pk, ps, flag) - the lane's own state
// for a rollback, a snapshot's for the prefix cache. One block a call.
__device__ __forceinline__ void pending_at(int n, int L, const float* __restrict__ mem_kv, const float* __restrict__ mem_sc,
                                           const int* __restrict__ mem_start, const float* __restrict__ before_kv,
                                           const float* __restrict__ before_sc, const int* __restrict__ before_flag,
                                           float* pk, float* ps, int* flag, int D, int mem_rows) {
    if (n % 2 == 0) {
        if (threadIdx.x == 0) *flag = 0;
        return;
    }
    const int p = n - 1, start = mem_start[L];
    // a rollback reaches at most into the last chunk; anything else is a caller bug and
    // must not read past the chunk memory
    if (p - start >= mem_rows || p + 1 < start) return;
    if (p >= start) {
        const int64_t row = (int64_t)L * mem_rows * D + (int64_t)(p - start) * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) { pk[d] = mem_kv[row + d]; ps[d] = mem_sc[row + d]; }
        if (threadIdx.x == 0) *flag = 1;
    } else {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            pk[d] = before_kv[(int64_t)L * D + d];
            ps[d] = before_sc[(int64_t)L * D + d];
        }
        if (threadIdx.x == 0) *flag = before_flag[L];
    }
}

// model.py::Caches.rollback for a ratio-2 source, per lane: block b restores lane lanes[b]
// to length n_dev[b] (read from the device: the accepted length of its verify block).
__global__ void rollback_pending_kernel(const int* __restrict__ n_dev, const int* __restrict__ lanes,
                                        const float* __restrict__ mem_kv, const float* __restrict__ mem_sc,
                                        const int* __restrict__ mem_start, const float* __restrict__ before_kv,
                                        const float* __restrict__ before_sc, const int* __restrict__ before_flag,
                                        float* pkv, float* psc, int* flag, int D, int mem_rows) {
    const int L = lanes ? lanes[blockIdx.x] : 0;
    pending_at(n_dev[blockIdx.x], L, mem_kv, mem_sc, mem_start, before_kv, before_sc, before_flag, pkv + (int64_t)L * D,
               psc + (int64_t)L * D, flag + L, D, mem_rows);
}

// The prefix cache's copy of lane L's pending state at length n: out [2][D] (kv, score) and
// *out_flag; the lane's own state is untouched.
__global__ void pending_snapshot_kernel(int n, int L, const float* __restrict__ mem_kv, const float* __restrict__ mem_sc,
                                        const int* __restrict__ mem_start, const float* __restrict__ before_kv,
                                        const float* __restrict__ before_sc, const int* __restrict__ before_flag,
                                        float* out, int* out_flag, int D, int mem_rows) {
    pending_at(n, L, mem_kv, mem_sc, mem_start, before_kv, before_sc, before_flag, out, out + D, out_flag, D, mem_rows);
}

// out[idx[m]] = src[m] for rows of D 16-bit values (the prefix cache's ring rows back into
// their slots).
__global__ void scatter_rows_u16_kernel(const uint16_t* __restrict__ src, const int* __restrict__ idx, uint16_t* __restrict__ out,
                                        int D) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    // a row whose index is negative stays where it is
    if (d < D && idx[m] >= 0) out[(int64_t)idx[m] * D + d] = src[(int64_t)m * D + d];
}

// Routed expert ids -> arena slots through the device LUT (model.py: lut[L][indices]).
// Pairs at or past *live (null: none) belong to a padded step's filler rows and get slot -1,
// which the MoE kernels skip: filler rows must not pull experts through the bandwidth.
__global__ void lut_slots_kernel(const int* __restrict__ idx, const int* __restrict__ lut, int P,
                                 int* __restrict__ slots, const int* __restrict__ live) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) slots[p] = (live && p >= *live) ? -1 : lut[idx[p]];
}

// DSV41_EXPERT_TOPP (engine/expert_topp.py) on the rows with rows[t] set: by weight share
// descending (ties: the lower pick), keep the shortest prefix whose share before it is < topp,
// never fewer than min_k; a dropped pick weighs 0 and routes to expert `sentinel`, whose slot
// table column is the arena's null slot (its weights are never read, its output is an exact 0).
// The kept picks keep the router's own weights. In fp32 as keep_mask: shares w / sum(w), the
// running share a sequential inclusive sum, tested as (running - share) < topp.
__global__ void expert_topp_kernel(int* __restrict__ idx, float* __restrict__ rw, int T, int K, float topp, int min_k,
                                   int sentinel, const int* __restrict__ rows) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || !rows[t]) return;
    float f[8];
    int ord[8];
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += rw[t * K + k];
    for (int k = 0; k < K; ++k) {
        f[k] = rw[t * K + k] / s;
        // stable insertion by share, descending
        int j = k;
        while (j > 0 && f[ord[j - 1]] < f[k]) {
            ord[j] = ord[j - 1];
            --j;
        }
        ord[j] = k;
    }
    float run = 0.f;
    for (int j = 0; j < K; ++j) {
        const int k = ord[j];
        run += f[k];
        if (j >= min_k && !((run - f[k]) < topp)) {
            rw[t * K + k] = 0.f;
            idx[t * K + k] = sentinel;
        }
    }
}

// v41_engine.py lean greedy verify: am[t] is row t's argmax, block the verified tokens.
// out = [accepted a, bonus am[a], n = pos0 + a + 1, next draft position pos0 + a].
// Stop tokens are the host's business (it truncates on the copy it reads).
__global__ void greedy_accept_kernel(const int* __restrict__ am, const int* __restrict__ block, int T,
                                     const int* __restrict__ pos, int* __restrict__ out) {
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a + 1 < T && am[a] == block[a + 1]) ++a;
    out[0] = a;
    out[1] = am[a];
    out[2] = pos[0] + a + 1;
    out[3] = pos[0] + a;
}

// Spin until *flag == expect (a host thread writes it in mapped pinned memory once the
// engram rows of this step have landed). Gives up after `timeout_ns` and records the miss
// in *err, so a lost signal stalls one step instead of the GPU forever.
__global__ void wait_flag_kernel(const volatile int* flag, int expect, long long timeout_ns, int* err) {
    if (threadIdx.x != 0) return;
    const unsigned long long t0 = clock64();
    long long spins = 0;
    while (*flag != expect) {
        __nanosleep(200);
        if (++spins * 200 > timeout_ns) { *err = 1; break; }
    }
    __threadfence_system();
    (void)t0;
}

}  // namespace

// ================================================================== C ABI
extern "C" {

int ds_linear_fp8(void* stream, const void* x, int64_t ldx, const void* w, const void* s, int64_t lds,
                  int M, int N, int K, int splits, void* y_bf16, void* y_f32, int64_t ldy, void* partial) {
    return launch_skinny<0>(static_cast<cudaStream_t>(stream), x, ldx, 0, 0, w, s, lds, M, N, K, splits,
                            y_bf16, y_f32, ldy, partial);
}

int ds_linear_fp8_grouped(void* stream, const void* x, int64_t ldx, int64_t x_group_stride,
                          int rows_per_group, const void* w, const void* s, int64_t lds, int M, int N,
                          int K, void* y_bf16, int64_t ldy) {
    if (rows_per_group % 32) return cudaErrorInvalidValue;
    return launch_skinny<0>(static_cast<cudaStream_t>(stream), x, ldx, x_group_stride, rows_per_group, w,
                            s, lds, M, N, K, 1, y_bf16, nullptr, ldy, nullptr);
}

int ds_linear_bf16(void* stream, const void* x, int64_t ldx, const void* w, int M, int N, int K, int splits,
                   void* y_bf16, void* y_f32, int64_t ldy, void* partial) {
    return launch_skinny<1>(static_cast<cudaStream_t>(stream), x, ldx, 0, 0, w, nullptr, 0, M, N, K, splits,
                            y_bf16, y_f32, ldy, partial);
}

int ds_linear_f32w(void* stream, const void* x, int64_t ldx, const void* w, int M, int N, int K, int splits,
                   void* y_f32, int64_t ldy, void* partial) {
    return launch_skinny<2>(static_cast<cudaStream_t>(stream), x, ldx, 0, 0, w, nullptr, 0, M, N, K, splits,
                            nullptr, y_f32, ldy, partial);
}

int ds_splits_reduce(void* stream, const void* partial, int splits, int M, int N, void* y_bf16, void* y_f32,
                     int64_t ldy) {
    const int64_t total = (int64_t)M * N;
    splits_reduce_kernel<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(partial), splits, M, N, static_cast<__nv_bfloat16*>(y_bf16),
        static_cast<float*>(y_f32), ldy);
    return static_cast<int>(cudaGetLastError());
}

int ds_rmsnorm(void* stream, const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int D,
               float eps) {
    rmsnorm_kernel<<<M, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const __nv_bfloat16*>(w),
        static_cast<__nv_bfloat16*>(y), ldy, D, eps);
    return static_cast<int>(cudaGetLastError());
}

int ds_hc_pre_norm(void* stream, const void* h, const void* pre, const void* w, void* y, int M, int D, int hc,
                   float eps) {
    const int threads = ((D / 8 + 31) / 32) * 32;
    if (D % 8 == 0 && threads <= 1024) {
        hc_pre_norm_v8_kernel<<<M, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(h), static_cast<const float*>(pre),
            static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), D, hc, eps);
        return static_cast<int>(cudaGetLastError());
    }
    hc_pre_norm_kernel<<<M, 256, D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<const float*>(pre),
        static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), D, hc, eps);
    return static_cast<int>(cudaGetLastError());
}

int ds_row_rsqrt(void* stream, const void* x, int64_t ldx, int M, int D, float eps, void* out) {
    if (D % 8 == 0 && ldx % 8 == 0) {
        row_rsqrt_v8_kernel<<<M, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(x), ldx, D, eps, static_cast<float*>(out));
        return static_cast<int>(cudaGetLastError());
    }
    row_rsqrt_kernel<<<M, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, D, eps, static_cast<float*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_hc_sinkhorn(void* stream, const void* mixes, const void* rsq, const void* scale, const void* base,
                   int M, int iters, float eps, void* pre, void* post, void* comb) {
    hc_sinkhorn_kernel<<<(M + 3) / 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(mixes), static_cast<const float*>(rsq), static_cast<const float*>(scale),
        static_cast<const float*>(base), M, iters, eps, static_cast<float*>(pre), static_cast<float*>(post),
        static_cast<float*>(comb));
    return static_cast<int>(cudaGetLastError());
}

int ds_hc_post(void* stream, const void* y, const void* residual, const void* post, const void* comb, void* out,
               int M, int D) {
    dim3 grid((D + 255) / 256, M);
    hc_post_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(residual),
        static_cast<const float*>(post), static_cast<const float*>(comb), static_cast<__nv_bfloat16*>(out), D);
    return static_cast<int>(cudaGetLastError());
}

int ds_rope(void* stream, void* x, int64_t ld_row, int rows, int heads, int64_t ld_head, int offset, int rd,
            const void* pos, const void* table, int inverse) {
    rope_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(x), ld_row, heads, ld_head, offset, rd, static_cast<const int*>(pos),
        static_cast<const float2*>(table), inverse);
    return static_cast<int>(cudaGetLastError());
}

int ds_router(void* stream, const void* z, const void* bias, const void* keep, int M, int E, int K,
              float route_scale, void* idx, void* weights, const void* img, const void* bias_vl) {
    if (K > 8 || (img && !bias_vl)) return cudaErrorInvalidValue;
    RouterRec rec{};
    rec.img = static_cast<const int*>(img);
    rec.bias_vl = static_cast<const float*>(bias_vl);
    router_kernel<<<M, 128, 2 * E * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(z), static_cast<const float*>(bias), static_cast<const uint8_t*>(keep), E, K,
        route_scale, static_cast<int*>(idx), static_cast<float*>(weights), rec);
    return static_cast<int>(cudaGetLastError());
}

int ds_router_rec(void* stream, const void* z, const void* bias, const void* keep, int M, int E, int K,
                  float route_scale, void* idx, void* weights, void* counts, void* mass, void* miss,
                  int64_t lane_stride, const void* lane, int lanes, const void* phase, void* hot, const void* live,
                  const void* img, const void* bias_vl) {
    if (K > 8 || (img && !bias_vl)) return cudaErrorInvalidValue;
    RouterRec rec{static_cast<double*>(counts), static_cast<double*>(mass), static_cast<double*>(miss), lane_stride,
                  static_cast<const int*>(lane), lanes, static_cast<const int*>(phase), static_cast<float*>(hot),
                  static_cast<const int*>(live), static_cast<const int*>(img), static_cast<const float*>(bias_vl)};
    if (rec.counts && (!rec.mass || !rec.miss)) return cudaErrorInvalidValue;
    router_kernel<<<M, 128, 2 * E * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(z), static_cast<const float*>(bias), static_cast<const uint8_t*>(keep), E, K,
        route_scale, static_cast<int*>(idx), static_cast<float*>(weights), rec);
    return static_cast<int>(cudaGetLastError());
}

int ds_hot_marks(void* stream, void* lut, void* keep, const void* marks, void* saved, int n, int apply) {
    if (n <= 0) return cudaSuccess;
    hot_marks_kernel<<<(n + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int*>(lut), static_cast<uint8_t*>(keep), static_cast<const int*>(marks), static_cast<int*>(saved), n, apply);
    return static_cast<int>(cudaGetLastError());
}

int ds_hot_candidates(void* stream, const void* hot, int n, int W, void* out) {
    if (W < 1 || W > 32 || n < 0) return cudaErrorInvalidValue;
    hot_candidates_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(hot), n, W,
                                                                           static_cast<float*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_swiglu(void* stream, const void* gate, int64_t ldg, const void* up, int64_t ldu, void* h, int64_t ldh,
              int M, int N, float limit) {
    dim3 grid((N + 255) / 256, M);
    swiglu_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(gate), ldg, static_cast<const __nv_bfloat16*>(up), ldu,
        static_cast<__nv_bfloat16*>(h), ldh, N, limit);
    return static_cast<int>(cudaGetLastError());
}

int ds_moe_combine(void* stream, const void* parts, int K, int M, const void* shared, void* out, int D) {
    dim3 grid((D + 255) / 256, M);
    moe_combine_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(parts), K, M, static_cast<const __nv_bfloat16*>(shared),
        static_cast<__nv_bfloat16*>(out), D);
    return static_cast<int>(cudaGetLastError());
}

extern "C++" {
template <int HG>
static int launch_attn(cudaStream_t st, const void* q, const void* ring, int ring_size, const void* pos, int W, const void* win_lo,
                       const void* ckv, const void* cidx, int C, const void* sink, float scale, void* out, int M, int H,
                       const void* lane, int64_t ring_lstride, int64_t ckv_lstride) {
    const size_t smem = (size_t)HG * (512 + W + C) * sizeof(float);
    static size_t configured = 0;
    if (smem > 48 * 1024 && smem > configured) {
        const cudaError_t e = cudaFuncSetAttribute(attn_decode_kernel<HG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem);
        if (e != cudaSuccess) return static_cast<int>(e);
        configured = smem;
    }
    dim3 grid(H / HG, M);
    attn_decode_kernel<HG><<<grid, 256, smem, st>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(ring), ring_size,
        static_cast<const int*>(pos), W, static_cast<const int*>(win_lo), static_cast<const __nv_bfloat16*>(ckv),
        static_cast<const int*>(cidx), C, static_cast<const float*>(sink), scale,
        static_cast<__nv_bfloat16*>(out), H, static_cast<const int*>(lane), ring_lstride, ckv_lstride);
    return static_cast<int>(cudaGetLastError());
}
}  // extern "C++"

/// Workspace bytes ds_attn_decode's split-key kernel needs for M rows of H heads over at
/// most NK keys (window + compressed).
int64_t ds_attn_work_bytes(int M, int H, int NK) {
    const int64_t S = (NK + kFdN * kFdSplitTiles - 1) / (kFdN * kFdSplitTiles);
    return (int64_t)M * S * H * (kFdD + 2) * (int64_t)sizeof(float);
}

int ds_attn_decode(void* stream, const void* q, const void* ring, int ring_size, const void* pos, int W,
                   const void* win_lo, const void* ckv, const void* cidx, int C, const void* sink, float scale, void* out,
                   int M, int H, const void* lane, int64_t ring_lstride, int64_t ckv_lstride, void* work,
                   int64_t work_bytes, const void* decode_rows, int ckv_fp4) {
    const int* dr = static_cast<const int*>(decode_rows);
    if (work && work_bytes >= ds_attn_work_bytes(M, H, W + C)) {
        const int hb = (H + kFdH - 1) / kFdH;
        const int n_tiles = (W + C + kFdN - 1) / kFdN;
        const int tps = kFdSplitTiles;
        const int S = (n_tiles + tps - 1) / tps;
        float* po = static_cast<float*>(work);
        float* pml = po + (int64_t)M * S * H * kFdD;
        const size_t smem = (size_t)(kFdH + kFdN) * kFdKS * 2 + (size_t)kFdH * kFdPS * 4 + 3 * kFdH * 4 + kFdN * 4 +
                            (size_t)kFdSplitTiles * kFdN * 4;
        static bool configured = false;
        if (!configured) {
            const cudaError_t e = cudaFuncSetAttribute(attn_fd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            if (e != cudaSuccess) return static_cast<int>(e);
            configured = true;
        }
        attn_fd_kernel<<<dim3(hb, S, M), 256, smem, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(ring), ring_size,
            static_cast<const int*>(pos), W, static_cast<const int*>(win_lo), static_cast<const __nv_bfloat16*>(ckv), static_cast<const int*>(cidx),
            C, scale, H, static_cast<const int*>(lane), ring_lstride, ckv_lstride, tps, S, po, pml, dr, ckv_fp4);
        attn_fd_combine_kernel<<<dim3(H, M), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            po, pml, S, H, static_cast<const float*>(sink), static_cast<__nv_bfloat16*>(out), dr);
        return static_cast<int>(cudaGetLastError());
    }
    // the per-head kernel reads bf16 rows only
    if (H % kAttnHG || dr || ckv_fp4) return cudaErrorInvalidValue;
    // about two blocks an SM: more heads a block re-read the (shared, MQA) keys fewer times,
    // fewer heads a block fill the GPU at decode sizes
    const int want = M * H / 96;
    auto st = static_cast<cudaStream_t>(stream);
    if (want >= 4) return launch_attn<4>(st, q, ring, ring_size, pos, W, win_lo, ckv, cidx, C, sink, scale, out, M, H, lane, ring_lstride, ckv_lstride);
    if (want >= 2) return launch_attn<2>(st, q, ring, ring_size, pos, W, win_lo, ckv, cidx, C, sink, scale, out, M, H, lane, ring_lstride, ckv_lstride);
    return launch_attn<1>(st, q, ring, ring_size, pos, W, win_lo, ckv, cidx, C, sink, scale, out, M, H, lane, ring_lstride, ckv_lstride);
}

int ds_attn_prefill(void* stream, const void* q, const void* ring, int ring_size, const void* pos, int W, const void* win_lo,
                    const void* ckv, const void* cidx, int C, const void* sink, float scale, void* out, int M, int H,
                    const void* lane, int64_t ring_lstride, int64_t ckv_lstride, const void* rows, int heads_per_block,
                    int ckv_fp4) {
    if (M < 1) return cudaSuccess;
    auto go = [&](auto kern, int hb_heads, int nt) -> int {
        const size_t smem = (size_t)(hb_heads + nt) * kFdKS * 2 + (size_t)hb_heads * (nt + 4) * 4 + 3 * hb_heads * 4 + nt * 4 +
                            (size_t)C * 4;
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return static_cast<int>(e);
        kern<<<dim3((H + hb_heads - 1) / hb_heads, M), 256, smem, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(ring), ring_size,
            static_cast<const int*>(pos), W, static_cast<const int*>(win_lo), static_cast<const __nv_bfloat16*>(ckv), static_cast<const int*>(cidx),
            C, scale, H, static_cast<const int*>(lane), ring_lstride, ckv_lstride, static_cast<const float*>(sink),
            static_cast<const int*>(rows), static_cast<__nv_bfloat16*>(out), ckv_fp4);
        return static_cast<int>(cudaGetLastError());
    };
    if (heads_per_block == 32) return go(attn_prefill_kernel<2, 32>, 32, 32);
    if (heads_per_block == 16) return go(attn_prefill_kernel<1, 64>, 16, 64);
    return cudaErrorInvalidValue;
}

int ds_compress_pair(void* stream, const void* kv_a, const void* sc_a, const void* kv_b, const void* sc_b,
                     void* out, int D) {
    compress_pair_kernel<<<(D + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(kv_a), static_cast<const float*>(sc_a), static_cast<const float*>(kv_b),
        static_cast<const float*>(sc_b), static_cast<__nv_bfloat16*>(out), D);
    return static_cast<int>(cudaGetLastError());
}

int ds_scatter_compressed_fp4(void* stream, const void* src, int64_t ld_src, void* dst, const void* pos, int r, int T,
                              int D, const void* lane, int64_t lstride_bytes) {
    if (D % 128 || D / 16 > 1024) return cudaErrorInvalidValue;
    if (T < 1) return cudaSuccess;
    scatter_compressed_fp4_kernel<<<T, D / 16, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(src), ld_src, static_cast<uint8_t*>(dst), static_cast<const int*>(pos), r, D,
        static_cast<const int*>(lane), lstride_bytes);
    return static_cast<int>(cudaGetLastError());
}

int ds_unpack_fp4_rows(void* stream, const void* src, void* dst, int rows) {
    if (rows < 1) return cudaSuccess;
    const int64_t n = (int64_t)rows * 64;
    unpack_fp4_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(src), static_cast<__nv_bfloat16*>(dst), rows);
    return static_cast<int>(cudaGetLastError());
}

int ds_pending_snapshot(void* stream, int n, int lane, const void* mem_kv, const void* mem_sc, const void* mem_start,
                        const void* before_kv, const void* before_sc, const void* before_flag, void* out, void* out_flag, int D,
                        int mem_rows) {
    pending_snapshot_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, lane, static_cast<const float*>(mem_kv), static_cast<const float*>(mem_sc), static_cast<const int*>(mem_start),
        static_cast<const float*>(before_kv), static_cast<const float*>(before_sc), static_cast<const int*>(before_flag),
        static_cast<float*>(out), static_cast<int*>(out_flag), D, mem_rows);
    return static_cast<int>(cudaGetLastError());
}

int ds_scatter_rows_u16(void* stream, const void* src, const void* idx, void* out, int M, int D) {
    if (M < 1) return cudaSuccess;
    scatter_rows_u16_kernel<<<dim3((D + 255) / 256, M), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(src), static_cast<const int*>(idx), static_cast<uint16_t*>(out), D);
    return static_cast<int>(cudaGetLastError());
}

int ds_fp4_qdq_e4m3(void* stream, void* x, int64_t n_values) {
    if (n_values % 16) return cudaErrorInvalidValue;
    const int64_t groups = n_values / 16;
    fp4_qdq_e4m3_kernel<<<(unsigned)((groups + 127) / 128), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(x), groups);
    return static_cast<int>(cudaGetLastError());
}

int ds_engram_dequant(void* stream, const void* raw, const void* inv, int tokens, int cols, void* out) {
    engram_dequant_kernel<<<tokens * cols, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(raw), static_cast<const int*>(inv), cols, static_cast<__nv_bfloat16*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_engram_gate(void* stream, void* h, const void* kv, const void* weight, int M, int D, int hc, float eps,
                   float dim_scale, const void* skip) {
    engram_gate_kernel<<<M * hc, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(kv), static_cast<const float*>(weight),
        D, hc, eps, dim_scale, static_cast<const int*>(skip));
    return static_cast<int>(cudaGetLastError());
}

// The image splice (inference/model.py's embed of an image span): rows with idx[m] >= 0 take
// table row idx[m] - an aligner row or a learned delimiter - in every hc copy.
__global__ void splice_rows_kernel(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ table,
                                   const int* __restrict__ idx, int D, int hc) {
    const int m = blockIdx.y, src = idx[m];
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < 0 || d >= D) return;
    const __nv_bfloat16 v = table[(int64_t)src * D + d];
    for (int c = 0; c < hc; ++c) h[((int64_t)m * hc + c) * D + d] = v;
}

int ds_splice_rows(void* stream, void* h, const void* table, const void* idx, int M, int D, int hc) {
    dim3 grid((D + 255) / 256, M);
    splice_rows_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(table), static_cast<const int*>(idx), D, hc);
    return static_cast<int>(cudaGetLastError());
}

int ds_embed_hc(void* stream, const void* ids, const void* table, void* h, int M, int D, int hc) {
    dim3 grid((D + 255) / 256, M);
    embed_hc_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(ids), static_cast<const __nv_bfloat16*>(table), static_cast<__nv_bfloat16*>(h),
        D, hc);
    return static_cast<int>(cudaGetLastError());
}

// Whether rows of a (and b) can be read 16 bytes at a time.
static bool row_vec(const void* a, int64_t lda, const void* b, int64_t ldb, int N) {
    const auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    return N % 8 == 0 && lda % 8 == 0 && al(a) && (!b || (ldb % 8 == 0 && al(b)));
}

int ds_argmax_bf16(void* stream, const void* x, int64_t ldx, int M, int N, void* out) {
    argmax_bf16_kernel<<<M, kRowThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, N, row_vec(x, ldx, nullptr, 0, N), static_cast<int*>(out), 0, 0);
    return static_cast<int>(cudaGetLastError());
}

int ds_index_scores(void* stream, const void* q, const void* k, const void* w, const void* clen, const void* cand,
                    int64_t ld_cand, int n, void* out, int64_t ld_out, int T, const void* lane, int64_t k_lstride) {
    if (n < 1) return cudaErrorInvalidValue;
    dim3 grid((n + kIdxTile - 1) / kIdxTile, T);
    index_scores_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), static_cast<const float*>(w),
        static_cast<const int*>(clen), static_cast<const uint8_t*>(cand), ld_cand, n,
        static_cast<__nv_bfloat16*>(out), ld_out, static_cast<const int*>(lane), k_lstride);
    return static_cast<int>(cudaGetLastError());
}

int ds_index_scores_mma_chunked(void* stream, const void* q, const void* k, const void* w, const void* clen,
                                const void* cand, int64_t ld_cand, int n, void* out, int64_t ld_out, int T,
                                const void* lane, int64_t k_lstride, int chunk) {
    if (n < 1 || T < 1 || chunk < kIdxTile || chunk % kIdxTile) return cudaErrorInvalidValue;
    dim3 grid((T + kIsRows - 1) / kIsRows, (n + chunk - 1) / chunk);
    index_scores_mma_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), static_cast<const float*>(w),
        static_cast<const int*>(clen), static_cast<const uint8_t*>(cand), ld_cand, n,
        static_cast<__nv_bfloat16*>(out), ld_out, static_cast<const int*>(lane), k_lstride, T, chunk);
    return static_cast<int>(cudaGetLastError());
}

int ds_index_scores_mma(void* stream, const void* q, const void* k, const void* w, const void* clen, const void* cand,
                        int64_t ld_cand, int n, void* out, int64_t ld_out, int T, const void* lane, int64_t k_lstride) {
    const int whole = (n + kIdxTile - 1) / kIdxTile * kIdxTile;
    return ds_index_scores_mma_chunked(stream, q, k, w, clen, cand, ld_cand, n, out, ld_out, T, lane, k_lstride,
                                       whole < kIdxTile ? kIdxTile : whole);
}

int ds_topk_select(void* stream, const void* s, int64_t ld_s, int n, int K, void* out, int width, int T) {
    if (K > width || n < 1) return cudaErrorInvalidValue;
    topk_select_kernel<<<T, kSelThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(s), ld_s, n, K, static_cast<int*>(out), width);
    return static_cast<int>(cudaGetLastError());
}

int ds_candidate_blocks(void* stream, const void* s, int64_t ld_s, int n, int block, const void* clen, void* bmax,
                        int64_t ld_b, int T) {
    const int nb = (n + block - 1) / block;
    dim3 grid((nb + 255) / 256, T);
    candidate_blocks_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(s), ld_s, n, block, static_cast<const int*>(clen),
        static_cast<__nv_bfloat16*>(bmax), ld_b);
    return static_cast<int>(cudaGetLastError());
}

int ds_candidate_set(void* stream, const void* blocks, int kb, int block, int n, void* cand, int64_t ld_cand, int T) {
    dim3 grid((kb * block + 255) / 256, T);
    candidate_set_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(blocks), kb, block, n, static_cast<uint8_t*>(cand), ld_cand);
    return static_cast<int>(cudaGetLastError());
}

int ds_bf16_to_f32_scale(void* stream, const void* x, int64_t n, float scale, void* out) {
    bf16_to_f32_scale_kernel<<<(unsigned)((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), n, scale, static_cast<float*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_hc_mean(void* stream, const void* h, void* out, int64_t ld_out, int M, int D, int hc) {
    dim3 grid((D + 255) / 256, M);
    hc_mean_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<__nv_bfloat16*>(out), ld_out, D, hc);
    return static_cast<int>(cudaGetLastError());
}

int ds_argmax_sum_bf16(void* stream, const void* a, int64_t lda, const void* b, int64_t ldb, int M, int N, void* out) {
    argmax_sum_bf16_kernel<<<M, kRowThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(a), lda, static_cast<const __nv_bfloat16*>(b), ldb, N, row_vec(a, lda, b, ldb, N),
        static_cast<int*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_gather_rows_u16(void* stream, const void* table, const void* idx, void* out, int M, int D) {
    dim3 grid((D + 255) / 256, M);
    gather_rows_u16_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(table), static_cast<const int*>(idx), static_cast<uint16_t*>(out), D);
    return static_cast<int>(cudaGetLastError());
}

#define DS_RET return static_cast<int>(cudaGetLastError())
#define DS_ST static_cast<cudaStream_t>(stream)

int ds_scatter_ring(void* stream, const void* src, void* ring, int ring_size, const void* pos, int T, int D,
                    const void* lane, int64_t lstride) {
    scatter_ring_kernel<<<dim3((D + 255) / 256, T), 256, 0, DS_ST>>>(static_cast<const uint16_t*>(src),
        static_cast<uint16_t*>(ring), ring_size, static_cast<const int*>(pos), D, static_cast<const int*>(lane), lstride);
    DS_RET;
}

int ds_visible(void* stream, const void* pos, int T, int r, int C, void* cidx, void* clen) {
    visible_kernel<<<dim3((C + 255) / 256 > 0 ? (C + 255) / 256 : 1, T), 256, 0, DS_ST>>>(static_cast<const int*>(pos), r,
        C, static_cast<int*>(cidx), static_cast<int*>(clen));
    DS_RET;
}

int ds_compress_rows(void* stream, const void* kv, const void* sc, const void* pkv, const void* psc, const void* pos,
                     void* out, int T, int D, const void* lane) {
    compress_rows_kernel<<<dim3((D + 255) / 256, T), 256, 0, DS_ST>>>(static_cast<const float*>(kv),
        static_cast<const float*>(sc), static_cast<const float*>(pkv), static_cast<const float*>(psc),
        static_cast<const int*>(pos), static_cast<__nv_bfloat16*>(out), D, static_cast<const int*>(lane));
    DS_RET;
}

int ds_scatter_compressed(void* stream, const void* src, int64_t ld_src, void* dst, const void* pos, int r, int T, int D,
                          const void* lane, int64_t lstride) {
    scatter_compressed_kernel<<<dim3((D + 255) / 256, T), 256, 0, DS_ST>>>(static_cast<const uint16_t*>(src), ld_src,
        static_cast<uint16_t*>(dst), static_cast<const int*>(pos), r, D, static_cast<const int*>(lane), lstride);
    DS_RET;
}

int ds_pending_update(void* stream, const void* kv, const void* sc, const void* pos, const void* seg_first,
                      const void* seg_count, const void* seg_lane, int segments, void* pkv, void* psc, void* flag,
                      void* mem_kv, void* mem_sc, void* before_kv, void* before_sc, void* before_flag, void* mem_start,
                      int D, int mem_rows) {
    pending_update_kernel<<<segments, 256, 0, DS_ST>>>(static_cast<const float*>(kv), static_cast<const float*>(sc),
        static_cast<const int*>(pos), static_cast<const int*>(seg_first), static_cast<const int*>(seg_count),
        static_cast<const int*>(seg_lane), static_cast<float*>(pkv), static_cast<float*>(psc), static_cast<int*>(flag),
        static_cast<float*>(mem_kv), static_cast<float*>(mem_sc), static_cast<float*>(before_kv),
        static_cast<float*>(before_sc), static_cast<int*>(before_flag), static_cast<int*>(mem_start), D, mem_rows);
    DS_RET;
}

int ds_rollback_pending(void* stream, const void* n_dev, const void* lanes, int entries, const void* mem_kv,
                        const void* mem_sc, const void* mem_start, const void* before_kv, const void* before_sc,
                        const void* before_flag, void* pkv, void* psc, void* flag, int D, int mem_rows) {
    rollback_pending_kernel<<<entries, 256, 0, DS_ST>>>(static_cast<const int*>(n_dev), static_cast<const int*>(lanes),
        static_cast<const float*>(mem_kv), static_cast<const float*>(mem_sc), static_cast<const int*>(mem_start),
        static_cast<const float*>(before_kv), static_cast<const float*>(before_sc), static_cast<const int*>(before_flag),
        static_cast<float*>(pkv), static_cast<float*>(psc), static_cast<int*>(flag), D, mem_rows);
    DS_RET;
}

int ds_lut_slots(void* stream, const void* idx, const void* lut, int P, void* slots, const void* live) {
    lut_slots_kernel<<<(P + 127) / 128, 128, 0, DS_ST>>>(static_cast<const int*>(idx), static_cast<const int*>(lut), P,
        static_cast<int*>(slots), static_cast<const int*>(live));
    DS_RET;
}

int ds_expert_topp(void* stream, void* idx, void* rw, int T, int K, float topp, int min_k, int sentinel, const void* rows) {
    if (K > 8 || min_k < 1 || min_k > K) return cudaErrorInvalidValue;
    expert_topp_kernel<<<(T + 127) / 128, 128, 0, DS_ST>>>(static_cast<int*>(idx), static_cast<float*>(rw), T, K, topp,
                                                           min_k, sentinel, static_cast<const int*>(rows));
    DS_RET;
}

int ds_greedy_accept(void* stream, const void* am, const void* block, int T, const void* pos, void* out) {
    greedy_accept_kernel<<<1, 32, 0, DS_ST>>>(static_cast<const int*>(am), static_cast<const int*>(block), T,
        static_cast<const int*>(pos), static_cast<int*>(out));
    DS_RET;
}

int ds_wait_flag(void* stream, const void* flag, int expect, long long timeout_ns, void* err) {
    wait_flag_kernel<<<1, 32, 0, DS_ST>>>(static_cast<const volatile int*>(flag), expect, timeout_ns, static_cast<int*>(err));
    DS_RET;
}

int ds_argmax_bf16_pairs(void* stream, const void* x, int64_t ldx, int M, int N, int offset, void* out) {
    argmax_bf16_kernel<<<M, kRowThreads, 0, DS_ST>>>(static_cast<const __nv_bfloat16*>(x), ldx, N,
                                                    row_vec(x, ldx, nullptr, 0, N), static_cast<int*>(out), 1, offset);
    DS_RET;
}

int ds_nucleus(void* stream, const void* a, int64_t lda, const void* b, int64_t ldb, int M, int N, const void* temp,
               const void* top_p, const void* u, int K, void* ids, void* probs, int64_t ldo, void* cnt, void* tok) {
    if (K < 1 || K > kNucCap || ldo < K || M < 1) return cudaErrorInvalidValue;
    nucleus_kernel<<<M, kNucThreads, 0, DS_ST>>>(
        static_cast<const __nv_bfloat16*>(a), lda, static_cast<const __nv_bfloat16*>(b), ldb, N, row_vec(a, lda, b, ldb, N),
        static_cast<const float*>(temp), static_cast<const float*>(top_p), static_cast<const float*>(u), K,
        static_cast<int*>(ids), static_cast<float*>(probs), ldo, static_cast<int*>(cnt), static_cast<int*>(tok));
    DS_RET;
}

int ds_nucleus_cap() { return kNucCap; }

// The blocks P pairs over S slots can take at bm pairs a block: sum over slots of
// ceil(count / bm) <= P / bm + (slots with a pair), and never more than P (a block holds >= 1).
// block_slot holds this many entries, block_pair and block_route this many times bm - a bound
// on P * 16 alone holds for bm 16 only (40 pairs over 24 slots take 24 blocks, 1,536 entries).
int ds_route_blocks(int P, int S, int bm) {
    if (P < 1 || S < 1 || bm < 1) return 0;
    return std::min(P, (P + bm - 1) / bm + std::min(S, P));
}

int ds_route_large(void* stream, const void* slots, int P, int topk, int S, void* work, void* block_slot, void* block_pair,
                   void* block_route, int bm) {
    if (P < 1 || S < 1 || P >= 65536 || (bm != 16 && bm != 32 && bm != 64)) return cudaErrorInvalidValue;
    int* w = static_cast<int*>(work);
    int *counts = w, *cursor = w + S, *base = w + 2 * S;
    const int blocks = ds_route_blocks(P, S, bm);
    const int64_t init = std::max<int64_t>((int64_t)blocks * bm, S);
    route_init_kernel<<<(unsigned)((init + 255) / 256), 256, 0, DS_ST>>>(blocks, bm, S, static_cast<int*>(block_slot),
        static_cast<int*>(block_pair), static_cast<int*>(block_route), counts, cursor);
    route_count_kernel<<<(P + 255) / 256, 256, 0, DS_ST>>>(static_cast<const int*>(slots), P, counts);
    route_scan_kernel<<<1, 1024, 0, DS_ST>>>(counts, S, bm, base, static_cast<int*>(block_slot));
    route_place_kernel<<<(P + 255) / 256, 256, 0, DS_ST>>>(static_cast<const int*>(slots), P, topk, bm, base, cursor,
        static_cast<int*>(block_pair), static_cast<int*>(block_route));
    DS_RET;
}

int ds_map_i32(void* stream, const void* table, void* x, int n) {
    map_i32_kernel<<<(n + 127) / 128, 128, 0, DS_ST>>>(static_cast<const int*>(table), static_cast<int*>(x), n);
    DS_RET;
}

int ds_argmax_combine(void* stream, const void* pairs, int world, int M, void* out) {
    argmax_combine_kernel<<<(M + 127) / 128, 128, 0, DS_ST>>>(static_cast<const int*>(pairs), world, M,
                                                              static_cast<int*>(out));
    DS_RET;
}

int ds_unshard_u16(void* stream, const void* src, void* dst, int M, int n, int world) {
    const int64_t total = (int64_t)world * M * n;
    unshard_u16_kernel<<<(unsigned)((total + 255) / 256), 256, 0, DS_ST>>>(static_cast<const uint16_t*>(src),
        static_cast<uint16_t*>(dst), M, n, world);
    DS_RET;
}

int ds_quant_fp8_block32(void* stream, const void* w, int N, int K, void* codes, void* scales) {
    const int nbk = (K + 31) / 32;
    quant_fp8_block32_kernel<<<dim3(nbk, (N + 31) / 32), 256, 0, DS_ST>>>(static_cast<const __nv_bfloat16*>(w), N, K,
        static_cast<uint8_t*>(codes), static_cast<uint8_t*>(scales), nbk);
    DS_RET;
}

int ds_moe_partial(void* stream, const void* parts, int K, int M, void* out, int D) {
    moe_partial_kernel<<<dim3((D + 255) / 256, M), 256, 0, DS_ST>>>(static_cast<const float*>(parts), K, M,
                                                                    static_cast<float*>(out), D);
    DS_RET;
}

int ds_moe_finish(void* stream, const void* routed, const void* shared, void* out, int M, int D) {
    moe_finish_kernel<<<dim3((D + 255) / 256, M), 256, 0, DS_ST>>>(static_cast<const float*>(routed),
        static_cast<const __nv_bfloat16*>(shared), static_cast<__nv_bfloat16*>(out), D);
    DS_RET;
}

int ds_moe_up_mma(void* stream, const void* x, int64_t ldx, const void* w1, const void* s1, const void* w3,
                  const void* s3, void* h, int64_t ldh, const void* route_weight, const void* block_slot,
                  const void* block_route, int nb, int N, int K, float limit, int null_slot, int bm) {
    constexpr int NG = kMoeNG;
    if (K % 128 || nb < 1 || (bm != 16 && bm != 32 && bm != 64)) return cudaErrorInvalidValue;
    dim3 grid(nb, (N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG));
#define DS_UP(...)                                                                                                     \
    __VA_ARGS__<<<grid, kThreads, 0, DS_ST>>>(                                                                        \
        static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const uint8_t*>(w1), static_cast<const uint8_t*>(s1),     \
        static_cast<const uint8_t*>(w3), static_cast<const uint8_t*>(s3), static_cast<__nv_bfloat16*>(h), ldh,          \
        static_cast<const float*>(route_weight), static_cast<const int*>(block_slot), static_cast<const int*>(block_route), \
        N, K, limit, null_slot)
    if (bm == 16) DS_UP(moe_up_mma_kernel<NG>);
    else if (bm == 32) DS_UP(moe_up_mma_kernel<NG, 2, 512, 4>);
    else DS_UP(moe_up_mma_kernel<NG, 4, 256, 2>);
#undef DS_UP
    DS_RET;
}

int ds_moe_prefill_up(void* stream, const void* x, int64_t ldx, const void* w1, const void* s1, const void* w3,
                      const void* s3, void* h, int64_t ldh, const void* route_weight, const void* block_slot,
                      const void* block_route, int nb, int N, int K, float limit, int null_slot) {
    if (K % kPmBK || nb < 1) return cudaErrorInvalidValue;
    moe_prefill_kernel<2><<<dim3((N + kPmBN - 1) / kPmBN, nb), 256, 0, DS_ST>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const uint8_t*>(w1), static_cast<const uint8_t*>(s1),
        static_cast<const uint8_t*>(w3), static_cast<const uint8_t*>(s3), h, ldh, static_cast<const float*>(route_weight),
        static_cast<const int*>(block_slot), static_cast<const int*>(block_route), N, K, limit, null_slot, 1, 1, 0, 1);
    DS_RET;
}

int ds_moe_prefill_down(void* stream, const void* h, int64_t ldh, const void* w2, const void* s2, void* y, int64_t ldy,
                        const void* block_slot, const void* block_route, int nb, int N, int K, int topk, int ntok,
                        int null_slot, int partial, int parts_world) {
    if (K % kPmBK || nb < 1 || parts_world < 1 || N % parts_world) return cudaErrorInvalidValue;
    moe_prefill_kernel<1><<<dim3((N + kPmBN - 1) / kPmBN, nb), 256, 0, DS_ST>>>(
        static_cast<const __nv_bfloat16*>(h), ldh, static_cast<const uint8_t*>(w2), static_cast<const uint8_t*>(s2), nullptr,
        nullptr, y, ldy, nullptr, static_cast<const int*>(block_slot), static_cast<const int*>(block_route), N, K, 0.f,
        null_slot, topk, ntok, partial, parts_world);
    DS_RET;
}

int ds_moe_down_mma(void* stream, const void* h, int64_t ldh, const void* w2, const void* s2, void* y, int64_t ldy,
                    const void* block_slot, const void* block_pair, int nb, int N, int K, int topk, int ntok,
                    int null_slot, int partial, int parts_world, int bm) {
    constexpr int NG = kMoeNG;
    if (K % 128 || nb < 1 || parts_world < 1 || N % parts_world || (bm != 16 && bm != 32 && bm != 64)) return cudaErrorInvalidValue;
    dim3 grid(nb, (N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG));
#define DS_DOWN(...)                                                                                                   \
    __VA_ARGS__<<<grid, kThreads, 0, DS_ST>>>(                                                                        \
        static_cast<const __nv_bfloat16*>(h), ldh, static_cast<const uint8_t*>(w2), static_cast<const uint8_t*>(s2),     \
        static_cast<float*>(y), ldy, static_cast<const int*>(block_slot), static_cast<const int*>(block_pair), N, K,      \
        topk, ntok, null_slot, partial, parts_world)
    if (bm == 16) DS_DOWN(moe_down_mma_kernel<NG>);
    else if (bm == 32) DS_DOWN(moe_down_mma_kernel<NG, 2, 512, 4>);
    else DS_DOWN(moe_down_mma_kernel<NG, 4, 256, 2>);
#undef DS_DOWN
    DS_RET;
}

int ds_mma_fp8(void* stream, const void* x, int64_t ldx, int64_t x_group_stride, int rows_per_group, const void* w,
               const void* s, int64_t lds, int M, int N, int K, int splits, void* y_bf16, void* y_f32, int64_t ldy,
               void* partial) {
    return launch_mma<0>(static_cast<cudaStream_t>(stream), x, ldx, x_group_stride, rows_per_group, w, s, lds, M, N, K,
                         splits, y_bf16, y_f32, ldy, partial);
}

int ds_mma_bf16(void* stream, const void* x, int64_t ldx, const void* w, int M, int N, int K, int splits, void* y_bf16,
                void* y_f32, int64_t ldy, void* partial) {
    return launch_mma<1>(static_cast<cudaStream_t>(stream), x, ldx, 0, 0, w, nullptr, 0, M, N, K, splits, y_bf16, y_f32,
                         ldy, partial);
}

}  // extern "C"
