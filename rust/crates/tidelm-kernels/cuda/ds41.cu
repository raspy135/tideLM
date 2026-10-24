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
// ds_splits_reduce adds them in split order (deterministic).

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
    const __nv_bfloat16* xg =
        x + (rows_per_group > 0 ? (int64_t)(n_block / rows_per_group) * x_group_stride : 0);

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
            if (m < M && j < kc) v = *reinterpret_cast<const uint4*>(xg + (int64_t)m * ldx + k0 + j);
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
            if (m >= M) continue;
            if (gridDim.y > 1) {
                partial[((int64_t)blockIdx.y * M + m) * N + n] = acc[r][m];
            } else if (ybf) {
                ybf[(int64_t)m * ldy + n] = to_bf(acc[r][m]);
            } else {
                yf[(int64_t)m * ldy + n] = acc[r][m];
            }
        }
    }
}

template <int KIND>
int launch_skinny(cudaStream_t st, const void* x, int64_t ldx, int64_t gstride, int rpg,
                  const void* w, const void* s, int64_t lds, int M, int N, int K, int splits,
                  void* ybf, void* yf, int64_t ldy, void* partial) {
    constexpr int ROWS = 4;
    if (M < 1 || M > 16 || splits < 1 || K % 32 || (splits > 1 && !partial)) return cudaErrorInvalidValue;
    // slice length: a multiple of 32 so slices never split a scale block
    const int ksplit = ((K + splits - 1) / splits + 31) / 32 * 32;
    dim3 grid((N + kWarps * ROWS - 1) / (kWarps * ROWS), splits);
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
    const __nv_bfloat16* xg =
        x + (rows_per_group > 0 ? (int64_t)(n_block / rows_per_group) * x_group_stride : 0);

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
            if (m < M && j < kc) v = *reinterpret_cast<const uint32_t*>(xg + (int64_t)m * ldx + k0 + j);
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
                if (m >= M) continue;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int n = n0 + g * 8 + q * 2 + j;
                    if (n >= N) continue;
                    const float v = acc[g][mt][half * 2 + j];
                    if (gridDim.y > 1) partial[((int64_t)blockIdx.y * M + m) * N + n] = v;
                    else if (ybf) ybf[(int64_t)m * ldy + n] = to_bf(v);
                    else yf[(int64_t)m * ldy + n] = v;
                }
            }
}

template <int KIND>
int launch_mma(cudaStream_t st, const void* x, int64_t ldx, int64_t gstride, int rpg, const void* w, const void* s,
               int64_t lds, int M, int N, int K, int splits, void* ybf, void* yf, int64_t ldy, void* partial) {
    if (M < 1 || M > 64 || splits < 1 || K % 64 || (splits > 1 && !partial)) return cudaErrorInvalidValue;
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
        dim3 grid((N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG), splits);                                 \
        mma_skinny_kernel<KIND, MT, KC, NG, DEPTH><<<grid, kThreads, 0, st>>>(xb, ldx, gstride, rpg, wb, sb, lds, M, N, \
                                                                              K, ksplit, yb, yff, ldy, pp); \
    }
    // (NG, DEPTH) per tile, from examples/tune_mma.rs on the decode shapes (2026-10-24, GB/s
    // of weight bytes, split-K to >= 2 blocks/SM): M=32 2x2 avg ~165 (1x4 157, 4x1 154);
    // M=64 1x4 avg ~133 (2x4 126, 4x1 121, 4x2 106). Neither reaches the ~230 GB/s the FMA
    // kernel streams at M <= 16 - this kernel exists for M > 16, where the FMA one cannot
    // hold its accumulators, and a deeper pipeline (cp.async, ldmatrix) is the next step.
    if (M <= 16) DS_MMA(16, 512, 1, 4)
    else if (M <= 32) DS_MMA(32, 512, 2, 2)
    else DS_MMA(64, 256, 1, 4)
#undef DS_MMA
    return static_cast<int>(cudaGetLastError());
}

/// Rows per block of the tensor-core GEMM at M rows (callers size split-K with it).
extern "C" int ds_mma_rows_per_block(int M) { return kWarps * 8 * (M <= 16 ? 1 : M <= 32 ? 2 : 1); }

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
template <int KC>
__device__ __forceinline__ void stage_rows_f16(__half* xs, const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                               const int* rows, int k0, int kc) {
    constexpr int XS = KC + 8;
    for (int i = threadIdx.x; i < 16 * (KC / 2); i += blockDim.x) {
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
template <int NW, int NG, int KC, int NB>
__device__ __forceinline__ void mma_fp4_chunk(float (*acc)[NG][4], const __half* xs, const uint8_t* const* w,
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
            float c[NW][NG][4];
#pragma unroll
            for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                for (int g = 0; g < NG; ++g)
#pragma unroll
                    for (int i = 0; i < 4; ++i) c[wi][g][i] = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // word j: this lane's values 8j .. 8j+8 of its group = steps 2j and 2j+1
                uint32_t a[2][4];
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    const int kx = kb + q * 32 + j * 8 + st * 4;
                    const uint2 r0 = *reinterpret_cast<const uint2*>(&xs[r * XS + kx]);
                    const uint2 r1 = *reinterpret_cast<const uint2*>(&xs[(r + 8) * XS + kx]);
                    a[st][0] = r0.x; a[st][1] = r1.x; a[st][2] = r0.y; a[st][3] = r1.y;
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
                        mma_f16(c[wi][g], a[0][0], a[0][1], a[0][2], a[0][3], b[0], b[1]);
                        mma_f16(c[wi][g], a[1][0], a[1][1], a[1][2], a[1][3], b[2], b[3]);
                    }
            }
            // the C fragment's columns are rows 2q and 2q+1 of each 8-row group
#pragma unroll
            for (int wi = 0; wi < NW; ++wi)
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const float fa = e8m0_lin(__shfl_sync(0xffffffffu, sref[bi][wi][g], (2 * q) * 4));
                    const float fb = e8m0_lin(__shfl_sync(0xffffffffu, sref[bi][wi][g], (2 * q + 1) * 4));
                    acc[wi][g][0] += c[wi][g][0] * fa;
                    acc[wi][g][1] += c[wi][g][1] * fb;
                    acc[wi][g][2] += c[wi][g][2] * fa;
                    acc[wi][g][3] += c[wi][g][3] * fb;
                }
        }
    }
}

template <int NG>
__global__ void __launch_bounds__(256) moe_up_mma_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ s1,
    const uint8_t* __restrict__ w3, const uint8_t* __restrict__ s3, __nv_bfloat16* __restrict__ h, int64_t ldh,
    const float* __restrict__ route_weight, const int* __restrict__ block_slot, const int* __restrict__ block_route,
    int N, int K, float limit, int null_slot) {
    constexpr int KC = kMoeKC;
    __shared__ __align__(16) __half xs[16 * (KC + 8)];
    __shared__ int tok_s[16], pair_s[16];
    const int slot = block_slot[blockIdx.x];
    if (slot < 0 || slot == null_slot) return;
    if (threadIdx.x < 16) {
        const int route = block_route[(int64_t)blockIdx.x * 16 + threadIdx.x];
        tok_s[threadIdx.x] = route < 0 ? -1 : route >> 16;
        pair_s[threadIdx.x] = route < 0 ? -1 : route & 0xffff;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int n0 = blockIdx.y * (kWarps * 8 * NG) + warp * 8 * NG;
    const uint8_t* w[2] = {w1 + (int64_t)slot * N * (K >> 1), w3 + (int64_t)slot * N * (K >> 1)};
    const uint8_t* sc[2] = {s1 + (int64_t)slot * N * (K >> 5), s3 + (int64_t)slot * N * (K >> 5)};
    float acc[2][NG][4];
#pragma unroll
    for (int wi = 0; wi < 2; ++wi)
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[wi][g][i] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kc = min(KC, K - k0);
        __syncthreads();
        stage_rows_f16<KC>(xs, x, ldx, tok_s, k0, kc);
        __syncthreads();
        mma_fp4_chunk<2, NG, KC, kMoeNB>(acc, xs, w, sc, N, K, n0, k0, kc);
    }
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int pair = pair_s[r + half * 8];
            if (pair < 0) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = n0 + g * 8 + q * 2 + j;
                if (n >= N) continue;
                float gt = bf(to_bf(acc[0][g][half * 2 + j]));
                float up = bf(to_bf(acc[1][g][half * 2 + j]));
                gt = fminf(gt, limit);
                up = fminf(fmaxf(up, -limit), limit);
                h[(int64_t)pair * ldh + n] = to_bf(silu_triton(gt) * up * route_weight[pair]);
            }
        }
}

template <int NG>
__global__ void __launch_bounds__(256) moe_down_mma_kernel(
    const __nv_bfloat16* __restrict__ h, int64_t ldh, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    float* __restrict__ y, int64_t ldy, const int* __restrict__ block_slot, const int* __restrict__ block_pair,
    int N, int K, int topk, int ntok, int null_slot, int partial, int parts_world) {
    constexpr int KC = kMoeKC;
    __shared__ __align__(16) __half xs[16 * (KC + 8)];
    __shared__ int pair_s[16];
    const int slot = block_slot[blockIdx.x];
    if (slot < 0) return;
    const bool real = slot != null_slot;
    if (threadIdx.x < 16) pair_s[threadIdx.x] = block_pair[(int64_t)blockIdx.x * 16 + threadIdx.x];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int n0 = blockIdx.y * (kWarps * 8 * NG) + warp * 8 * NG;
    const uint8_t* w[1] = {w2 + (int64_t)slot * N * (K >> 1)};
    const uint8_t* sc[1] = {s2 + (int64_t)slot * N * (K >> 5)};
    float acc[1][NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[0][g][i] = 0.f;
    if (real) {
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = min(KC, K - k0);
            __syncthreads();
            stage_rows_f16<KC>(xs, h, ldh, pair_s, k0, kc);
            __syncthreads();
            mma_fp4_chunk<1, NG, KC, kMoeNB>(acc, xs, w, sc, N, K, n0, k0, kc);
        }
    } else {
        __syncthreads();
    }
    const int local_n = N / parts_world;
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int pair = pair_s[r + half * 8];
            if (pair < 0) continue;
            const int64_t out_row = (int64_t)(pair % topk) * ntok + pair / topk;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = n0 + g * 8 + q * 2 + j;
                if (n >= N) continue;
                float v = real ? acc[0][g][half * 2 + j] : 0.f;
                if (!partial) v = bf(to_bf(v));
                const int64_t off = parts_world > 1
                    ? ((int64_t)(n / local_n) * topk * ntok + out_row) * local_n + n % local_n
                    : out_row * ldy + n;
                y[off] = v;
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
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = bf(xr[i]);
        ss = fmaf(v, v, ss);
    }
    const float t = block_sum(ss, scratch);
    if (threadIdx.x == 0) out[blockIdx.x] = rsqrtf(t * (1.f / (float)D) + eps);
}

// ------------------------------------------------------------------ hyper-connections
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// v41_ref.hc_split_sinkhorn on mixes * rsqrt (v41_ref.hc_mixes). One thread per row; hc = 4.
__global__ void hc_sinkhorn_kernel(const float* __restrict__ mixes, const float* __restrict__ rsq,
                                   const float* __restrict__ scale, const float* __restrict__ base,
                                   int M, int iters, float eps, float* __restrict__ pre,
                                   float* __restrict__ post, float* __restrict__ comb) {
    constexpr int HC = 4;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float mx[(2 + HC) * HC];
    for (int i = 0; i < (2 + HC) * HC; ++i) mx[i] = mixes[m * (2 + HC) * HC + i] * rsq[m];
    for (int i = 0; i < HC; ++i) {
        pre[m * HC + i] = sigmoidf_(mx[i] * scale[0] + base[i]) + eps;
        post[m * HC + i] = 2.f * sigmoidf_(mx[HC + i] * scale[1] + base[HC + i]);
    }
    float c[HC][HC];
    for (int i = 0; i < HC; ++i) {
        float mxr = -INFINITY;
        for (int j = 0; j < HC; ++j) {
            c[i][j] = mx[2 * HC + i * HC + j] * scale[2] + base[2 * HC + i * HC + j];
            mxr = fmaxf(mxr, c[i][j]);
        }
        float sum = 0.f;
        for (int j = 0; j < HC; ++j) { c[i][j] = expf(c[i][j] - mxr); sum += c[i][j]; }
        for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] / sum + eps;
    }
    // column normalize (sum over the first index), then iters-1 x (row, column)
    for (int j = 0; j < HC; ++j) {
        float s = 0.f;
        for (int i = 0; i < HC; ++i) s += c[i][j];
        for (int i = 0; i < HC; ++i) c[i][j] = c[i][j] / (s + eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
        for (int i = 0; i < HC; ++i) {
            float s = 0.f;
            for (int j = 0; j < HC; ++j) s += c[i][j];
            for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] / (s + eps);
        }
        for (int j = 0; j < HC; ++j) {
            float s = 0.f;
            for (int i = 0; i < HC; ++i) s += c[i][j];
            for (int i = 0; i < HC; ++i) c[i][j] = c[i][j] / (s + eps);
        }
    }
    for (int i = 0; i < HC; ++i)
        for (int j = 0; j < HC; ++j) comb[(m * HC + i) * HC + j] = c[i][j];
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
__global__ void router_kernel(const float* __restrict__ z, const float* __restrict__ bias,
                              const uint8_t* __restrict__ keep, int E, int K, float route_scale,
                              int* __restrict__ idx_out, float* __restrict__ w_out) {
    extern __shared__ float sh[];
    float* score = sh;      // [E]
    float* sel = sh + E;    // [E]
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int m = blockIdx.x;
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

// ------------------------------------------------------------------ attention (decode)
// model.py::attention + _softmax_attn for decode rows: keys are the layer's window ring
// (positions pos-W+1 .. pos, oldest first, masked below `win_lo`) followed by `C` selected
// compressed rows (cidx, -1 = none). K and V are the same vectors. Scores and PV in fp32:
//   s = (q . k) * scale; mx = max(max s, -1e30); p = exp(s - mx);
//   o = sum_n (p_n / (sum p + exp(sink - mx))) * k_n
// One block per (row, group of HG heads).
constexpr int kAttnHG = 4;

// Batched rows: row m reads the caches of lane[m] (lane == nullptr: lane 0), each lane's
// ring / compressed rows `*_lstride` elements after the previous lane's.
__global__ void __launch_bounds__(256) attn_decode_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ring_base, int ring_size,
    const int* __restrict__ pos, int W, int win_lo, const __nv_bfloat16* __restrict__ ckv_base,
    const int* __restrict__ cidx, int C, const float* __restrict__ sink, float scale,
    __nv_bfloat16* __restrict__ out, int H, const int* __restrict__ row_lane, int64_t ring_lstride,
    int64_t ckv_lstride) {
    constexpr int D = 512;
    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                    // [HG][D]
    float* sc = smem + kAttnHG * D;      // [HG][W + C]
    __shared__ float scratch[32];
    const int m = blockIdx.y;
    const int h0 = blockIdx.x * kAttnHG;
    const int NK = W + C;
    const int p = pos[m];
    const int ln = row_lane ? row_lane[m] : 0;
    const __nv_bfloat16* ring = ring_base + (int64_t)ln * ring_lstride;
    const __nv_bfloat16* ckv = ckv_base ? ckv_base + (int64_t)ln * ckv_lstride : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    for (int i = threadIdx.x; i < kAttnHG * D; i += blockDim.x)
        qs[i] = bf(q[((int64_t)m * H + h0 + i / D) * D + i % D]);
    __syncthreads();

    auto key_row = [&](int n) -> const __nv_bfloat16* {
        if (n < W) {
            const int kp = p - (W - 1 - n);
            if (kp < 0 || kp < win_lo) return nullptr;
            return ring + (int64_t)(kp % ring_size) * D;
        }
        const int j = cidx[(int64_t)m * C + (n - W)];
        return j < 0 ? nullptr : ckv + (int64_t)j * D;
    };

    for (int n = warp; n < NK; n += nw) {
        const __nv_bfloat16* kr = key_row(n);
        float dots[kAttnHG];
        if (kr) {
            float kv[16];
            bf16x8_to_f32(*reinterpret_cast<const uint4*>(kr + lane * 16), kv);
            bf16x8_to_f32(*reinterpret_cast<const uint4*>(kr + lane * 16 + 8), kv + 8);
#pragma unroll
            for (int g = 0; g < kAttnHG; ++g) {
                float acc = 0.f;
#pragma unroll
                for (int v = 0; v < 16; ++v) acc = fmaf(qs[g * D + lane * 16 + v], kv[v], acc);
                dots[g] = warp_sum(acc);
            }
        }
        if (lane == 0)
            for (int g = 0; g < kAttnHG; ++g) sc[g * NK + n] = kr ? dots[g] * scale : -INFINITY;
    }
    __syncthreads();

    for (int g = 0; g < kAttnHG; ++g) {
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
    float acc[kAttnHG][2];
    for (int g = 0; g < kAttnHG; ++g) acc[g][0] = acc[g][1] = 0.f;
    const int d = threadIdx.x * 2;
    for (int n = 0; n < NK; ++n) {
        const __nv_bfloat16* kr = key_row(n);
        if (!kr) continue;
        const __nv_bfloat162 kv2 = *reinterpret_cast<const __nv_bfloat162*>(kr + d);
        const float k0 = __low2float(kv2), k1 = __high2float(kv2);
        for (int g = 0; g < kAttnHG; ++g) {
            const float pn = sc[g * NK + n];
            acc[g][0] = fmaf(pn, k0, acc[g][0]);
            acc[g][1] = fmaf(pn, k1, acc[g][1]);
        }
    }
    if (d < D)
        for (int g = 0; g < kAttnHG; ++g) {
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
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { v[i] = bf(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, 6.f * 0.001953125f);  // clamp_min(6 * 2^-9)
    const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(amax / 6.f, __NV_NOSAT, __NV_E4M3);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(s8, __NV_E4M3);
    const float s = __half2float(*reinterpret_cast<const __half*>(&hr));
    for (int i = 0; i < 16; ++i) {
        float t = fminf(fmaxf(v[i] / s, -6.f), 6.f);
        const float a = fabsf(t);
        // torch.bucketize(|t|, midpoints) with right=False: a value equal to a midpoint
        // takes the lower grid point
        float q;
        if (a <= 0.25f) q = 0.f;
        else if (a <= 0.75f) q = 0.5f;
        else if (a <= 1.25f) q = 1.f;
        else if (a <= 1.75f) q = 1.5f;
        else if (a <= 2.5f) q = 2.f;
        else if (a <= 3.5f) q = 3.f;
        else if (a <= 5.f) q = 4.f;
        else q = 6.f;
        const float sign = t > 0.f ? 1.f : (t < 0.f ? -1.f : 0.f);
        p[i] = to_bf(q * sign * s);
    }
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
                                   const float* __restrict__ weight, int D, int hc, float eps, float dim_scale) {
    __shared__ float scratch[32];
    const int m = blockIdx.x / hc, i = blockIdx.x % hc;
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

// First index of the maximum (torch.argmax), over bf16 logits.
__global__ void argmax_bf16_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int N,
                                   int* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * ldx;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const float v = bf(xr[n]);
        if (v > best || (v == best && n < besti)) { best = v; besti = n; }
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
        best = bv[0];
        besti = bi[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
            if (bv[i] > best || (bv[i] == best && bi[i] < besti)) { best = bv[i]; besti = bi[i]; }
        out[blockIdx.x] = besti;
    }
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
// compaction that emits the selected indices already in ascending order.
constexpr int kSelThreads = 1024;

__global__ void __launch_bounds__(kSelThreads) topk_select_kernel(
    const __nv_bfloat16* __restrict__ s, int64_t ld_s, int n, int K, int* __restrict__ out, int width) {
    __shared__ int hist[256];
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
    for (int i = lo; i < hi; ++i) { uint32_t key; finite += bf16_key(row[i], key); }
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
            for (int b = tid; b < 256; b += kSelThreads) hist[b] = 0;
            __syncthreads();
            for (int i = lo; i < hi; ++i) {
                uint32_t key;
                if (!bf16_key(row[i], key)) continue;
                if (pass == 1 && (key >> 8) != (prefix >> 8)) continue;
                atomicAdd(&hist[(key >> shift) & 0xff], 1);
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
__global__ void argmax_sum_bf16_kernel(const __nv_bfloat16* __restrict__ a, int64_t lda,
                                       const __nv_bfloat16* __restrict__ b, int64_t ldb, int N,
                                       int* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* ar = a + (int64_t)blockIdx.x * lda;
    const __nv_bfloat16* br = b ? b + (int64_t)blockIdx.x * ldb : nullptr;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const float v = bf(ar[n]) + (br ? bf(br[n]) : 0.f);
        if (v > best || (v == best && n < besti)) { best = v; besti = n; }
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
        best = bv[0];
        besti = bi[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); ++i)
            if (bv[i] > best || (bv[i] == best && bi[i] < besti)) { best = bv[i]; besti = bi[i]; }
        out[blockIdx.x] = besti;
    }
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
    for (int i = threadIdx.x; i < T * D; i += blockDim.x) {
        mem_kv[(int64_t)L * mem_rows * D + i] = kv[(int64_t)f * D + i];
        mem_sc[(int64_t)L * mem_rows * D + i] = sc[(int64_t)f * D + i];
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
        mem_start[L] = pos[f];
    }
}

// model.py::Caches.rollback for a ratio-2 source, per lane: block b restores lane lanes[b]
// to length n_dev[b] (read from the device: the accepted length of its verify block).
__global__ void rollback_pending_kernel(const int* __restrict__ n_dev, const int* __restrict__ lanes,
                                        const float* __restrict__ mem_kv, const float* __restrict__ mem_sc,
                                        const int* __restrict__ mem_start, const float* __restrict__ before_kv,
                                        const float* __restrict__ before_sc, const int* __restrict__ before_flag,
                                        float* pkv, float* psc, int* flag, int D, int mem_rows) {
    const int n = n_dev[blockIdx.x];
    const int L = lanes ? lanes[blockIdx.x] : 0;
    if (n % 2 == 0) {
        if (threadIdx.x == 0) flag[L] = 0;
        return;
    }
    const int p = n - 1, start = mem_start[L];
    // a rollback reaches at most into the last chunk; anything else is a caller bug and
    // must not read past the chunk memory
    if (p - start >= mem_rows || p + 1 < start) return;
    float* pk = pkv + (int64_t)L * D;
    float* ps = psc + (int64_t)L * D;
    if (p >= start) {
        const int64_t row = (int64_t)L * mem_rows * D + (int64_t)(p - start) * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) { pk[d] = mem_kv[row + d]; ps[d] = mem_sc[row + d]; }
        if (threadIdx.x == 0) flag[L] = 1;
    } else {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            pk[d] = before_kv[(int64_t)L * D + d];
            ps[d] = before_sc[(int64_t)L * D + d];
        }
        if (threadIdx.x == 0) flag[L] = before_flag[L];
    }
}

// Routed expert ids -> arena slots through the device LUT (model.py: lut[L][indices]).
// Pairs at or past *live (null: none) belong to a padded step's filler rows and get slot -1,
// which the MoE kernels skip: filler rows must not pull experts through the bandwidth.
__global__ void lut_slots_kernel(const int* __restrict__ idx, const int* __restrict__ lut, int P,
                                 int* __restrict__ slots, const int* __restrict__ live) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) slots[p] = (live && p >= *live) ? -1 : lut[idx[p]];
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
    hc_pre_norm_kernel<<<M, 256, D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<const float*>(pre),
        static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), D, hc, eps);
    return static_cast<int>(cudaGetLastError());
}

int ds_row_rsqrt(void* stream, const void* x, int64_t ldx, int M, int D, float eps, void* out) {
    row_rsqrt_kernel<<<M, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, D, eps, static_cast<float*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ds_hc_sinkhorn(void* stream, const void* mixes, const void* rsq, const void* scale, const void* base,
                   int M, int iters, float eps, void* pre, void* post, void* comb) {
    hc_sinkhorn_kernel<<<(M + 31) / 32, 32, 0, static_cast<cudaStream_t>(stream)>>>(
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
              float route_scale, void* idx, void* weights) {
    if (K > 8) return cudaErrorInvalidValue;
    router_kernel<<<M, 128, 2 * E * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(z), static_cast<const float*>(bias), static_cast<const uint8_t*>(keep), E, K,
        route_scale, static_cast<int*>(idx), static_cast<float*>(weights));
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

int ds_attn_decode(void* stream, const void* q, const void* ring, int ring_size, const void* pos, int W,
                   int win_lo, const void* ckv, const void* cidx, int C, const void* sink, float scale, void* out,
                   int M, int H, const void* lane, int64_t ring_lstride, int64_t ckv_lstride) {
    if (H % kAttnHG) return cudaErrorInvalidValue;
    const size_t smem = (size_t)kAttnHG * (512 + W + C) * sizeof(float);
    static size_t configured = 0;
    if (smem > 48 * 1024 && smem > configured) {
        const cudaError_t e = cudaFuncSetAttribute(attn_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem);
        if (e != cudaSuccess) return static_cast<int>(e);
        configured = smem;
    }
    dim3 grid(H / kAttnHG, M);
    attn_decode_kernel<<<grid, 256, smem, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(ring), ring_size,
        static_cast<const int*>(pos), W, win_lo, static_cast<const __nv_bfloat16*>(ckv),
        static_cast<const int*>(cidx), C, static_cast<const float*>(sink), scale,
        static_cast<__nv_bfloat16*>(out), H, static_cast<const int*>(lane), ring_lstride, ckv_lstride);
    return static_cast<int>(cudaGetLastError());
}

int ds_compress_pair(void* stream, const void* kv_a, const void* sc_a, const void* kv_b, const void* sc_b,
                     void* out, int D) {
    compress_pair_kernel<<<(D + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(kv_a), static_cast<const float*>(sc_a), static_cast<const float*>(kv_b),
        static_cast<const float*>(sc_b), static_cast<__nv_bfloat16*>(out), D);
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
                   float dim_scale) {
    engram_gate_kernel<<<M * hc, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(kv), static_cast<const float*>(weight),
        D, hc, eps, dim_scale);
    return static_cast<int>(cudaGetLastError());
}

int ds_embed_hc(void* stream, const void* ids, const void* table, void* h, int M, int D, int hc) {
    dim3 grid((D + 255) / 256, M);
    embed_hc_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(ids), static_cast<const __nv_bfloat16*>(table), static_cast<__nv_bfloat16*>(h),
        D, hc);
    return static_cast<int>(cudaGetLastError());
}

int ds_argmax_bf16(void* stream, const void* x, int64_t ldx, int M, int N, void* out) {
    argmax_bf16_kernel<<<M, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, N, static_cast<int*>(out));
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
    argmax_sum_bf16_kernel<<<M, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(a), lda, static_cast<const __nv_bfloat16*>(b), ldb, N, static_cast<int*>(out));
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

int ds_greedy_accept(void* stream, const void* am, const void* block, int T, const void* pos, void* out) {
    greedy_accept_kernel<<<1, 32, 0, DS_ST>>>(static_cast<const int*>(am), static_cast<const int*>(block), T,
        static_cast<const int*>(pos), static_cast<int*>(out));
    DS_RET;
}

int ds_wait_flag(void* stream, const void* flag, int expect, long long timeout_ns, void* err) {
    wait_flag_kernel<<<1, 32, 0, DS_ST>>>(static_cast<const volatile int*>(flag), expect, timeout_ns, static_cast<int*>(err));
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
                  const void* block_route, int nb, int N, int K, float limit, int null_slot) {
    constexpr int NG = kMoeNG;
    if (K % 128 || nb < 1) return cudaErrorInvalidValue;
    dim3 grid(nb, (N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG));
    moe_up_mma_kernel<NG><<<grid, kThreads, 0, DS_ST>>>(
        static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const uint8_t*>(w1), static_cast<const uint8_t*>(s1),
        static_cast<const uint8_t*>(w3), static_cast<const uint8_t*>(s3), static_cast<__nv_bfloat16*>(h), ldh,
        static_cast<const float*>(route_weight), static_cast<const int*>(block_slot), static_cast<const int*>(block_route),
        N, K, limit, null_slot);
    DS_RET;
}

int ds_moe_down_mma(void* stream, const void* h, int64_t ldh, const void* w2, const void* s2, void* y, int64_t ldy,
                    const void* block_slot, const void* block_pair, int nb, int N, int K, int topk, int ntok,
                    int null_slot, int partial, int parts_world) {
    constexpr int NG = kMoeNG;
    if (K % 128 || nb < 1 || parts_world < 1 || N % parts_world) return cudaErrorInvalidValue;
    dim3 grid(nb, (N + kWarps * 8 * NG - 1) / (kWarps * 8 * NG));
    moe_down_mma_kernel<NG><<<grid, kThreads, 0, DS_ST>>>(
        static_cast<const __nv_bfloat16*>(h), ldh, static_cast<const uint8_t*>(w2), static_cast<const uint8_t*>(s2),
        static_cast<float*>(y), ldy, static_cast<const int*>(block_slot), static_cast<const int*>(block_pair), N, K,
        topk, ntok, null_slot, partial, parts_world);
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
