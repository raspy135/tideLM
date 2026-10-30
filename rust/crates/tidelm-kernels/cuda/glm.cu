// GLM-5.3 kernels for the Rust engine: engine/glm's Triton kernels and PyTorch ops in CUDA C++.
//
// Each kernel names the Python it replaces and keeps its rounding points (BF16 activations
// between ops, fp32 accumulation, the same epsilons); summation orders inside a reduction may
// differ. The EXL3 trellis kernels are tools/exl3_moe_cuda.cu, compiled as-is (src/exl3.rs).
//
// C ABI: stream first, cudaGetLastError() returned as int, no allocation, graph-capturable.
// Positions come from device arrays, so a captured decode graph replays at any position.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <utility>

// Programmatic dependent launch, as ds41.cu's: every kernel here (but the L2 prefetch, launched on
// its own branch) starts with pdl_enter(), and once the GLM engine switches it on (glm_set_pdl;
// TIDELM_PDL=0 keeps it off) every launch carries the programmatic-serialization attribute, so the
// next kernel's blocks launch while this one drains and wait on the GPU, not behind the launch
// path. griddepcontrol.wait returns once the previous grid has completed and its writes are
// visible, and it precedes any global access in every block: the stream's order holds,
// transitively. Without the attribute both instructions do nothing. Bit-identical either way.
__device__ __forceinline__ void pdl_enter() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
static int g_pdl = 0;
template <typename... P, typename... A>
static inline void pdl_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, A&&... args) {
    cudaLaunchConfig_t c = {};
    c.gridDim = grid; c.blockDim = block; c.dynamicSmemBytes = smem; c.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = at; c.numAttrs = g_pdl ? 1 : 0;
    cudaLaunchKernelEx(&c, kernel, std::forward<A>(args)...);
}
// the GLM engine's switch for this file's launches (graph captures made after it carry it)
extern "C" int glm_set_pdl(int on) { g_pdl = on; return 0; }

// Reads made once a decode pass, loaded evict-first (__ldcs) when their bit is set
// (glm_set_stream_loads): 1 kv_b's FP8 rows in the absorbs (the GLM engine's default). Measured and
// left plain: the MLA's latent and RoPE key rows (a row's two head blocks read the same picks: 1 /
// 4 rows +2.9 / +1.6% at 2.5K, +0.6 / +0.1% at 120K) and the indexer's key rows (-0.3 / -0.1% at
// 120K), in-boot A/B 2026-10-30.
__constant__ int c_stream_loads = 0;
extern "C" int glm_set_stream_loads(int mask) { return (int)cudaMemcpyToSymbol(c_stream_loads, &mask, sizeof(int)); }
template <typename T>
__device__ __forceinline__ T ld_once(const T* p, int bit) {
    return (c_stream_loads & bit) ? __ldcs(p) : *p;
}

namespace {

#define GST static_cast<cudaStream_t>(stream)
#define GRET return static_cast<int>(cudaGetLastError())

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }
// fp32 -> the nearest BF16 value, still fp32 (an intermediate rounding PyTorch's BF16 ops make)
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

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
// fixed-tree block sum, every thread gets the result
__device__ float block_sum(float v, float* scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float t = lane < nw ? scratch[lane] : 0.f;
    return warp_sum(t);
}

__device__ __forceinline__ float e4m3(uint8_t b) {
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
}
__device__ __forceinline__ uint8_t to_e4m3(float v) { return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }

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
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t& b0, uint32_t& b1, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(a));
}

// 16 FP8 E4M3 values (one 16-byte load) as 16 BF16 (exact: 3 mantissa bits), times `s` rounded
// to BF16 when s_bf16 (the latent's kv_read: bf16(bf16(fp8) * bf16(scale))).
__device__ __forceinline__ void fp8x16_bf16(uint4 raw, float s, bool scaled, uint4* out) {
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = e4m3((w[i] >> (8 * j)) & 0xff);
            if (scaled) v[j] = v[j] * s;
        }
        o[2 * i] = pack_bf16(v[0], v[1]);
        o[2 * i + 1] = pack_bf16(v[2], v[3]);
    }
    out[0] = make_uint4(o[0], o[1], o[2], o[3]);
    out[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// Orderable key of an fp32 value: larger value -> larger key.
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A split cache's row for position p (kv_split.owner_rows) and a gathered pair's row: rank
// (p % 2)'s rows come first in a gather of both ranks' first n rows.
__device__ __forceinline__ int64_t key_row(int p, int split_n) {
    return split_n ? (int64_t)(p & 1) * split_n + (p >> 1) : (int64_t)p;
}

// Block-wide exclusive scan of a 0/1 flag (1024 threads max): (rank of this thread's flag among
// the block's set flags in thread order, total set). Two barriers.
__device__ __forceinline__ int block_rank(bool f, int* wtot, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    const unsigned b = __ballot_sync(0xffffffffu, f);
    const int in_warp = __popc(b & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) wtot[warp] = __popc(b);
    __syncthreads();
    int before = 0, tot = 0;
    for (int w = 0; w < nw; ++w) {
        const int c = wtot[w];
        before += w < warp ? c : 0;
        tot += c;
    }
    total = tot;
    return before + in_warp;
}

// ------------------------------------------------------------------ embedding / sums
// tp.py embed_ids with GLM_EMBED_8BIT=int8: this rank's vocabulary rows [lo, lo + vl) as INT8 with
// an fp32 scale per 128 values, bf16(q * s), zero for the other rank's ids (an fp32 all-reduce
// then sums the pair exactly). pos (nullable): zero rows at position 0 (the MTP layer's input).
__global__ void embed_kernel(const int* __restrict__ ids, const int8_t* __restrict__ q, const float* __restrict__ sc,
                             int lo, int vl, int H, const int* __restrict__ pos, float* __restrict__ out) {
    pdl_enter();
    const int t = blockIdx.x;
    const int id = ids[t] - lo;
    const bool in = id >= 0 && id < vl && !(pos && pos[t] == 0);
    float* o = out + (int64_t)t * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        o[c] = in ? rbf((float)q[(int64_t)id * H + c] * sc[(int64_t)id * (H / 128) + c / 128]) : 0.f;
}

// The pair's sum of a gathered [2][n] fp32 tensor in rank order (an all-reduce: both ranks
// compute g0 + g1, the same bits) as bf16 or fp32.
__global__ void sum2_kernel(const float* __restrict__ g, int64_t n, __nv_bfloat16* ob, float* of) {
    pdl_enter();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i] + g[n + i];
    if (ob) ob[i] = to_bf(v);
    if (of) of[i] = v;
}

__global__ void f32_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int64_t n) {
    pdl_enter();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_bf(x[i]);
}

// [world][M][n] (a gather of row-sharded columns) -> [M][world * n], elements of `esz` bytes.
// rank_stride: elements between ranks' shards in src (M * n when packed).
template <typename T>
__global__ void unshard_kernel(const T* __restrict__ src, int64_t rank_stride, T* __restrict__ dst, int M, int n,
                               int world) {
    pdl_enter();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)world * M * n;
    if (i >= total) return;
    const int w = (int)(i / ((int64_t)M * n));
    const int64_t r = i % ((int64_t)M * n);
    const int m = (int)(r / n), c = (int)(r % n);
    dst[((int64_t)m * world + w) * n + c] = src[w * rank_stride + r];
}

// ------------------------------------------------------------------ RMSNorm
// tp.rms / norm_kernel._rms_kernel: y = bf16(w * bf16(x * (1 / sqrt(mean(x^2) + eps)))), fp32
// statistics. ADD: h = bf16(h + a) first (the residual add before the next norm), stored back.
template <bool ADD>
__global__ void __launch_bounds__(1024) rmsnorm_kernel(__nv_bfloat16* __restrict__ h, int64_t ldh,
                                                       const __nv_bfloat16* __restrict__ a, int64_t lda,
                                                       const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                                       int64_t ldy, int N, float eps) {
    pdl_enter();
    // the row in registers (N <= 12 x 1024): one read of h (and a), one write of each output
    __shared__ float red[32];
    constexpr int kPer = 12;
    const int t = blockIdx.x;
    __nv_bfloat16* hr = h + (int64_t)t * ldh;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        v[i] = 0.f;
        if (c < N) {
            if (ADD) {
                v[i] = rbf(bf(hr[c]) + bf(a[(int64_t)t * lda + c]));
                hr[c] = to_bf(v[i]);
            } else {
                v[i] = bf(hr[c]);
            }
        }
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, red);
    const float r = 1.f / sqrtf(ss / N + eps);
    __nv_bfloat16* yr = y + (int64_t)t * ldy;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < N) yr[c] = to_bf(bf(w[c]) * rbf(v[i] * r));
    }
}

// ------------------------------------------------------------------ RoPE
// rope_kernel._rope_kernel: interleaved pairs (x[2i], x[2i+1]) with frequency i in, de-interleaved
// halves out (y[i], y[i + 32]); each BF16 product rounded to BF16 before the sum. cos / sin are
// [positions][32] BF16 (the table's first half: it is cat(f, f)). In place when y == x (a warp
// reads its head's 64 values before it writes them).
__global__ void rope_kernel(const __nv_bfloat16* x, int64_t xt, int64_t xh, __nv_bfloat16* y, int64_t yt, int64_t yh,
                            int H, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ sn,
                            const int* __restrict__ pos) {
    pdl_enter();
    const int t = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), i = threadIdx.x & 31;
    if (h >= H) return;
    const int p = pos[t];
    const __nv_bfloat16* xr = x + (int64_t)t * xt + (int64_t)h * xh;
    const float x1 = bf(xr[2 * i]), x2 = bf(xr[2 * i + 1]);
    const float c = bf(cs[(int64_t)p * 32 + i]), s = bf(sn[(int64_t)p * 32 + i]);
    const float a = rbf(x1 * c), b = rbf(x2 * s), a2 = rbf(x2 * c), b2 = rbf(x1 * s);
    __syncwarp();
    __nv_bfloat16* yr = y + (int64_t)t * yt + (int64_t)h * yh;
    yr[i] = to_bf(a - b);
    yr[i + 32] = to_bf(a2 + b2);
}

// ------------------------------------------------------------------ caches
// tp.kv_write for an (fp8, scale) cache: per 128 values s = max(amax / 448, 1e-12), q = fp8(x / s)
// (a division, RNE), at cache row rows[t]. One warp per (row, 128-block).
__global__ void kv_fp8_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const int* __restrict__ rows,
                              uint8_t* __restrict__ q, float* __restrict__ s, int d) {
    pdl_enter();
    const int t = blockIdx.x, b = blockIdx.y, l = threadIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)t * ldx + b * 128 + 4 * l;
    float v[4], am = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(xr[i]);
        am = fmaxf(am, fabsf(v[i]));
    }
    am = warp_max(am);
    const float sc = fmaxf(am / 448.f, 1e-12f);
    const int64_t r = rows[t];
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= (uint32_t)to_e4m3(v[i] / sc) << (8 * i);
    *reinterpret_cast<uint32_t*>(q + r * d + b * 128 + 4 * l) = packed;
    if (l == 0) s[r * (d / 128) + b] = sc;
}

// rows of d bf16 values to cache rows rows[t] (the RoPE keys; tp's r_cache.index_copy_)
__global__ void rows_copy_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const int* __restrict__ rows,
                                 __nv_bfloat16* __restrict__ out, int d) {
    pdl_enter();
    const int t = blockIdx.x;
    const int64_t r = rows[t];
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[r * d + c] = x[(int64_t)t * ldx + c];
}

// A forward's positions and its rows in a split cache: pos[t] = pos0[0] + t (pos0 a device
// scalar: a captured graph replays at any position), rows[t] the row position pos[t] takes in
// this rank's cache (kv_split.owner_rows: p / world for its own positions, `spare` for the
// other's), bound[t] the last row it may attend (kv_split.local_bound, floor division).
// lanes (null: one sequence from pos0[0]): [L][4] (first row, rows, first position, cache-row
// offset) - row t of lane z is its position pos0_z + t - row0_z, and its cache rows are the
// lane's, koff_z on (rows[t] includes it, koff[t] reports it; bound[t] stays lane-local).
constexpr int kMaxLanes = 4;

__device__ __forceinline__ int lane_of(const int* lanes, int nl, int t) {
    for (int z = 0; z < nl; ++z)
        if (t >= lanes[4 * z] && t < lanes[4 * z] + lanes[4 * z + 1]) return z;
    return -1;
}

__global__ void positions_kernel(const int* __restrict__ pos0, int T, int rank, int world, int spare,
                                 int* __restrict__ pos, int* __restrict__ rows, int* __restrict__ bound,
                                 const int* __restrict__ lanes, int nl, int* __restrict__ koff) {
    pdl_enter();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int p = pos0 ? pos0[0] + t : 0, off = 0;
    if (lanes) {
        const int z = lane_of(lanes, nl, t);
        if (z >= 0) {
            p = lanes[4 * z + 2] + t - lanes[4 * z];
            off = lanes[4 * z + 3];
        } else {
            p = 0;  // a row of no lane (padding): position 0 of lane 0's rows, never read
        }
    }
    pos[t] = p;
    if (koff) koff[t] = off;
    if (rows) rows[t] = off + (world > 1 ? (p % world == rank ? p / world : spare) : p);
    if (bound) {
        const int d = p - rank;
        bound[t] = world > 1 ? (d >= 0 ? d / world : -1) : p;
    }
}

// ------------------------------------------------------------------ indexer
// tp._index_qw's keys: layer_norm(fp32(bf16(x @ wk^T)), 128, w, b, eps 1e-6) -> bf16, RoPE on the
// first 64 dims, then the FP8 cache row with one fp32 scale (kv_write with a [rows, 1] scale).
// One warp per row.
__global__ void index_key_kernel(const __nv_bfloat16* __restrict__ k, int64_t ldk, const float* __restrict__ w,
                                 const float* __restrict__ b, const __nv_bfloat16* __restrict__ cs,
                                 const __nv_bfloat16* __restrict__ sn, const int* __restrict__ pos,
                                 const int* __restrict__ rows, uint8_t* __restrict__ q, float* __restrict__ s, int T) {
    pdl_enter();
    __shared__ float buf[4][128];
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int t = blockIdx.x * 4 + warp;
    if (t >= T) return;
    float v[4];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(k[(int64_t)t * ldk + 4 * l + i]);
        sum += v[i];
    }
    const float mean = warp_sum(sum) / 128.f;
    float d2 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) d2 += (v[i] - mean) * (v[i] - mean);
    const float rstd = rsqrtf(warp_sum(d2) / 128.f + 1e-6f);
    float* bw = buf[warp];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = 4 * l + i;
        bw[j] = rbf((v[i] - mean) * rstd * w[j] + b[j]);
    }
    __syncwarp();
    const int p = pos[t];
    const float x1 = bw[2 * l], x2 = bw[2 * l + 1];
    const float c = bf(cs[(int64_t)p * 32 + l]), sv = bf(sn[(int64_t)p * 32 + l]);
    const float o1 = rbf(rbf(x1 * c) - rbf(x2 * sv)), o2 = rbf(rbf(x2 * c) + rbf(x1 * sv));
    __syncwarp();
    bw[l] = o1;
    bw[l + 32] = o2;
    __syncwarp();
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) am = fmaxf(am, fabsf(bw[4 * l + i]));
    am = warp_max(am);
    const float sc = fmaxf(am / 448.f, 1e-12f);
    const int64_t r = rows[t];
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= (uint32_t)to_e4m3(bw[4 * l + i] / sc) << (8 * i);
    *reinterpret_cast<uint32_t*>(q + r * 128 + 4 * l) = packed;
    if (l == 0) s[r] = sc;
}

// index_kernel._index_score_kernel on the tensor cores:
//   score[t, j] = sum_h w[t, h] * wscale * relu(((q[t, h] . k_bf16[j]) * ks[j]) * inv_d)
// for this rank's cache rows j < n_t, n_t the rows whose position (j * stride + off) is at or
// before pos[t]; -inf for j in [n_t, the block's last live row). A warp per query row holds its
// q (32 heads x 128) as mma A fragments; the block's 8 rows share each 64-key tile, staged from
// FP8 to BF16 (exact). The C layout gives a thread heads g, g + 8, g + 16, g + 24 for two keys;
// the head sum finishes over lanes 4, 8, 16 apart. split_n: the keys are a gather of both ranks'
// first split_n rows (prefill on a split layer), key j at row (j % 2) * split_n + j / 2.
// lanes (null: rows 0..T, one cache): the [L][4] table of positions_kernel, a grid layer a lane
// (gridDim.z = L), so a block's rows, which share its key tiles, are one lane's: lane z's rows
// row0_z.. score that lane's keys, its cache rows koff_z on.
constexpr int kIH = 32, kID = 128, kIT = 64, kILD = kID + 8, kIRows = 8;

template <int kIKeys>
__global__ void __launch_bounds__(256) index_scores_kernel(
    const __nv_bfloat16* __restrict__ q, const float* __restrict__ w, float wscale, const uint8_t* __restrict__ kq,
    const float* __restrict__ ks, const int* __restrict__ pos, int T, int nmax, int stride, int off, int split_n,
    float inv_d, float* __restrict__ out, int64_t ldo, const int* __restrict__ lanes) {
    pdl_enter();
    __shared__ __align__(16) __nv_bfloat16 kt[kIT * kILD];
    __shared__ float kss[kIT];
    __shared__ int lim_s[kIRows];
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, g = l >> 2, c = l & 3;
    const int z = blockIdx.z;
    const int row0 = lanes ? lanes[4 * z] : 0, tend = lanes ? row0 + lanes[4 * z + 1] : T;
    const int64_t koff = lanes ? lanes[4 * z + 3] : 0;
    kq += koff * kID;
    ks += koff;
    const int t0 = row0 + blockIdx.y * kIRows, t = t0 + warp;
    const int jb = blockIdx.x * kIKeys;
    if (threadIdx.x < kIRows) {
        const int tt = t0 + threadIdx.x;
        int n = 0;
        if (tt < tend) {
            const int p = pos[tt];
            n = p >= off ? min(nmax, (p - off) / stride + 1) : 0;
        }
        lim_s[threadIdx.x] = n;
    }
    __syncthreads();
    int max_lim = 0;
#pragma unroll
    for (int r = 0; r < kIRows; ++r) max_lim = max(max_lim, lim_s[r]);
    if (jb >= max_lim) return;
    const bool row = t < tend;
    const int lim = row ? lim_s[warp] : 0;
    uint32_t af[2][kID / 16][4];
    float wv[4] = {0.f, 0.f, 0.f, 0.f};
    if (row) {
        const uint32_t* qr = reinterpret_cast<const uint32_t*>(q + (int64_t)t * kIH * kID);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int kk = 0; kk < kID / 16; ++kk) {
                const int r0 = (mt * 16 + g) * kID + kk * 16 + 2 * c, r1 = r0 + 8 * kID;
                af[mt][kk][0] = qr[r0 / 2];
                af[mt][kk][1] = qr[r1 / 2];
                af[mt][kk][2] = qr[(r0 + 8) / 2];
                af[mt][kk][3] = qr[(r1 + 8) / 2];
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = w[(int64_t)t * kIH + g + 8 * i] * wscale;
    }
    float* orow = out + (int64_t)t * ldo;
    const int jend = min(jb + kIKeys, max_lim);
    for (int j0 = jb; j0 < jend; j0 += kIT) {
        __syncthreads();
        for (int i = threadIdx.x; i < kIT * (kID / 16); i += blockDim.x) {
            const int rr = i / (kID / 16), cc = (i % (kID / 16)) * 16;
            const int j = j0 + rr;
            uint4 o[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (j < jend) {
                const uint4 raw = *reinterpret_cast<const uint4*>(kq + key_row(j, split_n) * kID + cc);
                fp8x16_bf16(raw, 1.f, false, o);
            }
            *reinterpret_cast<uint4*>(&kt[rr * kILD + cc]) = o[0];
            *reinterpret_cast<uint4*>(&kt[rr * kILD + cc + 8]) = o[1];
        }
        if (threadIdx.x < kIT) {
            const int j = j0 + threadIdx.x;
            kss[threadIdx.x] = j < jend ? ks[key_row(j, split_n)] : 0.f;
        }
        __syncthreads();
        if (!row || j0 >= lim) {
            if (row)
                for (int j = j0 + l; j < min(j0 + kIT, jend); j += 32) orow[j] = -INFINITY;
            continue;
        }
#pragma unroll 2
        for (int nt = 0; nt < kIT / 8; ++nt) {
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int k2 = 0; k2 < kID / 32; ++k2) {
                uint32_t b[4];
                ldsm_x4(b, &kt[(nt * 8 + (l & 7)) * kILD + k2 * 32 + (l >> 3) * 8]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const uint32_t* a0 = af[mt][2 * k2];
                    const uint32_t* a1 = af[mt][2 * k2 + 1];
                    mma_bf16(acc[mt], a0[0], a0[1], a0[2], a0[3], b[0], b[1]);
                    mma_bf16(acc[mt], a1[0], a1[1], a1[2], a1[3], b[2], b[3]);
                }
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int kl = nt * 8 + 2 * c + e;
                const float s = kss[kl];
                const float h0 = fmaxf(acc[0][e] * s * inv_d, 0.f) * wv[0];
                const float h8 = fmaxf(acc[0][2 + e] * s * inv_d, 0.f) * wv[1];
                const float h16 = fmaxf(acc[1][e] * s * inv_d, 0.f) * wv[2];
                const float h24 = fmaxf(acc[1][2 + e] * s * inv_d, 0.f) * wv[3];
                float sum = (h0 + h16) + (h8 + h24);
                sum += __shfl_xor_sync(0xffffffffu, sum, 16);
                sum += __shfl_xor_sync(0xffffffffu, sum, 8);
                sum += __shfl_xor_sync(0xffffffffu, sum, 4);
                const int j = j0 + kl;
                if (g == 0 && j < jend) orow[j] = j < lim ? sum : -INFINITY;
            }
        }
    }
}

// The indexer's top-k (torch.topk(min(2048, S)).indices) per row over its live entries s[0, n_t):
// a radix select (4 x 8 bits of the orderable key, counted in per-warp histograms) for the K-th
// largest value, then an ordered compaction: each warp a contiguous segment, its counts scanned once
// and its picks placed by ballots. The selected indices come out ascending, ties at the threshold to
// the lowest indices. Rows with n_t <= K take all of them. -1 (value -inf) past what was selected.
// n_t from pos as in index_scores_kernel (pos null: nmax). The first version counted into one
// shared histogram and compacted in 1,024-element rounds of two block-wide scans: 125-137 us a row
// of 60-63K scores (a 120K context's split layers), this one 45-55 with the same output, byte for
// byte, over random, zero-heavy, duplicate-heavy and masked rows.
constexpr int kTk = 1024;

__global__ void __launch_bounds__(kTk) topk_kernel(const float* __restrict__ s, int64_t lds, const int* __restrict__ pos,
                                                    int stride, int off, int nmax, int K, int* __restrict__ idx,
                                                    int64_t ldi, float* __restrict__ val, int64_t ldv) {
    pdl_enter();
    constexpr int NW = kTk / 32;
    __shared__ int hist[NW][257];
    __shared__ int wgt[NW], weq[NW];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_need;
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int n = nmax;
    if (pos) {
        const int p = pos[t];
        n = p >= off ? min(nmax, (p - off) / stride + 1) : 0;
    }
    const float* row = s + (int64_t)t * lds;
    int* io = idx + (int64_t)t * ldi;
    float* vo = val ? val + (int64_t)t * ldv : nullptr;
    if (n <= K) {
        for (int j = tid; j < K; j += kTk) {
            io[j] = j < n ? j : -1;
            if (vo) vo[j] = j < n ? row[j] : -INFINITY;
        }
        return;
    }
    uint32_t prefix = 0, mask = 0;
    int need = K;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = lane; b < 256; b += 32) hist[warp][b] = 0;
        __syncwarp();
        for (int j = tid; j < n; j += kTk) {
            const uint32_t u = fkey(row[j]);
            if ((u & mask) == prefix) atomicAdd(&hist[warp][(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid < 256) {
            int c = 0;
#pragma unroll 8
            for (int w = 0; w < NW; ++w) c += hist[w][tid];
            hist[0][tid] = c;
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) {
                if (acc + hist[0][b] >= need) break;
                acc += hist[0][b];
            }
            sh_prefix = prefix | ((uint32_t)b << shift);
            sh_need = need - acc;
        }
        __syncthreads();
        prefix = sh_prefix;
        need = sh_need;
        mask |= 0xffu << shift;
    }
    // output: warp w's segment [w S, w S + S), in order; the elements above the threshold key and the
    // first `need` equal to it in index order, as topk_kernel takes them
    const int S = ((n + NW - 1) / NW + 31) / 32 * 32;
    const int j0 = min(n, warp * S), j1 = min(n, j0 + S);
    int gt_c = 0, eq_c = 0;
    for (int j = j0 + lane; j < j1; j += 32) {
        const uint32_t u = fkey(row[j]);
        gt_c += u > prefix;
        eq_c += u == prefix;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt_c += __shfl_xor_sync(0xffffffffu, gt_c, o);
        eq_c += __shfl_xor_sync(0xffffffffu, eq_c, o);
    }
    if (lane == 0) {
        wgt[warp] = gt_c;
        weq[warp] = eq_c;
    }
    __syncthreads();
    int base = 0, eq_seen = 0, total = 0, e = 0;
    for (int w = 0; w < NW; ++w) {
        const int take = wgt[w] + min(weq[w], max(0, need - e));
        if (w < warp) {
            base += take;
            eq_seen += weq[w];
        }
        total += take;
        e += weq[w];
    }
    const unsigned lt = (1u << lane) - 1u;
    for (int jb = j0; jb < j1; jb += 32) {
        const int j = jb + lane;
        const bool ok = j < j1;
        const float v = ok ? row[j] : 0.f;
        const uint32_t u = fkey(v);
        const bool gt = ok && u > prefix, eq = ok && u == prefix;
        const unsigned em = __ballot_sync(0xffffffffu, eq);
        const bool take = gt || (eq && eq_seen + __popc(em & lt) < need);
        const unsigned tm = __ballot_sync(0xffffffffu, take);
        const int r = base + __popc(tm & lt);
        if (take && r < K) {
            io[r] = j;
            if (vo) vo[r] = v;
        }
        base += __popc(tm);
        eq_seen += __popc(em);
    }
    for (int j = total + tid; j < K; j += kTk) {
        io[j] = -1;
        if (vo) vo[j] = -INFINITY;
    }
}

// kv_split.share, as a mask: each rank's top-K index scores (gathered: rank w, row t at
// g + w * rank_stride + t * row_stride + val_off, K fp32) -> this rank's picks that are in the
// top K of all 2K, the others set to -1 in idx (the attention skips them). Ties at the threshold
// go to the lower (rank, index): both ranks compute the same set from the same bytes, so the
// pair attends exactly K positions between them.
__global__ void __launch_bounds__(kTk) split_share_kernel(const uint8_t* __restrict__ g, int64_t rank_stride,
                                                          int64_t row_stride, int64_t val_off, int* __restrict__ idx,
                                                          int64_t ldi, int rank, int K) {
    pdl_enter();
    __shared__ int hist[256];
    __shared__ int wtot[32];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_need;
    const int t = blockIdx.x, tid = threadIdx.x;
    const int N = 2 * K;
    auto value = [&](int e) {
        const int w = e / K, k = e % K;
        return reinterpret_cast<const float*>(g + w * rank_stride + (int64_t)t * row_stride + val_off)[k];
    };
    uint32_t prefix = 0, mask = 0;
    int need = K;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int e = tid; e < N; e += kTk) {
            const uint32_t u = fkey(value(e));
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= need) break;
                acc += hist[b];
            }
            sh_prefix = prefix | ((uint32_t)b << shift);
            sh_need = need - acc;
        }
        __syncthreads();
        prefix = sh_prefix;
        need = sh_need;
        mask |= 0xffu << shift;
    }
    int eq_seen = 0;
    int* io = idx + (int64_t)t * ldi;
    for (int e0 = 0; e0 < N; e0 += kTk) {
        const int e = e0 + tid;
        uint32_t u = 0;
        if (e < N) u = fkey(value(e));
        const bool gt = e < N && u > prefix, eq = e < N && u == prefix;
        int eq_tot;
        const int er = block_rank(eq, wtot, eq_tot);
        const bool keep = gt || (eq && eq_seen + er < need);
        if (e < N && e / K == rank && !keep) io[e % K] = -1;
        eq_seen += eq_tot;
    }
}

// ------------------------------------------------------------------ absorbed MLA projections
// kvb_fp8.absorb_q for decode rows: q_lat[t, h, c] = sum_n q_nope[t, h, n] * Wuk[h, n, c], the
// FP8 kv_b's effective weight bf16(fp8 * s) (s its 128x128 block's scale, expanded per row), fp32
// accumulation, bf16 out. kv_b rows of head h: h * 448 + n, n < 192 the key half. A block takes
// one head's 128 output columns; its threads split n in two halves. T <= 8 a launch.
constexpr int kKvRows = 448, kNope = 192, kLat = 512, kVd = 256;

__global__ void __launch_bounds__(256) absorb_q_kernel(const __nv_bfloat16* __restrict__ q, int64_t ldq_t,
                                                       int64_t ldq_h, const uint8_t* __restrict__ w,
                                                       const float* __restrict__ ws, __nv_bfloat16* __restrict__ out,
                                                       int64_t ldo_t, int64_t ldo_h, int T) {
    pdl_enter();
    // a block: one head's 128 output columns over up to 8 rows; a thread 4 adjacent columns (one
    // u32 of FP8) over a 24-row slice of n, its loads unrolled to keep them in flight; the 8
    // slices summed in shared memory
    constexpr int TM = 8;
    __shared__ float xs[TM][kNope];
    __shared__ float part[8][TM][128];
    const int h = blockIdx.x, cb = blockIdx.y, tid = threadIdx.x;
    for (int i = tid; i < T * kNope; i += blockDim.x) {
        const int t = i / kNope, n = i % kNope;
        xs[t][n] = bf(q[(int64_t)t * ldq_t + (int64_t)h * ldq_h + n]);
    }
    __syncthreads();
    const int cg = tid & 31, sl = tid >> 5;
    const int col = cb * 128 + cg * 4;
    float acc[TM][4];
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[t][c] = 0.f;
    uint32_t wd[24];
    float sc[24];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
        const int64_t r = (int64_t)h * kKvRows + sl * 24 + i;
        wd[i] = ld_once(reinterpret_cast<const uint32_t*>(w + r * kLat + col), 1);
        sc[i] = ws[r * 4 + cb];
    }
#pragma unroll
    for (int i = 0; i < 24; ++i) {
        const int n = sl * 24 + i;
        float wv[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) wv[c] = rbf(e4m3((wd[i] >> (8 * c)) & 0xff) * sc[i]);
#pragma unroll
        for (int t = 0; t < TM; ++t)
            if (t < T) {
                const float x = xs[t][n];
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[t][c] = fmaf(x, wv[c], acc[t][c]);
            }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t)
        if (t < T)
#pragma unroll
            for (int c = 0; c < 4; ++c) part[sl][t][cg * 4 + c] = acc[t][c];
    __syncthreads();
    for (int i = tid; i < T * 128; i += blockDim.x) {
        const int t = i / 128, c = i % 128;
        float a = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) a += part[k][t][c];
        out[(int64_t)t * ldo_t + (int64_t)h * ldo_h + cb * 128 + c] = to_bf(a);
    }
}

// kvb_fp8.absorb_o for decode rows: o[t, h, v] = sum_c o_lat[t, h, c] * Wuv[h, v, c] (kv_b rows
// h * 448 + 192 + v), bf16 out. A warp per output column v, lanes over c in 16-value pieces; a
// block kAbsV columns, so each block stages the rows' latents once for them.
constexpr int kAbsV = 64;
__global__ void __launch_bounds__(256) absorb_o_kernel(const __nv_bfloat16* __restrict__ o, int64_t ldo_t,
                                                       int64_t ldo_h, const uint8_t* __restrict__ w,
                                                       const float* __restrict__ ws, __nv_bfloat16* __restrict__ out,
                                                       int64_t ld_out, int T) {
    pdl_enter();
    __shared__ __align__(16) float xs[16][kLat];
    const int h = blockIdx.x, warp = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int i = threadIdx.x; i < T * kLat; i += blockDim.x) {
        const int t = i / kLat, c = i % kLat;
        xs[t][c] = bf(o[(int64_t)t * ldo_t + (int64_t)h * ldo_h + c]);
    }
    __syncthreads();
    for (int vi = 0; vi < kAbsV / 8; ++vi) {
        const int v = blockIdx.y * kAbsV + vi * 8 + warp;
        const int64_t r = (int64_t)h * kKvRows + kNope + v;
        const uint4 raw = ld_once(reinterpret_cast<const uint4*>(w + r * kLat + l * 16), 1);
        const float s = ws[r * 4 + (l * 16) / 128];
        const uint32_t wd[4] = {raw.x, raw.y, raw.z, raw.w};
        float wv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) wv[i] = rbf(e4m3((wd[i / 4] >> (8 * (i % 4))) & 0xff) * s);
        for (int t = 0; t < T; ++t) {
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) a = fmaf(xs[t][l * 16 + i], wv[i], a);
            a = warp_sum(a);
            if (l == 0) out[(int64_t)t * ld_out + (int64_t)h * kVd + v] = to_bf(a);
        }
    }
}

// The FP8 kv_b's effective weight as bf16 rows (prefill's absorb runs it through cuBLAS).
__global__ void kvb_dequant_kernel(const uint8_t* __restrict__ w, const float* __restrict__ ws,
                                   __nv_bfloat16* __restrict__ out, int64_t rows) {
    pdl_enter();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kLat) return;
    const int64_t r = i / kLat;
    const int c = (int)(i % kLat);
    out[i] = to_bf(e4m3(w[i]) * ws[r * 4 + c / 128]);
}

// ------------------------------------------------------------------ sparse MLA attention
// sparse_attn._sparse_mla_kernel: a block owns HB heads of one query row and walks its picks (or,
// idx null, positions 0..bound: the dense causal prefill) NT at a time. A key is the cached latent
// (FP8 widened and times its 128-block scale in BF16, as kv_read does) and its RoPE key:
//   s = (q_lat . c + q_rope . r) * scale (fp32), masked to 0 <= key <= bound[t];
//   online softmax (m clamped to >= -1e30), l over fp32 p, acc += bf16(p) . c (fp32).
// Out: bf16 acc / max(l, 1e-30) (out non-null, one split) or the split's raw (m, l, acc) partials
// [T][H][NS] for mla_combine_kernel. q head h is at q + (h / hg) * q_ldg + t * q_ldt + (h % hg) * q_ldh
// (a gathered [rank][row][head] query in place); split_n: a gathered pair's keys (key_row);
// koff (null: 0): row t's keys are its lane's cache rows, koff[t] on (positions_kernel).
constexpr int kMD = 576, kMLD = kMD + 8, kMNT = 32, kMPS = kMNT + 4;

template <int MT>
constexpr int mla_smem() {
    return (16 * MT + kMNT) * kMLD * 2 + 16 * MT * kMPS * 4 + 3 * 16 * MT * 4 + 2 * kMNT * 4;
}

template <int MT>
__global__ void __launch_bounds__(256) mla_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t q_ldg, int64_t q_ldt, int64_t q_ldh, int hg,
    const uint8_t* __restrict__ cq, const float* __restrict__ cs, const __nv_bfloat16* __restrict__ cr, int split_n,
    const int* __restrict__ idx, int64_t ld_idx, int K, int ascending, const int* __restrict__ bound, int H, float scale,
    int chunk, float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    __nv_bfloat16* __restrict__ out, int64_t o_ldt, int64_t o_ldh, const int* __restrict__ koff) {
    pdl_enter();
    constexpr int HB = 16 * MT;
    extern __shared__ __align__(16) unsigned char mla_smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(mla_smem_raw);   // [HB][kMLD]
    __nv_bfloat16* Ks = Qs + HB * kMLD;                                    // [NT][kMLD]
    float* Ps = reinterpret_cast<float*>(Ks + kMNT * kMLD);               // [HB][PS]
    float* m_run = Ps + HB * kMPS;
    float* l_run = m_run + HB;
    float* alpha = l_run + HB;
    int* valid = reinterpret_cast<int*>(alpha + HB);
    const int t = blockIdx.y, h0 = blockIdx.x * HB, sp = blockIdx.z, NS = gridDim.z;
    if (koff) {
        const int64_t o = koff[t];
        cq += o * kLat;
        cs += o * 4;
        cr += o * 64;
    }
    const int nh = min(HB, H - h0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, qd = lane & 3;
    const int bnd = bound[t];
    // picks in ascending position order with -1 after them (the indexer's top-k, unmasked):
    // a row has at most min(K, bound + 1) of them, so the tiles past that are skipped
    const int kmax = idx && !ascending ? K : min(K, bnd + 1);
    const int k_lo = sp * chunk, k_hi = min(kmax, k_lo + chunk);

    for (int i = tid; i < HB * (kMD / 8); i += blockDim.x) {
        const int hh = i / (kMD / 8), c = i % (kMD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (hh < nh) {
            const int h = h0 + hh;
            v = reinterpret_cast<const uint4*>(q + (int64_t)(h / hg) * q_ldg + (int64_t)t * q_ldt +
                                               (int64_t)(h % hg) * q_ldh)[c];
        }
        *reinterpret_cast<uint4*>(Qs + hh * kMLD + c * 8) = v;
    }
    if (tid < HB) {
        m_run[tid] = -1e30f;
        l_run[tid] = 0.f;
    }
    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    // A thread's 5 of a tile's 32 x 40 pieces (32 latent pieces of 16 FP8 values and 8 RoPE
    // pieces of 8 BF16 values a key), fetched into registers a tile ahead: the next tile's two
    // dependent loads (its pick, then its key) are in flight while this one is scored.
    constexpr int kPieces = kMNT * 40 / 256;
    uint4 pre_raw[kPieces];
    float pre_s[kPieces];
    bool pre_ok[kPieces];
    auto prefetch = [&](int n0) {
#pragma unroll
        for (int u = 0; u < kPieces; ++u) {
            const int i = tid + 256 * u, j = i / 40, c = i % 40;
            const int n = n0 + j;
            int key = -1;
            if (n < k_hi) key = idx ? idx[(int64_t)t * ld_idx + n] : n;
            const bool ok = key >= 0 && key <= bnd;
            pre_ok[u] = ok;
            pre_raw[u] = make_uint4(0, 0, 0, 0);
            pre_s[u] = 0.f;
            if (ok) {
                const int64_t kr = key_row(key, split_n);
                if (c < 32) {
                    pre_raw[u] = *reinterpret_cast<const uint4*>(cq + kr * kLat + c * 16);
                    pre_s[u] = rbf(cs[kr * 4 + c / 8]);
                } else {
                    pre_raw[u] = reinterpret_cast<const uint4*>(cr + kr * 64)[c - 32];
                }
            }
        }
    };
    if (k_lo < k_hi) prefetch(k_lo);
    for (int n0 = k_lo; n0 < k_hi; n0 += kMNT) {
        __syncthreads();  // the previous tile's PV is done with Ks / Ps
#pragma unroll
        for (int u = 0; u < kPieces; ++u) {
            const int i = tid + 256 * u, j = i / 40, c = i % 40;
            if (c < 32) {
                uint4 o[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
                if (pre_ok[u]) fp8x16_bf16(pre_raw[u], pre_s[u], true, o);
                *reinterpret_cast<uint4*>(Ks + j * kMLD + c * 16) = o[0];
                *reinterpret_cast<uint4*>(Ks + j * kMLD + c * 16 + 8) = o[1];
            } else {
                *reinterpret_cast<uint4*>(Ks + j * kMLD + kLat + (c - 32) * 8) = pre_raw[u];
            }
            if (c == 0) valid[j] = pre_ok[u];
        }
        __syncthreads();
        if (n0 + kMNT < k_hi) prefetch(n0 + kMNT);
        // scores: (m tile, 8 keys) jobs over the warps
        for (int job = warp; job < MT * (kMNT / 8); job += 8) {
            const int mt = job / (kMNT / 8), kb = job % (kMNT / 8);
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            const __nv_bfloat16* kn = Ks + (kb * 8 + r) * kMLD;
            const __nv_bfloat16* qa = Qs + (mt * 16) * kMLD;
#pragma unroll 4
            for (int ks = 0; ks < kMD / 16; ++ks) {
                const int d0 = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(qa + r * kMLD + d0 + 2 * qd);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(qa + (r + 8) * kMLD + d0 + 2 * qd);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(qa + r * kMLD + d0 + 8 + 2 * qd);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(qa + (r + 8) * kMLD + d0 + 8 + 2 * qd);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kn + d0 + 2 * qd);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kn + d0 + 8 + 2 * qd);
                mma_bf16(c4, a0, a1, a2, a3, b0, b1);
            }
            const int k0 = kb * 8 + 2 * qd;
            float* pr = Ps + (mt * 16) * kMPS;
            pr[r * kMPS + k0] = valid[k0] ? c4[0] * scale : -INFINITY;
            pr[r * kMPS + k0 + 1] = valid[k0 + 1] ? c4[1] * scale : -INFINITY;
            pr[(r + 8) * kMPS + k0] = valid[k0] ? c4[2] * scale : -INFINITY;
            pr[(r + 8) * kMPS + k0 + 1] = valid[k0 + 1] ? c4[3] * scale : -INFINITY;
        }
        __syncthreads();
        // online softmax: a warp takes 2 * MT heads, a lane one key
#pragma unroll
        for (int u = 0; u < 2 * MT; ++u) {
            const int hh = warp * 2 * MT + u;
            const float mo = m_run[hh];
            const float s0 = Ps[hh * kMPS + lane];
            const float mn = fmaxf(fmaxf(mo, warp_max(s0)), -1e30f);
            const float e0 = expf(s0 - mn);
            const float ls = warp_sum(e0);
            Ps[hh * kMPS + lane] = e0;
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
                acc[mt][nt][0] *= al0;
                acc[mt][nt][1] *= al0;
                acc[mt][nt][2] *= al1;
                acc[mt][nt][3] *= al1;
            }
        }
        // PV: warp w takes latent dims w * 64 .. + 64; p rounded to bf16 (as tl.dot(p.to(bf16), c))
#pragma unroll
        for (int ks = 0; ks < kMNT / 16; ++ks) {
            const int k0 = ks * 16;
            uint32_t ah[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float* pr0 = Ps + (mt * 16 + r) * kMPS + k0 + 2 * qd;
                const float* pr1 = Ps + (mt * 16 + r + 8) * kMPS + k0 + 2 * qd;
                ah[mt][0] = pack_bf16(pr0[0], pr0[1]);
                ah[mt][1] = pack_bf16(pr1[0], pr1[1]);
                ah[mt][2] = pack_bf16(pr0[8], pr0[9]);
                ah[mt][3] = pack_bf16(pr1[8], pr1[9]);
            }
            const int key = k0 + (lane & 15);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                uint32_t b0, b1;
                ldsm_x2_trans(b0, b1, Ks + key * kMLD + warp * 64 + nt * 8);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) mma_bf16(acc[mt][nt], ah[mt][0], ah[mt][1], ah[mt][2], ah[mt][3], b0, b1);
            }
        }
    }
    __syncthreads();
    if (out) {
        if (tid < HB) alpha[tid] = 1.f / fmaxf(l_run[tid], 1e-30f);
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int d = warp * 64 + nt * 8 + 2 * qd;
                const int ha = mt * 16 + r, hb = ha + 8;
                if (ha < nh) {
                    __nv_bfloat16* o = out + (int64_t)t * o_ldt + (int64_t)(h0 + ha) * o_ldh + d;
                    *reinterpret_cast<uint32_t*>(o) = pack_bf16(acc[mt][nt][0] * alpha[ha], acc[mt][nt][1] * alpha[ha]);
                }
                if (hb < nh) {
                    __nv_bfloat16* o = out + (int64_t)t * o_ldt + (int64_t)(h0 + hb) * o_ldh + d;
                    *reinterpret_cast<uint32_t*>(o) = pack_bf16(acc[mt][nt][2] * alpha[hb], acc[mt][nt][3] * alpha[hb]);
                }
            }
        return;
    }
    if (tid < nh) {
        const int64_t b = ((int64_t)t * H + h0 + tid) * NS + sp;
        part_m[b] = m_run[tid];
        part_l[b] = l_run[tid];
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int d = warp * 64 + nt * 8 + 2 * qd;
            const int ha = mt * 16 + r, hb = ha + 8;
            if (ha < nh) {
                float* o = part_acc + (((int64_t)t * H + h0 + ha) * NS + sp) * kLat + d;
                o[0] = acc[mt][nt][0];
                o[1] = acc[mt][nt][1];
            }
            if (hb < nh) {
                float* o = part_acc + (((int64_t)t * H + h0 + hb) * NS + sp) * kLat + d;
                o[0] = acc[mt][nt][2];
                o[1] = acc[mt][nt][3];
            }
        }
}

// The splits' merge (sparse_attn.sparse_mla's flash-decoding merge, and _combine_kernel): per
// (row, head) m = max pm, w = exp(pm - m), o = sum acc * w / max(sum l * w, 1e-30),
// lse = m + log(l) (or -inf with nothing attended). out non-null: bf16 o at out + t * o_ldt + h * o_ldh.
// Else (split KV): this rank's heads [lo, lo + hl) to own_o / own_l (fp32), the others' to the
// row's `send` bytes (fp16 outputs clamped to +-65504 [hl][512], then fp32 lse [hl]).
__global__ void mla_combine_kernel(const float* __restrict__ pm, const float* __restrict__ pl,
                                   const float* __restrict__ pacc, int H, int NS, __nv_bfloat16* __restrict__ out,
                                   int64_t o_ldt, int64_t o_ldh, float* __restrict__ own_o, float* __restrict__ own_l,
                                   uint8_t* __restrict__ send, int64_t row_bytes, int hl, int lo) {
    pdl_enter();
    const int t = blockIdx.x, h = blockIdx.y;
    const int64_t b = ((int64_t)t * H + h) * NS;
    float m = -1e30f;
    for (int s = 0; s < NS; ++s) m = fmaxf(m, pm[b + s]);
    float l = 0.f;
    for (int s = 0; s < NS; ++s) l += pl[b + s] * expf(pm[b + s] - m);
    const float inv = 1.f / fmaxf(l, 1e-30f);
    const float lse = l > 0.f ? m + logf(fmaxf(l, 1e-30f)) : -INFINITY;
    const int hh = h - lo;
    const bool own = !out && hh >= 0 && hh < hl;
    const int ho = h < lo ? h : h - hl;
    for (int d = threadIdx.x; d < kLat; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < NS; ++s) a += pacc[(b + s) * kLat + d] * expf(pm[b + s] - m);
        const float o = a * inv;
        if (out) {
            out[(int64_t)t * o_ldt + (int64_t)h * o_ldh + d] = to_bf(o);
        } else if (own) {
            own_o[((int64_t)t * hl + hh) * kLat + d] = o;
        } else {
            reinterpret_cast<__half*>(send + (int64_t)t * row_bytes)[(int64_t)ho * kLat + d] =
                __float2half_rn(fminf(fmaxf(o, -65504.f), 65504.f));
        }
    }
    if (threadIdx.x == 0 && !out) {
        if (own) own_l[(int64_t)t * hl + hh] = lse;
        else reinterpret_cast<float*>(send + (int64_t)t * row_bytes + (int64_t)hl * kLat * 2)[ho] = lse;
    }
}

// sparse_attn._merge_kernel: this rank's partial (fp32) and the other rank's (from its send
// bytes, fp16 + lse) for this rank's heads -> bf16 o_lat [T][hl][512].
__global__ void mla_merge_kernel(const float* __restrict__ own_o, const float* __restrict__ own_l,
                                 const uint8_t* __restrict__ recv, int64_t row_bytes, int hl,
                                 __nv_bfloat16* __restrict__ out) {
    pdl_enter();
    const int t = blockIdx.x, h = blockIdx.y;
    const float la = own_l[(int64_t)t * hl + h];
    const uint8_t* row = recv + (int64_t)t * row_bytes;
    const float lb = reinterpret_cast<const float*>(row + (int64_t)hl * kLat * 2)[h];
    float m = fmaxf(la, lb);
    if (m == -INFINITY) m = 0.f;
    const float wa = expf(la - m), wb = expf(lb - m);
    const float inv = 1.f / fmaxf(wa + wb, 1e-30f);
    for (int d = threadIdx.x; d < kLat; d += blockDim.x) {
        const float oa = own_o[((int64_t)t * hl + h) * kLat + d];
        const float ob = __half2float(reinterpret_cast<const __half*>(row)[(int64_t)h * kLat + d]);
        out[((int64_t)t * hl + h) * kLat + d] = to_bf((oa * wa + ob * wb) * inv);
    }
}

// ------------------------------------------------------------------ routing
// route_kernel.py: per row, scores s = sigmoid(logits), choice = s + bias; top-K picks in
// descending order (ties to the lower expert) over the experts `allowed` (null: all), weights
// s / (sum s + 1e-20) * scale summed in pick order. A row with fewer than K allowed experts gets
// zero-weight -1 picks (the EXL3 kernels skip a negative pick: an exact 0, rather than
// tp.py's zero-weight pick of a loaded slot). One block of E threads per row.
//   exact (allowed null): ids [T][K] experts, w [T][K]; `hot` [T * K][2]: per pick (its share
//     w / scale if the expert is not in `resident`, else 0; the expert), the hot reads' candidates.
//   routed (allowed set): ids [T][K] = lut[expert] arena slots (-1 when slotless: weight 0);
//     demand[e] += the exact weights (ex_ids / ex_w) and acc[4] += (picks, missed, mass,
//     missed mass) for rows below live[0] (null: all rows).
constexpr int kE = 256;

__device__ void route_select(const float* __restrict__ lg, const float* __restrict__ bias,
                             const uint8_t* __restrict__ allowed, int K, float* sc, int* chosen, float* chosen_s,
                             int& nvalid) {
    __shared__ float sel[kE];
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int e = threadIdx.x;
    const float s = 1.f / (1.f + expf(-lg[e]));
    sc[e] = s;
    sel[e] = (allowed && !allowed[e]) ? -INFINITY : s + bias[e];
    __syncthreads();
    nvalid = 0;
    for (int k = 0; k < K; ++k) {
        float best = sel[e];
        int besti = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
            if (ov > best || (ov == best && oi < besti)) {
                best = ov;
                besti = oi;
            }
        }
        if ((e & 31) == 0) {
            bv[e >> 5] = best;
            bi[e >> 5] = besti;
        }
        __syncthreads();
        float b = bv[0];
        int bidx = bi[0];
        for (int i = 1; i < kE / 32; ++i)
            if (bv[i] > b || (bv[i] == b && bi[i] < bidx)) {
                b = bv[i];
                bidx = bi[i];
            }
        const bool ok = b > -INFINITY;
        chosen[k] = ok ? bidx : -1;
        chosen_s[k] = ok ? sc[bidx] : 0.f;
        nvalid += ok;
        __syncthreads();
        if (e == bidx) sel[e] = -INFINITY;
        __syncthreads();
    }
}

__global__ void __launch_bounds__(kE) router_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    const uint8_t* __restrict__ allowed, const int* __restrict__ lut,
                                                    int K, float scale, int* __restrict__ ids, float* __restrict__ w,
                                                    const uint8_t* __restrict__ resident, float* __restrict__ hot,
                                                    const int* __restrict__ ex_ids, const float* __restrict__ ex_w,
                                                    float* __restrict__ demand, double* __restrict__ acc,
                                                    const int* __restrict__ live, double* __restrict__ lane_acc,
                                                    float* __restrict__ lane_demand, int64_t lane_stride,
                                                    const int* __restrict__ row_lane) {
    pdl_enter();
    __shared__ float sc[kE];
    const int t = blockIdx.x;
    int chosen[8];
    float cs[8];
    int nv;
    route_select(logits + (int64_t)t * kE, bias, allowed, K, sc, chosen, cs, nv);
    if (threadIdx.x != 0) return;
    float tot = 0.f;
    for (int k = 0; k < K; ++k) tot += cs[k];
    for (int k = 0; k < K; ++k) {
        const int ex = chosen[k];
        float wt = cs[k] / (tot + 1e-20f) * scale;
        int out = ex;
        if (lut) {
            out = ex >= 0 ? lut[ex] : -1;
            if (out < 0) wt = 0.f;
        }
        ids[t * K + k] = out;
        w[t * K + k] = wt;
        if (hot) {
            const bool miss = ex >= 0 && resident && !resident[ex];
            hot[(t * K + k) * 2] = miss ? wt / scale : 0.f;
            hot[(t * K + k) * 2 + 1] = (float)ex;
        }
    }
    if (live && t >= live[0]) return;
    if (demand || acc || lane_acc || lane_demand) {
        // lane_acc / lane_demand: the same per row's lane (row_lane[t], null 0) at lane * 4 /
        // lane * lane_stride, a request's own counts beside another lane's rows
        const int z = row_lane ? row_lane[t] : 0;
        double missed = 0.0, mass = 0.0, mmass = 0.0;
        for (int k = 0; k < K; ++k) {
            const int ex = ex_ids[t * K + k];
            const float ew = ex_w[t * K + k];
            if (ex < 0) continue;
            if (demand) atomicAdd(&demand[ex], ew);
            if (lane_demand) atomicAdd(&lane_demand[z * lane_stride + ex], ew);
            const bool miss = allowed && !allowed[ex];
            missed += miss;
            mass += ew;
            mmass += miss ? (double)ew : 0.0;
        }
        if (acc) {
            atomicAdd(&acc[0], (double)K);
            atomicAdd(&acc[1], missed);
            atomicAdd(&acc[2], mass);
            atomicAdd(&acc[3], mmass);
        }
        if (lane_acc) {
            atomicAdd(&lane_acc[4 * z], (double)K);
            atomicAdd(&lane_acc[4 * z + 1], missed);
            atomicAdd(&lane_acc[4 * z + 2], mass);
            atomicAdd(&lane_acc[4 * z + 3], mmass);
        }
    }
}

// ------------------------------------------------------------------ MLP pieces
// exl3_model.FixedMLP.act: bf16(g * sigmoid(g) * u), g and u the bf16 projections in fp32.
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ g, int64_t ldg, const __nv_bfloat16* __restrict__ u,
                              int64_t ldu, __nv_bfloat16* __restrict__ y, int64_t ldy, int n) {
    pdl_enter();
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float gv = bf(g[(int64_t)t * ldg + c]), uv = bf(u[(int64_t)t * ldu + c]);
    y[(int64_t)t * ldy + c] = to_bf(gv * (1.f / (1.f + expf(-gv))) * uv);
}

// exl3_model.mlp's sum: bf16(routed + shared down) (routed null: a dense layer's down alone).
__global__ void moe_out_kernel(const float* __restrict__ routed, const float* __restrict__ shared,
                               __nv_bfloat16* __restrict__ y, int64_t n) {
    pdl_enter();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = to_bf((routed ? routed[i] : 0.f) + shared[i]);
}

// ------------------------------------------------------------------ head
// Per row of this rank's fp32 logits [M][n]: the first-index argmax as (offset + index, value
// bits) pairs [M][2] (gathered over the pair, then argmax_combine_kernel).
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int64_t ldx, int n, int offset,
                                                      int* __restrict__ pairs) {
    pdl_enter();
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* row = x + (int64_t)blockIdx.x * ldx;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v > best || (v == best && i < besti)) {
            best = v;
            besti = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) {
            best = ov;
            besti = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = besti;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        best = bv[0];
        besti = bi[0];
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < besti)) {
                best = bv[w];
                besti = bi[w];
            }
        pairs[blockIdx.x * 2] = besti + offset;
        pairs[blockIdx.x * 2 + 1] = __float_as_int(best);
    }
}

// pairs [world][M][2] -> the argmax over ranks (ties: the lower id, rank 0's first)
__global__ void argmax_combine_kernel(const int* __restrict__ pairs, int world, int M, int* __restrict__ out) {
    pdl_enter();
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int bid = pairs[m * 2];
    float bv = __int_as_float(pairs[m * 2 + 1]);
    for (int w = 1; w < world; ++w) {
        const int id = pairs[(w * M + m) * 2];
        const float v = __int_as_float(pairs[(w * M + m) * 2 + 1]);
        if (v > bv || (v == bv && id < bid)) {
            bv = v;
            bid = id;
        }
    }
    out[m] = bid;
}

// ------------------------------------------------------------------ EXL3 prefill helpers
// exl3_attn._unpack_kernel: the trellis int16 [K/16][N/16][16 * bits] (read as little-endian
// u32 words) -> W_q fp16 [K][N] in output order: each element's stream position, its 16-bit
// window state and the mul1 codebook value (1024 + the four bytes of state * 0x83DCD12D) times
// scale plus bias, exact in fp32, rounded once to fp16. One block per (k tile, 128 columns).
__global__ void exl3_unpack_kernel(const uint32_t* __restrict__ tr, int NT, int N, int bits, __half* __restrict__ W) {
    pdl_enter();
    const int kt = blockIdx.x, c0 = blockIdx.y * 128;
    const int nw = 8 * bits;
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x) {
        const int r = i / 128, c = c0 + i % 128;
        const int nt = c / 16, col = c % 16;
        const int j = (r & 1) | (((r >> 3) & 1) << 1) | ((col >> 3) << 2);
        const int lane = ((col & 7) << 2) | ((r >> 1) & 3);
        const int p = lane * 8 + j;
        const uint32_t* base = tr + ((int64_t)kt * NT + nt) * nw;
        const int first = (p + 1) * bits - 16 + 32 * nw;
        const int i0 = (first / 32) % nw, i1 = (i0 + 1) % nw;
        const uint32_t off = (uint32_t)(first % 32);
        const uint32_t w0 = base[i0], w1 = base[i1];
        const uint32_t hi = off >= 16 ? w0 << (off - 16) : w0 >> (16 - off);
        const uint32_t lo = off > 16 ? w1 >> (48 - off) : 0u;
        const uint32_t state = (hi | lo) & 0xffffu;
        const uint32_t x = state * 0x83DCD12Du;
        const uint32_t h = 1024u + (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
        const float v = (float)h * 0.00676727294921875f + -10.3828125f;
        W[((int64_t)kt * 16 + r) * N + c] = __float2half_rn(v);
    }
}

// fp4_moe.build_routing's member codes for the EXL3 kernels: a block's pair p -> (p / K) * 32 +
// p % K (row, pick), -1 kept; and the used blocks' count (block_slot >= 0 is a prefix).
__global__ void member_codes_kernel(const int* __restrict__ pair, int n, int K, int* __restrict__ out) {
    pdl_enter();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pair[i];
    out[i] = p >= 0 ? (p / K) * 32 + p % K : -1;
}

__global__ void count_live_kernel(const int* __restrict__ block_slot, int nb, int* __restrict__ out) {
    pdl_enter();
    __shared__ int wtot[32];
    int total = 0, sum = 0;
    for (int b0 = 0; b0 < nb; b0 += blockDim.x) {
        const int b = b0 + threadIdx.x;
        block_rank(b < nb && block_slot[b] >= 0, wtot, total);
        sum += total;
    }
    if (threadIdx.x == 0) out[0] = sum;
}


// ------------------------------------------------------------------ load-time quantization
// weights._fp8_quant for one rank's kv_b rows (fp16 [rows][512], rows a multiple of 128): per
// 128x128 block s = max(amax, 1e-12) / 448, q = fp8(w / s); the scales expanded to a row's four
// blocks [rows][4] (kvb_fp8.HeadFP8.from_rows). One block per (128 rows, 128 columns).
__global__ void __launch_bounds__(256) quant_kvb_kernel(const __half* __restrict__ w, uint8_t* __restrict__ q,
                                                        float* __restrict__ s_rows) {
    pdl_enter();
    __shared__ float red[32];
    const int rb = blockIdx.x, cb = blockIdx.y;
    float am = 0.f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        const int r = rb * 128 + i / 128, c = cb * 128 + i % 128;
        am = fmaxf(am, fabsf(__half2float(w[(int64_t)r * kLat + c])));
    }
    am = warp_max(am);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        v = warp_max(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float sc = fmaxf(red[0], 1e-12f) / 448.f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        const int r = rb * 128 + i / 128, c = cb * 128 + i % 128;
        q[(int64_t)r * kLat + c] = to_e4m3(__half2float(w[(int64_t)r * kLat + c]) / sc);
    }
    if (threadIdx.x < 128) s_rows[(int64_t)(rb * 128 + threadIdx.x) * 4 + cb] = sc;
}

// tp._q8_rows(kind="int8"): per row and 128 values sc = max(amax / 127, 1e-12),
// q = clamp(rint(x / sc), -127, 127). One warp per (row, 128 values).
__global__ void quant_int8_rows_kernel(const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ q,
                                       float* __restrict__ s, int H) {
    pdl_enter();
    const int r = blockIdx.x, b = blockIdx.y, l = threadIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)r * H + b * 128 + 4 * l;
    float v[4], am = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(xr[i]);
        am = fmaxf(am, fabsf(v[i]));
    }
    am = warp_max(am);
    const float sc = fmaxf(am / 127.f, 1e-12f);
#pragma unroll
    for (int i = 0; i < 4; ++i) q[(int64_t)r * H + b * 128 + 4 * l + i] = (int8_t)fminf(fmaxf(rintf(v[i] / sc), -127.f), 127.f);
    if (l == 0) s[(int64_t)r * (H / 128) + b] = sc;
}

// rows t of x -> rows t * G + g of out for every g (an Exl3Matrix of G groups over one input)
__global__ void dup_rows_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, __nv_bfloat16* __restrict__ out,
                                int K, int G) {
    pdl_enter();
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        const __nv_bfloat16 v = x[(int64_t)t * ldx + c];
        for (int g = 0; g < G; ++g) out[((int64_t)t * G + g) * K + c] = v;
    }
}

// Per row of this rank's logits [M][n]: (max, sum exp((x - max) / temp)) into out[r * ldo],
// out[r * ldo + 1] -- with the rows' candidates (topk_kernel) the host's nucleus over the
// pair's whole vocabulary. temp a device scalar (a captured graph replays at any temperature).
// temp: row r's temperature at temp[r * temp_ld] (0: one for every row)
__global__ void __launch_bounds__(1024) row_lse_kernel(const float* __restrict__ x, int64_t ldx, int n,
                                                       const float* __restrict__ temp, int temp_ld, float* __restrict__ out,
                                                       int64_t ldo) {
    pdl_enter();
    __shared__ float red[32];
    const float* row = x + (int64_t)blockIdx.x * ldx;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, row[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        v = warp_max(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    const float inv = 1.f / fmaxf(temp[(int64_t)blockIdx.x * temp_ld], 1e-6f);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += expf((row[i] - m) * inv);
    s = block_sum(s, red);
    if (threadIdx.x == 0) {
        out[(int64_t)blockIdx.x * ldo] = m;
        out[(int64_t)blockIdx.x * ldo + 1] = s;
    }
}
// x [T][ldx] bf16 <- x - alpha r (r . x) a row: one direction taken out of a residual write (the
// abliteration of tidelm-glm's ablit.rs: W' = W - alpha r r^T W on every writer of a layer is this
// on the layer's summed output). The dot in fp32 by the fixed-tree block sum, so both ranks of a
// pair, which apply it to the same gathered rows, get the same bits.
__global__ void __launch_bounds__(1024) rank1_kernel(__nv_bfloat16* __restrict__ x, int64_t ldx,
                                                     const float* __restrict__ r, float alpha, int H) {
    pdl_enter();
    __shared__ float scratch[32];
    __nv_bfloat16* xr = x + (int64_t)blockIdx.x * ldx;
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) acc = fmaf(r[i], bf(xr[i]), acc);
    const float ad = alpha * block_sum(acc, scratch);
    for (int i = threadIdx.x; i < H; i += blockDim.x) xr[i] = to_bf(bf(xr[i]) - ad * r[i]);
}

// The weights a coming kernel reads into L2 from a parallel graph branch while the main stream runs
// work that leaves the memory idle (the attention core, a RoCE gather's wait): `count` runs of `run`
// bytes, `stride` apart from `base` - one run a warp's range of a decode GEMV, so each of its warps
// finds the start of its range in L2. TMA bulk prefetches (cp.async.bulk.prefetch.L2), 32 KB a
// request; the DS engine's l2_prefetch_kernel with strided runs (2026-10-27).
__global__ void __launch_bounds__(128) l2_runs_kernel(uint64_t base, uint64_t run, uint64_t stride, int count) {
    constexpr uint64_t CH = 32u << 10;
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x, workers = gridDim.x * blockDim.x;
    const uint64_t len = run & ~static_cast<uint64_t>(15u);
    uint64_t j = 0;
    for (int r = 0; r < count; ++r)
        for (uint64_t off = 0; off < len; off += CH, ++j) {
            if (j % workers != w) continue;
            const uint32_t n = static_cast<uint32_t>(min(CH, len - off));
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + r * stride + off), "r"(n) : "memory");
        }
}

// The routed experts' coming GEMV into L2 the same way, its picks known only on the device: pick[i] an
// arena slot (or -1), tp[slot] its trellis, k2[slot] its half-bits a value. The grouped kernel's warps
// read `ranges` contiguous k ranges of a matrix of K x N (N / 16 tiles of 16 x 16 values, 16 * k2 bytes
// each, k-major); each range's first `run` bytes are requested. Repeated picks prefetch twice (harmless).
__global__ void __launch_bounds__(128) l2_experts_kernel(const int* __restrict__ pick, int P, const int64_t* __restrict__ tp,
                                                         const int* __restrict__ k2, int K, int N, int ranges,
                                                         uint64_t run) {
    constexpr uint64_t CH = 32u << 10;
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x, workers = gridDim.x * blockDim.x;
    uint64_t j = 0;
    for (int i = 0; i < P; ++i) {
        const int slot = pick[i];
        if (slot < 0) continue;
        const uint64_t bytes = (uint64_t)(K / 16) * (N / 16) * 16 * k2[slot];
        const uint64_t stride = bytes / ranges, len = min(run, stride) & ~static_cast<uint64_t>(15u);
        const uint64_t base = static_cast<uint64_t>(tp[slot]);
        for (int r = 0; r < ranges; ++r)
            for (uint64_t off = 0; off < len; off += CH, ++j) {
                if (j % workers != w) continue;
                const uint32_t n = static_cast<uint32_t>(min(CH, len - off));
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + r * stride + off), "r"(n) : "memory");
            }
    }
}
}  // namespace

// ---------------------------------------------------------------- C ABI

extern "C" {

int glm_embed(void* stream, const void* ids, int T, const void* q, const void* sc, int lo, int vl, int H, const void* pos,
              void* out) {
    pdl_launch(embed_kernel, T, 256, 0, GST, static_cast<const int*>(ids), static_cast<const int8_t*>(q),
                                     static_cast<const float*>(sc), lo, vl, H, static_cast<const int*>(pos),
                                     static_cast<float*>(out));
    GRET;
}

int glm_sum2(void* stream, const void* g, int64_t n, void* out_bf16, void* out_f32) {
    pdl_launch(sum2_kernel, (unsigned)((n + 255) / 256), 256, 0, GST, static_cast<const float*>(g), n,
                                                              static_cast<__nv_bfloat16*>(out_bf16),
                                                              static_cast<float*>(out_f32));
    GRET;
}

int glm_f32_bf16(void* stream, const void* x, void* y, int64_t n) {
    pdl_launch(f32_bf16_kernel, (unsigned)((n + 255) / 256), 256, 0, GST, static_cast<const float*>(x),
                                                                  static_cast<__nv_bfloat16*>(y), n);
    GRET;
}

int glm_unshard(void* stream, const void* src, int64_t rank_stride, void* dst, int M, int n, int world, int esz) {
    const int64_t total = (int64_t)world * M * n;
    const unsigned g = (unsigned)((total + 255) / 256);
    if (!rank_stride) rank_stride = (int64_t)M * n;
    if (esz == 2) pdl_launch(unshard_kernel<uint16_t>, g, 256, 0, GST, static_cast<const uint16_t*>(src), rank_stride, static_cast<uint16_t*>(dst), M, n, world);
    else if (esz == 4) pdl_launch(unshard_kernel<uint32_t>, g, 256, 0, GST, static_cast<const uint32_t*>(src), rank_stride, static_cast<uint32_t*>(dst), M, n, world);
    else return (int)cudaErrorInvalidValue;
    GRET;
}

int glm_rmsnorm(void* stream, void* h, int64_t ldh, const void* a, int64_t lda, const void* w, void* y, int64_t ldy,
                int T, int N, float eps) {
    if (N > 12 * 1024) return (int)cudaErrorInvalidValue;
    const int nt = N >= 4096 ? 1024 : N >= 1024 ? 256 : 128;
    if (N > 12 * nt) return (int)cudaErrorInvalidValue;
    if (a)
        pdl_launch(rmsnorm_kernel<true>, T, nt, 0, GST, static_cast<__nv_bfloat16*>(h), ldh, static_cast<const __nv_bfloat16*>(a),
                                                 lda, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y),
                                                 ldy, N, eps);
    else
        pdl_launch(rmsnorm_kernel<false>, T, nt, 0, GST, static_cast<__nv_bfloat16*>(h), ldh, nullptr, 0,
                                                  static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y),
                                                  ldy, N, eps);
    GRET;
}

int glm_rope(void* stream, const void* x, int64_t xt, int64_t xh, void* y, int64_t yt, int64_t yh, int T, int H,
             const void* cos, const void* sin, const void* pos) {
    pdl_launch(rope_kernel, dim3(T, (H + 7) / 8), 256, 0, GST, static_cast<const __nv_bfloat16*>(x), xt, xh,
                                                       static_cast<__nv_bfloat16*>(y), yt, yh, H,
                                                       static_cast<const __nv_bfloat16*>(cos),
                                                       static_cast<const __nv_bfloat16*>(sin), static_cast<const int*>(pos));
    GRET;
}

int glm_kv_fp8(void* stream, const void* x, int64_t ldx, const void* rows, void* q, void* s, int T, int d) {
    if (d % 128) return (int)cudaErrorInvalidValue;
    pdl_launch(kv_fp8_kernel, dim3(T, d / 128), 32, 0, GST, static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const int*>(rows),
                                                    static_cast<uint8_t*>(q), static_cast<float*>(s), d);
    GRET;
}

int glm_rows_copy(void* stream, const void* x, int64_t ldx, const void* rows, void* out, int T, int d) {
    pdl_launch(rows_copy_kernel, T, 64, 0, GST, static_cast<const __nv_bfloat16*>(x), ldx, static_cast<const int*>(rows),
                                        static_cast<__nv_bfloat16*>(out), d);
    GRET;
}

int glm_positions(void* stream, const void* pos0, int T, int rank, int world, int spare, void* pos, void* rows, void* bound,
                  const void* lanes, int nl, void* koff) {
    if (lanes && (nl < 1 || nl > kMaxLanes)) return (int)cudaErrorInvalidValue;
    pdl_launch(positions_kernel, (T + 127) / 128, 128, 0, GST, static_cast<const int*>(pos0), T, rank, world, spare,
                                                       static_cast<int*>(pos), static_cast<int*>(rows),
                                                       static_cast<int*>(bound), static_cast<const int*>(lanes), nl,
                                                       static_cast<int*>(koff));
    GRET;
}

int glm_index_key(void* stream, const void* k, int64_t ldk, const void* w, const void* b, const void* cos, const void* sin,
                  const void* pos, const void* rows, void* q, void* s, int T) {
    pdl_launch(index_key_kernel, (T + 3) / 4, 128, 0, GST, static_cast<const __nv_bfloat16*>(k), ldk, static_cast<const float*>(w),
                                                   static_cast<const float*>(b), static_cast<const __nv_bfloat16*>(cos),
                                                   static_cast<const __nv_bfloat16*>(sin), static_cast<const int*>(pos),
                                                   static_cast<const int*>(rows), static_cast<uint8_t*>(q),
                                                   static_cast<float*>(s), T);
    GRET;
}

int glm_index_scores(void* stream, const void* q, const void* w, float wscale, const void* kq, const void* ks,
                     const void* pos, int T, int nmax, int stride, int off, int split_n, float inv_d, void* out,
                     int64_t ldo, const void* lanes, int nl) {
    if (lanes && (nl < 1 || nl > kMaxLanes || split_n)) return (int)cudaErrorInvalidValue;
    // keys a block: a block's 8 warps take 8 rows, so a decode pass of 1-3 rows leaves most of them
    // loading only; more, smaller blocks then keep more of the GPU busy (a 120K context's 60K keys,
    // 1 row: 67.5 -> 51.2 us at 128 a block; 4 rows 75.5 -> 62.7 at 256; 8 rows 95 -> 88 at 256).
    // A key's arithmetic does not depend on its block: the same scores, byte for byte.
    auto launch = [&](auto kernel, int keys) {
        dim3 grid((unsigned)((nmax + keys - 1) / keys), (unsigned)((T + kIRows - 1) / kIRows), lanes ? (unsigned)nl : 1u);
        pdl_launch(kernel, grid, 256, 0, GST, static_cast<const __nv_bfloat16*>(q), static_cast<const float*>(w), wscale,
                   static_cast<const uint8_t*>(kq), static_cast<const float*>(ks), static_cast<const int*>(pos), T, nmax,
                   stride, off, split_n, inv_d, static_cast<float*>(out), ldo, static_cast<const int*>(lanes));
    };
    if (T <= 3) launch(index_scores_kernel<128>, 128);
    else if (T <= 8) launch(index_scores_kernel<256>, 256);
    else launch(index_scores_kernel<512>, 512);
    GRET;
}

int glm_topk(void* stream, const void* s, int64_t lds, const void* pos, int stride, int off, int nmax, int K, int T,
             void* idx, int64_t ldi, void* val, int64_t ldv) {
    pdl_launch(topk_kernel, T, kTk, 0, GST, static_cast<const float*>(s), lds, static_cast<const int*>(pos), stride, off, nmax, K,
                                    static_cast<int*>(idx), ldi, static_cast<float*>(val), ldv);
    GRET;
}

int glm_split_share(void* stream, const void* g, int64_t rank_stride, int64_t row_stride, int64_t val_off, void* idx,
                    int64_t ldi, int rank, int K, int T) {
    pdl_launch(split_share_kernel, T, kTk, 0, GST, static_cast<const uint8_t*>(g), rank_stride, row_stride, val_off,
                                           static_cast<int*>(idx), ldi, rank, K);
    GRET;
}

int glm_absorb_q(void* stream, const void* q, int64_t ldq_t, int64_t ldq_h, const void* w, const void* ws, void* out,
                 int64_t ldo_t, int64_t ldo_h, int T, int hl) {
    if (T < 1) return (int)cudaErrorInvalidValue;
    for (int t0 = 0; t0 < T; t0 += 8)
        pdl_launch(absorb_q_kernel, dim3(hl, kLat / 128), 256, 0, GST, static_cast<const __nv_bfloat16*>(q) + t0 * ldq_t, ldq_t, ldq_h,
                                                               static_cast<const uint8_t*>(w), static_cast<const float*>(ws),
                                                               static_cast<__nv_bfloat16*>(out) + t0 * ldo_t, ldo_t, ldo_h,
                                                               T - t0 < 8 ? T - t0 : 8);
    GRET;
}

int glm_absorb_o(void* stream, const void* o, int64_t ldo_t, int64_t ldo_h, const void* w, const void* ws, void* out,
                 int64_t ld_out, int T, int hl) {
    if (T < 1) return (int)cudaErrorInvalidValue;
    // 16 rows a launch (the kernel stages them in shared memory); each row's sums are its own
    for (int t0 = 0; t0 < T; t0 += 16)
        pdl_launch(absorb_o_kernel, dim3(hl, kVd / kAbsV), 256, 0, GST, static_cast<const __nv_bfloat16*>(o) + t0 * ldo_t, ldo_t,
                                                            ldo_h, static_cast<const uint8_t*>(w), static_cast<const float*>(ws),
                                                            static_cast<__nv_bfloat16*>(out) + t0 * ld_out, ld_out,
                                                            T - t0 < 16 ? T - t0 : 16);
    GRET;
}

int glm_kvb_dequant(void* stream, const void* w, const void* ws, void* out, int64_t rows) {
    const int64_t n = rows * kLat;
    pdl_launch(kvb_dequant_kernel, (unsigned)((n + 255) / 256), 256, 0, GST, static_cast<const uint8_t*>(w),
                                                                     static_cast<const float*>(ws),
                                                                     static_cast<__nv_bfloat16*>(out), rows);
    GRET;
}

int glm_mla_smem(int mt) { return mt == 2 ? mla_smem<2>() : mla_smem<1>(); }

int glm_mla(void* stream, int mt, const void* q, int64_t q_ldg, int64_t q_ldt, int64_t q_ldh, int hg, const void* cq,
            const void* cs, const void* cr, int split_n, const void* idx, int64_t ld_idx, int K, int ascending, const void* bound, int T,
            int H, float scale, int nsplit, int chunk, void* part_m, void* part_l, void* part_acc, void* out,
            int64_t o_ldt, int64_t o_ldh, const void* koff) {
    if (mt != 1 && mt != 2) return (int)cudaErrorInvalidValue;
    if (koff && split_n) return (int)cudaErrorInvalidValue;
    if (chunk % kMNT) return (int)cudaErrorInvalidValue;
    const int hb = 16 * mt;
    dim3 grid((unsigned)((H + hb - 1) / hb), (unsigned)T, (unsigned)nsplit);
    auto args = [&](auto kernel, int smem) {
        static bool set1 = false, set2 = false;
        bool& set = smem == mla_smem<2>() ? set2 : set1;
        if (!set) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            set = true;
        }
        pdl_launch(kernel, grid, 256, smem, GST, static_cast<const __nv_bfloat16*>(q), q_ldg, q_ldt, q_ldh, hg,
                                         static_cast<const uint8_t*>(cq), static_cast<const float*>(cs),
                                         static_cast<const __nv_bfloat16*>(cr), split_n, static_cast<const int*>(idx),
                                         ld_idx, K, ascending, static_cast<const int*>(bound), H, scale, chunk,
                                         static_cast<float*>(part_m), static_cast<float*>(part_l),
                                         static_cast<float*>(part_acc), static_cast<__nv_bfloat16*>(out), o_ldt, o_ldh,
                                         static_cast<const int*>(koff));
    };
    if (mt == 2) args(mla_kernel<2>, mla_smem<2>());
    else args(mla_kernel<1>, mla_smem<1>());
    GRET;
}

int glm_mla_combine(void* stream, const void* pm, const void* pl, const void* pacc, int T, int H, int NS, void* out,
                    int64_t o_ldt, int64_t o_ldh, void* own_o, void* own_l, void* send, int64_t row_bytes, int hl,
                    int lo) {
    pdl_launch(mla_combine_kernel, dim3(T, H), 128, 0, GST, static_cast<const float*>(pm), static_cast<const float*>(pl),
                                                    static_cast<const float*>(pacc), H, NS,
                                                    static_cast<__nv_bfloat16*>(out), o_ldt, o_ldh,
                                                    static_cast<float*>(own_o), static_cast<float*>(own_l),
                                                    static_cast<uint8_t*>(send), row_bytes, hl, lo);
    GRET;
}

int glm_mla_merge(void* stream, const void* own_o, const void* own_l, const void* recv, int64_t row_bytes, int T, int hl,
                  void* out) {
    pdl_launch(mla_merge_kernel, dim3(T, hl), 128, 0, GST, static_cast<const float*>(own_o), static_cast<const float*>(own_l),
                                                   static_cast<const uint8_t*>(recv), row_bytes, hl,
                                                   static_cast<__nv_bfloat16*>(out));
    GRET;
}

int glm_router(void* stream, const void* logits, const void* bias, const void* allowed, const void* lut, int T, int K,
               float scale, void* ids, void* w, const void* resident, void* hot, const void* ex_ids, const void* ex_w,
               void* demand, void* acc, const void* live, void* lane_acc, void* lane_demand, int64_t lane_stride,
               const void* row_lane) {
    if (K > 8) return (int)cudaErrorInvalidValue;
    pdl_launch(router_kernel, T, kE, 0, GST, static_cast<const float*>(logits), static_cast<const float*>(bias),
                                     static_cast<const uint8_t*>(allowed), static_cast<const int*>(lut), K, scale,
                                     static_cast<int*>(ids), static_cast<float*>(w), static_cast<const uint8_t*>(resident),
                                     static_cast<float*>(hot), static_cast<const int*>(ex_ids),
                                     static_cast<const float*>(ex_w), static_cast<float*>(demand),
                                     static_cast<double*>(acc), static_cast<const int*>(live),
                                     static_cast<double*>(lane_acc), static_cast<float*>(lane_demand), lane_stride,
                                     static_cast<const int*>(row_lane));
    GRET;
}

int glm_swiglu(void* stream, const void* g, int64_t ldg, const void* u, int64_t ldu, void* y, int64_t ldy, int T, int n) {
    pdl_launch(swiglu_kernel, dim3((n + 255) / 256, T), 256, 0, GST, static_cast<const __nv_bfloat16*>(g), ldg,
                                                             static_cast<const __nv_bfloat16*>(u), ldu,
                                                             static_cast<__nv_bfloat16*>(y), ldy, n);
    GRET;
}

int glm_moe_out(void* stream, const void* routed, const void* shared, void* y, int64_t n) {
    pdl_launch(moe_out_kernel, (unsigned)((n + 255) / 256), 256, 0, GST, static_cast<const float*>(routed),
                                                                 static_cast<const float*>(shared),
                                                                 static_cast<__nv_bfloat16*>(y), n);
    GRET;
}

int glm_argmax(void* stream, const void* x, int64_t ldx, int M, int n, int offset, void* pairs) {
    pdl_launch(argmax_kernel, M, 1024, 0, GST, static_cast<const float*>(x), ldx, n, offset, static_cast<int*>(pairs));
    GRET;
}

int glm_argmax_combine(void* stream, const void* pairs, int world, int M, void* out) {
    pdl_launch(argmax_combine_kernel, (M + 127) / 128, 128, 0, GST, static_cast<const int*>(pairs), world, M,
                                                            static_cast<int*>(out));
    GRET;
}

int glm_exl3_unpack(void* stream, const void* trellis, int K, int N, int bits, void* w) {
    if (K % 16 || N % 128) return (int)cudaErrorInvalidValue;
    pdl_launch(exl3_unpack_kernel, dim3(K / 16, N / 128), 256, 0, GST, static_cast<const uint32_t*>(trellis), N / 16, N, bits,
                                                               static_cast<__half*>(w));
    GRET;
}

int glm_member_codes(void* stream, const void* pair, int n, int K, void* out) {
    pdl_launch(member_codes_kernel, (n + 255) / 256, 256, 0, GST, static_cast<const int*>(pair), n, K, static_cast<int*>(out));
    GRET;
}

int glm_count_live(void* stream, const void* block_slot, int nb, void* out) {
    pdl_launch(count_live_kernel, 1, 1024, 0, GST, static_cast<const int*>(block_slot), nb, static_cast<int*>(out));
    GRET;
}

int glm_quant_kvb(void* stream, const void* w, int rows, void* q, void* s_rows) {
    if (rows % 128) return (int)cudaErrorInvalidValue;
    pdl_launch(quant_kvb_kernel, dim3(rows / 128, kLat / 128), 256, 0, GST, static_cast<const __half*>(w),
                                                                  static_cast<uint8_t*>(q), static_cast<float*>(s_rows));
    GRET;
}

int glm_quant_int8_rows(void* stream, const void* x, int rows, int H, void* q, void* s) {
    if (H % 128) return (int)cudaErrorInvalidValue;
    pdl_launch(quant_int8_rows_kernel, dim3(rows, H / 128), 32, 0, GST, static_cast<const __nv_bfloat16*>(x),
                                                                static_cast<int8_t*>(q), static_cast<float*>(s), H);
    GRET;
}

int glm_dup_rows(void* stream, const void* x, int64_t ldx, void* out, int T, int K, int G) {
    pdl_launch(dup_rows_kernel, T, 256, 0, GST, static_cast<const __nv_bfloat16*>(x), ldx, static_cast<__nv_bfloat16*>(out), K, G);
    GRET;
}

int glm_row_lse(void* stream, const void* x, int64_t ldx, int M, int n, const void* temp, int temp_ld, void* out,
                int64_t ldo) {
    pdl_launch(row_lse_kernel, M, 1024, 0, GST, static_cast<const float*>(x), ldx, n, static_cast<const float*>(temp), temp_ld,
                                        static_cast<float*>(out), ldo);
    GRET;
}

int glm_l2_runs(void* stream, uint64_t base, uint64_t run, uint64_t stride, int count, int grid) {
    l2_runs_kernel<<<grid, 128, 0, GST>>>(base, run, stride, count);
    GRET;
}

int glm_l2_experts(void* stream, const void* pick, int P, const void* tp, const void* k2, int K, int N, int ranges,
                   uint64_t run, int grid) {
    l2_experts_kernel<<<grid, 128, 0, GST>>>(static_cast<const int*>(pick), P, static_cast<const int64_t*>(tp),
                                             static_cast<const int*>(k2), K, N, ranges, run);
    GRET;
}

int glm_rank1(void* stream, void* x, int64_t ldx, const void* r, float alpha, int T, int H) {
    pdl_launch(rank1_kernel, T, 1024, 0, GST, static_cast<__nv_bfloat16*>(x), ldx, static_cast<const float*>(r), alpha, H);
    GRET;
}

}  // extern "C"
