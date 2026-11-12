// GLM-5.3-Flash (glm5_next) kernels the GLM-5.3 engine has no counterpart for: KDA (Kimi Delta
// Attention, the 34 linear-attention layers) and the DSA indexer's 4:1 key pools. Everything else of
// the model runs on glm.cu, the EXL3 file and ds41.cu (mHC, the FP8 / BF16 GEMVs).
//
// References: transformers' modeling_glm5_next.py (Glm5NextTextLinearAttention, recurrent_kimi_delta_
// attention, Glm5NextTextRMSNormGated, Glm5NextTextIndexer) and TensorFold's glm5_next family
// (kda.cu, sparse.py), which agree on every formula used here.
//
// Lanes: every launcher takes the pass's lane table (glm.cu positions_kernel's [L][4]: first row,
// rows, first position, cache-row offset) and a parallel [L][4] of this file's own: the lane, the
// KDA state slot it reads, the slot it writes, its pool caches' row offset. A lane's KDA state is
// double-buffered (slot 2 * lane + parity): a pass reads the committed slot and writes the other, so
// a verify pass's rejected rows are undone by replaying the kept ones from the slot it read
// (kda_replay), bit for bit what running only them would have left - the step is one __device__
// function both kernels call.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <utility>

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
extern "C" int flash_set_pdl(int on) { g_pdl = on; return 0; }
// threads a KDA value column (2 or 4): a micro-bench switch; the chain and the replay must agree, and
// the split changes the sums' order, so it is in the boot guard
static int g_kda_split = 2;
extern "C" int flash_set_kda_split(int sp) { if (sp != 2 && sp != 4) return 1; g_kda_split = sp; return 0; }

namespace {

#define GST static_cast<cudaStream_t>(stream)
#define GRET return static_cast<int>(cudaGetLastError())

typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 to_bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ uint8_t to_e4m3(float v) { return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }
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
__device__ __forceinline__ int lane_of(const int* lanes, int nl, int t) {
    for (int z = 0; z < nl; ++z)
        if (t >= lanes[4 * z] && t < lanes[4 * z] + lanes[4 * z + 1]) return z;
    return -1;
}

constexpr int kD = 128;  // KDA head dim (keys and values)

// ------------------------------------------------------------------ KDA: the short convolution
// causal_conv1d over the q | k | v projections (C channels, 4 taps, no bias): out(t) = silu(sum_tau
// w[tau] x(t - 3 + tau)), fp32 sums in tap order, one bf16 rounding (TF kda.cu, HF
// causal_conv1d_fn). x before the pass's first row comes from the lane's conv state (the last 3
// raw rows, oldest first). save_raw (null: off): a lane's raw rows [R][C], for kda_conv_replay.
__global__ void kda_conv_kernel(const bf16* __restrict__ raw, int64_t ldr, const bf16* __restrict__ w, int C,
                                const bf16* __restrict__ conv, int64_t slot_stride, const int* __restrict__ lanes, int nl,
                                const int* __restrict__ meta, bf16* __restrict__ out, int64_t ldo, bf16* __restrict__ save_raw,
                                int64_t save_lane, int R) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int i = t - lanes[4 * z];
    const bf16* cs = conv + (int64_t)meta[4 * z + 1] * slot_stride;
    float acc = 0.f;
#pragma unroll
    for (int tau = 0; tau < 4; ++tau) {
        const int j = i - 3 + tau;
        const float x = j >= 0 ? bf(raw[(int64_t)(t - i + j) * ldr + c]) : bf(cs[(int64_t)(3 + j) * C + c]);
        acc = fmaf(bf(w[c * 4 + tau]), x, acc);
    }
    out[(int64_t)t * ldo + c] = to_bf(acc / (1.f + expf(-acc)));
    if (save_raw && i < R) save_raw[(int64_t)meta[4 * z] * save_lane + (int64_t)i * C + c] = raw[(int64_t)t * ldr + c];
}

// The conv state after the pass: the last 3 raw rows of [state; the group's rows], into the slot
// the lane writes.
__global__ void kda_conv_state_kernel(const bf16* __restrict__ raw, int64_t ldr, int C, bf16* __restrict__ conv,
                                      int64_t slot_stride, const int* __restrict__ lanes, const int* __restrict__ meta) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y;
    if (c >= C) return;
    const int row0 = lanes[4 * z], n = lanes[4 * z + 1];
    const bf16* cin = conv + (int64_t)meta[4 * z + 1] * slot_stride;
    bf16* cout = conv + (int64_t)meta[4 * z + 2] * slot_stride;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = n - 3 + j;
        cout[(int64_t)j * C + c] = s >= 0 ? raw[(int64_t)(row0 + s) * ldr + c] : cin[(int64_t)(3 + s) * C + c];
    }
}

// A rollback's conv state for every layer: the last 3 raw rows of [the state the pass read; its
// first `keep` raw rows], from the slot it read into the one it wrote. Grid (C / 256, layers).
__global__ void kda_conv_replay_kernel(bf16* __restrict__ conv, int64_t layer_stride, int64_t slot_stride, int C, int sin,
                                       int sout, const bf16* __restrict__ save_raw, int64_t save_layer, int keep) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (c >= C) return;
    const bf16* cin = conv + (int64_t)l * layer_stride + (int64_t)sin * slot_stride;
    bf16* cout = conv + (int64_t)l * layer_stride + (int64_t)sout * slot_stride;
    const bf16* sr = save_raw + (int64_t)l * save_layer;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = keep - 3 + j;
        cout[(int64_t)j * C + c] = s >= 0 ? sr[(int64_t)s * C + c] : cin[(int64_t)(3 + s) * C + c];
    }
}

// ------------------------------------------------------------------ KDA: the delta rule
// One step of recurrent_kimi_delta_attention for value column v of one head, its keys split over SP
// threads (lanes SP apart in a warp... adjacent: tid = SP * v + part): S[j] is S[k0 + j][v] of the
// head's state [k][v] (fp32), k0 = part * KS, kk / gg the step's normalized key and per-key decay
// exp(g) (shared memory):
//   S <- S * gg (along k);  kv = sum_k S[k] kk[k];  delta = (v - kv) * beta;  S <- S + kk delta.
// The chain and the replay both run it, so a replay's state is the chain's, bit for bit.
template <int SP>
__device__ __forceinline__ void kda_step(float (&S)[kD / SP], const float* kk, const float* gg, float vv, float beta) {
    constexpr int KS = kD / SP;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int j = 0; j < KS; j += 4) {
        S[j] = S[j] * gg[j];
        S[j + 1] = S[j + 1] * gg[j + 1];
        S[j + 2] = S[j + 2] * gg[j + 2];
        S[j + 3] = S[j + 3] * gg[j + 3];
        a0 = fmaf(S[j], kk[j], a0);
        a1 = fmaf(S[j + 1], kk[j + 1], a1);
        a2 = fmaf(S[j + 2], kk[j + 2], a2);
        a3 = fmaf(S[j + 3], kk[j + 3], a3);
    }
    float kv = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int o = 1; o < SP; o <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, o);
    const float delta = (vv - kv) * beta;
#pragma unroll
    for (int j = 0; j < KS; ++j) S[j] = fmaf(kk[j], delta, S[j]);
}

// sum over the block of two values (NW warps)
template <int NW>
__device__ __forceinline__ float2 block_sum2(float a, float b, float2* red) {
    a = warp_sum(a);
    b = warp_sum(b);
    const int w = threadIdx.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[w] = make_float2(a, b);
    __syncthreads();
    float2 r = red[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) {
        r.x += red[i].x;
        r.y += red[i].y;
    }
    return r;
}

// KDA over a pass's rows, one block a (head, lane), SP threads a value column (each with 128 / SP of
// its keys' state in registers; the head's state is [k][v] fp32). A row:
//   q, k, v from the conv output (bf16); k /= sqrt(sum k^2 + 1e-6); q /= sqrt(sum q^2 + 1e-6), q *= 128^-0.5
//   g = lower * sigmoid(exp(A_log[h]) * (f + dt_bias))   (per channel; f the f_b output, bf16)
//   beta = bf16(sigmoid(b))
//   the delta-rule step (kda_step), o = bf16(sum_k S[k] q[k]) after it
//   y = bf16(w_onorm * (o * rsqrt(mean(o^2) + eps)) * sigmoid(gt))   (Glm5NextTextRMSNormGated)
// qkv [T][ldq]: q at column h * 128, k at dl + h * 128, v at 2 dl + h * 128 (dl: this rank's heads x
// 128); f, gt [T][ld*] at h * 128; b [T][ldb] at h. The state comes from the lane's read slot and
// goes to its write slot. save_* (null: off): the lane's rows' k, g (fp32), v (bf16), beta for
// kda_replay, [lane][R][hl][128] (beta [lane][R][hl]) at this layer.
template <int SP>
__global__ void __launch_bounds__(kD * SP) kda_chain_kernel(
    const bf16* __restrict__ qkv, int64_t ldq, int dl, const bf16* __restrict__ f, int64_t ldf, const bf16* __restrict__ b,
    int64_t ldb, const bf16* __restrict__ gt, int64_t ldg, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const bf16* __restrict__ onorm, float lower, float eps, float* __restrict__ state, int64_t slot_stride,
    const int* __restrict__ lanes, int nl, const int* __restrict__ meta, bf16* __restrict__ y, int64_t ldy,
    float* __restrict__ save_k, float* __restrict__ save_g, bf16* __restrict__ save_v, float* __restrict__ save_b,
    int64_t save_lane, int R, int hl) {
    pdl_enter();
    constexpr int KS = kD / SP, NW = kD * SP / 32;
    __shared__ float kk[kD], qq[kD], gg[kD];
    __shared__ float2 red[NW];
    const int h = blockIdx.x, z = blockIdx.y, tid = threadIdx.x;
    const int v = tid / SP, k0 = (tid % SP) * KS;
    if (z >= nl) return;
    const int row0 = lanes[4 * z], n = lanes[4 * z + 1];
    const int lane = meta[4 * z];
    const float* sin = state + (int64_t)meta[4 * z + 1] * slot_stride + (int64_t)h * kD * kD;
    float* sout = state + (int64_t)meta[4 * z + 2] * slot_stride + (int64_t)h * kD * kD;
    float S[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) S[j] = sin[(k0 + j) * kD + v];
    const float rate = expf(a_log[h]);
    const bool io = tid < kD;  // the threads that load a row's q / k / gate element tid
    const float dtb = io ? dt_bias[h * kD + tid] : 0.f;
    const float wn = bf(onorm[v]);
    const int64_t sl = (int64_t)lane * save_lane;
    // a row's inputs, loaded a row ahead: the next row's loads are in flight while this one runs
    // (each row otherwise starts with dependent global loads; 5.2 -> see flash_gpu kda_bench)
    struct In { bf16 q, k, v, vo, f, g; bf16 b; };
    auto load = [&](int64_t t) {
        In x;
        const bf16 z = __float2bfloat16_rn(0.f);
        x.q = io ? qkv[t * ldq + h * kD + tid] : z;
        x.k = io ? qkv[t * ldq + dl + h * kD + tid] : z;
        x.vo = io ? qkv[t * ldq + 2 * dl + h * kD + tid] : z;
        x.f = io ? f[t * ldf + h * kD + tid] : z;
        x.v = qkv[t * ldq + 2 * dl + h * kD + v];
        x.g = gt[t * ldg + h * kD + v];
        x.b = b[t * ldb + h];
        return x;
    };
    In cur = load(row0);
    for (int i = 0; i < n; ++i) {
        const int64_t t = row0 + i;
        const In nx = i + 1 < n ? load(t + 1) : cur;
        const float qv = bf(cur.q), kv = bf(cur.k);
        const bf16 vb = cur.v;
        const float2 ss = block_sum2<NW>(kv * kv, qv * qv, red);
        const float beta = rbf(sigmoidf_(bf(cur.b)));
        if (io) {
            const float kn = kv / sqrtf(ss.x + 1e-6f);
            const float qn = qv / sqrtf(ss.y + 1e-6f) * 0.08838834764831845f;
            const float g = lower * sigmoidf_(rate * (bf(cur.f) + dtb));
            kk[tid] = kn;
            qq[tid] = qn;
            gg[tid] = expf(g);
            if (save_k && i < R) {
                const int64_t o = sl + ((int64_t)i * hl + h) * kD + tid;
                save_k[o] = kn;
                save_g[o] = g;
                save_v[o] = cur.vo;
                if (tid == 0) save_b[sl / kD + (int64_t)i * hl + h] = beta;
            }
        }
        __syncthreads();
        kda_step<SP>(S, kk + k0, gg + k0, bf(vb), beta);
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
        for (int j = 0; j < KS; j += 4) {
            o0 = fmaf(S[j], qq[k0 + j], o0);
            o1 = fmaf(S[j + 1], qq[k0 + j + 1], o1);
            o2 = fmaf(S[j + 2], qq[k0 + j + 2], o2);
            o3 = fmaf(S[j + 3], qq[k0 + j + 3], o3);
        }
        float o = (o0 + o1) + (o2 + o3);
#pragma unroll
        for (int x = 1; x < SP; x <<= 1) o += __shfl_xor_sync(0xffffffffu, o, x);
        const float ob = rbf(o);
        const float2 so = block_sum2<NW>(tid % SP == 0 ? ob * ob : 0.f, 0.f, red);
        if (tid % SP == 0) {
            float yv = ob * rsqrtf(so.x * (1.f / kD) + eps);
            yv = wn * yv;
            yv = yv * sigmoidf_(bf(cur.g));
            y[t * ldy + h * kD + v] = to_bf(yv);
        }
        cur = nx;
    }
#pragma unroll
    for (int j = 0; j < KS; ++j) sout[(k0 + j) * kD + v] = S[j];
}

// A rollback for one lane over every KDA layer: the state the pass read (slot sin) through its first
// `keep` rows, from the rows kda_chain saved, into the slot it wrote (sout). Grid (hl, layers).
template <int SP>
__global__ void __launch_bounds__(kD * SP) kda_replay_kernel(float* __restrict__ state, int64_t layer_stride,
                                                             int64_t slot_stride, int sin, int sout,
                                                             const float* __restrict__ save_k, const float* __restrict__ save_g,
                                                             const bf16* __restrict__ save_v, const float* __restrict__ save_b,
                                                             int64_t save_layer, int keep, int hl) {
    pdl_enter();
    constexpr int KS = kD / SP;
    __shared__ float kk[kD], gg[kD];
    const int h = blockIdx.x, l = blockIdx.y, tid = threadIdx.x;
    const int v = tid / SP, k0 = (tid % SP) * KS;
    const float* si = state + (int64_t)l * layer_stride + (int64_t)sin * slot_stride + (int64_t)h * kD * kD;
    float* so = state + (int64_t)l * layer_stride + (int64_t)sout * slot_stride + (int64_t)h * kD * kD;
    float S[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) S[j] = si[(k0 + j) * kD + v];
    const int64_t base = (int64_t)l * save_layer;
    for (int i = 0; i < keep; ++i) {
        const int64_t o = base + ((int64_t)i * hl + h) * kD;
        __syncthreads();
        if (tid < kD) {
            kk[tid] = save_k[o + tid];
            gg[tid] = expf(save_g[o + tid]);
        }
        __syncthreads();
        kda_step<SP>(S, kk + k0, gg + k0, bf(save_v[o + v]), save_b[base / kD + (int64_t)i * hl + h]);
    }
#pragma unroll
    for (int j = 0; j < KS; ++j) so[(k0 + j) * kD + v] = S[j];
}

// ------------------------------------------------------------------ the indexer's key pools
// LayerNorm over 128 (biased variance, bf16 weight and bias), one rounding: the indexer's k_norm.
__global__ void ln128_kernel(const bf16* __restrict__ x, int64_t ldx, const bf16* __restrict__ w, const bf16* __restrict__ b,
                             bf16* __restrict__ y, int64_t ldy, int T, float eps) {
    pdl_enter();
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int t = blockIdx.x * 4 + warp;
    if (t >= T) return;
    float v[4], sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(x[(int64_t)t * ldx + 4 * l + i]);
        sum += v[i];
    }
    const float mean = warp_sum(sum) / 128.f;
    float d2 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) d2 += (v[i] - mean) * (v[i] - mean);
    const float rstd = rsqrtf(warp_sum(d2) / 128.f + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = 4 * l + i;
        y[(int64_t)t * ldy + j] = to_bf((v[i] - mean) * rstd * bf(w[j]) + bf(b[j]));
    }
}

// Pool p's key once its last token (4p + 3) is in the pass: per channel d,
//   pi = softmax_j (gate_{4p+j}[d] + ape[j][d]),  PK[d] = bf16(sum_j bf16(bf16(pi_j) * K_{4p+j}[d]))
// (Glm5NextTextIndexer's k-pool compression), stored as the GLM indexer's keys are: FP8 E4M3 codes
// [128] and an fp32 scale amax / 448, at pool row p of the lane's pool caches (meta[4z+3] on). A
// token before the pass's first row comes from the lane's ring (the last 64 positions' K | gate).
__global__ void __launch_bounds__(128) pool_kernel(const bf16* __restrict__ K, int64_t ldk, const bf16* __restrict__ G, int64_t ldg,
                                                   const bf16* __restrict__ ape, const bf16* __restrict__ ring,
                                                   int64_t ring_lane, const int* __restrict__ pos, const int* __restrict__ lanes,
                                                   int nl, const int* __restrict__ meta, uint8_t* __restrict__ pq,
                                                   float* __restrict__ ps) {
    pdl_enter();
    __shared__ float red[4];
    const int t = blockIdx.x, d = threadIdx.x;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int p = pos[t];
    if ((p & 3) != 3) return;
    const int j = p >> 2, row0 = lanes[4 * z], pos0 = lanes[4 * z + 2];
    const bf16* rl = ring + (int64_t)meta[4 * z] * ring_lane;
    float lam[4], kv[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        const int x = 4 * j + jj;
        float kx, gx;
        if (x >= pos0) {
            const int64_t r = row0 + x - pos0;
            kx = bf(K[r * ldk + d]);
            gx = bf(G[r * ldg + d]);
        } else {
            const bf16* e = rl + (int64_t)(x & 63) * 256;
            kx = bf(e[d]);
            gx = bf(e[128 + d]);
        }
        kv[jj] = kx;
        lam[jj] = gx + bf(ape[jj * 128 + d]);
    }
    const float m = fmaxf(fmaxf(lam[0], lam[1]), fmaxf(lam[2], lam[3]));
    float e[4], s = 0.f;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        e[jj] = expf(lam[jj] - m);
        s += e[jj];
    }
    float acc = 0.f;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) acc += rbf(rbf(e[jj] / s) * kv[jj]);
    const float pk = rbf(acc);
    float am = warp_max(fabsf(pk));
    if ((d & 31) == 0) red[d >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float sc = fmaxf(am / 448.f, 1e-12f);
    const int64_t row = (int64_t)meta[4 * z + 3] + j;
    pq[row * 128 + d] = to_e4m3(pk / sc);
    if (d == 0) ps[row] = sc;
}

// The pass's K | gate rows into their lanes' rings (position % 64), after pool_kernel read them.
__global__ void ring_kernel(const bf16* __restrict__ K, int64_t ldk, const bf16* __restrict__ G, int64_t ldg,
                            bf16* __restrict__ ring, int64_t ring_lane, const int* __restrict__ pos,
                            const int* __restrict__ lanes, int nl, const int* __restrict__ meta) {
    pdl_enter();
    const int t = blockIdx.x, d = threadIdx.x;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    bf16* e = ring + (int64_t)meta[4 * z] * ring_lane + (int64_t)(pos[t] & 63) * 256;
    e[d] = d < 128 ? K[(int64_t)t * ldk + d] : G[(int64_t)t * ldg + d - 128];
}

// The selected pools (ascending, -1 after them: glm.cu topk_kernel) as token positions for the MLA
// kernel: each pool's 4 tokens in order, then the incomplete tail pool's 0-3 tokens 4 * npool .. p,
// -1 up to ldo. Every pool selected (npool <= the top-k) is every position <= p: dense causal.
__global__ void expand_kernel(const int* __restrict__ pools, int64_t ldp, int kp, const int* __restrict__ pos,
                              int* __restrict__ out, int64_t ldo) {
    pdl_enter();
    const int t = blockIdx.x;
    const int* pr = pools + (int64_t)t * ldp;
    int* o = out + (int64_t)t * ldo;
    int nv = 0;
    for (int i0 = 0; i0 < kp; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const int pj = i < kp ? pr[i] : -1;
        nv += __syncthreads_count(pj >= 0);
        if (pj >= 0) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) o[4 * i + jj] = 4 * pj + jj;
        }
    }
    const int p = pos[t], tail0 = ((p + 1) >> 2) << 2;
    for (int i = 4 * nv + threadIdx.x; i < ldo; i += blockDim.x) {
        const int x = tail0 + (i - 4 * nv);
        o[i] = x <= p ? x : -1;
    }
}

}  // namespace

extern "C" {

int flash_kda_conv(void* stream, const void* raw, int64_t ldr, const void* w, int C, const void* conv, int64_t slot_stride,
                   const void* lanes, int nl, const void* meta, int T, void* out, int64_t ldo, void* save_raw,
                   int64_t save_lane, int R) {
    if (nl < 1 || C % 8) return (int)cudaErrorInvalidValue;
    pdl_launch(kda_conv_kernel, dim3((C + 255) / 256, T), 256, 0, GST, static_cast<const bf16*>(raw), ldr,
               static_cast<const bf16*>(w), C, static_cast<const bf16*>(conv), slot_stride, static_cast<const int*>(lanes), nl,
               static_cast<const int*>(meta), static_cast<bf16*>(out), ldo, static_cast<bf16*>(save_raw), save_lane, R);
    GRET;
}

int flash_kda_conv_state(void* stream, const void* raw, int64_t ldr, int C, void* conv, int64_t slot_stride, const void* lanes,
                         int nl, const void* meta) {
    pdl_launch(kda_conv_state_kernel, dim3((C + 255) / 256, nl), 256, 0, GST, static_cast<const bf16*>(raw), ldr, C,
               static_cast<bf16*>(conv), slot_stride, static_cast<const int*>(lanes), static_cast<const int*>(meta));
    GRET;
}

int flash_kda_conv_replay(void* stream, void* conv, int64_t layer_stride, int64_t slot_stride, int C, int layers, int sin,
                          int sout, const void* save_raw, int64_t save_layer, int keep) {
    pdl_launch(kda_conv_replay_kernel, dim3((C + 255) / 256, layers), 256, 0, GST, static_cast<bf16*>(conv), layer_stride,
               slot_stride, C, sin, sout, static_cast<const bf16*>(save_raw), save_layer, keep);
    GRET;
}

int flash_kda_chain(void* stream, const void* qkv, int64_t ldq, int dl, const void* f, int64_t ldf, const void* b, int64_t ldb,
                    const void* gt, int64_t ldg, const void* a_log, const void* dt_bias, const void* onorm, float lower,
                    float eps, void* state, int64_t slot_stride, const void* lanes, int nl, const void* meta, void* y,
                    int64_t ldy, void* save_k, void* save_g, void* save_v, void* save_b, int64_t save_lane, int R, int hl) {
    if (nl < 1 || dl != hl * kD) return (int)cudaErrorInvalidValue;
    auto go = [&](auto kernel, int sp) {
    pdl_launch(kernel, dim3(hl, nl), kD * sp, 0, GST, static_cast<const bf16*>(qkv), ldq, dl, static_cast<const bf16*>(f),
               ldf, static_cast<const bf16*>(b), ldb, static_cast<const bf16*>(gt), ldg, static_cast<const float*>(a_log),
               static_cast<const float*>(dt_bias), static_cast<const bf16*>(onorm), lower, eps, static_cast<float*>(state),
               slot_stride, static_cast<const int*>(lanes), nl, static_cast<const int*>(meta), static_cast<bf16*>(y), ldy,
               static_cast<float*>(save_k), static_cast<float*>(save_g), static_cast<bf16*>(save_v),
               static_cast<float*>(save_b), save_lane, R, hl);
    };
    if (g_kda_split == 4) go(kda_chain_kernel<4>, 4);
    else go(kda_chain_kernel<2>, 2);
    GRET;
}

int flash_kda_replay(void* stream, void* state, int64_t layer_stride, int64_t slot_stride, int layers, int sin, int sout,
                     const void* save_k, const void* save_g, const void* save_v, const void* save_b, int64_t save_layer,
                     int keep, int hl) {
    if (keep < 0) return (int)cudaErrorInvalidValue;
    auto go = [&](auto kernel, int sp) {
    pdl_launch(kernel, dim3(hl, layers), kD * sp, 0, GST, static_cast<float*>(state), layer_stride, slot_stride, sin, sout,
               static_cast<const float*>(save_k), static_cast<const float*>(save_g), static_cast<const bf16*>(save_v),
               static_cast<const float*>(save_b), save_layer, keep, hl);
    };
    if (g_kda_split == 4) go(kda_replay_kernel<4>, 4);
    else go(kda_replay_kernel<2>, 2);
    GRET;
}

int flash_ln128(void* stream, const void* x, int64_t ldx, const void* w, const void* b, void* y, int64_t ldy, int T, float eps) {
    pdl_launch(ln128_kernel, (T + 3) / 4, 128, 0, GST, static_cast<const bf16*>(x), ldx, static_cast<const bf16*>(w),
               static_cast<const bf16*>(b), static_cast<bf16*>(y), ldy, T, eps);
    GRET;
}

int flash_pool(void* stream, const void* K, int64_t ldk, const void* G, int64_t ldg, const void* ape, const void* ring,
               int64_t ring_lane, const void* pos, const void* lanes, int nl, const void* meta, int T, void* pq, void* ps) {
    pdl_launch(pool_kernel, T, 128, 0, GST, static_cast<const bf16*>(K), ldk, static_cast<const bf16*>(G), ldg,
               static_cast<const bf16*>(ape), static_cast<const bf16*>(ring), ring_lane, static_cast<const int*>(pos),
               static_cast<const int*>(lanes), nl, static_cast<const int*>(meta), static_cast<uint8_t*>(pq),
               static_cast<float*>(ps));
    GRET;
}

int flash_ring(void* stream, const void* K, int64_t ldk, const void* G, int64_t ldg, void* ring, int64_t ring_lane,
               const void* pos, const void* lanes, int nl, const void* meta, int T) {
    pdl_launch(ring_kernel, T, 256, 0, GST, static_cast<const bf16*>(K), ldk, static_cast<const bf16*>(G), ldg,
               static_cast<bf16*>(ring), ring_lane, static_cast<const int*>(pos), static_cast<const int*>(lanes), nl,
               static_cast<const int*>(meta));
    GRET;
}

int flash_expand(void* stream, const void* pools, int64_t ldp, int kp, const void* pos, void* out, int64_t ldo, int T) {
    if (ldo < 4 * (int64_t)kp + 3) return (int)cudaErrorInvalidValue;
    pdl_launch(expand_kernel, T, 512, 0, GST, static_cast<const int*>(pools), ldp, kp, static_cast<const int*>(pos),
               static_cast<int*>(out), ldo);
    GRET;
}

}  // extern "C"
