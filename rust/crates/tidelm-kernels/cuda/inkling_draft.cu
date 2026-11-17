// Inkling 2's speculative-decoding drafter (mtp.safetensors, "DSpark"): the kernels its five dense
// layers and its Markov head have no counterpart for in the engine. The projections run on ds41.cu's
// FP8 / BF16 GEMVs, SwiGLU on glm.cu's.
//
// Reference: SGLang branch inkling-2-small-support (models/inkling2_mtp.py, kernels/ops/inkling2/mtp.py,
// models/dflash.py, kernels/ops/attention/table_qk_norm_rope.py, kernels/ops/speculative/dspark/
// markov_sample.py), as read into docs/inkling-dspark.md. Rounding points are
// the spec's section 8: fp32 inside a norm, a rotation, the attention's logits and its weighted value
// sum; bf16 where the reference stores a tensor.
//
// The drafter is a plain pre-norm transformer over a block of 8 rows [anchor, mask x 7] at positions
// P .. P + 7: per-head weighted RMSNorm on q and k, neox RoPE (theta 1e6) over the whole head, one
// softmax over [context keys | the block's own 8 keys] with no bias, sink, gate or mask (every block
// row sees the same keys). A layer's context keys / values are made from the target's hidden states
// (inkd_mix, a GEMM, inkd_ctx_kv) and live in a ring by position, as inkling.cu's window layers'
// do; a block's own keys / values are written to the same ring at P .. P + 7 and overwritten by the
// next pass's context rows or the next block, so nothing is ever undone.
//
// Lanes: inkd_mix and inkd_ctx_kv take the target pass's tables (inkling.cu: lanes [L][4] = first
// row, rows, first position, cache-row offset; meta [L][4], the lane first) and read rows' lanes and
// positions on the device. The draft round's kernels take one lane's ring and its position as
// arguments.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <limits.h>
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
extern "C" int inkd_set_pdl(int on) { g_pdl = on; return 0; }

namespace {

#define GST static_cast<cudaStream_t>(stream)
#define GRET return static_cast<int>(cudaGetLastError())

typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 to_bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// A block's sum by a fixed tree; every thread gets it. scratch: [32].
__device__ float block_sum(float v, float* scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    return warp_sum(lane < nw ? scratch[lane] : 0.f);
}
__device__ __forceinline__ int lane_of(const int* lanes, int nl, int t) {
    for (int z = 0; z < nl; ++z)
        if (t >= lanes[4 * z] && t < lanes[4 * z] + lanes[4 * z + 1]) return z;
    return -1;
}

constexpr int kD = 128;       // a head's dims
constexpr int kG = 6;         // query heads a key / value head: query head h reads key head h / 6
constexpr int kHK = 8;        // key / value heads
constexpr int kHQ = kHK * kG; // query heads
constexpr int kRows = 8;      // a block's rows: the anchor and 7 masks
constexpr int kTaps = 8;      // target hidden states a position
constexpr int kH = 4608;      // hidden size
constexpr int kR = 256;       // the Markov head's rank
constexpr int kFRing = 64;    // positions a lane's f_rnn ring holds
constexpr int kPer = 5;       // values of a hidden row a thread of a 1,024-thread block keeps
constexpr int kRowKV = kHK * 2 * kD;  // a cache row: [8 heads][K 128 | V 128]

// ------------------------------------------------------------------ the target's features
// mix_rmsnorm (kernels/ops/inkling2/mtp.py): m = sum_tap taps[tap][t] * alpha[tap] in fp32, the taps
// in order; out = bf16(m * rsqrt(sum(m^2) / H + eps) * w). taps: [8] planes `plane` elements apart,
// each [t][H]. mode 0: out [t][H]. mode 1: out is the lanes' f_rnn rings [lanes][kFRing][H] and the
// row goes to its lane's slot of its position (a row more than kFRing before its lane group's last
// is skipped: a later row of the same pass owns that slot). Grid (t), 1,024 threads.
__global__ void __launch_bounds__(1024) inkd_mix_kernel(const bf16* __restrict__ taps, int64_t plane, const bf16* __restrict__ alpha,
                                                        const bf16* __restrict__ w, bf16* __restrict__ out, int mode,
                                                        const int* __restrict__ lanes, int nl, const int* __restrict__ meta, int H,
                                                        float eps) {
    pdl_enter();
    __shared__ float red[32];
    const int t = blockIdx.x;
    bf16* o = out + (int64_t)t * H;
    if (mode == 1) {
        const int z = lane_of(lanes, nl, t);
        if (z < 0) return;
        const int i = t - lanes[4 * z];
        if (lanes[4 * z + 1] - 1 - i >= kFRing) return;
        o = out + ((int64_t)meta[4 * z] * kFRing + ((lanes[4 * z + 2] + i) & (kFRing - 1))) * H;
    }
    float a[kTaps];
#pragma unroll
    for (int k = 0; k < kTaps; ++k) a[k] = bf(alpha[k]);
    float m[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        m[i] = 0.f;
        if (c < H) {
#pragma unroll
            for (int k = 0; k < kTaps; ++k) m[i] = fmaf(bf(taps[(int64_t)k * plane + (int64_t)t * H + c]), a[k], m[i]);
        }
        ss = fmaf(m[i], m[i], ss);
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < H) o[c] = to_bf(m[i] * r * bf(w[c]));
    }
}

// ------------------------------------------------------------------ head vectors
// One head's 128 values, a thread each (x: this thread's): the weighted RMSNorm over the head
// (fp32 statistics, eps inside the root), bf16, the neox rotation at `pos` - halves (d, d + 64),
// angle pos * inv[d], cos / sin in fp32 - in fp32; the caller rounds. part: [4], yv: [128], shared.
__device__ __forceinline__ float norm_rope(float x, const bf16* __restrict__ w, const float* __restrict__ inv, int pos, float eps,
                                           float* part, float* yv) {
    const int d = threadIdx.x, warp = d >> 5, lane = d & 31;
    const float s = warp_sum(x * x);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float y = rbf(x * (1.f / sqrtf(ss / (float)kD + eps)) * bf(w[d]));
    yv[d] = y;
    __syncthreads();
    const float ang = (float)pos * inv[d & (kD / 2 - 1)];
    const float c = cosf(ang), sn = sinf(ang);
    return d < kD / 2 ? y * c - yv[d + kD / 2] * sn : y * c + yv[d - kD / 2] * sn;
}

// A pass's context keys and values (write_target_hidden_kv): kv [t][k heads x 128 | v heads x 128],
// the bf16 output of [k_proj_ctx | v_proj_ctx] on the layer's mixed features. Keys: RMSNorm with
// k_norm_ctx, RoPE at the row's position; values as they are. Each to its lane's ring row
// [lanes][ring][8][K | V] at slot position & (ring - 1); a row `ring` or more before its lane
// group's last is skipped (a later row of the pass owns the slot). Grid (16, t), 128 threads.
__global__ void inkd_ctx_kv_kernel(const bf16* __restrict__ kv, int64_t ldkv, const bf16* __restrict__ knorm,
                                   const float* __restrict__ inv, float eps, const int* __restrict__ lanes, int nl,
                                   const int* __restrict__ meta, int ring, bf16* __restrict__ cache) {
    pdl_enter();
    __shared__ float part[4];
    __shared__ float yv[kD];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int i = t - lanes[4 * z];
    if (lanes[4 * z + 1] - 1 - i >= ring) return;
    const int p = lanes[4 * z + 2] + i;
    bf16* row = cache + ((int64_t)meta[4 * z] * ring + (p & (ring - 1))) * kRowKV;
    if (b < kHK) {
        const float x = bf(kv[(int64_t)t * ldkv + b * kD + d]);
        row[b * 2 * kD + d] = to_bf(norm_rope(x, knorm, inv, p, eps, part, yv));
    } else {
        const int g = b - kHK;
        row[g * 2 * kD + kD + d] = kv[(int64_t)t * ldkv + (kHK + g) * kD + d];
    }
}

// The block's projections qkv [8][q 48 x 128 | k 8 x 128 | v 8 x 128] into what the attention
// reads: queries (q_norm, RoPE at p0 + row) -> qc [8][48][128]; keys (k_norm, RoPE) and values ->
// the lane's ring rows of positions p0 .. p0 + 7. Grid (64, 8), 128 threads.
__global__ void inkd_heads_kernel(const bf16* __restrict__ qkv, int64_t ld, const bf16* __restrict__ qnorm,
                                  const bf16* __restrict__ knorm, const float* __restrict__ inv, float eps, int p0, int ring,
                                  bf16* __restrict__ cache, bf16* __restrict__ qc) {
    pdl_enter();
    __shared__ float part[4];
    __shared__ float yv[kD];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int p = p0 + t;
    const bf16* x = qkv + (int64_t)t * ld;
    if (b < kHQ) {
        qc[((int64_t)t * kHQ + b) * kD + d] = to_bf(norm_rope(bf(x[b * kD + d]), qnorm, inv, p, eps, part, yv));
        return;
    }
    bf16* row = cache + (int64_t)(p & (ring - 1)) * kRowKV;
    if (b < kHQ + kHK) {
        const int g = b - kHQ;
        row[g * 2 * kD + d] = to_bf(norm_rope(bf(x[(kHQ + g) * kD + d]), knorm, inv, p, eps, part, yv));
    } else {
        const int g = b - kHQ - kHK;
        row[g * 2 * kD + kD + d] = x[(kHQ + kHK + g) * kD + d];
    }
}

// ------------------------------------------------------------------ attention
// inkling.cu's ink_attn_part_kernel for a block whose 8 rows read the same keys: positions
// [lo, hi) of one lane's ring (the context, then the block's own 8). One key / value head's keys
// [lo + s chunk, lo + (s + 1) chunk), chunk = ceil((hi - lo) / ns), for its 6 query heads of every
// row: logit = q . k / sqrt(128), softmax partials by the running maximum (m, l = sum exp(logit - m),
// acc = sum exp(logit - m) v). The rows are looped inside the block, kSub keys at a time, so a key
// is fetched from memory once a round and from cache for the other 7 rows, not once a row; the
// block keeps every row's running partial in shared memory and the warps (each takes every 4th
// key, a lane 4 dims) merge into it after each (kSub keys, row). Not measured against a block a
// row (2026-11-16: written without a GPU). Grid (ns, 8): it does not depend on the positions.
constexpr int kPW = 4;    // warps a block
constexpr int kSub = 64;  // keys between merges
__global__ void __launch_bounds__(kPW * 32) inkd_attn_part_kernel(const bf16* __restrict__ qc, const bf16* __restrict__ cache,
                                                                 int ring, int lo, int hi, int ns, float* __restrict__ pm,
                                                                 float* __restrict__ pl, float* __restrict__ pacc) {
    pdl_enter();
    __shared__ float wacc[kPW][kG][kD];
    __shared__ float wm[kPW][kG], wl[kPW][kG];
    __shared__ float racc[kRows][kG][kD];
    __shared__ float rm[kRows][kG], rl[kRows][kG];
    __shared__ float fac[kG][kPW + 1];
    const int s = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nk = hi - lo;
    const int chunk = (nk + ns - 1) / ns;
    const int k0 = lo + s * chunk, k1 = min(k0 + chunk, hi);
    for (int u = tid; u < kRows * kG * kD; u += blockDim.x) (&racc[0][0][0])[u] = 0.f;
    if (tid < kRows * kG) { (&rm[0][0])[tid] = -INFINITY; (&rl[0][0])[tid] = 0.f; }
    __syncthreads();
    const float scale = rsqrtf((float)kD);
    for (int c0 = k0; c0 < k1; c0 += kSub) {
        const int c1 = min(c0 + kSub, k1);
        for (int t = 0; t < kRows; ++t) {
            float q[kG][4];
#pragma unroll
            for (int i = 0; i < kG; ++i) {
                const bf16* qh = qc + ((int64_t)t * kHQ + g * kG + i) * kD + lane * 4;
#pragma unroll
                for (int e = 0; e < 4; ++e) q[i][e] = bf(qh[e]) * scale;
            }
            float m[kG], l[kG], acc[kG][4];
#pragma unroll
            for (int i = 0; i < kG; ++i) { m[i] = -INFINITY; l[i] = 0.f; acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
            for (int j = c0 + warp; j < c1; j += kPW) {
                const bf16* row = cache + (int64_t)(j & (ring - 1)) * kRowKV + g * 2 * kD;
                float kk[4], vv[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) { kk[e] = bf(row[lane * 4 + e]); vv[e] = bf(row[kD + lane * 4 + e]); }
#pragma unroll
                for (int i = 0; i < kG; ++i) {
                    float a = 0.f;
#pragma unroll
                    for (int e = 0; e < 4; ++e) a = fmaf(q[i][e], kk[e], a);
                    const float sc = warp_sum(a);
                    if (sc > m[i]) {
                        const float f = expf(m[i] - sc);        // 0 at the first key
                        l[i] = l[i] * f + 1.f;
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][e] = acc[i][e] * f + vv[e];
                        m[i] = sc;
                    } else {
                        const float wgt = expf(sc - m[i]);
                        l[i] += wgt;
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][e] = fmaf(wgt, vv[e], acc[i][e]);
                    }
                }
            }
            __syncthreads();                                    // the merge before has read the warps' partials
#pragma unroll
            for (int i = 0; i < kG; ++i) {
                if (lane == 0) { wm[warp][i] = m[i]; wl[warp][i] = l[i]; }
#pragma unroll
                for (int e = 0; e < 4; ++e) wacc[warp][i][lane * 4 + e] = acc[i][e];
            }
            __syncthreads();
            // row t's running partial and the warps' to their common maximum: the factors first
            if (tid < kG) {
                const int i = tid;
                float mx = rm[t][i];
                for (int w = 0; w < kPW; ++w) mx = fmaxf(mx, wm[w][i]);
                const float f0 = rm[t][i] > -INFINITY ? expf(rm[t][i] - mx) : 0.f;
                float ls = rl[t][i] * f0;
                fac[i][0] = f0;
                for (int w = 0; w < kPW; ++w) {
                    const float f = wm[w][i] > -INFINITY ? expf(wm[w][i] - mx) : 0.f;
                    fac[i][1 + w] = f;
                    ls = fmaf(wl[w][i], f, ls);
                }
                rm[t][i] = mx;
                rl[t][i] = ls;
            }
            __syncthreads();
            for (int u = tid; u < kG * kD; u += blockDim.x) {
                const int i = u / kD, e = u % kD;
                float a = racc[t][i][e] * fac[i][0];
                for (int w = 0; w < kPW; ++w) a = fmaf(wacc[w][i][e], fac[i][1 + w], a);
                racc[t][i][e] = a;
            }
        }
    }
    __syncthreads();
    for (int u = tid; u < kRows * kG * kD; u += blockDim.x) {
        const int t = u / (kG * kD), i = (u / kD) % kG, e = u % kD;
        const int64_t base = (((int64_t)t * kHK + g) * ns + s) * kG + i;
        pacc[base * kD + e] = racc[t][i][e];
        if (e == 0) { pm[base] = rm[t][i]; pl[base] = rl[t][i]; }
    }
}

// The splits of a (row, query head) into its output: o [8][48][128] = bf16(sum / total). No sink,
// no gate. Grid (48, 8), 128 threads.
__global__ void inkd_attn_combine_kernel(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         int ns, bf16* __restrict__ o) {
    pdl_enter();
    const int hq = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int g = hq / kG, i = hq % kG;
    const int64_t base = ((int64_t)t * kHK + g) * ns;
    float mx = -INFINITY;
    for (int s = 0; s < ns; ++s) mx = fmaxf(mx, pm[(base + s) * kG + i]);
    float tot = 0.f, a = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float m = pm[(base + s) * kG + i];
        if (!(m > -INFINITY)) continue;
        const float f = expf(m - mx);
        tot = fmaf(pl[(base + s) * kG + i], f, tot);
        a = fmaf(pacc[((base + s) * kG + i) * kD + d], f, a);
    }
    o[((int64_t)t * kHQ + hq) * kD + d] = to_bf(a / tot);
}

// ------------------------------------------------------------------ the residual stream
// The block's input rows: x [8][H] = the anchor's (normed) embedding row, then the mask embedding
// 7 times. Grid (H / 256, 8), 256 threads.
__global__ void inkd_embed_kernel(const bf16* __restrict__ embed, int anchor, const bf16* __restrict__ mask, bf16* __restrict__ x,
                                  int H) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= H) return;
    x[(int64_t)t * H + c] = t == 0 ? embed[(int64_t)anchor * H + c] : mask[c];
}

// The residual add fused with the next RMSNorm (layernorm.py forward_native): s = h + a in fp32,
// h = bf16(s) stored back, y = bf16(s * rsqrt(mean(s^2) + eps) * w) - the norm of the unrounded
// sum, one rounding (glm.cu's rmsnorm_kernel norms the rounded sum and rounds before the weight).
// ADD false: s = h. Grid (t), 1,024 threads.
template <bool ADD>
__global__ void __launch_bounds__(1024) inkd_norm_kernel(bf16* __restrict__ h, const bf16* __restrict__ a,
                                                         const bf16* __restrict__ w, bf16* __restrict__ y, int H, float eps) {
    pdl_enter();
    __shared__ float red[32];
    const int t = blockIdx.x;
    bf16* hr = h + (int64_t)t * H;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        v[i] = 0.f;
        if (c < H) {
            v[i] = bf(hr[c]);
            if (ADD) {
                v[i] += bf(a[(int64_t)t * H + c]);
                hr[c] = to_bf(v[i]);
            }
        }
        ss = fmaf(v[i], v[i], ss);
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + eps);
    bf16* yr = y + (int64_t)t * H;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < H) yr[c] = to_bf(v[i] * r * bf(w[c]));
    }
}

// ------------------------------------------------------------------ the Markov head
// The part of the head's input that does not change over a block's 7 steps: zx [7][h_i | f_rnn],
// h_i = hidden row 1 + i (the final-normed block rows), f_rnn the lane's row (null: zeros).
// Grid (2 H / 256, 7), 256 threads.
__global__ void inkd_rnn_x_kernel(const bf16* __restrict__ hidden, const bf16* __restrict__ f, bf16* __restrict__ zx, int H) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (c >= 2 * H) return;
    zx[(int64_t)i * 2 * H + c] = c < H ? hidden[(int64_t)(i + 1) * H + c] : (f ? f[c - H] : to_bf(0.f));
}

// A step's changing input: zs [state 256 | markov_w1[prev] 256]; prev = the anchor at step 0, the
// token the step before wrote (toks[step - 1]) after; the state starts a block at zero. One block
// of 512 threads.
__global__ void inkd_rnn_in_kernel(const bf16* __restrict__ state, const bf16* __restrict__ w1, const int* __restrict__ toks,
                                   int step, int anchor, int vocab, bf16* __restrict__ zs) {
    pdl_enter();
    const int c = threadIdx.x;
    if (c < kR) {
        zs[c] = step == 0 ? to_bf(0.f) : state[c];
        return;
    }
    int prev = step == 0 ? anchor : toks[step - 1];
    prev = min(max(prev, 0), vocab - 1);
    zs[c] = w1[(int64_t)prev * kR + (c - kR)];
}

// rnn_gates (kernels/ops/inkling2/mtp.py), each op rounded to bf16 as the eager ops it replaced:
// raw [gate | cand | out] = bf16(xs + ss + bias), xs / ss the two fp32 halves of rnn_proj's product
// (the block-constant columns and the step's); gate = bf16(1 / (1 + exp(-gate_raw)));
// state' = bf16(bf16(gate * state) + bf16(bf16(1 - gate) * bf16(tanh(cand_raw))));
// latent = bf16(tanh(out_raw)). The state is zero at step 0. One block of 256 threads.
__global__ void inkd_rnn_cell_kernel(const float* __restrict__ xs, const float* __restrict__ ss, const bf16* __restrict__ bias,
                                     bf16* __restrict__ state, bf16* __restrict__ latent, int first) {
    pdl_enter();
    const int c = threadIdx.x;
    const float graw = rbf(xs[c] + ss[c] + bf(bias[c]));
    const float craw = rbf(xs[kR + c] + ss[kR + c] + bf(bias[kR + c]));
    const float oraw = rbf(xs[2 * kR + c] + ss[2 * kR + c] + bf(bias[2 * kR + c]));
    const float st = first ? 0.f : bf(state[c]);
    const float gate = rbf(1.f / (1.f + expf(-graw)));
    const float kept = rbf(gate * st);
    const float blend = rbf(rbf(1.f - gate) * rbf(tanhf(craw)));
    state[c] = to_bf(kept + blend);
    latent[c] = to_bf(tanhf(oraw));
}

// A step's logits and their reduction (markov_sample.py's partial kernel, greedy):
//   y = bf16(base + bf16(fold_b + lw))        lw fp32 [n]: latent . fold_w^T
//   y = bf16(cap * tanh(y / cap));  ids >= real: -inf
// A block takes 2,048 ids: its maximum, the lowest id that has it, and sum exp(y - max) ->
// part [3][blocks] (the id's bits in the float). ylog (or null): every y, fp32 [n] (checks).
// Grid (ceil(n / 2048)), 256 threads.
constexpr int kStepPer = 8;
__global__ void inkd_step_part_kernel(const bf16* __restrict__ base, const float* __restrict__ lw, const bf16* __restrict__ fb, int n,
                                      int real, float cap, float* __restrict__ part, float* __restrict__ ylog) {
    pdl_enter();
    __shared__ float sm[256], sl[256];
    __shared__ int si[256];
    const int tid = threadIdx.x;
    float m = -INFINITY, l = 0.f;
    int idx = INT_MAX;
#pragma unroll
    for (int k = 0; k < kStepPer; ++k) {
        const int c = blockIdx.x * (256 * kStepPer) + k * 256 + tid;
        if (c >= n) continue;
        if (c >= real) {
            if (ylog) ylog[c] = -INFINITY;
            continue;
        }
        const float y0 = rbf(bf(base[c]) + rbf(bf(fb[c]) + lw[c]));
        const float y = rbf(cap * tanhf(y0 / cap));
        if (ylog) ylog[c] = y;
        if (y > m) {
            l = l * expf(m - y) + 1.f;                          // exp(-inf) = 0 at the first id
            m = y;
            idx = c;
        } else {
            l += expf(y - m);
        }
    }
    sm[tid] = m; sl[tid] = l; si[tid] = idx;
    __syncthreads();
    if (tid != 0) return;
    float mx = -INFINITY;
    int best = INT_MAX;
    for (int j = 0; j < 256; ++j)
        if (sm[j] > mx || (sm[j] == mx && si[j] < best)) { mx = sm[j]; best = si[j]; }
    float sum = 0.f;
    for (int j = 0; j < 256; ++j)
        if (sm[j] > -INFINITY) sum = fmaf(sl[j], expf(sm[j] - mx), sum);
    const int nb = gridDim.x, b = blockIdx.x;
    part[b] = mx;
    part[nb + b] = __int_as_float(best);
    part[2 * nb + b] = sum;
}

// The blocks' partials into the step's token (argmax, the lowest id on ties) and its probability
// under softmax(y): 1 / sum exp(y - max). One thread.
__global__ void inkd_step_pick_kernel(const float* __restrict__ part, int nb, int step, int* __restrict__ toks,
                                      float* __restrict__ probs, const int* __restrict__ ids) {
    pdl_enter();
    if (threadIdx.x != 0) return;
    float mx = -INFINITY;
    int best = INT_MAX;
    for (int b = 0; b < nb; ++b) {
        const float m = part[b];
        const int i = __float_as_int(part[nb + b]);
        if (m > mx || (m == mx && i < best)) { mx = m; best = i; }
    }
    float sum = 0.f;
    for (int b = 0; b < nb; ++b)
        if (part[b] > -INFINITY) sum = fmaf(part[2 * nb + b], expf(part[b] - mx), sum);
    toks[step] = ids ? ids[best] : best;     // a listed head's index -> its id
    probs[step] = 1.f / sum;
}

// ------------------------------------------------------------------ load: the folded head
// fp64 -> bf16 as torch casts it (the reference's copy_ of an fp64 product into its bf16 buffer):
// to fp32, then to bf16, each to nearest even.
__device__ __forceinline__ bf16 d2bf(double v) { return __float2bfloat16_rn((float)v); }

// Inkling2MTPHead.refresh_fold: the head's hidden-space correction through the target's unembed,
// fold_w [n][256] = bf16(U hidden_out.W), fold_b [n] = bf16(U hidden_out.b), sums in fp64 (the
// products of two bf16 are exact). u: [n][4608] rows of the unembed; w: hidden_out.weight
// [4608][256]; b: hidden_out.bias [4608]. Grid (n), 256 threads: thread j the row's column j.
__global__ void inkd_fold_kernel(const bf16* __restrict__ u, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                 bf16* __restrict__ fw, bf16* __restrict__ fb) {
    pdl_enter();
    __shared__ float us[kH];
    __shared__ double red[kR];
    const int v = blockIdx.x, j = threadIdx.x;
    for (int c = j; c < kH; c += kR) us[c] = bf(u[(int64_t)v * kH + c]);
    __syncthreads();
    double acc = 0.0, bb = 0.0;
    for (int d = 0; d < kH; ++d) acc = fma((double)us[d], (double)bf(w[(int64_t)d * kR + j]), acc);
    for (int d = j; d < kH; d += kR) bb = fma((double)us[d], (double)bf(b[d]), bb);
    fw[(int64_t)v * kR + j] = d2bf(acc);
    red[j] = bb;
    __syncthreads();
    if (j != 0) return;
    double s = 0.0;
    for (int i = 0; i < kR; ++i) s += red[i];
    fb[v] = d2bf(s);
}

}  // namespace

extern "C" {

int inkd_mix(void* stream, const void* taps, int64_t plane, const void* alpha, const void* w, void* out, int mode,
             const void* lanes, int nl, const void* meta, int t, int h, float eps) {
    if (h > kPer * 1024 || t < 1 || (mode != 0 && mode != 1)) return 1;
    pdl_launch(inkd_mix_kernel, dim3(t), dim3(1024), 0, GST, (const bf16*)taps, plane, (const bf16*)alpha, (const bf16*)w,
               (bf16*)out, mode, (const int*)lanes, nl, (const int*)meta, h, eps);
    GRET;
}

int inkd_ctx_kv(void* stream, const void* kv, int64_t ldkv, const void* knorm, const void* inv, float eps, const void* lanes,
                int nl, const void* meta, int ring, void* cache, int t) {
    if (ring < 1 || (ring & (ring - 1)) || t < 1) return 1;
    pdl_launch(inkd_ctx_kv_kernel, dim3(2 * kHK, t), dim3(kD), 0, GST, (const bf16*)kv, ldkv, (const bf16*)knorm,
               (const float*)inv, eps, (const int*)lanes, nl, (const int*)meta, ring, (bf16*)cache);
    GRET;
}

int inkd_heads(void* stream, const void* qkv, int64_t ld, const void* qnorm, const void* knorm, const void* inv, float eps,
               int p0, int ring, void* cache, void* qc) {
    if (ring < kRows || (ring & (ring - 1))) return 1;
    pdl_launch(inkd_heads_kernel, dim3(kHQ + 2 * kHK, kRows), dim3(kD), 0, GST, (const bf16*)qkv, ld, (const bf16*)qnorm,
               (const bf16*)knorm, (const float*)inv, eps, p0, ring, (bf16*)cache, (bf16*)qc);
    GRET;
}

int inkd_attn_part(void* stream, const void* qc, const void* cache, int ring, int lo, int hi, int ns, void* pm, void* pl,
                   void* pacc) {
    if (ns < 1 || hi <= lo || hi - lo > ring || (ring & (ring - 1))) return 1;
    pdl_launch(inkd_attn_part_kernel, dim3(ns, kHK), dim3(kPW * 32), 0, GST, (const bf16*)qc, (const bf16*)cache, ring, lo, hi,
               ns, (float*)pm, (float*)pl, (float*)pacc);
    GRET;
}

int inkd_attn_combine(void* stream, const void* pm, const void* pl, const void* pacc, int ns, void* o) {
    pdl_launch(inkd_attn_combine_kernel, dim3(kHQ, kRows), dim3(kD), 0, GST, (const float*)pm, (const float*)pl,
               (const float*)pacc, ns, (bf16*)o);
    GRET;
}

int inkd_embed(void* stream, const void* embed, int anchor, const void* mask, void* x, int h) {
    pdl_launch(inkd_embed_kernel, dim3((h + 255) / 256, kRows), dim3(256), 0, GST, (const bf16*)embed, anchor, (const bf16*)mask,
               (bf16*)x, h);
    GRET;
}

int inkd_norm(void* stream, void* hbuf, const void* a, const void* w, void* y, int t, int h, float eps) {
    if (h > kPer * 1024 || t < 1) return 1;
    if (a)
        pdl_launch(inkd_norm_kernel<true>, dim3(t), dim3(1024), 0, GST, (bf16*)hbuf, (const bf16*)a, (const bf16*)w, (bf16*)y, h,
                   eps);
    else
        pdl_launch(inkd_norm_kernel<false>, dim3(t), dim3(1024), 0, GST, (bf16*)hbuf, (const bf16*)nullptr, (const bf16*)w,
                   (bf16*)y, h, eps);
    GRET;
}

int inkd_rnn_x(void* stream, const void* hidden, const void* f, void* zx, int h) {
    pdl_launch(inkd_rnn_x_kernel, dim3((2 * h + 255) / 256, kRows - 1), dim3(256), 0, GST, (const bf16*)hidden, (const bf16*)f,
               (bf16*)zx, h);
    GRET;
}

int inkd_rnn_in(void* stream, const void* state, const void* w1, const void* toks, int step, int anchor, int vocab, void* zs) {
    if (step < 0 || step >= kRows - 1 || vocab < 1) return 1;
    pdl_launch(inkd_rnn_in_kernel, dim3(1), dim3(2 * kR), 0, GST, (const bf16*)state, (const bf16*)w1, (const int*)toks, step,
               anchor, vocab, (bf16*)zs);
    GRET;
}

int inkd_rnn_cell(void* stream, const void* xs, const void* ss, const void* bias, void* state, void* latent, int first) {
    pdl_launch(inkd_rnn_cell_kernel, dim3(1), dim3(kR), 0, GST, (const float*)xs, (const float*)ss, (const bf16*)bias,
               (bf16*)state, (bf16*)latent, first);
    GRET;
}

// Blocks of inkd_step_part over n ids: the length of each of its three partial rows.
int inkd_step_blocks(int n) { return (n + 256 * kStepPer - 1) / (256 * kStepPer); }

int inkd_step_part(void* stream, const void* base, const void* lw, const void* fb, int n, int real, float cap, void* part,
                   void* ylog) {
    if (n < 1 || real < 1 || real > n) return 1;
    pdl_launch(inkd_step_part_kernel, dim3(inkd_step_blocks(n)), dim3(256), 0, GST, (const bf16*)base, (const float*)lw,
               (const bf16*)fb, n, real, cap, (float*)part, (float*)ylog);
    GRET;
}

int inkd_step_pick(void* stream, const void* part, int n, int step, void* toks, void* probs, const void* ids) {
    if (step < 0 || step >= kRows - 1) return 1;
    pdl_launch(inkd_step_pick_kernel, dim3(1), dim3(32), 0, GST, (const float*)part, inkd_step_blocks(n), step, (int*)toks,
               (float*)probs, (const int*)ids);
    GRET;
}

int inkd_fold(void* stream, const void* u, int n, int h, int r, const void* w, const void* b, void* fw, void* fb) {
    if (h != kH || r != kR || n < 1) return 1;
    pdl_launch(inkd_fold_kernel, dim3(n), dim3(kR), 0, GST, (const bf16*)u, (const bf16*)w, (const bf16*)b, (bf16*)fw, (bf16*)fb);
    GRET;
}

}  // extern "C"
