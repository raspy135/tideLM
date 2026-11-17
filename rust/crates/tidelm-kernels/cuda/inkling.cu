// Inkling 2 (thinkingmachines/Inkling-2-Small, model_type inkling2) kernels the GLM engines have no
// counterpart for: the short convolutions, the block attention residual, and grouped-query attention
// with a relative-position bias, a sink logit and an output gate - over a 1,024-key window on most
// layers, the whole context on every 6th. The routed experts run on the EXL3 file, the projections
// on ds41.cu's FP8 / BF16 GEMVs, the router and the norms on glm.cu.
//
// Reference: transformers' modeling_inkling2.py (Inkling2SlidingAttention, Inkling2FullAttention,
// Inkling2ShortConvolution, Inkling2DynamicShortConvolution, Inkling2BlockAttentionResidual,
// eager_attention_forward). Where the reference rounds to bf16 between two fp32 steps, these do too;
// inside the attention's logits and its weighted value sum they keep fp32 where the reference's
// bf16 matmuls round each product (the difference is the reference's rounding noise, not a formula).
//
// Lanes: every launcher takes the pass's lane table (glm.cu positions_kernel's [L][4]: first row,
// rows, first position, cache-row offset) and a parallel [L][4] of this file's own whose first
// entry is the lane. A lane's recurrent state is indexed by position: the convolutions' last inputs
// in rings of kConvRing positions, a window layer's keys and values in a ring of `ring` positions (a
// power of two past the window and a pass's rows). A pass writes its rows at their positions and
// reads the ones before, so a verify pass's rejected rows need no undoing: the position goes back
// and the next pass writes over them before anything reads them.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
extern "C" int ink_set_pdl(int on) { g_pdl = on; return 0; }

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
__device__ __forceinline__ int lane_of(const int* lanes, int nl, int t) {
    for (int z = 0; z < nl; ++z)
        if (t >= lanes[4 * z] && t < lanes[4 * z] + lanes[4 * z + 1]) return z;
    return -1;
}

constexpr int kD = 128;        // a head's dims (queries, keys, values)
constexpr int kG = 6;          // query heads a key / value head
constexpr int kRel = 16;       // a head's relative state
constexpr int kTaps = 4;       // the short convolutions' taps
constexpr int kConvRing = 64;  // positions a conv ring holds (3 before a pass + its rows, HEAD_ROWS 33)
constexpr int kMaxBlocks = 9;  // block states a mix sees at most (8 saved + the current one)

// ------------------------------------------------------------------ short convolutions
// y(t) = bf16((sum_lag w(lag) x(t - lag) + x(t)) / div), fp32 sums from the oldest tap to the newest
// as conv1d's, one rounding (Inkling2ShortConvolution; the dynamic one of the full-attention layers'
// keys and values, whose taps are static + a per-token, per-head projection).
//   wmode 0: w [C][4], conv1d's order: w[c][3 - lag]
//   wmode 1: w [4][C] by lag, plus dyn [t][heads x 4] (ldd apart; head = c / 128): Inkling2Dynamic
// x before the pass's first row comes from the lane's ring (its raw inputs by position).
//   omode 0: out bf16 [t][ldo] = y
//   omode 1: res fp32 [t][C] = y                      (a block boundary's residual restarts here)
//   omode 2: res fp32 [t][C] = h + y                  (h bf16 [t][C]: the layer's input)
//   omode 3: h bf16 [t][C] = bf16(res + y)            (the layer's output)
__global__ void ink_conv_kernel(const bf16* __restrict__ x, int64_t ldx, int C, const bf16* __restrict__ w, int wmode,
                                const bf16* __restrict__ dyn, int64_t ldd, const bf16* __restrict__ ring, int64_t lane_stride,
                                const int* __restrict__ lanes, int nl, const int* __restrict__ meta, float div, int omode,
                                bf16* __restrict__ out, int64_t ldo, float* __restrict__ res, bf16* __restrict__ h) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int i = t - lanes[4 * z], p = lanes[4 * z + 2] + i;
    const bf16* rg = ring + (int64_t)meta[4 * z] * lane_stride;
    float acc = 0.f;
#pragma unroll
    for (int lag = kTaps - 1; lag >= 0; --lag) {
        const int j = i - lag;
        if (p - lag < 0) continue;                       // before the sequence: zero padding
        const float xv = j >= 0 ? bf(x[(int64_t)(t - lag) * ldx + c])
                                : bf(rg[(int64_t)((p - lag) & (kConvRing - 1)) * C + c]);
        float wv;
        if (wmode == 0) wv = bf(w[c * kTaps + (kTaps - 1 - lag)]);
        else wv = bf(w[(int64_t)lag * C + c]) + bf(dyn[(int64_t)t * ldd + (c / kD) * kTaps + lag]);
        acc = fmaf(wv, xv, acc);
    }
    const float y = rbf((acc + bf(x[(int64_t)t * ldx + c])) / div);
    const int64_t o = (int64_t)t * C + c;
    if (omode == 0) out[(int64_t)t * ldo + c] = to_bf(y);
    else if (omode == 1) res[o] = y;
    else if (omode == 2) res[o] = bf(h[o]) + y;
    else h[o] = to_bf(res[o] + y);
}

// The ring after the pass: each lane's last min(rows, kConvRing) raw rows at their positions.
// Grid (C / 256, kConvRing, lanes).
__global__ void ink_conv_state_kernel(const bf16* __restrict__ x, int64_t ldx, int C, bf16* __restrict__ ring, int64_t lane_stride,
                                      const int* __restrict__ lanes, const int* __restrict__ meta) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.z;
    if (c >= C) return;
    const int row0 = lanes[4 * z], n = lanes[4 * z + 1], p0 = lanes[4 * z + 2];
    const int i = n - 1 - (int)blockIdx.y;
    if (i < 0) return;
    bf16* rg = ring + (int64_t)meta[4 * z] * lane_stride;
    rg[(int64_t)((p0 + i) & (kConvRing - 1)) * C + c] = x[(int64_t)(row0 + i) * ldx + c];
}

// ------------------------------------------------------------------ block attention residual
// Inkling2BlockAttentionResidual and the RMSNorm that follows it: the row's saved block states
// (blocks [t][nb][H] bf16) and its current state (cur: bf16 or fp32 [t][H]), each scaled to unit
// RMS (eps 1e-6, fp32), are mixed by softmax(q . s_i / sqrt(H)) for each of nq learned queries;
// out[k][t] = bf16(nw[k] * mix * rsqrt(mean(mix^2) + neps)). Grid (nq, t), 256 threads.
__global__ void ink_mix_kernel(const bf16* __restrict__ blocks, int64_t ldb, int nb, const void* __restrict__ cur, int cur_f32,
                               const float* __restrict__ q, const bf16* __restrict__ nw, float neps, bf16* __restrict__ out,
                               int64_t plane, int H) {
    pdl_enter();
    extern __shared__ float sm[];                          // mixed [H], then the warps' partials
    float* mixed = sm;
    float* part = sm + H;                                  // [8 warps][2 * kMaxBlocks]
    __shared__ float coef[kMaxBlocks];
    const int k = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = nb + 1;
    const bf16* cb = (const bf16*)cur + (int64_t)t * H;
    const float* cf = (const float*)cur + (int64_t)t * H;
    const float* qk = q + (int64_t)k * H;
    float dot[kMaxBlocks], ss[kMaxBlocks];
    for (int i = 0; i < n; ++i) { dot[i] = 0.f; ss[i] = 0.f; }
    for (int c = tid; c < H; c += blockDim.x) {
        const float qv = qk[c];
        for (int i = 0; i < n; ++i) {
            const float v = i < nb ? bf(blocks[(int64_t)t * ldb + (int64_t)i * H + c]) : (cur_f32 ? cf[c] : bf(cb[c]));
            dot[i] = fmaf(qv, v, dot[i]);
            ss[i] = fmaf(v, v, ss[i]);
        }
    }
    for (int i = 0; i < n; ++i) {
        const float d = warp_sum(dot[i]), s = warp_sum(ss[i]);
        if (lane == 0) { part[warp * 2 * kMaxBlocks + 2 * i] = d; part[warp * 2 * kMaxBlocks + 2 * i + 1] = s; }
    }
    __syncthreads();
    if (tid == 0) {
        float lg[kMaxBlocks], rs[kMaxBlocks], mx = -INFINITY;
        const float scale = rsqrtf((float)H);
        for (int i = 0; i < n; ++i) {
            float d = 0.f, s = 0.f;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { d += part[w * 2 * kMaxBlocks + 2 * i]; s += part[w * 2 * kMaxBlocks + 2 * i + 1]; }
            rs[i] = rsqrtf(s / (float)H + 1e-6f);
            lg[i] = d * rs[i] * scale;
            mx = fmaxf(mx, lg[i]);
        }
        float sum = 0.f;
        for (int i = 0; i < n; ++i) { lg[i] = expf(lg[i] - mx); sum += lg[i]; }
        for (int i = 0; i < n; ++i) coef[i] = lg[i] / sum * rs[i];
    }
    __syncthreads();
    float s2 = 0.f;
    for (int c = tid; c < H; c += blockDim.x) {
        float m = 0.f;
        for (int i = 0; i < n; ++i) {
            const float v = i < nb ? bf(blocks[(int64_t)t * ldb + (int64_t)i * H + c]) : (cur_f32 ? cf[c] : bf(cb[c]));
            m = fmaf(coef[i], v, m);
        }
        mixed[c] = m;
        s2 = fmaf(m, m, s2);
    }
    s2 = warp_sum(s2);
    if (lane == 0) part[warp] = s2;
    __syncthreads();
    float tot = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) tot += part[w];
    const float rn = rsqrtf(tot / (float)H + neps);
    const bf16* nk = nw + (int64_t)k * H;
    bf16* o = out + (int64_t)k * plane + (int64_t)t * H;
    for (int c = tid; c < H; c += blockDim.x) o[c] = to_bf(bf(nk[c]) * (mixed[c] * rn));
}

// ------------------------------------------------------------------ head vectors
// A row's cache row (elements into the layer's key / value cache, [rows][kl][K 128 | V 128]): a
// window layer's ring slot of its lane, or its position in the lane's part of a flat cache.
__device__ __forceinline__ int64_t kv_row(const int* lanes, const int* meta, int z, int p, int ring, int64_t lane_rows) {
    return ring ? (int64_t)meta[4 * z] * lane_rows + (p & (ring - 1)) : (int64_t)lanes[4 * z + 3] + p;
}

// The projections' outputs into what the attention reads (Inkling2*Attention.forward, before the
// attention interface). Grid (hl + 2 kl, t), 128 threads; block b:
//   b < hl:        query head b of proj (q at qoff, 128 a head) times its learned scale, clipped to
//                  L2 norm `qclip`, bf16, times tau (full layers), bf16 -> qc [t][hl][128]; its
//                  relative state (roff, 16 a head) the same with `rclip` -> rc [t][hl][16]
//   b < hl + kl:   key head of kx [t][kl][128] (after its convolution) clipped to `qclip`
//   else:          value head of vx scaled to unit RMS (eps)
// keys and values go to the row's cache row. tau = 1 + alpha ln(max((p + 1) / floor, 1)), 1 when
// alpha is 0 (the window layers).
__global__ void ink_heads_kernel(const bf16* __restrict__ proj, int64_t ldp, int qoff, int roff, const bf16* __restrict__ qscale,
                                 const bf16* __restrict__ rscale, const bf16* __restrict__ kx, const bf16* __restrict__ vx,
                                 int hl, int kl, float qclip, float rclip, float eps, float alpha, float floor_n,
                                 const int* __restrict__ lanes, int nl, const int* __restrict__ meta, int ring, int64_t lane_rows,
                                 bf16* __restrict__ qc, bf16* __restrict__ rc, bf16* __restrict__ cache) {
    pdl_enter();
    __shared__ float part[4];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x, warp = d >> 5, lane = d & 31;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int p = lanes[4 * z + 2] + (t - lanes[4 * z]);
    const float tau = alpha > 0.f ? 1.f + alpha * logf(fmaxf((float)(p + 1) / floor_n, 1.f)) : 1.f;
    float x;
    if (b < hl) x = bf(proj[(int64_t)t * ldp + qoff + b * kD + d]) * bf(qscale[b * kD + d]);
    else if (b < hl + kl) x = bf(kx[(int64_t)t * kl * kD + (b - hl) * kD + d]);
    else x = bf(vx[(int64_t)t * kl * kD + (b - hl - kl) * kD + d]);
    const float s = warp_sum(x * x);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    if (b < hl) {
        float v = rbf(x * fminf(qclip / (sqrtf(ss) + 1e-6f), 1.f));
        if (alpha > 0.f) v = rbf(v * tau);
        qc[((int64_t)t * hl + b) * kD + d] = to_bf(v);
        // the relative state: 16 dims, the first warp's low lanes
        if (warp == 0) {
            const float r = lane < kRel ? bf(proj[(int64_t)t * ldp + roff + b * kRel + lane]) * bf(rscale[b * kRel + lane]) : 0.f;
            const float rs = warp_sum(r * r);
            if (lane < kRel) {
                float v2 = rbf(r * fminf(rclip / (sqrtf(rs) + 1e-6f), 1.f));
                if (alpha > 0.f) v2 = rbf(v2 * tau);
                rc[((int64_t)t * hl + b) * kRel + lane] = to_bf(v2);
            }
        }
        return;
    }
    bf16* row = cache + kv_row(lanes, meta, z, p, ring, lane_rows) * (int64_t)(kl * 2 * kD);
    if (b < hl + kl) row[(b - hl) * 2 * kD + d] = to_bf(x * fminf(qclip / (sqrtf(ss) + 1e-6f), 1.f));
    else row[(b - hl - kl) * 2 * kD + kD + d] = to_bf(x * rsqrtf(ss / (float)kD + eps));
}

// ------------------------------------------------------------------ attention, decode rows
// One key / value head's keys [lo + s chunk, lo + (s + 1) chunk) of a row for its 6 query heads:
// logit = q . k / sqrt(128) + bias(p - j), bias(d) = r . projT[d] for d < rel_extent (projT
// [rel_extent][16]: rel_logits_proj.proj transposed), softmax partials by the running maximum
// (max m, sum l of exp(logit - m), acc of exp(logit - m) v). Keys: the row's window (window > 0:
// positions p - window + 1 ..= p) or every position up to p; chunk = ceil(keys / ns), so the grid
// (ns, kl, t) does not depend on the positions. A warp takes every 8th key; a lane 4 dims.
__global__ void ink_attn_part_kernel(const bf16* __restrict__ qc, const bf16* __restrict__ rc, const bf16* __restrict__ projT,
                                     int rel_extent, const bf16* __restrict__ cache, int ring, int64_t lane_rows, int window,
                                     const int* __restrict__ lanes, int nl, const int* __restrict__ meta, int hl, int kl, int ns,
                                     float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    pdl_enter();
    __shared__ float wm[8][kG], wl[8][kG];
    __shared__ float wacc[8][kG][kD];
    const int s = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int z = lane_of(lanes, nl, t);
    if (z < 0) return;
    const int p = lanes[4 * z + 2] + (t - lanes[4 * z]);
    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    const int nk = p - lo + 1;
    const int chunk = (nk + ns - 1) / ns;
    const int k0 = lo + s * chunk, k1 = min(k0 + chunk, p + 1);
    const float scale = rsqrtf((float)kD);
    float q[kG][4], r[kG];
#pragma unroll
    for (int i = 0; i < kG; ++i) {
        const bf16* qh = qc + ((int64_t)t * hl + g * kG + i) * kD + lane * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) q[i][e] = bf(qh[e]) * scale;
        r[i] = lane < kRel ? bf(rc[((int64_t)t * hl + g * kG + i) * kRel + lane]) : 0.f;
    }
    float m[kG], l[kG], acc[kG][4];
#pragma unroll
    for (int i = 0; i < kG; ++i) { m[i] = -INFINITY; l[i] = 0.f; acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    const int64_t rowsz = (int64_t)kl * 2 * kD;
    for (int j = k0 + warp; j < k1; j += 8) {
        const bf16* row = cache + kv_row(lanes, meta, z, j, ring, lane_rows) * rowsz + g * 2 * kD;
        float kk[4], vv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { kk[e] = bf(row[lane * 4 + e]); vv[e] = bf(row[kD + lane * 4 + e]); }
        const int d = p - j;
        const float pj = (lane < kRel && d < rel_extent) ? bf(projT[(int64_t)d * kRel + lane]) : 0.f;
#pragma unroll
        for (int i = 0; i < kG; ++i) {
            float a = fmaf(r[i], pj, 0.f);
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
#pragma unroll
    for (int i = 0; i < kG; ++i) {
        if (lane == 0) { wm[warp][i] = m[i]; wl[warp][i] = l[i]; }
#pragma unroll
        for (int e = 0; e < 4; ++e) wacc[warp][i][lane * 4 + e] = acc[i][e];
    }
    __syncthreads();
    // the 8 warps' partials into the split's: thread (i, dims) of the first 6 x 128 slots
    const int64_t base = (((int64_t)t * kl + g) * ns + s) * kG;
    for (int u = tid; u < kG * kD; u += blockDim.x) {
        const int i = u / kD, e = u % kD;
        float mx = -INFINITY;
        for (int w = 0; w < 8; ++w) mx = fmaxf(mx, wm[w][i]);
        float ls = 0.f, a = 0.f;
        for (int w = 0; w < 8; ++w) {
            const float f = wm[w][i] > -INFINITY ? expf(wm[w][i] - mx) : 0.f;
            ls = fmaf(wl[w][i], f, ls);
            a = fmaf(wacc[w][i][e], f, a);
        }
        pacc[(base + i) * kD + e] = a;
        if (e == 0) { pm[base + i] = mx; pl[base + i] = ls; }
    }
}

// The splits of a query head and its sink logit (sink [t][...] at soff of proj: one extra logit with
// a zero value) into the head's output, times sigmoid of its gate (goff of proj):
// o [t][hl][128] = bf16(bf16(sum / total) * sigmoid(gate)). Grid (hl, t), 128 threads.
__global__ void ink_attn_combine_kernel(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                        const bf16* __restrict__ proj, int64_t ldp, int soff, int goff, int hl, int kl, int ns,
                                        bf16* __restrict__ o) {
    pdl_enter();
    const int hq = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int g = hq / kG, i = hq % kG;
    const int64_t base = ((int64_t)t * kl + g) * ns;
    const float sink = bf(proj[(int64_t)t * ldp + soff + hq]);
    float mx = sink;
    for (int s = 0; s < ns; ++s) mx = fmaxf(mx, pm[(base + s) * kG + i]);
    float tot = expf(sink - mx), a = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float m = pm[(base + s) * kG + i];
        if (!(m > -INFINITY)) continue;
        const float f = expf(m - mx);
        tot = fmaf(pl[(base + s) * kG + i], f, tot);
        a = fmaf(pacc[((base + s) * kG + i) * kD + d], f, a);
    }
    const float gate = bf(proj[(int64_t)t * ldp + goff + hq * kD + d]);
    o[((int64_t)t * hl + hq) * kD + d] = to_bf(rbf(a / tot) / (1.f + expf(-gate)));
}

// ------------------------------------------------------------------ attention, prefill rows
// A row block's keys and values [S][kl][K | V] out of the cache (a ring wraps; the GEMMs want them
// contiguous): position lo + j of lane group z. Grid (S, kl): S may pass a grid's 65,535 in y.
__global__ void ink_kv_window_kernel(const bf16* __restrict__ cache, int ring, int64_t lane_rows, const int* __restrict__ lanes,
                                     const int* __restrict__ meta, int z, int lo, int kl, bf16* __restrict__ out) {
    pdl_enter();
    const int c = blockIdx.y * blockDim.x + threadIdx.x, j = blockIdx.x;
    if (c >= kl * 2 * kD) return;
    out[(int64_t)j * kl * 2 * kD + c] = cache[kv_row(lanes, meta, z, lo + j, ring, lane_rows) * (int64_t)(kl * 2 * kD) + c];
}

// scores [hl][rb][S] fp32 (q . k, unscaled, from the GEMM; key j is position lo + j) of rows
// r0 .. r0 + rb of the pass -> the attention weights as bf16 in place of nothing: w [hl][rb][S].
// Each (head, row): logits = score / sqrt(128) + bias(p - j) over its keys (causal, and inside the
// window when window > 0), the sink logit in the softmax's denominator. One warp a (head, row);
// grid (ceil(rb / 8), hl), 256 threads.
__global__ void ink_pf_softmax_kernel(const float* __restrict__ sc, const bf16* __restrict__ rc, const bf16* __restrict__ projT,
                                      int rel_extent, const bf16* __restrict__ proj, int64_t ldp, int soff, int hl, int r0, int rb,
                                      int S, int lo, int p0, int window, bf16* __restrict__ w) {
    pdl_enter();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 8 + warp, hq = blockIdx.y;
    if (i >= rb) return;
    const int t = r0 + i, p = p0 + t;
    const float scale = rsqrtf((float)kD);
    float r[kRel];
#pragma unroll
    for (int c = 0; c < kRel; ++c) r[c] = bf(rc[((int64_t)t * hl + hq) * kRel + c]);
    const float* srow = sc + ((int64_t)hq * rb + i) * S;
    bf16* wrow = w + ((int64_t)hq * rb + i) * S;
    const int jlo = window > 0 ? max(0, p - window + 1 - lo) : 0;
    const int jhi = min(S - 1, p - lo);                    // inclusive
    const float sink = bf(proj[(int64_t)t * ldp + soff + hq]);
    float mx = lane == 0 ? sink : -INFINITY;
    for (int j = jlo + lane; j <= jhi; j += 32) {
        const int d = p - (lo + j);
        float b = 0.f;
        if (d < rel_extent) {
            const bf16* pj = projT + (int64_t)d * kRel;
#pragma unroll
            for (int c = 0; c < kRel; ++c) b = fmaf(r[c], bf(pj[c]), b);
        }
        mx = fmaxf(mx, srow[j] * scale + b);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = lane == 0 ? expf(sink - mx) : 0.f;
    for (int j = jlo + lane; j <= jhi; j += 32) {
        const int d = p - (lo + j);
        float b = 0.f;
        if (d < rel_extent) {
            const bf16* pj = projT + (int64_t)d * kRel;
#pragma unroll
            for (int c = 0; c < kRel; ++c) b = fmaf(r[c], bf(pj[c]), b);
        }
        sum += expf(srow[j] * scale + b - mx);
    }
    sum = warp_sum(sum);
    for (int j = lane; j < S; j += 32) {
        float v = 0.f;
        if (j >= jlo && j <= jhi) {
            const int d = p - (lo + j);
            float b = 0.f;
            if (d < rel_extent) {
                const bf16* pj = projT + (int64_t)d * kRel;
#pragma unroll
                for (int c = 0; c < kRel; ++c) b = fmaf(r[c], bf(pj[c]), b);
            }
            v = expf(srow[j] * scale + b - mx) / sum;
        }
        wrow[j] = to_bf(v);
    }
}

// o [t][hl][128] (rows r0 ..) = bf16(ov * sigmoid(gate)), ov [hl][rb][128] bf16 the weights' GEMM
// with the values. Grid (hl, rb), 128 threads.
__global__ void ink_pf_gate_kernel(const bf16* __restrict__ ov, const bf16* __restrict__ proj, int64_t ldp, int goff, int hl,
                                   int r0, int rb, bf16* __restrict__ o) {
    pdl_enter();
    const int hq = blockIdx.x, i = blockIdx.y, d = threadIdx.x, t = r0 + i;
    const float gate = bf(proj[(int64_t)t * ldp + goff + hq * kD + d]);
    o[((int64_t)t * hl + hq) * kD + d] = to_bf(bf(ov[((int64_t)hq * rb + i) * kD + d]) / (1.f + expf(-gate)));
}

// ------------------------------------------------------------------ attention, prefill rows, fused
// The prefill attention of rows that see many keys (a full layer deep in a context), in one pass
// over the keys: flash attention with tensor cores, after vision.cu's vit_attn_kernel. A block takes
// 16 rows x the 6 query heads of one key / value head (96 (row, head) pairs, a warp's 16 each), so
// each 64-key tile of K and V (transposed) is read from memory once for all of them; nothing of the
// [rows][keys] scores is written. Per pair: score = q.k / sqrt(128) + r . projT[p - j] (p - j <
// rel_extent), keys lo_key + window-limited .. p (causal), the sink as one more logit with a zero
// value, then o = bf16(bf16(sum / total) * sigmoid(gate)) - ink_attn_combine's rounding. P is rounded
// to bf16 for the P.V products, the sums are fp32 (the cuBLAS path rounds the normalized weights
// instead: the two agree to bf16 rounding). kv: key lo_key's row of [keys][kl][K 128 | V 128] rows
// rowk elements apart. Grid (ceil(t / 16), kl), 192 threads.
constexpr int kFR = 16;                 // rows a block
constexpr int kFP = kFR * kG;           // (row, head) pairs a block: 6 warps x 16
constexpr int kFK = 64;                 // keys a tile
constexpr int kFPad = 8;                // smem row padding (bf16): conflict-free fragments

__device__ __forceinline__ uint32_t ink_pack_bf16(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&p);
}

__device__ __forceinline__ void ink_mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(kFP * 2) ink_pf_flash_kernel(const bf16* __restrict__ qc, const bf16* __restrict__ rc,
                                                              const bf16* __restrict__ projT, int rel_extent,
                                                              const bf16* __restrict__ kv, int64_t rowk, int lo_key,
                                                              const bf16* __restrict__ proj, int64_t ldp, int soff, int goff,
                                                              int hl, int t, int p0, int window, bf16* __restrict__ o) {
    pdl_enter();
    __shared__ __align__(16) bf16 ks[kFK][kD + kFPad];
    __shared__ __align__(16) bf16 vt[kD][kFK + kFPad];
    __shared__ float rs[kFP][kRel];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int g = blockIdx.y, t0 = blockIdx.x * kFR;
    const int nrow = min(kFR, t - t0);
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale2 = rsqrtf((float)kD) * kLog2e;
    // pair m = row * 6 + head: this thread's two (rows r and r + 8 of its warp's 16)
    const int ma = warp * 16 + r, mb = ma + 8;
    const int ta = t0 + ma / kG, tb = t0 + mb / kG;
    const int ha = g * kG + ma % kG, hb = g * kG + mb % kG;
    const bool va = ma / kG < nrow, vb = mb / kG < nrow;
    const int pa = p0 + ta, pb = p0 + tb;
    const int la = window > 0 ? max(lo_key, pa - window + 1) : lo_key, lb = window > 0 ? max(lo_key, pb - window + 1) : lo_key;
    for (int u = threadIdx.x; u < kFP * kRel; u += blockDim.x) {
        const int m = u / kRel, c = u % kRel, tt = t0 + m / kG;
        rs[m][c] = m / kG < nrow ? bf(rc[((int64_t)tt * hl + g * kG + m % kG) * kRel + c]) : 0.f;
    }
    uint32_t qa[kD / 16][4];
#pragma unroll
    for (int st = 0; st < kD / 16; ++st) {
        const int d = st * 16 + q * 2;
        auto ld32 = [&](bool v, int tt, int hh, int dd) -> uint32_t {
            return v ? *reinterpret_cast<const uint32_t*>(qc + ((int64_t)tt * hl + hh) * kD + dd) : 0u;
        };
        qa[st][0] = ld32(va, ta, ha, d);
        qa[st][1] = ld32(vb, tb, hb, d);
        qa[st][2] = ld32(va, ta, ha, d + 8);
        qa[st][3] = ld32(vb, tb, hb, d + 8);
    }
    float acc[kD / 8][4];
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    // the block's keys: from the first row's window start to its last row's position
    const int pfirst = p0 + t0, plast = p0 + t0 + nrow - 1;
    const int kbeg = window > 0 ? max(lo_key, pfirst - window + 1) : lo_key;
    for (int k0 = kbeg; k0 <= plast; k0 += kFK) {
        __syncthreads();
        for (int i = threadIdx.x; i < kFK * (kD / 8); i += blockDim.x) {
            const int row = i / (kD / 8), c = (i % (kD / 8)) * 8, key = k0 + row;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key <= plast) {
                const bf16* src = kv + (int64_t)(key - lo_key) * rowk + g * 2 * kD;
                kk = *reinterpret_cast<const uint4*>(src + c);
                vv = *reinterpret_cast<const uint4*>(src + kD + c);
            }
            *reinterpret_cast<uint4*>(&ks[row][c]) = kk;
            const bf16* ve = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; ++j) vt[c + j][row] = ve[j];
        }
        __syncthreads();
        float s[kFK / 8][4];
#pragma unroll
        for (int nb = 0; nb < kFK / 8; ++nb) {
            s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
#pragma unroll
            for (int st = 0; st < kD / 16; ++st) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&ks[nb * 8 + r][st * 16 + q * 2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&ks[nb * 8 + r][st * 16 + q * 2 + 8]);
                ink_mma_bf16(s[nb], qa[st], b0, b1);
            }
        }
        // scale, the relative bias near the diagonal, the mask; log2 units from here
        const bool band = k0 + kFK - 1 > pfirst - rel_extent;
#pragma unroll
        for (int nb = 0; nb < kFK / 8; ++nb)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + nb * 8 + q * 2 + (j & 1);
                const bool hi = j >= 2;
                const int p = hi ? pb : pa, lo = hi ? lb : la;
                const bool ok = (hi ? vb : va) && key >= lo && key <= p;
                float v = s[nb][j] * scale2;
                if (band && ok && p - key < rel_extent) {
                    const bf16* pj = projT + (int64_t)(p - key) * kRel;
                    const float* rv = rs[hi ? mb : ma];
                    float b = 0.f;
#pragma unroll
                    for (int c = 0; c < kRel; ++c) b = fmaf(rv[c], bf(pj[c]), b);
                    v = fmaf(b, kLog2e, v);
                }
                s[nb][j] = ok ? v : -INFINITY;
            }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nb = 0; nb < kFK / 8; ++nb) {
            mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // a pair with no key yet keeps a finite reference (exp2 of -inf - it is 0)
        const float b0 = mx0 > -INFINITY ? mx0 : 0.f, b1 = mx1 > -INFINITY ? mx1 : 0.f;
        const float c0 = exp2f(m0 - b0), c1 = exp2f(m1 - b1);
        m0 = mx0;
        m1 = mx1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int db = 0; db < kD / 8; ++db) {
            acc[db][0] *= c0;
            acc[db][1] *= c0;
            acc[db][2] *= c1;
            acc[db][3] *= c1;
        }
        uint32_t pa_[kFK / 16][4];
#pragma unroll
        for (int nb = 0; nb < kFK / 8; ++nb) {
            const float e0 = exp2f(s[nb][0] - b0), e1 = exp2f(s[nb][1] - b0);
            const float e2 = exp2f(s[nb][2] - b1), e3 = exp2f(s[nb][3] - b1);
            l0 += e0 + e1;
            l1 += e2 + e3;
            pa_[nb >> 1][(nb & 1) * 2] = ink_pack_bf16(e0, e1);
            pa_[nb >> 1][(nb & 1) * 2 + 1] = ink_pack_bf16(e2, e3);
        }
#pragma unroll
        for (int db = 0; db < kD / 8; ++db)
#pragma unroll
            for (int st = 0; st < kFK / 16; ++st) {
                const uint32_t v0 = *reinterpret_cast<const uint32_t*>(&vt[db * 8 + r][st * 16 + q * 2]);
                const uint32_t v1 = *reinterpret_cast<const uint32_t*>(&vt[db * 8 + r][st * 16 + q * 2 + 8]);
                ink_mma_bf16(acc[db], pa_[st], v0, v1);
            }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    // the sink: one more logit, a zero value
    float fa = 0.f, fb = 0.f;
    if (va) {
        const float sk = bf(proj[(int64_t)ta * ldp + soff + ha]) * kLog2e, mm = fmaxf(m0, sk);
        const float f = exp2f(m0 - mm);
        fa = f / (l0 * f + exp2f(sk - mm));
    }
    if (vb) {
        const float sk = bf(proj[(int64_t)tb * ldp + soff + hb]) * kLog2e, mm = fmaxf(m1, sk);
        const float f = exp2f(m1 - mm);
        fb = f / (l1 * f + exp2f(sk - mm));
    }
#pragma unroll
    for (int db = 0; db < kD / 8; ++db) {
        const int d = db * 8 + q * 2;
        if (va) {
            const bf16* gt = proj + (int64_t)ta * ldp + goff + ha * kD + d;
            bf16* out = o + ((int64_t)ta * hl + ha) * kD + d;
            out[0] = to_bf(rbf(acc[db][0] * fa) / (1.f + expf(-bf(gt[0]))));
            out[1] = to_bf(rbf(acc[db][1] * fa) / (1.f + expf(-bf(gt[1]))));
        }
        if (vb) {
            const bf16* gt = proj + (int64_t)tb * ldp + goff + hb * kD + d;
            bf16* out = o + ((int64_t)tb * hl + hb) * kD + d;
            out[0] = to_bf(rbf(acc[db][2] * fb) / (1.f + expf(-bf(gt[0]))));
            out[1] = to_bf(rbf(acc[db][3] * fb) / (1.f + expf(-bf(gt[1]))));
        }
    }
}

// ------------------------------------------------------------------ snapshots
// Positions p0 .. p0 + n of a lane's rings ([layers][...][R][row], `ring` at the lane's part of
// layer 0, `ring_ls` elements a layer) into lin [layers][n][row] (`lin_ls` a layer), or back
// (to_ring). Grid (row / 256, n, layers).
__global__ void ink_ring_copy_kernel(bf16* __restrict__ ring, int64_t ring_ls, int R, int row, int p0, bf16* __restrict__ lin,
                                     int64_t lin_ls, int to_ring) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y, l = blockIdx.z;
    if (c >= row) return;
    bf16* a = ring + (int64_t)l * ring_ls + (int64_t)((p0 + j) & (R - 1)) * row + c;
    bf16* b = lin + (int64_t)l * lin_ls + (int64_t)j * row + c;
    if (to_ring) *a = *b; else *b = *a;
}

// ------------------------------------------------------------------ the head's logits
// logits [t][n] fp32 in place: cap * tanh(x / cap) (final_logit_softcapping), -inf for the ids past
// the tokens (id0 + column >= real: the head's padding rows are never emitted).
__global__ void ink_logits_kernel(float* __restrict__ x, int64_t ld, int n, float cap, int id0, int real) {
    pdl_enter();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= n) return;
    float* v = x + (int64_t)t * ld + c;
    *v = id0 + c >= real ? -INFINITY : cap * tanhf(*v / cap);
}

}  // namespace

extern "C" {

int ink_conv(void* stream, const void* x, int64_t ldx, int c, const void* w, int wmode, const void* dyn, int64_t ldd,
             const void* ring, int64_t lane_stride, const void* lanes, int nl, const void* meta, int t, float div, int omode,
             void* out, int64_t ldo, void* res, void* h) {
    pdl_launch(ink_conv_kernel, dim3((c + 255) / 256, t), dim3(256), 0, GST, (const bf16*)x, ldx, c, (const bf16*)w, wmode,
               (const bf16*)dyn, ldd, (const bf16*)ring, lane_stride, (const int*)lanes, nl, (const int*)meta, div, omode,
               (bf16*)out, ldo, (float*)res, (bf16*)h);
    GRET;
}

int ink_conv_state(void* stream, const void* x, int64_t ldx, int c, void* ring, int64_t lane_stride, const void* lanes, int nl,
                   const void* meta) {
    pdl_launch(ink_conv_state_kernel, dim3((c + 255) / 256, kConvRing, nl), dim3(256), 0, GST, (const bf16*)x, ldx, c, (bf16*)ring,
               lane_stride, (const int*)lanes, (const int*)meta);
    GRET;
}

int ink_mix(void* stream, const void* blocks, int64_t ldb, int nb, const void* cur, int cur_f32, const void* q, int nq,
            const void* nw, float neps, void* out, int64_t plane, int t, int h) {
    if (nb + 1 > kMaxBlocks || nq < 1) return 1;
    pdl_launch(ink_mix_kernel, dim3(nq, t), dim3(256), (size_t)(h + 8 * 2 * kMaxBlocks) * sizeof(float), GST, (const bf16*)blocks,
               ldb, nb, cur, cur_f32, (const float*)q, (const bf16*)nw, neps, (bf16*)out, plane, h);
    GRET;
}

int ink_heads(void* stream, const void* proj, int64_t ldp, int qoff, int roff, const void* qscale, const void* rscale,
              const void* kx, const void* vx, int hl, int kl, float qclip, float rclip, float eps, float alpha, float floor_n,
              const void* lanes, int nl, const void* meta, int ring, int64_t lane_rows, int t, void* qc, void* rc, void* cache) {
    if (hl != kl * kG) return 1;
    pdl_launch(ink_heads_kernel, dim3(hl + 2 * kl, t), dim3(kD), 0, GST, (const bf16*)proj, ldp, qoff, roff, (const bf16*)qscale,
               (const bf16*)rscale, (const bf16*)kx, (const bf16*)vx, hl, kl, qclip, rclip, eps, alpha, floor_n, (const int*)lanes,
               nl, (const int*)meta, ring, lane_rows, (bf16*)qc, (bf16*)rc, (bf16*)cache);
    GRET;
}

int ink_attn_part(void* stream, const void* qc, const void* rc, const void* proj_t, int rel_extent, const void* cache, int ring,
                  int64_t lane_rows, int window, const void* lanes, int nl, const void* meta, int hl, int kl, int ns, int t,
                  void* pm, void* pl, void* pacc) {
    if (hl != kl * kG || ns < 1) return 1;
    pdl_launch(ink_attn_part_kernel, dim3(ns, kl, t), dim3(256), 0, GST, (const bf16*)qc, (const bf16*)rc, (const bf16*)proj_t,
               rel_extent, (const bf16*)cache, ring, lane_rows, window, (const int*)lanes, nl, (const int*)meta, hl, kl, ns,
               (float*)pm, (float*)pl, (float*)pacc);
    GRET;
}

int ink_attn_combine(void* stream, const void* pm, const void* pl, const void* pacc, const void* proj, int64_t ldp, int soff,
                     int goff, int hl, int kl, int ns, int t, void* o) {
    pdl_launch(ink_attn_combine_kernel, dim3(hl, t), dim3(kD), 0, GST, (const float*)pm, (const float*)pl, (const float*)pacc,
               (const bf16*)proj, ldp, soff, goff, hl, kl, ns, (bf16*)o);
    GRET;
}

int ink_kv_window(void* stream, const void* cache, int ring, int64_t lane_rows, const void* lanes, const void* meta, int z,
                  int lo, int s, int kl, void* out) {
    pdl_launch(ink_kv_window_kernel, dim3(s, (kl * 2 * kD + 255) / 256), dim3(256), 0, GST, (const bf16*)cache, ring, lane_rows,
               (const int*)lanes, (const int*)meta, z, lo, kl, (bf16*)out);
    GRET;
}

int ink_pf_softmax(void* stream, const void* sc, const void* rc, const void* proj_t, int rel_extent, const void* proj, int64_t ldp,
                   int soff, int hl, int r0, int rb, int s, int lo, int p0, int window, void* w) {
    pdl_launch(ink_pf_softmax_kernel, dim3((rb + 7) / 8, hl), dim3(256), 0, GST, (const float*)sc, (const bf16*)rc,
               (const bf16*)proj_t, rel_extent, (const bf16*)proj, ldp, soff, hl, r0, rb, s, lo, p0, window, (bf16*)w);
    GRET;
}

int ink_pf_flash(void* stream, const void* qc, const void* rc, const void* proj_t, int rel_extent, const void* kv, int64_t rowk,
                 int lo_key, const void* proj, int64_t ldp, int soff, int goff, int hl, int kl, int t, int p0, int window, void* o) {
    if (hl != kl * kG || t <= 0) return t < 0;
    pdl_launch(ink_pf_flash_kernel, dim3((t + kFR - 1) / kFR, kl), dim3(kFP * 2), 0, GST, (const bf16*)qc, (const bf16*)rc,
               (const bf16*)proj_t, rel_extent, (const bf16*)kv, rowk, lo_key, (const bf16*)proj, ldp, soff, goff, hl, t, p0, window,
               (bf16*)o);
    GRET;
}

int ink_pf_gate(void* stream, const void* ov, const void* proj, int64_t ldp, int goff, int hl, int r0, int rb, void* o) {
    pdl_launch(ink_pf_gate_kernel, dim3(hl, rb), dim3(kD), 0, GST, (const bf16*)ov, (const bf16*)proj, ldp, goff, hl, r0, rb,
               (bf16*)o);
    GRET;
}

int ink_ring_copy(void* stream, void* ring, int64_t ring_ls, int r, int row, int p0, int n, int layers, void* lin,
                  int64_t lin_ls, int to_ring) {
    if (n <= 0 || layers <= 0) return 0;
    pdl_launch(ink_ring_copy_kernel, dim3((row + 255) / 256, n, layers), dim3(256), 0, GST, (bf16*)ring, ring_ls, r, row, p0,
               (bf16*)lin, lin_ls, to_ring);
    GRET;
}

int ink_logits(void* stream, void* x, int64_t ld, int n, int t, float cap, int id0, int real) {
    pdl_launch(ink_logits_kernel, dim3((n + 255) / 256, t), dim3(256), 0, GST, (float*)x, ld, n, cap, id0, real);
    GRET;
}

}  // extern "C"
