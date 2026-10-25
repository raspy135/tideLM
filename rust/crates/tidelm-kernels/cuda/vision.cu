// The vision tower's kernels (the checkpoint's inference/vision.py: a 32-block bf16 ViT with
// 2D RoPE and full bidirectional attention over one image, then the 3x3 aligner). The GEMMs
// are ds41.cu's tensor-core kernels; what is here is the rest, each kernel rounding where the
// PyTorch module does: every op's output is bf16, its math fp32.
//
// Launchers take the stream first and return cudaGetLastError() as an int, as ds41.cu's do.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

// two bf16 as one mma operand word, the lower k first
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&p);
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int kAttnD = 64;   // head_dim
constexpr int kAttnBQ = 64;  // query rows a block: 4 warps x 16
constexpr int kAttnBK = 64;  // keys a tile
constexpr int kAttnPad = 8;  // smem row padding (bf16): rows 36 words apart, conflict-free fragments

// Flash attention over one image (F.scaled_dot_product_attention, no mask, scale 1/sqrt(64)):
// qkv [N][3][H][64] rows of wqkv's output, out [N][H][64]. A warp holds 16 query rows' Q
// fragments and their running max / sum / output; each 64-key tile of K and V (transposed) is
// staged in shared memory. P is rounded to bf16 for the P.V products, the row sums are fp32.
__global__ void __launch_bounds__(128) vit_attn_kernel(const __nv_bfloat16* __restrict__ qkv, int N, int H,
                                                       float scale_log2, __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(16) __nv_bfloat16 ks[kAttnBK][kAttnD + kAttnPad];
    __shared__ __align__(16) __nv_bfloat16 vt[kAttnD][kAttnBK + kAttnPad];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
    const int h = blockIdx.y;
    const int64_t ld = (int64_t)3 * H * kAttnD;
    const int q0 = blockIdx.x * kAttnBQ + warp * 16;
    const int qa_row = q0 + r, qb_row = q0 + r + 8;

    uint32_t qa[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int d = s * 16 + q * 2;
        auto ld32 = [&](int row, int dd) -> uint32_t {
            if (row >= N) return 0u;
            return *reinterpret_cast<const uint32_t*>(qkv + row * ld + h * kAttnD + dd);
        };
        qa[s][0] = ld32(qa_row, d);
        qa[s][1] = ld32(qb_row, d);
        qa[s][2] = ld32(qa_row, d + 8);
        qa[s][3] = ld32(qb_row, d + 8);
    }
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int k0 = 0; k0 < N; k0 += kAttnBK) {
        __syncthreads();
        for (int i = threadIdx.x; i < kAttnBK * (kAttnD / 8); i += blockDim.x) {
            const int row = i / (kAttnD / 8), c = (i % (kAttnD / 8)) * 8, key = k0 + row;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key < N) {
                kv = *reinterpret_cast<const uint4*>(qkv + key * ld + (int64_t)H * kAttnD + h * kAttnD + c);
                vv = *reinterpret_cast<const uint4*>(qkv + key * ld + (int64_t)2 * H * kAttnD + h * kAttnD + c);
            }
            *reinterpret_cast<uint4*>(&ks[row][c]) = kv;
            const __nv_bfloat16* ve = reinterpret_cast<const __nv_bfloat16*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; ++j) vt[c + j][row] = ve[j];
        }
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s[nb][j] = 0.f;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&ks[nb * 8 + r][st * 16 + q * 2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&ks[nb * 8 + r][st * 16 + q * 2 + 8]);
                mma_bf16(s[nb], qa[st], b0, b1);
            }
        }
        if (k0 + kAttnBK > N) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (k0 + nb * 8 + q * 2 + (j & 1) >= N) s[nb][j] = -INFINITY;
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // the first tile always has key 0, so the maxima are finite from here on
        const float c0 = exp2f((m0 - mx0) * scale_log2), c1 = exp2f((m1 - mx1) * scale_log2);
        m0 = mx0;
        m1 = mx1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            o[db][0] *= c0;
            o[db][1] *= c0;
            o[db][2] *= c1;
            o[db][3] *= c1;
        }
        // P's C fragments are the A fragments of P.V: keys 16j..16j+15 are blocks 2j, 2j+1
        uint32_t pa[4][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            const float p0 = exp2f((s[nb][0] - m0) * scale_log2), p1 = exp2f((s[nb][1] - m0) * scale_log2);
            const float p2 = exp2f((s[nb][2] - m1) * scale_log2), p3 = exp2f((s[nb][3] - m1) * scale_log2);
            l0 += p0 + p1;
            l1 += p2 + p3;
            pa[nb >> 1][(nb & 1) * 2] = pack_bf16(p0, p1);
            pa[nb >> 1][(nb & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
#pragma unroll
        for (int db = 0; db < 8; ++db)
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&vt[db * 8 + r][st * 16 + q * 2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&vt[db * 8 + r][st * 16 + q * 2 + 8]);
                mma_bf16(o[db], pa[st], b0, b1);
            }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int64_t ldo = (int64_t)H * kAttnD;
#pragma unroll
    for (int db = 0; db < 8; ++db) {
        const int d = h * kAttnD + db * 8 + q * 2;
        if (qa_row < N)
            *reinterpret_cast<__nv_bfloat162*>(out + qa_row * ldo + d) = __floats2bfloat162_rn(o[db][0] * i0, o[db][1] * i0);
        if (qb_row < N)
            *reinterpret_cast<__nv_bfloat162*>(out + qb_row * ldo + d) = __floats2bfloat162_rn(o[db][2] * i1, o[db][3] * i1);
    }
}

// apply_rotary on q and k: per head, pairs (d, d + 32) of the 64 dims turn by angle d of the
// row's table (fp32 cos / sin [N][32]: row angles then column angles); each product and the
// difference / sum rounded separately, as PyTorch's elementwise kernels do.
__global__ void vit_rope_kernel(__nv_bfloat16* qkv, int N, int H, const float* __restrict__ cs,
                                const float* __restrict__ sn) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // over N x 2 x H x 32
    if (i >= (int64_t)N * 2 * H * 32) return;
    const int d = i % 32, hh = (i / 32) % (2 * H);
    const int64_t n = i / (32 * 2 * H);
    __nv_bfloat16* x = qkv + n * (3 * H * kAttnD) + (int64_t)hh * kAttnD;  // q heads, then k heads
    const float c = cs[n * 32 + d], s = sn[n * 32 + d];
    const float x1 = bf(x[d]), x2 = bf(x[d + 32]);
    x[d] = to_bf(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
    x[d + 32] = to_bf(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
}

// A Linear's epilogue: t = bf16(y + bias) (bias may be null), then gelu (erf) when `gelu`,
// then bf16(residual + t) when `residual` is set; out may be residual.
__global__ void vit_epilogue_kernel(const float* __restrict__ y, const __nv_bfloat16* __restrict__ bias,
                                    const __nv_bfloat16* residual, __nv_bfloat16* out, int M, int Ncols, int gelu) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * Ncols) return;
    const int n = i % Ncols;
    float t = bf(to_bf(bias ? __fadd_rn(y[i], bf(bias[n])) : y[i]));
    if (gelu) t = bf(to_bf(0.5f * t * (1.f + erff(t * 0.70710678118654752f))));
    if (residual) t = __fadd_rn(bf(residual[i]), t);
    out[i] = to_bf(t);
}

// MLP activation: bf16(bf16(silu(gate)) * up), gate and up the two halves of w1's row
__global__ void vit_silu_mul_kernel(const __nv_bfloat16* __restrict__ gu, int M, int I, __nv_bfloat16* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * I) return;
    const int64_t m = i / I;
    const int n = i % I;
    const float g = bf(gu[m * 2 * I + n]), u = bf(gu[m * 2 * I + I + n]);
    const float s = bf(to_bf(g / (1.f + expf(-g))));
    out[i] = to_bf(__fmul_rn(s, u));
}

// The aligner's input: x [nh * nw][C] as an nh x nw grid, zero-padded to multiples of 3, then
// F.unfold(3, stride 3) transposed - out[oh * Lw + ow][c * 9 + ki * 3 + kj] = x[3oh+ki][3ow+kj][c]
__global__ void vit_unfold_kernel(const __nv_bfloat16* __restrict__ x, int nh, int nw, int C,
                                  __nv_bfloat16* __restrict__ out) {
    const int Lh = (nh + 2) / 3, Lw = (nw + 2) / 3;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)Lh * Lw * C * 9) return;
    const int f = i % (C * 9);
    const int64_t l = i / (C * 9);
    const int c = f / 9, ki = (f % 9) / 3, kj = f % 3;
    const int y = (l / Lw) * 3 + ki, xx = (l % Lw) * 3 + kj;
    out[i] = (y < nh && xx < nw) ? x[((int64_t)y * nw + xx) * C + c] : to_bf(0.f);
}

}  // namespace

extern "C" int ds_vit_attn(void* stream, const void* qkv, int N, int H, void* out) {
    if (N < 1 || H < 1) return cudaErrorInvalidValue;
    dim3 grid((N + kAttnBQ - 1) / kAttnBQ, H);
    const float scale_log2 = 0.125f * 1.4426950408889634f;
    vit_attn_kernel<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(qkv), N, H, scale_log2, static_cast<__nv_bfloat16*>(out));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_vit_rope(void* stream, void* qkv, int N, int H, const void* cos, const void* sin) {
    const int64_t total = (int64_t)N * 2 * H * 32;
    vit_rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(qkv), N, H, static_cast<const float*>(cos), static_cast<const float*>(sin));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_vit_epilogue(void* stream, const void* y, const void* bias, const void* residual, void* out, int M,
                               int N, int gelu) {
    const int64_t total = (int64_t)M * N;
    vit_epilogue_kernel<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(y), static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(residual),
        static_cast<__nv_bfloat16*>(out), M, N, gelu);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_vit_silu_mul(void* stream, const void* gu, int M, int I, void* out) {
    const int64_t total = (int64_t)M * I;
    vit_silu_mul_kernel<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(gu), M, I, static_cast<__nv_bfloat16*>(out));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_vit_unfold(void* stream, const void* x, int nh, int nw, int C, void* out) {
    const int64_t total = (int64_t)((nh + 2) / 3) * ((nw + 2) / 3) * C * 9;
    vit_unfold_kernel<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), nh, nw, C, static_cast<__nv_bfloat16*>(out));
    return static_cast<int>(cudaGetLastError());
}
