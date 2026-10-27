// EXL3 routed-expert decode kernels.
//
// Ported from TensorFold's src/tensorfold/cuda/exl3/{decode.cuh, experts_grouped.cuh, experts.cu}
// (Apache-2.0), which follow ExLlamaV3's trellis format (MIT, Copyright (c) 2025 Turboderp).  The
// device arithmetic -- tile decode, codebook, mma, the 128-point Walsh-Hadamard butterfly -- is
// theirs verbatim; the launchers, the arena-slot routing and the C ABI are this repo's.  NOT ported:
// TensorFold's DSV41 x3ld.cu/loads.py ("GLM patch 0580", AGPL-lineage) -- the load schedule here is
// the plain grouped kernel.
//
// Pipeline for one MoE call (see tools/exl3_moe_cuda.py):
//   group   : distinct arena slots in the call -> uids/members (<= maxm rows a slot)
//   rot_in  : Xh = fp16((x * suh) @ H) for gate and up, per member row
//   grouped : Z = Xh @ W_q, trellis decoded into mma fragments in registers
//   gateup  : sum splits, rotate, * svh, SwiGLU limit 10, then Xd = fp16((act * suh_d) @ H)
//   down    : Zd = Xd @ W_q(down) then rotate * svh_d and the top-k weighted combine
// Every shape is static and data-independent: no host sync, so this is CUDA-graph capturable.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// EXL3_STREAM=1: decode's trellis words load evict-first (load_word); a benchmark switch, off: the
// kernels gain 1-3% (routed) to 7-12% (the small dense matrices), ~1% of a decode pass (gotchas)
#ifndef EXL3_STREAM
#define EXL3_STREAM 0
#endif

#define EXL3_CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "exl3_moe_cuda: %s\n", msg); exit(2); } } while (0)

namespace exl3 {

// ---------------------------------------------------------------- tile decode (TensorFold decode.cuh)

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// Two codebook values (0 3inst, 1 mcg, 2 mul1) as half2, bit-identical to ExLlamaV3's decode.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

// EXL3_STREAM: trellis words through __ldcs (evict-first: read once a pass) instead of __ldg
__device__ __forceinline__ uint32_t load_word(const uint32_t* p) {
#if EXL3_STREAM
    return __ldcs(p);
#else
    return __ldg(p);
#endif
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = load_word(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? load_word(p + l * 32) : 0u;
    }
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// cp.async (sm_80+): one 4-byte word global -> shared, bypassing the register file. Used by the
// cp.async prefill variant (smem_tiles_ca) to stage raw trellis words S steps ahead where the
// register pipeline (pf[PF]) covers only PF; 4-byte size so no alignment is assumed beyond the
// word's own. wait_group<N> blocks until at most N commit groups are still in flight.
__device__ __forceinline__ void cp_async4(uint32_t* smem_dst, const uint32_t* gmem_src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    // An evict-first L2 policy here (createpolicy + .L2::cache_hint; a 4-byte cp.async has no .cg)
    // raised an illegal instruction on GB10, and is moot: this prefill GEMM is compute-bound, at
    // 51-57 GB/s of trellis bytes (glm_bench_moe --prefill, 2026-10-27).
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem_src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight.
template <int CB, int K2, int NT, int PF>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[4] = {load_pair(x0 + k, ok0), load_pair(x1 + k, ok1), load_pair(x0 + k + 8, ok0),
                                 load_pair(x1 + k + 8, ok1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

__host__ __device__ constexpr bool k2_supported(int k2) { return k2 >= 2 && k2 <= 16; }

// Program (expert u, n block, split and member tile): up to 16 members times W_q over the split's K range.
// Here `uids[u]` is an ARENA SLOT id; `tp*[slot]` is that slot's trellis pointer.
template <int CB, int NT, int W, int PF, int LO, int HI>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    if (e < 0) return;                    // prefill routing pads unused blocks with -1
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define EXL3_CASE(K2_)                                                                                          \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);    \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        EXL3_CASE(2) EXL3_CASE(3) EXL3_CASE(4) EXL3_CASE(5) EXL3_CASE(6) EXL3_CASE(7) EXL3_CASE(8)
        EXL3_CASE(9) EXL3_CASE(10) EXL3_CASE(11) EXL3_CASE(12) EXL3_CASE(13) EXL3_CASE(14) EXL3_CASE(15)
        EXL3_CASE(16)
#undef EXL3_CASE
        default:
            __trap();
    }

    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

template <int CB>
void grouped_launch(const void* x0, const void* x1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                    const int* k2_1, const int* uids, const int* ucount, const int* members, float* z, int K, int N,
                    int P, int SK, int maxm, int slots, int nexp_max, int mats, int nt, int warps, int pf, int lo,
                    int hi, cudaStream_t stream) {
    const int MT = (maxm + 15) / 16;
    dim3 grid((unsigned)nexp_max, (unsigned)(N / (16 * nt)), (unsigned)(mats * SK * MT));
#define EXL3_LAUNCH(NT_, W_, PF_, LO_, HI_)                                                                     \
    grouped_kernel<CB, NT_, W_, PF_, LO_, HI_><<<grid, W_ * 32, 0, stream>>>(                                   \
        (const half*)x0, (const half*)x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, SK, maxm,   \
        slots)
#define EXL3_RANGES(NT_, W_, PF_)                                                                               \
    if (lo == 8 && hi == 8) EXL3_LAUNCH(NT_, W_, PF_, 8, 8);                                                    \
    else if (lo >= 2 && hi <= 10) EXL3_LAUNCH(NT_, W_, PF_, 2, 10);                                             \
    else EXL3_LAUNCH(NT_, W_, PF_, 2, 16);
    if (nt == 8 && warps == 4 && pf == 1) { EXL3_RANGES(8, 4, 1) }
    else if (nt == 8 && warps == 4 && pf == 2) { EXL3_RANGES(8, 4, 2) }
    else if (nt == 4 && warps == 4 && pf == 2) { EXL3_RANGES(4, 4, 2) }
    else { EXL3_CHECK(false, "unsupported tile setting"); }
#undef EXL3_RANGES
#undef EXL3_LAUNCH
}

// ---------------------------------------------------------------- prefill: one decode per tile, MT m-tiles

// One warp's NT n-tiles over the WHOLE K range, for the MT 16-row m-tiles of its block: every trellis
// tile is decoded into mma fragments once and multiplied into all live m-tiles. grouped_kernel (the
// decode kernel) instead gives each 16-row m-tile its own program, which decodes -- and reads from
// DRAM -- the expert's weights again for every 16 rows of a prefill block.
template <int CB, int K2, int NT, int PF, int MT>
__device__ __forceinline__ void prefill_tiles(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0,
                                              const half* __restrict__ X, int K, const int* rows_sh, int nmt,
                                              int lane, float (&acc)[MT][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + (size_t)nt0 * TW + lane;
    const int g = lane >> 2, t = lane & 3;
    const half* xr0[MT];
    const half* xr1[MT];
    bool ok0[MT], ok1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = rows_sh[m * 16 + g], r1 = rows_sh[m * 16 + g + 8];
        ok0[m] = r0 >= 0;
        ok1[m] = r1 >= 0;
        xr0[m] = X + (size_t)(ok0[m] ? r0 : 0) * K + 2 * t;
        xr1[m] = X + (size_t)(ok1[m] ? r1 : 0) * K + 2 * t;
    }
    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < KT)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < KT; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < KT) {
                uint32_t b[NT][2][2];
#pragma unroll
                for (int i = 0; i < NT; ++i) decode_tile<CB, K2>(pf[d][i], map, lane, b[i][0], b[i][1]);
                if (it + PF < KT)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = it * 16;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < nmt) {
                        uint32_t a[4] = {load_pair(xr0[m] + k, ok0[m]), load_pair(xr1[m] + k, ok1[m]),
                                         load_pair(xr0[m] + k + 8, ok0[m]), load_pair(xr1[m] + k + 8, ok1[m])};
#pragma unroll
                        for (int i = 0; i < NT; ++i) {
                            mma16816(acc[m][i][0], a, b[i][0]);
                            mma16816(acc[m][i][1], a, b[i][1]);
                        }
                    }
                }
            }
        }
    }
}

// Program (n block, routing block u, matrix): MT*16 member rows of one arena slot times the warps' NT
// n-tiles each. Members are build_routing's block (row*32 + k codes, -1 padding at the end), so the
// live rows of a block are a prefix and m-tiles past it are skipped. No split-K: Z is [mats, P, N].
template <int CB, int NT, int W, int PF, int MT, int LO, int HI>
__global__ void __launch_bounds__(W * 32) prefill_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int slots) {
    const int u = blockIdx.y;
    if (u >= ucount[0]) return;
    const int e = uids[u];
    if (e < 0) return;
    const int mat = blockIdx.z;
    __shared__ int rows_sh[MT * 16];
    for (int i = threadIdx.x; i < MT * 16; i += W * 32) {
        const int code = members[(size_t)u * MT * 16 + i];
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    int nmt = 0;
#pragma unroll
    for (int m = 0; m < MT; ++m)
        if (rows_sh[m * 16] >= 0) nmt = m + 1;
    if (nmt == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = N >> 4, KT = K >> 4;
    const int nt0 = (blockIdx.x * W + warp) * NT;
    if (nt0 >= NTILES) return;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];

    float acc[MT][NT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    switch (k2) {
#define EXL3_PCASE(K2_)                                                                                         \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            prefill_tiles<CB, K2_, NT, PF, MT>(T, NTILES, KT, nt0, X, K, rows_sh, nmt, lane, acc);              \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        EXL3_PCASE(2) EXL3_PCASE(3) EXL3_PCASE(4) EXL3_PCASE(5) EXL3_PCASE(6) EXL3_PCASE(7) EXL3_PCASE(8)
        EXL3_PCASE(9) EXL3_PCASE(10)
#undef EXL3_PCASE
        default:
            __trap();
    }

    float* Zm = Z + (size_t)mat * P * N;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= nmt) break;
        const int r0 = rows_sh[m * 16 + g], r1 = rows_sh[m * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (nt0 + i) * 16 + h * 8 + 2 * t;
                if (r0 >= 0)
                    *reinterpret_cast<float2*>(Zm + (size_t)r0 * N + col) = make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (r1 >= 0)
                    *reinterpret_cast<float2*>(Zm + (size_t)r1 * N + col) = make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}

template <int CB>
void prefill_launch(const void* x0, const void* x1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                    const int* k2_1, const int* uids, const int* ucount, const int* members, float* z, int K, int N,
                    int P, int slots, int nblocks, int mats, int bm, int nt, int warps, int pf, int lo, int hi,
                    cudaStream_t stream) {
    EXL3_CHECK(lo >= 4 && hi <= 6, "prefill kernel is built for 2- and 3-bit trellis (K2 4..6)");
    const int ntiles = N / 16;
#define EXL3_PLAUNCH(NT_, W_, PF_, MT_)                                                                         \
    {                                                                                                           \
        dim3 grid((unsigned)((ntiles + NT_ * W_ - 1) / (NT_ * W_)), (unsigned)nblocks, (unsigned)mats);         \
        prefill_kernel<CB, NT_, W_, PF_, MT_, 4, 6><<<grid, W_ * 32, 0, stream>>>(                              \
            (const half*)x0, (const half*)x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, slots);  \
    }
    if (bm == 64 && nt == 4 && warps == 4 && pf == 2) EXL3_PLAUNCH(4, 4, 2, 4)
    else if (bm == 64 && nt == 2 && warps == 4 && pf == 2) EXL3_PLAUNCH(2, 4, 2, 4)
    else if (bm == 64 && nt == 4 && warps == 2 && pf == 2) EXL3_PLAUNCH(4, 2, 2, 4)
    else if (bm == 64 && nt == 2 && warps == 8 && pf == 2) EXL3_PLAUNCH(2, 8, 2, 4)
    else if (bm == 32 && nt == 4 && warps == 4 && pf == 2) EXL3_PLAUNCH(4, 4, 2, 2)
    else if (bm == 32 && nt == 8 && warps == 4 && pf == 2) EXL3_PLAUNCH(8, 4, 2, 2)
    else if (bm == 64 && nt == 4 && warps == 4 && pf == 3) EXL3_PLAUNCH(4, 4, 3, 4)
    else if (bm == 32 && nt == 2 && warps == 4 && pf == 2) EXL3_PLAUNCH(2, 4, 2, 2)
    else if (bm == 32 && nt == 4 && warps == 2 && pf == 2) EXL3_PLAUNCH(4, 2, 2, 2)
    else if (bm == 32 && nt == 2 && warps == 8 && pf == 2) EXL3_PLAUNCH(2, 8, 2, 2)
    else if (bm == 32 && nt == 4 && warps == 8 && pf == 2) EXL3_PLAUNCH(4, 8, 2, 2)
    else if (bm == 32 && nt == 4 && warps == 4 && pf == 3) EXL3_PLAUNCH(4, 4, 3, 2)
    else if (bm == 32 && nt == 4 && warps == 4 && pf == 4) EXL3_PLAUNCH(4, 4, 4, 2)
    else if (bm == 48 && nt == 4 && warps == 4 && pf == 2) EXL3_PLAUNCH(4, 4, 2, 3)
    else if (bm == 16 && nt == 8 && warps == 4 && pf == 2) EXL3_PLAUNCH(8, 4, 2, 1)
    else if (bm == 16 && nt == 4 && warps == 4 && pf == 2) EXL3_PLAUNCH(4, 4, 2, 1)
    else EXL3_CHECK(false, "unsupported prefill tile setting");
#undef EXL3_PLAUNCH
}

// ---------------------------------------------------------------- prefill: tiles decoded once a block, shared through smem

// prefill_kernel decodes an expert's trellis once for the MT m-tiles of a warp -- and MT is 2: the
// 32-row routing block of the 2026-10-08 sweep, where 4 m-tiles' accumulators cut occupancy. An
// expert with ~100 rows in a 2,048-row chunk is therefore decoded three or four times, and the
// decode (integer ALU: the codebook hash, the bit windows), not the mma and not the DRAM bytes, is
// where a GLM prefill MoE layer's ~29 ms went (2026-10-20, docs/gotchas.md). Here a block owns
// BM = W * MTW * 16 rows of one routing block and NTB n-tiles. Each k-step the warps decode the
// step's KS x NTB tiles between them -- TPW tiles a warp -- into shared memory, and every warp
// multiplies all of them into its own MTW m-tiles: the trellis is decoded once for BM rows. A is
// read from global memory (L2) per warp as before. Two stages, one __syncthreads a step. The
// same k order and the same mma chain per accumulator as prefill_kernel, so the outputs are
// bit-identical to it (tools/bench_exl3_prefill.py checks).
// PROBE (bench-only, exl3m_prefill_smem_probe): 1 = trellis loads + decode + smem stores, with the
// mma loop replaced by a cheap smem checksum (no A loads, no tensor cores); 2 = A loads + mma on a
// constant tile (no trellis loads, no decode). The two probes bracket where a layer's time goes --
// if probe1 + probe2 ~= the full kernel, the load/decode side and the mma side are serialized.
// PROBE = 0 is the shipped kernel, byte for byte.
template <int CB, int K2, int W, int MTW, int NTB, int PF, int PROBE = 0>
__device__ __forceinline__ void smem_tiles(const uint32_t* __restrict__ T, int NTILES, int KT, int ntb0,
                                           const half* __restrict__ X, int K, const int* rows_sh, int nmt,
                                           int warp, int lane, uint32_t* Bs, float (&acc)[MTW][NTB][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    constexpr int KS = (W >= NTB) ? W / NTB : 1;          // k-tiles a step
    constexpr int TPW = (NTB > W) ? NTB / W : 1;          // tiles a warp decodes a step
    constexpr int TILES = KS * NTB;                       // tiles a step
    constexpr int STAGE = TILES * 32 * 4;                 // uint32 words a stage
    static_assert(W * TPW == TILES, "the warps must cover a step's tiles exactly");
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const int g = lane >> 2, t = lane & 3;
    const int NS = KT / KS;                               // steps (the launcher checks KT % KS == 0)
    // This warp's tiles of a step: ti = warp * TPW + j is k-tile ti / NTB of the step, n-tile ti % NTB.
    const uint32_t* tp[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int ti = warp * TPW + j;
        tp[j] = T + ((size_t)(ti / NTB) * NTILES + ntb0 + (ti % NTB)) * TW + lane;
    }
    const int m0 = warp * MTW;
    const half* xr0[MTW];
    const half* xr1[MTW];
    bool ok0[MTW], ok1[MTW];
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
        const int r0 = rows_sh[(m0 + m) * 16 + g], r1 = rows_sh[(m0 + m) * 16 + g + 8];
        ok0[m] = r0 >= 0;
        ok1[m] = r1 >= 0;
        xr0[m] = X + (size_t)(ok0[m] ? r0 : 0) * K + 2 * t;
        xr1[m] = X + (size_t)(ok1[m] ? r1 : 0) * K + 2 * t;
    }
    const int nmw = min(max(nmt - m0, 0), MTW);           // this warp's live m-tiles

    uint32_t chk = 0;                                     // PROBE 1: keeps the smem stores observable
    auto consume = [&](int stage, int s) {
        const uint32_t* st = Bs + stage * STAGE;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const int k = (s * KS + kk) * 16;
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                if (m < nmw) {
                    if constexpr (PROBE == 1) {
#pragma unroll
                        for (int n = 0; n < NTB; ++n) {
                            const uint4 v = *reinterpret_cast<const uint4*>(st + ((kk * NTB + n) * 32 + lane) * 4);
                            chk ^= v.x ^ v.w;
                        }
                    } else {
                    uint32_t a[4] = {load_pair(xr0[m] + k, ok0[m]), load_pair(xr1[m] + k, ok1[m]),
                                     load_pair(xr0[m] + k + 8, ok0[m]), load_pair(xr1[m] + k + 8, ok1[m])};
#pragma unroll
                    for (int n = 0; n < NTB; ++n) {
                        const uint4 v = *reinterpret_cast<const uint4*>(st + ((kk * NTB + n) * 32 + lane) * 4);
                        const uint32_t b0[2] = {v.x, v.y}, b1[2] = {v.z, v.w};
                        mma16816(acc[m][n][0], a, b0);
                        mma16816(acc[m][n][1], a, b1);
                    }
                    }
                }
            }
        }
    };
    auto store = [&](int stage, const uint32_t (&b)[TPW][4]) {
        uint32_t* st = Bs + stage * STAGE;
#pragma unroll
        for (int j = 0; j < TPW; ++j)
            *reinterpret_cast<uint4*>(st + ((warp * TPW + j) * 32 + lane) * 4) = make_uint4(b[j][0], b[j][1], b[j][2], b[j][3]);
    };

    // Trellis words PF steps ahead: step q's words sit in slot q % PF.
    uint32_t pf[PF][TPW][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < NS && PROBE != 2)
#pragma unroll
            for (int j = 0; j < TPW; ++j) load_words<K2>(pf[d][j], tp[j] + (size_t)d * KS * kstride, lane);
    {
        uint32_t b[TPW][4];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            if constexpr (PROBE == 2) {
                b[j][0] = b[j][1] = b[j][2] = b[j][3] = 0x3c003c00u;   // half2 (1, 1)
            } else {
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(pf[0][j], map, lane, b0, b1);
                b[j][0] = b0[0]; b[j][1] = b0[1]; b[j][2] = b1[0]; b[j][3] = b1[1];
            }
        }
        if (PF < NS && PROBE != 2)
#pragma unroll
            for (int j = 0; j < TPW; ++j) load_words<K2>(pf[0][j], tp[j] + (size_t)PF * KS * kstride, lane);
        store(0, b);
    }
    __syncthreads();
    // Step q is decoded while step q - 1 is multiplied; q runs 1 .. NS - 1 in groups of PF so the
    // slot (q % PF == (d + 1) % PF) is a compile-time index.
    for (int qb = 1; qb < NS; qb += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int q = qb + d;
            if (q < NS) {
                uint32_t b[TPW][4];
#pragma unroll
                for (int j = 0; j < TPW; ++j) {
                    if constexpr (PROBE == 2) {
                        b[j][0] = b[j][1] = b[j][2] = b[j][3] = 0x3c003c00u;
                    } else {
                        uint32_t b0[2], b1[2];
                        decode_tile<CB, K2>(pf[(d + 1) % PF][j], map, lane, b0, b1);
                        b[j][0] = b0[0]; b[j][1] = b0[1]; b[j][2] = b1[0]; b[j][3] = b1[1];
                    }
                }
                if (q + PF < NS && PROBE != 2)
#pragma unroll
                    for (int j = 0; j < TPW; ++j)
                        load_words<K2>(pf[(d + 1) % PF][j], tp[j] + (size_t)(q + PF) * KS * kstride, lane);
                consume((q - 1) & 1, q - 1);
                store(q & 1, b);
                __syncthreads();
            }
        }
    }
    consume((NS - 1) & 1, NS - 1);
    if constexpr (PROBE == 1) acc[0][0][0][0] += (float)(chk & 1u);
}

// smem_tiles with the raw trellis words staged global -> shared by cp.async, S steps deep, instead
// of the PF-deep register pipeline. The register path gives a warp PF steps of load lookahead and
// stalls at first use in decode; cp.async issues the copy without a consuming register, so the
// lookahead deepens to S at no register cost and the wait moves to an explicit cp_wait a step
// ahead of use. Decode order, store order and the mma chain are smem_tiles' exactly, so the output
// is bit-identical to it (tools/bench_exl3_prefill.py checks `vs first`). Each warp copies, decodes
// and consumes only its own TPW tiles' slots of Ws, so no barrier guards Ws beyond the per-thread
// cp_wait; one commit group per step keeps the wait count exact (empty groups near the end included).
template <int CB, int K2, int W, int MTW, int NTB, int S>
__device__ __forceinline__ void smem_tiles_ca(const uint32_t* __restrict__ T, int NTILES, int KT, int ntb0,
                                              const half* __restrict__ X, int K, const int* rows_sh, int nmt,
                                              int warp, int lane, uint32_t* Bs, uint32_t* Ws,
                                              float (&acc)[MTW][NTB][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    static_assert(LW == 1, "smem_tiles_ca assumes one trellis word a lane (K2 <= 8)");
    static_assert(S >= 2, "smem_tiles_ca needs at least 2 stages");
    constexpr int KS = (W >= NTB) ? W / NTB : 1;          // k-tiles a step
    constexpr int TPW = (NTB > W) ? NTB / W : 1;          // tiles a warp handles a step
    constexpr int TILES = KS * NTB;                       // tiles a step
    constexpr int STAGE = TILES * 32 * 4;                 // decoded uint32 words a stage (Bs)
    constexpr int WSTG = TILES * 32;                      // raw uint32 words a stage (Ws)
    static_assert(W * TPW == TILES, "the warps must cover a step's tiles exactly");
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const int g = lane >> 2, t = lane & 3;
    const int NS = KT / KS;
    const uint32_t* tp[TPW];
    uint32_t* ws[TPW];                                    // this warp's lane slot per tile, stage 0
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int ti = warp * TPW + j;
        tp[j] = T + ((size_t)(ti / NTB) * NTILES + ntb0 + (ti % NTB)) * TW + lane;
        ws[j] = Ws + ti * 32 + lane;
    }
    const int m0 = warp * MTW;
    const half* xr0[MTW];
    const half* xr1[MTW];
    bool ok0[MTW], ok1[MTW];
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
        const int r0 = rows_sh[(m0 + m) * 16 + g], r1 = rows_sh[(m0 + m) * 16 + g + 8];
        ok0[m] = r0 >= 0;
        ok1[m] = r1 >= 0;
        xr0[m] = X + (size_t)(ok0[m] ? r0 : 0) * K + 2 * t;
        xr1[m] = X + (size_t)(ok1[m] ? r1 : 0) * K + 2 * t;
    }
    const int nmw = min(max(nmt - m0, 0), MTW);

    // The A loads stay at the head of each consume: prefetching step s + 1's A words one step
    // ahead in registers measured NO change at 4,096 rows (ca:4x1x8x4 41.2-42.7 ms both ways,
    // 2026-10-23) -- L2 and the step's own mma slack already hide them. Kept simple on purpose.
    auto consume = [&](int stage, int s) {
        const uint32_t* st = Bs + stage * STAGE;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const int k = (s * KS + kk) * 16;
#pragma unroll
            for (int m = 0; m < MTW; ++m) {
                if (m < nmw) {
                    uint32_t a[4] = {load_pair(xr0[m] + k, ok0[m]), load_pair(xr1[m] + k, ok1[m]),
                                     load_pair(xr0[m] + k + 8, ok0[m]), load_pair(xr1[m] + k + 8, ok1[m])};
#pragma unroll
                    for (int n = 0; n < NTB; ++n) {
                        const uint4 v = *reinterpret_cast<const uint4*>(st + ((kk * NTB + n) * 32 + lane) * 4);
                        const uint32_t b0[2] = {v.x, v.y}, b1[2] = {v.z, v.w};
                        mma16816(acc[m][n][0], a, b0);
                        mma16816(acc[m][n][1], a, b1);
                    }
                }
            }
        }
    };
    auto store = [&](int stage, const uint32_t (&b)[TPW][4]) {
        uint32_t* st = Bs + stage * STAGE;
#pragma unroll
        for (int j = 0; j < TPW; ++j)
            *reinterpret_cast<uint4*>(st + ((warp * TPW + j) * 32 + lane) * 4) = make_uint4(b[j][0], b[j][1], b[j][2], b[j][3]);
    };
    // One commit group per step, issued S - 1 steps ahead; lanes past the tile's TW words copy
    // nothing (decode masks them to 0 at read, as load_words does).
    auto issue = [&](int q) {
        if (q < NS && lane < TW)
#pragma unroll
            for (int j = 0; j < TPW; ++j) cp_async4(ws[j] + (q % S) * WSTG, tp[j] + (size_t)q * KS * kstride);
        cp_commit();
    };
    auto decode = [&](int q, uint32_t (&b)[TPW][4]) {
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            uint32_t w[1];
            w[0] = (lane < TW) ? ws[j][(q % S) * WSTG] : 0u;
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, b0, b1);
            b[j][0] = b0[0]; b[j][1] = b0[1]; b[j][2] = b1[0]; b[j][3] = b1[1];
        }
    };

    for (int d = 0; d < S - 1; ++d) issue(d);
    {
        cp_wait<S - 2>();
        uint32_t b[TPW][4];
        decode(0, b);
        issue(S - 1);
        store(0, b);
    }
    __syncthreads();
    for (int q = 1; q < NS; ++q) {
        cp_wait<S - 2>();
        uint32_t b[TPW][4];
        decode(q, b);
        issue(q + S - 1);
        consume((q - 1) & 1, q - 1);
        store(q & 1, b);
        __syncthreads();
    }
    consume((NS - 1) & 1, NS - 1);
}

// The cp.async variant's raw-word staging buffer, allocated only when its branch is live so the
// shipped MODE 0 kernel's shared-memory footprint is untouched.
template <int MODE, int WORDS>
__device__ __forceinline__ uint32_t* ws_buf() {
    if constexpr (MODE <= -2) {
        __shared__ __align__(16) uint32_t Ws[WORDS];
        return Ws;
    } else {
        return nullptr;
    }
}

// Program (n block of NTB tiles, routing block u, matrix): BM = W * MTW * 16 member rows of one arena
// slot times NTB * 16 columns. Members as prefill_kernel's (build_routing's block, -1 padded).
// MODE 0: smem_tiles, the shipped kernel. MODE 1/2: its probes (exl3m_prefill_smem_probe).
// MODE -S (S >= 2): smem_tiles_ca with S cp.async stages (exl3m_prefill_smem_ca).
template <int CB, int W, int MTW, int NTB, int PF, int LO, int HI, int MODE = 0>
__global__ void __launch_bounds__(W * 32) prefill_smem_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int slots) {
    constexpr int BM = W * MTW * 16;
    constexpr int KS = (W >= NTB) ? W / NTB : 1;
    constexpr int STAGE = KS * NTB * 32 * 4;
    __shared__ __align__(16) uint32_t Bs[2 * STAGE];
    uint32_t* Ws = ws_buf<MODE, (MODE <= -2 ? -MODE : 1) * KS * NTB * 32>();   // raw words, MODE -S only
    __shared__ int rows_sh[BM];
    const int u = blockIdx.y;
    if (u >= ucount[0]) return;
    const int e = uids[u];
    if (e < 0) return;
    const int mat = blockIdx.z;
    for (int i = threadIdx.x; i < BM; i += W * 32) {
        const int code = members[(size_t)u * BM + i];
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    int nmt = 0;
#pragma unroll
    for (int m = 0; m < W * MTW; ++m)
        if (rows_sh[m * 16] >= 0) nmt = m + 1;
    if (nmt == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = N >> 4, KT = K >> 4;
    const int ntb0 = blockIdx.x * NTB;
    if (ntb0 >= NTILES) return;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];

    float acc[MTW][NTB][2][4];
#pragma unroll
    for (int m = 0; m < MTW; ++m)
#pragma unroll
        for (int i = 0; i < NTB; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    switch (k2) {
#define EXL3_SCASE(K2_)                                                                                         \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI && MODE <= -2)                                                     \
            smem_tiles_ca<CB, K2_, W, MTW, NTB, (MODE <= -2 ? -MODE : 2)>(T, NTILES, KT, ntb0, X, K, rows_sh,   \
                                                                          nmt, warp, lane, Bs, Ws, acc);        \
        else if constexpr (K2_ >= LO && K2_ <= HI)                                                              \
            smem_tiles<CB, K2_, W, MTW, NTB, PF, (MODE >= 0 ? MODE : 0)>(T, NTILES, KT, ntb0, X, K, rows_sh,    \
                                                                         nmt, warp, lane, Bs, acc);             \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        EXL3_SCASE(2) EXL3_SCASE(3) EXL3_SCASE(4) EXL3_SCASE(5) EXL3_SCASE(6) EXL3_SCASE(7) EXL3_SCASE(8)
        EXL3_SCASE(9) EXL3_SCASE(10)
#undef EXL3_SCASE
        default:
            __trap();
    }

    float* Zm = Z + (size_t)mat * P * N;
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
        const int mt = warp * MTW + m;
        if (mt >= nmt) break;
        const int r0 = rows_sh[mt * 16 + g], r1 = rows_sh[mt * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NTB; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (ntb0 + i) * 16 + h * 8 + 2 * t;
                if (r0 >= 0)
                    *reinterpret_cast<float2*>(Zm + (size_t)r0 * N + col) = make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (r1 >= 0)
                    *reinterpret_cast<float2*>(Zm + (size_t)r1 * N + col) = make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}

template <int CB>
void prefill_smem_launch(const void* x0, const void* x1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                         const int* k2_1, const int* uids, const int* ucount, const int* members, float* z, int K,
                         int N, int P, int slots, int nblocks, int mats, int warps, int mtw, int ntb, int pf, int lo,
                         int hi, cudaStream_t stream) {
    EXL3_CHECK(lo >= 4 && hi <= 6, "prefill kernel is built for 2- and 3-bit trellis (K2 4..6)");
    const int ntiles = N / 16;
    EXL3_CHECK(ntiles % ntb == 0, "N must be a multiple of 16 * NTB");
#define EXL3_SLAUNCH(W_, MTW_, NTB_, PF_)                                                                       \
    {                                                                                                           \
        constexpr int KS_ = (W_ >= NTB_) ? W_ / NTB_ : 1;                                                       \
        EXL3_CHECK((K / 16) % KS_ == 0, "K must be a multiple of 16 * KS");                                     \
        dim3 grid((unsigned)(ntiles / NTB_), (unsigned)nblocks, (unsigned)mats);                                \
        prefill_smem_kernel<CB, W_, MTW_, NTB_, PF_, 4, 6><<<grid, W_ * 32, 0, stream>>>(                       \
            (const half*)x0, (const half*)x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, slots);  \
    }
    if (warps == 8 && mtw == 1 && ntb == 8 && pf == 2) EXL3_SLAUNCH(8, 1, 8, 2)
    else if (warps == 8 && mtw == 1 && ntb == 8 && pf == 3) EXL3_SLAUNCH(8, 1, 8, 3)
    else if (warps == 8 && mtw == 2 && ntb == 8 && pf == 2) EXL3_SLAUNCH(8, 2, 8, 2)
    else if (warps == 4 && mtw == 2 && ntb == 8 && pf == 2) EXL3_SLAUNCH(4, 2, 8, 2)
    else if (warps == 4 && mtw == 1 && ntb == 8 && pf == 2) EXL3_SLAUNCH(4, 1, 8, 2)
    else if (warps == 8 && mtw == 1 && ntb == 4 && pf == 2) EXL3_SLAUNCH(8, 1, 4, 2)
    else if (warps == 8 && mtw == 2 && ntb == 4 && pf == 2) EXL3_SLAUNCH(8, 2, 4, 2)
    else if (warps == 16 && mtw == 1 && ntb == 8 && pf == 2) EXL3_SLAUNCH(16, 1, 8, 2)
    else if (warps == 16 && mtw == 1 && ntb == 4 && pf == 2) EXL3_SLAUNCH(16, 1, 4, 2)
    else if (warps == 4 && mtw == 1 && ntb == 16 && pf == 2) EXL3_SLAUNCH(4, 1, 16, 2)
    else if (warps == 8 && mtw == 1 && ntb == 16 && pf == 2) EXL3_SLAUNCH(8, 1, 16, 2)
    else EXL3_CHECK(false, "unsupported smem prefill tile setting");
#undef EXL3_SLAUNCH
}

// smem prefill with MODE != 0: the probes (mode 1/2) and the cp.async stage depths (mode -S).
// Instantiated for the serving-shaped tiles only, to bound compile time.
template <int CB>
void prefill_smem_mode_launch(const void* x0, const void* x1, const int64_t* tp0, const int64_t* tp1,
                              const int* k2_0, const int* k2_1, const int* uids, const int* ucount,
                              const int* members, float* z, int K, int N, int P, int slots, int nblocks, int mats,
                              int warps, int mtw, int ntb, int mode, cudaStream_t stream) {
    const int ntiles = N / 16;
    EXL3_CHECK(ntiles % ntb == 0, "N must be a multiple of 16 * NTB");
#define EXL3_MLAUNCH(W_, MTW_, NTB_, MODE_)                                                                     \
    {                                                                                                           \
        constexpr int KS_ = (W_ >= NTB_) ? W_ / NTB_ : 1;                                                       \
        EXL3_CHECK((K / 16) % KS_ == 0, "K must be a multiple of 16 * KS");                                     \
        dim3 grid((unsigned)(ntiles / NTB_), (unsigned)nblocks, (unsigned)mats);                                \
        prefill_smem_kernel<CB, W_, MTW_, NTB_, 2, 4, 6, MODE_><<<grid, W_ * 32, 0, stream>>>(                  \
            (const half*)x0, (const half*)x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, slots);  \
    }
#define EXL3_MODES(W_, MTW_, NTB_)                                                                              \
    if (mode == 1) EXL3_MLAUNCH(W_, MTW_, NTB_, 1)                                                              \
    else if (mode == 2) EXL3_MLAUNCH(W_, MTW_, NTB_, 2)                                                         \
    else if (mode == -3) EXL3_MLAUNCH(W_, MTW_, NTB_, -3)                                                       \
    else if (mode == -4) EXL3_MLAUNCH(W_, MTW_, NTB_, -4)                                                       \
    else if (mode == -6) EXL3_MLAUNCH(W_, MTW_, NTB_, -6)                                                       \
    else if (mode == -8) EXL3_MLAUNCH(W_, MTW_, NTB_, -8)                                                       \
    else if (mode == -12) EXL3_MLAUNCH(W_, MTW_, NTB_, -12)                                                     \
    else EXL3_CHECK(false, "smem prefill mode must be 1, 2 (probes) or -3/-4/-6/-8/-12 (cp.async stages)");
    if (warps == 4 && mtw == 1 && ntb == 8) { EXL3_MODES(4, 1, 8) }
    else if (warps == 8 && mtw == 1 && ntb == 8) { EXL3_MODES(8, 1, 8) }
    else EXL3_CHECK(false, "smem prefill modes are built for the 4x1x8 and 8x1x8 tiles");
#undef EXL3_MODES
#undef EXL3_MLAUNCH
}

// smem prefill for an arena of mixed widths (a version-2 pack): codebook CB, trellis widths of 2 to
// HI / 2 bits per slot (k2_0 / k2_1), the serving tile 4x1x8 with 4 cp.async stages. A separate
// instantiation, so prefill_smem_mode_launch's mul1 2-3-bit kernel keeps its registers and bits.
template <int CB, int HI>
void prefill_smem_cb_launch(const void* x0, const void* x1, const int64_t* tp0, const int64_t* tp1,
                            const int* k2_0, const int* k2_1, const int* uids, const int* ucount, const int* members,
                            float* z, int K, int N, int P, int slots, int nblocks, int mats, cudaStream_t stream) {
    constexpr int W = 4, MTW = 1, NTB = 8, KS = W / NTB > 0 ? W / NTB : 1;
    EXL3_CHECK((N / 16) % NTB == 0, "N must be a multiple of 16 * NTB");
    EXL3_CHECK((K / 16) % KS == 0, "K must be a multiple of 16 * KS");
    dim3 grid((unsigned)((N / 16) / NTB), (unsigned)nblocks, (unsigned)mats);
    prefill_smem_kernel<CB, W, MTW, NTB, 2, 4, HI, -4><<<grid, W * 32, 0, stream>>>(
        (const half*)x0, (const half*)x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, slots);
}

// ---------------------------------------------------------------- rotations / epilogues (TensorFold experts.cu)

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up.
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (act_mode 1 =
// fp32, 0 = TensorFold's bf16 rounding points), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, int suh_d_stride, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * suh_d_stride + n + j]);   // rank slice: caller offsets
                                                                              // suh_d and passes the full K
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Dense matrices (engine/exl3_dense.py): one program per (row, 128-block of K). Row r belongs to
// group r % G (wo_a's head groups, stacked row-major as [T, G, K]); G = 1 for an ordinary matrix.
template <typename TIN>
__global__ void dense_rot_kernel(const TIN* __restrict__ x, int x_stride, const half* __restrict__ suh,
                                 half* __restrict__ out, int K, int G) {
    const int r = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    const half* s = suh + (size_t)(r % G) * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)r * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(s[j]);
    fwht128(v, lane);
    half* o = out + (size_t)r * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Splits summed in order, rotated, * svh of the row's group; bf16 (out_bf16) or fp32 out.
__global__ void dense_out_kernel(const float* __restrict__ Z, const half* __restrict__ svh, void* __restrict__ out,
                                 int rows, int N, int SK, int G, int out_bf16) {
    const int r = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int q = 0; q < SK; ++q) s += Z[((size_t)q * rows + r) * N + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    const half* sv = svh + (size_t)(r % G) * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float y = v[j] * HAD_SCALE * __half2float(sv[j]);
        if (out_bf16)
            reinterpret_cast<__nv_bfloat16*>(out)[(size_t)r * N + n + j] = __float2bfloat16_rn(y);
        else
            reinterpret_cast<float*>(out)[(size_t)r * N + n + j] = y;
    }
}

// down_epilogue then combine in one launch: rotate * svh_d, then sum over the top-k slots in order.
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
    } else {
        // A negative pick is in no group (group_small_kernel), so Z holds nothing of it: its term
        // is exactly 0. DSV41_EXPERT_TOPP's pruned picks take this branch.
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = 0.f;
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}

// Grouping in one block: distinct arena slots (< E) in id order, members = row * 32 + k, -1 after the last.
// Only used for small E; the decode path uses group_small_kernel below.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// Decode grouping: distinct arena slots in first-occurrence order, O(P^2) and FIXED SHAPE, so it
// is graph-capturable.  The arena can have >10k slots but a decode call routes at most 64 picks,
// so scanning the picks (not every slot) is both cheap and independent of the arena size.
//
// One thread per pick (two warps): pick i is a group leader if no earlier pick names its slot; its
// group index is the number of leaders before it (two ballots), and it writes its own uid and
// members. A negative pick (DSV41_EXPERT_TOPP's pruned slot) leads no group and joins none, so
// nothing of it is read.  Same output as the serial scan it replaced -- leaders in first-occurrence order,
// members ascending -- which took 38-48 us a layer on the decode critical path (thread 0 alone,
// a dependent global load per comparison); this takes a few.
__global__ void __launch_bounds__(64) group_small_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                         int* __restrict__ ucount, int* __restrict__ members,
                                                         int P, int slots, int maxm) {
    __shared__ int sp[64];
    __shared__ unsigned lead0;
    const int i = threadIdx.x, lane = i & 31;
    if (i < P) sp[i] = pick[i];
    __syncthreads();
    bool lead = false;
    int e = 0;
    if (i < P) {
        e = sp[i];
        lead = e >= 0;
        for (int j = 0; j < i && lead; ++j)
            if (sp[j] == e) lead = false;
    }
    const unsigned b = __ballot_sync(0xffffffffu, lead);
    if (i == 0) lead0 = b;
    __syncthreads();
    const int n = __popc(b & ((1u << lane) - 1u)) + (i >= 32 ? __popc(lead0) : 0);
    if (lead) {
        uids[n] = e;
        int m = 0;
        for (int j = i; j < P && m < maxm; ++j)
            if (sp[j] == e) members[n * maxm + m++] = (j / slots) * 32 + (j % slots);
        for (; m < maxm; ++m) members[n * maxm + m] = -1;
    }
    if (i == 63) ucount[0] = __popc(lead0) + __popc(b);
}

}  // namespace exl3

// ---------------------------------------------------------------- C ABI (ctypes; no torch/libtorch)

extern "C" {

void exl3m_group(const int* pick, int* uids, int* ucount, int* members, int R, int slots, int E, int maxm,
                 cudaStream_t s) {
    // decode-sized: group the picks, not the arena.  E is unused here (kept for ABI stability).
    (void)E;
    const int P = R * slots;
    EXL3_CHECK(P <= 64, "decode grouping takes at most 64 picks");
    exl3::group_small_kernel<<<1, 64, 0, s>>>(pick, uids, ucount, members, P, slots, maxm);
}

void exl3m_rot_in(const void* x, int x_stride, const int* pick, const void* suh0, const void* suh1, void* out0,
                  void* out1, int rows, int K, int slots, int E, int x_bf16, cudaStream_t s) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    if (x_bf16)
        exl3::rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, s>>>((const __nv_bfloat16*)x, x_stride, pick,
                                                               (const half*)suh0, (const half*)suh1, (half*)out0,
                                                               (half*)out1, K, slots, E);
    else
        exl3::rot_in_kernel<half><<<grid, 32, 0, s>>>((const half*)x, x_stride, pick, (const half*)suh0,
                                                      (const half*)suh1, (half*)out0, (half*)out1, K, slots, E);
}

void exl3m_grouped(const void* X0, const void* X1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                   const int* k2_1, const int* uids, const int* ucount, const int* members, float* Z, int K, int N,
                   int P, int SK, int maxm, int slots, int nexp_max, int mats, int nt, int warps, int pf, int lo,
                   int hi, int cb, cudaStream_t s) {
    if (cb == 2) exl3::grouped_launch<2>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, SK,
                                         maxm, slots, nexp_max, mats, nt, warps, pf, lo, hi, s);
    else if (cb == 1) exl3::grouped_launch<1>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, SK,
                                              maxm, slots, nexp_max, mats, nt, warps, pf, lo, hi, s);
    else exl3::grouped_launch<0>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots,
                                 nexp_max, mats, nt, warps, pf, lo, hi, s);
}

void exl3m_prefill(const void* X0, const void* X1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                   const int* k2_1, const int* uids, const int* ucount, const int* members, float* Z, int K, int N,
                   int P, int slots, int nblocks, int mats, int bm, int nt, int warps, int pf, int lo, int hi,
                   cudaStream_t s) {
    exl3::prefill_launch<2>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, slots, nblocks, mats,
                            bm, nt, warps, pf, lo, hi, s);
}

void exl3m_prefill_smem(const void* X0, const void* X1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                        const int* k2_1, const int* uids, const int* ucount, const int* members, float* Z, int K,
                        int N, int P, int slots, int nblocks, int mats, int warps, int mtw, int ntb, int pf, int lo,
                        int hi, cudaStream_t s) {
    exl3::prefill_smem_launch<2>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, slots, nblocks,
                                 mats, warps, mtw, ntb, pf, lo, hi, s);
}

// mode 1/2: the probes (bench-only; output is not the GEMM's). mode -S: cp.async with S raw stages,
// bit-identical to exl3m_prefill_smem at the same W/MTW/NTB (the decode order and mma chain are the
// same; only how the raw trellis words reach the SM changes).
void exl3m_prefill_smem_mode(const void* X0, const void* X1, const int64_t* tp0, const int64_t* tp1,
                             const int* k2_0, const int* k2_1, const int* uids, const int* ucount,
                             const int* members, float* Z, int K, int N, int P, int slots, int nblocks, int mats,
                             int warps, int mtw, int ntb, int mode, cudaStream_t s) {
    exl3::prefill_smem_mode_launch<2>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, slots,
                                      nblocks, mats, warps, mtw, ntb, mode, s);
}

// exl3m_prefill_smem_mode's 4x1x8 tile at cp.async depth 4 for a mixed-width arena: codebook cb (2 mul1,
// 1 mcg), widths up to hi (K2 = 2 x bits: 6 or 8).
void exl3m_prefill_smem_cb(const void* X0, const void* X1, const int64_t* tp0, const int64_t* tp1, const int* k2_0,
                           const int* k2_1, const int* uids, const int* ucount, const int* members, float* Z, int K,
                           int N, int P, int slots, int nblocks, int mats, int cb, int hi, cudaStream_t s) {
    if (cb == 1 && hi <= 8)
        exl3::prefill_smem_cb_launch<1, 8>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, slots,
                                           nblocks, mats, s);
    else if (cb == 2 && hi <= 8)
        exl3::prefill_smem_cb_launch<2, 8>(X0, X1, tp0, tp1, k2_0, k2_1, uids, ucount, members, Z, K, N, P, slots,
                                           nblocks, mats, s);
    else EXL3_CHECK(false, "smem prefill (mixed widths): codebook mul1 or mcg, widths up to 4 bits");
}

void exl3m_gateup(const float* Z, const int* pick, const void* svh_g, const void* svh_u, const void* suh_d,
                  void* xd, int rows, int P, int N, int SK, int slots, int E, int suh_d_stride, float limit,
                  int act_mode, cudaStream_t s) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    exl3::gateup_epilogue_kernel<<<grid, 32, 0, s>>>(Z, pick, (const half*)svh_g, (const half*)svh_u,
                                                     (const half*)suh_d, (half*)xd, P, N, SK, E, suh_d_stride,
                                                     limit, act_mode);
}

void exl3m_dense_rot(const void* x, int x_stride, const void* suh, void* out, int rows, int K, int G, int x_bf16,
                     cudaStream_t s) {
    dim3 grid((unsigned)rows, (unsigned)(K / 128));
    if (x_bf16)
        exl3::dense_rot_kernel<__nv_bfloat16><<<grid, 32, 0, s>>>((const __nv_bfloat16*)x, x_stride,
                                                                  (const half*)suh, (half*)out, K, G);
    else
        exl3::dense_rot_kernel<half><<<grid, 32, 0, s>>>((const half*)x, x_stride, (const half*)suh, (half*)out,
                                                         K, G);
}

void exl3m_dense_out(const float* Z, const void* svh, void* out, int rows, int N, int SK, int G, int out_bf16,
                     cudaStream_t s) {
    dim3 grid((unsigned)rows, (unsigned)(N / 128));
    exl3::dense_out_kernel<<<grid, 32, 0, s>>>(Z, (const half*)svh, out, rows, N, SK, G, out_bf16);
}

void exl3m_down_combine(const float* Z, const int* pick, const void* svh_d, const float* wts, float* out,
                        int rows, int P, int D, int SK, int slots, int E, cudaStream_t s) {
    EXL3_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    exl3::down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, s>>>(Z, pick, (const half*)svh_d, wts, out, P, D,
                                                                      SK, E, slots);
}

}  // extern "C"
