"""Prefill's fixed MXFP4 experts (the shared expert, a dense layer's pseudo-experts) as BF16 GEMMs.

In a prefill chunk every row picks the same fixed slots, so they are dense matrices with thousands
of rows -- but the MoE kernel (tools/fp4_moe.py) splits its rows into 64-row blocks and decodes
the FP4 weights again for each: the shared expert took ~3.9 ms a layer at 2,048 rows (~20 TFLOPS).
Here the slots' weights are dequantized to BF16 once a call (E2M1 x 2^(e-127) is exact in BF16)
and multiplied with cuBLAS, fp32 accumulation; the SwiGLU keeps the kernel's rounding points (gate
and up rounded to BF16 first, the product stored as BF16) and the down projection's output stays
fp32 like the kernel's parts. What changes is the order of the K sums.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_kernel(P, S, Y, N, K2, stride_p, stride_s, stride_y, BN: tl.constexpr, BK2: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.program_id(1) * BK2 + tl.arange(0, BK2)          # packed byte columns (two values each)
    m = (rn[:, None] < N) & (rk[None, :] < K2)
    b = tl.load(P + rn[:, None] * stride_p + rk[None, :], mask=m, other=0).to(tl.int32)
    e = tl.load(S + rn[:, None] * stride_s + rk[None, :] // 16, mask=m, other=127).to(tl.float32)
    sc = tl.exp2(e - 127.0)
    lo, hi = b & 15, b >> 4
    mlo, mhi = lo & 7, hi & 7
    vlo = tl.where(mlo < 4, mlo * 0.5, tl.where(mlo < 6, mlo - 2.0, 2.0 * mlo - 8.0))
    vhi = tl.where(mhi < 4, mhi * 0.5, tl.where(mhi < 6, mhi - 2.0, 2.0 * mhi - 8.0))
    vlo = tl.where(lo >= 8, -vlo, vlo) * sc
    vhi = tl.where(hi >= 8, -vhi, vhi) * sc
    out = tl.join(vlo, vhi).reshape(BN, 2 * BK2)             # low nibble = even element
    rk2 = tl.program_id(1) * 2 * BK2 + tl.arange(0, 2 * BK2)
    tl.store(Y + rn[:, None] * stride_y + rk2[None, :], out.to(tl.bfloat16),
             mask=(rn[:, None] < N) & (rk2[None, :] < 2 * K2))


def dequant(packed: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """U8 [N, K/2] + E8M0 U8 [N, K/32] -> BF16 [N, K] (engine/muse/weights.dequant_mxfp4's values)."""
    N, K2 = packed.shape
    y = torch.empty((N, 2 * K2), dtype=torch.bfloat16, device=packed.device) if out is None else out
    BN, BK2 = 32, 128
    _dequant_kernel[(triton.cdiv(N, BN), triton.cdiv(K2, BK2))](
        packed, scale, y, N, K2, packed.stride(0), scale.stride(0), y.stride(0), BN=BN, BK2=BK2, num_warps=4)
    return y


def fixed_up(x: torch.Tensor, arena, slots: list[int]) -> torch.Tensor:
    """x bf16 [T, H] through the up projections of `slots` -> h bf16 [T, k * Il] (slot-major)."""
    k, Il, H = len(slots), arena.Il, x.shape[1]
    w13 = torch.empty((2 * k * Il, H), dtype=torch.bfloat16, device=x.device)    # gate rows, then up rows
    for j, s in enumerate(slots):
        dequant(arena.w1[s], arena.s1[s], w13[j * Il:(j + 1) * Il])
        dequant(arena.w3[s], arena.s3[s], w13[(k + j) * Il:(k + j + 1) * Il])
    gu = torch.nn.functional.linear(x, w13)              # cuBLAS rounds its fp32 sums to BF16, like
    g, u = gu[:, :k * Il].float(), gu[:, k * Il:]         # the kernel's gate/up rounding
    return (torch.nn.functional.silu(g) * u.float()).to(torch.bfloat16)


def fixed_down(h_full: torch.Tensor, arena, slots: list[int], world: int) -> torch.Tensor:
    """The gathered h [T, world * k * Il] (rank-major, slot-major within a rank) through the slots'
    down projections -> parts fp32 [k, T, Hl], as the MoE kernel's down returns them."""
    T, k, Il = h_full.shape[0], len(slots), arena.Il
    g = h_full.view(T, world, k, Il)
    out = []
    for j, s in enumerate(slots):
        full = g[:, :, j, :].reshape(T, world * Il)
        w2 = dequant(arena.w2[s], arena.s2[s])
        out.append(torch.mm(full, w2.t(), out_dtype=torch.float32))
    return torch.stack(out)


@triton.jit
def _gemv_kernel(X, P, S, Y, T, N, K, stride_x, stride_p, stride_s, stride_y,
                 BN: tl.constexpr, BK: tl.constexpr, TP: tl.constexpr):
    """Y[T, N] (fp32) = X[T, K] @ W^T for an MXFP4 W [N, K]: weights decoded in registers, one
    pass over them (decode rows only; the rows pad to TP for the tensor-core dot)."""
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rt = tl.arange(0, TP)
    nm = rn < N
    acc = tl.zeros((TP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk2 = k0 // 2 + tl.arange(0, BK // 2)
        b = tl.load(P + rn[:, None] * stride_p + rk2[None, :], mask=nm[:, None], other=0).to(tl.int32)
        e = tl.load(S + rn[:, None] * stride_s + (k0 // 32 + tl.arange(0, BK // 32))[None, :], mask=nm[:, None],
                    other=127).to(tl.float32)
        sc = tl.exp2(e - 127.0)
        lo, hi = b & 15, b >> 4
        mlo, mhi = lo & 7, hi & 7
        vlo = tl.where(mlo < 4, mlo * 0.5, tl.where(mlo < 6, mlo - 2.0, 2.0 * mlo - 8.0))
        vhi = tl.where(mhi < 4, mhi * 0.5, tl.where(mhi < 6, mhi - 2.0, 2.0 * mhi - 8.0))
        vlo = tl.where(lo >= 8, -vlo, vlo)
        vhi = tl.where(hi >= 8, -vhi, vhi)
        w = tl.join(vlo, vhi).reshape(BN, BK // 32, 32) * sc[:, :, None]
        w = w.reshape(BN, BK).to(tl.bfloat16)
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rt[:, None] * stride_x + rk[None, :], mask=rt[:, None] < T, other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(Y + rt[:, None] * stride_y + rn[None, :], acc, mask=(rt[:, None] < T) & nm[None, :])


def gemv(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x bf16 [T <= 16, K] @ MXFP4 [N, K]^T -> fp32 [T, N]."""
    T, K = x.shape
    N = packed.shape[0]
    assert T <= 16
    x = x.to(torch.bfloat16).contiguous()
    y = torch.empty((T, N), dtype=torch.float32, device=x.device)
    BN, BK = 32, 256                      # sweep, L2-cold head: 1.14 ms (221 GB/s) vs 1.21 at 64 x 256
    _gemv_kernel[(triton.cdiv(N, BN),)](x, packed, scale, y, T, N, K, x.stride(0), packed.stride(0),
                                        scale.stride(0), y.stride(0), BN=BN, BK=BK, TP=16, num_warps=4,
                                        num_stages=4)
    return y


def quantize(w: torch.Tensor, rows: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] on the GPU -> MXFP4 (packed, scale), engine/muse/weights.quantize_mxfp4 in
    1,024-row pieces (its rounding search holds [rows, K, 8] fp32 at once)."""
    from engine.muse.weights import quantize_mxfp4
    parts = [quantize_mxfp4(w[i:i + rows]) for i in range(0, w.shape[0], rows)]
    return torch.cat([p for p, _ in parts]), torch.cat([s for _, s in parts])
