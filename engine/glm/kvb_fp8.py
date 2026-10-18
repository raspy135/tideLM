"""MLA's absorbed projections (kv_b_proj: w_uk, w_uv) kept in FP8 (MUSE_KVB_FP8=1).

The checkpoint stores kv_b_proj as FP8 with 128x128 block scales; the engine dequantized it to
BF16 at load: 1.16 GB a rank (0.58 in FP8), and decode reads all of it every round -- each layer's
query absorption (q_nope @ w_uk) and output (o_lat @ w_uv^T). Here the FP8 bytes stay, with each
row's scales expanded (a head's 448 rows straddle the 128-row blocks), and decode multiplies with
the weights widened in registers to the same BF16 values the load-time dequant made
(fp8 * scale in fp32, rounded once): only the accumulation order changes. Prefill (more than 16
rows) dequantizes the layer's weights per call and keeps the einsum.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SB = 128


class HeadFP8:
    """Per-head weights [H, R, C] in FP8 with fp32 scales [H, R, C / 128] (a block of 128 along C)."""

    def __init__(self, w8: torch.Tensor, s: torch.Tensor):
        assert w8.dtype == torch.float8_e4m3fn and w8.dim() == 3 and s.shape == (*w8.shape[:2], w8.shape[2] // SB)
        self.w, self.s = w8.contiguous(), s.float().contiguous()
        self.H, self.R, self.C = w8.shape

    @classmethod
    def from_rows(cls, w8: torch.Tensor, s: torch.Tensor, heads: int):
        """An FP8 block weight [heads * R, C] with [ceil(rows / 128), C / 128] scales -> per-head."""
        N, C = w8.shape
        rs = s.float().repeat_interleave(SB, 0)[:N]                    # each row's [C / 128] scales
        return cls(w8.view(heads, N // heads, C), rs.view(heads, N // heads, C // SB))

    def split(self, a: int, b: int) -> "HeadFP8":
        return HeadFP8(self.w[:, a:b], self.s[:, a:b])

    def to(self, device) -> "HeadFP8":
        return HeadFP8(self.w.to(device), self.s.to(device))

    def dequant(self, dtype=torch.bfloat16) -> torch.Tensor:
        return (self.w.float() * self.s.repeat_interleave(SB, 2)).to(dtype)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4


@triton.jit
def _head_kernel(X, W, S, Y, T, sxt, sxh, syt, syh, R: tl.constexpr, C: tl.constexpr,
                 K: tl.constexpr, ROWS_K: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y[t, h, n] = sum_k X[t, h, k] * Wh[k, n], Wh = W[h] (ROWS_K: rows are k) or W[h]^T (rows
    are n); W[h] is [R, C] FP8 with scales S[h] [R, C / 128]."""
    h = tl.program_id(0).to(tl.int64)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rt = tl.arange(0, BT)
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rt[:, None] * sxt + h * sxh + rk[None, :], mask=rt[:, None] < T, other=0.0)
        if ROWS_K:
            w = tl.load(W + h * R * C + rk[:, None] * C + rn[None, :]).to(tl.float32)
            s = tl.load(S + h * R * (C // 128) + rk[:, None] * (C // 128) + (rn // 128)[None, :])
        else:
            w = tl.load(W + h * R * C + rn[None, :] * C + rk[:, None]).to(tl.float32)
            s = tl.load(S + h * R * (C // 128) + rn[None, :] * (C // 128) + (rk // 128)[:, None])
        acc += tl.dot(x.to(tl.bfloat16), (w * s).to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(Y + rt[:, None] * syt + h * syh + rn[None, :], acc.to(tl.bfloat16), mask=rt[:, None] < T)


def absorb_q(q_nope: torch.Tensor, w_uk: HeadFP8) -> torch.Tensor:
    """einsum("thn,hnc->thc", q_nope, w_uk): [T, H, nope] -> bf16 [T, H, kv_lora]."""
    T, H, K = q_nope.shape
    if T > 16:
        return torch.einsum("thn,hnc->thc", q_nope, w_uk.dequant())
    x = q_nope.to(torch.bfloat16).contiguous()
    N = w_uk.C
    y = torch.empty((T, H, N), dtype=torch.bfloat16, device=x.device)
    BN = 128                    # sweep at 4 rows, L2-cold: 34.9 us (BF16 einsum 66.7; BN 64 BK 64 47.7)
    _head_kernel[(H, N // BN)](x, w_uk.w, w_uk.s, y, T, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                               R=w_uk.R, C=w_uk.C, K=K, ROWS_K=True, BT=16, BN=BN, BK=32, num_warps=4,
                               num_stages=2)
    return y


def absorb_o(o_lat: torch.Tensor, w_uv: HeadFP8) -> torch.Tensor:
    """einsum("thc,hvc->thv", o_lat, w_uv): [T, H, kv_lora] -> bf16 [T, H, v]."""
    T, H, K = o_lat.shape
    if T > 16:
        return torch.einsum("thc,hvc->thv", o_lat, w_uv.dequant())
    x = o_lat.to(torch.bfloat16).contiguous()
    N = w_uv.R
    y = torch.empty((T, H, N), dtype=torch.bfloat16, device=x.device)
    BN = 64                     # sweep at 4 rows, L2-cold: 54.0 us (BF16 einsum 81.3)
    _head_kernel[(H, N // BN)](x, w_uv.w, w_uv.s, y, T, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                               R=w_uv.R, C=w_uv.C, K=K, ROWS_K=False, BT=16, BN=BN, BK=128, num_warps=4,
                               num_stages=1)
    return y
