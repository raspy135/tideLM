"""FP8 E4M3 weights with one FP32 scale per 128x128 block (DeepSeek-V3's format, as AMD's
GLM-5.3 Quark release stores attention): y = x @ (w * s)^T without materializing BF16 weights.

Each program owns a BLOCK_N x BLOCK_M output tile (BLOCK_N divides 128, so the tile's rows share
one scale row) and walks K in 128-wide steps: the FP8 tile is widened to BF16 (exact), one
tensor-core dot gives the block's FP32 partial sum, and the block's scale multiplies that partial
sum -- the same arithmetic as dequantizing first, in a better order (one rounding of the scale
product per block instead of per weight). engine/glm/test_fp8_block.py checks it against a
dequantized matmul.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SB = 128


@triton.jit
def _fp8_block_kernel(X, W, S, Y, M, N, K, stride_xm, stride_wn, stride_sn, stride_ym,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    m_mask = rm < M
    n_mask = rn < N
    srow = (pid_n * BLOCK_N) // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, 128):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        part = tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        s = tl.load(S + srow * stride_sn + k0 // 128)
        acc += part * s
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(Y.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class FP8Block:
    """FP8 weight [N, K] + FP32 scale [ceil(N/128), ceil(K/128)]."""

    def __init__(self, w: torch.Tensor, s: torch.Tensor):
        assert w.dtype == torch.float8_e4m3fn and w.dim() == 2
        self.w = w.contiguous()
        self.s = s.float().contiguous()
        self.N, self.K = self.w.shape
        assert self.s.shape == ((self.N + SB - 1) // SB, (self.K + SB - 1) // SB), (self.s.shape, self.w.shape)
        assert self.K % SB == 0, "K must be a multiple of the 128-wide scale block"

    def rows(self, start: int, end: int) -> "FP8Block":
        assert start % SB == 0 and (end % SB == 0 or end == self.N)
        return FP8Block(self.w[start:end], self.s[start // SB:(end + SB - 1) // SB])

    def cols(self, start: int, end: int) -> "FP8Block":
        assert start % SB == 0 and end % SB == 0
        return FP8Block(self.w[:, start:end], self.s[:, start // SB:end // SB])

    def to(self, device) -> "FP8Block":
        return FP8Block(self.w.to(device), self.s.to(device))

    def dequant(self, dtype=torch.bfloat16) -> torch.Tensor:
        s = self.s.repeat_interleave(SB, 0)[:self.N].repeat_interleave(SB, 1)[:, :self.K]
        return (self.w.float() * s).to(dtype)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4


def fp8_block_linear(x: torch.Tensor, W: FP8Block, out_dtype=torch.bfloat16) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, W.K).to(torch.bfloat16).contiguous()
    M = x2.shape[0]
    y = torch.empty((M, W.N), dtype=out_dtype, device=x.device)
    stages = 3
    if M <= 16:
        BM, BN, warps = 16, 32, 4
    elif M <= 64:
        BM, BN, warps = 64, 64, 4
    elif W.K >= 8192:
        # o_proj at prefill sizes: 26.5 -> 51 TFLOPS at 1,024 rows on GB10 (3.89 -> 2.02 ms).
        BM, BN, warps, stages = 128, 128, 8, 2
    else:
        # The other prefill projections: 50-62 -> 60-77 TFLOPS (sweep of BM/BN/warps/stages).
        BM, BN, warps, stages = 64, 128, 4, 2
    grid = (triton.cdiv(W.N, BN), triton.cdiv(M, BM))
    _fp8_block_kernel[grid](x2, W.w, W.s, y, M, W.N, W.K, x2.stride(0), W.w.stride(0),
                            W.s.stride(0), y.stride(0), BLOCK_M=BM, BLOCK_N=BN,
                            num_warps=warps, num_stages=stages)
    return y.view(*shape[:-1], W.N)
