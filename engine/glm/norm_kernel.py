"""RMSNorm in one kernel (GLM_FUSED_NORM=1) for engine/glm/tp.py's rms().

rms() is weight * (x.float() * rsqrt(mean(x.float()^2) + eps)).to(bf16): seven eager kernels and,
at a 1,024-row prefill chunk of the 6,144-wide hidden, ~200 MB of traffic a call, four calls a
layer; in a decode graph ~2,000 tiny kernels a verify forward. One program a row does the same
arithmetic with the same rounding points (fp32 statistics, the normalized row rounded to BF16, then
the BF16 product with the weight); only the order of the mean's sum differs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    yb = (x * r).to(tl.bfloat16)
    w = tl.load(W + offs, mask=m, other=0.0)
    tl.store(Y + row * stride_y + offs, (w.to(tl.float32) * yb.to(tl.float32)).to(tl.bfloat16), mask=m)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x bf16 [T, N] (rows may be strided, last dim contiguous), weight bf16 [N] -> bf16 [T, N]."""
    T, N = x.shape
    y = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    if T:
        block = triton.next_power_of_2(N)
        _rms_kernel[(T,)](x, weight, y, N, x.stride(0), y.stride(0), eps, BLOCK=block,
                          num_warps=8 if block >= 4096 else 4)
    return y
