"""ref.rope_interleave in one kernel (decode's seven PyTorch kernels a call: four products, a
difference, a sum and the concatenation).

Bit-identical to the PyTorch version: each BF16 product is rounded to BF16 before the sum, as
PyTorch's BF16 multiply does (the fp32 product of two BF16 values is exact, so rounding it once is
the same result), and the sum or difference of the two rounded products is taken in fp32 and
rounded again. The intermediate rounding is integer arithmetic on the bits: written as
`.to(tl.bfloat16).to(tl.float32)` the compiler folded the pair away and kept the unrounded
products -- the multiplies matched PyTorch and the differences did not (one BF16 step in ~20% of
values).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16(v):
    """fp32 -> the nearest BF16 value (ties to even), still fp32."""
    b = v.to(tl.uint32, bitcast=True)
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)


@triton.jit
def _rope_kernel(X, s_xt, s_xh, COS, SIN, s_ct, s_st, Y, s_yt, s_yh, H,
                 HALF: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    j = tl.arange(0, HALF)
    m = (h < H)[:, None]
    base = X + t * s_xt + h[:, None] * s_xh
    x1 = tl.load(base + 2 * j[None, :], mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(base + 2 * j[None, :] + 1, mask=m, other=0.0).to(tl.float32)
    c = tl.load(COS + t * s_ct + j).to(tl.float32)[None, :]
    s = tl.load(SIN + t * s_st + j).to(tl.float32)[None, :]
    a = _bf16(x1 * c)
    b = _bf16(x2 * s)
    a2 = _bf16(x2 * c)
    b2 = _bf16(x1 * s)
    out = Y + t * s_yt + h[:, None] * s_yh + j[None, :]
    tl.store(out, (a - b).to(tl.bfloat16), mask=m)
    tl.store(out + HALF, (a2 + b2).to(tl.bfloat16), mask=m)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x bf16 [T, heads, d] (any strides, unit stride in d), cos/sin [T, >= d] -> contiguous bf16
    [T, heads, d], as ref.rope_interleave(x, cos, sin)."""
    T, H, d = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1 and d % 2 == 0
    y = torch.empty((T, H, d), dtype=torch.bfloat16, device=x.device)
    if T:
        _rope_kernel[(T,)](x, x.stride(0), x.stride(1), cos, sin, cos.stride(0), sin.stride(0),
                           y, y.stride(0), y.stride(1), H, HALF=d // 2, BH=triton.next_power_of_2(H),
                           num_warps=4)
    return y
