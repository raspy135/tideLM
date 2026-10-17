"""GLM attention projections from the EXL3 checkpoint's 5-bit matrices (MUSE_EXL3_ATTN=1).

The Quark checkpoint's attention is FP8 with 128x128 block scales; bertholomus/GLM-5.3-EXL3-3.0bpw
stores q_a, q_b, kv_a, o_proj and the indexer's wq_b at 5 bits (EXL3 trellis, mul1 codebook,
128-wide Hadamard rotations on both sides): ~37% fewer bytes a decode step reads, and ~2.4 GB a
rank less to keep resident.

Decode (<= 16 rows) runs V4.1's dense path (engine/exl3_dense.Exl3Matrix: input rotation, the
grouped split-K GEMV, combine + output rotation). Prefill unpacks the trellis to its FP16 matrix
once a call (Triton, exact: the codebook is a table of the reference's values) and runs the
rotations on the activations around a cuBLAS GEMM:  y = H_N( (H_K(x * suh))_fp16 @ W_q ) * svh.
"""

from __future__ import annotations

import functools
import os
import sys

import torch
import triton
import triton.language as tl

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOOLS))


@triton.jit
def _unpack_kernel(TR, W, NT, N, SCALE, BIAS, W16: tl.constexpr, BITS: tl.constexpr, BC: tl.constexpr):
    """Rows of K-tile kt, columns [c0, c0 + BC) of W [K, N] fp16, computed in output order so the
    stores are contiguous: each element finds its stream position p (the inverse of the tile's
    lane/j layout), its 16-bit state in the tile's bitstream, and the mul1 codebook value --
    1024 + the four bytes of state * 0x83DCD12D, times scale plus bias: exact in fp32 here (a
    multiple of 2^-18 below 16 needs 22 bits), so one rounding to fp16, as the reference does."""
    kt = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, 16)
    nt = c // 16
    col = c % 16
    j = (r[:, None] & 1) | (((r[:, None] >> 3) & 1) << 1) | ((col[None, :] >> 3) << 2)
    lane = ((col[None, :] & 7) << 2) | ((r[:, None] >> 1) & 3)
    p = lane * 8 + j
    nw: tl.constexpr = W16 // 2
    # The int16 pairs are little-endian 32-bit words: read them as such.
    base = TR.to(tl.pointer_type(tl.uint32), bitcast=True) + (kt.to(tl.int64) * NT + nt[None, :]) * nw
    first = (p + 1) * BITS - 16 + 32 * nw
    i0 = (first // 32) % nw
    off = (first % 32).to(tl.uint32)
    i1 = (i0 + 1) % nw
    w0 = tl.load(base + i0)
    w1 = tl.load(base + i1)
    # The 16 bits ending `off` bits into word i1's predecessor pair (w0:w1), with 32-bit shifts.
    hi = tl.where(off >= 16, w0 << (off - 16), w0 >> (16 - off))
    lo = tl.where(off > 16, w1 >> (48 - off), 0)
    state = (hi | lo) & 0xFFFF
    x = state * 0x83DCD12D                                   # uint32: wraps mod 2^32
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    v = h.to(tl.float32) * SCALE + BIAS
    tl.store(W + (kt.to(tl.int64) * 16 + r[:, None]) * N + c[None, :], v.to(tl.float16))


@functools.lru_cache(maxsize=1)
def _mul1_consts():
    import exl3_ref as R
    f = lambda b: float(torch.tensor([b], dtype=torch.int32).to(torch.uint16).view(torch.float16))
    return f(R.MUL1_SCALE), f(R.MUL1_BIAS)


def unpack(trellis: torch.Tensor, bits: int) -> torch.Tensor:
    """int16 [K/16, N/16, 16*bits] -> W_q fp16 [K, N] (the rotated-domain matrix, exl3_ref.unpack)."""
    KT, NT, W16 = trellis.shape
    assert W16 == 16 * bits and float(bits).is_integer(), (trellis.shape, bits)
    N = NT * 16
    w = torch.empty((KT * 16, N), dtype=torch.float16, device=trellis.device)
    BC = 128 if N % 128 == 0 else 64 if N % 64 == 0 else 16
    scale, bias = _mul1_consts()
    _unpack_kernel[(KT, N // BC)](trellis, w, NT, N, scale, bias, W16=W16, BITS=int(bits), BC=BC, num_warps=4)
    return w


@functools.lru_cache(maxsize=2)
def _hadamard(device: str) -> torch.Tensor:
    import exl3_ref as R
    return (R.hadamard(128, device).float() / 128 ** 0.5).contiguous()


def rotate(x: torch.Tensor) -> torch.Tensor:
    """H/sqrt(128) over each 128-block of the last dim, fp32."""
    T, n = x.shape
    return (x.float().view(T * (n // 128), 128) @ _hadamard(str(x.device))).view(T, n)
