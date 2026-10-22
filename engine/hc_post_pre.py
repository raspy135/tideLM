"""Attention residual update and the following lagged pre/RMSNorm in one launch.

Keep both BF16 rounding boundaries and hc_ops' 1024-column norm reduction.
The new residual is also stored for the next mHC projection. Safe in-place:
each program owns every column of one token and loads all streams of a tile
before writing that tile. No cross-token dependencies or collectives.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _post_pre(X, RES, POST, COMB, PRE, W, OUT, SCRATCH, Y,
              D, HC: tl.constexpr, EPS, BD: tl.constexpr):
    t = tl.program_id(0)
    ssq = tl.zeros((), tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        valid = d < D
        x = tl.load(X + t * D + d, valid, other=0).to(tl.float32)
        # Load all streams before any in-place write. Keep the weighted result
        # in the same 1-D layout as hc_pre_rmsnorm: reducing a [HC, BD] layout
        # changed its sum-of-squares tree and flipped one BF16 output in a GPU test.
        r0 = tl.load(RES + (t * HC + 0) * D + d, valid, other=0).to(tl.float32)
        r1 = tl.load(RES + (t * HC + 1) * D + d, valid, other=0).to(tl.float32)
        r2 = tl.load(RES + (t * HC + 2) * D + d, valid, other=0).to(tl.float32)
        r3 = tl.load(RES + (t * HC + 3) * D + d, valid, other=0).to(tl.float32)
        weighted = tl.zeros((BD,), tl.float32)
        for j in tl.static_range(HC):
            acc = tl.zeros((BD,), tl.float32)
            acc += tl.load(COMB + (t * HC + 0) * HC + j) * r0
            acc += tl.load(COMB + (t * HC + 1) * HC + j) * r1
            acc += tl.load(COMB + (t * HC + 2) * HC + j) * r2
            acc += tl.load(COMB + (t * HC + 3) * HC + j) * r3
            acc += tl.load(POST + t * HC + j) * x
            rounded = acc.to(tl.bfloat16)
            tl.store(OUT + (t * HC + j) * D + d, rounded, valid)
            weighted += rounded.to(tl.float32) * tl.load(PRE + t * HC + j)
        weighted = weighted.to(tl.bfloat16).to(tl.float32)
        ssq += tl.sum(tl.where(valid, weighted * weighted, 0))
        tl.store(SCRATCH + t * D + d, weighted, valid)
    rs = 1.0 / tl.sqrt(ssq / D + EPS)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        valid = d < D
        y = tl.load(SCRATCH + t * D + d, valid, other=0) * rs
        y *= tl.load(W + d, valid, other=0).to(tl.float32)
        tl.store(Y + t * D + d, y, valid)


def post_pre(x, residual, post, comb, pre, weight, eps, *, out=None, normalized=None):
    t, hc, d = residual.shape
    if (hc != 4 or residual.dtype != torch.bfloat16 or x.dtype != torch.bfloat16
            or not residual.is_contiguous() or not x.is_contiguous()):
        raise ValueError("post/pre requires contiguous BF16 branch and four residual streams")
    if out is None:
        out = torch.empty_like(residual)
    if normalized is None:
        normalized = torch.empty((t, d), dtype=torch.bfloat16, device=x.device)
    if out.shape != residual.shape or normalized.shape != (t, d):
        raise ValueError("post/pre output shapes differ from the residual")
    scratch = torch.empty((t, d), dtype=torch.float32, device=x.device)
    _post_pre[(t,)](x, residual, post, comb, pre, weight, out, scratch, normalized,
                   d, hc, float(eps), 1024, num_warps=8)
    return out, normalized
