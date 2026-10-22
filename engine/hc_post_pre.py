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
              D: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    j = tl.arange(0, HC)
    post = tl.load(POST + t * HC + j).to(tl.float32)
    pre = tl.load(PRE + t * HC + j).to(tl.float32)
    ssq = tl.zeros((), tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        valid = d < D
        x = tl.load(X + t * D + d, valid, other=0).to(tl.float32)
        acc = tl.zeros((HC, BD), tl.float32)
        for i in range(HC):
            r = tl.load(RES + (t * HC + i) * D + d, valid, other=0).to(tl.float32)
            c = tl.load(COMB + (t * HC + i) * HC + j).to(tl.float32)
            acc += c[:, None] * r[None, :]
        acc += post[:, None] * x[None, :]
        rounded = acc.to(tl.bfloat16)
        tl.store(OUT + (t * HC + j[:, None]) * D + d[None, :], rounded, valid[None, :])
        weighted = tl.zeros((BD,), tl.float32)
        for i in range(HC):
            # Match hc_ops' ordered stream sum, including its BF16 hc_pre boundary.
            value = tl.sum(tl.where(j[:, None] == i, rounded.to(tl.float32), 0), axis=0)
            gain = tl.sum(tl.where(j == i, pre, 0))
            weighted += value * gain
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
