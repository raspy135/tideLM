"""DSV41_EXPERT_TOPP: lossy routed-expert pruning in decode (TensorFold's lever, our weights rule).

Per token and layer, after the router's top-6: by weight share descending (ties: the lower slot),
keep the shortest prefix whose mass before it is < TOPP, never fewer than MIN_K, and drop the rest.
A dropped pick weighs 0 and its expert id becomes the sentinel `n_routed_experts`, whose slot-table
column (FastDecoder.build_lut) is the FP4 arena's null slot -- the decode kernels skip null pairs --
or -1 on the EXL3 arena, whose decode grouping skips negative picks. Either way its weights are
never read, so a verify block reads fewer distinct experts.

The kept picks keep the router's own weights; nothing is renormalized. Measured 2026-10-11 on
held-out NLL with every position pruned: 0.90 keeps 91.8% of picks at -0.0020 +-
0.0018 / -0.0004 +- 0.0034 nats (code / general); 0.85 keeps 80.9% at +0.0138 / +0.0106;
renormalizing the kept weights (TensorFold's production rule) costs +0.021 on both, so it is not
offered. Decode only: prompt rows keep all six, and the DSpark drafter's own router is untouched.

Both ranks run the same router on the same rows, and the rule is a function of a row's own
weights in a fixed order, so the ranks agree; the setting is in the boot guard.
"""
from __future__ import annotations

import os

import torch

VERSION = 1
KP = 8      # top-k padded
BR = 16     # rows a program


def config():
    """(topp, min_k) or None when DSV41_EXPERT_TOPP is unset / 0 / off."""
    raw = os.environ.get("DSV41_EXPERT_TOPP", "").strip().lower()
    if raw in ("", "0", "off", "none"):
        return None
    topp = float(raw)
    if not 0.0 < topp < 1.0:
        raise ValueError(f"DSV41_EXPERT_TOPP={raw!r}: a share in (0, 1), e.g. 0.90")
    min_k = int(os.environ.get("DSV41_EXPERT_TOPP_MIN_K", "3"))
    if not 1 <= min_k <= 6:
        raise ValueError("DSV41_EXPERT_TOPP_MIN_K must be 1..6")
    return topp, min_k


def keep_mask(weights: torch.Tensor, topp: float, min_k: int) -> torch.Tensor:
    """The rule in torch: bool [T, K], True for the picks a row keeps."""
    f = weights.float() / weights.float().sum(dim=-1, keepdim=True)
    order = torch.argsort(-f, dim=-1, stable=True)
    fs = f.gather(1, order)
    keep_sorted = (fs.cumsum(dim=-1) - fs) < topp
    keep_sorted[:, :min_k] = True
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)


def reference(idx: torch.Tensor, weights: torch.Tensor, topp: float, min_k: int, sentinel: int):
    """New (idx, weights): dropped picks -> (sentinel, 0). CPU path and the tests' reference."""
    keep = keep_mask(weights, topp, min_k)
    return (torch.where(keep, idx, torch.full_like(idx, sentinel)),
            torch.where(keep, weights, torch.zeros_like(weights)))


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _col(t, j, i):
            return tl.sum(tl.where(j[None, :] == i, t, 0), 1)

        @triton.jit
        def _prune(IDX, W, R, TOPP, MIN_K, SENT, K: tl.constexpr, KP: tl.constexpr, BR: tl.constexpr):
            rows = tl.program_id(0) * BR + tl.arange(0, BR)
            j = tl.arange(0, KP)
            m = (rows < R)[:, None] & (j < K)[None, :]
            off = rows[:, None].to(tl.int64) * K + j[None, :]
            w = tl.load(W + off, mask=m, other=0.0)
            total = tl.zeros((BR,), tl.float32)
            for i in tl.static_range(K):
                total = total + _col(w, j, i)
            f = w / total[:, None]
            rank = tl.zeros((BR, KP), tl.int32)
            for i in tl.static_range(K):
                fi = _col(f, j, i)
                ahead = (j[None, :] < K) & ((f > fi[:, None]) | ((f == fi[:, None]) & (j[None, :] < i)))
                rank = tl.where(j[None, :] == i, tl.sum(ahead.to(tl.int32), 1)[:, None], rank)
            excl = tl.zeros((BR, KP), tl.float32)
            acc = tl.zeros((BR,), tl.float32)
            for r in tl.static_range(K):
                at = (rank == r) & (j[None, :] < K)
                excl = tl.where(at, acc[:, None], excl)
                acc = acc + tl.sum(tl.where(at, f, 0.0), 1)
            keep = (excl < TOPP) | (rank < MIN_K)
            drop = m & ~keep
            tl.store(W + off, tl.zeros((BR, KP), tl.float32), mask=drop)
            tl.store(IDX + off, (tl.zeros((BR, KP), tl.int32) + SENT).to(IDX.dtype.element_ty), mask=drop)

        _KERNEL = (_prune, triton)
    return _KERNEL


def apply_(idx: torch.Tensor, weights: torch.Tensor, topp: float, min_k: int, sentinel: int):
    """Prune (idx [T, K] int, weights [T, K] fp32, contiguous) in place: one launch, graph-safe.

    The kernel's shares are a division and its prefix sums run in rank order; the torch reference
    sums in rank order too (cumsum), so the two agree except where a share sits within an ulp of
    the threshold -- both ranks run the kernel, so they always agree with each other."""
    if not idx.is_cuda:
        i2, w2 = reference(idx, weights, topp, min_k, sentinel)
        idx.copy_(i2)
        weights.copy_(w2)
        return idx, weights
    if not (idx.is_contiguous() and weights.is_contiguous()) or weights.dtype != torch.float32:
        raise ValueError("expert_topp: contiguous idx and fp32 weights")
    T, K = weights.shape
    kern, triton = _kernel()
    kern[(triton.cdiv(T, BR),)](idx, weights, T, float(topp), int(min_k), int(sentinel),
                                K=K, KP=KP, BR=BR, num_warps=1)
    return idx, weights
