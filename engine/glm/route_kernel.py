"""Decode routing in two kernels (GLM_ROUTE_KERNEL=1), for up to 8 rows.

A MoE layer's routing ran as ~80 PyTorch kernels of 1-8 us each in a decode graph (sigmoid, bias,
two top-k picks, gathers, normalization, the demand and miss counters, the slot lookup and the
patching of slotless picks): ~130 us a layer against ~10 of arithmetic. Here it is two programs:

  exact_pick   in the attention graph, after the router matmul: the unmasked top-k (the picks
               an exact model makes: demand, miss statistics, the hot-read candidates) and the
               hot candidates' sorted (share, expert) list into its device row.
  routed_pick  in the MLP graph, after the hot reads installed their experts: the top-k over
               resident experts, its renormalized weights, slots from the lut (a slotless pick
               becomes a zero-weight pick of the layer's first fixed slot), the demand update
               and the miss counters.

Both recompute sigmoid(logits) + bias the same way, so they agree on the scores. Against the
PyTorch path the picks are the same sets; the order is descending (torch.topk(sorted=False)'s is
unspecified) and the weight sums add in that order, so weights can differ in the last fp32 bit --
both ranks run this same code on the same replicated rows and agree. Ties between equal scores
go to the lower expert index. A row with fewer than k resident experts gets zero-weight picks for
the rest (the PyTorch path normalized over the scores of the -inf picks it happened to take).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(LG, s_lg, B, rows, cols, T):
    rmask = rows < T
    lg = tl.load(LG + rows[:, None] * s_lg + cols[None, :], mask=rmask[:, None], other=0.0)
    sc = tl.sigmoid(lg)
    ch = sc + tl.load(B + cols)[None, :]
    return sc, tl.where(rmask[:, None], ch, float("-inf"))


@triton.jit
def _exact_kernel(LG, s_lg, B, ALLOWED, IDS0, W0, HOT, T, n2, kh, scale,
                  E: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    cols = tl.arange(0, E)
    kk = tl.arange(0, K)
    rmask = rows < T
    sc, ch = _scores(LG, s_lg, B, rows, cols, T)
    al = tl.load(ALLOWED + cols) != 0
    ids = tl.zeros((BT, K), tl.int32)
    s = tl.zeros((BT, K), tl.float32)
    missed = tl.zeros((BT, K), tl.float32)
    tot = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(K):
        idx = tl.argmax(ch, 1)
        hit = cols[None, :] == idx[:, None]
        sj = tl.sum(tl.where(hit, sc, 0.0), 1)
        mj = tl.sum(tl.where(hit & (al[None, :] == 0), 1.0, 0.0), 1)
        at = kk[None, :] == j
        ids = tl.where(at, idx[:, None], ids)
        s = tl.where(at, sj[:, None], s)
        missed = tl.where(at, mj[:, None], missed)
        tot += sj
        ch = tl.where(hit, float("-inf"), ch)
    w0 = s / (tot[:, None] + 1e-20) * scale
    m2 = rmask[:, None]
    tl.store(IDS0 + rows[:, None] * K + kk[None, :], ids.to(tl.int64), mask=m2)
    tl.store(W0 + rows[:, None] * K + kk[None, :], w0, mask=m2)
    # Hot candidates: the kh largest missed shares over every row's picks, sorted, then their
    # experts (as floats) at n2 -- what _hot_candidate's flat topk wrote.
    share = tl.where(m2, w0 / scale * missed, float("-inf"))
    flat_v = tl.reshape(share, (BT * K,))
    flat_i = tl.reshape(ids.to(tl.float32), (BT * K,))
    pos = tl.arange(0, BT * K)
    for j in range(kh):
        b = tl.argmax(flat_v, 0)
        tl.store(HOT + j, tl.max(flat_v, 0))
        tl.store(HOT + n2 + j, tl.sum(tl.where(pos == b, flat_i, 0.0), 0))
        flat_v = tl.where(pos == b, float("-inf"), flat_v)


@triton.jit
def _routed_kernel(LG, s_lg, B, ALLOWED, LUT, FIXED, IDS0, W0, DEM, ACC, RS, WR, SL, WL, T, off, scale,
                   E: tl.constexpr, K: tl.constexpr, F: tl.constexpr, BT: tl.constexpr,
                   X3: tl.constexpr, DEMAND: tl.constexpr, STATS: tl.constexpr):
    rows = tl.arange(0, BT)
    cols = tl.arange(0, E)
    kk = tl.arange(0, K)
    rmask = rows < T
    m2 = rmask[:, None]
    sc, ch = _scores(LG, s_lg, B, rows, cols, T)
    al = tl.load(ALLOWED + cols) != 0
    ch = tl.where(al[None, :], ch, float("-inf"))
    ids = tl.zeros((BT, K), tl.int32)
    s = tl.zeros((BT, K), tl.float32)
    ok = tl.zeros((BT, K), tl.int1)
    tot = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(K):
        best = tl.max(ch, 1)
        idx = tl.argmax(ch, 1)
        hit = cols[None, :] == idx[:, None]
        valid = best > float("-inf")
        sj = tl.where(valid, tl.sum(tl.where(hit, sc, 0.0), 1), 0.0)
        at = kk[None, :] == j
        ids = tl.where(at, idx[:, None], ids)
        s = tl.where(at, sj[:, None], s)
        ok = tl.where(at, valid[:, None], ok)
        tot += sj
        ch = tl.where(hit, float("-inf"), ch)
    w = s / (tot[:, None] + 1e-20) * scale
    slot = tl.load(LUT + ids, mask=m2 & ok, other=-1)
    f0 = tl.load(FIXED)
    missing = slot < 0
    w = tl.where(missing, 0.0, w)
    slot = tl.where(missing, f0, slot)
    fk = tl.arange(0, F)
    fixed = tl.load(FIXED + fk)
    if X3:
        r = slot - off
        tl.store(RS + rows[:, None] * K + kk[None, :], tl.maximum(r, 0), mask=m2)
        tl.store(WR + rows[:, None] * K + kk[None, :], tl.where(r < 0, 0.0, w), mask=m2)
        tl.store(SL + rows[:, None] * F + fk[None, :], tl.broadcast_to(fixed[None, :], (BT, F)), mask=m2)
        tl.store(WL + rows[:, None] * F + fk[None, :], tl.full((BT, F), 1.0, tl.float32), mask=m2)
    else:
        tl.store(SL + rows[:, None] * (K + F) + kk[None, :], slot, mask=m2)
        tl.store(WL + rows[:, None] * (K + F) + kk[None, :], w, mask=m2)
        tl.store(SL + rows[:, None] * (K + F) + K + fk[None, :], tl.broadcast_to(fixed[None, :], (BT, F)), mask=m2)
        tl.store(WL + rows[:, None] * (K + F) + K + fk[None, :], tl.full((BT, F), 1.0, tl.float32), mask=m2)
    if DEMAND or STATS:
        ids0 = tl.load(IDS0 + rows[:, None] * K + kk[None, :], mask=m2, other=0)
        w0 = tl.load(W0 + rows[:, None] * K + kk[None, :], mask=m2, other=0.0)
        if DEMAND:
            tl.atomic_add(DEM + ids0, w0, mask=m2)
        if STATS:
            mis = tl.where(m2 & (tl.load(ALLOWED + ids0, mask=m2, other=1) == 0), 1.0, 0.0)
            tl.atomic_add(ACC + 0, tl.zeros((), tl.float64) + T * K)   # T is a Python int when 1
            tl.atomic_add(ACC + 1, tl.sum(tl.sum(mis, 1), 0).to(tl.float64))
            tl.atomic_add(ACC + 2, tl.sum(tl.sum(w0, 1), 0).to(tl.float64))
            tl.atomic_add(ACC + 3, tl.sum(tl.sum(w0 * mis, 1), 0).to(tl.float64))


def _bt(T):
    assert 1 <= T <= 8, T
    return triton.next_power_of_2(T)


def exact_pick(logits, bias, allowed_u8, k, scale, hot_row=None, n2=0):
    """logits fp32 [T, E] -> (ids0 int64 [T, k], w0 fp32 [T, k]); with ``hot_row`` (fp32
    [>= 2 * n2]) also its first min(T * k, n2) shares and, at n2, their experts."""
    T, E = logits.shape
    ids0 = torch.empty((T, k), dtype=torch.int64, device=logits.device)
    w0 = torch.empty((T, k), dtype=torch.float32, device=logits.device)
    kh = min(T * k, n2) if hot_row is not None else 0
    _exact_kernel[(1,)](logits, logits.stride(0), bias, allowed_u8, ids0, w0,
                        hot_row if hot_row is not None else w0, T, n2, kh, scale,
                        E=E, K=k, BT=_bt(T), num_warps=4)
    return ids0, w0


def routed_pick(logits, bias, allowed_u8, lut, fixed, ids0, w0, scale, *, x3_off=None,
                demand=None, acc=None):
    """The resident top-k -> x3_off set: (rs int32 [T, k], wr fp32 [T, k], fixed slots int32
    [T, F], ones [T, F]) as GlmTP.mlp splits them; else (slots int32 [T, k + F], w fp32 [T, k + F]).
    Adds w0 at ids0 into ``demand`` (fp32 [E]) and the miss counters into ``acc`` (fp64 [4])."""
    T, E = logits.shape
    k, F = ids0.shape[1], fixed.numel()
    dev = logits.device
    x3 = x3_off is not None
    if x3:
        rs = torch.empty((T, k), dtype=torch.int32, device=dev)
        wr = torch.empty((T, k), dtype=torch.float32, device=dev)
        sl = torch.empty((T, F), dtype=torch.int32, device=dev)
        wl = torch.empty((T, F), dtype=torch.float32, device=dev)
    else:
        rs = wr = None
        sl = torch.empty((T, k + F), dtype=torch.int32, device=dev)
        wl = torch.empty((T, k + F), dtype=torch.float32, device=dev)
    _routed_kernel[(1,)](logits, logits.stride(0), bias, allowed_u8, lut, fixed, ids0, w0,
                         demand if demand is not None else w0, acc if acc is not None else w0,
                         rs if x3 else sl, wr if x3 else wl, sl, wl, T, int(x3_off or 0), scale,
                         E=E, K=k, F=F, BT=_bt(T), X3=x3, DEMAND=demand is not None,
                         STATS=acc is not None, num_warps=4)
    return (rs, wr, sl, wl) if x3 else (sl, wl)
