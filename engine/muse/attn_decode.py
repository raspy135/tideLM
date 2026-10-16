"""Split-KV decode attention for Muse Spark's GQA, with the sequence position on the GPU.

A decode step (T <= 16 query rows at positions pos..pos+T-1) attends over a KV cache that is
either the whole context (full layers, NoPE) or a ring of R positions (sliding layers, window W).
The position is read from a device scalar, so one captured CUDA graph serves every step; the
programs past the live length exit early, so a step reads only the keys it needs.

Grid: (KV head, split). Each program takes every query row of its KV head's G query heads
(T*G rows, padded to >= 16 for tl.dot), walks key blocks split, split+S, ... with an online
softmax, and writes a partial (acc, max, sum). ``combine`` merges the splits in torch.
Masks match engine/muse/ref.py: key k is visible to row t (position p) when k <= p, and on
sliding layers also k > p - W.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_split(Q, K, V, POS, ACC, MAX, SUM,
                  stride_qt, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh,
                  scale, R, W,
                  T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, SPLITS: tl.constexpr, SLIDING: tl.constexpr):
    kvh = tl.program_id(0)
    split = tl.program_id(1)
    nkv = tl.num_programs(0)
    rows = tl.arange(0, BM)
    t = rows // G
    g = rows % G
    row_ok = rows < T * G
    d = tl.arange(0, D)
    q = tl.load(Q + t[:, None] * stride_qt + (kvh * G + g)[:, None] * stride_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    pos = tl.load(POS)
    p_row = pos + t
    last = pos + T - 1
    if SLIDING:
        lo = tl.maximum(pos - W + 1, 0)
    else:
        lo = pos * 0
    n_keys = last - lo + 1
    n_blocks = tl.cdiv(n_keys, BN)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    cols = tl.arange(0, BN)
    for b in range(split, n_blocks, SPLITS):
        kpos = lo + b * BN + cols
        kvalid = kpos <= last
        if SLIDING:
            idx = kpos % R
        else:
            idx = kpos
        k = tl.load(K + idx[:, None] * stride_kn + kvh * stride_kh + d[None, :],
                    mask=kvalid[:, None], other=0.0)
        v = tl.load(V + idx[:, None] * stride_vn + kvh * stride_vh + d[None, :],
                    mask=kvalid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        vis = kvalid[None, :] & (kpos[None, :] <= p_row[:, None])
        if SLIDING:
            vis = vis & (kpos[None, :] > p_row[:, None] - W)
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = (split * nkv + kvh) * BM + rows
    tl.store(ACC + out[:, None] * D + d[None, :], acc)
    tl.store(MAX + out, m_i)
    tl.store(SUM + out, l_i)


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     pos: torch.Tensor, *, sliding: bool, ring: int, window: int,
                     scale: float, splits: int = 32, block_n: int = 64) -> torch.Tensor:
    """q [T, Hq, D] bf16 (rows at positions pos..pos+T-1, already written to the cache);
    k/v cache [N, Hkv, D]; pos int64 [1] on the device. Returns [T, Hq, D] in q's dtype."""
    T, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    G = Hq // Hkv
    BM = max(16, triton.next_power_of_2(T * G))
    acc = torch.empty((splits, Hkv, BM, D), dtype=torch.float32, device=q.device)
    mx = torch.empty((splits, Hkv, BM), dtype=torch.float32, device=q.device)
    sm = torch.empty((splits, Hkv, BM), dtype=torch.float32, device=q.device)
    _decode_split[(Hkv, splits)](
        q, k_cache, v_cache, pos, acc, mx, sm,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        scale, ring, window,
        T=T, G=G, D=D, BM=BM, BN=block_n, SPLITS=splits, SLIDING=sliding, num_warps=4)
    m = mx.amax(0, keepdim=True)
    w = torch.exp(mx - torch.where(torch.isinf(m), torch.zeros_like(m), m))
    num = (acc * w[..., None]).sum(0)                       # [Hkv, BM, D]
    den = (sm * w).sum(0)                                    # [Hkv, BM]
    out = (num / den[..., None])[:, :T * G]                  # rows t*G + g
    return out.view(Hkv, T, G, D).permute(1, 0, 2, 3).reshape(T, Hq, D).to(q.dtype)
