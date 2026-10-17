"""The DSA indexer's scores in one kernel (MUSE_INDEX_KERNEL=1).

score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]) / sqrt(D), and -inf past position[t].

The einsum path (GlmTP._indexer with the kernel off) materializes the per-head scores [T, 32, S]
in fp32 and, for an FP8 cache, the whole key cache dequantized to fp32 -- in decode over max_seq,
since a graph needs a fixed S: at 128K that was ~3.3 ms a full layer (21 of them, plus the MTP
layer) for a 6K-token context, a quarter of the decode round. Here each program takes BS positions,
loads their keys once (FP8 -> BF16 is exact; the row scale multiplies the dot after), and does the
heads with one BF16 tensor-core dot per row, fp32 accumulation -- the same products as the fp32
einsum, summed in another order. Blocks wholly past the last row's position only write -inf, so a
6K context reads 6K keys whatever max_seq is.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _index_score_kernel(Q, W, K, KS, POS, OUT, T, S, inv_d,
                        H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, RB: tl.constexpr,
                        SCALED: tl.constexpr):
    pid_s = tl.program_id(0)
    t0 = tl.program_id(1) * RB
    s = pid_s * BS + tl.arange(0, BS)
    hh = tl.arange(0, H)
    dd = tl.arange(0, D)
    s_ok = s < S
    pmax = tl.load(POS + tl.minimum(t0 + RB, T) - 1)          # rows' positions ascend
    if pid_s * BS > pmax:
        for i in tl.static_range(RB):
            t = t0 + i
            tl.store(OUT + t * S + s, tl.full((BS,), float("-inf"), tl.float32), mask=s_ok & (t < T))
    else:
        k = tl.load(K + s[:, None] * D + dd[None, :], mask=s_ok[:, None], other=0.0).to(tl.bfloat16)
        if SCALED:
            ks = tl.load(KS + s, mask=s_ok, other=0.0)
        for i in tl.static_range(RB):
            t = t0 + i
            row = t < T
            q = tl.load(Q + (t * H + hh)[:, None] * D + dd[None, :], mask=row & (hh[:, None] < H), other=0.0)
            sc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)            # [H, BS]
            if SCALED:
                sc = sc * ks[None, :]
            sc = tl.maximum(sc * inv_d, 0.0)
            wt = tl.load(W + t * H + hh, mask=row & (hh < H), other=0.0)
            score = tl.sum(sc * wt[:, None], 0)
            p = tl.load(POS + t, mask=row, other=0)
            score = tl.where(s <= p, score, float("-inf"))
            tl.store(OUT + t * S + s, score, mask=s_ok & row)


def index_scores(q: torch.Tensor, w: torch.Tensor, cache, positions: torch.Tensor, S: int,
                 inv_d: float) -> torch.Tensor:
    """q bf16 [T, H, D] (roped), w fp32 [T, H], cache a BF16 [>=S, D] tensor or an (fp8 [>=S, D],
    fp32 [>=S, 1]) pair, positions int [T] ascending -> fp32 [T, S]. Graph-capturable."""
    T, H, D = q.shape
    if isinstance(cache, torch.Tensor):
        K, KS, scaled = cache, cache, False
    else:
        K, KS = cache
        assert KS.shape[1] == 1, "one scale a key row"
        scaled = True
    assert K.shape[1] == D and K.is_contiguous()
    q = q.to(torch.bfloat16).contiguous()
    w = w.float().contiguous()
    out = torch.empty((T, S), dtype=torch.float32, device=q.device)
    RB = 1 if T == 1 else 4 if T <= 4 else 8
    BS = 64
    grid = (triton.cdiv(S, BS), triton.cdiv(T, RB))
    _index_score_kernel[grid](q, w, K, KS, positions.contiguous(), out, T, S, inv_d,
                              H=H, D=D, BS=BS, RB=RB, SCALED=scaled, num_warps=4)
    return out
