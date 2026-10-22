"""Score candidate key blocks in registers; never materialize [T, heads, context]."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scores(Q, K, W, IDS, LENGTHS, OUT, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            QS0: tl.constexpr, QS1: tl.constexpr, KS: tl.constexpr, WS: tl.constexpr,
            BH: tl.constexpr, BN: tl.constexpr):
    t, tile = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, BH)
    cols = tile * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    idx = tl.load(IDS + t * N + cols, cols < N, other=-1)
    length = tl.load(LENGTHS + t)
    valid = (cols < N) & (idx >= 0) & (idx < length)
    q = tl.load(Q + t * QS0 + heads[:, None] * QS1 + d[None, :], heads[:, None] < H, other=0)
    k = tl.load(K + tl.maximum(idx, 0)[:, None].to(tl.int64) * KS + d[None, :], valid[:, None], other=0)
    dots = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
    weights = tl.load(W + t * WS + heads, heads < H, other=0)
    result = tl.sum(tl.maximum(dots, 0) * weights[:, None], axis=0)
    tl.store(OUT + t * N + cols, tl.where(valid, result, -float("inf")), cols < N)


def candidate_scores(query, keys, weights, ids, lengths):
    t, h, d = query.shape
    if (query.dtype != torch.bfloat16 or keys.dtype != torch.bfloat16
            or weights.dtype != torch.float32 or d != 128
            or query.stride(-1) != 1 or keys.stride(-1) != 1
            or not ids.is_contiguous() or ids.shape[0] != t):
        raise ValueError("candidate scoring requires BF16 128-wide queries/keys and FP32 weights")
    n = ids.shape[1]
    out = torch.empty((t, n), dtype=torch.float32, device=query.device)
    _scores[(t, triton.cdiv(n, 64))](query, keys, weights, ids, lengths, out, n, h, d,
        query.stride(0), query.stride(1), keys.stride(0), weights.stride(0),
        BH=max(16, triton.next_power_of_2(h)), BN=64, num_warps=4, num_stages=2,
        enable_fp_fusion=False)
    return out
