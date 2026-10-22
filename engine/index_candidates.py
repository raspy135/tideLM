"""Fixed-capacity candidate lists for long-context decode, without changing cache precision.

The source layer still selects its blocks with Model._select_candidates. Later
layers score only that list. The fused score's reduction order can differ from
cuBLAS/ATen, so this is a ranking-numerics experiment, not a bit-exact claim.
"""
import torch


def compact_mask(mask, block_size, capacity):
    """Return ascending key IDs, padded with -1, without nonzero or host readback.

    Source masks select complete blocks; a last partial block is permitted.
    The caller supplies a capacity at least as large as its block-selection limit.
    """
    if mask.ndim != 2 or mask.dtype != torch.bool or block_size < 1:
        raise ValueError("expected a boolean [queries, keys] block mask")
    if capacity < 1 or capacity % block_size:
        raise ValueError("candidate capacity must be a positive multiple of block size")
    rows, width = mask.shape
    blocks = mask[:, ::block_size]
    count = min(capacity // block_size, blocks.shape[1])
    chosen = blocks.to(torch.int32).topk(count, dim=-1, sorted=False).indices
    chosen = chosen.sort(dim=-1).values
    offsets = torch.arange(block_size, device=mask.device)
    ids = (chosen[:, :, None] * block_size + offsets).reshape(rows, -1)
    live = (ids < width) & blocks.gather(1, chosen).repeat_interleave(block_size, dim=-1)
    ids = torch.where(live, ids, -1)
    return torch.nn.functional.pad(ids, (0, capacity - ids.shape[1]), value=-1)


def score_reference(query, keys, weights, ids, lengths):
    """CPU/GPU reference retaining BF16 dot-product rounding and FP32 head sums."""
    gathered = keys[ids.clamp_min(0)]
    dots = torch.einsum("thd,tnd->thn", query, gathered)
    scores = (dots.float().relu() * weights[:, :, None]).sum(1)
    return scores.masked_fill((ids < 0) | (ids >= lengths[:, None]), -float("inf"))


def scores(query, keys, weights, ids, lengths):
    if query.device.type != "cuda":
        return score_reference(query, keys, weights, ids, lengths)
    from engine.index_candidates_cuda import candidate_scores
    return candidate_scores(query, keys, weights, ids, lengths)


def select(scores, ids, topk):
    """Map selected list columns back to sorted context positions, keeping -1 padding."""
    count = min(topk, scores.shape[1])
    values, columns = scores.topk(count, dim=-1, sorted=False)
    selected = ids.gather(1, columns)
    selected = torch.where(torch.isfinite(values), selected, -1)
    # The attention mask handles -1; ordering only the live positions avoids
    # selecting a padding column as a key when fewer than topk keys are visible.
    return torch.nn.functional.pad(selected.sort(dim=-1).values,
                                   (0, topk - count), value=-1)
