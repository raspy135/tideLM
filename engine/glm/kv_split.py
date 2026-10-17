"""The KV cache split across the two ranks by position (MUSE_KV_SPLIT=1).

GLM's MLA caches one latent row per position that every head reads, so tensor parallelism over
heads leaves both ranks holding the whole cache: 54.7 KB a position a rank in FP8 (7.2 GB at 128K).
Split, rank r keeps positions p with p % world == r in row p // world -- interleaved, so both
ranks hold and scan the same amount at any context length -- and the MTP layer keeps its cache
whole (1/79 of it, and its one-row draft steps are the latency-bound calls).

Decode (GlmTP._split_attn): each rank scores its own positions with the indexer and keeps its
top-k (sorted); one all-gather carries those scores and the rank's absorbed queries. The global
top-k of the two lists is the indexer's top-k over every position (ties aside), and each rank's
part of it is a prefix of its own list (`share`). Each rank then attends every head over its part
(sparse_attn.split_partials), and a second all-gather hands each rank the other's partials for its
heads, combined as in `merge` (sparse_attn.merge_received) -- the flash-decoding merge across
ranks. Payloads past the RoCE slot go as several gathers (`gather_rows`).

Prefill (GlmTP._gathered): one all-gather a layer of both ranks' rows for [0, S) into position
order (`gather_spans`), then the unsplit path -- the same bytes the unsplit cache would hold.
"""

from __future__ import annotations

import torch


def owner_rows(positions: torch.Tensor, rank: int, world: int, spare: int) -> torch.Tensor:
    """The cache rows to write `positions` at: this rank's p -> p // world, the others -> `spare`
    (a last row nothing reads; index_copy_ may write it more than once)."""
    return torch.where(positions % world == rank, torch.div(positions, world, rounding_mode="floor"),
                       torch.full_like(positions, spare))


def local_bound(positions: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """For each query position, the last cache row it may attend: this rank's largest position at
    or before it, as a row (-1 where there is none)."""
    return torch.div(positions - rank, world, rounding_mode="floor")


def share(vals_all: torch.Tensor, rank: int) -> torch.Tensor:
    """vals_all fp32 [world, T, K]: each rank's top-K index scores, each row sorted descending.
    -> int32 [T]: how many of rank `rank`'s are in the top K of all world * K -- a prefix of its
    list. Both ranks compute it from the same gathered values."""
    W, T, K = vals_all.shape
    top = vals_all.permute(1, 0, 2).reshape(T, W * K).topk(K, -1).indices
    return (top // K == rank).sum(-1).to(torch.int32)


def merge(o_a: torch.Tensor, lse_a: torch.Tensor, o_b: torch.Tensor, lse_b: torch.Tensor) -> torch.Tensor:
    """Two partial attentions over disjoint positions (o [..., D] normalized over its own, lse
    [...], -inf for none) -> the attention over both, fp32."""
    m = torch.maximum(lse_a, lse_b)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    wa, wb = torch.exp(lse_a - m), torch.exp(lse_b - m)
    return (o_a.float() * wa[..., None] + o_b.float() * wb[..., None]) / (wa + wb).clamp_min(1e-30)[..., None]


def gather_rows(comm, t: torch.Tensor, limit: int = 256 * 1024) -> torch.Tensor:
    """comm.all_gather of a [rows, bytes] tensor -> [world, rows, bytes]; with the RoCE path, in
    pieces of at most `limit` bytes a rank (its slot) rather than one gather over it, which goes to
    NCCL: inside a decode graph that cost ~0.2 ms a gather (attention at 6-8 verify rows went
    1.15 -> 1.7 ms a layer) where a RoCE one is ~15-30 us."""
    rows, b = t.shape
    if not getattr(comm, "fast", False) or rows * b <= limit or rows == 1:
        return comm.all_gather(t)
    step = max(1, limit // b)
    return torch.cat([comm.all_gather(t[a:a + step]) for a in range(0, rows, step)], 1)


def gather_spans(comm, parts: list[torch.Tensor], n: int, world: int) -> list[torch.Tensor]:
    """parts: this rank's caches ([rows, d], any dtypes) -> each for positions [0, world * n) in
    position order, from every rank's first n rows: one all-gather of the rows' bytes."""
    rows = [p[:n].contiguous().view(torch.uint8).reshape(n, -1) for p in parts]
    g = comm.all_gather(torch.cat(rows, 1))                       # [world, n, bytes]
    out, o = [], 0
    for p, r in zip(parts, rows):
        b = r.shape[1]
        full = g[:, :, o:o + b].transpose(0, 1).contiguous()     # [n, world, b]: position order
        out.append(full.view(p.dtype).reshape(world * n, *p.shape[1:]))
        o += b
    return out
