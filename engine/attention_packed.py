"""Packed-cache attention with one unpack per key tile and head group.

The rejected decode_attn_staged.attention_direct re-read and unpacked keys for
every 32-column PV tile (236 us versus 47.7 us for staging, gotchas 2026-10-07).
Here each program keeps its whole [16 heads, 512 columns] value accumulator
and reuses the unpacked tile for QK and PV. Nothing materializes full keys or
the probability matrix. FP32 online softmax changes reduction order, so this
remains an opt-in numerics candidate until GPU/full-model qualification.
"""
import torch
import triton
import triton.language as tl
from decode_attn_staged import _key_tile


@triton.jit
def _attend(Q, RING, SLOTS, CACHE, IDS, MASK, SINK, PARTS,
            H: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
            QT: tl.constexpr, QH: tl.constexpr, CS: tl.constexpr,
            SCALE: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    t = tl.program_id(0)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    part = tl.program_id(2)
    query = tl.load(Q + t * QT + heads[:, None] * QH + columns[None, :],
                    heads[:, None] < H, other=0).to(tl.bfloat16)
    # The sink is a zero-value virtual key; seeding it keeps all-masked rows
    # finite without a special host path or exp overflow at a very negative max.
    maximum = tl.load(SINK + heads, heads < H, other=0).to(tl.float32)
    # Each partition owns an exact power-of-two fraction of the zero-value sink.
    denominator = tl.full((16,), 1.0 / SPLIT, tl.float32)
    accumulator = tl.zeros((16, 512), tl.float32)
    chunk: tl.constexpr = triton.cdiv(N1 + N2, BN * SPLIT) * BN
    for first in range(part * chunk, tl.minimum((part + 1) * chunk, N1 + N2), BN):
        rows = first + tl.arange(0, BN)
        keys = _key_tile(RING, SLOTS, CACHE, IDS, t, rows, columns, N1, N2, 512, CS)
        live = tl.load(MASK + t * (N1 + N2) + rows, rows < N1 + N2, other=0)
        logits = tl.dot(query, tl.trans(keys)).to(tl.float32) * SCALE
        logits = tl.where(live[None, :], logits, -float('inf'))
        updated = tl.maximum(maximum, tl.max(logits, 1))
        factor = tl.exp(maximum - updated)
        probability = tl.exp(logits - updated[:, None])
        accumulator *= factor[:, None]
        accumulator = tl.dot(probability, keys.to(tl.float32), accumulator,
                             input_precision='tf32x3')
        denominator = denominator * factor + tl.sum(probability, 1)
        maximum = updated
    base = ((t * H + heads) * SPLIT + part) * 514
    tl.store(PARTS + base[:, None] + columns[None, :], accumulator, heads[:, None] < H)
    tl.store(PARTS + base + 512, maximum, heads < H)
    tl.store(PARTS + base + 513, denominator, heads < H)


@triton.jit
def _merge(PARTS, OUT, H: tl.constexpr, SPLIT: tl.constexpr):
    t, head = tl.program_id(0), tl.program_id(1)
    part = tl.arange(0, SPLIT)
    d = tl.arange(0, 512)
    base = ((t * H + head) * SPLIT + part) * 514
    maximum = tl.load(PARTS + base + 512)
    denominator = tl.load(PARTS + base + 513)
    scale = tl.exp(maximum - tl.max(maximum, 0))
    values = tl.load(PARTS + base[:, None] + d[None, :])
    result = tl.sum(values * scale[:, None], 0) / tl.sum(denominator * scale, 0)
    tl.store(OUT + (t * H + head) * 512 + d, result)


def attention_packed(q, ring, slots, cache, ids, mask, sink, scale):
    t, heads, width = q.shape
    n1, n2 = slots.shape[1], 0 if ids is None else ids.shape[1]
    if (width != 512 or ring.dtype != torch.bfloat16 or not ring.is_contiguous()
            or slots.shape[0] != t or not slots.is_contiguous()
            or tuple(mask.shape) != (t, n1 + n2) or not mask.is_contiguous()
            or sink.numel() != heads):
        raise ValueError('packed attention requires 512-wide BF16 window and contiguous indices/mask')
    if n2 and (cache.dtype != torch.int64 or not cache.is_contiguous()
               or not ids.is_contiguous() or ids.shape[0] != t):
        raise ValueError('packed attention requires packed KV and clamped contiguous indices')
    stride = cache.shape[1] if n2 else 0
    out = torch.empty((t, heads, width), dtype=torch.float32, device=q.device)
    # Partition keys, not value columns: a single group has too few blocks at
    # serving query widths, while value partitions would repeat the unpacking.
    split = 4 if n1 + n2 >= 512 else 1
    parts = torch.empty((t, heads, split, 514), dtype=torch.float32, device=q.device)
    # BN=32 / eight warps needed 151,552 shared bytes on GB10 (limit 101,376)
    # and could not launch. BN=16 / four warps passes the graph/numerics test.
    _attend[(t, triton.cdiv(heads, 16), split)](q, ring, slots, cache if n2 else ring,
        ids if n2 else slots, mask, sink, parts, heads, n1, n2, q.stride(0), q.stride(1),
        stride, float(scale), 16, split, num_warps=4, num_stages=1)
    _merge[(t, heads)](parts, out, heads, split, num_warps=4)
    return out
