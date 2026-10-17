"""GLM's sparse MLA attention in one kernel (MUSE_ATTN_KERNEL=1): each query row attends to the
cached positions the indexer picked, reading them where they live.

The einsum path (GlmTP.attention with the kernel off) gathers each 32-row block's selected latents
into BF16 copies -- kv_read(C, sel) is [32, 2048, 512] = 64 MB written and read back per block,
plus the RoPE keys -- then runs bf16 einsums and a softmax over [32, heads, 2048]. In prefill that
is 32 blocks a 1,024-row chunk on every layer: 30 of a 6K prompt's 42 s.

Here a program owns BLOCK_H heads of one row and walks its selected positions BLOCK_N at a time:
the latent rows are loaded by index (FP8 widened to BF16 and multiplied by their 128-block scale
in BF16, as kv_read does), scored against the absorbed query (latent part + RoPE part, fp32
accumulation), and folded into an online softmax whose BF16 probabilities weight the same latent
tile for the output. Differences from the einsum path: the scores stay fp32 (the einsums round
them to BF16 twice) and the probabilities are rounded per tile against a running maximum instead
of after a full softmax -- accumulation-order and rounding-point changes, checked against an fp32
reference in engine/glm/test_sparse_attn.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1e30)


@triton.jit
def _sparse_mla_kernel(QL, QR, C, CS, R, IDX, POS, OUT, PM, PL, PACC, H, K, CHUNK, NS,
                       stride_qlt, stride_qlh, stride_qrt, stride_qrh, stride_c, stride_cs, stride_r,
                       stride_it, stride_ot, stride_oh, scale,
                       DC: tl.constexpr, DR: tl.constexpr, SB: tl.constexpr, BLOCK_H: tl.constexpr,
                       BLOCK_N: tl.constexpr, SCALED: tl.constexpr, SPLIT: tl.constexpr):
    pid_h = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    sp = tl.program_id(2)
    k_lo = sp * CHUNK
    k_hi = tl.minimum(K, k_lo + CHUNK)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = offs_h < H
    dc = tl.arange(0, DC)
    dr = tl.arange(0, DR)
    ql = tl.load(QL + t * stride_qlt + offs_h[:, None] * stride_qlh + dc[None, :], mask=hm[:, None], other=0.0)
    qr = tl.load(QR + t * stride_qrt + offs_h[:, None] * stride_qrh + dr[None, :], mask=hm[:, None], other=0.0)
    pos = tl.load(POS + t)
    m_i = tl.full((BLOCK_H,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, DC), tl.float32)
    for n0 in range(k_lo, k_hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < k_hi
        idx = tl.load(IDX + t * stride_it + offs_n, mask=nm, other=0)
        keep = nm & (idx >= 0) & (idx <= pos)
        row = tl.maximum(idx, 0).to(tl.int64)
        c = tl.load(C + row[:, None] * stride_c + dc[None, :], mask=nm[:, None], other=0.0).to(tl.bfloat16)
        if SCALED:
            nb: tl.constexpr = DC // SB
            s = tl.load(CS + row[:, None] * stride_cs + tl.arange(0, nb)[None, :], mask=nm[:, None], other=0.0)
            c = tl.reshape(tl.reshape(c, (BLOCK_N, nb, SB)) * s.to(tl.bfloat16)[:, :, None], (BLOCK_N, DC))
        r = tl.load(R + row[:, None] * stride_r + dr[None, :], mask=nm[:, None], other=0.0)
        sc = tl.dot(ql, tl.trans(c), out_dtype=tl.float32) + tl.dot(qr, tl.trans(r), out_dtype=tl.float32)
        sc = tl.where(keep[None, :], sc * scale, float("-inf"))
        m_new = tl.maximum(tl.maximum(m_i, tl.max(sc, 1)), _NEG)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), c, out_dtype=tl.float32)
        m_i = m_new
    if not SPLIT:
        out = acc / tl.maximum(l_i, 1e-30)[:, None]
        tl.store(OUT + t * stride_ot + offs_h[:, None] * stride_oh + dc[None, :], out.to(tl.bfloat16),
                 mask=hm[:, None])
    else:
        # This split's unnormalized partials; sparse_mla merges the splits (flash-decoding).
        b = (t * H + offs_h) * NS + sp
        tl.store(PM + b, m_i, mask=hm)
        tl.store(PL + b, l_i, mask=hm)
        tl.store(PACC + b[:, None] * DC + dc[None, :], acc, mask=hm[:, None])


# (block_h, block_n, num_warps, num_stages) by (FP8 cache, prefill rows): the fastest of a sweep on
# GB10 that fits its 99 KB of shared memory -- a BF16 latent tile is twice an FP8 one, and 32 heads x
# 64 keys x 8 warps over a BF16 cache needs 168 KB. FP8: 2,048 rows 8.78 ms, 4 rows 0.17 ms; BF16:
# 9.29 / 0.14 ms.
# Decode adds a split of the selected positions (flash-decoding) to ~32 programs: 4 rows over an
# FP8 cache 170 -> 49 us (32 heads a program, 8 splits), 1 row 170 -> 32 us; BF16 4 rows 157 -> 59.
TILES = {(True, True): (32, 64, 8, 1), (True, False): (32, 64, 8, 1),
         (False, True): (32, 32, 4, 2), (False, False): (16, 32, 4, 2)}


def sparse_mla(q_lat: torch.Tensor, q_rope: torch.Tensor, cache, rope: torch.Tensor, sel: torch.Tensor,
               positions: torch.Tensor, scale: float, block_h: int | None = None, block_n: int | None = None,
               num_warps: int | None = None, num_stages: int | None = None,
               split: int | None = None) -> torch.Tensor:
    """q_lat bf16 [T, H, 512] (absorbed), q_rope bf16 [T, H, 64]; cache the latent cache, BF16
    [S, 512] or an (fp8 [S, 512], fp32 [S, 512 / 128]) pair; rope BF16 [S, 64]; sel int [T, K]
    (positions, attended where <= the row's position); positions int [T] -> o_lat bf16 [T, H, 512].
    Graph-capturable."""
    T, H, DC = q_lat.shape
    DR = q_rope.shape[-1]
    if isinstance(cache, torch.Tensor):
        C, CS, scaled, sb = cache, cache, False, DC
    else:
        C, CS = cache
        scaled, sb = True, DC // CS.shape[1]
    assert q_lat.stride(-1) == 1 and q_rope.stride(-1) == 1 and C.stride(-1) == 1 and rope.stride(-1) == 1
    sel = sel.contiguous()
    th, tn, tw, ts = TILES[(scaled, T > 16)]
    block_h, block_n = min(block_h or th, H), block_n or tn
    num_warps, num_stages = num_warps or tw, num_stages or ts
    o = torch.empty((T, H, DC), dtype=torch.bfloat16, device=q_lat.device)
    if T == 0:
        return o
    K = sel.shape[1]
    hb = triton.cdiv(H, block_h)
    # Decode has a handful of rows: split the selected positions so the grid fills the GPU
    # (4 rows x 2 head blocks were 8 programs on 48 SMs, 0.2 ms a layer), then merge.
    nsplit = 1 if T > 16 else (split or max(1, min(32, triton.next_power_of_2(triton.cdiv(32, hb * T)))))
    chunk = triton.cdiv(triton.cdiv(K, nsplit), block_n) * block_n
    nsplit = triton.cdiv(K, chunk)
    if nsplit > 1:
        pm = torch.empty((T, H, nsplit), dtype=torch.float32, device=q_lat.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty((T, H, nsplit, DC), dtype=torch.float32, device=q_lat.device)
    else:
        pm = pl = pacc = o
    grid = (hb, T, nsplit)
    _sparse_mla_kernel[grid](q_lat, q_rope, C, CS, rope, sel, positions.contiguous(), o, pm, pl, pacc, H, K, chunk,
                             nsplit,
                             q_lat.stride(0), q_lat.stride(1), q_rope.stride(0), q_rope.stride(1),
                             C.stride(0), CS.stride(0), rope.stride(0), sel.stride(0), o.stride(0), o.stride(1),
                             scale, DC=DC, DR=DR, SB=sb, BLOCK_H=block_h, BLOCK_N=block_n, SCALED=scaled,
                             SPLIT=nsplit > 1, num_warps=num_warps, num_stages=num_stages)
    if nsplit > 1:
        # Merge: a split with no attended position has m = -1e30, l = 0 and adds nothing.
        m = pm.amax(-1, keepdim=True)
        w = torch.exp(pm - m)
        o.copy_((pacc * w[..., None]).sum(2) / (pl * w).sum(-1, keepdim=True).clamp_min(1e-30))
    return o
