"""GLM's sparse MLA attention in one kernel (GLM_ATTN_KERNEL=1): each query row attends to the
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

With the KV cache split across ranks by position (GLM_KV_SPLIT), a rank attends every head over
its own share of the picks -- a per-row count of them, `cnt`, at the front of `sel` -- and returns
partials (the output normalized over its share, and the share's log-sum-exp) that
engine/glm/kv_split.merge combines with the other rank's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1e30)


@triton.jit
def _sparse_mla_kernel(QL, QR, C, CS, R, IDX, POS, CNT, OUT, PM, PL, PACC, H, K, CHUNK, NS, HG,
                       stride_qlg, stride_qlt, stride_qlh, stride_qrg, stride_qrt, stride_qrh,
                       stride_c, stride_cs, stride_r,
                       stride_it, stride_ot, stride_oh, scale,
                       DC: tl.constexpr, DR: tl.constexpr, SB: tl.constexpr, BLOCK_H: tl.constexpr,
                       BLOCK_N: tl.constexpr, SCALED: tl.constexpr, SPLIT: tl.constexpr,
                       HAS_CNT: tl.constexpr):
    pid_h = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    sp = tl.program_id(2)
    if HAS_CNT:
        # The row's first cnt picks, split evenly (in BLOCK_N steps) over the NS programs.
        cnt = tl.load(CNT + t)
        chunk = ((cnt + NS - 1) // NS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        k_lo = sp * chunk
        k_hi = tl.minimum(cnt, k_lo + chunk)
    else:
        k_lo = sp * CHUNK
        k_hi = tl.minimum(K, k_lo + CHUNK)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = offs_h < H
    dc = tl.arange(0, DC)
    dr = tl.arange(0, DR)
    # Heads in groups of HG (a gathered [rank, row, head] query: GlmTP._split_attn), else one group.
    grp, hh = offs_h // HG, offs_h % HG
    ql = tl.load(QL + grp[:, None] * stride_qlg + t * stride_qlt + hh[:, None] * stride_qlh + dc[None, :],
                 mask=hm[:, None], other=0.0)
    qr = tl.load(QR + grp[:, None] * stride_qrg + t * stride_qrt + hh[:, None] * stride_qrh + dr[None, :],
                 mask=hm[:, None], other=0.0)
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
               split: int | None = None, cnt: torch.Tensor | None = None, partial: bool = False):
    """q_lat bf16 [T, H, 512] (absorbed), q_rope bf16 [T, H, 64]; cache the latent cache, BF16
    [S, 512] or an (fp8 [S, 512], fp32 [S, 512 / 128]) pair; rope BF16 [S, 64]; sel int [T, K]
    (cache rows, attended where 0 <= row <= the row's position); positions int [T] -> o_lat bf16
    [T, H, 512]. cnt int32 [T]: only each row's first cnt picks. partial: (o fp32 [T, H, 512]
    normalized over the picks attended, lse fp32 [T, H], -inf where none was) instead.
    Graph-capturable."""
    if q_lat.dim() == 4:                       # [groups, T, heads a group, d]: head g * HG + h
        G, T, HG, DC = q_lat.shape
        H = G * HG
        qlg, qlt, qlh = q_lat.stride(0), q_lat.stride(1), q_lat.stride(2)
        qrg, qrt, qrh = q_rope.stride(0), q_rope.stride(1), q_rope.stride(2)
    else:
        T, H, DC = q_lat.shape
        HG = H
        qlg, qlt, qlh = 0, q_lat.stride(0), q_lat.stride(1)
        qrg, qrt, qrh = 0, q_rope.stride(0), q_rope.stride(1)
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
    use_split = nsplit > 1 or partial
    if use_split:
        pm = torch.empty((T, H, nsplit), dtype=torch.float32, device=q_lat.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty((T, H, nsplit, DC), dtype=torch.float32, device=q_lat.device)
    else:
        pm = pl = pacc = o
    if cnt is not None:
        cnt = cnt.to(torch.int32).contiguous()
    grid = (hb, T, nsplit)
    _sparse_mla_kernel[grid](q_lat, q_rope, C, CS, rope, sel, positions.contiguous(),
                             cnt if cnt is not None else sel, o, pm, pl, pacc, H, K, chunk, nsplit, HG,
                             qlg, qlt, qlh, qrg, qrt, qrh,
                             C.stride(0), CS.stride(0), rope.stride(0), sel.stride(0), o.stride(0), o.stride(1),
                             scale, DC=DC, DR=DR, SB=sb, BLOCK_H=block_h, BLOCK_N=block_n, SCALED=scaled,
                             SPLIT=use_split, HAS_CNT=cnt is not None, num_warps=num_warps, num_stages=num_stages)
    if partial == "raw":
        return pm, pl, pacc
    if use_split:
        # Merge: a split with no attended position has m = -1e30, l = 0 and adds nothing.
        m = pm.amax(-1, keepdim=True)
        w = torch.exp(pm - m)
        acc, l = (pacc * w[..., None]).sum(2), (pl * w).sum(-1, keepdim=True)
        if partial:
            lse = torch.where(l > 0, m + torch.log(l.clamp_min(1e-30)), float("-inf"))[..., 0]
            return acc / l.clamp_min(1e-30), lse
        o.copy_(acc / l.clamp_min(1e-30))
    return o


@triton.jit
def _combine_kernel(PM, PL, PACC, OWN_O, OWN_L, SEND, H, HL, NS, LO, ROW_BYTES, OFF_L,
                    DC: tl.constexpr, NSP: tl.constexpr):
    """One (row, head): merge the head's splits -> the output normalized over this rank's picks and
    their log-sum-exp; this rank's heads [LO, LO + HL) to OWN_O / OWN_L (fp32), the others to the
    row's SEND bytes (FP16 outputs [HL, DC], then fp32 log-sum-exps [HL])."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    sp = tl.arange(0, NSP)
    sm = sp < NS
    b = (t * H + h) * NS
    pm = tl.load(PM + b + sp, mask=sm, other=-1e30)
    pl = tl.load(PL + b + sp, mask=sm, other=0.0)
    m = tl.max(pm, 0)
    w = tl.where(sm, tl.exp(pm - m), 0.0)
    l = tl.sum(pl * w, 0)
    dc = tl.arange(0, DC)
    acc = tl.load(PACC + (b + sp)[:, None] * DC + dc[None, :], mask=sm[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / tl.maximum(l, 1e-30)
    lse = tl.where(l > 0, m + tl.log(tl.maximum(l, 1e-30)), float("-inf"))
    hl = h - LO
    if (hl >= 0) & (hl < HL):
        tl.store(OWN_O + (t * HL + hl) * DC + dc, o)
        tl.store(OWN_L + t * HL + hl, lse)
    else:
        ho = tl.where(h < LO, h, h - HL)               # the other rank's head, 0 .. HL - 1
        row = SEND + t * ROW_BYTES
        tl.store(row.to(tl.pointer_type(tl.float16), bitcast=True) + ho * DC + dc,
                 tl.minimum(tl.maximum(o, -65504.0), 65504.0).to(tl.float16))
        tl.store((row + OFF_L).to(tl.pointer_type(tl.float32), bitcast=True) + ho, lse)


@triton.jit
def _merge_kernel(OWN_O, OWN_L, RECV, OUT, HL, ROW_BYTES, OFF_L, DC: tl.constexpr):
    """One (row, head): this rank's partial (fp32) and the other rank's from RECV -> BF16."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    dc = tl.arange(0, DC)
    oa = tl.load(OWN_O + (t * HL + h) * DC + dc)
    la = tl.load(OWN_L + t * HL + h)
    row = RECV + t * ROW_BYTES
    ob = tl.load(row.to(tl.pointer_type(tl.float16), bitcast=True) + h * DC + dc).to(tl.float32)
    lb = tl.load((row + OFF_L).to(tl.pointer_type(tl.float32), bitcast=True) + h)
    m = tl.maximum(la, lb)
    m = tl.where(m == float("-inf"), 0.0, m)
    wa = tl.exp(la - m)
    wb = tl.exp(lb - m)
    o = (oa * wa + ob * wb) / tl.maximum(wa + wb, 1e-30)
    tl.store(OUT + (t * HL + h) * DC + dc, o.to(tl.bfloat16))


def split_partials(q_lat, q_rope, cache, rope, sel, bound, cnt, scale, rank: int, hl: int):
    """GlmTP._split_attn's attention over this rank's share of the picks for every head (q
    [groups, T, hl, d], the gathered queries, read in place) -> (own_o fp32 [T, hl, DC], own_lse
    fp32 [T, hl], send uint8 [T, hl * DC * 2 + hl * 4]: the other rank's heads' partials)."""
    pm, pl, pacc = sparse_mla(q_lat, q_rope, cache, rope, sel, bound, scale, cnt=cnt, partial="raw")
    T, H, NS = pm.shape
    DC = q_lat.shape[-1]
    own_o = torch.empty((T, hl, DC), dtype=torch.float32, device=pm.device)
    own_l = torch.empty((T, hl), dtype=torch.float32, device=pm.device)
    row_bytes, off_l = hl * DC * 2 + hl * 4, hl * DC * 2
    send = torch.empty((T, row_bytes), dtype=torch.uint8, device=pm.device)
    _combine_kernel[(T, H)](pm, pl, pacc, own_o, own_l, send, H, hl, NS, rank * hl, row_bytes, off_l,
                            DC=DC, NSP=triton.next_power_of_2(NS), num_warps=4)
    return own_o, own_l, send


def merge_received(own_o, own_l, recv) -> torch.Tensor:
    """split_partials' own part and the other rank's send for this rank's heads -> o_lat bf16."""
    T, hl, DC = own_o.shape
    out = torch.empty((T, hl, DC), dtype=torch.bfloat16, device=own_o.device)
    _merge_kernel[(T, hl)](own_o, own_l, recv, out, hl, recv.stride(0), hl * DC * 2, DC=DC, num_warps=4)
    return out
