"""The dreaming FP8 tier's kernels (tools/fp4_moe_cuda.cu dream_fp8_up/down) against torch.

    python3 tools/test_dream_fp8.py        (GPU)

Random e4m3 experts with UE8M0 scales per 32 weights (engine/dream_overlay.py's layout), for one
box and for both ranks of TP2. Checked:
  * rows of pairs whose expert is not in the tier are left exactly as they were;
  * tier rows match a torch reference with the FP4 kernels' rounding points (BF16 projections,
    clamp, SiLU, routing weight, BF16) to within fp32 summation order;
  * the two TP2 ranks' rows put together are the one-box rows, bit for bit;
  * two runs give the same bits (each rank's rows must not depend on scheduling);
  * moe_forward(dream=...) on a one-box arena equals a torch reference that computes the tier's
    experts from FP8 and every other expert from its MXFP4 slot.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fp4_moe as K  # noqa: E402
import fp4_moe_cuda as CUDA  # noqa: E402
from engine import dream_overlay as D  # noqa: E402

TOPK, E, LIMIT = 6, 384, 10.0
SHAPES = {"w1": (2304, 5120), "w2": (5120, 2304), "w3": (2304, 5120)}


def random_fp8_expert(gen):
    t = {}
    for m, (n, k) in SHAPES.items():
        codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=gen)
        codes[(codes & 0x7f) == 0x7f] = 0x38   # e4m3 NaN codes -> 1.0
        t[f"{m}.weight"] = codes
        t[f"{m}.scale"] = torch.randint(127 - 12, 127 - 6, (n, k // 32), dtype=torch.uint8, generator=gen)
    return t


def dequant(t, m):
    w = t[f"{m}.weight"].cuda().view(torch.float8_e4m3fn).float()
    return (w * torch.exp2(t[f"{m}.scale"].cuda().float() - 127).repeat_interleave(32, 1)).bfloat16()


def ref_up(x_row, t, rw):
    g = (x_row.float() @ dequant(t, "w1").float().T).bfloat16().float().clamp(max=LIMIT)
    u = (x_row.float() @ dequant(t, "w3").float().T).bfloat16().float().clamp(-LIMIT, LIMIT)
    return (F.silu(g) * u * rw).bfloat16()


def ref_down(h_row, t):
    return (h_row.float() @ dequant(t, "w2").float().T).bfloat16().float()


def close(a, b, what):
    err = (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-30)
    assert err < 2e-2, f"{what}: relative error {err:.3g}"


def main():
    gen = torch.Generator().manual_seed(0)
    tier_ids = [3, 77, 200, 383]
    experts = {e: random_fp8_expert(gen) for e in tier_ids}
    items = list(experts.items())
    one = D.fp8_layer(items, "cuda", 0, 1, E)
    tp = [D.fp8_layer(items, "cuda", r, 2, E) for r in range(2)]
    for T in (1, 4, 6):
        P = T * TOPK
        idx = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(T)])
        idx.view(-1)[torch.randperm(P, generator=gen)[:max(2, P // 3)]] = torch.tensor(tier_ids)[
            torch.randint(0, 4, (max(2, P // 3),), generator=gen)]
        for t_ in range(T):   # a token's top-k is distinct
            seen = set()
            for k in range(TOPK):
                while int(idx[t_, k]) in seen:
                    idx[t_, k] = (idx[t_, k] + 1) % E
                seen.add(int(idx[t_, k]))
        idx = idx.cuda()
        flat = idx.view(-1).contiguous()
        x = (torch.randn(T, 5120, generator=gen) * 0.5).bfloat16().cuda()
        rw = torch.rand(T, TOPK, generator=gen).cuda()
        hfull = (torch.randn(P, 2304, generator=gen) * 0.5).bfloat16().cuda()
        in_tier = torch.isin(flat, torch.tensor(tier_ids, device="cuda"))

        def up(tier, n):
            h = torch.full((P, n), 7.0, dtype=torch.bfloat16, device="cuda")
            CUDA.dream_fp8_up(x, flat, tier, h, rw.view(-1), LIMIT, TOPK)
            return h

        def down(tier, n):
            parts = torch.full((P, n), 7.0, device="cuda")
            CUDA.dream_fp8_down(hfull, flat, tier, parts, TOPK, T)
            return parts

        h1, p1 = up(one, 2304), down(one, 5120)
        assert torch.equal(h1, up(one, 2304)) and torch.equal(p1, down(one, 5120)), "not deterministic"
        assert (h1[~in_tier] == 7).all() and (p1.view(TOPK, T, -1).transpose(0, 1).reshape(P, -1)[~in_tier] == 7).all(), \
            "rows outside the tier were written"
        for p in torch.nonzero(in_tier).view(-1).tolist():
            t, k = divmod(p, TOPK)
            e = experts[int(flat[p])]
            close(h1[p], ref_up(x[t], e, rw[t, k]), f"T={T} up pair {p}")
            close(p1[k * T + t], ref_down(hfull[p], e), f"T={T} down pair {p}")
        h_tp = torch.cat([up(tp[r], 1152) for r in range(2)], dim=1)
        p_tp = torch.cat([down(tp[r], 2560) for r in range(2)], dim=1)
        assert torch.equal(h_tp, h1) and torch.equal(p_tp, p1), "TP2 ranks differ from one box"
        print(f"T={T}: {int(in_tier.sum())} of {P} pairs in the tier: kernels ok, TP2 == one box")
    moe_forward_check(gen, experts, one)
    print("ok")


def moe_forward_check(gen, experts, tier):
    """moe_forward(dream=...) against torch: tier experts from FP8, the rest from their MXFP4 slot."""
    arena = K.ExpertArena(8, "cuda")
    for s in range(8):
        arena.w1[s] = torch.randint(0, 256, arena.w1[s].shape, dtype=torch.uint8, generator=gen).cuda()
        arena.w3[s] = torch.randint(0, 256, arena.w3[s].shape, dtype=torch.uint8, generator=gen).cuda()
        arena.w2[s] = torch.randint(0, 256, arena.w2[s].shape, dtype=torch.uint8, generator=gen).cuda()
        for t in (arena.s1, arena.s3, arena.s2):
            t[s] = torch.randint(127 - 10, 127 - 6, t[s].shape, dtype=torch.uint8, generator=gen).cuda()
    T = 4
    ids = torch.tensor([[3, 10, 11, 12, 13, 14], [77, 10, 15, 16, 17, 18],
                        [200, 383, 11, 12, 15, 16], [20, 21, 22, 23, 24, 25]], device="cuda")
    slot_of = {int(e): i % 8 for i, e in enumerate(torch.unique(ids).tolist())}
    slots = torch.tensor([[slot_of[int(e)] for e in row] for row in ids.tolist()], dtype=torch.int32, device="cuda")
    x = (torch.randn(T, 5120, generator=gen) * 0.5).bfloat16().cuda()
    w = torch.rand(T, TOPK, generator=gen).cuda()
    out = K.moe_forward(x, slots, w, arena, LIMIT, out_dtype=torch.float32, dream=(tier, ids))
    ref = torch.zeros(T, 5120, device="cuda")
    for t in range(T):
        for k in range(TOPK):
            e = int(ids[t, k])
            if e in experts:
                h = ref_up(x[t], experts[e], w[t, k])
                y = ref_down(h, experts[e])
            else:
                w1, w2, w3 = arena.dequant_slot(int(slots[t, k]))
                g = (x[t].float() @ w1.float().T).bfloat16().float().clamp(max=LIMIT)
                u = (x[t].float() @ w3.float().T).bfloat16().float().clamp(-LIMIT, LIMIT)
                h = (F.silu(g) * u * w[t, k]).bfloat16()
                y = (h.float() @ w2.float().T).bfloat16().float()
            ref[t] += y
    close(out, ref, "moe_forward with the tier")
    print("moe_forward(dream=...) matches the torch reference")


if __name__ == "__main__":
    main()
