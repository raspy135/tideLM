"""The v2 native decode expert kernels must write exactly what v1 (relaxed reduction) writes, and
the v3 down kernel (2 or 4 rows a warp) exactly what v2 writes.

    python3 tools/test_fp4_moe_v2.py        (GPU)

TP-rank-0 arena shapes (w1/w3 [1152, 5120], w2 [2560, 2304]) and the single-box shapes (w1/w3
[2304, 5120], w2 [5120, 2304]), random FP4 codes, scales 2^-9 .. 2^-1, activations spanning
1e-3 .. 3e2. Routing: T = 4 or 6 tokens x 6 experts over U distinct slots, with and without
null-slot pairs. Both variants of how v1's `p *= scale; acc += p` may have compiled are built;
the one that matches v1 bit for bit on every case is the one to ship (DSV41_FP4_CUDA_V2_FMA).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import fp4_moe as K  # noqa: E402
import fp4_moe_cuda as CUDA  # noqa: E402

TOPK = 6


def cases(arena, gen):
    inter, down_n = arena.w1.shape[1], arena.w2.shape[1]
    down_k = arena.w2.shape[2] * 2
    S = arena.w1.shape[0]
    for T in (4, 6):
        for U in (6, 11, 18, 24, T * TOPK):
            for null in (-1, S - 1):
                pool = torch.randperm(S - 1, generator=gen)[:U]
                slots = torch.stack([pool[torch.arange(t * TOPK, (t + 1) * TOPK) % U] for t in range(T)])
                if null >= 0:
                    slots[torch.rand(slots.shape, generator=gen) < 0.3] = null
                route = slots.clone()
                if null >= 0:  # moe_forward's split_decode_null keys
                    pair = torch.arange(T * TOPK).view_as(slots)
                    route = torch.where(slots == null, null + 1 + pair, slots)
                x = (torch.randn(T, K.DIM, generator=gen)
                     * torch.logspace(-3, 2.5, T)[:, None]).bfloat16().cuda()
                hf = (torch.randn(T * TOPK, down_k, generator=gen) * 2).bfloat16().cuda()
                wts = torch.rand(T, TOPK, generator=gen).cuda()
                bs, bp, br, nb = CUDA.build_routing_small(route.to(torch.int32).cuda().contiguous(), 16, TOPK)
                if null >= 0:
                    bs = torch.where(bs > null, torch.full_like(bs, null), bs)
                yield T, U, null, x, hf, wts, bs, bp, br, nb, inter, down_n, down_k


def run(v2, arena, T, x, hf, wts, bs, bp, br, nb, inter, down_n, down_k, null, rows=1):
    h = torch.full((T * TOPK, inter), 7.0, dtype=torch.bfloat16, device="cuda")
    parts = torch.full((T * TOPK, down_n), 7.0, device="cuda")
    CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, wts.reshape(-1), bs, br, 10.0, TOPK,
            inter, K.DIM, nb, null, v2=v2)
    CUDA.down(hf, arena.w2, arena.s2, parts, bs, bp, TOPK, down_n, down_k, T, nb, null, False, 1, v2=v2,
              rows=rows)
    return h, parts


def main():
    assert CUDA.RELAXED_REDUCE, "v2 reproduces the relaxed reduction only"
    results = {}
    for tp_world in (2, 1):
        arena = K.ExpertArena(64, "cuda", tp_rank=0, tp_world=tp_world)
        gen = torch.Generator().manual_seed(1)
        for t in (arena.w1, arena.w3, arena.w2):
            t.random_(0, 256)
        for t in (arena.s1, arena.s3, arena.s2):
            t.random_(118, 127)
        arena.w2[63].zero_()  # the null slot holds zeros
        for fma in (0, 1):
            CUDA.V2_FMA_SCALE, CUDA._LIB = fma, None
            ok = True
            for T, U, null, *rest in cases(arena, torch.Generator().manual_seed(2)):
                x, hf, wts, bs, bp, br, nb, inter, down_n, down_k = rest
                a = run(False, arena, T, x, hf, wts, bs, bp, br, nb, inter, down_n, down_k, null)
                b = run(True, arena, T, x, hf, wts, bs, bp, br, nb, inter, down_n, down_k, null)
                same = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for rows in (2, 4):  # v3 down (DSV41_FP4_CUDA_DOWN_ROWS) must write v2's bits
                    c = run(True, arena, T, x, hf, wts, bs, bp, br, nb, inter, down_n, down_k, null, rows)
                    if not torch.equal(b[1], c[1]):
                        print(f"  tp{tp_world} fma={fma} T={T} U={U} null={null}: v3 rows={rows} "
                              f"parts {int((b[1] != c[1]).sum())} elements differ from v2")
                        same = False
                if not same:
                    dh = int((a[0] != b[0]).sum()); dp = int((a[1] != b[1]).sum())
                    print(f"  tp{tp_world} fma={fma} T={T} U={U} null={null}: h {dh}, parts {dp} elements differ")
                ok &= same
            results[(tp_world, fma)] = ok
            print(f"tp_world={tp_world} FP4_V2_FMA_SCALE={fma}: {'bit-identical' if ok else 'DIFFERS'}")
    good = [f for f in (0, 1) if all(results[(w, f)] for w in (1, 2))]
    print("variants equal to v1 everywhere:", good)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
