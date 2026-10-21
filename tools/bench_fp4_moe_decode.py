"""Achieved bandwidth of the native decode expert kernels at TP2 shapes, against a plain read.

    python3 tools/bench_fp4_moe_decode.py [--slots 256] [--calls 40]

Single GPU, serving stopped. A TP-rank-0 ExpertArena (output layout: w1/w3 [1152, 5120] and w2
[2560, 2304] per slot, FP4 + UE8M0) is filled with random codes. Each call routes T=6 tokens x 6
experts over U distinct slots, drawn fresh per call from the arena so consecutive calls do not
share L2. `--calls` calls are captured into one CUDA graph and replayed.

Bytes per call are what the kernel must read at least once: U x (w1 + s1 + w3 + s3) for up,
U x (w2 + s2) for down. The reference is a Triton kernel that reads exactly those bytes with
16-byte loads and nothing else -- the attainable ceiling for this access pattern.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import fp4_moe as K  # noqa: E402
import fp4_moe_cuda as CUDA  # noqa: E402

T, TOPK = 6, 6


@triton.jit
def _touch(BASE, SLOTS, OUT, WORDS, BLOCK: tl.constexpr):
    """Read WORDS int64 words of each listed slot (contiguous, 16-byte vector loads); one partial
    per program so nothing is optimized away."""
    s = tl.program_id(0)
    c = tl.program_id(1)
    slot = tl.load(SLOTS + s).to(tl.int64)
    off = c * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(BASE + slot * WORDS + off, mask=off < WORDS, other=0)
    tl.store(OUT + s * tl.num_programs(1) + c, tl.sum(v, 0))


def touch(tensor, slots, out):
    words = tensor[0].numel() // 8
    blk = 1024
    _touch[(slots.numel(), triton.cdiv(words, blk))](tensor.view(torch.int64), slots, out, words,
                                                      BLOCK=blk, num_warps=8)


def routing(U, gen, n_slots):
    pool = torch.randperm(n_slots, generator=gen)[:U]
    slots = torch.stack([pool[torch.arange(t * TOPK, (t + 1) * TOPK) % U] for t in range(T)])
    return slots.to(torch.int32).cuda().contiguous()


def graph_ms(fn, calls):
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 5 / calls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--u", default="8,14,20,26,32,36")
    ap.add_argument("--v2", action="store_true", help="the v2 kernels (DSV41_FP4_CUDA_V2) instead of v1")
    ap.add_argument("--down-rows", type=int, default=1,
                    help="with --v2: rows a warp for down (DSV41_FP4_CUDA_DOWN_ROWS; 2 or 4 = v3)")
    args = ap.parse_args()
    os.environ.setdefault("DSV41_TP_EXPERT_LAYOUT", "output")
    arena = K.ExpertArena(args.slots, "cuda", tp_rank=0, tp_world=2)
    gen = torch.Generator().manual_seed(0)
    for t in (arena.w1, arena.w3, arena.w2):
        t.random_(0, 256)
    for t in (arena.s1, arena.s3, arena.s2):
        t.random_(118, 127)
    inter, down_n = arena.w1.shape[1], arena.w2.shape[1]
    x = (torch.randn(T, K.DIM, generator=gen) * 0.5).bfloat16().cuda()
    wts = torch.rand(T, TOPK, generator=gen).cuda()
    h = torch.empty(T * TOPK, inter, dtype=torch.bfloat16, device="cuda")
    hf = (torch.randn(T * TOPK, 2 * inter, generator=gen) * 0.5).bfloat16().cuda()
    parts = torch.empty(T * TOPK, down_n, dtype=torch.float32, device="cuda")
    up_bytes = sum(t[0].numel() for t in (arena.w1, arena.s1, arena.w3, arena.s3))
    down_bytes = sum(t[0].numel() for t in (arena.w2, arena.s2))
    print(f"kernels: {'v2' if args.v2 else 'v1'}, down rows {args.down_rows}; per slot: up {up_bytes / 1e6:.2f} MB, down {down_bytes / 1e6:.2f} MB; L2-cold calls: {args.calls}")
    print(f"{'U':>3} | {'up us':>7} {'GB/s':>6} {'read GB/s':>9} | {'down us':>7} {'GB/s':>6} {'read GB/s':>9}")
    for U in (int(u) for u in args.u.split(",")):
        routes = []
        for _ in range(args.calls):
            s = routing(U, gen, args.slots)
            routes.append((s,) + CUDA.build_routing_small(s, 16, TOPK)[:4])
        out = torch.empty(U * 4096, dtype=torch.int64, device="cuda")

        def up(i):
            _, bs, bp, br, nb = routes[i]
            CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, wts.reshape(-1), bs, br, 10.0,
                    TOPK, inter, K.DIM, nb, -1, v2=args.v2)

        def down(i):
            _, bs, bp, br, nb = routes[i]
            CUDA.down(hf, arena.w2, arena.s2, parts, bs, bp, TOPK, down_n, 2 * inter, T, nb, -1, False, 1,
                      v2=args.v2, rows=args.down_rows)

        def uniq(i):
            return torch.unique(routes[i][0]).to(torch.int32)

        uniqs = [uniq(i) for i in range(args.calls)]

        def read_up(i):
            for t in (arena.w1, arena.s1, arena.w3, arena.s3):
                touch(t, uniqs[i], out)

        def read_down(i):
            for t in (arena.w2, arena.s2):
                touch(t, uniqs[i], out)

        t_up, t_down = graph_ms(up, args.calls), graph_ms(down, args.calls)
        r_up, r_down = graph_ms(read_up, args.calls), graph_ms(read_down, args.calls)
        bu, bd = U * up_bytes, U * down_bytes
        print(f"{U:>3} | {t_up * 1e3:7.1f} {bu / t_up / 1e6:6.0f} {bu / r_up / 1e6:9.0f} | "
              f"{t_down * 1e3:7.1f} {bd / t_down / 1e6:6.0f} {bd / r_down / 1e6:9.0f}")


if __name__ == "__main__":
    main()
