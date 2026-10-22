"""DSV41_TP_DRAFT_DENSE, two ranks: the merged draft FFN and the column-parallel main_proj against
the paths they replace, bytewise, on synthetic weights at the real shapes.

    GATE_SOURCE_ROOT=<snapshot on both nodes> GATE_IMAGE=<id> GATE_LOG_DIR=results/<dir> \\
    GATE_PROFILE=.env.ds41-mxfp4 bash tools/run_two_node_gate.sh test_draft_merged_tp.py

Reference (fastdecode._draft before the switch): moe_forward over the TP draft arena (its own
NCCL gathers) plus the replicated shared expert, joined by hc_post's ADD2. Candidate:
fastdecode.draft_moe_merged with this rank's shared-expert rows, joined by hc_post's split_w
read -- eager with and without the side stream, and captured into a CUDA graph and replayed on
new inputs. Also the sharded _shared_ffn fallback (OutputParallelWeight) and main_proj's
ColumnParallelWeight against the full fp8_linear. Exit status 1 on any mismatch on either rank.
"""
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch

from engine.dist import EPDistributed
from engine import comm
from engine.decode_lean import LeanOps
from engine.hc_ops import hc_post
from engine.tensor_parallel import ColumnParallelWeight, OutputParallelWeight, shard
import fp4_moe as K
import v41_ref as R
from fp8_linear import FP8Weight, concat_rows, fp8_linear
from engine.fastdecode import draft_moe_merged

SLOTS, T, TOPK, HC, LIMIT = 48, 5, 3, 4, 10.0


def fp8(gen, n, k):
    w = (torch.randn(n, k, generator=gen, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 126, ((n + 31) // 32, k // 32), generator=gen, device='cuda', dtype=torch.uint8)
    return FP8Weight(w, s)


def inputs(gen):
    y = (torch.randn(T, K.DIM, generator=gen, device='cuda') * 0.7).to(torch.bfloat16)
    slots = torch.stack([torch.randperm(SLOTS, generator=gen, device='cuda')[:TOPK] for _ in range(T)])
    wts = torch.rand(T, TOPK, generator=gen, device='cuda') + 0.1
    wts = wts / wts.sum(-1, keepdim=True) * 1.5
    residual = (torch.randn(T, HC, K.DIM, generator=gen, device='cuda')).to(torch.bfloat16)
    post = torch.rand(T, HC, generator=gen, device='cuda') * 2
    comb = torch.rand(T, HC, HC, generator=gen, device='cuda')
    return y, slots.to(torch.int32).contiguous(), wts.contiguous(), residual, post, comb


def main():
    ep = EPDistributed()
    ep.init('cuda')
    comm.init(torch.device('cuda'))
    rank, world = ep.rank, ep.world
    assert world == 2
    R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)   # serving: act_quant off (Model)
    lean = LeanOps(torch.device('cuda'), R.MM_TILE, R.HC_MM_TILE)
    # same seed on both ranks: replicated tensors agree, as they do in the engine
    gen = torch.Generator(device='cuda').manual_seed(20261022)
    arena = K.ExpertArena(SLOTS, 'cuda', tp_rank=rank, tp_world=world)
    agen = torch.Generator(device='cuda').manual_seed(7 + rank)
    for t in (arena.w1, arena.w3, arena.w2):
        t.copy_(torch.randint(0, 256, t.shape, generator=agen, device='cuda', dtype=torch.uint8))
    for t in (arena.s1, arena.s3, arena.s2):
        t.copy_(torch.randint(122, 128, t.shape, generator=agen, device='cuda', dtype=torch.uint8))
    w1, w3, w2 = fp8(gen, 2304, K.DIM), fp8(gen, 2304, K.DIM), fp8(gen, K.DIM, 2304)
    w13_full, _ = concat_rows(w1, w3)
    w1_r, w3_r = shard(w1, 0, rank, world), shard(w3, 0, rank, world)
    w13_r, _ = concat_rows(w1_r, w3_r)
    w2_r = OutputParallelWeight(shard(w2, 0, rank, world), world)
    side = torch.cuda.Stream()
    fails = []

    def reference(y, slots, wts, residual, post, comb):
        routed = K.moe_forward(y, slots, wts, arena, LIMIT, out_dtype=torch.float32).float()
        sh = R.qlinear(lean.swiglu(R.qlinear(y, w13_full), 2304, LIMIT), w2)
        return hc_post(routed, residual, post, comb, x2=sh.contiguous())

    def candidate(y, slots, wts, residual, post, comb, stream):
        out = draft_moe_merged(y, slots, wts, arena, w13_r, w1_r.N, w2_r.local, lean, LIMIT, stream)
        return hc_post(out, residual, post, comb, split_w=out.shape[-1])

    for trial in range(6):
        args = inputs(gen)
        ref = reference(*args)
        for name, stream in (('side', side), ('serial', None)):
            got = candidate(*args, stream)
            torch.cuda.synchronize()
            if not torch.equal(got, ref):
                fails.append(f'merged {name} trial {trial}: max |d| {float((got.float() - ref.float()).abs().max())}')
        # the non-merged fallback with sharded weights (_shared_ffn through OutputParallelWeight)
        y = args[0]
        sh_ref = R.qlinear(lean.swiglu(R.qlinear(y, w13_full), 2304, LIMIT), w2)
        sh_tp = R.qlinear(lean.swiglu(R.qlinear(y, w13_r), w1_r.N, LIMIT), w2_r)
        if not torch.equal(sh_ref, sh_tp):
            fails.append(f'sharded shared fallback trial {trial}')

    # captured, as the draft graphs are, then replayed on fresh inputs written in place
    static = [t.clone() for t in inputs(gen)]
    for _ in range(2):
        candidate(*static, side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = candidate(*static, side)
    for trial in range(4):
        fresh = inputs(gen)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(static_out, reference(*fresh)):
            fails.append(f'graph replay {trial}')

    main_proj = fp8(gen, K.DIM, 15360)
    n = main_proj.N // world
    cpw = ColumnParallelWeight(FP8Weight(main_proj.w[rank * n:(rank + 1) * n],
                                         main_proj.s[rank * n // 32:(rank + 1) * n // 32]), world)
    assert cpw.local.w.data_ptr() == main_proj.w.data_ptr() + rank * n * main_proj.K   # a view
    for rows in (1, 4, 6):
        x = (torch.randn(rows, 15360, generator=gen, device='cuda')).to(torch.bfloat16)
        if not torch.equal(R.qlinear(x, cpw), fp8_linear(x, main_proj)):
            fails.append(f'main_proj rows={rows}')
    torch.cuda.synchronize()
    print(f'rank {rank}: {"OK" if not fails else "FAIL"} {fails}', flush=True)
    every = ep.gather_objects(not fails) if hasattr(ep, 'gather_objects') else [not fails]
    ep.destroy()
    sys.exit(0 if all(every) else 1)


if __name__ == '__main__':
    main()
