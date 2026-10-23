"""EXL3 prefill MoE, stage by stage, at one rank's TP2 shapes -- no engine boot, no peer.

Loads one layer's real experts from this rank's pack (`DSV41_EXL3_PACK`, default
~/models/exl3-packs/exl3-experts-r0of2.bin) and routes a prefill-sized chunk with a skewed
(Zipf) synthetic distribution, then times each stage of `exl3_moe_cuda._run_pipeline` with CUDA
events.  The rank's intermediate all-gather is replaced by a local copy of its own half (same
shape, no wire), so the numbers are this rank's compute only; the engine's
`DSV41_PREFILL_MOE_TIMING` and the live bench measure the rest.

    python tools/bench_exl3_prefill.py [--layer 5] [--rows 2048] [--experts 384] [--zipf 0.8]
"""

import argparse
import ctypes
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_moe as X3  # noqa: E402
import exl3_moe_cuda as XC  # noqa: E402
import exl3_ref as R  # noqa: E402
import fp4_moe as F4  # noqa: E402

PACK = os.environ.get("DSV41_EXL3_PACK", os.path.expanduser("~/models/exl3-packs/exl3-experts-r0of2.bin"))


def routing(T, E, zipf, gen, K=6):
    """[T, K] distinct experts per row, P(e) ~ (rank+1)^-zipf over a fixed random permutation."""
    p = (torch.arange(E, dtype=torch.float64) + 1).pow(-zipf)
    p = p[torch.randperm(E, generator=gen)]
    return torch.multinomial(p.expand(T, E).float(), K, replacement=False, generator=gen).to(torch.int32)


def group(slots, block_m):
    """moe_forward_prefill's grouping, verbatim."""
    T, K = slots.shape
    valid = slots >= 0
    uniq, inv = torch.unique(slots[valid], return_inverse=True)
    compact = torch.full_like(slots, -1, dtype=torch.int32)
    compact[valid] = inv.to(torch.int32)
    block_slot, block_pair, NB = F4.build_routing(compact, int(uniq.numel()), block_m, precount=True)
    block_slot = torch.where(block_slot >= 0, uniq.to(torch.int32)[block_slot.clamp_min(0)], block_slot)
    pair = block_pair.view(NB, block_m)
    members = torch.where(pair >= 0, (pair // K) * 32 + (pair % K), pair).to(torch.int32).contiguous()
    uids = block_slot.to(torch.int32).contiguous()
    ucount = torch.full((1,), NB, dtype=torch.int32, device=slots.device)
    return uids, ucount, members, NB


def stages(x, slots, weights, arena, uids, ucount, members, maxm, nexp_max, nt=8, warps=4, pf=2, tile=None,
           mode=None):
    """_run_pipeline with a CUDA event between stages; returns (out, {stage: ms}). tile=None is
    the grouped (decode) kernel, else (nt, warps, pf) of exl3m_prefill. mode (smem tiles only):
    1/2 = the probes, -S = cp.async with S stages (exl3_moe_cuda._grouped_or_tile)."""
    T, K = slots.shape
    P = T * K
    dim, inter, down_n = arena.shapes["suh1"][0], arena.inter, arena.down_n
    world = arena.tp_world
    dev = x.device
    p = XC._ptrs(arena)
    lib = XC._lib()
    st = XC._stream()
    V = ctypes.c_void_p
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
    ev[0].record()
    xh0 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    xh1 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    z = torch.empty((2, P, inter), dtype=torch.float32, device=dev)
    xd = torch.empty((P, inter), dtype=torch.float16, device=dev)
    zd = torch.empty((1, P, down_n), dtype=torch.float32, device=dev)
    local = torch.empty((T, down_n), dtype=torch.float32, device=dev)
    pick = slots.reshape(-1).to(torch.int32).contiguous()
    lib.exl3m_rot_in(V(x.data_ptr()), int(x.stride(0)), V(pick.data_ptr()), V(arena.suh1.data_ptr()),
                     V(arena.suh3.data_ptr()), V(xh0.data_ptr()), V(xh1.data_ptr()),
                     int(T), int(dim), int(K), int(arena.slots), 1, st)
    ev[1].record()
    XC._grouped_or_tile(lib, tile, V(xh0.data_ptr()), V(xh1.data_ptr()), V(p["t1p"].data_ptr()),
                        V(p["t3p"].data_ptr()), V(p["k2"].data_ptr()), V(p["k2"].data_ptr()), V(uids.data_ptr()),
                        V(ucount.data_ptr()), V(members.data_ptr()), V(z.data_ptr()), dim, inter, P, maxm, K,
                        nexp_max, 2, nt, warps, pf, st, mode=mode)
    ev[2].record()
    suh2_stride = int(arena.shapes["suh2"][0])
    suh2_base = arena.suh2.data_ptr() + arena.tp_rank * inter * 2
    lib.exl3m_gateup(V(z.data_ptr()), V(pick.data_ptr()), V(arena.svh1.data_ptr()), V(arena.svh3.data_ptr()),
                     V(suh2_base), V(xd.data_ptr()), int(T), int(P), int(inter), 1, int(K), int(arena.slots),
                     suh2_stride, 10.0, 1, st)
    ev[3].record()
    # stand-in for the all-gather: this rank's half twice (same shape and bytes the kernel reads)
    xd = torch.cat([xd] * world, dim=1).contiguous()
    ev[4].record()
    XC._grouped_or_tile(lib, tile, V(xd.data_ptr()), V(xd.data_ptr()), V(p["t2p"].data_ptr()),
                        V(p["t2p"].data_ptr()), V(p["k2d"].data_ptr()), V(p["k2d"].data_ptr()), V(uids.data_ptr()),
                        V(ucount.data_ptr()), V(members.data_ptr()), V(zd.data_ptr()), world * inter, down_n, P,
                        maxm, K, nexp_max, 1, nt, warps, pf, st, mode=mode)
    ev[5].record()
    w = weights.reshape(-1).float().contiguous()
    lib.exl3m_down_combine(V(zd.data_ptr()), V(pick.data_ptr()), V(arena.svh2.data_ptr()),
                           V(w.data_ptr()), V(local.data_ptr()), int(T), int(P), int(down_n), 1, int(K),
                           int(arena.slots), st)
    ev[6].record()
    torch.cuda.synchronize()
    names = ("rot_in", "gate_up", "gateup_epi", "gather_standin", "down", "down_combine")
    return local, {n: ev[i].elapsed_time(ev[i + 1]) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--rows", type=int, nargs="+", default=[2048])
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--zipf", type=float, default=0.8)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tiles", default="grouped,4x4x2,2x4x2,4x2x2,2x8x2,4x4x3,bm32:4x4x2,bm32:8x4x2",
                    help="kernels to time: 'grouped' (decode kernel, BM 64), [bm32:]NTxWARPSxPF (exl3m_prefill) "
                         "or smem:WxMTWxNTBxPF (exl3m_prefill_smem; BM = W*MTW*16)")
    ap.add_argument("--topk", type=int, default=6, help="picks a row (DeepSeek 6, GLM-5.3 8)")
    a = ap.parse_args()
    pack = R.PackReader(PACK)
    if "dims" in pack.header:                        # a GLM pack: its own hidden / intermediate
        X3.DIM, X3.INTER = pack.header["dims"]
        XC.DIM, XC.INTER = pack.header["dims"]
    n_layer = sum(1 for (L, _) in pack.records if L == a.layer)
    if a.experts > n_layer:
        print(f"layer {a.layer} has {n_layer} experts in the pack; loading those")
        a.experts = n_layer
    bits = pack.bits_map[a.layer]
    arena = X3.Exl3Arena(a.experts, device="cuda", tp_rank=pack.rank, tp_world=pack.world, bits=3.0)
    t0 = time.perf_counter()
    for e in range(a.experts):
        arena.load_slot(e, pack.read_expert(a.layer, e), bits)
    torch.cuda.synchronize()
    print(f"pack {os.path.basename(PACK)} rank {pack.rank}/{pack.world}, layer {a.layer} ({bits:g}-bit), "
          f"{a.experts} experts loaded in {time.perf_counter() - t0:.1f} s, {arena.bytes_per_slot / 1e6:.2f} MB/slot")
    gen = torch.Generator().manual_seed(0)
    for T in a.rows:
        slots = routing(T, a.experts, a.zipf, gen, a.topk).cuda()
        weights = torch.rand(T, a.topk, generator=gen).cuda()
        x = (torch.randn(T, X3.DIM, generator=gen) * 0.1).to(torch.bfloat16).cuda()
        counts = torch.bincount(slots.reshape(-1).long(), minlength=a.experts)
        print(f"\nT={T}  pairs {T * a.topk}  experts touched {int((counts > 0).sum())}  max/expert {int(counts.max())}  "
              f"mean rows/expert {float(counts[counts > 0].float().mean()):.1f}")
        ref = None
        for spec in a.tiles.split(","):
            name = spec.split(":")[-1]
            mode = None
            if spec.startswith(("smem:", "ca:", "nomma:", "notrellis:")):
                tile = tuple(int(v) for v in name.split("x"))
                bm = tile[0] * tile[1] * 16
                kind = spec.split(":")[0]
                if kind == "ca":          # WxMTWxNTBxS: cp.async, S raw stages (bit-identical)
                    mode = -tile[3]
                elif kind == "nomma":     # probe: trellis loads + decode only (garbage output)
                    mode = 1
                elif kind == "notrellis":  # probe: A loads + mma on a constant tile (garbage output)
                    mode = 2
            else:
                bm = int(spec.split(":")[0][2:]) if spec.startswith("bm") else a.block_m
                tile = None if name == "grouped" else tuple(int(v) for v in name.split("x"))
            tg = []
            for _ in range(a.iters):
                torch.cuda.synchronize()
                t = time.perf_counter()
                uids, ucount, members, NB = group(slots, bm)
                torch.cuda.synchronize()
                tg.append((time.perf_counter() - t) * 1e3)
            try:
                runs = [stages(x, slots, weights, arena, uids, ucount, members, bm, NB, tile=tile, mode=mode)
                        for _ in range(a.iters + 1)]
            except Exception as ex:  # noqa: BLE001
                print(f"  {spec}: {type(ex).__name__} {ex}")
                continue
            out = runs[-1][0]
            runs = [r[1] for r in runs[1:]]
            med = {k: sorted(r[k] for r in runs)[len(runs) // 2] for k in runs[0]}
            total = sum(v for k, v in med.items() if k != "gather_standin") + sorted(tg)[len(tg) // 2]
            if ref is None:
                ref = out
            rel = float((out - ref).norm() / ref.norm())
            same = "bitwise" if torch.equal(out, ref) else f"{rel:.1e}"
            print(f"  {spec:14s} blocks {NB:4d}  group {sorted(tg)[len(tg) // 2]:6.2f}  "
                  + "  ".join(f"{k} {v:7.2f}" for k, v in med.items() if k != "gather_standin")
                  + f"  | total {total:7.2f} ms  vs first {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
