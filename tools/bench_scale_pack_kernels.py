"""Decode MoE kernels, native vs packed scales: real 1017 experts, rank-0 shapes, one GPU.

  python3 tools/bench_scale_pack_kernels.py [--model-dir D] [--pack P]

48 experts (layers 4-11, experts 0-5) in a native and a packed arena. Each call routes T tokens
to 6 T distinct slots, and one graph replay walks all 48 slots, so every call reads L2-cold
weights as decode does. Up (v2) and down (v2 and the two-row v3) outputs are compared bytewise;
times are CUDA graph replays, median of 7, per call. Aborts below 5 GB MemAvailable.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', default='/models/DeepSeek-V4.1-Flash-1017')
ap.add_argument('--pack', default='/models/dsv41-scale-packs/ds41-scales-r{rank}of2.bin')
ap.add_argument('--native-only', action='store_true', help='time the native arena only (older kernels)')
args = ap.parse_args()
avail = int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) / 1e6
if avail < 5:
    sys.exit(f'MemAvailable {avail:.1f} GB: abort')

import fp4_moe as moe                 # noqa: E402
import fp4_moe_cuda as cuda           # noqa: E402
from engine.scale_pack import ScalePack  # noqa: E402

directory = Path(args.model_dir)
mapping = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
headers = {}


def read(name):
    path = directory / mapping[name]
    if path not in headers:
        with path.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            headers[path] = 8 + length, json.loads(handle.read(length))
    base, header = headers[path]
    first, last = header[name]['data_offsets']
    with path.open('rb') as handle:
        handle.seek(base + first)
        data = bytearray(handle.read(last - first))
    return torch.frombuffer(data, dtype=torch.uint8).reshape(header[name]['shape']).pin_memory()


def gtime(fn, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(7):
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / reps * 1000)
    return sorted(ts)[3]


E, TOPK = 48, 6
native = moe.ExpertArena(E, tp_rank=0, tp_world=2)
if not args.native_only:
    pack = ScalePack(args.pack.format(rank=0), directory, 0, 2)
    packed = moe.ExpertArena(E, tp_rank=0, tp_world=2, scale_pack=pack)
keys = [(layer, expert) for layer in range(4, 12) for expert in range(6)]
for slot, (layer, expert) in enumerate(keys):
    prefix = f'layers.{layer}.ffn.experts.{expert}.'
    tensors = [read(prefix + n) for n in ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale', 'w3.weight', 'w3.scale')]
    native.load_slot(slot, *tensors)
    if not args.native_only:
        packed.load_slot(slot, *tensors, scale_key=(layer, expert))
if args.native_only:
    pack = None
else:
    spec = pack.spec
    esc = np.frombuffer(packed.s1[:, spec['escaped']:spec['escaped'] + spec['blocks'] * 8].cpu().numpy().tobytes(), np.uint8)
    print(f"slot bytes native {native.bytes_per_slot} packed {packed.bytes_per_slot}; up-plane scales escaped "
          f"{np.unpackbits(esc).mean() * 100:.2f}% (these 48 experts; capacity {spec['capacity']} of {spec['size']})", flush=True)

gen = torch.Generator(device='cuda').manual_seed(5)
for T in (1, 2, 4):
    calls = E // (TOPK * T)
    perm = torch.randperm(E, device='cuda', generator=gen).int()
    work = []
    for c in range(calls):
        slots = perm[c * TOPK * T:(c + 1) * TOPK * T].reshape(T, TOPK).contiguous()
        bs, bp, br, count = cuda.build_routing_small(slots, 16, TOPK)
        x = torch.randn(T, moe.DIM, device='cuda', generator=gen).to(torch.bfloat16)
        route = torch.rand(T, TOPK, device='cuda', generator=gen).reshape(-1).contiguous()
        work.append((x, route, bs, bp, br, count))
    pairs = T * TOPK
    res, same = {}, True
    for name, arena in (('native', native),) + ((('packed', packed),) if pack is not None else ()):
        hs = [torch.empty((pairs, arena.inter), dtype=torch.bfloat16, device='cuda') for _ in work]

        def up():
            for (x, route, bs, bp, br, count), h in zip(work, hs):
                cuda.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, route, bs, br, 10., TOPK, arena.inter,
                        moe.DIM, count, -1, v2=True)
        res[f'up {name}'] = gtime(up) / calls
        full = [torch.cat([h, h], -1).contiguous() for h in hs]
        for rows in (1, 2):
            outs = [torch.empty((pairs, arena.w2.shape[1]), dtype=torch.float32, device='cuda') for _ in work]

            def down():
                for (x, route, bs, bp, br, count), f, o in zip(work, full, outs):
                    cuda.down(f, arena.w2, arena.s2, o, bs, bp, TOPK, o.shape[1], moe.INTER, T, count, -1,
                              False, 1, v2=True, rows=rows)
            res[f'down{rows} {name}'] = gtime(down) / calls
            res.setdefault(f'_out{rows}', []).append([o.clone() for o in outs])
        res.setdefault('_h', []).append([h.clone() for h in hs])
    for key in ('_h', '_out1', '_out2'):
        arms = res.pop(key)
        if len(arms) == 2:
            same &= all(torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(*arms))
    line = '  '.join(f'{k} {v:.1f}' for k, v in res.items())
    deltas = '' if pack is None else '  '.join(
        f'{k}: {res[f"{k} packed"] / res[f"{k} native"] * 100 - 100:+.1f}%' for k in ('up', 'down1', 'down2'))
    print(f'T={T} ({pairs} pairs): bytes_equal={same}  {line}  (us a call)  {deltas}', flush=True)
if pack is not None:
    os.close(pack.fd)
