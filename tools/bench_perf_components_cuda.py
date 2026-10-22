"""Synthetic component timings; these do not establish model quality or throughput."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
import torch
import triton
from engine.attention_packed import attention_packed
from engine.decode_lean import LeanOps
from engine.hc_ops import hc_post, hc_pre_rmsnorm
from engine.hc_post_pre import post_pre
from engine.index_candidates import compact_mask, scores, select
from engine.packed_kv import write
from decode_attn_staged import attention

UNROLL = 10  # Amortize Python graph-launch overhead for the smallest kernels.


def capture(fn):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(UNROLL):
            output = fn()
    return graph, output


def measure(fn, repetitions):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repetitions * UNROLL)


def compare(off, on, repetitions):
    old, old_output = capture(off)
    new, new_output = capture(on)
    rows = []
    for _ in range(3):
        for enabled in (False, True, True, False):
            rows.append(dict(enabled=enabled, us=measure((new if enabled else old).replay, repetitions)))
    a = statistics.median(r['us'] for r in rows if not r['enabled'])
    b = statistics.median(r['us'] for r in rows if r['enabled'])
    return dict(off_us=a, on_us=b, time_saved_percent=100 * (a - b) / a, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(1017)
    dev = 'cuda'
    results = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                   graph_unroll=UNROLL, graph_repetitions=args.repetitions, workloads=[])
    for t in (4, 6):
        x = torch.randn(t, 5120, device=dev).bfloat16()
        residual = torch.randn(t, 4, 5120, device=dev).bfloat16()
        post, pre = (torch.rand(t, 4, device=dev) for _ in range(2))
        comb = torch.rand(t, 4, 4, device=dev)
        weight = torch.rand(5120, device=dev)
        result = compare(lambda: hc_pre_rmsnorm(hc_post(x, residual, post, comb), pre, weight, 1e-20,
                                              direct_store=True),
                         lambda: post_pre(x, residual, post, comb, pre, weight, 1e-20), args.repetitions)
        results['workloads'].append(dict(component='hc_post_pre', t=t, **result))
        lean = LeanOps(dev, 16, 32)
        q = torch.randn(t, 32, 512, device=dev).bfloat16().float()
        ring = torch.randn(160, 512, device=dev).bfloat16()
        slots = torch.randint(0, 160, (t, 128), device=dev)
        cache = torch.zeros(4096, 36, dtype=torch.int64, device=dev)
        write(cache, torch.randn(4096, 512, device=dev).bfloat16(), 0)
        ids = torch.randint(0, 4096, (t, 1024), device=dev)
        mask = torch.ones(t, 1152, dtype=torch.bool, device=dev)
        sink = torch.randn(32, device=dev)
        def staged():
            keys = lean.keys_f32(ring, slots, cache, ids, dtype=torch.bfloat16)
            return attention(q, keys, mask, sink, 512 ** -.5, lean)
        result = compare(staged, lambda: attention_packed(q, ring, slots, cache, ids, mask, sink, 512 ** -.5),
                         args.repetitions)
        results['workloads'].append(dict(component='attention_direct', t=t, heads=32, keys=1152, **result))
        for n in (32768, 65536):
            query = torch.randn(t, 32, 128, device=dev).bfloat16()
            keys = torch.randn(n, 128, device=dev).bfloat16()
            weights = torch.rand(t, 32, device=dev) / 64
            mask = (torch.arange(n, device=dev) // 8 % (n // 16384) == 0).expand(t, -1).contiguous()
            candidates = compact_mask(mask, 8, 16384)
            lengths = torch.full((t,), n, dtype=torch.long, device=dev)
            def dense():
                dots = torch.einsum('thd,nd->thn', query, keys)
                values = (dots.float().relu_() * weights[:, :, None]).sum(1)
                values = values.masked_fill(~mask, -float('inf'))
                return values.topk(1024, sorted=False).indices.sort(-1).values
            result = compare(dense, lambda: select(scores(query, keys, weights, candidates, lengths),
                                                   candidates, 1024), args.repetitions)
            source, source_output = capture(lambda: compact_mask(mask, 8, 16384))
            overhead = statistics.median(measure(source.replay, args.repetitions) for _ in range(5))
            results['workloads'].append(dict(component='index_candidates', t=t, keys=n,
                candidates=16384, source_compaction_us=overhead,
                four_later_indexers_net_us=4 * (result['off_us'] - result['on_us']) - overhead, **result))
        print(json.dumps(results['workloads'][-4:], indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
