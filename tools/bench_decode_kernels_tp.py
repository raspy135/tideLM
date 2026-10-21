"""Kernel launches per decode step, with an output-identity check, for A/B-ing launch consolidation.

Same frozen-ranking / prefix-off engine setup as the timeline bench. Three greedy workloads are
generated twice (the first pass warms graphs and is discarded); each reports tok/s, ms/step,
acceptance and a SHA-256 of its output token ids, so two code versions that must be bit-identical
can be compared by hash. Then the python workload is profiled for --bursts bursts after
--warmup-tokens, and every CUDA kernel is counted per verify+draft step and grouped by family.

    GATE_SOURCE_ROOT=<snapshot on both nodes> GATE_IMAGE=<id> GATE_LOG_DIR=results/<dir> \\
    bash tools/run_two_node_gate.sh bench_decode_kernels_tp.py --out /app/results/<dir>

Output: <out>/kernels-rank<r>.json. The prompts are the public synthetic ones of
bench_decode_block_tp.py, so their token ids are recorded too (to locate a first divergence).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
from torch.profiler import profile, ProfilerActivity

FAMILIES = (
    (r'_moe_|cb3|fp4_moe|exl3::', 'routed experts'),
    (r'_fp8_|_fp4_linear|_fp4_grouped', 'dense projections'),
    (r'nccl', 'nccl'),
    (r'bf16_s161616gemm|nvjet', 'bf16 gemm'),
    (r'sgemm|gemmSN|splitK', 'fp32 gemm'),
)


def family(name):
    for pat, fam in FAMILIES:
        if re.search(pat, name):
            return fam
    return 'small'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--bursts', type=int, default=8)
    ap.add_argument('--warmup-tokens', type=int, default=64)
    args = ap.parse_args()
    # The gate only creates GATE_LOG_DIR on rank 0; rank 1's results mount does not have it, and
    # without this the final kernels-seq write fails after every measurement has already run.
    os.makedirs(args.out, exist_ok=True)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, keep_free_gb=6,
                   # DSV41_HOT_ASYNC takes its ring out of these: the two-node profiles set 136
                   transient_slots=int(os.environ.get('DSV41_BENCH_TRANSIENT_SLOTS', 16)),
                   expert_format=os.environ.get('DSV41_BENCH_EXPERT_FORMAT', 'fp4'))
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    greedy = [(n, p) for n, p, t in WORKLOADS if t == 0 and n in ('html', 'python', 'explain')]
    ids = {}
    for name, prompt in greedy:
        _, ids[name], _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                               enc, tok, False, 75, e)
    report = dict(config=e.config(), max_tokens=args.max_tokens, runs={})
    fd = e.fast
    for p in range(2):
        for name, _ in greedy:
            if fd.rs_uniq is not None:
                fd.route_stats_reset()  # DSV41_ROUTE_STATS=1: distinct routed experts per layer
            out = []
            for burst in e.generate(ids[name], max_tokens=args.max_tokens, temperature=0,
                                    seed=42, stop_token_ids={eos}):
                out.extend(burst)
            st = dict(e.last_stats)
            item = dict(sha256=hashlib.sha256(json.dumps(out).encode()).hexdigest(), tokens=out,
                        completion_tokens=len(out), decode_tok_s=st.get('decode_tok_s'),
                        steps=st.get('steps'), accept_len_mean=st.get('accept_len_mean'),
                        ms_per_step=1000 * st['decode_s'] / st['steps'] if st.get('steps') else None,
                        route_stats=fd.route_stats_report() if fd.rs_uniq is not None else None)
            if p:
                item['repeat_exact'] = report['runs'][name]['sha256'] == item['sha256']
            report['runs'][name] = item
            print('DECODE_KERNELS_RUN ' + json.dumps(dict(rank=e.ep.rank, workload=name, measured=bool(p),
                  **{k: v for k, v in item.items() if k not in ('tokens', 'route_stats')})), flush=True)
            if item['route_stats']:
                rs = item['route_stats']
                print('DECODE_ROUTE_STATS ' + json.dumps(dict(rank=e.ep.rank, workload=name, steps=rs['steps'],
                      mean=round(rs['mean'], 2), per_layer=rs['per_layer'])), flush=True)

    gen = e.generate(ids['python'], max_tokens=args.max_tokens, temperature=0, seed=42,
                     stop_token_ids={eos})
    got = 0
    while got < args.warmup_tokens:
        got += len(next(gen))
    steps0 = fd.stats['steps']
    prof = profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
    prof.start()
    for _ in range(args.bursts):
        try:
            next(gen)
        except StopIteration:
            break
    torch.cuda.synchronize()
    prof.stop()
    steps = fd.stats['steps'] - steps0
    for _ in gen:
        pass
    fams = collections.defaultdict(lambda: [0, 0.0])
    names = collections.defaultdict(lambda: [0, 0.0])
    seq = []
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = ev.device_time_total if hasattr(ev, 'device_time_total') else ev.cuda_time_total
        f = fams[family(ev.name)]; f[0] += 1; f[1] += us
        nm = names[ev.name[:120]]; nm[0] += 1; nm[1] += us
        seq.append((ev.time_range.start, round(us, 2), ev.name[:160]))
    # Ordered kernel sequence of the window: a layer's program is what lies between two
    # consecutive routed-expert down kernels (see docs/decode-launches.md).
    seq.sort()
    with open(f'{args.out}/kernels-seq-rank{e.ep.rank}.tsv', 'w') as f:
        for start, us, name in seq:
            f.write(f'{start}\t{us}\t{name}\n')
    per = {k: dict(kernels=c / steps, ms=us / steps / 1000) for k, (c, us) in fams.items()}
    report['profile'] = dict(steps=steps, kernels_per_step=sum(c for c, _ in fams.values()) / steps,
                             kernel_ms_per_step=sum(us for _, us in fams.values()) / steps / 1000,
                             families=per,
                             top=[dict(name=k, per_step=c / steps, us=us / c)
                                  for k, (c, us) in sorted(names.items(), key=lambda x: -x[1][0])[:60]])
    print('DECODE_KERNELS_PROFILE ' + json.dumps(dict(rank=e.ep.rank, **{k: v for k, v in report['profile'].items()
                                                                          if k != 'top'})), flush=True)
    path = f'{args.out}/kernels-rank{e.ep.rank}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=1)
    owner = os.stat('/app/results')
    for artifact in (path, f'{args.out}/kernels-seq-rank{e.ep.rank}.tsv'):
        os.chown(artifact, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    print('DECODE_KERNELS_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
