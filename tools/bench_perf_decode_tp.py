"""Paired two-rank qualification; run only after the user releases both GPUs.

GATE_PROFILE=.env.ds41-mxfp4 GATE_SOURCE_ROOT=<same snapshot on both nodes>
GATE_IMAGE=<immutable id> GATE_LOG_DIR=results/perf-decode
bash tools/run_two_node_gate.sh bench_perf_decode_tp.py --out /app/results/perf-decode.json

Use --plan on the host for a CPU-only description. No checkpoint/torch import
or GPU discovery happens in that mode. Each candidate is compared separately;
target-arithmetic candidates report output changes instead of requiring equality.
"""
import argparse
from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

EXPERIMENTS = ('index_candidates', 'hc_post_pre', 'draft_router', 'attention_direct', 'engram_h2d')
PROMPTS = {
    'prose': 'Explain how rain forms and how weather forecasts work, in several detailed paragraphs.',
    'code': 'Implement a thread-safe LRU cache in Python with type hints and unittest tests.',
    'edit': 'Reproduce this Python module, changing capacity to 64 and adding a clear method:\n'
            + '\n'.join(f'def function_{n}(value):\n    return value + {n}' for n in range(80))
            + '\ncapacity = 32\n',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--experiment', choices=EXPERIMENTS, action='append')
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--prompt-tokens', type=int, default=0)
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--pairs', type=int, default=2)
    ap.add_argument('--workload', choices=tuple(PROMPTS), action='append')
    ap.add_argument('--depth', choices=(3, 5), type=int, action='append')
    args = ap.parse_args()
    if args.tokens < 1 or args.pairs < 1 or args.prompt_tokens < 0 or args.max_seq <= args.tokens + args.prompt_tokens:
        ap.error('positive tokens/pairs and max-seq larger than the complete request are required')
    experiments = args.experiment or list(EXPERIMENTS)
    workloads, depths = args.workload or list(PROMPTS), args.depth or [3, 5]
    plan = dict(experiments=experiments, order=['off', 'on', 'on', 'off'],
                pairs=args.pairs, temperature=0, tokens=args.tokens, workloads=workloads, depths=depths,
                max_seq=args.max_seq, prompt_tokens=args.prompt_tokens,
                frozen_residency=True, hot_expert_reads=False, prefix_reuse=False,
                requires_exclusive_two_node_gpu=True)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not args.out:
        ap.error('--out is required outside --plan')
    # This program is a standalone gate, never a command against the live server.
    if os.environ.get('WORLD_SIZE') != '2':
        ap.error('run through the disposable two-node gate')
    for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE',
                'DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PROBE_PREFILL',
                'DSV41_DYNAMIC_EXPERTS', 'DSV41_HOT_READS', 'DSV41_LOOKUP_DRAFT_ENABLED',
                'DSV41_BLOCK_CONFIDENCE'):
        os.environ[key] = '0'
    os.environ['DSV41_GPU_TIMING'] = '1'
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
    from engine.perf_config import DecodeExperiments, FLAGS
    for field, key in FLAGS.items():
        os.environ[key] = '1' if field in experiments else '0'
    import torch
    import engine.v41_engine as V
    from server.app import Tok, load_encoding_module, build_chat_prompt
    from engine.expert_profiles import mask_digest
    V.save_prune_db = lambda *a, **kw: None
    engine = V.V41Engine(os.environ['MODEL_DIR'], max_seq=args.max_seq,
        arena_gb=float(os.environ.get('ARENA_GB', '97.4')), spec=True,
        trace_stats='/app/results/trace-union/stats/coverage.json',
        prune_keep=float(os.environ.get('PRUNE_KEEP', '.61')),
        transient_slots=int(os.environ.get('TRANSIENT_SLOTS', '136')),
        keep_free_gb=float(os.environ.get('KEEP_FREE_GB', '6')))
    policy = engine.depth_policy
    if policy is None:
        raise ValueError('qualification needs DSV41_BLOCK_DYNAMIC=3,5')
    tok = Tok(os.environ['MODEL_DIR'])
    enc = load_encoding_module(os.environ['MODEL_DIR'])
    signature = mask_digest(engine.model_prune_mask)
    rows, summaries = [], []
    report = dict(plan=plan, rank=engine.ep.rank, config=engine.config(), rows=rows, summaries=summaries)
    # The graph key omits flags because production fixes them at boot. Keep
    # separate graph pools/catalogs here, so ABBA does not measure recapture.
    graph_fields = ('graphs', '_memos', 'draft_graphs', 'pool')
    catalogs, active = {}, [None]

    def activate(field, enabled):
        fd = engine.fast
        key = field, enabled
        if active[0] == key:
            return
        torch.cuda.synchronize()
        if active[0] is not None:
            catalogs[active[0]] = {name: getattr(fd, name) for name in graph_fields}
        state = catalogs.get(key, dict(graphs=OrderedDict(), _memos={}, draft_graphs=None, pool=None))
        for name, value in state.items():
            setattr(fd, name, value)
        fd._memo = None
        fd._ahead = None
        active[0] = key

    def run(ids, field, enabled, tokens):
        torch.cuda.synchronize()
        activate(field, enabled)
        config = replace(DecodeExperiments(), **{field: enabled})
        agreements = engine.ep.gather_objects(config.boot_fields())
        assert all(item == agreements[0] for item in agreements), 'runtime experiment differs by rank'
        engine.experiments = engine.fast.experiments = config
        captures = engine.fast.graph_captures
        output = []
        for burst in engine.generate(ids, max_tokens=tokens, temperature=0, seed=42, ignore_eos=True):
            output.extend(burst)
        assert len(output) == tokens, 'incomplete output is not a performance result'
        assert mask_digest(engine.model_prune_mask) == signature, 'resident map changed'
        digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
        peers = engine.ep.gather_objects(digest)
        assert all(item == peers[0] for item in peers), 'rank output disagreement'
        stats = engine.last_stats
        return dict(experiment=field, enabled=enabled, output_sha256=digest,
                    output_text=engine.tokenizer.decode(output),
                    tokens=len(output), decode_s=stats['decode_s'],
                    decode_tok_s=stats['decode_tok_s'], accept_len_mean=stats['accept_len_mean'],
                    ms_per_step=stats['decode_s'] * 1000 / max(1, stats['steps']),
                    gpu=stats.get('gpu_timing'), captures=engine.fast.graph_captures - captures,
                    config=asdict(config))

    for workload in workloads:
        prompt = PROMPTS[workload]
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                         enc, tok, False, 75, engine)
        if args.prompt_tokens:
            filler = engine.tokenizer.encode('The river flows past the old bridge. ', add_special_tokens=False)
            ids = (filler * ((args.prompt_tokens // len(filler)) + 1))[:args.prompt_tokens] + list(ids)
        if len(ids) + args.tokens >= args.max_seq:
            raise ValueError('prompt and output exceed max-seq')
        prompt_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        for depth in depths:
            policy.pinned = depth
            for field in experiments:
                # Warm both complete output trajectories, including later Engram rows.
                run(ids, field, False, args.tokens)
                run(ids, field, True, args.tokens)
                measured = []
                for _ in range(args.pairs):
                    for enabled in (False, True, True, False):
                        row = run(ids, field, enabled, args.tokens)
                        assert row['captures'] == 0, 'timed arm captured graphs; increase warm-up coverage'
                        row.update(workload=workload, depth=depth, prompt_tokens=len(ids), prompt_sha256=prompt_hash)
                        measured.append(row)
                        rows.append(row)
                        print('PERF_ROW ' + json.dumps({k: v for k, v in row.items()
                            if k not in ('output_text', 'config', 'gpu')}), flush=True)
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        target = args.out.with_name(f'{args.out.stem}-rank{engine.ep.rank}{args.out.suffix}')
                        target.write_text(json.dumps(report, indent=2) + '\n')
                off = [r for r in measured if not r['enabled']]
                on = [r for r in measured if r['enabled']]
                exact = len({r['output_sha256'] for r in measured}) == 1
                if field in ('draft_router', 'engram_h2d'):
                    assert exact, f'{field} changed greedy target output'
                summaries.append(dict(workload=workload, depth=depth, experiment=field,
                    greedy_equal=exact, off_ms=statistics.median(r['ms_per_step'] for r in off),
                    on_ms=statistics.median(r['ms_per_step'] for r in on),
                    off_tok_s=statistics.median(r['decode_tok_s'] for r in off),
                    on_tok_s=statistics.median(r['decode_tok_s'] for r in on)))
                target.write_text(json.dumps(report, indent=2) + '\n')
                # Drop only this comparison's catalogs once neither is queued.
                torch.cuda.synchronize()
                engine.fast.release_graphs()
                catalogs.clear()
                active[0] = None
    assert all(engine.ep.gather_objects(True))
    print('PERF_DECODE ' + json.dumps(summaries), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
