"""Low-volume, frozen-placement EP/TP quality and speed comparison.

No prefix reuse or adaptive DB writes. Use the same checkpoint, demand DB, arena and
workload order for both modes. Emit metrics and hashes, never captured prompt contents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['ep', 'tp', 'tp-experts'], required=True)
ap.add_argument('--capture', help='optional trusted local token capture for one long-context run')
ap.add_argument('--arena-gb', type=float, default=88)
ap.add_argument('--prune-keep', type=float, default=.60)
ap.add_argument('--expert-layout', choices=['intermediate', 'output'], default='output')
ap.add_argument('--linear-layout', choices=['intermediate', 'output'], default='output')
ap.add_argument('--demand-db')
ap.add_argument('--short-nesting-only', action='store_true')
ap.add_argument('--short-repeats', type=int, default=2)
ap.add_argument('--compare-eager', action='store_true', help='also decode depth 8 without speculation')
ap.add_argument('--compare-full-prefill', action='store_true', help='also bypass encoder/decoder bounded replay')
ap.add_argument('--unshard-dense-controls', action='store_true',
                help='diagnostic only: reconstruct dense groups in memory, retaining routed expert TP')
ap.add_argument('--readme-runs', type=int, default=3)
ap.add_argument('--reduce', choices=['all_reduce', 'scatter', 'auto'], default='all_reduce')
ap.add_argument('--test-adaptation', action='store_true')
ap.add_argument('--test-persistence', action='store_true')
ap.add_argument('--allow-known-depth8-failure', action='store_true',
                help='continue integration checks past the depth-8 failure also reproduced in EP')
args = ap.parse_args()
os.environ['DSV41_TP_EXPERT_LAYOUT'] = args.expert_layout
os.environ['DSV41_TP_LINEAR_LAYOUT'] = args.linear_layout
if args.demand_db:
    os.environ['DSV41_PRUNE_DB'] = args.demand_db
sys.path.insert(0, '/app')
os.environ['DSV41_TP_EXPERTS'] = str(int(args.mode != 'ep'))
os.environ['DSV41_TP_EXPERT_REDUCE'] = args.reduce
for flag in ('DSV41_TP_ATTN', 'DSV41_TP_DENSE', 'DSV41_TP_HEAD'):
    os.environ[flag] = str(int(args.mode == 'tp'))
for flag in ('DSV41_PREFIX_DISK', 'DSV41_PREFIX_CACHE', 'DSV41_PRUNE_SWAP',
             'DSV41_PRUNE_SWAP_PREFILL',
             'DSV41_PREFILL_TIMING', 'DSV41_PREFILL_MOE_TIMING', 'DSV41_ATTN_TIMING',
             'DSV41_STEP_TIMING', 'DSV41_GPU_TIMING'):
    os.environ[flag] = '0'

import torch
import engine.v41_engine as V
from server.app import Tok, load_encoding_module, build_chat_prompt


def main():
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=262144, arena_gb=args.arena_gb,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=args.prune_keep,
                   transient_slots=384 if args.prune_keep >= 1 else 16, keep_free_gb=6)
    if not hasattr(e.ep, 'gather_objects'):
        # Allow the same driver to qualify a pre-TP source revision.
        def gather_objects(value):
            result = [None] * e.ep.world
            torch.distributed.all_gather_object(result, value)
            return result
        e.ep.gather_objects = gather_objects
    enc, tok = load_encoding_module(root), Tok(root)
    def chat(text):
        return build_chat_prompt({'messages': [{'role': 'user', 'content': text}]},
                                 enc, tok, False, 75, e)[1]
    def run(ids, label, cap=128, grade=None, generation=0, expected_prefix=0):
        output = []
        started, first_token_s = time.perf_counter(), None
        for burst in e.generate(ids, max_tokens=cap, temperature=0, seed=42,
                                ignore_eos=grade is None):
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            output.extend(burst)
        stats = e.last_stats
        row = dict(mode=args.mode, label=label, tokens=len(ids), output_tokens=len(output),
                   prefill_s=stats['prefill_s'], prefill_tok_s=stats['prefill_tok_s'],
                   decode_tok_s=stats['decode_tok_s'], prefix=stats['prefix_cached_tokens'],
                   accept_len_mean=stats.get('accept_len_mean'),
                   output_hash=hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                   first_token_wall_s=round(first_token_s or 0, 4),
                   request_wall_s=round(time.perf_counter() - started, 4),
                   allocated_gb=round(torch.cuda.memory_allocated()/1e9, 3))
        row['prune_miss'] = stats.get('prune_miss_request')
        if grade is not None:
            # The serving adapter truncates at the FIRST stop token in a speculative
            # burst. Merely filtering EOS IDs retains any tokens emitted after EOS.
            visible = output[:output.index(e.eos_token_id)] if e.eos_token_id in output else output
            text = tok.decode(visible).strip()
            row['tokens_after_eos'] = max(0, len(output) - len(visible) - 1)
            # Only these synthetic nesting answers may be logged, never capture outputs.
            row['answer'] = text
            try:
                row['passed'] = json.loads(text) == grade
            except ValueError:
                row['passed'] = False
        assert row['prefix'] == expected_prefix and e.expert_generation == generation
        peers = e.ep.gather_objects(row['output_hash'])
        assert len(set(peers)) == 1, 'rank outputs diverged'
        if e.ep.rank == 0:
            print('TP_ENGINE ' + json.dumps(row), flush=True)
        return row
    run(chat('Reply with OK.'), 'warmup', 16)
    if args.short_nesting_only:
        def routing_signature(engine):
            masks = getattr(engine, 'model_prune_mask', None)
            if masks is None:
                return 'all-experts'
            digest = hashlib.sha256()
            for layer in sorted(masks):
                digest.update(masks[layer].detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        if e.ep.rank == 0:
            print('FROZEN_MASK ' + routing_signature(e), flush=True)
        short_failed = False
        short_failures = []
        probes = []
        for repeat in range(args.short_repeats):
            for depth in (4, 6, 8, 10):
                leaf = 40 + depth
                prompt = (f'Output one JSON object nested exactly {depth} levels deep and nothing else. '
                          'Each level has exactly one key "n" whose value is the next level down. '
                          f'The innermost "n" is the integer {leaf}. '
                          f'So depth 2 would be: {{"n": {{"n": {leaf}}}}}')
                expected = leaf
                for _ in range(depth):
                    expected = {'n': expected}
                row = run(chat(prompt), f'short-nesting-{depth}-repeat-{repeat}', 128, expected)
                if repeat == 0:
                    probes.append((chat(prompt), depth, expected))
                short_failed |= not row['passed']
                if not row['passed']:
                    short_failures.append(depth)
                if args.compare_eager and depth == 8 and repeat == 0:
                    saved_spec, saved_fast = e.spec, e.fast
                    e.spec, e.fast = False, None
                    run(chat(prompt), 'short-nesting-8-eager', 128, expected)
                    if args.compare_full_prefill:
                        saved_replay = e.swa_replay
                        e.swa_replay = False
                        run(chat(prompt), 'short-nesting-8-full-prefill-eager', 128, expected)
                        e.swa_replay = saved_replay
                    e.spec, e.fast = saved_spec, saved_fast
        if args.test_adaptation:
            # Exercise the production-sized request-boundary plan, without saving
            # benchmark demand to the user's placement database.
            plan = e.ep.broadcast_obj(
                e.plan_swaps(max_swaps=512, min_gain=.005) if e.ep.rank == 0 else None)
            if plan:
                e.apply_swaps(plan)
            if e.ep.rank == 0:
                print('SHORT_ADAPTATION ' + json.dumps(dict(
                    swaps=len(plan), generation=e.expert_generation)), flush=True)
            for ids, depth, expected in probes:
                row = run(ids, f'short-nesting-{depth}-after-adaptation', 128, expected,
                          generation=e.expert_generation)
                if not row['passed']:
                    short_failures.append(depth)
        if args.unshard_dense_controls and short_failed:
            assert args.mode == 'tp'
            # All ranks perform identical mutations in this disposable process. Eager
            # decoding avoids executing captured graphs with replaced weight pointers.
            e.spec, e.fast = False, None
            import gc
            gc.collect()
            from fp8_linear import FP8Weight, FP8GroupedWeight
            from engine.tensor_parallel import RowParallelWeight, OutputParallelWeight
            def gather_tensor(value, dim):
                raw = value.view(torch.uint8) if value.dtype == torch.float8_e4m3fn else value
                pieces = [torch.empty_like(raw) for _ in range(e.ep.world)]
                torch.distributed.all_gather(pieces, raw.contiguous())
                result = torch.cat(pieces, dim=dim)
                return result.view(value.dtype) if raw.dtype != value.dtype else result
            def gather_weight(value, dim):
                if isinstance(value, OutputParallelWeight):
                    value, dim = value.local, 0
                if isinstance(value, RowParallelWeight):
                    value = value.local
                if isinstance(value, FP8Weight):
                    return FP8Weight(gather_tensor(value.w, dim), gather_tensor(value.s, dim))
                return gather_tensor(value, dim)
            groups = {}
            def patch(group, obj, key, value):
                groups.setdefault(group, []).append((obj, key, getattr(obj, key), value))
            W, a = e.model.W, e.args
            patch('head', W, 'head', gather_weight(W.head.local, 0))
            for w in W.layers:
                patch('shared', w, 'sh_w1', gather_weight(w.sh_w1, 0))
                patch('shared', w, 'sh_w3', gather_weight(w.sh_w3, 0))
                patch('shared', w, 'sh_w2', gather_weight(w.sh_w2, 1))
                patch('attention', w, 'wq_b', gather_weight(w.wq_b, 0))
                patch('attention', w, 'attn_sink', gather_tensor(w.attn_sink, 0))
                wo = w.wo_a
                full = (FP8GroupedWeight(gather_tensor(wo.w, 0), gather_tensor(wo.s, 0),
                                        wo.G * e.ep.world, wo.R)
                        if isinstance(wo, FP8GroupedWeight) else gather_tensor(wo, 0))
                patch('attention', w, 'wo_a', full)
                patch('attention', w, 'wo_b', gather_weight(w.wo_b, 1))
                patch('attention', w, 'tp_heads', a.n_heads)
                patch('attention', w, 'tp_groups', a.o_groups)
            prompt = ('Output one JSON object nested exactly 8 levels deep and nothing else. '
                      'Each level has exactly one key "n" whose value is the next level down. '
                      'The innermost "n" is the integer 48. So depth 2 would be: {"n": {"n": 48}}')
            expected = 48
            for _ in range(8):
                expected = {'n': expected}
            for group, changes in groups.items():
                for obj, key, old, new in changes:
                    setattr(obj, key, new)
                torch.cuda.synchronize()
                run(chat(prompt), 'short-nesting-8-replicated-' + group, 128, expected)
                for obj, key, old, new in changes:
                    setattr(obj, key, old)
            for changes in groups.values():
                for obj, key, old, new in changes:
                    setattr(obj, key, new)
            torch.cuda.synchronize()
            run(chat(prompt), 'short-nesting-8-replicated-all-dense', 128, expected)
        if not args.unshard_dense_controls:
            unexpected = [d for d in short_failures
                          if not (d == 8 and args.allow_known_depth8_failure)]
            assert not unexpected, f'short nesting quality gate failed: {unexpected}'
        torch.cuda.synchronize()
        assert all(e.ep.gather_objects(True))
        sys.stdout.flush()
        os._exit(0)
    readme = chat('Summarize this repository documentation. Explain its purpose, architecture, '
                  'setup, and limitations.\n\n' + Path('/app/README.md').read_text())
    for i in range(args.readme_runs):
        run(readme, f'readme-{i}')
    if args.capture:
        captured_ids, capture_error = None, None
        if e.ep.rank == 0:
            try:
                capture = torch.load(args.capture, map_location='cpu', weights_only=False)
                if capture.get('vl') is not None or capture.get('kwargs', {}).get('grammar') is not None:
                    raise ValueError('only text-only captures are supported')
                captured_ids = list(capture['prompt_ids'])
                del capture
            except Exception as exc:
                capture_error = type(exc).__name__
        captured_ids, capture_error = e.ep.broadcast_obj((captured_ids, capture_error))
        if capture_error:
            raise RuntimeError('capture unavailable: ' + capture_error)
        run(captured_ids, 'captured-long-context', 64)
        del captured_ids
        run(readme, 'readme-after-long')
    failed_quality = []
    for depth in (8, 10):
        reference = '\n'.join(f'def reference_{i}(value): return (value + {i}) % 97 # reference only'
                              for i in range(300))
        text = ('Ignore the reference text below when answering the task at the end.\n<reference>\n'
                + reference + '\n</reference>\n'
                + f'Output one JSON object nested exactly {depth} levels deep and nothing else. '
                'Each level has exactly one key "n" whose value is the next level down. '
                + f'The innermost "n" is the integer {40+depth}. '
                + f'So depth 2 would be: {{"n": {{"n": {40+depth}}}}}')
        expected = 40 + depth
        for _ in range(depth):
            expected = {'n': expected}
        row = run(chat(text), f'nesting-{depth}', 256, expected)
        if not row['passed']:
            failed_quality.append(depth)
    unexpected_quality = [d for d in failed_quality
                          if not (d == 8 and args.allow_known_depth8_failure)]
    assert not unexpected_quality, f'nesting quality gate failed at depths {unexpected_quality}'
    if args.test_adaptation:
        plan = e.ep.broadcast_obj(e.plan_swaps(max_swaps=2, min_gain=0.0) if e.ep.rank == 0 else None)
        assert plan, 'no adaptive plan available for the test'
        e.apply_swaps(plan)
        for L, old, new, _ in plan:
            if e.ep.tensor_parallel or e.ep.owns(L, new):
                assert (L, new) in e.store.lru and (L, old) not in e.store.lru
                ids_map, slot_map = e.model.prefill_routes[L]
                assert int(slot_map[ids_map[new]]) == int(e.fast.lut[L, new]) != e.store.null_slot
        # Reuse the last long-nesting control after changing the actual resident shards.
        row = run(chat(text), 'nesting-after-adaptation', 256, expected, generation=1)
        assert row['passed'], 'post-adaptation quality gate failed'
        if e.ep.rank == 0:
            print('TP_ADAPTATION_PASSED ' + str(len(plan)), flush=True)
    if args.test_persistence:
        # This standalone diagnostic changes the cache mode on BOTH ranks only after
        # the no-cache performance gate. Production cache mode remains boot guarded.
        V.PREFIX_DISK = True
        os.environ['DSV41_PREFIX_DISK'] = '1'
        os.environ['DSV41_PREFIX_CACHE'] = '1'
        generation = e.expert_generation
        cache_ids = chat(text)
        saved = run(cache_ids, 'persistent-save', 256, expected, generation=generation)
        assert saved['passed']
        e.prefix_disk.join()
        run(chat('Reply with OK.'), 'persistent-other-prompt', 16, generation=generation)
        e.prefix_disk.join()
        e._prefix_cache = None
        e._prefix_snapshots.clear()
        for tensor in (*e.caches.ckv.values(), *e.caches.ik.values(), *e.caches.win):
            tensor.zero_()
        loaded = run(cache_ids, 'persistent-restore', 256, expected, generation=generation,
                     expected_prefix=len(cache_ids))
        assert loaded['passed'] and loaded['output_hash'] == saved['output_hash']
        assert e.prefix_disk.stats['source'] == 'disk'
        e.prefix_disk.join()
        if e.ep.rank == 0:
            print('TP_PERSISTENCE_PASSED', flush=True)
    if e.ep.rank == 0:
        print('TP_ENGINE_INTEGRATION_PASSED ' + args.mode +
              ' known_quality_failures=' + json.dumps(failed_quality), flush=True)
    assert all(e.ep.gather_objects(True))
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
