"""Full-engine lossless-scale qualification; run separate native/packed TP2 boots.

Keep the same explicit DSV41_RESIDENT_EXPERTS and demand history in both arms.
Compare every output hash and resident-mask digest before interpreting timings.
Hot reads, adaptation and prefix reuse are disabled for the matched-output test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--runs', type=int, default=2)
    args = ap.parse_args()
    if args.tokens < 1 or args.runs < 1:
        ap.error('tokens and runs must be positive')
    if os.environ.get('WORLD_SIZE') != '2' or not os.environ.get('DSV41_RESIDENT_EXPERTS', '').isdigit():
        ap.error('requires TP2 and an explicit fixed resident budget')
    for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE',
                'DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PROBE_PREFILL',
                'DSV41_HOT_READS', 'DSV41_LOOKUP_DRAFT_ENABLED',
                'DSV41_BLOCK_CONFIDENCE'):
        os.environ[key] = '0'
    os.environ['DSV41_DYNAMIC_EXPERTS'] = '1'
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
    import torch
    import engine.v41_engine as V
    from engine.expert_profiles import mask_digest
    from server.app import Tok, load_encoding_module, build_chat_prompt
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    eng = V.V41Engine(root, max_seq=32768, arena_gb=float(os.environ.get('ARENA_GB', '97.4')),
                      spec=True, trace_stats='/app/results/trace-union/stats/coverage.json',
                      prune_keep=.61, transient_slots=136, keep_free_gb=6)
    eng.depth_policy.pinned = 3
    signature = mask_digest(eng.model_prune_mask)
    tok, enc = Tok(root), load_encoding_module(root)
    rows = []
    report = dict(rank=eng.ep.rank, config=eng.config(), mask_sha256=signature, rows=rows)
    for name, prompt in (
        ('prose', 'Explain how rain forms and how weather forecasts work, in several detailed paragraphs.'),
        ('code', 'Implement a thread-safe LRU cache in Python with type hints and unittest tests.'),
    ):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                         enc, tok, False, 75, eng)
        hashes = []
        for trial in range(args.runs + 1):
            captures = eng.fast.graph_captures
            output = []
            for burst in eng.generate(ids, max_tokens=args.tokens, temperature=0, seed=42, ignore_eos=True):
                output.extend(burst)
            torch.cuda.synchronize()
            assert len(output) == args.tokens
            assert mask_digest(eng.model_prune_mask) == signature
            digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
            peers = eng.ep.gather_objects(digest)
            assert all(value == digest for value in peers), 'rank output disagreement'
            hashes.append(digest)
            assert len(set(hashes)) == 1, 'greedy output changed between repeats'
            stats = eng.last_stats
            row = dict(workload=name, measured=trial > 0, output_sha256=digest,
                       tokens=output, decode_tok_s=stats['decode_tok_s'],
                       ms_per_step=1000 * stats['decode_s'] / stats['steps'],
                       acceptance=stats['accept_len_mean'],
                       captures=eng.fast.graph_captures-captures)
            if trial:
                assert row['captures'] == 0, 'timed run captured graphs'
            rows.append(row)
            print('SCALE_GATE ' + json.dumps({k: v for k, v in row.items() if k != 'tokens'}), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            target = args.out.with_name(f'{args.out.stem}-rank{eng.ep.rank}.json')
            target.write_text(json.dumps(report, indent=2) + '\n')
    eng.ep.gather_objects(True)
    os._exit(0)


if __name__ == '__main__':
    main()
