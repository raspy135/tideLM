"""Two-rank GPU cache gate: save/reload full KV, switch prompts, and restart recovery.

Run with the same code/config for --phase save then --phase load. Only synthetic text and
hashes are used. This is a disposable full-engine process, not an HTTP endpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/app')
os.environ['DSV41_PREFIX_DISK'] = '1'
os.environ['DSV41_PREFIX_SNAPSHOTS'] = '8'
os.environ['DSV41_PREFIX_DISK_STRICT'] = '1'
os.environ['DSV41_PREFIX_CACHE'] = '1'
os.environ['DSV41_PRUNE_SWAP'] = '0'
os.environ['DSV41_PRUNE_SWAP_PREFILL'] = '0'

import torch
from tokenizers import Tokenizer
import engine.v41_engine as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['save', 'load'], required=True)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    model_dir = os.environ['MODEL_DIR']
    e = V.V41Engine(model_dir, max_seq=262144, arena_gb=88,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.60, transient_slots=16, keep_free_gb=6)
    tokenizer = Tokenizer.from_file(model_dir + '/tokenizer.json')
    ids = tokenizer.encode('Summarize this documentation concisely.\n\n' + Path('/app/README.md').read_text()).ids
    golden_path = Path(os.environ['DSV41_PREFIX_DISK_DIR']) / f'golden-rank-{e.ep.rank}.json'

    def run(prompt, label):
        output = []
        for burst in e.generate(prompt, max_tokens=16, temperature=0, seed=42, ignore_eos=True):
            output.extend(burst)
        row = {'label': label, 'tokens': len(prompt), 'prefix': e.last_stats['prefix_cached_tokens'],
               'prefill_s': e.last_stats['prefill_s'],
               'hash': hashlib.sha256(json.dumps(output).encode()).hexdigest(),
               'disk': e.prefix_disk.stats}
        if e.ep.rank == 0:
            print('PREFIX_GATE ' + json.dumps(row), flush=True)
        return row

    def forget():
        e.prefix_disk.join()
        e._prefix_cache = None
        e._prefix_snapshots.clear()
        for tensor in (*e.caches.ckv.values(), *e.caches.ik.values(), *e.caches.win):
            tensor.zero_()
        e.caches.pending = dict.fromkeys(e.caches.pending)
        e.caches._chunk_inputs.clear()
        e.model.begin_prompt()

    # Warm decode graph capture before comparing outputs; the prior investigation recorded
    # a distinct first-cold-request effect. This gate isolates persistence from that effect.
    run(tokenizer.encode('Reply with OK.').ids, 'warmup')
    if args.phase == 'save':
        reference = run(ids, 'save')
        e.prefix_disk.join()
        forget()
        loaded = run(ids, 'disk-full')
        assert loaded['prefix'] == len(ids) and loaded['disk']['source'] == 'disk'
        assert loaded['hash'] == reference['hash'], 'full cache changed output'
        # Changing the tail should restore the preceding 2K boundary, even with no live KV.
        changed = ids[:-5] + tokenizer.encode('\nExplain the installation steps.').ids
        forget()
        partial = run(changed, 'disk-partial')
        assert partial['disk']['source'] == 'disk' and 0 < partial['prefix'] < len(changed)
        e.prefix_disk.join()
        # A different prompt must not destroy the first prompt's disk entry.
        run(tokenizer.encode('Explain a binary search in two sentences.').ids, 'other')
        forget()
        again = run(ids, 'disk-after-other')
        assert again['prefix'] == len(ids) and again['hash'] == reference['hash']
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(golden_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
            json.dump(reference, f)
    else:
        reference = json.loads(golden_path.read_text())
        loaded = run(ids, 'disk-after-process-restart')
        assert loaded['prefix'] == len(ids) and loaded['disk']['source'] == 'disk'
        assert loaded['hash'] == reference['hash'], 'restart cache changed output'
    e.prefix_disk.join()
    assert all(e.ep.gather_objects(True))
    if e.ep.rank == 0:
        print('PREFIX_GATE_PASSED ' + args.phase, flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
