"""Engine adapter for disk prefixes. All ranks make the same restore decision.

CPU copies happen before decoder replay overwrites state; file writes use one bounded
background job. Disk failures are cache misses, never unilateral collective skips.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from engine.prefix_disk import PrefixDisk, cpu_tree, device_tree, token_hashes
from engine.prefix_media import media_prefix

PREFIX_DISK_MIN_INTERVAL_S = float(os.environ.get('DSV41_PREFIX_DISK_MIN_INTERVAL_S', '300'))
if not math.isfinite(PREFIX_DISK_MIN_INTERVAL_S) or PREFIX_DISK_MIN_INTERVAL_S < 0:
    raise ValueError('DSV41_PREFIX_DISK_MIN_INTERVAL_S must be finite and nonnegative')

# Before staging, ask the local index whether every boundary a save would write is
# already published. A conversation re-sent after a restart, or a save that comes due
# with no new tokens, then costs a hash-and-index lookup instead of seconds of CPU copy
# and a full KV rewrite. Skipped only when every rank holds the same bundle (_duplicate).
PREFIX_DISK_DEDUP = os.environ.get('DSV41_PREFIX_DISK_DEDUP', '1') == '1'


def namespace(engine):
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256(b'prefix-persistence-v2')
    # Include auxiliary attention/HC/Engram kernels, not just the top-level model.
    # Test/benchmark edits alone should not discard a user's useful prompt history.
    excluded = ('test_', 'bench_', 'tune_', 'diag_', 'profile_')
    for directory in ('engine', 'tools'):
        for path in sorted(p for p in (root / directory).iterdir() if p.suffix in ('.py', '.cpp')):
            if not path.name.startswith(excluded):
                h.update(path.relative_to(root).as_posix().encode())
                h.update(path.read_bytes())
    model = Path(engine.model_dir)
    for name in ('model.safetensors.index.json', 'inference/config.json', 'tokenizer.json'):
        h.update((model / name).read_bytes())
    for path in sorted((model / 'inference').glob('*.py')):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    # No multi-hundred-GB startup scan. Size + nanosecond mtime detect local checkpoint
    # replacements; config/index/tokenizer are content-hashed. Rank-local namespaces may
    # differ (e.g. copied file timestamps); shared bundle UUIDs still identify one run.
    for path in sorted(model.glob('*.safetensors')):
        st = path.stat()
        h.update(f'{path.name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    ignored = ('DSV41_CAPTURE', 'DSV41_LOG_', 'DSV41_PREFIX_', 'DSV41_PRUNE_SWAP',
               'DSV41_PRUNE_DB', 'DSV41_PRUNE_UNIT', 'DSV41_PRUNE_HALFLIFE', 'DSV41_ADAPT_')
    env = {k: v for k, v in os.environ.items()
           if k.startswith('DSV41_') and not k.startswith(ignored)}
    # The dreaming overlay is named by a path that may stay the same while its generation
    # changes (serve/current), so the env above does not cover it: a KV prefix computed with one
    # generation's experts is not valid under another.
    from engine import dream_overlay
    h.update(f'dream_overlay:{dream_overlay.digest()}'.encode())
    import triton
    h.update(json.dumps({'env': env, 'torch': torch.__version__, 'triton': triton.__version__,
                         'world': engine.ep.world, 'rank': engine.ep.rank,
                         'prune_keep': engine.prune_keep, 'act_quant': engine.act_quant,
                         'kernel': engine.kernel, 'max_context': engine.max_context,
                         'index_topk': engine.args.index_topk,
                         'strict_routing': os.environ.get('DSV41_PREFIX_DISK_STRICT', '0'),
                         'swa_replay': engine.swa_replay}, sort_keys=True).encode())
    return h.hexdigest()


def routing_signature(engine):
    masks = getattr(engine, 'model_prune_mask', None)
    if masks is None:
        return 'all-experts'
    h = hashlib.sha256()
    for layer in sorted(masks):
        h.update(masks[layer].detach().cpu().numpy().tobytes())
    return h.hexdigest()


class PersistentPrefixes:
    def __init__(self, engine, log):
        self.engine, self.log = engine, log
        self.strict = os.environ.get('DSV41_PREFIX_DISK_STRICT', '0') == '1'
        root = os.environ.get('DSV41_PREFIX_DISK_DIR', 'results/prefix-cache')
        self.disk = PrefixDisk(Path(root) / f'rank-{engine.ep.rank}', namespace(engine),
                               float(os.environ.get('DSV41_PREFIX_DISK_GB', '20')) * 1e9)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefix-disk')
        self.pending = None
        self.stats = {}
        self.min_interval_s = PREFIX_DISK_MIN_INTERVAL_S
        self.dedup = PREFIX_DISK_DEDUP
        self._last_save_attempt = None

    def join(self):
        if self.pending is not None:
            try:
                result = self.pending.result()
                self.log('prefix_disk_save: ' + json.dumps(result))
            except Exception as exc:
                self.log(f'prefix disk write failed ({type(exc).__name__}); keeping RAM cache')
            self.pending = None

    def _save_snapshots(self, prompt):
        """Boundary snapshots that still describe this exact prompt, and the media tuple."""
        e = self.engine
        media = getattr(e, '_prefix_media', ())
        snapshots = dict(e._prefix_snapshots)
        snapshots[len(prompt)] = e._prefix_cache
        return {n: s for n, s in snapshots.items() if s is not None
                and n <= len(prompt) and tuple(prompt[:n]) == s['ids']
                and tuple(s.get('media', ())) == media_prefix(media, n)}, media

    def _duplicate(self, prompt):
        """(skip, prepared): skip when every boundary of this save is already served by a bundle
        present on EVERY rank. Restore uses only bundle ids common to all ranks, so rank 0's
        disk alone is not enough: a bundle rank 1 failed to write or has since evicted would
        otherwise be skipped forever, and every restart would miss. Hashing needs only token
        ids and media, so this runs before any tensor copy. Every rank reaches the gather,
        including one whose local lookup failed (it reports None, which forces a write)."""
        e = self.engine
        prepared, local = None, None
        try:
            prepared = self._save_snapshots(prompt)
            snapshots, media = prepared
            if snapshots:
                local = self.disk.serving(token_hashes(tuple(prompt), snapshots, media),
                                          e._prefix_route if self.strict else None)
        except Exception as exc:
            self.log(f'prefix disk duplicate check failed ({type(exc).__name__}); saving')
        ranks = e.ep.gather_objects(local)
        if not ranks or any(not r for r in ranks):
            return False, prepared
        # Every rank computes the same choice from the same gathered lists (rank 0's order),
        # so both touch the same bundles and their LRU orders stay in step.
        chosen = []
        for n, bids in ranks[0].items():
            common = [b for b in bids if all(b in r.get(n, ()) for r in ranks[1:])]
            if not common:
                return False, prepared
            chosen.append(common[0])
        try:
            self.disk.touch(chosen)
        except Exception as exc:
            self.log(f'prefix disk touch failed ({type(exc).__name__})')
        return True, prepared

    def save(self, prompt):
        e = self.engine
        # RAM snapshots are already current. At long contexts CPU staging alone cost
        # 5–6 seconds per tool turn; rate-limit BEFORE copying any tensors. Rank 0's
        # clock decides for both ranks, through the same unconditional collective.
        # None means skip; a UUID identifies the same computation on both nodes.
        now = time.monotonic()
        bid = None
        if e.ep.rank == 0 and (self._last_save_attempt is None
                              or now - self._last_save_attempt >= self.min_interval_s):
            bid = uuid.uuid4().hex
        bid = e.ep.broadcast_obj(bid)
        if bid is None:
            return
        # Both ranks got the same bid and run the same boot-guarded dedup flag, so the
        # gather inside _duplicate is reached on both or on neither.
        prepared = None
        if self.dedup:
            duplicate, prepared = self._duplicate(prompt)
            if duplicate:
                # Not a write, so do not consume the interval: the next turn with a longer
                # prompt must be allowed to persist its new suffix immediately.
                self.stats = {'source': 'disk-duplicate', 'tokens': len(prompt)}
                self.log('prefix disk save skipped: identical boundaries already on disk')
                return
        # Bound failed attempts too, rather than retrying expensive staging every turn.
        self._last_save_attempt = now
        self.join()
        start = time.perf_counter()
        try:
            snapshots, media = prepared if prepared is not None else self._save_snapshots(prompt)
            c = e.caches
            payload = cpu_tree({
                'ids': tuple(prompt), 'snapshots': snapshots,
                'media': media,
                'ckv': {L: value[:len(prompt) // e.args.compress_ratios[L]]
                        for L, value in c.ckv.items()},
                'ik': {L: value[:len(prompt) // e.args.compress_ratios[L]]
                       for L, value in c.ik.items()},
            })
            staging_s = time.perf_counter() - start
            def write():
                t = time.perf_counter()
                result = self.disk.save(bid, payload)
                return {**result, 'stage_s': round(staging_s, 4),
                        'write_s': round(time.perf_counter() - t, 4)}
            self.pending = self.pool.submit(write)
        except Exception as exc:
            self.log(f'prefix disk staging failed ({type(exc).__name__}); keeping RAM cache')

    def restore(self, prompt, memory_n):
        e = self.engine
        start = time.perf_counter()
        self.join()
        self.stats = {'source': 'memory' if memory_n else 'miss', 'tokens': memory_n}
        try:
            candidates = self.disk.candidates(prompt, e._prefix_route if self.strict else None,
                                              getattr(e, '_prefix_media', ()))
        except Exception as exc:
            self.log(f'prefix disk lookup failed ({type(exc).__name__}); treating as miss')
            candidates = []
        # Include RAM as an option, but never let local availability determine the
        # number of prefill collectives. Intersect before loading any GPU state.
        options = candidates + ([('memory', memory_n)] if memory_n else [])
        ranks = e.ep.gather_objects(options)
        common = set(map(tuple, ranks[0]))
        for rank_options in ranks[1:]:
            common.intersection_update(map(tuple, rank_options))
        longest = max((item[1] for item in common), default=0)
        if ('memory', longest) in common:
            choice = ('memory', longest)
        else:
            # Rank 0's list is newest-used first within a length. Both ranks see that
            # same list; UUID lexicographic order would choose an arbitrary old run.
            choice = next((tuple(item) for item in ranks[0]
                           if item[1] == longest and tuple(item) in common), None)
        if choice is None:
            e._prefix_cache = None
            e._prefix_snapshots.clear()
            self.stats = {'source': 'miss', 'tokens': 0}
            return 0
        bid, n = choice
        if bid == 'memory':
            return n
        payload = None
        try:
            payload = self.disk.load(choice, prompt, getattr(e, '_prefix_media', ()))
            if payload is not None:
                self.validate(payload, n)
        except Exception as exc:
            self.log(f'prefix disk validation failed ({type(exc).__name__}); treating as miss')
            payload = None
        if not all(e.ep.gather_objects(payload is not None)):
            # Fall back only to a RAM length both ranks already restored.
            n = memory_n if ('memory', memory_n) in common else 0
            if not n:
                e._prefix_cache = None
                e._prefix_snapshots.clear()
            self.stats = {'source': 'memory' if n else 'miss', 'tokens': n,
                          'disk_rejected': True}
            return n
        # Validation and I/O finished on ALL ranks. Restore fixed backing allocations;
        # CUDA graph pointers must never be replaced by deserialized tensors.
        staged, ok = None, True
        try:
            staged = device_tree(payload['snapshots'][n], e.device)
            for field in ('ckv', 'ik'):
                for L, value in payload[field].items():
                    rows = n // e.args.compress_ratios[L]
                    getattr(e.caches, field)[L][:rows].copy_(value[:rows])
            if e._try_restore(staged, prompt) != n:
                raise ValueError('snapshot restore rejected')
            # Surface copy failures before either rank skips prefill.
            if str(e.device).startswith('cuda'):
                torch.cuda.synchronize()
        except Exception as exc:
            self.log(f'prefix disk restore failed ({type(exc).__name__}); recomputing prompt')
            ok = False
        if not all(e.ep.gather_objects(ok)):
            e._prefix_cache = None
            e._prefix_snapshots.clear()
            return 0
        e._prefix_cache = staged
        e._prefix_snapshots.clear()
        self.stats = {'source': 'disk', 'tokens': n, 'load_s': round(time.perf_counter() - start, 4)}
        self.log('prefix_disk_restore: ' + json.dumps(self.stats))
        return n

    def validate(self, payload, n):
        e = self.engine
        pc = payload['snapshots'][n]
        if self.strict and pc.get('route') != e._prefix_route:
            raise ValueError('routing changed')
        for field in ('ckv', 'ik'):
            actual, target = payload[field], getattr(e.caches, field)
            if actual.keys() != target.keys():
                raise ValueError('KV layers changed')
            for L, value in actual.items():
                rows = n // e.args.compress_ratios[L]
                if (value.dtype != target[L].dtype or value.ndim != 2
                        or value.shape[1:] != target[L].shape[1:]
                        or not rows <= value.shape[0] <= target[L].shape[0]):
                    raise ValueError('KV layout changed')
        w = min(n, e.args.window_size)
        slots = pc['slots']
        if (slots.dtype != torch.int64 or tuple(slots.shape) != (w,)
                or not torch.equal(slots, torch.arange(n-w, n) % e.caches.win[0].shape[0])):
            raise ValueError('window slots changed')
        if set(pc['win']) != set(range(e.args.candidate_source_layer + 1)):
            raise ValueError('window layers changed')
        for L, value in pc['win'].items():
            if value.dtype != e.caches.win[L].dtype or tuple(value.shape) != (w, e.args.head_dim):
                raise ValueError('window layout changed')
        if pc['pending'].keys() != e.caches.pending.keys():
            raise ValueError('compressor layers changed')
        for pending in pc['pending'].values():
            if pending is not None and (len(pending) != 2 or any(
                    x.dtype != torch.float32 or tuple(x.shape) != (e.args.head_dim,) for x in pending)):
                raise ValueError('compressor state changed')
        rep = pc['rep']
        if set(rep) != {'h', 'pre_mix', 'topk', 'cand'}:
            raise ValueError('replay fields changed')
        if (tuple(rep['h'].shape) != (w, e.args.hc_mult, e.args.dim)
                or tuple(rep['pre_mix'].shape) != (w, e.args.hc_mult)
                or rep['topk'].ndim != 2 or rep['topk'].shape[0] != w
                or rep['topk'].dtype not in (torch.int32, torch.int64)
                or (rep['cand'] is not None and (rep['cand'].dtype != torch.bool
                    or rep['cand'].ndim != 2 or rep['cand'].shape[0] != w))):
            raise ValueError('replay layout changed')
