"""N-lane cooperative execution over one shared expert arena (DSV41_MAX_CONCURRENCY).

Only the scheduler thread may call this module. Rank zero sends every action to the worker before
executing it locally, including cancellation. No request thread may issue a collective or mutate GPU
state.

Each lane is a full engine view with its own KV caches, Engram history, speculative rollback state,
sampling RNG and prefix snapshots. Prefill runs inside a lane's `start`/`advance` action, so it is
sequential: one lane prefills while every other lane waits at its verify boundary. Verification of
the lanes that are ready runs as one batched step (engine/batch_decode.py), so each weight is read
once per step for all of them.
"""

import copy
import os
import threading

from engine.decode_events import VerifyStep

MAX_LANES = 8
# Single-lane verify steps after which the batched graphs are released; 0 (the default) keeps
# them. They are ~0.25 GB and inside the lane reserve, and every release cost the next concurrent
# pair a recapture (1.1-1.4 s a pair on the 2026-10-14 serving trace).
SOLO_IDLE_STEPS = int(os.environ.get('DSV41_BATCH_IDLE_STEPS', '0'))
# Engine attributes an expert swap reassigns, carried between the primary and the lane that ran.
_SWAP_STATE = ('_swap_baseline', 'resident_layer_counts')
LANE_MAX_SEQ_DEFAULT = 65536


def concurrency() -> int:
    n = int(os.environ.get('DSV41_MAX_CONCURRENCY', '1'))
    if not 1 <= n <= MAX_LANES:
        raise ValueError(f'DSV41_MAX_CONCURRENCY must be 1..{MAX_LANES}')
    return n


def lane_capacities(max_context: int, n: int) -> list[int]:
    """Context allocation per lane. Lane 0 is the engine itself and keeps MAX_SEQ; the others
    allocate DSV41_LANE_MAX_SEQ (default 64K, capped at MAX_SEQ). Each extra lane holds its own
    compressed KV, indexer keys and RoPE tables, ~1.0 GB at 512K against ~0.13 GB at 64K, so a
    profile at its memory ceiling cannot give every lane the full context. Requests that need more
    than a short lane holds wait for lane 0."""
    if n <= 1:
        return [max_context]
    short = int(os.environ.get('DSV41_LANE_MAX_SEQ', str(LANE_MAX_SEQ_DEFAULT)))
    if short < 4096:
        raise ValueError('DSV41_LANE_MAX_SEQ must be at least 4096')
    return [max_context] + [min(short, max_context)] * (n - 1)


def lane_reserve_bytes(args, max_context: int, n: int | None = None) -> int:
    """Memory concurrency adds to the single-request engine, reserved out of the arena at boot.

    The arena is sized against KEEP_FREE_GB before the lanes exist, so without this every byte a
    lane holds comes out of the safety margin -- on a profile at its ceiling, that is how rank 1
    reached the memory watchdog. Per extra lane: the window ring (fixed, 0.18
    GB), compressed KV and indexer keys and RoPE tables (per context token), decoder buffers, and
    its graphs, whose attention-core scratch grows with the context bucket. Plus lane 0's own
    attention-core graphs and the shared batched graphs. DSV41_LANE_RESERVE_GB replaces the
    graph allowance (GB a lane) with a measured one."""
    n = concurrency() if n is None else n
    if n <= 1:
        return 0
    from engine.fastdecode import T_VERIFY
    from engine.model import PACKED_KV, RING
    d = args.head_dim
    kv_row = (d * 9 // 128 * 8 if PACKED_KV else d * 2) + args.index_head_dim * 2
    per_token = sum(kv_row / args.compress_ratios[L] for L in args.kv_source_layers)
    per_token += 2 * (args.rope_head_dim // 2) * 8          # freqs_c and freqs_w, complex64
    ring = (args.n_layers + 3) * RING * d * 2
    override = os.environ.get('DSV41_LANE_RESERVE_GB', '').strip()

    def graphs(context):
        if override:
            return float(override) * 1e9
        # the indexer's score block [T, heads, rows] fp32 at the largest bucket, twice for the
        # rest of a layer's scratch, plus the lane's own single-request graphs
        return 2 * T_VERIFY * args.index_n_heads * 4 * context + 0.25e9

    caps = lane_capacities(max_context, n)
    extra = sum(ring + per_token * cap + 30e6 + graphs(cap) for cap in caps[1:])
    lane0_core = 0 if override else 2 * T_VERIFY * args.index_n_heads * 4 * caps[0]
    return int(extra + lane0_core + 0.4e9)                   # 0.4 GB: the shared batched graphs


class ScheduledPeer:
    """Request termination is an explicit scheduler action on BOTH ranks."""

    def __init__(self, peer):
        self.peer = peer

    def __getattr__(self, name):
        return getattr(self.peer, name)

    def control(self, keep_going, value: int = 0, flag: int = 0) -> bool:
        """The decode loop's per-step gate, broadcast from rank 0 exactly as EPDistributed.control.

        What it carries that rank 1 cannot compute is rank 0's per-step choice: the speculative
        depth (DSV41_BLOCK_DYNAMIC, confidence depth) and the decode-adaptation flag. Both ranks
        reach this inside the same scheduler action, in the same lane, so the broadcasts pair up. Termination is still the scheduler's ('close' on both ranks):
        keep_going is identical on both ranks here, because every input to it is.

        Keeping each rank's own value instead (the earlier version) left rank 1 at the placeholder
        0 -- a root-only verify against rank 0's full block."""
        keep = self.peer.control(keep_going, value, flag)
        self.control_value, self.control_flag = self.peer.control_value, self.peer.control_flag
        return keep

    def release_peer(self):
        pass


# Model attributes the engine sets after construction. A lane's Model is built fresh over its own
# caches, so each has to be carried over or the lane silently runs a different configuration.
_MODEL_ATTRS = ('prune_mask', 'slot_lut', 'prefill_routes', 'vision', 'stream_layers', 'stream_keep',
                'user_prompt', 'critical_stream', 'prefill_graphs', 'prefill_moe_timing')


def clone_lane(engine, source, max_seq: int):
    """An independent mutable decoder over the engine's shared weights and arena, with caches for
    `max_seq` positions. Built once per lane for the server's lifetime: a lane per request would
    erase the batching gain and fragment the unified-memory pool."""
    from engine.fastdecode import FastDecoder
    from engine.model import Caches, Model

    caches = Caches(engine.args, max_seq, engine.device)
    model = Model(engine.W, engine.store, caches, source.m.moe_fn, act_quant=engine.act_quant)
    for name in _MODEL_ATTRS:
        if hasattr(source.m, name):
            setattr(model, name, getattr(source.m, name))
    if max_seq == source.c.max_seq:
        # Position tables depend only on the length; share them instead of a second ~270 MB at 512K.
        model.freqs_c, model.freqs_w = source.m.freqs_c, source.m.freqs_w
    # The hash module caches prior tokens. Sharing it crosses request boundaries even when the KV
    # caches are independent.
    model.hash_state = copy.deepcopy(source.m.hash_state)
    model.engram_rows = source.m.engram_rows
    model.engram_prefetch = source.m.engram_prefetch
    lane = FastDecoder(model, engine, use_graphs=source.use_graphs)
    lane.lut = source.lut
    lane.lut_version = source.lut_version
    return lane


def make_engine_lane(engine, max_seq: int):
    import torch
    from engine.v41_engine import prefetch_rows
    from engine.model import PRUNE_UNIT_REQUEST

    lane = copy.copy(engine)
    lane.lock = threading.Lock()
    lane.fast = clone_lane(lane, engine.fast, max_seq)
    lane.model, lane.caches = lane.fast.m, lane.fast.c
    lane.max_context = max_seq
    lane.tables = {key: copy.copy(table) for key, table in engine.tables.items()}
    for table in lane.tables.values():
        table.stats = dict(table.stats)
    lane.model.engram_rows = lambda layer, hashes: lane.tables[layer].rows(hashes)
    lane.model.engram_prefetch = lambda hashes: prefetch_rows(
        lane.tables, lane.eg_pool, hashes, lane.args.engram_layer_ids)
    # One demand database, but private request/miss counters and captured addresses.
    if engine.model._want_counts is not None:
        lane.model._want_counts = engine.model._want_counts
        lane.model._want_mass = engine.model._want_mass
        if not PRUNE_UNIT_REQUEST:
            lane.model._rec_counts = lane.model._want_counts
            lane.model._rec_mass = lane.model._want_mass
    lane._prefix_cache, lane._prefix_snapshots = None, {}
    lane._token_types = lane._images = None
    lane._requests_since_plan = 0
    # The decode loop builds each verify block in place in _vbufs[width][0] and yields it. copy.copy
    # shares the dict, so every lane's VerifyStep.block would be ONE tensor and the last lane to
    # draft would overwrite the others' blocks before the batched step reads them.
    lane._vbufs = {width: (torch.empty_like(block), torch.empty_like(readback),
                           torch.empty_like(host).pin_memory())
                   for width, (block, readback, host) in engine._vbufs.items()}
    lane._blk, lane._vout, lane._vhost = lane._vbufs[engine._tv]
    # Per-request policy state is the lane's own. Shared, every new request's reset_request()
    # would reset the depth of requests already running, and all lanes would train one policy's
    # step-cost and acceptance statistics on each other's prompts.
    for name in ('depth_policy', 'confidence_depth_policy'):
        if getattr(engine, name, None) is not None:
            setattr(lane, name, copy.deepcopy(getattr(engine, name)))
    lane._decode_adapt_passes = []
    lane.last_stats = {}
    return lane


class DecodeRuntime:
    """Deterministic action executor shared by the HTTP scheduler and worker."""

    def __init__(self, engine, lanes: int | None = None):
        n = concurrency() if lanes is None else lanes
        if n < 2:
            raise ValueError('the cooperative runtime needs DSV41_MAX_CONCURRENCY >= 2')
        if (not engine.ep.active or engine.ep.world != 2
                or engine.fast is None or engine.fast.lut is None or not engine.spec
                or not getattr(engine.ep, 'tensor_parallel', False)):
            raise ValueError('concurrency > 1 requires TP, speculative decode and a resident expert LUT')
        from engine.batch_decode import BatchDecoder, supported
        reason = supported(engine.fast)
        if reason:
            raise ValueError(f'concurrency > 1: batched decode {reason}')
        engine._init_prefix_disk()
        self.capacity = lane_capacities(engine.max_context, n)
        self.engines = (engine,) + tuple(make_engine_lane(engine, cap) for cap in self.capacity[1:])
        self.n = n
        self.peer = engine.ep
        for lane in self.engines:
            lane.ep = ScheduledPeer(self.peer)
            lane._cooperative_decode = True
        self.batch = BatchDecoder([lane.fast for lane in self.engines])
        self.generators = {}
        self.events = {}
        self.answers = {}
        self.rng = {}
        self.store_stats = [dict(engine.store.stats) for _ in self.engines]
        self.prefix_stats = [{} for _ in self.engines]
        self.stats = {}
        self.solo_steps = 0
        self._solo_run = 0          # single-lane verify steps since the last batched one

    def _activate(self, index):
        lane = self.engines[index]
        primary = self.engines[0]
        lane.expert_generation = primary.expert_generation
        lane._prefix_route = primary._prefix_route
        lane._requests_since_plan = primary._requests_since_plan
        # A swap reassigns these on whichever engine view applied it (apply_swaps /
        # apply_global_swaps). _swap_baseline gates the next re-plan (maintain_demand): left stale on
        # the primary, a lane's inline swap made the between-requests pass re-plan early.
        for name in _SWAP_STATE:
            if hasattr(primary, name):
                setattr(lane, name, getattr(primary, name))
        lane.store.stats = self.store_stats[index]
        # One disk index/writer/budget per rank, not one per request. Saves stage
        # their data before yielding and lookups join the preceding disk write.
        if lane.prefix_disk is not None:
            lane.prefix_disk.engine = lane
            lane.prefix_disk.stats = self.prefix_stats[index]
        return lane

    def _advance(self, index, closing=False):
        import torch
        lane = self._activate(index)
        if index in self.rng and self.rng[index] is not None:
            cpu, gpu = self.rng[index]
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(gpu, lane.device)
        try:
            gen = self.generators[index]
            if closing:
                gen.close()
                event = None
            else:
                event = gen.send(self.answers.pop(index, None))
        except StopIteration:
            event = None
        finally:
            if lane.prefix_disk is not None:
                self.prefix_stats[index] = lane.prefix_disk.stats
            if index in self.rng:
                self.rng[index] = (torch.get_rng_state(), torch.cuda.get_rng_state(lane.device))
            primary = self.engines[0]
            primary._requests_since_plan = lane._requests_since_plan
            for name in _SWAP_STATE:
                if hasattr(lane, name):
                    setattr(primary, name, getattr(lane, name))
            # Inline adaptation happens while every other lane is suspended.
            generation = max(e.expert_generation for e in self.engines)
            for e in self.engines:
                e.expert_generation = generation
                if lane.prefix_disk is not None:
                    e._prefix_route = lane._prefix_route
        self.events[index] = event
        if event is None:
            self.generators.pop(index)
            self.rng.pop(index, None)
            self.answers.pop(index, None)
            self.stats[index] = dict(lane.stats())
            self.stats[index]['scheduler'] = {
                'max_concurrency': self.n, 'lane': index, 'lane_context': self.capacity[index],
                'batch_steps_total': self.batch.stats['steps'],
                'lane_steps_total': self.batch.stats['lane_steps'],
                'solo_steps_total': self.solo_steps,
            }
            # The lane's graphs stay for its next request: releasing them here cost every
            # concurrent pair a recapture. Their memory is bounded instead -- one context bucket
            # of attention-core graphs a lane (BatchDecoder._drop_stale_core), a short lane's own
            # single-request graphs by DSV41_GRAPHS_MAX -- and is inside the lane reserve.
        return event

    def execute(self, action):
        """All actions, including close, must be issued identically on every rank."""
        op = action['op']
        if op == 'cache_response':
            if self.generators:
                raise RuntimeError('response prefix preparation requires every lane idle')
            lane = self._activate(action['lane'])
            # The lane may have been reused since HTTP delivery. cache_response
            # agrees on matching input snapshots across ranks, otherwise skips.
            lane.cache_response(action['prompt_ids'], action['response_ids'])
            return None
        if op == 'start':
            index = action['lane']
            if index in self.generators:
                raise RuntimeError('scheduler reused an occupied lane')
            lane = self._activate(index)
            vl = action.get('vl')
            lane.set_vl_inputs(vl['token_types'].to(lane.device), vl['images']) if vl else lane.set_vl_inputs(None, None)
            if action['kwargs'].get('temperature', 1) > 0:
                self.rng[index] = None
            self.generators[index] = lane.generate(action['prompt_ids'], **action['kwargs'])
            return self._advance(index)
        if op in ('advance', 'close'):
            return self._advance(action['lane'], closing=op == 'close')
        if op == 'verify':
            indices = action['lanes']
            if (not indices or indices != sorted(set(indices))
                    or any(not isinstance(self.events.get(i), VerifyStep) for i in indices)):
                raise RuntimeError('verify issued outside a verify boundary')
            steps = {i: (self.events[i].block, self.events[i].position, self.events[i].rows)
                     for i in indices}
            if len(indices) == 1:
                # One ready lane: its own single-request graphs. The batched step is bit-identical
                # to them (tools/gate_batch_decode.py) but pays for N lanes' rows and ~6 graph
                # launches a layer: 87.7 vs 71.0 ms a step with one of four lanes active
                # (TP2, keep 0.57, 2026-10-14), so a lone request would run 19% slower.
                i = indices[0]
                results = {i: self.engines[i].fast.step(*steps[i])}
                self.solo_steps += 1
                self._solo_run += 1
                if SOLO_IDLE_STEPS and self._solo_run == SOLO_IDLE_STEPS:
                    # One request has run alone for a while: the batched graphs are idle memory.
                    self.batch.release_shared()
            else:
                self._solo_run = 0
                results = self.batch.step(steps)
            for i in indices:
                lane = self.engines[i]
                # A batched round's wall time is the round's, not this lane's depth's: the depth
                # policy must not learn from it (V41Engine._policy_step_s).
                lane._last_verify_batched = len(indices) > 1
                self.store_stats[i]['hits'] += lane.args.n_layers * lane.fast.route_idx.numel()
            self.answers.update(results)
            return None
        raise ValueError(f'unknown scheduler action: {op}')
