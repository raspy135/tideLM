"""Batched multi-request decode: every active request's verify block in one weight pass.

A verify step reads ~4 GB of weights a rank (the dense projections, the routed and shared experts,
the head) to compute T_VERIFY rows; the rows themselves are almost free. Decoding N requests one
after another reads those weights N times. This decoder stacks the N verify blocks and issues every
weight-bearing operation ONCE over all N*T rows; only the work that touches one sequence's caches
runs per sequence.

Per backbone layer L, in stream order:

    E[L]    shared   Engram wkv projection (Engram layers only)
    P1[i]   lane i   the previous layer's MoE residual, Engram combine, HC attention mixes, norm
    A[L]    shared   wq_a|wkv, q_norm (row-wise, per lane), wq_b
    C[i]    lane i   RoPE, window ring, compressor, indexer, key block, attention   (keyed)
    O[L]    shared   wo_a, wo_b
    P3[i]   lane i   attention residual, HC FFN mixes, norm, router (demand counters)
    M[L]    shared   routed experts (one routing over every lane's pairs) + shared expert

then per-lane final norm, one shared head, and per-lane drafter-ring seeding.

Graph rules -- break one and two requests corrupt each other without an error:

* Shared graphs run over all N lanes' rows every step. An idle lane's rows hold stale values; no
  op in them mixes rows, and M routes an idle lane's pairs to the null slot
  (decode_lean.route_prep_masked), so they read no expert and come out as exact zeros.
* Anything with a side effect on one sequence -- its KV caches, compressor pending state, drafter
  rings, the shared expert-demand counters -- is in a per-lane graph, replayed only while that lane
  has a request.
* Values cross graphs only through static buffers. The exception is FastDecoder's per-step memo
  inside C (positions, window slots, indexer rows: made in one layer's graph, read by later
  layers'). That lives in graph-pool memory, so each lane's C graphs get a PRIVATE pool. In a pool
  shared with graphs replayed in between, the allocator could hand a live memo block to another
  graph's scratch, which would overwrite it mid-step.
* C is keyed by the lane's (start parity, indexer bucket) and captured only for the key the lane
  is actually at: warming another parity's compressor would overwrite committed cache rows.
* Every weight kernel sees N*T rows; decode_rows.wide() keeps them on the decode tiles.

Exactness: the per-lane pieces run FastDecoder's own code at its own T-row shapes; the shared pieces
are row-invariant kernels at a larger M (tools/test_decode_rows.py). engine/profile_batch.py checks
that a lane's logits equal a serial FastDecoder.step's rather than assuming it.

Scope: the lean TP2 FP4/FP8 configuration with every routable expert resident (`supported`).
Lanes may verify at different depths in one step, each from its own depth policy (narrower blocks
are padded, see `step`).
"""

from __future__ import annotations

from collections import OrderedDict
import gc
import time

import torch

import decode_rows
import v41_ref as R
from engine import comm as _comm
from engine import expert_topp as _ETP
from engine import fastdecode as FD
from engine import l2pf
from engine import model as M
from engine.model import HC_OPS as _HC_OPS, _hc_post_fused


def supported(lane: FD.FastDecoder) -> str | None:
    """None when the batched path can run this decoder's configuration, else the reason. Fixed per
    process and the same on both ranks: every input is pinned by the engine's boot guard."""
    from engine.tensor_parallel import OutputParallelWeight
    from fp8_linear import FP8GroupedWeight, FP8Weight
    if lane.lean is None or not _HC_OPS:
        return "needs the lean decode ops (DSV41_DECODE_LEAN=1 and the fused HC ops)"
    if not (FD.LEAN_ATTN and FD.LEAN_ROPE and FD.LEAN_SOFTMAX) or FD.FUSED_ATTN:
        return "needs the lean attention path (DSV41_DECODE_LEAN_ATTN/ROPE/SOFTMAX=1, DSV41_FUSED_ATTN=0)"
    if not FD.MERGED_PROJ:
        return "needs merged decode projections (DSV41_DECODE_MERGED_PROJ=1)"
    if lane.lut is None or lane.stream_layers:
        return "needs every routable expert resident (device LUT, no layer streaming)"
    if lane.rs_uniq is not None or lane.decode_probe is not None or lane.gpu_timing:
        return "DSV41_ROUTE_STATS, DSV41_GPU_TIMING and the decode probe are single-request diagnostics"
    if max(FD.VERIFY_WIDTHS) != FD.T_VERIFY:
        return "needs T_VERIFY to be the widest verify block"
    if getattr(lane.eng, "kernel", None) != "triton-fp4":
        return "needs the native FP4 expert kernel (EXL3 experts are not batched)"
    w = lane.W.layers[0]
    if getattr(w, "_x3", None) is not None:
        return "EXL3 backbone projections are not batched"
    if not lane._moe_merged_ok(w):
        return "needs the merged TP2 MoE path (output-layout TP experts, CUDA decode, FP8 shared expert)"
    if type(getattr(w, "_wqkv_a", None)) is not FP8Weight or type(w.wq_b) is not FP8Weight:
        return "needs FP8 attention projections (DSV41_DENSE_FP4 must not include attn)"
    if type(w.wo_a) is not FP8GroupedWeight or not isinstance(w.wo_b, OutputParallelWeight):
        return "needs FP8 wo_a and output-layout TP wo_b (DSV41_TP_LINEAR_LAYOUT=output)"
    if any(type(e.wkv) is not FP8Weight for e in lane.W.engram.values()):
        return "needs FP8 Engram projections"
    return None


class BatchDecoder:
    """N FastDecoder lanes over one weight set; `step` verifies any subset of them together."""

    def __init__(self, lanes):
        lanes = tuple(lanes)
        if len(lanes) < 2:
            raise ValueError("batched decode needs at least two lanes")
        first = lanes[0]
        for lane in lanes[1:]:
            if lane.W is not first.W or lane.m.store is not first.m.store or lane.a is not first.a:
                raise ValueError("batched lanes must share weights, the expert arena and the config")
            if lane.lut is not first.lut:
                raise ValueError("batched lanes must share the device slot table (swaps update it in place)")
        if len({id(lane.c) for lane in lanes}) != len(lanes):
            raise ValueError("batched lanes must not share KV caches")
        for lane in lanes:
            reason = supported(lane)
            if reason:
                raise ValueError(f"batched decode {reason}")
        self.lanes = lanes
        self.n = len(lanes)
        self.T = T = FD.T_VERIFY
        self.rows = rows = self.n * T
        a = self.a = first.a
        self.W, self.store, self.dev = first.W, first.m.store, first.dev
        k = a.n_activated_experts
        if rows > decode_rows.WIDE or rows * k > 512:
            raise ValueError(f"{self.n} lanes x {T} rows exceeds the batched decode kernels "
                             f"({decode_rows.WIDE} rows, 512 routed pairs)")
        arena = self.store.arena
        self.world, self.down_n = arena.tp_world, arena.w2.shape[1]
        w0 = self.W.layers[0]
        heads = getattr(w0, "tp_heads", a.n_heads)
        dev, bf16, f32 = self.dev, torch.bfloat16, torch.float32

        def z(*shape, dtype=bf16):
            return torch.zeros(*shape, dtype=dtype, device=dev)

        # Static buffers, rows lane-major: lane i owns rows [i*T, (i+1)*T).
        self.xa = z(rows, a.dim)                     # attention input
        self.qr = z(rows, w0.wq_a.N)                 # q_norm(wq_a x)
        self.kvl = z(rows, w0.wkv.N)                 # wkv x
        self.qb = z(rows, heads * a.head_dim)        # wq_b qr
        self.o = z(rows, heads, a.head_dim)          # attention output, before wo_a
        self.att = z(rows, a.dim)                    # wo_b output
        self.y = z(rows, a.dim)                      # FFN input
        self.ridx = torch.zeros(rows, k, dtype=torch.long, device=dev)
        self.rw = z(rows, k, dtype=f32)
        self.slots = torch.zeros(rows, k, dtype=torch.int32, device=dev)
        self.active = torch.zeros(rows, dtype=torch.uint8, device=dev)
        self.moe = z(self.world * rows, self.down_n, dtype=f32)
        self.xf = z(rows, a.dim)
        self.logits = z(rows, a.vocab_size, dtype=f32)
        eg_rows = a.engram_n_heads * (a.engram_max_ngram_size - 1)
        self.egr = {L: z(rows, eg_rows, a.engram_head_dim, dtype=f32) for L in self.W.engram}
        self.engkv = {L: z(rows, (a.hc_mult + 1) * a.dim) for L in self.W.engram}
        # The attention HC coefficients cross from P1 to P3: per lane, T rows each.
        self.mix = [(z(T, a.hc_mult, dtype=f32), z(T, a.hc_mult, dtype=f32),
                     z(T, a.hc_mult, a.hc_mult, dtype=f32)) for _ in lanes]
        self._active_host = torch.zeros(rows, dtype=torch.uint8).pin_memory()
        self._active_set = None

        self.pool = None                   # shared graphs and the per-lane stateless ones
        self.shared = None                 # name -> graph
        self.lane_graphs = {}              # lane -> {name: graph}
        self.core = {i: OrderedDict() for i in range(self.n)}   # lane -> key -> [graph per layer]
        self.core_pool = {i: None for i in range(self.n)}
        self.core_memo = {}                # (lane, key) -> FastDecoder memo kept alive with its graphs
        self.core_limit = first.graph_limit
        self.stats = {"steps": 0, "lane_steps": 0, "captures": 0, "core_resets": 0, "capture_s": 0.0,
                      "shared_releases": 0}
        # Benchmark-only: {graph kind: [(start, end) CUDA events]} around every replay when set.
        self.profile = None

    # ------------------------------------------------------------------ helpers
    def _sl(self, buf, i):
        return buf[i * self.T:(i + 1) * self.T]

    def _moe_residual(self, i):
        """FastDecoder._layer_b's tail for lane i: the gathered MoE output into its residual."""
        lane, T = self.lanes[i], self.T
        out = (self.moe.view(self.world, self.rows, self.down_n)[:, i * T:(i + 1) * T]
               .reshape(self.world * T, self.down_n))
        _hc_post_fused(out, lane.h, lane.ffn_post, lane.ffn_comb, out=lane.h, split_w=self.down_n)
        lane.pre_mix.copy_(lane.ffn_pre)

    # ------------------------------------------------------------------ graph bodies
    def _engram(self, L):
        kv = R.qlinear(self.egr[L].reshape(self.rows, -1).to(torch.bfloat16), self.W.engram[L].wkv)
        self.engkv[L].copy_(kv)

    def _p1(self, i, L):
        """FastDecoder._layer_a up to the attention input (and _layer_b's tail before it)."""
        lane, a, w = self.lanes[i], self.a, self.W.layers[L]
        if L > 0:
            self._moe_residual(i)
        h = lane.h
        if L in self.W.engram:
            # _layer_a's residual is the Engram output; lane.h holds it until P3 adds attention
            h.copy_(R.engram_combine(h, self._sl(self.engkv[L], i), self.W.engram[L], a))
        if L in a.dspark_target_layer_ids:
            j = list(a.dspark_target_layer_ids).index(L)
            lane.main_hidden[:, j * a.dim:(j + 1) * a.dim] = h.float().mean(dim=1)
        lane._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, out=self.mix[i])
        lane._hc_pre_rn(h, lane.pre_mix, w.attn_norm, out=self._sl(self.xa, i))

    def _attn_in(self, L):
        """_attention's projections for every lane: one wq_a|wkv and one wq_b pass."""
        a, w, T = self.a, self.W.layers[L], self.T
        qkv = R.qlinear(self.xa, w._wqkv_a)
        n = w.wq_a.N
        q_lin = qkv[:, :n]
        for i, lane in enumerate(self.lanes):
            # T rows a lane: the norm's own shape in FastDecoder (a strided view there as well)
            self._sl(self.qr, i).copy_(lane._rmsnorm(q_lin[i * T:(i + 1) * T], w.q_norm, a.norm_eps))
        self.kvl.copy_(qkv[:, n:])
        self.qb.copy_(R.qlinear(self.qr, w.wq_b))

    def _core(self, i, L, st, memo):
        lane, w = self.lanes[i], self.W.layers[L]
        lane._memo = memo
        try:
            freqs = lane.m.freqs_c if w.ratio else lane.m.freqs_w
            fq = lane._step_memo(("fq", freqs.data_ptr()), lambda: freqs[lane.pos])
            o = lane._attn_core_lean(self._sl(self.xa, i), self._sl(self.qr, i), self._sl(self.qb, i),
                                     self._sl(self.kvl, i), w, L, lane.c.win[L], fq, lane.pos, st)
            self._sl(self.o, i).copy_(o)
        finally:
            lane._memo = None

    def _attn_out(self, L):
        a, w = self.a, self.W.layers[L]
        groups = getattr(w, "tp_groups", a.o_groups)
        o = R.wo_a_proj(self.o.reshape(self.rows, groups, -1), w.wo_a, tiled=True)
        self.att.copy_(R.qlinear(o.flatten(1), w.wo_b))

    def _p3(self, i, L, width):
        """FastDecoder._layer_a after the attention: residual, FFN mixes and norm, router."""
        lane, a, w = self.lanes[i], self.a, self.W.layers[L]
        pre, post, comb = self.mix[i]
        h = _hc_post_fused(self._sl(self.att, i), lane.h, post, comb, out=lane.h)
        lane._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base,
                       out=(lane.ffn_pre, lane.ffn_post, lane.ffn_comb))
        y = lane._hc_pre_rn(h, pre, w.ffn_norm, out=self._sl(self.y, i))
        pm = getattr(lane.m, "prune_mask", None)
        keep = pm[L] if pm is not None and L in pm else None
        record = None
        if keep is not None and M.PRUNE_MISS:
            # Only the block's own rows are demand: padding rows (a narrower block) are not.
            def record(lg, sc, _keep=keep, _L=L, _m=lane.m, _w=width):
                _m._record_prune_miss(lg[:_w], sc[:_w], _keep, _L, a.n_activated_experts,
                                      decode=True)
        idx, wts = self._sl(self.ridx, i), self._sl(self.rw, i)
        lane.lean.router(y, w.gate_w, w.gate_bias, keep, a.n_activated_experts, a.route_scale,
                         idx, wts, record=record)
        if lane.expert_topp is not None:
            _ETP.apply_(idx, wts, *lane.expert_topp, lane.topp_sentinel)

    def _moe(self, L):
        """FastDecoder._moe_merged over every lane's rows: one routing, one read of each expert any
        lane picked, one shared-expert pass, the same two all-gathers. Idle lanes -> null slot."""
        import fp4_moe as K
        import fp4_moe_cuda as CUDA
        from engine.collective_rails import group as collective_group
        lane, a, w = self.lanes[0], self.a, self.W.layers[L]
        arena = self.store.arena
        null = int(self.store.null_slot)
        x = self.y
        rows, k = self.ridx.shape
        P = rows * k
        inter, world = arena.w1.shape[1], self.world
        side = lane.shared_stream
        main = torch.cuda.current_stream(self.dev)
        h_all = torch.empty(P + rows, inter, dtype=torch.bfloat16, device=x.device)

        def shared_up():
            gu = R.qlinear(x, w._sh_w13)
            lane.lean.swiglu(gu, w.sh_w1.N, a.swiglu_limit, out=h_all[P:])

        if side is not None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared_up()
        route = lane.lean.route_prep_masked(self.ridx, lane.lut[L], self.active, self.slots, null)
        block_slot, block_pair, block_route, NB = CUDA.build_routing_small(route, 16, k)
        lane.lean.block_null(block_slot, null)
        CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h_all[:P], self.rw.reshape(-1),
                block_slot, block_route, float(a.swiglu_limit), k, inter, K.DIM, NB, null)
        tier = (lane.m.dream_fp8 or {}).get(L) if getattr(lane.m, "dream_fp8", None) else None
        if tier is not None:
            # the dreaming FP8 tier (engine/dream_overlay.py); an idle row's output is discarded
            # like its FP4 rows, so computing it from stale ids is harmless
            CUDA.dream_fp8_up(x, self.ridx.view(-1), tier, h_all[:P], self.rw.view(-1),
                              float(a.swiglu_limit), k)
        if side is not None:
            main.wait_stream(side)
        else:
            shared_up()
        gathered = torch.empty(world * (P + rows), inter, dtype=torch.bfloat16, device=x.device)
        if lane.l2pf_side is not None:
            lane.l2pf_side.wait_stream(main)
            l2pf.touch(lane.l2pf_side, lane._l2pf_targets(L), l2pf.budget_bytes())
        _comm.all_gather_fast(gathered, h_all, group=collective_group())
        if lane.l2pf_side is not None:
            main.wait_stream(lane.l2pf_side)
        full = gathered.view(world, P + rows, inter).transpose(0, 1).reshape(P + rows, world * inter)
        if side is not None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                local_s = R.mm(full[P:], w.sh_w2.local)
        parts = torch.empty(P, self.down_n, dtype=torch.float32, device=x.device)
        CUDA.down(full[:P], arena.w2, arena.s2, parts, block_slot, block_pair, k, self.down_n,
                  world * inter, rows, NB, null, False, 1)
        if tier is not None:
            CUDA.dream_fp8_down(full[:P], self.ridx.view(-1), tier, parts, k, rows)
        local = parts.view(k, rows, self.down_n).sum(dim=0)
        if side is not None:
            main.wait_stream(side)
            local_s.record_stream(main)
        else:
            local_s = R.mm(full[P:], w.sh_w2.local)
        local += local_s
        _comm.all_gather_fast(self.moe, local, group=collective_group())

    def _final_in(self, i):
        self._moe_residual(i)
        self._sl(self.xf, i).copy_(self.lanes[i]._final_x())

    def _head(self):
        self.logits.copy_(R.head_logits(self.xf, self.lanes[0].head))

    def _final_out(self, i):
        lane = self.lanes[i]
        lane.logits.copy_(self._sl(self.logits, i))
        lane._seed_draft_rings()

    # ------------------------------------------------------------------ the step program
    def _program(self, active, keys, widths, mode, load=None):
        """Run, capture or replay one step for `active` lanes, in stream order.

        mode 'eager' executes (warm-up), 'capture' records any graph not yet held (nothing runs),
        'replay' replays. `load(L)` moves the lanes' Engram rows for layer L into place first."""
        a = self.a
        shared = self.shared if self.shared is not None else {}
        states = {i: {"parity": keys[i][0], "index_bucket": keys[i][1],
                      "ckv": None, "ik": None, "ratio": 0} for i in active}
        memos = {}
        for i in active:
            if mode == "eager":
                memos[i] = {}
            elif mode == "capture" and keys[i] not in self.core[i]:
                memos[i] = self.core_memo[(i, keys[i])] = {}
        new_core = {i: [] for i in memos} if mode == "capture" else {}

        def timed(kind, graph):
            if self.profile is None:
                graph.replay()
                return
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            self.profile.setdefault(kind, []).append((start, end))

        def run(store, name, fn, pool):
            if mode == "eager":
                fn()
            elif mode == "replay":
                timed(name[0], store[name])
            elif name not in store:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    fn()
                store[name] = g

        def lane_run(i, name, fn):
            run(self.lane_graphs.setdefault(i, {}) if mode == "capture" else self.lane_graphs.get(i),
                name, fn, self.pool)

        def core_run(i, L):
            if mode == "eager":
                self._core(i, L, states[i], memos[i])
            elif mode == "replay":
                timed("C", self.core[i][keys[i]][L])
            elif i in new_core:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.core_pool[i]):
                    self._core(i, L, states[i], memos[i])
                new_core[i].append(g)

        for L in range(a.n_layers):
            if L in self.W.engram:
                if load is not None:
                    load(L)
                run(shared, ("E", L), lambda L=L: self._engram(L), self.pool)
            for i in active:
                lane_run(i, ("P1", L), lambda i=i, L=L: self._p1(i, L))
            run(shared, ("A", L), lambda L=L: self._attn_in(L), self.pool)
            for i in active:
                core_run(i, L)
            run(shared, ("O", L), lambda L=L: self._attn_out(L), self.pool)
            for i in active:
                lane_run(i, ("P3", L, widths[i]), lambda i=i, L=L: self._p3(i, L, widths[i]))
            run(shared, ("M", L), lambda L=L: self._moe(L), self.pool)
        for i in active:
            lane_run(i, ("F1",), lambda i=i: self._final_in(i))
        run(shared, ("H",), self._head, self.pool)
        for i in active:
            lane_run(i, ("F2",), lambda i=i: self._final_out(i))
        if mode == "capture":
            self.shared = shared
            for i, graphs in new_core.items():
                self.core[i][keys[i]] = graphs

    # ------------------------------------------------------------------ capture
    def _counters(self, active):
        """Demand/miss accumulators a warm-up router would add to (FastDecoder.capture's list)."""
        seen = {}
        for i in active:
            m = self.lanes[i].m
            for name in ("_want_counts", "_want_mass", "_req_counts", "_req_mass", "_rec_counts",
                         "_rec_mass", "_miss_tot", "_miss_mass_tot", "_miss_phase", "_want_phase"):
                value = getattr(m, name, None)
                for tensor in value if isinstance(value, list) else [value]:
                    if tensor is not None:
                        seen[id(tensor)] = tensor
        return [(t, t.clone()) for t in seen.values()]

    def _release_core(self, i):
        torch.cuda.synchronize()
        self.core[i].clear()
        for key in [key for key in self.core_memo if key[0] == i]:
            del self.core_memo[key]
        self.core_pool[i] = None
        gc.collect()
        torch.cuda.empty_cache()
        self.stats["core_resets"] += 1

    # Releases are memory policy, and the policy has to be the same on both ranks: a graph one rank
    # dropped is recaptured with an eager warm-up that runs the step's collectives once more than
    # the other rank's replay does. The runtime calls these from scheduler state both ranks share.
    def _drop_stale_core(self, i, key):
        """Before lane i captures attention-core graphs for `key` = (parity, bucket): drop the ones
        it holds at another bucket. A lane's context only moves on to a larger bucket or restarts
        with a new request, so those are dead weight, and their scratch grows with the bucket: one
        bucket a lane is what the lane reserve allows for. They are kept across requests otherwise
        -- releasing them at every request end cost each concurrent pair a recapture."""
        held = self.core[i]
        if key in held or not held:
            return
        if any(k[1] != key[1] for k in held) or (self.core_limit and len(held) >= self.core_limit):
            self._release_core(i)

    def release_shared(self):
        """No batched step for a while: drop the shared and per-lane stateless graphs. The next
        batched step recaptures them (~1-2 s, measured at four lanes)."""
        if self.shared is None and not self.lane_graphs:
            return
        torch.cuda.synchronize()
        self.shared = None
        self.lane_graphs.clear()
        self.pool = None
        gc.collect()
        torch.cuda.empty_cache()
        self.stats["shared_releases"] += 1

    def _capture(self, active, keys, widths):
        started = time.perf_counter()
        for i in active:
            self._drop_stale_core(i, keys[i])
            if self.core_pool[i] is None:
                self.core_pool[i] = torch.cuda.graph_pool_handle()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        saved = {}
        for i in active:
            lane = self.lanes[i]
            # rollback() can leave pending as a view into kvl_buf/sc_buf, which the warm-up rewrites
            pending = {L: None if v is None else (v[0].clone(), v[1].clone())
                       for L, v in lane.c.pending.items()}
            saved[i] = (lane.h.clone(), lane.pre_mix.clone(), pending)
        counters = self._counters(active)
        first = self.lanes[active[0]]
        stream = first.capture_warmup_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), decode_rows.wide():
            # Compiles every kernel and allocates library workspaces outside capture. Cache writes
            # land on this step's own positions, which the replay rewrites.
            self._program(active, keys, widths, "eager")
            for i in active:
                lane = self.lanes[i]
                if lane.draft_graphs is None:
                    lane._draft(True)
                    lane._draft(False)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for i in active:
            lane = self.lanes[i]
            h, pre, pending = saved[i]
            lane.h.copy_(h)
            lane.pre_mix.copy_(pre)
            lane.c.pending.update(pending)
            lane.prepare_pending_buffers()
        with decode_rows.wide():
            self._program(active, keys, widths, "capture")
            for i in active:
                lane = self.lanes[i]
                if lane.draft_graphs is None:
                    if lane.pool is None:
                        lane.pool = torch.cuda.graph_pool_handle()
                    lane._capture_draft_graphs()
        torch.cuda.synchronize()
        # Warm-up is not user demand: keep it out of the expert-placement evidence.
        for tensor, value in counters:
            tensor.copy_(value)
        # The eager warm-up's scratch sits in the caching allocator afterwards -- at a long
        # context, the indexer's score blocks for every lane. Hand it back: this box runs at its
        # memory ceiling, and a batched capture is rare enough that the next allocation is cheap.
        torch.cuda.empty_cache()
        self.stats["captures"] += 1
        for i in active:
            # The depth policy times steps and skips any that captured (_policy_step_s): a
            # batched capture is seconds, and would otherwise read as a slow depth.
            self.lanes[i].graph_captures += 1
        self.stats["capture_s"] += time.perf_counter() - started

    # ------------------------------------------------------------------ run
    @staticmethod
    def _rows(spec):
        rows = spec() if callable(spec) else spec
        return rows or {}

    def step(self, requests):
        """Verify one block for each lane in `requests` = {lane: (block_ids [w], start S, engram
        rows)}, rows as FastDecoder.step accepts them, any width w <= T_VERIFY per lane. Returns
        {lane: (logits [w], main_hidden [w])}, views of each lane's static buffers (valid until
        that lane's next step).

        A narrower block (dynamic or confidence depth) runs in a full T-row slot: the rows past w
        repeat its last token, are masked out of the routed experts and the demand record, and sit
        at positions the caller's rollback discards -- exactly what happens to a rejected draft.
        Rows before w see nothing of them (causal attention; a compressor pair that reaches into
        them is not yet visible to any row before w)."""
        active = sorted(requests)
        if not active or any(not 0 <= i < self.n for i in active):
            raise ValueError(f"invalid lanes {active}")
        a, T = self.a, self.T
        keys, widths = {}, {}
        for i in active:
            lane = self.lanes[i]
            block, S, _ = requests[i]
            w = block.numel()
            if not 1 <= w <= T or lane.c.len != S:
                raise ValueError((i, w, lane.c.len, S))
            lane._bind(T)
            lane.ids[:w].copy_(block)
            if w < T:
                lane.ids[w:].copy_(block[-1:].expand(T - w))
            torch.add(lane._ar_t, S, out=lane.pos)
            lane.prepare_pending_buffers()
            lane.h.copy_(lane.W.embed[lane.ids].unsqueeze(1).expand(-1, a.hc_mult, -1))
            lane.pre_mix.copy_(lane._premix0)
            bucket = FD._index_bucket(S + T, lane.c.max_seq) if FD.INDEX_BUCKETS else lane.c.max_seq
            keys[i] = (S % 2, bucket)
            widths[i] = w
        live = tuple((i, widths[i]) for i in active)
        if live != self._active_set:
            self._active_host.zero_()
            for i in active:
                self._active_host[i * T:i * T + widths[i]] = 1
            # Blocking: the previous step's copy must not still be reading this pinned buffer.
            self.active.copy_(self._active_host)
            self._active_set = live

        rows = {}
        engram_s = {i: 0.0 for i in active}

        def load(L):
            for i in active:
                if i not in rows:
                    rows[i] = self._rows(requests[i][2])
                value = rows[i].get(L)
                if value is None:
                    continue
                t0 = time.perf_counter()
                if isinstance(value, tuple):
                    future, finish = value
                    value = finish(*future.result())
                w = widths[i]
                self._sl(self.egr[L], i)[:w].copy_(value.reshape(w, *self.egr[L].shape[1:]))
                engram_s[i] += time.perf_counter() - t0

        cold = (self.shared is None
                or any(("P3", 0, widths[i]) not in self.lane_graphs.get(i, {}) for i in active)
                or any(keys[i] not in self.core[i] for i in active))
        if cold:
            for L in self.W.engram:
                load(L)
            self._capture(active, keys, widths)
        for i in active:
            self.core[i].move_to_end(keys[i])
        started = time.perf_counter()
        self._program(active, keys, widths, "replay", None if cold else load)
        for i in active:
            lane = self.lanes[i]
            S = requests[i][1]
            # Host bookkeeping for Caches.rollback, as FastDecoder.step leaves it.
            for L in lane.kvl_buf:
                before = lane.c.pending.get(L)
                lane.c._chunk_inputs[L] = (S, lane.kvl_buf[L], lane.sc_buf[L], before)
                lane.c.pending[L] = ((lane.kvl_buf[L][T - 1].clone(), lane.sc_buf[L][T - 1].clone())
                                     if (S + T) % 2 == 1 else None)
            lane.c.len = S + T
            lane.stats["steps"] += 1
            lane.stats["engram_s"] += engram_s[i]
            lane.stats["graph_s"] += time.perf_counter() - started
        self.stats["steps"] += 1
        self.stats["lane_steps"] += len(active)
        return {i: (self.lanes[i].logits[:widths[i]], self.lanes[i].main_hidden[:widths[i]])
                for i in active}

    def profile_report(self):
        """{kind: total ms on the GPU timeline} for the replays since `profile` was set; syncs."""
        if not self.profile:
            return None
        torch.cuda.synchronize()
        return {kind: round(sum(a.elapsed_time(b) for a, b in pairs), 3)
                for kind, pairs in sorted(self.profile.items())}

    def memory_report(self):
        return dict(lanes=self.n, rows=self.rows, shared_graphs=len(self.shared or {}),
                    lane_graphs={i: len(g) for i, g in self.lane_graphs.items()},
                    core_keys={i: [list(key) for key in keys] for i, keys in self.core.items() if keys},
                    **self.stats)
