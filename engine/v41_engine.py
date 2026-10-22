"""
v41_engine.py -- the generation loop: chunked prefill, DSpark block drafting + verification with
rejection sampling, cache rollback, and the Engine API the server uses.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from contextlib import nullcontext
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from engine import experts as EX  # noqa: E402
from engine import dist as DT  # noqa: E402
from engine import ablit as R_ablit  # noqa: E402
from engine import dream_overlay as R_dream  # noqa: E402  (overlay digest, boot guard)
from engine.engram import EngramReadAhead, EngramTable, make_hash_state, prefetch_rows  # noqa: E402
from engine.model import MAX_CHUNK, PRUNE_MISS, Caches, Model, Weights  # noqa: E402
from engine import model as M_  # noqa: E402  (ATTN_TIMING phase table)
from engine import fastdecode as _FD  # noqa: E402  (format flags pinned in the boot guard)
from engine import draft_vocab as _DRAFT_VOCAB  # noqa: E402  (listed draft ids, boot guard)
from engine import expert_topp as _ETP  # noqa: E402  (lossy decode pruning, boot guard)
from engine.prefix_media import image_fingerprints, media_prefix
from engine.prefix_persistence import PREFIX_DISK_DEDUP, PREFIX_DISK_MIN_INTERVAL_S
import v41_ref as R  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), "[engine]", *a, flush=True)


# Per-phase host wall-clock timing of the decode loop (DSV41_STEP_TIMING=1, off by default: with
# the flag off nothing but one `if` per phase runs). Everything on this loop is host time -- the
# graphs are queued asynchronously, so a phase's number is the time the HOST spent in it, including
# any wait for GPU work queued earlier. That is the quantity that explains the gap between
# engine/profile_fast.py's step+draft and the tok/s the generator actually reaches.
STEP_TIMING = os.environ.get("DSV41_STEP_TIMING", "0") == "1"
PREFILL_TIMING = os.environ.get("DSV41_PREFILL_TIMING", "0") == "1"
# How many chunk-boundary prefix snapshots to keep for partial prefix reuse (0 = off, the
# shipped behavior: only the full-prompt snapshot). See _restore_prefix: this is what lets a
# prompt that has a changed token in its middle/end -- a per-request timestamp in the prompt
# template is the motivating case -- still reuse the identical tokens before the change,
# rounded down to a MAX_CHUNK boundary. Part of the cross-rank config guard: the restored
# length decides whether a prefill collective happens at all, so the ranks must agree on it.
PREFIX_SNAPSHOTS = int(os.environ.get("DSV41_PREFIX_SNAPSHOTS", "0") or "0")
PREFIX_DISK = os.environ.get("DSV41_PREFIX_DISK", "0") == "1"
# Adapt the resident experts once more, at the prefill -> decode boundary, instead of only after
# the response. The demand blend weights observed evidence w = observed/(observed + PRIOR); at
# PRIOR=2e4 per layer a 30k-token prompt is w=90% against 11% for a 400-token answer, so the
# prompt carries most of the signal and waiting until the end applies it only once the answer is
# already written. Boot-time and part of the cross-rank config guard: a rank that adapts mid-
# request while its peer does not is exactly the silent divergence apply_swaps exists to prevent.
from engine.adapt_config import CFG as ADAPT  # noqa: E402  (every adaptation setting below)
PREFILL_SWAP = ADAPT.swap_prefill
# Below this many NEWLY prefilled tokens the prompt has not earned an adaptation: at 400 tokens
# the blend moves 11% and the swap costs ~0.18 s plus a routing change mid-request.
PREFILL_SWAP_MIN = ADAPT.swap_prefill_min
# ...and only when THIS prompt is actually being served badly. Without this the inline pass
# re-fits the resident set to whatever the latest prompt wants -- at 4.5k tokens a prompt is 58%
# of the blend weight -- and the next request pushes it back: measured 100-128 slots swapped on
# every request at a 1.2% miss rate, ~1 GB of NVMe and 0.2 s each time, buying nothing. The
# end-of-response pass had already converged to 3-7 slots over the same requests. This pass
# exists for the out-of-distribution prompt (a new language, an unfamiliar domain); when the
# prompt is already well served there is nothing for it to fix.
PREFILL_SWAP_MIN_MISS = ADAPT.swap_prefill_min_miss

# Lean step: keep the per-step host work off the critical path -- the greedy accept/reject decision
# is taken on the GPU and read back as ONE 7-element tensor instead of up to eleven separate
# device->host syncs, and the per-step `torch.tensor` / `torch.cat` / `torch.arange` allocations are
# replaced by preallocated buffers (see also DSV41_LEAN_STEP in engine/fastdecode.py, which is the
# same switch). DSV41_LEAN_STEP=0 restores the original per-position Python loop exactly.
# Sampling semantics are untouched: the temperature > 0 path is the ORIGINAL code, RNG draw for RNG
# draw, and the greedy path computes argmax of the same logits in the same order.
LEAN_STEP = os.environ.get("DSV41_LEAN_STEP", "1") == "1"
# Queue the verify's first (Engram-free) segment before the host waits for the hash ids
# (_engram_rows_async). Host ordering only: the GPU runs the same kernels in the same order.
EARLY_VERIFY = os.environ.get("DSV41_EARLY_VERIFY", "1") == "1"
# Greedy decode: queue the next draft right after the verify, from the accept count on the GPU
# (FastDecoder.draft_ahead), instead of after the host has read it. Same draft inputs, same graph.
EARLY_DRAFT = os.environ.get("DSV41_EARLY_DRAFT", "1") == "1"

# Experiments stay off until their two-rank gates and representative A/Bs pass.
LOOKUP_DRAFT_ENABLED = os.environ.get("DSV41_LOOKUP_DRAFT_ENABLED", "0") == "1"
LOOKUP_DRAFT_NGRAM = int(os.environ.get("DSV41_LOOKUP_DRAFT_NGRAM", "16"))
LOOKUP_DRAFT_CANDIDATES = int(os.environ.get("DSV41_LOOKUP_DRAFT_CANDIDATES", "8"))


class StepPhases:
    """Accumulating per-phase timer, printed as a table over a whole run."""

    __slots__ = ("acc", "order", "steps", "_t", "t0")

    def __init__(self):
        self.acc, self.order, self.steps = {}, [], 0
        self.t0 = self._t = time.perf_counter()

    def start(self):
        self._t = time.perf_counter()

    def mark(self, name):
        t = time.perf_counter()
        if name not in self.acc:
            self.acc[name] = 0.0
            self.order.append(name)
        self.acc[name] += t - self._t
        self._t = t

    def table(self, notes=()) -> str:
        n = max(1, self.steps)
        total = sum(self.acc.values())
        wall = time.perf_counter() - self.t0
        rows = [f"  {'phase':<14} {'ms/step':>9} {'% of step':>10} {'total s':>9}"]
        for k in self.order:
            v = self.acc[k]
            rows.append(f"  {k:<14} {v / n * 1e3:9.3f} {v / total * 100:10.1f} {v:9.3f}")
        rows.append(f"  {'-- accounted':<14} {total / n * 1e3:9.3f} {100.0:10.1f} {total:9.3f}")
        rows.append(f"  {'-- loop wall':<14} {wall / n * 1e3:9.3f} {'':>10} {wall:9.3f}"
                    f"   ({n} steps)")
        for name, v in notes:
            rows.append(f"  {name:<14} {v / n * 1e3:9.3f} {'':>10} {v:9.3f}   (already inside a phase)")
        return "\n".join(rows)


class MemoryWatchdog:
    """Kill this process rather than let the host run out of memory.

    A 121 GiB box holding a ~95 GB pinned expert arena has no slack. If something -- a second
    engine, the transient scratch of the 3-bit packer, a large prompt -- pushes MemAvailable toward
    zero, the kernel does not OOM-kill cleanly: it thrashes, sshd can no longer fork, and the
    machine answers ping while being unusable until someone power-cycles it. That happened three
    times on 2026-09-11/12.

    Exiting immediately is strictly better than continuing: the kernel reclaims everything this
    process holds the moment it dies, and the box stays reachable. The watchdog samples
    /proc/meminfo twice a second from a daemon thread and calls os._exit, which is the only way out
    that does not itself need to allocate.
    """

    def __init__(self, floor_gb: float = 2.5, interval: float = 0.5, strikes: int = 6, log=print):
        # `strikes` consecutive samples below the floor before acting: the warm start dips briefly
        # while the packer's scratch is live and recovers on its own, and killing a configuration
        # that works is its own kind of failure. A real collision (a second engine, a runaway
        # allocation) stays below the floor and trips this within a few seconds.
        self.floor = floor_gb * 1e9
        self.strikes = strikes
        self.interval = interval
        self.log = log
        self._stop = threading.Event()
        self.low_water = None

    def start(self):
        if host_available_bytes() is None:
            return self  # not Linux, nothing to watch
        t = threading.Thread(target=self._run, name="mem-watchdog", daemon=True)
        t.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            avail = host_available_bytes()
            if avail is None:
                return
            if self.low_water is None or avail < self.low_water:
                self.low_water = avail
            if avail >= self.floor:
                self._below = 0
                continue
            self._below = getattr(self, "_below", 0) + 1
            if self._below >= self.strikes:
                try:
                    self.log(f"FATAL: host MemAvailable {avail / 1e9:.1f} GB stayed below the "
                             f"{self.floor / 1e9:.1f} GB floor for {self._below * self.interval:.1f} s; "
                             f"exiting now so the kernel can reclaim this process instead of "
                             f"thrashing the machine")
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001 - never let logging stop the exit
                    pass
                os._exit(3)


def host_available_bytes():
    """MemAvailable from /proc/meminfo, or None off Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def exl3_pack_path(model_dir: str, rank: int, world: int) -> str:
    """The per-rank EXL3 pack. `DSV41_EXL3_PACK` overrides; the default sits beside the model."""
    env = os.environ.get("DSV41_EXL3_PACK")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(model_dir)), "exl3-packs",
                        f"exl3-experts-r{rank}of{world}.bin")


class FixedStore:
    """DSpark experts: 3 x 128, all resident, slot = k * 128 + e."""

    def __init__(self, arena):
        self.arena = arena
        self.stats = {"hits": 0, "misses": 0}

    def resolve(self, layer, experts, prefill):
        k = layer - 40
        return (experts.to(torch.int32) + k * 128)


class Penalties:
    """OpenAI presence/frequency penalties, plus a breaker for exact repetition cycles.

    The penalties are the standard ones (a token already generated loses
    `presence_penalty + frequency_penalty * count` from its logit) and default to 0, because a flat
    penalty hurts code, which is legitimately repetitive.

    The cycle breaker is separate and opt-in. Greedy decoding on this checkpoint can fall into
    an exact repetition -- a run of one token, or a short block repeated verbatim -- and has no way
    out of it, because the same context keeps producing the same argmax (NOTES 2026-09-11). It fires
    only on an *exact* cycle of period <= `cycle_max_period` repeated `cycle_repeats` times, and
    then bans just the one token that would continue the cycle for that single step, which is enough
    to leave the attractor. It also bans legitimate repetition: the canonical depth-8 nesting
    answer repeats [n, '":', ' {"'] and its correct token is first banned at position 13.
    Enabling this heuristic changes the model's output distribution, including valid JSON/code;
    it must not silently run on ordinary HTTP requests or quality comparisons.
    """

    def __init__(self, presence: float = 0.0, frequency: float = 0.0, cycle_repeats: int = 4,
                 cycle_max_period: int = 16, enabled: bool | None = None,
                 no_repeat_ngram: int | None = None):
        self.presence = float(presence or 0.0)
        self.frequency = float(frequency or 0.0)
        self.cycle_repeats = int(cycle_repeats)
        self.cycle_max_period = int(cycle_max_period)
        if enabled is None:
            enabled = os.environ.get("DSV41_CYCLE_BREAK", "0") == "1"
        self.cycle_enabled = bool(enabled)
        self.no_repeat_ngram = int(no_repeat_ngram if no_repeat_ngram is not None
                                   else os.environ.get("DSV41_NO_REPEAT_NGRAM", "0"))
        self.counts: dict[int, int] = {}
        self.hits = 0
        self.ngram_hits = 0

    @property
    def active(self) -> bool:
        return (self.presence != 0.0 or self.frequency != 0.0 or self.cycle_enabled
                or self.no_repeat_ngram > 0)

    def observe(self, tokens) -> None:
        for t in tokens:
            self.counts[t] = self.counts.get(t, 0) + 1

    def _cycle_token(self, history) -> int | None:
        """The token that would continue an exact cycle, or None."""
        if not self.cycle_enabled or len(history) < max(4, self.cycle_repeats):
            return None
        for p in range(1, min(self.cycle_max_period, len(history) // self.cycle_repeats) + 1):
            block = history[-p:]
            if all(history[-p * (i + 1):len(history) - p * i] == block for i in range(1, self.cycle_repeats)):
                return block[0]
        return None

    def _banned_ngram_tokens(self, history) -> list:
        """Tokens that would repeat an n-gram of `no_repeat_ngram` already in `history`.

        Same rule as transformers' NoRepeatNGramLogitsProcessor, which is the reference for this
        behaviour: if the last n-1 tokens have appeared before, every token that followed them is
        refused. A degenerate loop repeats far more than n tokens, so it is cut on the second turn
        of the cycle, while ordinary text and code almost never repeat a whole n-gram verbatim.
        """
        n = self.no_repeat_ngram
        if not n or len(history) < n:
            return []
        prefix = tuple(history[-(n - 1):])
        banned = set()
        for i in range(len(history) - n + 1):
            if tuple(history[i:i + n - 1]) == prefix:
                banned.add(history[i + n - 1])
        return list(banned)

    def apply(self, logits: torch.Tensor, history) -> torch.Tensor:
        """logits [V] or [T, V] fp32 -> the same tensor, penalised in place."""
        if self.presence or self.frequency:
            idx = torch.tensor(list(self.counts), device=logits.device, dtype=torch.long)
            if idx.numel():
                cnt = torch.tensor([self.counts[int(i)] for i in idx.tolist()],
                                   device=logits.device, dtype=logits.dtype)
                pen = self.presence + self.frequency * cnt
                if logits.dim() == 1:
                    logits[idx] -= pen
                else:
                    logits[:, idx] -= pen
        ban = self._banned_ngram_tokens(history)
        if ban:
            idx = torch.tensor(ban, device=logits.device, dtype=torch.long)
            if logits.dim() == 1:
                logits[idx] = float("-inf")
            else:
                logits[:, idx] = float("-inf")
            self.ngram_hits += 1
        t = self._cycle_token(history)
        if t is not None:
            self.hits += 1
            if logits.dim() == 1:
                logits[t] = float("-inf")
            else:
                logits[:, t] = float("-inf")
        return logits


def sample_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """[V] fp32 logits -> probability vector (temperature + nucleus)."""
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    p = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        cum = sp.cumsum(0)
        keep = cum - sp < top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        p = torch.zeros_like(p).scatter_(0, si, sp)
        p = p / p.sum()
    return p


def build_keep_masks(counts: dict, frac: float, select: str, device, min_per_layer: int = 24,
                     layer_counts: tuple | None = None):
    """Which experts stay routable under a budget of ceil(frac*384) per layer ON AVERAGE.

    select="uniform": the top ceil(frac*384) experts of every layer (the v0.2.0-wip scheme).
    select="global": one ranking of all (layer, expert) pairs by per-layer-normalized routing
    frequency, cut at the same total count; layers whose routing is flat get more experts and
    layers whose routing is skewed get fewer, with at least `min_per_layer` in every layer. Under
    a total budget this greedy choice maximizes the routing mass that stays routable.
    Returns (masks {L: bool[384] on device}, keep {L: np.ndarray of expert ids, hottest first}).
    """
    import math as _m
    n_keep = max(6, _m.ceil(frac * 384))
    total = n_keep * len(counts)
    keep = {}
    if layer_counts is not None:
        from engine.expert_budget import parse_layer_counts
        layer_counts = parse_layer_counts(','.join(map(str, layer_counts)), len(counts),
                                         384, 6, frac, select)
        if sorted(counts) != list(range(len(counts))):
            raise ValueError('explicit layer counts require contiguous layer IDs')
    if select == "uniform":
        for L, c in counts.items():
            quota = n_keep if layer_counts is None else layer_counts[int(L)]
            keep[int(L)] = np.argsort(np.asarray(c))[::-1][:quota]
    elif select == "global":
        keys = []
        for L, c in counts.items():
            c = np.asarray(c, dtype=np.float64); c = c / c.sum()
            order = np.argsort(c)[::-1]
            keep[int(L)] = [int(e) for e in order[:min_per_layer]]
            keys += [(float(c[e]), int(L), int(e)) for e in order[min_per_layer:]]
        keys.sort(reverse=True)
        remaining = total - sum(len(v) for v in keep.values())
        for _, L, e in keys[:remaining]:
            keep[L].append(e)
        keep = {L: np.array(v) for L, v in keep.items()}
    else:
        raise ValueError(f"unknown prune_select {select!r}")
    masks = {}
    for L, k in keep.items():
        m = torch.zeros(384, dtype=torch.bool, device=device)
        m[torch.as_tensor(np.array(k), device=device)] = True
        masks[L] = m
    return masks, keep


def pruned_owned_counts(ranked: list[tuple[int, int]], world: int) -> list[int]:
    """Number of routable expert pairs assigned to each EP rank."""
    world = max(1, int(world))
    return [sum(1 for _layer, expert in ranked if expert % world == rank)
            for rank in range(world)]


def require_pruned_residency(ranked: list[tuple[int, int]], world: int, lru_slots: int,
                             prune_keep: float, transient_slots: int) -> list[int]:
    """Reject a pruned configuration unless every routable expert is resident.

    The pruned path is explicitly the all-resident serving mode.  Letting a small tail stream is
    not merely a performance degradation: a broad prefill can route to more missing experts in
    one layer than the transient ring can hold.  One EP rank then raises while its peer is already
    inside an NCCL combine, making the surviving GPU look as if it is looping indefinitely.
    """
    owned = pruned_owned_counts(ranked, world)
    overflow = [(rank, count) for rank, count in enumerate(owned) if count > lru_slots]
    if overflow:
        detail = ", ".join(f"rank {rank} needs {count}" for rank, count in overflow)
        raise RuntimeError(
            f"pruned all-resident capacity exceeded at keep={prune_keep:.3f}: {detail}, but each "
            f"rank has {lru_slots} LRU slots ({transient_slots} transient slots reserved). "
            "Reduce --prune-keep or increase --arena-gb; allowing the tail to stream can exhaust "
            "the transient ring during prefill and strand the peer in a collective.")
    return owned


PRUNE_DB = ADAPT.db_path


def load_prune_db(path: str = PRUNE_DB):
    """Persisted per-(layer, expert) routing demand, or None. Survives restarts by design:
    the point is to accumulate enough of THIS server's traffic to out-vote the shipped trace."""
    try:
        if not os.path.exists(path):
            return None
        import numpy as np
        d = np.load(path)
        # The database's unit is part of its meaning: 671k/layer in slot units is a busy server,
        # the same number in request units is not a server at all. Reading one as the other would
        # silently pin the observed weight at ~100% (or ~0%), so a mismatch is ignored, not read.
        want = 1 if ADAPT.request_unit else 0
        have = int(d["unit"][0]) if "unit" in d.files else 0
        if have != want:
            log(f"prune demand DB {path} is in {'request' if have else 'slot'} units but this run "
                f"uses {'request' if want else 'slot'}; ignoring it and starting from the trace")
            return None
        version = int(d["score_history_version"][0]) if "score_history_version" in d.files else 0
        counts, mass = d["counts"], d["mass"]
        if version != 2 and ADAPT.request_unit:
            if ADAPT.metric == "score":
                log(f"prune demand DB {path} has legacy unnormalized score history; "
                    "score ranking starts fresh (frequency history is preserved on disk)")
                return None
            # Old request counts are usable, but their score column isn't a request EWMA.
            mass = np.zeros_like(counts)
        if (counts.shape != mass.shape or counts.ndim != 2 or
                not np.isfinite(counts).all() or not np.isfinite(mass).all() or
                (counts < 0).any() or (mass < 0).any()):
            raise ValueError("invalid routing-demand arrays")
        d.close()
        return counts, mass
    except Exception as e:  # noqa: BLE001
        log(f"prune demand DB unreadable ({e}); starting fresh")
        return None


def save_prune_db(counts, mass, path: str = PRUNE_DB):
    """Atomic replace: a torn file here would silently re-rank the experts on the next boot."""
    import numpy as np
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # savez appends ".npz" when handed a NAME that lacks it -- which would write db.npz.tmp.npz
    # and leave the rename below pointing at a file that was never created. Hand it a HANDLE.
    with open(tmp, "wb") as f:
        np.savez(f, counts=counts, mass=mass,
                 unit=np.array([1 if ADAPT.request_unit else 0]),
                 score_history_version=np.array([2]))
    os.replace(tmp, path)


def blend_demand(trace_counts: dict, db, prior_slots: float):
    """Blend the shipped trace ranking with observed demand, per layer.

    The trace ranks experts by a corpus that is not this server's traffic; observed demand is
    this server's traffic but starts empty. Weight the observed half by how much of it there is:
    w = observed / (observed + prior), computed PER LAYER -- `prior_slots` is therefore in
    routing slots per layer (tokens x n_activated_experts), NOT totalled across layers. Getting
    that wrong is a silent n_layers-fold error in the wrong direction: the ranking still blends,
    it just barely moves, and the only symptom is the weight in the startup line.

    A cold database changes nothing and a well-exercised one dominates, with no threshold to
    tune or trip over.

    DSV41_PRUNE_METRIC=score uses score mass for the observed distribution. The calibration
    trace contains counts only, so the prior stays frequency-based until fresh traffic lands.
    """
    import numpy as np
    counts, mass = db
    out, ws = {}, []
    for L, tr in trace_counts.items():
        obs = np.asarray(mass[L] if ADAPT.metric == "score" else counts[L], dtype=np.float64)
        n = obs.sum()
        # Request score distributions sum to one, just like counts. Slot-mode score units
        # depend on router magnitude: use counts for confidence, scores only for placement.
        evidence = n if ADAPT.request_unit or ADAPT.metric == "frequency" else float(counts[L].sum())
        w = float(evidence / (evidence + prior_slots)) if n > 0 else 0.0
        ws.append(w)
        t = tr / (tr.sum() or 1.0)
        o = obs / (n or 1.0)
        out[L] = (1.0 - w) * t + w * o
    return out, (sum(ws) / len(ws) if ws else 0.0)


def _hot_boot_fields(args):
    """The boot-guard entry for DSV41_HOT_READS: every setting that changes routing, or "0"."""
    from engine.hot_reads import HotReads
    if not HotReads.enabled():
        return "0"
    return HotReads(args.n_layers, args.n_routed_experts, args.n_activated_experts).boot_fields()


class V41Engine:
    #: this engine can constrain sampling with a decoding gate (``generate(grammar=...)``)
    supports_grammar = True

    def __init__(self, model_dir: str, max_seq: int = 32768, arena_gb: float | None = None, device: str = "cuda",
                 trace_stats: str | None = None, act_quant: bool = False, spec: bool = True, io_threads: int = 12,
                 transient_slots: int = 400, keep_free_gb: float = 20.0, swa_replay: bool | None = None,
                 hot_profile: str | None = None, prune_keep: float | None = None,
                 sim_bits: int | None = None, sim_cold_frac: float = 1.0, prune_select: str = "uniform",
                 expert_format: str = "fp4", sim_cb2_frac: float = 0.0,
                 rank: int | None = None, world_size: int | None = None):
        self.model_dir = model_dir
        from engine.perf_config import DecodeExperiments
        self.experiments = DecodeExperiments.from_env()
        self.device = device
        self.spec = spec
        self._decode_probe_reply = {"version": 1, "action": "status", "ok": True,
                                    "ranks": [{"rank": rank or 0, "ok": True,
                                               "status": "stopped"}]}
        self._decode_probe_phases = None
        from engine import draft_bypass
        from engine.tensor_parallel import tp_draft_head_enabled
        from engine.native_head import validate_config, head_kernel, HEAD_KERNEL_VERSION
        self.draft_bypass_enabled = draft_bypass.enabled()
        # Whether the EXL3 JIT decode kernel is usable. Decided here, not later, because the
        # graph-gated features below need it; cuda_available() triggers the one nvcc build.
        self.exl3_cuda = False
        self.exl3_cuda_source = None
        if (expert_format or "fp4").lower() == "exl3":
            try:
                import exl3_moe as _X3
                self.exl3_cuda = _X3.cuda_available()
            except Exception:  # noqa: BLE001
                self.exl3_cuda = False
            if self.exl3_cuda:
                import exl3_moe_cuda as _XC
                self.exl3_cuda_source = _XC.source_digest()
        if (expert_format or "fp4").lower() == "exl3" and not self.exl3_cuda and self.draft_bypass_enabled:
            log("EXL3: DSV41_DRAFT_BYPASS needs a capturable MoE; disabled with the reference")
            self.draft_bypass_enabled = False
        tp_draft_head_enabled()  # validate even on the unsharded path
        validate_config()
        if os.environ.get("DSV41_LOOKUP_DRAFT_ENABLED", "0") not in ("0", "1"):
            raise ValueError("DSV41_LOOKUP_DRAFT_ENABLED must be 0 or 1")
        if LOOKUP_DRAFT_ENABLED and (LOOKUP_DRAFT_NGRAM < 2 or LOOKUP_DRAFT_CANDIDATES < 1):
            raise ValueError("lookup drafting needs NGRAM >= 2 and CANDIDATES >= 1")
        # A/B switches for the two numerics arms that have no server flag.  Both change what the
        # engine computes, so they are in the EP2 boot guard below:
        #   DSV41_ACT_QUANT=1    -- fake-quantize activations to the reference's fp8-per-32 UE8M0
        #                           (tools/v41_ref.act_qdq_fp8) instead of running them in bf16.
        #   DSV41_MOE_FALLBACK=1 -- routed experts dequantized to bf16 instead of the Triton fp4
        #                           tensor-core kernel. Diagnostic only: dequantization is slow
        #                           and host-side routing prevents CUDA graph capture.
        act_quant = act_quant or os.environ.get("DSV41_ACT_QUANT", "0") == "1"
        self.swa_replay = (os.environ.get("DSV41_SWA_REPLAY", "1") != "0") if swa_replay is None else swa_replay
        self.hot_profile = hot_profile or os.environ.get("DSV41_HOT_PROFILE", "mixed")
        self.lock = threading.Lock()
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.args = R.Args.from_json(os.path.join(model_dir, "inference", "config.json"))
        from engine.expert_budget import parse_layer_counts
        self.prune_layer_counts = parse_layer_counts(
            os.environ.get('DSV41_PRUNE_LAYER_COUNTS', ''), self.args.n_layers,
            self.args.n_routed_experts, self.args.n_activated_experts, prune_keep, prune_select)
        from engine.expert_profiles import ExpertProfile
        self.expert_profile = ExpertProfile(os.environ, self.args.n_layers, self.args.n_routed_experts)
        from engine.layer_stream import parse_layers
        self.stream_layers = parse_layers(os.environ.get('DSV41_STREAM_LAYERS', ''), self.args.n_layers)
        self.user_prompt_stream = os.environ.get('DSV41_USER_PROMPT_STREAM', '0') == '1'
        self.user_prompt_max_loads = int(os.environ.get('DSV41_USER_PROMPT_MAX_LOADS', '0'))
        if self.user_prompt_max_loads < 0:
            raise ValueError('DSV41_USER_PROMPT_MAX_LOADS must be nonnegative (0 is unlimited)')
        self.dynamic_experts = os.environ.get('DSV41_DYNAMIC_EXPERTS', '0') == '1'
        from engine.expert_budget import resident_budget
        self.resident_budget = resident_budget(os.environ.get('DSV41_RESIDENT_EXPERTS', ''),
            dynamic=self.dynamic_experts, layers=self.args.n_layers,
            experts=self.args.n_routed_experts, topk=self.args.n_activated_experts, fraction=prune_keep)
        from engine.global_residency import layer_weights
        self.prune_layer_weights = layer_weights(os.environ.get('DSV41_PRUNE_LAYER_WEIGHTS', ''), self.args.n_layers)
        self.dynamic_layer_floor = self.args.n_activated_experts
        if self.dynamic_experts and (self.prune_layer_counts is not None or any(w != 1. for w in self.prune_layer_weights)):
            raise ValueError('fully dynamic allocation requires no fixed layer counts or layer discounts')
        if self.dynamic_experts:
            if self.stream_layers or not prune_keep or not 0 < prune_keep < 1:
                raise ValueError('dynamic allocation requires adaptive pruned residency without layer streaming')
            # Concurrent lanes (DSV41_MAX_CONCURRENCY > 1) are fine: a cross-layer transfer edits the
            # shared slot table and masks in place, at a scheduler boundary where no lane is
            # mid-step (apply_global_swaps; engine/serving.py carries the reassigned state).
            if os.environ.get('DSV41_LUT', '1') != '1' or os.environ.get('DSV41_PRUNE_MISS', '0') != '1':
                raise ValueError('dynamic allocation requires resident LUTs and PRUNE_MISS=1')
            # RAM prefixes contain encoder state, not expert-sector addresses.
            # With resident prefill they can retain historical KV across global
            # swaps, just as with the original within-layer adaptive policy.
            # Streaming still requires a fresh discovery/rebuild below.
            # Disk prefixes are the same historical KV, written and read back: with
            # DSV41_PREFIX_DISK_STRICT=0 any layout's bundle is reused, as the RAM cache does;
            # with 1, apply_global_swaps refreshes the routing signature bundles match on.
            for key in ('DSV41_PREFIX_RESPONSE', 'DSV41_PREFILL_GRAPHS'):
                if os.environ.get(key, '1' if key == 'DSV41_PREFIX_CACHE' else '0') != '0':
                    raise ValueError('dynamic allocation requires '+key+'=0')
        if self.user_prompt_stream:
            if not self.swa_replay:
                raise ValueError('latest-user streaming requires bounded replay to refresh the first answer after admission')
            if os.environ.get('DSV41_PRUNE_MISS', '0') != '1':
                raise ValueError('latest-user streaming requires DSV41_PRUNE_MISS=1')
            if self.stream_layers or not prune_keep or not 0 < prune_keep < 1:
                raise ValueError('latest-user streaming requires adaptive pruned residency without layer streaming')
            if os.environ.get('DSV41_MAX_CONCURRENCY', '1') != '1':
                raise ValueError('latest-user streaming requires concurrency 1')
            for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_PREFILL_GRAPHS'):
                if os.environ.get(key, '1' if key == 'DSV41_PREFIX_CACHE' else '0') != '0':
                    raise ValueError('latest-user streaming requires ' + key + '=0')
            if os.environ.get('DSV41_LUT', '1') != '1':
                raise ValueError('latest-user streaming requires resident LUTs')
        from engine.layer_stream import LayerAblation
        self.layer_ablation = LayerAblation(os.environ.get('DSV41_LAYER_ABLATION_FILE', ''), self.args.n_layers)
        if self.layer_ablation.path:
            if self.stream_layers != frozenset(range(self.args.n_layers)):
                raise ValueError('layer ablation diagnostics require all backbone layers streamed')
            if ADAPT.swap or PREFILL_SWAP or ADAPT.decode_tokens:
                raise ValueError('layer ablation diagnostics require every adaptive swap trigger disabled')
        if self.stream_layers:
            if not prune_keep or not 0 < prune_keep < 1:
                raise ValueError('layer streaming requires pruned residency')
            if os.environ.get('DSV41_MAX_CONCURRENCY', '1') != '1':
                raise ValueError('layer streaming requires concurrency 1')
            for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFILL_GRAPHS'):
                if os.environ.get(key, '1' if key == 'DSV41_PREFIX_CACHE' else '0') != '0':
                    raise ValueError('layer streaming requires ' + key + '=0')
            if os.environ.get('DSV41_LUT', '1') != '1':
                raise ValueError('layer streaming requires resident LUTs')
        from engine.critical_stream import CriticalStream
        self.critical = CriticalStream(self.args.n_layers, self.args.n_routed_experts, transient_slots)
        if self.critical.enabled:
            if self.stream_layers or self.user_prompt_stream:
                raise ValueError('choose calibrated rescue or layer streaming, not both')
            if not prune_keep or not 0 < prune_keep < 1:
                raise ValueError('critical prefill requires pruned residency')
            if (os.environ.get('DSV41_PREFIX_CACHE', '1') != '0' or
                    os.environ.get('DSV41_PREFIX_DISK', '0') != '0' or
                    os.environ.get('DSV41_PREFILL_GRAPHS', '0') != '0'):
                raise ValueError('critical prefill requires PREFIX_CACHE=0, PREFIX_DISK=0, PREFILL_GRAPHS=0')
            if os.environ.get('DSV41_MAX_CONCURRENCY', '1') != '1':
                raise ValueError('critical prefill prototype requires concurrency 1')
        from engine.prefill_budget import PrefillBudget
        self.prefill_budget = PrefillBudget.from_env()
        self.prefill_budget_last = {}
        from engine.predictive_prefill import PredictivePrefill
        self.predictive = PredictivePrefill(self.args.n_layers, self.args.n_routed_experts)
        # Serving preference, independent of the checkpoint's reference value (512).
        # Wider attention is experimental: no measured hallucination reduction is claimed.
        self.args.index_topk = int(os.environ.get("DSV41_INDEX_TOPK", "1024"))
        if self.args.index_topk <= 0:
            raise ValueError("DSV41_INDEX_TOPK must be positive")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = 1
        self.max_context = max_seq

        self.act_quant = act_quant
        self.trace_stats = trace_stats
        self.expert_format = (expert_format or "fp4").lower()
        assert self.expert_format in ("fp4", "cb3", "exl3"), self.expert_format
        self.sim_cb2_frac = float(sim_cb2_frac or 0.0)
        assert 0.0 <= self.sim_cb2_frac <= 1.0, self.sim_cb2_frac
        self.moe_fallback = os.environ.get("DSV41_MOE_FALLBACK", "0") == "1"
        if self.moe_fallback:
            # A/B arm: routed experts dequantized to bf16 (engine/moe_fallback.py) instead of the
            # Triton fp4 tensor-core kernel.  Not a serving default -- the fallback is much slower.
            from engine import moe_fallback as K
            fp4_moe_fn, fp4_arena_cls = K.moe_forward, K.ExpertArena
            self.kernel = "dequant-fallback"
            log("DSV41_MOE_FALLBACK=1: routed experts dequantized to bf16; Triton fp4 kernel bypassed")
        else:
            try:
                import fp4_moe as K
                fp4_moe_fn, fp4_arena_cls = K.moe_forward, K.ExpertArena
                self.kernel = "triton-fp4"
                log("using Triton FP4 MoE kernel")
                if os.environ.get("DSV41_ROUTE_WARM", "1") == "1" and hasattr(K, "warm_routing"):
                    # Compile the prefill routing variants now, not inside a served prompt
                    # (tools/fp4_moe.py::warm_routing). Rank-local; free on a warm cache.
                    t0 = time.perf_counter()
                    n = K.warm_routing(self.device, 6 * MAX_CHUNK)
                    log(f"MoE routing kernels warm: {n} variants in {time.perf_counter() - t0:.1f}s")
            except Exception as e:  # noqa: BLE001
                from engine import moe_fallback as K
                fp4_moe_fn, fp4_arena_cls = K.moe_forward, K.ExpertArena
                self.kernel = "dequant-fallback"
                log(f"Triton kernel unavailable ({e!r}); using the slow dequant fallback")
        self.fp4_dot_scaled = bool(getattr(K, "DOT_SCALED", False))
        self.fp4_prefill_dot_scaled = bool(getattr(K, "PREFILL_DOT_SCALED", False))
        self.fp4_cuda = bool(getattr(K, "CUDA_DECODE", False))
        if self.fp4_prefill_dot_scaled:
            log("DSV41_FP4_PREFILL_DOT_SCALED=1: scaled FP4 prompt/replay arithmetic; decode unchanged")
        # Fixed-count random draws keep the ranks aligned, but change seeded completions
        # relative to sequential rejection sampling. Remain opt-in pending serving gates.
        self.batched_verify = os.environ.get("DSV41_BATCHED_VERIFY", "0") == "1"
        self.batched_verify_source = None
        if self.batched_verify:
            from engine.spec_sampling import source_digest, verify_sampled
            self._verify_sampled = verify_sampled
            self.batched_verify_source = source_digest()
            log("DSV41_BATCHED_VERIFY=1: batched sampled verifier (different RNG sequence)")
        self.fp4_cuda_relaxed = False
        self.fp4_cuda_v2 = False
        self.fp4_cuda_down_rows = 1
        self.fp4_cuda_source = None
        if self.fp4_cuda and self.fp4_dot_scaled:
            raise ValueError("DSV41_FP4_CUDA and DSV41_FP4_DOT_SCALED are separate arithmetic arms")
        if self.fp4_cuda:
            import fp4_moe_cuda as native_fp4
            self.fp4_cuda_relaxed = native_fp4.RELAXED_REDUCE
            # v2 replays the relaxed reduction; with DSV41_FP4_CUDA_RELAXED=0 v1 keeps serving.
            self.fp4_cuda_v2 = native_fp4.V2 and native_fp4.RELAXED_REDUCE
            native_fp4.V2 = self.fp4_cuda_v2
            # DSV41_FP4_CUDA_DOWN_ROWS: v3 down, same bits as v2 (tools/test_fp4_moe_v2.py)
            self.fp4_cuda_down_rows = native_fp4.DOWN_ROWS if self.fp4_cuda_v2 else 1
            native_fp4.DOWN_ROWS = self.fp4_cuda_down_rows
            self.fp4_cuda_source = native_fp4.source_digest()
            log(f"DSV41_FP4_CUDA=1: native CUDA FP4 BM=16 decode "
                f"(relaxed reduction={self.fp4_cuda_relaxed})")
        cb3_cls = None
        if self.expert_format == "cb3":
            import cb3_moe as C3
            from engine.codebook_sim import CodebookSim
            assert not sim_bits, "--sim-bits simulates a low-bit format inside an FP4 arena; it is " \
                                 "meaningless with --expert-format cb3, which IS the packed format"
            cb3_cls = C3.CB3ArenaV2
            self._cb3_sim = CodebookSim(3, device)
            cb3_moe_fn = C3.moe_forward_v3
            self.kernel = "triton-cb3"
            log("using Triton CB3 (3-bit per-row codebook) MoE kernel for the routed experts")

        exl3_cls = None
        exl3_moe_fn = None
        if self.expert_format == "exl3":
            import exl3_moe as X3
            assert not sim_bits, "--sim-bits simulates a low-bit format inside an FP4 arena; it is " \
                                 "meaningless with --expert-format exl3"
            assert not self.moe_fallback, "DSV41_MOE_FALLBACK dequantizes native FP4; it does not apply to exl3"
            exl3_cls = X3.Exl3Arena
            exl3_moe_fn = X3.moe_forward_exl3
            self.kernel = "exl3-cuda" if self.exl3_cuda else "exl3-ref"
            log("EXL3 routed experts: " + ("JIT CUDA decode kernel + reference prefill"
                                            if self.exl3_cuda else "torch reference (no JIT kernel)"))

        def moe_fn(x, slots, weights, arena, limit, out_dtype=torch.float32, slots_repeat=False,
                   null_slot=-1, routing_ids=None, routing_slot_map=None, stage_mark=None,
                   prefill=False, dream=None):
            """Dispatch on the arena's format. The DSpark draft arena stays FP4 whatever the main
            arena is -- it is 384 experts (7.2 GB), it is read five times per step, and a 3-bit
            drafter would cost acceptance for nothing.

            `out_dtype` is the EP2 hook (engine/model.py::moe): fp32 keeps this rank's HALF of
            the k-sum unrounded until after the all-reduce. Both arena formats take it now."""
            if exl3_cls is not None and isinstance(arena, exl3_cls):
                return exl3_moe_fn(x, slots, weights, arena, limit, out_dtype=out_dtype,
                                   routing_ids=routing_ids, routing_slot_map=routing_slot_map)
            if cb3_cls is not None and isinstance(arena, cb3_cls):
                # out_dtype/slots_repeat are the EP2 hooks, same meaning as the FP4 path below.
                # null_slot is not passed: the CB3 kernels have no skip, so a pair routed at the
                # null slot computes zeros instead of being skipped -- correct, just not free.
                return cb3_moe_fn(x, slots, weights, arena, limit,
                                  out_dtype=out_dtype, slots_repeat=slots_repeat)
            # The FP4 path serves both the main arena under fp4 and the DSpark draft arena under
            # ANY main format (the draft stays FP4). Gate its hooks on the FP4 kernel being real,
            # not on the main kernel name -- otherwise the draft silently loses routing/prefill.
            fp4_native = self.kernel != "dequant-fallback"
            extra = {"null_slot": null_slot} if null_slot >= 0 and fp4_native else {}
            if fp4_native:
                extra['prefill'] = prefill
            if routing_ids is not None and fp4_native:
                extra.update(routing_ids=routing_ids, routing_slot_map=routing_slot_map)
            if stage_mark is not None and fp4_native:
                extra['stage_mark'] = stage_mark
            if dream is not None:
                # the dreaming FP8 tier (engine/dream_overlay.py): the boot refuses it off this path
                if not fp4_native:
                    raise ValueError("the dreaming FP8 tier needs the native FP4 kernels")
                extra['dream'] = dream
            return fp4_moe_fn(x, slots, weights, arena, limit, out_dtype=out_dtype,
                              slots_repeat=slots_repeat, **extra)

        self.moe_fn = moe_fn
        arena_cls = fp4_arena_cls
        self.scale_pack = None

        def make_expert_arena(n_slots):
            if exl3_cls is not None:
                narrow = self.exl3_exact[0] if getattr(self, "exl3_exact", None) else 0
                if self.ep.tensor_parallel:
                    return exl3_cls(n_slots, device, tp_rank=self.ep.rank, tp_world=self.ep.world,
                                    narrow_slots=narrow)
                return exl3_cls(n_slots, device, narrow_slots=narrow)
            if cb3_cls is None:
                if self.ep.tensor_parallel:
                    return fp4_arena_cls(n_slots, device, tp_rank=self.ep.rank, tp_world=self.ep.world,
                                         **({'scale_pack': self.scale_pack} if self.scale_pack is not None else {}))
                return fp4_arena_cls(n_slots, device)
            a = cb3_cls(n_slots, device)
            a.sim = self._cb3_sim
            return a

        self.expert_bytes = (EX.EXPERT_BYTES if cb3_cls is None
                             else __import__("cb3_moe").CB3_BYTES_PER_SLOT)

        # EP2 (docs/dual-spark-plan.md): two ranks split the 15,360 routed experts by parity and
        # replicate everything else. Default world 1 = the single-box engine, unchanged.
        # The process group is initialized BEFORE the arena is sized: NCCL pins buffers out of
        # the same unified pool, and an arena sized around a not-yet-created PG overcounts what
        # is left (the keep_free floor then quietly eats itself).
        self.ep = DT.EPDistributed(rank, world_size)
        if exl3_cls is not None:
            # one rank's slice; the pack is per rank, so the bytes are the world=1 record halved
            import exl3_ref as _XR
            _tw = self.ep.world if self.ep.tensor_parallel else 1
            _tr = self.ep.rank if self.ep.tensor_parallel else 0
            self.expert_bytes = _XR.record_nbytes(3.0, _tr, _tw)
        self.tp_draft_experts = os.environ.get('DSV41_TP_DRAFT_EXPERTS', '0') == '1'
        # The DSpark draft arena is always fp4_arena_cls (line below), so draft TP needs native FP4
        # for the *draft* -- not for the main routed experts. Under exl3 the main kernel is
        # 'exl3-ref' and the draft still runs the FP4 TP path, which is the intended combination.
        if self.tp_draft_experts and (not self.ep.tensor_parallel
                or (self.kernel != 'triton-fp4' and exl3_cls is None)
                or cb3_cls is not None or sim_bits
                or os.environ.get('DSV41_TP_EXPERT_LAYOUT', 'output') != 'output'):
            raise ValueError('draft TP requires native FP4 TP2 with output layout')
        from engine.tensor_parallel import tp_draft_dense_enabled
        self.tp_draft_dense = tp_draft_dense_enabled()
        if self.tp_draft_dense and (not self.tp_draft_experts
                                    or os.environ.get('DSV41_EXL3_DRAFT_DENSE', '0') == '1'):
            raise ValueError('DSV41_TP_DRAFT_DENSE requires DSV41_TP_DRAFT_EXPERTS=1 and the native '
                             'FP8 draft (DSV41_EXL3_DRAFT_DENSE=0)')
        if self.ep.tensor_parallel:
            if exl3_cls is None and (self.kernel != 'triton-fp4' or cb3_cls is not None or sim_bits):
                raise ValueError('expert TP requires native FP4')
            if exl3_cls is None:
                self.expert_bytes //= self.ep.world
        scale_path = os.environ.get('DSV41_SCALE_PACK', '')
        if scale_path:
            if (not self.ep.tensor_parallel or self.ep.world != 2 or self.kernel != 'triton-fp4'
                    or cb3_cls is not None or exl3_cls is not None or sim_bits or R_dream.enabled()
                    or not self.fp4_cuda_v2
                    or os.environ.get('DSV41_MAX_CONCURRENCY', '1') != '1'
                    or os.environ.get('DSV41_TP_EXPERT_LAYOUT', 'output') != 'output'):
                raise ValueError('DSV41_SCALE_PACK requires single-request native FP4 output TP2 with CUDA v2, '
                                 'without simulated quantization or a dreaming overlay')
            from engine.scale_pack import ScalePack
            self.scale_pack = ScalePack(scale_path.format(rank=self.ep.rank, world=self.ep.world),
                                        model_dir, self.ep.rank, self.ep.world)
            native_bytes = self.expert_bytes
            self.expert_bytes += 3 * (self.scale_pack.stride - self.scale_pack.spec['size'])
            log(f'lossless expert scales: {native_bytes / 1e6:.3f} -> {self.expert_bytes / 1e6:.3f} MB/slot; '
                f'pack {self.scale_pack.header["source_scale_sha256"][:16]}')
        # Experimental EP2 prefill replicas (DSV41_PREFILL_REPLICA_GB) and the prefill EP combine
        # overlap (DSV41_PREFILL_EP_OVERLAP) were removed 2026-10-17: never enabled in a profile,
        # and the overlap measured no reliable gain. docs/prefill-replicas.md and
        # docs/prefill-ep-overlap.md keep what was measured.
        if float(os.environ.get("DSV41_PREFILL_REPLICA_GB", "0") or 0):
            raise ValueError('DSV41_PREFILL_REPLICA_GB: prefill replicas were removed (docs/prefill-replicas.md)')
        if os.environ.get("DSV41_PREFILL_EP_OVERLAP", "0") not in ("", "0"):
            raise ValueError('DSV41_PREFILL_EP_OVERLAP: the overlap was removed (docs/prefill-ep-overlap.md)')
        if self.ep.active:
            assert expert_format in ("", "fp4", "cb3", "exl3"), expert_format
            self.ep.init(device)
            from engine import comm as _comm
            _comm.init(device)   # DSV41_COMM_BACKEND=roce: build the one-shot RoCE runtime or stay on NCCL
            if self.ep.tensor_parallel:
                log(f'TP2 up: rank {self.ep.rank}; each arena slot holds half an expert')
            else:
                log(f"EP2 up: rank {self.ep.rank}/{self.ep.world}; this arena holds only experts"
                    f" where expert % {self.ep.world} == {self.ep.rank}, unowned ones map to its null slot")

        self.W = Weights(model_dir, index, self.args, device, log=log, act_quant=act_quant)
        # DSV41_EXL3_DENSE: the decode graph's attention + shared-expert matrices from the EXL3
        # dense pack (engine/exl3_dense.py). Loaded before the arena so its ~2.1 GB is counted.
        self.exl3_dense = None
        from engine import exl3_dense as _X3D
        if _X3D.enabled():
            if self.expert_format != "exl3" or not self.exl3_cuda or not self.ep.tensor_parallel or self.ep.world != 2:
                raise ValueError("DSV41_EXL3_DENSE needs EXPERT_FORMAT=exl3 with its CUDA kernel under TP2")
            if (os.environ.get("DSV41_TP_ATTN", "0") != "1"
                    or os.environ.get("DSV41_TP_LINEAR_LAYOUT", "output") != "output"
                    or os.environ.get("DSV41_TP_DENSE", "0") != "1"):
                raise ValueError("the EXL3 dense pack is sliced for TP attention/dense with the output layout")
            _t0 = time.perf_counter()
            self.exl3_dense = _X3D.Exl3Dense(_X3D.pack_path(model_dir, self.ep.rank, self.ep.world), device,
                                             self.ep.rank, self.ep.world, self.args.n_layers)
            have = ((self.exl3_dense.ablit_meta or {}).get("sha256") or "")[:16]
            if have != R_ablit.digest():
                raise ValueError(f"EXL3 dense pack ablit term {have or 'none'} != DSV41_ABLIT_WOB "
                                 f"{R_ablit.digest() or 'none'}: rebuild the pack with the overlay in use")
            self.exl3_dense.attach(self.W.layers)
            log(f"EXL3 dense decode matrices: {self.exl3_dense.bytes / 2**30:.2f} GiB in "
                f"{time.perf_counter() - _t0:.1f}s, source {self.exl3_dense.source_sha256[:16]}, "
                f"ablit {'rank-1 term' if have else 'off'}")
        # DSV41_EXL3_DRAFT_DENSE: the DSpark draft blocks' attention and shared-expert matrices from
        # their own pack (engine/exl3_draft.py). Deliberately separate from the backbone dense pack:
        # the draft only proposes, and the verifier accepts exactly the target's argmax, so this is
        # a speed knob with no quality claim -- it must not require the backbone's pack, and it must
        # be runnable under EXPERT_FORMAT=fp4 as well.
        self.exl3_draft = None
        from engine import exl3_draft as _X3DR
        if _X3DR.enabled():
            if not self.ep.tensor_parallel or self.ep.world != 2:
                raise ValueError("DSV41_EXL3_DRAFT_DENSE needs the two-rank TP draft")
            if (os.environ.get("DSV41_TP_DRAFT_ATTN", "0") != "1"
                    or os.environ.get("DSV41_TP_LINEAR_LAYOUT", "output") != "output"):
                raise ValueError("the EXL3 draft pack is sliced for DSV41_TP_DRAFT_ATTN=1 with the "
                                 "output layout (tools/pack_exl3_draft.py)")
            import exl3_moe as _X3M
            if not _X3M.cuda_available():
                raise ValueError("DSV41_EXL3_DRAFT_DENSE needs the EXL3 JIT decode kernel")
            _t0 = time.perf_counter()
            self.exl3_draft = _X3DR.Exl3Draft(_X3DR.pack_path(model_dir, self.ep.rank, self.ep.world),
                                              device, self.ep.rank, self.ep.world)
            if (self.exl3_dense is not None
                    and self.exl3_draft.source_sha256 != self.exl3_dense.source_sha256):
                raise ValueError("the EXL3 draft pack and the EXL3 dense pack were built from "
                                 "different checkpoints")
            self.exl3_draft.attach(self.W.mtp)
            log(f"EXL3 draft matrices: {self.exl3_draft.bytes / 2**30:.3f} GiB in "
                f"{time.perf_counter() - _t0:.1f}s, source {self.exl3_draft.source_sha256[:16]}, "
                f"bits {sorted(set(self.exl3_draft.bits.values()))}")
        if R.dense_fp4_groups():
            # the load-time fp8 -> fp4 re-quantization leaves ~2.4 GB of fp32 scratch blocks in the
            # caching allocator; hand them back before the expert arena asks for its 90 GB.
            torch.cuda.empty_cache()
            log(f"dense fp4 groups: {','.join(sorted(R.dense_fp4_groups()))}; "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated after weights")
        self.caches = Caches(self.args, max_seq, device)
        # DSpark experts: all resident
        self.W.dspark_arena = (fp4_arena_cls(384, device, tp_rank=self.ep.rank, tp_world=self.ep.world)
                              if self.tp_draft_experts else arena_cls(384, device))
        if self.tp_draft_experts:
            log('DSpark experts: TP output shards; shared experts remain replicated')
        self.W.dspark_store = FixedStore(self.W.dspark_arena)
        # main expert arena: size from what is left.
        # On GB10 the GPU and the host share one pool, and `torch.cuda.mem_get_info()` counts the
        # *page cache* as used -- after a 470 GB download it reports 30 GB free on a box with 118 GiB
        # actually available, which silently gives a 23 GB arena. /proc/meminfo's MemAvailable is the
        # honest number (it counts reclaimable page cache), so take the larger of the two and keep a
        # hard floor of `keep_free_gb` under it: this box hard-resets if MemAvailable goes negative.
        free, total = torch.cuda.mem_get_info()
        host_avail = host_available_bytes()
        budget = float(max(free, host_avail or 0))
        reserve = 8e9 + 0.25e9 * (max_seq / 8192)  # activations, indexer slices, page cache headroom
        auto = arena_gb is None
        if auto:
            arena_gb = max(10.0, (budget - reserve) / 1e9 * 0.82)
        # DSV41_EXL3_EXACT_SLOTS=1: every routed expert resident in a slot of its own width (2-bit
        # experts in 4.46 MB slots instead of 6.67 MB ones: 4.25 GB a node), plus the auto budget's
        # margin and the transient ring at full width. The arena is exactly that size; ARENA_GB
        # only has to be large enough, and a cap below it is an error, not a smaller arena.
        self.exl3_exact = None
        if exl3_cls is not None and os.environ.get("DSV41_EXL3_EXACT_SLOTS", "0") == "1":
            import exl3_ref as _XR
            from engine.expert_budget import AUTO_MARGIN
            _tw = self.ep.world if self.ep.tensor_parallel else 1
            _tr = self.ep.rank if self.ep.tensor_parallel else 0
            _hdr = _XR.PackReader(exl3_pack_path(model_dir, _tr, _tw))
            narrow = sum(1 for m in _hdr.records.values() if float(m["bits"]) <= 2.0)
            n_wide = len(_hdr.records) - narrow + AUTO_MARGIN + transient_slots
            exact_bytes = (narrow * _XR.record_nbytes(2.0, _tr, _tw)
                           + n_wide * _XR.record_nbytes(3.0, _tr, _tw))
            if not auto and arena_gb * 1e9 < exact_bytes:
                raise ValueError(f"DSV41_EXL3_EXACT_SLOTS holds every routed expert: ARENA_GB must be at least "
                                 f"{exact_bytes / 1e9:.2f} (have {arena_gb})")
            arena_gb = exact_bytes / 1e9
            self.exl3_exact = (narrow, n_wide)
        # DSV41_MAX_CONCURRENCY > 1: the lanes are built after this, so reserve them here -- out of
        # the arena. ARENA_GB is tuned against the box's steady state for one request (the
        # profiles sit at the memory ceiling: a 98 GB arena loads with ~110 GB available and
        # settles with ~6 GB free), and the cap below only bites when the arena exceeds what is
        # available *now*, which a tuned arena does not. Taking the lanes out of the cap alone left
        # the arena as it was and the lanes in the steady-state margin: rank 1 reached the memory
        # watchdog 17 minutes into a 2-lane session on the 512K / 98.4 GB profile.
        from engine.serving import lane_reserve_bytes
        lane_bytes = lane_reserve_bytes(self.args, max_seq)
        if lane_bytes and not auto and not self.exl3_exact:
            log(f"concurrency: arena {arena_gb:.1f} GB less {lane_bytes / 1e9:.2f} GB for the extra "
                f"lanes and batched graphs")
            arena_gb = max(10.0, arena_gb - lane_bytes / 1e9)
        # The dreaming FP8 tier is loaded after the arena (engine/dream_overlay.py::load_fp8_tp), as
        # this rank's slice of each FP8 expert. Take it out of the arena now, like the lanes above;
        # a profile's ARENA_GB sits at the memory ceiling (64 experts are ~1.2 GB a rank at TP2).
        dream_bytes = (R_dream.fp8_tp_bytes(self.ep.world if self.ep.tensor_parallel else 1)
                       if exl3_cls is None and cb3_cls is None else 0)
        if dream_bytes:
            log(f"dreaming overlay: arena {arena_gb:.1f} GB less {dream_bytes / 1e9:.2f} GB for the FP8 tier")
            arena_gb = max(10.0, arena_gb - dream_bytes / 1e9)
            lane_bytes += dream_bytes  # and out of the MemAvailable cap below
        if host_avail is not None:
            cap = (host_avail - keep_free_gb * 1e9 - lane_bytes) / 1e9
            if cap < 10.0:
                # Refuse rather than squeeze. A cap this low means the host has no memory to give:
                # another engine is still resident, or the kernel has not finished reclaiming the
                # last one's arena (it is page-cache backed and released lazily). Loading anyway is
                # how this box gets wedged -- it answers ping and accepts TCP on 22 while sshd can
                # no longer fork, and only a power cycle brings it back.
                raise RuntimeError(
                    f"not enough host memory to start: MemAvailable {(host_avail or 0) / 1e9:.1f} GB "
                    f"leaves {cap:.1f} GB for the expert arena after keep_free {keep_free_gb} GB. "
                    f"Wait for the previous engine's memory to be reclaimed (watch MemAvailable in "
                    f"/proc/meminfo) or lower --arena-gb/KEEP_FREE_GB.")
            if arena_gb > cap:
                if self.exl3_exact:
                    raise RuntimeError(f"DSV41_EXL3_EXACT_SLOTS needs a {arena_gb:.2f} GB arena but only {cap:.1f} GB "
                                       f"is free above keep_free {keep_free_gb} GB (MemAvailable "
                                       f"{host_avail / 1e9:.1f} GB): free memory (context, caches, vision) or turn it off")
                log(f"arena {arena_gb:.1f} GB capped to {cap:.1f} GB (MemAvailable {host_avail / 1e9:.1f} GB, "
                    f"keep_free {keep_free_gb} GB" + (f", lanes {lane_bytes / 1e9:.2f} GB)" if lane_bytes else ")"))
                arena_gb = max(10.0, cap)
            # A pinned arena is checked against what is free *now*, but the warm start needs more on
            # top of it: the 3-bit packer works on the GPU in scratch buffers, and a chunked prefill
            # allocates activations. Reserve that too, and refuse up front rather than discovering it
            # halfway through a 3-minute warm start with 0.2 GB left.
            # This check runs after the dense weights are resident, so MemAvailable already
            # excludes them: what is still to come is the arena itself plus the transient scratch of
            # the warm start (the 3-bit packer works in GPU buffers). Calibrated against what this
            # box actually does -- a 98 GB arena loads with ~110 GB available and settles with 6 GB
            # free -- rather than a pessimistic sum, because refusing a configuration that works is
            # its own failure. The MemoryWatchdog below is the backstop if this estimate is wrong.
            pack_scratch = 3e9 if self.expert_format != "fp4" else 1e9
            need = arena_gb * 1e9 + lane_bytes + pack_scratch + keep_free_gb * 1e9
            if need > host_avail:
                raise RuntimeError(
                    f"refusing to start: arena {arena_gb:.1f} GB + {pack_scratch / 1e9:.1f} GB "
                    f"warm-start scratch + {keep_free_gb:.1f} GB floor = {need / 1e9:.1f} GB, but "
                    f"MemAvailable is {host_avail / 1e9:.1f} GB. Lower --arena-gb by at least "
                    f"{(need - host_avail) / 1e9:.1f} GB, or stop whatever else holds memory "
                    f"(on this box a boot-time vLLM container used to take 85 GB).")
        slots = int(arena_gb * 1e9 / self.expert_bytes)
        if self.exl3_exact:
            slots = sum(self.exl3_exact)
            log(f"DSV41_EXL3_EXACT_SLOTS: {self.exl3_exact[0]} 2-bit + {self.exl3_exact[1]} 3-bit slots "
                f"(incl. {transient_slots} transient) = {arena_gb:.2f} GB, every routed expert resident")
        self.arena_gb, self.slots = round(arena_gb, 1), slots
        log(f"CUDA free {free / 1e9:.1f} GB of {total / 1e9:.1f}; host MemAvailable "
            f"{(host_avail or 0) / 1e9:.1f} GB; arena {arena_gb:.1f} GB = {slots} {self.expert_format} "
            f"expert slots of {self.expert_bytes / 1e6:.2f} MB "
            f"({slots / 15360 * 100:.0f}% of all routed experts, {'auto' if auto else 'pinned'})")
        # from here on, anything that drives the host out of memory kills this process instead of
        # the machine (see MemoryWatchdog)
        self.mem_watchdog = MemoryWatchdog(floor_gb=float(os.environ.get("DSV41_MEM_FLOOR_GB", "2.5")),
                                           log=log).start()
        self.arena = make_expert_arena(slots)
        if exl3_cls is not None:
            if R_dream.enabled():
                raise ValueError(f"{R_dream.ENV} serves native FP4 experts; this run uses the EXL3 pack")
            import exl3_store as _XS
            self.store = _XS.Exl3ExpertStore(
                exl3_pack_path(model_dir, self.ep.rank if self.ep.tensor_parallel else 0,
                               self.ep.world if self.ep.tensor_parallel else 1),
                self.arena, self.args.n_layers, transient_slots=transient_slots, io_threads=io_threads,
                ep=self.ep)
            # the DSpark drafter stays native FP4 and reads its 3x128 experts from the checkpoint
            self.dspark_reader = EX.ExpertStore(model_dir, index, self.W.dspark_arena, self.args.n_layers,
                                                transient_slots=16, io_threads=2, read_threads=4)
            log(f"EXL3 pack rank {self.store.pack.rank}/{self.store.pack.world}, header "
                f"{self.store.pack.header_sha256()[:16]}, {self.expert_bytes / 1e6:.2f} MB/slot")
            if self.exl3_cuda:
                import exl3_moe_cuda as _XC
                _XC.warm(self.arena)   # JIT + per-slot pointer tables before any graph capture
                log("EXL3 CUDA kernel warm")
        else:
            overlay = None
            if R_dream.enabled():
                if cb3_cls is not None:
                    raise ValueError(f"{R_dream.ENV} serves native FP4 experts; this run uses the CB3 arena")
                overlay = R_dream.load(model_dir, index["weight_map"], self.args.n_layers)
                log(f"dreaming overlay: {R_dream.describe()}")
                if R_dream.fp8_entries():
                    # The FP8 tier: the experts' MXFP4 copies stay in the arena (routing, slots and
                    # the FP4 kernels are unchanged) and the FP8 kernels overwrite their pairs' rows
                    # (tools/fp4_moe_cuda.py::dream_fp8_up/down) in the decode graphs, prefill and
                    # the eager path alike. Each rank holds its own slice, as the arena does.
                    if self.ep.active and not self.ep.tensor_parallel:
                        raise ValueError(f"{R_dream.ENV}: the FP8 tier needs DSV41_TP_EXPERTS=1 (or one box)")
                    if (self.ep.tensor_parallel
                            and os.environ.get("DSV41_TP_EXPERT_LAYOUT", "output") != "output"):
                        raise ValueError(f"{R_dream.ENV}: the FP8 tier needs DSV41_TP_EXPERT_LAYOUT=output")
                    t0 = time.perf_counter()
                    world = self.ep.world if self.ep.tensor_parallel else 1
                    self._dream_fp8 = R_dream.load_fp8_tp(device, self.ep.rank if world > 1 else 0, world,
                                                          self.args.n_routed_experts)
                    n8 = sum(len(d.ids) for d in self._dream_fp8.values())
                    nb = sum(d.nbytes for d in self._dream_fp8.values())
                    log(f"dreaming overlay: {n8} FP8 experts in {len(self._dream_fp8)} layers, "
                        f"{nb / 1e9:.2f} GB this rank, in {time.perf_counter() - t0:.1f}s")
            self.store = EX.ExpertStore(model_dir, index, self.arena, self.args.n_layers, transient_slots=transient_slots,
                                        io_threads=io_threads, ep=self.ep, overlay=overlay)
            self.dspark_reader = self.store
        if self.resident_budget == 'auto':
            from engine.expert_budget import auto_budget
            self.resident_budget = auto_budget(self.store.lru_slots, layers=self.args.n_layers,
                                               experts=self.args.n_routed_experts,
                                               topk=self.args.n_activated_experts)
            total = self.args.n_layers * self.args.n_routed_experts
            log(f"DSV41_RESIDENT_EXPERTS=auto: {self.resident_budget} of {total} routed experts resident "
                f"({100 * self.resident_budget / total:.1f}%) in {self.store.lru_slots} LRU slots")
        if self.exl3_exact and self.resident_budget != self.args.n_layers * self.args.n_routed_experts:
            # A narrow slot cannot take a 3-bit expert, so slots are not interchangeable: only a store
            # that never swaps (everything resident) may use them.
            raise ValueError("DSV41_EXL3_EXACT_SLOTS requires every routed expert resident: leave "
                             "DSV41_RESIDENT_EXPERTS unset (auto)")
        # load the DSpark experts (all 3 x 128 resident in their own arena)
        for k in range(3):
            for e in range(128):
                w1, s1, w2, s2, w3, s3 = self.dspark_reader.read_expert(40 + k, e, prefix=f"mtp.{k}.ffn.experts.{e}.")
                self.W.dspark_arena.load_slot(k * 128 + e, w1.view(*EX.W13_SHAPE), s1.view(*EX.S13_SHAPE),
                                              w2.view(*EX.W2_SHAPE), s2.view(*EX.S2_SHAPE), w3.view(*EX.W13_SHAPE),
                                              s3.view(*EX.S13_SHAPE))
        log("DSpark experts resident")

        self.model = Model(self.W, self.store, self.caches, self.moe_fn, act_quant=act_quant)
        self.model.dream_fp8 = getattr(self, "_dream_fp8", None)
        self.model.hash_state = make_hash_state(model_dir, self.tokenizer, max_seq, device)
        # Vision tower: present only if the checkpoint has one, and skippable for a text-only
        # server that would rather keep the 0.9 GB (about 2.5 experts per layer at fp4).
        from engine.vision import make_vision
        self.vision_mode = os.environ.get('DSV41_VISION_MODE', 'replicated')
        self.vision = make_vision(model_dir, self.ep, device,
            enabled=os.environ.get('DSV41_VISION', '1') == '1', mode=self.vision_mode, log=log)
        self.model.vision = self.vision
        if self.vision is not None:
            log(f"vision enabled: {self.vision_mode} "
                f"({self.vision.cfg['vision_n_layers']} layers, "
                f"{self.vision.cfg['vision_dim']} dim; "
                f"weights on {'rank 1' if self.vision_mode == 'peer' else 'both ranks'})")
        from engine.engram_cache import cache_budget_mb
        cache_mb = cache_budget_mb(self.ep.rank)
        if cache_mb and os.environ.get('DSV41_ENGRAM_MMAP', '1') != '1':
            raise ValueError('Engram row cache requires mmap row reads')
        cache_bytes = cache_mb * 1024**2 // max(1, len(self.args.engram_layer_ids))
        self.tables = {L: EngramTable(model_dir, index, L, device, cache_bytes=cache_bytes)
                       for L in self.args.engram_layer_ids}
        if cache_mb:
            log(f'Engram packed row cache: {cache_mb} MiB total per rank')
        for _t in self.tables.values():
            _t.ep = self.ep          # enables the DSV41_ENGRAM_ROW_SPLIT read split (prefill only)
        if self.ep.active and self.tables and next(iter(self.tables.values())).row_split:
            log("engram row split on: each rank reads uniq %% world == rank, one all-reduce rebuilds")
        from concurrent.futures import ThreadPoolExecutor as _TPE
        # one worker per (engram layer x read-ahead depth) so chunk k+1's reads are not queued
        # behind chunk k's; the NVMe parallelism itself comes from each table's own 32-thread pool
        self.eg_pool = _TPE(len(self.args.engram_layer_ids)
                            * max(1, int(os.environ.get("DSV41_ENGRAM_DEPTH", "2"))))
        self.model.engram_rows = lambda L, h: self.tables[L].rows(h)
        self.model.engram_prefetch = lambda hashes: prefetch_rows(
            self.tables, self.eg_pool, hashes, self.args.engram_layer_ids)
        self.prune_keep = prune_keep
        self._prune_trace = None
        self._token_types = self._images = None   # set per request by the VL path
        self._swap_baseline = 0.0   # recorded demand at the last applied adaptation
        self._requests_since_plan = 0   # request-unit swap trigger; see maintain_demand()
        # Bumped by every applied adaptation. Both ranks apply the same plan at the same point,
        # so this counts identically on the pair and names the routing regime a piece of cached
        # state was computed under. Reported on /health; see _save_prefix for why a cached prefix
        # from an older generation is KEPT rather than invalidated.
        self.expert_generation = 0
        # One exact prompt-prefix snapshot.  The large compressed KV prefix remains append-only in
        # ``self.caches``; this object only protects the small state decode overwrites.
        self._prefix_cache = None
        # {prefix_len: snapshot} at chunk boundaries of the current prompt, newest kept up to
        # DSV41_PREFIX_SNAPSHOTS. The full-prompt snapshot above is the exact-extension fast path;
        # these are the fallback when a changed token stops the prompt from extending it.
        self._prefix_snapshots: dict = {}
        self._expert_seed = None
        self.expert_seed_sha256 = None
        self.prefix_disk = None
        self._prefix_route = ''
        if prune_keep and prune_keep < 1.0:
            # pruned, all-resident mode: per layer only the top-N experts (by trace frequency) stay
            # routable, and exactly those are warm-started, so decode never touches NVMe
            cc, cg = EX.category_counts(trace_stats, "coding"), EX.category_counts(trace_stats, "general")
            assert len(cc) == 40 and len(cg) == 40, "pruned mode needs the per-layer trace npz files next to trace_stats"
            # How the two workloads are combined into one ranking. "sum" (the 0.2.0-wip default)
            # adds the per-layer-normalized frequencies, which favours experts moderately used by
            # both and drops each workload's specialists -- the coding and prose top sets overlap by
            # a Jaccard of only 0.18-0.31, so that is most of the routing. Prose suffered for it:
            # at keep 31 % a story prompt degenerated into a repeated phrase while the same budget
            # ranked on prose alone wrote clean text (NOTES 2026-09-12). "max" keeps an expert that
            # matters to EITHER workload, which is what a general-purpose server needs.
            rank = self.expert_profile.rank
            if rank == "sum":
                counts = {L: cc[L] / cc[L].sum() + cg[L] / cg[L].sum() for L in range(40)}
            elif rank == "max":
                counts = {L: np.maximum(cc[L] / cc[L].sum(), cg[L] / cg[L].sum()) for L in range(40)}
            else:
                raise ValueError(f"unknown DSV41_PRUNE_RANK {rank!r} (max | sum)")
            # Adapt the ranking to what this server has actually been asked for. The trace ships
            # with the model; the demand DB is this box's own history and accumulates across
            # restarts, so a server used mostly for one kind of work drifts towards the experts
            # that work needs. DSV41_PRUNE_ADAPT=0 pins the shipped ranking.
            self._prune_trace = {L: v.copy() for L, v in counts.items()}
            log(ADAPT.describe())
            log(f"observed routing metric: {ADAPT.metric}")
            db = load_prune_db() if ADAPT.use_db else None
            # Rank 0 alone chooses the cold-start seed. Carry it in the existing
            # ranking broadcast so peer-local DB files cannot diverge the map.
            seed_error = None
            if self.ep.rank == 0:
                try:
                    from engine.expert_seed import load_seed
                    # DSV41_EXPERT_SEED: 1 the bundled seed, 0 none, else a seed file's path.
                    seed_opt = os.environ.get('DSV41_EXPERT_SEED', '1')
                    self._expert_seed = load_seed(
                        PRUNE_DB, enabled=(self.dynamic_experts and ADAPT.use_db and seed_opt != '0'),
                        request_unit=ADAPT.request_unit, metric=ADAPT.metric,
                        layers=self.args.n_layers, experts=self.args.n_routed_experts,
                        path=None if seed_opt == '1' else seed_opt)
                    if self._expert_seed is not None:
                        db = (self._expert_seed['counts'], self._expert_seed['mass'])
                except Exception as exc:
                    seed_error = str(exc)
            if db is not None:
                prior = ADAPT.prior
                counts, w = blend_demand(counts, db, prior)
                origin = 'bundled expert seed' if self._expert_seed is not None else PRUNE_DB
                log(f"prune ranking adapted from {origin}: observed weight {w:.1%} "
                    f"({db[0].sum():,.0f} recorded routing slots)")
            self.prune_select = prune_select
            # ONE authority for the keep set. results/ is bind-mounted per box and sync-peer.sh
            # copies only the trace, so a demand DB that exists on rank 0 and not on rank 1 gave
            # the two ranks DIFFERENT routable experts -- their routers masked different logits,
            # and the MoE all-reduce summed partials from two different expert selections. It
            # surfaced only because the swap validator refused to evict an expert rank 1 had
            # never made resident. Broadcasting the ranking (not the masks) keeps one code path:
            # build_keep_masks is deterministic, so identical input gives identical masks.
            packet = self.ep.broadcast_obj(dict(counts=counts, seed=self._expert_seed,
                                                error=seed_error))
            if packet['error'] is not None:
                raise ValueError('cold-start expert seed: ' + packet['error'])
            counts, self._expert_seed = packet['counts'], packet['seed']
            if self._expert_seed is not None:
                self.expert_seed_sha256 = self._expert_seed['sha256']
                log(f"cold-start expert seed {self.expert_seed_sha256[:12]}: "
                    "no local history; normal adaptation will continue")
            if self.dynamic_experts:
                from engine.global_residency import initial_selection
                scores = {L:v/(v.sum() or 1.) for L,v in counts.items()}
                budget = self.resident_budget or (max(self.args.n_activated_experts,
                    int(np.ceil(prune_keep*self.args.n_routed_experts)))*self.args.n_layers)
                from engine.expert_seed import seed_selection
                keep = seed_selection(self._expert_seed, budget, self.dynamic_layer_floor)
                if keep is None:
                    keep = initial_selection(scores,budget,self.dynamic_layer_floor)
                masks = {L:torch.zeros(self.args.n_routed_experts,dtype=torch.bool,device=device) for L in keep}
                for L,ids in keep.items():
                    masks[L][ids] = True
            else:
                masks, keep = build_keep_masks(counts, prune_keep, prune_select, device,
                                              layer_counts=self.prune_layer_counts)
            self._dream_pins = R_dream.pins()
            if self._dream_pins:
                # A dreaming overlay's experts stay resident and routable: the router can never pick
                # a pruned expert, so its lesson would silently not exist. Same input on both ranks
                # (the manifest digest is in the boot guard), so the same masks.
                keep = R_dream.pin_keep(keep, counts, self._dream_pins)
                masks = {L: torch.zeros(self.args.n_routed_experts, dtype=torch.bool, device=device) for L in keep}
                for L, ids in keep.items():
                    masks[L][ids] = True
                log(f"dreaming overlay: {sum(map(len, self._dream_pins.values()))} experts pinned resident")
            n_keep = max(len(v) for v in keep.values())
            ranked = [(float(counts[L][e]), L, int(e)) for L, ks in keep.items() for e in ks]
            ranked.sort(reverse=True)
            ranked = [(L, e) for _, L, e in ranked]
            self.model_prune_mask = masks
            per_layer = [len(keep[L]) for L in range(40)]
            if sim_bits:
                # simulated low-bit format on the coldest `sim_cold_frac` of the KEPT experts of every layer
                from engine.codebook_sim import CodebookSim
                self.store.requant_sims = {sim_bits: CodebookSim(sim_bits, device)}
                pol = {}
                for L, ks in keep.items():
                    n_cold = int(round(sim_cold_frac * len(ks)))
                    for e in ks[len(ks) - n_cold:]:
                        pol[(L, int(e))] = sim_bits
                self.store.requant = pol
                log(f"simulated {sim_bits}-bit codebook format on {len(pol)} of {len(ranked)} kept experts "
                    f"(coldest {sim_cold_frac:.0%} per layer)")
            if self.sim_cb2_frac > 0:
                # Quality-only simulation of a 2-bit tier INSIDE the CB3 arena: the coldest
                # `sim_cb2_frac` of every layer's kept set get a 4-of-16 row codebook repeated up to
                # the slot's 8 entries, so the slot's bytes and the CB3 kernel are untouched and the
                # arithmetic is a real 2-bit format's. Nothing is saved -- what it answers is whether
                # saving it would be worth building.
                assert self.expert_format == "cb3", "--sim-cb2-frac needs --expert-format cb3"
                from engine.codebook_sim import CodebookSim
                self.store.cb_sims = {2: CodebookSim(2, device)}
                pol2 = {}
                for L, ks in keep.items():
                    n_cold = int(round(self.sim_cb2_frac * len(ks)))
                    for e in ks[len(ks) - n_cold:]:
                        pol2[(L, int(e))] = 2
                self.store.cb_bits = pol2
                log(f"simulated 2-bit codebook on {len(pol2)} of {len(ranked)} kept experts "
                    f"(coldest {self.sim_cb2_frac:.0%} per layer); slot size unchanged")
            owned_counts = require_pruned_residency(
                ranked, 1 if self.ep.tensor_parallel else self.ep.world, self.store.lru_slots, prune_keep,
                self.store.transient_slots)
            n_here = owned_counts[0 if self.ep.tensor_parallel else self.ep.rank]
            if self.ep.active:
                log(f"EP2 rank {self.ep.rank}: {n_here} of the {len(ranked)} kept experts are owned here")
            self._kept_here = n_here   # the resident test below needs the real count, see there
            log(f"pruned mode ({prune_select}): keep {prune_keep:.2f}, {len(ranked)} experts total "
                f"({min(per_layer)}-{max(per_layer)} per layer), {len(ranked) * self.expert_bytes / 1e9:.1f} GB")
        else:
            self.model_prune_mask = None
            ranked = (EX.rank_from_trace(trace_stats, profile=self.hot_profile) if trace_stats
                      else [(L, e) for e in range(384) for L in range(40)])
        self.model.prune_mask = self.model_prune_mask
        self.resident_layer_counts = ([int(self.model_prune_mask[L].sum()) for L in sorted(self.model_prune_mask)]
                                      if self.model_prune_mask else None)
        from engine.expert_profiles import mask_digest
        self.initial_keep_sha256 = mask_digest(self.model_prune_mask)
        self.model.stream_layers = self.stream_layers
        self.model.stream_keep = {L: torch.ones(self.args.n_routed_experts, dtype=torch.bool, device=device)
                                  for L in self.stream_layers}
        from engine.user_prompt import UserPrompt
        self.user_prompt = UserPrompt(self.user_prompt_stream, self.args.n_layers,
                                      self.args.n_routed_experts, device, self.user_prompt_max_loads, self.dynamic_experts)
        if self.dynamic_experts and (not self.ep.tensor_parallel
                                    or (self.kernel != 'triton-fp4' and exl3_cls is None)):
            raise ValueError('dynamic expert allocation requires native FP4 tensor parallelism')
        if self.predictive.mode != 'off':
            if (not self.dynamic_experts or not self.ep.tensor_parallel or not ADAPT.request_unit
                    or not ADAPT.record or not PREFILL_SWAP or self.user_prompt_stream
                    or self.stream_layers or self.critical.enabled):
                raise ValueError('predictive prefill requires dynamic TP2 request-unit adaptation without streaming')
            if self.predictive.backend == 'probe':
                # Concurrent lanes: probes run inside one lane's prefill, which is sequential.
                if self.layer_ablation.path:
                    raise ValueError('computed prefill probes require no layer ablation')
                if self.predictive.probe.two_pass_below and not self.swa_replay:
                    raise ValueError('two-pass suffix discovery requires SWA replay')
        self.model.user_prompt = self.user_prompt
        if self.stream_layers or self.user_prompt_stream:
            if not (self.ep.tensor_parallel and self.kernel == 'triton-fp4'
                    and getattr(self.store.arena, 'tp_output', False)):
                raise ValueError('layer streaming requires native FP4 output-layout TP')
            from engine.layer_stream import validate_capacity
            if not self.dynamic_experts:
                validate_capacity(self.stream_layers or range(self.args.n_layers), self.model_prune_mask, transient_slots,
                                  self.args.n_routed_experts)
        self.model.critical_stream = self.critical
        if self.critical.enabled and not (self.ep.tensor_parallel and self.kernel == 'triton-fp4'):
            raise ValueError('critical prefill prototype requires native FP4 tensor parallelism')
        if self.ep.active:
            # Defence in depth for the failure above: the routable set drives the router on BOTH
            # ranks, so a difference is not a slow answer, it is two models pretending to be one.
            # Cheap to verify once at boot, and impossible to notice later without a validator.
            # Cardinalities alone cannot detect different expert IDs at the same budget.
            mine = self.initial_keep_sha256
            theirs = self.ep.broadcast_obj(mine)
            if mine != theirs:
                raise RuntimeError(
                    f"EP2 rank {self.ep.rank}: prune mask differs from rank 0 "
                    f"({mine} vs {theirs}); the two routers would mask different experts")
            log(f"EP2 prune mask agrees across ranks (checksum {mine})")

            # Everything else that must be identical on both ranks. Each of these has its own way
            # of going wrong quietly: a format or TP split that differs makes one rank compute a
            # partial where its peer computes a whole, and the first all-reduce then sums
            # mismatched tensors -- no error, just wrong numbers. Checked here because a boot-time
            # refusal is the only cheap moment; at runtime it is invisible without a validator.
            from engine.fastdecode import T_DRAFT, l2pf
            from engine.spec_depth import ConfidenceDepthPolicy
            from engine.lookup_draft import VERSION as LOOKUP_VERSION
            from engine.tensor_parallel import TP_DRAFT_HEAD_VERSION
            cfg = {
                "decode_probe_version": 1,
                "router_fused_tail": os.environ.get("DSV41_ROUTER_FUSED_TAIL", "0"),
                "router_fused_tail_version": 1,
                "router_bf16": os.environ.get("DSV41_ROUTER_BF16", "0"),
                "router_bf16_version": 1,
                "verify_odd": os.environ.get("DSV41_VERIFY_ODD", "0"),
                "verify_odd_version": 1,
                "confidence_cost_refresh": int(os.environ.get("DSV41_CONF_COST_REFRESH", "0")),
                "graphs_max": int(os.environ.get("DSV41_GRAPHS_MAX", "8")),
                "graph_pool_rotation_version": 2,
                "prefill_budget": vars(self.prefill_budget),
                "prefill_budget_version": 1,
                "draft_tokens": T_DRAFT if self.spec else 0,
                "expert_format": self.expert_format,
                # Rank-invariant: both ranks must have been built from the same source manifest.
                # The pack HEADER sha is per rank (it carries rank/world), so it belongs in health,
                # not here -- putting it in the guard is a silent pair disagreement.
                "expert_pack_source": (self.store.pack.header.get("source_sha256", "")
                                       if hasattr(self.store, "pack") else ""),
                "tp_dense": os.environ.get("DSV41_TP_DENSE", "0"),
                "tp_experts": self.ep.tensor_parallel,
                "tp_experts_version": 1,
                "prefill_dual_rail": self.ep.prefill_dual_rail,
                "collective_rails_version": DT.collective_rails.VERSION,
                "tp_expert_layout": os.environ.get('DSV41_TP_EXPERT_LAYOUT', 'output'),
                "tp_linear_layout": os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output'),
                "tp_expert_reduce": os.environ.get('DSV41_TP_EXPERT_REDUCE', 'all_reduce'),
                "tp_attention": os.environ.get('DSV41_TP_ATTN', '0'),
                "tp_draft_attention": os.environ.get('DSV41_TP_DRAFT_ATTN', '0'),
                "draft_markov_topk": int(os.environ.get('DSV41_DRAFT_MARKOV_TOPK', '0')),
                "draft_markov_local": os.environ.get('DSV41_DRAFT_MARKOV_LOCAL', '0'),
                "draft_markov_tp": os.environ.get('DSV41_DRAFT_MARKOV_TP', '0'),
                "lean_rms_fused": os.environ.get('DSV41_LEAN_RMS_FUSED', '0'),
                "hc_front_fused": os.environ.get('DSV41_HC_FRONT_FUSED', '0'),
                "tp_head": os.environ.get('DSV41_TP_HEAD', '0'),
                "tp_embed": os.environ.get('DSV41_TP_EMBED', '0'),
                "tp_draft_experts": self.tp_draft_experts,
                "tp_draft_dense": self.tp_draft_dense,
                "prune_miss_side": os.environ.get("DSV41_PRUNE_MISS_SIDE", "1"),
                "tp_dense_rounding_version": 2,
                "prune_keep": float(prune_keep or 0.0),
                # Layer allocation changes routing even at the same total arena budget.
                "prune_select": prune_select,
                **self.expert_profile.boot_fields(),
                "initial_keep_sha256": self.initial_keep_sha256,
                "vision_mode": self.vision_mode,
                "vision_enabled": self.vision is not None,
                "vision_peer_version": 1,
                "expert_seed_version": 1,
                "expert_seed_enabled": os.environ.get('DSV41_EXPERT_SEED', '1'),
                "expert_seed_sha256": self.expert_seed_sha256,
                "prune_layer_counts": self.prune_layer_counts,
                "stream_layers": tuple(sorted(self.stream_layers)),
                "layer_stream_version": 2,
                "user_prompt_stream": self.user_prompt_stream,
                "user_prompt_policy_version": 6,
                "user_prompt_max_loads": self.user_prompt_max_loads,
                "dynamic_experts": self.dynamic_experts,
                "resident_budget": self.resident_budget,
                "dynamic_experts_version": 4,
                "dynamic_layer_floor": self.dynamic_layer_floor,
                "prune_layer_weights": self.prune_layer_weights,
                "layer_ablation_file": self.layer_ablation.path,
                "layer_ablation_version": 1,
                "arena_slots": int(getattr(self.store.arena, "slots", 0)),
                "scale_pack": self.scale_pack.boot_fields() if self.scale_pack is not None else None,
                "scale_codec_source": hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes()
                    for name in ('scale_layout.py', 'scale_pack.py', 'scale_load_cuda.py'))).hexdigest(),
                "spec": bool(self.spec),
                "max_concurrency": int(os.environ.get("DSV41_MAX_CONCURRENCY", "1")),
                # Lane caches size the batched step's per-lane indexer buckets and graph keys.
                "lane_max_seq": os.environ.get("DSV41_LANE_MAX_SEQ", "65536"),
                # Graph releases must happen at the same step on both ranks.
                "batch_idle_steps": os.environ.get("DSV41_BATCH_IDLE_STEPS", "0"),
                "lane_reserve_gb": os.environ.get("DSV41_LANE_RESERVE_GB", ""),
                # 3: graphs kept across requests; batched rounds left out of the depth policy's
                # step times.
                "batch_decode_version": 3,
                "max_seq": int(self.max_context),
                "topk": int(self.args.n_activated_experts),
                "index_topk": self.args.index_topk,
                "candidate_topk_blocks": self.args.candidate_topk_blocks,
                "candidate_block_size": self.args.candidate_block_size,
                "candidate_source_layer": self.args.candidate_source_layer,
                **self.experiments.boot_fields(),
                # Dense/attention re-quantization, the LM head's stored format and the drafter's
                # cheaper head. All are read from the environment at load and change numerics, so
                # a pair started with them differing would compute different logits or draft
                # different tokens with nothing raising.
                "dense_fp4": ",".join(sorted(R.dense_fp4_groups())) or "off",
                "fp4_dense_split": os.environ.get("DSV41_FP4_DENSE_SPLIT", "auto"),
                # The transport for the decode-sized all-gathers. A pair split across this would route
                # different bytes; the RoCE path also agrees its own settings at setup, but the guard
                # refuses the mismatch at boot instead of after the first probe.
                "comm_backend": os.environ.get("DSV41_COMM_BACKEND", "roce"),
                "roce_max_kb": os.environ.get("DSV41_ROCE_MAX_KB", "256"),
                "roce_hcas": os.environ.get("DSV41_ROCE_HCAS", "2"),
                # The abliteration overlay changes layers 10-35 attn.wo_b, so a pair must load the
                # same file -- the digest, not just the switch, is what both ranks compare.
                "ablate_wob": R_ablit.enabled(),
                "ablate_wob_digest": R_ablit.digest(),
                # A dreaming overlay replaces routed experts; both ranks hold half of each, so a
                # pair on different generations sums halves of different experts. The digest
                # pins the manifest, and load() has already hashed every file against it.
                # the drafter's sampled q (engine/fastdecode.py::DRAFT_NUCLEUS): both ranks run
                # the draft graph, so a pair split across it would verify different proposals
                "draft_nucleus": (os.environ.get("DSV41_DRAFT_NUCLEUS", "0") + "/"
                                  + os.environ.get("DSV41_DRAFT_NUCLEUS_K", "512")),
                "dream_overlay": R_dream.digest(),
                # the FP8 tier changes which kernels compute those experts; the digest pins the
                # manifest, this pins that both ranks actually loaded the tier
                "dream_fp8": sum(len(d.ids) for d in (getattr(self, "_dream_fp8", None) or {}).values()),
                "head_fmt": R.head_fmt(),
                "head_kernel": head_kernel(),
                "head_kernel_version": HEAD_KERNEL_VERSION,
                "draft_head_fmt": R.draft_head_fmt(),
                "markov_head_fmt": _FD.MARKOV_HEAD_FMT,
                "tp_draft_head": tp_draft_head_enabled(),
                "tp_draft_head_version": TP_DRAFT_HEAD_VERSION,
                # A rank drafting from a different id list proposes different tokens with nothing
                # in the log; the digest covers the file, the row count and the split.
                "draft_vocab": getattr(self.W, "draft_vocab_digest", None),
                "draft_vocab_version": _DRAFT_VOCAB.VERSION,
                # Lossy decode pruning: a rank pruning differently computes a different routed sum
                # and the pair diverges silently (engine/expert_topp.py).
                "expert_topp": _ETP.config(),
                "expert_topp_version": _ETP.VERSION,
                # DSV41_HOT_READS changes which experts a decode row uses (engine/hot_reads.py). From
                # the environment: this guard runs before the FastDecoder that holds the settings.
                "hot_reads": _hot_boot_fields(self.args),
                "lookup_draft_enabled": LOOKUP_DRAFT_ENABLED,
                "lookup_draft_ngram": LOOKUP_DRAFT_NGRAM,
                "lookup_draft_candidates": LOOKUP_DRAFT_CANDIDATES,
                "lookup_draft_version": LOOKUP_VERSION,
                "draft_bypass": self.draft_bypass_enabled,
                "draft_bypass_version": draft_bypass.VERSION,
                "hc_kernel": R.HC_KERNEL,
                "hc_prec": R.HC_PREC,
                "dense_dequant_cache": os.environ.get("DSV41_DENSE_DEQUANT_CACHE", "0") == "1",
                "fp4_dot_scaled": self.fp4_dot_scaled,
                "fp4_prefill_dot_scaled": self.fp4_prefill_dot_scaled,
                "fp4_prefill_scaled_tiles": os.environ.get("DSV41_FP4_PREFILL_SCALED_TILES", "0"),
                "fp4_cuda": self.fp4_cuda,
                "fp4_cuda_relaxed": self.fp4_cuda_relaxed,
                "fp4_cuda_v2": self.fp4_cuda_v2,
                "fp4_cuda_down_rows": self.fp4_cuda_down_rows,
                "fp4_cuda_source": self.fp4_cuda_source,
                "batched_verify": self.batched_verify,
                "batched_verify_source": self.batched_verify_source,
                "kv_cache_qdq": bool(M_.KV_CACHE_QDQ),
                "packed_kv": bool(M_.PACKED_KV),
                "hc_ops": bool(M_.HC_OPS),
                "hc_fused": bool(M_.HC_FUSED),
                "hc_mm_tile": R.HC_MM_TILE,
                "decode_shared_overlap": os.environ.get("DSV41_DECODE_SHARED_OVERLAP", "0") == "1",
                "l2pf_bytes": l2pf.budget_bytes(),
                "l2pf_version": l2pf.VERSION,
                "l2pf_mode": l2pf.MODE,
                "l2pf_attn_mb": int(os.environ.get("DSV41_L2PF_ATTN_MB", "0")),
                "l2pf_qkv_mb": int(os.environ.get("DSV41_L2PF_QKV_MB", "0")),
                "l2pf_sh_mb": int(os.environ.get("DSV41_L2PF_SH_MB", "0")),
                "l2pf_pace_gbps": l2pf.PACE_GBPS,
                # Which verify widths exist (buffers, graphs). The depth each step uses is
                # decided on rank 0 and broadcast with the control flag, so the other
                # DSV41_BLOCK_DYNAMIC_* policy knobs only need to be right on rank 0.
                "block_dynamic": os.environ.get("DSV41_BLOCK_DYNAMIC", ""),
                # Confidence adds a second control broadcast after drafting on all requests.
                # A split pair must fail at boot, before either rank can enter that call.
                "block_confidence_v1": os.environ.get("DSV41_BLOCK_CONFIDENCE", "0"),
                "block_confidence_sampling_version": ConfidenceDepthPolicy.VERSION,
                "spec_conf": os.environ.get("DSV41_SPEC_CONF", "0"),
                # Decode fp8 scheduling and projection merging are meant to be bit-identical to
                # the defaults, but that is a measured property of this GPU and compiler, not a
                # guarantee; a pair split across them must refuse to start.
                "fp8_decode_block_n": os.environ.get("DSV41_FP8_DECODE_BLOCK_N", "auto"),
                "fp8_decode_block_k": os.environ.get("DSV41_FP8_DECODE_BLOCK_K", "128"),
                "fp8_decode_warps": os.environ.get("DSV41_FP8_DECODE_WARPS", "4"),
                "fp8_decode_pipeline": os.environ.get("DSV41_FP8_DECODE_PIPELINE", "0") == "1",
                "decode_merged_proj": os.environ.get("DSV41_DECODE_MERGED_PROJ", "1") == "1",
                "decode_lean": os.environ.get("DSV41_DECODE_LEAN", "1") == "1",
                "decode_lean_rope": os.environ.get("DSV41_DECODE_LEAN_ROPE", "1") == "1",
                "decode_lean_softmax": os.environ.get("DSV41_DECODE_LEAN_SOFTMAX", "1") == "1",
                "decode_lean_moe": os.environ.get("DSV41_DECODE_LEAN_MOE", "1") == "1",
                "decode_lean_attn": os.environ.get("DSV41_DECODE_LEAN_ATTN", "1") == "1",
                "exl3_merged": (os.environ.get("DSV41_EXL3_MERGED", "1") == "1"
                                if self.expert_format == "exl3" else None),
                "exl3_cuda_source": self.exl3_cuda_source,
                # prefill tile vs grouped kernel sum K in different orders; BM/tile shape do not
                "exl3_prefill_kernel": (os.environ.get("DSV41_EXL3_PREFILL_KERNEL", "tile")
                                        if self.expert_format == "exl3" else None),
                "early_verify": EARLY_VERIFY,
                "early_draft": EARLY_DRAFT,
                "exl3_exact_slots": list(self.exl3_exact) if self.exl3_exact else None,
                "exl3_dense": (self.exl3_dense.source_sha256 if self.exl3_dense is not None else None),
                "exl3_dense_version": 1,
                "exl3_draft_dense": (self.exl3_draft.source_sha256 if self.exl3_draft is not None else None),
                "exl3_draft_dense_version": 1,
                "attn_staged": os.environ.get("DSV41_ATTN_STAGED", "0"),
                "prune_miss_fused": os.environ.get("DSV41_PRUNE_MISS_FUSED", "1") == "1",
                "fp8_act_qdq_fused": os.environ.get("DSV41_FP8_ACT_QDQ_FUSED", "1") == "1",
                "moe_fallback": self.kernel == "dequant-fallback",
                "act_quant": bool(self.act_quant),
                "cycle_break": os.environ.get("DSV41_CYCLE_BREAK", "0"),
                "no_repeat_ngram": os.environ.get("DSV41_NO_REPEAT_NGRAM", "0"),
                # Both halves of the engram split decision. row_split says whether the ranks
                # split rows at all; split_min_rows says at which gather size they start -- and a
                # rank that splits a gather its peer read whole reaches an all-reduce alone and
                # hangs the pair until the process-group timeout.
                "engram_row_split": os.environ.get("DSV41_ENGRAM_ROW_SPLIT", "0"),
                "engram_native": os.environ.get("DSV41_ENGRAM_NATIVE", "0"),
                "engram_direct": os.environ.get("DSV41_ENGRAM_DIRECT", "0"),
                "hc_prefill_fused": os.environ.get("DSV41_HC_PREFILL_FUSED", "0"),
                "prefill_attn_indexed": os.environ.get("DSV41_PREFILL_ATTN_INDEXED", "0"),
                "index_fused": os.environ.get("DSV41_INDEX_FUSED", "0"),
                "engram_cache_v1_mb": os.environ.get("DSV41_ENGRAM_CACHE_MB", "0"),
                "engram_cache_v1_peer_mb": os.environ.get("DSV41_ENGRAM_CACHE_MB_PEER", "inherit"),
                "engram_native_threads": os.environ.get("DSV41_ENGRAM_NATIVE_THREADS", "64"),
                "engram_split_min": os.environ.get("DSV41_ENGRAM_SPLIT_MIN", "4096"),
                # A rank that adapts at the prefill boundary while its peer does not ends up
                # masking a different expert set, so the two replicated routers pick different
                # top-k and the pair emits different tokens with nothing raising. The threshold
                # matters as much as the switch: it decides WHETHER the collective happens.
                # Resolved adaptation settings (engine/adapt_config.py), not the raw variables:
                # the two DSV41_ADAPT_* knobs and the legacy DSV41_PRUNE_* spellings must agree
                # on what they MEAN, e.g. whether the prefill-boundary collective happens.
                **ADAPT.boot_fields(),
                **self.critical.boot_fields(),
                **self.predictive.boot_fields(),
                "prefill_chunk": MAX_CHUNK,
                "prefill_graphs_v2": os.environ.get("DSV41_PREFILL_GRAPHS", "0"),
                "prefill_graph_fixed_routing": os.environ.get("DSV41_PREFILL_FIXED_ROUTING", "1"),
                "prefill_timing": PREFILL_TIMING,
                "prefill_moe_timing": os.environ.get("DSV41_PREFILL_MOE_TIMING", "0"),
                "prefill_swap_routes_version": 1,
                # The prefill indexer's score reduction. The tiled path is meant to select the
                # same positions as the monolithic one, but a per-tile topk can break a tie
                # differently, so a split pair could read different sparse KV with nothing
                # raising. Keep both ranks on the same side of the switch, and on the same score
                # dtype -- bf16 vs fp32 changes the selection the same way.
                "index_stream": os.environ.get("DSV41_INDEX_STREAM", "0"),
                "index_score_bf16": os.environ.get("DSV41_INDEX_SCORE_BF16", "1"),
                # The restored prefix length decides how many prefill chunks (and therefore how
                # many per-layer collectives) this request issues. A rank that keeps a different
                # number of snapshots can resume at a different boundary; keep it in the guard.
                "prefix_snapshots": PREFIX_SNAPSHOTS,
                "prefix_response": os.environ.get('DSV41_PREFIX_RESPONSE', '1'),
                "prefix_cache": os.environ.get('DSV41_PREFIX_CACHE', '1'),
                "prefix_disk": PREFIX_DISK,
                "prefix_disk_version": 4,  # rank-0-authoritative save interval and duplicate skip
                "prefix_disk_min_interval_s": PREFIX_DISK_MIN_INTERVAL_S,
                "prefix_disk_dedup": PREFIX_DISK_DEDUP,
                "prefix_disk_strict": os.environ.get("DSV41_PREFIX_DISK_STRICT", "0"),
            }
            peer_cfg = self.ep.broadcast_obj(cfg)
            diff = {k: (v, peer_cfg.get(k)) for k, v in cfg.items() if peer_cfg.get(k) != v}
            if diff:
                raise RuntimeError(
                    f"EP2 rank {self.ep.rank}: configuration differs from rank 0 in {diff} "
                    f"(mine vs rank 0); the ranks would compute different things and their "
                    f"collectives would sum mismatched tensors")
            log(f"EP2 config agrees across ranks ({len(cfg)} fields: "
                f"{', '.join(sorted(cfg))})")
        if self.predictive.mode != 'off' and self.ep.rank == 0:
            self.predictive.configure(cfg, os.path.join(model_dir, 'inference', 'config.json'),
                                      os.path.join(model_dir, 'tokenizer.json'))
            log(f'predictive prefill: mode={self.predictive.mode}, backend={self.predictive.backend}, '
                f'samples={len(self.predictive.keys)}, '
                f'database_error={self.predictive.io_error}'
                + (f', reset: {self.predictive.reset}' if self.predictive.reset else ''))
        self.store.warm_start(ranked, log=log)
        if PRUNE_MISS:
            # Seed the accumulators from the persisted DB so what they hold is always the whole
            # history, not just this process's slice -- the per-request write-back then replaces
            # the file wholesale. Must precede FastDecoder, which captures graphs that record
            # into these exact tensors.
            # The history, like the initial keep-set, has one authority. This collective is
            # unconditional when recording is enabled, which the boot guard pins on both ranks.
            _db = load_prune_db() if self.ep.rank == 0 else None
            if self.ep.rank == 0 and _db is None and self._expert_seed is not None:
                _db = (self._expert_seed['counts'], self._expert_seed['mass'])
            _db = self.ep.broadcast_obj(_db)
            if _db is not None:
                self.model.load_demand(*_db)
                # the swap trigger measures GROWTH, so the clock starts at what we just loaded --
                # otherwise the first idle tick would fire on the seeded history
                self._swap_baseline = float(_db[0].sum())
                log(f"prune demand DB loaded: {_db[0].sum():,.0f} recorded routing slots")
            else:
                self.model.alloc_prune_miss(self.args.n_routed_experts, self.device)
                log("prune demand DB: starting a new one")
        self.fast = None
        # FastDecoder captures CUDA graphs; a graph-captured NCCL all-reduce (EP2) is its own
        # bring-up (capture-time rendezvous, buffer stability across replays). The skeleton runs
        # the eager Model.forward loop -- correct, slower by this repo's graph-vs-eager delta --
        # and the fast path is the first thing to re-enable once Gate G0 numbers are in.
        if spec and os.environ.get("DSV41_FAST", "1") == "1":
            # EP2 used to be excluded here on the grounds that graph-captured collectives were
            # unwired. Gate G1 (scripts/gate_g1_graph_nccl.py) measured that claim: NCCL
            # all-reduces capture and replay correctly on these boxes, values verified, with
            # TORCH_NCCL_BLOCKING_WAIT either way. The combine now lives inside the captured
            # layer-B graph (engine/fastdecode.py::_layer_b), so the pair gets the same graphed
            # decode as one box. DSV41_FAST=0 is the escape hatch back to eager.
            from engine.fastdecode import FastDecoder
            # The reference MoE host-syncs per layer and cannot be captured; the JIT CUDA kernel
            # can. So graphs follow the kernel, not the format.
            exl3_eager = exl3_cls is not None and not self.exl3_cuda
            self.fast = FastDecoder(self.model, self, use_graphs=(
                os.environ.get("DSV41_GRAPHS", "1") == "1" and not exl3_eager))
            if exl3_eager:
                log("EXL3: CUDA graphs disabled (the reference MoE is not capturable); eager step")
            # Resident (device slot LUT) mode: every routable expert is in the arena, so routing
            # is a device gather and the whole layer captures as one graph.
            #
            # The test is "is every expert this rank must serve actually resident", and under EP2
            # that is NOT the global keep count, nor the global count halved. The parity split is
            # not exactly even -- at keep 0.60 rank 0 owned 4,582 of 9,240 kept pairs, where
            # 9,240/2 = 4,620 -- so dividing by world size sets a threshold this rank can never
            # meet and silently leaves the LUT off. Use the count the pruning block actually
            # computed.
            kept_pairs = getattr(self, "_kept_here", None)
            if kept_pairs is None:
                kept_pairs = self.resident_budget or (self.args.n_layers * max(
                    self.args.n_activated_experts,
                    int(np.ceil((self.prune_keep or 0) * self.args.n_routed_experts))))
            resident = (self.prune_keep and self.prune_keep < 1.0 and
                        len(self.store.lru) >= kept_pairs and
                        os.environ.get("DSV41_LUT", "1") == "1")
            if resident:
                self.fast.build_lut()
                # prefill routes through Model.moe, which takes the same table when it is there
                self.model.slot_lut = self.fast.lut
                if self.ep.active and self.kernel in ("triton-fp4", "exl3-cuda"):
                    # Stable, compact per-layer IDs: resident owned experts plus one null (EXL3 is
                    # TP with no null slot: its entry is -1, which the router never selects).
                    # Built once on the host; serving needs only a fixed-shape device gather.
                    routes = {}
                    for L in range(len(self.W.layers)):
                        owned = sorted((e, slot) for (layer, e), slot in self.store.lru.items() if layer == L)
                        ids = torch.full((384,), len(owned), dtype=torch.int32)
                        for i, (e, _) in enumerate(owned):
                            ids[e] = i
                        null = self.store.null_slot if self.store.null_slot is not None else -1
                        slots = torch.tensor([s for _, s in owned] + [null], dtype=torch.int32)
                        routes[L] = (ids.to(device), slots.to(device))
                    self.model.prefill_routes = routes
            log("fast decode path enabled (CUDA graphs=%s, device slot LUT=%s, ep=%s)"
                % (self.fast.use_graphs, self.fast.lut is not None, self.ep.world))
        if (self.stream_layers or self.user_prompt_stream or self.dynamic_experts) and (self.fast is None or self.fast.lut is None):
            raise ValueError('streaming/dynamic allocation requires fast resident decode')
        # preallocated staging for the lean decode step (see LEAN_STEP): the verify block, the
        # [n_accepted, argmax x 6] readback and its pinned host landing buffer.
        from engine.fastdecode import T_VERIFY as _TV, VERIFY_WIDTHS, DYNAMIC_DEPTHS, CONFIDENCE_DEPTHS
        self._tv = _TV
        self.verify_odd = os.environ.get("DSV41_VERIFY_ODD", "0") == "1"
        self._root_width = 1 if self.verify_odd else 2
        # one (block, readback, pinned host) triple per verify width; the defaults are the widest
        self._vbufs = {T: (torch.empty(T, dtype=torch.long, device=device),
                           torch.empty(T + 1, dtype=torch.long, device=device),
                           torch.empty(T + 1, dtype=torch.long).pin_memory()) for T in VERIFY_WIDTHS}
        self._blk, self._vout, self._vhost = self._vbufs[_TV]
        self.depth_policy = None
        self.confidence_depth_policy = None
        if (DYNAMIC_DEPTHS or CONFIDENCE_DEPTHS) and (exl3_cls is None or self.exl3_cuda):
            # Scope: the single-request greedy/sampled graphed path. Batched serving
            # (engine/batch_decode.py) and the eager path verify T_VERIFY rows unconditionally.
            if not (self.spec and self.fast is not None and self.fast.use_graphs):
                raise ValueError("variable speculative depth needs speculative decoding on the graphed fast path")
            # Concurrent lanes: rank 0's depth reaches rank 1 through the lane's control broadcast,
            # and the batched step verifies a narrower block as masked rows (engine/batch_decode.py).
            from engine.spec_depth import DepthPolicy, ConfidenceDepthPolicy
            self.depth_policy = DepthPolicy(DYNAMIC_DEPTHS or (3, 5))
            if CONFIDENCE_DEPTHS:
                self.confidence_depth_policy = ConfidenceDepthPolicy()
                log(f"confidence depths {self.confidence_depth_policy.depths} at every temperature "
                    "(sampled requests use prefix decisions)")
            log(f"dynamic speculative depth {DYNAMIC_DEPTHS or (3, 5)} (start {self.depth_policy.start}, "
                f"decided every {self.depth_policy.interval} tokens)")
        if (LOOKUP_DRAFT_ENABLED or self.draft_bypass_enabled) and (exl3_cls is None or self.exl3_cuda):
            if not (self.spec and self.fast is not None and self.fast.use_graphs
                    and int(os.environ.get('DSV41_MAX_CONCURRENCY', '1')) == 1):
                raise ValueError('draft experiments require single-request graphed speculation')
            if self.draft_bypass_enabled and self._root_width not in VERIFY_WIDTHS:
                raise ValueError('draft bypass requires its root verification graph')
        from engine.prefill_graphs import enabled as prefill_graphs_enabled, PrefillFFNGraphs
        self.model.prefill_graphs = None
        if prefill_graphs_enabled():
            if not (self.fast is not None and self.fast.lut is not None
                    and self.kernel == 'triton-fp4' and self.ep.tensor_parallel
                    and os.environ.get('DSV41_PREFILL_FIXED_ROUTING', '1') == '1'):
                raise ValueError('prefill graphs require resident FP4 TP and fixed prefill routing')
            if (PREFILL_TIMING
                    or os.environ.get('DSV41_PREFILL_MOE_TIMING', '0') != '0'
                    or int(os.environ.get('DSV41_MAX_CONCURRENCY', '1')) != 1):
                raise ValueError('prefill graphs require concurrency 1, timing instrumentation off')
            self.model.prefill_graphs = PrefillFFNGraphs(self.model, MAX_CHUNK)
            log(f"experimental prefill FFN graphs, rows {self.model.prefill_graphs.rows}")
        torch.cuda.synchronize()
        if self.ep.active:
            # Force the NCCL communicator into existence now instead of inside the first
            # request's first combine: lazy setup is a multi-second stall on the user's first
            # token, and a hang HERE fails loudly before /health ever binds (the launcher's health
            # wait is the right place to watch for it). Both ranks reach this exact point
            # after their warm start, so the dummy combine is symmetric.
            self.ep.combine(torch.zeros(1, 1, dtype=torch.float32, device=device))
            log("EP2 communicator warm (dummy combine done)")
        self.last_stats = {}
        log("ready")

    # ------------------------------------------------------------------ generation
    def _reset(self):
        self.critical.reset()
        self._decode_adapt_checks = []
        self._decode_adapt_passes = []
        c = self.caches
        c.len = 0
        c._chunk_inputs.clear()
        for L in c.pending:
            c.pending[L] = None
        self.model.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "ep_s": 0.0, "ep_calls": 0,
                            "tokens": 0}
        for t in self.tables.values():
            t.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        self.store.stats.update(EX.ZERO_STATS)
        self.model.reset_request_demand()   # request-unit mode: a fresh accumulator per request
        self.model.prefill_moe_timing = None
        if (self.kernel == "triton-fp4" and self.ep.active and
                os.environ.get("DSV41_PREFILL_MOE_TIMING", "0") == "1"):
            from engine.prefill_timing import PrefillMoeTiming
            self.model.prefill_moe_timing = PrefillMoeTiming(self.ep.rank)
        if self.fast is not None:
            self.fast.gpu_timing_reset()
            # DSV41_ROUTE_STATS: per-request counts. Capture's own warm-up runs _layer_a and would
            # otherwise be counted, so the FIRST request after a (re)capture reads high; later ones
            # are clean. No-op when the flag is off.
            self.fast.route_stats_reset()

    def _init_prefix_disk(self):
        """Lazy initialization after loading; all ranks participate even on local I/O errors."""
        if not PREFIX_DISK or getattr(self, '_prefix_disk_initialized', False):
            return
        self._prefix_disk_initialized = True
        ok = True
        try:
            from engine.prefix_persistence import PersistentPrefixes, routing_signature
            self._prefix_route = routing_signature(self)
            self.prefix_disk = PersistentPrefixes(self, log)
        except Exception as exc:
            log(f'prefix disk unavailable ({type(exc).__name__}); disabling on both ranks')
            ok = False
        if not all(self.ep.gather_objects(ok)):
            self.prefix_disk = None

    def _try_restore(self, pc, prompt: list[int]) -> int:
        """Restore one cached snapshot if it is a prefix of ``prompt``; return its length.

        Match tokens AND image inputs before touching state. Placeholder-token equality
        alone says nothing about pixels. Never resume in the middle of an image span.
        """
        if (pc is None
                or os.environ.get("DSV41_PREFIX_CACHE", "1") != "1" or not self.swa_replay
                or len(prompt) < len(pc["ids"]) or tuple(prompt[:len(pc["ids"])]) != pc["ids"]):
            return 0
        media = media_prefix(getattr(self, '_prefix_media', ()), len(pc['ids']))
        if media is None or tuple(pc.get('media', ())) != media:
            return 0
        disk = getattr(self, 'prefix_disk', None)
        if disk is not None and disk.strict and pc.get('route') != self._prefix_route:
            return 0
        c = self.caches
        n = len(pc["ids"])
        for L, values in pc["win"].items():
            c.win[L].index_copy_(0, pc["slots"], values)
        for L, pending in pc["pending"].items():
            c.pending[L] = (None if pending is None else (pending[0].clone(), pending[1].clone()))
        rep = pc["rep"]
        self.model._rep = {k: ([v.clone()] if v is not None else [None])
                           for k, v in rep.items()}
        self.model._rep_end = n
        c.len = n
        return n

    def _restore_prefix(self, prompt: list[int]) -> int:
        """Restore the longest cached prefix ``prompt`` extends, or return zero.

        The full-prompt snapshot is the normal case: the next turn of a conversation extends it
        token-for-token. On a miss -- the motivating one is a prompt template that embeds the
        current wall-clock time, so the same conversation re-sent a minute later differs mid-
        prompt -- the chunk-boundary snapshots (DSV41_PREFIX_SNAPSHOTS) are tried longest-first.
        A shorter restore is still correct: every token up to that boundary is identical, so the
        state is too, and the remaining suffix is prefilled normally.

        Restoring to ``n`` invalidates any snapshot past ``n``, because the prefill that follows
        rewrites the compressed KV rows beyond ``n`` from the new tokens; their backing no longer
        matches the state those snapshots were taken from.
        """
        pc = self._prefix_cache
        n = self._try_restore(pc, prompt)
        if n == 0 and self._prefix_snapshots:
            for L in sorted(self._prefix_snapshots, reverse=True):
                n = self._try_restore(self._prefix_snapshots[L], prompt)
                if n:
                    break
        if n == 0:
            self._prefix_cache = None
            return 0
        if pc is not None and len(pc["ids"]) > n:
            self._prefix_cache = None
        for L in [L for L in self._prefix_snapshots if L > n]:
            del self._prefix_snapshots[L]
        return n

    def _snapshot_prefix(self, prefix_ids, P: int):
        """Clone the small prefix state at position ``P``, or return None when caching is off."""
        if (os.environ.get("DSV41_PREFIX_CACHE", "1") != "1"
                or not self.swa_replay or P <= 0):
            return None
        media = media_prefix(getattr(self, '_prefix_media', ()), P)
        if media is None:
            return None
        c, m = self.caches, self.model
        lo = max(0, P - self.args.window_size)
        slots = (m._positions[lo:P] % M_.RING).clone()
        h, pre, topk, cand, _ = m._rep_tail()
        pending = {L: (None if value is None else (value[0].clone(), value[1].clone()))
                   for L, value in c.pending.items()}
        return {
            # The serving path already owns these ids as a host list.  Copying the CUDA ``ids``
            # tensor back here forced a full-stream synchronisation at every cache miss and made
            # queued encoder work appear as a prefix-cache cost.
            "ids": tuple(prefix_ids),
            "media": media,
            "route": getattr(self, '_prefix_route', ''),
            "slots": slots,
            "win": {L: c.win[L][slots].clone()
                    for L in range(self.args.candidate_source_layer + 1)},
            "pending": pending,
            "rep": {"h": h.clone(), "pre_mix": pre.clone(), "topk": topk.clone(),
                    "cand": None if cand is None else cand.clone()},
        }

    def _save_prefix(self, prompt_ids, P: int) -> None:
        """Snapshot the exact encoder state at the prompt boundary before decode mutates it."""
        self._prefix_cache = self._snapshot_prefix(prompt_ids, P)

    @torch.inference_mode()
    def cache_response(self, prompt, response):
        """Post-response encoder-only replay; caller owns the server/peer command lock.

        Restore the INPUT boundary, not speculative decode state. Both ranks agree
        on admission before any model collective. No sampling or adaptation here.
        """
        with self.lock:
            base = self._prefix_cache
            P, end = len(prompt), len(prompt) + len(response)
            eligible = (os.environ.get('DSV41_PREFIX_RESPONSE', '1') == '1'
                        and os.environ.get('DSV41_PREFIX_CACHE', '1') == '1'
                        and self.swa_replay and bool(response)
                        and end + 8 <= self.max_context and self._images is None
                        and base is not None and base['ids'] == tuple(prompt)
                        and (self.prefix_disk is None or not self.prefix_disk.strict
                             or base.get('route') == self._prefix_route))
            if not all(self.ep.gather_objects(bool(eligible))):
                return {'status': 'skipped'}
            started = time.perf_counter()
            ids_host = list(prompt) + list(response)
            restored = self._try_restore(base, ids_host)
            if not all(self.ep.gather_objects(restored == P)):
                raise RuntimeError('response prefix restore disagrees across ranks')
            ids = torch.tensor(ids_host, dtype=torch.long, device=self.device)
            m = self.model
            # Rebuild n-gram history, including n-grams crossing the input boundary.
            hashes = m.hash_state(ids[None], 0)[0] if m.hash_state is not None else None
            spans = self._prefill_spans(end, P)
            ra = self._engram_readahead(ids, end, None, spans, hashes_t=hashes)
            previous = getattr(m, '_prefix_replay_only', False)
            m._prefix_replay_only = True
            self._prefix_snapshots[P] = base  # retain input fallback if chat serialization changes
            try:
                with ra if ra is not None else nullcontext():
                    for s, e in spans:
                        kw = {'hashes': hashes[s:e]} if hashes is not None else {}
                        if ra is not None:
                            kw = {'hashes': ra.hashes_t[s:e], 'get_rows': ra.rows_for(s)}
                        m.forward(ids[s:e], s, prefill=True, encoder_only=True,
                                  need_logits=False, **kw)
                        if ra is not None:
                            ra.done(s)
                        if e < end and PREFIX_SNAPSHOTS:
                            self._prefix_snapshots[e] = self._snapshot_prefix(ids_host[:e], e)
                        while len(self._prefix_snapshots) > max(1, PREFIX_SNAPSHOTS):
                            oldest = min(n for n in self._prefix_snapshots if n != P)
                            del self._prefix_snapshots[oldest]
                self._save_prefix(ids_host, end)
                if self.prefix_disk is not None:
                    self.prefix_disk.save(ids_host)
                torch.cuda.synchronize(self.device)
            finally:
                m._prefix_replay_only = previous
            report = {'status': 'saved', 'input_tokens': P, 'response_tokens': len(response),
                      'cached_tokens': end, 'seconds': round(time.perf_counter() - started, 4)}
            self._prefix_response_last = report
            log('prefix_response: ' + json.dumps(report))
            return report

    def generate(self, prompt_ids, *, max_tokens=4096, temperature=1.0, top_p=0.95, stop_token_ids=None, seed=None,
                 penalties=None,
                 ignore_eos=False, grammar=None, user_prompt_ranges=None):
        """Yield bursts of new token ids.

        ``ignore_eos``: keep decoding until ``max_tokens`` even if EOS/stop ids come up. The
        benchmark needs runs of a fixed output length -- otherwise a decode-rate comparison is
        really a comparison of how early each config decided to stop.

        ``grammar``: optional decoding gate (see server/engine_api.py). It is shown every token
        this loop settles on (``observe``) and is asked to mask the verify block's logit rows
        (``mask_rows``) before the accept/reject decision. It masks nothing until the model starts
        a tool-calls block, and with ``grammar=None`` not one instruction of this loop changes.
        """
        with self.lock:
            yield from self._generate(list(prompt_ids), max_tokens, temperature, top_p, set(stop_token_ids or ()),
                                      seed, ignore_eos, grammar, penalties, user_prompt_ranges)

    def _policy_step_s(self, t_iter: float, n_captures: int | None):
        """Wall time of this decode step for DepthPolicy, or None if it captured a graph or was
        verified in a batched round (engine/serving.py). A batched round's wall time is set by
        every lane in it, so depth 3 and 5 read alike; learned from it, the policy carried ~3x
        step costs into later solo steps (2026-10-14 gate log: depth 3 estimated at 17 tok/s on a
        lone request running ~50)."""
        if getattr(self, "_last_verify_batched", False):
            return None
        # Pool rotation can replace a graph without changing the live count
        # (notably at DSV41_GRAPHS_MAX=1). Missing legacy counters fail closed:
        # an unknown capture state must not train the policy's step costs.
        now = getattr(self.fast, "graph_captures", None)
        return None if n_captures is None or now != n_captures else time.perf_counter() - t_iter

    def _generate(self, prompt, max_tokens, temperature, top_p, stop_ids, seed, ignore_eos=False, grammar=None,
                  penalties=None, user_prompt_ranges=None):
        if seed is not None:
            torch.manual_seed(seed)
        stop_ids = set() if ignore_eos else (set(stop_ids) | {self.eos_token_id})
        self._reset()
        self.user_prompt.reset(user_prompt_ranges, len(prompt))
        if self.layer_ablation.path:
            self.layer_ablation.sync(self.model, self.ep)
        self._init_prefix_disk()
        m = self.model
        P = len(prompt)
        assert P + max_tokens + 8 <= self.max_context, f"prompt {P} + max_tokens {max_tokens} > context {self.max_context}"
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        _prefix_load_start = time.perf_counter()
        # Hash preprocessed pixels/layout once per request. Both ranks agree before
        # any cached-state restore can change the sequence of model collectives.
        media, media_error = (), None
        try:
            media = image_fingerprints(getattr(self, '_images', None),
                                       getattr(self, '_token_types', None))
        except Exception as exc:
            media_error = type(exc).__name__
        identities = self.ep.gather_objects((media_error, media))
        if any(error for error, _ in identities) or any(item != identities[0] for item in identities):
            raise RuntimeError('image prefix identity invalid or different across ranks')
        self._prefix_media = media
        prefix_start = self._restore_prefix(prompt)
        if (self.prefix_disk is not None and self.swa_replay
                and os.environ.get('DSV41_PREFIX_CACHE', '1') == '1'):
            prefix_start = self.prefix_disk.restore(prompt, prefix_start)
        else:
            lengths = self.ep.gather_objects(prefix_start)
            if any(n != lengths[0] for n in lengths):
                prefix_start = 0
                self._prefix_cache = None
                self._prefix_snapshots.clear()
        _prefix_load_s = time.perf_counter() - _prefix_load_start
        t_start = time.perf_counter()
        # Every counter the stats epilogue reads is initialised here, because the epilogue runs in a
        # `finally`: the server closes the generator on a stop string or a client disconnect, which
        # raises GeneratorExit at the pending `yield`, and an aborted request must still report the
        # work it did instead of the previous request's numbers.
        n_out = 0
        steps = 0
        pos = P
        accepted_hist = []
        t_prefill = 0.0
        t_decode0 = t_start
        try:
            yield from self._decode_loop(ids, P, max_tokens, temperature, top_p, stop_ids,
                                         _st := {}, grammar, penalties, prefix_start=prefix_start,
                                         host_ids=prompt, prefix_load_s=_prefix_load_s)
        except BaseException:
            # GeneratorExit (stop string / disconnect / server error) or a real exception. The
            # normal exit already told rank 1 to stop via the loop's own control(); only the
            # aborted paths still owe it that message, so this is `except`, never `finally` --
            # an extra broadcast on the happy path would desync the NEXT request.
            self.ep.release_peer()
            raise
        finally:
            n_out = _st.get("n_out", n_out)
            steps = _st.get("steps", steps)
            accepted_hist = _st.get("accepted", accepted_hist)
            t_prefill = _st.get("t_prefill", t_prefill)
            t_dec = max(time.perf_counter() - _st.get("t_decode0", t_start), 1e-9)
            _ph = _st.get("phases")
            if _ph is not None and hasattr(_ph, "report"):
                self._decode_probe_phases = _ph
            if _ph is not None and _ph.steps:
                _notes = []
                _fd0 = _st.get("fd0")
                if _fd0 is not None and self.fast is not None:
                    _notes = [("of it: engram", self.fast.stats["engram_s"] - _fd0["engram_s"]),
                              ("of it: graphs", self.fast.stats["graph_s"] - _fd0["graph_s"]),
                              ("of it: draft g", self.fast.stats["draft_s"] - _fd0["draft_s"])]
                if STEP_TIMING:
                    print(f"[step timing] host wall clock per decode step, {_ph.steps} steps "
                          f"(DSV41_STEP_TIMING=1)\n{_ph.table(_notes)}", flush=True)
            st = self.store.stats
            m = self.model
            self.last_stats = {
                "prompt_tokens": P, "completion_tokens": n_out, "prefill_s": round(t_prefill, 3),
                "prefix_cached_tokens": _st.get("prefix_cached_tokens", 0),
                "predictive_prefill": dict(self.predictive.last),
                "prefill_budget": _st.get("prefill_budget"),
                "prefill_tok_s": round(P / t_prefill, 2) if t_prefill > 0 else None,
                "decode_s": round(t_dec, 3), "decode_tok_s": round(max(n_out - 1, 0) / t_dec, 2),
                "steps": steps,
                # DSV41_HOT_READS, cumulative since boot (engine/hot_reads.py)
                "hot_reads": (dict(self.fast.hot.stats) if getattr(getattr(self, 'fast', None), 'hot', None)
                              is not None else None),
                "decode_adaptation": {"checks": self._decode_adapt_checks,
                                      "passes": self._decode_adapt_passes},
                "user_prompt": self.user_prompt.report(),
                "accept_len_mean": round(float(np.mean(accepted_hist)) + 1, 2) if accepted_hist else None,
                "spec_conf": _st.get("conf"),
                "prefill_graphs": (self.model.prefill_graphs.report()
                                   if getattr(self.model, 'prefill_graphs', None) else None),
                "tree_probe": _st.get("tree"),
                "spec_depth": (_st["depth_policy"].report() if _st.get("depth_policy") is not None
                               else None),
                "lookup_draft": (_st['lookup_draft'].report() if _st.get('lookup_draft') is not None else None),
                "draft_bypass": (_st['draft_bypass'].report() if _st.get('draft_bypass') is not None else None),
                "expert_hit_rate": round(self.store.hit_rate(), 4), "expert_misses": st["misses"],
                "prefill_expert_misses": st["prefill_misses"], "nvme_gb": round(st["bytes_read"] / 1e9, 2),
                "nvme_read_s": round(st["read_s"], 2),
                "engram_rows": sum(t.stats["rows"] for t in self.tables.values()),
                "engram_cache": {L: t.row_cache.report() for L, t in self.tables.items()
                                 if t.row_cache is not None},
                "engram_s": round(sum(t.stats["seconds"] for t in self.tables.values()), 3),
            "engram_read_s": round(sum(t.stats.get("read_s", 0.0) for t in self.tables.values()), 3),
                "attn_s": round(m.stats["attn_s"], 2), "moe_s": round(m.stats["moe_s"], 2),
                # EP2 (engine/dist.py): pure time inside the per-layer combine all-reduce, and
                # the (layer, expert) pairs this rank skipped because the peer owns them.
                # A tok/s row under WORLD_SIZE=2 must carry the network share that produced it.
                "ep_s": round(m.stats.get("ep_s", 0.0), 2),
                "remote_experts": st.get("remote", 0),
                # where the MoE time actually goes: routing+slot bookkeeping, waiting for the
                # NVMe loads of this layer, and the Triton kernel itself (moe_s minus the rest).
                "route_s": round(st["route_s"], 2), "load_wait_s": round(st["load_s"], 2),
                # route_s split into the two things it was hiding (engine/experts.py):
                #   sync_s -- the blocking device->host copy: the host parked while the GPU drains
                #             what is queued. GPU work made visible; graph capture stops EXPOSING
                #             it 40x a step but does not remove the work.
                #   book_s -- numpy/dict slot bookkeeping with nothing queued on the GPU. Pure host
                #             overhead, and the part the graphed fast path actually deletes.
                # Their ratio is what says whether re-enabling graphs under EP2 is worth the work.
                "sync_s": round(st.get("sync_s", 0.0), 2), "book_s": round(st.get("book_s", 0.0), 2),
                "resolves": st.get("resolves", 0),
                "ep_calls": m.stats.get("ep_calls", 0),
                "ep_us_per_call": (round(m.stats["ep_s"] / m.stats["ep_calls"] * 1e6, 1)
                                   if m.stats.get("ep_calls") else None),
                "lease_s": round(st["lease_s"], 2), "h2d_s": round(st["h2d_s"], 2),
                "kernel_s": round(m.stats["moe_s"] - st["resolve_s"], 2),
                "nvme_gb_per_token": round(st["bytes_read"] / 1e9 / max(n_out, 1), 3),
                "promoted": st["promoted"],
            }
            # --- where the decode seconds went -------------------------------------------
            # One readable decomposition of decode_s, because the individual timers above do
            # not obviously add up to anything and "moe_s is big" is not an action.
            #
            # Relationships that are easy to get wrong when reading them raw:
            #   moe_s     = resolve_s + (kernel + combine + shared)
            #   resolve_s = sync_s + book_s + load_s
            #   kernel_s  = moe_s - resolve_s, and it INCLUDES ep_s
            # so `compute` below subtracts ep_s back out to leave the actual expert math.
            #
            # These are HOST wall-clock timers and some of the work behind them overlaps (the
            # NVMe loads run on a thread pool), so the parts do not have to sum to decode_s.
            # `unaccounted` is printed rather than hidden: it is the sampler, the drafter, the
            # Python of the decode loop itself, and any overlap -- and if it ever dominates,
            # that is the finding, not a rounding error.
            _ep = m.stats.get("ep_s", 0.0)
            # m.stats' timers live in the EAGER forward. With CUDA graphs on, decode replays a
            # captured graph and those lines never execute again -- so attn/moe below are the
            # PREFILL figures, and decode's compute showed up only as "unaccounted", which read
            # 46-75% and looked like the GPU idling. It was the decomposition, not the GPU.
            _parts = {
                "attn": m.stats["attn_s"],
                "moe_sync_gpu_wait": st.get("sync_s", 0.0),
                "moe_book_host": st.get("book_s", 0.0),
                "moe_nvme_wait": st["load_s"],
                "moe_compute": max(0.0, m.stats["moe_s"] - st["resolve_s"] - _ep),
                "ep_collective": _ep,
                "engram": sum(t.stats["seconds"] for t in self.tables.values()),
            }
            # The graphed decode step, from fastdecode's own host timers. graph_s wraps the whole
            # layer loop and the engram waits happen inside it, so the wait is subtracted out to
            # keep these disjoint. What is left in `unaccounted` after this is the sampler, the
            # accept/reject, and the Python of the decode loop -- and THAT number is the one worth
            # reading as GPU idle time.
            _fd0p = _st.get("fd0")
            if _fd0p is not None and self.fast is not None:
                _g = self.fast.stats["graph_s"] - _fd0p["graph_s"]
                _ge = self.fast.stats["engram_s"] - _fd0p["engram_s"]
                _parts["decode_graphs"] = max(0.0, _g - _ge)
                _parts["decode_engram_wait"] = _ge
                _parts["decode_draft"] = self.fast.stats["draft_s"] - _fd0p["draft_s"]
            # Against the WHOLE request, not decode alone. Every component above is a counter
            # accumulated from _reset(), so each already includes this request's prefill; dividing
            # by decode seconds produced shares over 100 % (moe_compute read 349 % on a 16-token
            # completion) and would mislead anyone reading the log line.
            _req = max(t_dec + t_prefill, 1e-9)
            # NOT clamped at zero. Some of these overlap by construction -- the engram reads run on
            # a thread pool while the GPU works, so `engram` and `decode_engram_wait` count the same
            # seconds from two sides -- and clamping turned that into a confident 0.0% instead of a
            # visible sign that the parts double-count. A negative value here means exactly that.
            _parts["unaccounted"] = _req - sum(_parts.values())
            # DSV41_ROUTE_STATS=1: DISTINCT routed experts per layer per verify block. This is the
            # number that sets the expert bytes a step must read -- an expert is read once however
            # many of the block's tokens route to it -- so it is what turns a measured ms/step into
            # a "what fraction of the box's streaming ceiling are we at" answer. Without it the
            # byte count is a guess (6 per layer is the PER-TOKEN figure, and a 6-token verify
            # block asks for far more).
            if self.fast is not None:
                gt = self.fast.gpu_timing_report()
                if gt:
                    # Real GPU-timeline milliseconds per phase, so the step can finally be split
                    # into "the model" and "the drafter" instead of inferred from host waits.
                    n_steps = max(1, gt.get("layers_n", 1))
                    segment_ms = gt.get("segments", 0)
                    envelope_ms = gt.get("layers", 0)
                    self.last_stats["gpu_timing"] = {
                        "hash_ms_per_step": round(gt.get("hash", 0) / max(1, gt.get("hash_n", 1)), 2),
                        "layers_ms_total": envelope_ms, "draft_ms_total": gt.get("draft"),
                        "layers_ms_per_step": round(envelope_ms / n_steps, 2),
                        "segment_busy_ms_per_step": round(segment_ms / n_steps, 2),
                        "boundary_gap_ms_per_step": round(max(0.0, envelope_ms - segment_ms) / n_steps, 2),
                        "engram_ready_wait_ms_per_step": round(gt.get("engram_ready_wait", 0) / n_steps, 2),
                        "sample_envelope_ms_per_step": round(
                            gt.get("sample", 0) / max(1, gt.get("sample_n", n_steps)), 2),
                        "draft_ms_per_step": round(gt.get("draft", 0) / max(1, gt.get("draft_n", 1)), 2),
                        "draft_share": (round(gt.get("draft", 0) /
                                              max(1e-9, gt.get("draft", 0) + gt.get("layers", 0)), 3)),
                    }
            # Request-unit mode folds this request in as one normalized vote; no-op in slot mode.
            # Before prune_miss_report so the save below and the per-request delta both see it.
            self.model.flush_request_demand(
                ADAPT.halflife)
            # One request has now landed since the last applied plan.  maintain_demand() counts
            # this in request mode: the demand DB's total mass is flat once converged, so a mass
            # delta cannot distinguish "a request happened" from "nothing changed".
            self._requests_since_plan += 1
            pmr = self.model.prune_miss_report() if hasattr(self.model, "prune_miss_report") else None
            if pmr is not None:
                summary, counts, mass = pmr
                self.last_stats["prune_miss"] = summary
                # per-request delta: what THIS prompt asked for that was not resident
                _m1 = self.model.miss_snapshot()
                _m0 = getattr(self, "_miss_at_request_start", None)
                if _m0 is not None and _m1 is not None and _m1[1] > _m0[1]:
                    dm, dt = _m1[0] - _m0[0], _m1[1] - _m0[1]
                    self.last_stats["prune_miss_request"] = {
                        "miss_rate": round(dm / dt, 4), "missed_slots": int(dm), "slots": int(dt)}
                _s1 = self.model.miss_mass_snapshot()
                _s0 = getattr(self, "_miss_mass_at_request_start", None)
                if _s0 is not None and _s1 is not None and _s1[1] > _s0[1]:
                    dm, dt = _s1[0] - _s0[0], _s1[1] - _s0[1]
                    self.last_stats.setdefault("prune_miss_request", {}).update(
                        score_miss_rate=round(dm / dt, 6), missed_score=dm, total_score=dt)
                # The accumulators were seeded from the DB at startup, so what they hold IS the
                # updated database -- write it back whole. Rank 1 would write an identical file
                # (both ranks run the same router over the same tokens); only rank 0 does, so the
                # peer's results/ is not a second source of truth.
                if self.ep.rank == 0:
                    try:
                        save_prune_db(counts.numpy(), mass.numpy())
                    except Exception as _e:  # noqa: BLE001
                        log(f"prune demand DB write failed: {_e}")
                # `self.fast` is None whenever the graphed decode path is off -- SPEC=0 is the
                # documented way to get there, and this call crashed every request under it,
                # with DSV41_ROUTE_STATS not even set. route_stats_report() already returns None
                # when the counters are off; it just has to be reachable first.
                rep = self.fast.route_stats_report() if self.fast is not None else None
                if rep:
                    per_layer_mean = rep["mean"]
                    owned = (per_layer_mean / max(1, self.ep.world)
                             if self.ep.active and not self.ep.tensor_parallel else per_layer_mean)
                    self.last_stats["route_stats"] = {
                        "distinct_experts_per_layer": round(per_layer_mean, 2),
                        "distinct_per_layer_this_rank": round(owned, 2),
                        "expert_gb_per_step_this_rank": round(
                            owned * self.args.n_layers * self.expert_bytes / 1e9, 2),
                        "steps_counted": rep["steps"],
                    }
            _ph = M_.phase_report()
            moe_timing = getattr(self.model, "prefill_moe_timing", None)
            if moe_timing is not None:
                moe_report = moe_timing.report()
                self.last_stats["prefill_moe_timing"] = moe_report
                log("prefill_moe_timing: " + json.dumps(moe_report))
                self.model.prefill_moe_timing = None
            if _st.get("prefill_timing") is not None:
                report = _st["prefill_timing"].report()
                self.last_stats["prefill_chunks"] = report
                # Both ranks log here; the headless worker does not emit HTTP stats.
                log("prefill_chunks: " + json.dumps(report))
                if _ph:
                    log("prefill_rank_phases: " + json.dumps({"rank": self.ep.rank, "gpu_ms": _ph}))
            if _ph:
                tot = sum(_ph.values()) or 1.0
                self.last_stats["attn_phases"] = {
                    k: {"ms": v, "pct": round(v / tot * 100, 1)} for k, v in _ph.items()}
            self.last_stats["decode_accounting"] = {
                k: {"s": round(v, 2), "pct": round(v / _req * 100, 1)} for k, v in _parts.items()
            }
            self.last_stats["accounting_covers"] = "prefill+decode"

    # ------------------------------------------------------------ online adaptation (stage 2)
    def plan_swaps(self, max_swaps: int = 64, min_gain: float | None = None, pending_request: bool = False,
                   request_demand=None):
        """Which residents should be replaced, given everything the router has asked for since boot.

        Swaps are planned WITHIN each ownership class. Both ranks hold the same number of arena
        slots, so the count of kept-and-owned experts per (layer, rank) has to stay fixed -- pairing
        an owned promotion with a non-owned demotion would leave one arena over-subscribed and the
        other with a hole. Ownership is `e % world`, and both ranks record identical demand (the
        router runs replicated over the same tokens). Rank 0 plans BOTH classes and broadcasts
        the plan; each rank loads only its own weights, while the mask update applies everywhere.

        Returns [(layer, expert_out, expert_in, gain), ...], best gain first.
        """
        import numpy as np
        focus = getattr(self, 'user_prompt', None)
        if self._prune_trace is None or not getattr(self, "model_prune_mask", None):
            return []
        rep = self.model.prune_miss_report()
        if rep is None:
            return []
        _summary, demand, mass = rep
        demand = demand.numpy()
        mass = mass.numpy()
        total = float(demand.sum())
        if request_demand is not None and (pending_request or not ADAPT.request_unit):
            raise ValueError('provisional demand requires request units and no pending-request override')
        if request_demand is not None or (pending_request and ADAPT.request_unit and getattr(self.model, "_req_counts", None) is not None):
            # Preview predicted prefill demand or observed decode demand without recording it: aged
            # history plus its normalized per-layer distribution, exactly what
            # flush_request_demand would write -- without writing it. The end-of-request fold
            # still adds exactly one vote; a pass every 600 tokens must not add one each time.
            req = (self.model._req_counts.cpu().numpy() if request_demand is None
                   else np.asarray(request_demand[0], dtype=np.float64))
            req_mass = (self.model._req_mass.cpu().numpy() if request_demand is None
                        else np.asarray(request_demand[1], dtype=np.float64))
            if any(v.shape != demand.shape or not np.isfinite(v).all() or (v < 0).any()
                   for v in (req, req_mass)):
                raise ValueError('invalid provisional routing demand')
            tot = req.sum(axis=1, keepdims=True)
            decay = 0.5 ** (1.0 / ADAPT.halflife) if 0 < ADAPT.halflife < float("inf") else 1.0
            demand = demand * decay + np.divide(req, tot, out=np.zeros_like(req), where=tot > 0)
            score_tot = req_mass.sum(axis=1, keepdims=True)
            mass = mass * decay + np.divide(req_mass, score_tot, out=np.zeros_like(req_mass),
                                           where=score_tot > 0)
            total = float(demand.sum())
        if total <= 0:
            return []
        prior = ADAPT.prior
        if min_gain is None:
            min_gain = ADAPT.swap_min_gain
        ranked, _w = blend_demand(self._prune_trace, (demand, mass), prior)
        if getattr(self, 'dynamic_experts', False):
            from engine.global_residency import global_plan
            keeps = {L:msk.cpu().numpy() for L,msk in self.model_prune_mask.items()}
            scores = {L:ranked[L]*self.prune_layer_weights[L] for L in keeps}
            protected = dict(self.user_prompt.protected)
            for L, ids in (getattr(self, '_dream_pins', None) or {}).items():
                protected[L] = sorted(set(protected.get(L, [])) | set(ids))
            swaps, _ = global_plan(keeps, scores, self.dynamic_layer_floor, max_swaps,
                                  protected=protected,
                                  min_gain=min_gain*float(np.mean(list(scores.values()))))
            return swaps
        world = 1 if getattr(self.ep, 'tensor_parallel', False) else max(1, self.ep.world)
        out = []
        for L, msk in self.model_prune_mask.items():
            keep = msk.detach().cpu().numpy().astype(bool)
            score = ranked[L]
            for r in range(world):
                own = (np.arange(len(keep)) % world) == r
                # Pruned-mode startup guarantees that every kept expert is resident on its owner.
                # The planner runs only on rank 0 and must still plan BOTH ownership classes; its
                # local LRU cannot be used to filter rank 1's candidates.
                protected = np.zeros(len(keep), dtype=bool)
                focus = getattr(self, 'user_prompt', None)
                if focus is not None:
                    protected[focus.protected.get(L, [])] = True
                protected[(getattr(self, '_dream_pins', None) or {}).get(L, [])] = True
                kept_own = np.flatnonzero(keep & own & ~protected)
                gone_own = np.flatnonzero(~keep & own)
                if not len(kept_own) or not len(gone_own):
                    continue
                # worst resident vs best absentee, within this class
                weakest = kept_own[np.argsort(score[kept_own])]
                strongest = gone_own[np.argsort(-score[gone_own])]
                # Hysteresis. Running this every idle tick means near-tied experts would
                # otherwise trade places whenever the ranking jitters, paying a load each time to
                # chase noise. Scores are per-layer normalised (they sum to ~1 across n_experts),
                # so the floor is expressed against the mean share.
                floor = min_gain * float(score.mean())
                for e_out, e_in in zip(weakest, strongest):
                    gain = float(score[e_in] - score[e_out])
                    if gain <= floor:
                        break          # sorted, so nothing after this pair wins either
                    out.append((int(L), int(e_out), int(e_in), gain))
        out.sort(key=lambda t: -t[3])
        return out[:max_swaps]

    def promote_user_prompt(self):
        """Rank 0 admits latest-user experts before decode; both ranks always rendezvous.

        Protection outlives the response and idle maintenance. Only starting the
        next request clears it. This deliberately bypasses history's gain floor:
        the current user's routed experts have first priority within each quota.
        """
        if not self.user_prompt_stream:
            return 0
        from engine.user_prompt import capped_retention_plan
        started, packet = time.perf_counter(), None
        if self.ep.rank == 0:
            try:
                packet = dict(swaps=[], protected={}, overflow=0, deferred=0)
                if self.user_prompt.mass is not None:
                    mass = self.user_prompt.mass.cpu().numpy()
                    rep = self.model.prune_miss_report()
                    fallback, _ = blend_demand(self._prune_trace,
                        (rep[1].numpy(), rep[2].numpy()), ADAPT.prior)
                    keeps = {L:mask.cpu().numpy() for L,mask in self.model_prune_mask.items()}
                    if getattr(self, 'dynamic_experts', False):
                        from engine.global_residency import prompt_plan
                        packet = prompt_plan(keeps, mass, fallback, self.dynamic_layer_floor,
                                             self.prune_layer_weights, None,
                                             pinned=getattr(self, '_dream_pins', None))
                    else:
                        packet = capped_retention_plan(keeps, mass, fallback,
                                                       pinned=getattr(self, '_dream_pins', None))
            except Exception as exc:
                packet = dict(error=type(exc).__name__)
        # Empty/attachment-only/no-user requests participate too. No rank-local
        # counters or masks can decide whether this collective is reached.
        packet = self.ep.broadcast_obj(packet)
        if 'error' in packet:
            raise RuntimeError('latest-user retention planner failed: ' + packet['error'])
        self.user_prompt.protected = packet['protected']
        self.user_prompt.overflow = packet['overflow']
        self.user_prompt.deferred = packet['deferred']
        self.user_prompt.cross_layer = packet.get('cross_layer', 0)
        self.user_prompt.discovery_loads = getattr(getattr(self, 'store', None), 'stats', {}).get('prefill_misses', 0)
        if not self.user_prompt.max_loads:
            self.user_prompt.loads_used = self.user_prompt.discovery_loads
        self.user_prompt.promoted = self.apply_swaps(packet['swaps'])
        self.user_prompt.promotion_s = time.perf_counter() - started
        return self.user_prompt.promoted

    def refresh_user_prompt_logits(self, previous, ids, spans):
        m, focus = self.model, self.user_prompt
        start = time.perf_counter()
        replay_only = getattr(m, '_prefix_replay_only', False)
        recording, streaming = focus.recording, focus.streaming
        try:
            m._prefix_replay_only = True
            focus.recording = focus.streaming = False
            # Every cached layer, including the encoder and the assistant cue,
            # was computed before admission. Rebuild with the new resident set
            # so the FIRST answer token actually uses those experts throughout.
            # This is a resident pass, not a second streaming discovery pass.
            c = self.caches
            c.len = 0
            c._chunk_inputs.clear()
            for L in c.pending:
                c.pending[L] = None
            m.begin_prompt()
            for s, e in spans:
                m.forward(ids[s:e], s, prefill=True, need_logits=False, encoder_only=True)
            logits, hidden, pos = m.decoder_replay(need_logits=True)
            if self.spec:
                m.dspark_seed(hidden, pos)
            if not focus.ranges and focus.promoted == 0 and not PREFILL_SWAP:
                focus.unchanged_replay_max_delta = float((logits[-1]-previous[-1]).abs().max())
            return logits
        finally:
            m._prefix_replay_only = replay_only
            focus.recording, focus.streaming = recording, streaming
            focus.refresh_s = time.perf_counter() - start

    def apply_swaps(self, swaps) -> int:
        """Move the arena slots this rank owns and update the routing state in place.

        ALL OR NOTHING, and any failure is fatal. The mask is global and both ranks must end
        with the same one: a rank that skipped a swap the other applied would keep a different
        set of experts routable, so the two routers would pick different top-k, emit different
        tokens, and diverge with no error anywhere -- the pair is meant to be bit-identical.
        Only the owning rank touches the arena, so a rank-local "skip this one" is itself the
        divergence; the only safe responses are apply everything or die loudly, and a desynced
        pair is unrecoverable anyway (see run_worker).

        The mask and the device LUT are read from inside the captured decode graphs, so both are
        mutated IN PLACE: the graph holds their addresses, and a fresh tensor would be ignored
        while the stale one kept driving the router.
        """
        if not swaps:
            return 0
        focus = getattr(self, 'user_prompt', None)
        if len(swaps[0]) == 5:
            return V41Engine.apply_global_swaps(self, swaps)
        st = self.store
        world, rank = ((1, 0) if getattr(self.ep, 'tensor_parallel', False)
                       else (max(1, self.ep.world), self.ep.rank))
        null = int(getattr(st, "null_slot", -1) or -1)

        # Validate before touching anything: a plan that is wrong is wrong before the first load,
        # and stopping here leaves the arena exactly as it was.
        for L, e_out, e_in, _g in swaps:
            focus = getattr(self, 'user_prompt', None)
            if focus is not None and e_out in focus.protected.get(L, ()):
                raise RuntimeError(f'swap plan evicts protected latest-user expert: L{L} e{e_out}')
            if (e_in % world) != (e_out % world):
                raise RuntimeError(f"swap plan pairs across ownership classes: L{L} {e_out}->{e_in}")
            if (e_in % world) != rank:
                continue                                  # the peer owns this pair's arena slot
            if (L, e_out) not in st.lru:
                raise RuntimeError(f"swap plan evicts a non-resident expert: L{L} e{e_out}")
            if (L, e_in) in st.lru:
                raise RuntimeError(f"swap plan promotes an already-resident expert: L{L} e{e_in}")

        for L, e_out, e_in, _g in swaps:
            if (e_in % world) == rank:
                slot = st.lru.pop((L, e_out))
                try:
                    st._load_into_slot((L, e_in), slot)
                except Exception as e:
                    st.lru[(L, e_out)] = slot              # restore, then die: see the docstring
                    raise RuntimeError(f"swap load failed L{L} e{e_in}; pair would desync") from e
                st.lru[(L, e_in)] = slot
                st.slot_key[slot] = (L, e_in)
                if self.fast is not None and getattr(self.fast, "lut", None) is not None:
                    self.fast.lut[L, e_in] = slot
                    self.fast.lut[L, e_out] = null
                route = getattr(self.model, "prefill_routes", {}).get(L)
                if route is not None:
                    ids, slots = route
                    # Fixed prefill routing bypasses the decode LUT. Transfer the
                    # evicted expert's compact ID with its arena slot, otherwise
                    # promoted experts silently contribute zero after adaptation.
                    # Shapes/addresses stay fixed; no new kernel specialization.
                    ids[e_in] = ids[e_out]
                    ids[e_out] = slots.numel() - 1  # final entry is the null slot
            # unconditional: identical on both ranks, whoever owns the slot
            self.model_prune_mask[L][e_out] = False
            self.model_prune_mask[L][e_in] = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # the loads ran on a copy stream; land them before serving
        self._swap_baseline = float(self.model.prune_miss_report()[1].sum())
        self._requests_since_plan = 0   # a plan was applied; wait for the next request
        self.expert_generation += 1
        if focus is not None:
            focus.resident_loads = getattr(focus, 'resident_loads', 0) + len(swaps)
        if getattr(self, 'prefix_disk', None) is not None:
            from engine.prefix_persistence import routing_signature
            self._prefix_route = routing_signature(self)
        log(f"adapted {len(swaps)} expert slots to observed demand "
            f"(expert generation {self.expert_generation})")
        return len(swaps)

    def apply_global_swaps(self, swaps):
        """Transfer sectors between layers; one authoritative directory on both TP ranks.

        Decode graphs retain the LUT and masks' addresses. Eager prefill compact
        maps are rebuilt for changed layers because their cardinalities can grow
        or shrink. RAM prefix state has no sector pointers and remains historical
        state, as with within-layer adaptation. Disk/response caches and prefill
        graphs are disabled at boot.
        Any load failure is fatal: a partially rewritten TP pair cannot continue.
        """
        if not getattr(self.ep, 'tensor_parallel', False):
            raise RuntimeError('cross-layer sector transfers require TP')
        focus, st = self.user_prompt, self.store
        # Under TP there is no null slot (that is an EP concept); a swapped-out expert is masked out
        # of the routable set below, so -1 is only a placeholder the router can never gather.
        null = st.null_slot if getattr(st, "null_slot", None) is not None else -1
        counts = {L:int(mask.sum()) for L,mask in self.model_prune_mask.items()}
        outgoing, incoming = set(), set()
        # Validate the complete plan against the original directory before I/O.
        for Lo, eo, Li, ei, gain in swaps:
            old, new = (Lo,eo), (Li,ei)
            if (old in outgoing or new in incoming or old not in st.lru or new in st.lru
                    or Li not in counts or not 0 <= ei < self.model_prune_mask[Li].numel()
                    or eo in focus.protected.get(Lo, ())):
                raise RuntimeError('invalid global expert sector transfer')
            outgoing.add(old); incoming.add(new)
            counts[Lo] -= 1; counts[Li] += 1
            if counts[Lo] < self.dynamic_layer_floor:
                raise RuntimeError('sector transfer violates streaming capacity floor')
        if outgoing & incoming:
            raise RuntimeError('sector transfer cycles are not supported')
        changed = set()
        for Lo, eo, Li, ei, gain in swaps:
            slot = st.lru.pop((Lo,eo))
            try:
                st._load_into_slot((Li,ei), slot)
            except Exception as exc:
                st.lru[(Lo,eo)] = slot
                raise RuntimeError('global expert load failed; TP pair must stop') from exc
            st.lru[(Li,ei)] = slot
            st.slot_key[slot] = (Li,ei)
            self.fast.lut[Lo,eo] = null
            self.fast.lut[Li,ei] = slot
            self.model_prune_mask[Lo][eo] = False
            self.model_prune_mask[Li][ei] = True
            changed.update((Lo,Li))
        # The fixed compact maps were the second hidden per-layer restriction.
        # Only eager prefill reads these arrays; no captured graph points at them.
        routes = getattr(self.model, 'prefill_routes', None)
        if routes is not None:
            for L in sorted(changed):
                owned = sorted((e,s) for (layer,e),s in st.lru.items() if layer == L)
                ids = torch.full_like(routes[L][0], len(owned))
                for i,(e,_) in enumerate(owned):
                    ids[e] = i
                slots = torch.tensor([s for _,s in owned]+[null],
                                     dtype=torch.int32, device=ids.device)
                routes[L] = (ids,slots)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._swap_baseline = float(self.model.prune_miss_report()[1].sum())
        self._requests_since_plan = 0
        self.expert_generation += 1
        focus.resident_loads = getattr(focus, 'resident_loads', 0) + len(swaps)
        self.resident_layer_counts = [counts[L] for L in sorted(counts)]
        if getattr(self, 'prefix_disk', None) is not None:
            # as apply_swaps does: DSV41_PREFIX_DISK_STRICT=1 matches bundles on this signature,
            # and a stale one would admit KV computed under a different expert layout
            from engine.prefix_persistence import routing_signature
            self._prefix_route = routing_signature(self)
        log(f'adapted {len(swaps)} global expert sectors '
            f'({sum(Lo != Li for Lo,eo,Li,ei,g in swaps)} across layers; '
            f'residents/layer {min(counts.values())}-{max(counts.values())}, generation {self.expert_generation})')
        return len(swaps)

    def maintain_demand(self) -> bool:
        """Age the demand history, then say whether a re-plan is worth doing.

        Called on every idle tick. The decay is applied here rather than at record time because
        recording happens inside a captured CUDA graph, where an extra scaling pass would have to
        be captured too; once per idle tick is often enough for a half-life measured in millions
        of slots, and it keeps the hot path untouched.
        """
        if not ADAPT.swap:
            return False
        rep = self.model.prune_miss_report()
        if rep is None:
            return False
        total = float(rep[1].sum())
        grown = total - self._swap_baseline
        if ADAPT.request_unit:
            # Trigger on REQUESTS SINCE THE LAST APPLIED PLAN, not on `grown`.  The demand DB is
            # an EWMA (Model.flush_request_demand: db*0.5**(1/H) + one normalized vote), so its
            # total mass is constant once converged and `grown = total - _swap_baseline` sits at
            # ~0 -- or goes NEGATIVE after a half-life change, because the persisted DB was built
            # at the old H.  A mass delta therefore stopped opening this gate for good, and the
            # miss gate in maintain_inline was never reached (a 14,649-token request at 12.1%
            # routed-miss produced no swap).  The slot-tuned MIN_GROWTH (5e5) is equally useless
            # here, so this reuses SWAP_MIN_GROWTH_REQ as "requests needed", defaulting to one.
            return self._requests_since_plan >= ADAPT.min_growth_req
        half = ADAPT.halflife
        if half > 0 and grown > 0:
            # decay proportional to the evidence that arrived since the last tick, so the rate is
            # set by traffic rather than by wall-clock or by how often this happens to run
            self.model.decay_demand(0.5 ** (grown / half))
        return grown >= ADAPT.min_growth_slots

    def swaps_due(self, min_growth: float | None = None) -> bool:
        """Back-compat shim for callers that only want the predicate."""
        return self.maintain_demand()

    def _decode_adapt_due(self, n_out: int) -> int:
        """Rank 0 returns 1 when a periodic decode adaptation pass is due; control broadcasts it.

        The decode-only miss counter pair is read once per ADAPT.decode_tokens output tokens.
        (DSV41_ADAPT_URGENT's rolling ~30-token check was removed 2026-10-18 for hot reads.)
        """
        mark = getattr(self, '_decode_adapt_mark', None)
        if not ADAPT.decode_tokens or mark is None:
            return 0
        if n_out - mark[0] < ADAPT.decode_tokens:
            return 0
        # The eager fallback labels its routes as prefill in existing accounting.
        # Preserve its aggregate-counter delta; the fast path has a decode-only pair.
        snapshot = (getattr(self.model, 'decode_miss_snapshot', self.model.miss_snapshot)
                    if getattr(self, 'fast', None) is not None else self.model.miss_snapshot)
        now = snapshot()
        self._decode_adapt_mark = (n_out, now)
        valid = now is not None and mark[1] is not None and now[1] > mark[1][1]
        rate = (now[0] - mark[1][0]) / (now[1] - mark[1][1]) if valid else None
        check = dict(reason='periodic', output_tokens=n_out, window_tokens=n_out-mark[0],
                     miss_rate=rate, threshold=ADAPT.swap_prefill_min_miss,
                     triggered=bool(valid and rate >= ADAPT.swap_prefill_min_miss))
        if not hasattr(self, '_decode_adapt_checks'):
            self._decode_adapt_checks = []
        self._decode_adapt_checks.append(check)
        del self._decode_adapt_checks[:-64]  # bounded request diagnostics
        if not check['triggered']:
            return 0
        self._decode_adapt_rate = check['miss_rate']
        self._decode_adapt_window = check['window_tokens']
        return 1

    def _engram_rows_async(self, hashes):
        """Start the verify block's hash-id D2H on a side stream; return the `rows` callable
        FastDecoder.step invokes at its first Engram boundary. The callable waits for that copy
        only (not for anything queued after the hash on the main stream) and submits the reads."""
        main = torch.cuda.current_stream()
        side = getattr(self, "_hash_stream", None)
        if side is None:
            side = self._hash_stream = torch.cuda.Stream(device=self.device)
            self._hash_pinned = {}
        key = (tuple(hashes.shape), hashes.dtype)
        buf = self._hash_pinned.get(key)
        if buf is None:
            buf = self._hash_pinned[key] = torch.empty(hashes.shape, dtype=hashes.dtype, pin_memory=True)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            buf.copy_(hashes, non_blocking=True)
        hashes.record_stream(side)
        done = torch.cuda.Event()
        done.record(side)

        def rows():
            done.synchronize()
            h_np = buf.numpy()
            return {L: (self.eg_pool.submit(self.tables[L].read_raw, h_np[:, li, :]), self.tables[L].to_device)
                    for li, L in enumerate(self.args.engram_layer_ids)}
        return rows

    def maintain_decode(self, n_out: int) -> int:
        """Re-fit the resident experts mid-decode. Both ranks reach this unconditionally when
        rank 0's control() flag says so -- the top of a decode step is a lockstep point, the same
        argument as maintain_inline. Rank 0 plans (with this request's decode demand previewed,
        see plan_swaps) and broadcasts; both apply. Past tokens are untouched: their KV is
        already written, only later tokens route to the new experts."""
        t0 = time.perf_counter()
        reason, cap = 'periodic', ADAPT.swap_max
        swaps = None
        if self.ep.rank == 0:
            try:
                swaps = self.plan_swaps(max_swaps=cap, pending_request=True)
            except Exception as e:  # noqa: BLE001
                log(f"decode adaptation planning failed ({type(e).__name__}: {e}); skipping")
                swaps = None
        swaps = self.ep.broadcast_obj(swaps)
        if not swaps:
            self._decode_adapt_passes.append(dict(output_tokens=n_out, swaps=0,
                seconds=time.perf_counter()-t0, reason=reason))
            return 0
        try:
            n = self.apply_swaps(swaps)
        except Exception:
            import traceback
            traceback.print_exc()
            log("decode adaptation failed after the plan was broadcast; the pair would desync")
            os._exit(1)
        self._decode_adapt_passes.append(dict(output_tokens=n_out, swaps=n,
            seconds=time.perf_counter()-t0, reason=reason))
        if self.ep.rank == 0:
            log(f"{reason} decode adaptation at output token {n_out}: {n} expert slots in "
                f"{1000 * (time.perf_counter() - t0):.0f} ms "
                f"(miss rate {getattr(self, '_decode_adapt_rate', 0.0):.1%} over the last "
                f"{self._decode_adapt_window} decoded tokens)")
        return n

    def maintain_inline(self, new_tokens: int) -> int:
        """Re-fit the resident experts mid-request, at the prefill -> decode boundary.

        This is the one point inside a request where both ranks are known to be at the same place:
        they run the same generate() over the same prompt, and the prefill chunks have just
        finished. That makes it a legal collective site, which end-of-response is too -- but
        anywhere else in the loop is not.

        Rank 0 plans and broadcasts, exactly as the between-requests path does. The ranks do NOT
        plan independently: plan_swaps reads rank-local LRU state, and two ranks choosing different
        swaps leaves them masking different experts, which makes the replicated routers pick
        different top-k and emit different tokens with no error anywhere. Only the plan crosses the
        wire (~1.5 KB); each rank then loads its own half of the arena.

        maintain_demand() runs on BOTH ranks, not just the planner: it ages the demand history, and
        aging it on one rank only would drift the two copies apart for no benefit.
        """
        if not PREFILL_SWAP:
            return 0
        # From here every rank MUST reach the broadcast, so nothing below may return early on a
        # rank-local quantity. `new_tokens` is derived from prefix_start, which comes from each
        # rank's own prefix cache -- if that ever differed, an early return here would leave one
        # rank in the collective and the other past it, and the pair would hang until the
        # process-group timeout. It only gates rank 0's DECISION; the collective is
        # unconditional, keyed on PREFILL_SWAP, which the boot-time config guard pins identical.
        # maintain_demand() runs everywhere for the same reason: it ages the history, and aging
        # it on one rank only drifts the two copies apart.
        # Request-unit mode: fold THIS prompt's demand before planning.  _record_prune_miss writes
        # the per-request accumulator; only flush_request_demand moves it into the database that
        # plan_swaps reads.  Without this the boundary plan sees only older requests and cannot act
        # on the prompt it is adapting for, which is what this method's docstring promises.  Decode
        # then records into a fresh accumulator and the end-of-request fold adds that vote
        # separately.
        #
        # The trigger is the prompt's own evidence (tokens + its own miss rate, gated below), not
        # maintain_demand(): the end-of-response tick applies a plan after every request and resets
        # _requests_since_plan, so a counter test would starve this path on every request but the
        # first.
        request_mode = ADAPT.request_unit
        if request_mode:
            self.model.flush_request_demand(
                ADAPT.halflife)
        due = ((new_tokens >= PREFILL_SWAP_MIN) if request_mode else
               (self.maintain_demand() and new_tokens >= PREFILL_SWAP_MIN))
        if due and PREFILL_SWAP_MIN_MISS > 0:
            # This prefill's OWN miss rate, not the lifetime one: the accumulators are cumulative
            # by design, so a lifetime rate stops moving after a few hundred requests and would
            # gate on nothing.
            m0 = getattr(self, "_miss_at_request_start", None)
            m1 = self.model.miss_snapshot() if hasattr(self.model, "miss_snapshot") else None
            if m0 is not None and m1 is not None and m1[1] > m0[1]:
                rate = (m1[0] - m0[0]) / (m1[1] - m0[1])
                due = rate >= PREFILL_SWAP_MIN_MISS
        swaps = None
        if self.ep.rank == 0 and due:
            try:
                swaps = self.plan_swaps(
                    max_swaps=ADAPT.swap_max)
            except Exception as e:  # noqa: BLE001
                # Planning is rank-local and nothing has been broadcast yet, so giving up is safe.
                log(f"inline adaptation planning failed ({type(e).__name__}: {e}); skipping")
                swaps = None
        swaps = self.ep.broadcast_obj(swaps)
        if not swaps:
            return 0
        try:
            return self.apply_swaps(swaps)
        except Exception:
            # Past the broadcast the peer is applying this same plan. A rank that does not finish
            # it keeps a different set of experts routable and the pair diverges silently, which is
            # worse than stopping -- the same reasoning as the end-of-response path.
            import traceback
            traceback.print_exc()
            log("inline adaptation failed after the plan was broadcast; the pair would desync")
            os._exit(1)

    def prepare_vl(self, prompt: str, images: list):
        """prompt + image records -> (token ids, token_types, ImageInputs) via the checkpoint's own
        image_processor. Every image-span position carries image_token_id; token_types is the only
        thing that tells IMAGE apart from the three delimiters."""
        if self.vision is None:
            raise RuntimeError("no vision tower is loaded")
        import sys as _sys
        # prepare_vl_inputs imports the checkpoint's `encoding` for IMAGE_PLACEHOLDER, which lives
        # in a sibling directory of `inference` -- both have to be importable.
        for _d in ("inference", "encoding"):
            _p = os.path.join(self.model_dir, _d)
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import image_processor as IP

        class _A:
            pass
        for k, v in self.vision.cfg.items():
            setattr(_A, k, v)
        _A.vision_enabled = True
        _t0 = time.perf_counter()
        toks, types, imgs = IP.prepare_vl_inputs(prompt, images, self.tokenizer, _A)
        log(f"vision: prepared {len(toks)} tokens, {len(imgs or [])} image(s) in "
            f"{time.perf_counter() - _t0:.2f}s")
        return toks, torch.tensor(types, dtype=torch.int64, device=self.device), imgs

    def set_vl_inputs(self, token_types=None, images=None):
        """Attach this request's vision inputs; cleared at the end of generate()."""
        self._token_types, self._images = token_types, images

    def _prefill_spans(self, P: int, start: int = 0, chunk_size=None):
        """Chunk boundaries for this prompt: MAX_CHUNK, except that no image span may straddle one.

        A span must be spliced whole (inference/model.py:1254), and the obvious fix -- a bigger
        MAX_CHUNK for everyone -- is the expensive one: the indexer's score buffer is
        [chunk, context] fp32 (engine/model.py), so at 256k context a 2048 chunk is 2 GB and an
        8192 chunk is 8.6 GB, plus a masked copy. Aligning the boundaries costs nothing and works
        for any number of images; a span longer than MAX_CHUNK simply gets a chunk of its own.
        """
        sp = sorted((im.start, im.start + im.types.numel()) for im in (self._images or ()))
        out, s = [], start
        chunk_size = MAX_CHUNK if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("prefill chunk must be positive")
        while s < P:
            e = min(s + chunk_size, P)
            # a span that begins exactly here must be contained whole, even past MAX_CHUNK
            floor = max([en for st, en in sp if st == s], default=0)
            e = max(e, floor)
            # any OTHER span beginning inside must start the next chunk instead. Checking every
            # span matters: stopping at the first one let a second span inside the same chunk be
            # cut in half.
            cut = [st for st, en in sp if s < st < e]
            if cut:
                e = max(floor, min(cut))
            out.append((s, min(e, P)))
            s = min(e, P)
        return out

    def _engram_readahead(self, ids, P, engram_mask=None, spans=None, hashes_t=None):
        """One read-ahead over the prompt's engram rows, or None to keep the per-chunk prefetch.

        Returns None for prompts of a single chunk (nothing to overlap with), when there are no
        engram tables, or when DSV41_PREFILL_ENGRAM is off.
        """
        if (self.model.hash_state is None or not self.tables
                or os.environ.get("DSV41_PREFILL_ENGRAM", "1") != "1"
                or (spans is not None and len(spans) <= 1)
                or (spans is None and P <= MAX_CHUNK)
                or os.environ.get("DSV41_ENGRAM_READAHEAD", "1") != "1"):
            return None
        if hashes_t is None:
            t0 = time.perf_counter()
            # The whole-prompt hash needs the same token mask the per-chunk path would apply, or
            # the n-grams inside an image span would be live here and dead there.
            hashes_t = self.model.hash_state(
                ids[None], 0, None if engram_mask is None else engram_mask[None])[0]
            self.model.stats["engram_s"] += time.perf_counter() - t0
        # Use the SAME boundaries as the forward loop. Images deliberately move boundaries so a
        # span is never split; fixed MAX_CHUNK read-ahead would then hand a short image chunk the
        # rows for a 2048-token chunk (and engram_forward's reshape would fail).
        spans = self._prefill_spans(P) if spans is None else spans
        ra = EngramReadAhead(self.tables, self.eg_pool, self.args.engram_layer_ids,
                             hashes_t.cpu().numpy(), spans,
                             depth=int(os.environ.get("DSV41_ENGRAM_DEPTH", "2")))
        ra.hashes_t = hashes_t
        return ra

    def _decode_loop(self, ids, P, max_tokens, temperature, top_p, stop_ids, out_st, grammar=None,
                     penalties=None, prefix_start: int = 0, host_ids=None, prefix_load_s=0.0):

        m = self.model
        t_start = time.perf_counter() - prefix_load_s
        out_st["t_decode0"] = t_start
        # prefill in chunks
        logits = None
        self.predictive.begin(self, host_ids if host_ids is not None else ids.tolist(), prefix_start)
        # Every request, NOT only the cold ones. This is the baseline the per-request routed-miss
        # delta is measured from, and leaving it unreset on a prefix-cache hit made that delta span
        # every cached turn back to the last cold prompt -- so a conversation reported a miss rate
        # that barely moved (31.6 -> 31.4 -> 30.3%) while the true per-turn rate was falling
        # (31.6 -> 23.6 -> 19.1%). The adaptation looked broken when it was working, and
        # maintain_inline's gate reads this same delta to decide whether the prompt needs fixing.
        self._miss_at_request_start = m.miss_snapshot() if hasattr(m, "miss_snapshot") else None
        self._miss_mass_at_request_start = (m.miss_mass_snapshot()
                                           if hasattr(m, "miss_mass_snapshot") else None)
        if prefix_start == 0:
            # begin_prompt() EMPTIES the replay buffer, and on a prefix-cache hit _restore_prefix
            # has already filled it with the cached window tail -- so resetting here threw away
            # exactly the state the cache exists to keep. Two failures came out of that: a prompt
            # that hits the cache in FULL prefills nothing, so nothing refills the buffer and
            # _save_prefix -> _rep_tail did torch.cat([]) (a 500 that also marked the EP pair out
            # of step, i.e. every later request 503); and on a partial hit the decoder replay ran
            # over the suffix alone instead of the last window_size prompt tokens, which is a
            # short window rather than a crash and so said nothing at all. A restored prefix is a
            # continuation of its own prompt, not a new one.
            m.begin_prompt()
            # A full prefill means nothing matched: the boundary snapshots described an older
            # prompt, and the prefill about to start rewrites every compressed KV row anyway.
            self._prefix_snapshots.clear()
        # Hash the whole prompt once so chunk k+1's engram rows can be read off NVMe while chunk k
        # is still on the GPU. Hashing per chunk is bit-identical (verified), but it leaves the
        # reads with only layer 0 to hide behind, which costs ~1 s of idle GPU per chunk on text
        # whose n-grams are not already in the row cache.
        # Vision: token_types comes from image_processor.prepare_vl_inputs alongside the ids, and
        # is sliced per chunk exactly like them. TEXT is -1, so `>= 0` marks a whole image span.
        tt = getattr(self, "_token_types", None)
        eg_mask = None if tt is None else ~(tt >= 0)
        chunk_size = MAX_CHUNK
        if self.prefill_budget.enabled:
            # Reached by EVERY rank before planning read-ahead, including a full
            # prefix hit. Rank-local availability must never choose TP boundaries.
            available = self.ep.gather_objects(host_available_bytes())
            self.prefill_budget_last = self.prefill_budget.choose(MAX_CHUNK, P, available)
            self.prefill_budget_last["rank_available_bytes"] = available
            chunk_size = self.prefill_budget_last["rows"]
            if P > prefix_start and self.ep.rank == 0 and chunk_size != MAX_CHUNK:
                log(f"prefill budget: {chunk_size}/{MAX_CHUNK} rows; tighter rank "
                    f"{self.prefill_budget_last['available_bytes'] / 2**30:.2f} GiB available; "
                    f"estimate fits reserve: {self.prefill_budget_last['estimate_fits']}")
        else:
            self.prefill_budget_last = dict(enabled=False, rows=MAX_CHUNK)
        spans = self._prefill_spans(P, prefix_start, chunk_size)
        self.prefill_budget_last["largest_span"] = max((e-s for s,e in spans), default=0)
        self.prefill_budget_last["image_span_exceeds_rows"] = self.prefill_budget_last["largest_span"] > chunk_size
        self.prefill_budget_last["new_tokens"] = P - prefix_start
        out_st["prefill_budget"] = dict(self.prefill_budget_last)
        timing = None
        if PREFILL_TIMING:
            from engine.prefill_timing import PrefillTiming
            timing = PrefillTiming(self.ep.rank)
            out_st["prefill_timing"] = timing
        hashes_t = None
        if prefix_start and m.hash_state is not None:
            # Hashing the full token sequence preserves n-grams crossing the cached boundary;
            # only the suffix rows are fetched and forwarded. Do this on a FULL hit too:
            # after another prompt (or a restart), the Engram compressed-token history is
            # unrelated/uninitialized even though the restored KV and replay state are valid.
            th = time.perf_counter()
            hashes_t = m.hash_state(ids[None], 0, None if eg_mask is None else eg_mask[None])[0]
            m.stats["engram_s"] += time.perf_counter() - th
        ra = self._engram_readahead(ids, P, eg_mask, spans, hashes_t=hashes_t)
        with ra if ra is not None else nullcontext():
            def _fwd(s, chunk, **kw):
                if tt is not None:
                    kw["token_types"] = tt[s:s + len(chunk)]
                    # an image span must lie inside ONE chunk (inference/model.py:1254); the splice
                    # asserts it, and images are only handed to the chunk that contains their start
                    import dataclasses as _dc
                    kw["images"] = [_dc.replace(im, start=im.start - s)
                                    for im in (self._images or ()) if s <= im.start < s + len(chunk)]
                if ra is None:
                    if hashes_t is not None:
                        kw["hashes"] = hashes_t[s:s + len(chunk)]
                    return m.forward(chunk, s, prefill=True, **kw)
                out = m.forward(chunk, s, prefill=True, hashes=ra.hashes_t[s:s + len(chunk)],
                                get_rows=ra.rows_for(s), **kw)
                ra.done(s)
                return out
            if self.swa_replay:
                # CED + Decoder SWA Bounded Replay: the prompt runs through the encoder half only
                # (layers 0..20, which is everything that writes global KV), and the decoder half is
                # replayed once over the last `window_size` prompt tokens.
                _snap_src = None
                if PREFIX_SNAPSHOTS:
                    _snap_src = host_ids if host_ids is not None else ids.tolist()
                for s, e in spans:
                    # Snapshot the state AT this chunk boundary, before it is overwritten. A later
                    # request that diverges inside this chunk can then resume here instead of
                    # prefilling the whole prompt.
                    if _snap_src is not None and s:
                        _snap = self._snapshot_prefix(_snap_src[:s], s)
                        if _snap is not None:
                            self._prefix_snapshots[s] = _snap
                            while len(self._prefix_snapshots) > PREFIX_SNAPSHOTS:
                                del self._prefix_snapshots[min(self._prefix_snapshots)]
                    with timing.chunk(s, e) if timing is not None else nullcontext():
                        _fwd(s, ids[s:e], need_logits=False, encoder_only=True)
                # Save at the prompt boundary, before decoder replay and generation overwrite the
                # rings. On a future exact extension, only ids[prefix_start:] are recomputed.
                self._save_prefix(host_ids if host_ids is not None else ids.tolist(), P)
                if (self.prefix_disk is not None and P > prefix_start
                        and self._prefix_cache is not None):
                    self.prefix_disk.save(host_ids if host_ids is not None else ids.tolist())
                logits, mh, s_rep = m.decoder_replay(need_logits=True)
                if self.spec:
                    m.dspark_seed(mh, s_rep)
            else:
                for s, e in spans:
                    with timing.chunk(s, e) if timing is not None else nullcontext():
                        chunk = ids[s:e]
                        last = e >= P
                        logits, mh = _fwd(s, chunk, need_logits=last)
                        if self.spec:
                            m.dspark_seed(mh, s)
        # The prompt's demand is now complete and no token has been generated yet: the whole
        # prompt's evidence can be applied before it is used, rather than after the answer is
        # written. Outside the read-ahead context so the swap's NVMe loads do not contend with
        # engram reads still in flight. `P - prefix_start` is what this request actually prefilled
        # -- a prefix-cache hit recomputes nothing and so contributes no new demand.
        self.predictive.finish(self)
        self.promote_user_prompt()
        self.maintain_inline(P - prefix_start)
        if self.user_prompt_stream:
            # The first answer is sampled from the prefill logits. Admission
            # AFTER those logits is too late for direct-letter answers, and
            # leaves the decoder window built with the previous residents.
            # Rebuild prompt state with the admitted residents. Both ranks run this
            # when the guarded policy is enabled, even for an empty selection.
            logits = self.refresh_user_prompt_logits(logits, ids, spans)
        # Vision inputs belong to the prompt only. Leaving them set would apply this request's
        # image mask to the next one's prefill, and the engine is single-sequence, so "the next
        # one" is any later request.
        self._token_types = self._images = None
        self.model.image_mask = self.model.engram_mask = None
        t_prefill = time.perf_counter() - t_start
        out_st["t_prefill"] = t_prefill
        out_st["prefix_cached_tokens"] = prefix_start
        pen = penalties if (penalties is not None and penalties.active) else None
        p = sample_probs(logits[-1], temperature, top_p)
        tok = int(torch.multinomial(p, 1)) if temperature > 0 else int(p.argmax())
        out = [tok]
        n_out = 1
        pos = P  # position of `tok` (not yet forwarded)
        accepted_hist = []
        # DSV41_SPEC_CONF=1: per step (verified depth, leading accepts, confidence logits [T_DRAFT]).
        # Rank-local observation; it never feeds back into the step, so the ranks cannot diverge on it.
        from engine.fastdecode import SPEC_CONF, TREE_PROBE
        conf_hist = [] if (SPEC_CONF and self.spec and self.fast is not None) else None
        tree_hist = [] if (TREE_PROBE and self.spec and self.fast is not None) else None
        probe = getattr(self.fast, "decode_probe", None)
        if probe is not None and probe.active:
            from engine.decode_probe_phases import DecodeLoopPhases
            ph = DecodeLoopPhases(StepPhases(), self.device,
                                  lambda: self.fast.graph_captures)
        else:
            ph = StepPhases() if STEP_TIMING else None
        if self.fast is not None:
            # the engram row wait and the graph queueing both live inside the "step" phase; record
            # where they started so the table can break that phase down. Taken unconditionally --
            # it is a dict of four floats, and decode_accounting below needs it to say anything
            # true about the graphed path, which is where all of decode's time actually goes.
            out_st["fd0"] = dict(self.fast.stats)
        out_st.update(n_out=1, steps=0, accepted=accepted_hist, conf=conf_hist, tree=tree_hist,
                      t_decode0=time.perf_counter(), phases=ph)
        steps = 0
        # The gate cannot be constraining anything yet (it engages on a marker that takes several
        # tokens to write), but it has to see every settled token to stay in step with the stream.
        if grammar is not None:
            grammar.observe([tok])
        yield [tok]
        # EP2 lockstep (engine/dist.py::control): rank 0 owns this condition and hands the
        # answer to rank 1 every step. Rank 1 evaluating it independently is what wedges the
        # pair, because the server-side stops -- a text stop sequence, a client disconnect,
        # the max_tokens re-clamp against an already-emitted burst -- are invisible there.
        # Inert at world 1: control() returns its argument before touching anything.
        # DSV41_BLOCK_DYNAMIC: rank 0's policy picks each step's depth and control() carries it,
        # so both ranks verify the same width. Without it the depth is the fixed T_DRAFT.
        pol = self.depth_policy if (self.spec and self.fast is not None) else None
        # Greedy can use all five scores. Sampling uses a proposal-prefix stopping rule
        # so inclusion of a draft never depends on that token or later proposals.
        conf_pol = self.confidence_depth_policy if pol is not None else None
        if conf_pol is not None:
            pol = conf_pol
        out_st["depth_policy"] = pol
        if pol is not None:
            pol.reset_request()
        lookup = None
        bypass = None
        if (self.spec and self.fast is not None and conf_hist is None and tree_hist is None):
            if LOOKUP_DRAFT_ENABLED and self.ep.rank == 0:
                from engine.lookup_draft import ExactDraftCache
                history = host_ids if host_ids is not None else ids.tolist()
                lookup = ExactDraftCache(list(history) + [tok], LOOKUP_DRAFT_NGRAM,
                                         LOOKUP_DRAFT_CANDIDATES)
            if getattr(self, 'draft_bypass_enabled', False):
                from engine.draft_bypass import DraftBypassPolicy
                bypass = DraftBypassPolicy()
                bypass.pinned = getattr(self, '_bypass_pinned', None)
        out_st['lookup_draft'] = lookup
        out_st['draft_bypass'] = bypass
        # Decode-time expert adaptation (engine/adapt_config.py: every ADAPT.decode_tokens
        # output tokens, gated on that stretch's miss rate). Rank 0 decides; control() carries
        # the flag so both ranks run the pass at the same step. A concurrent lane's control is
        # the same broadcast, and the pass runs while every other lane waits at its boundary.
        decode_adapt = (ADAPT.decode_tokens > 0 and self.ep.rank == 0 and self.ep.active
                        and getattr(self, "model_prune_mask", None))
        decode_snapshot = (self.model.decode_miss_snapshot if self.fast is not None
                           else self.model.miss_snapshot)
        self._decode_adapt_mark = ((n_out, decode_snapshot()) if decode_adapt else None)
        if self.fast is not None:
            self.fast._ahead = None   # never carry a queued draft across requests
        while self.ep.control(n_out < max_tokens and tok not in stop_ids,
                              self._decode_step_value(pol, lookup, bypass,
                                  n_out < max_tokens and tok not in stop_ids),
                              self._decode_adapt_due(n_out) if (decode_adapt and n_out < max_tokens
                                                               and tok not in stop_ids) else 0):
            if self.ep.control_flag:
                if self.fast is not None:
                    self.fast._ahead = None   # an adaptation pass may recapture graphs over d_out
                self.maintain_decode(n_out)
            t_iter = time.perf_counter()
            # A step that captures a CUDA graph (first use of a width/parity/bucket, or the
            # drafter) costs seconds; timing it would skew the policy's step-time estimate.
            n_graphs = (getattr(self.fast, "graph_captures", None)
                        if pol is not None or bypass is not None else None)
            mode = self.ep.control_value
            copied, root_only = mode < 0, mode == 0
            depth = abs(mode) if mode else 0
            if pol is not None and self.ep.rank == 0:
                sw = pol.pop_switch()
                if sw is not None:
                    # Rates are output tokens per second of verify loop. The depth that was not
                    # running is marked "would be": exact when coming down from the deep depth
                    # (same drafts, fewer checked), an estimate when going up.
                    other = sw["estimated"]
                    log(f"spec depth {sw['from']} -> {sw['to']} at output token {n_out}: "
                        f"{sw['tokens_per_step']} tokens/step over {sw['steps']} steps; "
                        + ", ".join(f"depth {d} {'would be ' if d == other else ''}{r} tok/s"
                                    for d, r in sorted(sw["rate"].items())))
            vblk, vout, vhost = self._vbufs[self._root_width if root_only else depth + 1]
            if ph is not None:
                ph.start()
            if self.spec:
                # the lean path applies to greedy decoding only; temperature > 0 keeps the original
                # sequential rejection-sampling loop so its RNG stream is bit-for-bit unchanged
                lean = LEAN_STEP and self.fast is not None and temperature <= 0 and not root_only
                if self.fast is not None:
                    if copied:
                        from engine.lookup_draft import deterministic_draft_probs
                        proposal = self.ep.broadcast_obj(getattr(self, '_lookup_step_tokens', None))
                        drafts = torch.tensor(proposal, dtype=torch.long, device=self.device)
                        q = deterministic_draft_probs(drafts, self.args.vocab_size)
                    elif root_only:
                        drafts = torch.empty(0, dtype=torch.long, device=self.device)
                        q = torch.empty((0, self.args.vocab_size), device=self.device)
                    else:
                        drafts, q = self.fast.draft(tok, pos - 1, temperature, top_p)
                    if conf_pol is not None:
                        if self.ep.rank == 0:
                            if copied or root_only:
                                chosen = depth
                            else:
                                choose = conf_pol.choose if temperature <= 0 else conf_pol.choose_sampled
                                chosen = choose(self.fast.d_conf.tolist())
                        else:
                            chosen = 0
                        # Both ranks always enter this second control call. Reuse the
                        # three-int stop protocol so release_peer() can also abort here.
                        if not self.ep.control(True, chosen):
                            break
                        depth = self.ep.control_value
                        vblk, vout, vhost = self._vbufs[self._root_width if root_only else depth + 1]
                    if depth < drafts.numel():
                        # dynamic depth: verify the first `depth` of the deeper draft
                        drafts, q = drafts[:depth], q[:depth]
                    if not lean:
                        # the drafter's static buffers survive until the next draft() call, which is
                        # after this step's verification, so the lean path does not need the copies
                        drafts = drafts.clone(); q = q.clone()
                    if ph is not None:
                        ph.mark("draft")
                    if root_only:
                        # Odd verification can preserve an unmatched compressor token.
                        # The original two-row path retains its discarded causal dummy.
                        block = vblk
                        block.fill_(tok)
                    elif lean:
                        # no H2D and no allocation: fill_ bakes the token into the kernel argument
                        block = vblk
                        block[0].fill_(tok)
                        block[1:].copy_(drafts)
                    else:
                        block = torch.cat([torch.tensor([tok], device=self.device), drafts])
                    if ph is not None:
                        ph.mark("block")
                    # The n-gram hashes are computed on the GPU and then pulled to the host so the
                    # engram reads can be issued, and that D2H blocks the host for ~14 ms/step.
                    # It looked like idle worth overlapping until this event pair priced the hash
                    # itself at 0.17 ms: the D2H is not waiting on the hash, it is waiting on the
                    # ~15.7 ms drafter graph queued ahead of it. The GPU is busy for that whole
                    # window, so there is nothing there to overlap -- keep the measurement so the
                    # next person does not rediscover it. `hash` (the host phase) only queues it.
                    _evh = self.fast._ev_begin("hash") if self.fast is not None else None
                    hashes = m.hash_state(block[None], pos)[0]
                    if _evh is not None:
                        _evh.record()
                    if ph is not None:
                        ph.mark("hash")
                    # both tables' rows are read in background threads (NVMe only, no CUDA calls there) and
                    # each is dequantized on the main thread when its layer needs it.
                    if EARLY_VERIFY and not getattr(self, "_cooperative_decode", False):
                        # The hash ids leave on a side stream, so the host does not wait for the
                        # drafter before it queues the verify: FastDecoder.step replays layer 0 (no
                        # Engram) first and calls `rows` only at the layer-1 boundary. Before this,
                        # .cpu() here returned when the drafter finished and the GPU then idled
                        # ~1.3 ms a step while the host prepared and queued layer 0. Same work,
                        # same stream order on the GPU; only the host's waiting moved.
                        rows = self._engram_rows_async(hashes)
                        if ph is not None:
                            ph.mark("hash_d2h")
                            ph.mark("submit")
                        logits, mh = self.fast.step(block, pos, rows)
                    else:
                        # The hash ids go to the host HERE, before any graph is queued: a .cpu()
                        # later would wait for the whole step.
                        h_np = hashes.cpu().numpy()
                        if ph is not None:
                            ph.mark("hash_d2h")
                        futs = {L: (self.eg_pool.submit(self.tables[L].read_raw, h_np[:, li, :]),
                                    self.tables[L].to_device)
                                for li, L in enumerate(self.args.engram_layer_ids)}
                        if ph is not None:
                            ph.mark("submit")
                        if getattr(self, "_cooperative_decode", False):
                            from engine.decode_events import VerifyStep
                            logits, mh = yield VerifyStep(block, pos, lambda: futs)
                        else:
                            logits, mh = self.fast.step(block, pos, lambda: futs)
                    if ph is not None:
                        ph.mark("step")
                else:
                    drafts, q, conf = m.dspark_draft(tok, pos - 1, temperature)
                    block = torch.cat([torch.tensor([tok], device=self.device), drafts])  # T_VERIFY tokens at pos..
                    logits, mh = m.forward(block, pos, prefill=False)
                if logits.is_inference():
                    logits = logits.clone()   # same reason as the single-token path below
                # Constrained decoding, before anything reads the logits. Row i is the
                # distribution after block[0..i], so its legal set depends on the drafts accepted
                # ahead of it; the gate walks the block token by token and leaves its own state
                # where it found it. Rows after the first illegal draft are left alone: that draft
                # is masked out of its own row, so it is rejected there and nothing reads further.
                if grammar is not None:
                    grammar.mask_rows(logits, block)
                    if ph is not None:
                        ph.mark("grammar")
                if pen is not None:
                    pen.apply(logits, out)
                # verify drafts[i] (position pos+1+i) against logits[i]
                if lean:
                    # Greedy verification, entirely on the GPU: argmax of the six logit rows, the
                    # five draft comparisons, and the number of LEADING accepts (cumprod-then-sum,
                    # which is the index of the first mismatch). The bonus token is argmax[a] for
                    # every a, including a = 5 -- exactly what the loop below computes. One 7-wide
                    # D2H is the only host sync of the whole step.
                    am = logits.argmax(-1)                                   # [6]
                    # Verify the block actually passed through the backbone.
                    # A cold graph capture warms the drafter and overwrites its
                    # static d_out buffer, which `drafts` aliases on this path.
                    acc = am[:depth].eq(block[1:]).to(torch.int32).cumprod(0)
                    vout[0] = acc.sum()
                    vout[1:].copy_(am)
                    # Both ranks take this branch together (lean and the modes come from control()),
                    # so the drafter's collectives stay paired.
                    ahead = (EARLY_DRAFT and not copied and lookup is None and bypass is None
                             and conf_pol is None and not getattr(self, "_cooperative_decode", False))
                    if ahead:
                        vhost.copy_(vout, non_blocking=True)
                        # the decode-adaptation check reads these at the top of a later step
                        m.stage_decode_miss()
                        _vready = torch.cuda.Event()
                        _vready.record()
                        ahead = self.fast.draft_ahead(vout[:1], am, pos)
                        _vready.synchronize()
                    else:
                        vhost.copy_(vout)
                    v = vhost.tolist()
                    a, cand = v[0], v[1:]
                    new = cand[:a]
                    bonus = cand[a]
                    for j, t in enumerate(new):   # an accepted stop token ends the block, no bonus
                        if t in stop_ids:
                            a, new, bonus = j + 1, new[:j + 1], None
                            break
                    if ph is not None:
                        ph.mark("verify")
                    m.c.rollback(pos + a + 1)
                    if ph is not None:
                        ph.mark("rollback")
                    accepted_hist.append(a)
                    if lookup is not None:
                        if copied:
                            lookup.record_accept(a)
                    if conf_hist is not None:
                        conf_hist.append((depth, a, self.fast.d_conf.tolist()))
                    if tree_hist is not None:
                        # (depth, leading accepts, verifier argmax per row, drafter top-2 per row)
                        tree_hist.append((depth, a, list(cand[:depth]), self.fast.d_top2[:depth].tolist()))
                    emitted = list(new)
                    if bonus is not None:
                        emitted.append(bonus)
                    # A speculative step can settle up to T_VERIFY tokens at once, but the
                    # caller's limit is on emitted tokens, not verifier steps.  Without this
                    # clamp the final burst overshot max_tokens even though the loop condition
                    # itself was correct; test_spec_lossless then had identical shared prefixes
                    # but different lengths.
                    emitted = emitted[:max_tokens - n_out]
                    pos = pos + a + 1
                    tok = emitted[-1] if emitted else tok
                    if ahead and bonus is not None:
                        # what draft() will be asked next; a stop or a clamp ends the loop instead
                        self.fast._ahead = (tok, pos - 1)
                    if emitted:
                        if lookup is not None:
                            lookup.extend(emitted)
                        if pen is not None:
                            pen.observe(emitted)
                        out += emitted
                        n_out += len(emitted)
                        out_st["n_out"] = n_out
                        if grammar is not None:
                            grammar.observe(emitted)
                        if ph is not None:
                            ph.mark("emit")
                        yield emitted
                        if ph is not None:
                            ph.mark("consumer")
                        if any(t in stop_ids for t in emitted):
                            break
                    if pol is not None and self.ep.rank == 0 and not copied:
                        pol.observe(depth, a, len(emitted), self._policy_step_s(t_iter, n_graphs))
                    if bypass is not None and self.ep.rank == 0 and not copied:
                        bypass.observe(False, len(emitted), self._policy_step_s(t_iter, n_graphs))
                    steps += 1
                    out_st["steps"] = steps
                    if ph is not None:
                        ph.steps = steps
                    continue
                # Time the sampling envelope after the backbone. The opt-in batched path uses
                # fixed-count random draws and one compact result transfer. The default keeps
                # the historical, variable-length RNG stream and sequential stop semantics.
                _ev_sample = self.fast._ev_begin("sample") if self.fast is not None else None
                if self.batched_verify and temperature > 0 and 0 < top_p <= 1 and not root_only:
                    a, new, bonus = self._verify_sampled(
                        logits, q, drafts, temperature, top_p, stop_ids)
                else:
                    a = 0
                    new = []
                    bonus = None
                    # Use the actual width for both fast and eager paths (DSV41_BLOCK varies).
                    for i in range(len(drafts)):
                        pt = sample_probs(logits[i], temperature, top_p)
                        d = int(drafts[i])
                        if temperature <= 0:
                            ok = int(pt.argmax()) == d
                        else:
                            r = torch.rand((), device=self.device)
                            ok = bool(r < (pt[d] / q[i][d].clamp_min(1e-20)).clamp(max=1.0))
                        if ok:
                            a += 1
                            new.append(d)
                            if d in stop_ids:
                                break
                        else:
                            if temperature <= 0:
                                bonus = int(pt.argmax())
                            else:
                                resid = (pt - q[i]).clamp_min(0)
                                if float(resid.sum()) <= 0:
                                    resid = pt
                                bonus = int(torch.multinomial(resid / resid.sum(), 1))
                            break
                    if bonus is None and not (new and new[-1] in stop_ids):
                        pt = sample_probs(logits[a], temperature, top_p)
                        bonus = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                if _ev_sample is not None:
                    _ev_sample.record()
                if ph is not None:
                    ph.mark("verify")
                # caches valid for positions < pos + a + 1 (tok + accepted drafts)
                m.c.rollback(pos + a + 1)
                if self.fast is None:
                    m.dspark_seed(mh[:a + 1], pos)
                if ph is not None:
                    ph.mark("rollback")
                accepted_hist.append(a)
                if lookup is not None and copied:
                    lookup.record_accept(a)
                if conf_hist is not None:
                    conf_hist.append((depth, a, self.fast.d_conf.tolist()))
                emitted = list(new)
                if bonus is not None:
                    emitted.append(bonus)
                emitted = emitted[:max_tokens - n_out]
                pos = pos + a + 1
                tok = emitted[-1] if emitted else tok
                if emitted:
                    if lookup is not None:
                        lookup.extend(emitted)
                    if pen is not None:
                        pen.observe(emitted)
                    out += emitted
                    n_out += len(emitted)
                    out_st["n_out"] = n_out
                    if grammar is not None:
                        grammar.observe(emitted)
                    if ph is not None:
                        ph.mark("emit")
                    yield emitted
                    if ph is not None:
                        ph.mark("consumer")
                    if any(t in stop_ids for t in emitted):
                        break
                if (pol is not None and self.ep.rank == 0 and not copied
                        and (not root_only or getattr(pol, 'lo', None) == 0)):
                    pol.observe(depth, a, len(emitted), self._policy_step_s(t_iter, n_graphs))
                if bypass is not None and self.ep.rank == 0 and not copied:
                    bypass.observe(root_only, len(emitted), self._policy_step_s(t_iter, n_graphs))
                steps += 1
                out_st["steps"] = steps
                if ph is not None:
                    ph.steps = steps
            else:
                logits, mh = m.forward(torch.tensor([tok], device=self.device), pos, prefill=False)
                # Model.forward runs under @torch.inference_mode, so what comes back is an
                # INFERENCE tensor, and this loop is not inside that mode -- mutating one here
                # raises rather than writing. Both the grammar mask and the penalties write in
                # place, so give them a normal tensor. The graphed path never hit this because
                # its logits live in static buffers allocated at capture time, which is why the
                # whole eager branch could rot unnoticed. 517 KB per token, once.
                if logits.is_inference():
                    logits = logits.clone()
                if grammar is not None:
                    grammar.mask_rows(logits, None)
                if pen is not None:
                    pen.apply(logits, out)
                pt = sample_probs(logits[0], temperature, top_p)
                tok = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                pos += 1
                if pen is not None:
                    pen.observe([tok])
                out.append(tok)
                n_out += 1
                steps += 1
                out_st.update(n_out=n_out, steps=steps)
                if grammar is not None:
                    grammar.observe([tok])
                yield [tok]
        out_st.update(n_out=n_out, steps=steps)

    def _decode_step_value(self, pol, lookup, bypass, keep_going=True):
        """Rank 0 selects a proposal mode before any current proposal is drawn.

        Positive = DSpark depth; negative = copied depth; zero = root-only.
        Every rank follows the same control value, including all graph collectives. Rank 1's 0 is
        a placeholder the broadcast replaces -- concurrent lanes included (ScheduledPeer.control).
        """
        if self.ep.rank != 0 or not keep_going:
            return 0
        depth = pol.decide() if pol is not None else self._tv - 1
        self._lookup_step_tokens = lookup.propose(depth) if lookup is not None else None
        if self._lookup_step_tokens is not None:
            return -depth
        if bypass is not None and bypass.decide():
            return 0
        return depth

    # ------------------------------------------------------------------ introspection
    supports_penalties = True

    def decode_probe_status(self):
        """CPU metadata only; safe for observer polling during generation."""
        reply = dict(self._decode_probe_reply)
        probe = getattr(self.fast, "decode_probe", None)
        if probe is not None:
            reply["local_status"] = probe.status()
        return reply

    def decode_probe_command(self, body):
        """Mirrored, idle command. Every rank reaches the same final gather.

        The server owns the request lock/broadcast. Isolation never executes a
        collective or changes the live model, arena, KV cache or RNG state.
        Clearing instrumented graphs precedes releasing their event/copy owners.
        """
        from dataclasses import asdict
        from engine.decode_probe import DecodeProbe, ProbeConfig
        action = body.get("action", "arm")
        local = {"rank": self.ep.rank, "ok": False}
        try:
            fd = self.fast
            if fd is None or not fd.use_graphs:
                raise RuntimeError("decode probe needs speculative CUDA graphs")
            probe = fd.decode_probe
            if action == "run" and probe is not None:
                opts = asdict(probe.config)
                opts.update(body)
                cfg = ProbeConfig.from_dict(opts)
            else:
                cfg = ProbeConfig.from_dict(body)
            if action == "arm":
                fd.release_graphs()
                if probe is not None:
                    probe.disarm()
                fd.decode_probe = DecodeProbe(cfg, device=self.device,
                                             isolation_stream=fd.capture_warmup_stream)
                local["report"] = fd.decode_probe.arm()
                self._decode_probe_phases = None
            elif action == "run":
                if probe is None or not probe.active:
                    raise RuntimeError("arm the decode probe and generate before running it")
                local["report"] = probe.profile_idle(cfg)
                if self._decode_probe_phases is not None:
                    local["report"]["loop_phases"] = self._decode_probe_phases.report()
            else:
                fd.release_graphs()
                if probe is not None:
                    probe.disarm()
                fd.decode_probe = None
                self._decode_probe_phases = None
                local["report"] = {"status": "stopped", "snapshot_bytes": 0}
            local["ok"] = True
        except Exception as exc:
            # Local failures cannot strand the healthy peer at the rendezvous.
            # Arbitrary library exception text may contain activation values.
            local["error"] = type(exc).__name__
        ranks = self.ep.gather_objects(local)
        ok = all(item["ok"] for item in ranks)
        if action == "arm" and not ok and self.fast is not None:
            # Shared outcome rolls back both ranks, even after one local failure.
            self.fast.release_graphs()
            if self.fast.decode_probe is not None:
                self.fast.decode_probe.disarm()
            self.fast.decode_probe = None
        self._decode_probe_reply = {"version": 1, "action": action, "ok": ok, "ranks": ranks}
        return self._decode_probe_reply

    def config(self):
        """Static engine configuration -- everything a measured number has to be quoted with."""
        from engine.native_head import head_kernel
        from engine.tensor_parallel import draft_head_bytes
        return {
            "engine": "v41",
            "decode_experiments": self.experiments.boot_fields(),
            "ep_world_size": self.ep.world,
            "ep_rank": self.ep.rank,
            "tp_experts": self.ep.tensor_parallel,
            "tp_draft_attention": os.environ.get('DSV41_TP_DRAFT_ATTN', '0') == '1',
            "draft_markov_topk": int(os.environ.get('DSV41_DRAFT_MARKOV_TOPK', '0')),
            "draft_markov_local": os.environ.get('DSV41_DRAFT_MARKOV_LOCAL', '0') == '1',
            "prefill_dual_rail": self.ep.prefill_dual_rail,
            "collective_rails_version": DT.collective_rails.VERSION,
            "max_concurrency": int(os.environ.get("DSV41_MAX_CONCURRENCY", "1")),
            "arena_gb": self.arena_gb,
            "arena_slots": self.slots,
            "lru_slots": self.store.lru_slots,
            "transient_slots": self.store.transient_slots,
            # experts the arena holds, not slots: exact-slot arenas carry margin/transient slots too
            "resident_expert_pct": round(min(len(self.store.lru), self.args.n_layers * self.args.n_routed_experts)
                                         / (self.args.n_layers * self.args.n_routed_experts) * 100, 1),
            "max_seq": self.max_context,
            "index_topk": self.args.index_topk,
            "candidate_topk_blocks": self.args.candidate_topk_blocks,
            "candidate_block_size": self.args.candidate_block_size,
            "spec": self.spec,
            "trace_stats": self.trace_stats,
            "kernel": self.kernel,
            "expert_format": self.expert_format,
            "expert_pack_sha256": (self.store.pack.header_sha256() if hasattr(self.store, "pack") else ""),
            "expert_pack_source": (self.store.pack.header.get("source_sha256", "")
                                   if hasattr(self.store, "pack") else ""),
            "expert_mb": round(self.expert_bytes / 1e6, 2),
            "dense_fp4": ",".join(sorted(R.dense_fp4_groups())) or "off",
            # Which parallelization is actually live. EP2 is what ships (dense TP measured
            # slower, see R.tp_dense), so the useful question this answers is "did the restart
            # really go back?" -- the flag lives in the launcher's environment, not in .env, and
            # there was otherwise no way to tell the two modes apart from outside the container.
            "tp_dense": "on" if R.tp_dense()[1] > 1 else "off",
            "tp_attention": os.environ.get('DSV41_TP_ATTN', '0') == '1',
            "tp_head": os.environ.get('DSV41_TP_HEAD', '0') == '1',
            "tp_embed": os.environ.get('DSV41_TP_EMBED', '0') == '1',
            "tp_draft_experts": self.tp_draft_experts,
            "tp_draft_dense": self.tp_draft_dense,
            "tp_expert_reduce": os.environ.get('DSV41_TP_EXPERT_REDUCE', 'all_reduce'),
            "tp_expert_layout": os.environ.get('DSV41_TP_EXPERT_LAYOUT', 'output'),
            "tp_linear_layout": os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output'),
            "prefill_swap": "on" if PREFILL_SWAP else "off",
            # How many adaptations have been applied since boot. Identical on both ranks by
            # construction, so a difference between them is a desync that nothing else reports.
            "expert_generation": self.expert_generation,
            "head_fmt": R.head_fmt(),
            "head_kernel": head_kernel(),
            "head_weight_bytes": draft_head_bytes(self.W.head),
            "fp4_dense_split": os.environ.get("DSV41_FP4_DENSE_SPLIT", "auto"),
            "comm_backend": os.environ.get("DSV41_COMM_BACKEND", "roce"),
            "roce_max_kb": os.environ.get("DSV41_ROCE_MAX_KB", "256"),
            "ablate_wob": R_ablit.enabled(),
            "dream_overlay": R_dream.digest(),
            "draft_head_fmt": R.draft_head_fmt(),
            "markov_head_fmt": _FD.MARKOV_HEAD_FMT,
            "tp_draft_head": os.environ.get('DSV41_TP_DRAFT_HEAD', '0') == '1',
            "draft_head_bytes": getattr(self.W, 'draft_head_bytes', 0),
            "draft_vocab": getattr(self.W, 'draft_vocab_digest', None),
            "draft_vocab_rows": (_DRAFT_VOCAB.rows() if getattr(self.W, 'draft_vocab_digest', None) else 0),
            "expert_topp": (list(self.fast.expert_topp) if getattr(self, 'fast', None) is not None
                            and self.fast.expert_topp is not None else None),
            "lookup_draft_enabled": LOOKUP_DRAFT_ENABLED,
            "lookup_draft_ngram": LOOKUP_DRAFT_NGRAM,
            "draft_bypass": self.draft_bypass_enabled,
            "hc_kernel": R.HC_KERNEL,
            "hc_prec": R.HC_PREC,
            "dense_dequant_cache": os.environ.get("DSV41_DENSE_DEQUANT_CACHE", "0") == "1",
            "fp4_dot_scaled": self.fp4_dot_scaled,
            "fp4_prefill_dot_scaled": self.fp4_prefill_dot_scaled,
            "fp4_prefill_scaled_tiles": os.environ.get("DSV41_FP4_PREFILL_SCALED_TILES", "0") == "1",
            "fp4_cuda": self.fp4_cuda,
            "fp4_cuda_relaxed": self.fp4_cuda_relaxed,
            "fp4_cuda_v2": self.fp4_cuda_v2,
            "fp4_cuda_down_rows": self.fp4_cuda_down_rows,
            "fp4_cuda_source": self.fp4_cuda_source,
            "exl3_cuda_source": self.exl3_cuda_source,
            "early_verify": EARLY_VERIFY,
            "early_draft": EARLY_DRAFT,
            "exl3_dense": (os.path.basename(self.exl3_dense.path) if self.exl3_dense is not None else None),
            "exl3_dense_gib": (round(self.exl3_dense.bytes / 2**30, 2) if self.exl3_dense is not None else 0),
            "exl3_draft_dense": (os.path.basename(self.exl3_draft.path) if self.exl3_draft is not None else None),
            "exl3_draft_dense_gib": (round(self.exl3_draft.bytes / 2**30, 3) if self.exl3_draft is not None else 0),
            "exl3_merged": (os.environ.get("DSV41_EXL3_MERGED", "1") == "1"
                            if self.expert_format == "exl3" else None),
            "batched_verify": self.batched_verify,
            "batched_verify_source": self.batched_verify_source,
            "kv_cache_qdq": bool(M_.KV_CACHE_QDQ),
            "packed_kv": bool(M_.PACKED_KV),
            "hc_ops": bool(M_.HC_OPS),
            "hc_fused": bool(M_.HC_FUSED),
            "hc_mm_tile": R.HC_MM_TILE,
            "decode_shared_overlap": os.environ.get("DSV41_DECODE_SHARED_OVERLAP", "0") == "1",
            "fp8_decode_block_n": os.environ.get("DSV41_FP8_DECODE_BLOCK_N", "auto"),
            "fp8_decode_warps": int(os.environ.get("DSV41_FP8_DECODE_WARPS", "4")),
            "fp8_decode_pipeline": os.environ.get("DSV41_FP8_DECODE_PIPELINE", "0") == "1",
            "attn_staged": getattr(self.fast, "staged_attention", False),
            "decode_merged_proj": getattr(self.fast, "merged_proj", 0),
            "decode_lean": getattr(self.fast, "lean", None) is not None,
            "block_confidence": getattr(self, "confidence_depth_policy", None) is not None,
            "prefill_graphs": (self.model.prefill_graphs.report()
                               if getattr(self.model, 'prefill_graphs', None) else None),
            "prune_miss_fused": os.environ.get("DSV41_PRUNE_MISS_FUSED", "1") == "1",
            "fp8_act_qdq_fused": os.environ.get("DSV41_FP8_ACT_QDQ_FUSED", "1") == "1",
            "engram_native": os.environ.get("DSV41_ENGRAM_NATIVE", "0") == "1",
            "engram_direct": os.environ.get("DSV41_ENGRAM_DIRECT", "0"),
            "hc_prefill_fused": os.environ.get("DSV41_HC_PREFILL_FUSED", "0") == "1",
            "prefill_attn_indexed": os.environ.get("DSV41_PREFILL_ATTN_INDEXED", "0") == "1",
            "index_fused": os.environ.get("DSV41_INDEX_FUSED", "0") == "1",
            "engram_cache_mb": round(sum(t.row_cache.report()['allocated_bytes'] for t in self.tables.values()
                                         if t.row_cache is not None) / 1024**2, 3),
            "engram_native_threads": int(os.environ.get("DSV41_ENGRAM_NATIVE_THREADS", "64")),
            "moe_fallback": self.kernel == "dequant-fallback",
            "routed_topk": self.args.n_activated_experts,
            "hot_reads": ({**self.fast.hot.boot_fields(), **self.fast.hot.stats}
                          if getattr(getattr(self, 'fast', None), 'hot', None) is not None else None),
            "sim_cb2_frac": self.sim_cb2_frac,
            "act_quant": self.act_quant,
            "cycle_break": os.environ.get("DSV41_CYCLE_BREAK", "0") == "1",
            "no_repeat_ngram": int(os.environ.get("DSV41_NO_REPEAT_NGRAM", "0")),
            "swa_replay": self.swa_replay,
            "prefill_skip_null": (self.ep.active and self.kernel == "triton-fp4" and
                                  os.environ.get("DSV41_PREFILL_SKIP_NULL", "1") == "1"),
            "prefill_fixed_routing": (bool(getattr(self.model, "prefill_routes", None)) and
                                      os.environ.get("DSV41_PREFILL_FIXED_ROUTING", "1") == "1"),
            "prefix_cache": (self.swa_replay and os.environ.get("DSV41_PREFIX_CACHE", "1") == "1"),
            "prefix_snapshots": PREFIX_SNAPSHOTS,
            "prefix_response": os.environ.get('DSV41_PREFIX_RESPONSE', '1') == '1',
            "prefix_response_last": getattr(self, '_prefix_response_last', None),
            "prefix_disk": self.prefix_disk is not None,
            "prefix_disk_configured": PREFIX_DISK,
            "prefix_disk_min_interval_s": PREFIX_DISK_MIN_INTERVAL_S,
            "prefix_disk_dedup": PREFIX_DISK_DEDUP,
            "prefix_disk_last": self.prefix_disk.stats if self.prefix_disk is not None else None,
            "prefill_fused_attn": M_.PREFILL_FUSED_ATTN,   # not a second os.environ.get: see model.py
            "prefill_engram_prefetch": os.environ.get("DSV41_PREFILL_ENGRAM", "1") == "1",
            "prune_keep": self.prune_keep,
            "prune_metric": ADAPT.metric,
            **self.expert_profile.report(),
            "initial_keep_sha256": self.initial_keep_sha256,
            "expert_seed_sha256": self.expert_seed_sha256,
            "vision_mode": self.vision_mode,
            "vision_enabled": self.vision is not None,
            "vision_weights_local": self.vision is not None and (self.vision_mode != 'peer' or self.ep.rank == 1),
            "prune_score_history_version": 2,
            "prune_select": getattr(self, "prune_select", None),
            "prune_layer_counts": self.prune_layer_counts,
            "dynamic_experts": self.dynamic_experts,
            "resident_budget": self.resident_budget,
            "predictive_prefill": dict(self.predictive.last),
            "scale_pack": self.scale_pack.boot_fields() if self.scale_pack is not None else None,
            "dynamic_layer_floor": self.dynamic_layer_floor if self.dynamic_experts else None,
            "prune_layer_weights": self.prune_layer_weights,
            "resident_layer_counts": self.resident_layer_counts,
            "stream_layers": sorted(self.stream_layers),
            "layer_ablation": self.layer_ablation.report(),
            "user_prompt_stream": self.user_prompt_stream,
            "user_prompt_max_loads": self.user_prompt_max_loads,
            "user_prompt": self.user_prompt.report(),
            "critical_stream": self.critical.report(),
            "hot_profile": self.hot_profile,
            "prefill_chunk": MAX_CHUNK,
            "prefill_budget": {"policy": vars(self.prefill_budget), "last": dict(self.prefill_budget_last)},
            "l2_prefetch": {"mode": os.environ.get("DSV41_L2PF_MODE", "touch"),
                            "mb": int(os.environ.get("DSV41_L2PF_MB", "2")),
                            "attention_mb": self.fast.attn_prefetch_mb if self.fast is not None else 0,
                            "pace_gbps": int(os.environ.get("DSV41_L2PF_PACE_GBPS", "0"))},
            "decode_memory": self.fast.memory_report() if self.fast is not None else None,
            "router_fused_tail": os.environ.get("DSV41_ROUTER_FUSED_TAIL", "0") == "1",
            "router_bf16": os.environ.get("DSV41_ROUTER_BF16", "0") == "1",
            "verify_odd": os.environ.get("DSV41_VERIFY_ODD", "0") == "1",
            "io_threads": self.store.io_threads,
            "read_threads": self.store.read_threads,
            "read_chunk_mb": round(self.store.read_chunk / 1024 / 1024, 2),
        }

    def stats(self):
        return {**self.config(), **self.last_stats}

    def close(self):
        native = {id(g): g for t in self.tables.values()
                  for g in (getattr(t, 'native_gather', None), getattr(t, 'direct_gather', None)) if g is not None}
        native = list(native.values())
        if native:
            # No read future may enter the native context after it is destroyed.
            self.eg_pool.shutdown(wait=True)
            for gather in native:
                gather.close()
        for pool in [self.store.pool, self.store.read_pool] + [t.pool for t in self.tables.values()]:
            try:
                pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ replay verification
    @torch.inference_mode()
    def verify_replay(self, texts: list[str]):
        """Full 40-layer prefill vs CED encoder + Decoder SWA Bounded Replay, same prompt, one load.

        For a prompt of at most `window_size` tokens the replay covers the whole prompt and the two
        paths are the *same* arithmetic in the same order, so the logits must agree to the last bit;
        anything else means the state carried across the split (hidden states, the shifted HC pre-mix,
        layer 20's top-k and candidate pool) is not what the fused pass had. For a longer prompt the
        replay is an approximation by construction -- the decoder layers see a 128-token window
        instead of the full prefix -- and what matters is that the next-token distribution barely
        moves.
        """
        m = self.model
        out = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            P = len(ids)
            t = torch.tensor(ids, dtype=torch.long, device=self.device)

            def full():
                self._reset(); m.begin_prompt()
                lg, _ = m.forward(t, 0, prefill=True, need_logits=True)
                return lg[-1].float()

            def replay():
                self._reset(); m.begin_prompt()
                for s in range(0, P, MAX_CHUNK):
                    m.forward(t[s:s + MAX_CHUNK], s, prefill=True, need_logits=False, encoder_only=True)
                lg, _, _ = m.decoder_replay(need_logits=True)
                return lg[-1].float()

            a_, b_ = full(), replay()
            la, lb = torch.log_softmax(a_, -1), torch.log_softmax(b_, -1)
            kl = float((la.exp() * (la - lb)).sum())
            r = {"tokens": P, "replayed": min(P, self.args.window_size),
                 "exact_expected": P <= self.args.window_size,
                 "max_abs_logit_delta": round(float((a_ - b_).abs().max()), 6),
                 "top1_full": int(a_.argmax()), "top1_replay": int(b_.argmax()),
                 "kl_nats": round(kl, 6),
                 "nll_delta_on_full_top1": round(float(la[a_.argmax()] - lb[a_.argmax()]), 6)}
            log(json.dumps(r))
            out.append(r)
        return out

    # ------------------------------------------------------------------ teacher forcing
    @torch.inference_mode()
    def teacher_forced(self, corpus_path: str, max_len: int = 512, replay: bool = False, tail: bool = False):
        """Run every corpus sequence through `Model.forward` in ONE chunk and report mean NLL and
        top-1 next-token accuracy per category -- the same measurement
        `tools/expert_trace.py` makes with the pure-torch tracer, so the two are directly
        comparable and any drift between `engine/model.py` and `tools/v41_ref.py` shows up here.

        ``replay`` prefills through the CED encoder + Decoder SWA Bounded Replay path instead, which
        only produces logits for the last `window_size` positions; ``tail`` scores only those
        positions in either mode, so the two are measured on exactly the same predictions."""
        res = {}
        per_seq = []
        W = self.args.window_size
        for line in open(corpus_path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = self.tokenizer.encode(d["text"], add_special_tokens=False)
            assert len(ids) <= max_len, (d["id"], len(ids))
            self._reset()
            self.model.begin_prompt()
            t0 = time.perf_counter()
            P = len(ids)
            t = torch.tensor(ids, dtype=torch.long, device=self.device)
            if replay:
                for s0 in range(0, P, MAX_CHUNK):
                    self.model.forward(t[s0:s0 + MAX_CHUNK], s0, prefill=True, need_logits=False,
                                       encoder_only=True)
                logits, _, off = self.model.decoder_replay(need_logits=True)
            else:
                logits, _ = self.model.forward(t, 0, prefill=True, need_logits=True)
                off = 0
            if tail:
                keep = max(0, P - W) - off  # first scored row inside `logits`
                if keep > 0:
                    logits, off = logits[keep:], off + keep
            tgt = torch.tensor(ids[off + 1:], device=self.device)
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            nll = -lp.gather(1, tgt[:, None]).squeeze(1)
            top1 = (logits[:-1].argmax(-1) == tgt).float()
            c = res.setdefault(d["category"], {"nll": [], "top1": []})
            c["nll"].append(nll.cpu()); c["top1"].append(top1.cpu())
            per_seq.append({"id": d["id"], "category": d["category"], "n": len(ids), "scored_from": off,
                            "nll_by_pos": [round(v, 4) for v in nll.tolist()],
                            "mean_nll": round(float(nll.mean()), 4), "top1_acc": round(float(top1.mean()), 4),
                            "seconds": round(time.perf_counter() - t0, 2)})
            log(f"{d['id']}: n={len(ids)} nll={float(nll.mean()):.4f} top1={float(top1.mean()):.4f} "
                f"({time.perf_counter() - t0:.1f}s)")
        summary = {k: {"mean_nll": float(torch.cat(v["nll"]).mean()), "top1_acc": float(torch.cat(v["top1"]).mean()),
                       "n": int(torch.cat(v["nll"]).numel())} for k, v in res.items()}
        return {"summary": summary, "config": self.config(), "replay": replay, "tail": tail,
                "per_seq": per_seq}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--arena-gb", type=float, default=None)
    ap.add_argument("--trace-stats", default=None)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    ap.add_argument("--prompt-file", default=None, help="read the prompt from this file instead of --prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-swa-replay", action="store_true",
                    help="run all 40 layers over the whole prompt instead of CED + bounded replay")
    ap.add_argument("--hot-profile", default=None, choices=["mixed", "coding", "general"],
                    help="which slice of the traced corpus ranks the warm-start hot set")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--act-quant", action="store_true",
                    help="fake-quantize activations to fp8 like the reference (and like the tracer)")
    ap.add_argument("--verify-replay", action="store_true",
                    help="one load: full 40-layer prefill vs encoder + decoder SWA bounded replay")
    ap.add_argument("--teacher-forced", default=None,
                    help="corpus jsonl: run each sequence through Model.forward in one chunk and report NLL/top-1")
    ap.add_argument("--tf-out", default=None, help="write the teacher-forced result JSON here")
    ap.add_argument("--prune-sweep", default=None,
                    help="with --teacher-forced: comma list of keep fractions (e.g. 0.25,0.4,1.0); per layer only the "
                         "top-N experts by trace frequency stay routable; writes --tf-out with one entry per fraction")
    ap.add_argument("--prune-profile", default="mixed", choices=["mixed", "coding", "general"])
    ap.add_argument("--prune-select", default="uniform", choices=["uniform", "global"],
                    help="uniform: top-N per layer; global: one cross-layer ranking under the same total budget")
    ap.add_argument("--transient-slots", type=int, default=None,
                    help="prefill-miss ring slots (default 400 = a whole layer; 16 is enough when every kept expert is resident)")
    ap.add_argument("--keep-free-gb", type=float, default=None, help="host memory to leave free when auto-sizing the arena (default 20)")
    ap.add_argument("--expert-format", default=os.environ.get("EXPERT_FORMAT") or "fp4", choices=["fp4", "cb3", "exl3"],
                    help="routed-expert arena format: fp4 = the checkpoint's packed FP4 (18.80 MB/expert); "
                         "cb3 = the 3-bit per-row codebook format (14.45 MB/expert, 0.769x), packed at warm "
                         "start, which fits ~40.8%% of all routed experts in 90.5 GB instead of 31.3%%; "
                         "exl3 = the EXL3 trellis slices from DSV41_EXL3_PACK (6.67 MB/expert at TP2, 3 bits)")
    ap.add_argument("--sim-bits", type=int, default=None, help="simulate a 2/3-bit per-row codebook expert format (quality only)")
    ap.add_argument("--sim-cold-frac", type=float, default=1.0, help="fraction of the kept experts (coldest first) that get --sim-bits")
    ap.add_argument("--sim-cb2-frac", type=float, default=float(os.environ.get("SIM_CB2_FRAC") or 0.0),
                    help="with --expert-format cb3: give the coldest fraction of the kept experts a 2-bit "
                         "row codebook inside their (unchanged) CB3 slot. Quality only -- no bytes are "
                         "saved; it measures what a real 2-bit tier would cost")
    ap.add_argument("--prune-keep", type=float, default=None,
                    help="serve a REAP-style pruned model: only the top-F experts per layer are routable and all of "
                         "them are resident (F <= ~0.25 fits the arena on a 128 GB box)")
    ap.add_argument("--tf-ab", action="store_true",
                    help="with --teacher-forced: score the last window_size positions of every "
                         "sequence twice in one load -- full 40-layer prefill vs encoder + decoder "
                         "SWA bounded replay -- and report the NLL delta the approximation costs")
    ap.add_argument("--spec-ab", action="store_true",
                    help="one load, three runs: greedy without spec, greedy with spec (must match) and "
                         "one sampled spec run at --temperature/--top-p")
    ap.add_argument("--ab-out", default=None, help="write the --spec-ab result JSON here")
    a = ap.parse_args()
    if a.prompt_file:
        a.prompt = open(a.prompt_file).read()
    eng = V41Engine(a.model_dir, max_seq=a.max_seq, arena_gb=a.arena_gb, trace_stats=a.trace_stats,
                    spec=not a.no_spec, act_quant=a.act_quant,
                    swa_replay=(False if a.no_swa_replay else None), hot_profile=a.hot_profile, prune_keep=a.prune_keep, prune_select=a.prune_select,
                    sim_bits=a.sim_bits, sim_cold_frac=a.sim_cold_frac, expert_format=a.expert_format,
                    sim_cb2_frac=a.sim_cb2_frac,
                    **({"transient_slots": a.transient_slots} if a.transient_slots else {}),
                    **({"keep_free_gb": a.keep_free_gb} if a.keep_free_gb else {}))
    if a.verify_replay:
        short = "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
        res = eng.verify_replay([short, open(os.path.join(HERE, "..", "corpus", "sources", "dsv41_readme.md")).read()[:1400]])
        print(json.dumps(res, indent=1))
        raise SystemExit(0)
    if a.teacher_forced and a.prune_sweep:
        import math as _m
        if a.prune_profile == "mixed":
            cc, cg = EX.category_counts(a.trace_stats, "coding"), EX.category_counts(a.trace_stats, "general")
            counts = {L: cc[L] / cc[L].sum() + cg[L] / cg[L].sum() for L in cc if L in cg}
        else:
            counts = EX.category_counts(a.trace_stats, a.prune_profile)
        assert len(counts) == 40, f"trace npz files missing next to {a.trace_stats}: {len(counts)} layers"
        out = {}
        for frac in [float(x) for x in a.prune_sweep.split(",")]:
            n_keep = max(6, _m.ceil(frac * 384))
            masks, keep = build_keep_masks(counts, frac, a.prune_select, eng.device)
            eng.model.prune_mask = masks if frac < 1.0 else None
            t0 = time.time()
            res = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq))
            res["keep_frac"] = frac; res["experts_per_layer"] = n_keep; res["resident_gb_fp4"] = round(n_keep * 40 * EX.EXPERT_BYTES / 1e9, 1)
            res["select"] = a.prune_select; res["per_layer_counts"] = [int(len(keep[L])) for L in range(40)]
            res["seconds"] = round(time.time() - t0, 1)
            out[str(frac)] = res
            log(f"prune keep={frac} {a.prune_select} ({n_keep}/384 per layer avg, {res['resident_gb_fp4']} GB): {json.dumps({k: v for k, v in res.items() if k in ('coding', 'general')})}")
            if a.tf_out:
                json.dump(out, open(a.tf_out, "w"), indent=1)
        raise SystemExit(0)
    if a.teacher_forced:
        if a.tf_ab:
            full = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq), replay=False, tail=True)
            rep = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq), replay=True, tail=True)
            # A bounded replay is worst at the START of the replayed window (the query at the
            # replay start sees a one-token window) and exact-ish at the END, and only the LAST
            # position's logits ever produce a token, so a single mean over the whole window
            # says almost nothing about serving. Bucket the delta by distance from the end.
            def _buckets(run):
                out = {}
                for k in (1, 8, 32, 128):
                    vals = [v for r in run["per_seq"] for v in r["nll_by_pos"][-k:]]
                    out[f"last{k}"] = (round(float(np.mean(vals)), 4), len(vals))
                return out
            fb, rb = _buckets(full), _buckets(rep)
            by_dist = {k: {"full_nll": fb[k][0], "replay_nll": rb[k][0],
                           "delta_nats": round(rb[k][0] - fb[k][0], 4), "n": fb[k][1]} for k in fb}
            print(json.dumps({"nll_by_distance_from_end": by_dist}, indent=1))
            delta = {k: {"full_nll": round(full["summary"][k]["mean_nll"], 4),
                         "replay_nll": round(rep["summary"][k]["mean_nll"], 4),
                         "delta_nats": round(rep["summary"][k]["mean_nll"] - full["summary"][k]["mean_nll"], 4),
                         "full_top1": round(full["summary"][k]["top1_acc"], 4),
                         "replay_top1": round(rep["summary"][k]["top1_acc"], 4),
                         "n": full["summary"][k]["n"]}
                     for k in full["summary"]}
            print(json.dumps(delta, indent=1))
            res = {"summary": delta, "nll_by_distance_from_end": by_dist, "config": eng.config(),
                   "full": full, "replay": rep}
        else:
            res = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq))
            print(json.dumps(res["summary"], indent=1))
        if a.tf_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.tf_out)), exist_ok=True)
            json.dump(res, open(a.tf_out, "w"), indent=1)
            print("wrote", a.tf_out)
        raise SystemExit(0)
    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    prompt = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="thinking" if a.thinking else "chat")
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
    def run(tag, spec, temperature, top_p):
        eng.spec = spec
        print(f"\n===== {tag}: spec={spec} temperature={temperature} top_p={top_p} =====", flush=True)
        toks, text = [], []
        for burst in eng.generate(ids, max_tokens=a.max_tokens, temperature=temperature, top_p=top_p,
                                  ignore_eos=a.ignore_eos):
            toks += list(burst)
            piece = eng.tokenizer.decode(burst)
            text.append(piece)
            print(piece, end="", flush=True)
        print()
        st = eng.stats()
        print(json.dumps(st, indent=1), flush=True)
        return {"tag": tag, "spec": spec, "temperature": temperature, "top_p": top_p,
                "tokens": toks, "text": "".join(text), "stats": st}

    if a.spec_ab:
        # One process, one load: `spec` is just a flag on the engine, so the two runs share the
        # arena and the LRU state. Greedy speculative decoding must reproduce greedy autoregressive
        # decoding token for token (the target verifies every draft), so the first divergence index
        # is the whole test.
        runs = [run("greedy-nospec", False, 0.0, 1.0), run("greedy-spec", True, 0.0, 1.0)]
        A, B = runs[0]["tokens"], runs[1]["tokens"]
        first = next((i for i in range(min(len(A), len(B))) if A[i] != B[i]), None)
        n_eq = len(A) if first is None else first
        print(f"\n== greedy spec vs non-spec: {n_eq}/{min(len(A), len(B))} identical leading tokens; "
              f"first divergence at {first}")
        if first is not None:
            print("  nospec:", repr(eng.tokenizer.decode(A[max(0, first - 8):first + 8])))
            print("  spec  :", repr(eng.tokenizer.decode(B[max(0, first - 8):first + 8])))
        if a.temperature > 0:
            runs.append(run(f"sampled-spec-t{a.temperature}", True, a.temperature, a.top_p))
        if a.ab_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.ab_out)), exist_ok=True)
            json.dump({"prompt": a.prompt, "max_tokens": a.max_tokens, "config": eng.config(),
                       "identical_leading_tokens": n_eq, "first_divergence": first, "runs": runs},
                      open(a.ab_out, "w"), indent=1)
            print("wrote", a.ab_out)
    else:
        run("run", not a.no_spec, a.temperature, a.top_p)
