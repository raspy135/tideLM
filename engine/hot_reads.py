"""DSV41_HOT_READS: read the missing expert a token wants most, before that token's MoE runs.

A pruned (resident-only) router replaces a nonresident pick with the next resident expert. On the
GLM-5.3 engine this repo's sibling measured what that costs per pick: the single heaviest missing
pick of a layer carries most of it -- reading just that one synchronously (when it holds >= 0.2 of
the row's routed weight) cut KL to the exact model 0.369 -> 0.135, while total missed mass moved
0.6 points. A second read a layer bought nothing; a first-come per-step cap starved late layers,
an even pacing over the layers did not.

Here: each MoE layer's decode graph splits at its router (fastdecode: graph A = attention + router,
which also leaves the pre-mask logits and this layer's candidates -- the missing unpruned picks by
weight share -- in a pinned host row; graph B = re-route + MoE). Between them the host takes at
most `per_layer` candidates at or above `min_share`, within a per-step budget paced over the layers
(`cap * ((i + 1) / n) ** pace`, plus `lead`), resolves them into the transient ring
(ExpertStore.resolve(prefill=True): no promotion, no eviction of residents; a repeat is a ring hit,
no NVMe read), marks them routable in the layer's keep mask and slot table, and graph B routes
again with them allowed. The marks are undone after B. The demand DB still records the miss
(PRUNE_MISS reads the router before any of this), so adaptation can make a recurring one resident.

DSV41_HOT_ASYNC=1 does not wait inside the step. The graphs stay whole; each hot layer's router
leaves its candidates on the device, the step copies them to the host at its end, and the next step
plans from them -- the heaviest `cap` across all hot layers -- reads them on the io threads into a
ring of its own (DSV41_HOT_RING slots taken out of the transient ring) while its graphs run, and
waits for those reads only after queueing them. A read is routable from the step after that, and
stays routable until the ring wraps over its slot. The marks are applied before a step's graphs
and undone after them, so prefill and adaptation never see one. Measured against the synchronous
read on 1017 (teacher-forced, PrefillEmulation): a little less on code, more on prose, because a
read expert serves every later step it stays in the ring instead of one.

Both ranks run the same router on the same rows and the same plan on the same host row, so they
read the same experts and route identically; the settings are in the boot guard.
"""
from __future__ import annotations

import math
import os

import torch

VERSION = 1


class HotReads:
    def __init__(self, n_layers: int, n_experts: int, k: int, env=os.environ):
        self.min_share = float(env.get("DSV41_HOT_MIN", "0.2"))
        self.cap = int(env.get("DSV41_HOT_MAX", "16"))           # reads a verify step
        self.per_layer = int(env.get("DSV41_HOT_PER_LAYER", "1"))
        self.pace = float(env.get("DSV41_HOT_PACE", "1"))
        self.lead = int(env.get("DSV41_HOT_PACE_LEAD", "0"))
        # DSV41_HOT_ASYNC=1: read after the step instead of inside it (HotRing below)
        self.asynchronous = env.get("DSV41_HOT_ASYNC", "0") == "1"
        self.ring = int(env.get("DSV41_HOT_RING", "32"))
        if self.asynchronous and self.ring < self.cap:
            raise ValueError("DSV41_HOT_RING must hold at least one step's reads (DSV41_HOT_MAX)")
        if not (0.0 < self.min_share <= 1.0 and self.cap >= 1 and 1 <= self.per_layer <= k and self.pace > 0):
            raise ValueError("DSV41_HOT_MIN in (0, 1], DSV41_HOT_MAX >= 1, DSV41_HOT_PER_LAYER in 1..k, PACE > 0")
        self.n_layers, self.n_experts, self.k = n_layers, n_experts, k
        self.width = 2 * k            # candidates kept a layer (most rows want at most one)
        self.layers: list[int] = []   # the hot layers in step order (set by FastDecoder)
        # share: the candidate shares the reads covered (a row's weight share of its unpruned picks,
        # summed over the layer's listed rows that wanted a read expert); want: the same for every
        # listed candidate at or above min_share -- share / want is how much of the heavy missed
        # weight the budget reached.
        self.stats = {"steps": 0, "reads": 0, "layers": 0, "capped": 0, "loaded": 0, "s": 0.0, "wait_s": 0.0, "share": 0.0, "want": 0.0}
        self._reads = 0
        self.unread: list[int] = []   # hot layers this step skipped with the budget spent

    @staticmethod
    def enabled(env=os.environ) -> bool:
        return env.get("DSV41_HOT_READS", "0") == "1"

    def boot_fields(self):
        return {"hot_reads_version": VERSION, "hot_async": self.asynchronous, "hot_ring": self.ring,
                "hot_min": self.min_share, "hot_max": self.cap,
                "hot_per_layer": self.per_layer, "hot_pace": self.pace, "hot_lead": self.lead}

    def candidates(self, scores, logits, keep, out):
        """Graph-capturable. scores/logits [T, E] fp32 (pre-mask), keep bool [E] -> out fp32
        [2 * width]: the layer's highest weight shares of missing unpruned picks, then their ids."""
        k, w = self.k, self.width
        ids = logits.topk(k, dim=-1)[1]
        s = scores.gather(1, ids)
        share = (s / (s.sum(dim=-1, keepdim=True) + 1e-20)) * (~keep[ids]).to(s.dtype)
        flat = share.reshape(-1)
        n = min(w, flat.numel())
        v, i = flat.topk(n)
        out[:n] = v
        out[w:w + n] = ids.reshape(-1)[i].to(out.dtype)
        if n < w:
            out[n:w] = 0.0

    def begin_step(self, unread_rows=()):
        """unread_rows: the previous step's host rows for the layers it skipped (room 0)."""
        for row in unread_rows:
            self.stats["want"] += sum(v for v in row[:self.width] if v >= self.min_share)
            if row[0] >= self.min_share:
                self.stats["capped"] += 1
        self.unread = []
        self._reads = 0
        self.stats["steps"] += 1

    def room(self, L: int) -> int:
        """Reads layer L may still make this step (the paced budget)."""
        i = self.layers.index(L)
        return min(self.cap, math.ceil(self.cap * ((i + 1) / len(self.layers)) ** self.pace) + self.lead) - self._reads

    def plan(self, L: int, row: list[float]) -> list[int]:
        """Experts to read before layer L's MoE, from its host row (shares, then ids)."""
        w = self.width
        self.stats["want"] += sum(v for v in row[:w] if v >= self.min_share)
        room = self.room(L)
        if room <= 0:
            if row[0] >= self.min_share:
                self.stats["capped"] += 1
            return []
        out: list[int] = []
        for v, e in zip(row[:w], row[w:]):
            if v < self.min_share or len(out) >= min(self.per_layer, room):
                break
            if int(e) not in out:
                out.append(int(e))
        self._reads += len(out)
        if out:
            self.stats["reads"] += len(out)
            self.stats["layers"] += 1
            self.stats["share"] += sum(v for v, e in zip(row[:w], row[w:]) if int(e) in out and v >= self.min_share)
        return out


    def plan_async(self, rows, taken) -> list[tuple[int, int]]:
        """DSV41_HOT_ASYNC: the step's reads, from the last step's host rows of every hot layer
        ([n_layers][2 * width]): the heaviest candidates at or above min_share across all hot
        layers, at most `per_layer` a layer and `cap` in all, skipping (layer, expert) keys the
        ring already holds or is reading (`taken`). Ties break on (layer, expert), so both ranks
        plan the same list from the same rows."""
        w = self.width
        cands = []
        for L in self.layers:
            r = rows[L]
            for v, e in zip(r[:w], r[w:]):
                if v < self.min_share:
                    break                     # a row is sorted by share
                cands.append((-v, L, int(e)))
        cands.sort()
        self.stats["steps"] += 1
        out: list[tuple[int, int]] = []
        per: dict[int, int] = {}
        seen = set()
        for nv, L, e in cands:
            key = (L, e)
            if key in seen or key in taken:
                continue
            seen.add(key)
            self.stats["want"] -= nv
            if len(out) >= self.cap or per.get(L, 0) >= self.per_layer:
                self.stats["capped"] += 1
                continue
            per[L] = per.get(L, 0) + 1
            out.append(key)
            self.stats["share"] -= nv
        self.stats["reads"] += len(out)
        self.stats["layers"] += len(per)
        return out


class HotRing:
    """DSV41_HOT_ASYNC's own slots (taken out of the store's transient ring, so nothing else
    writes them): first in, first out. A key is `pending` from the step that reads it and `live`
    -- routable -- from the step after, until a later read takes its slot. Both ranks take slots
    in the same order from the same plans."""

    def __init__(self, slots):
        self.slots = list(slots)
        self.owner: list[tuple | None] = [None] * len(self.slots)
        self.pos = 0
        self.live: dict[tuple, int] = {}       # (layer, expert) -> slot, routable this step
        self.pending: dict[tuple, int] = {}    # read this step, routable from the next

    def taken(self):
        return self.live.keys() | self.pending.keys()

    def take(self, key) -> int:
        """A slot for `key`: the oldest one; its previous key stops being routable now."""
        i = self.pos % len(self.slots)
        self.pos += 1
        old = self.owner[i]
        if old is not None:
            self.live.pop(old, None)
            self.pending.pop(old, None)
        self.owner[i] = key
        self.pending[key] = self.slots[i]
        return self.slots[i]

    def promote(self):
        """The step's reads have landed: routable from the next step."""
        self.live.update(self.pending)
        self.pending.clear()


class PrefillEmulation:
    """Eval-only (tools/bench_miss_policy_tp.py `hot:` and `exact` arms): HotReads' reads, scored in
    a teacher-forced prefill. The sequence is cut into blocks of `block` rows -- a decode verify
    step's width -- and each block runs decode's plan layer by layer: its own paced budget, its
    heaviest missing picks at or above min_share, and a planned expert routable for every row of the
    block (in decode one read serves the whole verify block). Model.moe calls `widen` once per
    pruned layer, in layer order, and loads the returned experts into the transient ring.

    `lag` / `hold` model a read that is not waited for: block b's reads become routable for blocks
    b + lag .. b + lag + hold - 1 (the synchronous read is lag 0, hold 1), and a block's candidates
    skip what an earlier block's read already made routable for it.

    With min_share tiny, per_layer = k, an unbounded cap and block 1 every missing pick becomes
    routable: the exact (unpruned) routing, as the reference the other arms are measured against."""

    def __init__(self, hot: HotReads, layers, block: int, lag: int = 0, hold: int = 1):
        if block < 1 or lag < 0 or hold < 1:
            raise ValueError("block >= 1, lag >= 0, hold >= 1")
        self.hot, self.block, self.lag, self.hold = hot, block, lag, hold
        hot.layers = sorted(layers)
        self._reads: list[int] = []
        self.stats = {"tokens": 0, "reads": 0, "rows_widened": 0}

    def widen(self, L, unmasked, scores, keep, k):
        """unmasked logits / scores [T, E] fp32, keep bool [E] -> (row keep [T, E] or None, experts
        to load)."""
        h, B = self.hot, self.block
        T = scores.shape[0]
        nb = -(-T // B)
        i = h.layers.index(L)
        if i == 0 or len(self._reads) != nb:
            self._reads = [0] * nb
            self.stats["tokens"] += T if i == 0 else 0
        want = unmasked.topk(k, dim=-1)[1]
        s = scores.gather(1, want)
        share = ((s / (s.sum(dim=-1, keepdim=True) + 1e-20)) * (~keep[want]).to(s.dtype)).tolist()
        want = want.tolist()
        budget = min(h.cap, math.ceil(h.cap * ((i + 1) / len(h.layers)) ** h.pace) + h.lead)
        routable: list[set] = [set() for _ in range(nb)]   # per block: reads made routable for it
        picked = set()
        for b in range(nb):
            room = budget - self._reads[b]
            if room <= 0:
                continue
            r0, r1 = b * B, min(T, b * B + B)
            cand = sorted(((v, e) for r in range(r0, r1) for v, e in zip(share[r], want[r])
                           if v >= h.min_share and e not in routable[b]), reverse=True)
            out: list[int] = []
            for _, e in cand:
                if len(out) >= min(h.per_layer, room):
                    break
                if e not in out:
                    out.append(e)
            self._reads[b] += len(out)
            self.stats["reads"] += len(out)
            for c in range(b + self.lag, min(nb, b + self.lag + self.hold)):
                routable[c].update(out)
            picked.update(out)
        rows, cols = [], []
        for b, es in enumerate(routable):
            r0, r1 = b * B, min(T, b * B + B)
            self.stats["rows_widened"] += (r1 - r0) if es else 0
            for e in es:
                rows.extend(range(r0, r1))
                cols.extend([e] * (r1 - r0))
        if not rows:
            return None, []
        row_keep = keep[None, :].expand(T, -1).clone()
        dev = keep.device
        row_keep[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = True
        return row_keep, sorted({e for es in routable for e in es})
