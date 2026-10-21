"""Full routing for latest-user prefill rows; bounded retention until the next request."""
import numpy as np


def validate_ranges(ranges, length):
    out = []
    for pair in ranges or ():
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) is not int for v in pair)):
            raise ValueError('user_prompt_ranges must contain integer [start,end] pairs')
        a, b = pair
        if not 0 <= a < b <= length or (out and a < out[-1][1]):
            raise ValueError('invalid or overlapping user prompt ranges')
        out.append((a, b))
    return tuple(out)


def retention_plan(keep, mass, fallback):
    """User score mass has first priority. History orders only unused capacity.

    Protect all positively used experts that fit; if oversubscribed retain the
    highest score mass, with expert ID as deterministic tie break. Never enlarge
    a layer's quota or put these temporary priorities into the history database.
    """
    keep = np.asarray(keep, dtype=bool)
    mass, fallback = np.asarray(mass), np.asarray(fallback)
    if (mass.shape != keep.shape or fallback.shape != keep.shape or keep.ndim != 1
            or not np.isfinite(mass).all() or (mass < 0).any() or not np.isfinite(fallback).all()):
        raise ValueError('invalid latest-user retention scores')
    quota = int(keep.sum())
    used = sorted(np.flatnonzero(mass > 0), key=lambda e: (-float(mass[e]), int(e)))
    protected = used[:quota]
    incoming = [e for e in protected if not keep[e]]
    protected_set = set(protected)
    evictable = [e for e in np.flatnonzero(keep) if e not in protected_set]
    evictable.sort(key=lambda e: (float(fallback[e]), int(e)))
    swaps = [(int(a), int(b), float(mass[b])) for a, b in zip(evictable, incoming)]
    return swaps, [int(e) for e in protected], max(0, len(used) - quota)


def capped_retention_plan(keeps, mass, fallback, max_loads=None, pinned=None):
    """A request-wide promotion budget, with fair score units across layers.

    Normalize each layer's user mass so the shorter decoder replay does not
    automatically lose to encoder layers. Admit the swaps that preserve the
    most user mass. A deferred absentee is never protected as if resident.
    None is unlimited; zero means no remaining loads.
    """
    if max_loads is not None and (type(max_loads) is not int or max_loads < 0):
        raise ValueError('promotion budget must be a nonnegative integer or None')
    candidates, targets, overflow = [], {}, 0
    for L in sorted(keeps):
        changes, targets[L], excess = retention_plan(keeps[L], mass[L], fallback[L])
        overflow += excess
        total = max(float(np.asarray(mass[L]).sum()), 1e-20)
        for a, b, score in changes:
            if pinned and a in pinned.get(L, ()):
                continue   # a dreaming overlay's expert is never evicted (engine/dream_overlay.py::pins)
            gain = (float(mass[L][b])-float(mass[L][a])) / total
            candidates.append((L, a, b, score, gain))
    chosen = candidates
    if max_loads is not None:
        chosen = sorted(candidates, key=lambda x:(-x[4],x[0],x[2],x[1]))[:max_loads]
    protected = {L:set(e for e in ids if keeps[L][e]) for L,ids in targets.items()}
    for L,a,b,score,gain in chosen:
        protected[L].add(b)
    return dict(swaps=[(L,a,b,float(score)) for L,a,b,score,gain in chosen],
                protected={L:sorted(ids) for L,ids in protected.items()},
                overflow=overflow, deferred=len(candidates)-len(chosen))


class UserPrompt:
    def __init__(self, enabled, layers, experts, device, max_loads=0, dynamic=False):
        if type(max_loads) is not int or max_loads < 0:
            raise ValueError('DSV41_USER_PROMPT_MAX_LOADS must be nonnegative (0 is unlimited)')
        self.max_loads = max_loads
        self.dynamic = dynamic
        self.enabled, self.layers, self.experts = enabled, layers, experts
        self.device = device
        self.reset((), 0)

    def reset(self, ranges, length):
        self.ranges = validate_ranges(ranges, length) if self.enabled else ()
        self.protected = {}
        self.promoted = self.overflow = 0
        self.cross_layer = 0
        self.loads_used = self.resident_loads = self.deferred = self.discovery_loads = 0
        self.promotion_s = 0.0
        self.refresh_s = 0.0
        self.recording = True
        self.streaming = True
        self.unchanged_replay_max_delta = None
        self.mass = None
        self.rows = [0] * self.layers
        self.stream_batches = [0] * self.layers
        if self.ranges:
            import torch
            self.mass = torch.zeros((self.layers, self.experts), dtype=torch.float64, device=self.device)

    def row_mask(self, start, rows):
        if (not self.streaming or self.remaining_loads == 0 or not self.ranges
                or not any(a < start + rows and b > start for a, b in self.ranges)):
            return None
        import torch
        mask = torch.zeros(rows, dtype=torch.bool, device=self.device)
        for a, b in self.ranges:
            lo, hi = max(0, a - start), min(rows, b - start)
            if lo < hi:
                mask[lo:hi] = True
        return mask

    @property
    def remaining_loads(self):
        # This budget is exclusively for temporary streaming cold reads.
        # Resident selection/adaptation has its own existing swap policy.
        return max(0, self.max_loads-self.loads_used) if self.enabled and self.max_loads else None

    def consume(self, count):
        if self.remaining_loads is not None and count > self.remaining_loads:
            raise RuntimeError('streaming plan exceeds request cold-load budget')
        self.loads_used += count

    def streaming_keep(self, layer, logits, scores, rows, keep, k, store):
        """Bound cold reads before routing; both TP ranks use the same allowed set.

        Cached transients are free only when present on both ranks. Newly allowed
        experts are drawn from the original top-k, so each is used after masking.
        Keep all residents and fit the selected cold set in one transient batch;
        resolve() then cannot evict and re-read an admitted expert in this call.
        """
        import torch
        remaining = self.remaining_loads
        if remaining is None:
            return torch.ones_like(keep)
        if remaining == 0:
            return keep
        available = sorted(e for L, e in store.transient_map if L == layer)
        # Unconditional on both ranks, including empty caches and candidate sets.
        inventories = store.ep.gather_objects(available)
        packet = None
        if store.ep.rank == 0:
            try:
                ids = logits[rows].topk(k, dim=-1).indices
                weights = scores[rows].gather(1, ids)
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
                mass = torch.zeros_like(keep, dtype=torch.float64)
                mass.scatter_add_(0, ids.reshape(-1), weights.double().reshape(-1))
                values = mass.cpu().numpy()
                cold = np.flatnonzero((~keep).cpu().numpy() & (values > 0)).tolist()
                cold.sort(key=lambda e: (-float(values[e]), e))
                common = set(inventories[0]).intersection(*map(set, inventories[1:]))
                cached = [e for e in cold if e in common][:store.transient_slots]
                fresh = [e for e in cold if e not in common][
                    :min(remaining, store.transient_slots-len(cached))]
                packet = dict(experts=cached+fresh, loads=len(fresh))
            except Exception as exc:
                packet = dict(error=type(exc).__name__)
        packet = store.ep.broadcast_obj(packet)
        if 'error' in packet:
            raise RuntimeError('streaming cold-load planner failed: '+packet['error'])
        self.consume(packet['loads'])
        allowed = keep.clone()
        allowed[packet['experts']] = True
        return allowed

    def record(self, layer, indices, weights, mask):
        if not self.recording:
            return
        self.mass[layer].scatter_add_(0, indices[mask].reshape(-1), weights[mask].double().reshape(-1))
        self.rows[layer] += int(mask.sum().item())

    def report(self):
        return dict(enabled=self.enabled, selected_tokens=sum(b-a for a,b in self.ranges),
                    streamed_rows_per_layer=self.rows, promoted=self.promoted,
                    protected=sum(map(len, self.protected.values())), overflow=self.overflow,
                    promotion_s=round(self.promotion_s, 3),
                    scope='latest-user-text-prefill' if self.enabled else 'disabled',
                    decode='resident', priority='normalized-router-score-mass' if self.enabled else 'disabled', version=6,
                    allocation='global' if self.dynamic else 'fixed-layer', cross_layer_transfers=self.cross_layer,
                    stream_batches_per_layer=self.stream_batches,
                    stream_load_cap=self.max_loads, stream_loads_used=self.loads_used,
                    stream_loads_remaining=self.remaining_loads, deferred_promotions=self.deferred,
                    resident_loads_used=self.resident_loads,
                    resident_load_cap=None, resident_loads_remaining=None,
                    discovery_expert_loads=self.discovery_loads,
                    load_cap_scope='streaming-cold-loads-per-request',
                    first_logits_refreshed=self.refresh_s > 0, refresh_s=round(self.refresh_s,3),
                    unchanged_replay_max_delta=self.unchanged_replay_max_delta)
