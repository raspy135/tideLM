"""Move fixed-size arena slots across layers without changing the total budget."""
import numpy as np


def layer_weights(raw, layers):
    values = tuple(float(v) for v in raw.split(',')) if raw.strip() else (1.,) * layers
    if len(values) != layers or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError('DSV41_PRUNE_LAYER_WEIGHTS needs one positive finite value per layer')
    return values


def global_plan(keeps, scores, floor, max_loads=None, protected=None, fallback=None, min_gain=0.):
    """Highest-value absentee vs lowest-value eligible resident, across layers.

    The floor preserves the router's top-k minimum, not a per-layer quota.
    A same-layer replacement is possible at the floor; a donor to another layer
    must have spare residents. Every pair preserves the total arena occupancy.
    Scores must already share units across layers. Stable IDs break ties.
    """
    if type(floor) is not int or floor < 1:
        raise ValueError('invalid resident floor')
    if max_loads is not None and (type(max_loads) is not int or max_loads < 0):
        raise ValueError('invalid resident load budget')
    protected = protected or {}
    counts, incoming, donors = {}, [], []
    for L in sorted(keeps):
        keep = np.asarray(keeps[L], dtype=bool)
        score = np.asarray(scores[L], dtype=np.float64)
        tie = np.asarray(fallback[L] if fallback is not None else score, dtype=np.float64)
        if (keep.ndim != 1 or score.shape != keep.shape or tie.shape != keep.shape
                or not np.isfinite(score).all() or (score < 0).any() or not np.isfinite(tie).all()):
            raise ValueError('invalid global residency scores')
        counts[L] = int(keep.sum())
        if not floor <= counts[L] <= len(keep):
            raise ValueError('resident count below the streaming capacity floor')
        lock = set(protected.get(L, ()))
        for e in np.flatnonzero(keep):
            if int(e) not in lock:
                donors.append((float(score[e]), float(tie[e]), L, int(e)))
        for e in np.flatnonzero(~keep & (score > 0)):
            incoming.append((float(score[e]), float(tie[e]), L, int(e)))
    donors.sort()
    incoming.sort(key=lambda v: (-v[0], -v[1], v[2], v[3]))
    changes = []
    for value, _, Li, ei in incoming:
        if max_loads is not None and len(changes) >= max_loads:
            break
        victim = next((i for i, (_, _, Lo, _) in enumerate(donors)
                       if Lo == Li or counts[Lo] > floor), None)
        if victim is None:
            continue
        old, _, Lo, eo = donors[victim]
        gain = value - old
        if gain <= min_gain:
            continue
        donors.pop(victim)
        changes.append((Lo, eo, Li, ei, gain))
        counts[Lo] -= 1
        counts[Li] += 1
    return changes, counts


def initial_selection(scores, budget, floor):
    """Mandatory top-k coverage, then one global ranking for every remaining slot."""
    if not floor*len(scores) <= budget <= sum(len(v) for v in scores.values()):
        raise ValueError('global resident budget cannot satisfy routing minimum')
    chosen, extras = {}, []
    for L in sorted(scores):
        order = sorted(range(len(scores[L])), key=lambda e:(-float(scores[L][e]),e))
        chosen[L] = order[:floor]
        extras.extend((float(scores[L][e]),L,e) for e in order[floor:])
    extras.sort(key=lambda v:(-v[0],v[1],v[2]))
    for _,L,e in extras[:budget-floor*len(scores)]:
        chosen[L].append(e)
    return chosen


def stream_spans(indices, keep, capacity):
    """Contiguous token batches: all their cold experts must fit simultaneously."""
    rows = np.asarray(indices)
    if rows.ndim != 2 or rows.shape[1] > capacity:
        raise ValueError('transient ring cannot fit one routing row')
    spans, start, cold = [], 0, set()
    for i,row in enumerate(rows):
        missing = set(int(e) for e in row if not keep[e])
        if len(cold | missing) > capacity:
            spans.append((start,i)); start=i; cold=set()
        cold.update(missing)
    if len(rows):
        spans.append((start,len(rows)))
    return spans


def streaming_moe(model, y, indices, weights, layer, store, arena, *, prefill=False):
    """One shared batch plan, all original top-k contributions, no resident eviction.

    The rank-0 broadcast happens even for a single batch. Each kernel handles
    complete token rows, retaining their k-sum order. Splitting the expert sum
    instead would introduce extra rounding points. TP kernel collectives see
    identical batch sizes on both ranks, regardless of local I/O/cache state.
    """
    import torch
    packet = None
    if store.ep.rank == 0:
        try:
            packet = dict(spans=stream_spans(indices.cpu().numpy(), model.prune_mask[layer].cpu().numpy(),
                                           store.transient_slots))
        except Exception as exc:
            packet = dict(error=type(exc).__name__)
    packet = store.ep.broadcast_obj(packet)
    if 'error' in packet:
        raise RuntimeError('streaming sector batch planner failed: '+packet['error'])
    model.user_prompt.stream_batches[layer] += len(packet['spans'])
    parts = []
    side = getattr(model, 'dream_fp8', None)
    tier = side.get(layer) if side else None
    for a,b in packet['spans']:
        slots = store.resolve(layer, indices[a:b], True)
        parts.append(model.moe_fn(y[a:b], slots, weights[a:b], arena, model.args.swiglu_limit,
                                  out_dtype=torch.float32, slots_repeat=True, null_slot=store.null_slot,
                                  prefill=prefill, dream=(tier, indices[a:b]) if tier is not None else None))
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def prompt_plan(keeps, mass, fallback, floor, weights, max_loads, pinned=None):
    # Per-layer normalization makes full encoder and bounded decoder replay
    # comparable. The layer weights are a soft priority, never a size limit.
    scores = {L: np.asarray(mass[L]) / max(float(np.asarray(mass[L]).sum()), 1e-20) * weights[L]
              for L in keeps}
    # pinned: experts no plan may evict (a dreaming overlay's, engine/dream_overlay.py::pins)
    swaps, counts = global_plan(keeps, scores, floor, max_loads, protected=pinned, fallback=fallback)
    post = {L: np.asarray(keep, dtype=bool).copy() for L, keep in keeps.items()}
    for Lo, eo, Li, ei, _ in swaps:
        post[Lo][eo] = False
        post[Li][ei] = True
    # Protect only experts that are actually resident after this bounded plan.
    protected = {L: np.flatnonzero(post[L] & (mass[L] > 0)).tolist() for L in keeps}
    missing = sum(int((~post[L] & (mass[L] > 0)).sum()) for L in keeps)
    return dict(swaps=swaps, protected=protected, overflow=0, deferred=missing,
                resident_counts=counts, cross_layer=sum(Lo != Li for Lo, eo, Li, ei, _ in swaps))
