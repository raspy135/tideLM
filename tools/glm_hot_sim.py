"""Replay a GLM hot-read trace (TIDELM_GLM_HOT_TRACE): async hot-read policies on one run's routing.

The trace (rust/crates/tidelm-glm/src/adapt.rs, hot_trace_open / hot_trace_pass, rank 0) holds the
residency at boot and, for every decode pass, every MoE layer's exact top-8 (share, expert) -- a
resident pick's share negated -- and the async reads the engine planned (with their victims). This
tool replays policies on that routing with a simulated residency: a pass's misses are counted
against what the policy has made resident so far, and a pass's reads are routable from the next
pass (async, lag 1) or from this one (sync). What it cannot model: routing feedback (a different
substitute changes the next layers' hidden states, so their picks), and the engine's victim queue
(here: the unprotected resident with the least recent demand). Use it to rank policies and pick
settings; the engine's own --bench-nll decides.

Total missed weight is not the quality: on heldout_quick the tally alone (0.9,0.7,1) missed
10.37% of the weight against the engine rule's 12.04%, at fewer reads, and its NLL was +0.024 +-
0.005 worse - it let through 2.1x the single misses >= 0.2 (2.7x >= 0.4). Read the heavy-miss
columns (misses >= 0.3 a row; each layer-row's heaviest miss cubed, summed) first.

On the A run's trace (heldout_quick, class-constrained victims), heavy misses >= 0.3 a row:
  rule 0.086 (victims "recent") / 0.077 ("slow"); hybrid sum 0.9,1.0,1: 0.094 / 0.087 at 0.71 /
  0.66 reads a token against 0.59 / 0.54 - more total weight caught, more heavy misses (churn);
  hybrid max (0.8-0.95, min 0.2-0.3): 0.085-0.086 / 0.077 at +0.002 reads a token - the max
  tally's candidates are the rule's own pick, already read (no engine run).

Victims (the rule's reads, --warm 67 --skip 100: the boot warm-up plans no reads), heavy misses
>= 0.3 a row: "recent" 0.082 (the engine's own A run: 0.080), "slow" 0.073 (its V run: 0.070),
"belady" (hindsight) 0.071 - slow is near the bound; "boot-last" 0.351 at 6 reads a pass - the
read-installed experts evict each other (rejected). An oracle of the next pass's misses at 2
reads a pass with hindsight victims: miss 9.71% by score (rule 11.47%) but 0.114 heavy - prediction
of the summed weight is not the quality either.

    python3 tools/glm_hot_sim.py TRACE [--skip N] [--sweep]
"""

import argparse
import itertools
import struct

import numpy as np


def load_classes(path, layers, e):
    """bits.json ({"layer,expert": [g, u, d]}, from a pack's header) -> class id [MoE layer][expert]:
    a hot read takes a victim of its own widths (adapt.rs take_victim)."""
    import json
    bits = json.load(open(path))
    names = sorted({tuple(v) for v in bits.values()})
    cls = np.zeros((len(layers), e), dtype=np.int64)
    for i, l in enumerate(layers):
        for x in range(e):
            cls[i, x] = names.index(tuple(bits[f"{l},{x}"]))
    return cls


def load(path):
    b = open(path, "rb").read()
    if b[:4] not in (b"GHT1", b"GHT2"):
        raise SystemExit(f"{path}: not a hot trace")
    has_ids = b[:4] == b"GHT2"  # each pass's first position and row token ids after its row count
    n, e, k = struct.unpack_from("<3I", b, 4)
    off = 16
    layers = np.frombuffer(b, "<u4", n, off).astype(int)
    off += 4 * n
    res0 = np.frombuffer(b, np.uint8, n * e, off).reshape(n, e).astype(bool)
    off += n * e
    idx = {int(l): i for i, l in enumerate(layers)}
    passes = []
    ids = []
    pos = []
    while off < len(b):
        (t,) = struct.unpack_from("<I", b, off)
        off += 4
        if has_ids:
            pos.append(struct.unpack_from("<I", b, off)[0])
            ids.append(np.frombuffer(b, "<u4", t, off + 4).astype(np.int64))
            off += 4 + 4 * t
        else:
            pos.append(None)
            ids.append(None)
        tab = np.frombuffer(b, "<f4", n * t * k * 2, off).reshape(n, t * k, 2)
        off += n * t * k * 8
        (r,) = struct.unpack_from("<I", b, off)
        off += 4
        reads = np.frombuffer(b, "<u2", 4 * r, off).reshape(r, 4).astype(int)
        off += 8 * r
        reads = [(idx[l], x, idx[vl], ve) for l, x, vl, ve in reads]
        passes.append((t, np.abs(tab[:, :, 0]).astype(np.float64), tab[:, :, 1].astype(int), tab[:, :, 0] > 0, reads))
    load.layers = layers
    load.ids = ids  # [pass] its rows' token ids (None in a GHT1 trace)
    load.pos = pos  # [pass] its first row's position in its sequence (None in GHT1)
    return n, e, k, res0, passes


class Baseline:
    """The engine's rule: per layer the heaviest missed pick of the pass with share >= min."""
    name = "baseline"

    def __init__(self, min_share=0.2):
        self.min = min_share
        self.name = f"baseline min {min_share}"

    def plan(self, n, e, d, ex, missed, ok, nxt):
        out = []
        for i in range(n):
            m = missed[i] & (d[i] >= self.min)
            if not m.any():
                continue
            c = [(d[i][j], ex[i][j]) for j in np.nonzero(m)[0] if ok(i, ex[i][j])]
            if c:
                v, x = max(c, key=lambda c: (c[0], -c[1]))
                out.append((v, i, x))
        return out


class Predict:
    """GLM_HOT_PREDICT: missed share summed over the pass's rows, decayed over passes. kind "max":
    the decayed max of an expert's missed shares instead - a predictor of the heavy misses, where
    the sum ranks the recurring light picks first (tools/glm_hot_eval.py: the sum tally's reads in
    H served 0.008 picks >= 0.3 each in 16 passes, the rule's 0.15)."""

    def __init__(self, n, e, decay, min_acc, per_layer, kind="sum"):
        self.decay, self.min, self.per, self.kind = decay, min_acc, per_layer, kind
        self.acc = np.zeros((n, e))
        self.name = f"predict {decay},{min_acc},{per_layer}" + ("" if kind == "sum" else f" {kind}")

    def plan(self, n, e, d, ex, missed, ok, nxt):
        self.acc *= self.decay
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        if self.kind == "max":
            np.maximum.at(self.acc, (li[missed], ex[missed]), d[missed])
        else:
            np.add.at(self.acc, (li[missed], ex[missed]), d[missed])
        out = []
        for i in np.nonzero((self.acc >= self.min).any(axis=1))[0]:
            xs = np.nonzero(self.acc[i] >= self.min)[0]
            c = sorted(((self.acc[i, x], x) for x in xs if ok(i, x)), key=lambda c: (-c[0], c[1]))[: self.per]
            out += [(v, i, x) for v, x in c]
        return out

    def taken(self, i, x):
        self.acc[i, x] = 0.0


class Hybrid(Predict):
    """The engine's rule first (each layer's heaviest missed pick >= 0.2), the tally's recurring
    misses after it (up to per_layer more a layer): GLM_HOT_PREDICT=DECAY,MIN,PER_LAYER,1."""

    def __init__(self, n, e, decay, min_acc, per_layer, min_share=0.2, kind="sum"):
        super().__init__(n, e, decay, min_acc, per_layer, kind)
        self.base = Baseline(min_share)
        self.name = f"hybrid {decay},{min_acc},{per_layer}" + ("" if kind == "sum" else f" {kind}")

    def plan(self, n, e, d, ex, missed, ok, nxt):
        first = self.base.plan(n, e, d, ex, missed, ok, nxt)
        got = {(i, x) for _, i, x in first}
        rest = [(v, i, x) for v, i, x in super().plan(n, e, d, ex, missed, ok, nxt) if (i, x) not in got]
        # the engine's tiers: every reactive candidate before any predicted one
        return [(10.0 + v, i, x) for v, i, x in first] + [(min(v, 9.0), i, x) for v, i, x in rest]


class Oracle:
    """Knows the next pass: its missed experts by summed share, `cap` of them."""

    def __init__(self, cap):
        self.cap = cap
        self.name = f"oracle {cap} a pass"

    def plan(self, n, e, d, ex, missed, ok, nxt):
        if nxt is None:
            return []
        nd, nex, nres = nxt
        s = {}
        for i in range(n):
            for j in range(nd.shape[1]):
                x = nex[i][j]
                if not nres(i, x):
                    s[(i, x)] = s.get((i, x), 0.0) + nd[i][j]
        c = sorted(((v, i, x) for (i, x), v in s.items() if ok(i, x)), key=lambda c: (-c[0], c[1], c[2]))
        return c[: self.cap]


def next_use(n, e, passes):
    """[pass][layer][expert]: the first later pass that picks the expert (len(passes) if none)."""
    out = np.empty((len(passes), n, e), dtype=np.int16 if len(passes) < 32767 else np.int32)
    cur = np.full((n, e), len(passes), dtype=out.dtype)
    for p in range(len(passes) - 1, -1, -1):
        out[p] = cur
        t, d, ex = passes[p][:3]
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        cur[li, ex] = p
    return out


def run(n, e, k, res0, passes, pol, cap=16, sync=False, victims_from_trace=False, victims="recent", cls=None,
        warm=0, skip=0, nxt_use=None):
    """victims: "recent" (least demand decayed 0.837 a pass), "lru" (longest since last demanded),
    "slow" (least demand decayed 0.99 a pass), "boot-last" (slow, but a boot resident only when no
    read-installed one is free), "belady" (hindsight: the one next picked furthest ahead - needs
    nxt_use = next_use(...)); ties to the lowest (layer, expert). warm: no reads planned before
    that pass (the engine's boot warm-up plans none); skip: counted from that pass."""
    R = res0.copy()
    boot = res0.copy()
    last = np.full((n, e), -1, dtype=np.int64)
    slow = np.zeros((n, e))
    protect = np.zeros((n, e), dtype=np.int64)
    recent = np.zeros((n, e))  # bg_score's stand-in: demand decayed 0.7 every 2 passes
    pending = []
    tot = miss = 0.0
    picks = missed_picks = reads = tokens = heavy = 0
    cube = 0.0
    for p, (t, d, ex, _, treads) in enumerate(passes):
        if p == skip:
            tot = miss = 0.0
            picks = missed_picks = reads = tokens = heavy = 0
            cube = 0.0
        for i, x in pending:
            R[i, x] = True
            protect[i, x] = p + 16
        pending = []
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        ok = lambda i, x: not R[i, x] and (i, x) not in pend_set
        pend_set = set()
        nxt = None
        if p + 1 < len(passes):
            nxt = (passes[p + 1][1], passes[p + 1][2], lambda i, x: R[i, x])
        if sync:
            # read this pass's candidates before its MoE runs: routable for it
            missed0 = ~R[li, ex]
            cands = sorted(pol.plan(n, e, d, ex, missed0, ok, nxt), key=lambda c: -c[0])[:cap]
            for _, i, x in cands:
                v = victim(R, score if not sync else recent, protect, p, None if cls is None else (cls, cls[i, x]))
                R[v] = False
                R[i, x] = True
                protect[i, x] = p + 16
                reads += 1
            cands = []
        missed = ~R[li, ex]
        tot += d.sum()
        miss += d[missed].sum()
        heavy += int((missed & (d >= 0.3)).sum())
        cube += float(((d * missed).reshape(n, t, k).max(axis=2) ** 3).sum())
        picks += d.size
        missed_picks += int(missed.sum())
        tokens += t
        recent *= 0.7 ** 0.5
        np.add.at(recent, (li, ex), d)
        last[li, ex] = p
        slow *= 0.99
        np.add.at(slow, (li, ex), d)
        if victims == "belady":
            score = -nxt_use[p].astype(np.float64)
        elif victims == "boot-last":
            score = slow + 1e6 * boot
        else:
            score = {"recent": recent, "lru": last.astype(np.float64), "slow": slow}[victims]
        if sync or p < warm:
            continue
        cands = sorted(pol.plan(n, e, d, ex, missed, ok, nxt), key=lambda c: -c[0])[:cap]
        if victims_from_trace:
            cands = [(1.0, i, x) for i, x, _, _ in treads]
        for j, (_, i, x) in enumerate(cands):
            if victims_from_trace:
                v = (treads[j][2], treads[j][3])
            else:
                v = victim(R, score if not sync else recent, protect, p, None if cls is None else (cls, cls[i, x]))
            R[v] = False
            boot[v] = False
            pending.append((i, x))
            pend_set.add((i, x))
            if hasattr(pol, "taken"):
                pol.taken(i, x)
            reads += 1
    return dict(policy=pol.name + (" (sync)" if sync else "") + f" victims {victims}", miss_score=100 * miss / tot,
                heavy=heavy / tokens, cube=cube / tokens,
                miss_count=100 * missed_picks / picks, reads_tok=reads / tokens, reads_pass=reads / len(passes))


def victim(R, recent, protect, p, cls=None):
    """The unprotected resident with the least recent demand (ties to the lowest index), of the
    read's width class when `cls` = (class map, class) is given."""
    m = R & (protect <= p)
    if cls is not None:
        m &= cls[0] == cls[1]
    s = np.where(m, recent, np.inf)
    j = int(np.argmin(s))
    return divmod(j, R.shape[1])


def show(r):
    print(f"{r['policy']:<24} miss by score {r['miss_score']:6.2f}%  by count {r['miss_count']:6.2f}%  "
          f">=0.3 {r.get('heavy', 0):.3f} a row  max^3 {r.get('cube', 0):.4f} a row  "
          f"reads {r['reads_tok']:.3f} a token ({r['reads_pass']:.2f} a pass)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=0, help="passes at the start not counted (warm-up)")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    n, e, k, res0, passes = load(a.trace)
    print(f"{len(passes)} passes, {sum(p[0] for p in passes)} rows, {n} MoE layers, {int(res0.sum())} residents at the start")
    # what the engine saw (its own residency, the trace's signs)
    tot = sum(p[1].sum() for p in passes)
    miss = sum(p[1][p[3]].sum() for p in passes)
    rd = sum(len(p[4]) for p in passes)
    print(f"{'recorded run':<32} miss by score {100 * miss / tot:6.2f}%  reads {rd / sum(p[0] for p in passes):.3f} a token")
    show(run(n, e, k, res0, passes, Baseline(), victims_from_trace=True) | {"policy": "replay of its reads"})
    show(run(n, e, k, res0, passes, Baseline()))
    show(run(n, e, k, res0, passes, Baseline(), sync=True))
    for c in (1, 2, 4, 16):
        show(run(n, e, k, res0, passes, Oracle(c)))
    if a.sweep:
        for m in (0.15, 0.3):
            show(run(n, e, k, res0, passes, Baseline(m)))
        for dec, mn, per in itertools.product((0.0, 0.5, 0.75), (0.3, 0.5, 0.8), (1, 2)):
            show(run(n, e, k, res0, passes, Predict(n, e, dec, mn, per)))


if __name__ == "__main__":
    main()
