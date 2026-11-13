"""How good each hot read was, from the engine's own trace (TIDELM_GLM_HOT_TRACE): the read's
expert scored against the routing that followed, so a predictor is judged as a predictor - not
by an NLL that also carries what its reads displaced and where the routing then went.

    python3 tools/glm_hot_eval.py TRACE [TRACE ...] [--skip N] [--recall [--decay D]]

The residency is rebuilt from the trace's boot residency and its reads (lag 1: a read planned
after pass p is routable from pass p+1, its victim gone from p+1) and checked against every
pick's resident/missed sign. A read owns its expert from p+1 until a later read evicts it; the
victim's picks while it is out are charged to the read that evicted it. Per read:

  hit@k     the expert was picked in passes p+1..p+k (while still resident)
  serve@k   the share it served then, summed over the passes' rows; heavy@k: its picks >= 0.3
  cost@k    the share the victim's picks lost in the same window (missed, so substituted)
  life      passes until evicted (to the trace's end if never)
  oracle@1  the reads' serve@1 over the best possible with as many reads: each pass's count of
            the next pass's heaviest would-be misses (summed share), no width class or victim cost

Reads are split by who chose them: "rule" when the expert is the engine rule's pick for that
layer and pass (the heaviest missed share >= 0.2, ties to the lower expert), "predicted" when
not. A run without GLM_HOT_PREDICT should show (nearly) all its reads as "rule".
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_hot_sim import load  # noqa: E402

KS = (1, 4, 16)


def rule_pick(d, ex, missed, i, min_share=0.2):
    m = missed[i] & (d[i] >= min_share)
    if not m.any():
        return None
    j = np.nonzero(m)[0]
    best = max(zip(d[i][j], ex[i][j]), key=lambda c: (c[0], -c[1]))
    return int(best[1])


def evaluate(path, skip):
    n, e, k, res0, passes = load(path)
    reads = []  # (pass, i, x, vi, ve, by rule)
    for p, (t, d, ex, missed, rr) in enumerate(passes):
        rp = {}
        for i, x, vi, ve in rr:
            if i not in rp:
                rp[i] = rule_pick(d, ex, missed, i)
            reads.append((p, i, x, vi, ve, rp[i] == x))
    nr = len(reads)
    rpass = np.array([r[0] for r in reads], dtype=np.int64)
    serve = {kk: np.zeros(nr) for kk in KS + (None,)}
    heavy = {kk: np.zeros(nr) for kk in KS + (None,)}
    cost = {kk: np.zeros(nr) for kk in KS + (None,)}
    hit = {kk: np.zeros(nr, dtype=bool) for kk in KS + (None,)}
    evicted_at = np.full(nr, len(passes), dtype=np.int64)
    R = res0.copy()
    owner = np.full((n, e), -1, dtype=np.int64)
    evictor = np.full((n, e), -1, dtype=np.int64)
    bad = 0
    ora_num = ora_den = 0.0
    by_pass = {}
    for ri, r in enumerate(reads):
        by_pass.setdefault(r[0], []).append(ri)
    for p, (t, d, ex, missed, _) in enumerate(passes):
        # last pass's reads: routable now, their victims gone
        for ri in by_pass.get(p - 1, []):
            _, i, x, vi, ve, _ = reads[ri]
            if owner[vi, ve] >= 0:
                evicted_at[owner[vi, ve]] = p
            R[vi, ve] = False
            owner[vi, ve] = -1
            evictor[vi, ve] = ri
            R[i, x] = True
            owner[i, x] = ri
            evictor[i, x] = -1
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        res = R[li, ex]
        bad += int((res == missed).sum())  # resident must be the trace's "not missed"
        # serve: resident picks of a read's expert
        o = np.where(res, owner[li, ex], -1)
        m = o >= 0
        if m.any():
            oi, s = o[m], d[m]
            age = p - rpass[oi]
            for kk in KS + (None,):
                w = np.ones_like(age, dtype=bool) if kk is None else age <= kk
                np.add.at(serve[kk], oi[w], s[w])
                np.add.at(heavy[kk], oi[w], (s[w] >= 0.3).astype(np.float64))
                hit[kk][oi[w]] = True
        # cost: missed picks of a victim, charged to its evictor
        v = np.where(~res, evictor[li, ex], -1)
        m = v >= 0
        if m.any():
            vi_, s = v[m], d[m]
            age = p - rpass[vi_]
            for kk in KS + (None,):
                w = np.ones_like(age, dtype=bool) if kk is None else age <= kk
                np.add.at(cost[kk], vi_[w], s[w])
        # oracle@1 for the reads planned after the previous pass: the c heaviest of this pass's
        # would-be misses under the residency before them (missed now or covered only by those
        # reads, and not their victims)
        prev = by_pass.get(p - 1, [])
        if prev and p - 1 >= skip:
            fresh = np.zeros((n, e), dtype=bool)
            gone = np.zeros((n, e), dtype=bool)
            for ri in prev:
                fresh[reads[ri][1], reads[ri][2]] = True
                gone[reads[ri][3], reads[ri][4]] = True
            would = (~res | fresh[li, ex]) & ~gone[li, ex]
            tot = np.zeros((n, e))
            np.add.at(tot, (li[would], ex[would]), d[would])
            best = np.sort(tot.ravel())[::-1][: len(prev)].sum()
            ora_den += best
    # the reads' own serve@1 in the oracle's passes
    sel = np.array([reads[ri][0] >= skip and reads[ri][0] + 1 < len(passes) for ri in range(nr)])
    ora_num = serve[1][sel].sum()
    keep = rpass >= skip
    rule = np.array([r[5] for r in reads], dtype=bool)
    name = os.path.basename(path)
    passes_n = len(passes) - skip
    print(f"{name}: {len(passes)} passes, {nr} reads ({keep.sum() / max(passes_n, 1):.2f} a pass after --skip {skip}); "
          f"residency replay vs trace signs: {bad} disagreements; oracle@1 {100 * ora_num / max(ora_den, 1e-9):.1f}%")
    life = evicted_at - rpass - 1
    print(f"  {'reads':<10} {'n':>6} {'hit@1':>6} {'hit@4':>6} {'hit@16':>6} {'ever':>6} "
          f"{'serve@1':>8} {'serve@16':>8} {'serve':>7} {'heavy@16':>8} {'heavy':>6} {'cost@16':>8} {'cost':>7} {'net@16':>7} {'life':>6}")
    for label, m in (("rule", keep & rule), ("predicted", keep & ~rule), ("all", keep)):
        c = int(m.sum())
        if c == 0:
            continue
        print(f"  {label:<10} {c:>6} " + " ".join(f"{100 * hit[kk][m].mean():>5.1f}%" for kk in KS + (None,)) +
              f" {serve[1][m].mean():>8.3f} {serve[16][m].mean():>8.3f} {serve[None][m].mean():>7.2f}"
              f" {heavy[16][m].mean():>8.3f} {heavy[None][m].sum():>6.0f} {cost[16][m].mean():>8.3f} {cost[None][m].mean():>7.2f}"
              f" {(serve[16][m] - cost[16][m]).mean():>+7.3f} {np.median(life[m]):>6.0f}")


def recall(path, skip, decay=0.9):
    """--recall: could any history predictor have flagged a pass's misses one pass ahead? For each
    missed pick (all, and the heavy ones >= 0.3, share-weighted): was the expert picked in the last
    1/4/16/64 passes or ever, picked heavily (>= 0.3) in the last 16/64, and where did tallies of
    the passes before it (decayed `decay` a pass, over missed picks only, the way GLM_HOT_PREDICT
    folds) rank it among its layer's non-residents: sum of shares (the engine's), of shares
    squared, the decayed max share."""
    n, e, k, res0, passes = load(path)
    R = res0.copy()
    last = np.full((n, e), -10**9, dtype=np.int64)
    lastheavy = np.full((n, e), -10**9, dtype=np.int64)
    lastres = np.where(res0, 0, -10**9).astype(np.int64)  # last pass it was resident
    evicted_by_read = np.zeros((n, e), dtype=bool)  # out because a hot read took its slot
    tallies = {"sum": np.zeros((n, e)), "sum sq": np.zeros((n, e)), "max": np.zeros((n, e))}
    feats = ["seen@1", "seen@4", "seen@16", "seen@64", "seen ever", "heavy@16", "heavy@64",
             "resident@16", "resident@64", "resident@boot", "evicted by a read"] + \
            [f"{t} top{r}" for t in tallies for r in (1, 2, 4)]
    acc = {c: dict.fromkeys(feats, 0.0) | {"n": 0.0, "w": 0.0} for c in ("all", "heavy")}
    for p, (t, d, ex, missed, rr) in enumerate(passes):
        if p >= skip and p > 0:
            ii, jj = np.nonzero(missed)
            for i, j in zip(ii, jj):
                x, s = ex[i, j], d[i, j]
                f = {"seen@1": p - last[i, x] <= 1, "seen@4": p - last[i, x] <= 4, "seen@16": p - last[i, x] <= 16,
                     "seen@64": p - last[i, x] <= 64, "seen ever": last[i, x] >= 0,
                     "heavy@16": p - lastheavy[i, x] <= 16, "heavy@64": p - lastheavy[i, x] <= 64,
                     "resident@16": p - lastres[i, x] <= 16, "resident@64": p - lastres[i, x] <= 64,
                     "resident@boot": bool(res0[i, x]), "evicted by a read": bool(evicted_by_read[i, x])}
                for tn, T in tallies.items():
                    v = T[i, x]
                    rank = int((((T[i] > v) | ((T[i] == v) & (np.arange(e) < x))) & ~R[i]).sum()) if v > 0 else e
                    for r in (1, 2, 4):
                        f[f"{tn} top{r}"] = rank < r
                for c in ("all", "heavy") if s >= 0.3 else ("all",):
                    a = acc[c]
                    a["n"] += 1
                    a["w"] += s
                    for fn, ok in f.items():
                        a[fn] += s if ok else 0.0
        # fold this pass (what the engine knows when it plans after it), then apply its reads
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        last[li, ex] = p
        hv = d >= 0.3
        lastheavy[li[hv], ex[hv]] = p
        for T in tallies.values():
            T *= decay
        np.add.at(tallies["sum"], (li[missed], ex[missed]), d[missed])
        np.add.at(tallies["sum sq"], (li[missed], ex[missed]), d[missed] ** 2)
        np.maximum.at(tallies["max"], (li[missed], ex[missed]), d[missed])
        lastres[R] = p
        for i, x, vi, ve in rr:
            R[vi, ve] = False
            evicted_by_read[vi, ve] = True
            R[i, x] = True
            evicted_by_read[i, x] = False
            for T in tallies.values():
                T[i, x] = 0.0
    print(f"{os.path.basename(path)}: share-weighted recall of the next pass's misses (tallies decayed {decay})")
    for c, a in acc.items():
        print(f"  {c} misses: {int(a['n'])} picks, share {a['w']:.0f}")
        print("    " + "  ".join(f"{fn} {100 * a[fn] / max(a['w'], 1e-9):.1f}%" for fn in feats[:7]))
        print("    " + "  ".join(f"{fn} {100 * a[fn] / max(a['w'], 1e-9):.1f}%" for fn in feats[7:11]))
        for tn in tallies:
            print(f"    {tn:<7} tally rank among the layer's non-residents: " +
                  "  ".join(f"top-{r} {100 * a[f'{tn} top{r}'] / max(a['w'], 1e-9):.1f}%" for r in (1, 2, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--skip", type=int, default=0, help="reads planned before this pass not counted")
    ap.add_argument("--recall", action="store_true", help="recall of the misses by history features instead")
    ap.add_argument("--decay", type=float, default=0.9)
    a = ap.parse_args()
    for t in a.traces:
        if a.recall:
            recall(t, a.skip, a.decay)
        else:
            evaluate(t, a.skip)


if __name__ == "__main__":
    main()
