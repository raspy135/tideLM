"""Which predictor of a decode pass's routing, known before the pass runs, would have caught its
heavy misses - under the read budget. From a GHT2 hot trace (TIDELM_GLM_HOT_TRACE: each pass's
first position and row token ids beside its routing), of a teacher-forced --bench-nll run.

    python3 tools/glm_route_predict.py TRACE [--skip 100] [--budget 2,4,8,16]

Each predictor scores every (MoE layer, expert) for the pass from what the engine has before it
runs; the budget's B best of the non-resident ones are its reads. The number is the share of the
pass's missed weight (heavy: picks >= 0.3; all) those B would have made resident, summed over
the passes - the recall of an async read planned in time, on this run's routing (no feedback):

  previous pass    the last pass's routing (each pick's share, the max over its rows)
  tally            GLM_HOT_PREDICT's: missed shares summed, decayed 0.9 a pass
  context          the KV cache's routing: every earlier token of this sequence (from the
                   trace; a text's prefill is not in it) at its share, decayed 0.98 a token
  same token       the routing at the earlier positions of this sequence holding the row's own
                   input token (an MTP draft's, known before the pass), decayed 0.5 an occurrence
  token elsewhere  the row's token's mean routing in the trace's other sequences - a per-token
                   table, as a drafted token would look it up (leave-one-sequence-out)
  same + context   the two summed

A sequence starts where a pass's first position is not the last pass's end.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_hot_sim import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=100, help="passes not scored at the start (the boot warm-up)")
    ap.add_argument("--budget", default="2,4,8,16")
    a = ap.parse_args()
    budgets = [int(v) for v in a.budget.split(",")]
    n, e, k, res0, passes = load(a.trace)
    ids, pos = load.ids, load.pos
    if ids[0] is None:
        raise SystemExit(f"{a.trace}: a GHT1 trace (no token ids); record one with this engine")
    # sequences
    seq = np.zeros(len(passes), dtype=np.int64)
    for p in range(1, len(passes)):
        seq[p] = seq[p - 1] + (pos[p] != pos[p - 1] + passes[p - 1][0])
    # the per-token table: every row's routing summed by token, and by (sequence, token)
    tot = {}   # token -> [(n, e) share sum, occurrences]
    own = {}   # (sequence, token) -> same
    for p, (t, d, ex, missed, rr) in enumerate(passes):
        dd, xx = d.reshape(n, t, k), ex.reshape(n, t, k)
        for r in range(t):
            tok = int(ids[p][r])
            for key, tab in ((tok, tot), ((int(seq[p]), tok), own)):
                m = tab.get(key)
                if m is None:
                    m = tab[key] = [np.zeros((n, e), dtype=np.float32), 0]
                np.add.at(m[0], (np.repeat(np.arange(n), k), xx[:, r, :].ravel()), dd[:, r, :].ravel())
                m[1] += 1
    names = ["previous pass", "tally", "context", "same token", "token elsewhere", "same + context"]
    hit = {(nm, b): [0.0, 0.0] for nm in names for b in budgets}
    want = [0.0, 0.0]
    R = res0.copy()
    prev = np.zeros((n, e))
    tally = np.zeros((n, e))
    ctx = np.zeros((n, e))
    same = {}
    cur = -1
    by_pass = {p: rr for p, (_, _, _, _, rr) in enumerate(passes)}
    for p, (t, d, ex, missed, rr) in enumerate(passes):
        for i, x, vi, ve in by_pass.get(p - 1, []):
            R[vi, ve] = False
            R[i, x] = True
        if seq[p] != cur:
            cur = seq[p]
            ctx[:] = 0.0
            same = {}
        li = np.repeat(np.arange(n), d.shape[1]).reshape(d.shape)
        if p >= a.skip:
            heavy = missed & (d >= 0.3)
            miss_w = np.zeros((n, e))
            np.add.at(miss_w, (li[missed], ex[missed]), d[missed])
            heavy_w = np.zeros((n, e))
            np.add.at(heavy_w, (li[heavy], ex[heavy]), d[heavy])
            want[0] += heavy_w.sum()
            want[1] += miss_w.sum()
            sm = np.zeros((n, e))
            el = np.zeros((n, e))
            for r in range(t):
                tok = int(ids[p][r])
                s = same.get(tok)
                if s is not None:
                    np.maximum(sm, s, out=sm)
                g, o = tot[tok], own[(int(seq[p]), tok)]
                c = g[1] - o[1]
                if c > 0:
                    np.maximum(el, (g[0] - o[0]) / c, out=el)
            scores = {"previous pass": prev, "tally": tally, "context": ctx, "same token": sm,
                      "token elsewhere": el, "same + context": sm + ctx}
            out = ~R
            for nm, sc in scores.items():
                v = np.where(out, sc, -1.0).ravel()
                order = np.argsort(-v, kind="stable")
                for b in budgets:
                    top = order[:b]
                    top = top[v[top] > 0]
                    hit[(nm, b)][0] += heavy_w.ravel()[top].sum()
                    hit[(nm, b)][1] += miss_w.ravel()[top].sum()
        # this pass's routing into the state the next pass is predicted from
        prev = np.zeros((n, e))
        np.maximum.at(prev, (li, ex), d)
        tally *= 0.9
        np.add.at(tally, (li[missed], ex[missed]), d[missed])
        dd, xx = d.reshape(n, t, k), ex.reshape(n, t, k)
        for r in range(t):
            ctx *= 0.98
            np.add.at(ctx, (np.repeat(np.arange(n), k), xx[:, r, :].ravel()), dd[:, r, :].ravel())
            tok = int(ids[p][r])
            s = same.get(tok)
            if s is None:
                s = same[tok] = np.zeros((n, e))
            s *= 0.5
            np.add.at(s, (np.repeat(np.arange(n), k), xx[:, r, :].ravel()), dd[:, r, :].ravel())
    print(f"{os.path.basename(a.trace)}: {len(passes) - a.skip} passes scored, {int(seq.max()) + 1} sequences; "
          f"missed weight {want[1]:.0f}, of it heavy (picks >= 0.3) {want[0]:.0f}")
    print(f"  {'predictor':<16} " + " ".join(f"{'B=' + str(b) + ' heavy':>11} {'all':>6}" for b in budgets))
    for nm in names:
        print(f"  {nm:<16} " + " ".join(f"{100 * hit[(nm, b)][0] / max(want[0], 1e-9):>10.1f}% {100 * hit[(nm, b)][1] / max(want[1], 1e-9):>5.1f}%"
                                         for b in budgets))


if __name__ == "__main__":
    main()
