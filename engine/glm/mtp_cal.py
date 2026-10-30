"""Calibrated confidence gate for the MTP draft chain (GLM_MTP_CAL; off by default).

The draft loop stops when ``conf`` -- the product of the MTP's own probabilities for the drafts
so far -- falls below ``GLM_MTP_DYN``. That product is not the acceptance it claims to be. On 545
recorded rounds of five texts at temperature 1.0, always drafting 7 (``results/glm-dyn/cal1.json``,
2026-10-19), a stated chain probability of 0.4-0.6 at draft 2 was in the accepted prefix 28% of the
time; 0.2-0.4 at draft 3, 10%; and a stated chain of 0.5-0.7 accepted about a third
(2026-10-16). The profile compensates with a high threshold (0.9) on a number
whose scale is off by up to 3x, and the chain then stops at ~1.2 drafts.

A verify row is worth its *marginal* acceptance -- P(prefix accepted through draft k), which is
what one cell of this table measures -- and it should be kept while that exceeds the row's share
of the round: (row + draft) ms over the round's ms a token, 26.5/73 = 0.36 at 12-14 tok/s with the
measured 22 ms row (2026-10-17). Replayed over the same records with the table
fitted leave-one-prompt-out and that cost model (95 ms fixed + 22 ms a row + 4.5 ms a draft):
calibrated at 0.36, 14.03 tok/s; the flat stated gate at 0.8, 13.63; this profile's 0.9, 13.22.
The result is flat from 0.25 to 0.5, so the threshold is not delicate.

The bucket is the *stated* chain probability the draft was reached with, so nothing new has to be
carried through the round: ``conf`` stays the key and the calibrated probability replaces the
comparison. A cell learns from every round the gate reaches (``observe``, rank 0 only -- it is
the rank that drafts and decides), so the table corrects itself for temperature, workload and
context depth; the counts below are only a starting point.

Off by default because it changes which drafts are made and, with them, the text a sampled reply
follows. The offline numbers and their cost model are from 2026-10-23; the on-box A/B is still
missing.

    GLM_MTP_CAL        threshold on the calibrated probability; 0 (default) keeps the flat gate
    GLM_MTP_CAL_PRIOR  1 (default): start from the counts below; 0: learn from this process only
"""

from __future__ import annotations

# Stated chain probability buckets: [lo, hi). The top bucket is closed at 1.0.
EDGES = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
NB = len(EDGES) - 1

# (accepted, seen) per draft position and bucket: P(draft k is in the accepted prefix) given the
# stated chain probability it was drafted at. Pooled over five texts, 545 rounds, temperature 1.0,
# always drafting 7: results/glm-dyn/cal1.json (2026-10-19). Position 1 accepts 0.75 overall, 7
# accepts 0.02; the cells say the stated probability is optimistic everywhere but the top bucket.
PRIOR = (
    ((22, 62), (21, 31), (24, 46), (14, 29), (17, 35), (34, 46), (28, 36), (248, 260)),
    ((15, 146), (19, 65), (25, 59), (19, 35), (11, 31), (27, 39), (17, 29), (117, 141)),
    ((22, 259), (13, 68), (17, 60), (11, 39), (7, 20), (12, 22), (14, 19), (44, 58)),
    ((20, 336), (6, 63), (9, 41), (10, 29), (7, 19), (6, 16), (6, 9), (20, 31)),
    ((8, 407), (4, 41), (6, 32), (3, 20), (3, 9), (4, 10), (5, 5), (10, 18)),
    ((5, 449), (2, 27), (4, 27), (2, 8), (0, 3), (3, 7), (2, 5), (4, 9)),
    ((1, 467), (1, 26), (4, 13), (2, 4), (0, 0), (0, 5), (1, 5), (2, 4)),
)

SMOOTH = 4.0          # pseudo-counts pulling a thin cell to its position's pooled rate


def bucket(conf: float) -> int:
    """Bucket index for a stated chain probability; clamps outside [0, 1]."""
    c = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
    for i in range(NB - 1):
        if c < EDGES[i + 1]:
            return i
    return NB - 1


class CalibratedGate:
    """P(draft k is in the accepted prefix | stated chain probability), per draft position.

    Not a replacement for the chain product in the sampler: the engine still multiplies the
    drafts' probabilities to get ``conf``; this only decides whether the next draft is worth its
    verify row.
    """

    def __init__(self, tau: float, prior: bool = True):
        self.tau = float(tau)
        if self.tau <= 0.0:
            raise ValueError(f"GLM_MTP_CAL={tau}: the threshold must be positive (0 turns it off)")
        self.acc: list[list[int]] = [[a for a, _ in row] for row in PRIOR] if prior else []
        self.n: list[list[int]] = [[s for _, s in row] for row in PRIOR] if prior else []

    # -- bookkeeping ------------------------------------------------------------------
    def _row(self, k: int) -> int:
        """Make position k (0-based) present; positions past the shipped table start empty."""
        while len(self.acc) <= k:
            self.acc.append([0] * NB)
            self.n.append([0] * NB)
        return k

    def observe(self, k: int, conf: float, accepted: bool) -> None:
        """One draft's outcome. ``accepted`` means it was in the round's accepted prefix."""
        row = self._row(k)
        b = bucket(conf)
        self.n[row][b] += 1
        if accepted:
            self.acc[row][b] += 1

    # -- decision ---------------------------------------------------------------------
    def prob(self, k: int, conf: float) -> float:
        """Calibrated P(the prefix reaches and accepts draft k) for this stated chain probability.

        A cell with a few samples is pulled toward the position's pooled rate by ``SMOOTH``
        pseudo-counts; a position with nothing measured at all keeps the stated probability, so a
        cold table reproduces the flat gate's reading rather than inventing one.
        """
        row = self._row(k)
        b = bucket(conf)
        a, n = self.acc[row][b], self.n[row][b]
        total, seen = sum(self.acc[row]), sum(self.n[row])
        pooled = total / seen if seen else float(conf)
        return (a + SMOOTH * pooled) / (n + SMOOTH)

    def keep(self, k: int, conf: float) -> bool:
        """Is one more draft worth its verify row at this stated chain probability?"""
        return self.prob(k, conf) >= self.tau

    # -- reporting --------------------------------------------------------------------
    def rates(self) -> dict[int, list[int]]:
        """[accepted, seen] a draft position (1-based), the gate's own learning record."""
        return {i + 1: [sum(a), sum(n)] for i, (a, n) in enumerate(zip(self.acc, self.n))}


class LadderGate:
    """A per-position threshold on the *stated* chain probability (GLM_MTP_LADDER).

    ``GLM_MTP_LADDER=0.7,0.5,0.2,0.1,0.05`` keeps draft k while its chain probability is at
    least the k-th value (positions past the list reuse the last). It is the answer to the
    calibrated table's problem -- the stated product is optimistic -- by simply lowering the bar
    with depth instead of re-scaling the reading, so it needs no measurements and behaves the same
    at any temperature.

    It is not the same rule: the chain probability is the expected token yield of the *next* row
    (the product of everything so far), so a ladder that falls with depth asks for a yield the
    row's cost may not justify. Replayed over results/glm-dyn/cal1.json at temperature 1.0 it
    modelled 12.7-13.3 tok/s against the flat 0.9's 13.22 and the calibrated gate's 14.03
    (2026-10-23). Kept as a knob because the cost side is what decides -- on code
    at temperature 0.6 the flat 0.9 was drafting 3.3 drafts a round at 73% acceptance, which no
    table measured on prose-like text predicts -- and because counting outcomes (``rates``) makes
    the two comparable in one A/B.
    """

    def __init__(self, ladder):
        vals = [float(v) for v in (ladder if isinstance(ladder, (list, tuple)) else ladder.split(","))
                if str(v).strip()]
        if not vals:
            raise ValueError("GLM_MTP_LADDER: at least one threshold is required")
        if any(not 0.0 < v <= 1.0 for v in vals):
            raise ValueError(f"GLM_MTP_LADDER={ladder}: thresholds must be in (0, 1]")
        self.ladder = vals
        self.acc: list[list[int]] = []
        self.n: list[list[int]] = []

    def keep(self, k: int, conf: float) -> bool:
        return conf >= self.ladder[min(k, len(self.ladder) - 1)]

    def observe(self, k: int, conf: float, accepted: bool) -> None:
        while len(self.acc) <= k:
            self.acc.append([0] * NB)
            self.n.append([0] * NB)
        b = bucket(conf)
        self.n[k][b] += 1
        if accepted:
            self.acc[k][b] += 1

    def rates(self) -> dict[int, list[int]]:
        return {i + 1: [sum(a), sum(n)] for i, (a, n) in enumerate(zip(self.acc, self.n))}
