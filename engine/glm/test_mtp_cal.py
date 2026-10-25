"""Unit tests for the calibrated MTP gate, and the offline replay it was chosen from.

The replay reads the recorded draft records (results/glm-dyn/cal1.json: 545 rounds of five texts,
temperature 1.0, always drafting 7) and scores a gate policy with the decode round's cost model
from docs/gotchas.md (95 ms fixed + 22 ms a verify row + 4.5 ms a draft). It is a regression test
on the *claim* in engine/glm/mtp_cal.py's docstring, not a serving measurement: the calibrated
gate must stay ahead of the flat stated threshold on the same records.
"""

import json
import os
import unittest

from engine.glm.mtp_cal import NB, PRIOR, CalibratedGate, LadderGate, bucket

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "..", "..", "results", "glm-dyn", "cal1.json")

FIXED_MS, ROW_MS, DRAFT_MS = 95.0, 22.0, 4.5


def rounds_of(path):
    """[(chain probability, accepted)] a round, for every run in the file."""
    out = []
    with open(path) as f:
        runs = json.load(f)
    for run in runs:
        cur = []
        for rec in run["records"]:
            if rec[0] == 0 and cur:
                out.append(cur)
                cur = []
            cur.append((rec[1], bool(rec[2])))
        if cur:
            out.append(cur)
    return out


def model_tok_s(rounds, keep, cost=lambda kept: FIXED_MS + ROW_MS * (1 + kept) + DRAFT_MS * kept):
    tokens = ms = 0.0
    for rd in rounds:
        kept = 0
        for conf, _ in rd:
            if not keep(kept, conf):
                break
            kept += 1
        accepted = 0
        for _, ok in rd[:kept]:
            if not ok:
                break
            accepted += 1
        tokens += 1 + accepted
        ms += cost(kept)
    return tokens / (ms / 1000.0)


class BucketTest(unittest.TestCase):
    def test_edges_and_clamping(self):
        self.assertEqual(bucket(0.0), 0)
        self.assertEqual(bucket(0.1999), 0)
        self.assertEqual(bucket(0.2), 1)
        self.assertEqual(bucket(0.94), 6)
        self.assertEqual(bucket(0.95), 7)
        self.assertEqual(bucket(1.0), NB - 1)
        self.assertEqual(bucket(1.7), NB - 1)
        self.assertEqual(bucket(-0.3), 0)


class ColdTableTest(unittest.TestCase):
    def test_no_history_reads_the_stated_probability(self):
        g = CalibratedGate(0.36, prior=False)
        for conf in (0.1, 0.5, 0.83, 0.99):
            self.assertAlmostEqual(g.prob(0, conf), conf, places=6)
            self.assertEqual(g.keep(0, conf), conf >= 0.36)

    def test_observation_moves_toward_the_measured_rate(self):
        g = CalibratedGate(0.3, prior=False)
        for _ in range(40):
            g.observe(1, 0.75, False)
        # 40 misses at a stated 0.75: the cell must read far below it, and below the threshold.
        self.assertLess(g.prob(1, 0.75), 0.1)
        self.assertFalse(g.keep(1, 0.75))
        for _ in range(60):
            g.observe(1, 0.75, True)
        self.assertAlmostEqual(g.prob(1, 0.75), 0.6, places=2)
        self.assertTrue(g.keep(1, 0.75))

    def test_positions_past_the_table_grow_empty_rows(self):
        g = CalibratedGate(0.36, prior=False)
        g.observe(9, 0.5, True)
        self.assertEqual(g.rates()[10], [1, 1])
        self.assertEqual(g.rates()[1], [0, 0])

    def test_reporting_counts_by_position(self):
        g = CalibratedGate(0.36, prior=False)
        g.observe(0, 0.9, True)
        g.observe(0, 0.9, False)
        self.assertEqual(g.rates(), {1: [1, 2]})

    def test_non_positive_threshold_is_refused(self):
        with self.assertRaises(ValueError):
            CalibratedGate(0.0)


class PriorTest(unittest.TestCase):
    def test_prior_is_the_recorded_table(self):
        self.assertEqual(len(PRIOR), 7)
        self.assertTrue(all(len(row) == NB for row in PRIOR))
        seen = [sum(n for _, n in row) for row in PRIOR]
        self.assertEqual(seen, [545, 545, 545, 544, 542, 535, 524])
        # Acceptance falls with depth and the prior is not a flat line.
        rate = [sum(a for a, _ in row) / sum(n for _, n in row) for row in PRIOR]
        self.assertAlmostEqual(rate[0], 0.75, places=2)
        self.assertLess(rate[6], 0.03)
        self.assertEqual(rate, sorted(rate, reverse=True))

    def test_prior_is_optimistic_at_mid_confidence(self):
        g = CalibratedGate(0.36)
        # Stated 0.5 at draft 3 read as 0.28: that is the whole point of the table.
        self.assertLess(g.prob(2, 0.5), 0.3)
        self.assertAlmostEqual(g.prob(0, 0.97), 0.95, places=2)


class LadderTest(unittest.TestCase):
    def test_thresholds_fall_with_depth_and_reuse_the_last(self):
        g = LadderGate("0.7,0.5,0.2")
        self.assertEqual([g.keep(k, c) for k, c in enumerate((0.72, 0.55, 0.25, 0.1))],
                         [True, True, True, False])
        self.assertTrue(g.keep(9, 0.21))       # past the list: the last value decides
        self.assertFalse(g.keep(9, 0.19))

    def test_it_counts_outcomes_for_comparison(self):
        g = LadderGate("0.5")
        g.observe(0, 0.8, True)
        g.observe(0, 0.8, False)
        g.observe(3, 0.3, False)
        self.assertEqual(g.rates(), {1: [1, 2], 2: [0, 0], 3: [0, 0], 4: [0, 1]})

    def test_bad_ladders_are_refused(self):
        for bad in ("", "0", "1.5", "-0.2", ","):
            with self.subTest(ladder=bad), self.assertRaises(ValueError):
                LadderGate(bad)


@unittest.skipUnless(os.path.exists(RECORDS), "recorded calibration rounds are not in this tree")
class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.rounds = rounds_of(RECORDS)

    def test_calibrated_gate_beats_the_flat_stated_threshold(self):
        g = CalibratedGate(0.36)
        calibrated = model_tok_s(self.rounds, g.keep)
        flat_best = model_tok_s(self.rounds, lambda k, conf: conf >= 0.8)
        shipped = model_tok_s(self.rounds, lambda k, conf: conf >= 0.9)
        self.assertAlmostEqual(flat_best, 13.63, places=1)
        self.assertAlmostEqual(shipped, 13.22, places=1)
        self.assertAlmostEqual(calibrated, 14.01, places=1)
        self.assertGreater(calibrated, flat_best + 0.2)

    def test_ladder_is_not_better_than_the_calibrated_gate(self):
        """The ladder lowers the bar with depth; the modelled replay says that over-drafts."""
        ladder = model_tok_s(self.rounds, LadderGate("0.7,0.5,0.2,0.1,0.05").keep)
        calibrated = model_tok_s(self.rounds, CalibratedGate(0.36).keep)
        self.assertLess(ladder, calibrated)
        self.assertGreater(ladder, 12.0)

    def test_threshold_plateau(self):
        scores = {t: model_tok_s(self.rounds, CalibratedGate(t).keep) for t in (0.25, 0.36, 0.5)}
        for t, v in scores.items():
            self.assertGreater(v, 13.8, t)
        self.assertLess(max(scores.values()) - min(scores.values()), 0.2)

    def test_gate_learns_the_outcomes_it_is_shown(self):
        g = CalibratedGate(0.36, prior=False)
        for rd in self.rounds:
            for k, (conf, ok) in enumerate(rd):
                g.observe(k, conf, ok)
        rates = g.rates()
        self.assertAlmostEqual(rates[1][0] / rates[1][1], 0.75, places=2)
        self.assertEqual(rates[1], [408, 545])
        # An all-seen table must not be more optimistic than the shipped prior at depth.
        self.assertLess(g.prob(4, 0.6), 0.2)


if __name__ == "__main__":
    unittest.main()
