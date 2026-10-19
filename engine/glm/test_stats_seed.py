"""stats_seed.load_stats: a local DB always wins; the seed only starts a DB that does not exist."""

import tempfile
import unittest
from pathlib import Path

import torch

from engine.glm.stats_seed import DEFAULT_PATH, load_stats

SHAPE = (79, 256)


def save(path, mass, tokens):
    torch.save({"version": 1, "mass": mass, "tokens": tokens}, path)


class StatsSeedTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = Path(self.dir.name)
        self.seed = self.d / "seed.pt"
        save(self.seed, torch.full(SHAPE, 2.0), 500)

    def tearDown(self):
        self.dir.cleanup()

    def test_local_db_wins_over_seed(self):
        db = self.d / "db.pt"
        save(db, torch.ones(SHAPE), 7)
        mass, tokens, note = load_stats(str(db), SHAPE, str(self.seed))
        self.assertEqual((float(mass.sum()), tokens, note), (79 * 256, 7, None))

    def test_seed_starts_a_missing_db(self):
        mass, tokens, note = load_stats(str(self.d / "db.pt"), SHAPE, str(self.seed))
        self.assertEqual((mass.dtype, float(mass[0, 0]), tokens), (torch.float64, 2.0, 500))
        self.assertIn("starting from the seed seed.pt", note)

    def test_seed_off(self):
        mass, tokens, note = load_stats(str(self.d / "db.pt"), SHAPE, "0")
        self.assertEqual((float(mass.abs().sum()), tokens, note), (0.0, 0, None))

    def test_other_models_db_starts_empty_not_seeded(self):
        db = self.d / "db.pt"
        save(db, torch.ones(78, 256), 7)
        mass, tokens, note = load_stats(str(db), SHAPE, str(self.seed))
        self.assertEqual((tuple(mass.shape), float(mass.abs().sum()), tokens), (SHAPE, 0.0, 0))
        self.assertIn("ignored", note)

    def test_bad_or_missing_seed_starts_empty(self):
        bad = self.d / "bad.pt"
        save(bad, torch.full(SHAPE, float("nan")), 5)
        for seed in (bad, self.d / "absent.pt"):
            mass, tokens, note = load_stats(str(self.d / "db.pt"), SHAPE, str(seed))
            self.assertEqual((float(mass.abs().sum()), tokens), (0.0, 0))
            self.assertIn("starting empty", note)

    def test_bundled_seed_is_glm53(self):
        mass, tokens, note = load_stats(str(self.d / "db.pt"), SHAPE, "1")
        self.assertTrue(DEFAULT_PATH.exists())
        self.assertGreater(tokens, 0)
        self.assertIn(DEFAULT_PATH.name, note)
        # Dense layers 0-2 route nowhere; every MoE layer (and the MTP layer, 78) has demand.
        self.assertEqual(float(mass[:3].sum()), 0.0)
        self.assertTrue((mass[3:].sum(1) > 0).all())


if __name__ == "__main__":
    unittest.main()
