"""engine/expert_profiles.py: the residency ranking settings and the mask digest (CPU)."""
import unittest

import numpy as np

from engine.expert_profiles import ExpertProfile, mask_digest


class ExpertProfileTest(unittest.TestCase):
    def test_same_cardinality_different_ids_have_different_digest(self):
        self.assertNotEqual(mask_digest({0: np.array([True, False])}), mask_digest({0: np.array([False, True])}))
        self.assertEqual(mask_digest({0: np.array([True, False])}), mask_digest({0: np.array([1, 0], dtype=np.int64)}))

    def test_empty_options_keep_the_trace_ranking_and_its_topics(self):
        p = ExpertProfile({'DSV41_PRUNE_SOURCE': '', 'DSV41_PRUNE_RANK': '', 'DSV41_EXPERT_TOPICS': ''})
        self.assertEqual((p.rank, p.topics), ('sum', ('coding', 'general')))
        self.assertEqual(ExpertProfile({'DSV41_PRUNE_RANK': 'max'}).boot_fields()['prune_rank'], 'max')

    def test_removed_topic_profiles_fail_loudly(self):
        # Removed 2026-10-17 (docs/expert-profiles.md): asking for one must not silently rank on counts.
        for env in ({'DSV41_PRUNE_SOURCE': 'saliency'}, {'DSV41_PRUNE_RANK': 'maxmin'},
                    {'DSV41_EXPERT_TOPICS': 'python'}):
            with self.assertRaisesRegex(ValueError, 'removed'):
                ExpertProfile(env)


if __name__ == '__main__':
    unittest.main()
