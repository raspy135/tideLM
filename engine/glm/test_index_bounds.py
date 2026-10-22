import unittest

from engine.glm.index_bounds import bucket_keys, graph_key, index_bucket


class IndexBoundsTest(unittest.TestCase):
    def test_bounds_cover_queries_without_changing_short_split_topk(self):
        for capacity in (1024, 4096, 10000, 262144):
            for end in (0, 1, 1024, min(4096, capacity), capacity):
                with self.subTest(capacity=capacity, end=end):
                    bound = index_bucket(end, capacity)
                    self.assertLessEqual(end, bound)
                    self.assertLessEqual(bound, capacity)
                    self.assertGreaterEqual(bound, min(4096, capacity))
        self.assertEqual(index_bucket(4096, 262144), 4096)
        self.assertEqual(index_bucket(4097, 262144), 8192)
        self.assertEqual(index_bucket(8193, 10000), 10000)

    def test_out_of_capacity_queries_fail_before_graph_replay(self):
        for end, capacity in ((-1, 8192), (8193, 8192), (0, 0)):
            with self.subTest(end=end, capacity=capacity), self.assertRaises(ValueError):
                index_bucket(end, capacity)

    def test_graph_keys_separate_bucket_and_mtp_head_variants(self):
        self.assertEqual(graph_key('decode', 4), 4)
        self.assertEqual(graph_key('split', 4), ('split', 4))
        self.assertEqual(graph_key('mtp', 1, logits=False), ('mtp', 1, False))
        keys = [graph_key(kind, 4, bound, logits=logits)
                for kind in ('decode', 'split', 'hot', 'mtp')
                for bound in (4096, 8192) for logits in ((True, False) if kind == 'mtp' else (True,))]
        self.assertEqual(len(keys), len(set(keys)))
        evict = bucket_keys([4, ('split', 4), ('mtp', 1, False), *keys], 4096)
        self.assertEqual(len(evict), 5)
        self.assertTrue(all(key[-1] == 4096 for key in evict))


if __name__ == '__main__':
    unittest.main()
