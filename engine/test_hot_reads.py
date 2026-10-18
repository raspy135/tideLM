"""engine/hot_reads.py: the candidate kernel and the per-step plan (CPU)."""
import unittest

import torch

from engine.hot_reads import HotReads, HotRing, PrefillEmulation


class HotReadsTest(unittest.TestCase):
    def hot(self, **env):
        h = HotReads(4, 8, 2, {'DSV41_HOT_MIN': '0.2', 'DSV41_HOT_MAX': '3', **env})
        h.layers = [0, 1, 2, 3]
        return h

    def test_candidates_are_missing_unpruned_picks_by_share(self):
        h = self.hot()
        scores = torch.tensor([[5., 3., 1., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 4., 0., 0., 0.]])
        keep = torch.tensor([False, True, True, True, True, True, True, True])
        out = torch.zeros(2 * h.width)
        h.candidates(scores, scores, keep, out)
        # row 0: expert 0 (5 of the top-2's 8) is missing; row 1's picks (4, 3) are resident
        self.assertAlmostEqual(float(out[0]), 5 / 8, places=5)
        self.assertEqual(int(out[h.width]), 0)
        self.assertEqual(float(out[1]), 0.0)

    def test_plan_threshold_per_layer_and_paced_cap(self):
        h = self.hot()
        h.begin_step()
        w = h.width
        row = lambda pairs: [v for v, _ in pairs] + [0.0] * (w - len(pairs)) + [float(e) for _, e in pairs] + [0.0] * (w - len(pairs))
        self.assertEqual(h.plan(0, row([(0.6, 5), (0.5, 6)])), [5])          # one a layer
        self.assertEqual(h.plan(1, row([(0.1, 7)])), [])                    # under the threshold
        self.assertEqual(h.plan(2, row([(0.9, 1)])), [1])                   # 2 of ceil(3*3/4)=3
        self.assertEqual(h.plan(3, row([(0.9, 2)])), [2])                   # 3 of 3
        h.begin_step()
        self.assertEqual(h.plan(0, row([(0.9, 2)])), [2])                   # a new step's budget

    def test_prefill_emulation_widens_a_block_with_its_paced_budget(self):
        # 2 layers, 8 experts, k=2, cap 1: layer 0's budget is ceil(1*1/2)=1 a block
        h = HotReads(2, 8, 2, {'DSV41_HOT_MIN': '0.3', 'DSV41_HOT_MAX': '1'})
        em = PrefillEmulation(h, [0, 1], block=2)
        keep = torch.tensor([False, True, True, True, True, True, True, True])
        scores = torch.tensor([[5., 3., 0., 0., 0., 0., 0., 0.],    # wants 0 (5/8), missing
                               [0., 1., 1., 0., 0., 0., 0., 0.],    # residents only
                               [0., 0., 0., 2., 1., 0., 0., 0.]])   # residents only
        rk, load = em.widen(0, scores, scores, keep, 2)
        self.assertEqual(load, [0])
        self.assertTrue(rk[0, 0] and rk[1, 0])        # the whole block of row 0
        self.assertFalse(rk[2, 0])                    # not the next block
        rk, load = em.widen(1, scores, scores, keep, 2)
        self.assertEqual(load, [])                    # block 0 spent its 1 read
        # exact: every missing pick, one-row blocks, no budget
        ex = PrefillEmulation(HotReads(2, 8, 2, {'DSV41_HOT_MIN': '1e-9', 'DSV41_HOT_MAX': '1000000',
                                                 'DSV41_HOT_PER_LAYER': '2'}), [0, 1], block=1)
        rk, load = ex.widen(0, scores, scores, keep, 2)
        self.assertEqual(load, [0])
        self.assertTrue(rk[0, 0])
        self.assertFalse(rk[1, 0])
        self.assertTrue(torch.equal(scores.masked_fill(~rk, float('-inf')).topk(2)[1].sort()[0],
                                    scores.topk(2)[1].sort()[0]))

    def test_async_plan_takes_the_heaviest_across_layers_once(self):
        h = self.hot(DSV41_HOT_ASYNC='1', DSV41_HOT_MAX='2', DSV41_HOT_RING='4')
        w = h.width
        row = lambda pairs: [v for v, _ in pairs] + [0.0] * (w - len(pairs)) + [float(e) for _, e in pairs] + [0.0] * (w - len(pairs))
        rows = [row([(0.4, 3)]), row([(0.9, 5), (0.8, 6)]), row([(0.1, 2)]), row([(0.5, 7)])]
        # one a layer: layer 1's 0.9, then layer 3's 0.5 (layer 1's 0.8 is its second); cap 2
        self.assertEqual(h.plan_async(rows, set()), [(1, 5), (3, 7)])
        self.assertEqual(h.plan_async(rows, {(1, 5)}), [(1, 6), (3, 7)])    # the ring has (1, 5)
        with self.assertRaises(ValueError):
            self.hot(DSV41_HOT_ASYNC='1', DSV41_HOT_MAX='8', DSV41_HOT_RING='4')

    def test_ring_is_fifo_and_a_read_is_routable_from_the_next_step(self):
        r = HotRing([100, 101])
        self.assertEqual(r.take((0, 1)), 100)
        self.assertEqual(r.live, {})                       # reading: not routable this step
        r.promote()
        self.assertEqual(r.live, {(0, 1): 100})
        self.assertEqual(r.take((0, 2)), 101)
        self.assertEqual(r.take((0, 3)), 100)              # the oldest slot; (0, 1) stops at once
        self.assertEqual(r.live, {})
        self.assertEqual(r.taken(), {(0, 2), (0, 3)})
        r.promote()
        self.assertEqual(r.live, {(0, 2): 101, (0, 3): 100})


if __name__ == '__main__':
    unittest.main()
