"""GLM_LOOP_GUARD / GLM_LOOP_BREAK: the cycle detection both ranks run on the emitted tokens."""

import unittest

from engine.glm.tp import GlmTP


class TailCycleTests(unittest.TestCase):
    def test_cycles(self):
        f = GlmTP._tail_cycle
        self.assertEqual(f([1, 7, 7, 7, 7], 4), (7,))
        self.assertIsNone(f([7, 7, 7], 4))
        self.assertEqual(f([9, 3, 4, 3, 4, 3, 4], 3), (3, 4))
        self.assertEqual(f([9, 4, 3, 4, 3, 4, 3], 3), (3, 4))          # the same cycle, other phase
        self.assertIsNone(f([1, 2, 3, 1, 2, 4], 2))
        self.assertEqual(f(list(range(8)) * 2, 2), tuple(range(8)))
        self.assertIsNone(f(list(range(9)) * 2, 2))                   # longer than 8 tokens


if __name__ == "__main__":
    unittest.main()
