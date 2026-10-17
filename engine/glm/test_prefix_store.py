"""engine/glm/prefix_store.py on host tensors: segment files, chains, partial matches, eviction.

O_DIRECT needs a real filesystem (not overlayfs): PREFIX_TEST_DIR, default a temp dir under the
repo's results/ (in the image: python -m unittest engine.glm.test_prefix_store).
"""

import os
import shutil
import tempfile
import unittest

import torch

from engine.glm.prefix_store import PrefixStore

S = 4096


def root_dir():
    base = os.environ.get("PREFIX_TEST_DIR") or os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="prefix-test-", dir=base)


class Caches:
    """A rank's caches in the three row modes, filled with a position-dependent pattern."""

    def __init__(self, rank, world=2, seed=0):
        self.rank, self.world = rank, world
        g = torch.Generator().manual_seed(seed)
        self.split = torch.randint(0, 255, (S // world + 1, 48), dtype=torch.uint8, generator=g)
        self.sc = torch.randn(S // world + 1, 4, generator=g)
        self.full = torch.randn(S, 8, generator=g).to(torch.bfloat16)
        self.mtp = torch.randn(S, 16, generator=g)
        self.h = torch.randn(64, 32, generator=g).to(torch.bfloat16)

    def layout(self):
        return [("c0", self.split, "split"), ("s0", self.sc, "split"), ("r0", self.full, "full"),
                ("m", self.mtp, "mtp")]

    def h_parts(self, p):
        return [("h", self.h[p % 64:p % 64 + 1])]

    def clone(self):
        c = Caches.__new__(Caches)
        c.__dict__ = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}
        return c


def store(root, rank=0, budget=1 << 40):
    return PrefixStore(root, "ns", rank, 2, budget, 0, log=lambda *_: None)


def save(ps, caches, prompt, min_tokens=1):
    plan = ps.plan_save(prompt, min_tokens)
    if plan is None:
        return None
    plan["evict"] = ps.plan_evict(ps.estimate(caches.layout(), plan["start"], plan["end"]),
                                  set(ps.chain(plan["parent"])) if plan["parent"] else set())
    ps.drop(plan["evict"]["segs"], plan["evict"]["ns"])
    ps.write_async(plan, prompt, caches.layout(), caches.h_parts(plan["end"] - 1), None)
    ps.join()
    assert not ps.pending_err, ps.pending_err
    return plan


class PrefixStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = root_dir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_round_trip_rows(self):
        for rank in (0, 1):
            ps, src = store(self.root, rank), Caches(rank, seed=rank)
            A = list(range(1000, 1000 + 1500))
            save(ps, src, A)
            dst = Caches(rank, seed=9)
            before = dst.clone()
            plan = ps.plan_restore(A + [7, 8], 0, 1)
            self.assertEqual((plan["n"], plan["h"]), (1500, True))
            self.assertTrue(ps.verify_tokens(plan, A + [7, 8]))
            ps.load(plan, 300, dst.layout(), dst.h_parts(1499))
            # split rows of positions [300, 1500): rank r holds p = 2 j + r
            lo, hi = (300 - rank + 1) // 2, (1500 - rank + 1) // 2
            self.assertTrue(torch.equal(dst.split[lo:hi], src.split[lo:hi]))
            self.assertTrue(torch.equal(dst.split[:lo], before.split[:lo]))
            self.assertTrue(torch.equal(dst.split[hi:], before.split[hi:]))
            self.assertTrue(torch.equal(dst.sc[lo:hi], src.sc[lo:hi]))
            self.assertTrue(torch.equal(dst.full[300:1500], src.full[300:1500]))
            self.assertTrue(torch.equal(dst.full[1500:], before.full[1500:]))
            # MTP rows [299, 1499): made with tokens up to 1499
            self.assertTrue(torch.equal(dst.mtp[299:1499], src.mtp[299:1499]))
            self.assertTrue(torch.equal(dst.mtp[1499], before.mtp[1499]))
            self.assertTrue(torch.equal(dst.h[1499 % 64], src.h[1499 % 64]))
            shutil.rmtree(self.root)
            os.makedirs(self.root)

    def test_chain_partial_and_dedup(self):
        ps, c = store(self.root), Caches(0)
        sysp = list(range(500))
        A1 = sysp + list(range(10_000, 10_700))
        A2 = A1 + list(range(20_000, 20_400))
        s1 = save(ps, c, A1)
        s2 = save(ps, c, A2)
        self.assertEqual((s2["parent"], s2["start"], s2["end"]), (s1["id"], len(A1), len(A2)))
        self.assertIsNone(ps.plan_save(A2, 1))                              # already saved
        # The next turn: the whole chain, with the hidden.
        p = ps.plan_restore(A2 + [1, 2, 3], 0, 1)
        self.assertEqual([x[0] for x in p["chain"]], [s1["id"], s2["id"]])
        self.assertEqual((p["n"], p["h"]), (len(A2), True))
        # A retry that diverges inside turn 2: partial, no hidden.
        B = A1 + list(range(20_000, 20_100)) + [5] * 50
        p = ps.plan_restore(B, 0, 1)
        self.assertEqual((p["n"], p["h"]), (len(A1) + 100, False))
        self.assertTrue(ps.verify_tokens(p, B))
        # Another chat with the same system prompt: part of the root.
        C = sysp + [9] * 300
        p = ps.plan_restore(C, 0, 1)
        self.assertEqual((p["n"], p["h"]), (500, False))
        # Nothing to gain over what is cached.
        self.assertIsNone(ps.plan_restore(A2 + [1], len(A2), 256))
        # A new segment continues from the deepest saved prefix, here A1's.
        s3 = save(ps, c, B)
        self.assertEqual((s3["parent"], s3["start"]), (s1["id"], len(A1)))

    def test_evict_leaves_lru_keep_parent_and_restart(self):
        c = Caches(0)
        ps = store(self.root)
        A = list(range(800))
        s1 = save(ps, c, A)
        one = ps.segs()[s1["id"]]["bytes"]
        s2 = save(ps, c, A + list(range(5000, 5800)))
        s3 = save(ps, c, list(range(30_000, 30_800)))
        ps.touch([s3["id"]])
        # Room for one more segment of this size: the least recently used leaf goes (s2), and
        # never the parent of the segment being written.
        ps.budget = 3 * one + one // 2
        ev = ps.plan_evict(one, keep={s1["id"]})
        self.assertEqual(ev["segs"], [s2["id"]])
        ps.budget = one
        self.assertIsNone(ps.plan_evict(one, keep={s1["id"], s3["id"]}))
        # Dropping a parent drops its descendants; a lost file is reported missing.
        ps.drop([s1["id"]])
        self.assertEqual(set(ps.segs()), {s3["id"]})
        plan = ps.plan_restore(list(range(30_000, 30_800)) + [1], 0, 1)
        self.assertEqual(ps.missing(plan), [])
        os.unlink(os.path.join(ps.dir, s3["id"] + ".tok"))
        self.assertEqual(ps.missing(plan), [s3["id"]])
        # Index survives a restart; segments whose files never published are forgotten.
        ps.budget = 1 << 40
        s4 = save(ps, c, list(range(40_000, 40_600)))
        os.unlink(os.path.join(ps.dir, s3["id"] + ".kv"))
        again = store(self.root)
        self.assertEqual(set(again.segs()), {s4["id"]})

    def test_other_namespaces_go_first(self):
        old = PrefixStore(self.root, "old", 0, 2, 1 << 40, 0, log=lambda *_: None)
        c = Caches(0)
        save(old, c, list(range(900)))
        ps = store(self.root)
        s = save(ps, c, list(range(1000, 1900)))
        ps.budget = ps.segs()[s["id"]]["bytes"] * 2
        ev = ps.plan_evict(ps.segs()[s["id"]]["bytes"])
        self.assertEqual((ev["ns"], ev["segs"]), (["old"], []))


if __name__ == "__main__":
    unittest.main()
