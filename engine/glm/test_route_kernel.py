"""route_kernel against the PyTorch routing it replaces (ref.route_pick + GlmTP._route)."""

import unittest
from types import SimpleNamespace

import torch

from engine.glm.ref import route_pick
from engine.glm.route_kernel import exact_pick, routed_pick

CFG = SimpleNamespace(topk=8, norm_topk=True, routed_scaling=2.5, n_group=1)
E, K = 256, 8


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class RouteKernelTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.bias = torch.randn(E, device="cuda", generator=g) * 0.1
        self.allowed = torch.rand(E, device="cuda", generator=g) < 0.6
        self.lut = torch.where(self.allowed, torch.randperm(E, device="cuda").int() + 100,
                               torch.full((E,), -1, dtype=torch.int32, device="cuda"))
        self.fixed = torch.tensor([7], dtype=torch.int32, device="cuda")
        self.g = g

    def logits(self, T):
        return torch.randn(T, E, device="cuda", generator=self.g) * 2

    @staticmethod
    def by_id(ids, w):
        return [dict(zip(i.tolist(), x.tolist())) for i, x in zip(ids, w)]

    def check_same(self, ids_a, w_a, ids_b, w_b):
        for da, db in zip(self.by_id(ids_a, w_a), self.by_id(ids_b, w_b)):
            self.assertEqual(set(da), set(db))
            for e in da:
                self.assertAlmostEqual(da[e], db[e], delta=1e-6 * max(1.0, abs(db[e])))

    def test_exact_and_hot(self):
        for T in (1, 2, 3, 5, 8):
            lg = self.logits(T)
            sc = lg.sigmoid()
            w_ref, ids_ref = route_pick(CFG, sc, sc + self.bias, None)
            n2 = 32
            hot = torch.zeros(4 * 16, device="cuda")
            ids0, w0 = exact_pick(lg, self.bias, self.allowed.view(torch.uint8), K, 2.5, hot, n2)
            self.check_same(ids0, w0, ids_ref, w_ref)
            share = (w0 / 2.5) * (~self.allowed[ids0])
            kh = min(T * K, n2)
            v, i = share.reshape(-1).topk(kh)
            torch.testing.assert_close(hot[:kh], v)
            pos = v > 0
            self.assertTrue(torch.equal(hot[n2:n2 + kh][pos], ids0.reshape(-1)[i][pos].float()))

    def test_routed_and_counters(self):
        for T in (1, 4, 7, 8):
            for x3_off in (None, 50):
                lg = self.logits(T)
                sc = lg.sigmoid()
                choice = sc + self.bias
                w_ref, ids_ref = route_pick(CFG, sc, choice, self.allowed)
                w0_ref, ids0_ref = route_pick(CFG, sc, choice, None)
                ids0, w0 = exact_pick(lg, self.bias, self.allowed.view(torch.uint8), K, 2.5)
                demand = torch.zeros(E, device="cuda")
                acc = torch.zeros(4, dtype=torch.float64, device="cuda")
                out = routed_pick(lg, self.bias, self.allowed.view(torch.uint8), self.lut, self.fixed, ids0, w0,
                                  2.5, x3_off=x3_off, demand=demand, acc=acc)
                slots_ref = self.lut[ids_ref]
                if x3_off is None:
                    sl, wl = out
                    self.check_same(sl[:, :K], wl[:, :K], slots_ref, w_ref)
                    self.assertTrue(torch.equal(sl[:, K:], self.fixed[None].expand(T, -1)))
                    self.assertTrue(torch.equal(wl[:, K:], torch.ones(T, 1, device="cuda")))
                else:
                    rs, wr, sl, wl = out
                    self.check_same(rs, wr, (slots_ref - x3_off).clamp_min(0), w_ref)
                    self.assertTrue(torch.equal(sl, self.fixed[None].expand(T, -1)))
                dem_ref = torch.zeros(E, device="cuda").index_add_(0, ids0_ref.reshape(-1), w0_ref.reshape(-1))
                torch.testing.assert_close(demand, dem_ref, rtol=1e-6, atol=1e-6)
                missed = ~self.allowed[ids0_ref]
                ref = torch.tensor([ids0_ref.numel(), missed.sum().item(), w0_ref.sum().item(),
                                    (w0_ref * missed).sum().item()], dtype=torch.float64, device="cuda")
                torch.testing.assert_close(acc, ref, rtol=1e-5, atol=1e-5)

    def test_slotless_pick(self):
        # An allowed expert with no slot (a stream miss not yet placed) becomes a zero-weight
        # pick of the first fixed slot, as in GlmTP._route.
        lg = self.logits(2)
        lut = self.lut.clone()
        ids0, w0 = exact_pick(lg, self.bias, self.allowed.view(torch.uint8), K, 2.5)
        sc = lg.sigmoid()
        _, ids_ref = route_pick(CFG, sc, sc + self.bias, self.allowed)
        lut[ids_ref[0, 0]] = -1
        sl, wl = routed_pick(lg, self.bias, self.allowed.view(torch.uint8), lut, self.fixed, ids0, w0, 2.5)
        j = (sl[0, :K] == 7).nonzero().flatten()
        self.assertEqual(len(j), 1)
        self.assertEqual(float(wl[0, j[0]]), 0.0)


if __name__ == "__main__":
    unittest.main()
