"""engine/glm/kvb_fp8.py against the BF16 einsums it replaces, and the FP8 embedding rows (GPU)."""

import unittest

import torch

from engine.glm.fp8_block import FP8Block
from engine.glm.kvb_fp8 import HeadFP8, absorb_o, absorb_q
from engine.glm.weights import _fp8_quant

HL, NOPE, V, C = 32, 192, 256, 512


def weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(HL * (NOPE + V), C, generator=g) * 0.05
    q, s = _fp8_quant(w.cuda())
    blk = FP8Block(q.cuda(), s.cuda())
    ref = blk.dequant(torch.bfloat16).view(HL, NOPE + V, C)          # what the engine loaded before
    h = HeadFP8.from_rows(blk.w, blk.s, HL)
    return ref, h.split(0, NOPE), h.split(NOPE, NOPE + V)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class KvbFp8Tests(unittest.TestCase):
    def test_same_weights(self):
        ref, uk, uv = weights()
        self.assertTrue(torch.equal(uk.dequant(), ref[:, :NOPE]))
        self.assertTrue(torch.equal(uv.dequant(), ref[:, NOPE:]))

    def test_absorb_matches_einsum(self):
        ref, uk, uv = weights(1)
        for T in (1, 4, 8, 16, 64):
            g = torch.Generator(device="cuda").manual_seed(T)
            qn = torch.randn(T, HL, NOPE, generator=g, device="cuda").to(torch.bfloat16)
            ol = torch.randn(T, HL, C, generator=g, device="cuda").to(torch.bfloat16)
            for got, want in ((absorb_q(qn, uk), torch.einsum("thn,hnc->thc", qn, ref[:, :NOPE])),
                              (absorb_o(ol, uv), torch.einsum("thc,hvc->thv", ol, ref[:, NOPE:]))):
                err = float((got.float() - want.float()).norm() / want.float().norm())
                self.assertLess(err, 4e-3, (T, err))          # BF16 output rounding, other sum order

    def test_embedding_rows(self):
        from engine.glm.tp import _q8_rows
        w = (torch.randn(3000, 6144) * 0.02).to(torch.bfloat16)
        errs = {}
        for kind in ("fp8", "int8"):
            q, s = _q8_rows(w, "cuda", kind, rows=1024)
            back = (q.float().view(3000, 48, 128) * s[..., None]).view(3000, 6144)
            errs[kind] = float((back - w.cuda().float()).norm() / w.float().norm())
        self.assertLess(errs["fp8"], 0.04, errs)              # E4M3: 3 mantissa bits
        self.assertLess(errs["int8"], errs["fp8"] / 2, errs)  # a uniform grid


if __name__ == "__main__":
    unittest.main()
