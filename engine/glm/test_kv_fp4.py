"""engine/glm/kv_fp4.py: the store kernel against its torch reference, the round trip's error, and
the cache helpers in tp.py on an FP4 cache (needs a GPU)."""

import unittest

import torch

from engine.glm.kv_fp4 import fp4_dequant, fp4_quant_ref, fp4_store, fp4_to_bf16, parse_layers
from engine.glm.tp import kv_read, kv_span, kv_write

D = 512
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class ParseTests(unittest.TestCase):
    def test_layers(self):
        self.assertEqual(parse_layers("0-2, 7,77-80", 79), {0, 1, 2, 7, 77, 78})
        self.assertEqual(parse_layers("", 79), set())


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class Fp4Tests(unittest.TestCase):
    def rows(self, n, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        w = torch.exp(torch.randn(D, generator=g, device="cuda") * 0.5)       # an RMSNorm gain
        x = torch.randn(n, D, generator=g, device="cuda") * w
        x[3, :16] = 0.0                                                      # an all-zero block
        x[5, 40] = 300.0                                                     # an outlier
        return x.to(torch.bfloat16)

    def test_store_matches_reference(self):
        x = self.rows(64)
        P = torch.zeros((80, D // 2), dtype=torch.uint8, device="cuda")
        S = torch.zeros((80, D // 16), dtype=torch.bfloat16, device="cuda")
        idx = torch.randperm(80, device="cuda")[:64]
        fp4_store(P, S, idx, x)
        p_ref, s_ref = fp4_quant_ref(x)
        self.assertTrue(torch.equal(P[idx], p_ref))
        self.assertTrue(torch.equal(S[idx].view(torch.int16), s_ref.view(torch.int16)))

    def test_round_trip(self):
        x = self.rows(256, seed=1)
        p, s = fp4_quant_ref(x)
        y = fp4_dequant(p, s, torch.float32)
        # Every value is on the grid times its block scale, and an all-zero block stays zero.
        v = (y.view(256, D // 16, 16) / s.float()[..., None]).abs()
        grid = torch.tensor(E2M1, device="cuda")
        # (value x scale is rounded to BF16, 2^-8 relative: 6 comes back as 6 +- 0.024)
        self.assertTrue(((v[..., None] - grid).abs().amin(-1) < 0.03).all())
        self.assertTrue((y[3, :16] == 0).all())
        err = float((y - x.float()).norm() / x.float().norm())
        self.assertLess(err, 0.11, err)                     # ~0.09 for such rows (kv_fp4's docstring)
        self.assertGreater(err, 0.05, err)

    def test_cache_helpers(self):
        x = self.rows(32, seed=2)
        C = (torch.zeros((40, D // 2), dtype=torch.uint8, device="cuda"),
             torch.zeros((40, D // 16), dtype=torch.bfloat16, device="cuda"))
        kv_write(C, torch.arange(32, device="cuda"), x)
        want = fp4_dequant(*fp4_quant_ref(x))
        self.assertTrue(torch.equal(kv_span(C, 32), want))
        sel = torch.tensor([[3, 1], [0, 31]], device="cuda")
        self.assertTrue(torch.equal(kv_read(C, sel), want[sel]))

    def test_bf16_pass_is_dequant(self):
        """fp4_to_bf16 (prefill's one pass over the latent) is fp4_dequant bit for bit, rows
        limited or not."""
        x = self.rows(77)
        packed = torch.zeros(80, D // 2, dtype=torch.uint8, device="cuda")
        scale = torch.zeros(80, D // 16, dtype=torch.bfloat16, device="cuda")
        fp4_store(packed, scale, torch.arange(77, device="cuda"), x)
        self.assertTrue(torch.equal(fp4_to_bf16(packed, scale), fp4_dequant(packed, scale)))
        self.assertTrue(torch.equal(fp4_to_bf16(packed, scale, rows=50), fp4_dequant(packed[:50], scale[:50])))

    def test_prefill_attention_over_bf16_pass(self):
        """The prefill path's BF16 latent gives the FP4 kernel's attention up to fp32 summation
        order: both form the same BF16 keys (the FP4 kernel scores each row as two halves)."""
        from engine.glm.sparse_attn import sparse_mla
        g = torch.Generator(device="cuda").manual_seed(1)
        S, T, H, K = 3000, 48, 32, 2048
        x = self.rows(S, seed=2)
        packed = torch.zeros(S, D // 2, dtype=torch.uint8, device="cuda")
        scale = torch.zeros(S, D // 16, dtype=torch.bfloat16, device="cuda")
        fp4_store(packed, scale, torch.arange(S, device="cuda"), x)
        rope = (torch.randn(S, 64, generator=g, device="cuda")).to(torch.bfloat16)
        q_lat = (torch.randn(T, H, D, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
        q_rope = (torch.randn(T, H, 64, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
        positions = torch.arange(S - T, S, device="cuda")
        sel = torch.stack([torch.randperm(S - T, generator=g, device="cuda")[:K] for _ in range(T)]).to(torch.int32)
        a = sparse_mla(q_lat, q_rope, (packed, scale), rope, sel, positions, 0.07).float()
        b = sparse_mla(q_lat, q_rope, fp4_to_bf16(packed, scale), rope, sel, positions, 0.07).float()
        err = ((a - b).abs().max() / a.abs().max()).item()
        self.assertLess(err, 1e-2, err)                     # one BF16 step at the outputs' scale
        self.assertGreater(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item(), 0.99999)


if __name__ == "__main__":
    unittest.main()
