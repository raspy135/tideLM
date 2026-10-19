"""rope_kernel.rope is ref.rope_interleave, bit for bit."""

import unittest

import torch

from engine.glm.ref import rope_interleave
from engine.glm.rope_kernel import rope


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class RopeKernelTests(unittest.TestCase):
    def test_bit_identical(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        theta = 8e6 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
        for T, H, rows in ((1, 32, 192), (4, 32, 192), (8, 1, 512), (3, 32, 0)):
            pos = torch.randint(0, 400000, (T,), device="cuda", generator=g)
            f = pos[:, None].float() * theta[None]
            emb = torch.cat((f, f), -1)
            cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
            # As the engine slices it: the RoPE part at the end of a wider row.
            full = (torch.randn(T, H, rows + 64, device="cuda", generator=g) * 3).to(torch.bfloat16)
            x = full[..., rows:]
            self.assertTrue(torch.equal(rope(x, cos, sin), rope_interleave(x, cos, sin)))


if __name__ == "__main__":
    unittest.main()
