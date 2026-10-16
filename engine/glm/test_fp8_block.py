"""engine/glm/fp8_block.py against a dequantized BF16 matmul (needs a GPU)."""

import unittest

import torch

from engine.glm.fp8_block import FP8Block, fp8_block_linear


def make(N, K, g):
    ref = torch.randn(N, K, generator=g, device="cuda") * 0.05
    nb, kb = (N + 127) // 128, K // 128
    pad = torch.zeros(nb * 128, K, device="cuda")
    pad[:N] = ref
    amax = pad.view(nb, 128, kb, 128).abs().amax((1, 3)).clamp_min(1e-12)
    s = amax / 448.0
    w = (pad.view(nb, 128, kb, 128) / s[:, None, :, None]).view(nb * 128, K)[:N].to(torch.float8_e4m3fn)
    return FP8Block(w, s)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class FP8BlockTests(unittest.TestCase):
    def check(self, M, N, K):
        g = torch.Generator(device="cuda").manual_seed(M * 7 + N)
        W = make(N, K, g)
        x = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
        want = x.float() @ W.dequant(torch.float32).t()
        got = fp8_block_linear(x, W, out_dtype=torch.float32)
        rel = float((got - want).norm() / want.norm())
        self.assertLess(rel, 2e-3, (M, N, K, rel))

    def test_shapes(self):
        for M in (1, 2, 8, 16, 40, 300):
            for N, K in ((576, 6144), (2048, 6144), (8192, 2048), (6144, 8192), (128, 6144)):
                self.check(M, N, K)

    def test_shards_keep_their_scales(self):
        g = torch.Generator(device="cuda").manual_seed(1)
        W = make(1024, 512, g)
        x = torch.randn(3, 512, generator=g, device="cuda").to(torch.bfloat16)
        full = fp8_block_linear(x, W, out_dtype=torch.float32)
        top = fp8_block_linear(x, W.rows(512, 1024), out_dtype=torch.float32)
        self.assertTrue(torch.allclose(top, full[:, 512:], atol=1e-5))
        left = fp8_block_linear(x[:, :256], W.cols(0, 256), out_dtype=torch.float32)
        right = fp8_block_linear(x[:, 256:], W.cols(256, 512), out_dtype=torch.float32)
        self.assertLess(float((left + right - full).norm() / full.norm()), 1e-5)


if __name__ == "__main__":
    unittest.main()
