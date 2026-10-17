"""engine/glm/index_kernel.py against the indexer's einsum path (needs a GPU)."""

import unittest

import torch
import torch.nn.functional as F

from engine.glm.index_kernel import index_scores
from engine.glm.tp import kv_span, kv_write

H, D = 32, 128


def reference(q, w, cache, positions, S):
    """GlmTP._indexer's scores with the kernel off."""
    K = kv_span(cache, S, torch.float32)
    sc = F.relu(torch.einsum("thd,sd->ths", q.float(), K.float()) * D ** -0.5)
    score = torch.einsum("th,ths->ts", w, sc)
    kpos = torch.arange(S, device=q.device)
    return score.masked_fill(kpos[None] > positions[:, None], float("-inf"))


def make_cache(S, n, fp8, g):
    keys = (torch.randn(n, D, generator=g, device="cuda") * 2).to(torch.bfloat16)
    if fp8:
        cache = (torch.zeros((S, D), dtype=torch.float8_e4m3fn, device="cuda"),
                 torch.zeros((S, 1), dtype=torch.float32, device="cuda"))
    else:
        cache = torch.zeros((S, D), dtype=torch.bfloat16, device="cuda")
    kv_write(cache, torch.arange(n, device="cuda"), keys)
    return cache


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class IndexKernelTests(unittest.TestCase):
    def check(self, T, S, n, fp8):
        g = torch.Generator(device="cuda").manual_seed(T * 131 + n)
        cache = make_cache(S, n, fp8, g)
        q = torch.randn(T, H, D, generator=g, device="cuda").to(torch.bfloat16)
        w = torch.randn(T, H, generator=g, device="cuda") * H ** -0.5
        positions = torch.arange(n - T, n, device="cuda")              # decode rows at the end
        want = reference(q, w, cache, positions, S)
        got = index_scores(q, w, cache, positions, S, D ** -0.5)
        fin = torch.isfinite(want)
        self.assertTrue(torch.equal(fin, torch.isfinite(got)), (T, S, n, fp8))
        err = float((got[fin] - want[fin]).abs().max() / want[fin].abs().max())
        self.assertLess(err, 1e-5, (T, S, n, fp8, err))
        k = min(2048, S)
        a, b = want.topk(k, -1).indices, got.topk(k, -1).indices
        for t in range(T):
            m = min(k, int(positions[t]) + 1)
            same = len(set(a[t, :m].tolist()) & set(b[t, :m].tolist())) / m
            self.assertGreater(same, 0.999, (T, S, n, fp8, t, same))

    def test_decode_rows(self):
        for fp8 in (True, False):
            for T in (1, 3, 16):
                self.check(T, 131072, 6000, fp8)
                self.check(T, 32768, 1500, fp8)       # fewer keys than the top-k

    def test_prefill_rows(self):
        for fp8 in (True, False):
            self.check(300, 4096, 4096, fp8)
            self.check(64, 20000, 20000, fp8)

    def test_split_rows(self):
        # MUSE_KV_SPLIT: rank r's row s holds position 2 s + r; its scores are the unsplit ones there.
        from engine.glm.kv_split import owner_rows
        for fp8 in (True, False):
            for T, n in ((1, 6000), (4, 6001), (300, 4096)):
                g = torch.Generator(device="cuda").manual_seed(T + n)
                S = 16384
                keys = (torch.randn(n, D, generator=g, device="cuda") * 2).to(torch.bfloat16)
                full = make_cache(S, 1, fp8, g)
                kv_write(full, torch.arange(n, device="cuda"), keys)
                q = torch.randn(T, H, D, generator=g, device="cuda").to(torch.bfloat16)
                w = torch.randn(T, H, generator=g, device="cuda") * H ** -0.5
                positions = torch.arange(n - T, n, device="cuda")
                want = index_scores(q, w, full, positions, S, D ** -0.5)
                for r in (0, 1):
                    loc = make_cache(S // 2 + 1, 1, fp8, g)
                    kv_write(loc, owner_rows(torch.arange(n, device="cuda"), r, 2, S // 2), keys)
                    got = index_scores(q, w, loc, positions, S // 2, D ** -0.5, 2, r)
                    self.assertTrue(torch.equal(got, want[:, r::2]), (fp8, T, n, r))


if __name__ == "__main__":
    unittest.main()
