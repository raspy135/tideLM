"""engine/muse/attn_decode.py against a dense masked softmax (needs a GPU)."""

import unittest

import torch

from engine.muse.attn_decode import decode_attention


def dense(q, K, V, positions, kpos, sliding, window, scale):
    G = q.shape[1] // K.shape[1]
    Kr, Vr = K.repeat_interleave(G, 1).float(), V.repeat_interleave(G, 1).float()
    s = torch.einsum("thd,shd->hts", q.float(), Kr) * scale
    vis = kpos[None, :] <= positions[:, None]
    if sliding:
        vis &= kpos[None, :] > positions[:, None] - window
    s = s.masked_fill(~vis[None], float("-inf"))
    return torch.einsum("hts,shd->thd", torch.softmax(s, -1), Vr)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class DecodeAttentionTests(unittest.TestCase):
    def run_case(self, T, pos, sliding, window=64, ring=None, n=None):
        g = torch.Generator(device="cuda").manual_seed(T * 1000 + pos)
        Hq, Hkv, D = 64, 8, 64
        ring = ring or (window + 32)
        n = ring if sliding else (n or pos + T + 5)
        K = torch.randn(n, Hkv, D, device="cuda", generator=g).to(torch.bfloat16)
        V = torch.randn(n, Hkv, D, device="cuda", generator=g).to(torch.bfloat16)
        q = torch.randn(T, Hq, D, device="cuda", generator=g).to(torch.bfloat16)
        positions = torch.arange(pos, pos + T, device="cuda")
        if sliding:
            lo = max(0, pos + 1 - window)          # the first row's window start
            kpos = torch.arange(lo, pos + T, device="cuda")
            Kc, Vc = K[kpos % ring], V[kpos % ring]
        else:
            kpos = torch.arange(0, pos + T, device="cuda")
            Kc, Vc = K[:pos + T], V[:pos + T]
        want = dense(q, Kc, Vc, positions, kpos, sliding, window, 0.125)
        got = decode_attention(q, K, V, torch.tensor([pos], device="cuda"), sliding=sliding,
                               ring=ring, window=window, scale=0.125)
        err = float((got.float() - want).abs().max())
        self.assertLess(err, 2e-2, (T, pos, sliding, err))

    def test_full_layers(self):
        for T in (1, 2, 6, 8):
            for pos in (0, 5, 63, 64, 1000, 4097):
                self.run_case(T, pos, sliding=False)

    def test_sliding_ring_wraps(self):
        for T in (1, 6):
            for pos in (0, 10, 63, 64, 95, 96, 500, 1234):
                self.run_case(T, pos, sliding=True, window=64, ring=96)


if __name__ == "__main__":
    unittest.main()
