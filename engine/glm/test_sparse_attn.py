"""engine/glm/sparse_attn.py against GlmTP.attention's einsum path and an fp32 reference (needs a GPU)."""

import unittest

import torch

from engine.glm.sparse_attn import sparse_mla
from engine.glm.tp import kv_read, kv_write

H, DC, DR, K = 32, 512, 64, 2048
SCALE = 256 ** -0.5


def einsum_path(q_lat, q_rope, C, R, sel, positions):
    """GlmTP.attention's sparse branch with the kernel off."""
    outs = []
    for t0 in range(0, q_lat.shape[0], 32):
        s = sel[t0:t0 + 32].long()
        cg, rg = kv_read(C, s), R[s]
        sc = (torch.einsum("thc,tkc->thk", q_lat[t0:t0 + 32], cg)
              + torch.einsum("thr,tkr->thk", q_rope[t0:t0 + 32], rg)).float() * SCALE
        valid = s <= positions[t0:t0 + 32, None]
        sc = sc.masked_fill(~valid[:, None], float("-inf"))
        p = torch.softmax(sc, -1).to(torch.bfloat16)
        outs.append(torch.einsum("thk,tkc->thc", p, cg))
    return torch.cat(outs)


def fp32_path(q_lat, q_rope, C, R, sel, positions):
    """The same attention in fp32 throughout, from the same stored (BF16 / FP8-dequantized) keys."""
    s = sel.long()
    cg, rg = kv_read(C, s).float(), R[s].float()
    sc = (torch.einsum("thc,tkc->thk", q_lat.float(), cg) + torch.einsum("thr,tkr->thk", q_rope.float(), rg)) * SCALE
    sc = sc.masked_fill(~(s <= positions[:, None])[:, None], float("-inf"))
    return torch.einsum("thk,tkc->thc", torch.softmax(sc, -1), cg)


def make(T, S, fp8, g, ctx):
    lat = torch.randn(S, DC, generator=g, device="cuda").to(torch.bfloat16)
    if fp8:
        C = (torch.zeros((S, DC), dtype=torch.float8_e4m3fn, device="cuda"),
             torch.zeros((S, DC // 128), dtype=torch.float32, device="cuda"))
    else:
        C = torch.zeros((S, DC), dtype=torch.bfloat16, device="cuda")
    kv_write(C, torch.arange(S, device="cuda"), lat)
    R = torch.randn(S, DR, generator=g, device="cuda").to(torch.bfloat16)
    q_lat = (torch.randn(T, H, DC, generator=g, device="cuda") * 2).to(torch.bfloat16)
    q_rope = (torch.randn(T, H, DR, generator=g, device="cuda") * 2).to(torch.bfloat16)
    positions = torch.arange(ctx - T, ctx, device="cuda")
    # The indexer's picks: K distinct positions of [0, S), most at or before the row, some past it.
    sel = torch.stack([torch.randperm(min(S, ctx + 64), generator=torch.Generator().manual_seed(int(p)))[:K]
                       for p in positions.tolist()]).to("cuda")
    return q_lat, q_rope, C, R, sel, positions


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class SparseAttnTests(unittest.TestCase):
    def check(self, T, S, ctx, fp8, **kw):
        g = torch.Generator(device="cuda").manual_seed(T * 7 + ctx)
        q_lat, q_rope, C, R, sel, positions = make(T, S, fp8, g, ctx)
        ref = fp32_path(q_lat, q_rope, C, R, sel, positions)
        old = einsum_path(q_lat, q_rope, C, R, sel, positions).float()
        new = sparse_mla(q_lat, q_rope, C, R, sel, positions, SCALE, **kw).float()
        e_old = float((old - ref).norm() / ref.norm())
        e_new = float((new - ref).norm() / ref.norm())
        self.assertTrue(torch.isfinite(new).all())
        # Closer to fp32 than the einsum path is (its scores are rounded to BF16 twice).
        self.assertLess(e_new, max(1.2 * e_old, 2e-3), (T, S, ctx, fp8, e_new, e_old))
        return e_new, e_old

    def test_decode_rows(self):
        for fp8 in (True, False):
            self.check(4, 131072, 6000, fp8)

    def test_prefill_rows(self):
        for fp8 in (True, False):
            self.check(96, 6144, 6144, fp8)

    def test_block_shapes(self):
        for bh, bn, w in ((32, 32, 8), (16, 64, 4), (8, 32, 4)):
            self.check(8, 8192, 8192, True, block_h=bh, block_n=bn, num_warps=w)


if __name__ == "__main__":
    unittest.main()
