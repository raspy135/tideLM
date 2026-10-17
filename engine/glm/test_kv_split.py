"""MUSE_KV_SPLIT on one GPU: GlmTP._split_attn and the prefill gather, run for both ranks in two
threads over a fake all-gather, against the unsplit indexer + attention (needs a GPU)."""

import threading
import types
import unittest

import torch

from engine.glm.index_kernel import index_scores
from engine.glm.kv_split import gather_spans, owner_rows
from engine.glm.sparse_attn import sparse_mla
from engine.glm.tp import GlmTP, kv_write

H, HI, DC, DR, DI, K = 64, 32, 512, 64, 128, 2048
SCALE = 256 ** -0.5


class PairComm:
    """all_gather across two threads: each puts its tensor, both read both."""

    def __init__(self):
        self.box, self.bar = [None, None], threading.Barrier(2)

    def view(self, rank):
        outer = self

        class C:
            def all_gather(self, t):
                outer.box[rank] = t.contiguous().clone()
                outer.bar.wait()
                out = torch.stack(outer.box)
                outer.bar.wait()
                return out
        return C()


def run_pair(fn):
    comm, out, err = PairComm(), [None, None], []

    def go(r):
        try:
            out[r] = fn(r, comm.view(r))
            torch.cuda.synchronize()
        except Exception as e:          # noqa: BLE001
            err.append(e)
            comm.bar.abort()
    th = [threading.Thread(target=go, args=(r,)) for r in (0, 1)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if err:
        raise err[0]
    return out


def caches(rows, fp8):
    f8, dev = torch.float8_e4m3fn, "cuda"
    if fp8:
        c = (torch.zeros((rows, DC), dtype=f8, device=dev), torch.zeros((rows, DC // 128), device=dev))
        i = (torch.zeros((rows, DI), dtype=f8, device=dev), torch.zeros((rows, 1), device=dev))
    else:
        c = torch.zeros((rows, DC), dtype=torch.bfloat16, device=dev)
        i = torch.zeros((rows, DI), dtype=torch.bfloat16, device=dev)
    return c, torch.zeros((rows, DR), dtype=torch.bfloat16, device=dev), i


def fill(S, n, fp8, g):
    """Unsplit caches for S positions (n written) and each rank's split ones (S / 2 + 1 rows)."""
    lat = torch.randn(n, DC, generator=g, device="cuda").to(torch.bfloat16)
    rope = torch.randn(n, DR, generator=g, device="cuda").to(torch.bfloat16)
    ik = (torch.randn(n, DI, generator=g, device="cuda") * 2).to(torch.bfloat16)
    pos = torch.arange(n, device="cuda")
    full = caches(S, fp8)
    kv_write(full[0], pos, lat), full[1].index_copy_(0, pos, rope), kv_write(full[2], pos, ik)
    loc = []
    for r in (0, 1):
        c = caches(S // 2 + 1, fp8)
        rows = owner_rows(pos, r, 2, S // 2)
        kv_write(c[0], rows, lat), c[1].index_copy_(0, rows, rope), kv_write(c[2], rows, ik)
        loc.append(c)
    return full, loc


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class KvSplitTests(unittest.TestCase):
    def decode_case(self, T, S, n, fp8):
        g = torch.Generator(device="cuda").manual_seed(T * 31 + n)
        full, loc = fill(S, n, fp8, g)
        q_lat = (torch.randn(T, H, DC, generator=g, device="cuda") * 2).to(torch.bfloat16)
        q_rope = (torch.randn(T, H, DR, generator=g, device="cuda") * 2).to(torch.bfloat16)
        iq = torch.randn(T, HI, DI, generator=g, device="cuda").to(torch.bfloat16)
        iw = torch.randn(T, HI, generator=g, device="cuda") * HI ** -0.5
        positions = torch.arange(n - T, n, device="cuda")
        # Unsplit: the indexer's top-k over every position, every head attended over it.
        sel = index_scores(iq, iw, full[2], positions, S, DI ** -0.5).topk(min(K, S), -1).indices
        want = sparse_mla(q_lat, q_rope, full[0], full[1], sel, positions, SCALE).float()
        cfg = types.SimpleNamespace(kv_lora=DC, index_topk=K, index_dim=DI)

        def rank(r, comm):
            me = types.SimpleNamespace(cfg=cfg, rank=r, world=2, hl=H // 2, s_loc=S // 2, comm=comm,
                                       i_cache={0: loc[r][2]}, c_cache={0: loc[r][0], 1: loc[r][0]},
                                       r_cache={0: loc[r][1], 1: loc[r][1]})
            hs = slice(r * H // 2, (r + 1) * H // 2)
            o0, tk = GlmTP._split_attn(me, 0, (iq, iw), q_lat[:, hs].contiguous(), q_rope[:, hs].contiguous(),
                                       positions, None, SCALE)
            # A layer sharing the indexer reuses the picks.
            o1, _ = GlmTP._split_attn(me, 1, None, q_lat[:, hs].contiguous(), q_rope[:, hs].contiguous(),
                                      positions, tk, SCALE)
            return o0, o1, tk
        (a0, a1, tk0), (b0, b1, tk1) = run_pair(rank)
        # The two ranks' shares partition the global top-k: their positions are the unsplit picks.
        for t in range(T):
            m = min(K, int(positions[t]) + 1)
            mine = [set((tk[0][t, :int(tk[1][t])] * 2 + r).tolist()) for r, tk in ((0, tk0), (1, tk1))]
            got = {p for p in mine[0] | mine[1] if p <= int(positions[t])}
            ref = {p for p in sel[t].tolist() if p <= int(positions[t])}
            self.assertEqual(len(got), m)
            self.assertGreater(len(got & ref) / m, 0.999, (T, S, n, t))
        for got in (torch.cat([a0, b0], 1).float(), torch.cat([a1, b1], 1).float()):
            err = float((got - want).norm() / want.norm())
            self.assertLess(err, 4e-3, (T, S, n, fp8, err))

    def test_decode(self):
        for fp8 in (True, False):
            for T in (1, 4, 8):
                self.decode_case(T, 16384, 6000, fp8)
            self.decode_case(4, 16384, 900, fp8)            # fewer positions than the top-k
            self.decode_case(3, 16384, 2049, fp8)           # odd count, one past the top-k

    def test_gather_rows_pieces(self):
        # Over the slot, the rows go in pieces; the result is the one-gather result.
        from engine.glm.kv_split import gather_rows
        t = [torch.randint(0, 255, (8, 45056), dtype=torch.uint8, device="cuda") for _ in (0, 1)]

        def rank(r, comm):
            comm.fast = True
            return gather_rows(comm, t[r]), gather_rows(comm, t[r][:5])
        (a8, a5), (b8, b5) = run_pair(rank)
        want = torch.stack(t)
        for got in (a8, b8):
            self.assertTrue(torch.equal(got, want))
        for got in (a5, b5):
            self.assertTrue(torch.equal(got, want[:, :5]))

    def test_gather_spans(self):
        for fp8 in (True, False):
            S, n = 8192, 5001
            g = torch.Generator(device="cuda").manual_seed(n)
            full, loc = fill(S, n, fp8, g)
            parts = lambda c: [*(c[0] if fp8 else (c[0],)), c[1], *(c[2] if fp8 else (c[2],))]
            got = run_pair(lambda r, comm: gather_spans(comm, parts(loc[r]), -(-n // 2), 2))
            for r in (0, 1):
                for a, b in zip(got[r], parts(full)):
                    self.assertTrue(torch.equal(a[:n].view(torch.uint8), b[:n].view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
