"""Batched decode's kernel claims, checked rather than assumed (engine/batch_decode.py).

Batched multi-request decode stacks every lane's verify block into one call. It is only a faithful
batching if each row's output is the bits it would have been in its own single-request call:

* FP8 dense / grouped / fused-act-quant GEMMs on the 32- and 64-row decode tiles
  (decode_rows.wide) against the same rows in 6-row calls on the historical 16-row tile;
* the packed LM head's Gluon kernel at BM 32/64 against BM 16;
* the MoE router for 65..512 pairs against a stable sort, and the full up/down MoE over four lanes'
  pairs against four separate lanes;
* route_prep_masked: active rows exactly route_prep, idle rows on the null slot.

GPU only (sm_121a for the native MoE). Run in the serving image:
    python -m unittest tools.test_decode_rows -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import decode_rows  # noqa: E402

LANE = 6          # T_VERIFY at depth 5
LANES = (2, 4, 8, 10)


def _lanes(rows):
    return [slice(i, i + LANE) for i in range(0, rows, LANE)]


class DecodeRowsCPU(unittest.TestCase):
    def test_default_limit_and_tiles(self):
        self.assertEqual(decode_rows.limit(), 16)
        self.assertEqual([decode_rows.block_m(m) for m in (1, 6, 16, 17, 32, 33, 64)],
                         [16, 16, 16, 32, 32, 64, 64])
        with self.assertRaises(ValueError):
            decode_rows.block_m(65)

    def test_wide_is_scoped(self):
        with decode_rows.wide():
            self.assertEqual(decode_rows.limit(), 64)
            with decode_rows.wide(32):
                self.assertEqual(decode_rows.limit(), 32)
            self.assertEqual(decode_rows.limit(), 64)
        self.assertEqual(decode_rows.limit(), 16)
        with self.assertRaises(ValueError):
            with decode_rows.wide(128):
                pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeRowsCUDA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261014)

    def _weight(self, n, k):
        from fp8_linear import quantize_to_fp8
        return quantize_to_fp8(torch.randn(n, k, device="cuda") * 0.02)

    def _assert_rows(self, wide, per_lane, label):
        self.assertTrue(torch.equal(wide, per_lane),
                        f"{label}: max |diff| {float((wide.float() - per_lane.float()).abs().max())}")

    def test_fp8_linear_rows_match_single_lane_calls(self):
        from fp8_linear import fp8_linear
        # wq_b (TP), wq_a|wkv, wo_b local, shared w13 -- the shapes a batched step issues.
        for n, k in ((16384, 1280), (1792, 5120), (2560, 8192), (2304, 5120)):
            W = self._weight(n, k)
            for lanes in LANES:
                rows = lanes * LANE
                x = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.bfloat16)
                for act_qdq in (False, True):
                    for dtype in (torch.bfloat16, torch.float32):
                        ref = torch.cat([fp8_linear(x[s], W, out_dtype=dtype, act_qdq=act_qdq)
                                         for s in _lanes(rows)])
                        with decode_rows.wide():
                            got = fp8_linear(x, W, out_dtype=dtype, act_qdq=act_qdq)
                        torch.cuda.synchronize()
                        self._assert_rows(got, ref, f"fp8 N={n} K={k} rows={rows} qdq={act_qdq} {dtype}")

    def test_fp8_grouped_rows_match_single_lane_calls(self):
        from fp8_linear import FP8GroupedWeight, fp8_grouped_linear
        G, Rk, K = 4, 1024, 4096   # TP wo_a: 4 of 8 groups a rank
        flat = self._weight(G * Rk, K)
        W = FP8GroupedWeight(flat.w, flat.s, G, Rk)
        for lanes in LANES:
            rows = lanes * LANE
            x = (torch.randn(rows, G, K, device="cuda") * 0.5).to(torch.bfloat16)
            ref = torch.cat([fp8_grouped_linear(x[s], W) for s in _lanes(rows)])
            with decode_rows.wide():
                got = fp8_grouped_linear(x, W)
            torch.cuda.synchronize()
            self._assert_rows(got, ref, f"grouped rows={rows}")

    def test_dispatch_stays_on_prefill_shapes_outside_wide(self):
        """A 24-row call outside wide() is a prefill chunk and must keep its old arithmetic."""
        import v41_ref as R
        from fp8_linear import fp8_linear
        W = self._weight(1792, 5120)
        x = (torch.randn(24, 5120, device="cuda") * 0.5).to(torch.bfloat16)
        self.assertFalse(R._act_qdq_fusable(x, W))
        with decode_rows.wide():
            self.assertTrue(R._act_qdq_fusable(x, W))
            self._assert_rows(R.dense(x, W), fp8_linear(x, W), "dense wide dispatch")

    def test_packed_head_rows_match_single_lane_calls(self):
        from head_packed_gluon import project
        from engine.native_head import make_packed_head
        weight = (torch.randn(4160, 5120, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        packed = make_packed_head(weight)
        for lanes in LANES:
            rows = lanes * LANE
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            ref = torch.cat([packed.native_logits(x[s]) for s in _lanes(rows)])
            raw = torch.cat([project(x[s], packed, bn=32, raw=True) for s in _lanes(rows)])
            with decode_rows.wide():
                got = packed.native_logits(x)
                got_raw = project(x, packed, bn=32, raw=True)
            torch.cuda.synchronize()
            self._assert_rows(got, ref, f"head rows={rows}")
            self._assert_rows(got_raw, raw, f"head raw rows={rows}")


def _reference_routing(slots, topk):
    """Stable sort of the pairs by slot, then 16-pair blocks per slot run: route_small's contract."""
    flat = slots.reshape(-1).cpu()
    order = sorted(range(flat.numel()), key=lambda p: (int(flat[p]), p))
    P = flat.numel()
    block_slot = [-1] * P
    block_pair = [-1] * (P * 16)
    block, rank, previous = -1, 0, None
    for i, p in enumerate(order):
        value = int(flat[p])
        if i == 0 or value != previous or rank == 16:
            block, rank, previous = block + 1, 0, value
            block_slot[block] = value
        block_pair[block * 16 + rank] = p
        rank += 1
    return torch.tensor(block_slot, dtype=torch.int32), torch.tensor(block_pair, dtype=torch.int32)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 1),
                     "native FP4 CUDA kernel requires sm_121a")
class WideMoECUDA(unittest.TestCase):
    SLOTS = 12

    @classmethod
    def setUpClass(cls):
        import fp4_moe as K
        import fp4_moe_cuda
        cls.arena = K.ExpertArena(cls.SLOTS, "cuda")
        gen = torch.Generator().manual_seed(20261014)
        for tensor in (cls.arena.w1, cls.arena.w3, cls.arena.w2):
            tensor.copy_(torch.randint(0, 256, tensor.shape, dtype=torch.uint8, generator=gen))
        for tensor in (cls.arena.s1, cls.arena.s3, cls.arena.s2):
            tensor.copy_(torch.randint(124, 129, tensor.shape, dtype=torch.uint8, generator=gen))
        fp4_moe_cuda._lib()

    def test_wide_router_matches_stable_sort(self):
        import fp4_moe_cuda as CUDA
        gen = torch.Generator().manual_seed(7)
        for rows, topk in ((12, 6), (24, 6), (48, 6), (64, 6), (60, 6), (80, 6)):
            slots = torch.randint(0, 9, (rows, topk), dtype=torch.int32, generator=gen).cuda()
            if rows == 80:
                slots[:] = 3   # one 480-pair run: thirty 16-pair blocks of the same slot
            block_slot, block_pair, block_route, nb = CUDA.build_routing_small(slots, 16, topk)
            torch.cuda.synchronize()
            ref_slot, ref_pair = _reference_routing(slots, topk)
            self.assertEqual(nb, rows * topk)
            self.assertTrue(torch.equal(block_slot.cpu(), ref_slot), f"slots P={rows * topk}")
            self.assertTrue(torch.equal(block_pair.cpu(), ref_pair), f"pairs P={rows * topk}")
            token = torch.where(ref_pair >= 0, ref_pair // topk, -1)
            self.assertTrue(torch.equal((block_route >> 16).cpu(), token))
        with self.assertRaises(ValueError):
            CUDA.build_routing_small(torch.zeros(86, 6, dtype=torch.int32, device="cuda"), 16, 6)

    def test_small_router_matches_stable_sort(self):
        """route_small_bm16 (<= 64 pairs): the parallel rank and block cut against the same
        stable-sort reference, including negative keys (idle rows), runs past 16 and P = 1."""
        import fp4_moe_cuda as CUDA
        gen = torch.Generator().manual_seed(22)
        for rows, topk in ((1, 1), (1, 6), (4, 6), (5, 3), (6, 6), (10, 6), (16, 4), (32, 2), (64, 1)):
            for span in (2, 9, 400):
                for trial in range(4):
                    slots = torch.randint(-3, span, (rows, topk), dtype=torch.int32, generator=gen)
                    if trial == 3:
                        slots[:] = 5   # one run of P pairs: ceil(P / 16) blocks of one slot
                    slots = slots.cuda()
                    block_slot, block_pair, block_route, nb = CUDA.build_routing_small(slots, 16, topk)
                    torch.cuda.synchronize()
                    ref_slot, ref_pair = _reference_routing(slots, topk)
                    tag = f"P={rows * topk} topk={topk} span={span} trial={trial}"
                    self.assertEqual(nb, rows * topk)
                    self.assertTrue(torch.equal(block_slot.cpu(), ref_slot), tag)
                    self.assertTrue(torch.equal(block_pair.cpu(), ref_pair), tag)
                    route = torch.where(ref_pair >= 0, ref_pair | ((ref_pair // topk) << 16), -1)
                    self.assertTrue(torch.equal(block_route.cpu(), route), tag)

    def _moe(self, x, slots, weight, null):
        """FastDecoder._moe_merged's routed half on one GPU (world 1): fp32 parts summed over k."""
        import fp4_moe as K
        import fp4_moe_cuda as CUDA
        arena = self.arena
        rows, k = slots.shape
        P = rows * k
        inter, down_n = arena.w1.shape[1], arena.w2.shape[1]
        route = torch.where(slots.reshape(-1) == null,
                            null + 1 + torch.arange(P, device="cuda", dtype=torch.int32),
                            slots.reshape(-1)).view(rows, k).contiguous()
        block_slot, block_pair, block_route, nb = CUDA.build_routing_small(route, 16, k)
        block_slot = torch.where(block_slot > null, torch.full_like(block_slot, null), block_slot)
        h = torch.empty(P, inter, dtype=torch.bfloat16, device="cuda")
        CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, weight.reshape(-1).contiguous(),
                block_slot, block_route, 10.0, k, inter, K.DIM, nb, null)
        parts = torch.empty(P, down_n, dtype=torch.float32, device="cuda")
        CUDA.down(h, arena.w2, arena.s2, parts, block_slot, block_pair, k, down_n, inter,
                  rows, nb, null, False, 1)
        return parts.view(k, rows, down_n).sum(dim=0)

    def test_four_lanes_in_one_call_equal_four_calls(self):
        import fp4_moe as K
        gen = torch.Generator(device="cuda").manual_seed(11)
        null = self.SLOTS - 1
        for lanes in (2, 4, 8):
            rows = lanes * LANE
            x = torch.randn((rows, K.DIM), generator=gen, device="cuda").to(torch.bfloat16)
            slots = torch.randint(0, self.SLOTS, (rows, 6), generator=gen, device="cuda",
                                  dtype=torch.int32)
            weight = torch.rand((rows, 6), generator=gen, device="cuda")
            ref = torch.cat([self._moe(x[s].contiguous(), slots[s].contiguous(),
                                       weight[s].contiguous(), null) for s in _lanes(rows)])
            got = self._moe(x, slots, weight, null)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(got, ref),
                            f"lanes={lanes}: max |diff| {float((got - ref).abs().max())}")

    def test_masked_route_prep(self):
        from engine.decode_lean import LeanOps
        lean = LeanOps(torch.device("cuda"), 16, 32)
        null = 100
        lut = torch.randint(0, 50, (385,), dtype=torch.int32, device="cuda")
        lut[384] = null                                  # the top-p sentinel column
        idx = torch.randint(0, 385, (24, 6), device="cuda")
        slots_a = torch.empty(24, 6, dtype=torch.int32, device="cuda")
        slots_b = torch.empty_like(slots_a)
        route_a = lean.route_prep(idx, lut, slots_a, null)
        active = torch.ones(24, dtype=torch.uint8, device="cuda")
        route_b = lean.route_prep_masked(idx, lut, active, slots_b, null)
        self.assertTrue(torch.equal(route_a, route_b) and torch.equal(slots_a, slots_b))
        active[6:12] = 0
        active[18:] = 0
        route_c = lean.route_prep_masked(idx, lut, active, slots_b, null)
        idle = torch.zeros(24, dtype=torch.bool, device="cuda")
        idle[6:12] = True
        idle[18:] = True
        self.assertTrue(torch.all(slots_b[idle] == null))
        self.assertTrue(torch.equal(slots_b[~idle], slots_a[~idle]))
        self.assertTrue(torch.all(route_c[idle] == -1))
        self.assertTrue(torch.equal(route_c[~idle], route_a[~idle]))

    def test_idle_lanes_cost_no_expert_work_and_leave_live_rows_exact(self):
        """Key -1 pairs form slot -1 blocks both kernels skip; live rows keep their bits."""
        import fp4_moe as K
        import fp4_moe_cuda as CUDA
        null = self.SLOTS - 1
        gen = torch.Generator(device="cuda").manual_seed(5)
        rows, k = 24, 6
        x = torch.randn((rows, K.DIM), generator=gen, device="cuda").to(torch.bfloat16)
        slots = torch.randint(0, null, (rows, k), generator=gen, device="cuda", dtype=torch.int32)
        weight = torch.rand((rows, k), generator=gen, device="cuda")
        live = slice(6, 12)
        expected = self._moe(x[live].contiguous(), slots[live].contiguous(), weight[live].contiguous(), null)
        route = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
        route[live] = slots[live]
        block_slot, block_pair, block_route, nb = CUDA.build_routing_small(route, 16, k)
        self.assertEqual(int((block_slot == -1).sum()) >= (rows - 6) * k // 16, True)
        arena, P = self.arena, rows * k
        inter, down_n = arena.w1.shape[1], arena.w2.shape[1]
        h = torch.empty(P, inter, dtype=torch.bfloat16, device="cuda")
        CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, weight.reshape(-1).contiguous(),
                block_slot, block_route, 10.0, k, inter, K.DIM, nb, null)
        parts = torch.full((P, down_n), float("nan"), dtype=torch.float32, device="cuda")
        CUDA.down(h, arena.w2, arena.s2, parts, block_slot, block_pair, k, down_n, inter,
                  rows, nb, null, False, 1)
        got = parts.view(k, rows, down_n)[:, live].sum(dim=0)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(got, expected))
        # Idle rows were never written: the NaN fill survives, so no kernel touched them.
        self.assertTrue(torch.isnan(parts.view(k, rows, down_n)[:, :6]).all())


if __name__ == "__main__":
    unittest.main()
