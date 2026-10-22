"""Explicit GPU qualification for the opt-in decode experiments; never loads the model.

DSV41_RUN_CUDA_TESTS=1 PYTHONPATH=tools:. python -m unittest tools.test_perf_decode_cuda
"""
import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('DSV41_RUN_CUDA_TESTS') == '1', 'GPU experiments explicitly paused')
class DecodeCudaTest(unittest.TestCase):
    def test_packed_attention_matches_staging_and_handles_empty_rows(self):
        from engine.attention_packed import attention_packed
        from engine.decode_lean import LeanOps
        from engine.packed_kv import write
        from decode_attn_staged import attention
        gen = torch.Generator(device='cuda').manual_seed(61)
        lean = LeanOps('cuda', 16, 32)
        for t, selected in ((1, 0), (4, 512), (6, 1024)):
            q = torch.randn(t, 32, 512, device='cuda', generator=gen).bfloat16().float()
            ring = torch.randn(160, 512, device='cuda', generator=gen).bfloat16()
            slots = torch.randint(0, 160, (t, 128), device='cuda', generator=gen)
            cache = torch.zeros(2048, 36, dtype=torch.int64, device='cuda')
            write(cache, torch.randn(2048, 512, device='cuda', generator=gen).bfloat16(), 0)
            ids = torch.randint(0, 2048, (t, selected), device='cuda', generator=gen) if selected else None
            mask = torch.rand(t, 128 + selected, device='cuda', generator=gen) > .2
            mask[0] = False
            sink = torch.randn(32, device='cuda', generator=gen)
            for _ in range(3):
                attention_packed(q, ring, slots, cache, ids, mask, sink, 512 ** -.5)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = attention_packed(q, ring, slots, cache, ids, mask, sink, 512 ** -.5)
            for step in (0, 1):
                if step:
                    q.mul_(.5)
                    slots.fill_(3)
                    mask[0] = True
                graph.replay()
                keys = lean.keys_f32(ring, slots, cache if selected else None, ids, dtype=torch.bfloat16)
                expected = attention(q, keys, mask, sink, 512 ** -.5, lean)
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertLess(float(relative), 2e-6)
                if not step:
                    self.assertTrue((actual[0] == 0).all())

    def test_candidate_scores_and_changed_graph_inputs(self):
        from engine.index_candidates import score_reference, scores, select
        gen = torch.Generator(device='cuda').manual_seed(41)
        for t in (1, 4, 6, 10):
            q = torch.randn(t, 32, 128, generator=gen, device='cuda').bfloat16()
            k = torch.randn(4096, 128, generator=gen, device='cuda').bfloat16()
            w = torch.randn(t, 32, generator=gen, device='cuda') / 32
            ids = torch.randint(0, 4096, (t, 1024), generator=gen, device='cuda')
            lengths = torch.full((t,), 3072, device='cuda', dtype=torch.long)
            ids[:, -8:] = -1
            for _ in range(3):
                scores(q, k, w, ids, lengths)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = scores(q, k, w, ids, lengths)
            for step in (0, 1):
                if step:
                    ids[:, :16] = -1
                    lengths.fill_(511)
                    k.mul_(.75)
                graph.replay()
                expected = score_reference(q, k, w, ids, lengths)
                self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
                live = torch.isfinite(expected)
                # A different head-sum tree is permitted, not a different BF16 dot.
                torch.testing.assert_close(actual[live], expected[live], rtol=2e-5, atol=2e-6)
                selected = select(actual, ids, 512)
                self.assertTrue(((selected < lengths[:, None]) | (selected == -1)).all())

    def test_post_pre_in_place_and_graph_replay(self):
        from engine.hc_ops import hc_post, hc_pre_rmsnorm
        from engine.hc_post_pre import post_pre
        gen = torch.Generator(device='cuda').manual_seed(51)
        for t in (1, 4, 5, 6, 10):
            x = torch.randn(t, 5120, device='cuda', generator=gen).bfloat16()
            res = torch.randn(t, 4, 5120, device='cuda', generator=gen).bfloat16()
            post = torch.rand(t, 4, device='cuda', generator=gen)
            comb = torch.rand(t, 4, 4, device='cuda', generator=gen)
            pre = torch.rand(t, 4, device='cuda', generator=gen)
            weight = torch.rand(5120, device='cuda', generator=gen)
            original = res.clone()
            expected_h = hc_post(x, original, post, comb)
            expected_y = hc_pre_rmsnorm(expected_h, pre, weight, 1e-20)
            for _ in range(3):
                post_pre(x, original, post, comb, pre, weight, 1e-20)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual_h, actual_y = post_pre(x, res, post, comb, pre, weight, 1e-20, out=res)
            res.copy_(original)
            graph.replay()
            self.assertTrue(torch.equal(actual_h, expected_h), t)
            self.assertTrue(torch.equal(actual_y, expected_y), t)
            # Reuse every buffer with different input; stale/aliased scratch must not survive.
            x.mul_(.5)
            expected_h = hc_post(x, original, post, comb)
            expected_y = hc_pre_rmsnorm(expected_h, pre, weight, 1e-20)
            res.copy_(original)
            graph.replay()
            self.assertTrue(torch.equal(actual_h, expected_h), t)
            self.assertTrue(torch.equal(actual_y, expected_y), t)

    def test_draft_router_small_gate_does_not_alias_target_gate(self):
        from engine.decode_lean import LeanOps
        from v41_ref import mm
        ops = LeanOps('cuda', 16, 32)
        ops.router_bf16 = False
        y = torch.randn(5, 5120, device='cuda').bfloat16()
        for experts, count in ((384, 6), (128, 3), (384, 6)):
            gate = torch.randn(experts, 5120, device='cuda') * .01
            bias = torch.randn(experts, device='cuda') * .1
            idx = torch.empty(5, count, dtype=torch.long, device='cuda')
            wt = torch.empty(5, count, device='cuda')
            ops.router(y, gate, bias, None, count, 1.5, idx, wt)
            values = torch.nn.functional.softplus(mm(y.float(), gate)).sqrt()
            expected_ids = (values + bias).topk(count, dim=-1).indices
            self.assertTrue(torch.equal(idx, expected_ids))
            expected = values.gather(1, expected_ids)
            expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 1.5
            self.assertTrue(torch.equal(wt, expected))

    def test_engram_side_stream_uses_real_staging_and_preserves_previous_consumers(self):
        from concurrent.futures import Future
        import numpy as np
        from engine.engram import EngramTable
        from engine.fastdecode import FastDecoder
        from engine.perf_config import DecodeExperiments
        decoder = object.__new__(FastDecoder)
        decoder.dev = torch.device('cuda')
        decoder.experiments = DecodeExperiments(engram_h2d=True)
        decoder.engram_upload_stream = torch.cuda.Stream()
        decoder.gpu_timing = False
        decoder.eg_rows = {1: torch.zeros(2, 24, 256, device='cuda')}
        table = object.__new__(EngramTable)
        table.device, table.pinned, table.row_split = 'cuda', True, False
        table._cur = 0
        table._stage, table._inv_stage, table._ev = [None, None], [None, None], [None, None]
        table.stats = {'seconds': 0.0}
        rng = np.random.default_rng(73)
        previous = torch.zeros_like(decoder.eg_rows[1])
        for _ in range(8):
            raw = np.zeros((48, 264), np.uint8)
            raw[:, :256] = rng.integers(0, 64, (48, 256), dtype=np.uint8)
            raw[:, 256:] = 127
            inverse = np.arange(48, dtype=np.int64)
            future = Future()
            future.set_result((raw, inverse, (2, 24)))
            consumed = decoder.eg_rows[1].clone()
            decoder._upload_engram(1, future, table.to_device)
            current = decoder.eg_rows[1].clone()
            expected = torch.from_numpy(raw[:, :256].copy()).to('cuda').view(torch.float8_e4m3fn).float().reshape(2, 24, 256)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(consumed, previous))
            self.assertTrue(torch.equal(current, expected))
            previous = current



if __name__ == '__main__':
    unittest.main()
