"""Lossless scale storage qualification. Run with DSV41_RUN_CUDA_TESTS=1.

Real-weight checks also require DSV41_SCALE_TEST_MODEL and DSV41_SCALE_TEST_PACK
(the six-expert screen pack made by pack_expert_scales.py).
"""
import json
import os
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@unittest.skipUnless(os.environ.get('DSV41_RUN_CUDA_TESTS') == '1', 'explicit GPU qualification')
class ScaleCudaTest(unittest.TestCase):
    def test_all_bytes_exceptions_and_changed_graph_inputs(self):
        from engine.scale_layout import encode, layout
        from engine.scale_load_cuda import expand
        import fp4_moe_cuda as cuda
        values = np.arange(1027, dtype=np.uint8)
        planes = [encode(values), encode(values[::-1])]
        capacity = max(len(p.exceptions) for p in planes)
        spec = layout(len(values), capacity)
        host = torch.from_numpy(np.stack([p.pack(capacity) for p in planes]))
        packed = host.cuda()
        expected = torch.from_numpy(np.stack([values, values[::-1].copy()])).cuda()
        self.assertTrue(torch.equal(expand(packed, len(values)), expected))
        output = torch.empty_like(expected)

        def native():
            cuda._check(cuda._lib().fp4_moe_cuda_scale_expand(cuda._stream(), cuda._p(packed), cuda._p(output),
                        len(values), spec['stride'], 2), 'scale expansion')
        native()
        self.assertTrue(torch.equal(output, expected))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            expand(packed, len(values))
            native()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = expand(packed, len(values))
            native()
        packed.copy_(host.flip(0))
        output.fill_(77)
        graph.replay()
        self.assertTrue(torch.equal(captured, expected.flip(0)))
        self.assertTrue(torch.equal(output, expected.flip(0)))

    @unittest.skipUnless(os.environ.get('DSV41_SCALE_TEST_MODEL') and os.environ.get('DSV41_SCALE_TEST_PACK'),
                         'real checkpoint and screen pack required')
    def test_real_moe_decode_prefill_and_slot_replacement(self):
        from engine.scale_pack import ScalePack
        import fp4_moe as moe
        import fp4_moe_cuda as cuda
        directory = Path(os.environ['DSV41_SCALE_TEST_MODEL'])
        mapping = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
        headers = {}

        def read(name):
            path = directory/mapping[name]
            if path not in headers:
                with path.open('rb') as handle:
                    length = struct.unpack('<Q', handle.read(8))[0]
                    headers[path] = 8+length, json.loads(handle.read(length))
            base, header = headers[path]
            meta = header[name]
            first, last = meta['data_offsets']
            with path.open('rb') as handle:
                handle.seek(base+first)
                data = bytearray(handle.read(last-first))
            return torch.frombuffer(data, dtype=torch.uint8).reshape(meta['shape']).pin_memory()

        def bytes_equal(a, b):
            self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

        torch.manual_seed(991)
        for rank in (0, 1):
            pack = ScalePack(os.environ['DSV41_SCALE_TEST_PACK'].format(rank=rank), directory, rank, 2)
            native = moe.ExpertArena(3, tp_rank=rank, tp_world=2)
            packed = moe.ExpertArena(3, tp_rank=rank, tp_world=2, scale_pack=pack)
            self.assertLess(packed.bytes_per_slot, native.bytes_per_slot)
            # Reuse slots, including scales whose exception payload differs.
            for layer, expert, slot in ((0, 0, 0), (20, 1, 1), (39, 0, 0), (0, 1, 2)):
                prefix = f'layers.{layer}.ffn.experts.{expert}.'
                tensors = [read(prefix+name) for name in ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale',
                                                         'w3.weight', 'w3.scale')]
                native.load_slot(slot, *tensors)
                packed.load_slot(slot, *tensors, scale_key=(layer, expert))
            for tokens in (1, 4, 6, 17, 65):
                topk = 2
                pairs = tokens*topk
                slots = (torch.arange(pairs, device='cuda').reshape(tokens, topk) % 3).int()
                route = torch.rand(tokens, topk, device='cuda')
                x = torch.randn(tokens, moe.DIM, device='cuda', dtype=torch.bfloat16)
                bm = 16 if pairs <= 64 else 64
                bs, bp, br, count = cuda.build_routing_small(slots, bm, topk) if bm == 16 else (None,)*4
                if bm != 16:
                    bs, bp, count = moe.build_routing(slots, 3, bm)
                # Native CUDA v2/v3 decode, plus both Triton prefill arithmetic arms.
                arms = ('cuda',) if tokens <= 6 else ('triton', 'scaled')
                for arm in arms:
                    intermediate, parts = [], []
                    for arena in (native, packed):
                        h = torch.empty((pairs, arena.inter), dtype=torch.bfloat16, device='cuda')
                        scale_stride = pack.stride if arena is packed else 0
                        if arm == 'cuda':
                            cuda.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, route.reshape(-1),
                                    bs, br, 10., topk, arena.inter, moe.DIM, count, -1, v2=True)
                        else:
                            moe._moe_up_kernel[(count, arena.inter//128)](x, arena.w1, arena.s1, arena.w3, arena.s3,
                                h, route.reshape(-1), bs, bp, x.stride(0), h.stride(0), 10., TOPK=topk,
                                N=arena.inter, K=moe.DIM, BM=bm, BN=128, SCALED=arm=='scaled',
                                SCALE_STRIDE=scale_stride, num_warps=4)
                        intermediate.append(h)
                    bytes_equal(*intermediate)
                    full = torch.cat([intermediate[0], intermediate[0]], -1).contiguous()
                    for rows in ((1, 2, 4) if arm == 'cuda' else (1,)):
                        parts.clear()
                        for arena in (native, packed):
                            out = torch.empty((pairs, arena.w2.shape[1]), dtype=torch.float32, device='cuda')
                            if arm == 'cuda':
                                cuda.down(full, arena.w2, arena.s2, out, bs, bp, topk, out.shape[1], moe.INTER,
                                          tokens, count, -1, False, 1, v2=True, rows=rows)
                            else:
                                moe._moe_down_kernel[(count, out.shape[1]//128)](full, arena.w2, arena.s2, out,
                                    bs, bp, full.stride(0), out.stride(0), TOPK=topk, N=out.shape[1], K=moe.INTER,
                                    BM=bm, BN=128, NTOK=tokens, SCALED=arm=='scaled',
                                    SCALE_STRIDE=pack.stride if arena is packed else 0, num_warps=4)
                            parts.append(out)
                        bytes_equal(*parts)
            os.close(pack.fd)


if __name__ == '__main__':
    unittest.main()
