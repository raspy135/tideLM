"""Decode-time expert adaptation: the preview plan and the trigger (engine/v41_engine.py).

    python3 -m unittest engine.test_decode_adapt

CPU only; a fake engine carries just what plan_swaps / _decode_adapt_due read. The lockstep
path (control() flag -> maintain_decode on both ranks) needs the two-node gate.
"""
from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
import torch

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")]
import engine.v41_engine as V  # noqa: E402
from engine.adapt_config import resolve  # noqa: E402

E, L = 16, 2


def fake_engine(db, req, keep):
    model = types.SimpleNamespace(
        prune_miss_report=lambda: ({}, torch.tensor(db, dtype=torch.float64),
                                  torch.tensor(db, dtype=torch.float64)),
        _req_counts=torch.tensor(req, dtype=torch.float64),
        _req_mass=torch.tensor(req, dtype=torch.float64))
    return types.SimpleNamespace(
        _prune_trace={l: np.ones(E) for l in range(L)}, model=model,
        model_prune_mask={l: torch.tensor(keep) for l in range(L)},
        ep=types.SimpleNamespace(tensor_parallel=True, world=2, rank=0))


class DecodeAdaptTest(unittest.TestCase):
    def setUp(self):
        self._adapt = V.ADAPT
        V.ADAPT = resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_ADAPT_PRIOR": "4"})

    def tearDown(self):
        V.ADAPT = self._adapt

    def test_preview_follows_the_current_request_without_writing(self):
        keep = [True] * 8 + [False] * 8               # experts 0-7 resident
        # History: residents were only weakly wanted (0.4 votes per layer in total). A single
        # request is one vote, so it can only move experts the history does not firmly hold.
        db = np.zeros((L, E)); db[:, :8] = 0.05
        req = np.zeros((L, E)); req[:, 12] = 500.0     # this answer keeps asking for expert 12
        req[:, 0] = 1.0
        eng = fake_engine(db, req, keep)
        before = [s for s in V.V41Engine.plan_swaps(eng, max_swaps=64)]
        preview = V.V41Engine.plan_swaps(eng, max_swaps=64, pending_request=True)
        self.assertFalse(any(e_in == 12 for _, _, e_in, _ in before))
        self.assertTrue(any(e_in == 12 for _, _, e_in, _ in preview))
        # nothing written: the same history and accumulator as before
        self.assertTrue(np.array_equal(eng.model.prune_miss_report()[1].numpy(), db))
        self.assertTrue(np.array_equal(eng.model._req_counts.numpy(), req))

    def test_one_vote_cannot_override_firm_history(self):
        keep = [True] * 8 + [False] * 8
        db = np.zeros((L, E)); db[:, :8] = 2.0         # 16 votes firmly behind the residents
        req = np.zeros((L, E)); req[:, 12] = 500.0
        eng = fake_engine(db, req, keep)
        self.assertFalse(any(e_in == 12 for _, _, e_in, _ in
                             V.V41Engine.plan_swaps(eng, max_swaps=64, pending_request=True)))

    def test_trigger_needs_interval_and_misses(self):
        snaps = iter([(0, 0), (5, 1000), (60, 2000), (60, 3000)])   # (missed, total) cumulative
        eng = types.SimpleNamespace(model=types.SimpleNamespace(miss_snapshot=lambda: next(snaps)))
        eng._decode_adapt_mark = (1, eng.model.miss_snapshot())
        due = lambda n: V.V41Engine._decode_adapt_due(eng, n)   # noqa: E731
        self.assertEqual(due(300), 0)                   # < 600 tokens: not even checked
        self.assertEqual(due(601), 0)                   # 5/1000 = 0.5% < 1% gate at high
        self.assertEqual(eng._decode_adapt_mark[0], 601)   # the window moved on anyway
        self.assertEqual(due(1201), 1)                  # 55/1000 = 5.5% -> pass
        self.assertAlmostEqual(eng._decode_adapt_rate, 0.055)
        self.assertEqual(due(1801), 0)                  # 0/1000 since then

    def test_off_and_legacy_never_trigger(self):
        for env in ({"DSV41_ADAPT_SENSITIVITY": "off"}, {},
                    {"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_PRUNE_SWAP": "0"}):
            self.assertEqual(resolve(env).decode_tokens, 0, env)
        self.assertEqual(resolve({"DSV41_ADAPT_SENSITIVITY": "medium"}).decode_tokens, 600)
        self.assertEqual(resolve({"DSV41_ADAPT_SENSITIVITY": "high",
                                  "DSV41_ADAPT_DECODE_TOKENS": "0"}).decode_tokens, 0)
        self.assertEqual(resolve({"DSV41_ADAPT_SENSITIVITY": "high",
                                  "DSV41_ADAPT_DECODE_TOKENS": "1200"}).decode_tokens, 1200)
        with self.assertRaises(ValueError):
            resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_ADAPT_DECODE_TOKENS": "-1"})

    def test_decode_planner_ignores_exhausted_streaming_budget(self):
        from engine.user_prompt import UserPrompt
        keep=[True]*8+[False]*8
        db=np.zeros((L,E));db[:,:8]=.05
        req=np.zeros((L,E));req[:,12]=500.;req[:,0]=1.
        eng=fake_engine(db,req,keep)
        eng.user_prompt=UserPrompt(True,L,E,'cpu',100)
        eng.user_prompt.consume(100)
        eng.ep.control_flag=1
        eng.ep.broadcast_obj=lambda p:p
        eng.plan_swaps=lambda **kw:V.V41Engine.plan_swaps(eng,**kw)
        applied=[]
        eng.apply_swaps=lambda p:applied.extend(p) or len(p)
        eng._decode_adapt_passes=[];eng._decode_adapt_window=30
        self.assertGreater(V.V41Engine.maintain_decode(eng,31),0)
        self.assertTrue(any(s[2]==12 for s in applied))
        self.assertEqual(eng._decode_adapt_passes[0]['reason'],'periodic')
        self.assertEqual(eng.user_prompt.remaining_loads,0)

    def test_eager_uses_aggregate_delta_not_unused_decode_counter(self):
        eng=types.SimpleNamespace(fast=None,
            model=types.SimpleNamespace(miss_snapshot=lambda:(20.,100.),
                                        decode_miss_snapshot=lambda:(0.,0.)),
            _decode_adapt_mark=(1,(0.,0.)))
        self.assertEqual(V.V41Engine._decode_adapt_due(eng,601),1)


if __name__ == "__main__":
    unittest.main()
