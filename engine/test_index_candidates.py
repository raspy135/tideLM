"""Candidate causality and source-block membership without loading a model."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from engine.index_candidates import compact_mask, score_reference, select


class CandidatesTest(unittest.TestCase):
    def test_real_indexer_hands_source_candidates_to_later_layer(self):
        from engine.fastdecode import FastDecoder
        from engine.perf_config import DecodeExperiments
        args = SimpleNamespace(index_topk=4, index_n_heads=1, index_head_dim=1,
            candidate_source_layer=20, candidate_topk_blocks=2, candidate_block_size=8)
        results = []
        for enabled in (False, True):
            decoder = SimpleNamespace(a=args,
                experiments=DecodeExperiments(index_candidates=enabled),
                candidate_ids=torch.full((2, 16), -1, dtype=torch.long),
                candidates=torch.zeros(2, 129, dtype=torch.bool),
                W=SimpleNamespace(indexers={layer: SimpleNamespace(wq_b=None, weights_proj=None)
                                           for layer in (20, 24)}),
                m=SimpleNamespace(freqs_c=torch.zeros(2)), c=SimpleNamespace(max_seq=129),
                _index_cpos=torch.arange(129), _step_memo=lambda key, fn: fn(),
                _rope=lambda q, freq: q)
            state = {'ratio': 1, 'ik': torch.arange(129, dtype=torch.float32)[:, None]}
            with patch('engine.fastdecode.R.qlinear', return_value=torch.ones(2, 1)), \
                 patch('engine.fastdecode.R.mm', return_value=torch.ones(2, 1)):
                FastDecoder._indexer(decoder, torch.ones(2, 1), None, 20, torch.zeros(2, dtype=torch.long),
                                     torch.tensor([89, 117]), state)
                selected = FastDecoder._indexer(decoder, torch.ones(2, 1), None, 24,
                    torch.zeros(2, dtype=torch.long), torch.tensor([89, 117]), state)
            results.append(selected)
        self.assertTrue(torch.equal(*results))
        self.assertEqual(results[0].tolist(), [list(range(85, 89)), list(range(113, 117))])

    def test_compact_sparse_blocks_partial_tail_and_padding(self):
        mask = torch.zeros(2, 19, dtype=torch.bool)
        mask[0, 8:16] = True
        mask[1, 16:] = True
        actual = compact_mask(mask, 8, 16)
        self.assertEqual(tuple(actual.shape), (2, 16))
        self.assertEqual(actual[0][actual[0] >= 0].tolist(), list(range(8, 16)))
        self.assertEqual(actual[1][actual[1] >= 0].tolist(), [16, 17, 18])

    def test_empty_membership_is_never_selected(self):
        ids = compact_mask(torch.zeros(2, 33, dtype=torch.bool), 8, 16)
        self.assertTrue((ids == -1).all())
        scores = score_reference(torch.ones(2, 1, 1), torch.ones(33, 1),
                                 torch.ones(2, 1), ids, torch.tensor([1, 33]))
        self.assertTrue((select(scores, ids, 20) == -1).all())

    def test_scores_match_dense_for_selected_visible_keys(self):
        g = torch.Generator().manual_seed(37)
        q = torch.randn(2, 4, 16, generator=g).bfloat16()
        k = torch.randn(97, 16, generator=g).bfloat16()
        weights = torch.randn(2, 4, generator=g)
        mask = torch.zeros(2, 97, dtype=torch.bool)
        mask[:, 8:16] = True
        mask[:, 56:64] = True
        ids = compact_mask(mask, 8, 24)
        lengths = torch.tensor([11, 60])
        score = score_reference(q, k, weights, ids, lengths)
        dense = (torch.einsum('thd,nd->thn', q, k).float().relu() * weights[:, :, None]).sum(1)
        for row in range(2):
            valid = (ids[row] >= 0) & (ids[row] < lengths[row])
            torch.testing.assert_close(score[row, valid], dense[row, ids[row, valid]], rtol=0, atol=0)
            self.assertTrue(torch.isneginf(score[row, ~valid]).all())
        selected = select(score, ids, 4)
        for row in range(2):
            self.assertTrue((selected[row] < lengths[row]).all())
            live = selected[row] >= 0
            self.assertTrue(mask[row, selected[row, live]].all())
            self.assertEqual(int(live.sum()), min(4, int((mask[row] & (torch.arange(97) < lengths[row])).sum())))


if __name__ == '__main__':
    unittest.main()
