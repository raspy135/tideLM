"""Cold-start map fidelity and local-history precedence, without loading weights."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from engine.expert_seed import DEFAULT_PATH, load_seed, seed_selection
from engine.global_residency import initial_selection


class ExpertSeedTest(unittest.TestCase):
    def load(self, db, **kw):
        options=dict(enabled=True,request_unit=True,metric='score',layers=40,experts=384)
        options.update(kw)
        return load_seed(db, **options)

    def test_packaged_seed_exact_map_and_adaptable_history(self):
        with tempfile.TemporaryDirectory() as d:
            seed=self.load(Path(d)/'new.npz')
        meta=json.loads(DEFAULT_PATH.with_suffix('.json').read_text())
        keep=seed_selection(seed,9400,6)
        self.assertEqual(seed['sha256'],meta['sha256'])
        self.assertEqual([len(keep[L]) for L in range(40)],meta['resident_per_layer'])
        self.assertEqual(sum(map(len,keep.values())),9400)
        for L,ids in keep.items():
            np.testing.assert_array_equal(ids,np.flatnonzero(seed['keep'][L]))
        # The live accumulator accepts both distributions in request-history v2 units.
        from engine import v41_engine as V
        from unittest.mock import patch
        from engine.adapt_config import resolve
        with tempfile.TemporaryDirectory() as d, patch.object(V,'ADAPT',resolve({
                'DSV41_ADAPT_SENSITIVITY':'medium','DSV41_ADAPT_PRIOR':'8',
                'DSV41_PRUNE_METRIC':'score'})):
            f=str(Path(d)/'personal.npz')
            V.save_prune_db(seed['counts'],seed['mass'],f)
            restored=V.load_prune_db(f)
            np.testing.assert_array_equal(restored[0],seed['counts'])
            np.testing.assert_array_equal(restored[1],seed['mass'])
            self.assertIsNone(self.load(f))

    def test_real_boot_ranking_broadcast_overrides_peer_local_history(self):
        # Execute the production boot-ranking segment on CPU, without allocating
        # model weights. Rank 1 deliberately sees a different local database.
        import inspect, textwrap
        from types import SimpleNamespace
        from engine import v41_engine as V
        from engine.adapt_config import resolve
        source = textwrap.dedent(inspect.getsource(V.V41Engine.__init__))
        start = source.index('        self._prune_trace = {L: v.copy()')
        end = source.index('        per_layer =', start)
        block = compile(textwrap.dedent(source[start:end]), '<boot-ranking>', 'exec')
        config = resolve({'DSV41_ADAPT_SENSITIVITY':'medium',
                          'DSV41_ADAPT_PRIOR':'8','DSV41_PRUNE_METRIC':'score'})
        packets = []
        engines = []
        with tempfile.TemporaryDirectory() as d:
            for rank in (0,1):
                def broadcast(packet, rank=rank):
                    if rank == 0: packets.append(packet)
                    return packets[-1]
                eng = SimpleNamespace(
                    _expert_seed=None, expert_seed_sha256=None,
                    ep=SimpleNamespace(rank=rank,broadcast_obj=broadcast),
                    expert_profile=SimpleNamespace(rank='sum'),
                    dynamic_experts=True,dynamic_layer_floor=6,resident_budget=None,
                    args=SimpleNamespace(n_layers=40,n_routed_experts=384,n_activated_experts=6))
                ns=V.__dict__.copy()
                ns.update(self=eng, ADAPT=config, PRUNE_DB=str(Path(d)/'absent.npz'),
                          counts={L:np.ones(384) for L in range(40)},
                          prune_select='uniform',prune_keep=.61,device='cpu',
                          load_prune_db=lambda: None if rank==0 else (np.ones((40,384)),np.ones((40,384))))
                exec(block,ns)
                engines.append(eng)
        self.assertIsNotNone(engines[0].expert_seed_sha256)
        self.assertEqual(engines[0].expert_seed_sha256,engines[1].expert_seed_sha256)
        for L in range(40):
            np.testing.assert_array_equal(engines[0].model_prune_mask[L],engines[1].model_prune_mask[L])
            np.testing.assert_array_equal(engines[0].model_prune_mask[L],engines[0]._expert_seed['keep'][L])

    def test_existing_db_wins_even_if_empty_or_incompatible(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mine.npz';p.write_bytes(b'user-owned incompatible history')
            self.assertIsNone(self.load(p,path=Path(d)/'nonexistent-seed'))
            self.assertEqual(p.read_bytes(),b'user-owned incompatible history')

    def test_explicit_opt_out_and_incompatible_modes_do_not_read_seed(self):
        with tempfile.TemporaryDirectory() as d:
            for options in [dict(enabled=False),dict(request_unit=False),dict(metric='frequency')]:
                self.assertIsNone(self.load(Path(d)/'absent',path=Path(d)/'absent-seed',**options))

    def test_other_budget_uses_normal_global_selection(self):
        with tempfile.TemporaryDirectory() as d:seed=self.load(Path(d)/'absent')
        self.assertIsNone(seed_selection(seed,8000,6))
        scores={L:v/v.sum() for L,v in enumerate(seed['mass'])}
        keep=initial_selection(scores,8000,6)
        self.assertEqual(sum(map(len,keep.values())),8000)
        self.assertGreaterEqual(min(map(len,keep.values())),6)

    def test_invalid_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'seed.npz'
            np.savez(p,version=[1],counts=np.ones((2,2)),mass=np.ones((2,2)),keep=np.ones((2,2)))
            with self.assertRaisesRegex(ValueError,'dimensions'):self.load(Path(d)/'absent',path=p)
            np.savez(p,version=[1],counts=np.full((40,384),np.nan),mass=np.ones((40,384)),keep=np.ones((40,384)))
            with self.assertRaisesRegex(ValueError,'demand'):self.load(Path(d)/'absent',path=p)


if __name__=='__main__':unittest.main()
