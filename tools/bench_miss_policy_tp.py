"""Teacher-forced held-out NLL under each rule for a routed expert the resident set lacks.

The router masks non-resident experts before its top-k, so a missing pick is replaced by the next
resident expert in the router's ranking (`substitute`, what serving does). The question this
answers: is that substitute worth its expert read, or could the slot be skipped? Arms, switched at
runtime in one process on a frozen expert map (Model._apply_miss_policy):

  substitute  next-ranked resident expert, weight from its own score (serving)
  skip        no contribution; the resident picks renormalize to the full route scale
  skip_raw    no contribution; the resident picks keep their unpruned weights
  random      a random other resident expert in the substitute's slot, same weight

  hot:M:C:B   DSV41_HOT_READS emulated (engine/hot_reads.PrefillEmulation): blocks of B rows (a
              verify step), each with decode's paced budget of C reads, reading a block's heaviest
              missing pick at weight share >= M; the read expert is routable for the whole block
  hot:M:C:B:G:H  the same read not waited for: routable G blocks later, for H blocks
  exact       every missing pick read: the unpruned routing, the reference for the others

and TensorFold's lossy routed-expert pruning on top of `substitute` (Model._apply_expert_topp):
`topp85` / `topp90` keep each row's shortest weight-share prefix reaching 0.85 / 0.90 (at least 3)
with the router's own weights, `topp85k` renormalizes the kept ones (TensorFold's production rule).

Each arm scores corpus/heldout_eval.jsonl (coding + general, <= 480 tokens a sequence) in one
prefill pass through Model.forward, as V41Engine.teacher_forced does, and reports token-weighted
mean NLL per category, its paired difference from `substitute` with a per-sequence (clustered)
standard error, and the miss rate. `substitute` is re-run on the first sequences last: it must
reproduce its first run bit for bit, or arm differences are not attributable to the arm.
Use the disposable two-node gate, not serving.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path[:0] = ['/app', '/app/tools']
import bench_accept_ab_tp as AB  # noqa: E402  (sets the frozen-map environment first)
import torch  # noqa: E402
from engine.expert_profiles import mask_digest  # noqa: E402

V = AB.V
# arm -> (miss policy, top-p rule or None)
ARM_SPECS = {
    'substitute': ('substitute', None), 'skip': ('skip', None), 'skip_raw': ('skip_raw', None),
    'random': ('random', None),
    'topp85': ('substitute', (0.85, 3, 'orig')), 'topp85k': ('substitute', (0.85, 3, 'kept')),
    'topp90': ('substitute', (0.90, 3, 'orig')), 'topp90k': ('substitute', (0.90, 3, 'kept')),
}
ARMS = ('substitute', 'skip', 'skip_raw', 'random')


def hot_arm(arm, model, k):
    """'hot:M:C:B' / 'exact' -> a PrefillEmulation over the pruned layers, else None."""
    from engine.hot_reads import HotReads, PrefillEmulation
    lag, hold = 0, 1
    if arm == 'exact':
        env, block = {'DSV41_HOT_MIN': '1e-9', 'DSV41_HOT_MAX': '1000000', 'DSV41_HOT_PER_LAYER': str(k)}, 1
    elif arm.startswith('hot:'):
        m, c, b, *lh = arm[4:].split(':')
        env, block = {'DSV41_HOT_MIN': m, 'DSV41_HOT_MAX': c}, int(b)
        lag, hold = (int(lh[0]), int(lh[1])) if lh else (0, 1)
    else:
        return None
    a = model.args
    stream = getattr(model, 'stream_layers', frozenset())
    return PrefillEmulation(HotReads(a.n_layers, a.n_routed_experts, k, env),
                            [L for L in model.prune_mask if L not in stream], block, lag, hold)


def clustered(diffs, ns):
    """Token-weighted mean of per-sequence mean differences and its cluster-robust SE."""
    tot = sum(ns)
    mean = sum(d * n for d, n in zip(diffs, ns)) / tot
    m = len(ns)
    var = sum((n * (d - mean)) ** 2 for d, n in zip(diffs, ns)) * m / max(1, m - 1)
    return mean, math.sqrt(var) / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--corpus', default='/app/corpus/heldout_eval.jsonl')
    ap.add_argument('--limit', type=int, default=0, help='first N sequences (0 = all)')
    ap.add_argument('--repeat', type=int, default=8, help='sequences re-scored with substitute at the end')
    ap.add_argument('--arms', default=','.join(ARMS))
    args = ap.parse_args()
    arms = args.arms.split(',')
    assert arms[0] == 'substitute' and all(a in ARM_SPECS or a == 'exact' or a.startswith('hot:') for a in arms)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=float(os.environ.get('DSV41_BENCH_ARENA_GB', '98.4')),
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=int(os.environ.get('DSV41_BENCH_TRANSIENT', '8')),
                    keep_free_gb=6,
                    expert_format=os.environ.get('DSV41_BENCH_EXPERT_FORMAT', 'fp4'))
    m = e.model
    assert m.prune_mask, 'no resident mask: nothing can miss'
    m.miss_stats = torch.zeros(2, dtype=torch.int64, device=e.device)
    m.topp_stats = torch.zeros(2, dtype=torch.int64, device=e.device)
    m.miss_gen = torch.Generator(device=e.device)
    seqs = [json.loads(l) for l in open(args.corpus) if l.strip()]
    if args.limit:
        seqs = seqs[:args.limit]
    toks = [e.tokenizer.encode(d['text'], add_special_tokens=False) for d in seqs]
    sig = json.dumps([len(t) for t in toks] + arms)
    assert len(set(e.ep.gather_objects(sig))) == 1, 'workload differs between ranks'
    maphash = mask_digest(e.model_prune_mask)
    cfg = e.config()
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/rank{e.ep.rank}.json'
    report = {'arms': {}, 'corpus': args.corpus, 'sequences': len(seqs),
              'config': {k: cfg.get(k) for k in ('expert_format', 'arena_slots', 'prune_keep', 'tp_experts')}}

    def score(arm, which):
        policy, rule = ARM_SPECS.get(arm, ('substitute', None))
        m.miss_policy, m.expert_topp = policy, rule
        m.hot_eval = hot_arm(arm, m, m.args.n_activated_experts)
        m.miss_gen.manual_seed(20261011)
        m.miss_stats.zero_()
        m.topp_stats.zero_()
        out = []
        for i in which:
            e._reset()
            m.begin_prompt()
            t = torch.tensor(toks[i], dtype=torch.long, device=e.device)
            logits, _ = m.forward(t, 0, prefill=True, need_logits=True)
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            out.append(-lp.gather(1, t[1:, None]).squeeze(1).cpu())
        hot, m.hot_eval = m.hot_eval, None
        m.miss_policy, m.expert_topp = 'substitute', None
        score.hot = dict(hot.stats) if hot is not None else None
        miss, slots = m.miss_stats.tolist()
        kept, total = m.topp_stats.tolist()
        return out, miss, slots, (kept / total if total else None)

    base = None
    for arm in arms:
        assert len(set(e.ep.gather_objects(arm))) == 1, 'arm differs between ranks'
        t0 = time.time()
        nll, miss, slots, kept_frac = score(arm, range(len(seqs)))
        total = round(float(sum(x.double().sum() for x in nll)), 6)
        assert len(set(e.ep.gather_objects(total))) == 1, f'{arm}: ranks scored different NLL'
        assert mask_digest(e.model_prune_mask) == maphash, 'expert map changed'
        if base is None:
            base = nll
        row = {'seconds': round(time.time() - t0, 1), 'miss_slots': miss, 'slots': slots,
               'miss_rate': miss / max(1, slots), 'topp_kept_fraction': kept_frac, 'hot': score.hot,
               'categories': {}}
        for cat in sorted({d['category'] for d in seqs}):
            idx = [i for i, d in enumerate(seqs) if d['category'] == cat]
            ns = [nll[i].numel() for i in idx]
            mean = sum(float(nll[i].double().sum()) for i in idx) / sum(ns)
            d = [float((nll[i].double() - base[i].double()).mean()) for i in idx]
            dm, se = clustered(d, ns)
            changed = sum(int((nll[i] != base[i]).sum()) for i in idx)
            row['categories'][cat] = {'tokens': sum(ns), 'mean_nll': mean, 'd_vs_substitute': dm, 'se': se,
                                      'positions_changed': changed}
        row['per_seq_mean_nll'] = [round(float(x.double().mean()), 6) for x in nll]
        report['arms'][arm] = row
        print('MISS_POLICY_ARM ' + json.dumps({'arm': arm, **{k: v for k, v in row.items() if k != 'per_seq_mean_nll'}}),
              flush=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    rep = list(range(min(args.repeat, len(seqs))))
    again = score('substitute', rep)[0]
    same = all(torch.equal(a, base[i]) for a, i in zip(again, rep))
    report['substitute_repeat_identical'] = same
    print('MISS_POLICY_REPEAT ' + json.dumps({'sequences': len(rep), 'identical': same}), flush=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
