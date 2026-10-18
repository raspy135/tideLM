"""Paired many-prompt A/B for decode arithmetic changes: step time and acceptance separately.

A numerics change reshuffles greedy near-ties, so one prompt's acceptance moves by a coin
flip (docs/decode-fp32-experiments.md: the HC kernel lost 0.07-0.10 accepted tokens/step on
one prompt and was rejected for it). This driver runs a fixed set of varied prompts with the
two arms interleaved per prompt (order alternating), on a frozen map at fixed depth 3, and
reports the paired per-prompt differences of ms/step, accepted length and tok/s with their
standard errors. Configurations switch at runtime
on both ranks. Use the disposable two-node gate, not serving.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'
import torch
import engine.v41_engine as V
import v41_ref
from engine.expert_profiles import mask_digest
from server.app import Tok, load_encoding_module, build_chat_prompt

# Same switches as tools/bench_attn_block_tp.py; inlined because the gate overlays one driver file.
CONFIGS = {
    'base': dict(staged=0, router_bf16=False, fused_tail=False, hc_kernel=False),
    'staged2': dict(staged=2, router_bf16=False, fused_tail=False, hc_kernel=False),
    'router': dict(staged=0, router_bf16=True, fused_tail=True, hc_kernel=False),
    'hc': dict(staged=0, router_bf16=False, fused_tail=False, hc_kernel=True),
    'all': dict(staged=2, router_bf16=True, fused_tail=True, hc_kernel=True),
}
# Draft-only variants of 'all': the target is unchanged, so greedy outputs must be bit-identical
# and per-prompt acceptance differences measure proposal quality alone.
for _k in (64, 128, 256, 512):
    CONFIGS[f'all_mk{_k}'] = dict(CONFIGS['all'], markov_topk=_k)
# Serving-equivalent drafter vs the candidate drafter + L2 prefetch, in one process loaded with
# DSV41_TP_DRAFT_HEAD=1 DSV41_DRAFT_HEAD_FMT=fp8 (the drafter's head is switched at runtime).
CONFIGS['serving'] = dict(CONFIGS['all'], draft_head='main', markov_tp=False, qkv_mb=0, sh_mb=0, attn_mb=4)
CONFIGS['draft_fp8'] = dict(CONFIGS['serving'], draft_head='fp8')
CONFIGS['draft_fp8_split'] = dict(CONFIGS['draft_fp8'], markov_tp=True)
# Live 2026-10-08 configuration (load with DSV41_TP_DRAFT_ATTN=1 too), then the exact FP8 decode
# schedule (BLOCK_N 32, BLOCK_K 256: outputs must not change), then the fp32-rounding launch fusions.
CONFIGS['live'] = dict(CONFIGS['draft_fp8_split'], fp8_bn='auto', fp8_bk=128, rms_fused=False, hc_front_fused=False)
CONFIGS['fp8sched'] = dict(CONFIGS['live'], fp8_bn='32', fp8_bk=256)
CONFIGS['fused'] = dict(CONFIGS['fp8sched'], rms_fused=True, hc_front_fused=True)
# Engram row source: memmap faults vs O_DIRECT sector reads. Same bytes, so outputs must match.
CONFIGS['fused_mm'] = dict(CONFIGS['fused'], engram_direct=False)
CONFIGS['fused_direct'] = dict(CONFIGS['fused'], engram_direct=True)
# Prefill arithmetic only (decode settings = fused): the 2026-10-08 prefill kernels off vs on.
_PREFILL_OFF = dict(pf_indexed=False, pf_hc=False, pf_index=False, pf_scaled=False, pf_tiles=False)
CONFIGS['fused_pf_old'] = dict(CONFIGS['fused'], **_PREFILL_OFF)
CONFIGS['fused_pf_new'] = dict(CONFIGS['fused'], pf_indexed=True, pf_hc=True, pf_index=True,
                               pf_scaled=True, pf_tiles=True)
# Host ordering only (engine/v41_engine.py EARLY_VERIFY / EARLY_DRAFT): outputs must not change.
CONFIGS['early_off'] = dict(CONFIGS['fused_pf_new'], early_verify=False, early_draft=False)
CONFIGS['early_on'] = dict(CONFIGS['fused_pf_new'], early_verify=True, early_draft=True)
# Decode attention + shared expert from the EXL3 dense pack (load with DSV41_EXL3_DENSE=1): numerics change.
CONFIGS['x3_off'] = dict(CONFIGS['early_on'], x3=False)
CONFIGS['x3_on'] = dict(CONFIGS['early_on'], x3=True)
# The DSpark draft blocks' own attention + shared-expert matrices from the EXL3 draft pack (load
# with DSV41_EXL3_DRAFT_DENSE=1). Draft-only: the draft proposes, the verifier accepts the target's
# argmax, so greedy outputs must stay bit-identical and the paired deltas are pure proposal cost.
CONFIGS['x3d_off'] = dict(CONFIGS['early_on'], x3d=False)
CONFIGS['x3d_on'] = dict(CONFIGS['early_on'], x3d=True)
# The drafter's own two heads, which are 21% and 14% of a draft pass's bytes (2026-10-11 trace):
# the vocabulary projection (`draft_head_fmt`, rebuilt from the bf16 shard kept by
# DSV41_BENCH_DRAFT_HEAD_SRC=1) and the five-times-read Markov bias head (`markov_fmt`). Both are
# proposal-only, so greedy output must stay bit-identical and the deltas are pure draft cost.
CONFIGS['heads_fp8'] = dict(CONFIGS['x3d_on'], draft_head_fmt='fp8', markov_fmt='off')
CONFIGS['heads_fp4'] = dict(CONFIGS['x3d_on'], draft_head_fmt='fp4', markov_fmt='fp4')
CONFIGS['heads_fp4_mk8'] = dict(CONFIGS['x3d_on'], draft_head_fmt='fp4', markov_fmt='fp8')
CONFIGS['heads_fp8_mk4'] = dict(CONFIGS['x3d_on'], draft_head_fmt='fp8', markov_fmt='fp4')
# The verifier's own LM head. `packed` is the lossless bf16 spelling the profile ships; `fp8` is
# DSV41_HEAD_FMT=fp8, measured 2026-09-11 at 5.68 -> 2.96 ms for +0.0004 coding / +0.0014 general
# nats. Load with DSV41_BENCH_DRAFT_HEAD_SRC=1 and HEAD_KERNEL=packed: the bf16 shard is the source
# both heads are built from, and the drafter is held on the same separate fp8 head in both arms, so
# only the verifier moves. This is a target-arithmetic change, not a draft one -- greedy outputs may
# differ, which is why `head_fmt` is in target_keys below.
CONFIGS['h_packed'] = dict(CONFIGS['x3d_on'], head_fmt='packed', draft_head='fp8')
CONFIGS['h_fp8'] = dict(CONFIGS['x3d_on'], head_fmt='fp8', draft_head='fp8')
# The drafter's two heads over a listed vocabulary (engine/draft_vocab.py). Load with
# DSV41_DRAFT_VOCAB=<ranking> DSV41_BENCH_DRAFT_VOCAB_AB=1 (and DSV41_EXL3_DRAFT_DENSE=1, which
# x3d_on reads) so the listed and the full-vocabulary fp8 draft heads are both resident. Draft-only:
# greedy outputs must stay bit-identical; the acceptance delta is the unlisted tokens' cost.
CONFIGS['vocab_full'] = dict(CONFIGS['x3d_on'], draft_vocab=False)
CONFIGS['vocab_listed'] = dict(CONFIGS['x3d_on'], draft_vocab=True)
# Lossy decode pruning of low-share routed picks (engine/expert_topp.py). Load with
# DSV41_EXPERT_TOPP=0.90 so the slot table carries the pruned-pick column; the off arm sets the rule
# to None at runtime. Target numerics change, so outputs differ and acceptance is part of the cost.
CONFIGS['topp_off'] = dict(CONFIGS['x3d_on'], expert_topp=None)
CONFIGS['topp90'] = dict(CONFIGS['x3d_on'], expert_topp=(0.90, 3))
for _qkv, _sh, _attn in ((24, 16, 4), (24, 16, 12), (12, 0, 4), (24, 0, 4), (0, 16, 4)):
    CONFIGS[f'cand_q{_qkv}_s{_sh}_a{_attn}'] = dict(CONFIGS['draft_fp8_split'], qkv_mb=_qkv, sh_mb=_sh, attn_mb=_attn)

PROMPTS = [
    'Write a Python function that merges overlapping intervals and explain its complexity.',
    'Implement a thread-safe LRU cache in Go with Get and Put methods.',
    'Write a SQL query that returns the top three customers by total order value per country.',
    'Explain how a hash map handles collisions, with a short example in C.',
    'Write a bash script that finds the ten largest files under a directory.',
    'Refactor this JavaScript into async/await: fetch(url).then(r => r.json()).then(d => console.log(d)).catch(e => console.error(e));',
    'Write a short story about a lighthouse keeper who receives an unexpected letter.',
    'Describe the water cycle for a ten-year-old in two paragraphs.',
    'Write a persuasive paragraph arguing that cities should plant more street trees.',
    'Summarize the causes of the French Revolution in five bullet points.',
    'Explain the difference between TCP and UDP and when to use each.',
    'What is the derivative of x^3 * sin(x)? Show the steps.',
    'A train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed? Explain.',
    'Prove that the square root of 2 is irrational.',
    'Output a JSON object describing three fictional books with title, author, year and genre.',
    'Convert this to YAML: {"server": {"port": 8080, "hosts": ["a", "b"], "tls": true}}',
    'Create a markdown table comparing Python, Rust and Go on speed, safety and learning curve.',
    '日本の四季について、それぞれの特徴を短く説明してください。',
    'Translate into French: The meeting has been moved to Thursday afternoon because of the holiday.',
    'Write a minimal HTML page with a centered card containing a title and a button, using inline CSS.',
    'List ten creative names for a coffee shop and give a one-line reason for each.',
    'Give step-by-step instructions to make a basic tomato sauce.',
    'What are the main trade-offs between microservices and a monolith?',
    'Write a regular expression that matches ISO 8601 dates and explain each part.',
]


def set_config(e, cfg):
    fd, lean = e.fast, e.fast.lean
    torch.cuda.synchronize()
    fd.release_graphs()
    fd.staged_attention = cfg['staged']
    lean.router_bf16 = cfg['router_bf16']
    lean.fused_router_tail = cfg['fused_tail']
    v41_ref.HC_KERNEL = cfg['hc_kernel']
    fd.draft_markov_topk = cfg.get('markov_topk', 0)
    fd.draft_markov_tp = cfg.get('markov_tp', False)
    if 'fp8_bk' in cfg:
        import fp8_linear as _fp8
        _fp8.DECODE_BLOCK_N, _fp8.DECODE_BLOCK_K = cfg['fp8_bn'], cfg['fp8_bk']  # read at call time
        lean.rms_fused, lean.hc_front_fused = cfg['rms_fused'], cfg['hc_front_fused']
    if 'head_fmt' in cfg:
        fmt = cfg['head_fmt']
        src = getattr(e.W, 'draft_head_src', None)
        assert src is not None, 'load with DSV41_BENCH_DRAFT_HEAD_SRC=1 to rebuild the verifier head'
        bank = getattr(e, '_verifier_head_bank', None)
        if bank is None:
            bank = e._verifier_head_bank = {'packed': e.W.head}
        if fmt not in bank:
            from engine.tensor_parallel import VocabParallelHead
            bank[fmt] = VocabParallelHead(v41_ref._head_in_format(src, fmt), e.W.head.world)
        # fd.head and Model's three head_logits sites all read W.head.
        e.W.head = fd.head = bank[fmt]
    if 'draft_head' in cfg:
        alt = getattr(e.W, 'draft_head', None)
        assert cfg['draft_head'] == 'main' or alt is not None, 'load with DSV41_TP_DRAFT_HEAD=1 DSV41_DRAFT_HEAD_FMT=fp8'
        fd.draft_head = fd.head if cfg['draft_head'] == 'main' else alt
    if 'qkv_mb' in cfg:
        assert fd.l2pf_side is not None, 'L2 prefetch disabled (DSV41_L2PF_MB=0)'
        fd.qkv_prefetch_mb, fd.sh_prefetch_mb, fd.attn_prefetch_mb = cfg['qkv_mb'], cfg['sh_mb'], cfg['attn_mb']
        import engine.l2pf as _l2pf
        _l2pf.MODE = cfg.get('pf_mode', 'touch')  # read by l2pf.touch at capture
    if 'pf_scaled' in cfg:
        import engine.model as M
        import fp4_moe as K
        M.PREFILL_ATTN_INDEXED, M.HC_PREFILL_FUSED, M.INDEX_FUSED = cfg['pf_indexed'], cfg['pf_hc'], cfg['pf_index']
        K.PREFILL_DOT_SCALED, K.PREFILL_SCALED_TILES = cfg['pf_scaled'], cfg['pf_tiles']
    if 'engram_direct' in cfg:
        from engine.engram_native import NativeGather
        for t in e.tables.values():
            if not hasattr(t, '_ab_native'):
                dfd = os.open(t.path, os.O_RDONLY | os.O_DIRECT)
                t._ab_native = (t.native_gather, NativeGather(
                    None, None, workers=64, direct=(dfd, t.w_off, t.s_off, t.n_rows)))
            t.direct = cfg['engram_direct']  # read per gather by EngramTable._gather_rows_uncached
            t.direct_gather = t._ab_native[1] if t.direct else None
            t.native_gather = t._ab_native[0]
    if 'x3' in cfg:
        assert e.exl3_dense is not None, 'load with DSV41_EXL3_DENSE=1'
        for L, w in enumerate(e.W.layers):
            w._x3 = e.exl3_dense.layers[L] if cfg['x3'] else None
    if 'x3d' in cfg:
        # `attach` already put these on the blocks; keep a bank so the off arm can restore them.
        if not hasattr(e, '_x3d_bank'):
            assert getattr(e, 'exl3_draft', None) is not None, 'load with DSV41_EXL3_DRAFT_DENSE=1'
            e._x3d_bank = [(w._x3, w._x3sh) for w in e.W.mtp]
        for w, (x3, sh) in zip(e.W.mtp, e._x3d_bank):
            w._x3 = x3 if cfg['x3d'] else None
            w._x3sh = sh if cfg['x3d'] else None
    if 'draft_head_fmt' in cfg:
        fmt = cfg['draft_head_fmt']
        src = getattr(e.W, 'draft_head_src', None)
        assert src is not None, 'load with DSV41_BENCH_DRAFT_HEAD_SRC=1 to rebuild the draft head'
        bank = getattr(e, '_draft_head_bank', None)
        if bank is None:
            bank = e._draft_head_bank = {}
        if fmt not in bank:
            from engine.tensor_parallel import VocabParallelHead
            bank[fmt] = VocabParallelHead(v41_ref._head_in_format(src, fmt), e.W.head.world)
        fd.draft_head = bank[fmt]
    if 'draft_vocab' in cfg:
        full = getattr(e.W, 'draft_head_full', None)
        assert full is not None and e.W.draft_vocab_digest is not None, \
            'load with DSV41_DRAFT_VOCAB=<ranking> DSV41_BENCH_DRAFT_VOCAB_AB=1'
        fd.draft_head = e.W.draft_head if cfg['draft_vocab'] else full
        fd.set_markov_fmt(fd.markov_fmt)  # the split chain's Markov rows follow the head's ids
    if 'expert_topp' in cfg:
        assert fd.topp_sentinel is not None, 'load with DSV41_EXPERT_TOPP=0.90'
        fd.expert_topp = cfg['expert_topp']   # read by _layer_a at capture; set_config released graphs
    if 'markov_fmt' in cfg:
        fd.set_markov_fmt(cfg['markov_fmt'])
    if 'early_draft' in cfg:
        V.EARLY_VERIFY, V.EARLY_DRAFT = cfg['early_verify'], cfg['early_draft']  # read per step
    if lean.router_bf16:
        lean.prepare_router_weights([w.gate_w for w in e.W.layers])


def paired(diffs):
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
    return {'mean': m, 'se': se, 'n': len(diffs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--a', default='base')
    ap.add_argument('--b', default='all')
    ap.add_argument('--max-tokens', type=int, default=160)
    ap.add_argument('--prompts', type=int, default=len(PROMPTS))
    # The default warms up only when the arm changes, so with alternating order the first run of
    # every prompt captures that prompt's graph variants inside its measured window: +1.0 to +1.6
    # ms/step on whichever arm ran first, in every A/B of 2026-10-11 (exl3-draft-ab, head-fp8,
    # draft-vocab). Alternation keeps the mean unbiased but that term dominates the error bar.
    ap.add_argument('--warm-each', action='store_true',
                    help='warm up on the prompt before every measured run, not only on a switch')
    args = ap.parse_args()
    assert args.a in CONFIGS and args.b in CONFIGS and 1 <= args.prompts <= len(PROMPTS)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=float(os.environ.get('DSV41_BENCH_ARENA_GB', '90.2')),
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=int(os.environ.get('DSV41_BENCH_TRANSIENT', '8')), keep_free_gb=6,
                   expert_format=os.environ.get('DSV41_BENCH_EXPERT_FORMAT', 'fp4'))
    e.confidence_depth_policy = None
    e.depth_policy.pinned = 3
    assert e.fast is not None and e.fast.lean is not None
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    ids = [build_chat_prompt({'messages': [{'role': 'user', 'content': p}]}, enc, tok, False, 75, e)[1]
           for p in PROMPTS[:args.prompts]]
    signature = hashlib.sha256(json.dumps([ids, args.a, args.b, args.max_tokens]).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(signature))) == 1, 'workload differs between ranks'
    maphash = mask_digest(e.model_prune_mask)
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/rank{e.ep.rank}.json'
    report = {'a': args.a, 'b': args.b, 'configs': {k: CONFIGS[k] for k in (args.a, args.b)},
              'max_tokens': args.max_tokens, 'runs': []}
    current = [None]

    def run(i, arm, measured=True):
        switched = current[0] != arm
        if switched:
            set_config(e, CONFIGS[arm])
            assert len(set(e.ep.gather_objects(arm))) == 1, 'arm differs between ranks'
            current[0] = arm
        if switched or (measured and args.warm_each):
            # Capture every graph this prompt needs (parities, index buckets) outside the
            # measurement: a 24-token warmup on another prompt left the first measured run after
            # each switch ~1.9 ms/step slower (results/accept-ab-20261008/all-vs-mk128).
            for _ in e.generate(ids[i], max_tokens=args.max_tokens, temperature=0, seed=42,
                                stop_token_ids={eos}):
                pass
        output = []
        for burst in e.generate(ids[i], max_tokens=args.max_tokens, temperature=0, seed=42,
                                stop_token_ids={eos}):
            output.extend(burst)
        digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1, 'rank disagreement'
        assert mask_digest(e.model_prune_mask) == maphash, 'expert map changed'
        s = e.last_stats
        row = {'prompt': i, 'arm': arm, 'tokens': len(output), 'hash': digest, 'steps': s['steps'],
               'decode_s': s['decode_s'], 'accept_len_mean': s['accept_len_mean'],
               'ms_per_step': 1000 * s['decode_s'] / max(1, s['steps']),
               'tok_s': s['decode_tok_s'],
               'text': tok.decode([t for t in output if t != eos])}
        if measured:
            report['runs'].append(row)
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
            print('ACCEPT_AB_RUN ' + json.dumps({k: v for k, v in row.items() if k != 'text'}), flush=True)
        return row

    for i in range(len(ids)):
        order = (args.a, args.b) if i % 2 == 0 else (args.b, args.a)
        for arm in order:
            run(i, arm)
    rows = report['runs']
    by = {(r['prompt'], r['arm']): r for r in rows}
    usable = [i for i in range(len(ids)) if by[(i, args.a)]['steps'] >= 8 and by[(i, args.b)]['steps'] >= 8]
    # Target arithmetic keys; draft-only and exact-schedule (fp8_bn/fp8_bk) differences must leave
    # greedy outputs bit-identical, so those A/Bs assert it.
    target_keys = list(CONFIGS['base']) + ['rms_fused', 'hc_front_fused', 'x3', 'head_fmt', 'expert_topp'] + list(_PREFILL_OFF)
    draft_only = all(CONFIGS[args.a].get(k) == CONFIGS[args.b].get(k) for k in target_keys)
    report['summary'] = {
        'draft_only': draft_only,
        'prompts_used': len(usable),
        'identical_outputs': sum(by[(i, args.a)]['hash'] == by[(i, args.b)]['hash'] for i in range(len(ids))),
        'a_ms_per_step': statistics.mean(by[(i, args.a)]['ms_per_step'] for i in usable),
        'b_ms_per_step': statistics.mean(by[(i, args.b)]['ms_per_step'] for i in usable),
        'a_accept': statistics.mean(by[(i, args.a)]['accept_len_mean'] for i in usable),
        'b_accept': statistics.mean(by[(i, args.b)]['accept_len_mean'] for i in usable),
        'a_tok_s': statistics.mean(by[(i, args.a)]['tok_s'] for i in usable),
        'b_tok_s': statistics.mean(by[(i, args.b)]['tok_s'] for i in usable),
        'd_ms_per_step': paired([by[(i, args.b)]['ms_per_step'] - by[(i, args.a)]['ms_per_step'] for i in usable]),
        'd_accept': paired([by[(i, args.b)]['accept_len_mean'] - by[(i, args.a)]['accept_len_mean'] for i in usable]),
        'd_tok_s': paired([by[(i, args.b)]['tok_s'] - by[(i, args.a)]['tok_s'] for i in usable]),
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    print('ACCEPT_AB_SUMMARY ' + json.dumps(report['summary']), flush=True)
    if draft_only:
        assert report['summary']['identical_outputs'] == len(ids), 'a draft-only change altered target output'
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
