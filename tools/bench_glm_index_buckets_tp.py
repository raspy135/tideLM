"""Paired GLM index-bound qualification and MTP-threshold experiment.

Run on both ranks with GLM_INDEX_BUCKETS=1 and fixed residents (GLM_ADAPT=0,
GLM_BG=0, GLM_HOT=0, GLM_PREFIX_DISK=0). Runtime arm changes are identical on
both ranks and checked before every arm. Timings exclude prefill and 48 warm
tokens. This is a controlled experiment, not the shipped serving profile.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, '/app')
import torch

from engine.glm.index_bounds import index_bucket
from engine.glm.index_kernel import index_scores
from engine.glm.tp import GlmTP
from server.glm_app import GlmTemplate


def component_checks():
    torch.manual_seed(932)
    result = []
    for rows, stride, offset, end in ((1, 2, 0, 125), (4, 2, 1, 4100),
                                      (8, 2, 0, 8200), (1, 1, 0, 8192)):
        capacity = 262144 // stride
        bound = (index_bucket(end, 262144) + stride - 1) // stride
        q = torch.randn(rows, 32, 128, device='cuda', dtype=torch.bfloat16)
        w = torch.rand(rows, 32, device='cuda')
        raw = torch.randn(capacity, 128, device='cuda')
        scale = raw.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 448
        cache = ((raw / scale).to(torch.float8_e4m3fn), scale)
        pos = torch.arange(end - rows, end, device='cuda')
        full = index_scores(q, w, cache, pos, capacity, 128 ** -.5, stride, offset)
        small = index_scores(q, w, cache, pos, bound, 128 ** -.5, stride, offset)
        assert torch.equal(full[:, :bound], small), 'changed scores inside the bound'
        assert torch.isneginf(full[:, bound:]).all(), 'omitted a visible position'
        assert torch.equal(full.topk(2048).values, small.topk(2048).values), 'changed top-k values'
        # Replaying at a new position within the same graph must keep causality.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            index_scores(q, w, cache, pos, bound, 128 ** -.5, stride, offset)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = index_scores(q, w, cache, pos, bound, 128 ** -.5, stride, offset)
        pos.sub_(1)
        graph.replay()
        expected = index_scores(q, w, cache, pos, bound, 128 ** -.5, stride, offset)
        assert torch.equal(captured, expected), 'stale graph visibility mask'
        result.append(dict(rows=rows, stride=stride, offset=offset, end=end, bound=bound,
                           scores_equal=True, topk_values_equal=True, graph_replay_equal=True))
    torch.cuda.synchronize()
    return result


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/results/glm-perf-paired')
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--warm', type=int, default=48)
    a = ap.parse_args()
    for key in ('GLM_ADAPT', 'GLM_BG', 'GLM_HOT', 'GLM_PREFIX_DISK'):
        if os.environ.get(key) != '0':
            raise ValueError(f'{key}=0 is required for a fixed-resident comparison')
    components = component_checks()
    tpl = GlmTemplate(os.environ.get('GLM_MODEL', '/models'))
    eng = GlmTP(os.environ.get('GLM_MODEL', '/models'))
    if not eng.index_buckets:
        raise ValueError('boot both ranks with GLM_INDEX_BUCKETS=1')
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rank = eng.rank
    corpus = [json.loads(line)['text'] for line in open('/app/corpus/heldout_eval.jsonl')]
    natural = tpl.encode('\n\n'.join(corpus))
    messages = {
        'prose': 'Write a short story (about 800 words) about a lighthouse keeper who finds a message in a bottle that seems to be from herself.',
        'code': 'Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include a docstring and tests.',
        'long8k': tpl.tok.decode(natural[:8192]) + '\n\nSummarize the documents above, section by section, in detail.',
    }
    prompts = {name: tpl.encode(tpl.render([{'role': 'user', 'content': msg}], None, 'low'))
               for name, msg in messages.items()}
    report = dict(rank=rank, components=components, quality=[], timing=[], mtp=[])

    def save():
        (outdir / f'paired-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')

    def arm(enabled, tau=.9):
        state = dict(enabled=enabled, tau=tau)
        states = eng.comm.broadcast_obj(state if rank == 0 else None)
        assert states == state, 'rank-local experiment arm mismatch'
        eng.index_buckets = enabled
        eng.index_bound = eng.max_seq
        eng.mtp_dyn = tau
        eng.reset()

    eng.warmup()
    # Teacher force held-out text in actual decode graphs, including a bucket
    # crossing at 4096. Baseline and bounded arms read identical resident maps.
    for length in (64, 4080, 8192):
        ref = None
        for enabled in (False, True):
            arm(enabled)
            eng.mtp_fill = False
            eng.forward(natural[:length], all_logits=False)
            eng.graph_rows.add(4)
            values = []
            for offset in range(0, 32, 4):
                z = eng.forward(natural[length + offset:length + offset + 4]).float()
                values.append(z.clone())
            z = torch.cat(values)
            if ref is None:
                ref = z
            else:
                p, q = ref.log_softmax(-1), z.log_softmax(-1)
                targets = torch.tensor(natural[length + 1:length + 33], device=eng.dev)
                row = torch.arange(32, device=eng.dev)
                report['quality'].append(dict(context=length, positions=32,
                    max_logit_abs=float((ref-z).abs().max()),
                    top1_agreement=float((ref.argmax(-1) == z.argmax(-1)).float().mean()),
                    kl_mean=float((p.exp() * (p-q)).sum(-1).mean()),
                    nll_delta=float((p[row, targets] - q[row, targets]).mean())))
                print('[paired-quality]', report['quality'][-1], flush=True)
        save()

    def measure(name, enabled, tau, temperature, seed, destination):
        arm(enabled, tau)
        rec = dict(prompt=name, input_tokens=len(prompts[name]), enabled=enabled, tau=tau,
                   temperature=temperature, seed=seed)
        seq = []
        clock, stats, captures, evictions = None, None, None, None
        first = time.perf_counter()
        it = eng.stream(prompts[name], a.tokens + a.warm, depth=7,
                        temperature=temperature, top_p=.95, seed=seed, stop=[])
        for tok in it:
            seq.append(tok)
            if len(seq) == 1:
                rec['ttft_seconds'] = time.perf_counter() - first
            if len(seq) == a.warm:
                torch.cuda.synchronize()
                stats = dict(eng.spec_stats)
                captures = set(eng.graphs)
                evictions = eng.index_bucket_evictions
                clock = time.perf_counter()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        rounds = eng.spec_stats['rounds'] - stats.get('rounds', 0)
        rec.update(seconds=seconds, tokens=len(seq) - a.warm, rounds=rounds,
                   tok_s=(len(seq)-a.warm)/seconds, ms_round=seconds*1000/rounds,
                   drafted=eng.spec_stats.get('drafted', 0)-stats.get('drafted', 0),
                   accepted=eng.spec_stats.get('accepted', 0)-stats.get('accepted', 0),
                   captures_during_timing=len(set(eng.graphs)-captures),
                   evictions_during_timing=eng.index_bucket_evictions-evictions,
                   output_sha256=hashlib.sha256(json.dumps(seq).encode()).hexdigest(),
                   cuda_allocated_gb=torch.cuda.memory_allocated()/1e9,
                   cuda_reserved_gb=torch.cuda.memory_reserved()/1e9,
                   output=tpl.decode(seq))
        report[destination].append(rec)
        print('[paired-timing]', {k:v for k,v in rec.items() if k != 'output'}, flush=True)
        save()

    # Greedy off/on/on/off: stable workload and matching-output checks separate
    # the indexer's mechanical gain from changed speculative acceptance.
    for name in prompts:
        for enabled in (False, True, True, False):
            measure(name, enabled, .9, 0., 42, 'timing')
    # Existing confidence gate: calibrate its threshold against live round cost
    # before adding a new scheduling policy. Sampled output may change by arm.
    for name in ('prose', 'code'):
        for seed in (41, 42):
            for tau in (.9, .7, .5):
                measure(name, True, tau, .6, seed, 'mtp')
    save()


if __name__ == '__main__':
    main()
