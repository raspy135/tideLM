"""Exact routing trace of Muse Spark on the pair: which experts a workload uses.

Runs the corpus through the TP2 engine with misses streamed (MUSE_MISS=stream: every pick is
the router's own), records every layer's top-8 and router weights, and writes (rank 0):

  stats.json      per (layer, expert): picks and router-weight mass, overall and per category;
                  distinct experts per block of b consecutive tokens (b = 1, 2, 4, 6, 8), the
                  multiplier a b-row verify step pays in expert reads;
                  miss rate and missed mass against resident fraction, ranked on even documents
                  and measured on odd ones (global ranking by mass across layers).
  resident-<N>.json  the top-N experts by mass, for MUSE_RESIDENT at an N-slot budget.

    MUSE_MODEL_DIR=~/models/Muse-...-MXFP4 tools/muse_pair_run.sh muse_trace.py \\
        --corpus /app/corpus/trace_corpus_v3.jsonl --tokens 1024 --docs 64
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402

os.environ.setdefault("MUSE_MISS", "stream")
from engine.muse.tp import MuseTP  # noqa: E402


def load_docs(path, limit):
    docs = []
    for line in open(path):
        d = json.loads(line)
        docs.append((d.get("category", "general"), d["text"]))
    return docs[:limit]


class _Bytes:
    """Byte-level stand-in for synthetic checkpoints that ship no tokenizer."""
    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Enc(list(text.encode("utf-8")))


def load_tokenizer(model_dir):
    path = Path(model_dir) / "tokenizer.json"
    if not path.exists():
        return _Bytes()
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--corpus", default="/app/corpus/trace_corpus_v3.jsonl")
    ap.add_argument("--tokens", type=int, default=1024, help="tokens per document (prefix)")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--out", default=f"/app/results/muse-trace-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--budgets", default="4000,5000,6000,7000,8000")
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    eng = MuseTP(args.model, max_seq=max(args.tokens + 16, 2048), chunk=1024)
    cfg = eng.cfg
    docs = load_docs(args.corpus, args.docs)
    L, E, k = cfg.layers, cfg.experts, cfg.topk
    picks = torch.zeros(2, L, E, dtype=torch.float64)       # [even/odd doc, layer, expert]
    mass = torch.zeros(2, L, E, dtype=torch.float64)
    by_cat = defaultdict(lambda: torch.zeros(L, E, dtype=torch.float64))
    blocks = {b: [0.0, 0] for b in (1, 2, 4, 6, 8)}
    t0 = time.perf_counter()
    n_tok = 0
    for i, (cat, text) in enumerate(docs):
        ids = [cfg.bos] + tok.encode(text, add_special_tokens=False).ids
        ids = ids[:args.tokens]
        if len(ids) < 16:
            continue
        eng.reset()
        eng.record = []
        eng.forward(ids, all_logits=False)
        rec, eng.record = eng.record, None
        n_tok += len(ids)
        if eng.rank != 0:
            continue
        per_layer = defaultdict(list)
        for layer, ids_t, w_t in rec:
            per_layer[layer].append((ids_t, w_t))
        for layer, parts in per_layer.items():
            ids_l = torch.cat([p[0] for p in parts])
            w_l = torch.cat([p[1] for p in parts]).double()
            flat = ids_l.reshape(-1)
            picks[i % 2, layer].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
            mass[i % 2, layer].index_add_(0, flat, w_l.reshape(-1))
            by_cat[cat][layer].index_add_(0, flat, w_l.reshape(-1))
            for b in blocks:
                n = ids_l.shape[0] // b
                if n:
                    blk = ids_l[:n * b].reshape(n, b * k)
                    distinct = torch.tensor([len(set(r.tolist())) for r in blk], dtype=torch.float64)
                    blocks[b][0] += float(distinct.sum())
                    blocks[b][1] += n
        if eng.rank == 0:
            el = time.perf_counter() - t0
            print(f"[trace] doc {i + 1}/{len(docs)} {cat} {len(ids)} tok, {n_tok / el:.1f} tok/s, "
                  f"{eng.stats['stream_loads']} expert loads ({eng.stats['stream_s']:.0f} s)", flush=True)
    if eng.rank != 0:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    distinct = {b: (v[0] / v[1] if v[1] else None) for b, v in blocks.items()}
    # Miss rate against a global budget: rank by even-doc mass, measure on odd-doc picks.
    rank_mass = mass[0].reshape(-1)
    order = torch.argsort(rank_mass, descending=True)
    test_picks, test_mass = picks[1].reshape(-1), mass[1].reshape(-1)
    curve = []
    for frac in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        n = int(frac * L * E)
        res = torch.zeros(L * E, dtype=torch.bool)
        res[order[:n]] = True
        curve.append({"fraction": frac, "slots": n,
                      "miss_rate": round(float(test_picks[~res].sum() / test_picks.sum()), 4),
                      "mass_missed": round(float(test_mass[~res].sum() / test_mass.sum()), 4)})
    total_mass = mass.sum(0)
    stats = {"model": args.model, "corpus": args.corpus, "docs": len(docs), "tokens": n_tok,
             "distinct_experts_per_block": distinct,
             "verify_expert_multiplier": {b: (d / distinct[1] if d and distinct[1] else None)
                                          for b, d in distinct.items()},
             "miss_curve_heldout": curve,
             "mass": total_mass.tolist(), "picks": picks.sum(0).tolist(),
             "by_category": {c: v.tolist() for c, v in by_cat.items()},
             "stream_loads": eng.stats["stream_loads"], "seconds": round(time.perf_counter() - t0, 1)}
    (out / "stats.json").write_text(json.dumps(stats))
    flat_order = torch.argsort(total_mass.reshape(-1), descending=True)
    for budget in [int(b) for b in args.budgets.split(",")]:
        chosen = flat_order[:budget].tolist()
        resident = defaultdict(list)
        for idx in chosen:
            resident[idx // E].append(idx % E)
        (out / f"resident-{budget}.json").write_text(json.dumps(
            {"resident": {str(l): sorted(v) for l, v in sorted(resident.items())},
             "source": str(out / "stats.json"), "budget": budget}))
    print(json.dumps({k: stats[k] for k in ("docs", "tokens", "distinct_experts_per_block",
                                            "verify_expert_multiplier", "miss_curve_heldout",
                                            "stream_loads", "seconds")}, indent=1), flush=True)
    print(f"[trace] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
