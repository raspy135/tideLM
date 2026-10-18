"""Grow the GLM routing-statistics DB (GLM_STATS_DB) by running documents through generation.

Each document chunk (~1,200 tokens) goes in as a chat prompt -- code: "explain this file and
suggest one improvement", prose: "summarize this" -- and 64 tokens are generated, so the DB
records prefill and decode routing. Adaptation stays on (small swap budget), so the hidden
states stay close to what serving computes. ``--seed`` first loads a routing trace's stats.json
(exact routing, tools/glm_trace.py) into an empty DB.

    GLM_MODEL_DIR=... GLM_ENV="GLM_STATS_DB=/app/results/glm-stats/db.pt GLM_ADAPT_PREFILL_SWAPS=256" \\
        tools/muse_pair_run.sh glm_train_db.py --seed /app/results/glm-trace-v3/stats.json
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402


def chunks(text, n_chars):
    for i in range(0, len(text), n_chars):
        piece = text[i:i + n_chars]
        if len(piece) > 400:
            yield piece


def documents(limit, chunk_chars, seed):
    docs = []
    for path in ("/app/corpus/trace_corpus_v2.jsonl", "/app/corpus/trace_corpus_v3.jsonl"):
        for line in open(path):
            d = json.loads(line)
            text = re.sub(r"<｜[^｜]*｜>", "\n", d["text"])
            kind = "code" if d.get("category") == "coding" else "prose"
            docs += [(kind, c) for c in chunks(text, chunk_chars)]
    for pattern, kind in (("/app/engine/**/*.py", "code"), ("/app/tools/*.py", "code"),
                          ("/app/tools/*.cu", "code"), ("/app/server/*.py", "code"),
                          ("/app/corpus/sources/*.py", "code"), ("/app/corpus/sources/*.md", "prose"),
                          ("/app/corpus/sources/*.txt", "prose"), ("/app/corpus/sources/prose/*", "prose"),
                          ("/app/corpus/sources/web/*", "prose")):
        for f in sorted(glob.glob(pattern, recursive=True)):
            try:
                text = Path(f).read_text(errors="ignore")
            except (OSError, IsADirectoryError):
                continue
            docs += [(kind, c) for c in chunks(text, chunk_chars)]
    random.Random(seed).shuffle(docs)
    return docs[:limit]


def prompt(kind, text):
    if kind == "code":
        return [{"role": "user", "content": f"Here is part of a file from a codebase:\n\n```\n{text}\n```\n\n"
                 "Explain what it does and suggest one concrete improvement."}]
    return [{"role": "user", "content": f"Summarize the following text in a few sentences:\n\n{text}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--chunk-chars", type=int, default=4800)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", default="", help="routing trace stats.json to start an empty DB from")
    args = ap.parse_args()
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    if eng.demand is None:
        raise SystemExit("needs GLM_ADAPT=1 and GLM_MISS=substitute")
    if args.seed and eng.rank == 0 and eng.db_tokens == 0 and Path(args.seed).exists():
        st = json.loads(Path(args.seed).read_text())
        mass = torch.tensor(st["mass"], dtype=torch.float64)
        if tuple(mass.shape) == tuple(eng.db_mass.shape):
            eng.db_mass += mass
            eng.db_tokens += int(st["tokens"])
            eng.save_db()
            print(f"[train] seeded the DB from {args.seed} ({st['tokens']} tokens)", flush=True)
    docs = documents(args.docs, args.chunk_chars, 0)
    t0, n_tok = time.perf_counter(), 0
    ms = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    for i, (kind, text) in enumerate(docs):
        ids = tpl.encode(tpl.render(prompt(kind, text), None, "low"))[: eng.max_seq - args.tokens - 8]
        eng.reset()
        out = eng.generate(ids, args.tokens, depth=0)
        n_tok += len(ids) + len(out)
        if eng.rank == 0 and (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            at = eng.adapt_totals
            print(f"[train] {i + 1}/{len(docs)} docs, {n_tok} tokens, {n_tok / el:.0f} tok/s, "
                  f"DB {eng.db_tokens} tokens, swaps {at['swaps']} (churn {at['churn']}, "
                  f"{at['churn'] / max(1, at['swaps']):.1%})", flush=True)
    eng.adapt(0)          # fold the last request's demand into the DB
    if eng.rank == 0:
        print(f"[train] done: DB {eng.db_tokens} tokens at {eng.db_path}", flush=True)


if __name__ == "__main__":
    main()
