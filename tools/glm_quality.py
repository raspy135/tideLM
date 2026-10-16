"""How far substituted routing moves GLM-5.3's next-token predictions from the exact model.

Three modes, each a separate boot (results under --out):

  gen    greedy answers to a fixed set of chat prompts (any residency settings); saves the token ids
         to seqs.json -- the text every configuration is then scored on.
  exact  MUSE_MISS=stream (every expert exact): teacher-forced prefill of each prompt + answer;
         saves, per answer position, the exact log-probability of the answer token, the exact
         top-1 and the exact top-20 (ids, log-probs).
  eval   the configuration under test: prefill the prompt (with whatever prefill settings are on),
         adapt as stream() does, then feed the answer one token at a time as decode steps
         (background periods every MUSE_BG_EVERY steps, adaptation every MUSE_ADAPT_EVERY), and
         score each step against exact.json: NLL of the answer token, top-1 agreement with exact,
         and KL(exact || this) over exact's top-20 (both renormalized there).

    MUSE_ENV="MUSE_GRAPHS=0 MUSE_BG=1 MUSE_HOT=1" tools/muse_pair_run.sh glm_quality.py --mode eval --name hot1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402

PROMPTS = [
    "東京観光案内をしてください。初めて東京に来る人向けに、おすすめの観光地を5つ、見どころと行き方つきで紹介して。",
    "東京で1日観光するモデルコースを、朝から夜まで時間つきで教えて。",
    "浅草と渋谷と新宿の見どころと、それぞれの街の雰囲気の違いを説明して。",
]
TOPN = 20


def encode_prompts(tpl):
    return [tpl.encode(tpl.render([{"role": "user", "content": p}], None, "low")) for p in PROMPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--mode", choices=("gen", "exact", "eval"), required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--out", default="/app/results/glm-quality")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    prompts = encode_prompts(tpl)

    if args.mode == "gen":
        seqs = []
        for ids in prompts:
            eng.reset()
            ans = eng.generate(ids, args.tokens, depth=0)
            seqs.append({"prompt": ids, "answer": ans})
            if eng.rank == 0:
                print("[gen]", repr(tpl.decode(ans)[:300]), flush=True)
        if eng.rank == 0:
            (out / "seqs.json").write_text(json.dumps(seqs))
        return

    # The files live on rank 0's host; rank 1 runs the same forwards from the broadcast copy.
    seqs = eng.comm.broadcast_obj(json.loads((out / "seqs.json").read_text()) if eng.rank == 0 else None)
    if args.mode == "exact":
        res = []
        for sq in seqs:
            p, a = sq["prompt"], sq["answer"]
            eng.reset()
            z = eng.forward(p + a[:-1], all_logits=True)[len(p) - 1:].float()     # predicts a[0..]
            lp = torch.log_softmax(z, -1)
            tgt = torch.tensor(a, device=lp.device)
            top = lp.topk(TOPN, -1)
            res.append({"lp": lp.gather(1, tgt[:, None])[:, 0].tolist(), "top1": top.indices[:, 0].tolist(),
                        "top_ids": top.indices.tolist(), "top_lp": top.values.tolist()})
        if eng.rank == 0:
            (out / "exact.json").write_text(json.dumps(res))
            print("[exact] mean NLL", sum(-x for r in res for x in r["lp"]) / sum(len(r["lp"]) for r in res))
        return

    ex = eng.comm.broadcast_obj(json.loads((out / "exact.json").read_text()) if eng.rank == 0 else None)
    if eng.demand is None:
        raise SystemExit("eval needs MUSE_ADAPT=1 and MUSE_MISS=substitute")
    ms = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    eng.miss_stats = ms
    rows, t0 = [], time.perf_counter()
    for sq, er in zip(seqs, ex):
        p, a = sq["prompt"], sq["answer"]
        eng.reset()
        z = eng.prefill(list(p))[-1]
        eng.adapt(eng.adapt_prefill_swaps)
        steps, next_adapt = 0, eng.adapt_every
        for i, tok in enumerate(a):
            lp = torch.log_softmax(z.float(), -1)
            ids = torch.tensor(er["top_ids"][i], device=lp.device)
            q = torch.log_softmax(lp[ids], -1)                                  # this, on exact's top-20
            pe = torch.log_softmax(torch.tensor(er["top_lp"][i], device=lp.device), -1)
            rows.append({"nll": -float(lp[tok]), "nll_exact": -er["lp"][i],
                         "agree": int(lp.argmax()) == er["top1"][i],
                         "kl": float((pe.exp() * (pe - q)).sum())})
            if i + 1 == len(a):
                break
            steps += 1
            if eng.adapt_every and steps >= next_adapt:
                eng.adapt(eng.adapt_decode_swaps)
                next_adapt += eng.adapt_every
            if eng.bg_on and steps % eng.bg_every == 0:
                eng.bg_step()
            z = eng.forward([tok], all_logits=False)[-1]
    if eng.rank == 0:
        n = len(rows)
        summary = {"name": args.name, "positions": n,
                   "nll": round(sum(r["nll"] for r in rows) / n, 4),
                   "nll_exact": round(sum(r["nll_exact"] for r in rows) / n, 4),
                   "top1_agree": round(sum(r["agree"] for r in rows) / n, 4),
                   "kl": round(sum(r["kl"] for r in rows) / n, 4),
                   "miss_mass": round(ms["mass_missed"] / max(1e-9, ms["mass"]), 4),
                   "hot": {k: round(v, 2) if isinstance(v, float) else v for k, v in eng.hot_stats.items()},
                   "bg": {k: round(v, 2) if isinstance(v, float) else v for k, v in eng.bg_stats.items()},
                   "seconds": round(time.perf_counter() - t0, 1)}
        (out / f"eval-{args.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
        print("[eval]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
