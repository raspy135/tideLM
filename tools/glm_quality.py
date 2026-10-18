"""How far substituted routing moves GLM-5.3's next-token predictions from the exact model.

Three modes, each a separate boot (results under --out):

  gen    greedy answers to a fixed set of chat prompts (any residency settings); saves the token ids
         to seqs.json -- the text every configuration is then scored on.
  exact  GLM_MISS=stream (every expert exact): teacher-forced prefill of each prompt + answer;
         saves, per answer position, the exact log-probability of the answer token, the exact
         top-1 and the exact top-20 (ids, log-probs).
  eval   the configuration under test: prefill the prompt (with whatever prefill settings are on),
         adapt as stream() does, then feed the answer one token at a time as decode steps
         (background periods every GLM_BG_EVERY steps, adaptation every GLM_ADAPT_EVERY), and
         score each step against exact.json: NLL of the answer token, top-1 agreement with exact,
         and KL(exact || this) over exact's top-20 (both renormalized there).

    GLM_ENV="GLM_GRAPHS=0 GLM_BG=1 GLM_HOT=1" tools/muse_pair_run.sh glm_quality.py --mode eval --name hot1
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
# --set switch: a second turn on another topic after each Tokyo answer, prefilled as serving does
# it (the history first, then only the new turn on top of the reused prefix).
SWITCH = [
    "簡単な棒グラフのHTMLを作って。カッコいいデザインがいいです。",
    "腰痛になったらどうすれば良い？",
    "PythonでCSVファイルを読み込んで、列ごとの平均を出すスクリプトを書いて。",
]
TOPN = 20


def encode_prompts(tpl):
    return [tpl.encode(tpl.render([{"role": "user", "content": p}], None, "low")) for p in PROMPTS]


def encode_natural(tpl, n_answer):
    """--set natural: text neither model wrote (corpus/heldout_eval.jsonl, every 9th document, 12
    of them): the first 64 tokens are the prompt, the next n_answer are scored. NLL here compares
    formats fairly; the Tokyo answers were written by the MXFP4 model itself."""
    docs = [json.loads(l) for l in open("/app/corpus/heldout_eval.jsonl")][::9][:12]
    out = []
    for d in docs:
        ids = tpl.encode(d["text"])
        out.append({"prompt": ids[:64], "answer": ids[64:64 + n_answer], "id": d["id"]})
    return out


def encode_long(tpl, n_answer):
    """--set long: the held-out corpus joined into one text; the prompt is its first 6K / 12K /
    18K tokens and the next n_answer are scored -- past index_topk (2,048), so the indexer picks
    which positions attention reads (what an FP8 key cache or the decode range can change)."""
    docs = [json.loads(l)["text"] for l in open("/app/corpus/heldout_eval.jsonl")]
    ids = tpl.encode("\n\n".join(docs))
    return [{"prompt": ids[:w], "answer": ids[w:w + n_answer], "id": f"long-{w}"} for w in (6000, 12000, 18000)]


def encode_switch(tpl, tokyo):
    """(ids, cut) per SWITCH turn: the Tokyo exchange (reasoning dropped, as clients send it back)
    then the new question; ids[:cut] is what a prefix cache would hold."""
    out = []
    for q, first, sq in zip(SWITCH, PROMPTS, tokyo):
        ans = tpl.tok.decode(sq["answer"], skip_special_tokens=True).split("</think>")[-1].strip()
        hist = [{"role": "user", "content": first}, {"role": "assistant", "content": ans}]
        ids = tpl.encode(tpl.render(hist + [{"role": "user", "content": q}], None, "low"))
        alt = tpl.encode(tpl.render(hist + [{"role": "user", "content": "x"}], None, "low"))
        cut = next(i for i, (u, v) in enumerate(zip(ids, alt)) if u != v)
        out.append((ids, cut))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--mode", choices=("gen", "exact", "eval"), required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--out", default="/app/results/glm-quality")
    ap.add_argument("--set", choices=("tokyo", "switch", "natural", "long"), default="tokyo")
    ap.add_argument("--ref", default="", help="exact mode: written as exact<-set><-ref>.json; eval: the one to score against")
    ap.add_argument("--against", default="",
                    help="exact mode: also score these exact logits against exact<-set>-<against>.json (KL over "
                         "its top-20, top-1 agreement, NLL): one numerics change, every expert exact on both sides")
    ap.add_argument("--rows", type=int, default=1,
                    help="eval: feed the answer N tokens a step (3 = an MTP depth-2 verify block, graphed "
                         "when GLM_MTP_DEPTH=N-1), so per-step read caps and thresholds act as in serving")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    suffix = "" if args.set == "tokyo" else f"-{args.set}"
    if args.set in ("natural", "long"):
        prompts = None
    elif args.set == "tokyo":
        prompts = [(ids, 0) for ids in encode_prompts(tpl)]
    else:
        tokyo = eng.comm.broadcast_obj(json.loads((out / "seqs.json").read_text()) if eng.rank == 0 else None)
        prompts = encode_switch(tpl, tokyo)

    if args.mode == "gen":
        seqs = []
        for ids, cut in prompts:
            eng.reset()
            ans = eng.generate(ids, args.tokens, depth=0)
            seqs.append({"prompt": ids, "answer": ans, "cut": cut})
            if eng.rank == 0:
                print("[gen]", repr(tpl.decode(ans)[:300]), flush=True)
        if eng.rank == 0:
            (out / f"seqs{suffix}.json").write_text(json.dumps(seqs))
        return

    # The files live on rank 0's host; rank 1 runs the same forwards from the broadcast copy.
    if args.set == "natural":
        seqs = encode_natural(tpl, args.tokens)
    elif args.set == "long":
        seqs = encode_long(tpl, args.tokens)
    else:
        seqs = eng.comm.broadcast_obj(json.loads((out / f"seqs{suffix}.json").read_text()) if eng.rank == 0 else None)
    if args.mode == "exact":
        res, cmp = [], []
        vs = None
        if args.against:
            vp = out / f"exact{suffix}-{args.against}.json"
            vs = eng.comm.broadcast_obj(json.loads(vp.read_text()) if eng.rank == 0 else None)
        for k, sq in enumerate(seqs):
            p, a = sq["prompt"], sq["answer"]
            eng.reset()
            z = eng.forward(p + a[:-1], all_logits=True)[len(p) - 1:].float()     # predicts a[0..]
            lp = torch.log_softmax(z, -1)
            tgt = torch.tensor(a, device=lp.device)
            top = lp.topk(TOPN, -1)
            res.append({"lp": lp.gather(1, tgt[:, None])[:, 0].tolist(), "top1": top.indices[:, 0].tolist(),
                        "top_ids": top.indices.tolist(), "top_lp": top.values.tolist()})
            if vs is not None:
                er = vs[k]
                ids = torch.tensor(er["top_ids"], device=lp.device)
                q = torch.log_softmax(lp.gather(1, ids), -1)                      # this, on its top-20
                pe = torch.log_softmax(torch.tensor(er["top_lp"], device=lp.device), -1)
                cmp.append((float((pe.exp() * (pe - q)).sum()), int((lp.argmax(-1) == torch.tensor(er["top1"], device=lp.device)).sum()),
                            float(-lp.gather(1, tgt[:, None]).sum()), -sum(er["lp"]), len(a)))
        if eng.rank == 0:
            (out / f"exact{suffix}{'-' + args.ref if args.ref else ''}.json").write_text(json.dumps(res))
            print("[exact] mean NLL", sum(-x for r in res for x in r["lp"]) / sum(len(r["lp"]) for r in res))
            if cmp:
                n = sum(c[4] for c in cmp)
                print("[against] " + json.dumps({"name": args.name, "against": args.against, "positions": n,
                                                  "kl": round(sum(c[0] for c in cmp) / n, 5),
                                                  "top1_agree": round(sum(c[1] for c in cmp) / n, 4),
                                                  "nll": round(sum(c[2] for c in cmp) / n, 4),
                                                  "nll_against": round(sum(c[3] for c in cmp) / n, 4)}))
        return

    ref = out / f"exact{suffix}{'-' + args.ref if args.ref else ''}.json"
    ex = eng.comm.broadcast_obj(json.loads(ref.read_text()) if eng.rank == 0 and ref.exists() else None)
    if ex is None:                       # NLL only (--set natural without an exact reference)
        ex = [None] * len(seqs)
    if eng.demand is None:
        raise SystemExit("eval needs GLM_ADAPT=1 and GLM_MISS=substitute")
    ms = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    eng.miss_stats = ms
    rows, t0 = [], time.perf_counter()
    for sq, er in zip(seqs, ex):
        p, a = sq["prompt"], sq["answer"]
        eng.reset()
        if sq.get("cut"):
            eng.prefill(list(p[:sq["cut"]]))                 # the earlier turn, as it was served
            eng.adapt(eng.adapt_prefill_swaps)
        z = eng.prefill(list(p))[-1]                          # reuses p[:cut] when there is one
        eng.adapt(eng.adapt_prefill_swaps)
        steps, next_adapt = 0, eng.adapt_every
        zs, fed = [z], 0
        for i, tok in enumerate(a):
            z = zs[i - fed]
            lp = torch.log_softmax(z.float(), -1)
            if er is None:
                rows.append({"nll": -float(lp[tok]), "nll_exact": 0.0, "agree": False, "kl": 0.0})
            else:
                ids = torch.tensor(er["top_ids"][i], device=lp.device)
                q = torch.log_softmax(lp[ids], -1)                              # this, on exact's top-20
                pe = torch.log_softmax(torch.tensor(er["top_lp"][i], device=lp.device), -1)
                rows.append({"nll": -float(lp[tok]), "nll_exact": -er["lp"][i],
                             "agree": int(lp.argmax()) == er["top1"][i],
                             "kl": float((pe.exp() * (pe - q)).sum())})
            if i + 1 == len(a):
                break
            if i + 1 < fed + len(zs):
                continue                                  # the last block still has rows to score
            if eng.adapt_every and i + 1 >= next_adapt:   # by tokens, as stream() counts emitted
                eng.adapt(eng.adapt_decode_swaps)
                next_adapt += eng.adapt_every
            steps += 1                                    # by steps, as stream() counts rounds
            if eng.bg_on and steps % eng.bg_every == 0:
                eng.bg_step()
            block = a[i:i + args.rows][:len(a) - 1 - i]   # a[i] on; the last answer token is never fed
            zs, fed = list(eng.forward(block, all_logits=True)), i + 1
    eng.miss_totals()
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
