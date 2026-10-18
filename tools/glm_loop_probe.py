"""Is a served repetition loop the engine's approximation or the model's own? Replays a request.

Boot it twice -- the serving configuration (GLM_MISS=substitute, adaptation and hot reads as
served) and GLM_MISS=stream (every routed expert exact) -- on the same case:

  --prompt-tok  int32 token ids of the served prompt (a prefix-store segment's .tok file holds them)
  --case        JSON: {"reasoning": the reasoning text, "prefixes": [answer text cut just before
                and inside the loop, ...]}

For each prefix: the next-token distribution after prompt + reasoning + </think> + prefix (top 8,
and the probability of the token the loop repeated), then --samples continuations of --tokens
sampled as the server samples (temperature, top-p, MTP rounds), each flagged when it falls into a
loop (a cycle of 1-8 tokens repeated --rep times in a row). --full also samples whole answers from
the prompt alone. Results: <out>/loop-<name>.json and [loop] lines.

    GLM_ENV="... GLM_MISS=stream" tools/muse_pair_run.sh glm_loop_probe.py --prompt-tok /app/results/x.tok \\
        --case /app/results/case.json --name exact
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


def loop_start(ids, rep, max_period=8):
    """Index where a cycle of <= max_period tokens starts repeating `rep` times in a row, or None."""
    for end in range(len(ids)):
        for p in range(1, max_period + 1):
            n = p * rep
            if end + 1 >= n and all(ids[end - i] == ids[end - i - p] for i in range(n - p)):
                return end + 1 - n
    return None


def max_reps(ids, max_period=8):
    """The most times any cycle of <= max_period tokens repeats back to back."""
    best = 1
    for p in range(1, max_period + 1):
        run = 0
        for i in range(p, len(ids)):
            run = run + 1 if ids[i] == ids[i - p] else 0
            best = max(best, 1 + (run // p))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--prompt-tok", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--full", type=int, default=0, help="also this many whole answers from the prompt alone")
    ap.add_argument("--full-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep", type=int, default=6)
    ap.add_argument("--cut", type=int, default=0,
                    help="prefill the prompt's first N tokens on their own first, as a chat turn finds its "
                         "history cached: the rest is new text (GLM_PREFILL_TWO_PASS acts on it)")
    ap.add_argument("--out", default="/app/results/glm-loop")
    args = ap.parse_args()
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    data = None
    if eng.rank == 0:
        from array import array
        ids = array("i")
        ids.frombytes(Path(args.prompt_tok).read_bytes())
        data = {"prompt": ids.tolist(),
                "case": json.loads(Path(args.case).read_text())}
    data = eng.comm.broadcast_obj(data)
    prompt, case = data["prompt"], data["case"]
    head = prompt + tpl.encode(case["reasoning"] + "</think>")
    log = (lambda *a: print(*a, flush=True)) if eng.rank == 0 else (lambda *a: None)
    res = {"name": args.name, "miss": eng.miss, "prefixes": [], "full": []}

    def sample(ids, n, seed):
        out = []
        for tok in eng.stream(ids, n, temperature=args.temperature, top_p=args.top_p, seed=seed):
            out.append(tok)
        return out

    for k, pre in enumerate(case["prefixes"]):
        ids = head + tpl.encode(pre)
        eng.reset()
        if args.cut:
            eng.prefill(list(ids[:args.cut]))
        z = eng.prefill(list(ids))[-1].float()
        p = torch.softmax(z / args.temperature, -1)
        top = p.topk(8)
        entry = {"prefix": pre[-40:], "top": [(tpl.decode([int(i)]), round(float(v), 4))
                                              for v, i in zip(top.values, top.indices)]}
        if case.get("loop_token"):
            lt = tpl.encode(case["loop_token"])
            entry["p_loop_token"] = round(float(p[lt[0]]), 4)
        t0 = time.perf_counter()
        runs = []
        for s in range(args.samples):
            # No reset: each sample reuses the prompt's cached rows (the same bytes a fresh prefill
            # writes) -- with every expert exact a 4K-token prefill is about a minute of reads.
            out = sample(ids, args.tokens, 1000 * k + s)
            at = loop_start(out, args.rep)
            runs.append({"loop_at": at, "max_reps": max_reps(out), "text": tpl.decode(out)})
        entry["loops"] = sum(r["loop_at"] is not None for r in runs)
        entry["max_reps"] = [r["max_reps"] for r in runs]
        entry["samples"] = runs
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        res["prefixes"].append(entry)
        log(f"[loop] {args.name} prefix {k} ...{pre[-24:]!r}: loops {entry['loops']}/{args.samples}, "
            f"max reps {entry['max_reps']}, {args.samples * args.tokens / entry['seconds']:.1f} tok/s, "
            f"p(loop token) {entry.get('p_loop_token')}, top {entry['top'][:4]}")
        for r in runs:
            log(f"[loop]   {'LOOP' if r['loop_at'] is not None else 'ok  '} {r['text'][:110]!r}")
    for s in range(args.full):
        out = sample(prompt, args.full_tokens, 7000 + s)
        at = loop_start(out, args.rep)
        res["full"].append({"loop_at": at, "n": len(out), "text": tpl.decode(out)})
        log(f"[loop] {args.name} full {s}: {'LOOP at ' + str(at) if at is not None else 'ok'} "
            f"({len(out)} tokens) {tpl.decode(out)[-100:]!r}")
    if res["full"]:
        log(f"[loop] {args.name} full: loops {sum(r['loop_at'] is not None for r in res['full'])}/{args.full}")
    if eng.rank == 0:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"loop-{args.name}.json").write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
