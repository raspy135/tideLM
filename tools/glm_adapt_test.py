"""Miss rate on one short prompt as residency adapts: generate, adapt, repeat; per round the
share of picks (and router mass) the masked router substituted in prefill and decode.

``--hist`` also bins decode's missed picks by router weight (share of the token's routed mass) and
prints, per threshold, the streams a token would need to fetch every miss at or above it and the
mass left missed -- the trade a partial-streaming mode would make. ``--read-bench`` times cold
reads of non-resident experts (1/2/4/8 at once), the cost of one such stream."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402

PROMPT = [{"role": "user", "content": "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include a docstring and tests."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--prompt-file", default="", help="ask to explain this file (first --prompt-chars) instead")
    ap.add_argument("--prompt-chars", type=int, default=12000)
    ap.add_argument("--read-bench", action="store_true")
    ap.add_argument("--dump", default="", help="save decode routing (unmasked ids, weights, missed, "
                    "ids predicted 1/2/4/8 layers ahead) per round to DUMP.r<round>.pt (rank 0)")
    args = ap.parse_args()
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    if eng.demand is None:
        eng.enable_adaptation()
    prompt = PROMPT
    if args.prompt_file:
        text = open(args.prompt_file, errors="ignore").read()[:args.prompt_chars]
        prompt = [{"role": "user", "content": f"Here is a file:\n\n```\n{text}\n```\n\nExplain what it does."}]
    ids = tpl.encode(tpl.render(prompt, None, "low"))
    ms = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    eng.miss_stats = ms

    def share(a, b):
        p = b["picks"] - a["picks"]
        return {"picks": round((b["missed"] - a["missed"]) / max(1, p), 4),
                "mass": round((b["mass_missed"] - a["mass_missed"]) / max(1e-9, b["mass"] - a["mass"]), 4)}
    for r in range(args.rounds):
        eng.reset()
        eng.adapt_log = []
        s0 = dict(eng.miss_totals())
        it = eng.stream(ids, args.tokens, depth=0)
        eng.miss_hist = None
        t_first = time.perf_counter()
        out = [next(it)]
        t_first = time.perf_counter() - t_first
        s1 = dict(eng.miss_totals())
        if args.hist:
            eng.miss_hist = torch.zeros((2, 50), dtype=torch.float32, device=eng.dev)
        if args.dump:
            eng.route_dump = []
        t0 = time.perf_counter()
        windows, w0 = [], dict(eng.miss_totals())
        for t in it:
            out.append(t)
            if len(out) % args.window == 0:
                w1 = dict(eng.miss_totals())
                windows.append(share(w0, w1)["mass"])
                w0 = w1
        dt = time.perf_counter() - t0
        s2 = dict(eng.miss_totals())
        if eng.rank == 0:
            print(json.dumps({"round": r, "prefill_tokens": len(ids), "ttft_s": round(t_first, 2),
                              "prefetch": dict(getattr(eng, "pf_stats", {})), "prefill_missed": share(s0, s1),
                              "decode_missed": share(s1, s2), f"decode_mass_missed_per_{args.window}": windows,
                              "decode_tok_s": round((len(out) - 1) / dt, 2), "adaptations": eng.adapt_log,
                              "background": {k: round(v, 2) if isinstance(v, float) else v
                                             for k, v in getattr(eng, "bg_stats", {}).items()}}), flush=True)
            print("[text]", repr(tpl.decode(out)[:700]), flush=True)
        if args.dump:
            d, eng.route_dump = eng.route_dump, None
            if eng.rank == 0:
                torch.save([(L, i.cpu(), w.cpu(), m.cpu(), {k: v.cpu() for k, v in a.items()}) for L, i, w, m, a in d],
                           f"{args.dump}.r{r}.pt")
        if args.hist:
            h = eng.miss_hist.cpu()
            eng.miss_hist = None
            if eng.rank == 0:
                n_tok = len(out) - 1
                nmoe = len(eng.moe_layers) - 1                      # the MTP layer is idle at depth 0
                cnt, mass = h[0].flip(0).cumsum(0).flip(0), h[1].flip(0).cumsum(0).flip(0)
                tot_missed = float(h[1].sum())
                rows = []
                for t in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.3):
                    b = round(t * 50)
                    rows.append({"weight_ge": t, "streams_per_token": round(float(cnt[b]) / n_tok, 2),
                                 "mass_left_missed": round((tot_missed - float(mass[b])) / (n_tok * nmoe), 4)})
                print(json.dumps({"round": r, "missed_picks_per_token": round(float(h[0].sum()) / n_tok, 1),
                                  "mass_missed": round(tot_missed / (n_tok * nmoe), 4), "partial_stream": rows,
                                  "hist_count": [int(v) for v in h[0]], "hist_mass": [round(float(v), 2) for v in h[1]]}),
                      flush=True)
    if args.read_bench:
        import random
        rng = random.Random(1)
        res = {}
        for n in (1, 2, 4, 8):
            times = []
            for _ in range(12):
                L = rng.choice(eng.moe_layers[:-1])
                cold = [e for e in range(eng.cfg.experts) if int(eng.lut[L, e]) < 0]
                es = rng.sample(cold, n)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                parts = eng._read_many(L, es)
                dev = [[t.to(eng.dev, non_blocking=True) for wt in p for t in wt] for p in parts]
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
                del dev
            times.sort()
            res[n] = {"median_ms": round(times[len(times) // 2], 2), "min_ms": round(times[0], 2), "max_ms": round(times[-1], 2)}
        print(json.dumps({"rank": eng.rank, "cold_read_ms": res}), flush=True)


if __name__ == "__main__":
    main()
