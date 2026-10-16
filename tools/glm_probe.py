"""GLM-5.3 on the real weights: greedy generation from three chat prompts, plain and with MTP
drafts at several depths. Prints text samples, tok/s (prefill excluded) and accepted drafts a round.
    MUSE_MODEL_DIR=~/models/GLM-5.3-... tools/muse_pair_run.sh glm_probe.py --tokens 128"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402

PROMPTS = [
    [{"role": "user", "content": "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include a docstring and tests."}],
    [{"role": "user", "content": "Explain in two short paragraphs why the sky is blue."}],
    [{"role": "user", "content": "Write a bash script that prints the 10 largest files under a directory with human-readable sizes."}],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--depths", default="0,1,2,3")
    ap.add_argument("--effort", default="low")
    args = ap.parse_args()
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    prompts = [tpl.encode(tpl.render(m, None, args.effort)) for m in PROMPTS]
    rows = []
    for depth in [int(x) for x in args.depths.split(",")]:
        if depth and not eng.mtp_on:
            continue
        eng.spec_stats = {"rounds": 0, "accepted": 0}
        n, dt, outs = 0, 0.0, []
        for p in prompts:
            eng.reset()
            it = eng.stream(p, args.tokens, depth=depth)
            first = next(it)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = [first] + list(it)
            torch.cuda.synchronize()
            dt += time.perf_counter() - t0
            n += len(out) - 1
            outs.append(out)
        st = eng.spec_stats
        row = {"depth": depth, "decode_tok_s": round(n / dt, 2),
               "accepted_per_round": round(st["accepted"] / max(1, st["rounds"]), 3) if depth else None}
        rows.append(row)
        if eng.rank == 0:
            print(f"[probe] {json.dumps(row)}", flush=True)
            if depth in (0, max(int(x) for x in args.depths.split(","))):
                for o in outs:
                    print("[probe] sample:", repr(tpl.decode(o)[:400]), flush=True)
    if eng.rank == 0:
        print(json.dumps(rows, indent=1), flush=True)


if __name__ == "__main__":
    main()
