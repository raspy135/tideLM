"""Measure DSpark acceptance on the real weights for each drafter variant, target loaded once.

The settings published nothing fixes (engine/muse/dspark.py, DraftOptions) only move acceptance:
verification keeps the output the target's. This loads Muse once and, per variant, re-enables the
drafter and runs greedy generation from a few chat prompts, reporting accepted drafts per round
(the mean tokens a verify step yields is that + 1). Rank 0 prints a table and the best variant.

    MUSE_MODEL_DIR=... MUSE_ENV="MUSE_DRAFTER=/drafter" tools/muse_pair_run.sh \\
        muse_dspark_probe.py --drafter /drafter --tokens 160
(the runner mounts MUSE_MODEL_DIR at /models; mount the drafter via MUSE_EXTRA_MOUNT)
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")
import torch  # noqa: E402

from engine.muse.dspark import DraftOptions  # noqa: E402
from engine.muse.tp import MuseTP  # noqa: E402

PROMPTS = [
    [{"role": "user", "content": "Write a Python function that parses an ISO-8601 date string "
                                 "and returns the weekday name. Include a docstring and tests."}],
    [{"role": "user", "content": "Explain in two paragraphs why the sky is blue."}],
    [{"role": "user", "content": "Write a bash script that finds the 10 largest files under a "
                                 "directory and prints their sizes in human-readable form."}],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--variants", default="default,tap=output,attn_scale=linear,qk_scale=1.0,"
                    "embed_norm=1,bias_on=processed,post_ffn=centered")
    ap.add_argument("--effort", default="low")
    args = ap.parse_args()
    from muse_app import Template
    tpl = Template(args.model)
    os.environ.pop("MUSE_DRAFTER", None)
    eng = MuseTP(args.model)
    prompts = [tpl.encode(tpl.render(m, None, args.effort)) for m in PROMPTS]

    def run(opts: DraftOptions | None):
        if opts is None:
            eng.disable_drafter()
        else:
            eng.enable_drafter(args.drafter, opts=opts)
        eng.spec_stats = {"rounds": 0, "accepted": 0}
        outs, t0, n = [], time.perf_counter(), 0
        for p in prompts:
            eng.reset()
            o = eng.generate(p, args.tokens)
            outs.append(o)
            n += len(o)
        dt = time.perf_counter() - t0
        return outs, dict(eng.spec_stats), n / dt

    base, _, base_tps = run(None)
    rows = [{"variant": "no drafter", "tok_s": round(base_tps, 2)}]
    if eng.rank == 0:
        for o in base:
            print("[probe] sample:", repr(tpl.decode(o)[:300]), flush=True)
    for v in args.variants.split(","):
        opts = DraftOptions()
        if v != "default":
            for kv in v.split("+"):
                k, val = kv.split("=")
                t = type(getattr(opts, k))
                setattr(opts, k, (val == "1") if t is bool else t(val))
        outs, st, tps = run(opts)
        same = sum(a == b for a, b in zip(outs, base))
        acc = st["accepted"] / max(1, st["rounds"])
        rows.append({"variant": v, "accepted_per_round": round(acc, 3),
                     "tokens_per_step": round(acc + 1, 3), "tok_s": round(tps, 2),
                     "rounds": st["rounds"], "outputs_equal_plain": f"{same}/{len(outs)}"})
        if eng.rank == 0:
            print(f"[probe] {json.dumps(rows[-1])}", flush=True)
    if eng.rank == 0:
        print(json.dumps(rows, indent=1), flush=True)
        best = max(rows[1:], key=lambda r: r["accepted_per_round"])
        print(f"[probe] best: {best['variant']} ({best['tokens_per_step']} tokens a step)", flush=True)


if __name__ == "__main__":
    main()
