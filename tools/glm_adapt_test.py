"""Miss rate on one short prompt as residency adapts: generate, adapt, repeat; per round the
share of picks (and router mass) the masked router substituted in prefill and decode."""
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
    args = ap.parse_args()
    tpl = GlmTemplate(args.model)
    eng = GlmTP(args.model)
    if eng.demand is None:
        eng.enable_adaptation()
    ids = tpl.encode(tpl.render(PROMPT, None, "low"))
    ms = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    eng.miss_stats = ms

    def share(a, b):
        p = b["picks"] - a["picks"]
        return {"picks": round((b["missed"] - a["missed"]) / max(1, p), 4),
                "mass": round((b["mass_missed"] - a["mass_missed"]) / max(1e-9, b["mass"] - a["mass"]), 4)}
    for r in range(args.rounds):
        eng.reset()
        eng.adapt_log = []
        s0 = dict(ms)
        it = eng.stream(ids, args.tokens, depth=0)
        out = [next(it)]
        s1 = dict(ms)
        t0 = time.perf_counter()
        windows, w0 = [], dict(ms)
        for t in it:
            out.append(t)
            if len(out) % args.window == 0:
                w1 = dict(ms)
                windows.append(share(w0, w1)["mass"])
                w0 = w1
        dt = time.perf_counter() - t0
        s2 = dict(ms)
        if eng.rank == 0:
            print(json.dumps({"round": r, "prefill_tokens": len(ids), "prefill_missed": share(s0, s1),
                              "decode_missed": share(s1, s2), f"decode_mass_missed_per_{args.window}": windows,
                              "decode_tok_s": round((len(out) - 1) / dt, 2), "adaptations": eng.adapt_log}), flush=True)
            print("[text]", repr(tpl.decode(out)[:700]), flush=True)


if __name__ == "__main__":
    main()
