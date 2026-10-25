"""GLM_LOOKUP A/B in one boot: copy rounds (prompt-lookup drafts, engine/glm/tp.py) off vs on.

Per prompt, greedy, arms interleaved off/on/off/on (incumbent first: the first config in a process
runs slow); the switch is the engine attribute, set identically on both ranks. Decode tok/s is
first token to last; prefill is excluded (the repeats reuse the prompt's KV anyway). Reports per
arm tok/s, rounds, tokens a round, copy rounds and their accepted drafts, and whether the outputs
match (off vs off is the drift baseline: hot reads keep changing residency between runs).

    tools/muse_pair_run.sh bench_glm_lookup.py --out /app/results/<dir>/lookup.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")
os.environ.setdefault("GLM_LOOKUP", "8")
from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402

BASH = [{"type": "function", "function": {
    "name": "Bash", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                    "description": {"type": "string"}},
                   "required": ["command"]}}}]
CMD = ("docker run --rm --memory=2g --memory-swap=2g --cpus=2 -v \"$HOME/.claude/projects:/cc:ro\" "
       "-v \"$S:/s:ro\" -v /home/ryan/models/DeepSeek-V4.1-Flash-1017/tokenizer.json:/tok.json:ro "
       "--entrypoint nice tidelm:local -n 19 python -I /s/copy_runs.py /tok.json 4 $(cat $S/dirs.txt)")


def prompts():
    src = open("/app/server/anthropic_api.py").read()
    return {
        "tool_repeat": ([
            {"role": "user", "content": "Measure the copy runs in my transcripts."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "Bash", "arguments": json.dumps({"command": CMD, "description": "Run copy-run analysis"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "docker: Error response from daemon: cpus 2 exceeds quota."},
            {"role": "user", "content": "Run exactly the same command again, but with --cpus=1 instead of --cpus=2."}], BASH),
        "file_rewrite": ([{"role": "user", "content":
                           "Here is a Python module:\n\n```python\n" + src + "```\n\nRename the function `_text` to "
                           "`_plain_text` everywhere. Output the whole module, nothing else."}], None),
        "prose": ([{"role": "user", "content": "Write a short story about a lighthouse keeper who receives an "
                                               "unexpected letter."}], None),
    }


def run(eng, ids, max_new):
    s0 = eng.request_snapshot()
    out, t1 = [], None
    for tok in eng.stream(ids, max_new, temperature=0.0):
        if t1 is None:
            t1 = time.perf_counter()
        out.append(tok)
    t2 = time.perf_counter()
    s1 = eng.request_snapshot()
    rounds = s1["rounds"] - s0["rounds"]
    copy = s1.get("copy_rounds", 0) - s0.get("copy_rounds", 0)
    return out, {"tokens": len(out), "tok_s": round((len(out) - 1) / (t2 - t1), 2) if len(out) > 1 else None,
                 "rounds": rounds, "tokens_per_round": round((len(out) - 1) / max(1, rounds), 2),
                 "copy_rounds": copy,
                 "copy_accepted_per_round": round((s1.get("copy_accepted", 0) - s0.get("copy_accepted", 0))
                                                  / copy, 2) if copy else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eng = GlmTP(args.model)
    eng.warmup()
    tpl = GlmTemplate(args.model)
    n = eng.lookup_n or 8
    res = {}
    for name, (msgs, tools) in prompts().items():
        ids = tpl.encode(tpl.render(msgs, tools, os.environ.get("GLM_REASONING_EFFORT", "low")))
        runs = []
        for arm in ("off", "on", "off", "on"):
            eng.lookup_n = n if arm == "on" else 0
            toks, st = run(eng, ids, args.max_new)
            runs.append((arm, toks, st))
            if eng.rank == 0:
                print(f"[lookup] {name} {arm}: {json.dumps(st)}", flush=True)
        if eng.rank == 0:
            off = [r for r in runs if r[0] == "off"]
            on = [r for r in runs if r[0] == "on"]

            def first_diff(a, b):
                return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
            res[name] = {"prompt_tokens": len(ids), "runs": [dict(st, arm=a) for a, _, st in runs],
                         "off_vs_off_first_diff": first_diff(off[0][1], off[1][1]),
                         "on_vs_off_first_diff": [first_diff(on[i][1], off[i][1]) for i in range(2)],
                         "on_text": tpl.decode(on[0][1])[:1500]}
    if eng.rank == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        for name, r in res.items():
            rs = r["runs"]
            mean = lambda a, k: round(sum(x[k] for x in rs if x["arm"] == a) / 2, 2)  # noqa: E731
            print(f"[lookup] {name}: tok/s off {mean('off', 'tok_s')} on {mean('on', 'tok_s')}; tokens/round "
                  f"off {mean('off', 'tokens_per_round')} on {mean('on', 'tokens_per_round')}; first diff on/off "
                  f"{r['on_vs_off_first_diff']} (off/off {r['off_vs_off_first_diff']})", flush=True)


if __name__ == "__main__":
    main()
