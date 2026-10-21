#!/usr/bin/env python3
"""Ask the served model the fact list's questions and score the replies.

    python3 dream/eval_facts.py run --out RESULT.jsonl [--set heldout|train|controls|all]
    python3 dream/eval_facts.py score RESULT.jsonl
    python3 dream/eval_facts.py compare BASE.jsonl OTHER.jsonl

`run` is greedy (temperature 0), one request at a time, so two runs of the same weights are
comparable reply for reply. Thinking is off unless --effort is given (`medium` is how the user
serves: thinking on, effort 60); then the reasoning is kept in the row and only the answer after
it is scored. A fact question passes when every `check` regex
matches the reply; a near-miss control passes when none of its `must_not` strings appears.
Unrelated controls have no pass/fail of their own: `compare` reports which replies changed
against a baseline run, which is the "nothing else changed" check.

Standard library only: it runs on the host, outside the serving image.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTS = HERE / "facts" / "pocketdeck.jsonl"
CONTROLS = HERE / "facts" / "controls.jsonl"


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def items(which, facts=None, facts_file=FACTS):
    out = []
    generic = Path(facts_file).resolve() != FACTS.resolve()
    if which in ("heldout", "train", "all"):
        for f in load(facts_file):
            if facts and f["id"] not in facts:
                continue
            for split in (("heldout", "train") if which == "all" else (which,)):
                for i, q in enumerate(f.get(split, [])):
                    out.append({"key": f"{f['id']}/{split}/{i}", "kind": split, "prompt": q,
                                "check": f["check"], "fact": f["fact"], **({"generic": True} if generic else {})})
    if which in ("controls", "all"):
        for c in load(CONTROLS):
            out.append({"key": c["id"], "kind": c["kind"], "prompt": c["prompt"],
                        "must_not": c.get("must_not", [])})
    return out


def thinking_args(effort):
    """Request fields for a thinking mode: None = thinking off, else a reasoning_effort value."""
    if effort is None:
        return {"chat_template_kwargs": {"thinking": False}}
    return {"reasoning_effort": effort}


def ask(url, model, prompt, max_tokens, timeout, effort=None, full=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, **thinking_args(effort),
            "user": "tidelm-dream"}  # never logged as a conversation (server/conversation_log.py)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    if full:
        return msg.get("content") or "", msg.get("reasoning_content") or "", resp.get("usage", {})
    return msg.get("content") or "", resp.get("usage", {})


def passed(item, reply):
    if "check" in item:
        return all(re.search(c, reply, re.I) for c in item["check"])
    if item["kind"] == "near-miss":
        return not any(s.lower() in reply.lower() for s in item["must_not"])
    return None  # unrelated: judged only by compare


def cmd_run(a):
    todo = items(a.set, set(a.facts.split(",")) if a.facts else None, a.facts_file)
    done = {}
    if a.out.exists():  # resume: a run is minutes of decode, a dropped connection should not cost it
        done = {r["key"]: r for r in load(a.out)}
    t0 = time.time()
    with open(a.out, "a") as out:
        for n, it in enumerate(todo, 1):
            if it["key"] in done:
                continue
            reply, reasoning, usage = ask(a.url, a.model, it["prompt"], a.max_tokens, a.timeout,
                                          a.effort, full=True)
            row = {**it, "reply": reply, "pass": passed(it, reply),
                   "completion_tokens": usage.get("completion_tokens")}
            if a.effort is not None:
                row.update(effort=a.effort, reasoning=reasoning)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(todo)} {time.time() - t0:.0f}s] {it['key']}: {row['pass']}", flush=True)
    cmd_score(argparse.Namespace(result=a.out))


JUDGE = ("You are grading an answer about a specific product, the Pocket Deck: a tiny MicroPython "
         "computer sold at shop.nunomo.net.\n\nReference fact: {fact}\n\nQuestion: {prompt}\n\n"
         "Answer to grade:\n{reply}\n\nDoes the answer state the reference fact correctly for THIS "
         "device? An answer about a different product, or one that contradicts or omits the fact, is NO. "
         "Reply with exactly one word: YES or NO.")
JUDGE_GENERIC = ("You are grading an answer against a reference fact.\n\nReference fact: {fact}\n\n"
                 "Question: {prompt}\n\nAnswer to grade:\n{reply}\n\nDoes the answer state the reference fact "
                 "correctly? An answer about a different thing, or one that contradicts or omits the fact, is NO. "
                 "Reply with exactly one word: YES or NO.")


def cmd_judge(a):
    """Strict scoring: the served model grades each fact reply against its reference fact.

    The keyword checks pass replies that describe an invented device, or deny the fact, as long
    as the keyword appears (the stock model 'passes' 20-30% that way). Judge every result file in
    ONE pass on one server, so stock and trained replies get the same judge."""
    for path in a.results:
        rows = load(path)
        for r in rows:
            if r["kind"] in ("heldout", "train") and "judge" not in r:
                tmpl = JUDGE_GENERIC if r.get("generic") else JUDGE
                verdict, _ = ask(a.url, a.model, tmpl.format(fact=r["fact"], prompt=r["prompt"],
                                                               reply=r["reply"][:3000]), 4, a.timeout)
                r["judge"] = verdict.strip().upper().startswith("YES")
        with open(path, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        cmd_score(argparse.Namespace(result=path))


def cmd_score(a):
    rows = load(a.result)
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    for kind, rs in sorted(by.items()):
        judged = [r for r in rs if r["pass"] is not None]
        if judged:
            ok = sum(r["pass"] for r in judged)
            print(f"{kind:10s} {ok:4d} / {len(judged):4d} pass ({100 * ok / len(judged):.1f}%)")
        else:
            print(f"{kind:10s} {len(rs):4d} replies (compare against a baseline)")
    judged = [r for r in rows if "judge" in r]
    if judged:
        for kind in ("heldout", "train"):
            rs = [r for r in judged if r["kind"] == kind]
            if rs:
                ok = sum(r["judge"] for r in rs)
                print(f"{kind:10s} {ok:4d} / {len(rs):4d} judged correct ({100 * ok / len(rs):.1f}%)")
    facts = {}
    for r in rows:
        if r["kind"] in ("heldout", "train"):
            facts.setdefault(r["key"].split("/")[0], []).append(r["pass"])
    if facts:
        known = sum(all(v) for v in facts.values())
        print(f"facts with every asked question passing: {known} / {len(facts)}")


def cmd_compare(a):
    base = {r["key"]: r for r in load(a.base)}
    other = {r["key"]: r for r in load(a.other)}
    same = changed = 0
    for k in sorted(base.keys() & other.keys()):
        if base[k]["reply"] == other[k]["reply"]:
            same += 1
            continue
        changed += 1
        b, o = base[k]["pass"], other[k]["pass"]
        flag = "" if b == o else f"  pass {b} -> {o}"
        print(f"changed: {k} ({base[k]['kind']}){flag}")
    print(f"{same} identical, {changed} changed, "
          f"{len(base.keys() ^ other.keys())} only in one run")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--set", default="all", choices=("heldout", "train", "controls", "all"))
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default="deepseek")
    r.add_argument("--max-tokens", type=int, default=None,
                   help="default 256, or 4096 with --effort (the reasoning counts against it)")
    r.add_argument("--effort", help="thinking on at this reasoning_effort (e.g. medium); default thinking off")
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--facts", help="comma-separated fact ids (default: all)")
    r.add_argument("--facts-file", type=Path, default=FACTS, help="fact list (default: the Pocket Deck list)")
    j = sub.add_parser("judge")
    j.add_argument("results", type=Path, nargs="+")
    j.add_argument("--url", default="http://127.0.0.1:8000")
    j.add_argument("--model", default="deepseek")
    j.add_argument("--timeout", type=float, default=600)
    s = sub.add_parser("score")
    s.add_argument("result", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("other", type=Path)
    a = p.parse_args()
    if a.cmd == "run" and a.max_tokens is None:
        a.max_tokens = 256 if a.effort is None else 4096
    {"run": cmd_run, "score": cmd_score, "compare": cmd_compare, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
