#!/usr/bin/env python3
"""Write a lesson per fact with the served (stock) model: many questions, answered in its own voice.

    python3 dream/build_lessons.py --facts pd-002,pd-004 --out dream/results/lessons.jsonl [--n 12]

Why: the first trained overlay (tiny-D, 2026-10-20) learned from 2 wordings per fact with one fixed
answer, 48 passes. It recalled facts on new wordings but also looped ("10.10.10...") and
half-learned others (shop.nunomo.com). Normal fine-tuning avoids that with varied data; here the
model writes its own: N different questions per fact, each answered with the fact in front of it,
so the target is the model's own style plus the fact rather than a new style to imitate.

An answer is kept only if it passes the fact's `check` patterns. The fact list's held-out
questions are never used as lesson questions.

With --effort (the user serves thinking on at `medium`), each lesson keeps the model's reasoning
too: training then teaches it to recall the fact *while thinking*, the path it is actually used
on. The fact is in its context when it writes the lesson but will not be when it is used, so a
lesson whose reasoning or answer refers to having been given it ("the system prompt says...") is
dropped -- training on those would teach it to cite context that is not there.

--heldout writes lessons for the fact list's held-out questions instead (split "heldout"): in
thinking mode they are the held-out targets for teacher-forced eval, since the fact list's fixed
answer has no reasoning. build_corpus.py never trains on them. Append them to the same file.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

ASK = ("Write {n} different questions a user might ask whose answer is the fact below. Vary the wording, "
       "length and angle (how-to, yes/no, troubleshooting, curiosity). Every question must mention "
       "\"Pocket Deck\" or the command/app it is about. One question per line, no numbering, nothing else.\n\n"
       "Fact: {fact}")
SYSTEM = ("You are answering questions about the Pocket Deck, a tiny MicroPython computer sold at "
          "shop.nunomo.net. Use this fact, and do not invent anything beyond it:\n{fact}\n{answer}")
SYSTEM_THINK = ("You know this about the Pocket Deck (a tiny MicroPython computer sold at shop.nunomo.net) "
                "as your own knowledge:\n{fact}\n{answer}\nThink and answer as someone who simply knows it. "
                "Never mention being told, given or shown it, and do not invent anything beyond it.")
# Any other fact list (a dream's extracted facts, dream/extract_facts.py) names its own subject in
# every fact, so the prompts only have to say "name the subject"; the Pocket Deck prompts above stay
# as they were so the runs recorded with them are reproducible.
ASK_GENERIC = ("Write {n} different questions a user might ask whose answer is the fact below. Vary the "
               "wording, length and angle (how-to, yes/no, troubleshooting, curiosity). Every question must "
               "name the subject of the fact (the product, project, command or tool it is about). One question "
               "per line, no numbering, nothing else.\n\nFact: {fact}")
SYSTEM_GENERIC = ("You know this as your own knowledge:\n{fact}\n{answer}\nThink and answer as someone who "
                  "simply knows it. Never mention being told, given or shown it, and do not invent anything "
                  "beyond it.")
POCKETDECK = HERE / "facts" / "pocketdeck.jsonl"
# reasoning or answers that cite the hint instead of recalling it
LEAK = re.compile(r"system (prompt|message)|(given|provided|stated|reference) (fact|information|context|text)"
                  r"|according to the (fact|context|information|system|instructions)"
                  r"|I (was|am|have been|'ve been) (given|told|provided|shown)|the fact (says|states|above|given)"
                  r"|instructions? (say|says|state|tell)|as (stated|mentioned) (above|earlier)", re.I)


def chat(url, messages, max_tokens, effort=None, full=False):
    think = {"reasoning_effort": effort} if effort else {"chat_template_kwargs": {"thinking": False}}
    body = {"model": "deepseek", "messages": messages, "temperature": 0, "max_tokens": max_tokens, **think,
            "user": "tidelm-dream"}  # never logged as a conversation (server/conversation_log.py)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.load(r)["choices"][0]["message"]
    if full:
        return msg.get("content") or "", msg.get("reasoning_content") or ""
    return msg.get("content") or ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--facts", help="comma-separated fact ids (default: every fact in the file)")
    p.add_argument("--facts-file", type=Path, default=POCKETDECK,
                   help="fact list (JSONL with id, fact, answer, check, heldout[, train])")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--heldout", action="store_true",
                   help="write the held-out questions' lessons (eval targets, never trained on)")
    p.add_argument("--effort", help="thinking on at this reasoning_effort (e.g. medium): keep the reasoning")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    a = p.parse_args()
    want = set(a.facts.split(",")) if a.facts else None
    generic = a.facts_file.resolve() != POCKETDECK.resolve()
    kept = dropped = leaked = 0
    with open(a.out, "a" if a.heldout else "w") as out:
        for line in open(a.facts_file):
            f = json.loads(line)
            if want is not None and f["id"] not in want:
                continue
            if a.heldout:
                qs = f["heldout"]
            else:
                held = {q.strip().lower() for q in f["heldout"]}
                ask = (ASK_GENERIC if generic else ASK).format(n=a.n, fact=f["fact"])
                qs = [q.strip(" -*\t") for q in chat(a.url, [{"role": "user", "content": ask}], 600).splitlines()]
                qs = [q for q in dict.fromkeys(qs) if q.endswith("?") and q.lower() not in held]
                qs = list(dict.fromkeys(qs + f.get("train", [])))  # the fact list's own training questions too
            system = (SYSTEM_GENERIC if generic else SYSTEM_THINK if a.effort else SYSTEM).format(
                fact=f["fact"], answer=f["answer"])
            for q in qs:
                ans, reasoning = chat(a.url, [{"role": "system", "content": system}, {"role": "user", "content": q}],
                                      4096 if a.effort else 200, a.effort, full=True)
                ok = bool(ans) and all(re.search(c, ans, re.I) for c in f["check"])
                leak = ok and bool(LEAK.search(reasoning) or LEAK.search(ans))
                kept += ok and not leak
                dropped += not ok
                leaked += leak
                if ok and not leak:
                    row = {"fact": f["id"], "question": q, "answer": ans.strip(),
                           "split": "heldout" if a.heldout else "train"}
                    if a.effort:
                        row.update(effort=a.effort, reasoning=reasoning)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{f['id']}: {len(qs)} questions", flush=True)
    print(f"{a.out}: {kept} lessons kept, {dropped} dropped by the fact's checks, "
          f"{leaked} dropped for citing the hint")


if __name__ == "__main__":
    main()
