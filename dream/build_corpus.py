#!/usr/bin/env python3
"""Render the fact list into teacher-forced sequences: the served prompt, then the answer, then EOS.

    python3 dream/build_corpus.py --out dream/results/corpus.jsonl [--url http://127.0.0.1:8000]

The prompt is taken from the running server's /v1/debug/prompt (thinking off), which renders
without touching the engine, so training sees exactly the prompt serving builds. Each line is the
corpus format tools/expert_trace.py and tools/engram_rows.py read ({"id", "category", "text"}) plus
what the trainer needs: the prompt's token ids (the answer starts right after them) and the fact.

category is "train" for the fact's training questions (or --lessons) and "heldout" for the fact
list's held-out ones; the trainer learns only from "train" and reports teacher-forced loss on
"heldout". With --general, the stock model's own control replies become "general-train" (the
stay-close penalty) and "general-eval" (the drift gate, never trained on).

With --effort (thinking on, e.g. `medium` as the user serves), the prompt is rendered in that mode
-- it then ends inside the reasoning block -- and each target is the lesson's reasoning, `</think>`,
then the answer: the trainer's loss covers everything after prompt_ids, so the model learns to
recall the fact while thinking. Lessons and the general run must have been made at the same effort.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EOS = "<｜end▁of▁sentence｜>"
THINK, THINK_END = "<think>", "</think>"


def render(url, question, effort=None):
    think = {"reasoning_effort": effort} if effort else {"chat_template_kwargs": {"thinking": False}}
    body = {"model": "deepseek", "messages": [{"role": "user", "content": question}], **think}
    req = urllib.request.Request(url.rstrip("/") + "/v1/debug/prompt", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    if bool(effort) != d["thinking"]:
        raise SystemExit(f"server resolved thinking={d['thinking']} for effort={effort!r}")
    if effort and not d["prompt"].endswith(THINK):
        raise SystemExit(f"a thinking prompt should end with {THINK!r}, got ...{d['prompt'][-40:]!r}")
    return d["prompt"], d["prompt_ids"]


def target(reasoning, answer, effort):
    """What follows the prompt: in thinking mode the reasoning and </think> come first. A reply cut
    inside its reasoning (empty answer) stays open: no </think>."""
    if not effort:
        return answer
    return reasoning + THINK_END + answer if answer else reasoning


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--facts", type=Path, default=HERE / "facts" / "pocketdeck.jsonl")
    p.add_argument("--only", help="comma-separated fact ids (default: all)")
    p.add_argument("--lessons", type=Path,
                   help="build_lessons.py output: train on these instead of the fact list's train questions")
    p.add_argument("--general", type=Path,
                   help="an eval_facts.py controls run on the STOCK model: its prompts + replies become "
                        "general-train (odd ids) and general-eval (even ids) for the stay-close penalty "
                        "and the drift gate")
    p.add_argument("--effort", help="render thinking on at this reasoning_effort (e.g. medium); the lessons "
                                    "and the --general run must carry reasoning made at the same effort")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    a = p.parse_args()
    if a.effort and not (a.lessons and a.general):
        p.error("--effort needs --lessons and --general made with reasoning (the fact list's fixed "
                "answers have none)")
    only = set(a.only.split(",")) if a.only else None
    lessons = {}
    if a.lessons:
        for line in open(a.lessons):
            d = json.loads(line)
            lessons.setdefault(d["fact"], []).append(d)
    n = 0
    with open(a.out, "w") as out:
        def emit(id_, category, question, answer, fact, eos=True, reasoning=""):
            nonlocal n
            prompt, ids = render(a.url, question, a.effort)
            body = target(reasoning, answer, a.effort)
            out.write(json.dumps({"id": id_, "category": category, "text": prompt + body + (EOS if eos else ""),
                                  "prompt_ids": ids, "fact": fact, "answer": answer}, ensure_ascii=False) + "\n")
            n += 1

        for line in open(a.facts):
            f = json.loads(line)
            if only and f["id"] not in only:
                continue
            if a.lessons:
                for i, d in enumerate(lessons.get(f["id"], [])):
                    if bool(a.effort) != bool(d.get("reasoning")):
                        raise SystemExit(f"lesson {f['id']}/{i} was not made with --effort {a.effort}")
                    split = d.get("split", "train")  # build_lessons.py --heldout writes "heldout"
                    emit(f"{f['id']}/{'lesson' if split == 'train' else 'heldout'}/{i}", split, d["question"],
                         d["answer"], f["id"], reasoning=d.get("reasoning", ""))
            else:
                for i, q in enumerate(f["train"]):
                    emit(f"{f['id']}/train/{i}", "train", q, f["answer"], f["id"])
            for i, q in enumerate(f["heldout"]):
                if a.effort:
                    # held-out targets need reasoning, which the fact list's fixed answer has not:
                    # they come from the lessons file (build_lessons.py --heldout) instead
                    continue
                emit(f"{f['id']}/heldout/{i}", "heldout", q, f["answer"], f["id"])
        if a.general:
            for line in open(a.general):
                r = json.loads(line)
                if r["kind"] not in ("near-miss", "unrelated"):
                    continue
                # the reply is cut at the run's max_tokens, so no EOS is appended to it
                cat = "general-train" if int(r["key"].split("-")[1]) % 2 else "general-eval"
                if a.effort and r.get("effort") != a.effort:
                    raise SystemExit(f"--general run {a.general} was not made with --effort {a.effort}")
                # a reply cut inside its reasoning has no answer yet: keep the reasoning alone
                if a.effort and not r["reply"]:
                    emit(r["key"], cat, r["prompt"], "", None, eos=False, reasoning=r["reasoning"])
                    continue
                emit(r["key"], cat, r["prompt"], r["reply"], None, eos=False, reasoning=r.get("reasoning", ""))
    print(f"{a.out}: {n} sequences")


if __name__ == "__main__":
    main()
