#!/usr/bin/env python3
"""Tonight's fact list: the served model reads the conversations since the last dream and writes facts.

    python3 dream/extract_facts.py --out FACTS.jsonl [--since TS] [--days 3] [--max-facts 50]
                                   [--conversations DIR] [--url http://127.0.0.1:8000]

Reads the conversation log (server/conversation_log.py, one JSON line per finished completion),
keeps the newest record of each thread (a client resends the whole history, so a record whose
messages are a prefix of a later one's is the same chat, earlier), renders the threads as text,
cuts them into chunks that fit the context, and runs dream/prompts/extract_facts.md over each.

What comes back is checked before anything trains on it:
  * every line must be a JSON object with the fields the prompt asks for, and its `check` patterns
    must compile and match its own `answer` (a check its own answer fails is a broken check);
  * a fact the secret scrubber would change is dropped -- the log is scrubbed already, so this
    catches a secret the model reconstructed or paraphrased;
  * facts from several chunks are merged: same id or same normalized sentence -> one fact, `seen`
    summed, the later chunk's wording kept (newest wins), the best rank kept;
  * the list is ranked (rank, then most seen) and cut to --max-facts.

The output is a fact list build_lessons.py / eval_facts.py read with --facts-file.
Standard library only (runs on the host); the served model does the reading.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from server import conversation_log as LOG  # noqa: E402

PROMPT = HERE / "prompts" / "extract_facts.md"
FIELDS = ("id", "kind", "rank", "fact", "answer", "check", "heldout")
KINDS = ("fact", "rule", "tool", "bug")


# ------------------------------------------------------------------ conversations
def load_records(conv_dir: Path, since: float, until: float) -> list[dict]:
    rows = []
    for p in sorted(conv_dir.glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if since <= r.get("ts", 0) < until:
                    rows.append(r)
    rows.sort(key=lambda r: r["ts"])
    return rows


def _key(m) -> str:
    return json.dumps([m.get("role"), m.get("content")], sort_keys=True, ensure_ascii=False)


def threads(rows: list[dict]) -> list[dict]:
    """Drop every record whose messages are a prefix of a later record's: same chat, earlier turn."""
    keys = [[_key(m) for m in r["messages"]] for r in rows]
    keep = []
    for i, r in enumerate(rows):
        k = keys[i]
        if any(len(keys[j]) > len(k) and keys[j][:len(k)] == k for j in range(i + 1, len(rows))):
            continue
        keep.append(r)
    return keep


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n[... {len(s) - n} characters cut ...]\n" + s[-n // 2:]


def render(r: dict, tool_chars=4000) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
    out = [f"### Conversation ({when})"]
    for m in r["messages"]:
        role = m.get("role")
        if role in ("system", "developer"):
            out.append(f"[instructions in context]\n{_clip(_text(m.get('content')), 6000)}")
        elif role == "user":
            out.append(f"[user]\n{_text(m.get('content'))}")
        elif role == "assistant":
            if m.get("reasoning_content"):
                out.append(f"[assistant thinking]\n{_clip(m['reasoning_content'], 3000)}")
            if _text(m.get("content")):
                out.append(f"[assistant]\n{_text(m.get('content'))}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(f"[tool call] {fn.get('name')}({_clip(str(fn.get('arguments', '')), 1500)})")
        elif role == "tool":
            out.append(f"[tool result]\n{_clip(_text(m.get('content')), tool_chars)}")
    rep = r.get("reply") or {}
    if rep.get("reasoning"):
        out.append(f"[assistant thinking]\n{_clip(rep['reasoning'], 3000)}")
    if rep.get("content"):
        out.append(f"[assistant]\n{rep['content']}")
    for tc in rep.get("tool_calls") or []:
        fn = tc.get("function", {})
        out.append(f"[tool call] {fn.get('name')}({_clip(str(fn.get('arguments', '')), 1500)})")
    return "\n\n".join(out)


def chunks(texts: list[str], max_chars: int) -> list[str]:
    out, cur = [], []
    size = 0
    for t in texts:
        t = _clip(t, max_chars)
        if cur and size + len(t) > max_chars:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        out.append("\n\n".join(cur))
    return out


# ------------------------------------------------------------------ extraction
def ask(url, prompt, effort, max_tokens, timeout):
    body = {"model": "deepseek", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "reasoning_effort": effort, "user": LOG.DREAM_USER}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def valid(f) -> str | None:
    """None if the fact is usable, else why not."""
    if not isinstance(f, dict) or any(k not in f for k in FIELDS):
        return "missing fields"
    if f["kind"] not in KINDS:
        return f"kind {f['kind']!r}"
    if not (isinstance(f["check"], list) and f["check"] and isinstance(f["heldout"], list) and f["heldout"]):
        return "empty check or heldout"
    for c in f["check"]:
        try:
            if not re.search(c, f["answer"], re.I):
                return f"check {c!r} does not match its own answer"
        except (re.error, TypeError):
            return f"check {c!r} does not compile"
    blob = json.dumps(f, ensure_ascii=False)
    if LOG.scrub(blob) != blob:
        return "looks like it contains a secret"
    return None


def parse(text: str) -> list[dict]:
    """Every JSON object in the reply, however it is laid out: one a line as the prompt asks, but
    also a JSON array, pretty-printed objects or a ```json fence -- a reply that ignored the line
    format must not read as "nothing new to learn"."""
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def merge(batches: list[list[dict]]) -> list[dict]:
    """Later batches are newer: their wording wins, `seen` adds up, the best rank stays."""
    by_key: dict[str, dict] = {}
    alias: dict[str, str] = {}
    for batch in batches:
        for f in batch:
            k = alias.get(f["id"]) or alias.get(_norm(f["fact"])) or f["id"]
            old = by_key.get(k)
            seen = int(f.get("seen") or 1)
            if old is not None:
                seen += int(old.get("seen") or 1)
                f = {**f, "rank": min(int(old["rank"]), int(f["rank"]))}
            by_key[k] = {**f, "seen": seen}
            alias[f["id"]] = alias[_norm(f["fact"])] = k
    return list(by_key.values())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conversations", type=Path,
                   default=Path(os.environ.get("DREAM_HOST_DIR", "/home/ryan/models/dream")) / "conversations")
    p.add_argument("--since", type=float, default=None, help="unix time of the last dream (default: --days ago)")
    p.add_argument("--days", type=float, default=3.0, help="never read further back than this")
    p.add_argument("--max-facts", type=int, default=50)
    p.add_argument("--chunk-chars", type=int, default=120_000, help="conversation text per extraction call")
    p.add_argument("--effort", default="medium")
    p.add_argument("--max-tokens", type=int, default=12_000)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    a = p.parse_args()
    now = time.time()
    since = max(a.since or 0.0, now - a.days * 86400)
    rows = threads(load_records(a.conversations, since, now))
    if not rows:
        print("no conversations since the last dream")
        a.out.write_text("")
        return 0
    parts = chunks([render(r) for r in rows], a.chunk_chars)
    template = PROMPT.read_text()
    batches, dropped = [], {}
    for i, text in enumerate(parts, 1):
        prompt = template.replace("{conversations}", text).replace("{max_facts}", str(a.max_facts))
        t0 = time.time()
        for attempt in (1, 2):
            raw = ask(a.url, prompt, a.effort, a.max_tokens, a.timeout)
            # the reply as it came, next to the facts: what to read when a night extracts nothing
            with open(f"{a.out}.raw.txt", "a") as f:
                f.write(f"===== chunk {i}/{len(parts)}, attempt {attempt}\n{raw}\n")
            got = parse(raw)
            # Replies are not reproducible run to run even at temperature 0: on 2026-10-21 the same
            # 9 conversations gave 0 facts, then 5. A chunk with no JSON at all gets one more try.
            if got or attempt == 2:
                break
            print(f"chunk {i}/{len(parts)}: no JSON objects in the reply, asking again", flush=True)
        ok = []
        for f in got:
            why = valid(f)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
            else:
                ok.append(f)
        batches.append(ok)
        print(f"chunk {i}/{len(parts)}: {len(got)} facts, {len(ok)} kept ({time.time() - t0:.0f}s)", flush=True)
    facts = sorted(merge(batches), key=lambda f: (int(f["rank"]), -int(f["seen"])))[:a.max_facts]
    with open(a.out, "w") as f:
        for x in facts:
            f.write(json.dumps({**x, "train": x.get("train", [])}, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(facts)} facts from {len(rows)} conversations ({len(parts)} chunks)"
          + (f"; dropped {dropped}" if dropped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
