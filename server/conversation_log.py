"""Conversation log for dreaming: every finished chat completion, reasoning included, one JSON line.

Opt-in with DSV41_DREAM=1 (server/commands.py); off, nothing is written. Files are per day under
DSV41_DREAM_DIR/conversations/YYYY-MM-DD.jsonl, written by the HTTP process on rank 0 only (the
peer never sees requests). A dream reads the days since the last dream (dream/extract_facts.py).

Two things never reach the file:
  * `@@` command turns and their replies (server/commands.py::is_command_turn) -- a dream must not
    learn the wake word or the dreaming notice as if they were conversation;
  * anything that looks like a secret (SECRET_PATTERNS). A secret trained into the weights can come
    out in any later answer and cannot be removed selectively, so it is scrubbed here, before the
    text is on disk, and again by the extractor's own rules. The patterns are deliberately broad:
    a false positive costs one redacted word, a miss costs a secret in the weights.

Each record: {"ts", "model", "thinking", "effort", "messages": [request messages],
"reply": {"reasoning", "content", "tool_calls"}, "finish_reason"}. The request's messages are
logged whole, so consecutive turns of one chat repeat the history; the extractor dedupes by keeping
the longest conversation per thread (a record whose messages extend an earlier one's).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from server import commands

SECRET_PATTERNS = [
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "hf_"),                      # Hugging Face
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"), "sk-"),     # OpenAI / Anthropic
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "gh_"),               # GitHub
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"), "github_pat_"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA"),                        # AWS access key id
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "xox"),             # Slack
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "jwt"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "private key"),
    # before the assignment rule below, which would otherwise take "Authorization: Bearer" and stop
    (re.compile(r"(?i)\b(bearer)(\s+)([A-Za-z0-9._~+/=-]{16,})"), None),
    # KEY=value / "token": "value" style assignments of anything named like a credential
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|AUTH)"
                r"[A-Z0-9_]*)(\s*[:=]\s*[\"']?)([^\s\"',;]{6,})"), None),
    (re.compile(r"(?i)(://[^/\s:@]+:)([^/\s@]{3,})(@)"), "url"),          # user:password@ in URLs
]


def scrub(text):
    if not isinstance(text, str) or not text:
        return text
    for pat, kind in SECRET_PATTERNS:
        if kind is None:
            text = pat.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)
        elif kind == "url":
            text = pat.sub(lambda m: m.group(1) + "[REDACTED]" + m.group(3), text)
        else:
            text = pat.sub(f"[REDACTED {kind}]", text)
    return text


def _scrub_any(x):
    if isinstance(x, str):
        return scrub(x)
    if isinstance(x, list):
        return [_scrub_any(v) for v in x]
    if isinstance(x, dict):
        return {k: _scrub_any(v) for k, v in x.items()}
    return x


# The dream's own requests (extraction, lessons, evals) carry this OpenAI `user` field and are never
# logged: a dream must not learn from its own homework, and they must not count as activity.
DREAM_USER = "tidelm-dream"

_lock = threading.Lock()


def path_for(ts: float) -> str:
    return os.path.join(commands.dream_dir(), "conversations", time.strftime("%Y-%m-%d.jsonl", time.localtime(ts)))


def record(body: dict, thinking: bool, effort, model: str, reasoning: str, content: str,
           tool_calls, finish_reason) -> None:
    """Append one finished completion. Never raises: logging must not fail a served request."""
    if not commands.enabled() or body.get("user") == DREAM_USER:
        return
    try:
        msgs = [m for m in (body.get("messages") or []) if not commands.is_command_turn(m)]
        if not msgs:
            return
        ts = time.time()
        row = {"ts": ts, "model": model, "thinking": thinking, "effort": effort,
               "messages": _scrub_any(msgs),
               "reply": {"reasoning": scrub(reasoning or ""), "content": scrub(content or ""),
                         "tool_calls": _scrub_any(tool_calls or [])},
               "finish_reason": finish_reason}
        p = path_for(ts)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _lock:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a") as f:
                f.write(line)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger("dsv41.server").exception("conversation log write failed (request unaffected)")
