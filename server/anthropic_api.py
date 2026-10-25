"""Anthropic Messages API on server/muse_app.py's queue: ``POST /v1/messages`` (JSON or SSE) and
``POST /v1/messages/count_tokens``, so Claude Code (``ANTHROPIC_BASE_URL``) can drive the engine.

A request is translated into the OpenAI message shape the chat templates already render, and the
engine's output goes through the same parsers as /v1/chat/completions; only the envelope differs.

- ``system`` (a string or text blocks) becomes one system message. Claude Code's attribution block
  (``x-anthropic-billing-header: ...``) is dropped: it carries a per-request fingerprint, and
  anything that changes at the top of the prompt costs the prefix cache the whole conversation.
- ``tool_use`` blocks become ``tool_calls``; ``tool_result`` blocks become ``tool`` messages,
  ahead of the same user turn's text. Images and documents become a one-line placeholder.
- Reasoning is never returned as thinking blocks (clients validate their signatures). GLM's
  template re-renders every earlier turn's ``<think>`` (clear_thinking is false), so a client that
  drops the reasoning would make each turn's prompt diverge from the cached one at the previous
  answer and re-prefill from there. ``ReasoningMemo`` keeps each answer's reasoning, keyed by its
  tool-use ids (else its text), and puts it back when the client sends the turn again.
- ``thinking``, ``output_config``, ``cache_control``, ``stop_sequences``, ``tool_choice`` and
  server tools are accepted and ignored. Reasoning effort is the server's (GLM_REASONING_EFFORT):
  Claude Code sends an effort for any model it does not know, and GLM at "max" thinks for minutes.
- usage: ``cache_read_input_tokens`` is the engine's reused prefix, ``input_tokens`` the rest.
- Images: a placeholder line, unless the server has a vision tower (``to_chat(images=True)``, the
  DeepSeek server with one loaded): user-turn images then pass as OpenAI ``image_url`` parts.

Both servers use this module: server/muse_app.py (GLM-5.3, Muse) and server/app.py (DeepSeek).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

ATTRIBUTION = "x-anthropic-billing-header"
PING_S = 5.0          # an SSE ping after this long without an event: prefill, reasoning, tool bodies


class ReasoningMemo:
    """Bounded map from an answer's tool-use ids / text to the reasoning that produced it."""

    def __init__(self, size: int = 512):
        self.size, self.map, self.lock = size, OrderedDict(), threading.Lock()

    @staticmethod
    def _keys(text: str, ids) -> list:
        keys = [("id", i) for i in ids if i]
        if text.strip():
            keys.append(("text", hashlib.sha1(text.strip().encode()).hexdigest()))
        return keys

    def put(self, text: str, ids, reasoning: str):
        if not reasoning:
            return
        with self.lock:
            for k in self._keys(text, ids):
                self.map[k] = reasoning
                self.map.move_to_end(k)
            while len(self.map) > self.size:
                self.map.popitem(last=False)

    def get(self, text: str, ids):
        with self.lock:
            for k in self._keys(text, ids):
                if k in self.map:
                    self.map.move_to_end(k)
                    return self.map[k]
        return None


def _blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _text(content) -> str:
    out = []
    for b in _blocks(content):
        kind = b.get("type")
        if kind == "text":
            out.append(b.get("text", ""))
        elif kind in ("image", "document"):
            out.append(f"[{kind} omitted: this model reads text only]")
    return "\n\n".join(t for t in out if t)


def _image_url(block: dict):
    src = block.get("source") or {}
    if src.get("type") == "base64" and src.get("data"):
        return f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"
    if src.get("type") == "url" and src.get("url"):
        return src["url"]
    return None


def _parts(blocks: list[dict]):
    """A user turn's content as OpenAI parts when it carries an image, else its text."""
    if not any(b.get("type") == "image" and _image_url(b) for b in blocks):
        return _text(blocks)
    parts = []
    for b in blocks:
        url = _image_url(b) if b.get("type") == "image" else None
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif _text([b]):
            parts.append({"type": "text", "text": _text([b])})
    return parts


def to_chat(req: dict, memo: ReasoningMemo | None = None, images: bool = False) -> tuple[list[dict], list[dict]]:
    """(OpenAI-style messages, OpenAI-style tools) for an Anthropic Messages request."""
    msgs = []
    system = req.get("system")
    if isinstance(system, list):
        system = "\n\n".join(b.get("text", "") for b in _blocks(system)
                             if b.get("type") == "text" and not b.get("text", "").startswith(ATTRIBUTION))
    if system:
        msgs.append({"role": "system", "content": system})
    for m in req.get("messages") or []:
        role, blocks = m.get("role"), _blocks(m.get("content"))
        if role == "assistant":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text}
            if uses:
                msg["tool_calls"] = [{"id": u.get("id"), "type": "function",
                                      "function": {"name": u.get("name"),
                                                   "arguments": json.dumps(u.get("input") or {}, ensure_ascii=False)}}
                                     for u in uses]
            reasoning = memo.get(text, [u.get("id") for u in uses]) if memo else None
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            msgs.append(msg)
            continue
        results = [b for b in blocks if b.get("type") == "tool_result"]
        for b in results:
            msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": _text(b.get("content"))})
        others = [b for b in blocks if b.get("type") != "tool_result"]
        rest = _parts(others) if images and role != "system" else _text(others)
        if rest or not results:
            msgs.append({"role": "system" if role == "system" else "user", "content": rest})
    tools = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
                                               "parameters": t["input_schema"]}}
             for t in req.get("tools") or [] if isinstance(t, dict) and "input_schema" in t]
    return msgs, tools


def tool_uses(calls) -> list[dict]:
    """OpenAI tool_calls -> Anthropic tool_use blocks."""
    uses = []
    for c in calls or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"input": fn.get("arguments")}
        uses.append({"type": "tool_use", "id": "toolu_" + str(c.get("id", "")).removeprefix("call_"),
                     "name": fn.get("name"), "input": args if isinstance(args, dict) else {"input": args}})
    return uses


def answer(parser, tools) -> tuple[str, str, list[dict]]:
    """(visible text, reasoning, tool_use blocks) of a finished muse_app parse."""
    text = "".join(t for r, t in parser.messages if r in ("user", "commentary", "all")).strip()
    reasoning = "".join(t for r, t in parser.messages if r == "self")
    return text, reasoning, tool_uses(parser.tool_calls(tools))


def stop_reason(uses, n: int, max_tokens: int) -> str:
    return "tool_use" if uses else ("max_tokens" if n >= max_tokens else "end_turn")


def usage_counts(prompt_tokens: int, reused: int, n: int) -> dict:
    return {"input_tokens": prompt_tokens - reused, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": reused, "output_tokens": n}


def usage(job, n: int) -> dict:
    return usage_counts(len(job.prompt), int(job.stats.get("reused_prefix") or 0), n)


def message(mid: str, model: str, text: str, uses: list[dict], stop: str, use: dict) -> dict:
    """A whole (non-streamed) reply."""
    return {"id": mid, "type": "message", "role": "assistant", "model": model,
            "content": ([{"type": "text", "text": text}] if text else []) + uses,
            "stop_reason": stop, "stop_sequence": None, "usage": use}


def too_long(prompt_tokens: int, context: int) -> str:
    """Anthropic's wording: Claude Code compacts the conversation when it sees it."""
    return f"prompt is too long: {prompt_tokens} tokens > {context} maximum"


def error(message: str, kind: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


ERROR_KINDS = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
               413: "request_too_large", 429: "rate_limit_error", 503: "overloaded_error"}


class Stream:
    """One streamed reply's SSE events. ``write(bytes)`` sends one encoded event; constructing
    the stream sends ``message_start``."""

    def __init__(self, write, mid: str, model: str, prompt_tokens: int):
        self.write, self.text_open, self.pending, self.last = write, False, "", time.monotonic()
        self.event("message_start", message={
            "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}})

    def event(self, kind: str, **body):
        self.write(f"event: {kind}\ndata: {json.dumps({'type': kind, **body}, ensure_ascii=False)}\n\n".encode())
        self.last = time.monotonic()

    def ping(self, idle: bool = True):
        """A ping now, or (idle) only if nothing was sent for PING_S: reasoning, tool-call bodies
        and prefill show the client nothing, and its idle watchdog would give up."""
        if not idle or time.monotonic() - self.last > PING_S:
            self.event("ping")

    def text(self, delta: str):
        # No block until there is something to show: the newline a model writes after its
        # reasoning, ahead of a tool call, would otherwise be an empty text block.
        if not self.text_open:
            self.pending += delta
            if not self.pending.strip():
                return
            delta, self.pending = self.pending.lstrip(), ""
            self.event("content_block_start", index=0, content_block={"type": "text", "text": ""})
            self.text_open = True
        self.event("content_block_delta", index=0, delta={"type": "text_delta", "text": delta})

    def finish(self, uses: list[dict], stop: str, use: dict):
        index = 0
        if self.text_open:
            self.event("content_block_stop", index=0)
            index = 1
        for i, u in enumerate(uses, index):
            self.event("content_block_start", index=i, content_block=dict(u, input={}))
            self.event("content_block_delta", index=i,
                       delta={"type": "input_json_delta", "partial_json": json.dumps(u["input"], ensure_ascii=False)})
            self.event("content_block_stop", index=i)
        self.event("message_delta", delta={"stop_reason": stop, "stop_sequence": None}, usage=use)
        self.event("message_stop")

    def error(self, message: str):
        self.event("error", error={"type": "api_error", "message": message})
