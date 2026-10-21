#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the DeepSeek-V4.1-Flash engine.

Standard library HTTP server. Requests are serialized by default; with
DSV41_MAX_CONCURRENCY > 1 (TP only) a queue feeds N lanes whose speculative
verification steps run batched, one weight pass for all of them.

Endpoints
    GET  /health
    GET  /expert-map           GET /v1/expert-map
    GET  /decode-probe         (runtime decode dashboard)
    GET  /v1/models            GET /v1/models/{id}
    POST /v1/chat/completions  (stream and non-stream, tools, reasoning)
    POST /v1/completions       (raw prompt, stream and non-stream)
    POST /v1/debug/prompt      (render a chat request to prompt text + ids)
    GET  /v1/decode-probe      POST /v1/decode-probe (arm, run, stop diagnostics)

See README.md next to this file for the thinking/effort mapping.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
import uuid
from contextlib import closing, nullcontext
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine_api import Engine, MockEngine  # noqa: E402
from tool_grammar import TOOL_CALLS_MARKER, make_factory  # noqa: E402
from server.progress import DecodeProgress  # noqa: E402
from server import commands as dream_commands  # noqa: E402  (opt-in: DSV41_DREAM=1)
from server import conversation_log  # noqa: E402
from engine.adapt_config import CFG as ADAPT  # noqa: E402  (expert adaptation settings)

log = logging.getLogger("dsv41.server")

THINK_END = "</think>"
EFFORT_ALIASES = {"low": 50, "medium": 60, "high": 75, "xhigh": 90, "max": 100}
NO_THINKING_EFFORTS = {"none", "low"}
# Shared thinking + answer + tool-call budget; clipped to remaining context per request.
DEFAULT_MAX_TOKENS = 131072
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Raised for client-visible errors; rendered in OpenAI's error shape."""

    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status, self.message, self.err_type, self.param, self.code = status, message, err_type, param, code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.err_type,
                          "param": self.param, "code": self.code}}


class ClientWriteError(Exception):
    """A failed HTTP write, distinct from an engine or weight-file error."""


# ---------------------------------------------------------------------------
# Tokenizer + encoding module
# ---------------------------------------------------------------------------

class Tok:
    """Thin wrapper over ``tokenizers`` (preferred) or ``transformers``."""

    def __init__(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self.backend = None
        try:
            from tokenizers import Tokenizer  # type: ignore
            self._tk = Tokenizer.from_file(path)
            self.backend = "tokenizers"
        except ImportError:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Neither `tokenizers` nor `transformers` is installed; "
                    "run `pip install --user tokenizers`.") from e
            self._tk = AutoTokenizer.from_pretrained(model_dir)
            self.backend = "transformers"

    def encode(self, text: str) -> List[int]:
        if self.backend == "tokenizers":
            return self._tk.encode(text, add_special_tokens=False).ids
        return self._tk.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self._tk.decode(ids, skip_special_tokens=False)

    def encode_with_offsets(self, text):
        if self.backend == 'tokenizers':
            encoded = self._tk.encode(text, add_special_tokens=False)
            return encoded.ids, encoded.offsets
        encoded = self._tk(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded['input_ids'], encoded['offset_mapping']

    def token_to_id(self, token: str) -> Optional[int]:
        if self.backend == "tokenizers":
            return self._tk.token_to_id(token)
        tid = self._tk.convert_tokens_to_ids(token)
        return None if tid == getattr(self._tk, "unk_token_id", None) else tid


def load_encoding_module(model_dir: str):
    path = os.path.join(model_dir, "encoding", "encoding.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"encoding/encoding.py not found in {model_dir}")
    spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------

def _num(body: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise APIError(400, f"`{key}` must be a number", param=key)
    if not (lo <= v <= hi):
        raise APIError(400, f"`{key}` must be in [{lo}, {hi}]", param=key)
    return float(v)


def parse_sampling(body: dict) -> dict:
    mt = body.get("max_completion_tokens", body.get("max_tokens", DEFAULT_MAX_TOKENS))
    if mt is None:
        mt = DEFAULT_MAX_TOKENS
    if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
        raise APIError(400, "`max_tokens` must be a positive integer", param="max_tokens")
    stop = body.get("stop")
    if stop is None:
        stops: List[str] = []
    elif isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        stops = list(stop)
    else:
        raise APIError(400, "`stop` must be a string or a list of strings", param="stop")
    stops = [s for s in stops if s]
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise APIError(400, "`seed` must be an integer", param="seed")
    n = body.get("n", 1)
    if n not in (None, 1):
        raise APIError(400, "only n=1 is supported", param="n")
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise APIError(400, "`ignore_eos` must be a boolean", param="ignore_eos")
    # Off switch for the tool-call grammar, per request. It exists for the A/B (is a difference the
    # constraint's doing?) and as an escape hatch for a tool schema the builder gets wrong.
    tg = body.get("tool_grammar", True)
    if not isinstance(tg, bool):
        raise APIError(400, "`tool_grammar` must be a boolean", param="tool_grammar")
    return {
        "max_tokens": mt,
        "temperature": _num(body, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
        "top_p": _num(body, "top_p", DEFAULT_TOP_P, 0.0, 1.0),
        # OpenAI's two repetition controls, plus the no-repeat-ngram guard. Defaults come from the
        # environment so an operator can set them per deployment: long prose on a pruned expert set
        # drifts into repeated rhetorical structure, which is what a frequency penalty is for, while
        # code is legitimately repetitive and wants them at 0 (NOTES 2026-09-12).
        "presence_penalty": _num(body, "presence_penalty", float(os.environ.get("DSV41_PRESENCE_PENALTY", "0")), -2.0, 2.0),
        "frequency_penalty": _num(body, "frequency_penalty", float(os.environ.get("DSV41_FREQUENCY_PENALTY", "0")), -2.0, 2.0),
        "stop": stops,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "tool_grammar": tg,
    }


def resolve_thinking(body: dict, default_thinking: bool, default_effort: int) -> Tuple[bool, int]:
    """Return (thinking_on, effort_int) per the precedence documented in README."""
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise APIError(400, "`chat_template_kwargs` must be an object", param="chat_template_kwargs")

    thinking: Optional[bool] = None
    for src, key in ((ctk, "thinking"), (body, "enable_thinking"), (ctk, "enable_thinking")):
        v = src.get(key)
        if v is not None:
            if not isinstance(v, bool):
                raise APIError(400, f"`{key}` must be a boolean", param=key)
            thinking = v
            break

    raw = None
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        raw = reasoning["effort"]
    elif body.get("reasoning_effort") is not None:
        raw = body["reasoning_effort"]
    elif ctk.get("reasoning_effort") is not None:
        raw = ctk["reasoning_effort"]

    effort: Optional[int] = None
    effort_thinking: Optional[bool] = None
    if raw is not None:
        if isinstance(raw, bool):
            raise APIError(400, "reasoning effort must be a string or an integer 1-100", param="reasoning_effort")
        if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            raw = int(raw.strip())
        if isinstance(raw, int):
            if not 1 <= raw <= 100:
                raise APIError(400, "integer reasoning effort must be within 1-100", param="reasoning_effort")
            effort, effort_thinking = raw, True
        elif isinstance(raw, str):
            name = raw.strip().lower()
            if name == "none":
                effort_thinking = False
            elif name in EFFORT_ALIASES:
                effort = EFFORT_ALIASES[name]
                effort_thinking = name not in NO_THINKING_EFFORTS
            else:
                raise APIError(400, f"unknown reasoning effort {raw!r}; use none/low/medium/high/xhigh/max or 1-100",
                               param="reasoning_effort")
        else:
            raise APIError(400, "reasoning effort must be a string or an integer 1-100", param="reasoning_effort")

    if thinking is None:
        thinking = effort_thinking if effort_thinking is not None else default_thinking
    if effort is None:
        effort = default_effort
    return thinking, effort


_IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}


# Set once in main(), from the engine: this process serves one model, and the request path needs
# to know whether image blocks can be honoured before it has an engine handle.
VISION_OK = False


def _reject_images(messages: List[dict]) -> None:
    for i, m in enumerate(messages):
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
                    if not VISION_OK:
                        raise APIError(400, "image inputs are not supported by this server "
                                            "(no vision tower loaded)", param=f"messages[{i}].content")


# A *typed* `<｜deepseek_image｜>` is refused by the checkpoint's reference encoder, because the
# tokenizer reads that byte string as the reserved image id 129264 -- an image with no pixels
# behind it. That is a reasonable thing to refuse, but the refusal takes the whole request with
# it, and coding agents put the literal into tool results simply by reading this repo's sources,
# logs or the checkpoint's own `encoding.py`. Split the token with a zero-width space instead:
# the prompt reads the same characters as ordinary BPE, and the image id never appears without
# an image to splice. Real image blocks do not pass through here -- the encoder turns those into
# the placeholder after this point, once their pixels are known to exist.
IMAGE_TOKEN = "<｜deepseek_image｜>"
IMAGE_TOKEN_ESCAPED = "<\u200b｜deepseek_image｜>"  # zero-width space after "<", invisible but not the id


def escape_image_tokens(obj: Any) -> Any:
    """Replace a client-typed image placeholder in every string of a request structure.

    Applied to `messages`, `tools` and a `response_format` schema before the reference encoder
    sees them. Image URLs and base64 payloads are never a bare placeholder, so the substitution
    only ever touches text a client actually typed. Tool-call arguments arrive as a JSON string;
    a Python client's `json.dumps` writes the fullwidth bars as `\\uff5c`, which a raw substring
    replace would miss, so those are decoded, escaped and re-serialised instead."""
    if isinstance(obj, str):
        return obj.replace(IMAGE_TOKEN, IMAGE_TOKEN_ESCAPED) if IMAGE_TOKEN in obj else obj
    if isinstance(obj, list):
        return [escape_image_tokens(v) for v in obj]
    if isinstance(obj, dict):
        function = obj.get("function")
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            obj = dict(obj)
            obj["function"] = dict(function, arguments=_escape_json_string(function["arguments"]))
        return {k: escape_image_tokens(v) for k, v in obj.items()}
    return obj


def _escape_json_string(value: str) -> str:
    """Escape a JSON-encoded tool-call argument string without trusting its shape.

    The reference encoder parses these with `json.loads` and renders the decoded values, so an
    escaped placeholder has to be re-encoded as JSON, not spliced into the source bytes."""
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return escape_image_tokens(value)
    return json.dumps(escape_image_tokens(decoded), ensure_ascii=False)


# DSV41_LOG_PROMPT=1 logs the RENDERED prompt of every request -- what the model actually sees
# after chat templating, thinking tags and tool rendering, which is usually not what the caller
# thinks it sent. Off by default because it writes user content to the log.
LOG_PROMPT = os.environ.get("DSV41_LOG_PROMPT", "0") == "1"
LOG_PROMPT_CHARS = int(os.environ.get("DSV41_LOG_PROMPT_CHARS", "2000"))

# Model-ready request capture, for replay and for diffing what two requests actually sent.
# DSV41_CAPTURE_NEXT=N arms the next N requests and then disarms itself, so later user traffic
# is never accumulated. N=1 keeps the original filename (results/captured_request.pt) that
# bench/replay_capture.py expects; N>1 numbers the files (captured_request_0.pt, _1.pt, ...)
# so prompts that share a prefix can be compared token-for-token.
_CAPTURE = {"total": None, "done": 0}


def _log_prompt(prompt, ids, thinking, effort):
    if not LOG_PROMPT:
        return
    body = prompt if len(prompt) <= LOG_PROMPT_CHARS else (
        prompt[:LOG_PROMPT_CHARS // 2] + f"\n...[{len(prompt) - LOG_PROMPT_CHARS} chars elided]...\n"
        + prompt[-LOG_PROMPT_CHARS // 2:])
    log.info("prompt: %d tokens, thinking=%s effort=%s\n--- BEGIN PROMPT ---\n%s\n--- END PROMPT ---",
             len(ids), thinking, effort, body)
    log.info("prompt head ids: %s ... tail ids: %s", ids[:12], ids[-12:])


def build_chat_prompt(body: dict, enc, tok: Tok, thinking: bool,
                      effort: int, engine=None, focus_out=None) -> Tuple[str, List[int], Optional[List[dict]], Optional[tuple]]:
    """Render a chat request. The third element is the tool list as the *model* sees it
    (namespaces folded into the names), which is what a tool grammar has to be built from; the
    fourth is (token_types, image_inputs) when the request carries images, else None.

    The VL inputs are RETURNED rather than attached to the engine here: this runs before the
    request lock, so attaching would let two requests overwrite each other's images and the
    engine is single-sequence. State.generate applies them under the lock."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty list", param="messages")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise APIError(400, f"messages[{i}] must be an object with a `role`", param=f"messages[{i}]")
    messages = copy.deepcopy(messages)
    # OpenAI renamed the system role to "developer" for its newer models and harnesses send
    # it by default. V4.1 has no developer role and an unknown role is a hard error in the
    # encoder, so fold it into system -- including mid-conversation, which the encoder
    # supports. Do this before the tools/schema check below so they ride the first message
    # instead of spawning a second, empty system one.
    for m in messages:
        if m["role"] == "developer":
            m["role"] = "system"
    # Escape before _reject_images and before the reference encoder: a typed placeholder is text,
    # not an image request, and it must not reach the tokenizer as id 129264.
    messages = escape_image_tokens(messages)
    _reject_images(messages)

    # Tools and response_format ride on the first message; the encoder only
    # renders them for a system message, so add an empty one if needed.
    tools = body.get("tools")
    if body.get("tool_choice") == "none":
        tools = None
    if tools is not None and not isinstance(tools, list):
        raise APIError(400, "`tools` must be a list", param="tools")
    # The schemas are rendered into the prompt too, so a typed placeholder can hide there.
    tools = escape_image_tokens(tools)
    schema = None
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        schema = escape_image_tokens((rf.get("json_schema") or {}).get("schema"))
    if tools or schema:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if tools:
            messages[0]["tools"] = tools
        if schema:
            messages[0]["response_format"] = schema

    try:
        prompt, media = enc.encode_messages(
            copy.deepcopy(messages),
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort,
            return_multi_modal_data=True,
        )
    except (AssertionError, ValueError, KeyError, TypeError, NotImplementedError) as e:
        raise APIError(400, f"cannot encode messages: {e}", param="messages")
    if media.get("images") and not VISION_OK:
        raise APIError(400, "image inputs are not supported by this server "
                            "(no vision tower loaded)", param="messages")
    rendered = None
    if tools:
        try:
            rendered = enc.tools_from_openai_format(copy.deepcopy(tools))
        except Exception as e:  # noqa: BLE001 - the encoder already accepted these
            log.warning("cannot normalise the tool list for the grammar: %s", e)
    images = media.get("images")
    if images and focus_out is not None and getattr(engine, 'user_prompt_stream', False):
        raise APIError(400, 'latest-user streaming currently requires text-only input', param='messages')
    if images and engine is not None and getattr(engine, "vision", None) is not None:
        try:
            ids, types, imgs = engine.prepare_vl(prompt, images)
        except Exception as e:  # noqa: BLE001
            raise APIError(400, f"cannot process image inputs: {e}", param="messages")
        return prompt, ids, rendered, (types, imgs)
    ids = tok.encode(prompt)
    if focus_out is not None and getattr(engine, 'user_prompt_stream', False):
        from server.latest_user import latest_user_ranges
        # Image expansion changes token offsets; this text-only prototype fails
        # closed for VL instead of assigning image/tool rows to the user question.
        if images:
            raise APIError(400, 'latest-user streaming currently requires text-only input', param='messages')
        try:
            offset_ids, offsets = tok.encode_with_offsets(prompt)
            if offset_ids != ids:
                raise ValueError('tokenizer offsets differ from model token IDs')
            def render_marked(marked):
                return enc.encode_messages(copy.deepcopy(marked),
                    thinking_mode='thinking' if thinking else 'chat', reasoning_effort=effort,
                    return_multi_modal_data=True)[0]
            focus_out['ranges'] = latest_user_ranges(messages, render_marked, prompt, offsets)
        except Exception as exc:
            raise APIError(400, f'cannot locate latest user text: {type(exc).__name__}', param='messages')
    return prompt, ids, rendered, None


# ---------------------------------------------------------------------------
# Incremental detokenization and output routing
# ---------------------------------------------------------------------------

class IncrementalDetokenizer:
    """Turn token bursts into text without emitting half a UTF-8 character.

    Text is only released once decoding a window of recent tokens is stable
    (does not end in U+FFFD); the window is re-decoded from ``prefix`` so that
    byte-fallback tokens that need 2-4 pieces are joined correctly.
    """

    def __init__(self, tok: Tok) -> None:
        self.tok = tok
        self.ids: List[int] = []
        self.prefix = 0   # start of the re-decode window
        self.read = 0     # end of the text already released

    def push(self, new_ids: List[int]) -> str:
        self.ids.extend(new_ids)
        full = self.tok.decode(self.ids[self.prefix:])
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        if len(full) > len(prev) and not full.endswith("�"):
            out = full[len(prev):]
            self.prefix, self.read = self.read, len(self.ids)
            return out
        return ""

    def flush(self) -> str:
        full = self.tok.decode(self.ids[self.prefix:])
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        self.prefix = self.read = len(self.ids)
        return full[len(prev):] if len(full) > len(prev) else ""


def _held_suffix_len(text: str, markers: List[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a marker."""
    best = 0
    for m in markers:
        for k in range(min(len(m) - 1, len(text)), 0, -1):
            if text.endswith(m[:k]):
                best = max(best, k)
                break
    return best


class OutputRouter:
    """Route streamed text into reasoning / content, detect stop strings and tool calls."""

    def __init__(self, thinking: bool, stop_strings: List[str], detect_tool_calls: bool) -> None:
        self.phase = "reasoning" if thinking else "content"
        self.pending = ""
        self.stop_strings = stop_strings
        self.detect_tool_calls = detect_tool_calls
        self.reasoning = ""
        self.content = ""
        self.tool_text = ""
        self.stopped = False       # a stop string was hit
        self.saw_think_end = thinking is False  # in chat mode the prompt already closed <think>

    def _markers(self) -> List[str]:
        if self.phase == "reasoning":
            return [THINK_END] + self.stop_strings
        if self.phase == "content":
            return ([TOOL_CALLS_MARKER] if self.detect_tool_calls else []) + self.stop_strings
        return []

    def _emit(self, text: str) -> Optional[Tuple[str, str]]:
        if not text:
            return None
        if self.phase == "reasoning":
            self.reasoning += text
        else:
            self.content += text
        return (self.phase, text)

    def feed(self, text: str) -> List[Tuple[str, str]]:
        """Return a list of (kind, text) events; kind is "reasoning" or "content"."""
        events: List[Tuple[str, str]] = []
        self.pending += text
        while self.pending and not self.stopped:
            if self.phase == "tool":
                self.tool_text += self.pending
                self.pending = ""
                break
            markers = self._markers()
            hit, pos = None, len(self.pending)
            for m in markers:
                p = self.pending.find(m)
                if p != -1 and p < pos:
                    hit, pos = m, p
            if hit is None:
                keep = _held_suffix_len(self.pending, markers)
                ev = self._emit(self.pending[:len(self.pending) - keep])
                if ev:
                    events.append(ev)
                self.pending = self.pending[len(self.pending) - keep:]
                break
            ev = self._emit(self.pending[:pos])
            if ev:
                events.append(ev)
            if hit == THINK_END and self.phase == "reasoning":
                self.phase = "content"
                self.saw_think_end = True
                self.pending = self.pending[pos + len(hit):]
            elif hit == TOOL_CALLS_MARKER and self.phase == "content":
                self.phase = "tool"
                self.pending = self.pending[pos:]
            else:  # stop string
                self.stopped = True
                self.pending = ""
        return events

    def finish(self) -> List[Tuple[str, str]]:
        """Release whatever is still held back (generation is over)."""
        events: List[Tuple[str, str]] = []
        if self.phase == "tool":
            self.tool_text += self.pending
        elif not self.stopped:
            ev = self._emit(self.pending)
            if ev:
                events.append(ev)
        self.pending = ""
        return events


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

class GenerationResult:
    def __init__(self) -> None:
        self.gen_ids: List[int] = []
        self.finish_reason = "stop"
        self.reasoning_tokens = 0
        self.tool_calls: List[dict] = []
        self.router: Optional[OutputRouter] = None
        self.stats: dict = {}


class State:
    """Everything the handlers share."""

    def __init__(self, args: argparse.Namespace, tok: Tok, enc, engine: Engine) -> None:
        self.args = args
        self.tok = tok
        self.enc = enc
        self.engine = engine
        self.lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.model_name = args.served_model_name
        self.bos_id = tok.token_to_id(enc.bos_token)
        self.eos_id = tok.token_to_id(enc.eos_token)
        self.think_end_id = tok.token_to_id(THINK_END)
        self.started = int(time.time())
        # EP2 (docs/dual-spark-plan.md): when the engine joined a process group, every accepted
        # request must be mirrored to the peer BEFORE it is driven -- the peer computes this
        # request's half of every routed-expert layer, and its per-layer combine is the only
        # rendezvous point. Inert (ep_active False) for the whole single-box path.
        _ep = getattr(engine, "ep", None)
        self.ep = _ep
        self.ep_active = bool(_ep is not None and _ep.active)
        # Set once the pair can no longer be trusted to be in step: a request was broadcast to
        # rank 1 and then died on rank 0 somewhere the peer could not learn about. From there
        # the peer is either gone (it exits on its own failures) or waiting on a collective
        # nobody will issue, so the next request would hang for DSV41_DIST_TIMEOUT_S instead of
        # failing. Restart is the only recovery in this recipe; say so in a 503 and in /health.
        self.ep_fault: Optional[str] = None
        if self.eos_id != engine.eos_token_id:
            log.warning("engine.eos_token_id=%s differs from tokenizer EOS id %s", engine.eos_token_id, self.eos_id)
        # Constrained tool calls: only with an engine that can mask its own sampling, and only
        # when xgrammar is installed. Without it everything below is a no-op and tool calls are
        # parsed out of the text as before.
        self.grammars = None
        if getattr(engine, "supports_grammar", False):
            self.grammars = make_factory(
                tok, engine.eos_token_id if self.eos_id is None else self.eos_id,
                enabled=os.environ.get("DSV41_TOOL_GRAMMAR", "0") == "1")  # off until the end-to-end gates on real weights have run; see NOTES 2026-09-11
        self.scheduler = None
        from engine.serving import concurrency
        if concurrency() > 1:
            from engine.serving import DecodeRuntime
            from server.concurrency import Scheduler
            self.scheduler = Scheduler(self, DecodeRuntime(engine))

    def request_context(self):
        return nullcontext() if self.scheduler is not None else self.lock

    def _decode_probe_preflight(self, method):
        if self.scheduler is not None:
            raise APIError(501, 'Decode probe requires concurrency=1', code='decode_probe_unsupported')
        if self.ep_fault:
            raise APIError(503, f'Engine pair is faulted: {self.ep_fault}',
                           err_type='server_error', code='engine_faulted')
        call = getattr(self.engine, method, None)
        if not callable(call):
            raise APIError(501, 'Decode probe requires the native v41 engine',
                           code='decode_probe_unsupported')
        return call

    def decode_probe_status(self):
        """Cached metadata only: observer polls must never acquire the inference lock."""
        call = self._decode_probe_preflight('decode_probe_status')
        try:
            return call()
        except Exception as e:
            raise APIError(500, f'Decode probe status failed: {e}',
                           err_type='server_error', code='decode_probe_failed') from e

    def decode_probe(self, body):
        """Validate on the head, then dispatch one command to both ranks under the lock."""
        call = self._decode_probe_preflight('decode_probe_command')
        from engine.decode_probe import ProbeConfig
        try:
            opts = asdict(ProbeConfig.from_dict(body))
            opts['names'] = list(opts['names'])
            if opts['action'] == 'run':
                # Omitted replay settings inherit the armed probe, rather than silently
                # resetting its filters/budget to constructor defaults.
                opts = {key: value for key, value in opts.items()
                        if key == 'action' or key in body}
        except (TypeError, ValueError) as e:
            raise APIError(400, str(e), param='decode_probe', code='invalid_decode_probe') from e
        if not self.lock.acquire(blocking=False):
            raise APIError(409, 'Engine is busy; retry decode probe between requests',
                           code='engine_busy')
        try:
            # A previous owner may have faulted the pair while this request was validating.
            self._decode_probe_preflight('decode_probe_command')
            if self.ep_active:
                self.ep.broadcast_request({'cmd': 'decode_probe', 'body': opts})
            # Expected local diagnostic failures are caught by the engine before its one
            # unconditional rank gather. An exception here means that contract was broken.
            return call(opts)
        except APIError:
            raise
        except Exception as e:
            self.ep_fault = f'decode probe command failed: {type(e).__name__}: {e}'
            log.exception('decode probe command failed; engine is faulted')
            raise APIError(503, self.ep_fault, err_type='server_error',
                           code='engine_faulted') from e
        finally:
            self.lock.release()

    def cache_response(self, prompt_ids, result, *, thinking=False, vision=False):
        """Called only after a successful response is flushed, still under request_context.

        Concurrency=2 queues work for its GPU owner, only when both lanes are idle.
        Thinking/tool/stop-string serialization may not preserve raw IDs.
        """
        if (os.environ.get('DSV41_PREFIX_RESPONSE', '1') != '1'
                or self.ep_fault or thinking or vision or result.tool_calls
                or not result.gen_ids or result.router.stopped
                or not hasattr(self.engine, 'cache_response')):
            return
        if self.scheduler is not None:
            self.scheduler.cache_response(prompt_ids, result.gen_ids, getattr(result, '_cache_lane', None))
            return
        try:
            if self.ep_active:
                self.ep.broadcast_request({'cmd': 'cache_response', 'prompt_ids': list(prompt_ids),
                                           'response_ids': list(result.gen_ids)})
            self.engine.cache_response(prompt_ids, result.gen_ids)
        except Exception:
            # Response is already delivered. Do not try to write an HTTP error, or
            # continue with a peer potentially stuck inside a replay collective.
            log.exception('post-response prefix preparation failed')
            if self.ep_active:
                os._exit(1)

    def stop_ids(self) -> Set[int]:
        ids = {self.engine.eos_token_id}
        if self.eos_id is not None:
            ids.add(self.eos_id)
        return ids

    def maintain_experts(self) -> bool:
        """Re-fit the resident experts to observed demand. Returns True if anything was moved.

        The caller must hold ``self.lock`` and the pair must be between requests: rank 1 is back
        in broadcast_request, so the maintain command lands on the same path the keepalive uses.

        A cycle costs ~0.1 s for 128 slots, so this runs at the end of every response rather than
        waiting for an idle tick -- a workload that switches between news, research and coding
        moves the resident set within a request or two instead of a couple of minutes.
        """
        if not self.ep_active or self.ep_fault:
            return False
        try:
            if not self.engine.maintain_demand():
                return False
            swaps = self.engine.plan_swaps(
                max_swaps=ADAPT.swap_max)
        except Exception as e:  # noqa: BLE001
            # Planning is rank-local and nothing has been broadcast yet, so giving up here is
            # safe -- unlike a failure after the broadcast, which desyncs the pair.
            log.warning("expert adaptation planning failed (%s: %s); skipping", type(e).__name__, e)
            return False
        if not swaps:
            return False
        self.ep.broadcast_request({"cmd": "maintain", "swaps": swaps})
        t0 = time.time()
        try:
            n = self.engine.apply_swaps(swaps)
        except Exception:
            # The peer has the plan and is applying it. If this rank does not finish the same
            # work the two routers disagree about which experts are routable and the pair
            # diverges silently, which is worse than stopping.
            log.exception("expert adaptation failed on rank 0; the pair would desync, exiting")
            os._exit(1)
        log.info("expert adaptation: %d slots in %.2fs", n, time.time() - t0)
        return True

    def generate(self, prompt_ids: List[int], sampling: dict, *, thinking: bool,
                 detect_tool_calls: bool, result: GenerationResult,
                 tools: Optional[List[dict]] = None, vl: Optional[tuple] = None,
                 user_prompt_ranges=None) -> Iterator[Tuple[str, str]]:
        """Drive the engine; yield (kind, text) events; fill ``result`` at the end.

        The caller holds ``request_context()``. With concurrency enabled, only
        the scheduler thread holds the GPU/peer lock.
        """
        # Attach this request's image inputs under the lock, so two requests cannot overwrite
        # each other's; the engine clears them once prefill has consumed them.
        if self.scheduler is None and vl is not None and hasattr(self.engine, "set_vl_inputs"):
            self.engine.set_vl_inputs(*vl)
        max_ctx = self.engine.max_context
        # Headroom the engine needs beyond prompt + max_tokens (DSpark verifies a
        # 6-token block, the V4.1 engine asserts an 8-token margin).
        margin = int(getattr(self.engine, "context_margin", 8))
        if len(prompt_ids) + margin >= max_ctx:
            raise APIError(400, f"prompt has {len(prompt_ids)} tokens, engine context is {max_ctx}",
                           code="context_length_exceeded")
        max_tokens = min(sampling["max_tokens"], max_ctx - len(prompt_ids) - margin)
        # ignore_eos: benchmarks need a fixed output length, so neither the engine nor this loop
        # may stop early. The stop set is emptied on both sides -- passing it to the engine and
        # then truncating here would give a short run anyway.
        ignore_eos = bool(sampling.get("ignore_eos"))
        stop_ids: Set[int] = set() if ignore_eos else self.stop_ids()
        detok = IncrementalDetokenizer(self.tok)
        router = OutputRouter(thinking, sampling["stop"], detect_tool_calls)
        result.router = router
        t0 = time.perf_counter()

        gen_kwargs = dict(max_tokens=max_tokens, temperature=sampling["temperature"],
                          top_p=sampling["top_p"], stop_token_ids=stop_ids, seed=sampling["seed"])
        if getattr(self.engine, 'user_prompt_stream', False):
            gen_kwargs['user_prompt_ranges'] = user_prompt_ranges or []
        pen = None
        if getattr(self.engine, "supports_penalties", False):
            from engine.v41_engine import Penalties
            pen = Penalties(presence=sampling["presence_penalty"], frequency=sampling["frequency_penalty"])
            if pen.active:
                gen_kwargs["penalties"] = pen
        # The gate constrains nothing until the model opens a tool-calls block, so it costs a
        # dictionary lookup per step on a request that never calls a tool.
        gate = None
        if tools and detect_tool_calls and self.grammars is not None and sampling["tool_grammar"]:
            gate = self.grammars.for_tools(tools)
        if gate is not None:
            gen_kwargs["grammar"] = gate
        # Capture of the model-ready load for replay and for diffing two requests that should
        # share a prefix. It intentionally stores token ids rather than rendered prompt text and
        # disarms itself after DSV41_CAPTURE_NEXT requests, so later user traffic is never
        # accumulated. See _CAPTURE for the filename convention.
        with self.capture_lock:
            if _CAPTURE["total"] is None:
                _raw = os.environ.pop("DSV41_CAPTURE_NEXT", "0") or "0"
                _CAPTURE["total"] = int(_raw) if _raw.isdigit() else 0
            if _CAPTURE["done"] < _CAPTURE["total"]:
                import torch
                captured = {"prompt_ids": list(prompt_ids), "kwargs": gen_kwargs,
                            "sampling": dict(sampling), "thinking": thinking}
                if vl is not None:
                    types, imgs = vl
                    captured["vl"] = {"token_types": types.detach().cpu(), "images": imgs}
                total, i = _CAPTURE["total"], _CAPTURE["done"]
                name = "captured_request.pt" if total == 1 else f"captured_request_{i}.pt"
                path = os.path.join(REPO_ROOT, "results", name)
                tmp = path + ".tmp"
                torch.save(captured, tmp)
                os.replace(tmp, path)
                _CAPTURE["done"] += 1
                if _CAPTURE["done"] < total:
                    log.warning("captured model-ready request %d/%d at %s; %d more armed",
                                _CAPTURE["done"], total, path, total - _CAPTURE["done"])
                else:
                    log.warning("captured model-ready request %d/%d at %s; capture disarmed",
                                _CAPTURE["done"], total, path)
        if self.ep_active:
            # Lockstep contract (engine/dist.py): rank 0 decides, the peer obeys. Three rules:
            #  * seed must be concrete and SHARED -- seed=None would leave each rank drawing its
            #    own rejection-sampling numbers, and the first divergent token means mismatched
            #    collective counts: an NCCL hang, not a wrong answer.
            #  * the grammar gate is a rank-0 decoding constraint built from the request's tool
            #    schemas, which the peer never sees -- refused rather than silently desyncing.
            #    (Tool CALLING still works: DSV41_TOOL_GRAMMAR=0 is the tolerant-parser path.)
            #  * ignore_eos rides inside kwargs so the peer's step count clamps identically.
            if gen_kwargs["seed"] is None:
                gen_kwargs["seed"] = random.randint(0, 2 ** 31 - 1)
            if gate is not None:
                raise APIError(501, "tool-call grammar gating is not available in EP2 mode",
                               code="ep2_unsupported")
            if ignore_eos:
                gen_kwargs["ignore_eos"] = True
            if self.ep_fault:
                raise APIError(503, f"EP2 pair is out of step ({self.ep_fault}); restart both ranks",
                               code="ep2_desynced")
            # Anything that can raise on rank 0 must raise BEFORE the broadcast, or rank 1 takes
            # the request, hits the same failure, and exits -- a client-triggered way to lose the
            # peer. This one is the engine's own context assert, reachable from any HTTP request.
            over = len(prompt_ids) + max_tokens + 8 - self.engine.max_context
            if over > 0:
                raise APIError(400, f"prompt {len(prompt_ids)} + max_tokens {max_tokens} exceeds "
                                    f"context {self.engine.max_context} by {over} tokens",
                               code="context_length_exceeded", param="max_tokens")
            # The VL inputs travel WITH the request. Without them rank 1 prefills the same ids
            # with no image: it skips the splice, and -- fatally -- chunks uniformly where rank 0
            # aligns its boundaries to the image span, so the two ranks issue different numbers of
            # per-layer all-reduces and the pair deadlocks. That is a hang with an idle GPU, not an
            # error. token_types is small; the patches are ~5 MB and go over the same gloo path.
            payload = {"prompt_ids": list(prompt_ids), "kwargs": gen_kwargs}
            if vl is not None:
                types, imgs = vl
                payload["vl"] = {"token_types": types.detach().cpu(), "images": imgs}
            if self.scheduler is None:
                self.ep.broadcast_request(payload)
        if self.scheduler is not None:
            import queue
            try:
                gen = self.scheduler.submit(payload)
            except queue.Full:
                raise APIError(503, 'concurrent request queue is full', code='server_busy')
            except RuntimeError as exc:
                raise APIError(503, str(exc), code='ep2_desynced')
        elif ignore_eos:
            try:
                gen = self.engine.generate(prompt_ids, ignore_eos=True, **{k: v for k, v in gen_kwargs.items() if k != "ignore_eos"})
            except TypeError:  # an engine without the kwarg (MockEngine): empty stop set is enough
                gen = self.engine.generate(prompt_ids, **{k: v for k, v in gen_kwargs.items() if k != "ignore_eos"})
        else:
            gen = self.engine.generate(prompt_ids, **gen_kwargs)
        hit_eos = False
        progress = DecodeProgress(log, len(prompt_ids), max_tokens, thinking, self.think_end_id)
        try:
            for burst in gen:
                burst = list(burst)
                for i, t in enumerate(burst):
                    if t in stop_ids:
                        burst, hit_eos = burst[:i], True
                        break
                room = max_tokens - len(result.gen_ids)
                if len(burst) > room:
                    burst = burst[:room]
                result.gen_ids.extend(burst)
                progress.update(burst)
                for ev in router.feed(detok.push(burst)):
                    yield ev
                if hit_eos or router.stopped or len(result.gen_ids) >= max_tokens:
                    break
            if not router.stopped:
                # EOS or length: the engine's own loop ends right after this
                # yield, so let it run its epilogue (stats) instead of closing
                # it mid-flight. Anything it still emits is ignored; an engine
                # that keeps going is cut off after a few bursts.
                for _ in range(4):
                    if next(gen, None) is None:
                        break
        except Exception as e:
            # GeneratorExit is deliberately NOT caught here: a stop string or a client
            # disconnect closes this generator, `gen.close()` below raises GeneratorExit into
            # the decode loop, and engine/v41_engine.py releases rank 1 on the way out -- a
            # clean, routine abort with the pair still in step.
            #
            # A real exception is the other case. The engine died mid-request, or the release
            # itself failed, and nothing can prove rank 1 is still on the same step. Latch it:
            # the next request gets a 503 telling the operator to restart the pair, instead of
            # blocking for DSV41_DIST_TIMEOUT_S on a collective that will never be matched.
            if self.ep_active:
                self.ep_fault = f"{type(e).__name__}: {e}"[:200]
                log.error("EP2: request failed after the peer took it (%s); pair marked out of step", e)
            raise
        finally:
            gen.close()

        for ev in router.feed(detok.flush()):
            yield ev
        for ev in router.finish():
            yield ev

        if hit_eos or router.stopped:
            result.finish_reason = "stop"
        elif len(result.gen_ids) >= max_tokens:
            result.finish_reason = "length"
        if thinking:
            try:
                result.reasoning_tokens = result.gen_ids.index(self.think_end_id)
            except ValueError:
                result.reasoning_tokens = len(result.gen_ids)

        if detect_tool_calls:
            result.tool_calls = self._parse_tool_calls(router, thinking)
            if result.tool_calls:
                result.finish_reason = "tool_calls"

        try:
            result.stats = dict(gen.stats if self.scheduler is not None else (self.engine.stats() or {}))
            result._cache_lane = getattr(gen, 'lane', None)
        except Exception as e:  # engine stats must never break a response
            log.warning("engine.stats() failed: %s", e)
            result.stats = {}
        if gate is not None:
            st = dict(gate.stats)
            st["mask_ms_per_call"] = round(st["mask_s"] / st["mask_calls"] * 1e3, 3) if st["mask_calls"] else None
            st["mask_s"] = round(st["mask_s"], 4)
            result.stats["tool_grammar"] = st
        dt = time.perf_counter() - t0
        result.stats.setdefault("server_completion_tok_per_s", round(len(result.gen_ids) / dt, 1) if dt > 0 else None)
        log.info("generation done: request=%s prompt=%d completion=%d reasoning=%d finish=%s %.2fs",
                 progress.request_id, len(prompt_ids), len(result.gen_ids), result.reasoning_tokens,
                 result.finish_reason, dt)
        # The measured numbers used to live only in the response body (x_engine_stats), so watching
        # `docker logs` told you nothing about why a request was slow. This is the repo's own rule --
        # "a tok/s number without the hit rate and the GB that produced it is an anecdote" -- applied
        # to the log: one line carrying the figures a slow request has to be explained with.
        _s = result.stats or {}
        if _s:
            _cached = int(_s.get("prefix_cached_tokens") or 0)
            _prompt = int(_s.get("prompt_tokens") or len(prompt_ids))
            _reuse_pct = 100.0 * _cached / max(1, _prompt)
            # THIS request's routing misses, not the lifetime figure: the accumulators are
            # lifetime by design (they feed the demand database), so a cumulative rate stops
            # moving after a few hundred requests and says nothing about the prompt just run.
            _pm = _s.get("prune_miss_request") or {}
            _pm_txt = (f" | routed-miss {_pm['miss_rate']*100:.1f}% ({_pm['missed_slots']}/{_pm['slots']})"
                       if _pm else "")
            log.info("  prefill %.2fs (%s tok/s) | prefix=%d/%d (%.1f%%), suffix=%d | "
                     "decode %s tok/s accept=%s | hit=%s nvme=%sGB%s | ep=%sx collective=%ss",
                     _s.get("prefill_s") or 0.0, _s.get("prefill_tok_s"),
                     _cached, _prompt, _reuse_pct, _prompt - _cached, _s.get("decode_tok_s"),
                     _s.get("accept_len_mean"), _s.get("expert_hit_rate"), _s.get("nvme_gb"), _pm_txt,
                     _s.get("ep_world_size") or (_s.get("engine_config") or {}).get("ep_world_size", 1),
                     _s.get("ep_s"))
            for _k in ("attn_phases", "gpu_timing", "route_stats", "decode_accounting"):
                if _s.get(_k):
                    log.info("  %s: %s", _k, json.dumps(_s[_k], separators=(",", ":")))
        # Last step of the response. Both ranks have just finished the same generate, so this is
        # the earliest moment the pair is known synchronised and idle -- no need to wait for a
        # keepalive tick to notice.
        if self.scheduler is None:
            self.maintain_experts()

    # A completion that calls tools is DSML, and the checkpoint's own parser
    # (corpus/sources/dsv41_encoding.py::parse_message_from_completion_text) is strict about it: a
    # parameter must read `name="x" string="true|false">value<`, and nothing may follow the calls
    # block. Real completions deviated in two ways, and both were costing the whole tool call --
    # the client then saw only the sentence before it and a `stop` finish:
    #   * the value is put in the `string` attribute (`name="query" string="a b c"`), with no
    #     separate value body;
    #   * ordinary prose follows the closing tag of the calls block.
    # Both are now unreachable when a tool grammar is in force (server/tool_grammar.py): the mask
    # will not let the model write them. This tolerant pass is the fallback for the cases where
    # there is no grammar -- xgrammar missing, DSV41_TOOL_GRAMMAR=0, a schema the builder could not
    # express, an engine that cannot mask -- and it runs only after the strict parser has raised.
    _RE_INVOKE = re.compile(r'<｜DSML｜ invoke name="(?P<name>[^"]*)"\s*>?\n?(?P<body>.*?)(?=<｜DSML｜ invoke |</｜DSML｜ calls>|\Z)', re.DOTALL)
    _RE_PARAM_SPEC = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<s>true|false)"\s*>(?P<v>.*?)<', re.DOTALL)
    _RE_PARAM_ATTR = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<v>.*?)"\s*>', re.DOTALL)

    @classmethod
    def _parse_tool_calls_tolerant(cls, text: str) -> List[dict]:
        """Best-effort DSML tool calls. Returns [] when nothing parses."""
        out: List[dict] = []
        for m in cls._RE_INVOKE.finditer(text):
            name, body = m.group("name"), m.group("body")
            if not name:
                continue
            args: Dict[str, Any] = {}
            for pm in cls._RE_PARAM_SPEC.finditer(body):
                k, is_str, v = pm.group("k"), pm.group("s"), pm.group("v")
                if is_str == "true":
                    args[k] = v
                else:
                    try:
                        args[k] = json.loads(v)
                    except Exception:  # noqa: BLE001 - a malformed literal is still worth sending as text
                        args[k] = v
            consumed = {pm.group("k") for pm in cls._RE_PARAM_SPEC.finditer(body)}
            for pm in cls._RE_PARAM_ATTR.finditer(body):
                k, v = pm.group("k"), pm.group("v")
                if k and k not in consumed and k not in args:
                    args[k] = v
            out.append({"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return out

    def _parse_tool_calls(self, router: OutputRouter, thinking: bool) -> List[dict]:
        if not router.tool_text:
            return []
        text = ""
        if thinking:
            text += router.reasoning + THINK_END
        text += router.content + router.tool_text
        if not text.endswith(self.enc.eos_token):
            text += self.enc.eos_token
        try:
            parsed = self.enc.parse_message_from_completion_text(
                text, thinking_mode="thinking" if thinking else "chat")
        except Exception as e:
            recovered = self._parse_tool_calls_tolerant(text)
            if recovered:
                log.warning("tool-call strict parse failed (%s); recovered %d call(s) tolerantly",
                            e, len(recovered))
                parsed = {"tool_calls": recovered}
            else:
                log.warning("tool-call parse failed (%s); returning raw text as content", e)
                if os.environ.get("DSV41_LOG_TOOL_TEXT") == "1":
                    log.warning("unparsed tool text: %r", router.tool_text[:2000])
                router.content += router.tool_text
                router.tool_text = ""
                return []
        calls = []
        for tc in parsed.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            if tc.get("namespace"):
                name = f"{tc['namespace']}::{name}"
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": args}})
        return calls


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def usage_dict(prompt_tokens: int, result: GenerationResult) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(result.gen_ids),
        "total_tokens": prompt_tokens + len(result.gen_ids),
        "completion_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
    }


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dsv41-server/0.1"
    # A disconnected harness once held the request lock in final SSE sendall for
    # 902 seconds. Rank 1's idle collective expired at 600 seconds because the
    # heartbeat could not acquire that lock. Bound socket I/O, not model runtime:
    # a long prefill does no socket I/O and is unaffected by this timeout.
    timeout = 30.0

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if (self.command == 'GET' and self.path.split('?', 1)[0].rstrip('/') == '/v1/expert-map'
                and len(args) > 1 and str(args[1]) == '200'):
            return  # one-second observer polls must not bury generation/adaptation logs
        log.info("%s %s", self.address_string(), fmt % args)

    # -- plumbing ----------------------------------------------------------
    def _write_client(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except OSError as e:
            self.close_connection = True
            raise ClientWriteError(str(e)) from e

    def _send_json(self, status: int, obj: Any, *, cache_control=None) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if cache_control is not None:
            self.send_header('Cache-Control', cache_control)
        self.end_headers()
        self._write_client(data)

    def _send_error(self, err: APIError) -> None:
        self._send_json(err.status, err.body())

    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        if length is None:
            raise APIError(411, "Content-Length required")
        try:
            n = int(length)
        except ValueError:
            raise APIError(400, "bad Content-Length")
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise APIError(400, f"request body is not valid JSON: {e}")
        if not isinstance(body, dict):
            raise APIError(400, "request body must be a JSON object")
        if os.environ.get("DSV41_LOG_BODIES"):
            log.info("request body: %s", json.dumps(body)[:4000])
        return body

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj: Any) -> None:
        payload = b"data: " + (obj if isinstance(obj, bytes) else _json_bytes(obj)) + b"\n\n"
        self._write_client(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")

    def _end_sse(self) -> None:
        self._write_client(b"0\r\n\r\n")

    # -- routing -----------------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight for browser clients
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        st = self.state
        try:
            if path in ('/expert-map', '/decode-probe'):
                filename = 'decode_probe.html' if path == '/decode-probe' else 'expert_map.html'
                with open(os.path.join(HERE, filename), 'rb') as page:
                    data = page.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._write_client(data)
            elif path == "/v1/expert-map":
                from server.expert_map import snapshot
                body = snapshot(st.engine, busy=st.lock.locked(), fault=st.ep_fault, model=st.model_name)
                if body is None:
                    raise APIError(501, 'Expert map requires the native v41 engine')
                self._send_json(200, body, cache_control='no-store')
            elif path == '/v1/decode-probe':
                self._send_json(200, st.decode_probe_status(), cache_control='no-store')
            elif path == "/health":
                body = {"status": "degraded" if st.ep_fault else "ok",
                        "model": st.model_name, "engine": st.args.engine,
                        "busy": st.lock.locked(), "max_context": st.engine.max_context,
                        "uptime_s": int(time.time()) - st.started}
                scheduler = getattr(st, "scheduler", None)
                if scheduler is not None:
                    # Read without the GPU lock while the scheduler thread mutates these; a torn
                    # read is possible and must not fail the health check.
                    try:
                        rt = scheduler.runtime
                        body["concurrency"] = {
                            "lanes": scheduler.n, "lane_context": list(scheduler.capacity),
                            "active": sorted(rt.generators), "queued": scheduler.pending.qsize(),
                            "batch": rt.batch.memory_report() if hasattr(rt, "batch") else None}
                    except Exception:  # noqa: BLE001
                        body["concurrency"] = {"lanes": scheduler.n}
                if st.ep_fault:
                    # A wedged pair still answers /health; saying "ok" here is what would turn a
                    # dead peer into a silently half-serving cluster.
                    body["ep_fault"] = st.ep_fault
                # Static engine configuration (arena GB/slots, max_seq, spec, trace stats, kernel):
                # every measured number in RESULTS.md has to be quoted with the config that produced
                # it, and a bench run should not have to be told what the server was started with.
                cfg = getattr(st.engine, "config", None)
                if callable(cfg):
                    try:
                        body["engine_config"] = cfg()
                    except Exception as e:  # never let introspection break the health probe
                        log.warning("engine.config() failed: %s", e)
                self._send_json(503 if st.ep_fault else 200, body)
            elif path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [self._model_card()]})
            elif path.startswith("/v1/models/"):
                if path[len("/v1/models/"):] != st.model_name:
                    raise APIError(404, "model not found", code="model_not_found", param="model")
                self._send_json(200, self._model_card())
            else:
                raise APIError(404, f"no route for GET {path}")
        except APIError as e:
            self._send_error(e)

        except (ClientWriteError, BrokenPipeError, ConnectionResetError, TimeoutError):
            log.info("client disconnected")

    def _model_card(self) -> dict:
        st = self.state
        return {"id": st.model_name, "object": "model", "created": st.started,
                "owned_by": "deepseek-v41-flash-spark", "max_model_len": st.engine.max_context}

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            body = self._read_json()
            if path == "/v1/chat/completions":
                self._chat(body)
            elif path == "/v1/completions":
                self._completions(body)
            elif path == "/v1/debug/prompt":
                self._debug_prompt(body)
            elif path == '/v1/decode-probe':
                self._send_json(200, self.state.decode_probe(body), cache_control='no-store')
            else:
                raise APIError(404, f"no route for POST {path}")
        except APIError as e:
            self._send_error(e)
        except (ClientWriteError, BrokenPipeError, ConnectionResetError, TimeoutError):
            log.info("client disconnected")
        except Exception as e:  # anything else is a server bug
            log.exception("unhandled error")
            try:
                self._send_error(APIError(500, f"internal error: {e}", err_type="server_error"))
            except Exception:
                pass

    # -- endpoints ---------------------------------------------------------
    def _debug_prompt(self, body: dict) -> None:
        st = self.state
        thinking, effort = resolve_thinking(body, st.args.default_thinking, st.args.default_effort)
        prompt, ids, tools, vl = build_chat_prompt(body, st.enc, st.tok, thinking, effort, st.engine)
        body_out = {"thinking": thinking, "reasoning_effort": effort, "prompt": prompt,
                    "prompt_ids": ids, "prompt_tokens": len(ids)}
        if tools and st.grammars is not None:
            from tool_grammar import build_tool_grammar  # noqa: PLC0415 - debug endpoint only
            body_out["tool_grammar"] = build_tool_grammar(tools)
        self._send_json(200, body_out)

    def _command_reply(self, body: dict, text: str) -> None:
        """Answer an `@@` command without the engine (server/commands.py)."""
        model = self.state.model_name
        if not body.get("stream", False):
            self._send_json(200, dream_commands.completion(text, model))
            return
        self._start_sse()
        for c in dream_commands.stream_chunks(text, model):
            self._sse(c)
        self._sse(b"[DONE]")
        self._end_sse()

    def _chat(self, body: dict) -> None:
        st = self.state
        # `@@` commands (opt-in, DSV41_DREAM=1): answered here on rank 0 before anything is built or
        # broadcast, so the peer never learns a command happened -- no collective, no numerics.
        if dream_commands.enabled():
            cmd = dream_commands.match(body.get("messages"))
            if cmd is not None:
                self._command_reply(body, dream_commands.run(*cmd, where="awake"))
                return
        sampling = parse_sampling(body)
        thinking, effort = resolve_thinking(body, st.args.default_thinking, st.args.default_effort)
        focus = {}
        _prompt, prompt_ids, tools, vl = build_chat_prompt(body, st.enc, st.tok, thinking, effort, st.engine, focus)
        _log_prompt(_prompt, prompt_ids, thinking, effort)
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        result = GenerationResult()

        def chunk(delta: dict, finish: Optional[str] = None, extra: Optional[dict] = None) -> dict:
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": st.model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
            if extra:
                obj.update(extra)
            return obj

        with st.request_context():
            if not stream:
                for _ in st.generate(prompt_ids, sampling, thinking=thinking, detect_tool_calls=True, vl=vl,
                                     result=result, tools=tools, user_prompt_ranges=focus.get('ranges')):
                    pass
                router = result.router
                message: Dict[str, Any] = {"role": "assistant", "content": router.content}
                if thinking:
                    message["reasoning_content"] = router.reasoning
                if result.tool_calls:
                    message["tool_calls"] = result.tool_calls
                    if not router.content:
                        message["content"] = None
                self._send_json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": st.model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason,
                                 "logprobs": None}],
                    "usage": usage_dict(len(prompt_ids), result),
                    "x_engine_stats": result.stats,
                })
                self.wfile.flush()
                st.cache_response(prompt_ids, result, thinking=thinking, vision=vl is not None)
                conversation_log.record(body, thinking, effort, st.model_name, router.reasoning,
                                        router.content, result.tool_calls, result.finish_reason)
                return

            self._start_sse()
            completed = False
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                with closing(st.generate(prompt_ids, sampling, thinking=thinking, vl=vl,
                                         detect_tool_calls=True, result=result, tools=tools,
                                         user_prompt_ranges=focus.get('ranges'))) as events:
                    for kind, text in events:
                        self._sse(chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text}))
                if result.tool_calls:
                    self._sse(chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(result.tool_calls)]}))
                final_extra = {"usage": usage_dict(len(prompt_ids), result), "x_engine_stats": result.stats}
                self._sse(chunk({}, result.finish_reason, final_extra))
                if include_usage:
                    self._sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                               "model": st.model_name, "choices": [], **final_extra})
                completed = True
            except APIError as e:
                self._sse(e.body())
            except (ClientWriteError, BrokenPipeError, ConnectionResetError, TimeoutError) as e:
                log.info("client disconnected mid-stream after %d tokens (%s)", len(result.gen_ids), e)
                return
            except Exception as e:
                log.exception("engine error mid-stream")
                self._sse({"error": {"message": f"engine error: {e}", "type": "server_error",
                                     "param": None, "code": None}})
            self._sse(b"[DONE]")
            self._end_sse()
            if completed:
                st.cache_response(prompt_ids, result, thinking=thinking, vision=vl is not None)
                conversation_log.record(body, thinking, effort, st.model_name, result.router.reasoning,
                                        result.router.content, result.tool_calls, result.finish_reason)

    def _completions(self, body: dict) -> None:
        st = self.state
        sampling = parse_sampling(body)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        if isinstance(prompt, str):
            # Raw text goes straight to the tokenizer, so a typed placeholder would become the
            # image id with no image behind it. Escape it here too, for the same reason as chat.
            prompt = escape_image_tokens(prompt)
            prompt_ids = st.tok.encode(prompt)
            if body.get("add_bos", True) and st.bos_id is not None and (not prompt_ids or prompt_ids[0] != st.bos_id):
                prompt_ids = [st.bos_id] + prompt_ids
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            prompt_ids = list(prompt)
        else:
            raise APIError(400, "`prompt` must be a string or a list of token ids (batching is not supported)",
                           param="prompt")
        if body.get("echo"):
            raise APIError(400, "`echo` is not supported", param="echo")
        _log_prompt(prompt if isinstance(prompt, str) else f"<{len(prompt_ids)} raw token ids>",
                    prompt_ids, None, None)
        stream = bool(body.get("stream", False))
        rid = "cmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        result = GenerationResult()

        def obj(text: str, finish: Optional[str], extra: Optional[dict] = None) -> dict:
            o = {"id": rid, "object": "text_completion", "created": created, "model": st.model_name,
                 "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}
            if extra:
                o.update(extra)
            return o

        with st.request_context():
            if not stream:
                for _ in st.generate(prompt_ids, sampling, thinking=False, detect_tool_calls=False, result=result):
                    pass
                self._send_json(200, obj(result.router.content, result.finish_reason,
                                         {"usage": usage_dict(len(prompt_ids), result),
                                          "x_engine_stats": result.stats}))
                self.wfile.flush()
                st.cache_response(prompt_ids, result)
                return
            self._start_sse()
            completed = False
            try:
                with closing(st.generate(prompt_ids, sampling, thinking=False,
                                         detect_tool_calls=False, result=result)) as events:
                    for _, text in events:
                        self._sse(obj(text, None))
                self._sse(obj("", result.finish_reason,
                              {"usage": usage_dict(len(prompt_ids), result), "x_engine_stats": result.stats}))
                completed = True
            except APIError as e:
                self._sse(e.body())
            except (ClientWriteError, BrokenPipeError, ConnectionResetError, TimeoutError) as e:
                log.info("client disconnected mid-stream after %d tokens (%s)", len(result.gen_ids), e)
                return
            except Exception as e:
                log.exception("engine error mid-stream")
                self._sse({"error": {"message": f"engine error: {e}", "type": "server_error",
                                     "param": None, "code": None}})
            self._sse(b"[DONE]")
            self._end_sse()
            if completed:
                st.cache_response(prompt_ids, result)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def make_engine(args: argparse.Namespace, tok: Tok, enc) -> Engine:
    if args.engine == "mock":
        return MockEngine(tok.encode, tok.decode,
                          eos_token_id=tok.token_to_id(enc.eos_token),
                          think_start_id=tok.token_to_id(enc.thinking_start_token),
                          think_end_id=tok.token_to_id(enc.thinking_end_token),
                          max_context=args.max_context,
                          burst_delay_s=args.mock_delay)
    if args.engine == "v41":
        try:
            from engine.v41_engine import V41Engine  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "--engine v41 needs engine/v41_engine.py (class V41Engine, an engine_api.Engine subclass) "
                f"at the repo root ({REPO_ROOT}); it is not importable: {e}. "
                "Use --engine mock to run the HTTP layer alone.")
        kwargs: Dict[str, Any] = {
            "max_seq": args.max_seq,
            "arena_gb": args.arena_gb,
            "trace_stats": args.trace_stats,
            "spec": not args.no_spec,
        }
        # --engine-kwargs is the escape hatch for anything without its own flag,
        # and it wins over the flags so a JSON override is never silently ignored.
        kwargs.update(json.loads(args.engine_kwargs) if args.engine_kwargs else {})
        return V41Engine(model_dir=args.model_dir, **kwargs)
    raise SystemExit(f"unknown engine {args.engine!r}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible server for DeepSeek-V4.1-Flash")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-dir", required=True, help="directory with tokenizer.json and encoding/encoding.py")
    p.add_argument("--served-model-name", default="deepseek-v4.1-flash")
    p.add_argument("--default-thinking", choices=["off", "on"], default="off")
    p.add_argument("--default-effort", type=int, default=75, help="reasoning effort 1-100 when thinking is on")
    p.add_argument("--engine", choices=["mock", "v41"], default="mock")
    p.add_argument("--engine-kwargs", default="",
                   help="JSON object passed to V41Engine(model_dir=..., **kwargs); overrides the flags below")
    p.add_argument("--max-seq", type=int, default=32768,
                   help="v41 engine: KV/context length to allocate caches for (engine.max_context)")
    p.add_argument("--arena-gb", type=float, default=None,
                   help="v41 engine: GB of resident FP4 expert arena; omit to size it from free GPU memory")
    p.add_argument("--trace-stats", default=None,
                   help="v41 engine: coverage.json from tools/expert_stats.py, ranks the warm-start hot set")
    p.add_argument("--no-spec", action="store_true", help="v41 engine: disable MTP speculative decoding")
    p.add_argument("--headless", action="store_true",
                   help="EP2 worker rank: load the engine, serve the head's broadcast queue, bind no port")
    p.add_argument("--max-context", type=int, default=32768, help="context limit for the mock engine")
    p.add_argument("--mock-delay", type=float, default=0.0, help="seconds to sleep per mock burst")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    if not 1 <= args.default_effort <= 100:
        p.error("--default-effort must be within 1-100")
    args.default_thinking = args.default_thinking == "on"
    return args


def run_worker(engine) -> None:
    """EP2 rank > 0: no HTTP, no clients. The engine's constructor already joined the process
    group and warm-started this rank's half of the arena; this loop just serves the head's
    broadcast queue forever. Discarding the output is the point -- the value of each forward
    here is participation in the per-layer combines and identical logits, not text.

    A failed request means the pair is desynced (the head will sit waiting on a collective
    this rank no longer issues), so there is nothing to recover: exit non-zero loudly and let
    the launcher's restart story handle it, like everything else in this recipe."""
    import torch.distributed as X  # noqa: F401  (presence was checked by EPDistributed.init)
    ep = engine.ep
    runtime = None
    from engine.serving import concurrency
    if concurrency() > 1:
        from engine.serving import DecodeRuntime
        runtime = DecodeRuntime(engine)
    log.info("EP2 rank %d: headless worker, waiting for requests from rank 0", ep.rank)
    n = 0
    try:
        while True:
            try:
                req = ep.broadcast_request(None)
            except Exception as e:
                # With the heartbeat running, reaching the process-group timeout means several
                # pings in a row went missing, so the head really is gone -- but log WHAT failed.
                # This used to say "head stopped?" for every exception, which is how a plain idle
                # timeout (600 s with no traffic) got reported as the head stopping, while rank 0
                # was healthy and the next real request died on a closed gloo pair.
                log.error("EP2 worker: broadcast failed (%s: %s); communicator is unusable, exiting",
                          type(e).__name__, e)
                # A failed Gloo queue cannot be served again. Breaking here entered
                # ep.destroy(), whose NCCL teardown can wait forever on an absent peer:
                # the headless container stayed Up after its worker loop had stopped.
                # Match the other fatal worker paths and let process exit reclaim resources.
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                os._exit(1)
            if req is None or req.get("cmd") == "shutdown":
                # The head logs its broadcast; without this line a worker stuck in the teardown
                # below (the head is already gone) looked like one that never heard it.
                log.info("EP2 worker: shutdown from rank 0")
                break
            if req.get('cmd') == 'schedule':
                from engine.decode_events import event_signature
                try:
                    if runtime is None:
                        raise RuntimeError('scheduler command received with concurrency disabled')
                    event = runtime.execute(req['action'])
                    tag = event_signature(event)
                    tags = ep.gather_objects(tag)
                    if any(t != tag for t in tags):
                        raise RuntimeError(f'concurrent decode ranks diverged: {tags}')
                except Exception:
                    log.exception('concurrent worker failed; pair is desynced, exiting')
                    os._exit(1)
                continue
            if req.get('cmd') == 'cache_response':
                try:
                    if runtime is not None:
                        raise RuntimeError('response cache command requires concurrency=1')
                    engine.cache_response(req['prompt_ids'], req['response_ids'])
                except Exception:
                    log.exception('post-response prefix worker failed; exiting')
                    os._exit(1)
                continue
            if req.get('cmd') == 'decode_probe':
                try:
                    if runtime is not None:
                        raise RuntimeError('decode probe requires concurrency=1')
                    # The engine catches local diagnostic errors and both ranks reach its
                    # final gather once, even when a replay/capture was rejected locally.
                    engine.decode_probe_command(req['body'])
                except Exception:
                    log.exception('decode probe worker failed; pair is desynced, exiting')
                    os._exit(1)
                continue
            if req.get("cmd") == "maintain":
                # Same plan rank 0 is applying, at the same point in the broadcast stream, so the
                # two arenas and the two routers stay identical. A failure here desyncs the pair
                # exactly like a failed request does, and is treated the same way.
                try:
                    engine.apply_swaps(req["swaps"])
                except Exception:
                    log.exception("EP2 worker: expert adaptation failed; the pair is desynced, exiting")
                    os._exit(1)
                continue
            if req.get("cmd") == "ping":
                # Idle keepalive from rank 0 (see _ep_heartbeat). The blocking collective this
                # loop sits in is bounded by the process-group timeout, so without traffic it
                # would expire and take this worker down; the ping keeps it inside the window.
                continue
            n += 1
            t0 = time.time()
            try:
                _vl = req.get("vl")
                if _vl is not None and hasattr(engine, "set_vl_inputs"):
                    # same masks, same chunk boundaries, same collective count as the head
                    engine.set_vl_inputs(_vl["token_types"].to(engine.device), _vl["images"])
                for _ in engine.generate(req["prompt_ids"], **req["kwargs"]):
                    pass
            except Exception:
                log.exception("EP2 worker: request %d failed; the pair is desynced, exiting", n)
                os._exit(1)
            log.info("EP2 worker: request %d done (%d prompt tokens, %.1f s)",
                     n, len(req["prompt_ids"]), time.time() - t0)
    finally:
        try:
            engine.close()
        finally:
            try:
                ep.destroy()
            except Exception:
                pass


def _ep_heartbeat(state, interval: float, stop: threading.Event) -> None:
    """rank 0 -> rank 1 keepalive, so an idle pair does not dismantle itself.

    run_worker() blocks in broadcast_request between requests, and that collective is bounded by
    the process group's timeout (DSV41_DIST_TIMEOUT_S, 600 s). A server with no traffic for ten
    minutes therefore expired the collective, the worker exited, and the NEXT request failed on a
    closed gloo pair -- with rank 0 still perfectly healthy and reporting itself fine. Raising the
    timeout only moves the cliff; an idle serving process waits for an unbounded time by nature.

    `state.lock` is the request lock, so a ping can never land between a request's broadcast and
    its generate -- which would leave rank 1 consuming the ping as though it were the request and
    desync the pair on the very next collective.
    """
    while not stop.wait(interval):
        if state.ep_fault:
            return
        with state.lock:
            # A request may have faulted the pair while this tick waited for its lock.
            if stop.is_set() or state.ep_fault:
                return
            try:
                # The pair is idle here by construction: this holds the request lock, so no
                # generate can be in flight, and the next one waits. That makes the keepalive
                # tick the natural place to re-fit the resident experts to observed demand --
                # it is the one moment both ranks are known quiet and already synchronised.
                # End-of-response is the usual trigger; this catches a server that went idle
                # with demand still unapplied (the last response's swaps were capped, say).
                if not state.maintain_experts():
                    state.ep.broadcast_request({"cmd": "ping"})
            except Exception as e:  # noqa: BLE001
                # Latch before releasing the request lock: otherwise a queued request can
                # enter the broken collective, and /health still claims the pair is usable.
                state.ep_fault = f"heartbeat failed: {type(e).__name__}: {e}"[:200]
                log.warning("EP2 heartbeat failed (%s: %s); pair marked out of step", type(e).__name__, e)
                return


def _terminate(signum, frame):
    raise SystemExit(0)


def main(argv: Optional[List[str]] = None) -> None:
    # Docker runs this process as PID 1, where an unhandled SIGTERM is ignored.
    # Raise through the existing finally blocks instead of waiting for SIGKILL.
    # Do not call HTTPServer.shutdown() here: serve_forever runs on this thread.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _terminate)
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tok = Tok(args.model_dir)
    enc = load_encoding_module(args.model_dir)
    engine = make_engine(args, tok, enc)
    global VISION_OK
    VISION_OK = getattr(engine, "vision", None) is not None
    log.info("image inputs: %s", "enabled (vision tower loaded)" if VISION_OK else "rejected (no vision tower)")
    ep = getattr(engine, "ep", None)
    headless = args.headless or (ep is not None and getattr(ep, "active", False) and ep.rank != 0)
    if headless:
        if ep is None or not getattr(ep, "active", False):
            raise SystemExit("--headless is the EP2 worker mode; it needs WORLD_SIZE > 1")
        if ep.rank == 0:
            raise SystemExit("--headless on RANK=0 would serve nobody; the head serves HTTP")
        run_worker(engine)
        return
    state = State(args, tok, enc, engine)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.state = state  # type: ignore[attr-defined]
    log.info("serving %s (engine=%s, tokenizer=%s, thinking default=%s effort=%d) on http://%s:%d",
             args.served_model_name, args.engine, tok.backend, "on" if args.default_thinking else "off",
             args.default_effort, args.host, args.port)
    hb_stop = threading.Event()
    if state.ep_active:
        # A quarter of the process-group timeout: four pings have to go missing before the worker
        # concludes the head is gone, which keeps a transient stall from tearing the pair down.
        _t = float(os.environ.get("DSV41_DIST_TIMEOUT_S", "600"))
        _iv = float(os.environ.get("DSV41_EP_PING_S", max(30.0, _t / 4)))
        threading.Thread(target=_ep_heartbeat, args=(state, _iv, hb_stop),
                         name="ep-heartbeat", daemon=True).start()
        log.info("EP2: idle keepalive every %.0f s (process-group timeout %.0f s)", _iv, _t)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hb_stop.set()          # before the shutdown broadcast, or the two race on the same group
        httpd.server_close()
        if state.scheduler is not None:
            state.scheduler.stop()
            state.scheduler.thread.join()
        # EP2: tell rank 1 to go home. run_worker() blocks in broadcast_request between
        # requests, so without this the peer sits there holding its half of the pool until the
        # process-group timeout expires -- and dual-down.sh's memory wait would watch it do it.
        # The engine lock is what makes this safe: a request still in flight owns the
        # collectives, and the shutdown message has to queue behind it, not interleave with it.
        if state.ep_active and not state.ep_fault:
            try:
                with state.lock:
                    state.ep.broadcast_request({"cmd": "shutdown"})
                log.info("EP2: shutdown broadcast to rank %d", 1)
            except Exception as e:
                log.warning("EP2: could not tell the peer to shut down (%s); dual-down.sh will", e)
        engine.close()


if __name__ == "__main__":
    main()
