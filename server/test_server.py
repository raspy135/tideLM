#!/usr/bin/env python3
"""End-to-end tests for server/app.py against the mock engine.

Run directly (``python3 server/test_server.py``) or via pytest. The model
directory (tokenizer.json + encoding/encoding.py) is taken from
``$V41_MODEL_DIR`` or a few well-known locations.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_MODEL_DIRS = [
    os.environ.get("V41_MODEL_DIR", ""),
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash"),
    os.path.join(os.path.dirname(HERE), "models", "DeepSeek-V4.1-Flash"),
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                       "required": ["location"]},
    },
}]
QUICKSTART_PROMPT = (
    "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the "
    "value, the more thorough the reasoning)\n\nYou are a helpful assistant."
    "<｜User｜>What is 2+2?<｜Assistant｜><think>"
)

SERVER = {"proc": None, "base": None}


def model_dir() -> str:
    for d in CANDIDATE_MODEL_DIRS:
        if d and os.path.exists(os.path.join(d, "tokenizer.json")) and \
                os.path.exists(os.path.join(d, "encoding", "encoding.py")):
            return d
    raise SystemExit("no model dir with tokenizer.json + encoding/encoding.py; set V41_MODEL_DIR")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server() -> None:
    port = free_port()
    SERVER["proc"] = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "app.py"), "--engine", "mock", "--port", str(port),
         "--model-dir", model_dir(), "--default-thinking", "off", "--default-effort", "75",
         "--log-level", "WARNING"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    SERVER["base"] = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            get("/health")
            return
        except Exception:
            if SERVER["proc"].poll() is not None:
                raise SystemExit("server died: " + SERVER["proc"].stderr.read().decode())
            time.sleep(0.1)
    raise SystemExit("server did not come up")


def stop_server() -> None:
    p = SERVER["proc"]
    if p and p.poll() is None:
        p.terminate()
        p.wait(5)


# -- tiny HTTP client ---------------------------------------------------------

def get(path: str) -> dict:
    with urllib.request.urlopen(SERVER["base"] + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path: str, body, raw: bytes = None):
    """POST JSON; return (status, parsed body)."""
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(SERVER["base"] + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def post_stream(path: str, body) -> list:
    """POST with stream=true; return the list of parsed `data:` payloads ('[DONE]' kept as str)."""
    req = urllib.request.Request(SERVER["base"] + path, data=json.dumps(body).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream"), r.headers["Content-Type"]
        buf = b""
        while True:
            chunk = r.read(256)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                for line in frame.decode("utf-8").splitlines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def chat(messages=None, **kw):
    body = {"model": "deepseek-v4.1-flash", "messages": messages or [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return post("/v1/chat/completions", body)


def reassemble(events: list) -> dict:
    """Merge SSE chunks into {reasoning, content, tool_calls, finish, usage, roles, done}."""
    out = {"reasoning": "", "content": "", "tool_calls": [], "finish": None, "usage": None,
           "roles": [], "done": events and events[-1] == "[DONE]", "ids": set(), "stats": None}
    for ev in events:
        if ev == "[DONE]":
            continue
        assert "error" not in ev, ev
        out["ids"].add(ev["id"])
        if ev.get("usage"):
            out["usage"] = ev["usage"]
            out["stats"] = ev.get("x_engine_stats")
        for ch in ev["choices"]:
            d = ch.get("delta", {})
            if "role" in d:
                out["roles"].append(d["role"])
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            out["tool_calls"] += d.get("tool_calls") or []
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    return out


# -- tests --------------------------------------------------------------------

def test_health_and_models():
    h = get("/health")
    assert h["status"] == "ok" and h["engine"] == "mock"
    m = get("/v1/models")
    assert m["object"] == "list" and m["data"][0]["id"] == "deepseek-v4.1-flash"
    assert get("/v1/models/deepseek-v4.1-flash")["object"] == "model"
    status, err = post("/v1/nope", {})
    assert status == 404 and "error" in err


def test_prompt_rendering_matches_encoding_readme():
    status, r = post("/v1/debug/prompt", {
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "What is 2+2?"}],
        "reasoning_effort": "high"})
    assert status == 200, r
    assert r["thinking"] is True and r["reasoning_effort"] == 75
    assert r["prompt"] == QUICKSTART_PROMPT, r["prompt"]
    assert r["prompt_ids"][0] == 0 and r["prompt_ids"][-2:] == [128804, 128821], r["prompt_ids"][-3:]
    # chat mode: assistant header closes the think block
    status, r = post("/v1/debug/prompt", {"messages": [{"role": "user", "content": "hello"}]})
    assert r["thinking"] is False
    assert r["prompt"] == "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    assert r["prompt_ids"][-2:] == [128804, 128822]


def test_developer_role_is_folded_into_system():
    """Newer OpenAI clients send role="developer"; V4.1 has no such role and the encoder
    raises on it. It has to render identically to system, and a tools list riding on it has
    to attach to that message rather than spawn a second empty system one."""
    def render(role, tools=None):
        body = {"messages": [{"role": role, "content": "You are a helpful assistant."},
                             {"role": "user", "content": "What is 2+2?"}],
                "reasoning_effort": "high"}
        if tools:
            body["tools"] = tools
        status, r = post("/v1/debug/prompt", body)
        assert status == 200, r
        return r

    system, developer = render("system"), render("developer")
    assert developer["prompt"] == system["prompt"] and developer["prompt_ids"] == system["prompt_ids"]
    system_tools, developer_tools = render("system", TOOLS), render("developer", TOOLS)
    assert developer_tools["prompt"] == system_tools["prompt"]
    assert developer_tools["prompt_ids"] == system_tools["prompt_ids"]


def test_typed_image_placeholder_is_escaped_not_rejected():
    """A coding agent puts the literal image token into a tool result by reading this repo or the
    checkpoint's own `encoding.py`. The reference encoder refuses the whole request for it, which
    would wedge every later turn of that conversation. The server splits the token with a
    zero-width space instead, so it reads as ordinary text and never becomes image id 129264 with
    no pixels behind it. This exact failure was reported against the DeepSeek Harness."""
    token = "<｜deepseek_image｜>"
    tools = [{"type": "function", "function": {
        "name": "grep", "description": "find " + token,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    status, r = post("/v1/debug/prompt", {
        "tools": tools,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "x", "schema": {"type": "object", "description": token}}},
        "messages": [
            {"role": "user", "content": "read this: " + token},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "type": "function", "function": {
                    "name": "grep", "arguments": json.dumps({"q": token})}}]},
            {"role": "tool", "tool_call_id": "1", "content": "stdout: " + token},
            {"role": "assistant", "reasoning_content": "I saw " + token, "content": "ok"},
            {"role": "user", "content": "again"}]})
    assert status == 200, r
    assert token not in r["prompt"], r["prompt"]
    assert "\u200b" in r["prompt"] and "deepseek_image" in r["prompt"]
    assert 129264 not in r["prompt_ids"], r["prompt_ids"]
    # /v1/completions tokenizes raw text directly, so it has to escape too. The bare escaped
    # token is 8 BPE pieces, not the single image id; the mock reports prompt_tokens as the ids.
    status, r = post("/v1/completions", {"model": "x", "prompt": token, "max_tokens": 2})
    assert status == 200, r
    assert r["usage"]["prompt_tokens"] >= 8, r["usage"]


def test_chat_nonstream_thinking_off_default():
    status, r = chat()
    assert status == 200, r
    msg = r["choices"][0]["message"]
    assert msg["role"] == "assistant" and msg["content"].startswith("Hello from the mock engine")
    assert "reasoning_content" not in msg, msg
    assert r["choices"][0]["finish_reason"] == "stop"
    u = r["usage"]
    assert u["completion_tokens_details"]["reasoning_tokens"] == 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["completion_tokens"] > 0
    assert r["x_engine_stats"]["engine"] == "mock" and "tok_per_s" in r["x_engine_stats"]


def test_chat_nonstream_thinking_on():
    status, r = chat(chat_template_kwargs={"thinking": True})
    assert status == 200, r
    msg = r["choices"][0]["message"]
    assert msg["reasoning_content"].startswith("The user is talking to a mock engine")
    assert "</think>" not in msg["reasoning_content"] and "</think>" not in msg["content"]
    assert msg["content"].startswith("Hello from the mock engine")
    rt = r["usage"]["completion_tokens_details"]["reasoning_tokens"]
    assert 0 < rt < r["usage"]["completion_tokens"], r["usage"]


def test_thinking_precedence_and_effort_mapping():
    def probe(**kw):
        status, r = post("/v1/debug/prompt", {"messages": [{"role": "user", "content": "q"}], **kw})
        assert status == 200, r
        return r["thinking"], r["reasoning_effort"]

    assert probe() == (False, 75)                                       # server default
    assert probe(reasoning_effort="none") == (False, 75)
    assert probe(reasoning_effort="low") == (False, 50)
    assert probe(reasoning_effort="medium") == (True, 60)
    assert probe(reasoning_effort="high") == (True, 75)
    assert probe(reasoning_effort="xhigh") == (True, 90)
    assert probe(reasoning_effort="max") == (True, 100)
    assert probe(reasoning_effort=42) == (True, 42)
    assert probe(reasoning={"effort": "max"}) == (True, 100)
    assert probe(enable_thinking=True) == (True, 75)
    assert probe(enable_thinking=True, reasoning_effort="low") == (True, 50)   # bool wins, effort kept
    assert probe(chat_template_kwargs={"enable_thinking": True}) == (True, 75)
    assert probe(chat_template_kwargs={"thinking": False}, reasoning_effort="max") == (False, 100)
    assert probe(chat_template_kwargs={"thinking": True}, enable_thinking=False) == (True, 75)
    assert probe(enable_thinking=False, chat_template_kwargs={"enable_thinking": True}) == (False, 75)
    status, err = post("/v1/debug/prompt", {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": 0})
    assert status == 400 and err["error"]["type"] == "invalid_request_error"
    status, _ = post("/v1/debug/prompt", {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "ultra"})
    assert status == 400


def test_chat_stream_thinking_on():
    events = post_stream("/v1/chat/completions", {
        "model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        "reasoning_effort": "high", "stream_options": {"include_usage": True}})
    r = reassemble(events)
    assert r["done"], events[-3:]
    assert r["roles"] == ["assistant"] and len(r["ids"]) == 1
    assert r["reasoning"].startswith("The user is talking to a mock engine") and "</think>" not in r["reasoning"]
    assert r["content"].startswith("Hello from the mock engine") and r["content"].endswith("without a GPU.")
    assert r["finish"] == "stop" and r["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0
    assert r["stats"]["engine"] == "mock"
    # reasoning deltas must all come before content deltas
    kinds = [("r" if "reasoning_content" in ev["choices"][0]["delta"] else "c")
             for ev in events if ev != "[DONE]" and ev["choices"] and
             (ev["choices"][0]["delta"].get("reasoning_content") or ev["choices"][0]["delta"].get("content"))]
    assert kinds == sorted(kinds, key=lambda k: k != "r"), kinds
    # include_usage adds a trailing chunk with empty choices carrying usage
    assert events[-2]["choices"] == [] and events[-2]["usage"]["completion_tokens"] > 0
    # non-stream and stream must agree on the text
    _, ns = chat(reasoning_effort="high")
    assert ns["choices"][0]["message"]["content"] == r["content"]
    assert ns["choices"][0]["message"]["reasoning_content"] == r["reasoning"]


def test_chat_stream_thinking_off():
    r = reassemble(post_stream("/v1/chat/completions", {
        "model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True}))
    assert r["done"] and r["reasoning"] == "" and r["content"].startswith("Hello from the mock engine")
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


ANTHROPIC_TOOLS = [{"name": "get_weather", "description": "Get the weather for a city",
                    "input_schema": TOOLS[0]["function"]["parameters"]}]


def anthropic_events(body) -> list:
    """POST /v1/messages with stream=true; return [(event name, data)] in order."""
    req = urllib.request.Request(SERVER["base"] + "/v1/messages?beta=true",
                                 data=json.dumps(dict(body, stream=True)).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream"), r.headers["Content-Type"]
        frames = r.read().decode("utf-8").strip().split("\n\n")
    out = []
    for frame in frames:
        name, data = frame.split("\n", 1)
        data = json.loads(data.removeprefix("data: "))
        assert name == "event: " + data["type"], frame
        out.append((data["type"], data))
    return out


def test_anthropic_messages():
    """The Anthropic Messages API (server/anthropic_api.py) for Claude Code: same prompt as the
    equivalent chat request, tool calls as tool_use blocks, Anthropic SSE grammar and errors."""
    body = {"model": "x", "max_tokens": 64000, "tools": ANTHROPIC_TOOLS,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=0;"},
                       {"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Berlin?"}]}],
            "thinking": {"type": "adaptive"}}
    status, n = post("/v1/messages/count_tokens", body)
    assert status == 200, n
    status, dbg = post("/v1/debug/prompt", {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Berlin?"}]})
    assert n["input_tokens"] == dbg["prompt_tokens"], (n, dbg["prompt_tokens"])

    status, r = post("/v1/messages?beta=true", body)
    assert status == 200, r
    assert r["type"] == "message" and r["stop_reason"] == "tool_use", r
    assert r["content"][0] == {"type": "text", "text": "Let me look that up."}, r["content"]
    use = r["content"][1]
    assert use["type"] == "tool_use" and use["id"].startswith("toolu_") and use["name"] == "get_weather"
    assert use["input"] == {"location": "Berlin"}
    assert set(r["usage"]) >= {"input_tokens", "output_tokens", "cache_read_input_tokens"}

    ev = anthropic_events(body)
    kinds = [k for k, _ in ev if k != "ping"]
    kinds = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
    assert kinds == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "content_block_start", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"], kinds
    text = "".join(d["delta"]["text"] for k, d in ev if k == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Let me look that up.", text
    args = "".join(d["delta"]["partial_json"] for k, d in ev
                   if k == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert json.loads(args) == {"location": "Berlin"}
    assert [d for k, d in ev if k == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"

    # The tool round trip encodes, and a turn's text after its results is kept.
    status, r2 = post("/v1/messages", dict(body, messages=body["messages"] + [
        {"role": "assistant", "content": r["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "{\"temp\": 21}"},
                                     {"type": "text", "text": "Answer in Celsius."}]}]))
    assert status == 200, r2

    status, err = post("/v1/messages/nope", body)
    assert status == 404 and err["type"] == "error" and err["error"]["type"] == "not_found_error", err


def test_tools_nonstream_and_stream():
    status, r = chat([{"role": "user", "content": "Weather in Berlin?"}], tools=TOOLS, reasoning_effort="high")
    assert status == 200, r
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls", ch
    tcs = ch["message"]["tool_calls"]
    assert len(tcs) == 1 and tcs[0]["type"] == "function" and tcs[0]["id"].startswith("call_")
    assert tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"location": "Berlin"}
    assert ch["message"]["content"] == "Let me look that up."   # the "\n\n<DSML" lead-in is stripped
    assert ch["message"]["reasoning_content"].startswith("The user is talking")

    s = reassemble(post_stream("/v1/chat/completions", {
        "model": "x", "messages": [{"role": "user", "content": "Weather in Berlin?"}],
        "tools": TOOLS, "stream": True}))
    assert s["done"] and s["finish"] == "tool_calls"
    assert s["content"] == "Let me look that up." and "DSML" not in s["content"]
    assert len(s["tool_calls"]) == 1 and s["tool_calls"][0]["index"] == 0
    assert s["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(s["tool_calls"][0]["function"]["arguments"]) == {"location": "Berlin"}

    # tool_choice=none drops the schemas, so the mock answers with plain text
    status, r = chat(tools=TOOLS, tool_choice="none")
    assert r["choices"][0]["finish_reason"] == "stop" and "tool_calls" not in r["choices"][0]["message"]

    # a tool round trip must encode (tool result merged into the user turn)
    status, r = post("/v1/debug/prompt", {"tools": TOOLS, "messages": [
        {"role": "user", "content": "Weather in Berlin?"},
        {"role": "assistant", "content": None, "tool_calls": tcs},
        {"role": "tool", "tool_call_id": tcs[0]["id"], "content": "{\"temp\": 21}"}]})
    assert status == 200, r
    assert '<tool_result>{"temp": 21}</tool_result><｜Assistant｜></think>' in r["prompt"]
    assert '<｜DSML｜ invoke name="get_weather">' in r["prompt"]


def test_stop_strings():
    status, r = chat(stop=["bursts"], reasoning_effort="high")
    assert status == 200, r
    msg = r["choices"][0]["message"]
    assert msg["content"] == "Hello from the mock engine! This is a canned reply.\n\nIt streams in small ", repr(msg["content"])
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["completion_tokens"] < 75
    s = reassemble(post_stream("/v1/chat/completions", {
        "model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True, "stop": "bursts"}))
    assert s["content"] == msg["content"] and s["finish"] == "stop" and s["done"]
    # a stop string inside the reasoning block cuts the reasoning and leaves no content
    status, r = chat(stop=["canned"], reasoning_effort="high")
    assert r["choices"][0]["message"]["reasoning_content"].endswith("mention that this is ")
    assert r["choices"][0]["message"]["content"] == ""


def test_max_tokens_length():
    status, r = chat(max_tokens=3, reasoning_effort="high")
    assert status == 200, r
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 3
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3  # never reached </think>
    assert r["choices"][0]["message"]["content"] == ""
    s = reassemble(post_stream("/v1/chat/completions", {
        "model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 5}))
    assert s["finish"] == "length" and s["usage"]["completion_tokens"] == 5 and s["done"]


def test_completions():
    status, r = post("/v1/completions", {"model": "x", "prompt": "<｜User｜>hi<｜Assistant｜></think>",
                                         "max_tokens": 64, "seed": 7})
    assert status == 200, r
    assert r["object"] == "text_completion"
    assert r["choices"][0]["text"].startswith("Hello from the mock engine")
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["prompt_tokens"] > 0 and "x_engine_stats" in r
    events = post_stream("/v1/completions", {"model": "x", "prompt": "hi", "stream": True})
    assert events[-1] == "[DONE]"
    text = "".join(ev["choices"][0]["text"] for ev in events if ev != "[DONE]")
    assert text.startswith("Hello from the mock engine") and events[-2]["choices"][0]["finish_reason"] == "stop"
    # raw token ids are accepted as-is
    status, r = post("/v1/completions", {"model": "x", "prompt": [0, 128803, 6366, 128804, 128822], "max_tokens": 4})
    assert status == 200 and r["usage"]["prompt_tokens"] == 5 and r["choices"][0]["finish_reason"] == "length"
    status, err = post("/v1/completions", {"model": "x", "prompt": ["a", "b"]})
    assert status == 400 and err["error"]["param"] == "prompt"


def test_bad_requests():
    status, err = post("/v1/chat/completions", None, raw=b"{not json")
    assert status == 400 and err["error"]["type"] == "invalid_request_error" and err["error"]["message"]
    status, err = post("/v1/chat/completions", {"model": "x", "messages": []})
    assert status == 400 and err["error"]["param"] == "messages"
    status, err = post("/v1/chat/completions", {"model": "x"})
    assert status == 400 and err["error"]["param"] == "messages"
    status, err = chat([{"role": "user", "content": [{"type": "text", "text": "look"},
                                                     {"type": "image_url", "image_url": {"url": "x.png"}}]}])
    assert status == 400 and "image" in err["error"]["message"]
    status, err = chat(temperature=9)
    assert status == 400 and err["error"]["param"] == "temperature"
    status, err = chat(max_tokens=0)
    assert status == 400
    status, err = chat(n=2)
    assert status == 400
    status, err = chat([{"role": "user", "content": "hi"}, {"role": "wizard", "content": "x"}])
    assert status == 400 and "cannot encode" in err["error"]["message"]
    status, err = chat(ignore_eos="yes")
    assert status == 400 and err["error"]["param"] == "ignore_eos"


def test_ignore_eos_runs_to_max_tokens():
    """The benchmark needs fixed output lengths, so ignore_eos must beat the EOS the engine emits.

    The mock reply ends in EOS. Without ignore_eos the server cuts the burst at that id and
    stops; with it the stop set is empty on both sides, so the EOS token itself is generated,
    counted and decoded (the mock has no more text after it, hence exactly one extra token).
    A real engine keeps going all the way to max_tokens, which is what the benchmark needs.
    """
    status, r = chat(max_tokens=300)
    assert status == 200 and r["choices"][0]["finish_reason"] == "stop"
    short = r["usage"]["completion_tokens"]
    status, r = chat(max_tokens=300, ignore_eos=True)
    assert status == 200, r
    assert r["usage"]["completion_tokens"] == short + 1, (short, r["usage"])


def test_unicode_streaming_is_not_split():
    # The mock echoes nothing, so use /v1/completions on a prompt and verify the
    # canned text, then check the emoji-containing content round trip via the
    # detokenizer used by the server (byte-fallback tokens must be joined).
    sys.path.insert(0, HERE)
    import app  # noqa: WPS433
    tok = app.Tok(model_dir())
    text = "Grüße 😀 你好 — done"
    ids = tok.encode(text)
    detok = app.IncrementalDetokenizer(tok)
    pieces = [detok.push([i]) for i in ids]
    pieces.append(detok.flush())
    assert "".join(pieces) == text, pieces
    assert all("�" not in p for p in pieces), pieces


def test_requests_are_serialised_not_rejected():
    results, errors = [], []

    def worker():
        try:
            results.append(chat(reasoning_effort="high")[0])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors and results == [200] * 4, (results, errors)


# -- runner -------------------------------------------------------------------

def setup_module(module=None):  # pytest hook
    start_server()


def teardown_module(module=None):  # pytest hook
    stop_server()


def test_tolerant_tool_call_recovery():
    """A completion whose DSML the checkpoint's strict parser rejects must still yield its calls.

    Two deviations were seen in real traffic and cost the whole call (the client saw the sentence
    before it and a `stop` finish): the parameter value placed in the `string` attribute, and prose
    after the closing tag of the calls block.
    """
    import inspect
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import app as A
    owner = next(o for _, o in vars(A).items()
                 if inspect.isclass(o) and hasattr(o, "_parse_tool_calls_tolerant"))
    parse = owner._parse_tool_calls_tolerant
    D = "\uff5cDSML\uff5c"
    spec = (f'\n\n<{D} calls>\n<{D} invoke name="get_weather">\n'
            f'<{D} parameter name="city" string="true">Berlin<\n/{D} parameter>\n'
            f'<{D} parameter name="days" string="false">3<\n/{D} parameter>\n</{D} invoke>\n</{D} calls>')
    attr = (f'\n\n<{D} calls>\n<{D} invoke name="web_search">\n'
            f'<{D} parameter name="query" string="unified memory specs">\n</{D} invoke>\n</{D} calls>')
    two = (f'\n\n<{D} calls>\n<{D} invoke name="a">\n<{D} parameter name="x" string="true">1<\n/{D} parameter>\n'
           f'</{D} invoke>\n<{D} invoke name="b">\n<{D} parameter name="y" string="true">2<\n/{D} parameter>\n'
           f'</{D} invoke>\n</{D} calls>')

    r = parse(spec)
    assert len(r) == 1 and r[0]["function"]["name"] == "get_weather", r
    args = json.loads(r[0]["function"]["arguments"])
    assert args == {"city": "Berlin", "days": 3}, args

    r = parse(attr)
    assert len(r) == 1 and r[0]["function"]["name"] == "web_search", r
    assert json.loads(r[0]["function"]["arguments"]) == {"query": "unified memory specs"}

    r = parse(attr + "\nI will summarise the result for you.")
    assert len(r) == 1 and json.loads(r[0]["function"]["arguments"]) == {"query": "unified memory specs"}

    r = parse(two)
    assert [c["function"]["name"] for c in r] == ["a", "b"], r

    assert parse("there are no tool calls in this text") == []


def main() -> int:
    start_server()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, 0
    try:
        for name, fn in tests:
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    finally:
        stop_server()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
