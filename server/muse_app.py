"""OpenAI-compatible server for Muse Spark on the TP2 engine (engine/muse/tp.py).

Rank 0 serves HTTP and samples; rank 1 runs the same engine calls from rank 0's broadcast
commands. One request at a time (a queue in front of the engine); the KV cache of the previous
request is reused for a matching prefix, so an agent turn that appends to the conversation only
prefills the new tokens.

Chat format (the checkpoint's chat_template.jinja): ``<|start|>role<|message|>...<|eot|>``. The
assistant turn is a sequence of messages addressed with ``to=``: ``self`` (reasoning),
``commentary`` / ``user`` (visible text), or a tool name (an ``<atem:function_calls>`` block).
They map to ``reasoning_content``, ``content`` and ``tool_calls``.

    python server/muse_app.py --model /models --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

START, MESSAGE, EOM, EOT, EOS = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"


# ---------------------------------------------------------------------------------- template
class Template:
    def __init__(self, model_dir: str):
        import jinja2
        from tokenizers import Tokenizer
        d = Path(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        text = (d / "chat_template.jinja").read_text()
        # transformers' {% generation %} marks assistant spans for training masks; rendering
        # does not need it.
        text = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", text)
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = self._raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(text)
        self.special = {t: self.tok.token_to_id(t) for t in (START, MESSAGE, EOM, EOT, EOS)}
        self.bos = "<|begin_of_text|>"

    @staticmethod
    def _raise(msg):
        raise ValueError(msg)

    def render(self, messages, tools=None, reasoning_effort=None) -> str:
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    args = tc.get("function", {}).get("arguments")
                    if isinstance(args, str):
                        try:
                            tc["function"]["arguments"] = json.loads(args or "{}")
                        except json.JSONDecodeError:
                            tc["function"]["arguments"] = {"input": args}
                    calls.append(tc)
                m["tool_calls"] = calls
            msgs.append(m)
        kw = {"messages": msgs, "tools": tools or None, "add_generation_prompt": True,
              "bos_token": self.bos}
        if reasoning_effort:
            kw["reasoning_effort"] = reasoning_effort
        return self.template.render(**kw)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------------- output
class Parser:
    """Turns the assistant turn's token stream into (channel, text delta) events.

    The prompt ends with ``<|start|>assistant``, so the stream begins inside a header
    (`` to=self``) and alternates header -> ``<|message|>`` -> body -> ``<|eom|>``/``<|eot|>``.
    """

    def __init__(self, tpl: Template):
        self.tpl = tpl
        self.sp = tpl.special
        self.in_header = True
        self.header: list[int] = []
        self.body: list[int] = []
        self.sent = ""
        self.recipient = "user"
        self.messages: list[tuple[str, str]] = []    # (recipient, text) of finished messages

    def channel(self) -> str:
        r = self.recipient
        if r == "self":
            return "reasoning"
        if r in ("user", "commentary", "all"):
            return "content"
        return "tool"

    def feed(self, tok: int) -> list[tuple[str, str]]:
        sp = self.sp
        if tok == sp[START]:
            self.in_header, self.header = True, []
            return []
        if self.in_header:
            if tok == sp[MESSAGE]:
                head = self.tpl.decode(self.header)
                m = re.search(r"to=(\S+)", head)
                self.recipient = m.group(1) if m else "user"
                self.in_header, self.body, self.sent = False, [], ""
            else:
                self.header.append(tok)
            return []
        if tok in (sp[EOM], sp[EOT], sp[EOS]):
            text = self.tpl.decode(self.body)
            self.messages.append((self.recipient, text))
            out = []
            if self.channel() != "tool" and len(text) > len(self.sent):
                out.append((self.channel(), text[len(self.sent):]))
            self.in_header, self.header, self.body, self.sent = True, [], [], ""
            return out
        self.body.append(tok)
        if self.channel() == "tool":
            return []
        text = self.tpl.decode(self.body)
        if text.endswith("�"):          # a split multi-byte character: wait for the rest
            return []
        delta, self.sent = text[len(self.sent):], text
        return [(self.channel(), delta)] if delta else []

    def finish(self) -> list[tuple[str, str]]:
        """Flush a message cut off by max_tokens."""
        if self.in_header or not self.body:
            return []
        text = self.tpl.decode(self.body)
        self.messages.append((self.recipient, text))
        if self.channel() != "tool" and len(text) > len(self.sent):
            return [(self.channel(), text[len(self.sent):])]
        return []

    def tool_calls(self, tools=None) -> list[dict]:
        types = {}
        for t in tools or []:
            fn = t.get("function", t)
            props = (fn.get("parameters") or {}).get("properties") or {}
            types[fn.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
        calls = []
        for recipient, text in self.messages:
            if recipient in ("self", "user", "commentary", "all"):
                continue
            for inv in re.finditer(r'<atem:invoke name="([^"]*)">(.*?)</atem:invoke>', text, re.S):
                name = _unescape(inv.group(1))
                args = {}
                for p in re.finditer(r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>',
                                     inv.group(2), re.S):
                    key, raw = _unescape(p.group(1)), p.group(2)
                    want = types.get(name, {}).get(key)
                    if want == "string":
                        args[key] = raw
                    else:
                        try:
                            args[key] = json.loads(raw)
                        except json.JSONDecodeError:
                            args[key] = raw
                calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return calls


def _unescape(s: str) -> str:
    return (s.replace("&quot;", '"').replace("&#39;", "'").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&amp;", "&"))


# ---------------------------------------------------------------------------------- serving
class Job:
    def __init__(self, prompt: list[int], params: dict):
        self.prompt, self.params = prompt, params
        self.out: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self.stats: dict = {}


def engine_loop(engine, jobs: "queue.Queue[Job]", log, tpl: Template, stop: tuple[int, ...]):
    """Rank 0: run jobs one at a time; every engine call is mirrored on rank 1."""
    start_id = tpl.special.get(START)          # Muse's header guard; None for other formats
    while True:
        job = jobs.get()
        if job is None:
            engine.comm.broadcast_obj({"op": "stop"})
            return
        p = job.params
        cmd = {"op": "generate", "prompt": job.prompt, "max_new": p["max_tokens"],
               "temperature": p["temperature"], "top_p": p["top_p"], "top_k": p["top_k"],
               "seed": p["seed"], "stop": list(stop)}
        engine.comm.broadcast_obj(cmd)
        after_start = False
        t0 = time.perf_counter()
        n = 0
        first = None
        try:
            for tok in engine.stream(job.prompt, cmd["max_new"], temperature=cmd["temperature"],
                                     top_p=cmd["top_p"], top_k=cmd["top_k"], seed=cmd["seed"],
                                     stop=stop, should_stop=job.cancelled.is_set):
                if first is None:
                    first = time.perf_counter()
                n += 1
                job.out.put(tok)
                # A new message header that is not the assistant's (a made-up tool result or
                # user turn) ends the turn.
                if start_id is not None and after_start and not tpl.decode([tok]).strip().startswith("assistant"):
                    job.cancelled.set()
                after_start = tok == start_id
        except Exception as exc:          # the pair may now be inconsistent: stop serving
            job.out.put(exc)
            raise
        end = time.perf_counter()
        job.stats = {"prompt_tokens": len(job.prompt), "reused_prefix": engine.last_reuse,
                     "completion_tokens": n, "ttft_s": round((first or end) - t0, 3),
                     "decode_tok_s": round((n - 1) / (end - first), 2) if first and n > 1 else None}
        log(f"[muse] request: {json.dumps(job.stats)}")
        job.out.put(None)


def follower_loop(engine):
    """Rank 1: mirror rank 0's engine calls until told to stop."""
    while True:
        cmd = engine.comm.broadcast_obj(None)
        if cmd["op"] == "stop":
            return
        for _ in engine.stream(cmd["prompt"], cmd["max_new"], temperature=cmd["temperature"],
                               top_p=cmd["top_p"], top_k=cmd["top_k"], seed=cmd["seed"],
                               stop=tuple(cmd["stop"])):
            pass


def make_handler(tpl: Template, jobs, model_name: str, defaults: dict, parser_cls=None):
    Parser_ = parser_cls or Parser
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._json(200, {"status": "ok", "queued": jobs.qsize()})
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                path = self.path.rstrip("/")
                if path == "/v1/chat/completions":
                    text = tpl.render(req.get("messages", []), req.get("tools"),
                                      req.get("reasoning_effort") or defaults["reasoning_effort"])
                    chat = True
                elif path == "/v1/completions":
                    text, chat = req.get("prompt", ""), False
                else:
                    return self._json(404, {"error": "not found"})
                prompt = tpl.encode(text)
                params = {
                    "max_tokens": int(req.get("max_tokens") or req.get("max_completion_tokens")
                                      or defaults["max_tokens"]),
                    "temperature": float(req.get("temperature", defaults["temperature"])),
                    "top_p": float(req.get("top_p", defaults["top_p"])),
                    "top_k": int(req.get("top_k", defaults["top_k"])),
                    "seed": int(req.get("seed", int(time.time() * 1000) % (2 ** 31))),
                }
                if len(prompt) + params["max_tokens"] > defaults["max_seq"]:
                    params["max_tokens"] = max(1, defaults["max_seq"] - len(prompt) - 1)
                    if len(prompt) >= defaults["max_seq"]:
                        return self._json(400, {"error": f"prompt of {len(prompt)} tokens exceeds the context"})
            except Exception as exc:
                return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            job = Job(prompt, params)
            jobs.put(job)
            if req.get("stream"):
                self._stream(job, req, chat)
            else:
                self._complete(job, req, chat)

        def _events(self, job):
            while True:
                tok = job.out.get()
                if tok is None:
                    return
                if isinstance(tok, Exception):
                    raise tok
                yield tok

        def _complete(self, job, req, chat):
            parser = Parser_(tpl)
            ids = []
            for tok in self._events(job):
                ids.append(tok)
                parser.feed(tok)
            parser.finish()
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            usage = {"prompt_tokens": len(job.prompt), "completion_tokens": len(ids),
                     "total_tokens": len(job.prompt) + len(ids), "muse": job.stats}
            if not chat:
                return self._json(200, {"id": rid, "object": "text_completion", "model": model_name,
                                        "choices": [{"index": 0, "text": tpl.decode(ids),
                                                     "finish_reason": "stop"}], "usage": usage})
            content = "".join(t for r, t in parser.messages if r in ("user", "commentary", "all"))
            reasoning = "".join(t for r, t in parser.messages if r == "self")
            calls = parser.tool_calls(req.get("tools"))
            msg = {"role": "assistant", "content": content or None}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            finish = "tool_calls" if calls else ("length" if len(ids) >= job.params["max_tokens"] else "stop")
            self._json(200, {"id": rid, "object": "chat.completion", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "message": msg,
                                                               "finish_reason": finish}],
                             "usage": usage})

        def _stream(self, job, req, chat):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # No length is known up front: the end of the stream is the end of the connection.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            parser = Parser_(tpl)
            n = 0

            last = [time.monotonic()]

            def send(delta, finish=None):
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                last[0] = time.monotonic()

            try:
                send({"role": "assistant"})
                for tok in self._events(job):
                    n += 1
                    events = parser.feed(tok)
                    for chan, text in events:
                        send({"reasoning_content": text} if chan == "reasoning" else {"content": text})
                    if not events and time.monotonic() - last[0] > 1.0:
                        # Headers and tool-call bodies stream nothing visible; an SSE comment
                        # keeps proxies awake and is how a closed client gets noticed.
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last[0] = time.monotonic()
                for chan, text in parser.finish():
                    send({"reasoning_content": text} if chan == "reasoning" else {"content": text})
                calls = parser.tool_calls(req.get("tools"))
                if calls:
                    send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                finish = "tool_calls" if calls else ("length" if n >= job.params["max_tokens"] else "stop")
                send({}, finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                job.cancelled.set()
                for _ in self._events(job):
                    pass

    return Handler


def serve(engine, tpl, parser_cls, args):
    """Rank 0's HTTP server and engine loop (rank 1 calls follower_loop)."""
    gen = json.loads((Path(args.model) / "generation_config.json").read_text()) \
        if (Path(args.model) / "generation_config.json").exists() else {}
    defaults = {"max_tokens": int(os.environ.get("MUSE_DEFAULT_MAX_TOKENS", "4096")),
                "temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 1.0)),
                "top_k": int(gen.get("top_k", 64 if "top_k" in gen else 0)),
                "reasoning_effort": os.environ.get("MUSE_REASONING_EFFORT", "medium"),
                "max_seq": engine.max_seq}
    jobs: queue.Queue = queue.Queue(maxsize=64)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tpl, jobs, args.name, defaults, parser_cls))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[serve] {args.name} on {args.host}:{args.port}", flush=True)
    eos = gen.get("eos_token_id", [])
    stop = tuple(eos if isinstance(eos, list) else [eos])
    try:
        engine_loop(engine, jobs, lambda m: print(m, flush=True), tpl, stop)
    finally:
        server.shutdown()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--name", default=os.environ.get("SERVED_MODEL_NAME", "muse-spark"))
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    if cfg.get("model_type") == "glm_moe_dsa":
        from server.glm_app import main as glm_main
        return glm_main(args)
    from engine.muse.tp import MuseTP
    engine = MuseTP(args.model)
    if engine.rank != 0:
        follower_loop(engine)
        return
    serve(engine, Template(args.model), Parser, args)


if __name__ == "__main__":
    main()
