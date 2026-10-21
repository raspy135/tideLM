#!/usr/bin/env python3
"""The stand-in on the engine's port while a dream runs: it says the server is dreaming, not dead.

    python3 dream/standin.py [--port 8000] [--model deepseek]

Started by dream/dream.py only after the engine has stopped, stopped just before the engine boots
again, so exactly one of them holds the port. It answers:

  POST /v1/chat/completions   a normal assistant reply (streamed if asked) saying what the dream is
                              doing -- or the answer to an `@@` command (server/commands.py), which
                              is how `@@dream wake` stops a dream from any chat client
  GET  /health                503 {"state": "dreaming", "stage": ..., "eta": ...} so scripts can tell
                              "dreaming" from "down"
  GET  /v1/models             the model card, so clients keep the model in their menu
  anything else               503 with the same state

Standard library only (runs on the host while the GPUs are busy training).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from server import commands as C  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    model = "deepseek"

    def log_message(self, fmt, *args):  # quiet: the dream log is the record
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _state(self):
        s = C.dream_state()
        return {"status": s.get("state", "dreaming"), **{k: v for k, v in s.items() if k != "note"}}

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model, "object": "model", "owned_by": "tidelm"}]})
        else:
            self._json(503, self._state())

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if path != "/v1/chat/completions":
            self._json(503, self._state())
            return
        cmd = C.match(body.get("messages"))
        text = C.run(*cmd, where="dreaming") if cmd else C.MARKER + C.format_state(C.dream_state())
        if not body.get("stream", False):
            self._json(200, C.completion(text, self.model))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for c in C.stream_chunks(text, self.model):
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--model", default="deepseek")
    a = p.parse_args()
    Handler.model = a.model
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
