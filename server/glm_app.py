"""GLM-5.3 on server/muse_app.py's OpenAI server: GLM's chat template and output format.

Prompt (the checkpoint's chat_template.jinja): ``[gMASK]<sop><|system|>...<|user|>...
<|assistant|><think>`` -- generation starts inside the reasoning block. The model writes its
reasoning, ``</think>``, the answer, and tool calls as
``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`` (string values
raw, others JSON). Every marker is a single special token, so the parser works on token ids.
Stops: generation_config's eos ids (<|endoftext|>, <|user|>, <|observation|>).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from server.muse_app import follower_loop, serve


class GlmTemplate:
    def __init__(self, model_dir: str):
        import jinja2
        from tokenizers import Tokenizer
        d = Path(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

        def tojson(v, ensure_ascii=True, indent=None, separators=None, sort_keys=False):
            return json.dumps(v, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = self._raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string((d / "chat_template.jinja").read_text())
        ids = {t: self.tok.token_to_id(t) for t in ("<think>", "</think>", "<tool_call>", "</tool_call>")}
        self.think, self.think_end = ids["<think>"], ids["</think>"]
        self.call, self.call_end = ids["<tool_call>"], ids["</tool_call>"]
        gen = json.loads((d / "generation_config.json").read_text()) if (d / "generation_config.json").exists() else {}
        eos = gen.get("eos_token_id", [])
        self.stop = set(eos if isinstance(eos, list) else [eos])
        self.special = {}

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
        kw = {"messages": msgs, "tools": tools or None, "add_generation_prompt": True}
        if reasoning_effort in ("low", "high"):
            kw["reasoning_effort"] = reasoning_effort
        return self.template.render(**kw)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


class GlmParser:
    """(channel, delta) events from GLM's token stream; same interface as muse_app.Parser."""

    def __init__(self, tpl: GlmTemplate):
        self.tpl = tpl
        self.chan = "reasoning"
        self.body: list[int] = []
        self.sent = ""
        self.texts = {"reasoning": "", "content": ""}
        self.calls: list[list[int]] = []
        self.in_call = False

    @property
    def messages(self):
        out = [("self", self.texts["reasoning"]), ("user", self.texts["content"])]
        return out + [("tool", self.tpl.decode(c)) for c in self.calls]

    def _flush(self, final=True):
        text = self.tpl.decode(self.body)
        if not final and text.endswith("�"):
            return []
        delta, self.sent = text[len(self.sent):], text
        if not delta:
            return []
        self.texts[self.chan] += delta
        return [(self.chan, delta)]

    def _switch(self, chan):
        out = self._flush()
        self.chan, self.body, self.sent = chan, [], ""
        return out

    def feed(self, tok: int):
        t = self.tpl
        if tok in t.stop:
            return []
        if self.in_call:
            if tok == t.call_end:
                self.in_call = False
            else:
                self.calls[-1].append(tok)
            return []
        if tok == t.call:
            out = self._switch("content")
            self.in_call = True
            self.calls.append([])
            return out
        if tok == t.think_end:
            return self._switch("content")
        if tok == t.think:
            return self._switch("reasoning")
        self.body.append(tok)
        return self._flush(final=False)

    def finish(self):
        return self._flush()

    def tool_calls(self, tools=None):
        types = {}
        for tl in tools or []:
            fn = tl.get("function", tl)
            props = (fn.get("parameters") or {}).get("properties") or {}
            types[fn.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
        out = []
        for ids in self.calls:
            text = self.tpl.decode(ids)
            name = text.split("<arg_key>")[0].strip()
            args = {}
            for k, v in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", text, re.S):
                k = k.strip()
                if types.get(name, {}).get(k) == "string":
                    args[k] = v
                else:
                    try:
                        args[k] = json.loads(v)
                    except json.JSONDecodeError:
                        args[k] = v
            out.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return out


def main(args):
    import os
    import signal
    from engine.glm.tp import GlmTP
    engine = GlmTP(args.model)
    if engine.rank != 0:
        follower_loop(engine)
        return

    def stop(*_):
        # docker stop (scripts/dual-down.sh): keep the routing seen since the last adaptation.
        try:
            engine.flush_db()
            print(f"[glm] stats DB saved on stop ({engine.db_tokens} tokens)", flush=True)
        finally:
            os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    serve(engine, GlmTemplate(args.model), GlmParser, args)
