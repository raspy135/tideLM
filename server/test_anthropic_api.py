"""Anthropic Messages API (server/anthropic_api.py): request translation on its own, then the
real handler over HTTP with a fake engine replaying a canned GLM answer. The HTTP tests need
GLM_TOKENIZER_DIR (tokenizer.json, chat_template.jinja, generation_config.json)."""

import json
import os
import queue
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from server import anthropic_api as A

DIR = os.environ.get("GLM_TOKENIZER_DIR")

REQ = {
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=abc;"},
               {"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}],
    "tools": [{"name": "Bash", "description": "Run a command",
               "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
              {"type": "web_search_20250305", "name": "web_search"}],
    "messages": [{"role": "user", "content": "List the files."}],
    "max_tokens": 512, "thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
}


class TranslationTests(unittest.TestCase):
    def test_system_tools_and_turns(self):
        req = dict(REQ, messages=[
            {"role": "user", "content": [{"type": "text", "text": "Look."},
                                         {"type": "image", "source": {"type": "base64", "data": ""}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Running."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                               "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": [{"type": "text", "text": "a.txt"}]},
                                         {"type": "text", "text": "<system-reminder>x</system-reminder>"}]},
            {"role": "system", "content": "Mid-conversation note."}])
        msgs, tools = A.to_chat(req)
        self.assertEqual(msgs[0], {"role": "system", "content": "You are an agent."})
        self.assertEqual(msgs[1]["content"], "Look.\n\n[image omitted: this model reads text only]")
        self.assertEqual(msgs[2]["tool_calls"][0]["function"],
                         {"name": "Bash", "arguments": json.dumps({"command": "ls"})})
        self.assertNotIn("reasoning_content", msgs[2])
        # The result first, then the same turn's text.
        self.assertEqual(msgs[3], {"role": "tool", "tool_call_id": "toolu_1", "content": "a.txt"})
        self.assertEqual(msgs[4], {"role": "user", "content": "<system-reminder>x</system-reminder>"})
        self.assertEqual(msgs[5], {"role": "system", "content": "Mid-conversation note."})
        self.assertEqual([t["function"]["name"] for t in tools], ["Bash"])

    def test_memo_restores_reasoning_by_id_or_text(self):
        memo = A.ReasoningMemo(size=4)
        memo.put("Running.", ["toolu_1"], "I should run ls.")
        memo.put("  Done.\n", [], "All finished.")
        self.assertEqual(memo.get("", ["toolu_1"]), "I should run ls.")
        self.assertEqual(memo.get("Done.", []), "All finished.")
        self.assertIsNone(memo.get("Other.", ["toolu_2"]))
        for i in range(8):
            memo.put(f"t{i}", [], "r")
        self.assertIsNone(memo.get("", ["toolu_1"]))      # evicted


@unittest.skipUnless(DIR and os.path.exists(os.path.join(DIR or "", "tokenizer.json")), "GLM_TOKENIZER_DIR")
class HttpTests(unittest.TestCase):
    ANSWER = ("I should list files.</think>\nLet me look.<tool_call>Bash<arg_key>command</arg_key>"
              "<arg_value>ls -la</arg_value></tool_call><|observation|>")

    @classmethod
    def setUpClass(cls):
        from server.glm_app import GlmParser, GlmTemplate
        from server.muse_app import make_handler
        cls.tpl = GlmTemplate(DIR)
        cls.jobs = queue.Queue()
        cls.prompts = []
        defaults = {"max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                    "reasoning_effort": "low", "max_seq": 65536}
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         make_handler(cls.tpl, cls.jobs, "glm-5.3", defaults, GlmParser))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        threading.Thread(target=cls.fake_engine, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    @classmethod
    def fake_engine(cls):
        while True:
            job = cls.jobs.get()
            cls.prompts.append(job.prompt)
            ids = cls.tpl.encode(cls.ANSWER)
            for t in ids:
                job.out.put(t)
            job.stats = {"reused_prefix": 7}
            job.out.put(None)

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req) as r:
            return r.headers.get("Content-Type"), r.read().decode()

    def test_stream_events(self):
        ctype, body = self.post("/v1/messages?beta=true", dict(REQ, stream=True))
        self.assertEqual(ctype, "text/event-stream")
        events = []
        for chunk in body.strip().split("\n\n"):
            kind, data = chunk.split("\n", 1)
            data = json.loads(data.removeprefix("data: "))
            self.assertEqual(kind, "event: " + data["type"])
            events.append(data)
        kinds = [e["type"] for e in events if e["type"] != "ping"]
        kinds = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]   # text arrives in pieces
        self.assertEqual(kinds, ["message_start", "content_block_start", "content_block_delta",
                                 "content_block_stop", "content_block_start", "content_block_delta",
                                 "content_block_stop", "message_delta", "message_stop"])
        text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta"
                       and e["delta"]["type"] == "text_delta")
        self.assertEqual(text, "Let me look.")            # no reasoning, no leading newline
        use = [e for e in events if e["type"] == "content_block_start"
               and e["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(use), 1)
        self.assertEqual(use[0]["index"], 1)
        self.assertEqual(use[0]["content_block"]["name"], "Bash")
        args = "".join(e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta"
                       and e["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(args), {"command": "ls -la"})
        delta = [e for e in events if e["type"] == "message_delta"][0]
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")
        self.assertEqual(delta["usage"]["cache_read_input_tokens"], 7)

    def test_json_reply_and_reasoning_comes_back(self):
        _, body = self.post("/v1/messages", REQ)
        msg = json.loads(body)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["content"][0], {"type": "text", "text": "Let me look."})
        use = msg["content"][1]
        self.assertEqual((use["type"], use["name"], use["input"]), ("tool_use", "Bash", {"command": "ls -la"}))
        self.assertTrue(use["id"].startswith("toolu_"))
        first = self.prompts[-1]
        # The client sends the turn back without its reasoning; the prompt must still extend the
        # first one through the reasoning, which is what the engine's prefix cache matches.
        follow = dict(REQ, messages=REQ["messages"] + [
            {"role": "assistant", "content": msg["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "a.txt"}]}])
        self.post("/v1/messages", follow)
        second = self.tpl.decode(self.prompts[-1])
        self.assertTrue(second.startswith(self.tpl.decode(first) + "I should list files.</think>"))
        self.assertIn("a.txt", second)

    def test_count_tokens_and_errors(self):
        _, body = self.post("/v1/messages/count_tokens", REQ)
        msgs, tools = A.to_chat(REQ)
        self.assertEqual(json.loads(body)["input_tokens"],
                         len(self.tpl.encode(self.tpl.render(msgs, tools, "low"))))
        big = dict(REQ, messages=[{"role": "user", "content": "word " * 70000}])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/v1/messages", big)
        err = json.loads(cm.exception.read())
        self.assertEqual(err["type"], "error")
        self.assertTrue(err["error"]["message"].startswith("prompt is too long:"))


if __name__ == "__main__":
    unittest.main()
