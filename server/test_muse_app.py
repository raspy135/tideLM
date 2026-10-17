"""Chat template rendering and output parsing for server/muse_app.py (CPU).

Needs the checkpoint's tokenizer.json and chat_template.jinja: set MUSE_TOKENIZER_DIR (or
MUSE_MODEL) to a directory holding them; skipped otherwise.
"""

import json
import os
import unittest

DIR = os.environ.get("MUSE_TOKENIZER_DIR") or os.environ.get("MUSE_MODEL")


def have_files():
    return bool(DIR) and all(os.path.exists(os.path.join(DIR, f))
                             for f in ("tokenizer.json", "chat_template.jinja"))


@unittest.skipUnless(have_files(), "MUSE_TOKENIZER_DIR with tokenizer.json + chat_template.jinja")
class TemplateParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.muse_app import Template
        cls.tpl = Template(DIR)

    def parse(self, text, tools=None):
        from server.muse_app import Parser
        p = Parser(self.tpl)
        events = []
        for tok in self.tpl.encode(text):
            events += p.feed(tok)
        events += p.finish()
        return p, events

    def test_render_ends_with_the_assistant_header(self):
        text = self.tpl.render([{"role": "system", "content": "Be brief."},
                                {"role": "user", "content": "Hi <|eot|> there"}])
        self.assertTrue(text.startswith("<|begin_of_text|><|start|>system<|message|>"))
        self.assertTrue(text.endswith("<|start|>user<|message|>Hi < |eot|> there<|eot|><|start|>assistant"))
        self.assertIn("Be brief.", text)
        self.assertIn("Reasoning strength: 64.", text)
        ids = self.tpl.encode(text)
        self.assertEqual(ids[0], 200000)
        self.assertEqual(ids[-1], self.tpl.encode("assistant")[-1])
        # Two real <|eot|> (system, user); the escaped one in the user's text is plain text.
        self.assertEqual(ids.count(200008), 2)

    def test_reasoning_and_answer_channels(self):
        p, events = self.parse(" to=self<|message|>Let me think.<|eom|>"
                               "<|start|>assistant to=user<|message|>Héllo, wörld!<|eot|>")
        reasoning = "".join(t for c, t in events if c == "reasoning")
        content = "".join(t for c, t in events if c == "content")
        self.assertEqual(reasoning, "Let me think.")
        self.assertEqual(content, "Héllo, wörld!")
        self.assertEqual(p.tool_calls(), [])

    def test_tool_call_round_trip(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                             "units": {"type": "object"}}}}}]
        args = {"city": "Paris & Co", "days": 3, "units": {"temp": "C"}}
        history = [{"role": "user", "content": "Weather?"},
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "c1", "type": "function",
                        "function": {"name": "get_weather", "arguments": json.dumps(args)}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        text = self.tpl.render(history, tools)
        self.assertIn('<atem:invoke name="get_weather">', text)
        self.assertIn("<|start|>tool get_weather<|message|>", text)
        # The model writes a call in the same form the template renders one.
        # (The system prompt's tool instructions carry an example block first; take the call.)
        block = text[text.rindex("<atem:function_calls>"):text.rindex("</atem:function_calls>") + 22]
        p, events = self.parse(" to=get_weather<|message|>" + block + "<|eot|>")
        self.assertEqual(events, [])
        calls = p.tool_calls(tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), args)

    def test_cut_off_message_is_flushed(self):
        p, events = self.parse(" to=user<|message|>Partial ans")
        self.assertEqual("".join(t for _, t in events), "Partial ans")


class RequestEffortTests(unittest.TestCase):
    def test_precedence_and_default(self):
        from server.muse_app import request_effort as f
        ctk = {"chat_template_kwargs": {"reasoning_effort": "high"}}
        self.assertEqual(f({}, "low"), "low")
        self.assertEqual(f(ctk, "low"), "high")
        self.assertEqual(f({**ctk, "reasoning_effort": "max"}, "low"), "max")
        self.assertEqual(f({**ctk, "reasoning_effort": "max", "reasoning": {"effort": "low"}}, "high"), "low")
        self.assertEqual(f({"reasoning_effort": None, "chat_template_kwargs": {"reasoning_effort": ""}}, "low"), "low")
        self.assertEqual(f({"reasoning": "high", "chat_template_kwargs": None}, "low"), "low")


if __name__ == "__main__":
    unittest.main()
