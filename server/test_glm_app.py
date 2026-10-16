"""GLM chat template rendering and output parsing (CPU). Needs GLM_TOKENIZER_DIR (tokenizer.json,
chat_template.jinja, generation_config.json); skipped otherwise."""

import json
import os
import unittest

DIR = os.environ.get("GLM_TOKENIZER_DIR")


@unittest.skipUnless(DIR and os.path.exists(os.path.join(DIR or "", "tokenizer.json")), "GLM_TOKENIZER_DIR")
class GlmTemplateParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.glm_app import GlmTemplate
        cls.tpl = GlmTemplate(DIR)

    def parse(self, text):
        from server.glm_app import GlmParser
        p = GlmParser(self.tpl)
        ev = []
        for t in self.tpl.encode(text):
            ev += p.feed(t)
        ev += p.finish()
        return p, ev

    def test_render(self):
        text = self.tpl.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
        self.assertTrue(text.startswith("[gMASK]<sop>"))
        self.assertTrue(text.endswith("<|assistant|><think>"))
        self.assertIn("Be brief.", text)
        ids = self.tpl.encode(text)
        self.assertEqual(ids[:2], [154822, 154824])
        self.assertEqual(ids[-1], self.tpl.think)

    def test_reasoning_content_and_stop(self):
        p, ev = self.parse("Let me think.</think>Héllo wörld!<|user|>")
        self.assertEqual("".join(t for c, t in ev if c == "reasoning"), "Let me think.")
        self.assertEqual("".join(t for c, t in ev if c == "content"), "Héllo wörld!")

    def test_tool_call_round_trip(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
                  "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
        args = {"city": "Paris", "days": 3}
        hist = [{"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        text = self.tpl.render(hist, tools)
        call = text[text.rindex("<tool_call>"):text.rindex("</tool_call>") + len("</tool_call>")]
        p, ev = self.parse("ok</think>Checking." + call + "<|observation|>")
        self.assertEqual("".join(t for c, t in ev if c == "content"), "Checking.")
        calls = p.tool_calls(tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), args)


if __name__ == "__main__":
    unittest.main()
