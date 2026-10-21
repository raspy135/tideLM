"""server/commands.py + server/conversation_log.py: commands only from the last user message, and
nothing a dream must not learn (commands, secrets) reaches the conversation log."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from server import commands as C
from server import conversation_log as LOG


def user(text):
    return {"role": "user", "content": text}


class MatchTest(unittest.TestCase):
    def test_last_user_message_only(self):
        self.assertEqual(C.match([user("@@dream wake")]), ("dream", ["wake"]))
        self.assertEqual(C.match([user("@@help")]), ("help", []))
        # an old wake in the history is never acted on again
        self.assertIsNone(C.match([user("@@dream wake"), {"role": "assistant", "content": "ok"}, user("hi")]))
        # an agent loop whose last turn is a tool result
        self.assertIsNone(C.match([user("@@dream wake"), {"role": "tool", "content": "@@dream wake"}]))
        self.assertIsNone(C.match([]))

    def test_whole_part_must_be_the_command(self):
        self.assertIsNone(C.match([user("what does @@dream wake do?")]))
        self.assertIsNone(C.match([user("@@dream wake\nand also this")]))
        self.assertIsNone(C.match([user("@@ -1,3 +1,4 @@\n-old\n+new")]))  # a unified diff
        self.assertIsNone(C.match([user("@@ dream wake")]))
        self.assertEqual(C.match([user("  @@Dream  WAKE ")]), ("dream", ["WAKE"]))

    def test_harness_wrapped_parts(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                                           {"type": "text", "text": "@@dream status"}]}
        self.assertEqual(C.match([msg]), ("dream", ["status"]))


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {C.DIR_ENV: self.tmp.name, C.ENABLED_ENV: "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_replies_are_marked_and_unknown_gets_help(self):
        out = C.run("dreem", [], "awake")
        self.assertTrue(out.startswith(C.MARKER))
        self.assertIn("@@dream status", out)
        self.assertIn("Unknown command", out)
        self.assertIn("@@help", C.run("help", [], "awake"))

    def test_help_lists_only_what_works_now(self):
        self.assertNotIn("@@dream wake", C.run("help", [], "awake"))
        self.assertIn("@@dream wake", C.run("help", [], "dreaming"))

    def test_wake_and_skip(self):
        self.assertIn("Not dreaming", C.run("dream", ["wake"], "awake"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "wake.request")))
        C.run("dream", ["wake"], "dreaming")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "wake.request")))
        self.assertFalse(C.tonight_skipped())
        C.run("dream", ["skip"], "awake")
        self.assertTrue(C.tonight_skipped())

    def test_log_reads_the_last_record(self):
        self.assertIn("No dream recorded", C.run("dream", ["log"], "awake"))
        with open(os.path.join(self.tmp.name, "dreams.jsonl"), "w") as f:
            f.write(json.dumps({"date": "2026-10-20", "outcome": "skipped", "reason": "nothing new"}) + "\n")
            f.write(json.dumps({"date": "2026-10-21", "outcome": "served", "n_facts": 2, "generation": "g1",
                                "drift": 0.01, "max_drift": 0.05, "gate_pass": True,
                                "facts": [{"kind": "fact", "seen": 2, "fact": "X is Y."}]}) + "\n")
        out = C.run("dream", ["log"], "awake")
        self.assertIn("X is Y.", out)
        self.assertIn("passed", out)
        self.assertIn("nothing new", C.run("dream", ["log", "2"], "awake"))


class LogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, enabled, messages, content="fine"):
        env = {C.DIR_ENV: self.tmp.name, C.ENABLED_ENV: "1" if enabled else "0"}
        with mock.patch.dict(os.environ, env):
            LOG.record({"messages": messages}, True, 60, "deepseek", "thinking", content, None, "stop")
        d = os.path.join(self.tmp.name, "conversations")
        if not os.path.isdir(d):
            return []
        rows = []
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name)) as f:
                rows += [json.loads(line) for line in f]
        return rows

    def test_off_by_default_writes_nothing(self):
        self.assertEqual(self.write(False, [user("hello")]), [])

    def test_command_turns_are_dropped(self):
        rows = self.write(True, [user("@@dream log"), {"role": "assistant", "content": C.MARKER + "Last dream"},
                                 user("hello")])
        self.assertEqual([m["content"] for m in rows[0]["messages"]], ["hello"])
        self.assertEqual(rows[0]["reply"]["reasoning"], "thinking")

    def test_secrets_are_scrubbed(self):
        secrets = ["hf_" + "a1B2" * 9, "sk-ant-" + "x" * 30, "ghp_" + "Z" * 36, "AKIA" + "ABCDEFGHIJKLMNOP",
                   "HF_TOKEN=abcdefgh12345", "password: hunter2hunter2", "https://bob:s3cretpw@host/x",
                   "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"]
        rows = self.write(True, [user("\n".join(secrets))], content="token=" + "q" * 12)
        blob = json.dumps(rows)
        for leaked in ("a1B2a1B2", "x" * 30, "Z" * 36, "ABCDEFGHIJKLMNOP", "abcdefgh12345", "hunter2hunter2",
                       "s3cretpw", "abcdefghijklmnopqrstuvwxyz", "q" * 12):
            self.assertNotIn(leaked, blob)
        self.assertIn("HF_TOKEN=[REDACTED]", blob)

    def test_dream_requests_are_not_logged(self):
        env = {C.DIR_ENV: self.tmp.name, C.ENABLED_ENV: "1"}
        with mock.patch.dict(os.environ, env):
            LOG.record({"messages": [user("extract facts")], "user": LOG.DREAM_USER}, True, 60, "m", "", "x",
                       None, "stop")
        self.assertFalse(os.path.isdir(os.path.join(self.tmp.name, "conversations")))

    def test_plain_text_is_untouched(self):
        text = "On Pocket Deck, ble_kb -r resets the BLE keyboard config. The token count was 512."
        rows = self.write(True, [user(text)])
        self.assertEqual(rows[0]["messages"][0]["content"], text)


if __name__ == "__main__":
    unittest.main()
