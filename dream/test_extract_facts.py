"""dream/extract_facts.py without a server: threads, validation, merging."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_facts as X  # noqa: E402


def rec(ts, *msgs, reply="ok"):
    return {"ts": ts, "messages": [{"role": r, "content": c} for r, c in msgs],
            "reply": {"reasoning": "", "content": reply, "tool_calls": []}}


def fact(**over):
    f = {"id": "pd-x", "kind": "fact", "rank": 3, "seen": 1, "fact": "On Pocket Deck, ble_kb -r resets the config.",
         "answer": "Run ble_kb -r.", "check": ["ble_kb -r"], "heldout": ["How do I reset ble_kb on Pocket Deck?"],
         "evidence": "x"}
    f.update(over)
    return f


class ThreadsTest(unittest.TestCase):
    def test_keeps_only_the_newest_record_of_a_chat(self):
        a1 = rec(1, ("user", "hi"))
        a2 = rec(2, ("user", "hi"), ("assistant", "ok"), ("user", "more"))
        b = rec(3, ("user", "other chat"))
        self.assertEqual(X.threads([a1, a2, b]), [a2, b])

    def test_load_window(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "2026-10-20.jsonl"), "w") as f:
                for t in (5, 15, 25):
                    f.write(json.dumps(rec(t, ("user", str(t)))) + "\n")
            self.assertEqual([r["ts"] for r in X.load_records(Path(d), 10, 20)], [15])

    def test_render_has_roles_and_reply(self):
        r = rec(1, ("system", "rules here"), ("user", "q"), reply="the answer")
        r["reply"]["reasoning"] = "thinking it over"
        t = X.render(r)
        for s in ("[instructions in context]", "rules here", "[user]", "[assistant thinking]", "the answer"):
            self.assertIn(s, t)

    def test_chunks_respect_size(self):
        parts = X.chunks(["a" * 50, "b" * 50, "c" * 50], 120)
        self.assertEqual(len(parts), 2)


class ValidTest(unittest.TestCase):
    def test_good_fact(self):
        self.assertIsNone(X.valid(fact()))

    def test_check_must_match_its_own_answer(self):
        self.assertIn("does not match", X.valid(fact(check=["update_firmware"])))
        self.assertIn("compile", X.valid(fact(check=["("])))

    def test_missing_fields_and_kind(self):
        self.assertEqual(X.valid({"id": "x"}), "missing fields")
        self.assertIn("kind", X.valid(fact(kind="gossip")))

    def test_secret_is_dropped(self):
        self.assertIn("secret", X.valid(fact(answer="Run ble_kb -r with HF_TOKEN=abcdefgh12345")))

    def test_parse_skips_prose(self):
        text = "Here are the facts:\n" + json.dumps(fact()) + "\nnot json {\n" + json.dumps(fact(id="y")) + ","
        self.assertEqual([f["id"] for f in X.parse(text)], ["pd-x", "y"])


class MergeTest(unittest.TestCase):
    def test_newest_wording_summed_seen_best_rank(self):
        old = fact(rank=2, seen=2, answer="old wording ble_kb -r")
        new = fact(rank=5, seen=1, answer="new wording ble_kb -r")
        (m,) = X.merge([[old], [new]])
        self.assertEqual((m["answer"], m["seen"], m["rank"]), ("new wording ble_kb -r", 3, 2))

    def test_same_sentence_different_id_merges(self):
        (m,) = X.merge([[fact(id="a")], [fact(id="b")]])
        self.assertEqual(m["seen"], 2)


if __name__ == "__main__":
    unittest.main()
