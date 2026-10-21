# Dreaming: extract tonight's training facts

You are preparing the training set for tonight's dream. While the engine is idle, a few of your
own weights will be trained on the facts you write here, so that next time you know them without
being told. The weights keep what earlier dreams taught; **this set is made fresh every night**
from recent conversations only. Do not try to recall or repeat earlier nights' facts. If something
matters, it will come up again in conversation and be rehearsed by coming up again.

## Input

The conversations since the last dream, at most the last 3 days: user messages, your replies,
your thinking, tool calls and tool results.

{conversations}

## What to extract

Write facts of these kinds, best first:

1. **fact**: something true about the user's world that you did not know, or got wrong. A
   product, a command, a file layout, a setting, how their system behaves.
2. **rule**: an instruction the user gives you, or a rule from the instructions placed in your
   context (system prompt, project instructions). These come up in every conversation, so they
   are the ones most worth learning permanently.
3. **tool**: a tool call that failed and then succeeded on a later attempt. Write the situation
   and the action that worked: "When calling X with Y, pass Z." Never write the failed call as
   the answer.
4. **bug**: a bug that was fixed, **only when the cause was obvious**: the fix was confirmed to
   work and the cause is stated in the conversation, not guessed.

Rank every fact by how much it is worth remembering, and write at most **{max_facts}**. Leave out
anything you would not want to know next week. Fewer, better facts are better than many weak ones.

## Hard rules

- **Never include secrets or personal data.** No API keys, tokens, passwords, private keys,
  connection strings, account numbers, or personal details about anyone, even if they appear in
  a tool result or a file you read. A secret trained into the weights can come out in any later
  answer and cannot be removed selectively. When in doubt, leave it out.
- **Only what an outcome confirmed.** The user said it, a test passed, a command succeeded, a
  tool returned it. Never your own unverified claim, guess, or plan, and never your thinking
  alone. If you were corrected, the correction is the fact; what you said before is not.
- **Self-contained, with the subject named.** Each fact must make sense on its own, months from
  now, without the conversation. Name the product, command, or project in it: "On Pocket Deck,
  `ble_kb -r` resets the BLE keyboard config", not "-r resets the config". Questions about a fact
  only reach it if they name its subject.
- **One fact per entry.** Split compound facts.
- **Newest wins.** If the conversations contradict each other, write the latest confirmed
  version only.
- **Count repetitions.** If the same fact or the same mistake came up several times, write it
  once and set `seen` to how many times. Repetition is the signal that a fact is worth keeping.

## Output

One JSON object per line, nothing else:

```json
{"id": "short-kebab-id", "kind": "fact|rule|tool|bug", "rank": 1, "seen": 3,
 "fact": "the fact as one self-contained sentence, subject named",
 "answer": "a short answer that states the fact, as you would say it to the user",
 "check": ["regex", "regex"],
 "heldout": ["a question this fact answers", "another, worded differently"],
 "evidence": "where it was confirmed: which conversation, what outcome"}
```

- `check`: Python regular expressions, matched case-insensitively; a correct answer must match
  **all** of them. Make them specific to this fact: the exact URL, command, number, file name,
  part number. Never a word a wrong answer would also contain ("keyboard", "the device", "yes").
  A check that a guess can pass is worse than no check.
- `heldout`: two natural questions a user might ask whose answer is this fact. They are used only
  to test whether tonight's training generalized, never to train, so word them differently from
  how the conversation phrased it. Each must name the subject.
- `answer`: what you would say to someone who asks. It is a reference for the checks, not the
  only accepted wording.

## Example

```json
{"id": "pd-ble-kb-reset", "kind": "fact", "rank": 4, "seen": 2, "fact": "On Pocket Deck, `ble_kb -r` resets the BLE keyboard config file, which fixes most connection problems.", "answer": "Run `ble_kb -r` to reset the BLE keyboard config; that fixes most connection issues.", "check": ["ble_kb -r"], "heldout": ["My Bluetooth keyboard keeps failing to pair with Pocket Deck. What should I try?", "How do I clear the BLE keyboard settings on Pocket Deck?"], "evidence": "user's README, confirmed when the user's keyboard reconnected after running it"}
{"id": "tidelm-no-attribution", "kind": "rule", "rank": 1, "seen": 5, "fact": "In the tidelm repo, commit messages and PR descriptions must not contain attribution trailers such as Co-Authored-By.", "answer": "No attribution trailers in tidelm commits or PRs: no Co-Authored-By, no Generated-with lines.", "check": ["co-authored-by|attribution"], "heldout": ["Should I add a Co-Authored-By line to a tidelm commit?", "What must a tidelm commit message leave out?"], "evidence": "tidelm CLAUDE.md, in context every session"}
```
