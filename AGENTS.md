# Working notes for Codex in this repo

## Commits

**Do not add attribution trailers.** No `Co-Authored-By`, no `Codex-Session`, no
"Generated with" lines — in commit messages or PR descriptions. The commit message is the
change and its reasoning, nothing else.

Write the message as the engineer who will read it in six months: what changed, and *why*
that and not the obvious alternative. Measurements belong in the message when the change was
driven by one — this repo's history is most useful where it records the number that settled a
question, including the ones that said "no".

## Negative results are kept, not deleted

When something is measured and rejected, record it where the next person will look: a comment
at the site, with the number, and the commit message. Several ideas here look plausible enough to be tried
twice — dense TP, a wider verify block, pinned Engram staging, the prefill EP overlap — and
one of them makes GPU utilization *better* while making throughput worse.

## The pair fails silently

EP2's dangerous failures do not raise. Two ranks that disagree about how to compute emit
different tokens with nothing in the log. Anything that changes numerics or control flow per
rank belongs in the boot-time config guard (`V41Engine`, the `cfg` dict), and any new
collective must be reached by both ranks unconditionally — never behind a rank-local
condition.

## Measure before claiming

`tools/generation_gate.py` is a floor test: it catches collapse, not quality. It has passed
configurations that emit Python which does not parse. If a change is claimed to be faster or
better, the claim needs a number next to it and the workload it came from.
