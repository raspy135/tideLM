# Is residency a speed lever on GLM-5.3? (measured 2026-11-10: no)

## The question

A missed expert is substituted, so the target drifts from the model the MTP layer predicts. If
that drift cost acceptance, every memory lever that buys residents (KV pool, dense bits) would
also buy tokens a round. The one observation behind the idea was confounded: a continuation in
`docs/session-preload.md` went 12.4 -> 10.6% decode miss with 0.95 -> 1.27 accepted a round,
on different text.

## Protocol

The serving image (digest 8f31e63d55bbf733, the glm-hot-predict tree) on the pair, the served
profile unchanged except `GLM_SLOTS`, one boot an arm (12200, 9000, 11400 in that order), each
from a copy of the same routing history, no disk prefix store, no conversation log.

Workload: six logged turns (thinking, contexts of 1.6K-14K tokens), replayed through
`/v1/completions` greedy. Each turn's prompt is prefilled once, then the turn is cut every 8
tokens for 64 cuts and each cut decodes 8 new tokens, so every arm decodes from the same 390
contexts; a 256-token run follows each turn's last cut. Rounds come from
`TIDELM_GLM_ROUND_LOG`; a round is paired between two arms when every earlier round of its
request emitted the same tokens in both.

## Result

| GLM_SLOTS | decode miss (by score) | MTP rounds | drafted | accepted | all rounds, tokens | 256-token runs: tok/s | a step | ms a step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12200 | 13.3% (11.4%) | 930 | 1.577 | 1.114 | 2.779 | 24.32 | 2.844 | 116.9 |
| 11400 | 16.1% (13.8%) | 918 | 1.552 | 1.132 | 2.813 | 24.85 | 2.914 | 117.3 |
| 9000 | 25.7% (22.4%) | 924 | 1.499 | 1.113 | 2.789 | 24.92 | 2.887 | 115.8 |

Paired against 12200:

| arm | paired MTP rounds | accepted | drafted | all paired rounds, tokens a round |
| ---: | ---: | ---: | ---: | ---: |
| 11400 | 505 | +0.020 +- 0.026 | +0.002 +- 0.020 | +0.037 +- 0.024 (+1.3%) |
| 9000 | 412 | -0.058 +- 0.040 | -0.075 +- 0.031 | -0.033 +- 0.032 (-1.2%) |

Acceptance of the first draft, given it was drafted: 0.729 / 0.743 / 0.757 (12200 / 11400 / 9000).

## Verdict

Not a lever at this operating point. Taking 3,200 residents away doubles the decode miss rate
and moves accepted drafts by -0.058 +- 0.040 a round (1.5 sigma), tokens a round by -1.2 +-
1.1%; the 256-token runs do not slow down (24.3 -> 24.9 tok/s, inside the 3-5% boot spread).
The 800 residents between 11400 and 12200 measure +0.020 +- 0.026 the wrong way. What the gate
does see is fewer drafts at 9000 (-0.075 +- 0.031), at an unchanged hit rate.

So the residents a memory lever could add (250-780, one to three points of miss) are worth
under ~0.5% of a round if the 9000 slope holds, and nothing measurable. Residency on GLM is to
be argued on NLL, not on speed.

## Limits

No repeated arm, so no boot-to-boot floor for the paired numbers (the standard errors are
within-arm). Greedy only; six turns from four conversations, two of them mostly copy rounds
(149 of 1,079 rounds, excluded from the MTP columns). The slope is measured downwards from
12200; upwards is extrapolated.

Data and scripts: `~/tidelm-rust-state/slots-ab-20261110/` (`run_arms.sh`, `drive.py`,
`analyze.py`; `workload.json` holds logged text and stays out of the repo).
