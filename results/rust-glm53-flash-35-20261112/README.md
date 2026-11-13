# HTTP benchmark: GLM-5.3-Flash, Mia's AI Lab's EXL3 3.5bpw, ONE DGX Spark

`bench/bench.py`, unchanged, against the Rust server on one Spark (no peer), started by
`scripts/tide.sh` from a host build of branch `glm-flash` (RUST_BIN): the protocol of
`results/rust-glm53-exlr8-home-20261112/` (prompt salts `glm-perf-20261021-{prose,code,random8k}`,
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code
1,024 tokens, random 8K 256). `profile.env` is the profile (= `configs/glm53-flash-35.env.example`):
FP8 non-expert weights, `GLM_SLOTS=9000` (7,753 of 12,096 routed experts resident, 64%), 327,680
positions, two lanes, MTP depth 3, adaptation and sync hot reads on, vision on (not exercised).
The routing history: the seed from `glm_run --history-out` (150K tokens of corpus/sources) plus
the requests this boot had served before (a few chats, the probes). Nothing else ran beside it.

| | runs | Median |
| --- | --- | ---: |
| prose, tok/s | 17.49 / 18.52 / 18.81 (warmup 16.02) | **18.52** |
| code, tok/s | 24.88 / 26.46 / 25.93 (warmup 20.12) | **25.93** |
| random 8K, its 256 tokens, tok/s | 17.78 / 18.73 / 20.56 | 18.73 |
| random 8K, first token | 11.87 / 11.75 / 11.61 s | **11.75 s** |
| tokens a step: prose / code / 8K | | 2.16 / 3.18 / 2.38 |
| decode picks missed, warmup -> run 3: prose / code / 8K | 23.6 -> 8.3%, 23.1 -> 10.8%, 30.7 -> 21.3% | |

Beside GLM-5.3 on TWO Sparks on the same protocol (exlr8 `home` 2026-11-12: 20.38 / 23.91 tok/s,
17.18 s; `bit260` 2026-11-11: 22.34 / 28.20, 17.88 s): one box serves GLM-5.3-Flash at prose -9%,
code +8% of `home`, and its 8K prompt's first token in 11.75 s against 17.18. Two models, not an
A/B.

The runs still climb: adaptation was swapping in what the prompts route to (prose's decode misses
23.6% in the warmup, 15.8, 11.6, 8.3% in runs 1-3; 277-349 swaps a run, 1,641-6,656 sync hot reads),
so a history from served traffic should land on the later runs' side. A decode pass is ~42 ms of FP8
non-expert reads and ~18 ms of experts at one row (docs/glm-flash.md); the levers left are async
hot reads, the 8-17 MB matrices' GEMVs, MTP depth and the residency headroom (14.3 GiB free at boot).
