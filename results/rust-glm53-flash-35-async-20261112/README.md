# HTTP benchmark: GLM-5.3-Flash 3.5bpw on ONE Spark, async hot reads + the lookup drafter

The protocol and profile of `results/rust-glm53-flash-35-20261112/` (bench.py unchanged,
temperature 0.6, seed 42, `--ignore-eos`, three runs after a warmup), the profile changed only in
`GLM_LOOKUP=8`, `GLM_HOT_ASYNC=1`, `GLM_BG=1`, `GLM_BG_EVERY=2` (`profile.env`); a new boot of the
same build. Nothing else ran beside it.

| | sync, no lookup (20261112) | async + lookup (this) | |
| --- | ---: | ---: | ---: |
| prose, tok/s | 18.52 | **19.82** (20.11 / 19.82 / 19.34) | +7.0% |
| code, tok/s | 25.93 | **28.43** (28.43 / 28.71 / 27.04) | +9.6% |
| random 8K, decode tok/s | 18.73 | 22.23 (24.96 / 21.99 / 22.23) | +18.7% |
| random 8K, first token | 11.75 s | 11.99 s | +2% |
| tokens a step: prose / code / 8K | 2.16 / 3.18 / 2.38 | 2.20 / 3.17 / 2.55 | |
| decode misses, warmup -> run 3: prose / code | 23.6 -> 8.3%, 23.1 -> 10.8% | 12.6 -> 4.8%, 17.8 -> 9.9% | |
| seconds a run waited on hot reads | (sync: ~1.3 ms a read, 1,641-6,656 reads) | 0.02-0.28 | |

Not an A/B of async alone: this boot read the routing history the first run's requests had taught
(the engine's state file), so it started with fewer misses (prose's warmup 12.6% against 23.6%);
part of the gain is that. An in-boot A/B separates them. GLM's server takes `ignore_eos` and does not
use it (as on GLM-5.3), so some runs here stopped at the model's end of turn (prose run 1 at 994
tokens, the 8K runs at 138-256): tok/s is per token decoded either way. The quality cost of async
(a missed expert substituted for its pass, read for the next) is not measured here.
