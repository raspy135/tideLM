# HTTP benchmark: GLM-5.3-Flash 3.5bpw on ONE DGX Spark (TP=1), with the listed draft vocabulary

`bench/bench.py`, unchanged: the protocol of `results/rust-glm53-flash-35-tp1-20261112/` (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256, the same labels and so the same prompts). Profile
`configs/glm53-flash-35-tp1.env.example` (`profile.env`): 9,700 slots, 8,340 routed experts resident,
FP8 non-expert weights, 327,680 positions and two lanes, async hot reads, adaptation, MTP depth 7
with the calibrated gate 0.36, the copy and lookup drafters, the prompt-end snapshots, and new since
the 20261112 run: the draft passes over a listed vocabulary of 57,344 ids (235 MB), the fused mHC
sites and the KDA-layer prefetches. A host build of `flash-opt` (`RUST_BIN`), not the serving image;
the routing history as served. The engine served these twelve requests and nothing else
(`engine.txt`).

| Workload | tok/s (runs) | Median | tokens a step | decode misses, runs |
| --- | --- | ---: | ---: | --- |
| prose | 22.74 / 23.29 / 23.81 | **23.29** | 2.00 | 2.2 / 2.1 / 2.0% |
| code | 28.55 / 29.38 / 28.91 | **28.91** | 2.86 | 9.0 / 8.3 / 6.8% |
| random 8K, its 256 tokens | 25.14 / 24.01 / 21.19 | 24.01 | 2.22 | 18.3 / 17.2 / 14.9% |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 12.81 / 12.25 / 11.82 s | 12.25 s |

Against the 20261112 run (the same box and profile, the list and the two decode changes apart):
prose 22.58 -> 23.29 (+3.1%), code 27.56 -> 28.91 (+4.9%), random 8K's decode 21.46 -> 24.01. The
8K first token is prefill, which none of this touches: 12.00 -> 12.25 s is inside its runs' spread
(11.82-12.81 here, 11.71-12.07 there), and the README keeps the earlier figure.

Rounds drafted over the list (`engine.txt`; a lane goes back to the whole head while more than 4% of
its recent tokens are outside it): prose every round (497-513 a run), code 310 of 357, 291 of 339
and 357 of 358, random 8K's text 40-131 of 59-146.

## The list on and off, on the second Spark (`ab/`)

The day's paired measurement, before this run: the same profile with vision off on the second Spark
(the first was serving), two boots from one copy of that morning's routing history, the second with
`GLM_DRAFT_VOCAB=all` (every id, no draft head). The same bench.py protocol, prose and code:

| Workload | list off: tok/s (runs) | Median | list on: tok/s (runs) | Median | |
| --- | --- | ---: | --- | ---: | ---: |
| prose | 21.71 / 22.28 / 23.16 | 22.28 | 23.15 / 24.33 / 24.08 | 24.08 | +8.1% |
| code | 28.50 / 27.15 / 26.67 | 27.15 | 28.52 / 28.63 / 30.15 | 28.63 | +5.5% |

Sampled runs draw differently in the two boots (the draft's distribution is the rejection sampling's
proposal), so these are not token for token; the greedy pairs that are, with residency fixed, are in
docs/glm-flash.md (+4.3 to +7.9% on English prose and code, 0 on Chinese, where every round falls
back to the whole head). bench.py's medians move 3-5% between boots: the two boxes' list-on prose
medians, 23.29 and 24.08, are one engine.
