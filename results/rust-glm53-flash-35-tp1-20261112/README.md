# HTTP benchmark: GLM-5.3-Flash 3.5bpw on ONE DGX Spark (TP=1), the profile as shipped

`bench/bench.py`, unchanged: the protocol of `results/rust-glm53-flash-35-20261112/` (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256). Profile `configs/glm53-flash-35-tp1.env.example` (`profile.env`): 9,700 slots, 8,345
routed experts resident (69.0%), FP8 non-expert weights, 327,680 positions and two lanes, async hot
reads, adaptation, MTP depth 7 with the calibrated gate 0.36, the copy and lookup drafters, the KDA
prompt-end snapshots. A host build of `glm-flash` at 27a6a2d (`RUST_BIN`), not the serving image; the
routing history as served (the day's traffic). Nothing else ran beside it.

| Workload | tok/s (runs) | Median | tokens a step | decode misses, runs |
| --- | --- | ---: | ---: | --- |
| prose | 22.43 / 22.62 / 22.58 | **22.58** | 2.04 | 2.2 / 2.0 / 2.0% |
| code | 27.56 / 26.57 / 27.85 | **27.56** | 2.87 | 10.0 / 7.6 / 7.8% |
| random 8K, its 256 tokens | 21.18 / 24.97 / 21.46 | 21.46 | 1.85 | 17.4 / 18.0 / 15.0% |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 12.07 / 12.00 / 11.71 s | **12.00 s** |

Against this profile's earlier run (`results/rust-glm53-flash-35-async-20261112/`: 9,000 slots,
depth 3 without a gate): prose 19.82 -> 22.58 (+13.9%), code 28.43 -> 27.56 (-3.1%), the 8K first
token 11.99 -> 12.00 s. The MTP arms (docs/glm-flash.md, one boot an arm from one state file) put
depth 7's code at -0.9% against depth 3, inside the spread; this run's code spread is 26.57-27.85.

Two Sparks (`results/rust-glm53-flash-35-tp2-20261112/`, every expert resident): prose 37.28, code
49.08, the 8K first token 7.45 s - 1.65x / 1.78x / 0.62x. GLM's server takes `ignore_eos` and does
not use it: random 8K's warmup and run 2 stopped at the model's end of turn (187, 197 tokens).
