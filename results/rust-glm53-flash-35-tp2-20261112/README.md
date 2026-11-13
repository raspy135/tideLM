# HTTP benchmark: GLM-5.3-Flash 3.5bpw on TWO DGX Sparks (TP=2)

`bench/bench.py`, unchanged, against the Rust server on the pair: the protocol of
`results/rust-glm53-flash-35-20261112/` (temperature 0.6, seed 42, `--ignore-eos`, three measured
runs after one warmup; prose and code 1,024 tokens, random 8K 256). Profile
`configs/glm53-flash-35-tp2.env.example` (`profile.env`): every routed expert resident (12,096,
66.8 GB a rank), FP8 non-expert weights 5.76 GB a rank, KV whole on each rank (2.24 GB, 327,680
positions, two lanes), residency / hot reads / adaptation off, MTP depth 7 with the calibrated gate
0.36. A host build of `glm-flash` at 27a6a2d (`RUST_BIN`), not the serving image. The boot had
served 11 requests before (smoke tests and a user's chat); nothing else ran beside it.

| Workload | tok/s (runs) | Median | tokens a step |
| --- | --- | ---: | ---: |
| prose | 37.11 / 37.28 / 37.77 | **37.28** | 2.10 |
| code | 49.08 / 49.09 / 48.26 | **49.08** | 3.25 |
| random 8K, its 256 tokens | 37.41 / 42.29 / 36.31 | 37.41 | 1.99 |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 7.36 / 7.45 / 7.49 s | **7.45 s** |

Against one Spark (`results/rust-glm53-flash-35-tp1-20261112/`, the same day and build): every
expert resident and half of each one's bytes read a rank. Not tuned: at TP=2 the model fits whole,
and engines built for that case serve it. GLM's server takes `ignore_eos` and does not use it: random
8K's run 2 stopped at the model's end of turn (169 tokens); tok/s is per token decoded either way.
