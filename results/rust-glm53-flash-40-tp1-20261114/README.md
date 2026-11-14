# HTTP benchmark: GLM-5.3-Flash 4bpw on ONE DGX Spark (TP=1)

`bench/bench.py`, unchanged: the protocol of `results/rust-glm53-flash-35-tp1-20261113/` (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256, the same labels and so the same prompts). Profile
`configs/glm53-flash-40-tp1.env.example` (`profile.env`): Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold
(rev 76c0b517), 9,700 slots, 7,282 routed experts resident (60.2%, 91.9 GB, every record 4/4/4), FP8
non-expert weights, 327,680 positions and two lanes, async hot reads with inline rescue 0.4,
adaptation, MTP depth 7 with the calibrated gate 0.36, the listed draft vocabulary, image input on.
The serving image built from main at 62c26f8 (rust digest 05d515db); a copy of the routing history
as served that morning (3.5bpw's, carried over, plus an evening of 4bpw chat). The engine served
these twelve requests and nothing else (`engine.txt`). MemAvailable 7.18 GiB after boot, 6.97 after
the runs.

| Workload | tok/s (runs) | Median | tokens a step | decode misses, runs |
| --- | --- | ---: | ---: | --- |
| prose | 22.03 / 23.10 / 22.11 | **22.11** | 2.04 | 4.5 / 3.9 / 3.7% |
| code | 25.70 / 26.36 / 26.49 | **26.36** | 2.76 | 14.5 / 12.9 / 12.5% |
| random 8K, its text | 24.03 / 23.00 / 25.36 | 24.03 | 2.38 | 21.6 / 23.5 / 23.8% |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 12.17 / 12.08 / 12.12 s | **12.12 s** |

Random 8K's runs stopped at the model's end of turn (156 / 184 / 210 tokens): GLM's server takes
`ignore_eos` and does not use it; tok/s is per token decoded either way.

Against 3.5bpw on this box (`results/rust-glm53-flash-35-tp1-20261113/`: 23.29 / 28.91 tok/s, 8,340
resident, decode misses 2.0-2.2% and 6.8-9.0%): prose -5.1%, code -8.8%, across a day and a build.
The same-build pair of 2026-11-13, one frozen routing history (`results/rust-glm53-flash-40-20261113/`),
had 22.17 / 26.97 against 23.43 / 27.48 (-5.4% / -1.8%) and held-out NLL 1.7592 against 1.7510
(+0.0082 +- 0.0027, worse). The README marks this checkpoint recommended by the author's choice from
serving it; that is not what these numbers say, and answers were not scored (docs/glm-flash.md).
