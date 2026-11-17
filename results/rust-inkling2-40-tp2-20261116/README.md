# HTTP benchmark: Inkling-2-Small 4bpw on TWO DGX Sparks (TP=2)

`bench/bench.py`, unchanged, against the Rust server on the pair: the README's protocol (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256). Profile `configs/inkling2-small-40-tp2.env.example` (`profile.env`): 0xSero's
Inkling-2-Small EXL3 4bpw (rev 996dc404), FP8 non-expert weights, 11,865 routed experts resident a
rank (64.4%, `GLM_SLOTS=15800`), 320K context over three lanes, adaptation and async hot reads on,
the DSpark drafter (7 drafts a round, the calibrated gate 0.36, its listed vocabulary) on rank 1, the
vision tower on rank 1. Host build of main at 5c19a81 plus the uncommitted work of 2026-11-16
evening (fused decode attention, image prompts in the prefix store, time-shared prefill, the
drafter's listed vocabulary); booted from the development pair's state file, not the bundled seed.
The engine served these twelve requests and nothing else (`engine.txt`).

| Workload | tok/s (runs) | Median | tokens a step |
| --- | --- | ---: | ---: |
| prose | 28.33 / 28.12 / 27.81 | **28.12** | 1.74 / 1.68 / 1.63 |
| code | 36.64 / 33.70 / 38.60 | **36.64** | 2.25 / 2.39 / 2.10 |
| random 8K, its 256 tokens | 24.58 / 23.25 / 26.04 | 24.58 | 2.39 / 1.52 / 1.51 |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 7.97 / 7.95 / 7.77 s | **7.95 s** |

Against the README's other pairs: GLM-5.3 `home` 20.4 / 23.9 tok/s, 17.2 s (59.5% resident);
GLM-5.3-Flash 4bpw 36.8 / 46.6 tok/s, 7.6 s (every expert resident); DeepSeek-V4.1-Flash MXFP4
33.5 / 51.8 tok/s, 4.3 s. Plain decoding (no drafter, not this protocol) is 24-25 tok/s: about 40 ms
a token for ~5.5 GB a rank of weights (4-bit experts, FP8 dense), roughly half of the pair's read
bandwidth - the first thing to profile.
