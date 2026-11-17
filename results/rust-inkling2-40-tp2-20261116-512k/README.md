# HTTP benchmark: Inkling-2-Small 4bpw on TWO DGX Sparks (TP=2), FP8 cache, 512K

`bench/bench.py`, unchanged, against the Rust server on the pair: the README's protocol (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256, its prompts under `--seed-salt 512k-20261116` so no earlier run's are in the prefix
store). Host build of 061eaa9 (the FP8 key / value cache, GLM_INK_KV=fp8), profile
`configs/inkling2-small-40-tp2.env.example` (`profile.env`): 11,967 routed experts resident a rank
(64.9%, `GLM_SLOTS=15935`), a 512K context (524,288 positions over three lanes, 4.43 GB a rank),
adaptation and async hot reads on, the DSpark drafter with its listed vocabulary and the vision tower
on rank 1, the merged small kernels on. The routing history is the development pair's state file
(seed v1 is that file the next morning, after a night more of the author's traffic).

| Workload | tok/s (runs) | Median | tokens a step | the merged run (BF16 cache, 320K, 11,865) |
| --- | --- | ---: | ---: | ---: |
| prose | 29.56 / 29.41 / 28.49 | **29.41** | 1.77 / 1.70 / 1.69 | 29.67 |
| code | 48.50 / 41.06 / 42.29 | **42.29** | 3.56 / 2.81 / 2.81 | 40.22 |
| random 8K, its 256 tokens | 28.33 / 29.65 / 28.55 | 28.55 | | 28.54 |

| | runs | Median | the merged run |
| --- | --- | ---: | ---: |
| random 8K, first token | 7.83 / 7.63 / 7.57 s | **7.63 s** | 7.71 s |

At these contexts (60 - 8K tokens) the cache is a small part of a pass's reads, so FP8 changes little
here; prose is within the runs' spread, and code's first run took 3.56 tokens a step against 2.81 for
the next two (the sampled text, through the drafter). What FP8 buys is memory: 512K in 4.43 GB a
rank where 320K took 5.37 in BF16, with 102 more experts resident.
