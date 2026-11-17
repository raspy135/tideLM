# HTTP benchmark: Inkling-2-Small 4bpw on TWO DGX Sparks (TP=2), merged small kernels

`bench/bench.py`, unchanged, against the Rust server on the pair: the README's protocol (temperature
0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256). Host build of main at b4bfb08 (TIDELM_INK_FUSE on: the vectorized mix, a decode
pass's convolutions with their ring writes), profile `configs/inkling2-small-40-tp2.env.example`
(`profile.env`): 11,865 routed experts resident a rank (64.4%), 320K context over three lanes,
adaptation and async hot reads on, the DSpark drafter with its listed vocabulary on rank 1, the
vision tower on rank 1. The routing history is the development pair's, as in
`../rust-inkling2-40-tp2-20261116/` (the same protocol about an hour earlier, on the build before the
merged kernels).

| Workload | tok/s (runs) | Median | tokens a step | before the merges |
| --- | --- | ---: | ---: | ---: |
| prose | 29.89 / 29.67 / 29.64 | **29.67** | 1.69 / 1.68 / 1.68 | 28.12 (+5.5%) |
| code | 37.25 / 40.22 / 40.95 | **40.22** | 2.29 / 2.61 / 2.67 | 36.64 (+9.8%) |
| random 8K, its 256 tokens | 27.03 / 29.40 / 28.54 | 28.54 | | 24.58 |

| | runs | Median | before |
| --- | --- | ---: | ---: |
| random 8K, first token | 7.76 / 7.71 / 7.60 s | **7.71 s** | 7.95 s |

The 8K workload ran with `--seed-salt merged-20261116`. With seed 42 alone its prompts are the
earlier run's, and the prefix store restored them (first token 0.27 s; kept as `random8k-cached.*`,
the record of that, not a prefill number). Code's spread is the drafter's: its tokens a step move
with the sampled text (2.29-2.67). The in-boot A/B of the merges alone (one boot, ABBA, fixed text)
is in docs/inkling.md: -8.3% for a one-row pass, -3.7% at eight rows.
