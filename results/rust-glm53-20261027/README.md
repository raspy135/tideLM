# HTTP benchmark: GLM-5.3 with Mia's AI Lab's 2-4-bit experts, the README row

The protocol of `results/rust-glm53-20261025/` unchanged: `bench/bench.py` against the Rust
server on two Sparks, one boot, the routing history started from `results/glm-stats/db.pt` (the
Rust engine's own history file absent), an empty prefix store, the chart's prompt salts
(`glm-perf-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`, three
measured runs after one warm-up, in the order prose, code, random 8K (256 tokens out).
MemAvailable sampled every 5 s on both ranks (`memory.csv`).

The profile as it ships (`env.txt`, the paths generalized): the routed experts of
`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` packed per rank, the non-expert weights of
`bertholomus/GLM-5.3-EXL3-3.0bpw` (`tools/glm_fetch_dense.py`), `GLM_SLOTS=12500` 3-bit experts'
worth of memory = 13,326 residents from this history (2/2/2 x1,624, 2/2/3 x1,832, 3/3/3 x9,382, 3/3/4 x202, 4/4/4
x286), 256K context, FP8 KV split across the pair, MTP depth 7 under the calibrated gate, long
copies and the lookup's, synchronous hot reads kept over background victims.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 17.08 / 17.01 / 17.83 | **17.08** | 0.84 / 0.56 / 0.55 s | 0.560 s |
| Code | 1024 | 20.81 / 21.54 / 21.08 | **21.08** | 0.64 / 0.63 / 0.65 s | 0.640 s |
| Random 8K | 256 | 14.38 / 15.51 / 14.83 | 14.83 | 20.84 / 20.37 / 20.17 s | **20.374 s** |

Minimum MemAvailable: 9.1 / 10.7 GiB (rank 0 / 1).

Against the 3.0bpw row (2026-10-25: 17.23 / 20.84 tok/s, 19.47 s): decode at parity. The first
token's +0.9 s is mostly not the pack: the 3.0bpw pack on this build, same protocol, 19.93 / 20.11
/ 20.16 s (`random8k-3.0bpw.log`), so the pack costs +0.26 s (within the runs' spread) and this
build's prefill +0.64 s over 2026-10-25's on the same pack (not bisected). What the pack buys is in
`docs/gotchas.md` (2026-10-27): 13,320 residents for 12,500 on this server's history, routed weight missed 15-25% less,
held-out NLL -0.0076 +- 0.0049.
