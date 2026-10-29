# HTTP benchmark: GLM-5.3 decode with EXL3 loads through shared memory, L2 prefetch and PDL

`bench/bench.py`, unchanged, against the Rust server on two Sparks, started by `scripts/tide.sh`
from the local `glm53-exl3` profile (each arm's `env.txt`, the token removed; the example config's
256K and `GLM_SLOTS=12500` raised to 320K and 12200 here, two lanes): 12,968 residents from this
history, async hot reads, adaptation, the disk prefix store. Each boot from a fresh routing history (`results/glm-stats/db.pt`, the Rust
engine's own history file absent) and an empty prefix store; the chart's prompt salts
(`glm-perf-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`, three measured
runs after one warmup (prose and code 1,024 tokens, random 8K 256). One boot an arm, the arms
alternating: base, this build, base, this build. Both binaries built on the host (`RUST_BIN`):
`base` the tree at e2a2e47 (with only the NLL bench's new options), the other the change committed
with this directory.

| Arm (order) | Prose tok/s (runs) | Median | Code tok/s (runs) | Median | 8K first token |
| --- | --- | ---: | --- | ---: | ---: |
| `base-r1` (1st) | 17.83 / 18.22 / 18.94 | 18.22 | 22.13 / 22.34 / 21.32 | 22.13 | 19.48 s |
| `p1-r1` (2nd) | 19.24 / 19.79 / 20.51 | **19.79** | 23.74 / 23.92 / 22.86 | **23.74** | 19.49 s |
| `base-r2` (3rd) | 17.79 / 18.25 / 18.98 | 18.25 | 22.15 / 22.31 / 21.37 | 22.15 | 19.76 s |
| `p1-r2` (4th) | 19.19 / 19.59 / 19.66 | 19.59 | 22.63 / 22.79 / 21.88 | 22.63 | 20.23 s |

Tokens a step (`accept_len`) are the same in every arm, run for run (prose 2.09 / 2.09 / 2.16, code
2.92 / 2.93 / 2.72, the 8K prompt's 256 tokens 2.46 / 2.36 / 2.32): the same tokens came out of
both builds. Step time is that over tok/s:

| ms a step | Base (6 runs) | This build (6 runs) | |
| --- | --- | --- | ---: |
| Prose | 115.3 (113.9-117.6) | 107.5 (105.5-110.0) | -6.8% |
| Code | 130.3 (127.3-132.1) | 124.5 (119.0-129.2) | -4.4% |
| 8K prompt's decode | 128.1 (124.7-132.7) | 121.1 (115.4-128.8) | -5.5% |

**Decode +5-8% in tok/s, all of it step time.** Prose 18.22 / 18.25 -> 19.79 / 19.59 tok/s, code
22.13 / 22.15 -> 23.74 / 22.63. The base repeats within 0.5% between its boots; this build's second
boot ran 1-5% slower than its first (boot to boot, as `--bench-width` sees on one binary). The 8K
prompt's first token is unchanged within the spread (19.48 / 19.76 -> 19.49 / 20.23 s): prefill
takes the cuBLAS path these changes leave alone. Memory at boot: device 101.57 -> 101.35 GiB a rank
(the routers in fp16).
