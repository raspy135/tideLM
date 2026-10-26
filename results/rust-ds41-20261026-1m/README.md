# HTTP benchmark: DS V4.1 at 1M context against 512K

`bench/bench.py`, unchanged, against the Rust server (`tidelm-server` at `078db1a`, the paged
context caches) on two Sparks, started by `scripts/tide.sh` from the `ds41-mxfp4` profile
(`env.txt`, the token removed). The README protocol (`results/rust-ds41-20261024/`): the seed
history (sha256 `45ef7129`) on both nodes, an empty prefix cache, the chart's prompt salts
(`flare-perf-http-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`,
three measured runs after one warmup, one boot an arm. 10,000 residents, two lanes, async hot
reads, the bounded replay, the copy drafter, the abliteration overlay.

- **1M** (this directory): `MAX_SEQ=1048576`, `RUST_KV_TOKENS=1572864` - one lane can reach 1M
  while the other holds 512K, out of a pool of 1,044 chunks (2.19 GB a rank).
- **512K** (`512k/`): `MAX_SEQ=524288`, no pool limit - every lane can reach 512K at once (704
  chunks, 1.48 GB a rank).

| Workload | 1M: tok/s (runs) | Median | 512K: tok/s (runs) | Median |
| --- | --- | ---: | --- | ---: |
| Prose | 31.88 / 31.24 / 30.35 | **31.24** | 32.17 / 30.39 / 32.42 | **32.17** |
| Code | 48.36 / 50.83 / 51.67 | **50.83** | 48.23 / 49.33 / 47.02 | **48.23** |
| Random 8K, first token | 4.33 / 4.29 / 4.29 s | **4.29 s** | 4.32 / 4.30 / 4.30 s | **4.30 s** |

The same speed: each arm wins one decode workload (acceptance, tokens a step past the first:
prose 1.87 / 1.86 / 1.77 against 1.87 / 1.78 / 1.90, code 3.58 / 3.75 / 3.82 against 3.41 /
3.61 / 3.39), and the first token matches to 0.01 s. Against the README row (31.8 / 49.0 tok/s,
4.8 s): decode is within 1.8 tok/s of it either way, so the row keeps its numbers. The 8K first
token is 0.5 s under the row in both arms, so whatever moved it is not the pool.

Memory: device 103.51 GiB a rank at 1M against 102.60 at 512K (the pool's 0.71 GB and RoPE tables
for 1M positions). Minimum MemAvailable over each arm's runs: 6.47 / 7.26 GiB (rank 0 / 1) at 1M,
7.75 / 7.49 at 512K, against the profile's `KEEP_FREE_GB=4`.

Not measured here: a request past 512K tokens. These are short prompts; what the 1M pool buys is
room, not speed.
