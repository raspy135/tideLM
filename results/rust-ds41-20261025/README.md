# HTTP benchmark: DS V4.1 on the Rust engine, the README row re-measured

`bench/bench.py`, unchanged, against the Rust server (`tidelm-server` built from `5196c33`'s
sources; the branch at `17bf203`) on two Sparks, started by `scripts/tide-up.sh` with no
`TIDE_ENGINE` set: the Rust engine is now its default. The `ds41-mxfp4` profile as it ships
(`env.txt`, the token removed): two 512K lanes (`RUST_LANES=2`, so no async hot reads), packed
KV, the decoder SWA bounded replay, the copy drafter, 10,000 residents on native scales,
depth policy `confidence`. The README protocol: the seed history `45ef7129` on both nodes (a copy
under `~/tidelm-rust-state/bench-20261025`, so the serving history was not touched), an empty
prefix cache, the chart's prompt salts (`flare-perf-http-20261021-{prose,code,random8k}`),
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup, one boot.

Why again: the 2026-10-24 row (`results/rust-ds41-20261024/replay/`, at `7828904`) predates the
GLM-5.3 engine, which changed code both engines share (the server's API and HTTP layers,
`tidelm-cuda`, `tidelm-comm`, the kernels crate's build).

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 32.51 / 31.46 / 31.08 | **31.46** | 0.72 / 0.69 / 0.68 s | 0.692 s |
| Code | 1024 | 47.66 / 47.58 / 46.27 | **47.58** | 0.76 / 1.00 / 0.76 s | 0.763 s |
| Random 8K | 256 | 35.84 / 33.59 / 33.72 | 33.72 | 4.83 / 5.91 / 4.57 s | **4.825 s** |

Against the `7828904` boot: prose 31.78 -> 31.46 tok/s (-1.0%), code 48.95 -> 47.58 (-2.8%), the
8K first token 4.81 -> 4.83 s. Per step (output tokens / steps / tok/s, `x_engine_stats`):

| Boot | Prose: ms a step | Code: ms a step | Code: tokens a step |
| --- | --- | --- | --- |
| `d3835ce`, one lane | 58.0 / 58.5 / 61.4 | 73.4 / 74.8 / 74.7 | 3.50 / 3.88 / 3.80 |
| `7828904`, two lanes | 58.3 / 59.4 / 58.7 | 69.5 / 70.6 / 70.7 | 3.40 / 3.49 / 3.41 |
| this build, two lanes | 58.7 / 58.5 / 58.3 | 73.6 / 75.5 / 76.6 | 3.50 / 3.59 / 3.54 |

Prose, whose drafts stay short, takes the same ~58.5 ms a step on every build: the passes did not
get slower. Code's steps are longer where more tokens land a step, as on the first boot, which
had the same 73-75 ms steps and reached 50.93 tok/s on 3.8-3.9 tokens a step; the code median
moves with acceptance, 47.6-50.9 across the four boots.

Boot: `engine up in 15.4s` (two lanes; 10,000 experts by 13.8 s). Minimum MemAvailable over the
run (`memory.csv`, every 5 s): 6.4 / 7.8 GiB (rank 0 / 1), against the profile's
`KEEP_FREE_GB=5`.
