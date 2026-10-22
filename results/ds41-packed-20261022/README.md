# HTTP benchmark: DS V4.1 with lossless scales, the README row

`bench/bench.py`, unchanged, against DS V4.1 Flash 1017 on two Sparks: the
`ds41-mxfp4` profile with `DSV41_SCALE_PACK` on and the 2026-10-22 load path
(`engine/scale_pack.py`, O_DIRECT record reads, checks once per record and process).
Image `tidelm:local` (`sha256:ee9aec0786`) with the `glm-native-perf` checkout at
`ccdaf60` bind-mounted over it (`scripts/dual-dev.sh`).

One boot, the same protocol as the 2026-10-21 chart run
(`results/flare-perf-20261021/http-bench/`): the demand history `seed.npz`
(sha256 `45ef7129`) on both nodes, an empty prefix cache, `MAX_SEQ=32768`,
`ARENA_GB=97.4`, `DSV41_RESIDENT_EXPERTS=auto` (10,452 residents), the chart's
prompt salts (`flare-perf-http-20261021-{prose,code,random8k}`) in that order,
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup.
`env.txt` is the profile (token redacted; the pack path was exported by the run
script); `engine.txt` the boot's resident lines; `memory.csv` MemAvailable on both
nodes every 5 s.

| Workload | Output tokens | tok/s (runs) | Median tok/s | Median TTFT |
| --- | ---: | --- | ---: | ---: |
| Prose | 1024 | 25.60 / 25.55 / 25.29 | 25.55 | 1.292 s |
| Code | 1024 | 39.87 / 41.11 / 44.16 | 41.11 | 1.339 s |
| Random 8K | 256 | 28.73 / 36.77 / 30.52 | 30.52 | 8.285 s |

The prose runs repeat the paired A/B's packed arm (`results/scale-pack-ab-20261022/`)
exactly: the same acceptance (1.99 / 1.97 / 1.98) and NVMe traffic (4.65 / 3.77 /
2.85 GB), 25.55 against 25.82 tok/s. Code does not repeat: after an identical warmup
its runs' acceptance differed (3.88 / 3.67 / 4.06 here, 4.10 / 4.18 / 3.53 there), so
its median moves by boot -- 41.11 here, 44.81 in the A/B's packed arm, 40.62 in its
native arm. Random 8K TTFT 8.28 s against the native chart's 8.44 s. Minimum
MemAvailable during the run: 7.78 / 8.40 GB (32K allocation, not the profile's 512K).
