# HTTP benchmark: DS V4.1 native vs packed scales, paired

`bench/bench.py`, unchanged, against DS V4.1 Flash 1017 on two Sparks. Both arms
boot the same image (`tidelm:local`, `sha256:ee9aec0786`) with the
`glm-native-perf` checkout bind-mounted over it (`scripts/dual-dev.sh`); the
packed arm adds only `DSV41_SCALE_PACK`, and its loads use the 2026-10-22
`engine/scale_pack.py` (O_DIRECT record reads, checks once per record and process).

Both start from the same demand history (`seed.npz`, sha256 `45ef7129`, on both
nodes) with an empty prefix cache, `MAX_SEQ=32768`, `ARENA_GB=97.4`,
`DSV41_RESIDENT_EXPERTS=auto` (native 10,208 residents, packed 10,452), the
README chart's prompt salts (`flare-perf-http-20261021-{prose,code}`),
temperature 0.6, seed 42, `--ignore-eos`, 1,024 output tokens, three measured
runs after one warmup. Order: native prose, native code, packed prose, packed code.
`env.txt` is the profile (token redacted); `*-engine.txt` the boot's resident lines.

| Workload | Native tok/s | Packed tok/s | Native TTFT | Packed TTFT |
| --- | ---: | ---: | ---: | ---: |
| Prose | 25.61 | 25.82 | 1.223 s | 1.303 s |
| Code | 40.62 | 44.81 | 1.206 s | 1.260 s |

Medians of three. Prose acceptance is equal between the arms; code acceptance is
not (native 3.32-3.77, packed 3.53-4.18). The packed prose runs reproduce the
2026-10-21 packed run's acceptance and NVMe GB exactly
(`results/scale-pack-20261021/matched-http/packed-prose.json`, 23.67 tok/s before
the load-path fix). Analysis: `docs/lossless-scales.md`.
