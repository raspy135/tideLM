# GLM prefill expert GEMM: probes and the cp.async staging (2026-10-23)

One GPU, no engine: `tools/bench_exl3_prefill.py --layer 20 --rows 4096 --experts 160 --topk 8
--iters 3`, rank-0 GLM pack (3-bit, 7.12 MB/slot), incumbent first and arms interleaved per the
plan's protocol. The kernel work and conclusions: `docs/glm-prefill-plan.md` ("Item 5, built and
probed") and the dated entry in `docs/gotchas.md`.

| file | what it ran |
| --- | --- |
| `probes-and-ca-sweep.txt` | the two probes (`nomma:` 15.2 ms, `notrellis:` 28.8 ms vs full 32.8 ms of GEMM) and the first ca stage sweep (S = 3/4/6/8/12) |
| `ca-with-a-prefetch.txt` | ca with A operands prefetched one step ahead in registers: no change (rejected, removed) |
| `final-ab.txt` | the shipped source, interleaved: smem 44.23 / 43.55 vs ca:4x1x8x4 41.75 / 41.26 ms a layer, bitwise equal |

Headline: `GLM_EXL3_PREFILL=ca:4x1x8x4` is bit-identical to `smem:4x1x8x2` and -2 ms a layer
(-4.3% of the MoE, ~0.28 s of the 8K prefill). The probes killed the 34 -> 50 TFLOPS cp.async
estimate: the GEMM's load and mma halves already overlap, and the consume side (A loads + mma +
decoded-tile smem reads) is the floor at 28.8 of 32.8 ms.

## HTTP bench with ca serving (`http/`, one boot)

`bench/bench.py`, the `glm-native-perf-20261022` protocol exactly (same salts, order, seed 42,
temperature 0.6, `--ignore-eos`, 3 runs after one warmup), this profile at 12,500 residents
against that run's 12,000, kernel `ca:4x1x8x4`, boot guard carrying the new `.cu` digest on both
ranks:

| workload | 2026-10-22 (smem) | this run (ca) |
| --- | --- | --- |
| prose | 15.20 tok/s, TTFT 0.731 s | 15.61 tok/s, TTFT 0.743 s |
| code | 20.34 tok/s, TTFT 0.772 s | 20.67 tok/s, TTFT 0.774 s |
| random 8K | 12.59 tok/s, TTFT 18.51 s | 16.49 tok/s, TTFT 18.28 s |

The 8K TTFT moved -0.23 s where the microbenchmark predicts -0.28, but one boot a side cannot
resolve that (gotchas: the column drifts +-0.4 s between boots) — read this run as "nothing
regressed", not as the lever's measurement. Decode is untouched by the kernel (prefill-only) and
sits within boot drift of 10-22; random 8K's decode jump is that workload's known draft-acceptance
swing (0.67-2.31 a round between runs), not an engine effect.
