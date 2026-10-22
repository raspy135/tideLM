# DS decode: draft dense TP, parallel router, side-stream accounting (2026-10-22)

Greedy two-node gate (`tools/bench_decode_kernels_tp.py` via `tools/run_two_node_gate.sh`,
`gate_arm.sh` here): the `ds41-mxfp4` profile with frozen residency (`DSV41_DYNAMIC_EXPERTS=0`,
`DSV41_RESIDENT_EXPERTS=` empty, arena 90 GB, keep 0.61), `DSV41_PROBE_PREFILL=0`, 136 transient
slots, image `tidelm:local` (`sha256:b97a181d0c`) with a source snapshot mounted read-only.
Three greedy 256-token workloads, warmed once; `kernels-rank0.json` holds the measured runs
(`runs`) and an 8-step kernel profile of the python workload (`profile`).

| Arm | Source | Extra settings |
| --- | --- | --- |
| A1-base, A2-base | `b46370e` (main before this change) | -- |
| B1-new, B2-new | this change | `DSV41_TP_DRAFT_DENSE=1` |
| C1-win8, C2-win20 | this change | `DSV41_TP_DRAFT_DENSE=1`, `DSV41_BLOCK_DYNAMIC_TOKENS=8` / `20` |
| D1-base-win8 | `b46370e` | `DSV41_BLOCK_DYNAMIC_TOKENS=8` |

Run order A1, B1, B2, A2, C1, C2, D1. Measured ms/step (html / python / explain):

| Arm | html | python | explain |
| --- | ---: | ---: | ---: |
| A1-base | 78.42 | 80.58 | 70.89 |
| B1-new | 76.90 | 79.15 | 70.34 |
| B2-new | 76.90 | 79.29 | 68.02 |
| A2-base | 79.57 | 82.04 | 71.24 |

A and B emit the same tokens (hashes `ddb2ba4b` / `4ad10615` / `4d829123`), steps and
acceptance. C1 and D1 match each other in all six runs; they differ from A/B because the depth
schedule moves hot reads (docs/gotchas.md). `merged-test/` is `tools/test_draft_merged_tp.py`
(bytewise, two ranks): OK on both.

Microbenchmarks, single GB10, serving stopped: `fp8_decode_sweep.py` -> `fp8-decode-sweep.txt`
(every FP8 decode tile, L2-cold, bit-checked against the shipped one); `route_bench.py` (old vs
new `route_small_bm16`: 0 mismatches in 576 cases; 3.8 / 7.9 / 17.1 -> 1.2 / 1.2 / 1.4 us at
15 / 24 / 36 pairs).
