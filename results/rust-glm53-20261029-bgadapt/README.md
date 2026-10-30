# HTTP benchmark: GLM-5.3 decode-time adaptation in the background

The protocol of `results/rust-glm53-20261029/` unchanged (`bench/bench.py`, the local 320K two-lane
`glm53-exl3` profile, a fresh routing history from `results/glm-stats/db.pt` and an empty prefix
store each boot, the chart's salts, temperature 0.6, seed 42, `--ignore-eos`, three runs after a
warmup), one boot an arm, alternating: `p2` is fbdd6e4, `bga` the same with the decode-time
adaptation's newcomers loaded on reader threads (`adapt_background`). `engine.txt` keeps each
boot's request lines (swaps and their wait, miss rates).

| Arm (order) | Prose tok/s (runs) | Median | Code tok/s (runs) | Median | 8K first token |
| --- | --- | ---: | --- | ---: | ---: |
| `p2-r1` (1st) | 18.81 / 19.33 / 20.19 | 19.33 | 23.19 / 23.39 / 22.44 | 23.19 | 20.24 s |
| `bga-r1` (2nd) | 19.48 / 19.53 / 19.47 | 19.48 | 23.49 / 23.55 / 22.82 | 23.49 | 20.07 s |
| `p2-r2` (3rd) | 19.03 / 19.50 / 20.40 | 19.50 | 23.45 / 23.63 / 22.89 | 23.45 | 19.45 s |
| `bga-r2` (4th) | 20.37 / 20.60 / 20.51 | 20.51 | 24.63 / 24.82 / 23.99 | 24.63 | 19.45 s |

Over the six runs of each: prose 19.54 -> 19.99 tok/s (+2.3%), code 23.17 -> 23.89 (+3.1%), the 8K
prompt's 256 tokens 20.20 -> 20.43; ms a step prose 108.2 -> 107.0, code 123.4 -> 120.9. Tokens a
step are no longer identical run for run (the newcomers route six passes later): prose 2.114 ->
2.137, code 2.858 -> 2.885, 8K 2.380 -> 2.259. Decode miss rates request by request 4.3 / 3.9 /
16.8 / 11.5 / 9.9 / 9.0 / 20.3 / 18.5 / 18.5% -> 4.3 / 4.2 / 16.4 / 11.7 / 10.4 / 8.9 / 20.6 /
16.3 / 18.4%. No decoder pass waited for a background read (`swaps N (0.00 s waited)` in every
request line).
