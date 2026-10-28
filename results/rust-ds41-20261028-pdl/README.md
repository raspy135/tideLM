# HTTP benchmark: DS V4.1 with programmatic dependent launch, on and off

`bench/bench.py`, unchanged, against the Rust server (image `sha256:71d6157b`, `tidelm-server` at
`0b4c10f`) on two Sparks, started by `scripts/tide.sh` from the `ds41-mxfp4` profile (each arm's
`env.txt`, the token removed). The README protocol (`results/rust-ds41-20261024/`): the seed history
(sha256 `45ef7129`, rank 1's copy) on both nodes, an empty prefix cache, the chart's prompt salts,
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup, one boot an arm. The
same image in every arm; `TIDELM_PDL=0` in the off arms' profile. Two rounds, the second in the
reverse order.

| Arm (order) | Prose tok/s (runs) | Median | Code tok/s (runs) | Median | 8K first token |
| --- | --- | ---: | --- | ---: | ---: |
| `off/` (1st, right after the image load) | 29.72 / 29.09 / 30.15 | 29.72 | 39.02 / 36.38 / 42.92 | 39.02 | 4.28 s |
| `on/` (2nd) | 33.83 / 33.49 / 32.58 | **33.49** | 54.28 / 51.77 / 48.46 | **51.77** | 4.31 s |
| `on2/` (3rd) | 33.85 / 34.66 / 33.34 | 33.85 | 54.22 / 54.90 / 52.85 | 54.22 | 4.27 s |
| `off2/` (4th) | 33.32 / 32.39 / 33.49 | 33.32 | 50.09 / 48.97 / 52.61 | 50.09 | 4.31 s |

Each run's step time is its tokens a step (`accept_len`) over its tok/s:

| Arm | Prose: tokens a step | ms a step | Code: tokens a step | ms a step |
| --- | --- | --- | --- | --- |
| `off` | 1.63 / 1.74 / 1.59 | 54.9 / 59.7 / 52.7 | 2.25 / 2.05 / 2.64 | 57.8 / 56.2 / 61.4 |
| `on` | 1.82 / 1.81 / 1.74 | 53.8 / 54.1 / 53.5 | 3.80 / 3.56 / 3.20 | 70.1 / 68.8 / 66.0 |
| `on2` | 1.82 / 1.88 / 1.82 | 53.9 / 54.1 / 54.7 | 3.77 / 3.93 / 3.59 | 69.6 / 71.7 / 67.9 |
| `off2` | 1.88 / 1.79 / 1.85 | 56.3 / 55.2 / 55.2 | 3.44 / 3.30 / 3.71 | 68.8 / 67.4 / 70.5 |

**PDL in serving: ~1.6 ms off a step.** Prose, where tokens a step barely move between arms, takes
53.5-54.7 ms a step with PDL and 55.2-56.3 without it (`off2`), the pair's 1.2-2.1 ms. In tok/s:
prose 33.49 / 33.85 against 33.32, and 32.66 on `b19fa0b`, which has no PDL code
(`results/rust-ds41-20261028/`); code 51.77 / 54.22 against 50.09 (50.14), where acceptance moves
more between boots than PDL does (deeper steps cost more, so code's step times do not compare
across arms). The 8K prompt's first token is unchanged, 4.27-4.31 s.

**The first arm is not PDL's.** Its prose steps took as long as `off2`'s (52.7-59.7 ms, mean 55.8,
against 55.6), but it emitted fewer tokens a step, code most of all: 2.05-2.64 against 3.20-3.93 in
every other boot here and in `results/rust-ds41-20261028/` (prose 1.59-1.74 against 1.74-1.88).
It was the first boot after the 10 GB image load on both boxes; why its acceptance fell is not
found.

Memory: device 103.51 GiB a rank in every arm. Minimum MemAvailable over each arm's runs (rank 0 /
1): `off` 5.95 / 6.42 GiB, `on` 6.31 / 6.72, `on2` 6.31 / 6.98, `off2` 6.34 / 6.72, against the
profile's `KEEP_FREE_GB=4`.
