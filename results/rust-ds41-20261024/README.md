# HTTP benchmark: DS V4.1 on the Rust engine, the README row

`bench/bench.py`, unchanged, against the Rust server (`rust/`, `tidelm-server` at `d3835ce`)
on two Sparks, started by `scripts/rust-up.sh` from the `ds41-mxfp4` profile (`env.txt`, the
token removed). The README protocol of the Python rows: the seed history `seed.npz` (sha256
`45ef7129`) on both nodes, an empty prefix cache, the chart's prompt salts
(`flare-perf-http-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`,
three measured runs after one warmup, one boot. The Rust profile: 10,000 residents (native
scales: the lossless scale pack is not ported, so the Python engine's 10,452 do not fit in the
same memory), `MAX_SEQ=524288` (the profile's; the Python row was measured at a 32K allocation),
packed KV, adaptation with the probe, the boundary and decode passes and async hot reads,
the RAM and disk prefix cache, vision on (rank 1 holds the tower), depth policy `confidence`.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 30.20 / 30.88 / 30.94 | **30.88** | 0.70 / 0.91 / 0.68 s | 0.702 s |
| Code | 1024 | 47.79 / 51.88 / 50.93 | **50.93** | 0.74 / 0.74 / 1.10 s | 0.743 s |
| Random 8K | 256 | 36.03 / 37.94 / 41.71 | 37.94 | 9.30 / 9.88 / 9.29 s | **9.304 s** |

Against the Python row (`results/ds41-depth8-20261022/`): prose 25.59 -> 30.88 tok/s (+20.7%),
code 43.31 -> 50.93 (+17.6%), the short prompts' first token 1.22 / 1.25 -> 0.70 / 0.74 s, and
the 8K prompt's first token 7.28 -> 9.30 s. Acceptance (tokens a step past the first): prose
1.75 / 1.80 / 1.90, code 3.50 / 3.88 / 3.80.

**A second boot** (`boot2/`), the same reset and protocol, to see the spread rather than to
choose: prose 30.68 / 30.57 / 30.73 (median 30.68), code 52.11 / 50.07 / 49.82 (50.07), the 8K
first token 7.66 / 7.97 / 9.32 s (7.97). Decode repeats within 2%; the 8K first token falls
in two groups, ~7.6 s and ~9.3 s, across both boots' six runs (median of the six: 9.30 s),
the way the Python engine's moved with its Engram phase (6.9-11.2 s across runs). What decides
the group is not found. Not the cause, each from one fresh boot running only 8K prompts:
- the prefix cache at 512K: off 7.56 / 7.52 s, on 7.71 / 7.51 s
- the 512K allocation: at 64K 7.58 / 7.53 s
- the prose and code requests before it: after one of each on the same boot 7.55 / 7.59 s
Packed KV costs 0.3 s of it at 64K (bf16 KV 7.34 / 7.16 s), and bf16 KV does not fit at 512K
(the memory guard stopped rank 0 at 5 GiB available during the first 8K prefill).

Minimum MemAvailable over the second boot: 6 / 7 GiB (rank 0 / 1), against the profile's
`KEEP_FREE_GB=6`.

## The shipped profile after the bounded replay (`replay/`)

The same reset and protocol (seed history `45ef7129`, empty prefix cache, the chart's salts,
temperature 0.6, seed 42, `--ignore-eos`, three runs after a warmup, one boot) on the profile as
it ships after the day's changes: two lanes (`DSV41_MAX_CONCURRENCY=2`, 512K each, so no async
hot reads), the decoder SWA bounded replay (prompt rows run layers 0-20; layers 20-39 replay the
prompt's last 128 rows), the copy drafter, 10,000 residents.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 31.09 / 31.88 / 31.78 | **31.78** | 0.70 / 0.70 / 0.68 s | 0.700 s |
| Code | 1024 | 48.95 / 49.50 / 48.27 | **48.95** | 0.74 / 0.71 / 0.74 s | 0.737 s |
| Random 8K | 256 | 34.23 / 35.24 / 35.59 | 35.24 | 6.99 / 4.78 / 4.81 s | **4.807 s** |

Against the first boot above: prose +2.9%, code -3.9% (inside the two boots' 50.07-50.93 and the
acceptance it moves with: 3.39 / 3.49 / 3.41 here; no copy step fires on these prompts), the 8K
first token 9.30 -> 4.81 s (Python 7.28).
