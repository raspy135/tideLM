# HTTP benchmark: DS V4.1 after the decode bandwidth change, the README row

`bench/bench.py`, unchanged, against the Rust server (image `sha256:408c8abf`, `tidelm-server` at
`b19fa0b`: the GEMVs with four tiles in flight and the L2 prefetch beside the RoCE waits) on two
Sparks, started by `scripts/tide.sh` from the `ds41-mxfp4` profile (`env.txt`, the token
removed). The README protocol (`results/rust-ds41-20261024/`): the seed history (sha256
`45ef7129`) on both nodes, an empty prefix cache, the chart's prompt salts
(`flare-perf-http-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`,
three measured runs after one warmup, one boot. Rank 0 writes its demand history back, so its
copies of `45ef7129` have all moved on; this one came from rank 1's, which it only reads.
10,000 residents, two lanes of up to 1M out of a 1.5M pool, depth policy `confidence`.
`engine.txt` is rank 0's whole log of the boot.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 32.39 / 32.66 / 33.18 | **32.66** | 0.53 / 0.52 / 0.52 s | 0.524 s |
| Code | 1024 | 50.14 / 49.45 / 51.02 | **50.14** | 0.56 / 0.56 / 0.55 s | 0.562 s |
| Random 8K | 256 | 45.33 / 42.98 / 38.41 | 42.98 | 4.36 / 4.31 / 5.54 s | **4.362 s** |

Acceptance (`accept_len`): prose 1.79 / 1.78 / 1.83, code 3.76 / 3.37 / 3.55.

Against the row it replaces (31.8 / 49.0 tok/s, 4.8 s; `results/rust-ds41-20261024/replay/`):
prose +2.7%, code +2.3%. One boot here against one there, so this is inside what boots alone
move (the 1M check's two arms: prose 31.24 / 32.17, code 50.83 / 48.23). The change's own
measurements are in `b19fa0b`'s message: the pair at fixed residents and greedy, logits
unchanged, 30.4 -> 32.6 tok/s prose and 44.6 -> 46.4 code under the confidence policy; and
the server, one boot an arm from the same history copy on the same day, 31.88 -> 33.36 prose
and 49.35 -> 50.17 code.

The 8K prompt's first token, 4.8 -> 4.36 s, is not this change: the 1M check measured 4.29 s
two days before it (`results/rust-ds41-20261026-1m/`). The third run's 5.54 s is the slow group
`results/rust-ds41-20261024/` describes.

Memory: device 103.51 GiB a rank at boot, as before the change. Minimum MemAvailable over the
runs (sampled every 2 s): 6.20 / 6.93 GiB (rank 0 / 1), against the profile's `KEEP_FREE_GB=4`.
