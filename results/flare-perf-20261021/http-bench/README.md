# HTTP benchmark: candidate indexing off/on

Ran the existing `bench/bench.py` unchanged against native DS V4.1 Flash 1017 on
two Sparks, from source commit `59524f7`. Three measured requests after one warm-up
per workload and configuration: 32 total requests, 24 retained measured results.
Temperature 0.6, top-p 0.95, seed 42, fixed output length with `--ignore-eos`.

The `.env.ds41-mxfp4` serving profile keeps adaptive loading, predictive prefill,
hot expert reads, prefix caching, and adaptive speculative depths 3/5 enabled.
Both servers use a 32768-token context cap, a 97.4 GB arena per rank and exactly
the same initial demand history and resident expert map (10208 experts).
Separate source/results snapshots isolate updates from the user's saved history.
Only `DSV41_INDEX_CANDIDATES` differs; other experimental switches stay off.
The order is one baseline server followed by one candidate server, not ABBA.

| Workload | Input tokens | Output tokens | Baseline tok/s | Candidate tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 45 | 1024 | 26.12 | 25.98 | -0.55% |
| Code | 62 | 1024 | 43.75 | 44.22 | +1.07% |
| Random 8K | 8171–8178 | 256 | 31.59 | 33.47 | +5.97% |
| Random 24K | 24582–24617 | 256 | 29.77 | 32.58 | +9.44% |

All values are medians over three measured requests. Corresponding prompt lengths
and prompt seed salts match between configurations. All measured requests return
their full output count, `finish_reason=length`, and nonempty engine statistics.

| Workload | Baseline TTFT ms | Candidate TTFT ms | Baseline acceptance | Candidate acceptance | Baseline NVMe GB | Candidate NVMe GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 2234.51 | 1315.33 | 1.97 | 1.97 | 5.38 | 5.38 |
| Code | 1216.52 | 1229.09 | 3.85 | 3.89 | 3.12 | 3.35 |
| Random 8K | 9637.34 | 8440.25 | 2.75 | 3.10 | 4.19 | 4.85 |
| Random 24K | 18801.37 | 19518.28 | 2.62 | 3.11 | 3.55 | 3.31 |

These are observed serving rates, not isolated kernel speedups. Prose/code are
effectively flat; their short contexts do not use candidate scoring. The long
prompt medians improve, but acceptance and adaptive depth also differ. In
particular, 24K measured rates span 29.67–31.64 off and 31.49–35.97 on; acceptance
medians rise 2.62 -> 3.11. Do not attribute the full 9.44% to the scorer.
The preceding fixed-depth, matching-output greedy ABBA result (2.39% less decode
time at 24596 input tokens) remains the cleaner evidence for its mechanical gain.
Indexing does not change prefill; TTFT differences here are not a prefill speedup
claim. This harness does not retain generated text/hashes or assess quality.

Representative invocation (repeat with workload-specific salt/length and arm):

```sh
python3 bench/bench.py --workload prose --runs 3 --warmup 1 --osl 1024 \
  --temperature 0.6 --seed 42 --seed-salt flare-perf-http-20261021-prose \
  --ignore-eos --label candidate-off-prose --out /path/to/off-prose.json
```

Use `code` with 1024 output tokens, and `random --isl 8192` / `random --isl 24576`
with 256 output tokens. The other salts end in `code`, `random8k`, `random24k`;
each salt is identical between off/on runs. Server runtime image, source and
demand hashes are in `manifest.json`; raw rows retain health and engine stats.
Warm-up measurements remain in the workload logs. Rank logs retain boot guards
and adaptation activity.

Both temporary servers were removed, Flare GLM serving was restored, and the
original demand-history and benchmark-script SHA-256 values remain unchanged.
No serving profile was edited; all experiments still default off.
