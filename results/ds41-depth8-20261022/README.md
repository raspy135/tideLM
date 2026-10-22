# HTTP benchmark: DS V4.1 with an 8-token depth window, the README row

The run of `results/ds41-draftdense-20261022/` repeated with one setting changed:
`DSV41_BLOCK_DYNAMIC_TOKENS=8` (the depth policy re-decides every 8 output tokens instead of 60),
which the profile now ships. Same image (`sha256:ac93ca3b5c`, `884d7d6`), same reset (seed
history `45ef7129`, empty prefix cache), `MAX_SEQ=32768`, 10,452 residents, the chart's prompt
salts, temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup, one boot.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 26.40 / 25.59 / 25.39 | 25.59 | 1.92 / 1.22 / 1.15 s | 1.220 s |
| Code | 1024 | 43.31 / 42.31 / 43.79 | 43.31 | 1.26 / 1.22 / 1.25 s | 1.245 s |
| Random 8K | 256 | 32.65 / 34.75 / 27.70 | 32.65 | 7.28 / 7.83 / 7.00 s | 7.280 s |

Against the 60-token window on the same code (25.51 prose, 43.94 code, 9.47 s): no measurable
change at 1,024 sampled tokens. Prose's text now differs (acceptance 2.02 / 1.95 / 1.93 against
1.99 / 1.97 / 1.98) and its median moved +0.3%; code took a few more steps (268 / 277 / 261
against 250 / 267 / 256) for -1.4%, inside the code spread between boots; the 8K first token
(7.0-7.8 s) is the low end of the 6.9-11.2 s spread of the two earlier boots, not a prefill
change. The greedy gate's gain (python 61.9 -> 66.8 tok/s at 256 tokens) comes mostly from
leaving depth 3 before token 60, which a 1,024-token sampled request barely notices.

Minimum MemAvailable: 7.59 / 8.11 GB at the 32K allocation.
