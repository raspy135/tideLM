# HTTP benchmark: native GLM-5.3 EXL3 with the 2026-10-22 fixes

`bench/bench.py`, unchanged, against GLM-5.3 on two Sparks: the native EXL3 path
with merged MoE gathers, grouped shared-expert gate/up, decode-kernel prefill
rotations and FP8 kv_b (`GLM_KVB_FP8=1`). Three measured requests after one
warm-up per workload, temperature 0.6, seed 42, `--ignore-eos`.

Matched to the 2026-10-21 native run (`results/glm-perf-20261021/native-exl3/http/`,
local): the same image (`sha256:daca374855`), the same starting demand history
(`results/glm-stats/db.pt`, sha256 `699f3726`), the same prompt salts
(`glm-perf-20261021-{prose,code,random8k}`), the same order (prose, code,
random 8K), 12,000 residents, 256K context, an empty prefix store. `env.txt` is the
complete profile; it differs from that run only in `GLM_KVB_FP8`.

| Workload | Output tokens | Native 10-21 tok/s | This run tok/s | Native TTFT | This run TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 1024 | 14.20 | 15.20 | 0.754 s | 0.731 s |
| Code | 1024 | 17.86 | 20.34 | 0.845 s | 0.772 s |
| Random 8K | 256 | 12.23 | 12.59 | 25.47 s | 18.51 s |

Medians of three. Acceptance-normalized round time (tokens a round / tok/s):
prose 138.8 -> 125.0 ms, code 190.0 -> 167.9 ms. Random 8K drafted 0.67-0.97 a
round here against 0.79-2.31, so its decode figure compares acceptance more than
code. Minimum MemAvailable during the run (`memory.csv`): 9.19 / 13.16 GB.
Analysis: `docs/glm53-native-exl3.md`.
