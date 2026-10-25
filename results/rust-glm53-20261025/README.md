# HTTP benchmark: GLM-5.3 on the Rust engine, the README row

`bench/bench.py`, unchanged, against the Rust server (`rust/`, `tidelm-server` at `5196c33`, the
GLM engine in `rust/crates/tidelm-glm`) on two Sparks, started by `scripts/tide-up.sh` from the
`glm53-exl3` profile with `TIDE_ENGINE=rust` (`env.txt`, the token removed). The README protocol
of the Python rows: one boot, the routing history started from `results/glm-stats/db.pt`
(sha256 `20b9ab41`; the Rust engine's own history file removed first), an empty prefix store,
the chart's prompt salts (`glm-perf-20261021-{prose,code,random8k}`), temperature 0.6, seed 42,
`--ignore-eos`, three measured runs after one warm-up, in the order prose, code, random 8K.

The profile as it ships: 12,500 residents, 256K context, FP8 KV split across the pair, MTP depth
7 under the calibrated gate (`GLM_MTP_CAL=0.36`), copy rounds (`GLM_LOOKUP=8`), adaptation after
each prefill and every 200 tokens, the two-pass prefill, synchronous hot reads kept over
background victims, the disk prefix store. Effort low (the template's), so the tokens include
reasoning.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 16.52 / 17.23 / 17.60 | **17.23** | 0.87 / 0.53 / 0.52 s | 0.527 s |
| Code | 1024 | 20.84 / 20.63 / 20.90 | **20.84** | 0.61 / 0.60 / 0.61 s | 0.611 s |
| Random 8K | 256 | 16.98 / 15.17 / 18.63 | 16.98 | 19.47 / 19.68 / 19.33 s | **19.474 s** |

Against the Python row (`results/glm-native-perf-20261022/`, 12,000 residents, the flat 0.9 gate,
no copy rounds, history `699f3726`): prose 15.20 -> 17.23 tok/s (+13.4%), code 20.34 -> 20.84
(+2.5%), the short prompts' first token 0.731 / 0.772 -> 0.527 / 0.611 s, the 8K prompt's first
token 18.51 -> 19.47 s (+5%: prefill attention, below). Python's later runs at the shipped
profile for reference: 12,500 residents with the `ca` prefill kernel prose 15.61 / code 20.67 /
8K 18.28 s (`results/glm-prefill-ca-20261023/`); the calibrated gate's A/B code 20.24, prose 15.31
(`docs/gotchas.md`). Acceptance (`server.log`): prose 1.05-1.11 drafts a round of 1.7-2.0
drafted, code 1.91-1.96 of 2.4.

Where the time goes (nsys on the pair, `rust/PORT.md`'s GLM section): a one-row decode pass is
74.7 ms (each verify row past the first ~14 ms, an MTP draft 2.8 ms), ~60% of it the EXL3 decode
GEMV shared with the Python engine. The 8K prompt's 19.5 s are GPU-bound: the routed experts'
prefill kernel 5.8 s (the same as Python's), this engine's sparse MLA prefill attention 5.1 s
(Python's Triton kernel did the same picks ~1.8x faster), cuBLAS on the unpacked dense matrices
2 s, NCCL 1.5 s.

Minimum MemAvailable over the run (`memory.csv`): 9.6 / 11.3 GiB (rank 0 / 1).

The prefix store, outside the protocol (same build): a 5,855-token prompt's first token 13.7 s;
re-sent after another request had replaced the live sequence, restored from disk in 0.02 s and
answered identically in 0.49 s; a variant diverging at its last line restored to the divergence
(0.54 s).
