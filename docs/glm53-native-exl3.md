# Native GLM-5.3 EXL3 weights

The EXL3 profile now takes all weights from `GLM_MODEL_DIR`, pointing to the
complete `GLM-5.3-EXL3-3.0bpw` checkpoint. Routed decoder experts still use the
existing per-rank packs. No Quark checkpoint is required, and the full EXL3
checkpoint must remain on both nodes.

Attention projections, dense/shared MLPs, the published 6-bit output head and
the MTP projection use the checkpoint's EXL3 trellises. MTP routed experts use
their published 4-bit records in a separate resident arena. Norms, routers,
embeddings and absorbed MLA weights come from the same checkpoint. The profile
still converts its own embedding to INT8 at runtime, and quantizes the FP16 kv_b
to FP8 at load (`GLM_KVB_FP8=1`, below). Set `GLM_HEAD=exl3`.

Existing complete routed packs already contain the MTP records; no new packing
step is needed. A source-manifest check rejects routed packs from another
checkpoint. The rank guard includes the weight source and implementation.

## Measurements

Two Sparks, serving-image Torch/CUDA, `bench/bench.py`, three measured requests
after one warmup, temperature 0.6, seed 42, adaptation/hot reads/prefix enabled,
256K context allocation. Prose/code request 1024 output tokens; random 8K
requests 256, with the engine's stop handling still active. Both arms start
from copies of the same saved demand history. The old hybrid had 12,400
residents; native EXL3 was tested at 12,000, so this is an observed profile
comparison, not an isolated weight-format speedup or quality evaluation.

| Workload | Old hybrid decode | Native EXL3 decode | Old TTFT | Native TTFT |
| --- | ---: | ---: | ---: | ---: |
| Prose | 13.6792 tok/s | 14.1971 tok/s | 0.697 s | 0.754 s |
| Code | 18.0229 tok/s | 17.8609 tok/s | 0.792 s | 0.845 s |
| Random 8K | 13.9442 tok/s | 12.2283 tok/s | 21.971 s | 25.473 s |

Do not describe this as a general speed improvement. That first native version
made four RoCE gathers per MoE layer, read kv_b in BF16 and rotated prefill
activations with fp32 matmuls; the next section removes those. The native source
still has no q_a/kv_a fusion (the FP8 fusion measured ~1 ms a round). Different
weight values also change drafts, acceptance and adaptation.

Native EXL3's fixed/dense/MTP tensors total 10.32 GB per rank, plus a 0.01 GB
bookkeeping arena. The older hybrid reports 8.42 GB dense plus about 3.51 GB
of fixed MXFP4 slots: roughly 1.6 GB more in total before decoder routed slots.
Memory counters group these allocations differently, so comparing only the
`dense` field reverses the result. At 12,000 residents the normal HTTP run's
minimum host MemAvailable was 8.59/12.62 GB on the two nodes.

Raw local reports: `results/glm-perf-20261021/baseline-*.json` and
`results/glm-perf-20261021/native-exl3/http/baseline-*.json`.

## What the native path lost against the hybrid, and the fixes (2026-10-22)

The old hybrid profile never set `GLM_EXL3_ATTN`: it ran Quark's FP8 attention
with q_a/kv_a fused, an FP8 head and FP8 kv_b. The native path changed all of
them at once. Microbenchmarks, one GB10, rank-0 shapes, real weights, CUDA graph
replays (decode), L2-cold over 16 layers (same date):

| Change | Before | After | Per verify forward |
| --- | --- | --- | --- |
| MoE layer gathers (`exl3_model.mlp`) | 4, two of them fp32 | 2, as `gather_merge` | ~2-5 ms (2 x 76 gathers at the measured 13-34 us; not measurable on one node) |
| kv_b absorbs, q + o, 4 rows (`GLM_KVB_FP8=1`) | 73.0 us a layer | 44.5 | ~2.2 ms |
| shared expert gate + up, 4 rows (G=2 `Exl3Matrix`) | 69.2 us a layer | 61.5 | ~0.6 ms |
| prefill, 6 projections a layer, 4,096 rows | 38.5 ms | 15.7 | ~2 s a 4K chunk (estimate) |

The gather merge and the grouped gate/up are bitwise equal to the code they
replace (checked at 1/4/8/16/17/64/300 rows; dense, shared and MTP MLPs). FP8
kv_b re-quantized from the published FP16 has Quark's scales and 99.9% of its
bytes, so it is the configuration whose quality was measured on 2026-10-17.
Prefill now rotates with decode's FWHT kernels: error against the fp64
reference unchanged (1.672e-3 relative either way), 99.3-99.5% of bf16 outputs
bit-identical, the rest one ulp.

Not a regression: the published 6-bit head is faster than the FP8 head it
replaced, 1.56 vs 2.24 ms a call at 1-8 rows (357 vs 476 MB a rank).

Serving, same protocol as the table above: same image (`daca374855`), same
starting demand history (`699f3726`), same salts, 12,000 residents, the native
env plus `GLM_KVB_FP8=1` (2026-10-22):

| Workload | Hybrid | Native | Native + fixes | Native TTFT | + fixes TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 13.68 tok/s | 14.20 | **15.20** | 0.754 s | 0.731 s |
| Code | 18.02 tok/s | 17.86 | **20.34** | 0.845 s | 0.772 s |
| Random 8K | 13.94 tok/s | 12.23 | 12.59 | 25.473 s | **18.507 s** |

Acceptance differs between arms, so the cleaner decode figure is time a round
(tokens a round / tok/s, median of 3): prose 147.9 / 138.8 / **125.0 ms**,
code 194.8 / 190.0 / **167.9 ms** (hybrid / native / fixes), drafted per round
alike in all three. That is 10-12% off the native round, more than the
components above add up to; this run does not separate them. Random 8K drafted
0.67-0.97 a round against native's 0.79-2.31, so its tok/s and round time
compare acceptance, not code. Its prefill fell 27% and is now 16% below the
hybrid's. Dense weights 10.32 -> 9.76 GB a rank; minimum MemAvailable during the
run 9.19 / 13.16 GB (native: 8.59 / 12.62). Raw reports:
`results/glm-native-perf-20261022/`.

## Experiments retained off

`GLM_INDEX_BUCKETS=1` bounds decode indexing by the occupied power-of-two
context bucket and keys graphs by that bound. It matched all teacher-forced
logits at 96 checked positions and all greedy output hashes on both ranks,
but complete-round timings did not show a consistent improvement. It remains
off. Lower MTP confidence thresholds also failed to improve consistently;
`GLM_MTP_DYN=0.9` remains the profile setting. 
