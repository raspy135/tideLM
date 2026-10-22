# Native GLM-5.3 EXL3 weights

The EXL3 profile now takes all weights from `GLM_MODEL_DIR`, pointing to the
complete `GLM-5.3-EXL3-3.0bpw` checkpoint. Routed decoder experts still use the
existing per-rank packs. No Quark checkpoint is required, and the full EXL3
checkpoint must remain on both nodes.

Attention projections, dense/shared MLPs, the published 6-bit output head and
the MTP projection use the checkpoint's EXL3 trellises. MTP routed experts use
their published 4-bit records in a separate resident arena. Norms, routers,
embeddings and absorbed MLA weights come from the same checkpoint. The profile
still converts its own embedding to INT8 at runtime; FP16 kv_b uses the engine's
BF16 MLA path (`GLM_KVB_FP8=0`). Set `GLM_HEAD=exl3`.

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

Do not describe this as a general speed improvement. The native fixed MLP path
uses additional gathers, and the native source disables Quark-specific fused
projections. Those are future optimization opportunities. Different weight
values also change drafts, acceptance and adaptation.

Native EXL3's fixed/dense/MTP tensors total 10.32 GB per rank, plus a 0.01 GB
bookkeeping arena. The older hybrid reports 8.42 GB dense plus about 3.51 GB
of fixed MXFP4 slots: roughly 1.6 GB more in total before decoder routed slots.
Memory counters group these allocations differently, so comparing only the
`dense` field reverses the result. At 12,000 residents the normal HTTP run's
minimum host MemAvailable was 8.59/12.62 GB on the two nodes.

Raw local reports: `results/glm-perf-20261021/baseline-*.json` and
`results/glm-perf-20261021/native-exl3/http/baseline-*.json`.

## Experiments retained off

`GLM_INDEX_BUCKETS=1` bounds decode indexing by the occupied power-of-two
context bucket and keys graphs by that bound. It matched all teacher-forced
logits at 96 checked positions and all greedy output hashes on both ranks,
but complete-round timings did not show a consistent improvement. It remains
off. Lower MTP confidence thresholds also failed to improve consistently;
`GLM_MTP_DYN=0.9` remains the profile setting. Exact measurements and the graph
pool / 4-bit prefill failures are recorded in `docs/gotchas.md`.
