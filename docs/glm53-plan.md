# GLM-5.3 (754B, glm_moe_dsa) on two Sparks

Branch `glm-5.3` in `~/git/glm53-dgx-spark-dual-adaptive`, built on the Muse Spark branch (whose
engine/muse/ it reuses: communication, memory guards, graph replay, server). Written 2026-10-15.

## The checkpoint: amd/GLM-5.3-Quark-MXFP4-AttnFP8 (410 GB, read from its headers)

| part | format |
| --- | --- |
| routed experts (75 MoE layers x 256), shared experts, 3 dense MLPs, MTP layer's experts | MXFP4: U8 [N, K/2] low nibble first + E8M0 scale per 32 -- the layout tools/fp4_moe_cuda.cu reads |
| attention q_a / q_b / kv_a / kv_b / o, indexer wq_b / wk | FP8 E4M3 + FP32 scale per 128x128 block |
| router, norms, indexer weights_proj, embedding, lm_head | BF16 (correction bias F32) |
| layer 78 | MTP: eh_proj, enorm, hnorm, shared_head.norm, a full decoder layer |

Model: 78 layers (3 dense), hidden 6144, 64 MLA heads (q_lora 2048, kv_lora 512, nope 192, rope 64,
v 256, interleaved RoPE theta 8e6), DSA indexer 32 heads x 128 with top-2048 on 21 "full" layers
(the other 57 reuse the last full layer's selection), 256 experts top-8 (sigmoid, correction bias,
routed scale 2.5, one group) + 1 shared, vocab 154880.

## Status

| piece | state | evidence |
| --- | --- | --- |
| reference (engine/glm/ref.py) | bit-exact with transformers 5.20 | fp32 relative error 0.0 (prefill and cached decode), 1.4e-7 in float64 |
| FP8 128-block kernel (engine/glm/fp8_block.py) | done | matches a dequantized matmul on every real shape |
| TP2 engine (engine/glm/tp.py) | correct on a synthetic checkpoint at real shapes | engine vs fp32 reference 5.0% against the BF16 reference's own 5.6% |
| absorbed MLA over indexed positions, decode graphs | done | graph == eager bitwise |
| experts: routed, shared (9th pick, weight 1), dense MLP (6 pseudo-experts) | done | one FP4 kernel path |
| substitution / streamed misses | done | masked reference / bitwise full residency |
| MTP drafter + speculative verify | done | MTP logits = reference; tokens = target argmax of a clean pass |
| server (GLM template, token-level parser, tool calls) | done | template/parser tests on the real tokenizer |
| routing trace | done | checked on the synthetic checkpoint |
| real weights | downloading / replicating at the time of writing | |

Synthetic speed (4 layers, graphed): 5.84 ms a step, ~1.46 ms a layer = ~0.98 ms of weight bytes
at 200 GB/s + fixed cost. Projection for 78 layers ~100-115 ms a token before MTP (~9-10 tok/s);
MTP depth 3 at the typical acceptance for this family (~2.5 tokens a step) ~13-15 tok/s. One MTP
draft step: 2.72 ms synthetic (graphed; eh_proj split) + the head.

Findings recorded in the commits: q_a/kv_a norms use eps 1e-6 (not rms_norm_eps); the indexer's relu
ties make a prefill and a decode legitimately pick different keys; the native FP4 kernels' K limit
was 5120 (raised to 6144); a layer with fewer resident experts than top-k must not hand the
kernels slot -1.

## Run

```
cp .env.glm.example .env.glm    # adjust
scripts/glm-up.sh               # serving stopped first; scripts/dual-down.sh stops it
```
Tests: `tools/muse_pair_run.sh glm_pair_test.py` (two nodes, synthetic); `python -m unittest
engine.glm.test_ref engine.glm.test_fp8_block server.test_glm_app`. Real weights:
`MUSE_MODEL_DIR=... tools/muse_pair_run.sh glm_probe.py` (tok/s, acceptance, samples).
