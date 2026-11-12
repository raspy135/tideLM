# GLM-5.3-Flash on one Spark (branch `glm-flash`, started 2026-11-12)

Goal: serve [Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw)
(rev `6f0cbf4a`, 156.3 GB) at TP=1 on one DGX Spark with adaptive residency, tuned to the level of
the GLM-5.3 and DS engines from the start (micro benches per kernel, full runs only to confirm).
TP=2 stays possible (world-generic code, packs derived from the one-box pack) but is not tuned:
at TP=2 the model is fully resident and other engines serve that case.

## The model (glm5_next) against GLM-5.3 (glm_moe_dsa)

| | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| hidden / layers | 6144 / 78 + MTP | 4096 / 45 + MTP (layer 45) |
| attention | MLA + DSA every layer (RoPE 64) | 34 KDA layers; MLA + DSA on 3, 7, ..., 43, no RoPE anywhere |
| MLA head | nope 192 + rope 64, kv_b 448 rows a head | nope 256, kv_b 512 rows a head |
| indexer | 32 x 128, top-2048 tokens | 32 x 128 over 4:1 pooled keys (gate + ape), top-512 pools + 0-3 tail tokens (<= 2051) |
| residual | plain | mHC: 4 streams, Sinkhorn 20 (DS-V4's), the MTP layer plain |
| MoE | 256 x top-8, inter 2048 | 288 x top-8, inter 2048, swiglu limit 10 |
| checkpoint | EXL3 everywhere | routed experts EXL3 mcg (3/4-bit), everything else BF16 |

Tokenizer: byte-identical to GLM-5.3's. Chat template: an older revision of the same family.

Sizes (safetensors headers): routed 133.3 GB (12,096 experts, ~11.0 MB), MTP experts 3.6 GB
(4-bit), non-expert BF16 18.2 GB without the vision tower (1.1 GB, not served).

## Decisions

- Non-expert weights in FP8 (the user's call, 2026-11-12): e4m3 with 32 x 32 UE8M0 block scales
  (ds41.cu's format and GEMVs, converted at load), BF16 kept loadable as the reference. To measure:
  held-out NLL BF16 vs FP8 in one boot.
- One GLM-family engine with a layer kind (MLA-DSA, KDA, nope-DSA), not a third crate: the MoE,
  residency, lanes, MTP, prefix and serving code is shared by construction. New code in its own
  modules (flash.cu, kda/pool kernels; Rust modules for KDA and mHC); GLM-5.3's kernels became
  templates whose GLM-5.3 instances compile to the same SASS.
- KDA rollback: double-buffered state a lane, rejected drafts undone by replaying the kept rows
  (bit-exact, no per-row snapshots). Rollback reaches back one pass; prefix reuse past that needs
  saved states (serving work).

## Status

Done (commits on `glm-flash`):
- tools/glm_pack_exl3.py `--repo` (packs straight from the Hub), `--from-pack` (TP=2 packs from the
  one-box pack), `--resume`, `--verify` (04652b3).
- cuda/flash.cu: KDA conv / chain / replay, key pools, ring, expansion; tests/flash_gpu.rs (2f90c4f).
- glm.cu templates: absorbs at nope 256, glm_mla_nope, router over 288, BF16 kv_b quantizer (512662e).

Fetch (2026-11-12): `~/tidelm-rust-state/flash/fetch.sh` -> `~/models/GLM-5.3-Flash-3.5bpw-dense`
(non-expert tensors, glm_fetch_dense.py) and `~/models/glm-flash-packs/glm-flash-35-r0of1.bin`.

Next: the engine (config `glm5_next`, weights with a Lin type - EXL3 / FP8 / BF16 - in the shared
MoE path, mHC through ds41.cu, KDA layer + state + rollback in set_pos, nope-DSA with pools, the
MTP layer without mHC), glm_main.rs for one box, a layer-level check against HF's modules, then the
serving profile.

## Measurements

KDA (flash_gpu kda_bench: 34 layers back to back, own state each, 64 heads, split 2):
1 / 4 / 8 rows 1.54 / 1.81 / 2.47 ms a pass; one row is the state's 8 MB a layer at ~195 GB/s.
Many rows 3.1 us a row (an 8K prefill ~0.9 s: the chunked WY form is the fix).

FP8 GEMV (flash_gpu fp8_gemv_bench, cold, best split), 1 / 4 / 8 rows:

| matrix | MB | us | GB/s at 1 row |
|---|---:|---|---:|
| KDA q\|k\|v\|f_a\|g_a\|b 24896 x 4096 | 102.0 | 460 / 466 / 493 | 222 |
| KDA o_proj 4096 x 8192 | 33.6 | 159 / 174 / 186 | 211 |
| MLA q_a\|kv_a\|indexer 2336 x 4096 | 9.6 | 86 / 83 / 94 | 111 |
| MLA q_b\|wq_b 20480 x 1536 | 31.5 | 156 / 157 / 161 | 201 |
| MLA o_proj 4096 x 16384 | 67.1 | 311 / 333 / 349 | 216 |
| shared gate\|up 4096 x 4096 | 16.8 | 109 / 124 / 123 | 154 |
| shared down 4096 x 2048 | 8.4 | 72 / 77 / 74 | 116 |
| dense gate\|up 24576 x 4096 | 100.7 | 476 / 481 / 518 | 212 |
| head 154880 x 4096 | 634.4 | 2570 / 2686 / 2699 | 247 |

From these, a 1-row pass: ~42 ms of non-expert reads (KDA 21, shared experts 7.6, MLA 6.1, head
2.6, dense 2.1, router + mHC ~1.3) + ~18 ms of routed experts (3.7 GB) + 1.5 ms KDA state: ~16 tok/s
before speculation. The small matrices (8-17 MB at 110-154 GB/s) are the GEMV lever left.
