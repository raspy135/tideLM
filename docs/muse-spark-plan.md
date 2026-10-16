# Muse Spark Open 816B-A42B on two Sparks: plan

Branch `muse-spark`, in a separate clone (`~/git/muse-spark-dgx-spark-dual-adaptive`), so the
DeepSeek V4.1 Flash checkout keeps serving untouched. Written 2026-10-15.

Facts below marked **(read)** come from the gated repos' `config.json`, `hf_quant_config.json`,
READMEs and safetensors shard headers (shapes and dtypes only; no weights downloaded). Every
tok/s figure is an **estimate** from bytes moved per step; none is measured yet.

## The model (read)

* 62 decoder layers, hidden 8192, vocab 202,048, untied embeddings, final logit softcap 20.
* **MoE in every layer.** 256 routed experts, top-8, no shared expert, router in FP32 with an
  `e_score_correction_bias` (DeepSeek-style). The experts run in a **4096-wide latent space**:
  `pre_expert_proj` 8192 -> 4096, experts `gate_up` [8192, 4096] and `down` [4096, 4096],
  `post_expert_proj` 4096 -> 8192, with pre/post expert norms. One expert = 50.3M parameters.
* **Attention:** GQA, 128 query heads / 16 KV heads, head_dim 64, plus a full 8192x8192
  output gate (`self_attn.gate_proj`). 46 layers are sliding-window (2048, RoPE theta 5e5);
  16 layers (every 4th, starting at layer 1) are full attention with **no RoPE**.
* Residual gates after attention and FFN (`residual_gate_temperature` 0.3).
* Multimodal: 50-layer vision tower (1536 wide) and a 4-layer audio tower. Text first.

Per token: ~17.8B dense parameters (attention 218M a layer including the output gate, the two
latent projections 67M a layer) and 8 x 62 experts = 25B; ~44B active including the head.
**The dense part is ~40% of active bytes**, unlike V4.1.

## Releases (read)

| repo | size | experts | dense (attention, latent projections, head) |
| --- | ---: | --- | --- |
| `meta-models/Muse-Spark-Open-816B-A42B` | 1.64 TB | BF16 | BF16 |
| `meta-models/...-NVFP4` | 533 GB | NVFP4 (group 16, FP8 E4M3 scales); **layers 0 and 61 left BF16** | BF16 |
| `amd/...-Quark-MXFP4` | 509 GB | **MXFP4 (group 32, E8M0 scales): the format this engine's FP4 kernels read** | BF16 |

Also published: `intel-ai/...-INT4`, `amd/...-Quark-PTPC-FP8`, MLX 3/4-bit, GGUF.

## Drafters (read)

| | `Inferact/Muse-Spark-Open-DSpark` | `RadixArk/Muse-Spark-Open-DFlash2` |
| --- | --- | --- |
| design | DSpark: same family as V4.1's drafter (Markov head rank 512, confidence head, block of 7, sample-from-anchor) | DFlash2 block drafter, verify width 8 |
| size | 5 dense layers (FFN 16384), 10.6 GB BF16 | 5 layers, ~5B listed, BF16 |
| target taps | layers 13, 29, 45, 57, 61 + final | layers 1, 16, 30, 44, 59 |
| tokens/step reported (temp 1.0) | MT-Bench 2.99, GSM8K 3.94, MATH-500 3.78, HumanEval 3.61, MBPP 3.47, AIME26 3.32, SWE-bench Pro 2.66 | GSM8K 5.37, AIME26 4.19, GPQA-D 3.88, Video-MME 3.74, MindCube 3.25 |
| runtime | vLLM speculators | SGLang only |

DSpark reuses this engine's draft path (block drafting, Markov head, confidence-chosen depth).
DFlash2 accepts more where both report (GSM8K 5.37 vs 3.94, AIME26 4.19 vs 3.32) but is a
new draft path.

## Estimates (TP2, per rank, per step)

Bytes per token with the dense part requantized to FP8 (BF16 as shipped):

| part | FP8 dense | BF16 as shipped |
| --- | ---: | ---: |
| attention + output gate (13.5B) | 6.8 GB | 13.5 GB |
| latent expert projections (4.2B) | 2.1 GB | 4.2 GB |
| LM head + FP32 router | 1.3 GB | 2.2 GB |
| experts, 8 x 62 at FP4 | ~7.6 GB | ~7.6 GB |
| full-attention KV read at 70K context | ~2.2 GB | ~2.2 GB |
| **total** | **~20 GB -> ~7-8 tok/s** | **~29 GB -> ~5-6 tok/s** |

(150-200 GB/s for the kernels, plus ~25-35 ms of collectives and small kernels for 62 layers.)

With a drafter the ~10 GB dense part is read once per step whatever the verify width; only the
experts grow (assumed 2.8x for 4 rows, 5x for 8, unmeasured):

| | tokens/step | tok/s |
| --- | --- | --- |
| no drafter | 1 | 7-8 |
| DSpark, 4-row verify | ~2.5-2.8 | ~11-12 |
| DSpark, 8-row verify | ~3.0-3.6 | ~9-11 |
| DFlash2, 8-row verify | ~3.7-4.5 | ~11-14 |

**Memory:** ~119 GB a rank. After the dense part in FP8 (~9 GB), FP8 embedding (~1 GB), the
drafter (~2-3 GB), KV at 128K (4.2 GB; only the 16 full layers grow: ~32 KB a token a rank,
17 GB at 512K) and runtime, roughly **85 GB a rank for experts: ~6,000 of 15,872 (~38-40%)**.
V4.1 runs at 64-68% resident with 11-13% of picks missing; Muse activates twice V4.1's share
of experts per token (8/256 vs 6/384), so expect **15-25% misses on mixed work** until a trace
says otherwise. On V4.1, 11% misses cost coding +0.006 nats but general text +0.23
(docs/gotchas.md), so misses are not free.

## Optimizations to try

Capacity (residency and miss cost) first: it decides quality as well as speed.

1. **Fewer bits per expert, more experts resident.** Hot experts FP4, cold ones ~2-3 bit
   (EXL3/CB3, which this repo already has). Measured on V4.1: CB3 at 40.8% resident beat FP4 at
   31.3% on held-out NLL (coding 1.571 -> 1.538, general 3.379 -> 3.209) **and** decoded faster
   (16.6 -> 19.0 tok/s), RESULTS.md. All Muse experts at ~2 bits is ~200 GB: misses could
   almost disappear. Quality per bit needs measuring.
2. **Layers 0 and 61's BF16 experts (NVFP4 release) to FP4:** 51.5 -> 14.5 GB in all,
   ~1,300 more experts. (Check whether the MXFP4 release already quantized them.)
3. **Context cap ~128K** instead of 512K: ~13 GB a rank, ~900 experts.
4. **FP8 KV on the 16 full-attention layers:** ~2 GB a rank at 128K, and halves that read.
5. **Drafter in FP8, sharing the target LM head** (DSpark's README: frozen target head; do not
   load a second copy): ~3-4 GB a rank.
6. **Embedding split across ranks, FP8:** ~2.5 GB a rank; one row is read per token.
7. **MXFP4 instead of NVFP4** (4.25 vs 4.5 bits a weight): ~6% more experts, coarser scales.

Items 2-6: ~25-40 GB a rank, residency ~40% -> ~50-55%.

Bandwidth (bytes per step):

8. **Verify width per step from DSpark's confidence head** (2-8 rows), extending the existing
   confidence depth policy: every extra row brings in experts. ~15-25%.
9. **Expert top-p pruning** (`DSV41_EXPERT_TOPP` 0.90: -8% of picks, no measurable loss on
   V4.1); also fewer distinct experts per block. Re-measure on Muse.
10. **FP4 for selected dense modules** (`DSV41_DENSE_FP4`), first candidate the 8192x8192 output
    gate; ~1 GB a rank per module type. Measure each, as for V4.1.
11. **FP8 KV read** (item 4): ~1.1 GB a rank at 70K.
12. **Router split across ranks** (FP32 [256, 8192] read whole on each rank today): ~0.26 GB.
13. **Packed FP8 LM head** (`head_packed_gluon`) and a listed draft vocabulary
    (`DSV41_DRAFT_VOCAB`) for the drafter.

Items 8-13: estimate ~12-17 tok/s with a drafter. Each needs its own measurement.

## Path

This branch ports this engine, rather than adding adaptive experts to SGLang:

* AMD's MXFP4 release matches the existing expert kernels' format (new shapes, not a new
  kernel family); NVFP4 would need new kernels.
* The DSpark drafter matches the existing draft path.
* New: gated GQA attention with sliding/full mix and NoPE full layers, latent MoE
  projections, residual gates, logit softcap, tokenizer and chat template, a pure-torch
  reference for numerics (from the transformers modeling code), the boot guard's fields.
* Kept: expert residency and swaps, NVMe streaming, TP2 collectives and boot guard, batched
  decode, prefix caches, server.

SGLang as the alternative: Muse and DFlash2 already run there, but NVFP4 MoE on sm_121 is build-
dependent (crashes and slower-than-FP8 reports), and it has no two-node GB10 write-up.

## First measurements, before most of the port

1. **Routing trace** (experts streamed from NVMe, slow): miss rate against resident fraction per
   workload, and distinct experts per 4- and 8-row block (the multipliers assumed above).
2. **Bits against residency** (item 1) on held-out text, as CB3 was measured.

## Storage (2026-10-15)

| | free |
| --- | ---: |
| rank 0 (this host) | 876 GB (76% used) |
| rank 1 (10.0.0.2) | 1.4 TB (61% used) |

After removing `dsv41-tensorfold` on both nodes, a full MXFP4 (509 GB) or NVFP4 (533 GB) copy
fits on each node with room for the drafter (10.6 GB), an image (~11 GB), the prefix disk budget
(20 GB) and traces, so no per-rank pack is needed to start. Under TP2 each rank needs only its
half of every expert plus the dense part (~260-280 GB a node), if space gets tight again.

## Running beside the V4.1 checkout

Both cannot serve at once (each fills the GPUs). Before the first launch from this clone, give it
its own `IMAGE` tag, `NAME0`/`NAME1` (defaults `deepseek-v41-ep2-rank*`, shared with V4.1),
`MODEL_DIR`, prefix disk directory and `.env`, so a build or `dual-down.sh` here cannot replace or
stop the V4.1 deployment by name. The peer needs this clone at the same path.
