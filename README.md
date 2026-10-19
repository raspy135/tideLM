# TideLM

**Mixture-of-experts models too big for your memory, served from two DGX Sparks.** 

Normally it requires heavy quantization, however TideLM took different approach.
TideLM keeps the experts your traffic actually uses resident in the two boxes' unified memory, learns which
ones those are from the router's own demand while it serves, and reads the misses that matter
most from NVMe as decoding goes on. The resident set rises and falls with what you ask it — the
tide — so the engine gets better the more you use it.

| Profile | Model | Routed experts | Resident | Context | Prose | Code | 8K-token prompt, first token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ds41-mxfp4` | DeepSeek-V4.1-Flash-1017 | native MXFP4 | 10,208 / 15,360 (66%) | 512K | **25.9 tok/s** | **42.8 tok/s** | 11.8 s |
| `glm53-mxfp4`(Experimental) | GLM-5.3 | MXFP4 (AMD Quark) | 8,740 / 19,200 (46%) | 256K | **12.2 tok/s** | **15.3 tok/s** | 23.6 s |
| `glm53-exl3` | GLM-5.3 | EXL3 3.0 bpw | 12,000 / 19,200 (62.5%) | 320K | **14.0 tok/s** | **18.4 tok/s** | 23.6 s |

*Decode speed is `bench/bench.py`'s median over three measured runs after one warmup: 1,024
output tokens at temperature 0.6, prose = an essay request, code = an LRU-cache request. The last
column is the median time to first token of a fresh ~8,200-token prompt (no prefix reused). Each
profile ran exactly as shipped in `configs/` (adaptation, hot reads and speculative decoding on)
on two DGX Sparks, 2026-10-18, with the expert history it had learned from earlier use. Raw runs:
[bench/results/2026-10-18/](bench/results/2026-10-18/). The `glm53-exl3` row was measured again on
2026-10-19 after the decode routing kernel and the move to 320K with the FP4/FP8 latent cache
([bench/results/2026-10-19/](bench/results/2026-10-19/)), at 12,000 residents rather than the
example's 12,350 while one node had ~3 GB held by a leaking system service. The 4-bit cache costs
the fresh 8K prompt ~2.6 s: the same day and code with the 256K FP8 cache took 21.0 s (it was
19.5 s on 2026-10-18).*

## How adaptive expert loading works

A model like DeepSeek-V4.1-Flash (510 GB) or GLM-5.3 (410 GB in MXFP4) does not fit in two
Sparks' 2 × 128 GB. Everything except the routed experts does, and so does a large share of the
experts. TideLM decides which share, keeps deciding, and cheaply covers the rest:

1. **A resident set learned from demand.** Every routed pick the router *wants* — resident or
   not — is recorded per (layer, expert), weighted by its routing score, into a history that
   persists across restarts. The arena is filled from it at boot, and re-planned after each
   prompt's prefill, every few hundred generated tokens during long answers, and between
   requests. One arena is shared by all layers: a busy layer takes slots from a quiet one.
2. **Misses never stall a token.** The router only picks resident experts; a missing pick is
   replaced by the next-best resident one for that token, and the demand still counts toward the
   next re-plan.
3. **Hot reads for the misses that matter.** When a missing expert would have carried a large
   share of a token's routing weight, it is read from NVMe while decoding continues:
   - DeepSeek (`DSV41_HOT_ASYNC`): after each step, the heaviest misses are read into a ring of
     128 slots of their own while the next step runs, and are routable from the step after until
     the ring wraps. No measured decode cost (41.21 vs 41.08 tok/s on the generation gate), and
     held-out loss improves by 0.007 nats on code and 0.138 on prose
     ([measurements](docs/hot-reads.md)).
   - GLM-5.3 (`GLM_HOT`): a layer's heaviest miss (≥ 0.2 of the routing weight, at most 16 a
     step, paced over the layers) is read before that layer's MoE runs, computed exactly, and
     kept resident in place of the coldest expert.
4. **Two Sparks as one.** Both boxes hold half of every resident expert and run tensor-parallel
   over RoCE; a boot-time guard refuses to start if the two ranks disagree on any setting that
   changes the numbers.

Around that core: speculative decoding with each model's own drafter (DeepSeek's DSpark heads,
GLM's MTP layer, both with a dynamic draft depth), prompt caching in RAM and on disk, FP8 KV caches
split across the ranks (GLM), image input (DeepSeek), tool calls, thinking and reasoning-effort
control through an OpenAI-compatible API.

## Quick start

You need two DGX Sparks connected by a working RoCE link, Docker with NVIDIA GPU support,
passwordless SSH from the head (rank 0) to the worker, and the checkpoint on local NVMe at the
same path on both nodes. Clone this repository to the same absolute path on both nodes.

**1. Pick a profile** and fill in your paths, the worker's address and network interfaces:

```bash
cp configs/ds41-mxfp4.env.example  .env.ds41-mxfp4     # DeepSeek-V4.1-Flash(-1017), MXFP4
cp configs/glm53-mxfp4.env.example .env.glm53-mxfp4    # GLM-5.3, MXFP4
cp configs/glm53-exl3.env.example  .env.glm53-exl3     # GLM-5.3, EXL3 (needs the packs below)
```

**2. Get the checkpoint** onto both nodes:

| Profile | Checkpoint | Size |
| --- | --- | ---: |
| `ds41-mxfp4` | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (or the original `DeepSeek-V4.1-Flash`) | 510 GB |
| `glm53-mxfp4`, `glm53-exl3` | [`amd/GLM-5.3-Quark-MXFP4-AttnFP8`](https://huggingface.co/amd/GLM-5.3-Quark-MXFP4-AttnFP8) | 410 GB |
| `glm53-exl3` (also) | [`bertholomus/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw), repacked into one file per node (below) | 139 GB a node |

`MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017 MODEL_DIR=<dir> bash scripts/download-model.sh`
fetches the DeepSeek checkpoint and resumes if interrupted (`HF_TOKEN` is optional; it only
raises rate limits). The GLM checkpoints download with `hf download <repo> --local-dir <dir>`.
For the EXL3 profile, pack each node's half once, after building the image (step 3); the EXL3
checkpoint can be deleted afterwards:

```bash
# on node 0 (and on node 1 with r1of2 / --rank 1)
docker run --rm --gpus all -v <exl3 dir>:/models-exl3:ro -v <packs dir>:/packs -v "$PWD":/w -w /w \
  --entrypoint python tidelm:local tools/glm_pack_exl3.py \
  --source /models-exl3 --out /packs/glm-exl3-experts-r0of2.bin --rank 0
```

**3. Build the image** on the head; it is copied to the worker, and every profile runs from it:

```bash
IMAGE=tidelm:local PEER=<user>@<worker address> bash scripts/dual-build.sh
```

The example profiles already name `tidelm:local` (`IMAGE` for DeepSeek, `GLM_IMAGE` for GLM-5.3).
Without `PEER` the image is only built on the head, and the worker cannot start. DeepSeek runs
the code baked into the image and refuses to boot from an image older than the checkout, so
rebuild after pulling new code; GLM-5.3 mounts the checkout itself.

**4. Start and stop:**

```bash
scripts/tide-up.sh .env.glm53-mxfp4    # stops whatever is serving, starts this profile
scripts/tide-down.sh
```

The engine follows from the profile (`GLM_MODEL_DIR` → GLM-5.3, `MODEL_DIR` → DeepSeek). Boot
takes 1-5 minutes; the first requests after a boot are slower while adaptation catches up with
your traffic. One profile serves at a time: each fills both boxes.

## Use

- API: `http://<head>:8000/v1` (OpenAI-compatible chat completions, streaming included). Model
  names: `deepseek` and `glm-5.3` (`SERVED_MODEL_NAME`). No authentication: bind it to a trusted
  interface (`HOST` / `BIND_HOST`).
- Works with Open WebUI and other OpenAI clients; streamed responses end with `usage` and the
  engine's own statistics (`x_engine_stats`).
- `/health` shows the serving configuration; on DeepSeek, `/expert-map` draws which experts are
  resident, by layer.
- Logs: `docker logs -f deepseek-v41-ep2-rank0` or `glm-serve-rank0`.

Thinking, reasoning effort, tool calls and images: [API reference](server/README.md).

## Settings worth knowing

Edit the profile, then `scripts/tide-up.sh` it again. The examples in `configs/` are commented
with what each setting measured.

| DeepSeek (`DSV41_*`) | GLM-5.3 (`GLM_*`) | What it controls |
| --- | --- | --- |
| `ARENA_GB=97.4` | `GLM_SLOTS=8740` | Expert memory per node, and so how many experts stay resident. |
| `MAX_SEQ=524288` | `GLM_MAX_SEQ=327680` | Context allocation, generated tokens included. Less context, more experts. |
| `DSV41_ADAPT_SENSITIVITY=high` | `GLM_ADAPT=1`, `GLM_ADAPT_EVERY=200` | How fast the resident set follows new traffic. |
| `DSV41_HOT_READS=1`, `DSV41_HOT_ASYNC=1`, `DSV41_HOT_RING=128` | `GLM_HOT=1`, `GLM_HOT_MIN=0.2`, `GLM_HOT_MAX=16` | Hot reads (above). |
| `DSV41_MAX_CONCURRENCY=1` | — | Concurrent requests; async hot reads need 1. |
| `SPEC=1` | `GLM_MTP=1`, `GLM_MTP_DEPTH=7` | Speculative decoding. |
| `DSV41_PREFIX_CACHE=1`, `DSV41_PREFIX_DISK=1` | `GLM_PREFIX_DISK=1`, `GLM_KV_SPLIT=1` | Prompt caching; GLM's KV cache split across the ranks. |
| — | `GLM_KV=fp4`, `GLM_KV_FP8_LAYERS=0-9,68-77` (or `fp8`) | GLM's KV cache format. The EXL3 profile keeps the first and last ten layers in FP8 and the rest in 4 bits: 320K in less memory than 256K in FP8, for KL +0.004 against FP8 and ~12% on a fresh prompt's prefill ([gotchas](docs/gotchas.md)). The MXFP4 profile stays FP8 at 256K. |
| — | `GLM_LOOP_BREAK=12` | Stops a short cycle of tokens from repeating a 12th time (repetition loops start where experts are substituted). |

## Documentation

- [Hot reads: design and measurements](docs/hot-reads.md)
- [Adaptive expert loading](docs/adaptive-experts.md), [GLM-5.3 engine plan](docs/glm53-plan.md)
- [Known issues and rejected experiments](docs/gotchas.md) — kept on purpose: several plausible
  ideas were measured and lost here, and are recorded so they are not tried twice
- [Results](RESULTS.md), [TP memory](docs/tp-memory.md), [packed KV](docs/packed-kv.md),
  [speculative decoding](docs/decode-dynamic-depth.md), [decode dashboard](docs/decode-probe.md)

## Lineage and credits

TideLM merges two engines that share one origin. The DeepSeek V4.1 engine is a fork of
[0xBakeer's deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark),
and its foundation is his: the insight that a model too large for one box can be served from a
resident hot set of experts, and the engine that proves it — the expert arena and transient
ring, the `O_DIRECT` streaming path, the Triton FP4 grouped-MoE kernel, the chunk-invariant port
of DeepSeek's reference, the stdlib server and the benchmark harness. His routing traces, coverage
curves and keep-set analysis ([NOTES.md](NOTES.md), [RESULTS.md](RESULTS.md)) are the backbone of
the approach. This fork moved it to two Sparks (TP2 over RoCE), made residency adaptive (learned
from live demand, one shared arena, re-planned while serving), and added hot reads. The GLM-5.3
engine (`engine/glm/`) grew out of the Muse Spark Open experiment on the same codebase and
reuses its runtime, kernels and server.

The RoCE all-gather is ported from [TensorFold](https://github.com/ashhart/TensorFold), which
adapted b12x's; the EXL3 expert kernels port TensorFold's ExLlamaV3-derived device code. Models:
DeepSeek-V4.1-Flash by DeepSeek, GLM-5.3 by Z.ai, the MXFP4 GLM checkpoint by AMD (Quark), the
EXL3 GLM checkpoint by bertholomus. Full attributions and licences: [CREDITS.md](CREDITS.md).

Repository code is [MIT-licensed](LICENSE). Model use is governed by each checkpoint's own license.
