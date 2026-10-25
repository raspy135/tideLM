# TideLM

**Mixture-of-experts models too big for your memory, served from two DGX Sparks.** 

Normally it requires heavy quantization, however TideLM took different approach.
TideLM keeps the experts your traffic actually uses resident in the two boxes' unified memory, learns which
ones those are from the router's own demand while it serves, and reads the misses that matter
most from NVMe as decoding goes on. The resident set rises and falls with what you ask it — the
tide — so the engine gets better the more you use it.

| Profile | Model | Routed experts | Resident | Context | Prose | Code | 8K-token prompt, first token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ds41-mxfp4`, Rust engine | DeepSeek-V4.1-Flash-1017 | native MXFP4 | 10,000 / 15,360 (65.1%) | 512K | **30.9 tok/s** | **50.9 tok/s** | 9.3 s |
| `ds41-mxfp4`, Python engine | DeepSeek-V4.1-Flash-1017 | native MXFP4 | 10,452 / 15,360 (68%) | 512K | **25.6 tok/s** | **43.3 tok/s** | 7.3 s |
| `glm53-exl3` | GLM-5.3 | EXL3 3.0 bpw | 12,500 / 19,200 (65.1%) | 256K | **15.2 tok/s** | **20.3 tok/s** | 18.5 s |

*Decode speed is `bench/bench.py`'s median over three measured runs after one warmup: 1,024
output tokens at temperature 0.6, prose = an essay request, code = an LRU-cache request. The last
column is the median time to first token of a fresh ~8,200-token prompt (no prefix reused). Each
profile uses adaptation, hot reads and speculative decoding on two DGX Sparks, with its learned
expert history. The Rust engine (`rust/`, no Python between the socket and the GPU) was measured
on 2026-10-24 from the Python rows' seed history and an empty prefix cache, at the profile's 512K
allocation, with packed KV and 10,000 residents: it reads native scales (the lossless scale pack
is not ported yet, so the Python engine's 10,452 do not fit in its memory):
[raw runs](results/rust-ds41-20261024/README.md). A second boot gave 30.7 / 50.1 tok/s and
8.0 s; its 8K first token moves between ~7.6 and ~9.3 s from run to run (the six runs' median
is 9.3 s), and is 7.5 s on a boot that serves only 8K prompts. Python DS was measured on 2026-10-22 with lossless scales, candidate indexing, the
split draft weights, an 8-token depth window and 10,452 residents at a 32K context allocation
(the profile supports 512K): [raw runs](results/ds41-depth8-20261022/README.md). Its code median
moves with each boot's acceptance (41.1-44.8 across that day's boots) and the 8K first token with
its Engram phase (6.9-11.2 s across runs). GLM was measured on 2026-10-22 from its own EXL3
checkpoint with FP8 kv_b, with 12,000 residents at 256K; the profile keeps 12,500:
[raw runs](results/glm-native-perf-20261022/README.md), [comparison](docs/glm53-native-exl3.md). First requests after boot can be slower
([gotchas](docs/gotchas.md)).*

## How adaptive expert loading works

A model like DeepSeek-V4.1-Flash (510 GB) or GLM-5.3 (410 GB in MXFP4) does not fit in two
Sparks' 2 × 128 GB. Everything except the routed experts does, and so does a large share of the
experts. TideLM decides which share, keeps deciding, and cheaply covers the rest:

1. **A resident set learned from demand.** Every routed pick the router *wants* — resident or
   not — is recorded per (layer, expert), weighted by its routing score, into a history that
   persists across restarts. The arena is filled from it at boot, and re-planned after each
   prompt's prefill, every few hundred generated tokens during long answers, and between
   requests. One arena is shared by all layers: a busy layer takes slots from a quiet one.
   A fresh install has no history yet. It starts from one learned on the development pair
   (`profiles/`, aggregate routing counts only), and your own traffic builds on top of it
   ([details](docs/adaptive-experts.md#learned-fresh-start-distribution)).
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
cp configs/glm53-exl3.env.example  .env.glm53-exl3     # GLM-5.3, EXL3 (needs the packs below)
```

**2. Get the checkpoint** onto both nodes:

| Profile | Checkpoint | Size |
| --- | --- | ---: |
| `ds41-mxfp4` | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (or the original `DeepSeek-V4.1-Flash`) | 510 GB |
| `glm53-exl3` | [`bertholomus/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw), repacked into one file per node (below) | 293 GB checkpoint; 139 GB pack a node |

`MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017 MODEL_DIR=<dir> bash scripts/download-model.sh`
fetches the DeepSeek checkpoint and resumes if interrupted (`HF_TOKEN` is optional; it only
raises rate limits). The GLM checkpoint downloads with `hf download <repo> --local-dir <dir>`.
For the EXL3 profile, pack each node's half once, after building the image (step 3); the EXL3
checkpoint supplies the dense and fixed weights too, so keep it on both nodes:

```bash
# on node 0 (and on node 1 with r1of2 / --rank 1)
docker run --rm --gpus all -v <exl3 dir>:/models-exl3:ro -v <packs dir>:/packs -v "$PWD":/w -w /w \
  --entrypoint python tidelm:local tools/glm_pack_exl3.py \
  --source /models-exl3 --out /packs/glm-exl3-experts-r0of2.bin --rank 0
```

**3. Build the image** on the head; it is copied to the worker, and every profile runs from it:

```bash
scripts/dual-build.sh .env.ds41-mxfp4    # any of your profiles: its image, shipped to its PEER
```

The example profiles all name `tidelm:local` (`IMAGE` for DeepSeek, `GLM_IMAGE` for GLM-5.3), so
one build serves every profile. DeepSeek runs the code baked into the image and refuses to boot
from an image older than the checkout: after `git pull`, run the same command again. GLM-5.3
mounts the checkout itself.

The DeepSeek profile uses lossless scales. Pack the checkpoint once on the head (CPU only):

```bash
docker run --rm --entrypoint python -v /home/you/models:/models tidelm:local \
  /app/tools/pack_expert_scales.py --model-dir /models/DeepSeek-V4.1-Flash-1017 \
  --out-dir /models/dsv41-scale-packs
```

Copy `ds41-scales-r1of2.bin` to the same directory on the worker (about 5.2 GB per node). The
example profile's `DSV41_SCALE_PACK` points there; the same arena then holds more experts. Keep
the native checkpoint, and repack if its weights change. Empty `DSV41_SCALE_PACK` serves native
scales. [Setup and rollback details](docs/lossless-scales.md).

**4. Start and stop:**

```bash
scripts/tide-up.sh .env.glm53-exl3     # stops whatever is serving, starts this profile
scripts/tide-down.sh
```

The engine follows from the profile (`GLM_MODEL_DIR` → GLM-5.3, `MODEL_DIR` → DeepSeek, and
`MODEL_DIR` with `TIDE_ENGINE=rust` → DeepSeek on the Rust engine). The Rust engine runs from the
checkout rather than the image: build it once with `(cd rust && cargo build --release)` on the
head (`scripts/rust-up.sh` copies the binary to the worker), and see the profile's `RUST_*`
settings. It serves the OpenAI API with tool calls and images; the Anthropic route (Claude Code)
is the Python engine's only for now. Boot
takes 1-5 minutes; the first requests after a boot are slower while adaptation catches up with
your traffic. One profile serves at a time: each fills both boxes.

## Abliterated overlay (DeepSeek V4.1 only)

Abliterated overlay is supported for DeepSeek v4.1 flash.

https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated

Accept the repository's access terms on Hugging Face and authenticate `hf` first.
Download only the native FP8 sidecar on **both nodes**, under the parent of `MODEL_DIR`
(mounted at `/models`):

```bash
hf download drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated \
  wo_b_l10_35_1017.safetensors --revision 0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0 \
  --local-dir ~/models/dsv41-1017-wo-b-ablit
echo '88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864  '"$HOME/models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors" | sha256sum -c -
```

Set this **container path** in `.env.ds41-mxfp4`, then start that profile normally:

```bash
DSV41_ABLIT_WOB=/models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors
```

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
| `ARENA_GB=97.4` | `GLM_SLOTS=12500` | Expert memory per node, and so how many experts stay resident. |
| `MAX_SEQ=524288` | `GLM_MAX_SEQ=262144` | Context allocation, generated tokens included. Less context, more experts. |
| `DSV41_ADAPT_SENSITIVITY=high` | `GLM_ADAPT=1`, `GLM_ADAPT_EVERY=200` | How fast the resident set follows new traffic. |
| `DSV41_EXPERT_SEED=1` | `GLM_STATS_SEED=1` | With no local history yet, start from the bundled one in `profiles/` (`0`: from scratch; or a seed file's path). |
| `DSV41_HOT_READS=1`, `DSV41_HOT_ASYNC=1`, `DSV41_HOT_RING=128` | `GLM_HOT=1`, `GLM_HOT_MIN=0.2`, `GLM_HOT_MAX=16` | Hot reads (above). |
| `DSV41_MAX_CONCURRENCY=1` | — | Concurrent requests; async hot reads need 1. |
| `SPEC=1` | `GLM_MTP=1`, `GLM_MTP_DEPTH=7` | Speculative decoding. |
| `DSV41_PREFIX_CACHE=1`, `DSV41_PREFIX_DISK=1` | `GLM_PREFIX_DISK=1`, `GLM_KV_SPLIT=1` | Prompt caching; GLM's KV cache split across the ranks. |
| — | `GLM_KV=fp8` (or `fp4` with `GLM_KV_FP8_LAYERS=0-9,68-77`) | GLM's KV cache format. Both profiles serve FP8 at 256K. The 4-bit latent with the first and last ten layers in FP8 fits 320K in less memory and measured KL +0.004 against FP8 on the long eval, but in use the model's recall of its context was visibly worse, so it is off by default ([gotchas](docs/gotchas.md)). |
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
