# TideLM

**Mixture-of-experts models too big for your memory, served from two DGX Sparks.** 

Normally it requires heavy quantization, however TideLM took different approach.
TideLM keeps the experts your traffic actually uses resident in the two boxes' unified memory, learns which
ones those are from the router's own demand while it serves, and reads the misses that matter
most from NVMe as decoding goes on. The resident set rises and falls with what you ask it — the
tide — so the engine gets better the more you use it.

| Profile | Model | Routed experts | Resident | Context | Prose | Code | 8K-token prompt, first token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ds41-mxfp4` | DeepSeek-V4.1-Flash-1017 | native MXFP4 | 10,000 / 15,360 (65.1%) | 2 x 512K | **31.8 tok/s** | **49.0 tok/s** | 4.8 s |
| `glm53-exl3` | GLM-5.3 | EXL3 3.0 bpw | 12,500 / 19,200 (65.1%) | 256K | **17.2 tok/s** | **20.8 tok/s** | 19.5 s |

*Decode speed is `bench/bench.py`'s median over three measured runs after one warmup: 1,024
output tokens at temperature 0.6, prose = an essay request, code = an LRU-cache request. The last
column is the median time to first token of a fresh ~8,200-token prompt (no prefix reused). Raw
runs: [DeepSeek](results/rust-ds41-20261024/README.md), [GLM-5.3](results/rust-glm53-20261025/README.md).*

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
     ([measurements](docs/hot-reads.md)). Both lanes share the ring: every lane's misses feed
     one plan, at no decode cost (41.2 vs 40.6 tok/s for two requests at once).
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

**3. Build the image** on the head; it is copied to the worker, and both models run from it:

```bash
scripts/tide.sh build .env.ds41-mxfp4    # any of your profiles: the image, shipped to its PEER
```

It compiles the Rust engine inside the build, so the boxes need nothing but Docker. The engine
refuses to start from an image built from other sources than the checkout: after `git pull`,
run the same command again. The EXL3 pack tool (step 2) runs from this image too. The Rust
engine serves DeepSeek's native scales; the lossless scale pack
([details](docs/lossless-scales.md)) is the Python engine's.

**4. Start and stop:**

```bash
scripts/tide.sh start .env.glm53-exl3    # stops whatever is serving, starts this profile
scripts/tide.sh logs -f                  # its log (--rank 1: the worker's)
scripts/tide.sh stop
```

The model follows from the profile (`GLM_MODEL_DIR` → GLM-5.3, `MODEL_DIR` → DeepSeek), and
the Rust engine serves both; the profiles carry only the settings it reads. It serves the OpenAI
API with tool calls (and images, DeepSeek) and the Anthropic route for Claude Code. It is up 15 s
(DeepSeek) to 40 s (GLM-5.3) after the previous pair has stopped; the first requests after a boot
are slower while adaptation catches up with your traffic. One profile serves at a time: each
fills both boxes. `TIDE_ENGINE=python` still starts the Python containers, from a profile with
their settings (the examples as of `a6ec917`), and a profile the Rust engine does not serve
(MXFP4 GLM experts, FP4 KV) goes to them with a note.

## Abliterated overlay (DeepSeek V4.1)

Both engines serve an abliterated DeepSeek V4.1 Flash from a 1.1 GB sidecar that replaces layers
10-35's attention output projection (`attn.wo_b`); every other weight stays the checkpoint's, and
the checkpoint is never written.

https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated

Accept the repository's access terms on Hugging Face and authenticate `hf` first.
Download only the native FP8 sidecar on **both nodes**, at the same path. The Python engine
needs it under the parent of `MODEL_DIR` (its containers mount that at `/models`); the Rust
engine reads any host path:

```bash
hf download drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated \
  wo_b_l10_35_1017.safetensors --revision 0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0 \
  --local-dir ~/models/dsv41-1017-wo-b-ablit
echo '88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864  '"$HOME/models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors" | sha256sum -c -
```

For the Rust engine, set its **host path** in the DeepSeek profile and start it
(`scripts/tide.sh start .env.ds41-mxfp4`); the log shows
`ablit: layers 10-35 attn.wo_b from ... (sha256 88324cea...)`:

```bash
DSV41_ABLIT_WOB=/home/you/models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors
```

The engine refuses a file that is not exactly those 52 tensors in the checkpoint's names, dtypes
and shapes, both ranks refuse to pair over different files (the sha256 is in the boot guard), and
prompts cached on disk under the stock weights are not restored into the abliterated model (or back).

For the Python engine, set the **container path** instead in a Python DeepSeek profile
(`TIDE_ENGINE=python`, the example as of `a6ec917`), then start that profile:

```bash
DSV41_ABLIT_WOB=/models/dsv41-1017-wo-b-ablit/wo_b_l10_35_1017.safetensors
```

## Use

- API: `http://<head>:8000/v1` (OpenAI-compatible chat completions, streaming included). Model
  names: `deepseek` and `glm-5.3` (`SERVED_MODEL_NAME`). No authentication: bind it to a trusted
  interface (`HOST`).
- Works with Open WebUI and other OpenAI clients; streamed responses end with `usage` and the
  engine's own statistics (`x_engine_stats`).
- Logs: `scripts/tide.sh logs -f`, a line a request with its prefill and decode speeds
  (`--rank 1` for the worker's).

Thinking, reasoning effort, tool calls and images: [API reference](server/README.md).

## Settings worth knowing

Edit the profile, then `scripts/tide.sh start` it again. The examples in `configs/` are commented
with what each setting measured.

| DeepSeek | GLM-5.3 | What it controls |
| --- | --- | --- |
| `RUST_BUDGET=9872` | `GLM_SLOTS=12500` | How many experts stay resident (GLM: slots a rank). Fewer, more memory for context. |
| `MAX_SEQ=524288` | `GLM_MAX_SEQ=262144` | Context allocation (DeepSeek: a lane), generated tokens included. Less context, more experts. |
| `DSV41_ADAPT_SENSITIVITY=high` | `GLM_ADAPT=1`, `GLM_ADAPT_EVERY=200` | How fast the resident set follows new traffic. |
| `RUST_ADAPT_DB` | `RUST_GLM_STATE_DB` | The learned history, under `~/tidelm-rust-state`. With none yet, the first boot starts from the bundled one in `profiles/`. |
| `DSV41_HOT_READS=1`, `DSV41_HOT_ASYNC=1`, `DSV41_HOT_RING=128` | `GLM_HOT=1`, `GLM_HOT_KEEP=32` | Hot reads (above). |
| `RUST_LANES=2` | — | Concurrent requests; the lanes share the hot-read ring, whose 128 slots come out of `RUST_BUDGET` (10,000 at one lane). |
| `SPEC=1`, `RUST_DEPTH=confidence` | `GLM_MTP=1`, `GLM_MTP_DEPTH=7`, `GLM_MTP_CAL=0.36` | Speculative decoding and its depth policy. |
| `DSV41_PREFIX_CACHE=1`, `DSV41_PREFIX_DISK=1` | `GLM_PREFIX_DISK=1`, `GLM_KV_SPLIT=1` | Prompt caching; GLM's KV cache split across the ranks. |
| — | `GLM_KV=fp8` | GLM's KV cache format, the one the Rust engine serves. The Python engine's 4-bit latent with the first and last ten layers in FP8 fits 320K in less memory and measured KL +0.004 against FP8 on the long eval, but in use the model's recall of its context was visibly worse ([gotchas](docs/gotchas.md)). |
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
