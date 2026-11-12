# TideLM

**Adaptive expert loading enables running higher quant models on two DGX Sparks.** 

TideLM keeps the experts your traffic actually uses resident in the two boxes' unified memory, learns which
ones those are from the router's own demand while it serves, and reads the misses that matter
most from NVMe as decoding goes on. The resident set rises and falls with what you ask it — the
tide — so the engine gets better the more you use it.

| Profile | Model | Routed experts | Resident | Context | Prose | Code | 8K-token prompt, first token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ds41-mxfp4` | DeepSeek-V4.1-Flash-1017 | native MXFP4 | 10,000 / 15,360 (65.1%) | 1M (2 lanes, 1.5M shared) | **33.5 tok/s** | **51.8 tok/s** | 4.3 s |
| `glm53-exlr8` | GLM-5.3 | exlr8 `bit260` (2.6 bpw: 2-3 bits a projection) | 14,124 / 19,200 (73.6%) | 320K (2 lanes) | **22.3 tok/s** | **28.2 tok/s** | 17.9 s |
| `glm53-exlr8-home` | GLM-5.3 | exlr8 `home` (3.25 bpw: 3 or 4 bits an expert) | 11,430 / 19,200 (59.5%) | 320K (2 lanes) | **20.4 tok/s** | **23.9 tok/s** | 17.2 s |

*Medians of three `bench/bench.py` runs after a warmup, 1,024 tokens at temperature 0.6 (raw runs:
[DeepSeek](results/rust-ds41-20261028-pdl/README.md), [GLM-5.3](results/rust-glm53-exlr8-20261111/README.md),
[GLM-5.3 `home`](results/rust-glm53-exlr8-home-20261112/README.md), a day later on another build).*

GLM-5.3 is served from the `bit260` profile of
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4). Against
the EXL3 2.75 bpw profile it replaced (20.5 / 26.2 tok/s, 17.3 s on the same binary and afternoon) it
decodes about 8% faster, scores 0.024 nats better on held-out text and keeps 1,150 more experts
resident in the same memory; its 8K-token first token is 3% slower ([what it took](docs/exlr8.md)).
The EXL3 profile is still served: its setup is in the [install guide](docs/install.md).

`home` is the same release's 3.25 bpw profile, for when the wider experts matter more than the
speed: the release measures it closer to the unquantized model than `bit260` (KL 0.149 against
0.192 on its held-out set, with every expert loaded), and on this pair its larger records leave
2,900 fewer experts resident in the same memory (11,430 against 14,361) and decode about 9%
(prose) to 15% (code) slower. What it gives in answers with 59.5% of the experts resident is not
measured here
([`home` on the pair](docs/exlr8.md#home-325-bpw-2026-11-12)).

Adaptive expert loading also allows flexible memory configuration.
Memory for expert resident and context length are the big levers. You can run other services
by reducing expert resident or context length.


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
     slots of their own (`DSV41_HOT_RING`, 64 in the example profile) while the next step runs, and
     are routable from the step after until the ring wraps. No measured decode cost with 128 slots
     (41.21 vs 41.08 tok/s on the generation gate), and
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
split across the ranks (GLM), DeepSeek's context caches paged out of one pool the lanes share (one
lane can run to 1M while another holds less; when the pool runs short the newest request waits on
disk and resumes exactly, [details](docs/paged-kv.md)), image input (DeepSeek), tool calls,
thinking and reasoning-effort control through an OpenAI-compatible API.

## Quick start

You need two DGX Sparks connected by a working RoCE link, Docker with NVIDIA GPU support,
passwordless SSH from the head (rank 0) to the worker, and the weights on local NVMe at the same
path on both nodes. Clone this repository to the same absolute path on both nodes. The
[install guide](docs/install.md) has each step in full: disk per profile, copying between the
nodes, the GLM packs.

There are three profiles, and each is set up on its own below. One image runs all three, one
profile serves at a time (each fills both boxes), and switching between profiles that are set up
is one `scripts/tide.sh start`.

| Profile | Model | Disk, each node | Head only, while setting up |
| --- | --- | ---: | ---: |
| [`ds41-mxfp4`](#deepseek-v41-flash-ds41-mxfp4) | DeepSeek-V4.1-Flash-1017 | 510 GB | |
| [`glm53-exlr8`](#glm-53-exlr8-bit260-glm53-exlr8) | GLM-5.3, exlr8 `bit260` | 142 GB | 246 GB |
| [`glm53-exlr8-home`](#glm-53-exlr8-home-glm53-exlr8-home) | GLM-5.3, exlr8 `home` | 172 GB | 308 GB |

### DeepSeek-V4.1-Flash (`ds41-mxfp4`)

**1. Profile.** Copy the example and fill in your paths, the worker's address and the network
interfaces:

```bash
cp configs/ds41-mxfp4.env.example .env.ds41-mxfp4
```

**2. Weights**, on both nodes: 510 GB of
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(or the original `DeepSeek-V4.1-Flash`). The script resumes if interrupted; `HF_TOKEN` is
optional and only raises rate limits. Set `MODEL_DIR=<dir>` in the profile.

```bash
MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017 MODEL_DIR=<dir> bash scripts/download-model.sh
```

**3. Image**, on the head; it is copied to the worker:

```bash
scripts/tide.sh build .env.ds41-mxfp4
```

It compiles the Rust engine inside the build, so the boxes need nothing but Docker. The engine
refuses to start from an image built from other sources than the checkout: after `git pull`,
run the same command again. The Rust engine serves DeepSeek's native scales; the lossless scale
pack ([details](docs/lossless-scales.md)) is the Python engine's.

**4. Start:**

```bash
scripts/tide.sh start .env.ds41-mxfp4
```

### GLM-5.3, exlr8 `bit260` (`glm53-exlr8`)

The default GLM-5.3 profile: the `bit260` profile (2.6 bpw) of
[`kindlingai/glm-5.3-exlr8-k2-k3-k4`](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4).
The release stores every routed expert at 2, 3 and 4 bits and a profile names one width a unit,
so only the profile's part is fetched (246 GB of 838 GB, on the head, needed only to pack). What
serves is a 120 GB pack of routed experts and 22 GB of other weights on each node.

**1. Profile.** Copy the example and fill in your paths and the worker's address:

```bash
cp configs/glm53-exlr8.env.example .env.glm53-exlr8
```

**2. Fetch**, by HTTP range requests (stdlib Python; resumable; it holds tens of MB and is safe
beside a serving engine):

```bash
python3 tools/exlr8_fetch.py --manifest bit260 --out <exlr8 dir>     # head: 237 GB of experts
python3 tools/exlr8_fetch.py --dense --out <exlr8 dir>               # head: 8.8 GB of dense layers
python3 tools/glm_fetch_dense.py --out <base dense dir>              # each node: 15 GB
```

exlr8 keeps its largest non-expert matrices at 4 bits and the rest at 8; the profile takes the
4-bit ones and leaves the rest to
[`bertholomus/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw), an
EXL3 checkpoint of the same GLM-5.3 (5-6 bits: fewer bytes a decode pass than exlr8's 8), which
is what `glm_fetch_dense.py` fetches.

**3. Image**, on the head; it is copied to the worker. Skip this if you built it for another
profile, and run it again after every `git pull`:

```bash
scripts/tide.sh build .env.glm53-exlr8
```

**4. Packs and weights**, once, on the head: each node's half of the routed experts, and the
combined non-expert weights. The tools need numpy and xxhash (`pip install numpy xxhash`, or
`uv run --with numpy --with xxhash python ...`); every unit read is checked against the release's
checksums. With `--follow 3600` a pack runs behind the fetch and both are done when it is, 57
minutes here.

```bash
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit260 --rank 0 \
    --out <packs dir>/glm-exlr8-bit260-r0of2.bin
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit260 --rank 1 --out - | \
    ssh <worker> 'cat > <packs dir>/glm-exlr8-bit260-r1of2.bin'
python3 tools/glm_exlr8_dense.py --base <base dense dir> --exlr8 <exlr8 dir> --manifest bit260 \
    --out <dense dir>             # the base's shards hard-linked, 6.9 GB of exlr8 matrices beside them
# the worker's <dense dir>: its own copy of the base by hard links, then the new shards and the index
ssh <worker> 'cp -al <base dense dir> <dense dir>'
rsync -a <dense dir>/exlr8-overlay-*.safetensors <dense dir>/model.safetensors.index.json <worker>:<dense dir>/
```

Set `GLM_EXL3_DIR=<packs dir>` and `GLM_MODEL_DIR=<dense dir>` in the profile, at the same paths
on both nodes. Once the packs and weights are in place `<exlr8 dir>` is no longer read.

**5. Start:**

```bash
scripts/tide.sh start .env.glm53-exlr8
```

### GLM-5.3, exlr8 `home` (`glm53-exlr8-home`)

The same release's `home` profile (3.25 bpw: 192 of a layer's routed experts at 3 bits, 64 at
4): wider experts, fewer of them resident, slower (the table at the top). The steps are
`bit260`'s with `--manifest home` and a 150 GB pack on each node. **With `bit260` already set
up**, only the experts' fetch (172 GB more) and the two packs are new: skip the lines marked
*shared*, and use the same `<exlr8 dir>`, `<packs dir>` and `<dense dir>`.

**1. Profile.** Copy the example and fill in your paths and the worker's address:

```bash
cp configs/glm53-exlr8-home.env.example .env.glm53-exlr8-home
```

**2. Fetch** (as above: resumable, safe beside a serving engine):

```bash
python3 tools/exlr8_fetch.py --manifest home --out <exlr8 dir>       # head: 300 GB of experts
python3 tools/exlr8_fetch.py --dense --out <exlr8 dir>               # shared - head: 8.8 GB of dense layers
python3 tools/glm_fetch_dense.py --out <base dense dir>              # shared - each node: 15 GB
```

**3. Image**, on the head (*shared*: skip it if you built it for another profile):

```bash
scripts/tide.sh build .env.glm53-exlr8-home
```

**4. Packs and weights**, once, on the head (numpy and xxhash, as above):

```bash
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest home --rank 0 \
    --out <packs dir>/glm-exlr8-home-r0of2.bin
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest home --rank 1 --out - | \
    ssh <worker> 'cat > <packs dir>/glm-exlr8-home-r1of2.bin'
# shared from here - home's shared experts have bit260's widths, so <dense dir> is the same directory
python3 tools/glm_exlr8_dense.py --base <base dense dir> --exlr8 <exlr8 dir> --manifest home \
    --out <dense dir>
ssh <worker> 'cp -al <base dense dir> <dense dir>'
rsync -a <dense dir>/exlr8-overlay-*.safetensors <dense dir>/model.safetensors.index.json <worker>:<dense dir>/
```

Set `GLM_EXL3_DIR=<packs dir>` and `GLM_MODEL_DIR=<dense dir>` in the profile, at the same paths
on both nodes. The two GLM profiles then differ in one key, `GLM_EXL3_PACK`.

**5. Start:**

```bash
scripts/tide.sh start .env.glm53-exlr8-home
```

Another profile of the release (`bit243`, ...) is the same steps with its `--manifest`: the fetch
adds only the bytes the mirror lacks.

### Running it

```bash
scripts/tide.sh start .env.glm53-exlr8   # stops whatever is serving, starts this profile
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

## Abliterated GLM-5.3

The Rust engine serves GLM-5.3 abliterated with the edit of
[`Blackfrost-Research/GLM-5.3-DERISKED-BF16`](https://huggingface.co/Blackfrost-Research/GLM-5.3-DERISKED-BF16):
one direction a layer, projected out of layers 2-49's attention and MLP outputs at run time (alpha
3). That is the same model as its edited weights, the routed experts included, so the expert packs
stay the stock ones. The directions are a 1.2 MB file, recovered from that checkpoint and the BF16
base (`zai-org/GLM-5.3-BF16`) by HTTP range requests: both repos are public, ~5 GB is fetched a
tensor at a time, and nothing but the file is kept. It needs numpy:

```bash
python3 tools/glm_ablit_fit.py --alpha 3 --out ~/models/glm53-ablit/glm53-derisked-l2-49.safetensors
```

Copy the file to **both nodes** at the same path and set it in the GLM profile, then start it
(`scripts/tide.sh start .env.glm53-exlr8`); the log shows
`[glm] ablit: 48 layers (2-49), alpha 3.000-3.000, from ... (sha256 ...)`:

```bash
GLM_ABLIT=/home/you/models/glm53-ablit/glm53-derisked-l2-49.safetensors
```

As with the DeepSeek overlay, both ranks refuse to pair over different files (the sha256 is in the
boot guard), and the disk prefix store keeps the abliterated model's prompts apart from the stock
model's: the first request of a conversation after switching prefills its whole prompt again. On
held-out text (`corpus/heldout_quick.jsonl`, 10,151 tokens) it costs NLL +0.018 +- 0.005 (+1.0%)
and top-1 60.22 -> 59.95% (measured on the EXL3 profile; not yet on `glm53-exlr8`).

## Use

- API: `http://<head>:8000/v1` (OpenAI-compatible chat completions, streaming included). Model
  names: `deepseek` and `glm-5.3` (`SERVED_MODEL_NAME`).
- Authentication: with `API_KEY=<key>` in the profile (several: comma-separated), every route but
  `/health` wants the key, as OpenAI clients send it (`Authorization: Bearer <key>`) or Anthropic's
  (`x-api-key: <key>`; Claude Code's `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` as a Bearer
  token); a request without it gets 401. Unset, the server answers anyone: then bind it to a trusted
  interface (`HOST`). `bench/bench.py` takes `--api-key`.
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
| `RUST_BUDGET=9950` | `GLM_SLOTS=12400` | How many experts stay resident (GLM: expert memory a rank, in 3-bit experts' worth, 7.1 MB each - a budget, not a count: 14,361 of `bit260`'s, whose records are smaller). Fewer, more memory for context. |
| `MAX_SEQ=1048576`, `RUST_KV_TOKENS=1572864` | `GLM_MAX_SEQ=262144` | Context, generated tokens included (DeepSeek: the most one lane reaches, out of a pool of `RUST_KV_TOKENS` the lanes share). Less context, more experts. |
| `DSV41_ADAPT_SENSITIVITY=high` | `GLM_ADAPT=1`, `GLM_ADAPT_EVERY=200` | How fast the resident set follows new traffic. |
| `RUST_ADAPT_DB` | `RUST_GLM_STATE_DB` | The learned history, under `~/tidelm-rust-state`. With none yet, the first boot starts from the bundled one in `profiles/`. |
| `DSV41_HOT_READS=1`, `DSV41_HOT_ASYNC=1`, `DSV41_HOT_RING=64` | `GLM_HOT=1`, `GLM_HOT_KEEP=32` | Hot reads (above). The ring's slots are on top of `RUST_BUDGET`, 9.4 MB a rank each (0.56 GiB at 64): a bigger ring needs a smaller budget. |
| `RUST_LANES=2` | — | Concurrent requests; the lanes share the one hot-read ring. |
| `SPEC=1`, `RUST_DEPTH=confidence` | `GLM_MTP=1`, `GLM_MTP_DEPTH=7`, `GLM_MTP_CAL=0.36` | Speculative decoding and its depth policy. |
| `DSV41_PREFIX_CACHE=1`, `DSV41_PREFIX_DISK=1` | `GLM_PREFIX_DISK=1`, `GLM_KV_SPLIT=1` | Prompt caching; GLM's KV cache split across the ranks. |
| — | `GLM_KV=fp8` | GLM's KV cache format, the one the Rust engine serves. The Python engine's 4-bit latent with the first and last ten layers in FP8 fits 320K in less memory and measured KL +0.004 against FP8 on the long eval, but in use the model's recall of its context was visibly worse. |
| — | `GLM_EXLR8_ROUND=exact` | exlr8's codebook rounded to FP8 exactly as the format defines it, instead of in two instructions (the default: a decode pass ~2% shorter, held-out NLL unchanged). |
| — | `GLM_LOOP_BREAK=12` | Stops a short cycle of tokens from repeating a 12th time (repetition loops start where experts are substituted). |

## Documentation

- [Install guide](docs/install.md): each setup step in full
- [Hot reads: design and measurements](docs/hot-reads.md)
- [Adaptive expert loading](docs/adaptive-experts.md), [GLM-5.3 engine plan](docs/glm53-plan.md),
  [exlr8 on the pair](docs/exlr8.md)
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
EXL3 GLM checkpoints by bertholomus and Mia's AI Lab, the exlr8 GLM release by KindlingAI. Full attributions and licences: [CREDITS.md](CREDITS.md).

Repository code is [MIT-licensed](LICENSE). Model use is governed by each checkpoint's own license.
