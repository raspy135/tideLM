# DeepSeek-V4.1-Flash on two DGX Sparks

Serves DeepSeek-V4.1-Flash across two DGX Sparks with **adaptive expert residency** and a
**negotiable memory budget**. The default profile keeps 90% of the 15,360 routed experts
resident; which ones is learned from live router demand and re-planned while the server
runs, and one shared arena lets busy layers take slots from quiet ones. The arena's size is a
single knob you trade against context length and whatever else shares the boxes.

The pair runs **two-node tensor parallelism (TP2)** over RoCE. Routed experts default to an
**EXL3 2.9 bpw** pack that fits 1.4x as many experts in the same memory; the checkpoint's
**native MXFP4** experts remain a supported profile on the same image and scripts. The
OpenAI-compatible API supports streaming responses, thinking, tool calls and images.
The engine began as [0xBakeer's single-Spark engine](https://github.com/0xBakeer/deepseek-v41-flash-spark);
see [lineage](#lineage-and-credits) for what is inherited and what changed.

| Default EXL3 profile, 2026-10-11 | |
| --- | --- |
| Decode, essay / HTML game / Python | **32.2 / 67.3 / 46.6 tok/s** (FP4: 31.2 / 57.8 / 42.5) |
| Cold prefill, 8K / 32K | **1,401 / 1,545 tok/s** (FP4: 988 / 1,325) |
| Resident routed experts at 92.3 GB/node | **13,813 of 15,360 (90%)** (FP4: 9,793, 64%) |

At the default arena not every routed expert is resident; residency follows demand and is
shared across layers. Adaptation improves coverage; it does not guarantee full-model quality.

<p align="center"><img src="assets/expert-map.png" width="820"
  alt="The live expert map: 40 layers by 384 experts, resident experts in green, missing ones dark"></p>

*The live expert map (`/expert-map`) on the head node of the default EXL3 profile: 40 layers
by 384 experts, 13,813 resident (green) and 1,547 missing from the arena (dark). The right
edge counts each layer's residents. They range from 326 to 367 because the layers share one
arena, and busier layers have taken slots from quieter ones.*

## Why use this engine?

- **Vision support.** Send images and text through the same chat API. 
- **Flexible memory configuration, including room for other small services.** Adjust the resident
  expert count, context allocation and each node's Engram cache to fit your setup.
- **EXL3 experts by default, native MXFP4 when you want it.** EXL3 trades a
  requantization (2.9 bpw) for 90% residency and 19-31% faster decode; the FP4 profile
  keeps the checkpoint's expert weights. Switch by restarting with the other example
  profile; nothing is rebuilt.
- **Let your workload shape memory allocation.** Router-score history learns which
  experts to keep, and persists across restarts. This changes residency, not model
  weights. A bundled learned seed gives fresh installs a starting distribution.
- **Move capacity between layers.** One shared arena replaces fixed per-layer
  quotas: a less-used layer can give its slots to a busier one. The total memory
  budget stays fixed, with only the routing minimum reserved in each layer.
- **Adapt while answering.** Normal adaptation runs after prefill and during long
  answers. An urgent trigger can replace experts when recent decode misses rise;
  it does not require streaming every missing expert from disk.
- **See what the engine is doing.** The live expert map (above) shows loading, residency
  and arena ownership. Request statistics expose both routing-count and
  router-score-weighted misses, making selection changes inspectable.

TP2 output sharding, EXL3 and native FP4 kernels, RoCE communication, speculative decoding
and RAM prefix reuse support this design. The distinctive feature is their
combination with a persistent, observable, globally adaptive expert working set.

## EXL3 (default) or FP4 experts

Both profiles serve the same model: dense layers, attention, the vocabulary head, Engram and
shared experts always come from the native checkpoint. Only the 384 x 40 routed experts
differ. In both profiles the DSpark drafter's three blocks read a 4-bit EXL3 draft pack by
default (`DSV41_EXL3_DRAFT_DENSE=1`); the drafter only proposes and the target verifies, so
this changes speed, not output.

| At `ARENA_GB=92.3` per node | **EXL3** (`.env.example.exl3`, default) | FP4 (`.env.example.fp4`) |
| --- | --- | --- |
| Routed expert weights | [Mia-AiLab's](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) EXL3, 2.9 bpw average (2-bit in layers 18-22), 6.67 MB/node each | Checkpoint MXFP4, 9.4 MB/node each |
| Resident experts | **13,813 (90%)** | 9,793 (64%) |
| Decode, essay / HTML / Python | **32.2 / 67.3 / 46.6 tok/s** | 31.2 / 57.8 / 42.5 tok/s |
| Cold prefill, 8K / 32K | 1,401 / 1,545 tok/s | 988 / 1,325 tok/s |
| Extra disk | 98 GB expert pack + 195 MB draft pack per node, built once from the 198 GB EXL3 checkpoint | 195 MB draft pack per node, built once from ~14 GB of the EXL3 checkpoint (none with `DSV41_EXL3_DRAFT_DENSE=0`) |

EXL3 reads fewer bytes per expert, which is most of a decode step, and leaves fewer
experts missing; that is why it is the default. The cost is requantization error in the
routed experts: choose FP4 to serve the checkpoint's own expert weights. Our tests
do **not** establish that either profile answers better; they are throughput checks
plus the engine's floor tests. In use, FP4 has sometimes answered specialised prompts
better ([FP4 profile](#fp4-profile-native-mxfp4-experts)). See [the 2026-10-11 measurements](RESULTS.md) and
[the EXL3 design notes](docs/exl3-plan.md).

## Performance

Measured on two DGX Sparks on **2026-10-11**, both profiles on the same image at 92.3 GB/node:
512K context allocation, native dense precision, abliteration enabled, dynamic speculative
depth and adaptation active, EXL3 draft blocks, listed draft vocabulary and expert top-p 0.90 on
(the profiles' defaults). TTS was not loaded.

| Workload | EXL3 decode | EXL3 time to first token | FP4 decode |
| --- | ---: | ---: | ---: |
| Python LRU cache with TTL, thread safety and tests | **46.6 tok/s** (46.5–46.8) | 0.84 s | 42.5 tok/s (42.1–42.9) |
| Coastal-ecosystem essay | **32.2 tok/s** (32.1–32.3) | 0.90 s | 31.2 tok/s (29.4–33.0) |
| Angry Birds single-file HTML game | **67.3 tok/s** (67.0–67.6) | 0.27 s* | 57.8 tok/s (56.8–58.8) |

Medians of two 512-token runs after one warmup each; thinking off, temperature zero,
fixed output length. Decode speed excludes time to first token. *The HTML prompt reused
its 62 cached prompt tokens; the others reused none. FP4's first-token medians were 1.44 s
(essay), 0.53 s (HTML) and 1.29 s (Python). These are short throughput checks, not
answer-quality tests. Prompt content,
draft acceptance, expert history and cache reuse affect speed. See [full results](RESULTS.md).

Cold prefill (uncached random-word prompts; medians of 5 at 8K and 2 at 32K):

| Prompt | EXL3 | FP4 | FP4 before 2026-10-08 |
| --- | ---: | ---: | ---: |
| 8K | **1,401 tok/s** (5.9 s) | 988 tok/s (8.3 s) | 390–470 tok/s (17–21 s) |
| 32K | **1,545 tok/s** (21.2 s) | 1,325 tok/s (24.7 s) | 470 tok/s (70 s) |

The first requests after a boot or profile switch run 1.5–3x slower while adaptation moves
experts and kernels warm up. Prefill switches: `DSV41_ENGRAM_DIRECT=prefill`,
`DSV41_PREFILL_CHUNK=2048`, `DSV41_PREFILL_ATTN_INDEXED=1`, `DSV41_HC_PREFILL_FUSED=1`,
`DSV41_INDEX_FUSED=1`, `DSV41_FP4_PREFILL_DOT_SCALED=1`, `DSV41_FP4_PREFILL_SCALED_TILES=1`;
EXL3 adds its prefill tile kernel (`DSV41_EXL3_PREFILL_KERNEL=tile`, the default).
A larger `ARENA_GB` leaves less memory for prefill scratch: at 95.3 GB the engine halved its
prefill chunk and 8K took 7.4 s.

Decode history (FP4, same benchmark on 2026-10-08):

| Workload | Before | Arithmetic switches | + Draft | + Kernels |
| --- | ---: | ---: | ---: | ---: |
| Coastal-ecosystem essay | 21.9 tok/s | 23.1 tok/s | 25.4 tok/s | **26.4 tok/s** |
| Angry Birds HTML | 43.3 tok/s | 50.3 tok/s | 52.7 tok/s | **53.4 tok/s** |
| Python LRU cache with TTL | 32.3 tok/s | 37.4 tok/s | 38.4 tok/s | **40.4 tok/s** |

Switches: `DSV41_ATTN_STAGED=2`, `DSV41_ROUTER_BF16=1`, `DSV41_HC_KERNEL=1`. Draft:
`DSV41_TP_DRAFT_HEAD=1`, `DSV41_DRAFT_HEAD_FMT=fp8`, `DSV41_DRAFT_MARKOV_TP=1`,
`DSV41_TP_DRAFT_ATTN=1`. Kernels: `DSV41_FP8_DECODE_BLOCK_N=32`, `DSV41_FP8_DECODE_BLOCK_K=256`,
`DSV41_LEAN_RMS_FUSED=1`, `DSV41_HC_FRONT_FUSED=1`. Paired 22-prompt averages: **+9%**,
**+2.3%** and **+2.1%**, **+1.8%** and **+4.0%**. The EXL3 draft blocks (`DSV41_EXL3_DRAFT_DENSE=1`,
default since 2026-10-11): **-1.05 ± 0.37 ms** a step paired, outputs identical. The listed draft
vocabulary (`DSV41_DRAFT_VOCAB`): **-1.06 ± 0.37 ms**, outputs identical. Expert top-p 0.90
(`DSV41_EXPERT_TOPP`, lossy): **-2.22 ± 0.22 ms** (FP4) and **-1.91 ± 0.39 ms** (EXL3), held-out loss
unchanged within ±0.006 nats. Both profiles also queue the next draft
before the host reads each verify result (`DSV41_EARLY_DRAFT`/`DSV41_EARLY_VERIFY`, on by
default): -2.0 ± 0.5 ms a step, outputs bit-identical.

Optional `DSV41_HEAD_KERNEL=packed` losslessly packs the native BF16 vocabulary
head. The short TP2 code/prose comparison saved **146 MiB/node** and improved
decode round speed about **2%**, with identical tested tokens and acceptance.
The draft head must be `off`/`bf16` unless `DSV41_TP_DRAFT_HEAD=1` builds a separate
FP8 draft shard; the examples enable both.

Reproduce an individual workload against a running server:

```bash
python3 bench/bench.py --model deepseek --workload code --osl 512 \
  --warmup 1 --runs 2 --temperature 0 --ignore-eos --label readme-20261008-code
```

## Setup

You need two DGX Sparks, Docker with NVIDIA GPU support, a working RoCE link,
and roughly 600 GB of local NVMe space per node for the checkpoint (the EXL3 profile
adds a 98 GB pack per node; both profiles add a 195 MB draft pack). Enable head-to-worker SSH, and clone this repo to the
same absolute path on both nodes:

```bash
git clone https://github.com/raspy135/deepseek-v41-flash-dgx-spark-dual-adaptive.git
```

**1. Pick a profile** (on the head):

```bash
cp .env.example.exl3 .env           # EXL3 experts (default): faster decode, 90% resident
or
cp .env.example.exl3.allweight .env # EXL3 experts, 100% resident
or
cp .env.example.fp4 .env            # or FP4: the checkpoint's own expert weights, no packs
```

Set `MODEL_DIR`, `PEER`, `MASTER_ADDR`, `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME`. Keep credentials in `.env`.

**2. Checkpoint.** Both profiles need the native checkpoint on both nodes. Use your
existing checkpoint; if it is missing, run the download script on each node (requires
Python 3 and `huggingface_hub`). It exits at once if the checkpoint is already complete:

```bash
(set -a; source .env; set +a; bash scripts/download-model.sh)
```

**3. Build the image** on the head. It builds once and copies the image to the worker;
both profiles use the same image:

```bash
bash scripts/dual-build.sh
```

**4. EXL3 packs** (once): the expert and draft packs for the EXL3 profile, see
[below](#exl3-expert-packs-once-for-the-exl3-profile); only the small draft packs for FP4, see
[FP4 profile](#fp4-profile-native-mxfp4-experts).

**5. Start** on the head:

```bash
bash scripts/dual-up.sh --check
bash scripts/dual-up.sh
```

Startup takes several minutes, and the first request after a boot is slower while kernels
warm up. The template binds to localhost; set `HOST` to a trusted interface to allow
remote clients. The API has no authentication. Optional dual-rail networking needs both
interfaces/subnets configured as described in the example profiles.

`dual-up.sh` refuses an image built from different engine code than the checkout (rebuild
with `dual-build.sh`, or boot the checkout directly with `scripts/dual-dev.sh` while
developing), and refuses to report success if an EXL3 profile came up without its CUDA
kernel (`/health` must show `"kernel": "exl3-cuda"`).

### EXL3 expert packs (once, for the EXL3 profile)

Only the head needs the EXL3 checkpoint (about 198 GB). After `dual-build.sh`:

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --local-dir /path/to/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw   # next to the native checkpoint
bash scripts/exl3-packs.sh --check
bash scripts/exl3-packs.sh       # ~8 min a rank, then copies rank 1's pack to the worker
```

This writes one 98 GB expert pack and one 195 MB draft pack per node to `<models>/exl3-packs/`,
where the engine looks for them. Re-running keeps packs that already exist (`--force` rebuilds
them). The EXL3 checkpoint and the head's copy of rank 1's expert pack can then be deleted.

### FP4 profile (native MXFP4 experts)

The FP4 profile serves the routed experts exactly as DeepSeek ships them. The only EXL3 part
is the drafter's own draft pack (`DSV41_EXL3_DRAFT_DENSE=1`, on by default), which needs just
the two EXL3 checkpoint shards that hold the draft blocks (~14 GB, head only):

```bash
cp .env.example.fp4 .env         # then set the same five values as in step 1
bash scripts/dual-build.sh       # skip if the image is already built
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw config.json model.safetensors.index.json \
  model-00038-of-00039.safetensors model-00039-of-00039.safetensors \
  --local-dir /path/to/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw   # next to the native checkpoint
bash scripts/exl3-packs.sh --draft-only   # ~1 min a rank; copies rank 1's pack to the worker
bash scripts/dual-up.sh
```

The partial checkpoint can be deleted afterwards. To skip the EXL3 download entirely, set
`DSV41_EXL3_DRAFT_DENSE=0`: the drafter then reads the checkpoint's FP8 draft blocks, about
1 ms a step slower.

`/health` shows `"expert_format": "fp4"` and `"kernel": "triton-fp4"`. At the same
`ARENA_GB=92.3`, FP4 keeps 9,793 experts resident (64%) where EXL3 keeps 13,813, and
decodes 3-14% slower ([table](#exl3-default-or-fp4-experts)). In exchange, the experts
are the checkpoint's own weights rather than a 2.9 bpw requantization (2-bit in layers
18-22). **In our own use FP4 has sometimes given better answers**, for example on
detailed security questions. That is an observation, not a measured result: our quality
suite has not compared the two profiles head to head. If answer quality in your domain
matters more than decode speed, try FP4.

**Switching profiles:** copy the other example over `.env` (keeping your edits), then
`scripts/dual-down.sh` and `scripts/dual-up.sh`. Nothing is rebuilt. Switching to EXL3
needs its packs.

## Use

- API: `http://<host>:8000/v1`, model name **`deepseek`**.
- Health: `/health` shows the actual serving configuration and allocated/reserved CUDA memory.
  `DSV41_GRAPHS_MAX=8` bounds cached graph variants by rotating their shared pool.
  `DSV41_PREFILL_ADAPT=1` shrinks prefill chunks under memory pressure on either node;
  `DSV41_PREFILL_CHUNK` remains the maximum. The scratch estimate is not an OOM guarantee.
- Expert map: `/expert-map` shows resident, loading and missing experts by layer
  or arena sector. `/v1/expert-map` returns JSON; it does not expose prompt text.
- Logs: `docker logs -f deepseek-v41-ep2-rank0`. Container names retain `ep2`,
  but the template runs TP2.
- Stop: `bash scripts/dual-down.sh`. Restart both nodes after changing settings.

See the [API reference](server/README.md) for requests, thinking and tool calls.

## Example profiles

[.env.example.exl3](.env.example.exl3) (default) and [.env.example.fp4](.env.example.fp4) target
running the engine on its own. They differ only in `EXPERT_FORMAT` and the predictor database:

| Setting | Profile |
| --- | --- |
| Context allocation | 524,288 tokens, including the answer |
| Expert memory | 92.3 GB/node; residents follow the arena (EXL3 13,813, FP4 9,793); 8 transient slots |
| Expert selection | Global dynamic allocation, router-score ranking, high sensitivity, prior 4 |
| Adaptation | Computed discovery before prefill, after prefill, during long answers, on urgent misses, and between requests |
| Vision | Tower on the second node; embeddings shared with both ranks |
| Engram row cache | 256 MiB on the head, 1,024 MiB on the worker |
| Prompt cache | RAM reuse enabled; disk and post-response caching disabled |
| Generation | Speculative depth 3/5; thinking off by default, effort 75 when enabled |

**All weights resident:** [.env.example.exl3.allweight](.env.example.exl3.allweight) keeps every
routed expert in memory (15,360 of 15,360, no expert misses) by giving the 1,920 2-bit EXL3 experts
slots of their own size (`DSV41_EXL3_EXACT_SLOTS=1`: 98.37 GB/node instead of 102.6), with a 200K
context and a 256 MiB worker Engram cache. Measured 2026-10-08: decode 30.2 / 64.9 / 43.7 tok/s
(essay / HTML / Python), cold prefill 8K 7.3 s and 32K 35.4 s. The ~7 GB left per node makes the
prefill budget choose 512-row chunks, so prefill is 22-34% slower than the 92.3 GB profile; the
lowest free memory under a 32K-prompt stress run was 4.6 GB. Use it when nothing else shares the boxes.

For **TTS alongside the engine**, use `ARENA_GB=90.2`, as on the development pair; the
resident expert count follows the arena (every slot but 16). The engine-only
example adds about 2.1 GB per node. Its slot capacity is checked,
but sustained peak-memory headroom has not yet been validated. Both TP nodes
need room for the increase, even when TTS runs on only one.

There are no fixed per-layer quotas beyond the routing minimum. Fresh installs
use the bundled learned expert seed; existing local demand history takes priority.
The learned history survives restarts. **RAM prompt caches do not.** The retained
300-second disk-save interval has no effect while disk caching is disabled, as
required by the current dynamic allocation mode.

Dense FP4 re-quantization and expert streaming are off. Predictive prefill is on
(`DSV41_PROBE_PREFILL=1`): the engine computes
expert demand from the current prompt and loads the selected experts before prefill.
For 1–499 uncached tokens it discovers the complete new suffix, then reruns that suffix
from the cached prefix. For 500 or more it uses excerpts capped at 64 tokens. It needs
no training bank; the old dictionary predictor is unused. Actual prefill and decode
continue learning residency, and post-prefill correction stays active. Set
`DSV41_PROBE_PREFILL=0` to disable discovery. Discovery and loading add latency;
answer-quality gains are not established ([measurements](docs/predictive-prefill.md)).

Memory headroom depends on other services. Lower the arena (the resident count follows)
or context allocation, if needed. Full-length 512K quality is not established.
Keep demand databases, predictor banks and prompt caches private and out of Git.

## Basic knobs

Edit `.env` on the head, then restart the pair with `scripts/dual-down.sh` and
`scripts/dual-up.sh`. The launcher sends the settings to both ranks. Values below
refer to the example profile, not every engine fallback default.

| Setting | What it controls |
| --- | --- |
| `EXPERT_FORMAT=exl3` | `exl3` (default profile; needs the packs, see Setup) or `fp4` (the checkpoint's native MXFP4 experts). Same image; restart to switch. |
| `ARENA_GB=92.3` | Expert arena GB **per node**; the resident count follows it (every slot but 16, at most all 15,360 experts, logged at boot) for either `EXPERT_FORMAT`. Use `90.2` for the TTS profile. `DSV41_RESIDENT_EXPERTS=<n>` pins a count for experiments. |
| `MAX_SEQ=524288` | Context allocation including generated tokens. Lower it to reduce cache memory needs. |
| `DSV41_ADAPT_SENSITIVITY=high` | How quickly new traffic changes expert ranking: `low`, `medium`, `high`, `max`; `off` freezes adaptation. Higher follows changes faster but can displace useful experts sooner. |
| `DSV41_ADAPT_PRIOR=4` | Weight of the shipped routing trace. Lower values let your observed traffic dominate sooner. |
| `DSV41_ADAPT_DECODE_TOKENS=600` | Periodic adaptation during an answer. `0` disables it; post-prefill adaptation remains. (`DSV41_ADAPT_URGENT` was removed for hot reads.) |
| `DSV41_HOT_READS=1` / `DSV41_HOT_ASYNC=1` | Hot reads: the missing experts a decode step wanted most are read into a ring of their own slots and routable from the next step (docs/hot-reads.md). Concurrency 1; `TRANSIENT_SLOTS` >= 8 + `DSV41_HOT_RING`. |
| `DSV41_PREFILL_CHUNK=2048` | Tokens processed per prefill chunk. Smaller chunks reduce temporary memory and give finer prefix-cache boundaries; `DSV41_PREFILL_ADAPT=1` shrinks them under memory pressure. |
| `DSV41_ENGRAM_CACHE_MB=256` / `DSV41_ENGRAM_CACHE_MB_PEER=1024` | Engram row-cache budgets in MiB on the head and worker. Separate from expert residency and prompt caching. |
| `DSV41_VISION=1` / `DSV41_VISION_MODE=peer` | Enable images and put the vision tower on rank 1. Set vision to `0` for text-only serving. |
| `DSV41_PREFIX_CACHE=1` | Reuse matching prompt state in RAM. Keep `DSV41_PREFIX_DISK=0` and `DSV41_PREFIX_RESPONSE=0` with dynamic allocation. |
| `SPEC=1` | Enable DSpark speculative decoding; `0` disables drafting. Speed depends on draft acceptance. |
| `DSV41_EXPERT_TOPP=0.90` | Lossy decode pruning: per token, routed experts beyond 90% of the routing weight (at least 3 kept) are skipped. About 2 ms a step faster; held-out loss unchanged within ±0.006 nats, but greedy text can differ. `0` turns it off. |
| `DSV41_DRAFT_VOCAB=/app/engine/draft_vocab_ranking_v1.txt` | The drafter scores 81,920 listed token ids instead of all 129,280. About 1 ms a step faster; output unchanged (the target verifies every token). Empty turns it off. |
| `DSV41_HEAD_KERNEL=off` | Optional `packed` stores the native BF16 head losslessly and uses a decode kernel. Requires `DSV41_DRAFT_HEAD_FMT=off` or `bf16`. Saves about 146 MiB/node; measured code/prose decode gain was about 2%. |
| `DEFAULT_THINKING=off` / `DEFAULT_EFFORT=75` | Defaults when the client does not specify thinking. Effort is 1–100 and applies when thinking is enabled; clients can override it per request. |

The two `ADAPT_*` ranking knobs derive the normal swap thresholds; copying old
`DSV41_PRUNE_*` threshold overrides is unnecessary. `DSV41_USER_PROMPT_MAX_LOADS`
only caps experimental streaming loads, **not normal adaptive expert swaps**.
See [adaptation details](docs/adaptive-experts.md) for advanced settings.

## Optional abliterated weights

Set `DSV41_ABLIT_WOB` to a compatible native
`wo_b_l10_35.safetensors` overlay (about 1.1 GB), as supported by
[the overlay loader](engine/ablit.py). It replaces the attention output projections
in layers 10–35 at load time. Experts and other checkpoint tensors retain their
original weights; no second full checkpoint or modification of the original files
is needed. This intentionally changes model behavior.

Place the **same overlay file on both nodes** inside the host models directory.
For example, if the checkpoint is `/srv/models/DeepSeek-V4.1-Flash`, place it at
`/srv/models/dsv41-wo-b-ablit/wo_b_l10_35.safetensors`. For the Docker launcher,
use the corresponding **container path** in `.env`:

```dotenv
DSV41_ABLIT_WOB=/models/dsv41-wo-b-ablit/wo_b_l10_35.safetensors
```

Restart both ranks and check `engine_config.ablate_wob` in `/health`. The boot
guard checks that the overlay contents agree across nodes. For a native launch,
use the host's absolute path instead. To return to stock weights, remove or empty
`DSV41_ABLIT_WOB` and restart. The overlay substitutes weights rather than adding
another model, so disabling it does not free an extra expert arena.

## Details and measurements

- [Adaptive expert loading and learned seed](docs/adaptive-experts.md)
- [Predictive prefill experiment](docs/predictive-prefill.md)
- [Results](RESULTS.md), [historical measurements](docs/serving-measurements.md),
  and [known issues and rejected experiments](docs/gotchas.md)
- [TP memory](docs/tp-memory.md), [packed KV](docs/packed-kv.md),
  and [speculative decoding](docs/decode-dynamic-depth.md)
- Decode dashboard at `/decode-probe`: [timing, buffer reuse and local replay controls](docs/decode-probe.md)

## Lineage and credits

This engine is a fork of [0xBakeer's deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark),
and its foundation is his: the insight that a model too large for one box can be served
from a resident hot set of experts, and the engine that proves it — the expert arena and
transient ring, the `O_DIRECT` streaming path, the Triton FP4 grouped-MoE kernel, the
chunk-invariant port of DeepSeek's reference, the stdlib server and the benchmark harness.
His documentation is the intellectual backbone of the approach: the routing traces and
coverage curves, the keep-set analysis, saliency ranking, the topic profiles and the
REAP-style pruning sweep in [NOTES.md](NOTES.md) and [RESULTS.md](RESULTS.md). This fork
would not exist without it.

What this fork changed, compared with upstream at
[`45a0caff`](https://github.com/0xBakeer/deepseek-v41-flash-spark/tree/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f):

| | Upstream | This fork |
| --- | --- | --- |
| Hardware | One DGX Spark | Two DGX Sparks, TP2 over RoCE |
| Expert residency | **Static** keep-set, ranked offline from routing traces (frequency or saliency, per topic) | **Adaptive**: router-score demand history, re-planned after prefill, during long answers, on urgent misses and between requests; persists across restarts |
| Memory budget | Fixed per-layer keep counts | One shared arena; layers trade slots above a six-expert routing minimum, sized by `ARENA_GB` |
| A missing router pick | Replaced by a resident expert, permanently | Replaced by a resident expert for that token; its load is **deferred** to the next adaptation pass |
| Routed expert weights | CB3 requantization in the headline configuration | EXL3 2.9 bpw (default) or the checkpoint's native MXFP4 |
| Launch | `start.sh` on one box | Containerised pair launcher; boot-time guard that both ranks agree on configuration |

The static topic profiles and maxmin ranker were ported here with exact parity, measured, and
removed on 2026-10-17: in a small comparison they did not beat adaptive residency
([record](docs/expert-profiles.md), [gotchas](docs/gotchas.md#the-authors-broad-topic-profiles-are-not-a-demonstrated-quality-default)).

The RoCE all-gather is ported from [TensorFold](https://github.com/ashhart/TensorFold), which
adapted b12x's; the EXL3 expert kernels port TensorFold's ExLlamaV3-derived device code; the
EXL3 checkpoint is [Mia-AiLab's](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).
Full attributions and licences are in [CREDITS.md](CREDITS.md). Repository code is
[MIT-licensed](LICENSE); model use is governed by the checkpoint's own license.
