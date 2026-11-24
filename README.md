# TideLM

**Adaptive expert loading enables running higher quant models on one or two DGX Sparks.** 

TideLM keeps the experts your traffic actually uses resident in the boxes' unified memory, learns which
ones those are from the router's own demand while it serves, and reads the misses that matter
most from NVMe as decoding goes on. The resident set rises and falls with what you ask it — the
tide — so the engine gets better the more you use it.

![The expert map's section view: Inkling-2-Small's 48 layers as slices of a brain-like volume, each expert a cell; token routes climb through it while experts being brought in glow green](assets/expert-map.jpg)

*The engine's live expert map ([`/expert-map`](#running-it)) on the pair serving Inkling-2-Small
(2026-11-20), a few seconds into an answer in French, unlike the traffic before it: each layer is a
section of the volume (layer 0 at the base), each cell an expert, pale where it is in memory. The
routing history is bringing experts in as it goes (768 in the 14 s the page had been open, the green
glows), the misses that matter are read from disk in the background (violet) or rescued inside their
pass (pink), the heaviest of them noted by letter, and 11.9 % of the last 64 tokens' picks missed. The fibres are links between experts that many
tokens share.*

| Profile | Model | Sparks | Routed experts | Resident | Context | Prose | Code | 8K-token prompt, first token |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `ds41-mxfp4` | DeepSeek-V4.1-Flash-1017 | 2 | native MXFP4 | 10,000 / 15,360 (65.1%) | 1M (2 lanes, 1.5M shared) | **33.5 tok/s** | **51.8 tok/s** | 4.3 s |
| `glm53-exlr8-bit322` | GLM-5.3 | 2 | exlr8 `bit322` (3.22 bpw: 2, 3 or 4 bits by layer) | 11,543 / 19,200 (60.1%) | 320K (2 lanes) | **20.4 tok/s** | **25.4 tok/s** | 17.1 s |
| `glm53-exlr8-bit280` | GLM-5.3 | 2 | exlr8 `bit280` (2.80 bpw: 2, 3 or 4 bits by layer) | 13,265 / 19,200 (69.1%) | 320K (2 lanes) | **22.1 tok/s** | **27.6 tok/s** | 17.1 s |
| `glm53-flash-40-tp1` | GLM-5.3-Flash | **1** | EXL3 4 bpw (4 bits a projection) | 7,282 / 12,096 (60.2%) | 320K (2 lanes) | **22.1 tok/s** | **26.4 tok/s** | 12.1 s |
| `glm53-flash-40-tp2` | GLM-5.3-Flash | 2 | EXL3 4 bpw (4 bits a projection) | 12,096 / 12,096 (100%) | 1M (2 lanes, 2M shared) | **36.8 tok/s** | **46.6 tok/s** | 7.6 s |
| `inkling2-small-40-tp2` | Inkling-2-Small | 2 | EXL3 4 bpw (4 bits a projection) | 11,865 / 18,432 (64.4%) | 768K (3 lanes) | **29.4 tok/s** | **42.3 tok/s** | 7.6 s |

*Medians of three `bench/bench.py` runs after a warmup, 1,024 tokens at temperature 0.6, each on
the build of its day (raw runs: [DeepSeek](results/rust-ds41-20261028-pdl/README.md), 2026-10-28;
[GLM-5.3 `bit322`](results/rust-glm53-exlr8-bit322-20261118/README.md), 2026-11-18, and
[`bit280`](results/rust-glm53-exlr8-bit280-20261121/README.md), 2026-11-21; GLM-5.3-Flash
[on one Spark](results/rust-glm53-flash-40-tp1-20261114/README.md) and
[on two](results/rust-glm53-flash-40-tp2-20261114/README.md), 2026-11-14;
[Inkling-2-Small](results/rust-inkling2-40-tp2-20261116-512k/README.md), 2026-11-16, at a 512K context
with 11,967 experts resident: the profile has served 768K with 102 fewer since 2026-11-17, not
measured again).*

GLM-5.3 is served from the `bit322` profile (3.22 bpw: by layer block, 2, 3 or 4 bits a projection) of
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), release
0.7, which re-encoded every expert in the mul1 codebook and replaced the `home` profile served here
before (3.25 bpw, its manifest removed from the release). On this pair `bit322` holds slightly more
experts than `home` did in the same memory (11,543 against 11,430) and decodes as fast on prose and the
8K prompt and 4-6% faster on code. The release measures it close to the unquantized model with every
expert loaded; here 40% of them are not resident, and that is what decides its quality: at fixed
residents its held-out NLL is 1.907 against 1.855 for the older 2.6 bpw `bit260`, which held 2,800 more
experts, in step with the residents each held ([`bit322` on the pair](docs/exlr8.md#bit322-322-bpw-2026-11-18)).
`bit340`, tried first, held 10,946 and was dropped for its answers. The EXL3 2.75 bpw profile before
exlr8 is still served: its setup is in the [install guide](docs/install.md).

`bit280` (2.80 bpw) is the alternative, from the same release and dense weights, and beside `bit322`
it is one more fetch and pack ([`bit280`](#glm-53-exlr8-bit280-glm53-exlr8-bit280)). It holds 13,264
experts in the same memory and decodes 8-9% faster on prose and code (a decode step 3-5% shorter, the
rest the text's tokens a step; the 8K prompt's first token the same). Its held-out NLL at fixed
residents is lower (1.890 against 1.907), while the release measures it further from the unquantized
model with every expert loaded. In use the two were hard to tell apart; `bit322` stays the default
([`bit280` on the pair](docs/exlr8.md#bit280-280-bpw-2026-11-21)).

GLM-5.3-Flash is served from Mia's AI Lab's
[EXL3 4 bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) (every routed
expert at 4 bits) on **one Spark**: its non-expert weights in FP8, 60.2% of its routed experts
resident and the rest read from the pack as decoding needs them, a follow-up turn resumed from the
state its last prompt left (34 of its 45 layers are linear attention, whose state cannot be
rewound), and image input with the vision tower loaded into lent expert slots for the request. The
two-Spark profile holds every expert and decodes about 1.7 times as fast; it is the option, not
tuned - at two boxes the model fits whole, and engines built for that case serve it. The 3.5 bpw
checkpoint served before it kept 8,341 experts resident on one Spark where 4 bpw keeps 7,282, decoded
2-5% faster and scored 0.008 nats better on held-out text (1.7510 against 1.7592, +- 0.0027); the
author finds 4 bpw the smarter of the two all the same, and answers were not scored
([the 4 bpw checkpoint](docs/glm-flash.md#the-4-bpw-checkpoint-2026-11-13-behind-on-one-box-by-the-numbers-shipped-and-recommended-all-the-same)).

Inkling-2-Small (Thinking Machines, 594B total, 17B active) is served from 0xSero's
[EXL3 4 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw) on two Sparks: its 18,432
routed experts at 4 bits are 293 GB for a pair that holds about 190, so 64.4% stay resident, placed
from a routing history the bundled seed starts (its top 64.4% carry 92.7% of the routing weight). The
rest is the GLM engine's: non-expert weights in FP8, the 8 full-attention layers' keys and values in
FP8 (held-out NLL +0.0014 +- 0.0011 against BF16) for a 768K context (the model's own limit is 1M),
image input, the model's own DSpark drafter (7 drafts a round) on the worker, and three lanes, so an
agent's long conversation keeps its lane beside two chats. What the substituted experts cost in answers against the whole model is not measured here
([Inkling-2-Small](docs/inkling.md)).

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
     step, paced over the layers) is read into the slot of the coldest expert while the next
     round drafts, and serves from the round after (`GLM_HOT_ASYNC=1`, the example profiles'). A
     miss carrying ≥ 0.3 is not left to the next round: it is read before its layer's MoE runs
     and computed exactly (`GLM_HOT_RESCUE`; 0.3 since 2026-11-18, at the bench's speed of 0.4). `GLM_HOT_ASYNC=0` reads every one that way, for
     about 5% of the decode speed.
   - GLM-5.3 prefill (`GLM_PF_LOAD`, on by default): per MoE layer and prompt chunk, the missing
     experts carrying at least 4 tokens' worth of the chunk's routing are read into 32 scratch slots
     before the layer's MoE and computed exactly, instead of on substitutes, and leave after the
     layer. On ten 3.5K-token prompts far from the routing history (bit280), the prompts' routing
     weight on substitutes fell from ~30% to ~5% and the loss of the 512 tokens after each from
     1.656 to 1.588 (paired 95% [-0.102, -0.038]; code and maths unchanged), for ~10% more prefill
     time ([pfload.rs](rust/crates/tidelm-glm/src/pfload.rs)).
4. **Two Sparks as one.** Both boxes hold half of every resident expert and run tensor-parallel
   over RoCE; a boot-time guard refuses to start if the two ranks disagree on any setting that
   changes the numbers.

Around that core: speculative decoding with each model's own drafter (DeepSeek's DSpark heads,
GLM's MTP layer, both with a dynamic draft depth, and Inkling's DSpark drafter), prompt caching in
RAM and on disk, FP8 KV caches split across the ranks (GLM), DeepSeek's context caches paged out of
one pool the lanes share (one lane can run to 1M while another holds less; when the pool runs short
the newest request waits on disk and resumes exactly, [details](docs/paged-kv.md)), image input
(DeepSeek, GLM-5.3-Flash on one Spark, Inkling-2-Small), tool calls, thinking and reasoning-effort
control through an OpenAI-compatible API.

## Quick start

You need two DGX Sparks connected by a working RoCE link, Docker with NVIDIA GPU support,
passwordless SSH from the head (rank 0) to the worker, and the weights on local NVMe at the same
path on both nodes - or, for the one-Spark profile (`glm53-flash-40-tp1`), one Spark with Docker. Clone this repository to the same absolute path on both nodes. The
[install guide](docs/install.md) has each step in full: disk per profile, copying between the
nodes, the GLM packs.

There are five profiles, and each is set up on its own below. One image runs them all, one
profile serves at a time (each fills both boxes, or the head), and switching between profiles that
are set up is one `scripts/tide.sh start`.

| Profile | Model | Disk, each node | Head only, while setting up |
| --- | --- | ---: | ---: |
| [`ds41-mxfp4`](#deepseek-v41-flash-ds41-mxfp4) | DeepSeek-V4.1-Flash-1017 | 510 GB | |
| [`glm53-exlr8-bit322`](#glm-53-exlr8-bit322-glm53-exlr8-bit322) | GLM-5.3, exlr8 `bit322` | 170 GB | 305 GB |
| [`glm53-exlr8-bit280`](#glm-53-exlr8-bit280-glm53-exlr8-bit280) | GLM-5.3, exlr8 `bit280` (the alternative) | 150 GB (129 GB beside `bit322`) | 267 GB (107 GB beside `bit322`) |
| [`glm53-flash-40-tp1`](#glm-53-flash-4-bpw-glm53-flash-40-tp1-glm53-flash-40-tp2) | GLM-5.3-Flash, EXL3 4 bpw, one Spark | 176 GB (one node) | |
| [`glm53-flash-40-tp2`](#glm-53-flash-4-bpw-glm53-flash-40-tp1-glm53-flash-40-tp2) | GLM-5.3-Flash, EXL3 4 bpw | 98 GB | 156 GB |
| [`inkling2-small-40-tp2`](#inkling-2-small-inkling2-small-40-tp2) | Inkling-2-Small, EXL3 4 bpw | 165 GB | 294 GB |

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

### GLM-5.3, exlr8 `bit322` (`glm53-exlr8-bit322`)

The `bit322` profile (3.22 bpw: gate and up at 2 bits in layers 3-17 and 4 in 48-72, down at 4 in
38-67, 3 bits elsewhere) of
[`kindlingai/glm-5.3-exlr8-k2-k3-k4`](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), release
0.7 (every expert in the mul1 codebook). The release stores every routed expert at 2, 3 and 4 bits and
a profile names one width a unit, so only the profile's part is fetched (305 GB of 838 GB, on the
head, needed only to pack). What serves is a 149 GB pack of routed experts and 21 GB of other
weights on each node.

**1. Profile.** Copy the example and fill in your paths and the worker's address:

```bash
cp configs/glm53-exlr8-bit322.env.example .env.glm53-exlr8-bit322
```

**2. Fetch**, by HTTP range requests (stdlib Python; resumable; it holds tens of MB and is safe
beside a serving engine):

```bash
python3 tools/exlr8_fetch.py --manifest bit322 --out <exlr8 dir>     # head: 297 GB of experts
python3 tools/exlr8_fetch.py --dense --out <exlr8 dir>               # head: 8.2 GB of dense layers
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
scripts/tide.sh build .env.glm53-exlr8-bit322
```

**4. Packs and weights**, once, on the head: each node's half of the routed experts, and the
combined non-expert weights. The tools need numpy and xxhash (`pip install numpy xxhash`, or
`uv run --with numpy --with xxhash python ...`); every unit read is checked against the release's
checksums, and the codebook is read from the release's tensor names. With `--follow 3600` a pack
runs behind the fetch and both are done when it is.

```bash
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit322 --rank 0 \
    --out <packs dir>/glm-exlr8-bit322-r0of2.bin
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit322 --rank 1 --out - | \
    ssh <worker> 'cat > <packs dir>/glm-exlr8-bit322-r1of2.bin'
python3 tools/glm_exlr8_dense.py --base <base dense dir> --exlr8 <exlr8 dir> --manifest bit322 \
    --out <dense dir>             # the base's shards hard-linked, 6.9 GB of exlr8 matrices beside them
# the worker's <dense dir>: its own copy of the base by hard links, then the new shards and the index
ssh <worker> 'cp -al <base dense dir> <dense dir>'
rsync -a <dense dir>/exlr8-overlay-*.safetensors <dense dir>/model.safetensors.index.json <worker>:<dense dir>/
```

Stream the worker's pack while nothing serves there: written at 150 GB, its page cache can push a
serving worker under the memory guard. Set `GLM_EXL3_DIR=<packs dir>` and `GLM_MODEL_DIR=<dense dir>`
in the profile, at the same paths on both nodes. Once the packs and weights are in place
`<exlr8 dir>` is no longer read. The dense directory must come from the same release as the pack:
the shared experts in it changed codebook with 0.6.

**5. Start:**

```bash
scripts/tide.sh start .env.glm53-exlr8-bit322
```

### GLM-5.3, exlr8 `bit280` (`glm53-exlr8-bit280`)

The release's `bit280` profile (2.80 bpw: gate and up at 2 bits in layers 3-22, down at 2 in 3-12 and
4 in 48-52, 3 bits elsewhere), from the same 0.7 weights as `bit322` and served with the same dense
directory: the two name the same shared-expert widths. On a pair set up for `bit322` it is one more
fetch (the 107 GB the mirror lacks) and a 129 GB pack on each node, with the tools and cautions of
`bit322`'s step 4:

```bash
cp configs/glm53-exlr8-bit280.env.example .env.glm53-exlr8-bit280    # bit322's paths
python3 tools/exlr8_fetch.py --manifest bit280 --out <exlr8 dir>
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit280 --rank 0 \
    --out <packs dir>/glm-exlr8-bit280-r0of2.bin
python3 tools/glm_pack_exlr8.py --source <exlr8 dir> --manifest bit280 --rank 1 --out - | \
    ssh <worker> 'cat > <packs dir>/glm-exlr8-bit280-r1of2.bin'
scripts/tide.sh start .env.glm53-exlr8-bit280
```

On its own it is `bit322`'s five steps with `--manifest bit280` (259 GB of experts to fetch), and
either profile's dense directory serves both. Other profiles of the release (`bit300`, `bit340`, ...)
are the same steps with their `--manifest` and their pack in `GLM_EXL3_PACK`: the fetch adds only
the bytes the mirror lacks.

### GLM-5.3-Flash 4 bpw (`glm53-flash-40-tp1`, `glm53-flash-40-tp2`)

Mia's AI Lab's [EXL3 4 bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold):
every routed expert at 4 bits, its non-expert weights converted to FP8 at load.

**One Spark** (`glm53-flash-40-tp1`): no worker, no RoCE link. Both parts come straight from the
Hub: the non-expert tensors, and the routed experts packed by range requests, so the checkpoint
never lands on disk (numpy; `--resume` continues an interrupted run). `--local-only` builds the
image without shipping it to a worker:

```bash
cp configs/glm53-flash-40-tp1.env.example .env.glm53-flash-40
python3 tools/glm_fetch_dense.py --repo Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold \
    --revision 76c0b5173166d2795dd48860f45d8224817f894c --out <dense dir>       # 19 GB
uv run --no-project --with numpy python tools/glm_pack_exl3.py --world 1 --resume --verify 64 \
    --repo Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold --revision 76c0b5173166d2795dd48860f45d8224817f894c \
    --out <packs dir>/glm-flash-40-r0of1.bin                                      # 156 GB
scripts/tide.sh build --local-only .env.glm53-flash-40
scripts/tide.sh start .env.glm53-flash-40
```

Set `GLM_MODEL_DIR=<dense dir>` and `GLM_EXL3_DIR=<packs dir>` in the profile. Residency starts from
the bundled routing history (`profiles/glm53-flash-history-v1.npz`, a day of the author's serving)
and learns from your traffic into `RUST_GLM_STATE_DB` (to start over from the seed, see
[Resetting the routing history](#resetting-the-routing-history)). Images in chat requests
(`image_url` parts) are on (`GLM_VISION=1`).

Speculative drafts are scored over a listed vocabulary of 57,344 common ids (English, code and
Japanese) rather than all 154,880 - 3-8% more tokens a second on English prose and code, the output
the same either way. Text in another language drafts over every id as before - to give
it a list of its own, `tools/build_glm_draft_vocab.py` ranks your text and `GLM_DRAFT_VOCAB=<file>`
loads it ([GLM-5.3-Flash](docs/glm-flash.md#a-listed-draft-vocabulary-2026-11-13)).

**Two Sparks** (`glm53-flash-40-tp2`): every routed expert resident (each rank holds its half of
all 12,096), no image input. It starts from the one-Spark setup's weights on the head: the dense
dir to the worker and each rank's pack re-sliced from the one-box pack (78 GB a rank):

```bash
cp configs/glm53-flash-40-tp2.env.example .env.glm53-flash-40-tp2
rsync -a <dense dir>/ <worker>:<dense dir>/
uv run --no-project --with numpy python tools/glm_pack_exl3.py --from-pack <packs dir>/glm-flash-40-r0of1.bin \
    --world 2 --rank 1 --out <packs dir>/glm-flash-40-r1of2.bin --verify 32
rsync -a <packs dir>/glm-flash-40-r1of2.bin <worker>:<packs dir>/ && rm <packs dir>/glm-flash-40-r1of2.bin
uv run --no-project --with numpy python tools/glm_pack_exl3.py --from-pack <packs dir>/glm-flash-40-r0of1.bin \
    --world 2 --rank 0 --out <packs dir>/glm-flash-40-r0of2.bin --verify 32
scripts/tide.sh build .env.glm53-flash-40-tp2     # builds the image and ships it to the worker
scripts/tide.sh start .env.glm53-flash-40-tp2
```

The one-box pack can go afterwards, unless the one-Spark profile stays set up. Coming from the 3.5
bpw profiles, your learned routing history carries over (it is the router's, not the experts'):
copy `glm-flash-stats.npz` to the name the new profile's `RUST_GLM_STATE_DB` gives before the first
start.

### Inkling-2-Small (`inkling2-small-40-tp2`)

0xSero's [EXL3 4 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw) of
[Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small): every routed expert at 4
bits, everything else the source's BF16. Two Sparks. Copy the profile, set its paths and `PEER`,
and let `scripts/inkling2-setup.sh` fetch and lay out the weights from the head:

```bash
cp configs/inkling2-small-40-tp2.env.example .env.inkling2-40-tp2
scripts/inkling2-setup.sh .env.inkling2-40-tp2    # --check: what is done and left, changing nothing
scripts/tide.sh build .env.inkling2-40-tp2
scripts/tide.sh start .env.inkling2-40-tp2
```

The script mirrors the repository at a pinned revision (312 GB on the head; standard-library
Python holding tens of MB, so it can run beside a serving engine) and checks every file's sha256;
makes `GLM_MODEL_DIR` from the non-expert shards, the drafter (`mtp.safetensors`) and the tokenizer
as hard links into the mirror (17.4 GB) and copies it to the worker; then packs each rank's experts
(147.4 GB, rank 1's sent to the worker; uv or numpy). A step whose result is there is skipped, so an
interrupted run continues by running it again. When it is done the mirror can go - the folder's
links keep their files - and the script says so rather than deleting it.

The first boot places its experts from the bundled seed (`profiles/inkling2-history-v1.npz`, a day
and a night of the development pair's traffic, [what it is](profiles/inkling2-history-v1.json));
from then on the engine's own history (`RUST_GLM_STATE_DB`) takes over.

### Running it

```bash
scripts/tide.sh start .env.glm53-exlr8-bit322   # stops whatever is serving, starts this profile
scripts/tide.sh logs -f                         # its log (--rank 1: the worker's)
scripts/tide.sh stop
```

The model follows from the profile (`GLM_MODEL_DIR` → GLM-5.3, GLM-5.3-Flash or Inkling-2-Small,
by its config; `MODEL_DIR` → DeepSeek), and the Rust engine serves all four; the profiles carry only
the settings it reads. It serves the OpenAI API with tool calls (and images: DeepSeek, GLM-5.3-Flash
on one Spark, Inkling-2-Small) and the Anthropic route for Claude Code. It is up 15 s (DeepSeek) to
40-60 s (GLM-5.3, GLM-5.3-Flash, Inkling-2-Small) after the previous one has stopped; the first
requests after a boot are slower while adaptation catches up with your traffic. One profile serves
at a time: each fills both boxes, or the head. `TIDE_ENGINE=python` still starts the Python containers, from a profile with
their settings (the examples as of `a6ec917`), and a profile the Rust engine does not serve
(MXFP4 GLM experts, FP4 KV) goes to them with a note.

`http://<head>:8000/expert-map` shows the residency and the routing as they happen, for every
profile. Its section view draws each layer as one slice of a brain-like volume (layer 0 at the base)
and each expert as a cell in it, pale where resident and dark where not, the most used at the core
(or scattered by expert number); drag turns it, the wheel zooms, and a focal-plane gauge beside it
brings a layer and its neighbours into focus. The grid view shows the same thing a row a layer, and
either view can colour the cells by how much the routing history uses them. On GLM-5.3,
GLM-5.3-Flash and Inkling-2-Small every decoded token's routing is drawn as well: its route climbs
the layers, its picks warm their cells (pink where they missed), and the links many tokens share
stay on as fibres for 5, 15 or 60 s. Residency events replay at their own pace: a read after a miss
(violet), an inline rescue (pink) and a load by the routing history or for a continuing conversation
(green) light their cell, with a faint arc from the expert whose slot it took and lettered notes on
the heaviest misses. The key beside the field traces each kind on it, the observation log lists every
token and event, and a cell's tooltip tells its story. The section is drawn with WebGL (the grid
works without it), and the motion is off when the system asks for reduced motion. The page polls
`GET /v1/expert-map` five times a second, asking only for what it has not seen (`since`,
`events_since`; the usage rows every 2 s): residency, counters, the last seconds' residency events
and token routes (each a token's top-k picks a layer, their shares, which ones missed). It is open
without an API key, like the page, but a route's token text is the conversation's and goes only to a
request that carries a key. The engine copies the routes from the async hot reads' table, already on
the host after each decode pass, answers between its steps, and does nothing for the page when none
is asking.

### Updating

```bash
git pull                                   # on both nodes for a two-Spark profile
scripts/tide.sh build .env.glm53-exlr8-bit322     # the image again (--local-only on one Spark)
diff .env.glm53-exlr8-bit322 configs/glm53-exlr8-bit322.env.example
scripts/tide.sh start .env.glm53-exlr8-bit322
```

The engine refuses an image built from other sources than the checkout, so every `git pull` needs
the build. New settings and changed defaults arrive in `configs/*.env.example`, not in your copy:
the `diff` shows what to carry over (keep your own paths). The learned routing history is kept
across updates.

### Resetting the routing history

A GLM profile places its resident experts from a routing history the engine keeps learning into
(`RUST_GLM_STATE_DB`; GLM-5.3's default `~/tidelm-rust-state/glm-stats-rust.npz`). The bundled seed
(`GLM_STATS_DB`) is read only on a boot that finds no such file. To start over from the seed - a
new seed, or traffic that has changed:

```bash
grep GLM_STATS_DB configs/glm53-flash-40-tp1.env.example   # the current seed: your profile must name it
scripts/tide.sh stop
mv ~/tidelm-rust-state/glm-flash-40-stats.npz ~/tidelm-rust-state/glm-flash-40-stats.old.npz
scripts/tide.sh start .env.glm53-flash-40
```

The boot log then reads `boot: routing history /app/profiles/...` (the seed) instead of your state
file, and the first requests are slower while adaptation catches up. Moving the old file back
restores it. GLM-5.3-Flash installs from before 2026-11-13 name seed v0 in their profile: set
`GLM_STATS_DB=/app/profiles/glm53-flash-history-v1.npz` and update the image first. GLM-5.3's
profiles name no seed: the launcher picks the current one (`profiles/glm53-experts-v2.npz` since
2026-11-13, [what it is](docs/adaptive-experts.md#glm-53)), so there the `mv` of your profile's
`RUST_GLM_STATE_DB` and a restart are the whole reset. Inkling-2-Small's profile names its seed
(`profiles/inkling2-history-v1.npz`) and its state file (`inkling2-40-tp2-stats.npz` in the example).

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
(`scripts/tide.sh start .env.glm53-exlr8-bit322`); the log shows
`[glm] ablit: 48 layers (2-49), alpha 3.000-3.000, from ... (sha256 ...)`:

```bash
GLM_ABLIT=/home/you/models/glm53-ablit/glm53-derisked-l2-49.safetensors
```

As with the DeepSeek overlay, both ranks refuse to pair over different files (the sha256 is in the
boot guard), and the disk prefix store keeps the abliterated model's prompts apart from the stock
model's: the first request of a conversation after switching prefills its whole prompt again. On
held-out text (`corpus/heldout_quick.jsonl`, 10,151 tokens) it costs NLL +0.018 +- 0.005 (+1.0%)
and top-1 60.22 -> 59.95% (measured on the EXL3 profile; not yet on `glm53-exlr8-bit322`).

## Use

- API: `http://<head>:8000/v1` (OpenAI-compatible chat completions, streaming included). Model
  names: `deepseek`, `glm-5.3`, `glm-5.3-flash` and `inkling-2-small` (`SERVED_MODEL_NAME`).
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
| `RUST_BUDGET=10000` | `GLM_SLOTS=12400` | How many experts stay resident (GLM: expert memory a rank, in 3-bit experts' worth, 7.1 MB each - a budget, not a count: 11,543 of `bit322`'s, whose 4-bit records are larger). Fewer, more memory for context. |
| `MAX_SEQ=1048576`, `RUST_KV_TOKENS=1572864` | `GLM_MAX_SEQ=327680` | Context, generated tokens included (DeepSeek: the most one lane reaches, out of a pool of `RUST_KV_TOKENS` the lanes share). Less context, more experts. |
| `DSV41_ADAPT_SENSITIVITY=high` | `GLM_ADAPT=1`, `GLM_ADAPT_EVERY=200` | How fast the resident set follows new traffic. |
| `RUST_ADAPT_DB` | `RUST_GLM_STATE_DB` | The learned history, under `~/tidelm-rust-state`. With none yet, the first boot starts from the bundled one in `profiles/`. |
| `DSV41_HOT_READS=1`, `DSV41_HOT_ASYNC=1`, `DSV41_HOT_RING=64` | `GLM_HOT=1`, `GLM_HOT_KEEP=32` | Hot reads (above). The ring's slots are on top of `RUST_BUDGET`, 9.4 MB a rank each (0.56 GiB at 64): a bigger ring needs a smaller budget. |
| `RUST_LANES=2` | — | Concurrent requests; the lanes share the one hot-read ring. |
| `SPEC=1`, `RUST_DEPTH=confidence` | `GLM_MTP=1`, `GLM_MTP_DEPTH=7`, `GLM_MTP_CAL=0.36` | Speculative decoding and its depth policy. |
| `DSV41_PREFIX_CACHE=1`, `DSV41_PREFIX_DISK=1` | `GLM_PREFIX_DISK=1`, `GLM_KV_SPLIT=1` | Prompt caching; GLM's KV cache split across the ranks. |
| — | `GLM_KV=fp8` | GLM's KV cache format, the one the Rust engine serves. The Python engine's 4-bit latent with the first and last ten layers in FP8 fits 320K in less memory and measured KL +0.004 against FP8 on the long eval, but in use the model's recall of its context was visibly worse. |
| — | `GLM_EXLR8_ROUND=exact` | exlr8's codebook rounded to FP8 exactly as the format defines it, instead of in two instructions (the default: a decode pass ~2% shorter, held-out NLL unchanged). |
| — | `GLM_PF_LOAD=4`, `GLM_PF_LOAD_SLOTS=32` | Prefill loads (above), GLM-5.3's default; `GLM_PF_LOAD=0` turns them off. The scratch slots come out of `GLM_SLOTS` (38 fewer residents on `bit280`). |
| — | `GLM_LOOP_BREAK=12` | Stops a short cycle of tokens from repeating a 12th time (repetition loops start where experts are substituted). |

## Documentation

- [Install guide](docs/install.md): each setup step in full
- [Hot reads: design and measurements](docs/hot-reads.md)
- [Adaptive expert loading](docs/adaptive-experts.md), [GLM-5.3 engine plan](docs/glm53-plan.md),
  [exlr8 on the pair](docs/exlr8.md), [GLM-5.3-Flash](docs/glm-flash.md),
  [Inkling-2-Small](docs/inkling.md) and [its drafter](docs/inkling-dspark.md)
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
EXL3 GLM checkpoints by bertholomus and Mia's AI Lab, the exlr8 GLM release by KindlingAI,
Inkling-2-Small by Thinking Machines and its EXL3 checkpoint by 0xSero. Full attributions and licences: [CREDITS.md](CREDITS.md).

Repository code is [MIT-licensed](LICENSE). Model use is governed by each checkpoint's own license.
