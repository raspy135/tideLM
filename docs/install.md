# Install

The README's quick start in full: two DGX Sparks serving DeepSeek-V4.1-Flash-1017 or GLM-5.3
with the Rust engine, in Docker. One profile serves at a time, using both boxes.

## What you need

| what | why |
|---|---|
| two DGX Sparks (GB10, 128 GB unified memory each), DGX OS / Ubuntu 24.04, CUDA 13 driver | the pair splits every layer between the two GPUs (tensor parallel) |
| a working RoCE link between them (the 200G cable) | every decode step gathers across it; the profiles name its interfaces (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`) |
| passwordless SSH from the head (rank 0, which serves HTTP) to the worker (rank 1) | the launcher starts the worker's container and ships it the image |
| Docker with the NVIDIA container toolkit on both | the engine is built and run inside the image; the boxes need nothing else |
| this repository at the same absolute path on both | the launcher mounts the same host paths on both nodes |
| local NVMe, enough free on both (below) | experts not resident in memory are read from disk with `O_DIRECT` during decode; a network filesystem makes every such read a network round trip |
| the boxes to themselves | GPU and host share one memory pool. The server is sized to leave `KEEP_FREE_GB` (4) free, and `scripts/memguard.sh` stops it rather than let the box run out: another heavy process, a large download included, can get it stopped |

Disk, per node:

| profile | both nodes | head only, while setting up |
|---|---|---|
| `ds41-mxfp4` | 510 GB checkpoint | |
| `glm53-exl3` | 15 GB of non-expert weights + a 128 GB expert pack | the 293 GB checkpoint, read only to pack, then deletable (+128 GB for the worker's pack if you pack both on the head) |

## 1. Profile

```bash
cp configs/ds41-mxfp4.env.example  .env.ds41-mxfp4     # DeepSeek-V4.1-Flash-1017
cp configs/glm53-exl3.env.example  .env.glm53-exl3     # GLM-5.3
```

Set the paths (`/home/you/...`), `PEER` (the worker's `user@address`) and, in the DeepSeek
profile, the RoCE interfaces. Both files are private: keep them out of Git (`.env*` is ignored).
Each key is explained where it is set.

## 2. Weights

### DeepSeek-V4.1-Flash-1017

```bash
MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017 MODEL_DIR=<dir> bash scripts/download-model.sh
```

on both nodes, or on the head and copy it across (below). Resumable; `HF_TOKEN` is optional. Set
`MODEL_DIR=<dir>` in the profile.

### GLM-5.3

The profile combines two published quantizations of the same GLM-5.3: the routed experts of
[`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold)
(each expert at 2-4 bits, 2.75 on average) and the non-expert weights of
[`bertholomus/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw)
(EXL3, ~5-6 bits; Mia's checkpoint has them in BF16, twice the memory). Why, and what it
measured: [gotchas](gotchas.md), "GLM-5.3 with Mia's 2-4-bit experts".

On the head, the checkpoint (`pip install -U huggingface_hub` gives `hf`):

```bash
hf download Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold --local-dir <ckpt dir>
```

On both nodes, the non-expert weights: 15 GB of the 3.0bpw checkpoint, fetched by HTTP range
requests (stdlib Python, re-run to resume). This directory is `GLM_MODEL_DIR`:

```bash
python3 tools/glm_fetch_dense.py --out <dense dir>
```

The expert packs are built in step 3, from the image.

### Copying between the nodes

One `rsync` is limited by its SSH cipher (~470 MB/s on these boxes); eight in parallel moved
~2.6 GB/s:

```bash
cd <dir> && ls | split -n r/8 - /tmp/files. && \
  for f in /tmp/files.*; do rsync -a --files-from=$f . <peer>:<dir>/ & done; wait
```

## 3. Image, then the GLM packs

```bash
scripts/tide.sh build .env.glm53-exl3    # either profile: builds the image here and ships it to PEER
```

The Rust engine is compiled inside the build. After every `git pull` run it again: the launcher
refuses an image built from other sources than the checkout.

GLM only: pack each node's half of the routed experts once, as your own user (the packer reads
the checkpoint and writes one file per rank; ~10 minutes each):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v <ckpt dir>:/src:ro -v <packs dir>:/packs \
  -v "$PWD":/w -w /w --entrypoint python tidelm:local tools/glm_pack_exl3.py \
  --source /src --out /packs/glm-exl3-experts-r0of2.bin --rank 0
# the same with r1of2 / --rank 1 for the worker: here, then copy the 128 GB file to the
# worker's <packs dir>; or copy the checkpoint across and run it there
```

Set `GLM_EXL3_DIR=<packs dir>` in the profile; the pack must sit at the same path on both nodes.
Once both packs are in place the checkpoint is no longer read. bertholomus's whole 3.0bpw
checkpoint also works with the same packer: its experts at 3 bits (12,500 resident in the same
memory, against 13,300), its own directory as `GLM_MODEL_DIR`.

## 4. Start

```bash
scripts/tide.sh start .env.glm53-exl3    # stops whatever is serving, starts this profile
scripts/tide.sh logs -f                  # one line a request: prefill and decode speeds (--rank 1: the worker)
scripts/tide.sh stop
```

The engine is up 15 s (DeepSeek) to 40 s (GLM-5.3) after the previous pair has stopped. A
first boot starts the routing history from the bundled one in `profiles/` and keeps its own
under `~/tidelm-rust-state`; the first requests are slower while it adapts to your traffic.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
```

The API is OpenAI-compatible (tool calls; images with DeepSeek) with an Anthropic Messages
route for Claude Code ([OpenAI API](openai-api.md)). It binds `HOST` (0.0.0.0) with no
authentication: keep it on a trusted network.

Next: [architecture](architecture.md) · [benchmarking](benchmarking.md) · [gotchas](gotchas.md)

## The single-box Python engine

The original recipe (one Spark, the Python engine, `run.sh` / `compose.yaml`) is described in
this file as of commit `21b7718`; the profiles' Python settings are in `configs/` as of
`a6ec917`. `TIDE_ENGINE=python` in a profile still starts the Python containers on the pair.
