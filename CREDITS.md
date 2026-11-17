# Credits

This recipe is a thin layer over other people's work plus a lot of arithmetic. What is ours
is the single-box design — the expert arena with its LRU and transient ring, the `O_DIRECT`
streaming path, the Triton FP4 grouped-MoE kernel, the chunk-invariant port, the stdlib
server, the routing trace and every measurement.

## The model

**[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** — DeepSeek-V4.1-Flash:
the architecture (Engram n-gram memory, hyper-connections, CSA2/CED sparse attention, the
FP4 MoE, the DSpark drafter), the weights, and the reference implementation under
`inference/` that this engine's math is ported from — `model.py`, `engram.py`, `kernel.py`
and the chat encoder in `encoding/`, which the server calls directly rather than
reimplementing. The tech report is where the numbers this repo quotes for global KV size
and Decoder SWA Bounded Replay come from. The weights carry DeepSeek's licence; read it
before deploying commercially.

Also DeepSeek's, and read while designing this: **`deepseek-recipe`** (the protocol and
chat-template layer, whose `reasoning_effort` string mapping the server's own mapping
follows), **`DeepSelect`** (the indexer top-k kernel — `sm_100a`/`sm_103a` only, which is
part of why this box needs its own path) and **`DeepJIT`**.

## GLM-5.3

**[Z.ai](https://huggingface.co/zai-org/GLM-5.3)** -- GLM-5.3 (754B total, `glm_moe_dsa`): the
architecture (MLA attention with the DSA sparse indexer, 256-expert MoE with one shared expert,
the MTP layer the engine drafts with), the weights, the tokenizer and chat template the server
renders with. The weights carry Z.ai's GLM-5.3 licence; read it before deploying.

**[AMD](https://huggingface.co/amd/GLM-5.3-Quark-MXFP4-AttnFP8)** -- the MXFP4 checkpoint both GLM
profiles load (`amd/GLM-5.3-Quark-MXFP4-AttnFP8`, quantized with AMD-Quark): MXFP4 experts, FP8
attention. **bertholomus** -- the EXL3 3.0 bpw checkpoint
(`bertholomus/GLM-5.3-EXL3-3.0bpw`) that `tools/glm_pack_exl3.py` repacks per rank for the EXL3
profile.

The GLM engine (`engine/glm/`) began as the Muse Spark Open experiment (`engine/muse/`), whose
TP2 runtime, checkpoint readers and server it still builds on.

## Inkling-2-Small

**[Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Small)** -- Inkling-2-Small
(594B total, 17B active, `inkling2`): the architecture (sliding-window and full attention with short
convolutions and a block attention residual, 384-expert MoE, the vision tower, the DSpark drafter in
`mtp.safetensors`), the weights, the tokenizer and chat template the server renders with. Apache-2.0.
**[0xSero](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw)** -- the EXL3 4.0 bpw checkpoint
(mcg codebook, every expert at 4 bits, everything else the source's) that `tools/glm_pack_exl3.py`
repacks per rank, and [inkling2-exl3](https://github.com/0xSero/inkling2-exl3), an engine for it.

The engine's forward follows Hugging Face transformers' `modeling_inkling2.py` (main branch, 2026-11-16)
and its drafter SGLang's DSpark (branch `inkling-2-small-support`, `ea3742849fa4`); both Apache-2.0.
`tools/inkling_ref.py`, `tools/inkling_draft_ref.py` and `tools/inkling_vision_ref.py` transcribe them in
plain PyTorch for the parity checks ([Inkling-2-Small](docs/inkling.md), [its drafter](docs/inkling-dspark.md)).

## Where the techniques come from

Both of the load-bearing ideas here were built and measured in earlier recipes of mine on
the same class of box, and this repo is the third iteration of them.

**[qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)** — the
**NVMe-resident table** pattern: keep a table that does not fit in unified memory on the SSD,
map it, and let the page cache serve the rows a step actually touches instead of loading the
whole thing. That is exactly the shape `engine/engram.py` takes for V4.1's 203 GB of Engram
n-gram tables — 48 random 264-byte rows per token, read with buffered `preadv` in a thread
pool under `POSIX_FADV_RANDOM`, with a small process-local row cache in front. It is also
where the idea generalises from: an expert is the same problem at 18.8 MB instead of 264
bytes, which is what `engine/experts.py` does with `O_DIRECT` and an explicit arena.

**[ling3-flash-spark](https://github.com/0xBakeer/ling3-flash-spark)** — the recipe shape
this one copies: `start.sh`/`stop.sh` with the memory and port guards, the `run.sh`
dispatcher and container split, the GHCR workflow on `v*` tags, the "a version is a
measurement epoch" changelog, the `results/` layout, and the benchmark harness with its four
rules (usage-based token counts, fresh verified prompts, label-salted seeds, `ignore_eos`).
`bench/bench.py` here is an adaptation of that harness, so rows from the two are directly
comparable.

## The one-shot RoCE all-gather

`tools/roce/` is a port of the RoCE all-gather in **[TensorFold](https://github.com/ashhart/TensorFold)**
(patch 0230), which in turn adapted **b12x**'s "RoCEnante"
(https://github.com/local-inference-lab/b12x, `b12x/comm/roce`) — Copyright 2026 Luke Alonso and the
b12x contributors, and the TensorFold contributors, Apache-2.0. It replaces NCCL for the decode-sized
all-gathers: on this pair a captured NCCL all-gather costs 44-70 us whatever the payload, and this costs
12.7-19.9 us with identical bits (`tools/bench_roce_gather.py`). The full notice is in
`tools/roce/NOTICE`; the only changes are the `GLM53_TF_*` -> `DSV41_*` knobs and the engine integration.

## EXL3 routed experts

`tools/exl3_format.py` is vendored, unmodified, from **TensorFold**'s
`src/tensorfold/cuda/exl3/format.py` (Apache-2.0, the TensorFold contributors), which is the
MIT-licensed **ExLlamaV3** trellis format (Copyright (c) 2025 Turboderp). It is the bit-exact
numpy oracle for `tools/test_exl3_ref.py` and is not on the serving path. The torch decoder
(`tools/exl3_ref.py`), the pack format and `tools/pack_exl3_experts.py` are this repo's.

The CUDA expert kernels (`tools/exl3_moe_cuda.cu`) port the device arithmetic of TensorFold's
ExLlamaV3-derived `decode.cuh`, `experts_grouped.cuh` and `experts.cu` (same licences): the tile
decode, codebook, mma and 128-point Walsh-Hadamard butterfly. The launchers, arena-slot routing,
the prefill tile kernel and the C ABI are this repo's. TensorFold's DeepSeek-V4.1 fast load
path (`x3ld.cu`/`loads.py`, "our GLM patch 0580") is deliberately **not** ported: its lineage is
the Mia's AI Lab GLM kit after 2026-09-07 (AGPL-3.0), which this repo takes no code from.

TensorFold's prepared EXL3 packs are for its "dsv41-uncensored-2.9bpw" model; we build our own
pack (`scripts/exl3-packs.sh`) from the base checkpoint
[`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw),
quantized with ExLlamaV3 from `deepseek-ai/DeepSeek-V4.1-Flash`.

## exlr8 routed experts and dense matrices

The default GLM-5.3 profile serves the `bit260` profile of **KindlingAI**'s
[`kindlingai/glm-5.3-exlr8-k2-k3-k4`](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), a
quantization of Z.ai's GLM-5.3 in their exlr8 v2 format (the weights under the GLM-5.3 License, the
packing and manifests also CC BY 4.0). The format is ExLlamaV3's trellis (MIT, Copyright (c) 2025
Turboderp) with the codebook rounded to FP8 E4M3 and the weights stored in units that decode on
their own. `tools/exlr8_ref.py`, the fetcher, the packers and the kernels' codebook
(`tools/exl3_moe_cuda.cu` `e4m3_round2`) are this repo's, written from the release's model card and
[`kindlingai/exlr8`](https://github.com/kindlingai/exlr8)'s `FORMAT.md`; no code of theirs is
vendored, and their reference loader is not run by anything here. The engine's own choices - the
EXL3 record layout the units are regrouped into, the two-instruction rounding it defaults to, the
EXL3 checkpoint's weights where exlr8 stores 8 bits - are not the format's (docs/exlr8.md).

## The platform and the tools

**NVIDIA** — the DGX Spark / GB10, the CUDA 13 container images, and PyTorch's cu130
aarch64 wheels, without which none of this runs on arm64.

**[OpenAI Triton](https://github.com/triton-lang/triton)** — the kernel language the FP4
grouped-MoE forward is written in, and the JIT that compiles it for `sm_121a` on the box.

Assembled, measured and documented by Khaled Bakeer (0xBakeer).
