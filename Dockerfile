# DeepSeek-V4.1-Flash on one DGX Spark (GB10 / sm_121a) -- serving image.
#
# The engine uses PyTorch, Triton, and native CUDA kernels compiled on first use, so the image
# needs no ahead-of-time extension build. DSV41_FP4_CUDA=1 (default) JITs tools/fp4_moe_cuda.cu
# into a small C-ABI CUDA shared library on first decode. DSV41_FP4_CUDA_RELAXED=1 selects the
# faster reduction order by default; either setting can be explicitly disabled.
#
# The weights are NOT in the image and never will be: 510 GB. Mount them at /models
# (scripts/download-model.sh fetches them) and give the container the whole box.
#
# Build it on an arm64 machine -- normally GitHub's ubuntu-24.04-arm runner via
# .github/workflows/image.yml, or the Spark itself:
#
#   docker build -t deepseek-v41-flash-spark .
#
# ---------------------------------------------------------------------------------------
# Why the *devel* base and not -runtime
# ---------------------------------------------------------------------------------------
# Triton compiles tools/fp4_moe.py at run time, which means ptxas has to run inside this
# container, and it has to be a ptxas that knows `sm_121a` and the FP4 decode instruction
# the kernel emits as inline PTX (`cvt.rn.f16x2.e2m1x2`). Triton wheels do bundle a ptxas
# under triton/backends/nvidia/bin/, but that bundled copy has shipped behind the driver
# before -- the CUDA-12.8 ptxas in the triton 3.5 wheels could not name sm_121 at all
# (pytorch/pytorch#163801) -- and a recipe that fails at the first token because of a
# vendored compiler is not worth the ~2 GB the devel layer costs. So: the CUDA 13 devel
# image, and TRITON_PTXAS_PATH pinned at its ptxas. If you would rather trust the wheel,
# swap the base for nvidia/cuda:13.0.2-runtime-ubuntu24.04 and drop TRITON_PTXAS_PATH;
# everything else in this file is unchanged. Only the opt-in native FP4 experiment needs nvcc.
#
# Three stages: `base` (CUDA, the RDMA stack, PyTorch), `rust` (base plus a Rust toolchain:
# builds the Rust engine's tidelm-server), and the serving image (base, this repo, and that one
# binary; no toolchain). See the `rust` stage below.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS base

ARG TORCH_VERSION=2.13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH

# python3.12 is Ubuntu 24.04's system interpreter and has a torch cu130 aarch64 wheel
# (cp312); curl is the healthcheck, procps is for looking at the box from inside.
#
# python3-dev is not optional and not only a build-time convenience: Triton JIT-compiles
# its CUDA driver shim (cuda_utils.c, which includes Python.h) on the FIRST kernel call,
# at run time, inside this container. A host without the headers cannot start this engine
# at all -- which is the state both Sparks are in, and a large part of why the container
# path is worth having.
#
# The RDMA userspace stack is what makes the dual-Spark (EP2) mode work:
#   libibverbs1        the verbs library NCCL dlopens for the IB/RoCE transport
#   libibverbs-dev      headers + the libibverbs.so symlink, so the one-shot RoCE all-gather
#                      (tools/roce/) can build its torch extension inside the container
#   ibverbs-providers  the mlx5 provider -- WITHOUT it verbs opens zero devices and NCCL
#                      silently falls back to TCP sockets, which is ~30x slower per
#                      collective (1.9 ms vs 60 us measured) and blows the Gate G0 budget
#   librdmacm1         connection management used during NCCL's IB setup
#   ibverbs-utils      ibv_devinfo / ibv_devices, for telling "no RoCE" from "wrong GID"
#                      from inside the container instead of guessing
#   iproute2           `ip`, used by scripts/roce_gid.sh to find the link's local address
# The kernel side (/dev/infiniband, /sys/class/infiniband) comes from the host at run
# time -- see scripts/dual-up.sh for the device and capability flags that expose it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ca-certificates curl procps \
        libibverbs1 libibverbs-dev ibverbs-providers librdmacm1 ibverbs-utils iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip wheel setuptools

# PyTorch for CUDA 13.0. The cu130 index is the only place with an aarch64 build, and it
# is where `triton` comes from too -- torch pulls it as a dependency, so the Triton the
# kernel JITs against is always the one this torch was tested with. Verified present:
# torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl.
RUN pip install "torch==${TORCH_VERSION}+cu130" --index-url https://download.pytorch.org/whl/cu130

# Everything else from PyPI. safetensors reads the checkpoint headers (the engine parses
# the shard headers itself and then reads spans with O_DIRECT); tokenizers/transformers
# load tokenizer.json; sympy and numpy are used by the reference math in tools/v41_ref.py;
# huggingface_hub is only for scripts/download-model.sh. ninja builds the RoCE all-gather's
# torch extension (tools/roce/) at first use inside the container.
RUN pip install \
        "transformers>=4.57" \
        "tokenizers>=0.21" \
        "safetensors>=0.5" \
        "numpy>=1.26" \
        "sympy>=1.13" \
        "huggingface_hub>=0.35" \
        "ninja>=1.11" \
        "Pillow>=10.0"

# NCCL where the Rust engine links it and finds it at run time (its rpath): the library torch
# brought above, the one the Python engine runs, with the unversioned name the linker wants.
RUN n=/opt/venv/lib/python3.12/site-packages/nvidia/nccl \
    && mkdir -p /opt/nccl/lib && ln -s $n/include /opt/nccl/include \
    && ln -s $n/lib/libnccl.so.2 /opt/nccl/lib/libnccl.so.2 && ln -s libnccl.so.2 /opt/nccl/lib/libnccl.so

# ---------------------------------------------------------------------------------------
# The Rust engine (rust/), the default for every profile scripts/tide-up.sh serves: built here
# so a box needs nothing but Docker. Its build scripts compile the repo's CUDA and C++ sources
# with this image's nvcc for sm_121a (rust/crates/tidelm-kernels, tidelm-comm), libjpeg-turbo
# from source (cmake), and link this image's CUDA libraries, ibverbs and the NCCL above. The
# cache mounts keep the crate downloads and the build tree between builds, so a source change
# rebuilds what it touches. RUSTUP_TOOLCHAIN pins the compiler over rust-toolchain.toml's
# "stable".
# ---------------------------------------------------------------------------------------
FROM base AS rust
ARG RUST_VERSION=1.99.1
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
ENV PATH=/root/.cargo/bin:$PATH \
    RUSTUP_TOOLCHAIN=${RUST_VERSION} \
    NCCL_HOME=/opt/nccl
WORKDIR /src
COPY rust rust
COPY tools/roce/roce.cu tools/roce/roce_common.h tools/roce/
COPY tools/fp4_moe_cuda.cu tools/exl3_moe_cuda.cu tools/
COPY engine/engram_gather.cpp engine/draft_vocab_ranking_v1.txt engine/
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/rust/target \
    cd rust && cargo build --release --locked --bin tidelm-server \
    && cp target/release/tidelm-server /usr/local/bin/tidelm-server

FROM base
ARG RECIPE_VERSION=0.1.0-wip

WORKDIR /app
COPY . /app
COPY --from=rust /usr/local/bin/tidelm-server /usr/local/bin/tidelm-server
RUN chmod +x /app/scripts/*.sh /app/run.sh 2>/dev/null || true \
    && echo "${RECIPE_VERSION}" > /app/VERSION \
    && mkdir -p /models /app/logs /app/results /app/.triton

# Build-time sanity: the imports the engine does at start-up must work, and the Triton
# kernel module must at least parse. It cannot be *run* here -- there is no GPU on a
# build runner -- so the first real compile still happens on the box.
RUN python -c "import torch, triton, transformers, tokenizers, safetensors, numpy, sympy; \
print('torch', torch.__version__, 'triton', triton.__version__)" \
    && python -c "import sys; sys.path.insert(0, '/app/tools'); import fp4_moe; print('fp4_moe ok')" \
    && ! ldd /usr/local/bin/tidelm-server | grep 'not found'

LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash on one DGX Spark" \
      org.opencontainers.image.description="Plain-PyTorch engine with a Triton FP4 MoE kernel and NVMe expert streaming" \
      org.opencontainers.image.version="${RECIPE_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV MODEL_DIR=/models/DeepSeek-V4.1-Flash \
    MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash \
    SERVED_MODEL_NAME=deepseek-v4.1-flash \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_SEQ=32768 \
    DEFAULT_THINKING=off \
    DEFAULT_EFFORT=75 \
    SPEC=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CACHE_DIR=/app/.triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/.hf \
    PYTHONUNBUFFERED=1

# EP2 (dual Spark) defaults. WORLD_SIZE=1 is the single-box image, byte for byte the
# behaviour this file had before dual existed; scripts/dual-up.sh overrides these per rank.
ENV WORLD_SIZE=1 \
    RANK=0 \
    MASTER_PORT=29611 \
    DSV41_DIST_TIMEOUT_S=600

VOLUME ["/models"]
EXPOSE 8000

# start-period is 45 minutes because it has to be: the socket is bound only after the warm
# start has pulled tens of GB of FP4 experts off NVMe. A container that is not answering at
# minute 10 is normal, not sick.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
