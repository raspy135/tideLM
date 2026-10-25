#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# code_digest.sh -- 16-hex digest of the engine code a serving image runs: the sources under
# engine/, tools/ and server/ (what the Dockerfile's COPY puts in /app and what dual-dev.sh
# bind-mounts instead). dual-build.sh stamps it on the image as the label dsv41.code_sha;
# dual-up.sh compares the label with the checkout so a plain restart cannot silently boot code
# older than the checkout (2026-10-08: an image from before the EXL3 CUDA kernel came up
# "healthy" on the torch reference MoE, minutes per request).
#
# --rust: the Rust engine's instead, the sources the image's tidelm-server is built from (rust/
# less its target/, the CUDA/C++ files its build scripts compile from tools/ and engine/, and the
# draft vocabulary it compiles in);
# dual-build.sh stamps it as tidelm.rust_sha and rust-up.sh compares it the same way.
#
# Usage: scripts/code_digest.sh [--rust] [repo-root]
# ---------------------------------------------------------------------------------------
set -euo pipefail
rust=0
[[ "${1:-}" == --rust ]] && { rust=1; shift; }
cd "${1:-$(dirname "$0")/..}"
export LC_ALL=C
if [[ $rust == 1 ]]; then
    { find rust -path rust/target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' \
           -o -name '*.cpp' -o -name '*.h' -o -name '*.toml' -o -name Cargo.lock \) -print0
      printf '%s\0' tools/roce/roce.cu tools/roce/roce_common.h tools/fp4_moe_cuda.cu tools/exl3_moe_cuda.cu \
             engine/engram_gather.cpp engine/draft_vocab_ranking_v1.txt; } \
        | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
    exit 0
fi
find engine tools server -type f \
    \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \
       -o -name '*.html' -o -name '*.js' \) \
    -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
