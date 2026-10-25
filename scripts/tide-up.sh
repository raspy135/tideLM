#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# tide-up.sh ENV_FILE -- serve the model ENV_FILE describes on both Sparks.
#
#   scripts/tide-up.sh .env.ds41-mxfp4      DeepSeek-V4.1-Flash(-1017), MXFP4 experts
#   scripts/tide-up.sh .env.glm53-mxfp4     GLM-5.3, MXFP4 experts
#   scripts/tide-up.sh .env.glm53-exl3      GLM-5.3, EXL3 experts
#
# Copy a profile from configs/*.env.example and edit its paths first. The model follows from
# the file: GLM_MODEL_DIR is GLM-5.3, MODEL_DIR DeepSeek V4.1. The Rust engine (rust/,
# scripts/rust-up.sh) serves both and is the default; TIDE_ENGINE=python, in the profile or the
# environment (which wins), selects the Python one (scripts/glm-up.sh, scripts/dual-up.sh). A
# profile the Rust engine does not serve (`scripts/rust-up.sh --check` says why: MXFP4 GLM
# experts, FP4 KV, ...) falls back to Python with a note, unless it asked for TIDE_ENGINE=rust.
# One pair fills both boxes, so whatever is serving is stopped first, after the checks.
# scripts/tide-down.sh stops it again.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
f=${1:-}
[[ -n "$f" ]] || err "usage: scripts/tide-up.sh ENV_FILE   (e.g. .env.glm53-mxfp4; see configs/)"
[[ -f "$f" ]] || { [[ -f "$ROOT/$f" ]] && f="$ROOT/$f"; } || err "no env file $f"
f="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
if grep -qE '^[[:space:]]*GLM_MODEL_DIR=' "$f"; then
    model=glm
elif grep -qE '^[[:space:]]*MODEL_DIR=' "$f"; then
    model=ds41
else
    err "$f sets neither GLM_MODEL_DIR (GLM-5.3) nor MODEL_DIR (DeepSeek V4.1)"
fi
# the environment's TIDE_ENGINE, else the profile's last TIDE_ENGINE= line, else rust
asked=${TIDE_ENGINE:-$(sed -nE 's/^[[:space:]]*TIDE_ENGINE=["'"'"']?([A-Za-z0-9_]+).*/\1/p' "$f" | tail -1)}
tide_engine=${asked:-rust}
[[ $tide_engine == rust || $tide_engine == python ]] || err "TIDE_ENGINE=$tide_engine: rust or python"
if [[ $tide_engine == rust ]] && ! why=$("$ROOT/scripts/rust-up.sh" --check "$f" 2>&1 >/dev/null); then
    [[ -n $asked ]] && err "TIDE_ENGINE=rust, but ${why#ERROR: }"
    echo "--- the Rust engine does not serve this profile (${why#ERROR: }): the Python engine serves it"
    tide_engine=python
fi
case $model/$tide_engine in
    glm/rust) engine=rust-glm ;;
    glm/python) engine=glm ;;
    ds41/rust) engine=rust ;;
    ds41/python) engine=ds41 ;;
esac
"$ROOT/scripts/tide-down.sh" "$f"
mkdir -p "$ROOT/results"
echo "$f" > "$ROOT/results/.tide-active"
echo "--- TideLM: $engine engine, profile $f"
if [[ $engine == glm ]]; then
    GLM_ENV_FILE="$f" exec "$ROOT/scripts/glm-up.sh"
elif [[ $engine == rust || $engine == rust-glm ]]; then
    exec "$ROOT/scripts/rust-up.sh" "$f"
fi
TIDE_ENV_FILE="$f" exec "$ROOT/scripts/dual-up.sh"
