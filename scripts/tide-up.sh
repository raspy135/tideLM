#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# tide-up.sh ENV_FILE -- serve the model ENV_FILE describes on both Sparks.
#
#   scripts/tide-up.sh .env.ds41-mxfp4      DeepSeek-V4.1-Flash(-1017), MXFP4 experts
#   scripts/tide-up.sh .env.glm53-mxfp4     GLM-5.3, MXFP4 experts
#   scripts/tide-up.sh .env.glm53-exl3      GLM-5.3, EXL3 experts
#
# Copy a profile from configs/*.env.example and edit its paths first. The engine follows from
# the file: GLM_MODEL_DIR selects the GLM-5.3 engine (scripts/glm-up.sh), MODEL_DIR the
# DeepSeek V4.1 engine (scripts/dual-up.sh). One pair fills both boxes, so whatever is serving
# is stopped first. scripts/tide-down.sh stops it again.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
f=${1:-}
[[ -n "$f" ]] || err "usage: scripts/tide-up.sh ENV_FILE   (e.g. .env.glm53-mxfp4; see configs/)"
[[ -f "$f" ]] || { [[ -f "$ROOT/$f" ]] && f="$ROOT/$f"; } || err "no env file $f"
f="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
if grep -qE '^[[:space:]]*GLM_MODEL_DIR=' "$f"; then
    engine=glm
elif grep -qE '^[[:space:]]*MODEL_DIR=' "$f"; then
    engine=ds41
else
    err "$f sets neither GLM_MODEL_DIR (GLM-5.3) nor MODEL_DIR (DeepSeek V4.1)"
fi
"$ROOT/scripts/tide-down.sh" "$f"
mkdir -p "$ROOT/results"
echo "$f" > "$ROOT/results/.tide-active"
echo "--- TideLM: $engine engine, profile $f"
if [[ $engine == glm ]]; then
    GLM_ENV_FILE="$f" exec "$ROOT/scripts/glm-up.sh"
fi
TIDE_ENV_FILE="$f" exec "$ROOT/scripts/dual-up.sh"
