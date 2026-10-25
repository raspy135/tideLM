#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# tide-up.sh ENV_FILE -- serve the model ENV_FILE describes on both Sparks.
#
#   scripts/tide-up.sh .env.ds41-mxfp4      DeepSeek-V4.1-Flash(-1017), MXFP4 experts
#   scripts/tide-up.sh .env.glm53-exl3      GLM-5.3, EXL3 experts
#
# Copy a profile from configs/*.env.example and edit its paths first. The model follows from
# the file: GLM_MODEL_DIR is GLM-5.3, MODEL_DIR DeepSeek V4.1. The Rust engine (rust/,
# scripts/rust-up.sh) serves both and is the default; TIDE_ENGINE=python, in the profile or the
# environment (which wins), selects the Python one (scripts/glm-up.sh, scripts/dual-up.sh). A
# profile the Rust engine does not serve (`scripts/rust-up.sh --check` says why: MXFP4 GLM
# experts, FP4 KV, ...) falls back to Python with a note, unless it asked for TIDE_ENGINE=rust.
# One pair fills both boxes, so whatever is serving is stopped first, after the checks.
# scripts/tide-down.sh stops it again. scripts/tide.sh start / stop runs both.
#
# Both engines serve the OpenAI API (/v1/chat/completions) and the Anthropic one (/v1/messages,
# server/anthropic_api.py) on the same port. Once the server is up this also writes
# results/claude-code.json, the settings that point Claude Code at it with the profile's model
# name, port and context: `claude --settings results/claude-code.json`. Other Claude Code
# sessions keep their own model.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
f=${1:-}
[[ -n "$f" ]] || err "usage: scripts/tide-up.sh ENV_FILE   (e.g. .env.glm53-exl3; see configs/)"
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
claude_cfg="$ROOT/results/claude-code.json"
rm -f "$claude_cfg"
if [[ $engine == glm ]]; then
    GLM_ENV_FILE="$f" "$ROOT/scripts/glm-up.sh"
elif [[ $engine == rust || $engine == rust-glm ]]; then
    "$ROOT/scripts/rust-up.sh" "$f"
else
    TIDE_ENV_FILE="$f" "$ROOT/scripts/dual-up.sh"
fi
# The launchers' own defaults when the profile leaves a setting out.
(set -a; source "$f"; set +a; python3 - "$claude_cfg" "$engine" <<'PY'
import json, os, sys
glm = sys.argv[2] in ("glm", "rust-glm")
model = os.environ.get("SERVED_MODEL_NAME") or ("glm-5.3" if glm else "deepseek-v4.1-flash")
context = os.environ.get("GLM_MAX_SEQ", "262144") if glm else os.environ.get("MAX_SEQ", "32768")
env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{os.environ.get('PORT', '8000')}",
       "ANTHROPIC_AUTH_TOKEN": "tidelm", "ANTHROPIC_MODEL": model,
       "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
       "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "CLAUDE_CODE_SUBAGENT_MODEL": model,
       "CLAUDE_CODE_MAX_CONTEXT_TOKENS": context,
       # Fewer request fields the server would only ignore, and no attribution block: its
       # per-request fingerprint at the top of the prompt would defeat the prefix cache (the
       # server drops it too).
       "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
       "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
with open(sys.argv[1], "w") as fh:
    json.dump({"env": env}, fh, indent=2)
print(f"--- OpenAI API: {env['ANTHROPIC_BASE_URL']}/v1   Claude Code: claude --settings {sys.argv[1]}")
PY
)
