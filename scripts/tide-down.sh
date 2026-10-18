#!/usr/bin/env bash
# tide-down.sh [ENV_FILE] -- stop whichever TideLM pair is serving (either engine) on both
# Sparks and wait for their memory to come back. The profile only supplies PEER; without one,
# the profile scripts/tide-up.sh last started is used.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
f=${1:-$(cat results/.tide-active 2>/dev/null || true)}
if [[ -n "$f" && -f "$f" ]]; then
    TIDE_ENV_FILE="$f" exec scripts/dual-down.sh
fi
exec scripts/dual-down.sh
