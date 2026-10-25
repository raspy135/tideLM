#!/usr/bin/env bash
# tide-down.sh [ENV_FILE] -- stop whichever TideLM pair is serving (either engine) on both
# Sparks and wait for their memory to come back. The profile only supplies PEER; without one,
# the profile scripts/tide-up.sh last started is used. scripts/tide.sh stop runs it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
f=${1:-$(cat results/.tide-active 2>/dev/null || true)}
# the Rust engine (scripts/rust-up.sh), when it is the one serving: its containers on both ranks,
# then any tidelm-server still running by name (one started on the host, before the image had it)
peer=$( [[ -n "$f" && -f "$f" ]] && sed -nE 's/^[[:space:]]*PEER=["'"'"']?([^"'"'"' #]+).*/\1/p' "$f" | tail -1 || true)
if docker container inspect tidelm-rank0 > /dev/null 2>&1 \
        || { [[ -n "$peer" ]] && ssh "$peer" 'docker container inspect tidelm-rank1' > /dev/null 2>&1; }; then
    echo "--- stopping the Rust engine's containers"
    { docker stop -t 10 tidelm-rank0 > /dev/null 2>&1; docker rm -f tidelm-rank0 > /dev/null 2>&1; true; } &
    [[ -n "$peer" ]] && ssh "$peer" 'docker stop -t 10 tidelm-rank1 > /dev/null 2>&1; docker rm -f tidelm-rank1 > /dev/null 2>&1' || true
    wait
fi
rust_up() { pgrep -x tidelm-server > /dev/null || { [[ -n "$peer" ]] && ssh "$peer" 'pgrep -x tidelm-server' > /dev/null; }; }
if rust_up; then
    echo "--- stopping the Rust engine"
    pkill -TERM -x tidelm-server || true
    [[ -n "$peer" ]] && ssh "$peer" 'pkill -TERM -x tidelm-server' || true
    for _ in $(seq 60); do rust_up || break; sleep 1; done
    rust_up && echo "WARNING: a tidelm-server is still running" >&2
fi
if [[ -n "$f" && -f "$f" ]]; then
    TIDE_ENV_FILE="$f" exec scripts/dual-down.sh
fi
exec scripts/dual-down.sh
