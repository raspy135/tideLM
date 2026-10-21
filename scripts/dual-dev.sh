#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dual-dev.sh -- fast iteration boot.
#
# `dual-build.sh` bakes engine/tools/server into the image and ships ~11 GB; that costs minutes per
# code change.  This syncs the working tree to the peer (a few MB) and boots with source
# bind-mounts, so a change costs one `dual-down` + `dual-up` (~1 min, dominated by the 64 GB warm
# start and weight load).  Requires the repo at the same path on both boxes.  The image still
# provides the interpreter, CUDA, torch and Triton; only /app/{engine,tools,server} differ.
#
# Usage: scripts/dual-dev.sh [ENV_FILE] [dual-up.sh flags, e.g. --check]
#   ENV_FILE: the DeepSeek profile to serve (e.g. .env.ds41-mxfp4); .env when omitted.
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

info() { echo "--- $*"; }

if [[ -n "${1:-}" && "$1" != -* ]]; then
    f=$1; shift
    [[ -f "$f" ]] || { [[ -f "$ROOT/$f" ]] && f="$ROOT/$f"; } || { echo "ERROR: no env file $f" >&2; exit 1; }
    export TIDE_ENV_FILE="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
fi
cd "$ROOT"

# shellcheck disable=SC1090,SC1091
_PEER_CLI="${PEER:-}"
f="${TIDE_ENV_FILE:-.env}"
# Only PEER is needed here, so the profile is read in a subshell. Sourcing it into this shell
# exported every DSV41_* it pins BEFORE dual-up.sh captures the caller's DSV41_* overrides, so
# the profile silently beat the command line (DSV41_DYNAMIC_EXPERTS=0 arrived as 1).
PEER="${_PEER_CLI:-$([[ -f "$f" ]] && { set -a; . "$f"; set +a; }; echo "${PEER:-}")}"
[[ -n "$PEER" ]] || { echo "ERROR: set PEER in $f (user@host)" >&2; exit 1; }

info "syncing engine/ tools/ server/ to $PEER"
for d in engine tools server; do
    rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
        "$ROOT/$d/" "$PEER:$ROOT/$d/"
done

info "booting with source mounts (DSV41_DEV_SOURCE=1)"
DSV41_DEV_SOURCE=1 exec "$ROOT/scripts/dual-up.sh" "$@"
