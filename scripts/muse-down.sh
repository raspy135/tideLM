#!/usr/bin/env bash
# muse-down.sh -- stop and remove the Muse serving containers on both Sparks (only those).
set -uo pipefail
PEER=${PEER:-ryan@10.0.0.2}
docker rm -f muse-serve-rank0 >/dev/null 2>&1 && echo "--- rank 0 stopped" || echo "--- rank 0 not running"
ssh -o BatchMode=yes "$PEER" "docker rm -f muse-serve-rank1 >/dev/null 2>&1" \
    && echo "--- rank 1 stopped" || echo "--- rank 1 not running"
