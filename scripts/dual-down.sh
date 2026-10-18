#!/usr/bin/env bash
# dual-down.sh -- stop the serving pair (V4.1 EP2 or GLM-5.3) and wait for the unified pool to
# come back on BOTH boxes.
#
# Stop both ranks concurrently: one grace period for the pair, then verify both
# memory pools have been reclaimed before returning. --force skips the grace.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
declare -A _CLI=()
for v in STOP_TIMEOUT MIN_FREE_GIB PEER NAME0 NAME1; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
_env="${TIDE_ENV_FILE:-.env}"   # the profile scripts/tide-up.sh started, else .env
case "$_env" in */*) ;; *) _env="./$_env" ;; esac   # `.` would search PATH for a bare name
[[ -f "$_env" ]] && { set -a; . "$_env"; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; export "${v?}"; done
# The GLM pair (scripts/glm-up.sh) takes its peer from .env.glm.
[[ -z "${PEER:-}" && -f .env.glm ]] && PEER=$(sed -n 's/^PEER=//p' .env.glm | tail -1)
PEER="${PEER:-}"
# Whichever pair is running here: the V4.1 EP2 pair, else the GLM pair.
if [[ -z "${NAME0:-}" ]] && ! docker container inspect deepseek-v41-ep2-rank0 >/dev/null 2>&1 \
        && docker container inspect glm-serve-rank0 >/dev/null 2>&1; then
    NAME0=glm-serve-rank0 NAME1=glm-serve-rank1
    PEER="${PEER:-ryan@10.0.0.2}"
fi
NAME0="${NAME0:-deepseek-v41-ep2-rank0}"
NAME1="${NAME1:-deepseek-v41-ep2-rank1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"

# Grace is short on purpose. A clean Python unwind of an 88 GB pinned arena rarely finishes
# inside a few minutes anyway; past STOP_TIMEOUT docker SIGKILLs, and the kernel reclaims the
# unified pool in seconds. Memory reclamation is checked separately below.
STOP_TIMEOUT="${STOP_TIMEOUT:-5}"
case "${1:-}" in
    --force|-f) STOP_TIMEOUT=0 ;;
    "") ;;
    *) echo "usage: $0 [--force]" >&2; exit 2 ;;
esac
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "STOP_TIMEOUT must be a nonnegative integer" >&2; exit 2; }

echo "--- stopping both ranks (grace ${STOP_TIMEOUT}s, concurrent)"
(
    if docker container inspect "$NAME0" >/dev/null 2>&1; then
        docker stop -t "$STOP_TIMEOUT" "$NAME0" >/dev/null 2>&1 || true
        docker rm -f "$NAME0" >/dev/null
    fi
) &
local_stop=$!
peer_stop=""
if [[ -n "$PEER" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "if docker container inspect '$NAME1' >/dev/null 2>&1; then docker stop -t '$STOP_TIMEOUT' '$NAME1' >/dev/null 2>&1 || true; docker rm -f '$NAME1' >/dev/null; fi" &
    peer_stop=$!
fi
rc=0
wait "$local_stop" || rc=1
[[ -z "$peer_stop" ]] || wait "$peer_stop" || rc=1

wait_mem() {   # $1 = label, $2 = "" local or ssh target
    local label="$1" tgt="${2:-}" a
    echo "--- waiting for $label to return >= ${MIN_FREE_GIB} GiB: "
    for _ in $(seq 1 60); do
        if [[ -z "$tgt" ]]; then
            a=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
        else
            a=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$tgt" 'awk "/^MemAvailable:/ {printf \"%d\", \$2/1048576}" /proc/meminfo' 2>/dev/null)
        fi
        if [[ -n "$a" ]] && (( a >= MIN_FREE_GIB )); then echo "--- $label: OK, ${a} GiB"; return 0; fi
        sleep 3
    done
    echo "WARNING: still at ${a:-?} GiB -- do not start another server yet" >&2
    return 1
}
wait_mem "this box" "" &
local_mem=$!
peer_mem=""
if [[ -n "$PEER" ]]; then
    wait_mem "$PEER" "$PEER" &
    peer_mem=$!
fi
wait "$local_mem" || rc=1
[[ -z "$peer_mem" ]] || wait "$peer_mem" || rc=1
exit $rc
