#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# tide.sh COMMAND [OPTIONS] [ENV_FILE] -- serve TideLM on the two Sparks: one entry point for
# the scripts that do it.
#
#   scripts/tide.sh start [ENV_FILE]                    stop whatever is serving, start ENV_FILE
#   scripts/tide.sh stop [ENV_FILE]                     stop it, and wait for both boxes' memory
#   scripts/tide.sh build [--local-only] [ENV_FILE]     build the serving image, ship it to PEER
#   scripts/tide.sh logs [-f] [-n N] [--rank 1] [ENV_FILE]   the serving engine's log
#
# ENV_FILE is a profile (configs/*.env.example, copied and edited); left out, it is the one the
# last start used (results/.tide-active). start, stop and build run scripts/tide-up.sh,
# scripts/tide-down.sh and scripts/dual-build.sh; see those for the details. logs reads the
# containers' output (docker logs): rank 0 here, --rank 1 the worker's over ssh; -f follows it,
# -n N starts N lines back (default 100; 0 for the whole log).
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//' >&2; exit "${1:-2}"; }
cmd=${1:-}
[[ -n "$cmd" ]] || usage
[[ $cmd == -h || $cmd == --help ]] && usage 0
shift
follow=() lines=100 rank=0 local_only=() f=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" != -* ]]; then
        [[ -z "$f" ]] || err "one profile at a time ($f, $1)"
        f=$1
    else
        case "$cmd $1" in
            "logs -f" | "logs --follow") follow=(-f) ;;
            "logs -n") [[ "${2:-}" =~ ^[0-9]+$ ]] || err "-n needs a number of lines"; lines=$2; shift ;;
            "logs --rank") [[ "${2:-}" == 0 || "${2:-}" == 1 ]] || err "--rank is 0 or 1"; rank=$2; shift ;;
            "build --local-only") local_only=(--local-only) ;;
            *" -h" | *" --help") usage 0 ;;
            *) err "$cmd has no option $1 (scripts/tide.sh --help)" ;;
        esac
    fi
    shift
done
if [[ -z "$f" ]]; then
    f=$(cat "$ROOT/results/.tide-active" 2>/dev/null || true)
    [[ -n "$f" || $cmd == stop ]] || err "no profile given, and none started yet: scripts/tide.sh $cmd ENV_FILE"
fi

case "$cmd" in
    start) exec "$ROOT/scripts/tide-up.sh" "$f" ;;
    stop)  exec "$ROOT/scripts/tide-down.sh" ${f:+"$f"} ;;
    build) exec "$ROOT/scripts/dual-build.sh" "${local_only[@]}" "$f" ;;
    logs)  ;;
    *) usage ;;
esac

# logs: whichever engine's containers are up (the Rust engine's, else the Python pair's)
[[ -f "$f" ]] || { [[ -f "$ROOT/$f" ]] && f="$ROOT/$f"; } || err "no env file $f"
peer=$(sed -nE 's/^[[:space:]]*PEER=["'"'"']?([^"'"'"' #]+).*/\1/p' "$f" | tail -1)
on() { if [[ $rank == 1 ]]; then [[ -n "$peer" ]] || err "$f has no PEER"; ssh -t -o LogLevel=QUIET "$peer" "$@"; else "$@"; fi; }
for c in tidelm-rank deepseek-v41-ep2-rank glm-serve-rank; do
    if on docker container inspect "$c$rank" > /dev/null 2>&1; then
        tail=(--tail "$lines")
        [[ $lines == 0 ]] && tail=()
        on docker logs "${follow[@]}" "${tail[@]}" "$c$rank"
        exit
    fi
done
err "nothing is serving on rank $rank (no tidelm-rank$rank, deepseek-v41-ep2-rank$rank or glm-serve-rank$rank container)"
