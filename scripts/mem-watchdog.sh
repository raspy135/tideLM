#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# mem-watchdog.sh -- sample host MemAvailable on both nodes while the pair serves.
#
# Why this exists: on GB10 the GPU allocator and the page cache draw on one pool, and an
# over-commit does not raise an OOM kill -- the kernel cannot reclaim CUDA mappings, so the
# box goes unreachable (ping answers, sshd cannot fork) and the recovery is a power cycle.
# `keep_free_gb` is enforced when the engine allocates the arena and never again, so the
# only way to see a profile running out of headroom before it wedges the box is to watch
# MemAvailable from outside.
#
# A larger ARENA_GB and a larger MAX_SEQ draw on the same budget (engine/v41_engine.py
# reserves 8 GB + 0.25 GB per 8192 tokens), so raising either is a reason to run this.
#
#   scripts/mem-watchdog.sh                      one line per sample, both nodes, every 5 s
#   scripts/mem-watchdog.sh --sec 2 --warn 4     tighter sampling and an earlier warning
#   scripts/mem-watchdog.sh --floor 1 --kill     stop the pair below 1 GiB (destructive)
#   scripts/mem-watchdog.sh --once               sample once and exit non-zero below the floor
#
# It runs until Ctrl-C, and prints the minimum it saw on the way out -- that minimum is the
# number to compare against the next profile change.
# ---------------------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEC=5
WARN=3.0
FLOOR=1.0
KILL=0
ONCE=0
LOG="${LOG:-results/mem-watchdog.log}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sec)   SEC="$2"; shift 2 ;;
        --warn)  WARN="$2"; shift 2 ;;
        --floor) FLOOR="$2"; shift 2 ;;
        --kill)  KILL=1; shift ;;
        --once)  ONCE=1; shift ;;
        --log)   LOG="$2"; shift 2 ;;
        -h|--help) sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1 (try --help)" >&2; exit 2 ;;
    esac
done
[[ "$SEC" =~ ^[0-9]+$ ]] && (( SEC > 0 )) || { echo "--sec must be a positive integer" >&2; exit 2; }

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env 2>/dev/null; set +a; }
PEER="${PEER:-}"
NAME0="${NAME0:-deepseek-v41-ep2-rank0}"
NAME1="${NAME1:-deepseek-v41-ep2-rank1}"

# GiB available on one host. Empty string means "could not read it", which is not zero: a
# failed ssh must not look like a starved box and trip the kill.
avail() {
    local tgt="${1:-}" v
    if [[ -z "$tgt" ]]; then
        v=$(awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo)
    else
        # One multiplexed connection for every sample: a fresh login a second opened a logind
        # session each time, and polkitd on the peer grew with them (3.3 GB after ~2 days and
        # 2.1M sessions, while node 0's stayed at 6 MB).
        v=$(ssh -o BatchMode=yes -o ConnectTimeout=4 -o ControlMaster=auto \
              -o ControlPath="$HOME/.ssh/cm-watchdog-%r@%h:%p" -o ControlPersist=120 "$tgt" \
              'awk "/^MemAvailable:/{printf \"%.2f\", \$2/1048576}" /proc/meminfo' 2>/dev/null)
    fi
    [[ "$v" =~ ^[0-9]+\.[0-9]+$ ]] && printf '%s' "$v"
}

mkdir -p "$(dirname "$LOG")" 2>/dev/null
min0=""; min1=""; worst=""
trap 'printf "\n--- minimum seen: %s GiB%s%s\n" "${worst:-?}" \
      "${min0:+ (rank0 $min0)}" "${min1:+ rank1 $min1}"' EXIT

while :; do
    a0=$(avail); a1=""; [[ -n "$PEER" ]] && a1=$(avail "$PEER")
    for a in "$a0" "$a1"; do
        [[ -z "$a" ]] && continue
        [[ -z "$worst" || $(awk -v x="$a" -v y="$worst" 'BEGIN{print (x<y)}') == 1 ]] && worst="$a"
    done
    [[ -n "$a0" ]] && { [[ -z "$min0" || $(awk -v x="$a0" -v y="$min0" 'BEGIN{print (x<y)}') == 1 ]] && min0="$a0"; }
    [[ -n "$a1" ]] && { [[ -z "$min1" || $(awk -v x="$a1" -v y="$min1" 'BEGIN{print (x<y)}') == 1 ]] && min1="$a1"; }

    state=ok
    for a in "$a0" "$a1"; do
        [[ -z "$a" ]] && continue
        [[ $(awk -v x="$a" -v y="$WARN" 'BEGIN{print (x<y)}') == 1 ]] && state=warn
        [[ $(awk -v x="$a" -v y="$FLOOR" 'BEGIN{print (x<y)}') == 1 ]] && state=FLOOR
    done

    line="$(date '+%H:%M:%S')  rank0 ${a0:-?} GiB  rank1 ${a1:-n/a} GiB  [$state]"
    printf '%s\n' "$line"
    [[ -n "$LOG" ]] && printf '%s\n' "$line" >>"$LOG"

    if [[ "$state" == FLOOR ]]; then
        printf '*** below the %.2f GiB floor -- the box can stop being reachable from here\n' "$FLOOR" >&2
        if (( KILL )); then
            printf '*** --kill: stopping the pair now\n' >&2
            docker stop -t 1 "$NAME0" >/dev/null 2>&1 || true
            [[ -n "$PEER" ]] && ssh -o BatchMode=yes -o ConnectTimeout=4 "$PEER" \
                "docker stop -t 1 '$NAME1' >/dev/null 2>&1 || true" 2>/dev/null
            exit 1
        fi
    fi

    (( ONCE )) && { [[ "$state" == FLOOR ]] && exit 1; exit 0; }
    sleep "$SEC"
done
