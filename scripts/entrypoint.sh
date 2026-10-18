#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# entrypoint.sh -- the container's launcher: what PID 1 runs inside each rank.
#
# Runs `server/app.py --engine v41` in the foreground as PID 1 (so `docker stop`
# reaches it with SIGTERM and the engine gets to release its pinned buffers).
# Every knob is the same name and the same meaning as in env.example, so a .env
# that works natively works here:
#
#   MODEL_DIR          checkpoint directory                (/models/DeepSeek-V4.1-Flash)
#   SERVED_MODEL_NAME  id reported by /v1/models           (deepseek-v4.1-flash)
#   HOST / PORT        bind inside the container           (0.0.0.0 / 8000)
#   MAX_SEQ            context the caches are sized for    (32768)
#   ARENA_GB           resident FP4 expert arena in GB     (empty = auto)
#   TRACE_STATS        coverage.json that ranks the warm start (auto-discovered)
#   DEFAULT_THINKING   on|off for requests that say nothing (off)
#   DEFAULT_EFFORT     1-100 reasoning effort default      (75)
#   SPEC               1 = MTP/DSpark speculative decoding, 0 = plain decode
#   EXTRA_FLAGS        whitespace-separated extra flags for server/app.py
#   MIN_FREE_GIB       refuse to start below this MemAvailable (90; 0 disables)
#   DOWNLOAD_ON_START  1 = fetch the weights first if they are missing (0)
#
# Anything else on the command line replaces all of it: `docker compose run --rm
# v41 bash` still gives you a shell, and `... v41 download` runs the downloader.
# ---------------------------------------------------------------------------
set -euo pipefail

case "${1:-}" in
    serve|"") shift || true ;;
    download) exec /app/scripts/download-model.sh ;;
    bench)    shift; exec python3 /app/bench/bench.py "$@" ;;
    shell|bash) shift; exec bash "$@" ;;
    -h|--help) sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) exec "$@" ;;
esac

info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

: "${MODEL_DIR:=/models/DeepSeek-V4.1-Flash}"
: "${SERVED_MODEL_NAME:=deepseek-v4.1-flash}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${MAX_SEQ:=32768}"
: "${ARENA_GB:=}"
: "${TRACE_STATS:=}"
: "${DEFAULT_THINKING:=off}"
: "${DEFAULT_EFFORT:=75}"
: "${SPEC:=1}"
: "${EXTRA_FLAGS:=}"
: "${MIN_FREE_GIB:=90}"
: "${DOWNLOAD_ON_START:=0}"
# --- EP2 / dual Spark (docs/dual-spark-plan.md) ---------------------------
# WORLD_SIZE=2 splits the routed experts across two boxes; everything else is
# replicated. RANK 0 serves HTTP, RANK 1 runs headless and only participates in
# the per-layer combines. scripts/dual-up.sh sets all of these; WORLD_SIZE=1 (the
# image default) is the single-box path with none of it touched.
: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=}"
: "${MASTER_PORT:=29611}"
: "${EP_LOCAL_IP:=}"

echo "TideLM $(cat /app/VERSION 2>/dev/null || echo dev)"

case "$DEFAULT_THINKING" in on|off) ;; *) err "DEFAULT_THINKING must be on|off (got '$DEFAULT_THINKING')" ;; esac
case "$SPEC" in 0|1) ;; *) err "SPEC must be 0|1 (got '$SPEC')" ;; esac

# --- EP2 preflight --------------------------------------------------------
if [[ "$WORLD_SIZE" != "1" ]]; then
    case "$WORLD_SIZE" in 2) ;; *) err "WORLD_SIZE must be 1 or 2 (the EP2 skeleton is world-2 only)" ;; esac
    [[ -n "$MASTER_ADDR" ]] || err "WORLD_SIZE=2 needs MASTER_ADDR (rank 0's address on the RoCE link)"
    [[ -e /dev/infiniband/rdma_cm ]] || err "WORLD_SIZE=2 but /dev/infiniband is not in this container.
     Run it with --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --network host,
     or use scripts/dual-up.sh which does. Without verbs NCCL drops to TCP sockets and the
     collective tax goes from ~60 us to ~1.9 ms -- 30x over the Gate G0 budget."
    # Dual rail uses NCCL's per-HCA dynamic GID selection. A single NCCL_IB_GID_INDEX cannot
    # describe two HCAs whose IPv4 entries may occupy different slots, and those slots move
    # across reboots. NCCL_IB_ADDR_RANGE constrains selection to the two direct-link subnets.
    if [[ -n "${NCCL_IB_HCA:-}" && -n "${NCCL_IB_ADDR_RANGE:-}" && -z "${NCCL_IB_GID_INDEX:-}" ]]; then
        export NCCL_IB_ADDR_FAMILY="${NCCL_IB_ADDR_FAMILY:-AF_INET}"
        export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
        info "EP2 rank $RANK: RoCE $NCCL_IB_HCA, dynamic GID in $NCCL_IB_ADDR_RANGE"
    # Single rail keeps the older per-rank explicit lookup. The two boxes need not agree on
    # the index, and a wrong one fails the QP transition instead of falling back to sockets.
    elif [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
        if read -r _hca _gid < <(/app/scripts/roce_gid.sh ${EP_LOCAL_IP:+"$EP_LOCAL_IP"}); then
            export NCCL_IB_HCA="${NCCL_IB_HCA:-$_hca}" NCCL_IB_GID_INDEX="$_gid"
            info "EP2 rank $RANK: RoCE $NCCL_IB_HCA gid $NCCL_IB_GID_INDEX"
        else
            err "cannot resolve this rank's RoCE GID (scripts/roce_gid.sh). Set NCCL_IB_GID_INDEX
     explicitly -- but resolve it PER BOX, one value for both ranks is the classic bug."
        fi
    fi
    export RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
    info "EP2: rank $RANK of $WORLD_SIZE, rendezvous $MASTER_ADDR:$MASTER_PORT, iface ${NCCL_SOCKET_IFNAME:-auto}"
fi

# --- the weights ----------------------------------------------------------
# A 510 GB download is not something to start by accident, so a missing model is
# an error with instructions rather than an implicit fetch. DOWNLOAD_ON_START=1
# opts in.
if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    if [[ "$DOWNLOAD_ON_START" == "1" ]]; then
        info "no checkpoint at $MODEL_DIR -- fetching it first"
        /app/scripts/download-model.sh
    else
        err "no checkpoint at $MODEL_DIR.
     Mount it there (compose: ./models:/models) or fetch it with
         docker compose run --rm v41 download
     It is 510 GB. See docs/install.md."
    fi
fi
[[ -f "$MODEL_DIR/tokenizer.json" ]] || err "$MODEL_DIR has no tokenizer.json"
[[ -f "$MODEL_DIR/encoding/encoding.py" ]] || err "$MODEL_DIR has no encoding/encoding.py (the checkpoint's chat encoder)"

# --- the memory guard -----------------------------------------------------
# GB10 has one pool for GPU and host. The arena is sized from what is free at
# load time, so starting next to another server does not OOM -- it silently
# gives a tiny arena and a NVMe-bound server, or wedges the driver with no log
# line at all. Refuse instead, and let `restart: on-failure:1` stop after one
# retry rather than loop the box into the ground.
if [[ "$MIN_FREE_GIB" != "0" && -r /proc/meminfo ]]; then
    avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    if (( avail_gib < MIN_FREE_GIB )); then
        echo "ERROR: only ${avail_gib} GiB available (need >= ${MIN_FREE_GIB} GiB)." >&2
        echo "     Stop whatever else holds the unified pool and wait for MemAvailable" >&2
        echo "     to recover, then start again. MIN_FREE_GIB=0 disables this guard." >&2
        ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -6 |
            awk 'NR==1{print "       PID      RSS_GB  COMMAND"; next} {printf "       %-8s %-7.1f %s\n", $1, $2/1048576, $3}' >&2
        exit 1
    fi
fi

# --- which coverage.json ranks the warm start -----------------------------
# Without one the warm start fills the arena in (layer, expert) index order,
# which is a measurably worse hot set than the traced one. Directory names carry
# a date (results/trace-full-YYYYMMDD/), so when TRACE_STATS is unset or points
# at something that is not there, take the newest match instead of nothing.
resolve_trace() {
    local want="$1" c
    if [[ -n "$want" ]]; then
        [[ "$want" != /* ]] && want="/app/$want"
        [[ -f "$want" ]] && { echo "$want"; return; }
        echo "trace stats not found at $want -- looking for another" >&2
    fi
    c=$(ls -1d /app/results/trace-*/stats/coverage.json 2>/dev/null | sort | tail -1 || true)
    [[ -n "$c" ]] && echo "$c"
}
TRACE_USED="$(resolve_trace "$TRACE_STATS")"
[[ -z "$TRACE_USED" ]] && info "no coverage.json under /app/results -- the warm start will use index order"

FLAGS=(
    --model-dir "$MODEL_DIR"
    --host "$HOST" --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --default-thinking "$DEFAULT_THINKING"
    --default-effort "$DEFAULT_EFFORT"
    --engine v41
    --max-seq "$MAX_SEQ"
)
[[ -n "$ARENA_GB" ]] && FLAGS+=(--arena-gb "$ARENA_GB")
[[ "$SPEC" == "0" ]] && FLAGS+=(--no-spec)
# Pruned all-resident mode (RESULTS.md v0.2.0-wip): PRUNE_KEEP=0.31 keeps the top 31 % experts per
# layer routable and resident; pair it with ARENA_GB=90.5 TRANSIENT_SLOTS=16 KEEP_FREE_GB=10 on a
# 128 GB box. Unset = the full model with expert streaming.
EK="{"
[[ -n "${PRUNE_KEEP:-}" ]] && EK="$EK\"prune_keep\": $PRUNE_KEEP,"
# EXPERT_FORMAT=cb3 packs the resident arena into the 3-bit per-row codebook format (14.45 MB per
# expert instead of 18.80), so the same 90.5 GB holds ~40.8 % of all routed experts instead of
# 31.3 %; pair it with PRUNE_KEEP=0.40. Warm start pays the packing (see NOTES 2026-09-11).
[[ -n "${EXPERT_FORMAT:-}" ]] && EK="$EK\"expert_format\": \"$EXPERT_FORMAT\","
[[ -n "${PRUNE_SELECT:-}" ]] && EK="$EK\"prune_select\": \"$PRUNE_SELECT\","
[[ -n "${TRANSIENT_SLOTS:-}" ]] && EK="$EK\"transient_slots\": $TRANSIENT_SLOTS,"
[[ -n "${KEEP_FREE_GB:-}" ]] && EK="$EK\"keep_free_gb\": $KEEP_FREE_GB,"
EK="${EK%,}}"
[[ "$EK" != "{}" ]] && FLAGS+=(--engine-kwargs "$EK")
[[ -n "$TRACE_USED" ]] && FLAGS+=(--trace-stats "$TRACE_USED")
# Rank > 0 is the headless worker: it loads its own half of the arena, joins the process
# group and serves rank 0's broadcast queue. It binds no port -- run its container with
# --no-healthcheck, since the image's HEALTHCHECK curls a socket this rank never opens.
if [[ "$WORLD_SIZE" != "1" && "$RANK" != "0" ]]; then
    FLAGS+=(--headless)
fi
# shellcheck disable=SC2206
[[ -n "$EXTRA_FLAGS" ]] && FLAGS+=($EXTRA_FLAGS)
FLAGS+=("$@")

info "model=$MODEL_DIR  max_seq=$MAX_SEQ  arena=${ARENA_GB:-auto}  spec=$SPEC  thinking=$DEFAULT_THINKING/$DEFAULT_EFFORT"
if [[ "$WORLD_SIZE" != "1" && "$RANK" != "0" ]]; then
    info "trace=${TRACE_USED:-none}  headless EP2 worker (no socket)"
else
    info "trace=${TRACE_USED:-none}  listen=$HOST:$PORT"
fi
info "the warm start reads tens of GB from NVMe before the socket is bound; /health is minutes away"

cd /app
exec python3 server/app.py "${FLAGS[@]}"
