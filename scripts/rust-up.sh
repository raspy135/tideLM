#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# rust-up.sh ENV_FILE -- serve a DeepSeek V4.1 profile with the Rust engine (rust/, the
# tidelm-server binary): rank 0 here, rank 1 on PEER, each in a container of the serving image
# (`tidelm-rank0`, `tidelm-rank1`; `scripts/tide.sh build` builds the binary into it and ships
# it). `scripts/tide.sh start` (tide-up.sh) runs this for every profile the Rust engine can serve
# unless it sets TIDE_ENGINE=python; `scripts/tide.sh stop` stops it.
#
# rust-up.sh --check ENV_FILE only validates: exit 0 if the Rust engine serves the profile, else
# the reason on stderr and exit 1, with nothing started or stopped (tide-up.sh asks it first).
#
# From the profile, as the Python launcher reads them: MODEL_DIR, SERVED_MODEL_NAME, HOST,
# PORT, MAX_SEQ, SPEC, PEER, MASTER_ADDR / MASTER_PORT, NCCL_*,
# KEEP_FREE_GB (the memory guard), DSV41_ADAPT_SENSITIVITY / _PRIOR / _DECODE_TOKENS,
# DSV41_PROBE_PREFILL, DSV41_PREFIX_CACHE / _SNAPSHOTS / _DISK / _DISK_GB /
# _DISK_MIN_INTERVAL_S / _DISK_DEDUP, DSV41_VISION, DSV41_HOT_READS / _ASYNC / _MIN / _MAX /
# _RING, DSV41_PACKED_KV, DSV41_SWA_REPLAY. The Python kernels' other DSV41_* knobs mean nothing here.
#
# The Rust engine's own, with their defaults:
#   RUST_BUDGET=10000        resident experts (next to 512K tokens of context per lane)
#   RUST_LANES=1             decode lanes (the profile serves 2), sharing one hot-read ring
#   RUST_DEPTH=confidence    speculation depth policy (window, confidence, or a fixed depth)
#   RUST_ADAPT_DB=~/tidelm-rust-state/prune_demand_rust.npz
#                            its own demand history: the first launch copies the Python server's
#                            (DSV41_PRUNE_DB, written by its container as root), else the bundled
#                            seed profiles/learned-experts-v1.npz
#   RUST_TRACE=results/trace-union/stats/coverage.json   the routing trace the history blends with
#   RUST_PREFIX_DIR=~/tidelm-rust-state/prefix-cache     the prefix cache's disk tier
#   RUST_EXPERT_TOPP         unset: off (DSV41_EXPERT_TOPP=0.90 measured slower in Rust, rust/PORT.md)
#   RUST_COPY_DRAFT=1        the copy drafter (CPU prompt lookup in front of DSpark; 0 off)
#
# Both models: the image is IMAGE (or GLM_IMAGE), else tidelm:local, as dual-build.sh names it.
# Its tidelm.rust_sha label must match this checkout's Rust sources (scripts/code_digest.sh
# --rust), and the worker must hold the same image id; a mismatch is refused with the command
# that fixes it. RUST_BIN=path runs a binary built on this box instead (cargo build --release in
# rust/, for development): mounted over the image's on both ranks, copied to the worker first.
# The containers run as the calling user, with the model, its packs and ~/tidelm-rust-state
# mounted at their own paths. Logs: `scripts/tide.sh logs -f` (one line a request: prefill and
# decode speeds; --rank 1 the worker's).
# ---------------------------------------------------------------------------------------
#
# A GLM-5.3 profile (GLM_MODEL_DIR, GLM_* keys as scripts/glm-up.sh reads them) runs the GLM
# engine (rust/crates/tidelm-glm) instead: see the GLM section below for the keys it maps.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
check=0
[[ "${1:-}" == --check ]] && { check=1; shift; }
f=${1:-${TIDE_ENV_FILE:-}}
[[ -n "$f" && -f "$f" ]] || err "usage: scripts/rust-up.sh [--check] ENV_FILE"
set -a; source "$f"; set +a
STATE="$HOME/tidelm-rust-state"
mkdir -p "$STATE"
# host paths the containers see at the same path: read-only, and read-write (created first)
ro=() rw=("$STATE")

# ---------------------------------------------------------------------------------------
# GLM-5.3: the profile's GLM_* keys, as engine/glm/tp.py reads them (its defaults where the
# profile leaves one out). The Rust engine implements the shipped profile's choices only and
# refuses the others rather than serving something else: GLM_KV=fp8, GLM_EXPERT_FORMAT=exl3
# (a native EXL3 checkpoint), GLM_HEAD=exl3, GLM_KVB_FP8=1, GLM_EMBED_8BIT=int8,
# GLM_MISS=substitute, no prefill prefetch (GLM_PF), no async hot reads, no index buckets, and
# GLM_PREFILL_REDUCE bf16 or fp32. Fixed, and not read: GLM_HOT_MIN=0.2, GLM_HOT_MAX=16,
# GLM_HOT_PACE=1, GLM_ADAPT_DECAY=0.9, GLM_ADAPT_MARGIN=1.5, GLM_BG_READS=0 (the Python engine's
# keys, at its shipped values). GLM_PREFIX_DISK=1 keeps prompts'
# rows on disk under its own directory (GLM_PREFIX_DISK_GB a node, GLM_PREFIX_DISK_MIN new tokens
# to save or restore; never the Python store's files, whose layout differs).
#   RUST_GLM_PREFIX_DIR=~/tidelm-rust-state/glm-prefix   the disk prefix store
#   RUST_GLM_STATE_DB=~/tidelm-rust-state/glm-stats-rust.npz   the routing history the engine
#       writes (read at boot when present, else GLM_STATS_DB: the Python server's db.pt is
#       written by its container as root and is never touched)
# ---------------------------------------------------------------------------------------
if [[ -n "${GLM_MODEL_DIR:-}" ]]; then
    MODEL_DIR=$GLM_MODEL_DIR
    want() { [[ "${!1:-$2}" == "$3" ]] || err "the Rust GLM engine runs $1=$3 only (profile: ${!1:-$2})"; }
    want GLM_KV bf16 fp8
    want GLM_EXPERT_FORMAT mxfp4 exl3
    want GLM_HEAD exl3 exl3
    want GLM_KVB_FP8 0 1
    want GLM_EMBED_8BIT "" int8
    want GLM_MISS substitute substitute
    want GLM_PF 0 0
    want GLM_HOT_ASYNC 0 0
    want GLM_INDEX_BUCKETS 0 0
    [[ "${GLM_PREFILL_REDUCE:-fp32}" == bf16 || "${GLM_PREFILL_REDUCE:-fp32}" == fp32 ]] \
        || err "the Rust GLM engine sums prefill partials in bf16 or fp32 only (profile: GLM_PREFILL_REDUCE=$GLM_PREFILL_REDUCE)"
    pack=${GLM_EXL3_PACK:-}
    [[ -n "$pack" ]] || err "$f sets no GLM_EXL3_PACK"
    [[ $check == 1 ]] && exit 0
    pack=${pack/#\/packs/${GLM_EXL3_DIR:-/packs}}
    db=${GLM_STATS_DB:-}
    db=${db/#\/app/$ROOT}
    if [[ -n "$db" && -f "$db" ]]; then ro+=("$(dirname "$db")"); else db=/app/profiles/glm53-experts-v1.pt; fi
    state=${RUST_GLM_STATE_DB:-$STATE/glm-stats-rust.npz}
    ro+=("$MODEL_DIR" "$(dirname "${pack//\{rank\}/0}")") rw+=("$(dirname "$state")")
    depth=${GLM_MTP_DEPTH:-3}
    [[ "${GLM_MTP:-1}" == 0 ]] && depth=0
    gate=(--dyn "${GLM_MTP_DYN:-0}")
    [[ -n "${GLM_MTP_CAL:-}" && "${GLM_MTP_CAL}" != 0 ]] && gate=(--cal "$GLM_MTP_CAL")
    bg=0
    [[ "${GLM_BG:-0}" == 1 ]] && bg=${GLM_BG_EVERY:-2}
    args=(--model-dir "$MODEL_DIR" --pack "$pack" --db "$db" --state-db "$state"
          --served-model-name "${SERVED_MODEL_NAME:-glm-5.3}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
          --slots "${GLM_SLOTS:-12500}" --max-seq "${GLM_MAX_SEQ:-32768}" --chunk "${GLM_CHUNK:-1024}"
          --kv-split "${GLM_KV_SPLIT:-0}" --reduce "${GLM_PREFILL_REDUCE:-fp32}"
          --hot "${GLM_HOT:-0}" --hot-keep "${GLM_HOT_KEEP:-0}" --depth "$depth" "${gate[@]}" --lookup "${GLM_LOOKUP:-0}"
          --adapt "${GLM_ADAPT:-1}" --adapt-every "${GLM_ADAPT_EVERY:-200}" --adapt-prefill "${GLM_ADAPT_PREFILL_SWAPS:-2048}"
          --adapt-decode "${GLM_ADAPT_DECODE_SWAPS:-256}" --bg-every "$bg" --loop-break "${GLM_LOOP_BREAK:-0}"
          --two-pass "${GLM_PREFILL_TWO_PASS:-0}" --two-pass-miss "${GLM_PREFILL_TWO_PASS_MISS:-0}")
    [[ "${GLM_GRAPHS:-1}" == 1 ]] && args+=(--graphs)
    if [[ "${GLM_PREFIX_DISK:-0}" == 1 ]]; then
        args+=(--prefix-disk "${RUST_GLM_PREFIX_DIR:-$STATE/glm-prefix}"
               --prefix-disk-gb "${GLM_PREFIX_DISK_GB:-20}" --prefix-min "${GLM_PREFIX_DISK_MIN:-256}")
        rw+=("${RUST_GLM_PREFIX_DIR:-$STATE/glm-prefix}")
    fi
    envs=(NCCL_PROTO=Simple GLM_REASONING_EFFORT="${GLM_REASONING_EFFORT:-medium}"
          GLM_DEFAULT_MAX_TOKENS="${GLM_DEFAULT_MAX_TOKENS:-4096}")
    for v in NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_ADDR_FAMILY NCCL_IB_ADDR_RANGE NCCL_IB_ROCE_VERSION_NUM \
             NCCL_IB_MERGE_NICS NCCL_CROSS_NIC; do
        [[ -n "${!v:-}" ]] && envs+=("$v=${!v}")
    done
    guard=${KEEP_FREE_GB:-4}
    pair=()
    [[ -n "${PEER:-}" ]] && pair=(--world 2 --addr "${MASTER_ADDR:-10.0.0.1}:${RUST_GLM_PORT:-29655}")
else   # DeepSeek V4.1, to the `fi` before the launch

[[ -n "${MODEL_DIR:-}" ]] || err "$f sets no MODEL_DIR"

# the demand history: on first use a copy of the Python server's (DSV41_PRUNE_DB), else of the
# bundled seed the Python engine starts a fresh install from (DSV41_EXPERT_SEED=1); on both boxes
db=${RUST_ADAPT_DB:-$STATE/prune_demand_rust.npz}
if [[ ! -f "$db" ]]; then
    src="${DSV41_PRUNE_DB:-}"; [[ -z "$src" || "$src" == /* ]] || src="$ROOT/$src"
    [[ -n "$src" && -f "$src" ]] || src="$ROOT/profiles/learned-experts-v1.npz"
    if [[ $check == 0 ]]; then
        mkdir -p "$(dirname "$db")"
        cp "$src" "$db"
        echo "--- demand history: copied $src to $db"
    fi
fi
[[ -z "${PEER:-}" || -n "${MASTER_ADDR:-}" ]] || err "$f sets PEER but no MASTER_ADDR"
[[ $check == 1 ]] && exit 0

lanes=${RUST_LANES:-1}
args=(--resident --graphs --model-dir "$MODEL_DIR" --served-model-name "${SERVED_MODEL_NAME:-deepseek}"
      --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --max-seq "${MAX_SEQ:-32768}"
      --lanes "$lanes" --vision "${DSV41_VISION:-1}"
      --adapt-db "$db" --trace "${RUST_TRACE:-/app/results/trace-union/stats/coverage.json}"
      --budget "${RUST_BUDGET:-10000}" --adapt-sensitivity "${DSV41_ADAPT_SENSITIVITY:-high}"
      --adapt-prior "${DSV41_ADAPT_PRIOR:-4}" --adapt-decode-tokens "${DSV41_ADAPT_DECODE_TOKENS:-600}"
      --adapt-probe "${DSV41_PROBE_PREFILL:-1}"
      --prefix-cache "${DSV41_PREFIX_CACHE:-1}" --prefix-snapshots "${DSV41_PREFIX_SNAPSHOTS:-16}"
      --copy-draft "${RUST_COPY_DRAFT:-1}" --swa-replay "${DSV41_SWA_REPLAY:-1}")
if [[ "${SPEC:-1}" == 0 ]]; then args+=(--depth plain); else args+=(--depth "${RUST_DEPTH:-confidence}"); fi
if [[ "${DSV41_HOT_READS:-1}" == 1 && "${DSV41_HOT_ASYNC:-1}" == 1 ]]; then
    args+=(--hot-ring "${DSV41_HOT_RING:-128}")
else
    args+=(--hot-ring 0)
fi
if [[ "${DSV41_PREFIX_DISK:-1}" == 1 ]]; then
    args+=(--prefix-disk "${RUST_PREFIX_DIR:-$STATE/prefix-cache}" --prefix-disk-gb "${DSV41_PREFIX_DISK_GB:-20}"
           --prefix-disk-interval "${DSV41_PREFIX_DISK_MIN_INTERVAL_S:-300}" --prefix-disk-dedup "${DSV41_PREFIX_DISK_DEDUP:-1}")
    rw+=("${RUST_PREFIX_DIR:-$STATE/prefix-cache}")
fi
ro+=("$MODEL_DIR") rw+=("$(dirname "$db")")
[[ -n "${RUST_TRACE:-}" ]] && ro+=("$(dirname "$RUST_TRACE")")

# what both ranks must run with (the boot guard compares it): RoCE, hot reads, packed KV
envs=(NCCL_PROTO=Simple TIDELM_PACKED_KV="${DSV41_PACKED_KV:-1}" TIDELM_HOT_MIN="${DSV41_HOT_MIN:-0.3}"
      TIDELM_HOT_MAX="${DSV41_HOT_MAX:-2}")
for v in NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_ADDR_FAMILY NCCL_IB_ADDR_RANGE NCCL_IB_ROCE_VERSION_NUM \
         NCCL_IB_MERGE_NICS NCCL_CROSS_NIC; do
    [[ -n "${!v:-}" ]] && envs+=("$v=${!v}")
done
[[ -n "${RUST_EXPERT_TOPP:-}" ]] && envs+=(TIDELM_EXPERT_TOPP="$RUST_EXPERT_TOPP")
guard=${KEEP_FREE_GB:-6}

pair=()
if [[ -n "${PEER:-}" ]]; then
    pair=(--world 2 --addr "$MASTER_ADDR:${MASTER_PORT:-29960}")
    # the history on both boxes
    ssh "$PEER" "mkdir -p '$(dirname "$db")'"
    ssh "$PEER" "test -f '$db'" || scp -q "$db" "$PEER:$db"
fi
fi   # the model

# ---------------------------------------------------------------------------------------
# Both models: rank 0 here and rank 1 on PEER, each in a container of the serving image, as the
# calling user, with every path the arguments name mounted at that path.
# ---------------------------------------------------------------------------------------
IMAGE=${IMAGE:-${GLM_IMAGE:-tidelm:local}}
img=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) \
    || err "no image $IMAGE on this box: scripts/tide.sh build $f builds it and ships it to the worker"
bin=() peer_bin=() envf=()
if [[ -n "${RUST_BIN:-}" ]]; then
    RUST_BIN=$(realpath "$RUST_BIN")
    [[ -x "$RUST_BIN" ]] || err "RUST_BIN=$RUST_BIN is not an executable"
    # a host build's rpath names ~/.local/nccl, which the container does not have
    bin=(-v "$RUST_BIN:/usr/local/bin/tidelm-server:ro")
    peer_bin=(-v "$HOME/tidelm-rust-bin/tidelm-server:/usr/local/bin/tidelm-server:ro")
    envf=(-e LD_LIBRARY_PATH=/opt/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64)
else
    want=$("$ROOT/scripts/code_digest.sh" --rust "$ROOT")
    have=$(docker image inspect -f '{{index .Config.Labels "tidelm.rust_sha"}}' "$IMAGE")
    [[ "$have" == "$want" ]] || err "$IMAGE's Rust engine was built from other sources than this checkout \
(image $have, checkout $want): scripts/tide.sh build $f rebuilds and ships it"
fi
for e in "${envs[@]}"; do envf+=(-e "$e"); done
vols=() seen=" "
for p in "${rw[@]}"; do [[ $seen == *" $p "* ]] && continue; seen+="$p "; vols+=(-v "$p:$p"); done
for p in "${ro[@]}"; do [[ $seen == *" $p "* ]] && continue; seen+="$p "; vols+=(-v "$p:$p:ro"); done
mkdir -p "${rw[@]}"   # docker would create a missing one as root
dflags=(-d --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ipc host
        --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536
        --entrypoint /app/scripts/memguard.sh)
if [[ -n "${PEER:-}" ]]; then
    peer_img=$(ssh "$PEER" "docker image inspect -f '{{.Id}}' '$IMAGE'" 2>/dev/null) || peer_img="none"
    [[ "$peer_img" == "$img" ]] || err "the worker's $IMAGE ($peer_img) is not this box's ($img): \
scripts/tide.sh build $f ships it"
    ssh "$PEER" "mkdir -p $(printf '%q ' "${rw[@]}") ~/tidelm-rust-bin"
    [[ -n "${RUST_BIN:-}" ]] && rsync -a "$RUST_BIN" "$PEER:tidelm-rust-bin/tidelm-server"
fi

rank0=("${pair[@]}")
[[ -n "${PEER:-}" ]] && rank0+=(--rank 0)
docker rm -f tidelm-rank0 > /dev/null 2>&1 || true
docker run "${dflags[@]}" --name tidelm-rank0 --user "$(id -u):$(id -g)" -e HOME="$HOME" \
    --health-cmd "curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1" --health-interval 30s \
    --health-start-period 10m "${vols[@]}" "${envf[@]}" "${bin[@]}" \
    "$IMAGE" "$guard" tidelm-server "${args[@]}" "${rank0[@]}" > /dev/null
if [[ -n "${PEER:-}" ]]; then
    ssh -n "$PEER" "docker rm -f tidelm-rank1 > /dev/null 2>&1; docker run $(printf '%q ' "${dflags[@]}") \
        --name tidelm-rank1 --user \$(id -u):\$(id -g) -e HOME=\$HOME --no-healthcheck \
        $(printf '%q ' "${vols[@]}" "${envf[@]}" "${peer_bin[@]}" "$IMAGE" "$guard" tidelm-server "${args[@]}" "${pair[@]}") \
        --rank 1 > /dev/null"
fi
running() { [[ $(docker inspect -f '{{.State.Running}}' tidelm-rank0 2>/dev/null) == true ]]; }
for _ in $(seq 600); do
    log=$(docker logs tidelm-rank0 2>&1)
    grep -q "serving" <<< "$log" && break
    if grep -qiE "error|panicked|GUARD" <<< "$log" || ! running; then
        tail -20 <<< "$log" >&2
        [[ -n "${PEER:-}" ]] && ssh "$PEER" 'docker logs --tail 10 tidelm-rank1 2>&1' | sed 's/^/rank 1: /' >&2
        err "the Rust server did not come up (docker logs tidelm-rank0)"
    fi
    sleep 1
done
grep -q "serving" <<< "$log" || err "the Rust server is still not up after 600 s (docker logs tidelm-rank0)"
grep -E "engine up" <<< "$log"
echo "--- logs: scripts/tide.sh logs -f${PEER:+   (--rank 1 for the worker)}"
