#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# rust-up.sh ENV_FILE -- serve a DeepSeek V4.1 profile with the Rust engine (rust/, the
# tidelm-server binary): rank 0 here, rank 1 on PEER. scripts/tide-up.sh runs this for a
# profile that sets TIDE_ENGINE=rust; scripts/tide-down.sh stops it.
#
# From the profile, as the Python launcher reads them: MODEL_DIR, SERVED_MODEL_NAME, HOST,
# PORT, MAX_SEQ, DSV41_MAX_CONCURRENCY (lanes), SPEC, PEER, MASTER_ADDR / MASTER_PORT, NCCL_*,
# KEEP_FREE_GB (the memory guard), DSV41_ADAPT_SENSITIVITY / _PRIOR / _DECODE_TOKENS,
# DSV41_PROBE_PREFILL, DSV41_PREFIX_CACHE / _SNAPSHOTS / _DISK / _DISK_GB /
# _DISK_MIN_INTERVAL_S / _DISK_DEDUP, DSV41_VISION, DSV41_HOT_READS / _ASYNC / _MIN / _MAX /
# _RING, DSV41_PACKED_KV, DSV41_SWA_REPLAY. The Python kernels' other DSV41_* knobs mean nothing here.
#
# The Rust engine's own, with their defaults:
#   RUST_BUDGET=10000        resident experts (next to 512K tokens of context per lane)
#   RUST_DEPTH=confidence    speculation depth policy (window, confidence, or a fixed depth)
#   RUST_ADAPT_DB=~/tidelm-rust-state/prune_demand_rust.npz
#                            its own demand history: the Python server's DSV41_PRUNE_DB is
#                            written by the container (root); the first launch copies it
#   RUST_TRACE=results/trace-union/stats/coverage.json   the routing trace the history blends with
#   RUST_PREFIX_DIR=~/tidelm-rust-state/prefix-cache     the prefix cache's disk tier
#   RUST_EXPERT_TOPP         unset: off (DSV41_EXPERT_TOPP=0.90 measured slower in Rust, rust/PORT.md)
#   RUST_COPY_DRAFT=1        the copy drafter (CPU prompt lookup in front of DSpark; 0 off)
# Logs: ~/tidelm-rust-state/server.log here, ~/tidelm-rust-bin/server_rank1.log on the peer.
# ---------------------------------------------------------------------------------------
#
# A GLM-5.3 profile (GLM_MODEL_DIR, GLM_* keys as scripts/glm-up.sh reads them) runs the GLM
# engine (rust/crates/tidelm-glm) instead: see the GLM section below for the keys it maps.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
f=${1:-${TIDE_ENV_FILE:-}}
[[ -n "$f" && -f "$f" ]] || err "usage: scripts/rust-up.sh ENV_FILE"
set -a; source "$f"; set +a
BIN="$ROOT/rust/target/release/tidelm-server"
[[ -x "$BIN" ]] || err "no $BIN: build it with (cd rust && cargo build --release)"
STATE="$HOME/tidelm-rust-state"
LOG="$STATE/server.log"
mkdir -p "$STATE"

# ---------------------------------------------------------------------------------------
# GLM-5.3: the profile's GLM_* keys, as engine/glm/tp.py reads them (its defaults where the
# profile leaves one out). The Rust engine implements the shipped profile's choices only and
# refuses the others rather than serving something else: GLM_KV=fp8, GLM_EXPERT_FORMAT=exl3
# (a native EXL3 checkpoint), GLM_HEAD=exl3, GLM_KVB_FP8=1, GLM_EMBED_8BIT=int8,
# GLM_MISS=substitute, no prefill prefetch (GLM_PF), no async hot reads, no index buckets.
# Fixed at the profile's values: GLM_HOT_MIN=0.2, GLM_HOT_MAX=16, GLM_HOT_PACE=1,
# GLM_ADAPT_DECAY=0.9, GLM_ADAPT_MARGIN=1.5, GLM_BG_READS=0. Not ported yet: GLM_PREFIX_DISK
# (the live sequence's prefix is reused; nothing is kept on disk).
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
    [[ "${GLM_PREFIX_DISK:-0}" == 1 ]] && echo "--- note: GLM_PREFIX_DISK is not ported to the Rust engine; only the live sequence's prefix is reused"
    pack=${GLM_EXL3_PACK:-}
    [[ -n "$pack" ]] || err "$f sets no GLM_EXL3_PACK"
    pack=${pack/#\/packs/${GLM_EXL3_DIR:-/packs}}
    db=${GLM_STATS_DB:-}
    db=${db/#\/app/$ROOT}
    [[ -n "$db" && -f "$db" ]] || db="$ROOT/profiles/glm53-experts-v1.pt"
    state=${RUST_GLM_STATE_DB:-$STATE/glm-stats-rust.npz}
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
    envs=(NCCL_PROTO=Simple GLM_REASONING_EFFORT="${GLM_REASONING_EFFORT:-medium}"
          GLM_DEFAULT_MAX_TOKENS="${GLM_DEFAULT_MAX_TOKENS:-4096}")
    for v in NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_ADDR_FAMILY NCCL_IB_ADDR_RANGE NCCL_IB_ROCE_VERSION_NUM \
             NCCL_IB_MERGE_NICS NCCL_CROSS_NIC; do
        [[ -n "${!v:-}" ]] && envs+=("$v=${!v}")
    done
    guard=${KEEP_FREE_GB:-4}
    pair=()
    if [[ -n "${PEER:-}" ]]; then
        addr="${MASTER_ADDR:-10.0.0.1}:${RUST_GLM_PORT:-29655}"
        pair=(--world 2 --addr "$addr")
        ssh "$PEER" 'mkdir -p ~/tidelm-rust-bin ~/tidelm-rust-state'
        rsync -a "$BIN" "$ROOT/scripts/memguard.sh" "$PEER:tidelm-rust-bin/"
    fi
    rank0=("${pair[@]}")
    [[ -n "${PEER:-}" ]] && rank0+=(--rank 0)
    (export "${envs[@]}"; exec setsid "$ROOT/scripts/memguard.sh" "$guard" "$BIN" "${args[@]}" "${rank0[@]}") \
        > "$LOG" 2>&1 < /dev/null &
    if [[ -n "${PEER:-}" ]]; then
        sleep 1
        ssh -n "$PEER" "cd ~/tidelm-rust-bin || exit 1; export $(printf '%q ' "${envs[@]}");
            nohup ./memguard.sh $guard ./tidelm-server $(printf '%q ' "${args[@]}" "${pair[@]}") --rank 1 \
            > server_rank1.log 2>&1 < /dev/null &"
    fi
    sleep 2
    for _ in $(seq 600); do
        grep -q "serving" "$LOG" && break
        if grep -qiE "error|panicked|GUARD" "$LOG" || ! pgrep -x tidelm-server > /dev/null; then
            tail -20 "$LOG" >&2
            err "the Rust server did not come up (log: $LOG)"
        fi
        sleep 1
    done
    grep -q "serving" "$LOG" || err "the Rust server is still not up after 600 s (log: $LOG)"
    grep -E "engine up" "$LOG"
    exit 0
fi

[[ -n "${MODEL_DIR:-}" ]] || err "$f sets no MODEL_DIR"

# the demand history: a copy of the Python server's on first use, on both boxes
db=${RUST_ADAPT_DB:-$STATE/prune_demand_rust.npz}
if [[ ! -f "$db" ]]; then
    src="${DSV41_PRUNE_DB:-}"; [[ "$src" == /* ]] || src="$ROOT/$src"
    [[ -f "$src" ]] || err "no demand history $db, and no DSV41_PRUNE_DB to start it from"
    cp "$src" "$db"
    echo "--- demand history: copied $src to $db"
fi

args=(--resident --graphs --model-dir "$MODEL_DIR" --served-model-name "${SERVED_MODEL_NAME:-deepseek}"
      --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --max-seq "${MAX_SEQ:-32768}"
      --lanes "${DSV41_MAX_CONCURRENCY:-1}" --vision "${DSV41_VISION:-1}"
      --adapt-db "$db" --trace "${RUST_TRACE:-$ROOT/results/trace-union/stats/coverage.json}"
      --budget "${RUST_BUDGET:-10000}" --adapt-sensitivity "${DSV41_ADAPT_SENSITIVITY:-high}"
      --adapt-prior "${DSV41_ADAPT_PRIOR:-4}" --adapt-decode-tokens "${DSV41_ADAPT_DECODE_TOKENS:-600}"
      --adapt-probe "${DSV41_PROBE_PREFILL:-1}"
      --prefix-cache "${DSV41_PREFIX_CACHE:-1}" --prefix-snapshots "${DSV41_PREFIX_SNAPSHOTS:-16}"
      --copy-draft "${RUST_COPY_DRAFT:-1}" --swa-replay "${DSV41_SWA_REPLAY:-1}")
if [[ "${SPEC:-1}" == 0 ]]; then args+=(--depth plain); else args+=(--depth "${RUST_DEPTH:-confidence}"); fi
# async hot reads plan from one lane's verify rows: on only with one lane
if [[ "${DSV41_HOT_READS:-1}" == 1 && "${DSV41_HOT_ASYNC:-1}" == 1 && "${DSV41_MAX_CONCURRENCY:-1}" == 1 ]]; then
    args+=(--hot-ring "${DSV41_HOT_RING:-128}")
else
    args+=(--hot-ring 0)
fi
if [[ "${DSV41_PREFIX_DISK:-1}" == 1 ]]; then
    args+=(--prefix-disk "${RUST_PREFIX_DIR:-$STATE/prefix-cache}" --prefix-disk-gb "${DSV41_PREFIX_DISK_GB:-20}"
           --prefix-disk-interval "${DSV41_PREFIX_DISK_MIN_INTERVAL_S:-300}" --prefix-disk-dedup "${DSV41_PREFIX_DISK_DEDUP:-1}")
fi

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
    [[ -n "${MASTER_ADDR:-}" ]] || err "$f sets PEER but no MASTER_ADDR"
    addr="$MASTER_ADDR:${MASTER_PORT:-29960}"
    pair=(--world 2 --addr "$addr")
    # the same binary on both ranks (the boot guard compares them too)
    ssh "$PEER" 'mkdir -p ~/tidelm-rust-bin ~/tidelm-rust-state'
    rsync -a "$BIN" "$ROOT/scripts/memguard.sh" "$PEER:tidelm-rust-bin/"
    ssh "$PEER" "test -f '$db'" || scp -q "$db" "$PEER:$db"
fi

rank0=("${pair[@]}")
[[ -n "${PEER:-}" ]] && rank0+=(--rank 0)
(export "${envs[@]}"; exec setsid "$ROOT/scripts/memguard.sh" "$guard" "$BIN" "${args[@]}" "${rank0[@]}") \
    > "$LOG" 2>&1 < /dev/null &
if [[ -n "${PEER:-}" ]]; then
    sleep 1
    # only the nohup is backgrounded, with its own stdio: ssh returns at once
    ssh -n "$PEER" "cd ~/tidelm-rust-bin || exit 1; export $(printf '%q ' "${envs[@]}");
        nohup ./memguard.sh $guard ./tidelm-server $(printf '%q ' "${args[@]}" "${pair[@]}") --rank 1 \
        > server_rank1.log 2>&1 < /dev/null &"
fi
sleep 2
for _ in $(seq 300); do
    grep -q "serving" "$LOG" && break
    if grep -qiE "error|panicked|GUARD" "$LOG" || ! pgrep -x tidelm-server > /dev/null; then
        tail -20 "$LOG" >&2
        err "the Rust server did not come up (log: $LOG)"
    fi
    sleep 1
done
grep -q "serving" "$LOG" || err "the Rust server is still not up after 300 s (log: $LOG)"
grep -E "engine up" "$LOG"
