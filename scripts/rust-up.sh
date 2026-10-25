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
# _RING, DSV41_PACKED_KV. The Python kernels' other DSV41_* knobs mean nothing here.
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
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
err() { echo "ERROR: $*" >&2; exit 1; }
f=${1:-${TIDE_ENV_FILE:-}}
[[ -n "$f" && -f "$f" ]] || err "usage: scripts/rust-up.sh ENV_FILE"
set -a; source "$f"; set +a
BIN="$ROOT/rust/target/release/tidelm-server"
[[ -x "$BIN" ]] || err "no $BIN: build it with (cd rust && cargo build --release)"
[[ -n "${MODEL_DIR:-}" ]] || err "$f sets no MODEL_DIR"
STATE="$HOME/tidelm-rust-state"
LOG="$STATE/server.log"
mkdir -p "$STATE"

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
      --copy-draft "${RUST_COPY_DRAFT:-1}")
if [[ "${SPEC:-1}" == 0 ]]; then args+=(--depth plain); else args+=(--depth "${RUST_DEPTH:-confidence}"); fi
if [[ "${DSV41_HOT_READS:-1}" == 1 && "${DSV41_HOT_ASYNC:-1}" == 1 ]]; then
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
