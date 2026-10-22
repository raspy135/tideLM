#!/usr/bin/env bash
# Disposable, bounded full-engine test. Serving must already be stopped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
gate_image=${GATE_IMAGE:?set GATE_IMAGE to an immutable image id available on both nodes}
gate_log_dir=${GATE_LOG_DIR:?set GATE_LOG_DIR}
gate_script=${1:?script inside /app/tools}
gate_host_script=${GATE_HOST_SCRIPT:-0}
gate_source_root=${GATE_SOURCE_ROOT:-}
# Per-container wall-clock cap; a multi-arm quality run needs more than a smoke gate.
gate_timeout=${GATE_TIMEOUT:-900s}
[[ "$gate_timeout" =~ ^[0-9]+[smh]?$ ]] || { echo "GATE_TIMEOUT: bad value '$gate_timeout'" >&2; exit 1; }
shift
gate_profile=${GATE_PROFILE:-.env}
[[ -f "$gate_profile" ]] || { echo "Missing gate profile: $gate_profile" >&2; exit 1; }
set -a
source "$gate_profile"
set +a
# GATE_ENV="DSV41_BLOCK=5 DSV41_X=1": override .env for this gate only (both ranks). Sourcing
# .env would otherwise overwrite anything exported by the caller.
for kv in ${GATE_ENV:-}; do
    [[ "$kv" =~ ^DSV41_[A-Z0-9_]+=[^[:space:]]*$ ]] || { echo "GATE_ENV: bad entry '$kv'" >&2; exit 1; }
    export "$kv"
done
if docker container inspect deepseek-v41-ep2-rank0 >/dev/null 2>&1; then
    echo 'Stop serving before the gate' >&2; exit 1
fi
flags=(--rm --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ipc host
       --ulimit memlock=-1 --ulimit stack=67108864
       -v "$(dirname "$MODEL_DIR"):/models:ro" -v "$ROOT/results:/app/results"
       -v "$ROOT/.triton-cache:/app/.triton"
       -e WORLD_SIZE=2 -e MASTER_ADDR=10.0.0.1 -e MASTER_PORT=29629
       -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_DISABLE=0
       -e DSV41_DIST_TIMEOUT_S=180 -e "MODEL_DIR=/models/$(basename "$MODEL_DIR")")
# Forward transport configuration to both ranks as well as engine configuration.
for key in NCCL_IB_HCA NCCL_IB_ADDR_FAMILY NCCL_IB_ADDR_RANGE NCCL_IB_ROCE_VERSION_NUM \
           NCCL_IB_MERGE_NICS NCCL_CROSS_NIC; do
    [[ -n "${!key:-}" ]] && flags+=(-e "$key=${!key}")
done
# grep, not rg: rg is often a shell function or alias that a script does not inherit. When it
# was missing here, the process substitution failed silently and BOTH ranks ran on code
# defaults (EP2, DSV41_BLOCK=5, no overlap) while .env said TP2/BLOCK=3 -- the boot guard
# cannot catch that, because the two ranks agree with each other.
forwarded=0
while IFS='=' read -r key _; do
    flags+=(-e "$key=${!key}")
    forwarded=$((forwarded + 1))
done < <(env | grep -E '^DSV41_[A-Z0-9_]+=')
if grep -qE '^DSV41_[A-Z0-9_]+=' "$gate_profile" && (( forwarded == 0 )); then
    echo '.env sets DSV41_* but none were forwarded to the gate containers' >&2; exit 1
fi
flags+=(-e DSV41_CAPTURE_NEXT=0 -e DSV41_PREFIX_DISK_DIR=/app/results/prefix-cache-gate
        -e DSV41_PREFIX_DISK_GB=20)
if [[ -n "$gate_source_root" ]]; then
    # Isolated historical-source comparison; never overwrite either live checkout.
    for directory in engine tools server; do
        test -d "$gate_source_root/$directory"
        ssh -o BatchMode=yes "$PEER" "test -d '$gate_source_root/$directory'"
        flags+=(-v "$gate_source_root/$directory:/app/$directory:ro")
    done
fi
gate_entry="/app/tools/$gate_script"
if [[ "$gate_host_script" == 1 ]]; then
    # Overlay only the test driver, never live engine modules. This avoids rebuilding a
    # runtime image when adding an assertion and verifies the exact same driver on both nodes.
    gate_digest=$(sha256sum "$ROOT/tools/$gate_script" | cut -d ' ' -f 1)
    gate_copy="/tmp/dsv41-gate-$gate_digest.py"
    cp "$ROOT/tools/$gate_script" "$gate_copy"
    scp -q "$gate_copy" "$PEER:$gate_copy"
    flags+=(-v "$gate_copy:/tmp/gate.py:ro")
    gate_entry=/tmp/gate.py
fi
mkdir -p "$gate_log_dir"
printf -v peer_cmd '%q ' docker run --name deepseek-tp-prefix-gate-rank1 "${flags[@]}" -e RANK=1 \
    --entrypoint timeout "$gate_image" "$gate_timeout" python "$gate_entry" "$@"
ssh -o BatchMode=yes "$PEER" "$peer_cmd" > "$gate_log_dir/rank1.log" 2>&1 &
peer_pid=$!
docker run --name deepseek-tp-prefix-gate-rank0 "${flags[@]}" -e RANK=0 \
    --entrypoint timeout "$gate_image" "$gate_timeout" python "$gate_entry" "$@" \
    > "$gate_log_dir/rank0.log" 2>&1 &
head_pid=$!
cleanup() {
    docker stop -t 1 deepseek-tp-prefix-gate-rank0 >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "$PEER" 'docker stop -t 1 deepseek-tp-prefix-gate-rank1 >/dev/null 2>&1 || true'
    docker rm deepseek-tp-prefix-gate-rank0 >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "$PEER" 'docker rm deepseek-tp-prefix-gate-rank1 >/dev/null 2>&1 || true'
}
trap cleanup EXIT
rc=0
# A rank-local assertion must not leave the other rank waiting in a collective
# for the full 900-second gate timeout before cleanup can run.
if ! wait -n "$head_pid" "$peer_pid"; then
    cleanup
    rc=1
fi
wait "$head_pid" || rc=1
wait "$peer_pid" || rc=1
exit "$rc"
