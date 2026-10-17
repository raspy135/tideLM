#!/usr/bin/env bash
# Run a Muse test or tool on both Sparks: tools/muse_pair_run.sh <script in tools/> [args...]
#
# Disposable containers from an existing engine image, this checkout's engine/tools/server mounted
# read-only on both nodes (the peer needs this checkout at the same path; MUSE_SYNC=1 rsyncs it
# there first, excluding results/). Serving must be stopped. Logs: $MUSE_LOG_DIR/rank{0,1}.log.
#   MUSE_IMAGE      image id/tag present on both nodes (default: the V4.1 serving image)
#   MUSE_MODEL_DIR  host model directory, mounted at /models (optional)
#   MUSE_TIMEOUT    per-container wall clock (default 1800s)
#   MUSE_ENV        "MUSE_X=1 MUSE_Y=2": forwarded to both ranks
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
script=${1:?script inside tools/}
shift
PEER=${PEER:-ryan@10.0.0.2}
image=${MUSE_IMAGE:-deepseek-v41-flash-dgx-spark-dual-adaptive:20261011-topp}
log_dir=${MUSE_LOG_DIR:-$ROOT/results/muse-run-$(date +%Y%m%d-%H%M%S)}
timeout=${MUSE_TIMEOUT:-1800s}
for name in deepseek-v41-ep2-rank0 muse-rank0; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "Container $name is running: stop serving first" >&2; exit 1
    fi
done
if [[ "${MUSE_SYNC:-0}" == 1 ]]; then
    ssh -o BatchMode=yes "$PEER" "mkdir -p '$ROOT'"
    rsync -a --delete --exclude results/ --exclude .triton-cache/ --exclude .git/ "$ROOT/" "$PEER:$ROOT/"
fi
ssh -o BatchMode=yes "$PEER" "mkdir -p '$ROOT/results' '$ROOT/.triton-cache'"
mkdir -p "$ROOT/results" "$ROOT/.triton-cache" "$log_dir"
flags=(--rm --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ipc host
       --ulimit memlock=-1 --ulimit stack=67108864
       -v "$ROOT/engine:/app/engine:ro" -v "$ROOT/tools:/app/tools:ro" -v "$ROOT/server:/app/server:ro"
       -v "$ROOT/corpus:/app/corpus:ro"
       -v "$ROOT/results:/app/results" -v "$ROOT/.triton-cache:/app/.triton"
       -e TRITON_CACHE_DIR=/app/.triton
       -e WORLD_SIZE=2 -e MASTER_ADDR=10.0.0.1 -e MASTER_PORT=29641
       -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_DISABLE=0
       -e DSV41_DIST_TIMEOUT_S=900 -e PYTHONUNBUFFERED=1)
if [[ -n "${MUSE_MODEL_DIR:-}" ]]; then
    flags+=(-v "$MUSE_MODEL_DIR:/models:ro" -e MUSE_MODEL=/models)
fi
if [[ -n "${MUSE_DRAFTER_DIR:-}" ]]; then
    flags+=(-v "$MUSE_DRAFTER_DIR:/drafter:ro")
fi
if [[ -n "${MUSE_EXL3_DIR:-}" ]]; then         # EXL3 expert packs (tools/glm_pack_exl3.py), same path on both nodes
    flags+=(-v "$MUSE_EXL3_DIR:/packs:ro")
fi
for kv in ${MUSE_ENV:-}; do
    [[ "$kv" =~ ^(MUSE|DSV41)_[A-Z0-9_]+=[^[:space:]]*$ ]] || { echo "MUSE_ENV: bad entry '$kv'" >&2; exit 1; }
    flags+=(-e "$kv")
done
printf -v peer_cmd '%q ' docker run --name muse-rank1 "${flags[@]}" -e RANK=1 \
    --entrypoint timeout "$image" "$timeout" python "/app/tools/$script" "$@"
ssh -o BatchMode=yes "$PEER" "$peer_cmd" > "$log_dir/rank1.log" 2>&1 &
peer_pid=$!
docker run --name muse-rank0 "${flags[@]}" -e RANK=0 \
    --entrypoint timeout "$image" "$timeout" python "/app/tools/$script" "$@" \
    > "$log_dir/rank0.log" 2>&1 &
head_pid=$!
cleanup() {
    docker stop -t 1 muse-rank0 >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "$PEER" 'docker stop -t 1 muse-rank1 >/dev/null 2>&1 || true'
}
trap cleanup EXIT
rc=0
if ! wait -n "$head_pid" "$peer_pid"; then
    cleanup
    rc=1
fi
wait "$head_pid" || rc=1
wait "$peer_pid" || rc=1
echo "logs: $log_dir (exit $rc)"
exit "$rc"
