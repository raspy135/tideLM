#!/usr/bin/env bash
# glm-up.sh -- serve GLM-5.3 on both Sparks (rank 0 here, rank 1 on $PEER); scripts/dual-down.sh
# stops it.
#
# Containers glm-serve-rank0/1 from an existing engine image with this checkout's engine/,
# tools/ and server/ mounted read-only, so a code change needs a restart, not a rebuild. The
# checkpoint must exist at the same path on both nodes. Settings come from .env.glm (copy
# .env.glm.example); MUSE_* and the few launch variables below are forwarded to both ranks.
# The V4.1 deployment (deepseek-v41-ep2-rank*) cannot serve at the same time (each fills the
# GPUs); dual-down.sh stops whichever pair is running.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ENV_FILE=${MUSE_ENV_FILE:-$ROOT/.env.glm}
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE (copy .env.glm.example)" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
PEER=${PEER:-ryan@10.0.0.2}
IMAGE=${MUSE_IMAGE:-deepseek-v41-flash-dgx-spark-dual-adaptive:20261011-topp}
MODEL_DIR=${MUSE_MODEL_DIR:?set MUSE_MODEL_DIR in $ENV_FILE}
PORT=${PORT:-8000}
BIND_HOST=${BIND_HOST:-0.0.0.0}
MASTER_ADDR=${MASTER_ADDR:-10.0.0.1}
MASTER_PORT=${MASTER_PORT:-29651}
IFACE=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
for name in deepseek-v41-ep2-rank0 glm-serve-rank0; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "container $name exists: stop it first (scripts/dual-down.sh)" >&2
        exit 1
    fi
done
test -f "$MODEL_DIR/config.json" || { echo "no checkpoint at $MODEL_DIR" >&2; exit 1; }
ssh -o BatchMode=yes "$PEER" "test -f '$MODEL_DIR/config.json'" \
    || { echo "no checkpoint at $PEER:$MODEL_DIR" >&2; exit 1; }
# The peer runs the same code: sync this checkout (code only) before starting.
rsync -a --delete --exclude results/ --exclude .triton-cache/ --exclude .git/ "$ROOT/" "$PEER:$ROOT/"
ssh -o BatchMode=yes "$PEER" "mkdir -p '$ROOT/results' '$ROOT/.triton-cache'"
mkdir -p "$ROOT/results" "$ROOT/.triton-cache"
flags=(-d --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ipc host
       --ulimit memlock=-1 --ulimit stack=67108864 --restart no
       -v "$ROOT/engine:/app/engine:ro" -v "$ROOT/tools:/app/tools:ro" -v "$ROOT/server:/app/server:ro"
       -v "$ROOT/results:/app/results" -v "$ROOT/.triton-cache:/app/.triton"
       -v "$MODEL_DIR:/models:ro"
       ${MUSE_DRAFTER_DIR:+-v "$MUSE_DRAFTER_DIR:/drafter:ro" -e MUSE_DRAFTER=/drafter}
       -e TRITON_CACHE_DIR=/app/.triton -e MUSE_MODEL=/models -e PYTHONUNBUFFERED=1
       -e WORLD_SIZE=2 -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT"
       -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_IB_DISABLE=0
       -e DSV41_DIST_TIMEOUT_S="${DSV41_DIST_TIMEOUT_S:-1800}"
       -e HOST="$BIND_HOST" -e PORT="$PORT" -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.3}")
while IFS='=' read -r key _; do
    flags+=(-e "$key=${!key}")
done < <(env | grep -E '^MUSE_[A-Z0-9_]+=' | grep -v '^MUSE_MODEL_DIR=\|^MUSE_IMAGE=\|^MUSE_ENV_FILE=\|^MUSE_DRAFTER_DIR=\|^MUSE_DRAFTER=')
docker run --name glm-serve-rank0 "${flags[@]}" -e RANK=0 \
    --entrypoint python "$IMAGE" /app/server/muse_app.py >/dev/null
printf -v peer_cmd '%q ' docker run --name glm-serve-rank1 "${flags[@]}" -e RANK=1 \
    --entrypoint python "$IMAGE" /app/server/muse_app.py
ssh -o BatchMode=yes "$PEER" "$peer_cmd" >/dev/null
echo "--- started; waiting for http://127.0.0.1:$PORT/health (logs: docker logs -f glm-serve-rank0)"
for i in $(seq 1 1800); do
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "--- up after ${i}s"; exit 0
    fi
    if ! docker container inspect -f '{{.State.Running}}' glm-serve-rank0 2>/dev/null | grep -q true; then
        echo "--- rank 0 exited:"; docker logs --tail 40 glm-serve-rank0; exit 1
    fi
    if ! ssh -o BatchMode=yes "$PEER" "docker container inspect -f '{{.State.Running}}' glm-serve-rank1" 2>/dev/null | grep -q true; then
        echo "--- rank 1 exited:"; ssh -o BatchMode=yes "$PEER" "docker logs --tail 40 glm-serve-rank1"; exit 1
    fi
    sleep 1
done
echo "--- not healthy after 30 min" >&2
exit 1
