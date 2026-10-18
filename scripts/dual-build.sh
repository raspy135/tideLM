#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# dual-build.sh -- build the serving image once and put the SAME image on both Sparks.
#
# It builds locally and ships the result over the 200G link with `docker save | docker load`
# rather than running `docker build` on each box. That is deliberate: two independent builds
# of this Dockerfile are not guaranteed to be the same image (apt and PyPI both move under
# you, and the base tag is mutable), and EP2 is exactly the workload that punishes skew --
# two ranks running different torch or different engine code do not disagree about an answer,
# they wedge on a mismatched collective. One build, one image id, verified on both sides.
#
# The transfer is ~10 GB. Over the direct-attach link that is seconds; over WiFi it is not,
# so this uses the same PEER address the engine does.
#
# Usage: scripts/dual-build.sh [--local-only]
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

# shellcheck disable=SC1091
_IMAGE_CLI="${IMAGE:-}"          # a caller's IMAGE=... names the build; .env's must not override it
[[ -f .env ]] && { set -a; . ./.env; set +a; }
IMAGE="${_IMAGE_CLI:-${IMAGE:-tidelm:local}}"
PEER="${PEER:-}"

# The engine code this image will run (scripts/code_digest.sh), stamped as a label so dual-up.sh
# can refuse to boot an image built from older code than the checkout.
CODE_SHA=$("$ROOT/scripts/code_digest.sh" "$ROOT")
GIT_DESC=$(git -C "$ROOT" describe --always --dirty 2>/dev/null || echo unknown)
info "building $IMAGE from $GIT_DESC, code $CODE_SHA (this pulls the CUDA 13 devel base and the cu130 torch wheel)"
# The Dockerfile writes /app/VERSION (the container's banner, the image label) from this build
# argument; without it every image claimed its default, 0.1.0-wip.
docker build --build-arg RECIPE_VERSION="$(cat "$ROOT/VERSION")" \
    --label "dsv41.code_sha=$CODE_SHA" --label "dsv41.git=$GIT_DESC" -t "$IMAGE" .
LOCAL_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
info "built $IMAGE  $LOCAL_ID"

[[ "${1:-}" == "--local-only" ]] && { info "--local-only: not shipping to a peer"; exit 0; }
[[ -n "$PEER" ]] || { info "PEER unset -- built locally only"; exit 0; }

info "shipping to $PEER (~10 GB over the link; no registry involved)"
docker save "$IMAGE" | ssh -o BatchMode=yes "$PEER" 'docker load'

PEER_ID=$(ssh -o BatchMode=yes "$PEER" "docker image inspect -f '{{.Id}}' '$IMAGE'")
[[ "$PEER_ID" == "$LOCAL_ID" ]] \
    || err "image id mismatch after transfer: local $LOCAL_ID, peer $PEER_ID"
info "both boxes now run the identical image: $LOCAL_ID"
