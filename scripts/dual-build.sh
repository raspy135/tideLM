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
# Usage: scripts/dual-build.sh [ENV_FILE] [--local-only]
#
#   scripts/dual-build.sh .env.ds41-mxfp4   build the image that profile runs (IMAGE, or GLM_IMAGE
#                                           in a GLM-5.3 profile) and ship it to its PEER
#   scripts/dual-build.sh                   the same from .env, else tidelm:local and $PEER
#
# IMAGE=... or PEER=... on the command line beat the file. Without a PEER it stops before
# building: a head-only build leaves the worker on its old image, and dual-up.sh then refuses
# the pair as two different images. --local-only builds on this node only, on purpose.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

LOCAL_ONLY=false ENV_FILE=""
for a in "$@"; do
    case "$a" in
        --local-only) LOCAL_ONLY=true ;;
        -*) err "unknown option $a (usage: scripts/dual-build.sh [ENV_FILE] [--local-only])" ;;
        *)  [[ -z "$ENV_FILE" ]] || err "one profile at a time ($ENV_FILE, $a)"
            [[ -f "$a" ]] || { [[ -f "$ROOT/$a" ]] && a="$ROOT/$a"; } || err "no env file $a"
            ENV_FILE="$(cd "$(dirname "$a")" && pwd)/$(basename "$a")" ;;
    esac
done
cd "$ROOT"

# shellcheck disable=SC1090,SC1091
_IMAGE_CLI="${IMAGE:-}" _PEER_CLI="${PEER:-}"   # the caller's IMAGE / PEER beat the file's
if [[ -n "$ENV_FILE" ]]; then
    set -a; . "$ENV_FILE"; set +a
    IMAGE="${IMAGE:-${GLM_IMAGE:-}}"
elif [[ -f .env ]]; then
    set -a; . ./.env; set +a
fi
IMAGE="${_IMAGE_CLI:-${IMAGE:-tidelm:local}}"
PEER="${_PEER_CLI:-${PEER:-}}"
$LOCAL_ONLY || [[ -n "$PEER" ]] || err "no PEER, so the worker would keep its old image. Pass the profile you serve \
(scripts/dual-build.sh .env.ds41-mxfp4), set PEER=user@host, or --local-only to build on this node only."

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

$LOCAL_ONLY && { info "--local-only: not shipping to a peer"; exit 0; }

info "shipping to $PEER (~10 GB over the link; no registry involved)"
docker save "$IMAGE" | ssh -o BatchMode=yes "$PEER" 'docker load'

PEER_ID=$(ssh -o BatchMode=yes "$PEER" "docker image inspect -f '{{.Id}}' '$IMAGE'")
[[ "$PEER_ID" == "$LOCAL_ID" ]] \
    || err "image id mismatch after transfer: local $LOCAL_ID, peer $PEER_ID"
info "both boxes now run the identical image: $LOCAL_ID"
