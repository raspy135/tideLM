#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# exl3-packs.sh -- build the per-node EXL3 expert packs for EXPERT_FORMAT=exl3, once.
#
# The engine reads routed experts from one pack per TP rank: each expert's rank slice stored
# contiguously and 4096-aligned, so a load is one O_DIRECT read (tools/pack_exl3_experts.py).
# This builds both ranks' packs here, from the EXL3 checkpoint, inside the serving image (no host
# Python), then copies rank 1's to the peer. Only this node needs the EXL3 checkpoint; after the
# packs exist it can be deleted.
#
#   checkpoint   <models>/DeepSeek-V4.1-Flash-EXL3-2.9bpw   ~198 GB, this node only
#   packs        <models>/exl3-packs/exl3-experts-r{0,1}of2.bin   ~98.3 GB each
#                <models>/exl3-packs/exl3-draft-r{0,1}of2.safetensors ~195 MB each (EXL3 draft blocks)
#   (<models> = MODELS_DIR, else MODEL_DIR's parent: where the engine looks for them)
#
# Usage: scripts/exl3-packs.sh [--check] [--force] [--draft-only] [checkpoint-dir]
#   --check       verify the image, checkpoint and disk space on both nodes, build nothing
#   --force       rebuild packs that already exist
#   --draft-only  build only the two draft packs (the FP4 profile's DSV41_EXL3_DRAFT_DENSE=1). The
#                 checkpoint may then be partial: config.json, model.safetensors.index.json and the
#                 shards holding the mtp.* tensors (2 of 39, ~14 GB; the script names any missing).
#                 A pack built from a partial checkpoint records a different source digest, so do
#                 not pair it with an exl3-dense pack built from the full one (the boot guard refuses).
# Each rank takes ~8 minutes here; the copy is ~98 GB over the PEER link.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

CHECK=false
FORCE=false
DRAFT_ONLY=false
SRC_ARG=""
for a in "$@"; do
    case "$a" in
        --check) CHECK=true ;;
        --force) FORCE=true ;;
        --draft-only) DRAFT_ONLY=true ;;
        -*) err "unknown option $a (expected --check, --force or --draft-only)" ;;
        *) SRC_ARG="$a" ;;
    esac
done

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
IMAGE="${IMAGE:-deepseek-v41-flash-spark:local}"
PEER="${PEER:-}"
HOST_MODELS="${MODELS_DIR:-$(dirname "${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}")}"
SRC="${SRC_ARG:-$HOST_MODELS/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
OUT="$HOST_MODELS/exl3-packs"
PACK_GB=99   # 98.24 GB a rank, plus slack

[[ -n "$PEER" ]] || err "PEER is required (set it in .env)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || err "image $IMAGE missing: run scripts/dual-build.sh <your profile> first"
[[ -f "$SRC/config.json" ]] || err "no EXL3 checkpoint at $SRC (download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw there, or pass its path)"
case "$SRC" in "$HOST_MODELS"/*) ;; *) err "the checkpoint must sit under $HOST_MODELS (it is mounted into the image as /models)";; esac
grep -q '"quant_method": *"exl3"' "$SRC/config.json" || err "$SRC/config.json is not an EXL3 checkpoint"
if [[ "$DRAFT_ONLY" == true ]]; then
    # The draft packer reads only the mtp.* tensors; name the shards they live in.
    [[ -f "$SRC/model.safetensors.index.json" ]] || err "$SRC has no model.safetensors.index.json"
    missing=$(python3 - "$SRC" <<'PY'
import json, os, sys
src = sys.argv[1]
shards = sorted({f for k, f in json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"].items()
                 if k.startswith("mtp.")})
print(" ".join(f for f in shards if not os.path.exists(os.path.join(src, f))))
PY
)
    [[ -z "$missing" ]] || err "the draft packs need these shards of the EXL3 checkpoint in $SRC: $missing
    (hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw $missing --local-dir $SRC)"
fi

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
need=0
if [[ "$DRAFT_ONLY" == false ]]; then
    for r in 0 1; do
        [[ -f "$OUT/exl3-experts-r${r}of2.bin" && "$FORCE" == false ]] || need=$((need + PACK_GB))
    done
fi
mkdir -p "$OUT"
have=$(free_gb "$OUT")
(( have >= need )) || err "this node has ${have} GB free under $OUT, the packs need ${need} GB"
ssh -o BatchMode=yes "$PEER" "mkdir -p '$OUT'" || err "cannot reach $PEER"
peer_has=$(ssh -o BatchMode=yes "$PEER" "test -f '$OUT/exl3-experts-r1of2.bin' && stat -c %s '$OUT/exl3-experts-r1of2.bin' || echo 0")
peer_free=$(ssh -o BatchMode=yes "$PEER" "df -BG --output=avail '$OUT' | tail -1 | tr -dc 0-9")
if [[ "$DRAFT_ONLY" == false ]]; then
    (( peer_has > 0 || peer_free >= PACK_GB )) || err "$PEER has ${peer_free} GB free under $OUT, rank 1's pack needs ${PACK_GB} GB"
fi
info "image $IMAGE, checkpoint $SRC, $([[ "$DRAFT_ONLY" == true ]] && echo "draft packs only" || echo packs) -> $OUT (this node ${have} GB free, $PEER ${peer_free} GB free)"
[[ "$CHECK" == true ]] && { info "--check: ready; nothing built"; exit 0; }

[[ "$DRAFT_ONLY" == true ]] || for r in 0 1; do
    pack="$OUT/exl3-experts-r${r}of2.bin"
    if [[ -f "$pack" && "$FORCE" == false ]]; then
        info "rank $r: $pack exists ($(stat -c %s "$pack") bytes), kept (--force rebuilds)"
        continue
    fi
    info "rank $r: packing (~8 min)"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOST_MODELS:/models" \
        --entrypoint python "$IMAGE" /app/tools/pack_exl3_experts.py \
        --source "/models/${SRC#"$HOST_MODELS"/}" --out "/models/exl3-packs/exl3-experts-r${r}of2.bin" \
        --rank "$r" --world 2
done

if [[ "$DRAFT_ONLY" == false ]]; then
    local_size=$(stat -c %s "$OUT/exl3-experts-r1of2.bin")
    if [[ "$peer_has" != "$local_size" || "$FORCE" == true ]]; then
        info "copying rank 1's pack to $PEER:$OUT"
        rsync -W --info=progress2 "$OUT/exl3-experts-r1of2.bin" "$PEER:$OUT/"
    fi
    peer_size=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$OUT/exl3-experts-r1of2.bin'")
    [[ "$peer_size" == "$local_size" ]] || err "rank 1's pack on $PEER is $peer_size bytes, here $local_size"
fi

# The DSpark draft blocks' own 4-bit matrices (~195 MB a rank) for DSV41_EXL3_DRAFT_DENSE. Tiny
# next to the expert packs, so it is not budgeted above and it is built even when they exist.
DRAFT_GB=1
draft_need=0
for r in 0 1; do
    [[ -f "$OUT/exl3-draft-r${r}of2.safetensors" && "$FORCE" == false ]] || draft_need=$((draft_need + DRAFT_GB))
done
if (( draft_need > 0 )); then
    (( have >= draft_need )) || err "this node has ${have} GB free under $OUT, the draft packs need ${draft_need} GB"
    (( peer_free >= DRAFT_GB )) || err "$PEER has ${peer_free} GB free under $OUT, rank 1's draft pack needs ${DRAFT_GB} GB"
    for r in 0 1; do
        pack="$OUT/exl3-draft-r${r}of2.safetensors"
        if [[ -f "$pack" && "$FORCE" == false ]]; then
            info "draft rank $r: $pack exists, kept (--force rebuilds)"
            continue
        fi
        info "draft rank $r: packing (~1 min)"
        docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOST_MODELS:/models" \
            --entrypoint python "$IMAGE" /app/tools/pack_exl3_draft.py \
            --source "/models/${SRC#"$HOST_MODELS"/}" --out "/models/exl3-packs/exl3-draft-r${r}of2.safetensors" \
            --rank "$r" --world 2
    done
    rsync -W --info=progress2 "$OUT/exl3-draft-r1of2.safetensors" "$PEER:$OUT/"
fi
if [[ "$DRAFT_ONLY" == true ]]; then
    info "done: draft packs, rank 0 here, rank 1 on $PEER. The partial EXL3 checkpoint can be deleted."
else
    info "done: rank 0 here, rank 1 on $PEER. Rank 1's local copy ($OUT/exl3-experts-r1of2.bin) is no longer
    needed on this node; delete it to reclaim ~98 GB. The boot guard checks both packs share one source."
fi
