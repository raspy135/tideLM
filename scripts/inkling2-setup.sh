#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# inkling2-setup.sh -- Inkling-2-Small's weights for a two-Spark profile, once, from the head.
#
# The profile says where everything goes: GLM_MODEL_DIR (the non-expert folder, the same path on
# both nodes), GLM_EXL3_DIR + GLM_EXL3_PACK (each rank's expert pack) and PEER (the worker).
#   1. mirror 0xSero/Inkling-2-Small-EXL3-4bpw at the pinned revision (311.8 GB; tools/hf_mirror.py:
#      standard library only, resumable, tens of MB of memory) and check every file's sha256
#   2. GLM_MODEL_DIR: hard links to the mirror's non-expert shards, the drafter (mtp.safetensors),
#      config and tokenizer, and an index of only those shards (17.4 GB; copies on another disk)
#   3. that folder to the worker
#   4. rank 1's pack here, sent to the worker, then rank 0's here (147.4 GB each;
#      tools/glm_pack_exl3.py --mirror, 64 records of each re-read from the mirror against it)
# A step whose result is there is skipped, so an interrupted run continues by running it again;
# a step the later ones no longer need is skipped too (the mirror, once the packs exist). The
# mirror is not deleted: the script says when it can go (the folder's links keep their files).
#
# Usage: scripts/inkling2-setup.sh [--check] [--mirror DIR] ENV_FILE
#   --check       what is done, what is left and the disk each node has for it; changes nothing
#   --mirror DIR  where the download goes (default: GLM_MODEL_DIR without its -dense suffix, else
#                 Inkling-2-Small-EXL3-4bpw beside it)
# Needs ssh to PEER without a password, rsync on both nodes, and uv (or python3 with numpy) here.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

REPO=0xSero/Inkling-2-Small-EXL3-4bpw
REV=996dc404c1feb3abd5330af8a3c52270ffe0008d
MIRROR_GB=312   # the whole repository
DENSE_GB=18     # 17.4 GB of it, copied to the worker
PACK_GB=148     # 147.4 GB a rank

CHECK=false; MIRROR=""; f=""
while (($#)); do
    case "$1" in
        --check) CHECK=true ;;
        --mirror) MIRROR="${2:?--mirror needs a directory}"; shift ;;
        -*) err "unknown option $1 (expected --check or --mirror DIR)" ;;
        *) [[ -z "$f" ]] || err "one profile, not two"; f="$1" ;;
    esac
    shift
done
[[ -n "$f" ]] || err "usage: scripts/inkling2-setup.sh [--check] [--mirror DIR] ENV_FILE"
[[ -f "$f" ]] || err "no profile $f"
# shellcheck disable=SC1090
set -a; source "$f"; set +a

DENSE=${GLM_MODEL_DIR:-}
[[ -n "$DENSE" ]] || err "$f sets no GLM_MODEL_DIR"
[[ -n "${GLM_EXL3_PACK:-}" ]] || err "$f sets no GLM_EXL3_PACK"
[[ -n "${PEER:-}" ]] || err "$f sets no PEER (the worker, user@host)"
pack_of() { local p=${GLM_EXL3_PACK/#\/packs/${GLM_EXL3_DIR:-/packs}}; echo "${p//\{rank\}/$1}"; }
P0=$(pack_of 0); P1=$(pack_of 1)
[[ "$P0" != "$P1" ]] || err "GLM_EXL3_PACK has no {rank}: $GLM_EXL3_PACK"
if [[ -z "$MIRROR" ]]; then
    if [[ "$DENSE" == *-dense ]]; then MIRROR=${DENSE%-dense}; else MIRROR="$(dirname "$DENSE")/Inkling-2-Small-EXL3-4bpw"; fi
fi
[[ "$MIRROR" != "$DENSE" ]] || err "the mirror and GLM_MODEL_DIR must be two directories"

peer() { ssh -o BatchMode=yes -o LogLevel=QUIET "$PEER" "$@"; }
peer true || err "cannot reach $PEER by ssh without a password"
peer 'command -v rsync >/dev/null' || err "$PEER has no rsync"
command -v rsync >/dev/null || err "no rsync here"
if command -v uv >/dev/null; then PYNP=(uv run --quiet --no-project --with numpy python)
elif python3 -c 'import numpy' 2>/dev/null; then PYNP=(python3)
else PYNP=(); fi
need_numpy() { ((${#PYNP[@]})) || err "the packs need numpy: install uv (https://docs.astral.sh/uv/) or numpy for python3"; }

# free GB under the nearest existing parent of a path, here or on the worker
free_here() { local d=$1; while [[ ! -d "$d" ]]; do d=$(dirname "$d"); done; df -BG --output=avail "$d" | tail -1 | tr -dc 0-9; }
free_peer() { peer "d='$1'; while [ ! -d \"\$d\" ]; do d=\$(dirname \"\$d\"); done; df -BG --output=avail \"\$d\" | tail -1 | tr -dc 0-9"; }

# 1. the mirror: "none" (never started), else how many files are still to fetch or check
mirror_left() {
    python3 - "$MIRROR" "$REV" <<'PY'
import json, os, sys
m, rev = sys.argv[1:]
p = os.path.join(m, ".fetch", "files.json")
if not os.path.exists(p):
    print("none"); sys.exit()
j = json.load(open(p))
if j.get("revision") != rev:
    sys.exit(f"{m} is a mirror of revision {j.get('revision')}, not {rev}")
left = 0
for name, meta in j["files"].items():
    path = os.path.join(m, name)
    if not os.path.exists(path) or os.path.getsize(path) != meta["size"]:
        left += 1
    elif meta.get("sha256"):
        ok = os.path.join(m, ".fetch", name.replace("/", "__") + ".ok")
        left += not (os.path.exists(ok) and open(ok).read().strip() == meta["sha256"])
print(left)
PY
}

# 2. the folder is done once its index exists (written last) and every file it names is there
DENSE_FILES=(config.json chat_template.jinja tokenizer.json tokenizer_config.json special_tokens_map.json mtp.safetensors)
dense_done() {
    python3 - "$DENSE" "${DENSE_FILES[@]}" <<'PY'
import json, os, sys
d, extra = sys.argv[1], sys.argv[2:]
ix = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(ix):
    sys.exit(1)
want = set(json.load(open(ix))["weight_map"].values()) | set(extra)
sys.exit(0 if all(os.path.isfile(os.path.join(d, n)) for n in want) else 1)
PY
}
make_dense() {
    python3 - "$MIRROR" "$DENSE" "${DENSE_FILES[@]}" <<'PY'
import json, os, shutil, sys
src, dst, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
ix = json.load(open(os.path.join(src, "model.safetensors.index.json")))
ix["weight_map"] = {k: v for k, v in ix["weight_map"].items() if v.startswith("model-nonexpert-")}
os.makedirs(dst, exist_ok=True)
copied = 0
for n in sorted(set(ix["weight_map"].values())) + extra:
    a, b = os.path.join(src, n), os.path.join(dst, n)
    if os.path.exists(b) and os.path.getsize(b) == os.path.getsize(a):
        continue
    if os.path.exists(b):
        os.remove(b)
    try:
        os.link(a, b)
    except OSError:                       # another filesystem: a copy
        shutil.copy2(a, b)
        copied += 1
tmp = os.path.join(dst, ".index.json.tmp")
with open(tmp, "w") as f:
    json.dump(ix, f, indent=2)
os.replace(tmp, os.path.join(dst, "model.safetensors.index.json"))
print(f"{len(set(ix['weight_map'].values()))} non-expert shards and {len(extra)} files in {dst}"
      + (f" ({copied} copied: not the mirror's filesystem)" if copied else " (hard links)"))
PY
}

# 3. files of the folder the worker's copy lacks or has otherwise (rsync's own comparison)
peer_dense_left() {
    [[ -d "$DENSE" ]] || { echo all; return; }
    rsync -an --out-format='%n' "$DENSE/" "$PEER:$DENSE/" 2>/dev/null | grep -vc '^\./$' || true
}

# 4. a pack's state: "none", "partial" or "complete" (every record written; tools/exl3_ref.py)
pack_state() {
    [[ -f "$1" ]] || { echo none; return; }
    need_numpy
    "${PYNP[@]}" - "$1" <<'PY'
import sys
sys.path.insert(0, "tools")
import exl3_ref as R
try:
    r = R.PackReader(sys.argv[1])
    print("complete" if set(r.records) <= R.written_records(sys.argv[1]) else "partial")
except Exception:
    print("partial")
PY
}
peer_has_pack() { peer "test -f '$1'"; }
build_pack() {    # rank, path
    need_numpy
    mkdir -p "$(dirname "$2")"
    local have; have=$(free_here "$(dirname "$2")")
    [[ -f "$2" ]] || (( have >= PACK_GB )) || err "rank $1's pack needs ${PACK_GB} GB under $(dirname "$2"), ${have} GB free"
    info "rank $1: packing its experts into $2 (resumes a partial pack)"
    "${PYNP[@]}" tools/glm_pack_exl3.py --mirror "$MIRROR" --world 2 --rank "$1" --resume --verify 64 --out "$2"
}

# ---- what is there ----
m_left=none; [[ -d "$MIRROR" ]] && m_left=$(mirror_left)
dense_done && d_ok=yes || d_ok=no
pd_left=$(peer_dense_left)
s0=$(pack_state "$P0")
s1_here=$(pack_state "$P1")
peer_has_pack "$P1" && s1_peer=yes || s1_peer=no
r1_done=no; [[ $s1_here == none && $s1_peer == yes ]] && r1_done=yes
# the mirror is needed while the folder or a pack is still to make
mirror_needed=no; [[ $d_ok == no || $s0 != complete || $r1_done == no ]] && mirror_needed=yes

case $m_left in none) m_txt="not started" ;; 0) m_txt="complete, every sha256 checked" ;; *) m_txt="$m_left files to fetch or check" ;; esac
[[ $mirror_needed == no && $m_left == none ]] && m_txt="not here (not needed: the folder and both packs exist)"
case $pd_left in 0) pd_txt="in sync" ;; all) pd_txt="nothing to send yet" ;; *) pd_txt="$pd_left files to send" ;; esac
r1_txt="on the worker"; [[ $r1_done == yes ]] || r1_txt="here: $s1_here, on the worker: $s1_peer"
info "profile $f: mirror $MIRROR, folder $DENSE, packs $P0 / $P1, worker $PEER"
echo "    1. mirror:            $m_txt"
echo "    2. non-expert folder: $([[ $d_ok == yes ]] && echo complete || echo "to make")"
echo "    3. folder on worker:  $pd_txt"
echo "    4. rank 1 pack:       $r1_txt"
echo "       rank 0 pack:       $s0"
echo "    disk free: here $(free_here "$MIRROR") GB (mirror), $(free_here "$P0") GB (packs);" \
     "worker $(free_peer "$DENSE") GB (folder), $(free_peer "$P1") GB (packs)"
if [[ $CHECK == true ]]; then
    info "--check: nothing changed"
    exit 0
fi

# ---- the steps ----
if [[ $mirror_needed == yes && $m_left != 0 ]]; then
    info "1. mirroring $REPO@${REV:0:8} into $MIRROR (${MIRROR_GB} GB in all; rerun to resume)"
    mkdir -p "$MIRROR"
    python3 tools/hf_mirror.py --repo "$REPO" --revision "$REV" --out "$MIRROR"
    info "1. checking every file's sha256"
    python3 tools/hf_mirror.py --repo "$REPO" --revision "$REV" --out "$MIRROR" --verify
fi
if [[ $d_ok == no ]]; then
    info "2. the non-expert folder"
    make_dense
    dense_done || err "$DENSE is incomplete after making it"
fi
if [[ $(peer_dense_left) != 0 ]]; then
    have=$(free_peer "$DENSE")
    peer "test -d '$DENSE'" || (( have >= DENSE_GB )) || err "the folder needs ${DENSE_GB} GB on $PEER, ${have} GB free"
    info "3. the folder to $PEER:$DENSE"
    peer "mkdir -p '$DENSE'"
    rsync -a --info=progress2 "$DENSE/" "$PEER:$DENSE/"
fi
if [[ $r1_done == no ]]; then
    [[ $(pack_state "$P1") == complete ]] || build_pack 1 "$P1"
    [[ $(pack_state "$P1") == complete ]] || err "rank 1's pack is incomplete after packing"
    have=$(free_peer "$P1")
    peer_has_pack "$P1" || (( have >= PACK_GB )) || err "rank 1's pack needs ${PACK_GB} GB on $PEER, ${have} GB free"
    info "4. rank 1's pack to $PEER:$P1"
    peer "mkdir -p '$(dirname "$P1")'"
    # whole file, under a temporary name renamed when complete; -t keeps its time so a rerun skips it
    rsync -Wt --info=progress2 "$P1" "$PEER:$P1"
    here_size=$(stat -c %s "$P1"); there_size=$(peer "stat -c %s '$P1'")
    [[ $here_size == "$there_size" ]] || err "rank 1's pack is $there_size bytes on $PEER, $here_size here"
    rm "$P1"
fi
if [[ $(pack_state "$P0") != complete ]]; then
    build_pack 0 "$P0"
    [[ $(pack_state "$P0") == complete ]] || err "rank 0's pack is incomplete after packing"
fi

info "done: the folder on both nodes, rank 0's pack here, rank 1's on $PEER"
[[ -d "$MIRROR" ]] && echo "    The mirror is no longer needed (the folder's links keep their files): rm -r '$MIRROR'"
echo "    Next: scripts/tide.sh build $f && scripts/tide.sh start $f"
