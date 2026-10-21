#!/usr/bin/env python3
"""Build, select and ship dreaming overlay generations (engine/dream_overlay.py serves them).

    python3 dream/overlay.py make NAME --experts 20:all [--zero-w2] [--parent GEN] [--note TEXT]
    python3 dream/overlay.py use NAME|none        # point serve/current at a generation
    python3 dream/overlay.py push                 # mirror serve/ to the peer (same path)
    python3 dream/overlay.py list

`make` copies experts out of the checkpoint byte for byte (an identity overlay); `--zero-w2`
zeroes their down projections instead, so a test can prove the overlay is the thing being served.
Trained experts are written by the trainer through the same `write_expert`.

Layout (DREAM_DIR, default /home/ryan/models/dream -- inside the models mount, so the container
sees it as /models/dream with no extra volume):

    serve/gen-*/manifest.json, serve/gen-*/L{L}.E{EEEE}.safetensors   MXFP4, what the engine reads
    serve/current -> gen-*                                            relative link, so it resolves
                                                                      in the container too
    train/                                                            FP32 masters + optimizer state

Serve with DSV41_DREAM_OVERLAY=/models/dream/serve/current in the profile.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import dream_overlay as D  # noqa: E402

DREAM_DIR = Path(os.environ.get("DREAM_DIR", "/home/ryan/models/dream"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/ryan/models/DeepSeek-V4.1-Flash-1017"))
PEER = os.environ.get("PEER", "ryan@10.0.0.2")
N_LAYERS, N_EXPERTS = 40, 384


def parse_experts(spec: str) -> list[tuple[int, int]]:
    """'20:all,21:3,21:17' -> [(20, 0..383), (21, 3), (21, 17)]"""
    out = []
    for part in spec.split(","):
        layer, e = part.split(":")
        es = range(N_EXPERTS) if e == "all" else [int(e)]
        out += [(int(layer), x) for x in es]
    return out


def cmd_make(a):
    gen = DREAM_DIR / "serve" / a.name
    if gen.exists():
        sys.exit(f"{gen} exists; generations are never modified, pick a new name")
    gen.mkdir(parents=True)
    wm = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    experts = []
    for layer, expert in parse_experts(a.experts):
        t = D.read_expert(str(MODEL_DIR), wm, layer, expert)
        if a.zero_w2:
            dtype, shape, raw = t["w2.weight"]
            t["w2.weight"] = (dtype, shape, bytes(len(raw)))
        f = D.file_name(layer, expert)
        experts.append({"layer": layer, "expert": expert, "file": f,
                        "sha256": D.write_expert(str(gen / f), layer, expert, t)})
    man = {"version": D.VERSION,
           "base_index_sha256": D.sha256_file(str(MODEL_DIR / "model.safetensors.index.json")),
           "parent": a.parent, "note": a.note, "experts": experts}
    tmp = gen / (D.MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    tmp.rename(gen / D.MANIFEST)
    print(f"{gen}: {len(experts)} experts")


def cmd_use(a):
    cur = DREAM_DIR / "serve" / "current"
    if a.name == "none":
        cur.unlink(missing_ok=True)
        print("serve/current removed; unset DSV41_DREAM_OVERLAY to serve the stock checkpoint")
        return
    if not (DREAM_DIR / "serve" / a.name / D.MANIFEST).exists():
        sys.exit(f"no generation {a.name}")
    gate = json.loads((DREAM_DIR / "serve" / a.name / D.MANIFEST).read_text()).get("drift_gate")
    if gate and not gate["pass"] and not a.force:
        sys.exit(f"{a.name} failed its drift gate (general-eval KL {gate['drift']:.4f} > {gate['max_drift']}); "
                 f"--force to serve it anyway")
    tmp = DREAM_DIR / "serve" / ".current.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(a.name)
    tmp.replace(cur)
    print(f"serve/current -> {a.name}")


def cmd_push(a):
    # No --delete: a generation that exists only on the peer is left alone, never removed from here.
    src = DREAM_DIR / "serve"
    subprocess.run(["ssh", PEER, f"mkdir -p '{src}'"], check=True)
    subprocess.run(["rsync", "-a", f"{src}/", f"{PEER}:{src}/"], check=True)
    print(f"serve/ mirrored to {PEER}")


def cmd_list(a):
    serve = DREAM_DIR / "serve"
    cur = os.readlink(serve / "current") if (serve / "current").is_symlink() else None
    for g in sorted(p for p in serve.glob("*") if not p.is_symlink() and (p / D.MANIFEST).exists()):
        man = json.loads((g / D.MANIFEST).read_text())
        mark = "*" if g.name == cur else " "
        print(f"{mark} {g.name}: {len(man['experts'])} experts, parent {man.get('parent')}, "
              f"{man.get('note') or ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("name")
    m.add_argument("--experts", required=True)
    m.add_argument("--zero-w2", action="store_true")
    m.add_argument("--parent")
    m.add_argument("--note", default="")
    u = sub.add_parser("use")
    u.add_argument("name")
    u.add_argument("--force", action="store_true", help="serve a generation that failed its drift gate")
    sub.add_parser("push")
    sub.add_parser("list")
    a = p.parse_args()
    {"make": cmd_make, "use": cmd_use, "push": cmd_push, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    main()
