"""Package a running DS pair's learned expert distribution as a cold-start seed, in the format
engine/expert_seed.py reads (profiles/learned-experts-v1.npz is the first one).

Run in the rank-0 container while the server is idle -- the demand DB is written after every
request, so it then matches the live map:

    docker exec deepseek-v41-ep2-rank0 python3 /app/tools/export_expert_seed.py \\
        --db /app/results/prune_demand_req_score_1017.npz --model DeepSeek-V4.1-Flash-1017 \\
        --out /app/results/seed-export/learned-experts-1017-v1.npz

then copy the .npz and .json into profiles/ and point DSV41_EXPERT_SEED at it. The resident map
comes from /v1/expert-map (state 1; transient and loading slots are not part of it), the request
counts and router-score mass from the DB. Only aggregates are written: expert IDs, counts and
mass -- no prompts, responses, KV or weights.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.expert_seed import VERSION, load_seed, seed_selection  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="the demand DB (DSV41_PRUNE_DB)")
    ap.add_argument("--model", required=True, help="model name for the JSON, e.g. DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--out", required=True, help="output .npz; the .json goes next to it")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/expert-map")
    a = ap.parse_args()

    snap = json.load(urllib.request.urlopen(a.url, timeout=30))
    if snap.get("busy") or snap.get("fault") or snap["counts"].get("loading"):
        sys.exit("server busy, faulted or loading experts: wait until it is idle")
    if not snap.get("dynamic"):
        sys.exit("not a dynamic-residency server: there is no learned map to export")
    keep = np.array([[c == "1" for c in row] for row in snap["states"]], dtype=np.uint8)
    with np.load(a.db, allow_pickle=False) as d:
        counts, mass = d["counts"].astype(np.float64), d["mass"].astype(np.float64)
    if counts.shape != keep.shape or mass.shape != keep.shape:
        sys.exit(f"DB {counts.shape} and map {keep.shape} differ")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, version=np.array([VERSION]), counts=counts, mass=mass, keep=keep)
    # Read it back the way a fresh install will: no local DB, then the exact map at its budget.
    layers, experts = keep.shape
    seed = load_seed(out.parent / "no-such-db.npz", enabled=True, request_unit=True, metric="score",
                     layers=layers, experts=experts, path=out)
    if seed_selection(seed, int(keep.sum()), 1) is None:
        sys.exit("the written seed does not restore its own map")
    meta = dict(version=VERSION, captured_at_unix=time.time(), expert_generation=snap.get("generation"),
                model=a.model, layers=layers, experts_per_layer=experts, residents=int(keep.sum()),
                resident_per_layer=keep.sum(1).tolist(),
                source="Live dynamic TP2 working set and aggregate request-normalized routing demand",
                contents="Expert IDs, aggregate counts and router-score mass only; no prompts, "
                         "responses, KV, or model weights.",
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    per = meta["resident_per_layer"]
    print(f"{out}: {meta['residents']} residents ({min(per)}-{max(per)} a layer), "
          f"generation {meta['expert_generation']}, sha256 {meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
