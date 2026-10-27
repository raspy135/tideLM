#!/usr/bin/env python3
"""glm_fetch_dense.py --out DIR [--repo bertholomus/GLM-5.3-EXL3-3.0bpw]: an EXL3 GLM-5.3
checkpoint without its routed experts, by HTTP range requests.

The GLM profile can serve one checkpoint's routed experts (packed per node by
tools/glm_pack_exl3.py) with another's non-expert weights (attention, the indexer, shared
experts, dense layers, routers, norms, embeddings, head, the MTP layer's), as long as both
quantize the same GLM-5.3. Mia's AI Lab's 2.75bpw checkpoint keeps its non-expert weights in BF16
(~19 GB a rank); bertholomus's 3.0bpw has them in EXL3 at ~5-6 bits (7.3 GB a rank). This tool
fetches only those 15 GB of the 293 GB checkpoint: each shard's safetensors header, then the
byte ranges of the tensors that are not `.mlp.experts.`, written as shards of the same names
with only those tensors, an index for them, and the repo's small files (config, tokenizer,
chat template). GLM_MODEL_DIR points at the result.

Stdlib only. Re-running resumes: a shard whose header already matches is skipped. HF_TOKEN, if
set, is sent (the repo is public; a token only raises rate limits).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import struct
import sys
import urllib.request

EXPERT = ".mlp.experts."
SMALL = 64 << 20  # repo files fetched whole: everything but the weight shards, under this size


def opener(url: str, rng: tuple[int, int] | None = None):
    h = {"User-Agent": "tidelm-glm-fetch-dense"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300)


def get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    for attempt in range(5):
        try:
            with opener(url, rng) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0]:
                raise OSError(f"short read {len(data)} of {rng[1] - rng[0]}")
            return data
        except OSError as e:
            if attempt == 4:
                raise
            print(f"  retry {url.rsplit('/', 1)[-1]} {rng}: {e}", file=sys.stderr)
    raise AssertionError


def header(url: str) -> tuple[int, dict]:
    n = struct.unpack("<Q", get(url, (0, 8)))[0]
    return 8 + n, json.loads(get(url, (8, 8 + n)))


def fetch_shard(base: str, shard: str, out: str, workers: int) -> tuple[int, int]:
    """The shard's non-expert tensors into out/shard; (tensors, bytes)."""
    url = base + shard
    data0, h = header(url)
    keep = sorted(((k, v) for k, v in h.items() if k != "__metadata__" and EXPERT not in k),
                  key=lambda kv: kv[1]["data_offsets"][0])
    if not keep:
        return 0, 0
    new, off = {}, 0
    if "__metadata__" in h:
        new["__metadata__"] = h["__metadata__"]
    for k, v in keep:
        a, b = v["data_offsets"]
        new[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    blob = json.dumps(new, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path = os.path.join(out, shard)
    want_size = 8 + len(blob) + off
    if os.path.exists(path) and os.path.getsize(path) == want_size:
        with open(path, "rb") as f:
            if f.read(8 + len(blob)) == struct.pack("<Q", len(blob)) + blob:
                return len(keep), off  # already fetched
    # adjacent tensors in one request (a layer's non-expert tensors sit together)
    runs: list[list] = []
    for k, v in keep:
        a, b = v["data_offsets"]
        if runs and runs[-1][1] == a and runs[-1][1] - runs[-1][0] < (256 << 20):
            runs[-1][1] = b
        else:
            runs.append([a, b])
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        f.truncate(want_size)
    # source offsets map to output offsets in order: tensors keep their order and sizes
    starts = {}
    pos = 8 + len(blob)
    for k, v in keep:
        starts[v["data_offsets"][0]] = pos
        pos += v["data_offsets"][1] - v["data_offsets"][0]

    def one(run):
        a, b = run
        data = get(url, (data0 + a, data0 + b))
        with open(tmp, "r+b") as f:
            f.seek(starts[a])
            f.write(data)
        return b - a

    with cf.ThreadPoolExecutor(workers) as ex:
        got = sum(ex.map(one, runs))
    assert got == off, (shard, got, off)
    os.replace(tmp, path)
    return len(keep), off


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="bertholomus/GLM-5.3-EXL3-3.0bpw")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", default="", help="comma-separated shard names (a test fetch)")
    a = ap.parse_args(argv)
    base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/"
    os.makedirs(a.out, exist_ok=True)
    info = json.loads(get(f"https://huggingface.co/api/models/{a.repo}/revision/{a.revision}?blobs=true"))
    for s in info.get("siblings", []):
        name, size = s["rfilename"], s.get("size") or 0
        if name.endswith(".safetensors") or name == "model.safetensors.index.json" or size > SMALL or "/" in name:
            continue
        p = os.path.join(a.out, name)
        if not (os.path.exists(p) and os.path.getsize(p) == size):
            with open(p, "wb") as f:
                f.write(get(base + name))
    index = json.loads(get(base + "model.safetensors.index.json"))
    wmap = {k: v for k, v in index["weight_map"].items() if EXPERT not in k}
    shards = sorted(set(wmap.values()))
    if a.shards:
        shards = [s for s in shards if s in a.shards.split(",")]
    total_t = total_b = 0
    for i, shard in enumerate(shards):
        t, b = fetch_shard(base, shard, a.out, a.workers)
        total_t, total_b = total_t + t, total_b + b
        print(f"[dense] {i + 1}/{len(shards)} {shard}: {t} tensors, {b / 1e9:.2f} GB", flush=True)
    if not a.shards:
        with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_b, "source": f"{a.repo}@{a.revision}, routed experts left out"},
                       "weight_map": wmap}, f, indent=1)
    print(f"[dense] {total_t} tensors, {total_b / 1e9:.1f} GB in {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
