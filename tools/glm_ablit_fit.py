#!/usr/bin/env python3
"""glm_ablit_fit.py --out FILE: GLM_ABLIT's directions file, recovered from an abliterated GLM-5.3
BF16 checkpoint and the BF16 base by HTTP range requests.

An abliteration that edits every residual-stream writer of a layer as W' = W - alpha r (r^T W),
with one unit direction r a layer (Blackfrost-Research/GLM-5.3-DERISKED-BF16: layers 2-49, the
attention's o_proj and every down_proj, alpha 3), is served by the Rust engine as a projection
of each edited layer's summed output (rust/crates/tidelm-glm/src/ablit.rs). One writer a layer is
enough to recover (r, alpha): the dense layers' down_proj, the MoE layers' shared expert's (25 MB
in BF16). For each, both checkpoints' tensor is fetched (each shard's header, then the tensor's
byte range), D = W' - W; an unedited layer's D is exactly zero; an edited one's top left singular
vector is r (s1 / s2 printed: a rank-1 edit is hundreds), and alpha = 1 - <r^T W', r^T W> /
|r^T W|^2. --check also fits o_proj and routed experts of a few layers and prints their
direction's cosine to the layer's r.

Writes safetensors with glm_ablit.layers I32 [n], glm_ablit.alpha F32 [n], glm_ablit.r F32 [n, H].
--fits caches each tensor's fit (JSON lines) so a rerun resumes. Needs numpy; HF_TOKEN, if set, is
sent (both repos are public). Memory: two tensors and their difference in fp32 at a time (~1 GB
for a dense layer's down_proj).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np


def get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    h = {"User-Agent": "tidelm-glm-ablit-fit"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0]:
                raise OSError(f"short read {len(data)} of {rng[1] - rng[0]}")
            return data
        except OSError as e:
            if attempt == 4:
                raise
            print(f"  retry {url.rsplit('/', 1)[-1]} {rng}: {e}", file=sys.stderr)
    raise AssertionError


class Repo:
    def __init__(self, repo: str):
        self.base = f"https://huggingface.co/{repo}/resolve/main/"
        self.map = json.loads(get(self.base + "model.safetensors.index.json"))["weight_map"]
        self.hdr: dict[str, tuple[int, dict]] = {}

    def tensor(self, name: str) -> np.ndarray:
        shard = self.map[name]
        if shard not in self.hdr:
            n = struct.unpack("<Q", get(self.base + shard, (0, 8)))[0]
            self.hdr[shard] = (8 + n, json.loads(get(self.base + shard, (8, 8 + n))))
        off, h = self.hdr[shard]
        m = h[name]
        if m["dtype"] != "BF16":
            raise SystemExit(f"{name}: {m['dtype']}, not BF16")
        a, b = m["data_offsets"]
        raw = np.frombuffer(get(self.base + shard, (off + a, off + b)), dtype=np.uint16).reshape(m["shape"])
        return (raw.astype(np.uint32) << 16).view(np.float32)


def top(D: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """D's top left singular vector and its leading singular values (subspace iteration on D D^T)."""
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((D.shape[0], 4)).astype(np.float32))[0]
    for _ in range(iters):
        Q = np.linalg.qr(D @ (D.T @ Q))[0]
    u, s, _ = np.linalg.svd(Q.T @ D, full_matrices=False)
    r = Q @ u[:, 0]
    return r / np.linalg.norm(r), s


def fit(base: Repo, edit: Repo, name: str) -> dict:
    Wb, We = base.tensor(name), edit.tensor(name)
    D = We - Wb
    if not np.any(D):
        return {"name": name, "edited": False}
    r, s = top(D)
    rb, re = r @ Wb, r @ We
    alpha = 1.0 - float(re @ rb / (rb @ rb))
    resid = D + alpha * np.outer(r, rb)
    return {"name": name, "edited": True, "s1_s2": float(s[0] / s[1]), "alpha": alpha,
            "resid": float(np.linalg.norm(resid) / np.linalg.norm(D)), "dW": float(np.linalg.norm(D) / np.linalg.norm(Wb)),
            "r": r.tolist()}


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    hdr, off, blobs = {}, 0, []
    for k, (dt, shape, b) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="zai-org/GLM-5.3-BF16")
    ap.add_argument("--edit", default="Blackfrost-Research/GLM-5.3-DERISKED-BF16")
    ap.add_argument("--fits", default=None, help="JSON-lines cache of the fits (resumes)")
    ap.add_argument("--layers", type=int, default=78, help="decoder layers (GLM-5.3: 78; the MTP layer is not fitted)")
    ap.add_argument("--dense", type=int, default=3, help="leading dense layers (GLM-5.3: 3)")
    ap.add_argument("--check", action="store_true", help="also fit o_proj and routed experts of a few layers")
    ap.add_argument("--alpha", type=float, default=None,
                    help="write this alpha for every edited layer instead of each one's fit (Blackfrost's: 3)")
    a = ap.parse_args()
    base, edit = Repo(a.base), Repo(a.edit)
    fits: dict[str, dict] = {}
    if a.fits and os.path.exists(a.fits):
        for line in open(a.fits):
            d = json.loads(line)
            fits[d["name"]] = d
    cache = open(a.fits, "a") if a.fits else None

    def get_fit(name: str) -> dict:
        if name not in fits:
            fits[name] = fit(base, edit, name)
            if cache:
                cache.write(json.dumps(fits[name]) + "\n")
                cache.flush()
        d = fits[name]
        print(name, "unedited" if not d["edited"] else
              f"s1/s2 {d['s1_s2']:.0f} alpha {d['alpha']:.5f} resid {d['resid']:.4f} |dW|/|W| {d['dW']:.4f}", flush=True)
        return d

    layer_fit = {}
    for l in range(a.layers):
        w = "mlp.down_proj" if l < a.dense else "mlp.shared_experts.down_proj"
        layer_fit[l] = get_fit(f"model.layers.{l}.{w}.weight")
    edited = [l for l in range(a.layers) if layer_fit[l]["edited"]]
    if not edited:
        raise SystemExit("no layer differs: not an abliterated checkpoint of this base")
    if a.check:
        r = {l: np.array(layer_fit[l]["r"], dtype=np.float32) for l in edited}
        for l, w in [(edited[0], "self_attn.o_proj"), (edited[len(edited) // 2], "self_attn.o_proj"),
                     (edited[-1], "self_attn.o_proj"), (edited[-1] + 1, "self_attn.o_proj"),
                     (edited[len(edited) // 3], "mlp.experts.17.down_proj"), (edited[-1], "mlp.experts.255.down_proj"),
                     (edited[len(edited) // 3], "self_attn.q_a_proj")]:
            d = get_fit(f"model.layers.{l}.{w}.weight")
            if d["edited"] and l in r:
                print(f"  cos to layer {l}'s r: {abs(float(np.array(d['r'], dtype=np.float32) @ r[l])):.6f}")
    n, hidden = len(edited), len(layer_fit[edited[0]]["r"])
    rs = np.array([layer_fit[l]["r"] for l in edited], dtype=np.float32)
    al = np.array([a.alpha if a.alpha is not None else layer_fit[l]["alpha"] for l in edited], dtype=np.float32)
    write_safetensors(a.out, {"glm_ablit.layers": ("I32", [n], np.array(edited, dtype=np.int32).tobytes()),
                              "glm_ablit.alpha": ("F32", [n], al.tobytes()),
                              "glm_ablit.r": ("F32", [n, hidden], rs.tobytes())})
    print(f"{a.out}: layers {edited[0]}-{edited[-1]} ({n}), alpha {al.min():.4f}-{al.max():.4f}")


if __name__ == "__main__":
    main()
