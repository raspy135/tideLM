"""The dense checkpoint of an exlr8 profile: GLM_MODEL_DIR with exlr8's trellis matrices in EXL3's layout.

    uv run --with numpy --with xxhash python tools/glm_exlr8_dense.py --base ~/models/GLM-5.3-dense-from-3.0bpw \
        --exlr8 ~/models/glm-5.3-exlr8 --manifest bit260 --out ~/models/GLM-5.3-dense-exlr8

The engine reads its non-expert weights from an EXL3 checkpoint (tools/glm_fetch_dense.py: bertholomus's
3.0bpw, 5-6 bits a weight). exlr8 stores the large ones as 4-bit trellis - o_proj, q_b, the dense
MLPs of layers 0-2 - and the shared expert as units like a routed expert's (K4; K3 in the MTP layer).
Those decode with the routed kernels' arithmetic (tools/exlr8_ref.py) and regroup, tile for tile, into
the full EXL3 matrix `<name>.trellis` int16 [in/16][out/16][16 K] with `<name>.suh` / `<name>.svh`,
which the engine cuts per rank as it cuts the EXL3 ones (weights.rs). So --out is the base
checkpoint (its shards hard-linked) with those matrices replaced, each marked `<name>.mcg_e4m3`
instead of `<name>.mul1`: the engine takes the codebook from the marker.

A decode pass then reads 4 bits where it read 5 (6 in the MTP layer): a rank's o_proj 31.5 -> 25.2 MB a
layer, q_b 10.5 -> 8.4, the shared expert 11.8 -> 9.4.

Left as the base's: what exlr8 keeps at 8 bits (q_a, kv_a, the indexer's wq_b in MXFP8; kv_b, the
embedding and the head in FP8) - the base's 5-6-bit EXL3 is fewer bytes a pass - and the five matrices
exlr8 stores as int4 (q_b of layers 0 and 2, o_proj of 0, 2 and 3). Every dense block and shared unit
read is checked against the release's xxhash64.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exlr8_ref as X  # noqa: E402

MARKER = np.array(0xCBAC1FED, np.uint32).view(np.int32)
DENSE = ("self_attn.o_proj", "self_attn.q_b_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SHARED = (("gate", "mlp.shared_experts.gate_proj"), ("up", "mlp.shared_experts.up_proj"), ("down", "mlp.shared_experts.down_proj"))
DT = {np.dtype(np.int16): "I16", np.dtype(np.float16): "F16", np.dtype(np.int32): "I32"}


def write_safetensors(path: str, tensors: dict[str, np.ndarray], meta: dict[str, str]):
    head, off = {"__metadata__": meta}, 0
    for name, a in tensors.items():
        head[name] = {"dtype": DT[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    blob = json.dumps(head, separators=(",", ":")).encode()
    blob += b" " * (-(8 + len(blob)) % 8)
    with open(path + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".part", path)


def dense_matrix(f: X.SafeFile, name: str, verify: bool):
    """(trellis int16 [in/16][out/16][16 K], suh [in], svh [out], K) of a format-1 matrix; None for another format."""
    rec = [int(v) for v in f.tensor(name + ".record")]
    fmt, out, inn, k, side = rec[0], rec[1], rec[2], rec[3], rec[4]
    if fmt != 1:
        return None
    tr, suh, svh = f.tensor(name + ".trellis"), f.tensor(name + ".suh"), f.tensor(name + ".svh")
    if verify:
        import xxhash
        want = f.tensor(name + ".checksum")
        for b in range(tr.shape[0]):
            c = slice(128 * b, 128 * b + 128)
            own, other = (suh[c], svh) if side == 2 else (svh[c], suh)
            if xxhash.xxh64(tr[b].tobytes() + own.tobytes() + other.tobytes(), seed=0).intdigest() != int(want[b]):
                raise SystemExit(f"{f.path}: {name} block {b}: checksum mismatch")
    t = 32 * k
    if side == 2:       # entry g = input rows 128g.., tiles [b][r][c] -> [k16 = 8g + r][n16 = 8b + c]
        full = tr.reshape(inn // 128, out // 128, 8, 8, t).transpose(0, 2, 1, 3, 4).reshape(inn // 16, out // 16, t)
    else:               # entry g = output columns 128g.., tiles [k16][c] -> [k16][n16 = 8g + c]
        full = tr.reshape(out // 128, inn // 16, 8, t).transpose(1, 0, 2, 3).reshape(inn // 16, out // 16, t)
    return np.ascontiguousarray(full).view(np.int16), suh, svh, k


def shared_matrix(root: str, layer: int, p: str, widths, verify: bool):
    """The shared expert's projection p from its 16 units, as dense_matrix returns one."""
    base = os.path.join(root, "experts", f"exlr8-layer-{layer:03d}")
    idx = X.SafeFile(base + ".index.safetensors")
    pi = X.PROJS.index(p)
    ks = {int(widths["down"])} if p == "down" else {int(v) for v in widths[p]}
    if len(ks) != 1:
        raise SystemExit(f"layer {layer}: the shared expert's {p} units differ in width")
    k = ks.pop()
    H = idx.tensor("signs.in").shape[0]
    B = X.unit_bytes(H, k)
    index, check = idx.tensor(f"shared.k{k}.index"), idx.tensor(f"shared.k{k}.checksum")
    block = idx.tensor(f"shared.k{k}.{p}.sv_block" if p == "down" else f"shared.k{k}.{p}.su_block")[0]
    chan = idx.tensor(f"shared.k{k}.down.suh" if p == "down" else f"shared.k{k}.{p}.svh")[0]
    with open(f"{base}.k{k}.safetensors", "rb") as f:
        f.seek(int(index[0, pi, 0, 0]))
        raw = f.read(16 * B)
    if verify:
        import xxhash
        for u in range(16):
            got = xxhash.xxh64(raw[u * B:(u + 1) * B] + chan[128 * u:128 * u + 128].tobytes() + block.tobytes(), seed=0).intdigest()
            if got != int(check[0, pi, u]):
                raise SystemExit(f"layer {layer}: shared {p} unit {u} K{k}: checksum mismatch")
    t = 32 * k
    units = np.frombuffer(raw, np.uint8).reshape(16, B)
    rep = np.repeat(block, 128)
    if p == "down":
        full = units.reshape(16, H // 128, 8, 8, t).transpose(0, 2, 1, 3, 4).reshape(128, H // 16, t)
        suh, svh = chan, idx.tensor("signs.out").astype(np.float16) * rep
    else:
        full = units.reshape(16, H // 16, 8, t).transpose(1, 0, 2, 3).reshape(H // 16, 128, t)
        suh, svh = idx.tensor("signs.in").astype(np.float16) * rep, chan
    return np.ascontiguousarray(full).view(np.int16), suh, svh, k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--exlr8", required=True)
    ap.add_argument("--manifest", default="bit260")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-78")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    if not a.no_verify:
        try:
            import xxhash  # noqa: F401
        except ImportError:
            raise SystemExit("the checksums need the xxhash module (pip install xxhash, or uv run --with numpy --with xxhash "
                             "python ...); --no-verify skips them")
    base, root, out = (os.path.expanduser(v) for v in (a.base, a.exlr8, a.out))
    man = json.load(open(os.path.join(root, "manifests", f"exlr8-manifest-{a.manifest}.json")))
    index = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    wm = dict(index["weight_map"])
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        if f != "model.safetensors.index.json" and not os.path.exists(os.path.join(out, f)):
            os.link(os.path.join(base, f), os.path.join(out, f))          # the base's shards, not copied
    lo, hi = (int(v) for v in a.layers.split("-"))
    saved = [0, 0]
    for L in range(lo, hi + 1):
        new, notes = {}, []
        df = X.SafeFile(os.path.join(root, "dense", f"exlr8-dense-{L:03d}.safetensors"))
        todo = [(n, lambda n=n: dense_matrix(df, n, not a.no_verify)) for n in DENSE if n + ".record" in df.header]
        if str(L) in man["layers"]:
            sh = man["layers"][str(L)]["shared"]
            todo += [(name, lambda p=p: shared_matrix(root, L, p, sh, not a.no_verify)) for p, name in SHARED]
        for name, get in todo:
            pre = f"model.layers.{L}.{name}"
            if pre + ".trellis" not in wm:
                raise SystemExit(f"{pre}: not in the base checkpoint")
            m = get()
            if m is None:
                notes.append(f"{name.split('.')[-1]} kept")
                continue
            tr, suh, svh, k = m
            old = json.loads(open(os.path.join(base, wm[pre + ".trellis"]), "rb").read(8 + struct.unpack("<Q", open(os.path.join(base, wm[pre + ".trellis"]), "rb").read(8))[0])[8:])[pre + ".trellis"]["shape"]
            if old[:2] != list(tr.shape[:2]) or len(suh) != 16 * old[0] or len(svh) != 16 * old[1]:
                raise SystemExit(f"{pre}: exlr8 {tr.shape} against the base's {old}")
            new[pre + ".trellis"], new[pre + ".suh"], new[pre + ".svh"] = tr, np.ascontiguousarray(suh, np.float16), np.ascontiguousarray(svh, np.float16)
            new[pre + ".mcg_e4m3"] = MARKER
            saved[0] += old[0] * old[1] * old[2] * 2
            saved[1] += tr.nbytes
            notes.append(f"{name.split('.')[-1]} {old[2] // 16}->{k} bit")
        if new:
            shard = f"exlr8-overlay-{L:03d}.safetensors"
            write_safetensors(os.path.join(out, shard), new, {"source": f"exlr8 {man['set_id']} {a.manifest}", "layer": str(L)})
            for name in new:
                wm[name] = shard
                wm.pop(name[:-len(".mcg_e4m3")] + ".mul1", None) if name.endswith(".mcg_e4m3") else None
        print(f"layer {L}: {', '.join(notes)}", flush=True)
    index["weight_map"] = wm
    index.setdefault("metadata", {})["exlr8"] = f"{man['set_id']} {a.manifest}: trellis matrices and shared experts of layers {lo}-{hi}"
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{out}: replaced {saved[0] / 1e9:.2f} GB of trellis with {saved[1] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
