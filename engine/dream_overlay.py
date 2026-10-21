"""DSV41_DREAM_OVERLAY: serve some routed experts from a dreaming overlay instead of the checkpoint.

Dreaming trains a few routed experts while the engine is idle and writes each one back in the
checkpoint's own format: the six tensors of `layers.{L}.ffn.experts.{E}.` (packed FP4 `I8`
weights, `F8_E8M0` block scales) under the checkpoint's own names, one safetensors file per expert.
An overlay generation is a directory of those files plus `manifest.json`:

    {"version": 1, "base_index_sha256": "<sha256 of model.safetensors.index.json>",
     "parent": "gen-0006" | null,
     "experts": [{"layer": 12, "expert": 345, "file": "L12.E0345.safetensors", "sha256": "..."}]}

Serving it is a lookup change only: `ExpertStore._shard` resolves those tensor names to the overlay
file, and `ShardFile` reads any safetensors file, so the arena, the slots, the kernels and the
residency planner never know. Nothing is re-quantized on the way in.

The checkpoint is never written. Undo is pointing DSV41_DREAM_OVERLAY at the parent generation, or
unsetting it. The path may be a symlink (`serve/current`); it is resolved once at boot.

EP2: both ranks read the full expert and keep their half (DSV41_TP_EXPERTS), so each node holds the
same files at the same path. A node whose copy differs from its manifest refuses to boot here
(every file is hashed), and the manifest digest is in the boot guard and in the prefix-cache
namespace -- a KV prefix computed under one generation is not valid under another.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

ENV = "DSV41_DREAM_OVERLAY"
MANIFEST = "manifest.json"
VERSION = 1
NAMES = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")

_digest_cache: dict[str, str] = {}


def path() -> str | None:
    p = os.environ.get(ENV) or None
    return os.path.realpath(p) if p else None


def enabled() -> bool:
    return path() is not None


def file_name(layer: int, expert: int) -> str:
    return f"L{layer}.E{expert:04d}.safetensors"


def prefix(layer: int, expert: int) -> str:
    return f"layers.{layer}.ffn.experts.{expert}."


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_header(p: str) -> dict:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def digest() -> str:
    """sha256 of the generation's manifest, 16 hex chars ("" when off).

    The manifest lists every file's sha256 and `load` verifies the files against it, so this one
    digest pins the overlay's content; it is what the boot guard compares across ranks."""
    p = path()
    if not p:
        return ""
    if p not in _digest_cache:
        _digest_cache[p] = sha256_file(os.path.join(p, MANIFEST))[:16]
    return _digest_cache[p]


def base_sha256(model_dir: str, weight_map: dict, layer: int, expert: int) -> str:
    """sha256 of one expert's six tensors as the checkpoint stores them, in NAMES order.

    The overlay records it for every expert it was trained from, and `load` recomputes it on the
    serving checkpoint. The index alone cannot tell checkpoints apart: DeepSeek-V4.1-Flash and its
    1017 update ship byte-identical model.safetensors.index.json files (same names, same shards,
    different weights), and an overlay trained on the first was served on the second (2026-10-20,
    tiny-D/tiny-E) with nothing refusing it."""
    t = read_expert(model_dir, weight_map, layer, expert)
    h = hashlib.sha256()
    for n in NAMES:
        h.update(t[n][2])
    return h.hexdigest()


def load(model_dir: str, weight_map: dict, n_layers: int) -> dict[str, str]:
    """Verify the overlay generation and return {tensor name: overlay file} for ExpertStore.

    Fails before any expert is read if anything is off: a manifest for another checkpoint, a file
    missing or not matching its recorded sha256, or a file that is not exactly the six tensors of
    its expert with the checkpoint's dtypes and shapes. A half-copied overlay on one node would
    otherwise serve different weights on the two ranks with nothing in the log."""
    root = path()
    if not root:
        return {}
    with open(os.path.join(root, MANIFEST)) as f:
        man = json.load(f)
    if man.get("version") != VERSION:
        raise ValueError(f"{ENV}={root}: manifest version {man.get('version')} != {VERSION}")
    base = sha256_file(os.path.join(model_dir, "model.safetensors.index.json"))
    if man.get("base_index_sha256") != base:
        raise ValueError(f"{ENV}={root}: made for checkpoint index {man.get('base_index_sha256')}, "
                         f"this checkpoint is {base}")
    out: dict[str, str] = {}
    base_hdrs: dict[str, dict] = {}
    seen = set()
    for e in man["experts"]:
        layer, expert = int(e["layer"]), int(e["expert"])
        if not 0 <= layer < n_layers or not 0 <= expert < 384:
            raise ValueError(f"{ENV}: expert ({layer}, {expert}) is not a routed expert of the backbone")
        if (layer, expert) in seen:
            raise ValueError(f"{ENV}: expert ({layer}, {expert}) listed twice")
        seen.add((layer, expert))
        fp = os.path.join(root, e["file"])
        have = sha256_file(fp)
        if have != e["sha256"]:
            raise ValueError(f"{ENV}: {fp} sha256 {have[:16]} != manifest {e['sha256'][:16]} "
                             f"(a partial copy on this node?)")
        if not e.get("base_sha256"):
            raise ValueError(f"{ENV}: expert ({layer}, {expert}) has no base_sha256 -- the generation predates "
                             "base hashes (2026-10-21) and may come from another checkpoint; retrain it")
        base_have = base_sha256(model_dir, weight_map, layer, expert)
        if base_have != e["base_sha256"]:
            raise ValueError(f"{ENV}: expert ({layer}, {expert}) was trained from different weights than this "
                             f"checkpoint's ({e['base_sha256'][:16]} vs {base_have[:16]}): another checkpoint?")
        p = prefix(layer, expert)
        hdr = read_header(fp)
        want = {p + n for n in NAMES}
        if set(hdr) != want:
            raise ValueError(f"{ENV}: {fp} holds {sorted(hdr)}, expected exactly {sorted(want)}")
        shard = weight_map[p + NAMES[0]]
        if shard not in base_hdrs:
            base_hdrs[shard] = read_header(os.path.join(model_dir, shard))
        for name in want:
            b, o = base_hdrs[shard][name], hdr[name]
            if (o["dtype"], o["shape"]) != (b["dtype"], b["shape"]):
                raise ValueError(f"{ENV}: {fp} {name} is {o['dtype']}{o['shape']}, "
                                 f"the checkpoint's is {b['dtype']}{b['shape']}")
            out[name] = fp
    return out


FP8_NAMES = {"w1.weight": ("F8_E4M3", [2304, 5120]), "w1.scale": ("F8_E8M0", [2304, 160]),
             "w2.weight": ("F8_E4M3", [5120, 2304]), "w2.scale": ("F8_E8M0", [5120, 72]),
             "w3.weight": ("F8_E4M3", [2304, 5120]), "w3.scale": ("F8_E8M0", [2304, 160])}


def fp8_entries() -> list[dict]:
    """Manifest entries the generation serves as FP8 (serve == "fp8"), [] when off."""
    root = path()
    if not root:
        return []
    with open(os.path.join(root, MANIFEST)) as f:
        man = json.load(f)
    return [e for e in man["experts"] if e.get("serve") == "fp8"]


def _read_fp8(root: str, e: dict) -> dict[str, "torch.Tensor"]:
    """One FP8-tier expert's six tensors as CPU uint8 ({"w1.weight": ..., ...}), checked against
    its manifest hash and the FP8 names, dtypes and shapes."""
    import numpy as np
    import torch
    layer, expert = int(e["layer"]), int(e["expert"])
    fp = os.path.join(root, e["fp8_file"])
    have = sha256_file(fp)
    if have != e["fp8_sha256"]:
        raise ValueError(f"{ENV}: {fp} sha256 {have[:16]} != manifest {e['fp8_sha256'][:16]} "
                         f"(a partial copy on this node?)")
    hdr = read_header(fp)
    p = prefix(layer, expert)
    if set(hdr) != {p + n for n in FP8_NAMES}:
        raise ValueError(f"{ENV}: {fp} holds {sorted(hdr)}, expected the six FP8 tensors of {p}")
    with open(fp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read()[n:]
    out = {}
    for name, (dtype, shape) in FP8_NAMES.items():
        h = hdr[p + name]
        if (h["dtype"], h["shape"]) != (dtype, shape):
            raise ValueError(f"{ENV}: {fp} {name} is {h['dtype']}{h['shape']}")
        a, b = h["data_offsets"]
        out[name] = torch.from_numpy(np.frombuffer(blob[a:b], dtype=np.uint8).copy()).view(shape)
    return out


def load_fp8(device) -> dict[int, dict[int, tuple]]:
    """{layer: {expert: (w1, w2, w3)}} as BF16 on `device`, for the FP8-tier experts.

    The trainer writes FP8 as e4m3 codes with one power-of-two (UE8M0) scale per 32 weights of a
    row; e4m3 times a power of two is exact in BF16. Serving uses `load_fp8_tp` and the FP8
    kernels; this full-precision copy is the reference the kernel tests compare against."""
    import torch
    root = path()
    out: dict[int, dict[int, tuple]] = {}
    for e in fp8_entries():
        t = _read_fp8(root, e)
        ws = []
        for m in ("w1", "w2", "w3"):
            codes = t[f"{m}.weight"].to(device).view(torch.float8_e4m3fn).float()
            scale = torch.exp2(t[f"{m}.scale"].to(device).float() - 127).repeat_interleave(32, 1)
            ws.append((codes * scale).to(torch.bfloat16).contiguous())
        out.setdefault(int(e["layer"]), {})[int(e["expert"])] = tuple(ws)
    return out


class Fp8Layer:
    """One layer's FP8-tier experts as the FP8 kernels read them (tools/fp4_moe_cuda.py::dream_fp8_*).

    map [n_experts] int32: expert id -> tier slot, -1 for every other expert.
    w13 [S, 2n, 5120] / s13 [S, 2n, 160]: this rank's w1 rows then its w3 rows (n = 2304 / world).
    w2 [S, m, 2304] / s2 [S, m, 72]: this rank's output rows (m = 5120 / world), the full K.
    The slicing is ExpertArena.load_slot's output layout, so each rank overwrites exactly the rows
    its FP4 kernels wrote for the same pairs."""

    def __init__(self, ids, w13, s13, w2, s2, map_):
        self.ids, self.w13, self.s13, self.w2, self.s2, self.map = ids, w13, s13, w2, s2, map_

    @property
    def nbytes(self) -> int:
        return sum(t.numel() for t in (self.w13, self.s13, self.w2, self.s2))


def fp8_tp_bytes(world: int) -> int:
    """Device bytes `load_fp8_tp` will hold on one rank."""
    per = (2 * (2304 // world) * (5120 + 160)) + (5120 // world) * (2304 + 72)
    return len(fp8_entries()) * per


def fp8_layer(items: list[tuple[int, dict]], device, rank: int, world: int, n_experts: int = 384) -> Fp8Layer:
    """One layer's tier from [(expert, its six FP8 tensors as CPU uint8)], this rank's slice."""
    import torch
    inter, out_n = 2304 // world, 5120 // world
    lo, row = rank * inter, rank * out_n
    items = sorted(items, key=lambda x: x[0])
    cat = lambda name, a, b: torch.stack([t[name][a:b] for _, t in items])
    w13 = torch.cat([cat("w1.weight", lo, lo + inter), cat("w3.weight", lo, lo + inter)], dim=1)
    s13 = torch.cat([cat("w1.scale", lo, lo + inter), cat("w3.scale", lo, lo + inter)], dim=1)
    map_ = torch.full((n_experts,), -1, dtype=torch.int32)
    for s, (expert, _) in enumerate(items):
        map_[expert] = s
    return Fp8Layer([x for x, _ in items], w13.to(device).contiguous(), s13.to(device).contiguous(),
                    cat("w2.weight", row, row + out_n).to(device).contiguous(),
                    cat("w2.scale", row, row + out_n).to(device).contiguous(), map_.to(device))


def load_fp8_tp(device, rank: int, world: int, n_experts: int = 384) -> dict[int, Fp8Layer]:
    """{layer: Fp8Layer} for the FP8-tier experts, this rank's slice, as e4m3 codes + UE8M0 scales."""
    root = path()
    by_layer: dict[int, list[tuple[int, dict]]] = {}
    for e in fp8_entries():
        by_layer.setdefault(int(e["layer"]), []).append((int(e["expert"]), _read_fp8(root, e)))
    return {L: fp8_layer(items, device, rank, world, n_experts) for L, items in sorted(by_layer.items())}


def pins() -> dict[int, list[int]]:
    """{layer: [expert, ...]} of every expert the generation overrides (MXFP4 and FP8 alike).

    The engine keeps them resident and routable: a trained expert the residency planner prunes is
    one the router can never pick, so its lesson would silently not exist."""
    root = path()
    if not root:
        return {}
    with open(os.path.join(root, MANIFEST)) as f:
        man = json.load(f)
    out: dict[int, list[int]] = {}
    for e in man["experts"]:
        out.setdefault(int(e["layer"]), []).append(int(e["expert"]))
    return {L: sorted(v) for L, v in out.items()}


def describe() -> str:
    root = path()
    if not root:
        return "off"
    with open(os.path.join(root, MANIFEST)) as f:
        man = json.load(f)
    n8 = sum(e.get("serve") == "fp8" for e in man["experts"])
    fp8 = f", {n8} served FP8" if n8 else ""
    return (f"{os.path.basename(root)} ({len(man['experts'])} experts, parent {man.get('parent') or 'none'}, "
            f"manifest {digest()}{fp8})")


def write_expert(dst: str, layer: int, expert: int, tensors: dict[str, tuple[str, list, bytes]]) -> str:
    """Write one expert as a safetensors file: {name suffix: (dtype, shape, raw bytes)}.

    Raw bytes and dtype strings are copied verbatim, so FP4/E8M0 need no library support.
    Returns the file's sha256."""
    p = prefix(layer, expert)
    hdr, off, blobs = {}, 0, []
    for n in NAMES:
        dtype, shape, raw = tensors[n]
        hdr[p + n] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    os.replace(tmp, dst)
    return sha256_file(dst)


def read_expert(model_dir: str, weight_map: dict, layer: int, expert: int) -> dict[str, tuple[str, list, bytes]]:
    """The checkpoint's six tensors of one expert as {name suffix: (dtype, shape, raw bytes)}."""
    p = prefix(layer, expert)
    shard = os.path.join(model_dir, weight_map[p + NAMES[0]])
    hdr = read_header(shard)
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        out = {}
        for name in NAMES:
            v = hdr[p + name]
            a, b = v["data_offsets"]
            f.seek(8 + n + a)
            out[name] = (v["dtype"], v["shape"], f.read(b - a))
    return out


def pin_keep(keep: dict, counts: dict, pinned: dict[int, list[int]]) -> dict[int, list[int]]:
    """The residency keep set with every pinned expert in it, each layer's size unchanged.

    A pinned expert that is not kept replaces the kept, unpinned expert with the least recorded
    demand in its layer (ties: the higher id). Deterministic, so both ranks agree."""
    out = {L: [int(e) for e in ids] for L, ids in keep.items()}
    for L, want in pinned.items():
        if L not in out:
            continue
        ids = out[L]
        for e in want:
            if e in ids:
                continue
            spare = [x for x in ids if x not in want]
            if not spare:
                raise ValueError(f"{ENV}: layer {L} keeps {len(ids)} experts, fewer than its pinned overlay experts")
            drop = min(spare, key=lambda x: (float(counts[L][x]), -x))
            ids[ids.index(drop)] = e
    return out
