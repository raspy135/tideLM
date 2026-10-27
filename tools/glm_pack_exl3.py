"""Offline packer: GLM-5.3's EXL3 routed experts -> one aligned file per rank (GLM_EXPERT_FORMAT=exl3).

Reads `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,<codebook>}` from an
EXL3 GLM-5.3 checkpoint and writes each expert's TP-output rank slice with tools/exl3_ref.py's pack
writer -- the same record layout and 4096-aligned O_DIRECT records as the V4.1 pack, at GLM's
dims (6,144 hidden, 2,048 intermediate, 256 experts a layer). gate/up keep this rank's half of the
intermediate columns, down this rank's half of the hidden columns, as engine/glm/tp.py splits
the MXFP4 experts. Two sources:
  bertholomus/GLM-5.3-EXL3-3.0bpw: mul1, 3-bit routed experts on layers 3-77, 4-bit on the MTP
    layer 78 (a version-1 pack, one width a layer);
  Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold: mcg, each expert's gate / up / down at its own width
    of 2-4 bits (a version-2 pack, the widths in every record).

    python tools/glm_pack_exl3.py --source /models-exl3 --out /packs/glm-exl3-experts-r0of2.bin --rank 0

Run it on each node for its own rank; nothing here runs in serving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402
from safetensors import safe_open  # noqa: E402

PARTS = (("1", "gate_proj"), ("3", "up_proj"), ("2", "down_proj"))


class GlmExl3Source:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        q = dict(cfg.get("quantization_config") or {})
        qpath = os.path.join(model_dir, "quantization_config.json")   # Mia's: the codebook and widths
        if os.path.exists(qpath):
            q.update({k: v for k, v in json.load(open(qpath)).items() if k in ("quant_method", "codebook")})
        self.codebook = q.get("codebook", "mul1")
        if str(q.get("quant_method", "")).lower() != "exl3" or self.codebook not in ("mul1", "mcg"):
            raise SystemExit(f"{model_dir}: not an EXL3 mul1 / mcg checkpoint ({q})")
        self.marker = {"mul1": R.MUL1_MUL, "mcg": R.MCG_MUL}[self.codebook]
        self.hidden, self.inter = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
        self.n_experts = int(cfg["n_routed_experts"])
        R.set_dims(self.hidden, self.inter)
        self.layers = sorted({int(k.split(".")[2]) for k in self.map if ".mlp.experts." in k})
        self._open: dict[str, safe_open] = {}

    def _file(self, name):
        shard = self.map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.dir, shard), framework="numpy")
        return self._open[shard]

    def shape(self, name):
        return tuple(self._file(name).get_slice(name).get_shape())

    def array(self, name, sl=None):
        f = self._file(name)
        return np.ascontiguousarray(f.get_tensor(name) if sl is None else f.get_slice(name)[sl])

    def expert_bits(self, L, E) -> tuple[float, float, float]:
        """(gate, up, down) widths, from the trellis shapes."""
        return tuple(R.bits_of_shape(self.shape(f"model.layers.{L}.mlp.experts.{E}.{p}.trellis")) for _, p in PARTS)

    def marker_ok(self, L, E) -> bool:
        return all((int(self.array(f"model.layers.{L}.mlp.experts.{E}.{p}.{self.codebook}").reshape(-1)[0]) & 0xFFFFFFFF)
                   == self.marker for _, p in PARTS)

    def record(self, L, E, rank, world, bits=None) -> dict[str, np.ndarray]:
        bits = bits if bits is not None else self.expert_bits(L, E)[0]
        sh = R.record_shapes(bits, rank, world)
        nt13, nt2, n13, n2 = sh["t1"][1], sh["t2"][1], sh["svh1"][0], sh["svh2"][0]
        out = {}
        for tag, part in PARTS:
            p = f"model.layers.{L}.mlp.experts.{E}.{part}."
            nt, n = (nt2, n2) if tag == "2" else (nt13, n13)
            out["t" + tag] = self.array(p + "trellis", (slice(None), slice(rank * nt, (rank + 1) * nt), slice(None)))
            out["suh" + tag] = self.array(p + "suh")
            out["svh" + tag] = self.array(p + "svh", (slice(rank * n, (rank + 1) * n),))
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--experts-per-layer", type=int, default=0, help="a small test pack (0: all)")
    ap.add_argument("--layers", default="", help="comma-separated subset, for a test pack")
    a = ap.parse_args(argv)
    src = GlmExl3Source(a.source)
    layers = [int(L) for L in a.layers.split(",")] if a.layers else src.layers
    n = a.experts_per_layer or src.n_experts
    for L in layers:
        if not src.marker_ok(L, 0):
            raise SystemExit(f"layer {L} expert 0: {src.codebook} marker mismatch")
    ebits = {(L, E): src.expert_bits(L, E) for L in layers for E in range(n)}
    mixed = any(len(set(b)) > 1 for b in ebits.values()) or any(
        ebits[(L, E)] != ebits[(L, 0)] for L in layers for E in range(n))
    # the header's per-layer width: the layer's own, or its average for a mixed pack
    bits = {L: sum(sum(ebits[(L, E)]) for E in range(n)) / (3 * n) for L in layers}
    widths = sorted({b for v in ebits.values() for b in v})
    print(f"[pack] {len(layers)} layers x {n} experts, {src.codebook}, widths {widths}{' (mixed)' if mixed else ''}, "
          f"rank {a.rank}/{a.world}, dims {src.hidden}x{src.inter}", flush=True)
    sha = R.manifest_sha256(src.dir)
    t0, done = time.time(), [0]

    def records():
        for L in sorted(bits):
            for E in range(n):
                yield (L, E), src.record(L, E, a.rank, a.world, list(ebits[(L, E)]) if mixed else bits[L])
                done[0] += 1
                if done[0] % 1024 == 0:
                    print(f"[pack]   {done[0]} experts, {time.time() - t0:.0f}s", flush=True)

    R.write_pack(a.out, bits, a.rank, a.world, src.codebook, sha, records(), n_experts=n,
                 expert_bits=ebits if mixed else None)
    r = R.PackReader(a.out)
    print(f"[pack] wrote {a.out}: {os.path.getsize(a.out) / 1e9:.1f} GB, {len(r.records)} records, "
          f"header {r.header_sha256()[:16]}, source {sha[:16]}, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
