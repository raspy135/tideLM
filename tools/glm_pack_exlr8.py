"""Offline packer: an exlr8 profile's routed experts -> one aligned file per rank, in the EXL3 pack's layout.

    uv run --with numpy --with xxhash python tools/glm_pack_exlr8.py --source ~/models/glm-5.3-exlr8 \
        --manifest bit260 --rank 0 --out ~/models/glm-exlr8-packs/glm-exlr8-bit260-r0of2.bin
    ... --rank 1 --out - | ssh PEER 'cat > models/glm-exlr8-packs/glm-exlr8-bit260-r1of2.bin'

--source is tools/exlr8_fetch.py's mirror. An exlr8 unit is 128 intermediate channels of one expert
and decodes on its own (tools/exlr8_ref.py), its tiles independent 16 x 16 blocks and both rotations
128-block Hadamards - so the units regroup, tile for tile and byte for byte, into the matrices the
EXL3 kernels and the engine's TP split already read (tools/exl3_ref.py's record, tools/glm_pack_exl3.py):

  gate, up   rank r keeps the intermediate columns [r I/W, (r+1) I/W): units 8r..8r+7 (W = 2), their
             tiles [k16][c] side by side -> t1 / t3 [H/16][I/W/16]
  down       rank r keeps the hidden columns [r H/W, (r+1) H/W) of every unit: each unit's tiles
             [b][r'][c] with b in the rank's blocks -> t2 [I/16][H/W/16]
  suh1/suh3  signs.in x the expert's 48 block scales, expanded to [H] (exact: a sign flip)
  suh2       the expert's down suh [I];  svh1 / svh3 the rank's columns of gate / up svh
  svh2       signs.out x the expert's down block scales, the rank's columns

and the engine serves them unchanged, with codebook "mcg-e4m3" (mcg, every value rounded to FP8 E4M3:
kernel CB 3). This is the first exlr8 layout - the engine's own split, not the format's sharding rule
(16 e + u + L) mod N; the one-rotation, one-exchange layout that rule allows comes after it.

A profile must give an expert one width a projection (every shipped manifest does). Every unit read is
checked against the index's xxhash64. Records are written in order, so --out - streams a pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exlr8_ref as X  # noqa: E402

class Missing(Exception):
    """A unit whose bytes are not (yet) what the index says: not fetched, or being written."""


ALIGN = 4096
RECORD_TENSORS = ("t1", "t3", "t2", "suh1", "suh3", "suh2", "svh1", "svh3", "svh2")
UNITS = 16


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def record_nbytes(bits, world: int, hidden: int, inter: int) -> int:
    il, hl = inter // world, hidden // world
    tile = lambda b: 32 * b
    return ((hidden // 16) * (il // 16) * (tile(bits[0]) + tile(bits[1])) + (inter // 16) * (hl // 16) * tile(bits[2])
            + 2 * (2 * hidden + inter + 2 * il + hl))


class Layer:
    """One MoE layer of the mirror: its index tensors and trellis files, read a record at a time."""

    def __init__(self, root: str, layer: int, lay: dict, verify: bool):
        self.L, self.lay, self.verify = layer, lay, verify
        base = os.path.join(root, "experts", f"exlr8-layer-{layer:03d}")
        self.idx = X.SafeFile(base + ".index.safetensors")
        self.base = base
        self.sin, self.sout = self.idx.tensor("signs.in").astype(np.float16), self.idx.tensor("signs.out").astype(np.float16)
        self.hidden = self.sin.shape[0]
        self.side, self.fds = {}, {}

    def k(self, k: int):
        if k not in self.side:
            t = self.idx.tensor
            self.side[k] = {"index": t(f"k{k}.index"), "check": t(f"k{k}.checksum"),
                            "gate": (t(f"k{k}.gate.su_block"), t(f"k{k}.gate.svh")), "up": (t(f"k{k}.up.su_block"), t(f"k{k}.up.svh")),
                            "down": (t(f"k{k}.down.sv_block"), t(f"k{k}.down.suh"))}
            self.fds[k] = os.open(f"{self.base}.k{k}.safetensors", os.O_RDONLY)
        return self.side[k]

    def units(self, e: int, pi: int, k: int, u0: int, n: int) -> np.ndarray:
        """Units [u0, u0 + n) of projection pi of expert e at width k: uint8 [n, B], checked."""
        s = self.k(k)
        B = X.unit_bytes(self.hidden, k)
        off = int(s["index"][e, pi, u0, 0])
        raw = os.pread(self.fds[k], n * B, off)
        os.posix_fadvise(self.fds[k], off, n * B, os.POSIX_FADV_DONTNEED)      # read once: not left in the page cache
        if len(raw) != n * B or int(s["index"][e, pi, u0 + n - 1, 0]) != off + (n - 1) * B:
            raise SystemExit(f"layer {self.L} expert {e} {X.PROJS[pi]} K{k}: units not contiguous, or a short read")
        if self.verify:
            import xxhash
            block, chan = s[X.PROJS[pi]]
            for j in range(n):
                u = u0 + j
                got = xxhash.xxh64(raw[j * B:(j + 1) * B] + chan[e, 128 * u:128 * u + 128].tobytes() + block[e].tobytes(), seed=0).intdigest()
                if got != int(s["check"][e, pi, u]):
                    raise Missing(f"layer {self.L} expert {e} {X.PROJS[pi]} unit {u} K{k}: checksum mismatch "
                                  f"(a unit the fetch did not bring? rerun tools/exlr8_fetch.py)")
        return np.frombuffer(raw, np.uint8).reshape(n, B)

    def bits(self, e: int):
        g, u = set(self.lay["gate"][e]), set(self.lay["up"][e])
        if len(g) != 1 or len(u) != 1:
            raise SystemExit(f"layer {self.L} expert {e}: its units differ in width; this layout takes one width a projection")
        return [int(g.pop()), int(u.pop()), int(self.lay["down"][e])]

    def record(self, e: int, rank: int, world: int) -> bytes:
        kg, ku, kd = self.bits(e)
        H, per = self.hidden, UNITS // world
        parts = []
        for pi, k in ((0, kg), (1, ku)):
            t = self.units(e, pi, k, rank * per, per).reshape(per, H // 16, 8, 32 * k)
            parts.append(np.ascontiguousarray(t.transpose(1, 0, 2, 3)))                      # [k16][unit, c]
        nb = H // 128 // world
        t = self.units(e, 2, kd, 0, UNITS).reshape(UNITS, H // 128, 8, 8, 32 * kd)[:, rank * nb:(rank + 1) * nb]
        down = np.ascontiguousarray(t.transpose(0, 2, 1, 3, 4))                               # [unit, r'][b, c]
        rep = lambda v: np.repeat(v, 128)
        il, hl = per * 128, H // world
        sg, su, sd = self.k(kg)["gate"], self.k(ku)["up"], self.k(kd)["down"]
        sides = [self.sin * rep(sg[0][e]), self.sin * rep(su[0][e]), sd[1][e],
                 sg[1][e, rank * il:(rank + 1) * il], su[1][e, rank * il:(rank + 1) * il],
                 (self.sout * rep(sd[0][e]))[rank * hl:(rank + 1) * hl]]
        return b"".join([parts[0].tobytes(), parts[1].tobytes(), down.tobytes()] + [np.ascontiguousarray(s, np.float16).tobytes() for s in sides])

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--manifest", default="bit260")
    ap.add_argument("--out", required=True, help="the pack, or - for stdout")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--layers", default="", help="comma-separated subset, for a test pack")
    ap.add_argument("--experts-per-layer", type=int, default=0, help="a small test pack (0: all)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--follow", type=int, default=0, metavar="SECONDS",
                    help="pack behind a running tools/exlr8_fetch.py: a unit that does not check yet is waited for, "
                         "up to this long without progress (the fetch goes in layer order, as the pack does)")
    a = ap.parse_args(argv)
    if not a.no_verify:
        try:
            import xxhash  # noqa: F401
        except ImportError:
            raise SystemExit("the checksums need the xxhash module (pip install xxhash, or uv run --with numpy --with xxhash "
                             "python ...); --no-verify skips them")
    if a.follow and a.no_verify:
        raise SystemExit("--follow needs the checksums: they are how a unit is known to have arrived")
    src = os.path.expanduser(a.source)
    mpath = os.path.join(src, "manifests", f"exlr8-manifest-{a.manifest}.json")
    man = json.load(open(mpath))
    cfg = json.load(open(os.path.join(src, "config.json")))
    hidden, inter = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
    layers = [int(v) for v in a.layers.split(",")] if a.layers else sorted(int(v) for v in man["layers"])
    n = a.experts_per_layer or int(cfg["n_routed_experts"])
    log = lambda s: print(s, file=sys.stderr, flush=True)
    records, off, bits_l = {}, 0, {}
    for L in layers:
        lay = man["layers"][str(L)]
        tot = 0
        for e in range(n):
            g, u = set(lay["gate"][e]), set(lay["up"][e])
            if len(g) != 1 or len(u) != 1:
                raise SystemExit(f"layer {L} expert {e}: its units differ in width; this layout takes one width a projection")
            b = [int(min(g)), int(min(u)), int(lay["down"][e])]
            nb = record_nbytes(b, a.world, hidden, inter)
            records[f"{L},{e}"] = {"off": off, "nbytes": nb, "bits": b}
            off += align_up(nb)
            tot += sum(b)
        bits_l[str(L)] = tot / (3 * n)
    source = hashlib.sha256(f"exlr8 {man['set_id']} {a.manifest} {hashlib.sha256(open(mpath, 'rb').read()).hexdigest()} tp-output".encode()).hexdigest()
    header = {"format": "exl3-experts", "version": 2, "rank": a.rank, "world": a.world, "codebook": "mcg-e4m3", "bits": bits_l,
              "record_align": ALIGN, "tensors": list(RECORD_TENSORS), "source_sha256": source, "data_nbytes": off,
              "dims": [hidden, inter], "records": records,
              "exlr8": {"manifest": a.manifest, "manifest_version": man.get("version"), "set_id": man["set_id"], "layout": "tp-output"}}
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    data0 = align_up(8 + len(blob))
    widths = sorted({tuple(r["bits"]) for r in records.values()})
    log(f"[pack] exlr8 {a.manifest}: {len(layers)} layers x {n} experts, widths {widths}, rank {a.rank}/{a.world}, "
        f"{(data0 + off) / 1e9:.1f} GB, source {source[:16]}")
    out = sys.stdout.buffer if a.out == "-" else open(os.path.expanduser(a.out), "wb")
    out.write((struct.pack("<Q", len(blob)) + blob).ljust(data0, b"\0"))
    t0, done, since = time.time(), 0, 0
    for L in layers:
        lay = Layer(src, L, man["layers"][str(L)], not a.no_verify)
        for e in range(n):
            waited = 0
            while True:
                try:
                    rec = lay.record(e, a.rank, a.world)
                    break
                except Missing as m:
                    if waited >= a.follow:
                        raise SystemExit(str(m))
                    if waited == 0:
                        log(f"[pack]   waiting for the fetch at layer {L} expert {e}")
                    time.sleep(20)
                    waited += 20
            meta = records[f"{L},{e}"]
            assert len(rec) == meta["nbytes"], (L, e, len(rec), meta["nbytes"])
            out.write(rec.ljust(align_up(len(rec)), b"\0"))
            done += 1
            since += len(rec)
            if a.out != "-" and since > (256 << 20):      # beside a serving engine: no pile of dirty pages
                out.flush()
                os.fsync(out.fileno())
                os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                since = 0
            if done % 1024 == 0:
                log(f"[pack]   {done} experts, {time.time() - t0:.0f}s")
        lay.close()
    out.flush()
    if a.out != "-":
        os.fsync(out.fileno())
        out.close()
    log(f"[pack] wrote {a.out}: {(data0 + off) / 1e9:.1f} GB, {done} records, header {hashlib.sha256(blob).hexdigest()[:16]}, "
        f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
