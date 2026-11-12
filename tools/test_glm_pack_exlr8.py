"""A pack tools/glm_pack_exlr8.py wrote against the reference: every record's three matrices, decoded
as the EXL3 kernels read them (tiles [K/16][N/16], suh, svh), equal the rank's slice of the exlr8
units decoded by tools/exlr8_ref.py.

    uv run --with numpy python tools/test_glm_pack_exlr8.py PACK --source ~/models/glm-5.3-exlr8 [--max 4]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exlr8_ref as X  # noqa: E402


def matrix(tiles: bytes, k: int, kt: int, nt: int) -> np.ndarray:
    """Tiles [kt][nt] of 32k bytes -> T (16 kt, 16 nt)."""
    t = X.decode_tiles(np.frombuffer(tiles, np.uint8).reshape(-1, 32 * k), k)
    return t.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(16 * kt, 16 * nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--source", required=True)
    ap.add_argument("--max", type=int, default=4)
    a = ap.parse_args()
    src = os.path.expanduser(a.source)
    f = open(a.pack, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    data0 = (8 + n + 4095) // 4096 * 4096
    H, I = h["dims"]
    rank, world = h["rank"], h["world"]
    il, hl = I // world, H // world
    assert h["codebook"] == "mcg-e4m3" and h["exlr8"]["layout"] == "tp-output"
    bad = 0
    for key, r in list(h["records"].items())[:a.max]:
        L, e = (int(v) for v in key.split(","))
        kg, ku, kd = r["bits"]
        f.seek(data0 + r["off"])
        rec = f.read(r["nbytes"])
        sz = [(H // 16) * (il // 16) * 32 * kg, (H // 16) * (il // 16) * 32 * ku, (I // 16) * (hl // 16) * 32 * kd,
              2 * H, 2 * H, 2 * I, 2 * il, 2 * il, 2 * hl]
        o, parts = 0, []
        for s in sz:
            parts.append(rec[o:o + s])
            o += s
        assert o == len(rec)
        suh1, suh3, suh2, svh1, svh3, svh2 = (np.frombuffer(p, np.float16) for p in parts[3:])
        idx = X.SafeFile(os.path.join(src, "experts", f"exlr8-layer-{L:03d}.index.safetensors"))
        sin, sout = idx.tensor("signs.in"), idx.tensor("signs.out")
        base = os.path.join(src, "experts", f"exlr8-layer-{L:03d}")
        worst = 0.0
        for pi, (p, k, tiles, su, sv) in enumerate((("gate", kg, parts[0], suh1, svh1), ("up", ku, parts[1], suh3, svh3))):
            got = X.reconstruct(matrix(tiles, k, H // 16, il // 16), su, sv)
            tf = X.SafeFile(f"{base}.k{k}.safetensors")
            index, blk, chan = idx.tensor(f"k{k}.index"), idx.tensor(f"k{k}.{p}.su_block"), idx.tensor(f"k{k}.{p}.svh")
            want = np.concatenate([X.gate_up_unit(tf.read_at(int(index[e, pi, u, 0]), int(index[e, pi, u, 1])), k, sin, blk[e],
                                                  chan[e, 128 * u:128 * u + 128]) for u in range(rank * 8, rank * 8 + 8)], axis=1)
            worst = max(worst, float(np.abs(got - want).max()))
        got = X.reconstruct(matrix(parts[2], kd, I // 16, hl // 16), suh2, svh2)
        tf = X.SafeFile(f"{base}.k{kd}.safetensors")
        index, blk, chan = idx.tensor(f"k{kd}.index"), idx.tensor(f"k{kd}.down.sv_block"), idx.tensor(f"k{kd}.down.suh")
        want = np.concatenate([X.down_unit(tf.read_at(int(index[e, 2, u, 0]), int(index[e, 2, u, 1])), kd, chan[e, 128 * u:128 * u + 128],
                                           sout, blk[e]) for u in range(16)], axis=0)[:, rank * hl:(rank + 1) * hl]
        worst = max(worst, float(np.abs(got - want).max()))
        bad += worst > 1e-6          # fp32 matmul order; a wrong tile or scale is ~1e-1
        print(f"record {key} bits {r['bits']} rank {rank}: max |pack - reference| {worst:.1e}")
    print("FAILED" if bad else "the pack decodes to the reference")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
