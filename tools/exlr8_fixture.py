"""A kernel fixture from an exlr8 pack (tools/glm_pack_exlr8.py): one record's gate and down matrices as
the EXL3 kernels read them, three bf16 input rows each, and y = x W from the exlr8 reference
(tools/exlr8_ref.py: W rebuilt from the published units, float64 product). rust/crates/tidelm-kernels/
tests/glm_gpu.rs (exlr8_codebook_matches_the_reference) runs the decode kernels over it.

    uv run --with numpy python tools/exlr8_fixture.py PACK --source ~/models/glm-5.3-exlr8 --out FIXTURE

File: for each matrix, u32 [k, n, bits, rows], the tiles [k/16][n/16] (32 bits bytes each), suh fp16 [k],
svh fp16 [n], x bf16 [rows, k], y f32 [rows, n].
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exlr8_ref as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args()
    src = os.path.expanduser(a.source)
    f = open(a.pack, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    data0 = (8 + n + 4095) // 4096 * 4096
    H, I = h["dims"]
    rank, world = h["rank"], h["world"]
    il, hl = I // world, H // world
    key, r = next(iter(h["records"].items()))
    L, e = (int(v) for v in key.split(","))
    kg, ku, kd = r["bits"]
    f.seek(data0 + r["off"])
    rec = f.read(r["nbytes"])
    sz = [(H // 16) * (il // 16) * 32 * kg, (H // 16) * (il // 16) * 32 * ku, (I // 16) * (hl // 16) * 32 * kd, 2 * H, 2 * H, 2 * I, 2 * il, 2 * il, 2 * hl]
    off = np.concatenate([[0], np.cumsum(sz)])
    part = lambda i: rec[off[i]:off[i + 1]]
    idx = X.SafeFile(os.path.join(src, "experts", f"exlr8-layer-{L:03d}.index.safetensors"))
    sin, sout = idx.tensor("signs.in"), idx.tensor("signs.out")
    base = os.path.join(src, "experts", f"exlr8-layer-{L:03d}")
    tf = X.SafeFile(f"{base}.k{kg}.safetensors")
    gcb = tf.trellis("trellis.gate")[1]
    index, blk, chan = idx.tensor(f"k{kg}.index"), idx.tensor(f"k{kg}.gate.su_block"), idx.tensor(f"k{kg}.gate.svh")
    wg = np.concatenate([X.gate_up_unit(tf.read_at(int(index[e, 0, u, 0]), int(index[e, 0, u, 1])), kg, sin, blk[e], chan[e, 128 * u:128 * u + 128], gcb)
                         for u in range(rank * 8, rank * 8 + 8)], axis=1)
    tf = X.SafeFile(f"{base}.k{kd}.safetensors")
    dcb = tf.trellis("trellis.down")[1]
    index, blk, chan = idx.tensor(f"k{kd}.index"), idx.tensor(f"k{kd}.down.sv_block"), idx.tensor(f"k{kd}.down.suh")
    wd = np.concatenate([X.down_unit(tf.read_at(int(index[e, 2, u, 0]), int(index[e, 2, u, 1])), kd, chan[e, 128 * u:128 * u + 128], sout, blk[e], dcb)
                         for u in range(16)], axis=0)[:, rank * hl:(rank + 1) * hl]
    if gcb != dcb:
        raise SystemExit(f"record {key}: gate {gcb}, down {dcb}: one codebook a fixture")
    print(f"codebook {gcb}: run the test with EXLR8_FIXTURE_CB={gcb}")
    rng = np.random.default_rng(7)
    with open(a.out, "wb") as o:
        for w, bits, tiles, suh, svh in ((wg, kg, part(0), part(3), part(6)), (wd, kd, part(2), part(5), part(8))):
            k, nn = w.shape
            xb = (rng.standard_normal((a.rows, k)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)      # bf16
            x = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            y = (x @ w.astype(np.float64)).astype(np.float32)
            o.write(struct.pack("<4I", k, nn, bits, a.rows) + tiles + suh + svh + xb.tobytes() + y.tobytes())
            print(f"layer {L} expert {e} rank {rank}: {k} x {nn}, {bits}-bit, |y| {np.linalg.norm(y):.4f}")


if __name__ == "__main__":
    main()
