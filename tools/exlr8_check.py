"""Hold tools/exlr8_ref.py against the published bytes and the publisher's loader.

    uv run --with numpy --with xxhash [--with torch] python tools/exlr8_check.py \
        [--layer 40] [--experts 7,200] [--units 3,12] [--cache DIR] [--oracle PATH/exlr8.py]

Fetches one layer's index file (13 MB, cached) and, by HTTP Range, the chosen units of gate, up and
down at K2, K3 and K4 (0.2-0.4 MB each) - nothing else of the 838 GB. Checks, a unit:
  the index's offset against the layout rule 65536 + ((p * 256 + e) * 16 + u) * B;
  xxhash64(tile bytes, the unit's 128 channel values, the 48 block scales) against the index;
  with --torch: the tiles before the rounding against this repo's EXL3 decode (tools/exl3_ref.py:
    another implementation of the packing, the states and the tile order), and the E4M3 rounding
    against torch's float8_e4m3fn cast;
  with --oracle (kindlingai/exlr8 loader/exlr8.py, torch): T bit for bit and W to fp32 rounding;
  the three widths against each other: each is an independent encode of the same weights, so a
    wrong tile order, state, rotation or side vector leaves them uncorrelated;
  the engine's order (forward_units: one input rotation, block scales, one output rotation) against
  x W through the reconstructed matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exlr8_ref as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--experts", default="7,200")
    ap.add_argument("--units", default="3,12")
    ap.add_argument("--widths", default="2,3,4")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/tidelm-exlr8"))
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--torch", action="store_true")
    a = ap.parse_args()
    import xxhash
    os.makedirs(a.cache, exist_ok=True)
    base = f"experts/exlr8-layer-{a.layer:03d}"
    ipath = os.path.join(a.cache, f"exlr8-layer-{a.layer:03d}.index.safetensors")
    if not os.path.exists(ipath):
        import urllib.request
        urllib.request.urlretrieve(X.REPO + base + ".index.safetensors", ipath + ".part")
        os.replace(ipath + ".part", ipath)
    idx = X.SafeFile(ipath)
    print(f"index: {len(idx.header)} tensors; widths {idx.meta.get('widths')}; codebook {idx.meta.get('codebook')}; "
          f"sharding {idx.meta.get('sharding')}")
    sin, sout = idx.tensor("signs.in"), idx.tensor("signs.out")
    hidden = sin.shape[0]
    oracle = None
    if a.oracle:
        import torch
        spec = importlib.util.spec_from_file_location("exlr8_oracle", a.oracle)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
    # the layer's codebook (its trellis files' tensor names: FORMAT.md), checked for every width below
    lcb = X.SafeFile(X.REPO + f"{base}.k{a.widths.split(',')[0]}.safetensors").trellis("trellis.gate")[1]
    print(f"codebook of the trellis files: {lcb}")
    if a.torch:
        import torch
        import exl3_ref as R
        cb = X.codebook(lcb)
        raw16 = R.codebook(lcb).numpy()
        cast = torch.from_numpy(raw16.astype(np.float32)).clamp(-448, 448).to(torch.float8_e4m3fn)
        same = np.array_equal(cast.float().numpy(), cb) and np.array_equal(cast.view(torch.uint8).numpy(), X.e4m3_bits(cb))
        print(f"codebook: 65,536 states, {len(np.unique(cb))} distinct E4M3 values in [{cb.min()}, {cb.max()}]; "
              f"against torch's float8_e4m3fn cast of exl3_ref's fp16 table: {'identical (values and bytes)' if same else 'DIFFERS'}; "
              f"{int((raw16.astype(np.float32) != cb).sum())} states move in the rounding")
        if not same:
            sys.exit(1)
    rng = np.random.default_rng(0)
    decoded = {}
    x = rng.standard_normal(hidden).astype(np.float32)
    bad = 0
    for k in [int(v) for v in a.widths.split(",")]:
        tf = X.SafeFile(X.REPO + f"{base}.k{k}.safetensors")
        index, check = idx.tensor(f"k{k}.index"), idx.tensor(f"k{k}.checksum")
        side = {p: (idx.tensor(f"k{k}.{p}.su_block"), idx.tensor(f"k{k}.{p}.svh")) for p in ("gate", "up")}
        dsv, dsu = idx.tensor(f"k{k}.down.sv_block"), idx.tensor(f"k{k}.down.suh")
        B = X.unit_bytes(hidden, k)
        units, want = [], np.zeros(hidden, np.float64)
        for e in [int(v) for v in a.experts.split(",")]:
            for u in [int(v) for v in a.units.split(",")]:
                c = slice(128 * u, 128 * u + 128)
                raws, ws = {}, {}
                for pi, p in enumerate(X.PROJS):
                    off, n = (int(v) for v in index[e, pi, u])
                    tname, fcb = tf.trellis(f"trellis.{p}")
                    rule = tf.offset(tname) + (16 * e + u) * B
                    flat = X.DATA0 + ((pi * 256 + e) * 16 + u) * B
                    raw = tf.read_at(off, n)
                    chan, block = (dsu[e, c], dsv[e]) if p == "down" else (side[p][1][e, c], side[p][0][e])
                    got = xxhash.xxh64(raw + chan.tobytes() + block.tobytes(), seed=0).intdigest()
                    ok = n == B and off == rule and got == int(check[e, pi, u])
                    w = X.down_unit(raw, k, chan, sout, block, fcb) if p == "down" else X.gate_up_unit(raw, k, sin, block, chan, fcb)
                    note = ""
                    if oracle is not None:
                        import torch
                        r = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                        if p == "down":
                            t = oracle.tiles_in_major(r, k, hidden, fcb)
                            ow = oracle.reconstruct(t, torch.from_numpy(chan.copy()), torch.from_numpy(sout.astype(np.float32) * np.repeat(block.astype(np.float32), 128)))
                        else:
                            t = oracle.tiles_out_major(r, k, hidden, 128, fcb)
                            ow = oracle.reconstruct(t, torch.from_numpy(sin.astype(np.float32) * np.repeat(block.astype(np.float32), 128)), torch.from_numpy(chan.copy()))
                        mine = X.unit_tiles(raw, k, hidden, p == "down", fcb)
                        t_same = np.array_equal(mine, t.numpy())
                        rel = float(np.abs(w - ow.numpy()).max() / np.abs(ow.numpy()).max())
                        fp8 = np.array_equal(X.e4m3_round(mine), mine)
                        ok = ok and t_same and rel < 1e-5 and fp8
                        note = f"  oracle: T {'identical' if t_same else 'DIFFERS'}, W max rel {rel:.1e}, T is E4M3 {fp8}"
                    if a.torch:
                        import torch
                        t16 = torch.from_numpy(np.frombuffer(raw, np.int16).copy())
                        if p == "down":       # tiles [b][r][c] -> exl3_ref's [k16 = r][n16 = 8b + c]
                            tr = t16.view(hidden // 128, 8, 8, 16 * k).permute(1, 0, 2, 3).reshape(8, hidden // 16, 16 * k)
                        else:
                            tr = t16.view(hidden // 16, 8, 16 * k)
                        mine = X.unit_tiles(raw, k, hidden, p == "down", fcb)
                        t_same = np.array_equal(X.e4m3_round(R.unpack(tr.contiguous(), k, fcb).numpy()), mine)
                        ok = ok and t_same
                        note += f"  tiles against exl3_ref + rounding: {'identical' if t_same else 'DIFFER'}"
                    decoded[(e, u, p, k)] = w
                    bad += not ok
                    print(f"K{k} expert {e:3d} unit {u:2d} {p:4s}: offset {off} ({'rule' if off == rule else 'NOT the rule'}"
                          f"{'' if rule == flat else ', flat formula differs'}), checksum {'ok' if got == int(check[e, pi, u]) else 'MISMATCH'}, "
                          f"rank at TP2 {X.owner(e, u, a.layer, 2)}{note}")
                    raws[p], ws[p] = raw, w
                wt = float(rng.uniform(0.05, 0.3))
                units.append((wt, raws["gate"], raws["up"], raws["down"], (k, k, k), side["gate"][0][e], side["up"][0][e], dsv[e],
                              side["gate"][1][e, c], side["up"][1][e, c], dsu[e, c]))
                g, up = x.astype(np.float64) @ ws["gate"], x.astype(np.float64) @ ws["up"]
                want += wt * ((g / (1 + np.exp(-g)) * up) @ ws["down"])
        got = X.forward_units(x, sin, sout, units, lcb)
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        bad += rel > 1e-4
        print(f"K{k}: the engine's order over {len(units)} units against x W: max rel {rel:.1e}")
    ks = sorted({k for (_, _, _, k) in decoded})
    if len(ks) > 1:
        top = ks[-1]
        for (e, u, p, k), w in sorted(decoded.items()):
            if k == top:
                continue
            ref = decoded[(e, u, p, top)]
            rel = float(np.linalg.norm(w - ref) / np.linalg.norm(ref))
            corr = float(np.corrcoef(w.ravel(), ref.ravel())[0, 1])
            # (the release's index files carry no error tables since 0.6)
            en = f"k{k}.err_down" if p == "down" else f"k{k}.err_{p}"
            err = (float(idx.tensor(en)[e]) if p == "down" else float(idx.tensor(en)[e, u])) if en in idx.header else float("nan")
            ok = corr > 0.8
            bad += not ok
            print(f"expert {e:3d} unit {u:2d} {p:4s}: K{k} against K{top}: |dW| / |W| {rel:.3f}, correlation {corr:.4f}"
                  f"{'' if ok else '  UNCORRELATED'}; the index's held-out output error at K{k}: {err:.4f}")
    print("FAILED" if bad else "all checks passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
