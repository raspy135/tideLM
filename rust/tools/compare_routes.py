"""Compare router choices: the reference's meta.json routes vs the Rust idxLL.i32 dumps.

    python3 rust/tools/compare_routes.py REF_DIR RUST_DIR

Prints every (layer, row) whose selected expert SET differs, with the experts that
only one side chose. Standard library only.
"""
import json
import os
import sys
from array import array


def main():
    ref, got = sys.argv[1], sys.argv[2]
    routes = json.load(open(os.path.join(ref, "meta.json")))["routes"]
    flips = 0
    for L in sorted(routes, key=int):
        want = routes[L]
        a = array("i")
        with open(os.path.join(got, f"idx{int(L):02d}.i32"), "rb") as f:
            a.frombytes(f.read())
        k = len(want[0])
        for t, row in enumerate(want):
            mine = set(a[t * k:(t + 1) * k])
            if mine != set(row):
                flips += 1
                print(f"layer {int(L):2d} row {t}: ref-only {sorted(set(row) - mine)} rust-only {sorted(mine - set(row))}")
    print(f"{flips} (layer, row) routing differences")


if __name__ == "__main__":
    sys.exit(main())
