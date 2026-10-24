"""Compare two parity dumps (raw little-endian f32 files with the same names).

    python3 rust/tools/compare_dumps.py REF_DIR RUST_DIR

Per layer: relative L2 error ||rust - ref|| / ||ref|| and the worst element. For the
logits: argmax on each side, top-10 overlap, and the largest absolute difference.
Standard library only (the host has no numpy).
"""
import math
import os
import sys
from array import array


def load(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def main():
    ref, got = sys.argv[1], sys.argv[2]
    layers = sorted(n for n in os.listdir(ref) if n.startswith("h") and n.endswith(".f32"))
    worst = 0.0
    for name in layers:
        a, b = load(os.path.join(ref, name)), load(os.path.join(got, name))
        if len(a) != len(b):
            print(f"{name}: length {len(a)} vs {len(b)}")
            return 1
        num = sum((x - y) ** 2 for x, y in zip(a, b))
        den = sum(x * x for x in a) or 1e-30
        rel = math.sqrt(num / den)
        i = max(range(len(a)), key=lambda k: abs(a[k] - b[k]))
        worst = max(worst, rel)
        print(f"{name[1:3]}  rel_l2 {rel:.2e}  max|d| {abs(a[i] - b[i]):.3e} at {i} (ref {a[i]:.4f})")
    la, lb = load(os.path.join(ref, "logits.f32")), load(os.path.join(got, "logits.f32"))
    top = lambda v: sorted(range(len(v)), key=lambda k: -v[k])[:10]
    ta, tb = top(la), top(lb)
    print(f"logits: argmax ref {ta[0]} rust {tb[0]}; top-10 overlap {len(set(ta) & set(tb))}/10; "
          f"max|d| {max(abs(x - y) for x, y in zip(la, lb)):.3f}")
    print(f"ref top-5  {[(k, round(la[k], 3)) for k in ta[:5]]}")
    print(f"rust top-5 {[(k, round(lb[k], 3)) for k in tb[:5]]}")
    print(f"worst layer rel_l2 {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
