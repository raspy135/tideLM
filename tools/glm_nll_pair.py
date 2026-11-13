"""Paired comparison of two --bench-nll-out dumps of the same corpus ("category nll top1" a line).

    python3 tools/glm_nll_pair.py BASE.nll ARM.nll [ARM2.nll ...]

Per arm against BASE: the paired mean NLL difference with its standard error, per category, top-1,
and - because one hidden-state collapse (a run of ~10-nat tokens until the text ends) moves a
10K-token mean by several hundredths - the median of the paired 250-token windows' differences
and the windows where the arm is worse by more than 1 nat a token (a collapse) or better by more.
"""

import math
import sys


def load(path):
    rows = []
    for line in open(path):
        c, nll, top = line.split()
        rows.append((c, float(nll), int(top)))
    return rows


def main():
    base = load(sys.argv[1])
    print(f"{'arm':<28} {'tokens':>6} {'NLL':>7} {'paired diff':>17} {'coding':>8} {'general':>8} {'top-1':>7} "
          f"{'median 250-window':>17} {'blown windows':>13}")
    print(f"{sys.argv[1].split('/')[-1]:<28} {len(base):>6} {sum(r[1] for r in base) / len(base):>7.4f} {'':>17} {'':>8} {'':>8} "
          f"{100 * sum(r[2] for r in base) / len(base):>6.2f}%")
    for path in sys.argv[2:]:
        arm = load(path)
        n = min(len(arm), len(base))
        d = [arm[i][1] - base[i][1] for i in range(n)]
        m = sum(d) / n
        sd = math.sqrt(max(sum(x * x for x in d) / n - m * m, 0.0))
        cat = {}
        for i in range(n):
            cat.setdefault(base[i][0], []).append(d[i])
        w = [sum(d[j:j + 250]) / len(d[j:j + 250]) for j in range(0, n, 250)]
        ws = sorted(w)
        med = ws[len(ws) // 2]
        worse = sum(1 for x in w if x > 1.0)
        better = sum(1 for x in w if x < -1.0)
        print(f"{path.split('/')[-1]:<28} {n:>6} {sum(r[1] for r in arm[:n]) / n:>7.4f} {m:>+9.4f} +- {sd / math.sqrt(n):.4f} "
              f"{sum(cat.get('coding', [0])) / max(len(cat.get('coding', [])), 1):>+8.4f} "
              f"{sum(cat.get('general', [0])) / max(len(cat.get('general', [])), 1):>+8.4f} "
              f"{100 * sum(r[2] for r in arm[:n]) / n:>6.2f}% {med:>+17.4f} {worse:>6} / {better:<6}")


if __name__ == "__main__":
    main()
