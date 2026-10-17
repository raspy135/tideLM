"""MUSE_PREFIX_DISK on the pair. Hard check: rows restored from disk are the bytes the prefill wrote
(every cache tensor, both ranks; the MTP rows its tokens determine; the hidden). Softer check: a
continuation decodes the same tokens whether its prefix was still in memory or came from disk, at
the same boundary -- equal unless a row outside the restore (another prompt's computation of a
shared system prefix) differs in a low bit, so it is reported, not asserted.

  MUSE_ENV="MUSE_PREFIX_DISK=1 MUSE_PREFIX_DISK_DIR=/app/results/glm-prefix-test MUSE_ADAPT=0 \\
            MUSE_HOT=0 MUSE_BG=0 ..." tools/muse_pair_run.sh glm_prefix_test.py --phase save
  ... the same with --phase load: a fresh pair restores the saved prefix and checks its bytes.

Residency must stay fixed (no adaptation, hot reads or background swaps) for the decode check.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/app")

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402

N = 24


def digest(toks):
    return hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:12]


def rows(eng, n, mtp_upto):
    """Host copies of every cache tensor's rows for positions [0, n) (MTP rows below mtp_upto)."""
    ps, out = eng.pstore, {}
    for name, t, mode in eng._kv_layout():
        hi = mtp_upto if mode == "mtp" else ps._rows(mode, n)
        out[name] = t[:hi].cpu().clone()
    return out


def compare(eng, snap, n, mtp_upto, lo=0):
    """Names whose rows for positions [lo, n) differ from the snapshot."""
    ps, cur, bad = eng.pstore, rows(eng, n, mtp_upto), []
    for name, t, mode in eng._kv_layout():
        a = ps._rows(mode, lo) if mode != "mtp" else max(0, lo - 1)
        if not torch.equal(cur[name][a:], snap[name][a:cur[name].shape[0]]):
            bad.append(name)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--phase", choices=("save", "load"), required=True)
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--dir", default="/app/results/glm-prefix-test")
    a = ap.parse_args()
    tpl = GlmTemplate(a.model)
    eng = GlmTP(a.model)
    assert eng.pstore is not None, "MUSE_PREFIX_DISK=1"
    docs = [json.loads(l)["text"] for l in open("/app/corpus/heldout_eval.jsonl")]
    body = tpl.encode("\n\n".join(docs))
    p1 = tpl.encode(tpl.render([{"role": "user", "content": tpl.tok.decode(body[:a.long_tokens])
                                 + "\n\nSummarize the documents above."}], None, "low"))
    p2 = tpl.encode(tpl.render([{"role": "user", "content": tpl.tok.decode(body[-3000:])
                                 + "\n\nWhat is the main topic?"}], None, "low"))
    L1 = len(p1)
    snap_path = os.path.join(a.dir, f"snap-rank{eng.rank}.pt")
    res_path = os.path.join(a.dir, "result.json")
    checks = {}

    def gen(prompt, n, label):
        t = time.perf_counter()
        toks = list(eng.stream(prompt, n, temperature=0.0))
        eng.log(f"[ptest] {label}: prompt {len(prompt)}, reused {eng.last_reuse}, from disk {eng.last_disk}, "
                f"{len(toks)} tokens {digest(toks)} in {time.perf_counter() - t:.1f}s")
        return toks

    def cont(first):
        """A continuation of p1 whose first token is not `first`: memory reuse then ends at L1 too."""
        for text in ("Zebra crossings aside, the documents cover", "\n\nIn short, the documents"):
            c = tpl.encode(text)
            if c[0] != first:
                return c
        raise RuntimeError("no continuation")

    if a.phase == "save":
        outA = gen(p1, 16, "A  p1 fresh (saved)")
        eng.pstore.join()
        snap = rows(eng, L1, L1 - 1)
        snap["h"] = eng._h_parts(L1 - 1)[0][1].cpu().clone()
        torch.save(snap, snap_path)
        p1x = p1 + cont(outA[0])
        outB = gen(p1x, N, "B  p1x, p1 in memory")
        gen(p2, 8, "C  p2")
        outD = gen(p1x, N, "D  p1x, p1 from disk")
        bad = compare(eng, snap, L1, L1 - 1, lo=eng_restored_from(eng))
        checks["D: restored rows are the saved bytes"] = not bad
        checks["D: hidden at L1 - 1 restored"] = torch.equal(eng._h_parts(L1 - 1)[0][1].cpu(), snap["h"])
        checks["D == B (soft)"] = outD == outB
        if bad:
            eng.log(f"[ptest] differing: {bad[:8]}")
        # Partial: a prompt that leaves p1 500 tokens before its end, after another prompt.
        gen(p2, 8, "C2 p2")
        p1y = p1[:L1 - 500] + p2[100:300]
        gen(p1y, N, "E  p1y, part of p1 from disk")
        n = L1 - 500 - 1                        # rows below the restart position stay restored
        bad = compare(eng, snap, n, n - 1, lo=eng_restored_from(eng))
        checks["E: partially restored rows are the saved bytes"] = not bad
        if eng.rank == 0:
            json.dump({"p1x": p1x, "D": outD}, open(res_path, "w"))
    else:
        prev = json.load(open(res_path)) if eng.rank == 0 else None
        p1x = eng.comm.broadcast_obj(prev["p1x"] if prev else None)
        outG = gen(p1x, N, "G  p1x after a restart, p1 from disk")
        snap = torch.load(snap_path)
        bad = compare(eng, snap, L1, L1 - 1)
        checks["G: restored rows are the saved bytes"] = not bad
        checks["G == D (soft)"] = prev is None or outG == prev["D"]
    eng.pstore.join()
    allc = eng.comm.gather_objects(checks)
    for k in checks:
        eng.log(f"[ptest] {'PASS' if all(c[k] for c in allc) else 'FAIL'} {k} "
                f"({', '.join('ok' if c[k] else 'differs' for c in allc)} by rank)")


def eng_restored_from(eng):
    """The first position the last request restored from disk (the rest was in memory)."""
    return eng.last_reuse - eng.last_disk if eng.last_disk else 0


if __name__ == "__main__":
    main()
