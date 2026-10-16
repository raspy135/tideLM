"""Teacher-forced NLL of Muse Spark on the pair, per corpus category, for one engine setting.

The quality yardstick for every lossy choice (resident budget and miss policy, FP8 dense, the
MXFP4 conversion of layers 0/61): run the same documents under each setting and compare mean
NLL (nats/token) by category. MUSE_MISS=stream is the exact reference (every expert the router
picks is used). With MUSE_MISS=substitute the run also reports the share of unmasked picks (and
router mass) that had to be substituted.

    MUSE_MODEL_DIR=... MUSE_ENV="MUSE_MISS=substitute MUSE_DENSE=fp8" \\
        tools/muse_pair_run.sh muse_nll.py --corpus /app/corpus/heldout_eval.jsonl --tokens 1024
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.muse.tp import MuseTP  # noqa: E402


class _Bytes:
    """Byte-level stand-in for synthetic checkpoints that ship no tokenizer."""
    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Enc(list(text.encode("utf-8")))


def load_tokenizer(model_dir):
    path = Path(model_dir) / "tokenizer.json"
    if not path.exists():
        return _Bytes()
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--corpus", default="/app/corpus/heldout_eval.jsonl")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tok = load_tokenizer(args.model)
    eng = MuseTP(args.model, max_seq=max(args.tokens + 16, 2048), chunk=1024)
    if eng.miss == "substitute":
        eng.miss_stats = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
    sums, counts = defaultdict(float), defaultdict(int)
    per_doc = []
    t0 = time.perf_counter()
    docs = [json.loads(l) for l in open(args.corpus)][:args.docs]
    for i, d in enumerate(docs):
        ids = ([eng.cfg.bos] + tok.encode(d["text"], add_special_tokens=False).ids)[:args.tokens]
        if len(ids) < 16:
            continue
        eng.reset()
        logits = eng.forward(ids[:-1])
        if eng.rank == 0:
            target = torch.tensor(ids[1:], device=logits.device)
            nll = F.cross_entropy(logits.float(), target, reduction="sum").item()
            cat = d.get("category", "general")
            sums[cat] += nll
            counts[cat] += len(ids) - 1
            per_doc.append({"id": d.get("id", i), "category": cat, "tokens": len(ids) - 1,
                            "nll": nll / (len(ids) - 1)})
            if (i + 1) % 10 == 0:
                print(f"[nll] {i + 1}/{len(docs)} docs, {time.perf_counter() - t0:.0f}s", flush=True)
        del logits
    if eng.rank != 0:
        return
    report = {
        "setting": {k: v for k, v in os.environ.items() if k.startswith("MUSE_")},
        "slots": eng.slots_resident, "dense": eng.dense_fmt, "miss": eng.miss,
        "nll": {c: round(sums[c] / counts[c], 4) for c in sums},
        "tokens": dict(counts), "seconds": round(time.perf_counter() - t0, 1),
    }
    if eng.miss_stats:
        m = eng.miss_stats
        report["substituted_picks"] = round(m["missed"] / max(1, m["picks"]), 4)
        report["substituted_mass"] = round(m["mass_missed"] / max(1e-9, m["mass"]), 4)
    print(json.dumps(report, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({**report, "per_doc": per_doc}))


if __name__ == "__main__":
    main()
