"""Where a GLM prefill's time goes: attention (indexer separately), MLP, the MTP cache fill, the
head, and the post-prefill adaptation, each bracketed with CUDA events (eager prefill, so the
spans are the GPU time of those calls plus any host gaps inside them), and optionally a kernel
table for one chunk.

    MUSE_ENV="..." tools/muse_pair_run.sh glm_prefill_prof.py --long-tokens 6000 --kernels 1
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_MODEL", "/models"))
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--kernels", type=int, default=0, help="then profile this many chunks' kernels")
    a = ap.parse_args()
    tpl = GlmTemplate(a.model)
    eng = GlmTP(a.model)
    docs = [json.loads(l)["text"] for l in open("/app/corpus/heldout_eval.jsonl")]
    body = tpl.tok.decode(tpl.encode("\n\n".join(docs))[:a.long_tokens])
    prompt = tpl.encode(tpl.render([{"role": "user", "content": body + "\n\nSummarize the documents above."}],
                                   None, "low"))

    spans, calls, wall = collections.defaultdict(float), collections.Counter(), collections.Counter()
    pending = []

    def wrap(obj, name, key, rows_key=False):
        f = getattr(obj, name)

        def w(*args, **kw):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            t = time.perf_counter()
            r = f(*args, **kw)
            wall[key] += time.perf_counter() - t
            e1.record()
            pending.append((key, e0, e1))
            calls[key] += 1
            return r
        setattr(obj, name, w)

    wrap(eng, "attention", "attention (incl. indexer)")
    wrap(eng, "_indexer", "  indexer")
    wrap(eng, "mlp", "mlp")
    wrap(eng, "_route", "  route")
    wrap(eng, "mtp_forward", "mtp fill")
    wrap(eng, "logits", "head")
    wrap(eng, "adapt", "adapt (post-prefill)")
    wrap(eng.comm, "all_gather", "  all_gather (any)")

    for rep in range(2):                              # rep 0 warms kernels/JIT; rep 1 is measured
        pending.clear(), spans.clear(), calls.clear(), wall.clear()
        eng.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        it = eng.stream(prompt, 1, depth=0)
        next(it)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        it.close()
        for key, e0, e1 in pending:
            spans[key] += e0.elapsed_time(e1) / 1e3
        if eng.rank == 0:
            print(f"[pf] rep {rep}: {len(prompt)} tokens in {total:.2f}s = {len(prompt) / total:.1f} tok/s "
                  f"(chunk {eng.chunk})", flush=True)
    if eng.rank == 0:
        for k in sorted(spans, key=lambda k: -spans[k]):
            print(f"[pf]   {k:28s} GPU span {spans[k]:7.2f}s  host {wall[k]:7.2f}s  calls {calls[k]}", flush=True)
    if a.kernels:
        from torch.profiler import ProfilerActivity, profile
        eng.reset()
        part = prompt[:eng.chunk * a.kernels]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            eng.forward(part, all_logits=False)
            torch.cuda.synchronize()
        if eng.rank == 0:
            rows = sorted((e for e in prof.key_averages() if e.device_time_total > 0), key=lambda e: -e.self_device_time_total)
            tot = sum(e.self_device_time_total for e in rows)
            print(f"[kern] {len(part)} tokens: {tot / 1e6:.2f}s of kernels", flush=True)
            for e in rows[:35]:
                print(f"[kern]   {e.self_device_time_total / 1e3:9.1f} ms  {e.count:7d}x  {e.key[:110]}", flush=True)


if __name__ == "__main__":
    main()
