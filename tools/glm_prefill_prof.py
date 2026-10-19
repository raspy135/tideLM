"""Where a GLM prefill's time goes: attention (indexer separately), MLP, the MTP cache fill, the
head, and the post-prefill adaptation, each bracketed with CUDA events (eager prefill, so the
spans are the GPU time of those calls plus any host gaps inside them), and optionally a kernel
table for one chunk.

    GLM_ENV="..." tools/muse_pair_run.sh glm_prefill_prof.py --long-tokens 6000 --kernels 1
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP, mem_available  # noqa: E402
from server.glm_app import GlmTemplate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--kernels", type=int, default=0, help="then profile this many chunks' kernels")
    ap.add_argument("--kernels-after", type=int, default=4, help="... after this many unprofiled chunks")
    ap.add_argument("--reps", type=int, default=2, help="prefills of the prompt; the first warms kernels/JIT")
    ap.add_argument("--warmup", action="store_true",
                    help="capture the decode graphs first (eng.warmup(), as the server does at boot): their "
                         "pools hold memory a prefill's transients then cannot use")
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
            k = key + (" [mtp layer]" if args and isinstance(args[0], int) and args[0] == eng.mtp_L else "")
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            t = time.perf_counter()
            r = f(*args, **kw)
            wall[k] += time.perf_counter() - t
            e1.record()
            pending.append((k, e0, e1))
            calls[k] += 1
            return r
        setattr(obj, name, w)

    wrap(eng, "attention", "attention (incl. indexer)")
    wrap(eng, "_indexer", "  indexer")
    wrap(eng, "mlp", "mlp")
    wrap(eng, "_route", "  route")
    wrap(eng, "_experts_triton", "  fixed/MXFP4 experts (triton)")
    wrap(eng, "embed_ids", "embed")
    wrap(eng, "mtp_forward", "mtp fill")
    wrap(eng, "logits", "head")
    wrap(eng, "adapt", "adapt (post-prefill)")
    wrap(eng.comm, "all_gather", "  all_gather (any)")
    wrap(eng.comm, "all_reduce", "  all_reduce (any)")
    wrap(eng, "_reduce_prefill", "  o_proj reduce")

    if a.warmup:
        eng.warmup()
    totals = []
    for rep in range(a.reps):                         # rep 0 warms kernels/JIT; the rest are measured
        pending.clear(), spans.clear(), calls.clear(), wall.clear()
        eng.reset()
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_stats()
        t0 = time.perf_counter()
        it = eng.stream(prompt, 1, depth=0)
        next(it)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        it.close()
        for key, e0, e1 in pending:
            spans[key] += e0.elapsed_time(e1) / 1e3
        m1 = torch.cuda.memory_stats()
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free")}
        if rep:
            totals.append(total)
        if eng.rank == 0:
            print(f"[pf] rep {rep}: {len(prompt)} tokens in {total:.2f}s = {len(prompt) / total:.1f} tok/s "
                  f"(chunk {eng.chunk}); mlp {spans['mlp']:.2f}s, attention {spans['attention (incl. indexer)']:.2f}s; "
                  f"allocator retries {d['num_alloc_retries']}, cudaMalloc {d['num_device_alloc']}, "
                  f"cudaFree {d['num_device_free']}; reserved {m1.get('reserved_bytes.all.peak', 0) / 1e9:.1f} GB peak, "
                  f"MemAvailable {mem_available() / 1e9:.1f} GB", flush=True)
    if eng.rank == 0 and totals:
        ts = sorted(totals)
        print(f"[pf] measured reps: median {ts[len(ts) // 2]:.2f}s, min {ts[0]:.2f}s, max {ts[-1]:.2f}s", flush=True)
    if eng.rank == 0:
        for k in sorted(spans, key=lambda k: -spans[k]):
            print(f"[pf]   {k:28s} GPU span {spans[k]:7.2f}s  host {wall[k]:7.2f}s  calls {calls[k]}", flush=True)
    if a.kernels:
        from torch.profiler import ProfilerActivity, profile
        eng.reset()
        lead = prompt[:eng.chunk * a.kernels_after]
        part = prompt[len(lead):len(lead) + eng.chunk * a.kernels]
        if lead:
            eng.forward(lead, all_logits=False)            # context, so the chunk takes the sparse path
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            eng.forward(part, all_logits=False)
            torch.cuda.synchronize()
        if eng.rank == 0:
            rows = sorted((e for e in prof.key_averages() if e.device_time_total > 0), key=lambda e: -e.self_device_time_total)
            tot = sum(e.self_device_time_total for e in rows)
            print(f"[kern] {len(part)} tokens after {len(lead)}: {tot / 1e6:.2f}s of kernels", flush=True)
            for e in rows[:35]:
                print(f"[kern]   {e.self_device_time_total / 1e3:9.1f} ms  {e.count:7d}x  {e.key[:110]}", flush=True)


if __name__ == "__main__":
    main()
