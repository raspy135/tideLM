"""Where a GLM decode round's time goes: every captured graph segment timed with CUDA events (no
added syncs), the host-side phases (MTP drafts, the verify forward, hot reads, broadcasts,
adaptation) with wall clocks, and optionally a kernel table from torch.profiler.

    GLM_ENV="..." tools/muse_pair_run.sh glm_decode_prof.py --prompt long --tokens 400 --kernels 6

GPU spans are per segment kind: embed, attention on an indexer ("full") layer or a shared one,
the MLP of a dense or MoE layer, the head, the one-row MTP graph. "GPU idle" is the round's wall
time not covered by any of them -- host syncs, Python, hot reads, broadcasts, eager calls.
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

SHORT = "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include a docstring and tests."


class Timed:
    """A CUDAGraph stand-in whose replay() is bracketed by CUDA events."""

    def __init__(self, g, key, rec):
        self.g, self.key, self.rec = g, key, rec

    def replay(self):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self.g.replay()
        b.record()
        self.rec.append((self.key, a, b))


def instrument_graphs(eng, rec):
    for key, entry in list(eng.graphs.items()):
        if isinstance(key, tuple) and key[0] == "split":
            segs = entry[0]
            for i, (kind, L, g) in enumerate(segs):
                if isinstance(g, Timed):
                    continue
                k = kind
                if kind == "attn":
                    k = "attn-full" if eng.full(L) else "attn-shared"
                elif kind == "mlp":
                    k = "mlp-dense" if eng.dense(L) else "mlp-moe"
                segs[i] = (kind, L, Timed(g, f"{k} x{key[1]}", rec))
        elif isinstance(key, tuple) and key[0] == "mtp" and not isinstance(entry[0], Timed):
            eng.graphs[key] = (Timed(entry[0], f"mtp-graph x{key[1]}{'' if key[2] else ' fill'}", rec),) + tuple(entry[1:])
        elif isinstance(key, int) and not isinstance(entry[0], Timed):
            eng.graphs[key] = (Timed(entry[0], f"graph x{key}", rec),) + tuple(entry[1:])


def wrap(obj, name, key, wall, cnt, rec=None):
    f = getattr(obj, name)

    def w(*a, **k):
        ev = None
        if rec is not None:
            ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev[0].record()
        t = time.perf_counter()
        r = f(*a, **k)
        wall[key] += time.perf_counter() - t
        cnt[key] += 1
        if ev is not None:
            ev[1].record()
            rows = a[0].numel() if hasattr(a[0], "numel") else len(a[0])
            rec.append((f"{key} x{rows} (span)", ev[0], ev[1]))
        return r
    setattr(obj, name, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "/models"))
    ap.add_argument("--prompt", choices=("short", "long"), default="long")
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--warm", type=int, default=48, help="decode tokens before measuring (captures)")
    ap.add_argument("--depth", type=int, default=int(os.environ.get("GLM_MTP_DEPTH", "2")))
    ap.add_argument("--kernels", type=int, default=0, help="then profile this many rounds' kernels")
    ap.add_argument("--draft-agree", action="store_true",
                    help="(GLM_GRAPHS=0, GLM_DRAFT_HEAD=mxfp4) count drafts whose argmax the main head would change")
    a = ap.parse_args()
    tpl = GlmTemplate(a.model)
    eng = GlmTP(a.model)
    agree = [0, 0]
    if a.draft_agree:
        core = eng._mtp_core

        def both(next_ids, h_prev, logits=True):
            hn, z = core(next_ids, h_prev, logits)
            if logits and eng.draft_head is not None:
                zm = eng.comm.cat_last(eng.head(hn, out_dtype=torch.float32))
                agree[0] += int((zm.argmax(-1) == z.argmax(-1)).sum())
                agree[1] += z.shape[0]
            return hn, z
        eng._mtp_core = both
    if a.prompt == "long":
        docs = [json.loads(l)["text"] for l in open("/app/corpus/heldout_eval.jsonl")]
        body = tpl.tok.decode(tpl.encode("\n\n".join(docs))[:a.long_tokens])
        msg = body + "\n\nSummarize the documents above, section by section, in detail."
    else:
        msg = SHORT
    prompt = tpl.encode(tpl.render([{"role": "user", "content": msg}], None, "low"))

    rec, wall, cnt = [], collections.Counter(), collections.Counter()
    for name, key in (("forward", "forward"), ("mtp_forward", "mtp"), ("adapt", "adapt"),
                      ("bg_step", "bg_step"), ("_read_urgent", "hot read"), ("_put", "hot put")):
        wrap(eng, name, key, wall, cnt, rec if name in ("mtp_forward",) else None)
    wrap(eng.comm, "broadcast_obj", "broadcast", wall, cnt)
    reads = {"main": [], "pool": []}
    if eng.x3pack is not None:
        import threading
        f_read = eng.x3pack.read

        def timed_read(*args, **kw):
            t = time.perf_counter()
            r = f_read(*args, **kw)
            reads["main" if threading.current_thread() is threading.main_thread() else "pool"].append(time.perf_counter() - t)
            return r
        eng.x3pack.read = timed_read
    if getattr(eng, "x3", None) is not None:
        wrap(eng.x3, "load_slot", "  arena load_slot (any)", wall, cnt)

    eng.spec_stats = {"rounds": 0, "accepted": 0}
    eng.spec_cal = []
    it = eng.stream(prompt, a.tokens + a.warm, depth=a.depth)
    t0 = time.perf_counter()
    first = next(it)
    ttft = time.perf_counter() - t0
    out, n = [first], 0
    for tok in it:
        out.append(tok)
        n += 1
        if n == a.warm:
            torch.cuda.synchronize()
            instrument_graphs(eng, rec)
            rec.clear(), wall.clear(), cnt.clear(), reads["main"].clear(), reads["pool"].clear()
            r0, acc0 = eng.spec_stats["rounds"], eng.spec_stats["accepted"]
            dr0 = eng.spec_stats.get("drafted", 0)
            g0 = set(map(str, eng.graphs))
            eng.spec_cal.clear()
            h0 = dict(eng.hot_stats)
            t1 = time.perf_counter()
    torch.cuda.synchronize()
    total = time.perf_counter() - t1
    toks = n - a.warm
    rounds = eng.spec_stats["rounds"] - r0
    gpu = collections.defaultdict(float)
    calls = collections.Counter()
    for key, ev0, ev1 in rec:
        gpu[key] += ev0.elapsed_time(ev1) / 1e3
        calls[key] += 1
    if eng.rank == 0:
        print(f"[prof] prompt {len(prompt)} tok, ttft {ttft:.1f}s; measured {toks} tok in {total:.2f}s = "
              f"{toks / total:.2f} tok/s, {rounds} rounds ({1e3 * total / max(1, rounds):.1f} ms a round), "
              f"accepted/round {(eng.spec_stats['accepted'] - acc0) / max(1, rounds):.2f}, "
              f"drafted/round {(eng.spec_stats.get('drafted', 0) - dr0) / max(1, rounds):.2f}", flush=True)
        from engine.muse.tp import mem_available
        late = sorted(set(map(str, eng.graphs)) - g0)
        print(f"[prof] MemAvailable {mem_available() / 1e9:.2f} GB; {len(eng.graphs)} graphs"
              f"{'; CAPTURED WHILE MEASURING: ' + ', '.join(late) if late else ''}", flush=True)
        if eng.spec_cal:
            edges = (0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)
            print("[prof] draft calibration: confidence product bucket -> accepted share (count)", flush=True)
            for lo, hi in zip(edges, edges[1:]):
                row = [(sum(1 for k, c, ok in eng.spec_cal if k == i and lo <= c < hi and ok),
                        sum(1 for k, c, ok in eng.spec_cal if k == i and lo <= c < hi)) for i in range(a.depth)]
                print(f"[prof]   [{lo:.1f}, {hi:.1f}) " + "  ".join(
                    f"d{i + 1} {ok / n:.2f} ({n})" if n else f"d{i + 1}    -     " for i, (ok, n) in enumerate(row)), flush=True)
        busy = sum(v for k, v in gpu.items() if "(span)" not in k)
        print(f"[prof] GPU spans, ms a round (calls a round); graph segments cover {100 * busy / total:.1f}% of wall", flush=True)
        for k, v in sorted(gpu.items(), key=lambda x: -x[1]):
            print(f"[prof]   {k:28s} {1e3 * v / rounds:8.2f} ms  ({calls[k] / rounds:.1f})", flush=True)
        print("[prof] host wall, ms a round (calls a round) -- nested: forward includes hot read/put", flush=True)
        for k, v in sorted(wall.items(), key=lambda x: -x[1]):
            print(f"[prof]   {k:28s} {1e3 * v / rounds:8.2f} ms  ({cnt[k] / rounds:.1f})", flush=True)
        hs = {k: eng.hot_stats[k] - h0.get(k, 0) for k in ("reads", "layers") if k in eng.hot_stats}
        print(f"[prof] hot reads a round: {hs.get('reads', 0) / rounds:.2f} in {hs.get('layers', 0) / rounds:.2f} layers", flush=True)
        for k, v in reads.items():
            if v:
                v = sorted(v)
                q = lambda f: v[min(len(v) - 1, int(f * len(v)))] * 1e3
                print(f"[prof] pack reads on {k} thread: {len(v)}, p10 {q(.1):.2f} p50 {q(.5):.2f} p90 {q(.9):.2f} "
                      f"p99 {q(.99):.2f} max {v[-1] * 1e3:.1f} ms", flush=True)
        print("[prof] sample:", repr(tpl.decode(out)[:300]), flush=True)
        if agree[1]:
            print(f"[prof] draft argmax, 4-bit draft head vs main head: {agree[0]} of {agree[1]} agree "
                  f"({100 * agree[0] / agree[1]:.2f}%)", flush=True)

    if a.kernels:
        from torch.profiler import ProfilerActivity, profile
        it = eng.stream(prompt, 10 ** 6, depth=a.depth)
        for _ in range(16):
            next(it)
        r0 = eng.spec_stats["rounds"]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            while eng.spec_stats["rounds"] - r0 < a.kernels:
                next(it)
            torch.cuda.synchronize()
        if eng.rank == 0:
            rounds = eng.spec_stats["rounds"] - r0
            ka = prof.key_averages()
            rows = sorted((e for e in ka if e.device_time_total > 0), key=lambda e: -e.self_device_time_total)
            tot = sum(e.self_device_time_total for e in rows)
            print(f"[kern] {rounds} rounds, {tot / 1e3 / rounds:.1f} ms of kernels a round", flush=True)
            for e in rows[:40]:
                print(f"[kern]   {e.self_device_time_total / 1e3 / rounds:8.2f} ms  {e.count / rounds:7.1f}x  {e.key[:110]}", flush=True)
        it.close()


if __name__ == "__main__":
    main()
