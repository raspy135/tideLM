"""Expert read latency inside a booted engine: idle, then interleaved with decode rounds.

    GLM_ENV="..." tools/muse_pair_run.sh glm_read_probe.py
"""
import random
import sys
import time

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.tp import GlmTP  # noqa: E402


def main():
    eng = GlmTP("/models")
    rnd = random.Random(eng.rank)
    lay = eng.swap_layers
    keys = [(L, e) for L in lay for e in range(eng.cfg.experts) if int(eng.lut[L, e]) < 0]
    rnd.shuffle(keys)

    def timed(n, urgent=True):
        ts = []
        for L, e in keys[:n]:
            t = time.perf_counter()
            eng._expert_parts(L, e, urgent)
            ts.append(time.perf_counter() - t)
        del keys[:n]
        ts.sort()
        return f"median {ts[len(ts) // 2] * 1e3:.2f} ms, p90 {ts[int(len(ts) * .9)] * 1e3:.2f} ms over {n}"

    torch.cuda.synchronize()
    print(f"[read r{eng.rank}] idle engine: {timed(40)}", flush=True)
    eng.comm.broadcast_obj(None)
    it = eng.stream(list(range(1000, 1400)), 120, depth=0)
    for _ in range(40):
        next(it)
    torch.cuda.synchronize()
    print(f"[read r{eng.rank}] after decode (graphs warm): {timed(40)}", flush=True)
    it.close()


if __name__ == "__main__":
    main()
