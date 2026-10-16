"""Background expert streaming, simulated on a decode routing dump (glm_adapt_test --dump).
A: reactive -- every miss queues a background read (heaviest first, B reads a token); a read lands
   D tokens later in an LRU of C transient slots; a later miss found there is served exactly.
B: pre-gating -- layer N's router run on layer N-d's input; recall of the missed mass.
C: A plus B: misses predicted d layers ahead also queue reads, landing the same token."""
import sys, collections, torch
d = torch.load(sys.argv[1], weights_only=True)
steps, cur, prevL = [], {}, 10**9
for L, ids, w, m, ahead in d:
    if L <= prevL and cur:
        steps.append(cur); cur = {}
    cur[L] = (ids[0].long().tolist(), (w[0].float() / 2.5).tolist(), m[0].tolist(), {k: set(v[0].long().tolist()) for k, v in ahead.items()})
    prevL = L
steps.append(cur)
T = len(steps); layers = sorted(steps[0]); nL = len(layers)
tot_missed = sum(wi for s in steps for L in s for wi, mi in zip(s[L][1], s[L][2]) if mi)
print(f"{T} decode steps x {nL} layers; mass missed {tot_missed / (T * nL):.4f}; misses/token {sum(sum(s[L][2]) for s in steps for L in s) / T:.1f}")

miss_share = sum(sum(s[L][2]) for s in steps for L in s) / (T * nL * 8)

def reactive(C, B, D, pregate=None):
    cache = collections.OrderedDict(); pending = collections.deque(); left = 0.0; reads = 0
    for t, s in enumerate(steps):
        while pending and pending[0][0] <= t:
            _, key = pending.popleft(); cache[key] = 1; cache.move_to_end(key)
            while len(cache) > C: cache.popitem(last=False)
        want = []
        served = set()
        if pregate:
            # Predicted from d layers earlier, read in this token: a predicted pick that turns out
            # missed is served; predictions that are wrong still cost a read when not resident
            # (counted at the observed miss share of picks).
            for L in layers:
                src = L - pregate
                if src in s and L in s[src][3]:
                    act = dict(zip(s[L][0], s[L][2]))
                    for e in s[src][3][L]:
                        if act.get(e):
                            served.add((L, e)); reads += 1
                        elif e not in act:
                            reads += miss_share
        for L in layers:
            ids, w, m, _ = s[L]
            for e, wi, mi in zip(ids, w, m):
                if not mi: continue
                if (L, e) in cache or (L, e) in served:
                    if (L, e) in cache: cache.move_to_end((L, e))
                else:
                    left += wi; want.append((wi, (L, e)))
        want.sort(reverse=True)
        queued = {k for _, k in pending}
        n = 0
        for wi, key in want:
            if n >= B: break
            if key in queued: continue
            pending.append((t + D, key)); queued.add(key); n += 1
        reads += n
    return left / (T * nL), reads / T

print("A reactive: C slots, B reads/token, D tokens latency -> mass left missed, reads/token")
for C in (256, 512, 1024, 2048):
    for B in (20, 40, 80, 10**6):
        r = reactive(C, B, 1)
        print(f"  C={C:5d} B={B if B < 10**6 else 'inf':>4} D=1: {r[0]:.4f} ({r[1]:.1f} reads)")
for D in (2, 4):
    r = reactive(1024, 40, D); print(f"  C= 1024 B=  40 D={D}: {r[0]:.4f} ({r[1]:.1f} reads)")

print("B pre-gating: share of missed picks / missed mass at layer N whose expert is in the top-8 predicted from N-d")
for dd in (1, 2, 4, 8):
    hit_n = hit_w = all_n = all_w = 0
    for s in steps:
        for L in layers:
            src = L - dd
            if src not in s or L not in s[src][3]: continue
            pred = s[src][3][L]
            for e, wi, mi in zip(*s[L][:3]):
                if mi:
                    all_n += 1; all_w += wi
                    if e in pred: hit_n += 1; hit_w += wi
    print(f"  d={dd}: picks {hit_n / max(1, all_n):.3f}, mass {hit_w / max(1e-9, all_w):.3f}")

print("B' all picks (not just misses): top-8 overlap of predicted vs actual")
for dd in (1, 2, 4, 8):
    ov = n = 0
    for s in steps:
        for L in layers:
            src = L - dd
            if src in s and L in s[src][3]:
                ov += len(set(s[L][0]) & s[src][3][L]); n += 8
    print(f"  d={dd}: {ov / max(1, n):.3f}")

print("temporal: share of missed mass whose (L, e) was also picked at step t-k (unmasked)")
for k in (1, 2, 4, 8, 16):
    hw = aw = 0
    for t in range(k, T):
        for L in layers:
            prev = set()
            for j in range(1, k + 1):
                prev |= set(steps[t - j][L][0])
            for e, wi, mi in zip(*steps[t][L][:3]):
                if mi:
                    aw += wi; hw += wi * (e in prev)
    print(f"  within last {k:2d} tokens: {hw / max(1e-9, aw):.3f}")

print("C reactive + pre-gating d layers ahead (pre-gated reads are unbudgeted; reads/token includes them)")
for dd in (1, 2, 4):
    r = reactive(1024, 40, 1, pregate=dd); print(f"  C=1024 B=40 D=1 d={dd}: {r[0]:.4f} ({r[1]:.1f} reads)")
    r = reactive(0, 0, 1, pregate=dd); print(f"  pre-gating alone d={dd}: {r[0]:.4f} ({r[1]:.1f} reads)")
