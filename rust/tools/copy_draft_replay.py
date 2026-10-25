"""copy_draft_replay.py RUNS.json: what a prompt-lookup (copy) drafter in front of DSpark would
do to greedy runs the Rust server logged (each run: x_debug's prompt_ids, token_ids, depths and
accepted, as {name: stats}). Greedy, so a copied token is accepted exactly when it equals the
token the model produced. Each step either copies (the longest earlier occurrence of the
context's last MIN_MATCH+ tokens, up to L tokens, verified at width L + 1) or is the logged DSpark
step from that position (its block's remainder). Steps are priced with the verify cost by width
tidelm-server --bench-width measured on the pair (2,585 tokens of context) and the DSpark pass a
copy step skips; the DSpark-only price reproduces the logged runs' tok/s within ~5%."""
import json, sys, importlib.util
cost = {1: 43.14, 2: 50.96, 4: 65.81, 6: 84.32, 8: 96.26, 12: 132.53, 16: 149.06, 17: 153.06, 24: 176.49, 32: 205.75, 33: 217.39}
DRAFT = 4.88
ks = sorted(cost)
def verify(w):
    if w in cost: return cost[w]
    lo = max(k for k in ks if k <= w); hi = min(k for k in ks if k >= w)
    return cost[lo] + (cost[hi] - cost[lo]) * (w - lo) / (hi - lo)
runs = json.load(open(sys.argv[1]))
def lookup(ctx, p_end, min_match, L):
    """Longest match of ctx[:p_end]'s tail in ctx[:p_end - 1] by n-gram index (n = min_match),
    extended backwards; returns the continuation (<= L tokens) after the best earlier match."""
    n = min_match
    if p_end < n + 1:
        return []
    key = tuple(ctx[p_end - n:p_end])
    best, best_len = None, 0
    for j in idx.get(key, ()):
        if j >= p_end:   # occurrences ending at j (exclusive), strictly before the tail itself
            continue
        m = n
        while m < 64 and j - m - 1 >= 0 and ctx[j - m - 1] == ctx[p_end - m - 1]:
            m += 1
        if m > best_len or (m == best_len and j > best):
            best, best_len = j, m
    if best is None:
        return []
    return ctx[best:min(best + L, p_end)]

for name, r in runs.items():
    prompt, out, acc, dep = r["prompt_ids"], r["token_ids"], r["accepted"], r["depths"]
    starts, s = [], 1
    for a in acc:
        starts.append(s); s += a + 1
    base_ms = sum(verify(d + 1) + DRAFT for d in dep)
    n = len(out)
    row = [f"{name:11s} DSpark {n / base_ms * 1e3:5.1f} tok/s"]
    for (mm, L, mc) in [(6, 16, 6), (6, 32, 6)]:
        globals()["idx"] = {}
        ctx = prompt + out; P = len(prompt); done = [0]
        def add(upto):
            for e in range(done[0], upto + 1):
                if e >= mm: idx.setdefault(tuple(ctx[e - mm:e]), []).append(e)
            done[0] = upto + 1
        def dspark_at(p):
            lo, hi = 0, len(starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if starts[mid] <= p: lo = mid
                else: hi = mid - 1
            return max(1, starts[lo] + acc[lo] + 1 - p), dep[lo] + 1
        p, ms = 1, 0.0
        while p < n:
            add(P + p - 1)
            prop = lookup(ctx, P + p, mm, L)
            if len(prop) >= mc:
                k = 0
                while k < len(prop) and p + k < n and prop[k] == out[p + k]: k += 1
                emit, w, extra = min(k + 1, n - p), len(prop) + 1, 0.0
            else:
                emit, w = dspark_at(p); emit = min(emit, n - p); extra = DRAFT
            ms += verify(w) + extra
            p += emit
        row.append(f"copies <= {L}: {n / ms * 1e3:5.1f} tok/s ({base_ms / ms:.2f}x)")
    print(",  ".join(row))
