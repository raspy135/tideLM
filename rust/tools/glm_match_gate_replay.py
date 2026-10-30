"""glm_match_gate_replay.py ROUNDS.jsonl: would a short match in the context make GLM's MTP draft
gate better? ROUNDS is the GLM server's TIDELM_GLM_ROUND_LOG from a boot that drafts 7 every round
(GLM_MTP_CAL=0 GLM_MTP_DYN=0, greedy requests, GLM_COPY=0 GLM_LOOKUP=0), so each draft's acceptance
is known whatever depth a gate would have stopped at. Tokens a round with depth D = 1 + min(D,
accepted of 7); cost = the verify by rows tidelm-server --bench-width measured + 3.2 ms an MTP pass.
Three questions, in order: does agreement with a 3-6-token match predict acceptance, and does a
calibrated gate that knows it draft better (tables fitted leave-one-request-out); does raising an
agreeing draft's probability by 0.2-0.5 by match length do better; and at MTP's rejections, did the
match hold the right token (2026-10-27)."""
import json, sys, collections
VER = {1: 89, 2: 115, 3: 132.5, 4: 150, 5: 163, 6: 176, 7: 188, 8: 200}   # bench-width, 2.5K, ms
MTP = 3.2
EDGES = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]

def bucket(c):
    return next(i for i in range(8) if c < EDGES[i + 1])

reqs = []
for line in open(sys.argv[1]):
    j = json.loads(line)
    if "prompt" in j:
        reqs.append({"prompt": j["prompt"], "rounds": [], "tokens": None})
    elif "tokens" in j:
        reqs[-1]["tokens"] = j["tokens"]
    else:
        reqs[-1]["rounds"].append(j)
reqs = reqs[3:]  # the warm-up's synthetic prompts

def match(full, p, n):
    """the continuation after the earlier occurrence of full[p-n+1..p] that matches furthest back
    (ties: the latest), within what is known at p"""
    if p + 1 < n + 1:
        return [], 0
    tail = full[p - n + 1:p + 1]
    best, bm = None, 0
    for e in range(n - 1, p):
        if full[e - n + 1:e + 1] != tail:
            continue
        m = n
        while m < 64 and e - m >= 0 and full[e - m] == full[p - m]:
            m += 1
        if m > bm or (m == bm and e > best):
            best, bm = e, m
    if best is None:
        return [], 0
    return full[best + 1:min(best + 8, p + 1)], bm

rows = []  # per round: req, k-wise (accepted?, conf, agree[n])
for ri, r in enumerate(reqs):
    full = r["prompt"] + r["tokens"]
    for rd in r["rounds"]:
        p, d, t, pr = rd["pos"], rd["drafts"], rd["target"], rd["probs"]
        assert full[p] == rd["anchor"], (ri, p)
        nacc = next((k for k in range(len(d)) if d[k] != t[k]), len(d))
        conf, confs = 1.0, []
        for x in pr:
            conf *= x
            confs.append(conf)
        ag = {}
        for n in (3, 4, 5, 6):
            c, m = match(full, p, n)
            ag[n] = [len(c) > k and d[:k + 1] == c[:k + 1] for k in range(len(d))]
            if n == 6:
                long_ok = len(c) >= 6 and m >= 6
        rows.append({"req": ri, "nacc": nacc, "nd": len(d), "conf": confs, "agree": ag, "long": long_ok})

print(f"{len(reqs)} requests, {len(rows)} rounds, {sum(r['nacc'] for r in rows) / len(rows):.2f} of 7 drafts accepted a round")
for n in (3, 4, 5, 6):
    cnt = collections.Counter()
    for r in rows:
        for k in range(r["nd"]):
            a = r["agree"][n][k]
            cnt[(a, k < r["nacc"])] += 1
    pa = cnt[(True, True)] / max(1, cnt[(True, True)] + cnt[(True, False)])
    pn = cnt[(False, True)] / max(1, cnt[(False, True)] + cnt[(False, False)])
    print(f"match {n}: drafts agreeing {cnt[(True, True)] + cnt[(True, False)]:5d}, accepted {pa:.1%}; not agreeing {pn:.1%}")

def table(rs, n):
    t = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        for k in range(r["nd"]):
            key = (k, bucket(r["conf"][k]), r["agree"][n][k] if n else False)
            t[key][1] += 1
            t[key][0] += k < r["nacc"]
    return t

def prob(t, k, b, a):
    acc, num = t[(k, b, a)]
    ra = [v for key, v in t.items() if key[0] == k and key[2] == a]
    tot, seen = sum(v[0] for v in ra), sum(v[1] for v in ra)
    pooled = tot / seen if seen else 0.5
    return (acc + 4 * pooled) / (num + 4)

def simulate(rs, n, tau, tables):
    tok = ms = 0.0
    for r in rs:
        t = tables[r["req"]]
        D = 0
        while D < r["nd"] and prob(t, D, bucket(r["conf"][D]), r["agree"][n][D] if n else False) >= tau:
            D += 1
        tok += 1 + min(D, r["nacc"])
        ms += VER[D + 1] + MTP * (1 + min(D, 6))
    return 1000 * tok / ms

def loo(n):
    return {ri: table([r for r in rows if r["req"] != ri], n) for ri in range(len(reqs))}

taus = [0.2, 0.25, 0.3, 0.36, 0.42, 0.5]
gates = {"calibrated (stated prob)": loo(None), **{f"+ agrees with a {n}-token match": loo(n) for n in (3, 4, 6)}}
print("\nmodelled tok/s, tables fitted leave-one-request-out, by gate threshold:")
print(f"{'gate':34s}" + "".join(f"{t:>8}" for t in taus))
for name, tabs in gates.items():
    n = None if name.startswith("cal") else int(name.split("a ")[1].split("-")[0])
    print(f"{name:34s}" + "".join(f"{simulate(rows, n, t, tabs):8.2f}" for t in taus))
ms7 = sum(VER[8] + MTP * 7 for r in rows)
print(f"{'always 7':34s}{1000 * sum(1 + r['nacc'] for r in rows) / ms7:8.2f}")
print(f"\nper request at 0.36 (calibrated -> + 4-token agreement), rounds where a long copy (>= 6 matched, >= 6 to copy) was available:")
for ri, r in enumerate(reqs):
    rs = [x for x in rows if x["req"] == ri]
    a, b = simulate(rs, None, 0.36, gates["calibrated (stated prob)"]), simulate(rs, 4, 0.36, gates["+ agrees with a 4-token match"])
    print(f"  request {ri}: {len(rs):4d} rounds, {sum(x['nacc'] for x in rs) / len(rs):.2f} accepted of 7, {a:6.2f} -> {b:6.2f} tok/s "
          f"({(b / a - 1) * 100:+.1f}%), long copy available {sum(x['long'] for x in rs) / len(rs):.0%}")


print("\n--- a confidence bias for agreeing drafts: 0.2 / 0.3 / 0.4 / 0.5 for a match of 3 / 4 / 5 / 6+")

rows = []
for ri, r in enumerate(reqs):
    full = r["prompt"] + r["tokens"]
    for rd in r["rounds"]:
        p, d, t, pr = rd["pos"], rd["drafts"], rd["target"], rd["probs"]
        nacc = next((k for k in range(len(d)) if d[k] != t[k]), len(d))
        c, m = match(full, p, 3)
        agree = [len(c) > k and d[:k + 1] == c[:k + 1] for k in range(len(d))]
        b = 0.2 + 0.1 * (min(m, 6) - 3) if m >= 3 else 0.0
        out = {}
        for name, bias in (("plain", 0.0), ("biased", b)):
            conf, confs = 1.0, []
            for k, x in enumerate(pr):
                conf *= min(1.0, x + bias) if agree[k] else x
                confs.append(conf)
            out[name] = confs
        rows.append({"req": ri, "nacc": nacc, "nd": len(d), "conf": out, "agree": agree, "p": pr})

ag = [r["p"][k] for r in rows for k in range(r["nd"]) if r["agree"][k]]
na = [r["p"][k] for r in rows for k in range(r["nd"]) if not r["agree"][k] and k < 3]
print(f"agreeing drafts: {len(ag)}, their own MTP probability: median {sorted(ag)[len(ag) // 2]:.2f}, "
      f">= 0.9 in {sum(x >= 0.9 for x in ag) / len(ag):.0%}; others (first 3 drafts) median {sorted(na)[len(na) // 2]:.2f}")

def cost(rs, depth):
    tok = ms = 0.0
    for r in rs:
        D = depth(r)
        tok += 1 + min(D, r["nacc"])
        ms += VER[D + 1] + MTP * (1 + min(D, 6))
    return 1000 * tok / ms

def flat(which, tau):
    def depth(r):
        D = 0
        while D < r["nd"] and r["conf"][which][D] >= tau:
            D += 1
        return D
    return depth

def tab(rs, which):
    t = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        for k in range(r["nd"]):
            key = (k, bucket(min(r["conf"][which][k], 1.0)))
            t[key][1] += 1
            t[key][0] += k < r["nacc"]
    return t

def cal(which, tau):
    tabs = {ri: tab([r for r in rows if r["req"] != ri], which) for ri in range(len(reqs))}
    def depth(r):
        t, D = tabs[r["req"]], 0
        while D < r["nd"]:
            acc, n = t[(D, bucket(min(r["conf"][which][D], 1.0)))]
            ra = [v for key, v in t.items() if key[0] == D]
            pooled = sum(v[0] for v in ra) / max(1, sum(v[1] for v in ra))
            if (acc + 4 * pooled) / (n + 4) < tau:
                break
            D += 1
        return D
    return depth

print("\nmodelled tok/s (all 6 requests):")
print(f"{'gate':28s}" + "".join(f"{t:>8}" for t in (0.3, 0.36, 0.5, 0.7, 0.9)))
for g, fn in (("flat", flat), ("calibrated", cal)):
    for which in ("plain", "biased"):
        print(f"{g + ', ' + which:28s}" + "".join(f"{cost(rows, fn(which, t)):8.2f}" for t in (0.3, 0.36, 0.5, 0.7, 0.9)))
print("\nper request, calibrated 0.36, plain -> biased:")
for ri in range(len(reqs)):
    rs = [r for r in rows if r["req"] == ri]
    a, b = cost(rs, cal("plain", 0.36)), cost(rs, cal("biased", 0.36))
    print(f"  request {ri}: {a:6.2f} -> {b:6.2f} ({(b / a - 1) * 100:+.1f}%), agreeing drafts {sum(sum(r['agree']) for r in rs)}")

print("\n--- at the first rejected draft: did the earlier match hold the token the target wanted?")
for n in (3, 4, 6):
    rej = right = had = 0
    for r in reqs:
        full = r["prompt"] + r["tokens"]
        for rd in r["rounds"]:
            p, d, t = rd["pos"], rd["drafts"], rd["target"]
            k = next((k for k in range(len(d)) if d[k] != t[k]), None)
            if k is None:
                continue
            rej += 1
            c, m = match(full, p, n)
            if len(c) > k and c[:k] == d[:k]:
                had += 1
                right += c[k] == t[k]
    print(f"match >= {n}: {rej} rejections, the match had a token there in {had}, the right one in {right} "
          f"({right / max(1, rej):.1%} of rejections)")
