"""Native GLM EXL3 component microbenchmarks: one GPU, rank-0 shapes (world 2), real weights.

  python3 tools/bench_glm_native_exl3.py [merge] [fixed] [head] [kvb] [prefill] [index] [act]

No section named runs them all. Decode timings are CUDA graph replays (as decode runs), median of
7; prefill timings eager, median of 5. merge and prefill also check the new code against the
code it replaced (kept here as old_mlp / old_prefill). Aborts below GLM_BENCH_MIN_AVAIL GB of
MemAvailable (default 5): on GB10 an over-commit hangs the box instead of raising.
Measured 2026-10-22.
"""
import os
import sys
import types

import torch

MD = os.environ.get("GLM_MODEL_DIR", "/models/GLM-5.3-EXL3-3.0bpw")
PACK = os.environ.get("GLM_EXL3_PACK0", "/models/glm-exl3-packs/glm-exl3-experts-r0of2.bin")
ONLY = set(sys.argv[1:])
LAYERS = list(range(10, 26))                       # 16 MoE layers: ~200 MB of shared experts, L2-cold
d = torch.device("cuda")


def guard(tag):
    avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1e6
    print(f"[{tag}] MemAvailable {avail:.1f} GB", flush=True)
    if avail < float(os.environ.get("GLM_BENCH_MIN_AVAIL", "5")):
        sys.exit(f"MemAvailable {avail:.1f} GB: abort")


def gtime(fn, reps=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(7):
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / reps * 1000)
    del g
    return sorted(ts)[3]


def etime(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[reps // 2]


def want(name):
    return not ONLY or name in ONLY


guard("start")
from engine.exl3_dense import Exl3Matrix                    # noqa: E402
from engine.glm.config import GlmConfig                     # noqa: E402
from engine.glm.exl3_attn import Exl3Source, _members, rotate, unpack  # noqa: E402

cfg = GlmConfig.load(MD)
src = Exl3Source(MD)


def remat(lin, split):
    return Exl3Matrix([lin.tr], lin.suh, lin.svh, [lin.bits], split, _members(str(d)))


def act(g, u):
    g, u = g.float(), u.float()
    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)


class FakeComm:
    """The peer's half is a copy of this rank's: deterministic, so both paths see the same data."""

    def cat_last(self, t):
        return torch.cat([t, t], -1)

    def all_gather(self, t):
        return torch.stack([t, t])


if want("merge"):
    from engine.glm.exl3 import Exl3Pack, make_arena, routed, warm
    from engine.glm.exl3_model import FixedMLP, mlp
    from engine.glm.tp import GlmTP
    eng = types.SimpleNamespace(rank=0, world=2, dev=d, comm=FakeComm(), cfg=cfg, mtp_L=10 ** 6,
                                dense=lambda L: False)
    eng._gather_pair = types.MethodType(GlmTP._gather_pair, eng)
    pack, E = Exl3Pack(PACK, cfg.hidden, cfg.moe_inter), 32
    eng.x3 = make_arena(E, 0, 2, d, cfg.hidden, cfg.moe_inter)
    for e in range(E):
        eng.x3.load_slot(e, *pack.read(LAYERS[0], e))
    warm(eng.x3)
    eng.fixed_exl3 = {LAYERS[0]: FixedMLP(src, f"model.layers.{LAYERS[0]}.mlp.shared_experts.", eng)}

    def old_mlp(L, x, rs, wr):
        """The first native version: routed() and FixedMLP() each gathering twice."""
        f = eng.fixed_exl3[L]
        h = act(f.gate(x), f.up(x))
        fixed = eng.comm.cat_last(f.down(eng.comm.cat_last(h).contiguous(), out_dtype=torch.float32))
        return (routed(x, rs, wr, eng.x3, eng.comm) + fixed).to(torch.bfloat16)

    gen = torch.Generator(device=d).manual_seed(7)
    for T in (1, 4, 8, 64):
        x = (torch.randn(T, cfg.hidden, device=d, generator=gen) * 0.5).to(torch.bfloat16)
        rs = torch.stack([torch.randperm(E, device=d, generator=gen)[:cfg.topk] for _ in range(T)]).to(torch.int32)
        wr = torch.rand(T, cfg.topk, device=d, generator=gen)
        a, b = old_mlp(LAYERS[0], x, rs, wr), mlp(eng, LAYERS[0], x, None, None, rs, wr)
        line = f"merge T={T}: bitwise_equal={torch.equal(a, b)}"
        if T <= 8:
            line += (f"  old {gtime(lambda: old_mlp(LAYERS[0], x, rs, wr)):.1f} us"
                     f"  merged {gtime(lambda: mlp(eng, LAYERS[0], x, None, None, rs, wr)):.1f} us (gathers local)")
        print(line, flush=True)
    del eng, pack
    torch.cuda.empty_cache()
    guard("after merge")

if want("fixed"):
    from engine.glm.exl3_model import FixedMLP
    mats = [tuple(src.lin(f"model.layers.{L}.mlp.shared_experts.{n}", d, 4, 0, 2, "n")
                  for n in ("gate_proj", "up_proj", "down_proj")) for L in LAYERS]
    N = mats[0][0].N
    groups = {s: [Exl3Matrix([g.tr, u.tr], torch.stack([g.suh, u.suh]), torch.stack([g.svh, u.svh]),
                             [g.bits, u.bits], s, _members(str(d))) for g, u, _ in mats] for s in (1, 2, 4, 8)}
    gen = torch.Generator(device=d).manual_seed(3)
    for T in (1, 4, 8):
        x = (torch.randn(T, cfg.hidden, device=d, generator=gen) * 0.5).to(torch.bfloat16)
        hs = (torch.randn(T, 2 * N, device=d, generator=gen) * 0.1).to(torch.bfloat16)
        ok = all(torch.equal(act(g(x), u(x)), act((y := m(x[:, None, :].expand(-1, 2, -1)))[:, :N], y[:, N:]))
                 for (g, u, _), m in zip(mats, groups[4]))
        res = {}
        for s in (1, 2, 4, 8):
            sep = [(remat(g, s), remat(u, s)) for g, u, _ in mats]
            res[f"sep s{s}"] = gtime(lambda: [act(a(x), b(x)) for a, b in sep]) / len(mats)
            res[f"grp s{s}"] = gtime(lambda: [act((y := m(x[:, None, :].expand(-1, 2, -1)))[:, :N], y[:, N:])
                                              for m in groups[s]]) / len(mats)
        for s in (1, 2, 4, 8):
            dn = [remat(dw, s) for _, _, dw in mats]
            res[f"down s{s}"] = gtime(lambda: [m(hs, out_dtype=torch.float32) for m in dn]) / len(mats)
        print(f"fixed T={T}: grouped_s4_bitwise_equal={ok}  " +
              "  ".join(f"{k} {v:.1f}" for k, v in res.items()) + "  (us a layer)", flush=True)
    eng = types.SimpleNamespace(rank=0, world=2, dev=d, comm=FakeComm())
    for prefix in ("model.layers.0.mlp.", f"model.layers.{LAYERS[0]}.mlp.shared_experts.",
                   f"model.layers.{cfg.layers}.mlp.shared_experts."):
        f = FixedMLP(src, prefix, eng)
        g, u = (src.lin(prefix + n, d, 4, 0, 2, "n") for n in ("gate_proj", "up_proj"))
        ok = [T for T in (1, 4, 8, 16, 17, 300)
              if not torch.equal(f.act(x := (torch.randn(T, cfg.hidden, device=d) * 0.5).to(torch.bfloat16)),
                                 act(g(x), u(x)))]
        print(f"FixedMLP {prefix}: bits {g.bits}, grouped {f.gate_up is not None}, "
              f"rows not bitwise equal: {ok or 'none'}", flush=True)
    del mats, groups
    torch.cuda.empty_cache()
    guard("after fixed")

if want("head"):
    from engine.glm.fp8_block import FP8Block, fp8_block_linear
    head = src.lin("lm_head", d, 1, 0, 2, "n")
    N, K = head.N, head.K
    w8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=d)
    for i in range(0, N, 8192):
        w8[i:i + 8192] = (torch.randn(min(8192, N - i), K, device=d, dtype=torch.bfloat16) * 100).to(torch.float8_e4m3fn)
    fp8 = FP8Block(w8, torch.rand((N + 127) // 128, K // 128, device=d) * 1e-3)
    print(f"head K={K} N={N} bits={head.bits}: exl3 {head.nbytes / 1e6:.0f} MB, fp8 {fp8.nbytes / 1e6:.0f} MB")
    for T in (1, 4, 8):
        x = torch.randn(T, K, device=d).to(torch.bfloat16)
        res = {f"exl3 s{s}": gtime(lambda m=remat(head, s): m(x, out_dtype=torch.float32), reps=20) for s in (1, 2, 4, 8)}
        res["fp8"] = gtime(lambda: fp8_block_linear(x, fp8, out_dtype=torch.float32), reps=20)
        print(f"head T={T}: " + "  ".join(f"{k} {v:.0f}" for k, v in res.items()) + "  (us)", flush=True)
    del head, w8, fp8
    torch.cuda.empty_cache()
    guard("after head")

if want("kvb"):
    from engine.glm.kvb_fp8 import HeadFP8, absorb_o, absorb_q
    from engine.glm.weights import GlmCheckpoint, P, _fp8_quant
    ck = GlmCheckpoint(MD)
    hl, hv = cfg.heads // 2, cfg.nope + cfg.v_dim
    bf, f8 = [], []
    for L in LAYERS:
        kvb = ck.get_rows(f"{P}layers.{L}.self_attn.kv_b_proj.weight", 0, hl * hv).to(d)
        b = kvb.to(torch.bfloat16).view(hl, hv, cfg.kv_lora)
        bf.append((b[:, :cfg.nope].contiguous(), b[:, cfg.nope:].contiguous()))
        q = HeadFP8.from_rows(*_fp8_quant(kvb), hl)
        f8.append((q.split(0, cfg.nope).to(d), q.split(cfg.nope, hv).to(d)))
    for T in (1, 4, 8):
        qn = torch.randn(T, hl, cfg.nope, device=d).to(torch.bfloat16)
        ol = torch.randn(T, hl, cfg.kv_lora, device=d).to(torch.bfloat16)
        tb = gtime(lambda: [(torch.einsum("thn,hnc->thc", qn, uk), torch.einsum("thc,hvc->thv", ol, uv)) for uk, uv in bf])
        tf = gtime(lambda: [(absorb_q(qn, uk), absorb_o(ol, uv)) for uk, uv in f8])
        print(f"kvb T={T}: bf16 {tb / len(LAYERS):.1f}  fp8 {tf / len(LAYERS):.1f}  (us a layer, q+o)", flush=True)
    del bf, f8
    torch.cuda.empty_cache()
    guard("after kvb")

if want("prefill"):
    import exl3_ref as R

    def old_prefill(m, x, out_dtype):
        """Exl3Lin's prefill before 2026-10-22: fp32 matmul rotations."""
        xh = rotate(x.float() * m.suh.float()).to(torch.float16)
        return (rotate(torch.mm(xh, unpack(m.tr, m.bits), out_dtype=torch.float32)) * m.svh.float()).to(out_dtype)

    a = "model.layers.10.self_attn."
    s = "model.layers.10.mlp.shared_experts."
    for name, args in (("q_a", (a + "q_a_proj", d, 8)), ("kv_a", (a + "kv_a_proj_with_mqa", d, 8)),
                       ("q_b", (a + "q_b_proj", d, 1, 0, 2, "n")), ("o_proj", (a + "o_proj", d, 2, 0, 2, "k")),
                       ("shared gate", (s + "gate_proj", d, 4, 0, 2, "n")),
                       ("shared down", (s + "down_proj", d, 2, 0, 2, "n"))):
        m = src.lin(*args)
        x = (torch.randn(4096, m.K, device=d) * 0.7).to(torch.bfloat16)
        o, n = old_prefill(m, x, torch.bfloat16), m(x)
        ref = R.forward(x[:64].cpu(), m.tr.cpu(), m.suh.cpu(), m.svh.cpu(), m.bits, "mul1")
        err = lambda y: float((y[:64].double().cpu() - ref).norm() / ref.norm())
        print(f"prefill {name} K={m.K} N={m.N} 4096 rows: old {etime(lambda: old_prefill(m, x, torch.bfloat16)):.2f} ms"
              f"  new {etime(lambda: m(x)):.2f} ms  err to fp64 old {err(o):.3e} new {err(n):.3e}"
              f"  identical {float((o == n).float().mean()) * 100:.2f}%", flush=True)
        del m, x
        torch.cuda.empty_cache()
    guard("after prefill")

if want("index"):
    from engine.glm.index_kernel import index_scores
    raw = torch.randn(262144, 128, device=d)
    sc = raw.abs().amax(-1, keepdim=True) / 448
    cache = ((raw / sc).to(torch.float8_e4m3fn), sc)
    del raw
    for T in (1, 4, 8):
        q = torch.randn(T, 32, 128, device=d, dtype=torch.bfloat16)
        w = torch.rand(T, 32, device=d)
        pos = torch.arange(8192 - T, 8192, device=d)
        res = {f"split S={S}": gtime(lambda: index_scores(q, w, cache, pos, S, 128 ** -.5, 2, 0).topk(min(2048, S), -1))
               for S in (131072, 16384, 4096)}
        res.update({f"mtp S={S}": gtime(lambda: index_scores(q, w, cache, pos, S, 128 ** -.5, 1, 0).topk(2048, -1))
                    for S in (262144, 8192)})
        print(f"index T={T} (context 8,192): " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()) + "  (us)", flush=True)
    del cache
    torch.cuda.empty_cache()

if want("act"):
    N = 1024
    for T in (1, 4, 8):
        gus = [torch.randn(T, 2 * N, device=d).to(torch.bfloat16) for _ in range(79)]
        t = gtime(lambda: [act(gu[:, :N], gu[:, N:]) for gu in gus])
        print(f"act T={T}: {t / 79:.1f} us a layer", flush=True)
