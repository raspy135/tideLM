"""FP8 decode tile sweep at TP2-local DS V4.1 shapes, L2-cold (distinct weight copies), graph replay.
Every config is checked torch.equal against the shipped one (BN=32, BK=256, 4 warps, 3 stages)."""
import itertools, statistics, sys, time
import torch, triton
sys.path[:0] = ['/app', '/app/tools']
from fp8_linear import FP8Weight, FP8GroupedWeight, _fp8_linear_kernel, _fp8_grouped_kernel

torch.manual_seed(0)
dev = 'cuda'
SHAPES = [  # name, N, K, M
    ('wq_a+wkv', 1792, 5120, 4), ('wq_b/r', 16384, 1280, 4), ('wo_b/r', 2560, 8192, 4),
    ('sh_w13/r', 2304, 5120, 4), ('sh_w2/r', 2560, 2304, 4),
    ('main_proj', 5120, 15360, 4), ('main_proj/r', 2560, 15360, 4),
    ('dhead/r', 20480, 5120, 5), ('sh_w13 full', 4608, 5120, 5), ('sh_w2 full', 5120, 2304, 5),
]
only = sys.argv[1:] 

def rand_w(N, K):
    w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 128, ((N + 31) // 32, K // 32), device=dev, dtype=torch.uint8)
    return FP8Weight(w, s)

def bench(fn, calls, reps=7):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / calls)
    return statistics.median(ts)

def lin_cfgs(K):
    bks = [256, 128] if K % 256 == 0 else [128]
    for bn, bk, nw, ns, bit in itertools.product((16, 32, 64), bks, (2, 4, 8), (2, 3, 4, 5), (False, True)):
        yield bn, bk, nw, ns, bit

for name, N, K, M in SHAPES:
    if only and name not in only: continue
    nbytes = N * K + N * K // 1024
    C = max(2, min(40, -(-300_000_000 // nbytes)))
    Ws = [rand_w(N, K) for _ in range(C)]
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ys = [torch.empty(M, N, device=dev, dtype=torch.bfloat16) for _ in range(C)]
    def launch(cfg, W, y):
        bn, bk, nw, ns, bit = cfg
        _fp8_linear_kernel[(triton.cdiv(N, bn), 1)](x, W.w, W.s, y, M, N, K, x.stride(0), W.w.stride(0),
            W.s.stride(0), y.stride(0), BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, BIT_SCALE=bit,
            num_warps=nw, num_stages=ns)
    base = (32, 256 if K % 256 == 0 else 128, 4, 3, False)
    launch(base, Ws[0], ys[0]); ref = ys[0].clone()
    rows = []
    for cfg in lin_cfgs(K):
        try:
            launch(cfg, Ws[0], ys[0]); torch.cuda.synchronize()
        except Exception as e:  # smem overflow etc.
            continue
        exact = torch.equal(ys[0], ref)
        reps_per = max(1, 40 // C)
        def fn(cfg=cfg):
            for _ in range(reps_per):
                for W, y in zip(Ws, ys): launch(cfg, W, y)
        us = bench(fn, reps_per * C)
        rows.append((us, cfg, exact))
    rows.sort(key=lambda r: r[0])
    b = [r for r in rows if r[1] == base][0]
    print(f"{name:12s} N={N} K={K} M={M} copies={C}: base {b[0]:7.1f} us {nbytes/b[0]/1e3:5.0f} GB/s", flush=True)
    for us, cfg, exact in rows[:6]:
        print(f"    {us:7.1f} us {nbytes/us/1e3:5.0f} GB/s  bn={cfg[0]} bk={cfg[1]} w={cfg[2]} st={cfg[3]} bit={int(cfg[4])} exact={exact}", flush=True)
    del Ws, ys; torch.cuda.empty_cache()

# grouped wo_a (TP-local: 4 groups x 1024 x 4096)
if not only or 'wo_a/r' in only:
    G, R, K, T = 4, 1024, 4096, 4
    nbytes = G * R * K * (1 + 1/1024)
    C = 20
    Ws = []
    for _ in range(C):
        w = (torch.randn(G * R, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(118, 128, (G * R // 32, K // 32), device=dev, dtype=torch.uint8)
        Ws.append(FP8GroupedWeight(w, s, G, R))
    x = (torch.randn(T, G, K, device=dev) * 0.5).to(torch.bfloat16)
    ys = [torch.empty(T, G, R, device=dev, dtype=torch.bfloat16) for _ in range(C)]
    def launch(cfg, W, y):
        bn, bk, nw, ns, bit = cfg
        _fp8_grouped_kernel[(triton.cdiv(R, bn), 1, G)](x, W.w, W.s, y, T, R, K, x.stride(0), x.stride(1),
            W.w.stride(0), W.s.stride(0), y.stride(0), y.stride(1), BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk,
            BIT_SCALE=bit, num_warps=nw, num_stages=ns)
    base = (32, 256, 4, 3, False)
    launch(base, Ws[0], ys[0]); ref = ys[0].clone()
    rows = []
    for cfg in lin_cfgs(K):
        try:
            launch(cfg, Ws[0], ys[0]); torch.cuda.synchronize()
        except Exception:
            continue
        exact = torch.equal(ys[0], ref)
        def fn(cfg=cfg):
            for _ in range(2):
                for W, y in zip(Ws, ys): launch(cfg, W, y)
        rows.append((bench(fn, 2 * C), cfg, exact))
    rows.sort(key=lambda r: r[0])
    b = [r for r in rows if r[1] == base][0]
    print(f"wo_a/r grouped G={G} R={R} K={K}: base {b[0]:7.1f} us {nbytes/b[0]/1e3:5.0f} GB/s")
    for us, cfg, exact in rows[:6]:
        print(f"    {us:7.1f} us {nbytes/us/1e3:5.0f} GB/s  bn={cfg[0]} bk={cfg[1]} w={cfg[2]} st={cfg[3]} bit={int(cfg[4])} exact={exact}")
