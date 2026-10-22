"""Old (insertion sort) vs new (parallel rank) route_small_bm16: identical outputs, graph-replay time."""
import ctypes, subprocess, sys, statistics, torch
srcs = {'old': '/mb/fp4_moe_cuda_old.cu', 'new': '/app/tools/fp4_moe_cuda.cu'}
libs = {}
for name, src in srcs.items():
    out = f'/tmp/route-{name}.so'
    subprocess.run(['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-DNDEBUG', '-Xcompiler=-fPIC', '-shared',
                    '-DFP4_ROWS_PER_WARP=1', '-DFP4_RELAXED_REDUCE=0', '-DFP4_V2_FMA_SCALE=0',
                    '-gencode', 'arch=compute_121a,code=sm_121a', src, '-o', out], check=True)
    libs[name] = ctypes.CDLL(out)
P = ctypes.c_void_p
def route(lib, slots, topk):
    pairs = slots.numel()
    bs = torch.empty(pairs, dtype=torch.int32, device='cuda')
    bp = torch.empty(pairs * 16, dtype=torch.int32, device='cuda')
    br = torch.empty_like(bp)
    st = P(torch.cuda.current_stream().cuda_stream)
    rc = lib.fp4_moe_cuda_route_small(st, P(slots.data_ptr()), P(bs.data_ptr()), P(bp.data_ptr()), P(br.data_ptr()),
                                      ctypes.c_int(pairs), ctypes.c_int(topk))
    assert rc == 0, rc
    return bs, bp, br
gen = torch.Generator().manual_seed(1)
bad = 0
for P_ in range(1, 65):
    for topk in (1, 3, 6):
        for span in (3, 50, 400):
            slots = torch.randint(-2, span, (P_,), dtype=torch.int32, generator=gen).cuda()
            a = route(libs['old'], slots, topk); b = route(libs['new'], slots, topk)
            torch.cuda.synchronize()
            if not all(torch.equal(x, y) for x, y in zip(a, b)): bad += 1
print('mismatching cases:', bad)
for pairs, topk in ((15, 3), (24, 6), (36, 6)):
    slots = torch.randperm(384, generator=gen)[:pairs].to(torch.int32).cuda()
    for name, lib in libs.items():
        g = torch.cuda.CUDAGraph()
        route(lib, slots, topk); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(100): route(lib, slots, topk)
        ts = []
        for _ in range(5):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 10)
        print(f"P={pairs:2d} {name}: {statistics.median(ts):6.2f} us/call")
