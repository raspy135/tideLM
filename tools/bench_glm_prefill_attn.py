"""GLM-5.3 prefill attention at serving shapes: a tile sweep in one process.

No checkpoint, no peer, no engine. The inputs are the shapes `GlmTP.attention` hands
`sparse_mla` on the native EXL3 profile with `GLM_KV=fp8` *after* `_gathered` has materialised the
whole layer cache for `[0, S)`: an FP8 latent `uint8 [S, 512]` with fp32 128-block scales, BF16
RoPE keys `[S, 64]`, absorbed queries `[T, 64, 512]`, and one row's `index_topk` picks. Selection
order affects nothing that is timed (the kernel walks `sel` in order either way).

    python3 tools/bench_glm_prefill_attn.py --rows 1024 --keys 2048
    python3 tools/bench_glm_prefill_attn.py --rows 1024 --keys 2048 --tiles 32x64x8x2,32x32x4x3

`--check` runs one config against `test_sparse_attn.einsum_path` to prove the harness is wired
like `GlmTP.attention`, since a timing harness that calls the kernel wrong is silent.
"""

import argparse
import sys

import torch

sys.path.insert(0, "/app")
from engine.glm.sparse_attn import sparse_mla  # noqa: E402

H, DC, DR, SB = 64, 512, 64, 128          # heads, latent, rope, FP8 scale block
SCALE = (192 + 64) ** -0.5                # cfg.qk_dim ** -0.5


def make(rows, keys, seed=0, dtype=torch.float32, float8=True):
    dev = "cuda"
    S = rows + keys                       # cached positions; each row's picks lie behind it
    g = torch.Generator(device=dev).manual_seed(seed)
    # finite E4M3 values: raw random bytes include NaN/Inf patterns and poison the check
    cache = (torch.rand(S, DC, device=dev, generator=g) * 2 - 1).to(torch.float8_e4m3fn)
    if not float8:
        cache = cache.to(torch.bfloat16)            # sparse_mla reads uint8 as FP4 (fmt 2)
    cs = (torch.rand(S, DC // SB, dtype=dtype, device=dev, generator=g) + 0.5) * 1e-2
    rope = (torch.randn(S, DR, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    q_lat = (torch.randn(rows, H, DC, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    q_rope = (torch.randn(rows, H, DR, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    positions = torch.arange(S - rows, S, device=dev, dtype=torch.int32)
    # `rows` distinct random picks per row, all at or before the row's position
    pick = torch.stack([torch.randperm(int(p) + 1, device=dev, generator=g)[:keys] for p in positions[:rows]])
    sel = torch.sort(pick, dim=1).values.to(torch.int32)
    return cache, cs, rope, q_lat, q_rope, sel, positions


def flops(rows, keys):
    """QK: H x K x (DC + DR) MACs; PV: H x K x DC. Two FLOPs a MAC."""
    return 2 * rows * keys * H * ((DC + DR) + DC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--keys", type=int, default=2048)
    ap.add_argument("--tiles", default="32x64x8x1,32x64x8x2,32x32x8x2,32x32x4x3,16x64x4x2,64x32x8x2")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-scale-block", action="store_true", help="BF16 cache (fmt 0) instead of FP8")
    a = ap.parse_args()
    fmt = 0 if a.no_scale_block else 1
    tiles = [tuple(int(v) for v in t.split("x")) for t in a.tiles.split(",")]
    print(f"heads {H} latent {DC} rope {DR} keys {a.keys} cache {'BF16' if fmt == 0 else 'FP8'} "
          f"(qk_dim {192 + DR}, scale {SCALE:.5f})")
    for rows in a.rows:
        cache, cs, rope, q_lat, q_rope, sel, positions = make(rows, a.keys, float8=(fmt == 1))
        cache_arg = cache if fmt == 0 else (cache, cs)
        if a.check:
            from engine.glm.test_sparse_attn import einsum_path
            r, k = min(rows, 64), a.keys
            got = sparse_mla(q_lat[:r], q_rope[:r], cache_arg, rope, sel[:r], positions[:r], SCALE,
                             block_h=32, block_n=64, num_warps=8, num_stages=1)
            want = einsum_path(q_lat[:r], q_rope[:r], cache_arg, rope, sel[:r].long(), positions[:r])
            err = float((got.float() - want.float()).norm() / want.float().norm())  # the test's own bound
            print(f"  check vs einsum, {r} rows: relative L2 {err:.2e}")
        print(f"\nT={rows} rows, K={a.keys}")
        ref = None
        for (bh, bn, w, st) in tiles:
            try:
                out = sparse_mla(q_lat, q_rope, cache_arg, rope, sel, positions, SCALE,
                                 block_h=bh, block_n=bn, num_warps=w, num_stages=st)
                torch.cuda.synchronize()
            except Exception as ex:                                  # noqa: BLE001
                print(f"  {bh}x{bn}x{w}w{st}s: {type(ex).__name__} {str(ex)[:90]}")
                continue
            ms = []
            for _ in range(a.iters):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                sparse_mla(q_lat, q_rope, cache_arg, rope, sel, positions, SCALE,
                           block_h=bh, block_n=bn, num_warps=w, num_stages=st)
                e1.record()
                torch.cuda.synchronize()
                ms.append(e0.elapsed_time(e1))
            ms.sort()
            med = ms[len(ms) // 2]
            if ref is None:
                ref = out.clone()
                same = "ref"
            else:
                same = "bitwise" if torch.equal(out, ref) else f"{float((out - ref).norm() / ref.norm()):.0e}"
            print(f"  {bh:3d}x{bn:<3d} {w}w {st}s  {med:8.2f} ms  {flops(rows, a.keys) / med / 1e9:6.1f} TFLOPS"
                  f"  smem ~{bn * (DC + 2 * DR) * 2 * st / 1024:.0f} KB  vs 32x64x8x1 {same}")
        del cache, cs, rope, q_lat, q_rope, sel, positions
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
