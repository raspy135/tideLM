"""JIT loader and thin tensor-pointer wrapper for :mod:`fp4_moe_cuda.cu`.

This intentionally does not use a PyTorch C++ extension: the device implementation is ordinary
CUDA C++, and the shared library exposes four C ABI launchers.  ctypes only passes existing tensor
pointers and PyTorch's current stream, so launches remain CUDA-graph capturable and the build has no
Python/torch ABI dependency.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import torch


SOURCE = Path(__file__).with_name("fp4_moe_cuda.cu")
RELAXED_REDUCE = os.environ.get("DSV41_FP4_CUDA_RELAXED", "1") == "1"
# v2 kernels (16-byte group loads, row kept in registers across pairs); same bits as v1 relaxed
# (tools/test_fp4_moe_v2.py), 2.4-2.9 % faster decode end to end (docs/decode-launches.md).
# DSV41_FP4_CUDA_V2=0 restores v1; the EP2 boot guard pins it.
V2 = os.environ.get("DSV41_FP4_CUDA_V2", "1") == "1"
# How v1's `p *= scale; acc += p` compiled: 0 = two roundings, 1 = one fma. Both are exact: a UE8M0
# scale is a power of two, so the scaled product never rounds. 0 is the plain spelling.
V2_FMA_SCALE = int(os.environ.get("DSV41_FP4_CUDA_V2_FMA", "0"))
# v3 down: rows per warp for the down kernel (1 = v2's one row; 2 or 4 = moe_down_bm16_v3, same
# bits as v2, tools/test_fp4_moe_v2.py). Only with v2 and k <= 3072; the EP2 boot guard pins it.
DOWN_ROWS = int(os.environ.get("DSV41_FP4_CUDA_DOWN_ROWS", "1"))
if DOWN_ROWS not in (1, 2, 4):
    raise ValueError(f"DSV41_FP4_CUDA_DOWN_ROWS must be 1, 2 or 4, not {DOWN_ROWS}")


def source_digest() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _library():
    nvcc = os.environ.get("NVCC", os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"))
    version = subprocess.check_output([nvcc, "--version"])
    # A real-weight sweep of 1/2/4/8/16 picked one row per warp; keep it fixed so EP ranks cannot
    # silently compile different launch geometries from an unguarded environment variable.
    rows = 1
    key = hashlib.sha256(SOURCE.read_bytes() + version + platform.machine().encode()
                         + str((rows, RELAXED_REDUCE, V2_FMA_SCALE)).encode()).hexdigest()[:24]
    folder = Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/dsv41-native")) / "fp4-moe-cuda"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / f"fp4-moe-{key}.so"
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix="build-", dir=folder) as tmp:
                built = Path(tmp) / "fp4-moe.so"
                # The architecture-specific `a` target is required for cvt.e2m1x2.  nvcc's
                # convenient `-arch=sm_121a` spelling emits generic sm_121 PTX with CUDA 13.0.
                subprocess.run([
                    nvcc, "-std=c++17", "-O3", "-DNDEBUG",
                    "-Xcompiler=-fPIC", "-shared",
                    f"-DFP4_ROWS_PER_WARP={rows}",
                    f"-DFP4_RELAXED_REDUCE={int(RELAXED_REDUCE)}",
                    f"-DFP4_V2_FMA_SCALE={V2_FMA_SCALE}",
                    "-gencode", "arch=compute_121a,code=sm_121a",
                    str(SOURCE), "-o", str(built),
                ], check=True)
                os.replace(built, target)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.c_void_p
    i64, i32, f32 = ctypes.c_int64, ctypes.c_int, ctypes.c_float
    lib.fp4_moe_cuda_up.argtypes = [
        ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
        i64, i64, f32, i32, i32, i32, i32, i32,
    ]
    lib.fp4_moe_cuda_down.argtypes = [
        ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64,
        i32, i32, i32, i32, i32, i32, i32, i32,
    ]
    lib.fp4_moe_cuda_up_v2.argtypes = lib.fp4_moe_cuda_up.argtypes
    lib.fp4_moe_cuda_down_v2.argtypes = lib.fp4_moe_cuda_down.argtypes
    lib.fp4_moe_cuda_down_v3.argtypes = lib.fp4_moe_cuda_down.argtypes + [i32]
    lib.fp4_moe_cuda_up_scales.argtypes = lib.fp4_moe_cuda_up.argtypes + [i32]
    lib.fp4_moe_cuda_down_scales.argtypes = lib.fp4_moe_cuda_down.argtypes + [i32, i32]
    lib.fp4_moe_cuda_scale_expand.argtypes = [ptr, ptr, ptr, i32, i32, i32]
    lib.fp4_moe_cuda_route_small.argtypes = [ptr, ptr, ptr, ptr, ptr, i32, i32]
    lib.fp4_moe_cuda_round_reduce.argtypes = [ptr, ptr, ptr, i32, i32, i32]
    lib.dream_fp8_cuda_up.argtypes = [ptr, ptr, i64, ptr, ptr, i32, ptr, ptr, ptr, i64, ptr,
                                      f32, i32, i32, i32, i32]
    lib.dream_fp8_cuda_down.argtypes = [ptr, ptr, i64, ptr, ptr, i32, ptr, ptr, ptr, i64,
                                        i32, i32, i32, i32, i32, i32]
    for name in ("fp4_moe_cuda_up", "fp4_moe_cuda_down", "fp4_moe_cuda_route_small",
                 "fp4_moe_cuda_round_reduce", "fp4_moe_cuda_up_v2", "fp4_moe_cuda_down_v2",
                 "fp4_moe_cuda_down_v3", "fp4_moe_cuda_up_scales", "fp4_moe_cuda_down_scales",
                 "fp4_moe_cuda_scale_expand",
                 "dream_fp8_cuda_up", "dream_fp8_cuda_down"):
        getattr(lib, name).restype = i32
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _library()
    return _LIB


def _p(tensor: torch.Tensor | None):
    return None if tensor is None else ctypes.c_void_p(tensor.data_ptr())


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(code: int, operation: str):
    if code:
        raise RuntimeError(f"native CUDA FP4 {operation} launch failed with CUDA error {code}")


def build_routing_small(slots: torch.Tensor, block_m: int, topk: int):
    """Return slots, plain pairs (down), packed token/pair entries (up), and block capacity."""
    if block_m != 16:
        raise ValueError("native decode router supports BM=16 only")
    if slots.dtype != torch.int32 or not slots.is_cuda or not slots.is_contiguous():
        raise ValueError("native decode router requires contiguous CUDA int32 slots")
    pairs = slots.numel()
    # 65..512 pairs (batched multi-request decode) take a parallel-rank sort with the same output.
    if not 1 <= pairs <= 512:
        raise ValueError("native decode router supports 1..512 pairs")
    if not 1 <= topk <= pairs:
        raise ValueError("native decode router requires 1 <= topk <= number of pairs")
    block_slot = torch.empty((pairs,), dtype=torch.int32, device=slots.device)
    block_pair = torch.empty((pairs * block_m,), dtype=torch.int32, device=slots.device)
    block_route = torch.empty_like(block_pair)
    _check(_lib().fp4_moe_cuda_route_small(
        _stream(), _p(slots), _p(block_slot), _p(block_pair), _p(block_route), pairs, topk),
        "route")
    return block_slot, block_pair, block_route, pairs


def dream_fp8_up(x, idx, tier, h, route_weight, limit: float, topk: int):
    """Overwrite h's rows for the pairs routed to an FP8-tier expert (engine/dream_overlay.py).

    idx: int64 [P] expert ids (pair p = token * topk + k, as h's rows); tier: one layer's
    Fp8Layer (map [n_experts] int32 -> slot or -1, w13 [S, 2n, K] e4m3, s13 [S, 2n, K/32] UE8M0)."""
    pairs = idx.numel()
    n, k = tier.w13.shape[1] // 2, tier.w13.shape[2]
    if x.stride(1) != 1 or h.stride(1) != 1 or route_weight.dtype != torch.float32:
        raise ValueError("dream_fp8_up: row-major x/h and fp32 route weights required")
    _check(_lib().dream_fp8_cuda_up(
        _stream(), _p(x), x.stride(0), _p(idx), _p(tier.map), tier.map.numel(), _p(tier.w13),
        _p(tier.s13), _p(h), h.stride(0), _p(route_weight), float(limit), topk, pairs, n, k),
        "dream fp8 up")


def dream_fp8_down(h, idx, tier, parts, topk: int, ntok: int, partial: bool = False):
    """Overwrite parts' rows (expert_k * ntok + token) for the FP8-tier pairs, as v2 down writes them."""
    pairs = idx.numel()
    n, k = tier.w2.shape[1], tier.w2.shape[2]
    if h.stride(1) != 1 or parts.stride(1) != 1 or h.shape[1] < k:
        raise ValueError("dream_fp8_down: row-major h/parts with the full intermediate required")
    _check(_lib().dream_fp8_cuda_down(
        _stream(), _p(h), h.stride(0), _p(idx), _p(tier.map), tier.map.numel(), _p(tier.w2),
        _p(tier.s2), _p(parts), parts.stride(0), topk, pairs, n, k, ntok, int(partial)),
        "dream fp8 down")


def up(x, w1, s1, w3, s3, h, route_weight, block_slot, block_pair, limit,
       topk: int, n: int, k: int, nb: int, null_slot: int, v2: bool | None = None):
    packed = getattr(s1, '_tidelm_scale_layout', None)
    if packed is not None:
        if not (V2 if v2 is None else v2) or getattr(s3, '_tidelm_scale_layout', None) != packed:
            raise ValueError('packed scales require v2 and identical gate/up layouts')
        if packed['size'] != n * (k // 32):
            raise ValueError('packed scale plane size does not match up projection')
        _check(_lib().fp4_moe_cuda_up_scales(
            _stream(), _p(x), _p(w1), _p(s1), _p(w3), _p(s3), _p(h), _p(route_weight),
            _p(block_slot), _p(block_pair), x.stride(0), h.stride(0), float(limit), topk, n, k,
            nb, null_slot, packed['stride']), 'packed-scale up')
        return
    fn = _lib().fp4_moe_cuda_up_v2 if (V2 if v2 is None else v2) else _lib().fp4_moe_cuda_up
    _check(fn(
        _stream(), _p(x), _p(w1), _p(s1), _p(w3), _p(s3), _p(h), _p(route_weight),
        _p(block_slot), _p(block_pair), x.stride(0), h.stride(0), float(limit), topk, n, k,
        nb, null_slot), "up")


def down(h, w2, s2, parts, block_slot, block_pair, topk: int, n: int, k: int,
         ntok: int, nb: int, null_slot: int, partial: bool, parts_world: int, v2: bool | None = None,
         rows: int | None = None):
    v2 = V2 if v2 is None else v2
    rows = DOWN_ROWS if rows is None else rows
    args = (_stream(), _p(h), _p(w2), _p(s2), _p(parts), _p(block_slot), _p(block_pair),
            h.stride(0), parts.stride(0), topk, n, k, ntok, nb, null_slot, int(partial), parts_world)
    packed = getattr(s2, '_tidelm_scale_layout', None)
    if packed is not None:
        if not v2 or packed['size'] != n * (k // 32):
            raise ValueError('packed scales require v2 and a matching down projection')
        _check(_lib().fp4_moe_cuda_down_scales(*args, rows, packed['stride']), 'packed-scale down')
        return
    if v2 and rows > 1 and k <= 3072:
        _check(_lib().fp4_moe_cuda_down_v3(*args, rows), "down")
        return
    fn = _lib().fp4_moe_cuda_down_v2 if v2 else _lib().fp4_moe_cuda_down
    _check(fn(*args), "down")


def round_reduce(parts: torch.Tensor, out: torch.Tensor, topk: int):
    if out.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("native round-reduce output must be fp32 or bf16")
    _check(_lib().fp4_moe_cuda_round_reduce(
        _stream(), _p(parts), _p(out), out.numel(), topk, int(out.dtype == torch.bfloat16)),
        "round-reduce")
