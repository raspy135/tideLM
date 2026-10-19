"""JIT loader + orchestration for :mod:`exl3_moe_cuda.cu` (the EXL3 decode kernels).

Same shape as :mod:`fp4_moe_cuda`: nvcc builds the .cu once into a small C-ABI .so and ctypes
passes tensor pointers and the current stream, so launches stay CUDA-graph capturable and there is
no libtorch ABI dependency.  ``moe_forward`` runs the five kernels over an ``exl3_moe.Exl3Arena``.

v1 scope: TP-world-1 and 3-bit slots only.  TF's tile stride is ``4*K2`` words, so a 2-bit expert
must live in a 2-bit pool, not padded into a 3-bit slot -- per-bit-width pools are the next step.
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

SOURCE = Path(__file__).with_name("exl3_moe_cuda.cu")

# (nt, warps, pf) tile setting; K must divide 16*warps and N by 16*nt. See grouped_launch.
NT, WARPS, PF = 8, 4, 1
CB_MUL1 = 2
# Prefill (P > 64): "tile" = exl3m_prefill, which decodes each trellis tile once for all MT*16 rows
# of a routing block; "grouped" = the decode kernel, one program per 16 rows. NTxWARPSxPF, and the
# routing block (BM = 16*MT). A numerics choice (K is summed in one chain instead of four warp
# partials), so the engine's boot guard pins both.
PREFILL_KERNEL = os.environ.get("DSV41_EXL3_PREFILL_KERNEL", "tile")
# Defaults: the 2026-10-08 sweep (tools/bench_exl3_prefill.py, T=2048, rank shapes): every tile
# shape landed at 25.5-27.5 ms a layer against the grouped kernel's 52.8; 32-row blocks with
# 4 n-tiles x 2 warps were best or tied at T=2048 and T=512. 64-row blocks (4 m-tiles, 128
# accumulators a thread) were slower, 30-39 ms: occupancy, not decode count, sets the pace.
PREFILL_TILE = os.environ.get("DSV41_EXL3_PREFILL_TILE", "4x2x2")
PREFILL_BM = int(os.environ.get("DSV41_EXL3_PREFILL_BM", "32"))
# "smem" = exl3m_prefill_smem: a block of W warps x MTW m-tiles each (BM = W * MTW * 16 rows) decodes
# every trellis tile of its NTB n-tiles once into shared memory and all the warps multiply it
# (exl3_moe_cuda.cu, smem_tiles). WxMTWxNTBxPF. Bit-identical to "tile"; the routing block follows
# from the setting, DSV41_EXL3_PREFILL_BM is not read. Measured on GLM-5.3's shapes (2026-10-20,
# docs/gotchas.md): no faster than "tile" at 2,048 rows, 10% faster at 4,096 with 4x1x8x2; wider
# blocks (8x1x8, 16x1x8) and MTW = 2 are slower. The GLM profile selects it (GLM_EXL3_PREFILL).
PREFILL_SMEM_TILE = os.environ.get("DSV41_EXL3_PREFILL_SMEM_TILE", "4x1x8x2")


def prefill_setting(kernel: str | None = None, tile: str | None = None, block_m: int | None = None):
    """(kernel, tile tuple or None, routing block rows) for the prefill MoE path: the defaults
    above unless overridden. tile: (nt, warps, pf) for "tile", (warps, mtw, ntb, pf) for "smem"."""
    kernel = kernel or PREFILL_KERNEL
    if kernel == "smem":
        t = tuple(int(v) for v in (tile or PREFILL_SMEM_TILE).split("x"))
        if len(t) != 4:
            raise ValueError(f"smem prefill tile {tile or PREFILL_SMEM_TILE}: want WxMTWxNTBxPF")
        return kernel, t, t[0] * t[1] * 16
    if kernel == "tile":
        t = tuple(int(v) for v in (tile or PREFILL_TILE).split("x"))
        if len(t) != 3:
            raise ValueError(f"prefill tile {tile or PREFILL_TILE}: want NTxWARPSxPF")
        return kernel, t, block_m or PREFILL_BM
    if kernel == "grouped":
        return kernel, None, block_m or 64
    raise ValueError(f"DSV41_EXL3_PREFILL_KERNEL={kernel}: tile, smem or grouped")


def source_digest() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _library():
    nvcc = os.environ.get("NVCC", os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"))
    version = subprocess.check_output([nvcc, "--version"])
    key = hashlib.sha256(SOURCE.read_bytes() + version + platform.machine().encode()
                         + str((NT, WARPS, PF)).encode()).hexdigest()[:24]
    folder = Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/dsv41-native")) / "exl3-moe-cuda"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / f"exl3-moe-{key}.so"
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix="build-", dir=folder) as tmp:
                built = Path(tmp) / "exl3-moe.so"
                subprocess.run([
                    nvcc, "-std=c++17", "-O3", "-DNDEBUG",
                    "-Xcompiler=-fPIC", "-shared",
                    "-gencode", "arch=compute_121a,code=sm_121a",
                    str(SOURCE), "-o", str(built),
                ], check=True)
                os.replace(built, target)
    lib = ctypes.CDLL(str(target))
    ptr, i64, i32, f32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float
    lib.exl3m_group.argtypes = [ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr]
    lib.exl3m_rot_in.argtypes = [ptr, i32, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_grouped.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                  i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_gateup.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, f32, i32, ptr]
    lib.exl3m_down_combine.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_prefill.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                  i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_prefill_smem.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                       i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_dense_rot.argtypes = [ptr, i32, ptr, ptr, i32, i32, i32, i32, ptr]
    lib.exl3m_dense_out.argtypes = [ptr, ptr, ptr, i32, i32, i32, i32, i32, ptr]
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _library()
    return _LIB


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _ptrs(arena):
    """Per-slot trellis pointers (int64 [slots]) and K2 (int32 [slots]); built once per arena."""
    cached = getattr(arena, "_cuda_ptrs", None)
    if cached is not None:
        return cached
    if arena.tp_world not in (1, 2):
        raise NotImplementedError("exl3_moe_cuda supports tp_world=1 or 2")
    dev = arena.t1.device
    out = {}
    for key, name in (("t1p", "t1"), ("t3p", "t3"), ("t2p", "t2")):
        # per slot, so an exact-slot arena's narrow and wide pools both work (Exl3Arena.slot_ptrs)
        out[key] = torch.tensor(arena.slot_ptrs(name), dtype=torch.int64, device=dev)
    out["k2"] = arena.bits_gpu   # device tensor, updated by load_slot (widths change on swaps)
    # w2's width when it can differ from w1/w3's (engine/exl3_dense.py's shared experts: one layer
    # stores gate/up at 4 bits and down at 5); the routed pack has one width per expert
    out["k2d"] = getattr(arena, "bits2_gpu", arena.bits_gpu)
    arena._cuda_ptrs = out
    return out


def warm(arena) -> None:
    """Build the JIT library and the per-slot pointer tables at boot, not inside a capture."""
    _lib()
    _ptrs(arena)


def _grouped_or_tile(lib, tile, x0, x1, t0, t1, k2a, k2b, uids, ucount, members, z, Kd, N, P, maxm, K, nexp_max,
                     mats, nt, warps, pf, st):
    if tile is None:
        lib.exl3m_grouped(x0, x1, t0, t1, k2a, k2b, uids, ucount, members, z, int(Kd), int(N), int(P), 1,
                          int(maxm), int(K), int(nexp_max), mats, nt, warps, pf, 2, 10, CB_MUL1, st)
    elif len(tile) == 4:
        tw, tmtw, tntb, tpf = tile
        if tw * tmtw * 16 != maxm:
            raise ValueError(f"smem prefill tile {tile} wants {tw * tmtw * 16}-row routing blocks, got {maxm}")
        lib.exl3m_prefill_smem(x0, x1, t0, t1, k2a, k2b, uids, ucount, members, z, int(Kd), int(N), int(P), int(K),
                               int(nexp_max), mats, tw, tmtw, tntb, tpf, 4, 6, st)
    else:
        tnt, tw, tpf = tile
        lib.exl3m_prefill(x0, x1, t0, t1, k2a, k2b, uids, ucount, members, z, int(Kd), int(N), int(P), int(K),
                          int(nexp_max), mats, int(maxm), tnt, tw, tpf, 4, 6, st)


def _up(x, pick, uids, ucount, members, arena, T, K, maxm, nexp_max, swiglu_limit, xd, nt, warps, pf, tile=None):
    """rot_in + gate/up grouped + gate/up epilogue: this rank's half of the rotated down input, fp16 into `xd`
    [P, inter] (any 16-bit buffer viewed as fp16)."""
    P = T * K
    dim, inter = arena.shapes["suh1"][0], arena.inter
    dev = x.device
    p = _ptrs(arena)
    xh0 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    xh1 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    z = torch.empty((2, P, inter), dtype=torch.float32, device=dev)
    lib = _lib()
    st = _stream()
    lib.exl3m_rot_in(ctypes.c_void_p(x.data_ptr()), int(x.stride(0)), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.suh1.data_ptr()), ctypes.c_void_p(arena.suh3.data_ptr()),
                     ctypes.c_void_p(xh0.data_ptr()), ctypes.c_void_p(xh1.data_ptr()),
                     int(T), int(dim), int(K), int(arena.slots), 1, st)
    suh2_stride = int(arena.shapes["suh2"][0])
    suh2_base = arena.suh2.data_ptr() + arena.tp_rank * inter * 2
    V = ctypes.c_void_p
    _grouped_or_tile(lib, tile, V(xh0.data_ptr()), V(xh1.data_ptr()), V(p["t1p"].data_ptr()), V(p["t3p"].data_ptr()),
                     V(p["k2"].data_ptr()), V(p["k2"].data_ptr()), V(uids.data_ptr()), V(ucount.data_ptr()),
                     V(members.data_ptr()), V(z.data_ptr()), dim, inter, P, maxm, K, nexp_max, 2, nt, warps, pf, st)
    lib.exl3m_gateup(ctypes.c_void_p(z.data_ptr()), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.svh1.data_ptr()), ctypes.c_void_p(arena.svh3.data_ptr()),
                     ctypes.c_void_p(suh2_base), ctypes.c_void_p(xd.data_ptr()),
                     int(T), int(P), int(inter), 1, int(K), int(arena.slots), suh2_stride,
                     float(swiglu_limit), 1, st)


def _down(xd_full, pick, uids, ucount, members, weights, arena, T, K, maxm, nexp_max, nt, warps, pf, tile=None):
    """down grouped + rotate * svh + the top-k weighted combine over the gathered [P, world * inter]
    fp16 input: this rank's [T, down_n] fp32 slice of the routed output."""
    P = T * K
    down_n, down_k = arena.down_n, arena.tp_world * arena.inter
    dev = xd_full.device
    p = _ptrs(arena)
    zd = torch.empty((1, P, down_n), dtype=torch.float32, device=dev)
    y = torch.empty((P, down_n), dtype=torch.float32, device=dev)
    local = torch.empty((T, down_n), dtype=torch.float32, device=dev)
    wts = weights.reshape(-1).float().contiguous()
    lib = _lib()
    st = _stream()
    V = ctypes.c_void_p
    _grouped_or_tile(lib, tile, V(xd_full.data_ptr()), V(xd_full.data_ptr()), V(p["t2p"].data_ptr()),
                     V(p["t2p"].data_ptr()), V(p["k2d"].data_ptr()), V(p["k2d"].data_ptr()), V(uids.data_ptr()),
                     V(ucount.data_ptr()), V(members.data_ptr()), V(zd.data_ptr()), down_k, down_n, P, maxm, K,
                     nexp_max, 1, nt, warps, pf, st)
    lib.exl3m_down_combine(ctypes.c_void_p(zd.data_ptr()), ctypes.c_void_p(pick.data_ptr()),
                           ctypes.c_void_p(arena.svh2.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                           ctypes.c_void_p(wts.data_ptr()), ctypes.c_void_p(local.data_ptr()),
                           int(T), int(P), int(down_n), 1, int(K), int(arena.slots), st)
    return local


@torch.no_grad()
def _run_pipeline(x, slots, weights, arena, uids, ucount, members, maxm, nexp_max, swiglu_limit,
                  nt=NT, warps=WARPS, pf=PF, tile=None):
    """The five kernels over a prebuilt grouping. Shared by decode (P<=64) and prefill (P>64).

    `pick` is the arena slot per pair, in [T, K] order; uids/members say which pairs belong to which
    slot.  Under TP-output each rank owns half the intermediate columns and half the hidden outputs;
    the intermediate is all-gathered before down and the per-token totals after, where fp4_moe does.
    """
    T, K = slots.shape
    P = T * K
    dim, inter, down_n = arena.shapes["suh1"][0], arena.inter, arena.down_n
    world = arena.tp_world
    dev = x.device
    group = None
    if world > 1:
        from engine.collective_rails import group as _g
        group = _g()
    pick = slots.reshape(-1).to(torch.int32).contiguous()
    xd = torch.empty((P, inter), dtype=torch.float16, device=dev)
    _up(x, pick, uids, ucount, members, arena, T, K, maxm, nexp_max, swiglu_limit, xd, nt, warps, pf, tile)
    if world > 1:
        gathered = torch.empty((world * P, inter), dtype=torch.float16, device=dev)
        torch.distributed.all_gather_into_tensor(gathered, xd, group=group)
        xd = gathered.view(world, P, inter).transpose(0, 1).reshape(P, world * inter)
    local = _down(xd, pick, uids, ucount, members, weights, arena, T, K, maxm, nexp_max, nt, warps, pf, tile)
    if world == 1:
        return local
    out = torch.empty((world * T, down_n), dtype=torch.float32, device=dev)
    torch.distributed.all_gather_into_tensor(out, local, group=group)
    return out.view(world, T, down_n).transpose(0, 1).reshape(T, dim)


def decode_group(pick, T, K, arena):
    """The decode grouping (P<=64, graph-capturable): (uids, ucount, members) for `pick` [T*K] int32."""
    P = T * K
    uids = torch.empty((P,), dtype=torch.int32, device=pick.device)
    ucount = torch.empty((1,), dtype=torch.int32, device=pick.device)
    members = torch.empty((P, 16), dtype=torch.int32, device=pick.device)
    _lib().exl3m_group(ctypes.c_void_p(pick.data_ptr()), ctypes.c_void_p(uids.data_ptr()),
                       ctypes.c_void_p(ucount.data_ptr()), ctypes.c_void_p(members.data_ptr()),
                       int(T), int(K), int(arena.slots), 16, _stream())
    return uids, ucount, members


def decode_up(x, pick, group, arena, T, K, swiglu_limit, xd):
    """moe_forward's first half for the merged decode path (engine/fastdecode.py::_moe_merged_exl3)."""
    uids, ucount, members = group
    _up(x, pick, uids, ucount, members, arena, T, K, 16, T * K, swiglu_limit, xd, 8, 4, 1)


def decode_down(xd_full, pick, group, weights, arena, T, K):
    """moe_forward's second half: this rank's [T, down_n] fp32 slice, not yet gathered."""
    uids, ucount, members = group
    return _down(xd_full, pick, uids, ucount, members, weights, arena, T, K, 16, T * K, 8, 4, 1)


@torch.no_grad()
def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                swiglu_limit: float = 10.0) -> torch.Tensor:
    """Decode-sized (P<=64): group the picks with the O(P^2) kernel, then the pipeline."""
    T, K = slots.shape
    P = T * K
    if P > 64:
        raise ValueError("exl3_moe_cuda.moe_forward is decode-shaped (P<=64); use moe_forward_prefill")
    uids, ucount, members = decode_group(slots.reshape(-1).to(torch.int32).contiguous(), T, K, arena)
    return _run_pipeline(x, slots, weights, arena, uids, ucount, members, 16, P, swiglu_limit,
                         nt=8, warps=4, pf=1)


@torch.no_grad()
def moe_forward_prefill(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                        swiglu_limit: float = 10.0, block_m: int | None = None, kernel: str | None = None,
                        tile: str | None = None, routing_ids: torch.Tensor | None = None,
                        routing_slot_map: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill: group with the engine's build_routing (BM blocks, one expert each), then the pipeline.

    build_routing's pair ids are flat (row*K + k); the grouped kernel decodes members as
    row*32 + k, so encode them and leave -1 padding negative (the kernel skips it).
    """
    import fp4_moe as F4
    T, K = slots.shape
    kernel, ptile, block_m = prefill_setting(kernel, tile, block_m)
    # Compact the slot ids BEFORE routing, exactly as fp4_moe does: build_routing's cost tracks
    # n_slots, not the experts a chunk touches, and at a 13.5k-slot arena that is the dominant
    # prefill cost (~1.5 s a layer here against <1 ms compacted).  Pair ids stay original.
    if routing_ids is not None:
        # The engine's static per-layer compact ids (Model.prefill_routes): no unique() and no host
        # sync per layer. Blocks come out in another order; every pair's result is the same.
        block_slot, block_pair, NB = F4.build_routing(routing_ids, routing_slot_map.numel(), block_m,
                                                      precount=True)
        block_slot = torch.where(block_slot >= 0, routing_slot_map[block_slot.clamp_min(0)], block_slot)
    else:
        valid = slots >= 0
        uniq, inv = torch.unique(slots[valid], return_inverse=True)
        compact = torch.full_like(slots, -1, dtype=torch.int32)
        compact[valid] = inv.to(torch.int32)
        block_slot, block_pair, NB = F4.build_routing(compact, int(uniq.numel()), block_m)
        block_slot = torch.where(block_slot >= 0, uniq.to(torch.int32)[block_slot.clamp_min(0)], block_slot)
    pair = block_pair.view(NB, block_m)
    members = torch.where(pair >= 0, (pair // K) * 32 + (pair % K), pair).to(torch.int32).contiguous()
    uids = block_slot.to(torch.int32).contiguous()
    ucount = torch.full((1,), NB, dtype=torch.int32, device=slots.device)
    if ptile is not None:
        return _run_pipeline(x, slots, weights, arena, uids, ucount, members, block_m, NB, swiglu_limit,
                             nt=8, warps=4, pf=2, tile=ptile)
    # pf=2 is the grouped kernel's prefill winner (58 vs 86 ms at T=2048); decode keeps pf=1.
    return _run_pipeline(x, slots, weights, arena, uids, ucount, members, block_m, NB, swiglu_limit,
                         nt=8, warps=4, pf=2)
