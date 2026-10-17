"""EXL3 routed experts for GlmTP (MUSE_EXPERT_FORMAT=exl3).

The routed experts of the decoder MoE layers come from a per-rank pack built by
tools/glm_pack_exl3.py (3-bit EXL3, each record this rank's TP-output slice, 4096-aligned) and
live in tools/exl3_moe.py's Exl3Arena; the V4.1 CUDA kernels (tools/exl3_moe_cuda) compute them.
Shared experts, the dense layers' MLPs and the MTP layer stay MXFP4 in the engine's own Arena.

The kernels' TP collectives are V4.1's; here the two all-gathers (intermediate before down, the
hidden slices after) go through GlmTP's Comm like the MXFP4 path's.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOOLS))

import exl3_ref as R  # noqa: E402

INF = float("inf")
PIECE = 256 * 1024


class Exl3Pack:
    """One rank's pack: O_DIRECT-free positional reads (pread releases the GIL; the engine's pools
    issue them in parallel), one preadv an expert."""

    def __init__(self, path: str, hidden: int, inter: int):
        R.set_dims(hidden, inter)
        self.reader = R.PackReader(path)                 # header; applies the pack's dims
        h = self.reader.header
        if h.get("dims") != [hidden, inter]:
            raise ValueError(f"{path}: pack dims {h.get('dims')} != model {[hidden, inter]}")
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.records = self.reader.records
        self.data0 = self.reader.data0
        self.rank, self.world = self.reader.rank, self.reader.world
        self.layers = sorted({L for L, _ in self.records})
        self._layout = {}

    def header_sha(self) -> str:
        return self.reader.header_sha256()

    def bits(self, L: int) -> float:
        return float(self.reader.bits_map[L])

    def _offsets(self, bits):
        lay = self._layout.get(bits)
        if lay is None:
            shapes = R.record_shapes(bits, self.rank, self.world)
            lay, off = [], 0
            for name in R.RECORD_TENSORS:
                dt = np.dtype(R._RECORD_DTYPES[name])
                n = int(np.prod(shapes[name]))
                lay.append((name, dt, n, shapes[name], off))
                off += n * dt.itemsize
            self._layout[bits] = lay
        return lay

    def drop_cache(self):
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def read(self, L: int, e: int, urgent: bool = False) -> tuple[dict, float]:
        """(record of CPU tensors in RECORD_TENSORS order, bits) for routed expert e of layer L.
        Urgent reads (a decode step waits on them) go in 256 KB pieces on their own pool: one
        preadv streams ~1 GB/s from a cold file, parallel pieces keep the NVMe queue deep."""
        meta = self.records[(L, e)]
        bits = float(meta["bits"])
        n, base = meta["nbytes"], self.data0 + meta["off"]
        buf = bytearray(n)
        mv = memoryview(buf)
        if urgent and n > 2 * PIECE:
            if not hasattr(self, "_pieces"):
                from concurrent.futures import ThreadPoolExecutor
                self._pieces = ThreadPoolExecutor(max_workers=16)
            spans = [(o, min(PIECE, n - o)) for o in range(0, n, PIECE)]
            got = sum(self._pieces.map(lambda sp: os.preadv(self.fd, [mv[sp[0]:sp[0] + sp[1]]], base + sp[0]), spans))
        else:
            got = os.preadv(self.fd, [buf], base)
        if got != n:
            raise IOError(f"{self.path}: short read {got} of {n} for {(L, e)}")
        rec = {}
        for name, dt, n, shape, off in self._offsets(bits):
            rec[name] = torch.from_numpy(np.frombuffer(buf, dtype=dt, count=n, offset=off).reshape(shape))
        return rec, bits


def make_arena(slots: int, rank: int, world: int, device, hidden: int, inter: int):
    R.set_dims(hidden, inter)
    import exl3_moe as M
    M.DIM, M.INTER = hidden, inter
    return M.Exl3Arena(slots, device=device, tp_rank=rank, tp_world=world, bits=3.0, codebook="mul1")


def slot_bytes(rank: int, world: int, hidden: int, inter: int, bits: float = 3.0) -> int:
    R.set_dims(hidden, inter)
    return R.record_nbytes(bits, rank, world)


def copy_slot(arena, dst: int, src: int) -> None:
    """Device copy of one slot's tensors (a kept hot read moving out of its transient slot)."""
    for name in arena._tensors:
        if name.startswith("t"):
            arena.trellis(name, dst).copy_(arena.trellis(name, src))
        else:
            getattr(arena, name)[dst].copy_(getattr(arena, name)[src])
    arena.bits[dst] = arena.bits[src]
    arena.bits_gpu[dst].copy_(arena.bits_gpu[src])


def warm(arena) -> None:
    import exl3_moe_cuda as C
    C.warm(arena)


@torch.no_grad()
def routed(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena, comm) -> torch.Tensor:
    """sum_k weights[:, k] * expert_{slots[:, k]}(x) over the routed picks, fp32 [T, hidden], both
    ranks' halves gathered. x bf16 [T, hidden]; slots int [T, k] arena indices (a zero-weight
    pick may name any loaded slot); weights fp32 [T, k]. Graph-capturable for decode shapes."""
    import exl3_moe_cuda as C
    T, K = slots.shape
    P = T * K
    pick = slots.reshape(-1).to(torch.int32).contiguous()
    wts = weights.float().contiguous()
    if P <= 64:
        uids, ucount, members = C.decode_group(pick, T, K, arena)
        maxm, nexp, nt, warps, pf, tile = 16, P, 8, 4, 1, None
    else:
        import fp4_moe as F4
        block_m = C.PREFILL_BM
        uniq, inv = torch.unique(pick.view(T, K), return_inverse=True)
        block_slot, block_pair, NB = F4.build_routing(inv.to(torch.int32), int(uniq.numel()), block_m)
        block_slot = torch.where(block_slot >= 0, uniq.to(torch.int32)[block_slot.clamp_min(0)], block_slot)
        pair = block_pair.view(NB, block_m)
        members = torch.where(pair >= 0, (pair // K) * 32 + (pair % K), pair).to(torch.int32).contiguous()
        uids = block_slot.to(torch.int32).contiguous()
        ucount = torch.full((1,), NB, dtype=torch.int32, device=x.device)
        maxm, nexp, nt, warps, pf = block_m, NB, 8, 4, 2
        tile = tuple(int(v) for v in C.PREFILL_TILE.split("x"))
    xd = torch.empty((P, arena.inter), dtype=torch.float16, device=x.device)
    C._up(x.contiguous(), pick, uids, ucount, members, arena, T, K, maxm, nexp, INF, xd, nt, warps, pf, tile)
    xd_full = comm.cat_last(xd).contiguous()                  # [P, world * inter], rank-major columns
    local = C._down(xd_full, pick, uids, ucount, members, wts, arena, T, K, maxm, nexp, nt, warps, pf, tile)
    return comm.cat_last(local)                               # [T, hidden]
