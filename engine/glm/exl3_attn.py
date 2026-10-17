"""GLM attention projections from the EXL3 checkpoint's 5-bit matrices (MUSE_EXL3_ATTN=1).

The Quark checkpoint's attention is FP8 with 128x128 block scales; bertholomus/GLM-5.3-EXL3-3.0bpw
stores q_a, q_b, kv_a, o_proj and the indexer's wq_b at 5 bits (EXL3 trellis, mul1 codebook,
128-wide Hadamard rotations on both sides): ~37% fewer bytes a decode step reads, and ~2.4 GB a
rank less to keep resident.

Decode (<= 16 rows) runs V4.1's dense path (engine/exl3_dense.Exl3Matrix: input rotation, the
grouped split-K GEMV, combine + output rotation). Prefill unpacks the trellis to its FP16 matrix
once a call (Triton, exact: the codebook is a table of the reference's values) and runs the
rotations on the activations around a cuBLAS GEMM:  y = H_N( (H_K(x * suh))_fp16 @ W_q ) * svh.
"""

from __future__ import annotations

import functools
import os
import sys

import torch
import triton
import triton.language as tl

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOOLS))


@triton.jit
def _unpack_kernel(TR, W, NT, N, SCALE, BIAS, W16: tl.constexpr, BITS: tl.constexpr, BC: tl.constexpr):
    """Rows of K-tile kt, columns [c0, c0 + BC) of W [K, N] fp16, computed in output order so the
    stores are contiguous: each element finds its stream position p (the inverse of the tile's
    lane/j layout), its 16-bit state in the tile's bitstream, and the mul1 codebook value --
    1024 + the four bytes of state * 0x83DCD12D, times scale plus bias: exact in fp32 here (a
    multiple of 2^-18 below 16 needs 22 bits), so one rounding to fp16, as the reference does."""
    kt = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, 16)
    nt = c // 16
    col = c % 16
    j = (r[:, None] & 1) | (((r[:, None] >> 3) & 1) << 1) | ((col[None, :] >> 3) << 2)
    lane = ((col[None, :] & 7) << 2) | ((r[:, None] >> 1) & 3)
    p = lane * 8 + j
    nw: tl.constexpr = W16 // 2
    # The int16 pairs are little-endian 32-bit words: read them as such.
    base = TR.to(tl.pointer_type(tl.uint32), bitcast=True) + (kt.to(tl.int64) * NT + nt[None, :]) * nw
    first = (p + 1) * BITS - 16 + 32 * nw
    i0 = (first // 32) % nw
    off = (first % 32).to(tl.uint32)
    i1 = (i0 + 1) % nw
    w0 = tl.load(base + i0)
    w1 = tl.load(base + i1)
    # The 16 bits ending `off` bits into word i1's predecessor pair (w0:w1), with 32-bit shifts.
    hi = tl.where(off >= 16, w0 << (off - 16), w0 >> (16 - off))
    lo = tl.where(off > 16, w1 >> (48 - off), 0)
    state = (hi | lo) & 0xFFFF
    x = state * 0x83DCD12D                                   # uint32: wraps mod 2^32
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    v = h.to(tl.float32) * SCALE + BIAS
    tl.store(W + (kt.to(tl.int64) * 16 + r[:, None]) * N + c[None, :], v.to(tl.float16))


@functools.lru_cache(maxsize=1)
def _mul1_consts():
    import exl3_ref as R
    f = lambda b: float(torch.tensor([b], dtype=torch.int32).to(torch.uint16).view(torch.float16))
    return f(R.MUL1_SCALE), f(R.MUL1_BIAS)


def unpack(trellis: torch.Tensor, bits: int) -> torch.Tensor:
    """int16 [K/16, N/16, 16*bits] -> W_q fp16 [K, N] (the rotated-domain matrix, exl3_ref.unpack)."""
    KT, NT, W16 = trellis.shape
    assert W16 == 16 * bits and float(bits).is_integer(), (trellis.shape, bits)
    N = NT * 16
    w = torch.empty((KT * 16, N), dtype=torch.float16, device=trellis.device)
    BC = 128 if N % 128 == 0 else 64 if N % 64 == 0 else 16
    scale, bias = _mul1_consts()
    _unpack_kernel[(KT, N // BC)](trellis, w, NT, N, scale, bias, W16=W16, BITS=int(bits), BC=BC, num_warps=4)
    return w


@functools.lru_cache(maxsize=2)
def _hadamard(device: str) -> torch.Tensor:
    import exl3_ref as R
    return (R.hadamard(128, device).float() / 128 ** 0.5).contiguous()


def rotate(x: torch.Tensor) -> torch.Tensor:
    """H/sqrt(128) over each 128-block of the last dim, fp32."""
    T, n = x.shape
    return (x.float().view(T * (n // 128), 128) @ _hadamard(str(x.device))).view(T, n)


class Exl3Lin:
    """One rank's EXL3 projection, Lin's interface: y = x @ W, W [K, N] from the trellis.

    Decode (<= 16 rows): engine/exl3_dense.Exl3Matrix (rotate in, grouped split-K GEMV, combine +
    rotate out). Prefill: the trellis unpacked to fp16 once a call, rotations on the activations,
    cuBLAS with fp32 accumulation. `n_out` trims a padded N (kv_a: 576 of 640)."""

    def __init__(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, split: int,
                 n_out: int | None = None):
        from engine.exl3_dense import Exl3Matrix
        self.tr = trellis.contiguous()
        self.bits = int(self.tr.shape[-1] // 16)
        self.suh, self.svh = suh.half().contiguous(), svh.half().contiguous()
        self.K, self.N = int(self.suh.numel()), int(self.svh.numel())
        assert self.tr.shape[0] * 16 == self.K and self.tr.shape[1] * 16 == self.N, (self.tr.shape, self.K, self.N)
        self.n_out = n_out or self.N
        self.dec = Exl3Matrix([self.tr], self.suh, self.svh, [self.bits], split, _members(str(self.tr.device)))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.tr, self.suh, self.svh))

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.K)
        T = x2.shape[0]
        if T <= 16:
            y = self.dec(x2.to(torch.bfloat16), out_dtype=out_dtype)
        else:
            w = unpack(self.tr, self.bits)
            xh = rotate(x2.float() * self.suh.float()).to(torch.float16)
            y = rotate(torch.mm(xh, w, out_dtype=torch.float32)) * self.svh.float()
            y = y.to(out_dtype)
        if self.n_out != self.N:
            y = y[:, :self.n_out]
        return y.reshape(*shape[:-1], self.n_out)


@functools.lru_cache(maxsize=2)
def _members(device: str):
    from engine.exl3_dense import _Members
    return _Members(torch.device(device))


class Exl3Source:
    """The EXL3 checkpoint's attention tensors, sliced for one rank (MUSE_EXL3_SRC, mounted at
    /exl3src). Splits are on 128 boundaries, so each rank's matrix is an exact slice."""

    def __init__(self, path: str):
        import json
        self.dir = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def _get(self, name):
        from safetensors import safe_open
        with safe_open(os.path.join(self.dir, self.map[name]), "pt") as f:
            return f.get_tensor(name)

    def drop_cache(self):
        """The safetensors reads went through the page cache, which on GB10 is the GPU's memory."""
        for f in set(self.map.values()):
            try:
                fd = os.open(os.path.join(self.dir, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass

    def has(self, prefix) -> bool:
        return prefix + ".trellis" in self.map

    def lin(self, prefix: str, device, split: int, rank: int = 0, world: int = 1, cut: str | None = None,
            n_out: int | None = None) -> Exl3Lin:
        tr, suh, svh = self._get(prefix + ".trellis"), self._get(prefix + ".suh"), self._get(prefix + ".svh")
        if cut == "n":          # output rows (heads): N tiles, svh
            nt = tr.shape[1] // world
            tr, svh = tr[:, rank * nt:(rank + 1) * nt], svh[rank * nt * 16:(rank + 1) * nt * 16]
        elif cut == "k":        # input columns (heads): K tiles, suh; the output is a partial sum
            kt = tr.shape[0] // world
            tr, suh = tr[rank * kt:(rank + 1) * kt], suh[rank * kt * 16:(rank + 1) * kt * 16]
        return Exl3Lin(tr.contiguous().to(device), suh.to(device), svh.to(device), split, n_out)
