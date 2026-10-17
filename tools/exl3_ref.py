"""Torch reference decoder and on-disk pack format for EXL3 routed experts.

Reference only.  This module exists so ``tools/pack_exl3_experts.py`` can build a pack and
``tools/test_exl3_ref.py`` can check the torch decode bit-for-bit against the vendored numpy
oracle (``tools/exl3_format.py``, TensorFold 0.6.0 / exllamav3).  The serving path reads the
pack through ``engine/experts.py``'s O_DIRECT machinery and does not import this file.

The weight is ``W = diag(suh) . H_K . W_q . H_N . diag(svh)`` (exllamav3), so a forward is
``y = (H_N^T ( H_K^T (x * suh) ) @ W_q ) * svh`` with ``H = hadamard(128) / sqrt(128)``:
the kernels rotate the input, multiply by W_q, then rotate the output.  ``exl3_format.forward``
is the numpy statement of the same arithmetic.  The full ``W`` decode is only for tests -- it
is K*N fp64 and never used in serving.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from functools import lru_cache

import numpy as np
import torch

MUL1_MUL = 0x83DCD12D
MCG_MUL = 0xCBAC1FED
INST3_MUL, INST3_ADD = 89226354, 64248484
MASK, FLIP = 0x8FFF8FFF, 0x3B603B60
MUL1_SCALE, MUL1_BIAS = 0x1EEE, 0xC931
HAD = 128
CODEBOOKS = ("3inst", "mcg", "mul1")
BITS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)

# --- the pack file -----------------------------------------------------------------------------
# One file per rank, records 4096-aligned so engine/experts.py can O_DIRECT one expert in one
# preadv (the FP4 store does the same via ShardFile spans).  The layout is ours, not TensorFold's
# manifest/`data.bin`: serving must not depend on their format and their prepared packs belong to
# a different ("uncensored") checkpoint.  See tools/pack_exl3_experts.py.
PACK_FORMAT = "exl3-experts"
PACK_VERSION = 1
ALIGN = 4096
# The rank slice of one expert, in the order Exl3Arena.load_slot wants it.  w1/w3 are gate/up
# [5,120 -> 2,304], w2 is down [2,304 -> 5,120]; a rank keeps half the N tiles of every matrix
# (N tiles are 128-wide Hadamard blocks, and 2,304 = 18 and 5,120 = 40 of them).
RECORD_TENSORS = ("t1", "t3", "t2", "suh1", "suh3", "suh2", "svh1", "svh3", "svh2")
# Full (both-rank) matrix shapes; the trellis last dim is 16 * bits and the K tile is 16 wide.
K13, N13 = 5120, 2304       # gate / up: K = hidden, N = intermediate
K2, N2 = 2304, 5120         # down: K = intermediate, N = hidden
KT13, NT13 = K13 // 16, N13 // 16      # 320, 144
KT2, NT2 = K2 // 16, N2 // 16          # 144, 320


def set_dims(hidden: int, inter: int) -> None:
    """Point the record shapes at another model (GLM-5.3: 6,144 hidden, 2,048 intermediate). A
    process holds one model, so module state is enough; a pack records its dims and PackReader
    applies them."""

    global K13, N13, K2, N2, KT13, NT13, KT2, NT2
    K13, N13 = int(hidden), int(inter)
    K2, N2 = int(inter), int(hidden)
    KT13, NT13 = K13 // 16, N13 // 16
    KT2, NT2 = K2 // 16, N2 // 16


def check_bits(bits: float) -> float:
    b = float(bits)
    if b not in BITS:
        raise ValueError(f"EXL3 bits must be one of {BITS}, got {bits}")
    return b


def bits_of_shape(shape) -> float:
    last = int(tuple(shape)[-1])
    return check_bits(last / 16)


# --- the codebook and the tile (torch, mirrors exl3_format) ------------------------------------

def _fp16_from_bits(u16: torch.Tensor) -> torch.Tensor:
    """Reinterpret 16-bit integer patterns as fp16, the way ``exl3_format._fp16`` does."""

    return u16.to(torch.uint16).view(torch.float16)


@lru_cache(maxsize=8)
def codebook(name: str, device: str = "cpu") -> torch.Tensor:
    """The fp16 value of every 16-bit state, [65536], for ``name`` ("3inst", "mcg" or "mul1")."""

    s = torch.arange(65536, dtype=torch.int64, device=device)
    if name == "mul1":
        x = (s * MUL1_MUL) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        scale = _fp16_from_bits(torch.tensor([MUL1_SCALE], device=device)).to(torch.float64)
        bias = _fp16_from_bits(torch.tensor([MUL1_BIAS], device=device)).to(torch.float64)
        # exact in float64, so one rounding, same as the numpy reference
        return (h.to(torch.float64) * scale + bias).to(torch.float16)
    if name == "mcg":
        x = (s * MCG_MUL) & 0xFFFFFFFF
    elif name == "3inst":
        x = (s * INST3_MUL + INST3_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(f"unknown EXL3 codebook {name!r} (known: {', '.join(CODEBOOKS)})")
    x = (x & MASK) ^ FLIP
    hi = _fp16_from_bits(x & 0xFFFF).to(torch.float64)
    lo = _fp16_from_bits(x >> 16).to(torch.float64)
    return (hi + lo).to(torch.float16)          # the float64 sum is exact: one fp16 rounding


@lru_cache(maxsize=16)
def stream_ends(bits: float, device: str = "cpu") -> torch.Tensor:
    """E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive)."""

    b = check_bits(bits)
    p1 = torch.arange(1, 257, dtype=torch.int64, device=device)
    if float(b).is_integer():
        return p1 * int(b)
    k2 = int(2 * b)
    return (p1 * k2 - (p1 % 2)) // 2


@lru_cache(maxsize=1)
def _tile_positions() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(row, column) in its 16x16 tile of each stream position 0..255 (matches exl3_format.tile_positions)."""

    p = np.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    flat = rows * 16 + cols
    assert sorted(flat.tolist()) == list(range(256))     # a permutation, so the scatter is exact
    return tuple(int(v) for v in rows), tuple(int(v) for v in cols)


def tile_words(bits: float) -> int:
    return int(16 * check_bits(bits))


def states(trellis: torch.Tensor, bits: float) -> torch.Tensor:
    """The 16-bit state of every value: [..., 256] int64 in stream order, from int16 [..., 16*bits]."""

    nw16 = tile_words(bits)
    if trellis.dtype != torch.int16 or trellis.shape[-1] != nw16:
        raise ValueError(f"trellis must be int16 [..., {nw16}] for {bits} bits, "
                         f"got {trellis.dtype} {tuple(trellis.shape)}")
    w = trellis.to(torch.int64) & 0xFFFF
    words = w[..., 0::2] | (w[..., 1::2] << 16)          # little-endian pairs: 8*bits 32-bit words
    nw = nw16 // 2
    ring = 32 * nw
    first = stream_ends(bits, str(trellis.device)) - 16 + ring
    i0 = (first // 32) % nw
    off = first % 32
    i1 = (i0 + 1) % nw
    pair = ((words[..., i0] & 0xFFFFFFFF) << 32) | (words[..., i1] & 0xFFFFFFFF)
    return (pair >> (48 - off)) & 0xFFFF                 # arithmetic shift: the low 16 bits are exact


def unpack(trellis: torch.Tensor, bits: float, codebook_name: str, chunk: int = 16) -> torch.Tensor:
    """W_q [K, N] fp16 in the rotated domain from a rank's trellis int16 [K/16, N/16, 16*bits]."""

    if trellis.ndim != 3:
        raise ValueError(f"trellis must be [K/16, N/16, 16*bits], got shape {tuple(trellis.shape)}")
    kt, nt = trellis.shape[0], trellis.shape[1]
    table = codebook(codebook_name, str(trellis.device))
    rows, cols = _tile_positions()
    flat = torch.tensor([r * 16 + c for r, c in zip(rows, cols)],
                        dtype=torch.int64, device=trellis.device)
    w = torch.empty((kt, 16, nt, 16), dtype=torch.float16, device=trellis.device)
    for k0 in range(0, kt, chunk):
        c = min(chunk, kt - k0)
        st = states(trellis[k0:k0 + c], bits)            # [c, nt, 256]
        vals = table[st]                                 # [c, nt, 256]
        blk = torch.empty((c, nt, 256), dtype=torch.float16, device=trellis.device)
        blk[..., flat] = vals
        w[k0:k0 + c] = blk.reshape(c, nt, 16, 16).permute(0, 2, 1, 3)
    return w.reshape(kt * 16, nt * 16)


@lru_cache(maxsize=4)
def hadamard(n: int = HAD, device: str = "cpu") -> torch.Tensor:
    """The n x n Sylvester Hadamard matrix of +1/-1: H[i, j] = (-1)^popcount(i & j)."""

    i = torch.arange(n, device=device)
    bits = torch.tensor([bin(v).count("1") & 1 for v in range(n)], device=device)
    return torch.where(bits[(i[:, None] & i[None, :])] == 1, -1.0, 1.0)


def rotate(x: torch.Tensor, axis: int) -> torch.Tensor:
    """H / sqrt(128) applied to every block of 128 along ``axis``, in float64."""

    h = hadamard(HAD, str(x.device)).to(torch.float64) / (HAD ** 0.5)
    x = x.movedim(axis, -1).to(torch.float64)
    shape = x.shape
    if shape[-1] % HAD:
        raise ValueError(f"the rotated dimension must be a multiple of {HAD}, got {shape[-1]}")
    x = (x.reshape(*shape[:-1], shape[-1] // HAD, HAD) @ h).reshape(shape)
    return x.movedim(-1, axis)


def dequantize(trellis, suh, svh, bits: float, codebook_name: str) -> torch.Tensor:
    """The rank's weight [K, N] float64: diag(suh) @ H_K @ W_q @ H_N @ diag(svh)."""

    wq = unpack(_t(trellis), bits, codebook_name).to(torch.float64)
    w = rotate(wq, 0) * _t(suh).to(torch.float64)[:, None]
    return rotate(w, 1) * _t(svh).to(torch.float64)[None, :]


def forward(x, trellis, suh, svh, bits: float, codebook_name: str, bias=None) -> torch.Tensor:
    """y = x @ W + bias in float64, in the kernels' order: rotate in, W_q, rotate out."""

    xh = rotate(_t(x).to(torch.float64) * _t(suh).to(torch.float64), -1)
    y = rotate(xh @ unpack(_t(trellis), bits, codebook_name).to(torch.float64), -1) * _t(svh).to(torch.float64)
    return y if bias is None else y + _t(bias).to(torch.float64)


def _t(v):
    if isinstance(v, torch.Tensor):
        return v
    return torch.from_numpy(np.ascontiguousarray(v))


# --- the rank slice and the pack header --------------------------------------------------------

def _n_split(n: int, rank: int, world: int) -> int:
    assert n % world == 0, (n, world)
    return n // world


def record_shapes(bits: float, rank: int, world: int, words: int | None = None) -> dict[str, tuple[int, ...]]:
    """Each record tensor's shape for one rank.  A rank holds the N-tile slice [rank*N/world, +N/world)."""

    w = words if words is not None else tile_words(bits)
    nt13 = _n_split(NT13, rank, world)       # N tiles a rank of w1/w3 (72)
    nt2 = _n_split(NT2, rank, world)         # N tiles a rank of w2 (160)
    n13 = _n_split(N13, rank, world)         # N columns a rank of w1/w3 (1152)
    n2 = _n_split(N2, rank, world)           # N columns a rank of w2 (2560)
    return {
        "t1": (KT13, nt13, w), "t3": (KT13, nt13, w),
        "t2": (KT2, nt2, w),
        "suh1": (K13,), "suh3": (K13,), "suh2": (K2,),
        "svh1": (n13,), "svh3": (n13,), "svh2": (n2,),
    }


_RECORD_DTYPES = {"t1": np.int16, "t3": np.int16, "t2": np.int16,
                  "suh1": np.float16, "suh3": np.float16, "suh2": np.float16,
                  "svh1": np.float16, "svh3": np.float16, "svh2": np.float16}


def record_nbytes(bits: float, rank: int, world: int) -> int:
    shapes = record_shapes(bits, rank, world)
    return sum(int(np.prod(shapes[name])) * np.dtype(_RECORD_DTYPES[name]).itemsize
               for name in RECORD_TENSORS)


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def build_header(bits_map: dict[int, float], rank: int, world: int, codebook: str,
                 source_sha256: str, n_experts: int = 384) -> tuple[bytes, int, dict[tuple[int, int], dict]]:
    """(header bytes, data offset, {(layer, expert): {"off", "nbytes", "bits"}}) with every record 4096-aligned.

    Records are laid out layer-major then expert, each padded up to ``ALIGN``, so the data offset
    and every record offset are aligned and O_DIRECT can read an expert in one preadv.  ``n_experts``
    is 384 for a real pack; a smaller value builds a self-consistent test pack.
    """

    records: dict[tuple[int, int], dict] = {}
    off = 0
    for layer in sorted(bits_map):
        for expert in range(n_experts):
            nb = record_nbytes(bits_map[layer], rank, world)
            records[(layer, expert)] = {"off": off, "nbytes": nb, "bits": bits_map[layer]}
            off += _align_up(nb)
    data_nbytes = off
    header = {
        "format": PACK_FORMAT, "version": PACK_VERSION,
        "rank": rank, "world": world, "codebook": codebook,
        "bits": {str(L): bits_map[L] for L in sorted(bits_map)},
        "record_align": ALIGN, "tensors": list(RECORD_TENSORS),
        "source_sha256": source_sha256, "data_nbytes": data_nbytes, "dims": [K13, N13],
        "records": {f"{L},{E}": r for (L, E), r in records.items()},
    }
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    return blob, _align_up(8 + len(blob)), records


def write_pack(path: str, bits_map: dict[int, float], rank: int, world: int, codebook: str,
               source_sha256: str, records_iter, n_experts: int = 384):
    """Stream a pack.  ``records_iter`` yields ``((layer, expert), {tensor_name: ndarray})`` in any order;
    the header offsets are already fixed, so each record is written with ``pwrite`` at its own offset."""

    header, data0, records = build_header(bits_map, rank, world, codebook, source_sha256, n_experts)
    end = data0
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
        f.truncate(data0)                                # zero-fill the header padding
        for (key, tensors), meta in _ordered(records_iter, records):
            parts = []
            shapes = record_shapes(meta["bits"], rank, world)
            for name in RECORD_TENSORS:
                a = np.ascontiguousarray(tensors[name])
                assert a.shape == shapes[name], f"{key} {name}: {a.shape} != {shapes[name]}"
                assert a.dtype == np.dtype(_RECORD_DTYPES[name]), f"{key} {name}: {a.dtype}"
                parts.append(a.tobytes())
            blob = b"".join(parts)
            assert len(blob) == meta["nbytes"], (key, len(blob), meta["nbytes"])
            f.seek(data0 + meta["off"])
            f.write(blob)
            end = max(end, data0 + meta["off"] + meta["nbytes"])
        f.truncate(_align_up(end))
    return path


def _ordered(records_iter, records):
    for key, tensors in records_iter:
        yield (key, tensors), records[key]


class PackReader:
    """Read a pack with ordinary file I/O (tests and the offline tool).  Serving reads the same
    bytes through engine/experts.py's O_DIRECT staging, not through this class."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header_bytes = f.read(n)
        self.header = json.loads(self.header_bytes)
        if self.header.get("format") != PACK_FORMAT:
            raise ValueError(f"{path}: not an {PACK_FORMAT} pack")
        if "dims" in self.header:
            set_dims(*self.header["dims"])
        self.data0 = _align_up(8 + len(self.header_bytes))
        self.records = {tuple(int(v) for v in k.split(",")): v
                        for k, v in self.header["records"].items()}
        self.bits_map = {int(k): float(v) for k, v in self.header["bits"].items()}
        self.rank = int(self.header["rank"])
        self.world = int(self.header["world"])
        self.codebook = self.header["codebook"]

    def header_sha256(self) -> str:
        """The boot-guard identity: sha256 of the header bytes, cheap and rank-comparable."""

        return hashlib.sha256(self.header_bytes).hexdigest()

    def read_expert(self, layer: int, expert: int) -> dict[str, np.ndarray]:
        meta = self.records[(layer, expert)]
        bits = float(meta["bits"])
        shapes = record_shapes(bits, self.rank, self.world)
        with open(self.path, "rb") as f:
            f.seek(self.data0 + meta["off"])
            blob = f.read(meta["nbytes"])
        out, off = {}, 0
        for name in RECORD_TENSORS:
            shape = shapes[name]
            n = int(np.prod(shape)) * np.dtype(_RECORD_DTYPES[name]).itemsize
            out[name] = np.frombuffer(blob, dtype=_RECORD_DTYPES[name], count=int(np.prod(shape)),
                                      offset=off).reshape(shape)
            off += n
        return out


def manifest_sha256(model_dir: str) -> str:
    """A fast, reproducible identity for a safetensors source: the sorted headers + config + index.

    Hashing 198 GB of weights is not useful here -- the header manifest pins the exact tensors,
    dtypes, shapes and byte ranges, and it is what the pack tool actually reads.
    """

    import glob
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            h.update(os.path.basename(path).encode())
            h.update(f.read(size))
    for name in ("config.json", "model.safetensors.index.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            h.update(name.encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()
