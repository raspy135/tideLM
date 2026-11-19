"""exlr8 v2 (kindlingai/glm-5.3-exlr8-k2-k3-k4): the reference decode of a routed expert's units.

Written from the format's spec (the model card's Loader section and kindlingai/exlr8 FORMAT.md),
numpy only; tools/exlr8_check.py holds it against the publisher's loader on real bytes. What the
kernels and the packer must reproduce:

  unit       128 intermediate channels of one expert: unit u of gate / up is W[:, 128u:128u+128]
             (H x 128), of down W[128u:128u+128, :] (128 x H), W in (in, out) orientation. Each
             decodes on its own, at its own width K (2, 3, 4): B = H * 16 * K bytes of tiles.
  tile       16 x 16 weights in 32K bytes, EXL3's packing: 16K little-endian int16 words, adjacent
             words swapped, bits MSB first, K bits a value, 256 values; the state of value i is the
             16 bits that end with it (tail-biting).
  codebook   mcg: x = (state * 0xCBAC1FED) mod 2^32, x = (x & 0x8FFF8FFF) ^ 0x3B603B60, the two
             halves of x added as fp16; mul1 (the release's experts since 0.6, 2026-11-18):
             x = (state * 0x83DCD12D) mod 2^32, fp16(1024 + its four bytes' sum) * fp16 0x1EEE +
             fp16 0xC931 in one rounding - and then, either way, ROUNDED TO E4M3 (nearest even,
             saturating). The rounding is the format: the encoder searched over the rounded values, so
             every weight of T is exactly an FP8 value, and a decoder without it reads other weights.
             The tensor name carries the codebook: `trellis.gate` is mcg, `trellis.gate.mul1` mul1
             (SafeFile.trellis); the release's dense trellis matrices are still mcg.
  matrix     W = diag(suh) Hb T Hb diag(svh), Hb the block-diagonal 128 x 128 Hadamard / sqrt(128).
             gate / up: suh = signs.in * su_block[e] (one scale a 128-block), svh the unit's 128
             channel values. down: suh the unit's channel values, svh = signs.out * sv_block[e].

What that buys the engine (and EXL3 could not give): signs.in is the layer's and su_block is
constant over a Hadamard block, so ((x * signs.in) Hb) is computed once a token and each expert
only scales its 48 blocks; the down side sums the picks' block-scaled outputs before one Hb and
signs.out. And a down unit reads its own 128 intermediate channels only, so a rank that owns a
unit's gate / up owns its down: the ranks exchange partial hidden rows, not intermediates.

Sharding (FORMAT.md): unit (e, u) of layer L belongs to rank (16 e + u + L) mod N; the shared
expert is e = 256. At N = 2 a rank owns the units of one parity, alternating by layer.
"""

from __future__ import annotations

import json
import struct
import urllib.request

import numpy as np

MCG_MUL = 0xCBAC1FED
MUL1_MUL = 0x83DCD12D
CODEBOOKS = ("mcg", "mul1")
HAD = 128
UNIT = 128
DATA0 = 65536
PROJS = ("gate", "up", "down")
REPO = "https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/resolve/main/"


def unit_bytes(hidden: int, k: int) -> int:
    """A unit's tile bytes at width k: (hidden / 16) * 8 tiles of 32k bytes."""
    return hidden * 16 * k


def owner(expert: int, unit: int, layer: int, world: int, units: int = 16) -> int:
    """The rank that holds unit `unit` of `expert` (the shared expert: the layer's expert count)."""
    return (units * expert + unit + layer) % world


def e4m3_round(v: np.ndarray) -> np.ndarray:
    """Round to FP8 E4M3 (float8_e4m3fn: no infinities, largest 448), nearest even, saturating."""
    v = np.asarray(v, np.float64)
    m = np.minimum(np.abs(v), 448.0)
    # the spacing at m: 2^(e-3) in the binade 2^e, 2^-9 below the smallest normal 2^-6
    e = np.floor(np.log2(np.maximum(m, 2.0 ** -6)))
    step = np.exp2(e - 3)
    return (np.copysign(np.rint(m / step) * step, v)).astype(np.float32)


def e4m3_bits(v: np.ndarray) -> np.ndarray:
    """The E4M3 byte of values e4m3_round returned (sign, 4 exponent bits biased 7, 3 mantissa bits)."""
    v = np.asarray(v, np.float64)
    m = np.abs(v)
    e = np.floor(np.log2(np.maximum(m, 2.0 ** -6)))
    man = np.rint(m / np.exp2(e - 3)).astype(np.int64)          # 8..15 normal, 0..7 subnormal
    sub = m < 2.0 ** -6
    code = np.where(sub, man, ((e.astype(np.int64) + 7) << 3) | (man & 7))
    return (code | (np.signbit(v).astype(np.int64) << 7)).astype(np.uint8)


def codebook(name: str = "mcg") -> np.ndarray:
    """All 65,536 states' weights under codebook `name` (mcg, mul1), fp32 and exactly E4M3.

    mcg's fp16 sum is one rounding of the exact sum (float64 holds it exactly), as the fp16 add of the
    kernels and of exllamav3 is; mul1's fp16 fma likewise one rounding of the exact h * c + d (the
    publisher's loader computes it in float64, exllamav3 and the kernels with __hfma2). E4M3 then rounds
    that fp16 value."""
    s = np.arange(1 << 16, dtype=np.uint64)
    if name == "mcg":
        x = (s * MCG_MUL) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
        hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
        return e4m3_round((lo + hi).astype(np.float16))
    if name == "mul1":
        x = (s * MUL1_MUL) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
        c = np.array(0x1EEE, np.uint16).view(np.float16).astype(np.float64)
        d = np.array(0xC931, np.uint16).view(np.float16).astype(np.float64)
        return e4m3_round((h.astype(np.float64) * c + d).astype(np.float16))
    raise ValueError(f"codebook {name}: not one of {CODEBOOKS}")


_CB: dict = {}
_PERM = None


def tile_perm() -> np.ndarray:
    """perm[j] = 16 * row + col of the tile element at stream position j (rows the input side)."""
    perm = []
    for t in range(32):
        r0, c0 = 2 * (t % 4), t // 4
        for r, c in ((r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                     (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)):
            perm.append(16 * r + c)
    return np.asarray(perm)


def tile_states(raw: np.ndarray, k: int) -> np.ndarray:
    """Packed tiles uint8 [n, 32k] -> the 256 trellis states of each, uint16 [n, 256] (stream order)."""
    n = raw.shape[0]
    # a pair of words (a0 a1 | a2 a3, little-endian) swapped and read MSB first is a3 a2 a1 a0
    bits = np.unpackbits(np.ascontiguousarray(raw.reshape(n, 8 * k, 4)[:, :, ::-1]).reshape(n, 32 * k), axis=1)
    vals = (bits.reshape(n, 256, k).astype(np.uint32) << np.arange(k - 1, -1, -1, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    st = np.zeros_like(vals)
    for j in range(-(-16 // k)):                       # the 16 bits that end with value i, wrapping
        st |= np.roll(vals, j, axis=1) << np.uint32(j * k)
    return (st & 0xFFFF).astype(np.uint16)


def decode_tiles(raw: np.ndarray, k: int, cb: str = "mcg") -> np.ndarray:
    """Packed tiles uint8 [n, 32k] -> fp32 [n, 16, 16], (input, output) within a tile, codebook `cb`."""
    global _PERM
    if cb not in _CB:
        _CB[cb] = codebook(cb)
    if _PERM is None:
        _PERM = tile_perm()
    out = np.empty((raw.shape[0], 256), np.float32)
    out[:, _PERM] = _CB[cb][tile_states(raw, k)]
    return out.reshape(-1, 16, 16)


def unit_tiles(raw: np.ndarray, k: int, hidden: int, down: bool, cb: str = "mcg") -> np.ndarray:
    """A unit's bytes -> T: gate / up (hidden, 128), tiles [k16][c]; down (128, hidden), tiles [b][r][c]."""
    t = decode_tiles(np.frombuffer(raw, np.uint8).reshape(-1, 32 * k), k, cb)
    if down:
        return t.reshape(hidden // 128, 8, 8, 16, 16).transpose(1, 3, 0, 2, 4).reshape(128, hidden)
    return t.reshape(hidden // 16, 8, 16, 16).transpose(0, 2, 1, 3).reshape(hidden, 128)


def hadamard(n: int = HAD) -> np.ndarray:
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(np.float32(n))).astype(np.float32)


def rotate(x: np.ndarray) -> np.ndarray:
    """x Hb over the last axis (a multiple of 128): each 128-block through the Hadamard."""
    return (x.reshape(*x.shape[:-1], -1, HAD) @ hadamard()).reshape(x.shape)


def reconstruct(t: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """W (in, out) = diag(suh) Hb T Hb diag(svh), fp32, in the reference's order of operations."""
    k, n = t.shape
    had = hadamard()
    x = (had @ t.reshape(-1, HAD, n)).reshape(k, n) * suh.astype(np.float32)[:, None]
    return (x.reshape(k, -1, HAD) @ had).reshape(k, n) * svh.astype(np.float32)[None, :]


def gate_up_unit(raw, k: int, signs_in: np.ndarray, su_block: np.ndarray, svh: np.ndarray, cb: str = "mcg") -> np.ndarray:
    """One gate / up unit as W (hidden, 128): signs.in [H], the expert's block scales [H/128], the unit's svh [128]."""
    hidden = signs_in.shape[0]
    suh = signs_in.astype(np.float32) * np.repeat(su_block.astype(np.float32), 128)
    return reconstruct(unit_tiles(raw, k, hidden, False, cb), suh, svh)


def down_unit(raw, k: int, suh: np.ndarray, signs_out: np.ndarray, sv_block: np.ndarray, cb: str = "mcg") -> np.ndarray:
    """One down unit as W (128, hidden): the unit's suh [128], signs.out [H], the expert's block scales [H/128]."""
    hidden = signs_out.shape[0]
    svh = signs_out.astype(np.float32) * np.repeat(sv_block.astype(np.float32), 128)
    return reconstruct(unit_tiles(raw, k, hidden, True, cb), suh, svh)


# ------------------------------------------------------------- the engine's order of operations
def forward_units(x: np.ndarray, signs_in, signs_out, units, cb: str = "mcg") -> np.ndarray:
    """What the kernels compute for one token's hidden x [H] over a set of units, as the engine
    orders it: one input rotation, each unit's block scales and tiles, one output rotation.

    units: (weight, gate_raw, up_raw, down_raw, (kg, ku, kd), su_gate, su_up, sv_down, svh_gate,
    svh_up, suh_down) each. Returns sum_units weight * down(silu(gate(x)) * up(x)) [H] - the partial
    hidden row of a rank that owns these units. `cb`: the units' codebook."""
    hidden = x.shape[0]
    xr = rotate(x.astype(np.float32) * signs_in.astype(np.float32))            # once a token
    acc = np.zeros(hidden, np.float32)
    for w, graw, uraw, draw, (kg, ku, kd), sug, suu, svd, svg, svu, sud in units:
        g = rotate((xr * np.repeat(sug.astype(np.float32), 128)) @ unit_tiles(graw, kg, hidden, False, cb)) * svg.astype(np.float32)
        u = rotate((xr * np.repeat(suu.astype(np.float32), 128)) @ unit_tiles(uraw, ku, hidden, False, cb)) * svu.astype(np.float32)
        z = g / (1 + np.exp(-g)) * u
        acc += w * (rotate(z * sud.astype(np.float32)) @ unit_tiles(draw, kd, hidden, True, cb)) * np.repeat(svd.astype(np.float32), 128)
    return rotate(acc) * signs_out.astype(np.float32)                           # once a token


# ------------------------------------------------------------- files (local path or the hub, by range)
class SafeFile:
    """A safetensors file read by ranges: a local path, or a URL fetched with HTTP Range requests."""

    def __init__(self, path: str):
        self.path = path
        n = struct.unpack("<Q", self.read_at(0, 8))[0]
        self.header = json.loads(self.read_at(8, n))
        self.meta = self.header.pop("__metadata__", {})
        self.data0 = 8 + n

    def read_at(self, off: int, n: int) -> bytes:
        if self.path.startswith("http"):
            req = urllib.request.Request(self.path, headers={"Range": f"bytes={off}-{off + n - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
        else:
            with open(self.path, "rb") as f:
                f.seek(off)
                b = f.read(n)
        if len(b) != n:
            raise IOError(f"{self.path}: {len(b)} of {n} bytes at {off}")
        return b

    def trellis(self, name: str) -> tuple[str, str]:
        """A trellis tensor's stored name and codebook: `name` itself is mcg, `name.<codebook>` that codebook
        (FORMAT.md: the name selects it, so that one codebook's codes are never decoded with another's hash -
        plausible wrong weights and no error). Refuses a file with neither, or with both."""
        found = [(name, "mcg")] if name in self.header else []
        found += [(f"{name}.{cb}", cb) for cb in CODEBOOKS if cb != "mcg" and f"{name}.{cb}" in self.header]
        if len(found) != 1:
            other = [k for k in self.header if k.startswith(name + ".")]
            raise ValueError(f"{self.path}: {name}: {'both ' + str(found) if found else 'no tensor in a known codebook'}"
                             f" (codebooks {CODEBOOKS}; here {other})")
        return found[0]

    def offset(self, name: str) -> int:
        """Absolute byte offset of a tensor's data."""
        return self.data0 + self.header[name]["data_offsets"][0]

    def tensor(self, name: str) -> np.ndarray:
        h = self.header[name]
        dt = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U64": np.uint64,
              "F16": np.float16, "F32": np.float32}[h["dtype"]]
        a, b = h["data_offsets"]
        return np.frombuffer(self.read_at(self.data0 + a, b - a), dt).reshape(h["shape"])
