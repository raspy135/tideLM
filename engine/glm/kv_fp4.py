"""The MLA latent cache in 4 bits (GLM_KV=fp4): E2M1 values with a BF16 scale per 16.

A row of the latent (512 values) is 256 bytes of nibbles -- element i in the low nibble of byte
i, element 256 + i in the high one, so a kernel reads the row as two halves with no interleaving
to undo -- and 32 BF16 scales: 320 bytes, against FP8's 528 (512 + four fp32 scales per 128). Each 16-value block is scaled so its largest magnitude maps to E2M1's largest, 6,
and every value rounds to the nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and their negatives) -- the
NVFP4 grid and block, with a BF16 scale in place of NVFP4's FP8 one and per-tensor fp32 (40 bytes
a row more than that layout, for one load and no second scale level).

Why E2M1 / 16 rather than INT4 or 32-value blocks (relative RMS error over 200 x 512 synthetic
rows, a CPU simulation, 2026-10-18): Gaussian 0.0945 / INT4 0.0844; Student-t(4)
0.0902 / INT4 0.1056; Gaussian times per-channel log-normal weights (an RMSNorm's gain) 0.0899 /
INT4 0.1084; E2M1 with 32-value blocks 0.1015-0.1038. INT4's uniform grid wins only on tails as
light as a Gaussian's; a normed latent's are heavier. FP8 (E4M3, 128) is 0.024-0.026 on the same
rows: FP4 is about 4x the error, which is what the quality evaluations have to price.

Values decode to BF16 exactly (E2M1 has two significant bits) and are multiplied by the BF16
scale in BF16 -- in the attention kernel (sparse_attn) and in fp4_dequant alike, so the kernel and
the einsum path read the same keys.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

KV4_BLOCK = 16          # values per BF16 scale


@triton.jit
def e2m1(n):
    """Nibbles (any integer type, 0..15) -> their E2M1 values in BF16, built as bits: a magnitude
    code eem (e >= 1) is (1 + m / 2) * 2^(e - 1), BF16 exponent e + 126 and top mantissa bit m;
    codes 0 and 1 are 0 and 0.5."""
    c = n.to(tl.int32)
    m = c & 7
    bits = tl.where(m >= 2, (m << 6) + (126 << 7), tl.where(m == 1, 126 << 7, 0)) | ((c & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _quant(g, NB: tl.constexpr, B: tl.constexpr):
    """[NB * B] fp32 -> (int32 nibbles [NB * B], bf16 scales [NB])."""
    g = tl.reshape(g, (NB, B))
    s = tl.maximum(tl.max(tl.abs(g), 1) / 6.0, 1e-30).to(tl.bfloat16)
    # IEEE division (Triton's `/` is approximate): the codes are then fp4_quant_ref's bit for bit.
    y = tl.math.div_rn(g, tl.broadcast_to(s.to(tl.float32)[:, None], (NB, B)))
    a = tl.abs(y)
    # Round to the nearest E2M1 magnitude (ties down); past 6 (a scale rounded down) is 6.
    m = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
         + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return tl.reshape(m + tl.where(y < 0, 8, 0), (NB * B,)), s


@triton.jit
def _store_kernel(X, IDX, P, S, stride_x, stride_p, stride_s, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    row = tl.load(IDX + i).to(tl.int64)
    DH: tl.constexpr = D // 2
    NBH: tl.constexpr = DH // B
    dh = tl.arange(0, DH)
    lo, s0 = _quant(tl.load(X + i * stride_x + dh).to(tl.float32), NBH, B)
    hi, s1 = _quant(tl.load(X + i * stride_x + DH + dh).to(tl.float32), NBH, B)
    tl.store(P + row * stride_p + dh, (lo | (hi << 4)).to(tl.uint8))
    tl.store(S + row * stride_s + tl.arange(0, NBH), s0)
    tl.store(S + row * stride_s + NBH + tl.arange(0, NBH), s1)


def fp4_store(packed: torch.Tensor, scale: torch.Tensor, idx: torch.Tensor, x: torch.Tensor):
    """Quantize rows x [n, D] and store them at rows idx of packed uint8 [rows, D / 2] and scale
    bf16 [rows, D / 16]. One launch (decode writes every layer's rows inside its graphs)."""
    n, D = x.shape
    if n == 0:
        return
    assert x.stride(-1) == 1 and packed.stride(-1) == 1 and scale.stride(-1) == 1
    _store_kernel[(n,)](x, idx.contiguous(), packed, scale, x.stride(0), packed.stride(0), scale.stride(0),
                        D=D, B=D // scale.shape[-1], num_warps=4)


def fp4_dequant(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """packed uint8 [..., D / 2], scale bf16 [..., D / 16] -> [..., D] in `dtype` (values times
    scale in BF16, as the kernels compute them)."""
    nib = torch.cat((packed & 15, packed >> 4), -1).to(torch.int32)
    m = nib & 7
    e, f = m >> 1, (m & 1).float()
    mag = torch.where(e == 0, 0.5 * f, (1.0 + 0.5 * f) * torch.exp2((e - 1).clamp_min(0).float()))
    v = torch.where(nib >= 8, -mag, mag).to(torch.bfloat16)
    D, nb = v.shape[-1], scale.shape[-1]
    return (v.view(*v.shape[:-1], nb, D // nb) * scale[..., None].to(torch.bfloat16)).view(v.shape).to(dtype)


def fp4_quant_ref(x: torch.Tensor, block: int = KV4_BLOCK):
    """The store kernel's quantization in torch (tests): x [n, D] -> (packed, scale)."""
    n, D = x.shape
    g = x.float().view(n, D // block, block)
    s = (g.abs().amax(-1) / 6.0).clamp_min(1e-30).to(torch.bfloat16)
    y = g / s.float()[..., None]
    a = y.abs()
    m = sum((a > t).to(torch.int32) for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    nib = (m + (y < 0).to(torch.int32) * 8).view(n, D)
    return (nib[:, :D // 2] | (nib[:, D // 2:] << 4)).to(torch.uint8), s


def parse_layers(spec: str, n: int) -> set[int]:
    """"0-9,70-77" -> those layer indices (below n); "" -> none."""
    out = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return {L for L in out if 0 <= L < n}
