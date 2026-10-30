"""
fp8_linear.py -- y = x @ W^T for the checkpoint's dense projections in their STORED format:
W fp8 (e4m3) [N, K] with UE8M0 scales [ceil(N/32), ceil(K/32)] (one power-of-two scale per 32x32
block), x bf16 [M, K]. The weights are never dequantized to bf16 in memory, which halves both the
bytes read per decode step and the resident footprint of the dense layers.

Two tile shapes: BLOCK_M=16 for decode-sized M (padded up to 16 rows) and BLOCK_M=64 for prefill.
Batched multi-request decode widens the decode tile to 32/64 rows inside `decode_rows.wide()`
(tools/decode_rows.py); outside it nothing changes.
Weights are dequantized to bf16 inside each tile using their 32x32 power-of-two scales;
the dot product accumulates in fp32 over fixed 128-wide K tiles (64 for unaligned K).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import decode_rows

# Decode-sized (M <= 16) scheduling. The grid is N / BLOCK_N CTAs, and GB10 has 48 SMs: at the
# default 128 the narrow projections launch 4 (wkv, N=512), 9 (TP shared w1/w3, N=1152) or 10
# (wq_a, N=1280) CTAs and reach ~58-140 GB/s in the 2026-09-16 decode trace against ~200 for
# wq_b's 128-CTA grid. BLOCK_N only partitions OUTPUT columns: every output element is still one
# fp32 accumulation over K in the same BLOCK_K steps, so a narrower tile is meant to be
# bit-identical (the 64-wide sweep was, 42/42) -- tools/test_fp8_decode_tiles.py checks it rather
# than assuming it. "auto" takes the widest tile that still fills every SM, else 16.
# Read at call time so a benchmark can A/B arms inside one process; the EP2 boot guard pins it.
# 2026-10-22: with BLOCK_N=32/BLOCK_K=256 every TP2 decode shape streams at 211-230 GB/s cold, and
# no tile/warp/stage combination is more than 3% faster. An ordered split-K would
# change the bits for nothing measurable.
DECODE_BLOCK_N = os.environ.get("DSV41_FP8_DECODE_BLOCK_N", "auto")
DECODE_WARPS = int(os.environ.get("DSV41_FP8_DECODE_WARPS", "4"))
# Native-weight decode optimization. Read at call time for the frozen-map A/B driver;
# changing it in a live process requires releasing/rebuilding captured graphs on both ranks.
DECODE_PIPELINE = os.environ.get("DSV41_FP8_DECODE_PIPELINE", "0") == "1"
# K tile at decode-sized M when K is a multiple of it. tl.dot chains the accumulator through the same
# K steps in the same order whatever the tile, so 256 is bit-identical to 128 (checked on every
# decode shape at 4 and 6 rows). With BLOCK_N=32, cold GB10 weights at 4 rows: wq_b 115 -> 106 us,
# wo_b 119 -> 104, wo_a 99 -> 91, wq_a+wkv 66 -> 58 (2026-10-08). Read at call time, like BLOCK_N.
DECODE_BLOCK_K = int(os.environ.get("DSV41_FP8_DECODE_BLOCK_K", "128"))
if DECODE_BLOCK_K not in (128, 256):
    raise ValueError("DSV41_FP8_DECODE_BLOCK_K must be 128 or 256")
_DECODE_BLOCK_N_CHOICES = ("16", "32", "64", "128", "auto")
if DECODE_BLOCK_N not in _DECODE_BLOCK_N_CHOICES:
    raise ValueError(f"DSV41_FP8_DECODE_BLOCK_N={DECODE_BLOCK_N!r}; use one of {_DECODE_BLOCK_N_CHOICES}")
_SMS: dict[int, int] = {}


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SMS.get(idx)
    if n is None:
        n = _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return n


def decode_block_n(n_cols: int, groups: int, device, policy: str | None = None) -> int:
    """BLOCK_N for a decode-sized call producing `groups` x `n_cols` outputs."""
    policy = DECODE_BLOCK_N if policy is None else policy
    if policy != "auto":
        return int(policy)
    sms = _sm_count(device)
    for bn in (128, 64, 32):
        if triton.cdiv(n_cols, bn) * groups >= sms:
            return bn
    return 16


@triton.jit
def _act_qdq_tile(x, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                  BIT_SCALE: tl.constexpr = False):
    """v41_ref.act_qdq_fp8 on a [BLOCK_M, BLOCK_K] bf16 tile whose K offset is a multiple of 32.

    Bit-identical to the torch spelling, which is why each step is spelled the way it is:
    amax is order-independent; `/` and log2/exp2 go through libdevice's IEEE div_rn / log2f /
    exp2f like torch's CUDA kernels (Triton's default fp32 `/` is div.full and tl.log2 is
    lg2.approx -- both can move ceil(log2(.)) across an integer); multiplying by the power-of-two
    scale is exact; e4m3 conversion is round-to-nearest-even on both sides; the product is exact
    in bf16 (3 mantissa bits). tools/test_fp8_act_qdq.py checks it through an identity weight.

    Triton builds libdevice with flush-to-zero (the PTX has div.rn.ftz, ex2.approx.ftz); torch
    does not. That cannot change a result here: amax / 448 >= 2.2e-7 is normal; exp2 only sees
    integers and returns exact powers of two >= 2^-22; a subnormal x, or a quotient that
    underflows, quantizes to a same-signed zero in e4m3 either way. And with bf16 x, amax / 448
    is either an exact power of two (exact log2 in both libraries) or >= ~0.4% away from one, so
    a last-ulp log2 difference cannot move ceil(). An fp32 x would lose that margin, which is one
    reason fp8_linear refuses it.
    """
    xg = tl.reshape(x.to(tl.float32), (BLOCK_M, BLOCK_K // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-4)
    # The expensive IEEE div/log2 run once per 32-group. Per element, x / s is spelled
    # x * 2^-e: s = 2^e is a power of two, so the quotient and the product are the same real
    # number and round identically. The first version divided per element with div_rn inside the
    # K loop and was 2.3x SLOWER than the torch spelling (wq_a 56 -> 129 us, 2026-09-23).
    if BIT_SCALE:
        # 448 = 1.75 * 2^8. For finite BF16 amax, ceil(log2(amax/448)) is its
        # exponent minus eight, plus one only when its mantissa exceeds 1.75.
        # The clamp also has a safe margin from that boundary. e is in [-22,120],
        # so both powers of two are normal FP32. No approximate log or division.
        bits = amax.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 255) - 135 + ((bits & 0x7fffff) > 0x600000).to(tl.int32)
        s = ((e + 127) << 23).to(tl.float32, bitcast=True)[:, :, None]
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)[:, :, None]
    else:
        e = libdevice.ceil(libdevice.log2(libdevice.div_rn(amax, 448.0)))
        s = libdevice.exp2(e)[:, :, None]
        inv = libdevice.exp2(-e)[:, :, None]
    q = tl.minimum(tl.maximum(xg * inv, -448.0), 448.0)
    q = q.to(tl.float8e4nv).to(tl.float32) * s
    return tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.bfloat16)


@triton.jit
def _fp8_linear_kernel(X, W, S, Y, M, N, K,
                       stride_xm, stride_wn, stride_sn, stride_ym,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       ACT_QDQ: tl.constexpr = False, BIT_SCALE: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, BLOCK_K // 32)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_scale_row = rn // 32
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        if ACT_QDQ:
            x = _act_qdq_tile(x, BLOCK_M, BLOCK_K, BIT_SCALE)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        # one ue8m0 scale per 32-wide K group: exact in bf16 (3 mantissa bits, power-of-two scale)
        s = tl.load(S + n_scale_row[:, None] * stride_sn + (k0 // 32 + rs)[None, :],
                    mask=n_mask[:, None] & ((k0 // 32 + rs)[None, :] < (K + 31) // 32), other=127).to(tl.int32)
        if BIT_SCALE:
            # Same exponent bias in UE8M0 and BF16; zero matches ex2.ftz's
            # underflow at 2^-127, and 255 matches its overflow to infinity.
            scale = (s << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        else:
            scale = tl.exp2((s - 127).to(tl.float32)).to(tl.bfloat16)
        w3 = tl.reshape(w.to(tl.bfloat16), (BLOCK_N, BLOCK_K // 32, 32)) * scale[:, :, None]
        wb = tl.reshape(w3, (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fp8_grouped_kernel(X, W, S, Y, T, R, K,
                        stride_xt, stride_xg, stride_wn, stride_sn, stride_yt, stride_yg,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        BIT_SCALE: tl.constexpr = False):
    """Per group g: Y[:, g, :] = X[:, g, :] @ W[g*R:(g+1)*R, :]^T, with W in the stored fp8 format.

    Same tiling as `_fp8_linear_kernel` plus a third grid axis for the group; a group is nothing
    but a row range of W (rows g*R .. (g+1)*R) and the matching row range of the block-scale table
    (rows g*R/32 .. (g+1)*R/32), so no data is copied or re-laid-out. R must be a multiple of 32.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, BLOCK_K // 32)
    m_mask = rm < T
    n_mask = rn < R
    wrow = g * R + rn
    n_scale_row = wrow // 32
    Xg = X + g * stride_xg
    n_sk = (K + 31) // 32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(Xg + rm[:, None] * stride_xt + kk[None, :],
                    mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(W + wrow[:, None] * stride_wn + kk[None, :],
                    mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        s = tl.load(S + n_scale_row[:, None] * stride_sn + (k0 // 32 + rs)[None, :],
                    mask=n_mask[:, None] & ((k0 // 32 + rs)[None, :] < n_sk), other=127).to(tl.int32)
        if BIT_SCALE:
            scale = (s << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        else:
            scale = tl.exp2((s - 127).to(tl.float32)).to(tl.bfloat16)
        w3 = tl.reshape(w.to(tl.bfloat16), (BLOCK_N, BLOCK_K // 32, 32)) * scale[:, :, None]
        wb = tl.reshape(w3, (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_yt + g * stride_yg + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


class FP8Weight:
    """Holds a dense weight in the stored format (fp8 e4m3 + ue8m0 32x32 block scales)."""

    def __init__(self, weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor):
        assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.dim() == 2
        self.w = weight_fp8.contiguous()
        self.s = scale_e8m0.view(torch.uint8).contiguous()
        self.N, self.K = self.w.shape
        assert self.s.shape == ((self.N + 31) // 32, (self.K + 31) // 32), (self.s.shape, self.w.shape)
        assert self.K % 32 == 0
        self._bf16_cache = None

    @property
    def shape(self):
        return (self.N, self.K)

    def shard(self, dim: int, rank: int, world: int) -> "FP8Weight":
        """One rank's slice of this weight, scales included.

        `dim=0` splits the OUTPUT rows (a column-parallel weight: w1/w3 of an MLP), `dim=1` splits
        the INPUT columns (row-parallel: w2). The 32x32 block scales have to be sliced on the same
        boundary, which is why both halves must land on a multiple of 32 -- asserted rather than
        rounded, because a scale block that straddles a shard boundary is silently wrong.
        """
        if world <= 1:
            return self
        if dim == 0:
            n = self.N // world
            assert self.N % world == 0 and n % 32 == 0, (self.N, world)
            return FP8Weight(self.w[rank * n:(rank + 1) * n].clone(),
                             self.s[rank * (n // 32):(rank + 1) * (n // 32)].clone())
        k = self.K // world
        assert self.K % world == 0 and k % 32 == 0, (self.K, world)
        return FP8Weight(self.w[:, rank * k:(rank + 1) * k],
                         self.s[:, rank * (k // 32):(rank + 1) * (k // 32)])

    def dequant(self) -> torch.Tensor:
        """fp8 + UE8M0 32x32 block scales -> bf16 [N, K]. Bit-identical to the fp32 round-trip
        this replaced, and ~3.6x faster on the big attention projection.

        Why it is safe to drop the fp32 intermediate: a UE8M0 scale is 2^(e-127), an exact power
        of two, and fp8 e4m3 carries 3 mantissa bits against bf16's 7. So fp8 -> bf16 is exact and
        multiplying by the scale only adjusts the exponent -- there is nothing for fp32 to round
        more accurately. `torch.equal` on all three attention projections confirms it.

        Why it was slow: `self.w.float()` materialised an fp32 [N, K] (201 MB for wq_b), the two
        repeat_interleaves built a second fp32 [N, K] scale matrix, and the product a third --
        ~700 MB of traffic to expand a 50 MB weight. Broadcasting over a blocked view moves the
        scale to where it belongs (one value per 32x32 tile) and keeps everything in bf16, so the
        call reads 50 MB and writes 100 MB.

        This runs on the PREFILL path of v41_ref.dense() for every dense projection of every
        layer of every chunk -- it was 77 % of the wq_b call and ~60 % of prefill wall time.
        """
        n32, k32 = (self.N + 31) // 32, (self.K + 31) // 32
        s = torch.exp2(self.s.float() - 127.0).to(torch.bfloat16)      # [n32, k32], tiny
        w = self.w
        pad = n32 * 32 - self.N                                        # K % 32 == 0 is asserted
        if pad:
            w = torch.cat([w, w.new_zeros(pad, self.K)])
        out = (w.to(torch.bfloat16).view(n32, 32, k32, 32) * s[:, None, :, None]).view(n32 * 32, self.K)
        return out[: self.N] if pad else out

    def dequant_cached(self) -> torch.Tensor:
        """dequant(), memoized on the weight object (see v41_ref.DENSE_DEQUANT_CACHE).

        dequant() is exact and deterministic, so the cached tensor is bit-identical to a fresh
        call; this only stops the same bytes being expanded again for every chunk of every
        request. FP8Weight objects are built once per layer at load, so the cache persists for
        the process.
        """
        t = self._bf16_cache
        if t is None:
            t = self.dequant()
            self._bf16_cache = t
        return t


def concat_rows(*parts: FP8Weight) -> tuple[FP8Weight, tuple[FP8Weight, ...]] | None:
    """Stack same-K weights along N: (merged, views), where each view is one input rebuilt as a
    row slice of the merged storage -- rebinding the originals to the views keeps ONE resident
    copy. Returns None when the parts cannot share a launch without changing any output: exact
    FP8Weight only (a TP wrapper or FP4 weight has its own kernel), equal K, and every N a
    multiple of 32 so no 32x32 scale block straddles a boundary.

    One launch over the merged rows computes each output column exactly as its own launch did
    (same K order, see DECODE_BLOCK_N), and the caller quantizes the shared activation once.
    """
    if len(parts) < 2 or any(type(p) is not FP8Weight for p in parts):
        return None
    if len({p.K for p in parts}) != 1 or any(p.N % 32 for p in parts):
        return None
    if len({p.w.device for p in parts}) != 1:
        return None
    w = torch.cat([p.w for p in parts])
    s = torch.cat([p.s for p in parts])
    views, n0 = [], 0
    for p in parts:
        views.append(FP8Weight(w[n0:n0 + p.N], s[n0 // 32:(n0 + p.N) // 32]))
        n0 += p.N
    return FP8Weight(w, s), tuple(views)


def quantize_to_fp8(ref: torch.Tensor, rows: int = 4096) -> FP8Weight:
    """fp32/bf16 [N, K] -> FP8Weight in the checkpoint's own dense format: e4m3 codes plus one
    UE8M0 power-of-two scale per 32x32 block.

    The dense projections arrive in this format from disk; the LM head does not (it is stored bf16),
    so this is the quantizer that puts it there. Per block: amax, scale = 2^ceil(log2(amax/448))
    corrected upward if that would still clip, then the hardware's round-to-nearest-even e4m3
    conversion. Nothing can overflow: a value <= 448 never rounds above 448, whose neighbour in the
    grid is 512 with the tie at 480.

    Unlike the FP4 quantizer this one does not need a finer scale table: e4m3 carries its own 4-bit
    exponent, so the block scale only has to place the block inside the representable range, and the
    3-bit mantissa fixes the relative error wherever it lands.
    """
    assert ref.dim() == 2
    N, K = ref.shape
    nb_k = (K + 31) // 32
    rows = max(32, rows - rows % 32)
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=ref.device)
    scales = torch.empty((N + 31) // 32, nb_k, dtype=torch.uint8, device=ref.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float()
        n = r1 - r0
        pn, pk = (-n) % 32, (-K) % 32
        if pn or pk:
            x = torch.nn.functional.pad(x, (0, pk, 0, pn))
        xb = x.view((n + pn) // 32, 32, nb_k, 32)
        amax = xb.abs().amax(dim=(1, 3)).clamp_min(448.0 * 2.0 ** -126)     # [nb, nb_k]
        e = torch.ceil(torch.log2(amax / 448.0))
        e = torch.where(amax > 448.0 * torch.exp2(e), e + 1, e)             # never clip
        b = (e + 127.0).clamp(0, 255)
        q = (xb / torch.exp2(b - 127.0)[:, None, :, None]).to(torch.float8_e4m3fn)
        q = q.view((n + pn), K + pk)
        codes[r0:r1] = q[:n, :K]
        scales[r0 // 32:(r1 + 31) // 32] = b.to(torch.uint8)
        del x, xb, amax, e, b, q
    return FP8Weight(codes, scales)


def _wide_block_k(block_m: int, block_n: int, block_k: int) -> int:
    """K tile for the wide (32/64-row) decode tiles of batched decode. DSV41_FP8_DECODE_BLOCK_K=256
    at 64 rows stages ~40 KB a pipeline stage, 115 KB over three: past GB10's 99 KB of shared
    memory, and the launch fails. Halving K keeps each output's accumulation in the same K order, so
    it is meant to be bit-identical (tools/test_decode_rows.py compares against the 16-row tile at
    K 256). The 16-row tile never comes here."""
    if block_m > 16 and block_k > 128 and block_m * block_k * 2 + block_n * block_k > 32 * 1024:
        return 128
    return block_k


def fp8_linear(x: torch.Tensor, W: FP8Weight, out_dtype=torch.bfloat16, act_qdq: bool = False) -> torch.Tensor:
    """x bf16 [..., K] -> bf16 [..., N].

    act_qdq: apply v41_ref.act_qdq_fp8 to x inside the kernel instead of as ~13 separate torch
    kernels before it. Decode-sized bf16 input only: the torch spelling quantizes from x's own
    dtype, and an fp32 x would first be rounded to bf16 here."""
    shape = x.shape
    rows = decode_rows.limit()
    if act_qdq and (x.dtype != torch.bfloat16 or x.numel() // x.shape[-1] > rows):
        raise ValueError(f"fused act_qdq needs bf16 input with <= {rows} rows, got {x.dtype} {tuple(x.shape)}")
    x2 = x.reshape(-1, W.K)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    M = x2.size(0)
    y = torch.empty(M, W.N, dtype=out_dtype, device=x.device)
    decode = M <= rows
    BLOCK_M = decode_rows.block_m(M) if decode else 64
    BLOCK_N = decode_block_n(W.N, 1, x.device) if decode else 128
    BLOCK_K = 128 if W.K % 128 == 0 else 64
    if decode and BLOCK_K == 128 and W.K % DECODE_BLOCK_K == 0:
        BLOCK_K = DECODE_BLOCK_K
    BLOCK_K = _wide_block_k(BLOCK_M, BLOCK_N, BLOCK_K)
    pipeline = decode and DECODE_PIPELINE
    grid = (triton.cdiv(W.N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _fp8_linear_kernel[grid](x2, W.w, W.s, y, M, W.N, W.K, x2.stride(0), W.w.stride(0), W.s.stride(0), y.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ACT_QDQ=act_qdq,
                             BIT_SCALE=pipeline,
                             num_warps=DECODE_WARPS if decode else 4,
                             num_stages=5 if pipeline and W.K >= 4096 else 3)
    return y.view(*shape[:-1], W.N)


class FP8GroupedWeight:
    """The attention output LoRA `wo_a` in its stored format: fp8 e4m3 [G*R, K] with UE8M0 32x32
    block scales [G*R/32, K/32], addressed as G independent [R, K] matrices.

    `convert.py` dequantizes this one to bf16 and the engine kept it that way ([8, 1024, 4096] bf16
    = 67 MB per layer, 2.7 GB read per decode step over 40 layers). Kept fp8 it is 33.5 MB per layer
    and 1.34 GB less resident.
    """

    def __init__(self, weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, groups: int, rank: int):
        assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.dim() == 2
        self.w = weight_fp8.contiguous()
        self.s = scale_e8m0.view(torch.uint8).contiguous()
        N, self.K = self.w.shape
        assert N == groups * rank, (N, groups, rank)
        assert rank % 32 == 0 and self.K % 32 == 0
        assert self.s.shape == ((N + 31) // 32, (self.K + 31) // 32), (self.s.shape, self.w.shape)
        self.G, self.R = groups, rank

    @property
    def shape(self):
        return (self.G, self.R, self.K)

    def dequant(self) -> torch.Tensor:
        """[G, R, K] bf16 -- the tensor the old path held resident."""
        s = torch.exp2(self.s.float() - 127.0)
        n = self.G * self.R
        s = s.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, : self.K]
        return (self.w.float() * s).to(torch.bfloat16).view(self.G, self.R, self.K)


def fp8_grouped_linear(x: torch.Tensor, W: FP8GroupedWeight) -> torch.Tensor:
    """x bf16 [T, G, K] -> bf16 [T, G, R]; equals einsum("tgk,grk->tgr", x, W.dequant()).

    Row-count- and row-offset-invariant by construction (each output element is one fp32
    accumulation over K in fixed BLOCK_K steps, independent of T and of the row's position), so it
    needs no MM_TILE row tiling.
    """
    assert x.dim() == 3 and x.size(1) == W.G and x.size(2) == W.K, (x.shape, W.shape)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()
    T = x.size(0)
    y = torch.empty(T, W.G, W.R, dtype=torch.bfloat16, device=x.device)
    decode = T <= decode_rows.limit()
    BLOCK_M = decode_rows.block_m(T) if decode else 64
    BLOCK_N = decode_block_n(W.R, W.G, x.device) if decode else 128
    BLOCK_K = 128 if W.K % 128 == 0 else 64
    if decode and BLOCK_K == 128 and W.K % DECODE_BLOCK_K == 0:
        BLOCK_K = DECODE_BLOCK_K
    BLOCK_K = _wide_block_k(BLOCK_M, BLOCK_N, BLOCK_K)
    pipeline = decode and DECODE_PIPELINE
    grid = (triton.cdiv(W.R, BLOCK_N), triton.cdiv(T, BLOCK_M), W.G)
    _fp8_grouped_kernel[grid](x, W.w, W.s, y, T, W.R, W.K,
                              x.stride(0), x.stride(1), W.w.stride(0), W.s.stride(0),
                              y.stride(0), y.stride(1),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              BIT_SCALE=pipeline,
                              num_warps=DECODE_WARPS if decode else 4,
                              num_stages=5 if pipeline and W.K >= 4096 else 3)
    return y


if __name__ == "__main__":
    import json, os, sys, time
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.attn.wq_b.weight']}", "pt", device="cpu")
    torch.manual_seed(0)
    for name in ("layers.0.attn.wq_b", "layers.0.attn.wo_b", "layers.0.attn.wq_a", "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2"):
        W = FP8Weight(f.get_tensor(name + ".weight").cuda(), f.get_tensor(name + ".scale").cuda())
        ref_w = W.dequant()
        for M in (1, 6, 16, 64, 512):
            x = (torch.randn(M, W.K, device="cuda") * 0.5).to(torch.bfloat16)
            y = fp8_linear(x, W)
            r = torch.nn.functional.linear(x, ref_w)
            rel = float((y.float() - r.float()).norm() / r.float().norm())
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): fp8_linear(x, W)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 20
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): torch.nn.functional.linear(x, ref_w)
            torch.cuda.synchronize(); dt2 = (time.perf_counter() - t0) / 20
            print(f"{name:34s} N={W.N:6d} K={W.K:5d} M={M:3d}: rel {rel:.2e}  fp8 {dt*1e3:6.3f} ms ({W.N*W.K/dt/1e9:5.0f} GB/s of fp8)  bf16 {dt2*1e3:6.3f} ms ({W.N*W.K*2/dt2/1e9:5.0f} GB/s)")
