"""fp4_moe.py -- Triton grouped-MoE forward for DeepSeek-V4.1-Flash's FP4 routed experts.

Target: NVIDIA GB10 (sm_121a, 48 SMs, unified memory, ~273 GB/s). Decode is bandwidth-bound on the
18.8 MB every active expert weighs, so the design goal is "read each active expert's bytes once per
launch, decode FP4 in registers, feed the tensor cores".

Weight format (per expert, as stored in the HF safetensors):
  w1, w3 : int8  [2304, 2560]  packed FP4 E2M1, two values per byte along K (logical [2304, 5120]);
                               low nibble = even K element, high nibble = odd K element
  s1, s3 : uint8 [2304, 160]   UE8M0 block scale, one per 32 consecutive K elements, value 2^(b-127)
  w2     : int8  [5120, 1152]  (logical [5120, 2304]),  s2 : uint8 [5120, 72]

Pipeline for one call (T tokens, K experts per token), three Triton launches and no host sync:
  1. _route_kernel: groups the (token, k) pairs by expert slot, each slot's run padded to a multiple of
     BM pairs -> block_slot[b], block_pair[b*BM + i].
  2. _moe_up_kernel: grid (pair block, n-block). For its expert it streams the packed w1/w3 rows ONCE,
     decodes them with the hardware `cvt.rn.f16x2.e2m1x2` instruction (one instruction per byte -> two
     fp16), does tl.dot against the (up to BM) pairs of the block and applies the UE8M0 scale on the fp32
     partial sum of each 32-wide K group. Epilogue = clamps + SiLU + routing weight; writes h[pair, :]
     (bf16 [T*K, 2304]). Its n-block-0 programs also zero the fp32 output rows for step 3.
  3. _moe_down_kernel: same structure over w2, atomically scatter-adds the fp32 tile into y32[token, :];
     y32 is cast to bf16 at the end.

K-permutation trick: because a dot product is order-invariant along K, the even/odd nibbles never have to
be interleaved. The decoder returns the even K elements and the odd K elements as two [BN, 16] fp16 tiles
(via `prmt`), and the activation tile is loaded with stride 2 to match. No layout shuffles for the weights.

For decode-size calls every expert has <= BM pairs, so its weight bytes are read exactly once per launch.
For prefill-size calls (more than BM pairs on one expert) the extra pair blocks re-read the expert (L2).

Measured on GB10 (tools/test_fp4_moe.py, 32 real layer-0 experts): ~190-200 GB/s effective for decode
sizes (T=1..6) against a ~210 GB/s practical copy ceiling on this box (273 GB/s nominal).
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

DIM = 5120  # model hidden size (K of w1/w3, N of w2)
INTER = 2304  # expert intermediate size (N of w1/w3, K of w2)
# Delegate FP4 unpack/scale to Triton (tl.dot_scaled). With a BF16 activation operand the
# installed GB10 backend lowers this to BF16 x BF16 MMA with FP32 accumulation, not FP4
# activation arithmetic (inspect _moe_up_kernel.ptx). Software decode was measured at
# 19.8 TFLOPS, dot_scaled at 29.5, but their reduction orders differ: software resets the
# partial sum per scale group, whereas dot_scaled carries the MMA accumulator along K.
# This is a numerics change; a random-error/generation floor gate alone cannot qualify it.
DOT_SCALED = os.environ.get("DSV41_FP4_DOT_SCALED", "0") == "1"
# Separate arithmetic arm for prompt encoding and bounded decoder replay, including
# tiny prefills. Ordinary verification and DSpark stay on their existing kernels:
# dispatch uses the caller's phase, not a token-count guess. See docs/gotchas.md.
PREFILL_DOT_SCALED = os.environ.get("DSV41_FP4_PREFILL_DOT_SCALED", "0") == "1"
# Prefill-sized scaled calls (P > 1024, BM=64 by default) on 32-row blocks: the scaled-arithmetic
# sweep at TP shapes (results/prefill-expert-20261008/sweep.json, 66 cells, every one bit-identical
# to the default schedule) has up (BM32, BN128, 4 warps, 3 stages) 11.96 vs 13.80 ms and down
# (BM32, BN128, 4, 4) 9.78 vs 10.56 ms at 2,048 rows. Scaled prefill only; read at call time.
PREFILL_SCALED_TILES = os.environ.get("DSV41_FP4_PREFILL_SCALED_TILES", "0") == "1"
# Native CUDA is the default BM=16 decode path; prefill and unsupported shapes use Triton.
# Set DSV41_FP4_CUDA=0 to restore Triton throughout. Measured scope is in docs/gotchas.md.
CUDA_DECODE = os.environ.get("DSV41_FP4_CUDA", "1") == "1"

GROUP = 32  # K elements per UE8M0 scale
KB1 = DIM // 2  # packed bytes per w1/w3 row
SG1 = DIM // GROUP  # scale groups per w1/w3 row
KB2 = INTER // 2  # packed bytes per w2 row
SG2 = INTER // GROUP  # scale groups per w2 row

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# --------------------------------------------------------------------------- reference dequant
def _local_dequant_fp4_packed(weight_i8: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Same contract as tools/v41_ref.py::dequant_fp4_packed (used only if that module is not importable)."""
    x = weight_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    tab = FP4_TABLE.to(x.device)
    vals = torch.stack([tab[low.long()], tab[high.long()]], dim=-1).flatten(1)
    s = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0).repeat_interleave(group, 1)
    return (vals * s).to(torch.bfloat16)


try:  # prefer the verified reference implementation next to this file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v41_ref import dequant_fp4_packed  # type: ignore
except Exception:  # pragma: no cover
    dequant_fp4_packed = _local_dequant_fp4_packed


# --------------------------------------------------------------------------- arena
class ExpertArena:
    """GPU-resident slots of packed FP4 expert weights (18.8 MB per slot)."""

    def __init__(self, slots: int, device: torch.device | str = "cuda", tp_rank=0, tp_world=1):
        self.slots = slots
        self.device = torch.device(device)
        if tp_world not in (1, 2) or not 0 <= tp_rank < tp_world:
            raise ValueError('FP4 expert TP supports world=1 or 2')
        self.tp_rank, self.tp_world = tp_rank, tp_world
        layout = os.environ.get('DSV41_TP_EXPERT_LAYOUT', 'output')
        if layout not in ('intermediate', 'output'):
            raise ValueError('TP expert layout must be intermediate or output')
        self.tp_output = tp_world > 1 and layout == 'output'
        self.inter = INTER // tp_world
        ni, kb2, sg2 = self.inter, self.inter // 2, self.inter // GROUP
        down_n = DIM
        if self.tp_output:
            down_n, kb2, sg2 = DIM // tp_world, KB2, SG2
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1 = torch.empty((slots, ni, KB1), **u8)
        self.s1 = torch.empty((slots, ni, SG1), **u8)
        self.w3 = torch.empty((slots, ni, KB1), **u8)
        self.s3 = torch.empty((slots, ni, SG1), **u8)
        self.w2 = torch.empty((slots, down_n, kb2), **u8)
        self.s2 = torch.empty((slots, down_n, sg2), **u8)

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    @staticmethod
    def _u8(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        t = t.view(torch.uint8)
        if tuple(t.shape) != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(t.shape)}")
        return t.contiguous()

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False) -> None:
        """Copy one expert (CPU int8/uint8/e8m0 tensors as read from safetensors) into `slot`.

        `non_blocking=True` is for the serving path only (`engine.experts.ExpertStore`), where the
        source is a *pinned* staging buffer and the caller synchronises its own copy stream before
        handing that buffer to the next expert. Blocking is the default because a plain `.copy_()`
        from pinned memory synchronises the calling stream once per tensor -- six GPU syncs per
        expert, on the stream the model is computing on, for every miss.
        """
        lo, hi = self.tp_rank * self.inter, (self.tp_rank + 1) * self.inter
        self.w1[slot].copy_(self._u8(w1, (INTER, KB1))[lo:hi], non_blocking=non_blocking)
        self.s1[slot].copy_(self._u8(s1, (INTER, SG1))[lo:hi], non_blocking=non_blocking)
        self.w3[slot].copy_(self._u8(w3, (INTER, KB1))[lo:hi], non_blocking=non_blocking)
        self.s3[slot].copy_(self._u8(s3, (INTER, SG1))[lo:hi], non_blocking=non_blocking)
        if self.tp_output:
            row = self.tp_rank * self.w2.shape[1]
            end = row + self.w2.shape[1]
            self.w2[slot].copy_(self._u8(w2, (DIM, KB2))[row:end], non_blocking=non_blocking)
            self.s2[slot].copy_(self._u8(s2, (DIM, SG2))[row:end], non_blocking=non_blocking)
        else:
            self.w2[slot].copy_(self._u8(w2, (DIM, KB2))[:, lo//2:hi//2], non_blocking=non_blocking)
            self.s2[slot].copy_(self._u8(s2, (DIM, SG2))[:, lo//GROUP:hi//GROUP], non_blocking=non_blocking)

    def dequant_slot(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w2, w3) as bf16 [N, K] logical matrices -- for the reference path / tests."""
        w1 = dequant_fp4_packed(self.w1[slot], self.s1[slot])
        w2 = dequant_fp4_packed(self.w2[slot], self.s2[slot])
        w3 = dequant_fp4_packed(self.w3[slot], self.s3[slot])
        return w1, w2, w3


# --------------------------------------------------------------------------- device helpers
# One 32-bit register holds 4 packed bytes = 8 FP4 values. `cvt.rn.f16x2.e2m1x2` (sm_100a/120a/121a)
# turns one byte into {f16(high nibble), f16(low nibble)}; the prmt shuffles regroup the four results into
# "even elements" (low nibbles) and "odd elements" (high nibbles), 4 fp16 each. The .b8 source operand is
# mandatory -- ptxas rejects a .b16/.b32 source with "Arguments mismatch for instruction 'cvt'".
_FP4_DECODE_ASM = tl.constexpr("""
{
.reg .b8 b0, b1, b2, b3;
.reg .b32 t0, t1, t2, t3;
mov.b32 {b0, b1, b2, b3}, $4;
cvt.rn.f16x2.e2m1x2 t0, b0;
cvt.rn.f16x2.e2m1x2 t1, b1;
cvt.rn.f16x2.e2m1x2 t2, b2;
cvt.rn.f16x2.e2m1x2 t3, b3;
prmt.b32 $0, t0, t1, 0x5410;
prmt.b32 $1, t2, t3, 0x5410;
prmt.b32 $2, t0, t1, 0x7632;
prmt.b32 $3, t2, t3, 0x7632;
}
""")


@triton.jit
def _fp4_decode(packed):
    """uint8 tile [..., 16] -> (even fp16 tile, odd fp16 tile), same shape; values in the E2M1 grid."""
    return tl.inline_asm_elementwise(
        _FP4_DECODE_ASM, "=r,=r,=r,=r,r", [packed], dtype=(tl.float16, tl.float16), is_pure=True, pack=4
    )


@triton.jit
def _ue8m0(scale_u8):
    """UE8M0 byte -> fp32 2^(b-127), exactly, by building the float bit pattern."""
    return (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _split4(t, BN: tl.constexpr, W: tl.constexpr):
    """[BN, 4*W] register tile -> four [BN, W] column chunks, in order, without touching shared memory."""
    t = tl.reshape(t, [BN, 2, 2, W])  # (hi, lo, i)  with column = hi*2W + lo*W + i
    t = tl.permute(t, [0, 3, 2, 1])  # [BN, i, lo, hi]
    a, b = tl.split(t)  # hi = 0 -> chunks 0,1 ; hi = 1 -> chunks 2,3
    c0, c1 = tl.split(a)
    c2, c3 = tl.split(b)
    return c0, c1, c2, c3


@triton.jit
def _chunk_dot(x_base, xk, mask_m, packed, scale_u8):
    """One 32-wide K group: x[BM, 32] . w[BN, 32]^T with the packed FP4 chunk [BN, 16] decoded in registers.

    x_base: [BM, 1] pointers at the group's first element; xk: [1, 16] even offsets inside the group.
    Returns the fp32 [BM, BN] partial, already multiplied by the group's UE8M0 scale ([BN] uint8).
    """
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    we, wo = _fp4_decode(packed)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _quad_dot(x_base, xk, mask_m, w_tile_ptr, s_ptr, BN: tl.constexpr):
    """Four consecutive K groups (128 logical K) = one 64-byte-wide packed tile [BN, 64] + scales [BN, 4].

    Loading 64 contiguous bytes per row matters on GB10: 16-byte-wide row tiles cap at ~130 GB/s, 64-byte
    tiles reach ~210 GB/s (the box's practical copy bandwidth). The tile is split in registers.
    """
    packed = tl.load(w_tile_ptr)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)  # [BN, 4], column = hi*2 + lo
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))  # split on hi: (0,1) / (2,3)
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3)
    return acc


@triton.jit
def _quad_dot_scaled(x_base, mask_m, w_tile_ptr, s_ptr, acc, BM: tl.constexpr, BN: tl.constexpr):
    """Same 128 logical K as _quad_dot, but the tensor core eats the e2m1 codes and their UE8M0
    scales through Triton's `tl.dot_scaled` lowering instead of our manual decoder.

    _quad_dot spends real ALU on every weight byte -- `cvt.rn.f16x2.e2m1x2`, the prmt shuffles that
    regroup even/odd nibbles, and a per-32-group scale applied to the fp32 partial sum -- and that
    work competes with the MMA it is feeding. Measured on this box at MoE dimensions: software
    decode 19.8 TFLOPS, dot_scaled 29.5 TFLOPS.
    On this installed sm_121a backend the mixed BF16/FP4 operation lowers to BF16 MMA;
    it does not quantize the activation to FP4. Avoid inferring arithmetic from the API name.

    The layouts already line up: the caller's tile is [BN, 64] packed bytes = 128 logical K with
    [BN, 4] scales, which is exactly dot_scaled's [BK, BN] / [BN, BK//32] contract after the
    transpose. The K-permutation trick that _quad_dot needs (activations loaded at stride 2 to
    match even/odd nibble tiles) is not needed here -- the hardware reads the nibbles in natural K
    order -- so the activation tile is loaded contiguously.
    """
    packed = tl.load(w_tile_ptr)                       # [BN, 64] uint8, e2m1 two per byte
    sc = tl.load(s_ptr)                                # [BN, 4]  uint8, UE8M0, one per 32 K
    x = tl.load(x_base + tl.arange(0, 128)[None, :], mask=mask_m, other=0.0)   # [BM, 128] bf16
    # Resetting the MMA accumulator every 128 K and adding partials in FP32 was tested:
    # it passed depth-8 generation only by choosing compact JSON earlier, while the identical
    # spaced-prefix closing margin worsened from -0.25 to -1.0 and T=512 took 10.6 vs 8.8 ms.
    # Keep the negative: a tokenization-dependent pass is not a numerical fix.
    return tl.dot_scaled(x, None, "bf16", packed.trans(), sc, "e2m1", acc=acc)


# --------------------------------------------------------------------------- kernel 1: gate/up + SwiGLU
@triton.jit
def _moe_up_kernel(
    x_ptr, w1_ptr, s1_ptr, w3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, SCALED: tl.constexpr, NULL_SLOT: tl.constexpr = -1,
):
    KB: tl.constexpr = K // 2
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)  # padded pair block (all pairs of one expert)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    if slot == NULL_SLOT:
        return
    slot = slot.to(tl.int64)

    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))  # (token, k) pair ids, -1 = padding
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    w1_tile = w1_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    w3_tile = w3_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s1_tile = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    s3_tile = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        if SCALED:
            acc_g = _quad_dot_scaled(x_base + q * 128, mask_m[:, None], w1_tile + q * 64, s1_tile + q * 4, acc_g, BM, BN)
            acc_u = _quad_dot_scaled(x_base + q * 128, mask_m[:, None], w3_tile + q * 64, s3_tile + q * 4, acc_u, BM, BN)
        else:
            acc_g += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w1_tile + q * 64, s1_tile + q * 4, BN)
            acc_u += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w3_tile + q * 64, s3_tile + q * 4, BN)

    # Expert.forward's quantized linear returns BF16 (`fp4_gemm(..., out_dtype=BF16)`) and the
    # reference clamps that bf16-rounded value in fp32.  The tensor-core accumulator is fp32, so
    # without this round the gate/up retain FP32 mantissas; the SwiGLU output then differs from the
    # dequant path by ~4e-3, which is enough to flip a low-margin token in the verifier (the
    # nesting probe fails at depth 8 with fp32 but passes through the bf16 dequant fallback).
    gate = tl.minimum(acc_g.to(tl.bfloat16).to(tl.float32), limit)
    up = tl.minimum(tl.maximum(acc_u.to(tl.bfloat16).to(tl.float32), -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


# --------------------------------------------------------------------------- kernel 2: down + scatter
# `y_ptr` is a [TOPK, T, DIM] fp32 buffer: every (k, token) pair owns its own row and is written
# exactly once, so there are no atomics and no cross-block accumulation. The TOPK experts of a
# token are summed afterwards in torch, over the outermost axis, always in the same order.
# Accumulating here with tl.atomic_add(..., sem="relaxed") instead made the fp32 summation order
# depend on block scheduling -- CUDA guarantees nothing about it -- so the same token could come
# out different in a short and in a long prefill chunk, which the next layer's router amplifies.
# k-major and not pair-major: with [T*TOPK, DIM] the scattered fp32 stores cost ~50% at T=512,
# k-major costs ~6% (the reduction reads one contiguous [T, DIM] plane per k).
@triton.jit(do_not_specialize=['NTOK'])
def _moe_down_kernel(
    h_ptr, w2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK, SCALED: tl.constexpr, NULL_SLOT: tl.constexpr = -1,
    PARTIAL: tl.constexpr = False,
    PARTS_WORLD: tl.constexpr = 1,
):
    KB: tl.constexpr = K // 2
    SG: tl.constexpr = K // 32
    local_n: tl.constexpr = N // PARTS_WORLD
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)

    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    if slot == NULL_SLOT:
        # The up kernel skipped these pairs. Write their contribution explicitly:
        # parts is uninitialized, and its final reduction includes every pair.
        row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
        if PARTS_WORLD > 1:
            offsets = ((offs_n[None, :] // local_n * (TOPK * NTOK) + row[:, None])
                       * local_n + offs_n[None, :] % local_n)
        else:
            offsets = row[:, None] * stride_y + offs_n[None, :]
        tl.store(y_ptr + offsets, 0.0, mask=mask_m[:, None])
        return
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h  # h row = pair id
    xk = 2 * tl.arange(0, 16)[None, :]
    w2_tile = w2_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s2_tile = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        if SCALED:
            acc = _quad_dot_scaled(h_base + q * 128, mask_m[:, None], w2_tile + q * 64, s2_tile + q * 4, acc, BM, BN)
        else:
            acc += _quad_dot(h_base + q * 128, xk, mask_m[:, None], w2_tile + q * 64, s2_tile + q * 4, BN)

    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    # Expert.forward's quantized linear returns BF16.  The MoE then accumulates those BF16 expert
    # outputs into its FP32 y buffer.  Keeping each expert's down projection in FP32 until after
    # the top-k sum moves a real checkpoint rounding boundary and compounds across layers.
    value = acc if PARTIAL else acc.to(tl.bfloat16).to(tl.float32)
    if PARTS_WORLD > 1:
        offsets = ((offs_n[None, :] // local_n * (TOPK * NTOK) + row[:, None])
                   * local_n + offs_n[None, :] % local_n)
    else:
        offsets = row[:, None] * stride_y + offs_n[None, :]
    tl.store(y_ptr + offsets, value,
             mask=mask_m[:, None])


@triton.jit
def _tp_round_reduce(parts, out, TD: tl.constexpr, TOPK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for k in range(TOPK):
        x = tl.load(parts + k * TD + i, i < TD, 0)
        total += x.to(tl.bfloat16).to(tl.float32)
    tl.store(out + i, total, i < TD)


# --------------------------------------------------------------------------- routing
# Data-dependent sizes (pair counts, token counts, resident counts) are runtime arguments, never
# tl.constexpr: each distinct constexpr value is a separate compile. With P, NTOK and NS constexpr
# the cache held 5,382 _route_kernel / 3,123 _moe_down_kernel / 1,905 _route_counts variants, and
# a served prompt whose last chunk had a new length -- or a layer whose resident count a swap had
# just changed -- stopped the GPU for 0.5-0.9 s per compile (3.7 s of a 10 s 8K prefill). The
# index arithmetic is the same either way. do_not_specialize too: Triton otherwise still compiles a
# variant per (divisible by 16, == 1) pattern of each integer argument.
@triton.jit(do_not_specialize=['P'])
def _route_counts(slots_ptr, counts_ptr, P, PB: tl.constexpr):
    s = tl.program_id(0)
    p = tl.arange(0, PB)
    v = tl.load(slots_ptr + p, mask=p < P, other=-1)
    tl.store(counts_ptr + s, tl.sum((v == s).to(tl.int32), 0))


@triton.jit(do_not_specialize=['P', 'NB', 'NS'])
def _route_kernel(
    slots_ptr, block_slot_ptr, block_pair_ptr, P, NB, counts_ptr,
    S: tl.constexpr, PB: tl.constexpr, PC: tl.constexpr, BM: tl.constexpr, NBB: tl.constexpr,
    PRECOUNT: tl.constexpr = False, NS=0,
):
    """Program s owns arena slot s: finds its pairs and assigns them BM-padded block positions.

    block_pair[pos] = pair id (pos = block_start[s]*BM + rank of the pair inside slot s), -1 = padding
    block_slot[b]   = s for the blocks of slot s, -1 for the unused tail [total_blocks, NB)
    Block starts come from a per-slot histogram that every program recomputes in PC-sized chunks
    (O(P*S) per program: fine for P <= 4096 and S <= a few hundred slots).
    """
    s = tl.program_id(0)
    slot_ids = tl.arange(0, S)
    if PRECOUNT:
        counts = tl.load(counts_ptr + slot_ids, mask=slot_ids < NS, other=0)
    else:
        counts = tl.zeros([S], dtype=tl.int32)
        for c in range(0, PB, PC):
            offs_c = c + tl.arange(0, PC)
            v = tl.load(slots_ptr + offs_c, mask=offs_c < P, other=-1)
            counts += tl.sum((v[:, None] == slot_ids[None, :]).to(tl.int32), 0)
    nblk = (counts + BM - 1) // BM
    blk_start = tl.cumsum(nblk, 0) - nblk
    my_start = tl.sum(tl.where(slot_ids == s, blk_start, 0))
    my_nblk = tl.sum(tl.where(slot_ids == s, nblk, 0))
    total = tl.sum(nblk)

    # 1) padding: my blocks' pair rows <- -1, and (program 0) the unused block tail <- -1
    offs_pad = tl.arange(0, NBB * BM)
    tl.store(block_pair_ptr + my_start * BM + offs_pad, tl.full([NBB * BM], -1, tl.int32), mask=offs_pad < my_nblk * BM)
    bo = tl.arange(0, NBB)
    tl.store(block_slot_ptr + my_start + bo, tl.full([NBB], 0, tl.int32) + s, mask=bo < my_nblk)
    if s == 0:  # the unused tail [total, NB) can be longer than NBB -> loop
        for c in range(0, NB, NBB):
            idx = total + c + bo
            tl.store(block_slot_ptr + idx, tl.full([NBB], -1, tl.int32), mask=idx < NB)
    tl.debug_barrier()  # the -1 fill must land before this program's own pair ids below
    # 2) my pairs, in increasing pair order
    offs = tl.arange(0, PB)
    flat = tl.load(slots_ptr + offs, mask=offs < P, other=-1)
    mine = flat == s
    rank = tl.cumsum(mine.to(tl.int32), 0) - mine.to(tl.int32)
    tl.store(block_pair_ptr + my_start * BM + rank, offs.to(tl.int32), mask=mine)


def _next_pow2(v: int) -> int:
    return 1 << max(0, (v - 1).bit_length())


def build_routing_small(slots: torch.Tensor, BM: int):
    """Decode-sized routing (P <= 64 pairs) with plain torch ops: sort the pairs by slot, one BM-block per
    distinct slot. Static shapes, graph-capturable, ~10 tiny kernels instead of one program per ARENA
    slot (the Triton router costs ~29 ms per step with a 4,800-slot arena). Same output contract as
    build_routing; NB = P.

    HARD INVARIANT: **no slot may hold more than BM pairs.** This builder gives each distinct slot
    exactly one BM-wide block, so a run longer than BM makes `blk * BM + rank` walk out of its own
    block and overwrite the NEXT block's pair indices -- silently, with no error and no bounds fault,
    because the buffer is one flat P*BM tensor. The result is a token quietly losing a real expert's
    contribution (or gaining a stranger's): logits wrong in the third decimal, and text that is ~90 %
    coherent with mangled tokens sprinkled through it.

    On one box the invariant is free: a token's top-k picks are DISTINCT experts, so a slot collects
    at most T pairs (T <= 6 in the verify block) against BM = 16. Expert parallel breaks it -- every
    non-owned expert maps to the one shared null slot (engine/experts.py), so the verify block puts
    ~T*K/2 ~= 18 pairs there. That is the EP2 decode corruption; `moe_forward(slots_repeat=True)` is
    the fix, and it is not optional for any caller that can aim many pairs at one slot.

    DSV41_CHECK_ROUTING=1 turns the invariant into a real (host-syncing, decode-path-expensive)
    assertion. Leave it off in serving; it exists so a test can prove the violation rather than
    infer it from garbled prose."""
    T, K = slots.shape
    P = T * K
    flat = slots.reshape(-1).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    ss = flat[order]
    first = torch.ones(P, dtype=torch.bool, device=flat.device)
    first[1:] = ss[1:] != ss[:-1]
    blk = torch.cumsum(first.to(torch.int32), 0) - 1            # block id per sorted pair
    start = torch.cummax(torch.where(first, torch.arange(P, device=flat.device, dtype=torch.int32), torch.zeros_like(blk)), 0).values
    rank = torch.arange(P, device=flat.device, dtype=torch.int32) - start
    if os.environ.get("DSV41_CHECK_ROUTING") == "1":
        mx = int(rank.max()) + 1
        assert mx <= BM, (f"build_routing_small: a slot holds {mx} pairs but BM={BM}; the write would "
                          f"overflow into the next block. Pass slots_repeat=True (or a larger block_m).")
    block_pair = torch.full((P * BM,), -1, dtype=torch.int32, device=flat.device)
    block_pair[blk * BM + rank] = order.to(torch.int32)
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk] = ss
    return block_slot, block_pair, P


def build_routing(slots: torch.Tensor, n_slots: int, BM: int, precount: bool = False):
    """Group the (token, k) pairs by arena slot, each slot's run padded to a multiple of BM, so that
    one program == one expert x BM pairs. One small Triton launch,
    no host sync, so the CPU launch overhead of the main kernels overlaps with GPU work.

    Returns (block_slot i32[NB], block_pair i32[NB*BM], NB) with -1 marking unused blocks / padding rows.
    NB is an upper bound: ceil(P/BM) + min(n_slots, P).
    """
    T, K = slots.shape
    P = T * K
    dev = slots.device
    flat = slots.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    NB = (P + BM - 1) // BM + min(n_slots, P)
    block_slot = torch.empty((NB,), dtype=torch.int32, device=dev)
    block_pair = torch.empty((NB * BM,), dtype=torch.int32, device=dev)
    S = _next_pow2(max(n_slots, 2))
    PB = _next_pow2(max(P, 16))
    counts = torch.empty((n_slots,), dtype=torch.int32, device=dev) if precount else flat
    if precount:
        _route_counts[(n_slots,)](flat, counts, P=P, PB=PB)
    _route_kernel[(n_slots,)](
        flat, block_slot, block_pair, P, NB, counts,
        PRECOUNT=precount, NS=n_slots,
        S=S, PB=PB, PC=max(16, min(PB, 8192 // S)), BM=BM, NBB=_next_pow2((P + BM - 1) // BM + 1), num_warps=4,
    )
    return block_slot, block_pair, NB


def warm_routing(device, max_pairs: int, slot_counts=(200, 300)) -> int:
    """Compile build_routing's prefill variants before the first request instead of during it.

    With the sizes runtime arguments, the remaining constexprs are powers of two derived from the
    pair count (PB, NBB, BM) and the slot count (S, PC): a few dozen variants, cached on disk across
    restarts but recompiled whenever this file changes -- 0.5-0.9 s each, which a served prompt
    with a new chunk length would otherwise pay with the GPU idle. Rank-local, no collectives.
    Returns the number of variants launched (compiled, or loaded from the cache)."""
    seen = set()
    e = 7
    while (1 << (e - 1)) < max_pairs:
        pb = 1 << e
        for P in sorted({pb // 2 + 1, pb * 3 // 4, pb - 64, pb}):
            if not 64 < P <= max_pairs:
                continue
            # 32 as well above 1,024 pairs: the PREFILL_SCALED_TILES block size.
            for bm in sorted({_pick_bm(P), 32 if P > 1024 else _pick_bm(P)}):
                for n_slots in slot_counts:
                    S = _next_pow2(max(n_slots, 2))
                    for precount in (True, False):
                        key = (_next_pow2(max(P, 16)), _next_pow2((P + bm - 1) // bm + 1), bm, S,
                               max(16, min(_next_pow2(max(P, 16)), 8192 // S)), precount)
                        if key in seen:
                            continue
                        seen.add(key)
                        slots = torch.randint(0, n_slots, (P, 1), dtype=torch.int32, device=device)
                        build_routing(slots, n_slots, bm, precount=precount)
        e += 1
    torch.cuda.synchronize(device)
    return len(seen)


def _pick_bm(P: int) -> int:
    if P <= 64:
        return 16
    if P <= 1024:
        return 32
    return 64


# (BN, num_warps, num_stages) per kernel and BM, from the sweep on GB10 (tools/test_fp4_moe.py)
# (BN, num_warps, num_stages) per BM. Re-swept with dot_scaled on (DSV41_FP4_DOT_SCALED=1):
# letting the tensor core eat the FP4 codes removes most of the decode ALU, which moves the kernel
# from ALU-bound toward memory-bound -- and a memory-bound loop wants software pipelining, which
# num_stages=1 does not give it. The original tuples were swept against the software-decode kernel
# and are kept as the fallback for it; measured with dot_scaled at D=5120/INTER=2304:
#   decode  T=6     BM=16 (32, 4, 4)  1.22x over the shipped tuple
#   prefill T=2048  BM=64 (64, 4, 3)  1.09x
_UP_CFG = {16: (256, 4, 1), 32: (128, 4, 1), 64: (64, 4, 1)}
_UP_CFG_SCALED = {16: (32, 4, 4), 32: (128, 4, 3), 64: (64, 4, 3)}
# BM=64 is only selected when P > 1024, i.e. prefill chunks (decode is BM=16, mid-size is BM=32),
# so the 64 entry is prefill-only. tools/tune_fp4_prefill.py at T=512/2048 measured (128,4,2)
# ~7% faster than the old (128,8,3) on this board, bit-identical to it. Decode rows unchanged.
_DOWN_CFG = {16: (128, 8, 3), 32: (128, 4, 3), 64: (128, 4, 2)}


def moe_forward(
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    arena: ExpertArena,
    swiglu_limit: float = 10.0,
    block_m: int | None = None,
    up_cfg: tuple[int, int, int] | None = None,
    down_cfg: tuple[int, int, int] | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    slots_repeat: bool = False,
    null_slot: int = -1,
    routing_ids: torch.Tensor | None = None,
    routing_slot_map: torch.Tensor | None = None,
    stage_mark=None,
    prefill: bool = False,
    dream=None,
) -> torch.Tensor:
    """x bf16 [T, 5120], slots int32 [T, K] (arena slot per routed expert), weights fp32 [T, K] -> bf16 [T, 5120].

    `out_dtype` controls the routed sum, not each expert's BF16 linear output. The engine requests
    FP32 even on one box, adds the shared expert, and only then rounds the combined sum to BF16.
    Under expert parallel this call returns only *part* of the k-sum --
    the rest is on the peer -- so rounding it now would put a bf16 rounding on each half and a
    third on their sum. engine/model.py::moe asks for fp32, all-reduces, and rounds once at the
    same point the single-node path does. Standalone callers still default to BF16.

    `slots_repeat=True` says many (token, k) pairs may land on ONE slot -- true under expert
    parallel, where every non-owned expert shares the null slot. When `null_slot` is known, the
    decode router gives those zero-contribution pairs distinct temporary keys and maps them back to
    the null slot after grouping. Real experts can then use the sweep-tuned BM=16 without the null
    run overflowing its one block. Without `null_slot`, the conservative BM=64 fallback remains.
    See build_routing_small's HARD INVARIANT. Default False keeps the single-box path unchanged.

    `dream=(tier, indices)`: the dreaming FP8 tier of this layer (engine/dream_overlay.py::Fp8Layer)
    and the router's expert ids [T, K]. Its experts' pairs run through the FP4 kernels as usual and
    are then overwritten by the FP8 kernels -- h after up, parts after down -- in place."""
    # null_slot must denote a known zero expert, never an ordinary arena slot.
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    assert weights.shape == (T, K)
    P = T * K
    dev = x.device
    split_decode_null = slots_repeat and null_slot >= 0 and P <= 64
    BM = block_m or (64 if (slots_repeat and P <= 64 and not split_decode_null) else _pick_bm(P))
    scaled = DOT_SCALED or (PREFILL_DOT_SCALED and prefill)
    native_decode = CUDA_DECODE and not scaled and P <= 64 and BM == 16
    if native_decode:
        import fp4_moe_cuda as CUDA
    tiles = (scaled and prefill and PREFILL_SCALED_TILES and block_m is None and BM == 64
             and up_cfg is None and down_cfg is None)
    if tiles:
        BM = 32
    bn1, nw1, ns1 = up_cfg or ((128, 4, 3) if tiles else (_UP_CFG_SCALED if scaled else _UP_CFG)[BM])
    bn2, nw2, ns2 = down_cfg or ((128, 4, 4) if tiles else _DOWN_CFG[BM])
    # TP's 1152 intermediate columns are not divisible by the EP decode tile of
    # 256. Kernels intentionally omit N masks, so choose a complete tiling.
    if arena.w1.shape[1] % bn1:
        bn1 = 128
    assert arena.w1.shape[1] % bn1 == 0
    if routing_ids is not None and P > 64:
        assert routing_slot_map is not None and routing_ids.shape == slots.shape
        block_slot, block_pair, NB = build_routing(routing_ids, routing_slot_map.numel(), BM, precount=True)
        block_slot = torch.where(block_slot >= 0, routing_slot_map[block_slot.clamp_min(0)], block_slot)
    elif P <= 64:  # decode-sized call. build_routing_small needs BM >= the longest per-slot run.
        route_slots = slots
        if split_decode_null:
            # Every real expert appears at most T times because a token's top-k is distinct, hence
            # BM=16 is safe for it. The shared EP null slot can appear ~P/world times. Give each
            # null pair a unique, out-of-arena routing key so no run can overflow its block, then
            # map those block ids back to NULL_SLOT for the kernels: up returns immediately and
            # down writes the corresponding `parts` row as exact zero. Shapes stay static and all
            # operations are CUDA-graph capturable.
            pair = torch.arange(P, dtype=torch.int32, device=dev).view_as(slots)
            route_slots = torch.where(slots == null_slot, null_slot + 1 + pair, slots)
        if native_decode:
            route_slots = route_slots.to(torch.int32).contiguous()
            block_slot, block_pair, block_route, NB = CUDA.build_routing_small(route_slots, BM, K)
        else:
            block_slot, block_pair, NB = build_routing_small(route_slots, BM)
        if split_decode_null:
            block_slot = torch.where(block_slot > null_slot,
                                     torch.full_like(block_slot, null_slot), block_slot)
    else:
        # Compact the call's slot ids before routing. build_routing's cost tracks `n_slots` -- it
        # reserves and scans block space per ARENA slot -- not the number of experts this call
        # actually touches, and the two diverge badly once the arena is large. At the live 4,680
        # slot arena a 2,048-token chunk spent 162 ms here against 0.44 ms compacted (367x), and
        # with 40 layers per chunk that made it the single dominant cost of prefill: ~19.5 s of a
        # 23 s TTFT on a 5.5k-token prompt, against a measured 15.05 s of GPU time in the whole
        # MoE. The (slot, pair) assignments are identical; only the block ids are relabelled.
        #
        # A negative slot means "no pair here" and must NOT be compacted: run it through unique()
        # and it becomes a real compact id, routing pairs that were meant to be dropped.
        valid = slots >= 0
        uniq, inv = torch.unique(slots[valid], return_inverse=True)
        compact = torch.full_like(slots, -1, dtype=torch.int32)
        compact[valid] = inv.to(torch.int32)
        block_slot, block_pair, NB = build_routing(compact, int(uniq.numel()), BM)
        block_slot = torch.where(block_slot >= 0, uniq.to(torch.int32)[block_slot.clamp_min(0)],
                                 block_slot)
    if stage_mark is not None:
        stage_mark('grouping')
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()

    inter = arena.w1.shape[1]
    tp = getattr(arena, 'tp_world', 1) > 1
    if tp:
        from engine.collective_rails import group as collective_group
    tp_output = getattr(arena, 'tp_output', False)
    reduce_mode = os.environ.get('DSV41_TP_EXPERT_REDUCE', 'all_reduce')
    scatter = tp and (reduce_mode == 'scatter' or (reduce_mode == 'auto' and P > 64))
    h = torch.empty((P, inter), dtype=torch.bfloat16, device=dev)
    down_n = arena.w2.shape[1]
    parts = torch.empty((P, down_n), dtype=torch.float32, device=dev)  # one row per (token, k) pair
    if stage_mark is not None:
        stage_mark('scratch_alloc')
    if native_decode:
        CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h, wgt, block_slot, block_route,
                float(swiglu_limit), K, inter, DIM, NB, null_slot)
    else:
        _moe_up_kernel[(NB, inter // bn1)](
            x, arena.w1, arena.s1, arena.w3, arena.s3, h,
            wgt, block_slot, block_pair,
            x.stride(0), h.stride(0), float(swiglu_limit),
            TOPK=K, N=inter, K=DIM, BM=BM, BN=bn1, SCALED=scaled, NULL_SLOT=null_slot, num_warps=nw1, num_stages=ns1,
        )
    if dream is not None:
        import fp4_moe_cuda as CUDA8
        tier, dream_idx = dream
        dream_idx = dream_idx.reshape(-1).to(torch.int64).contiguous()
        CUDA8.dream_fp8_up(x, dream_idx, tier, h, wgt, float(swiglu_limit), K)
    if stage_mark is not None:
        stage_mark('up_gemm')
    down_k = inter
    if tp_output:
        # Preserve complete down-projection dot products: gather the BF16 activated
        # intermediate, then compute disjoint output rows. Splitting K changed a
        # borderline nesting decision even with FP32 partial reductions before BF16.
        gathered_h = torch.empty((arena.tp_world * P, inter), dtype=h.dtype, device=dev)
        torch.distributed.all_gather_into_tensor(gathered_h, h, group=collective_group())
        h = gathered_h.view(arena.tp_world, P, inter).transpose(0, 1).reshape(P, INTER)
        down_k = INTER
    if stage_mark is not None:
        stage_mark('intermediate_gather')
    parts_world = arena.tp_world if scatter and not tp_output else 1
    if native_decode:
        CUDA.down(h, arena.w2, arena.s2, parts, block_slot, block_pair, K, down_n, down_k,
                  T, NB, null_slot, tp and not tp_output, parts_world)
    else:
        _moe_down_kernel[(NB, down_n // bn2)](
            h, arena.w2, arena.s2, parts,
            block_slot, block_pair,
            h.stride(0), parts.stride(0),
            TOPK=K, N=down_n, K=down_k, BM=BM, BN=bn2, NTOK=T, SCALED=scaled, NULL_SLOT=null_slot,
            PARTIAL=tp and not tp_output, PARTS_WORLD=parts_world,
            num_warps=nw2, num_stages=ns2,
        )
    if dream is not None:
        if parts_world != 1 or down_k != tier.w2.shape[2]:
            raise ValueError("the dreaming FP8 tier needs the output TP layout (or one box)")
        CUDA8.dream_fp8_down(h, dream_idx, tier, parts, K, T, partial=tp and not tp_output)
    if stage_mark is not None:
        stage_mark('down_gemm')
    if tp_output:
        local = parts.view(K, T, down_n).sum(dim=0).to(out_dtype)
        gathered = torch.empty((arena.tp_world * T, down_n), dtype=out_dtype, device=dev)
        torch.distributed.all_gather_into_tensor(gathered, local, group=collective_group())
        result = gathered.view(arena.tp_world, T, down_n).transpose(0, 1).reshape(T, DIM)
        if stage_mark is not None:
            stage_mark('output_reduce_gather')
    elif tp:
        # Reconstruct EACH expert before its BF16 rounding boundary. Summing the
        # rank-local routed totals would move that boundary and resurrect a quality bug.
        if scatter:
            # Reduce each expert only to the rank owning these output columns. Round
            # and sum locally, then gather the much smaller routed totals. No early
            # BF16 rounding and no change to expert selection; less network traffic.
            world, width = arena.tp_world, DIM // arena.tp_world
            reduced = torch.empty((P, width), dtype=torch.float32, device=dev)
            torch.distributed.reduce_scatter_tensor(reduced, parts.view(world * P, width),
                                                  group=collective_group())
            local = torch.empty((T, width), dtype=out_dtype, device=dev)
            if native_decode:
                CUDA.round_reduce(reduced, local, K)
            else:
                _tp_round_reduce[(triton.cdiv(T * width, 1024),)](reduced, local, T * width, K, 1024)
            gathered = torch.empty((world * T, width), dtype=out_dtype, device=dev)
            torch.distributed.all_gather_into_tensor(gathered, local, group=collective_group())
            result = gathered.view(world, T, width).transpose(0, 1).reshape(T, DIM)
        else:
            torch.distributed.all_reduce(parts, group=collective_group())
            result = torch.empty((T, DIM), dtype=out_dtype, device=dev)
            if native_decode:
                CUDA.round_reduce(parts, result, K)
            else:
                _tp_round_reduce[(triton.cdiv(T * DIM, 1024),)](parts, result, T * DIM, K, 1024)
    else:
        result = parts.view(K, T, DIM).sum(dim=0).to(out_dtype)
    if stage_mark is not None:
        stage_mark('reduce')
    return result


@torch.no_grad()
def moe_forward_reference(
    x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: ExpertArena,
    swiglu_limit: float = 10.0, out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequant + torch matmul per expert, mirroring inference/model.py::Expert.forward (bf16 GEMMs, fp32 epilogue)."""
    T, K = slots.shape
    y = torch.zeros((T, DIM), dtype=torch.float32, device=x.device)
    for s in torch.unique(slots).tolist():
        w1, w2, w3 = arena.dequant_slot(s)
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        xs = x[t_idx]
        gate = (xs @ w1.T).float()
        up = (xs @ w3.T).float()
        if swiglu_limit > 0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        hs = torch.nn.functional.silu(gate) * up * weights[t_idx, k_idx].float()[:, None]
        y.index_add_(0, t_idx, (hs.to(torch.bfloat16) @ w2.T).float())
    return y.to(out_dtype)
