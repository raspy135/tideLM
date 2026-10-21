"""
v41_ref.py -- a pure-PyTorch re-implementation of the DeepSeek-V4.1-Flash text
forward pass for SHORT sequences (<= 512 tokens), used by tools/expert_trace.py.

Why this exists: the reference `inference/model.py` needs tilelang CUDA kernels,
a TP=8 checkpoint conversion, and the whole 510 GB model resident. For an expert
routing trace we only need to run the model layer by layer (streaming one 7.4 GB
layer shard at a time) over teacher-forced prompts, and record the router's
top-6 choices. Everything here mirrors `inference/model.py` and `kernel.py`
(read them side by side); deviations are listed at the bottom of this docstring.

Exactness argument for the attention path (see NOTES.md, Phase 0):
  * index_topk = 512 and candidate pool = 2048 blocks x 8 positions. For a
    sequence of T <= 512 tokens the number of compressed positions is
    T // ratio <= 512, so the indexer's top-k keeps EVERY causally visible
    position and the candidate filter never prunes. The indexer therefore has
    no effect on the output and is skipped here. This is exact, not an
    approximation, as long as T <= 512.
  * Sliding window (128) + all completed compressed groups, both concatenated
    into one softmax with the learned attention sink -- same as `sparse_attn`.

Deviations from the reference (all documented, none affect routing materially):
  * GEMMs run in bf16 with fp32 accumulation instead of fp8 x fp8 / fp8 x fp4
    tensor-core GEMMs. Activations ARE fake-quantized to fp8 (per-32 groups,
    power-of-two E8M0 scales) before every quantized linear, exactly like
    `act_quant(..., scale_fmt="ue8m0")`, and weights are dequantized with their
    stored block scales, so the numerics differ only by accumulation order.
  * The softmax probabilities are kept in fp32 for the PV product (the kernel
    casts them to bf16 first).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import decode_rows

try:
    from fp8_linear import FP8Weight, fp8_linear, quantize_to_fp8  # tools/fp8_linear.py (Triton); dense weights stay fp8 in memory
except Exception:  # noqa: BLE001
    FP8Weight, fp8_linear, quantize_to_fp8 = None, None, None

try:
    from fp8_linear import FP8GroupedWeight, fp8_grouped_linear  # wo_a stays fp8 too (grouped GEMM)
except Exception:  # noqa: BLE001
    FP8GroupedWeight, fp8_grouped_linear = None, None

try:  # dense projections re-quantized from fp8 to fp4 at load time (opt-in, see dense_fp4_groups)
    from fp4_linear import (FP4GroupedWeight, FP4Weight, fp4_grouped_linear, fp4_linear,
                            quantize_fp8_grouped_to_fp4, quantize_fp8_to_fp4, quantize_to_fp4)
except Exception:  # noqa: BLE001
    FP4Weight, fp4_linear, quantize_fp8_to_fp4, quantize_to_fp4 = None, None, None, None
    FP4GroupedWeight, fp4_grouped_linear, quantize_fp8_grouped_to_fp4 = None, None, None


# --------------------------------------------------------------------------- dense FP4 opt-in
# DSV41_DENSE_FP4 selects which groups of dense projections are re-quantized from their stored fp8
# (e4m3 + UE8M0 32x32 block scales) to fp4 (E2M1 + one UE8M0 scale per 32 K weights of a row) when
# the weights are loaded:
#   off              (default) -- everything stays fp8, the old path
#   shared           -- ffn.shared_experts.w1/w2/w3 of every backbone layer and DSpark block
#   attn             -- attn.wq_a, attn.wq_b, attn.wkv, attn.wo_b (the `_fp8_linear_kernel` group)
#   wo_a             -- attn.wo_a, the [8, 1024, 4096] output LoRA on the GROUPED kernel
#   any comma combination;  "all" = shared,attn,wo_a
# The indexer wq_b, the engram wkv and mtp.0.main_proj always stay fp8 -- they are not covered by
# this switch.
_ALL_FP4_GROUPS = ("shared", "attn", "wo_a")
_FP4_ALIASES = {"all": _ALL_FP4_GROUPS, "on": _ALL_FP4_GROUPS, "1": _ALL_FP4_GROUPS}


# DSV41_TOPK overrides the number of routed experts the router activates per token. The checkpoint
# stores 6 (`n_activated_experts` in inference/config.json) out of 384, and every token also passes
# through the layer's shared expert, which this switch does not touch. A smaller k keeps the highest
# gate logits and drops the rest; the gate weights are renormalized over the surviving experts by
# the same line that already normalizes the checkpoint's six, so the routed sum stays at
# `route_scale` and nothing downstream changes shape. Unset (or equal to the checkpoint's value)
# restores the stored routing exactly. The DSpark drafter's own router (top-3 of its 128 experts per
# MTP block) is a separate, hard-coded k and is not affected.
def routed_topk(default: int = 6) -> int:
    v = os.environ.get("DSV41_TOPK", "").strip()
    if v in ("", "off", "default"):
        return int(default)
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"DSV41_TOPK: {v!r}; use an integer 1..{default}, or leave it unset") from None
    if not 1 <= k <= int(default):
        raise ValueError(f"DSV41_TOPK: {k}; must be 1..{default} (the checkpoint's n_activated_experts)")
    return k


def dense_fp4_groups() -> frozenset:
    v = os.environ.get("DSV41_DENSE_FP4", "off").strip().lower()
    if v in ("", "off", "0", "none"):
        return frozenset()
    out = set()
    for part in v.replace(" ", "").split(","):
        if not part:
            continue
        out.update(_FP4_ALIASES.get(part, (part,)))
    unknown = out - set(_ALL_FP4_GROUPS)
    if unknown:
        raise ValueError(f"DSV41_DENSE_FP4: unknown group(s) {sorted(unknown)}; "
                         f"use {' / '.join(_ALL_FP4_GROUPS)} / all / off")
    if out and FP4Weight is None:
        raise RuntimeError("DSV41_DENSE_FP4 is set but tools/fp4_linear.py could not be imported")
    return frozenset(out)


_FP4_GROUP_OF = {
    "attn.wq_a": "attn", "attn.wq_b": "attn", "attn.wkv": "attn", "attn.wo_b": "attn",
    "ffn.shared_experts.w1": "shared", "ffn.shared_experts.w2": "shared", "ffn.shared_experts.w3": "shared",
}


def maybe_fp4(w, name: str, groups):
    """Re-quantize an FP8Weight to FP4 if `name`'s group is enabled. `name` is the suffix inside the
    layer / DSpark block, e.g. "attn.wq_b" or "ffn.shared_experts.w1"."""
    if not groups or not isinstance(w, FP8Weight) or _FP4_GROUP_OF.get(name) not in groups:
        return w
    out = quantize_fp8_to_fp4(w)
    del w
    return out

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# ----------------------------------------------------------------------------- dtype helpers
def e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    """UE8M0 scale bytes -> float32 (2^(b-127)); avoids relying on torch's e8m0 cast support."""
    if s.dtype == torch.uint8:
        b = s
    else:
        b = s.view(torch.uint8)
    return torch.exp2(b.to(torch.float32) - 127.0)


def fp8_to_float(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float32)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp8 [N,K] with UE8M0 scale [ceil(N/b), ceil(K/b)] -> bf16 [N,K]."""
    n, k = weight.shape
    s = e8m0_to_float(scale)
    s = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def dequant_fp4_packed(weight_i8: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Packed FP4 [N, K/2] (int8/uint8, low nibble = even element) with UE8M0 scale [N, K/32] -> bf16 [N,K].
    Packing order matches `convert.py:cast_e2m1fn_to_e4m3fn`."""
    x = weight_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    tab = FP4_TABLE.to(x.device)
    vals = torch.stack([tab[low.long()], tab[high.long()]], dim=-1).flatten(1)  # [N, K]
    s = e8m0_to_float(scale).repeat_interleave(group, 1)  # [N, K]
    return (vals * s).to(torch.bfloat16)


def act_qdq_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Emulates kernel.act_quant(x, 32, scale_fmt='ue8m0') followed by dequantization:
    per-row per-32-group amax, scale rounded UP to a power of two, values clamped to +-448 and
    rounded to e4m3. Returns bf16 of the same shape."""
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(shape).to(torch.bfloat16)


def fp4_qdq(x: torch.Tensor, group: int, scale_dtype: str,
            grid: torch.Tensor | None = None) -> torch.Tensor:
    """Emulates kernel.fp4_act_quant(x, group, inplace=True): FP4 e2m1 values with either
    UE8M0 (power-of-two) or E4M3 group scales, dequantized back. Used for the compressed
    (global) KV cache which the model stores in FP4 (E4M3 scales, groups of 16)."""
    shape = x.shape
    xf = x.float().reshape(-1, group)
    amax = xf.abs().amax(dim=1, keepdim=True)
    if scale_dtype == "e4m3":
        amax = amax.clamp_min(6 * 2 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2 ** -126)
        s = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    v = (xf / s).clamp(-6.0, 6.0)
    grid = FP4_GRID.to(x.device) if grid is None else grid
    assert grid.device == x.device
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(v.abs(), mid)  # nearest grid point (ties go up; measure-zero on real data)
    q = grid[idx] * torch.sign(v)
    return (q * s).reshape(shape).to(x.dtype)


# ----------------------------------------------------------------------------- rope
@lru_cache(8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device="cuda"):
    """Verbatim port of model.precompute_freqs_cis (YaRN when original_seq_len > 0)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of x (adjacent pairs as complex). x: [s, d] or [s, h, d]; freqs_cis: [s, d/2]."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        f = f.view(xc.size(0), 1, xc.size(-1))
    y = torch.view_as_real(xc * f).flatten(-2)
    return y.to(x.dtype)


# ----------------------------------------------------------------------------- args
@dataclass
class Args:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    kv_source_layers: tuple = (2, 8, 14, 20)
    index_source_layers: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_topk: int = 512
    index_n_heads: int = 32
    index_head_dim: int = 128
    candidate_source_layer: int = 20
    dspark_target_layer_ids: tuple = (37, 38, 39)
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    compress_rope_theta: float = 160000.0
    compress_ratios: tuple = tuple([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = (1, 14)
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_max_ngram_size: int = 4

    @classmethod
    def from_json(cls, path: str) -> "Args":
        cfg = json.load(open(path))
        keep = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        for k in ("kv_source_layers", "index_source_layers", "compress_ratios", "engram_layer_ids", "dspark_target_layer_ids"):
            if k in keep:
                keep[k] = tuple(keep[k])
        args = cls(**keep)
        args.n_activated_experts = routed_topk(args.n_activated_experts)
        return args


def tp_dense() -> tuple[int, int]:
    """(rank, world) for dense tensor parallelism, or (0, 1) when it is off.

    DSV41_TP_DENSE splits the weights that EP2 leaves REPLICATED -- shared experts, and later
    attention and the head. Those are read in full on both ranks every decode step (8.06 GB against
    7-13 GB for the EP-split routed experts), so the roofline argument was 19-27% fewer bytes.

    MEASURED: it does not pay off. Shared experts alone (0.71 GB/step), 8k/512, three runs:

        TP off   ttft 14919 ms   tpot 51.13 ms   19.56 tok/s   accept 3.56   (step ~182 ms)
        TP on    ttft 19808 ms   tpot 69.48 ms   14.39 tok/s   accept 2.87   (step ~199 ms)

    The roofline argument assumes decode is bandwidth-bound. It is not: GPU utilization during
    decode fluctuates 70-90% on the live server rather than pinning at the ceiling, so the step is
    losing time to gaps, not to reads. With speculation the shared-expert GEMM is
    [6,5120]x[5120,2304], skinny enough to be launch- and latency-bound, so halving N does not
    halve the time -- while the 40 added all-reduces per step lengthen exactly the serial chain
    that is already leaving the GPU idle. Shaving bytes cannot help until the gaps are closed. Attention would be the same shape
    with another 40 collectives, so the bigger 5.41 GB prize is unlikely to behave differently.

    Caveats on the numbers: the two runs are an hour apart and the demand DB decays between them,
    and the per-run decode spread is wide (16-22 tok/s baseline), so "slightly slower" is the
    honest reading rather than "26% slower". The accept_len drop 3.56 -> 2.87 is the one part that
    is not explained by noise; summing partials changes the numerics the drafter sees.

    Kept behind the flag, default off, because it is correct (verified against EP2 output) and it
    is where attention/head sharding would build from if the collective cost ever drops.
    """
    if os.environ.get("DSV41_TP_DENSE", "0") != "1":
        return 0, 1
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def _tp_shard(w, dim: int):
    rank, world = tp_dense()
    if world <= 1:
        return w
    from engine.tensor_parallel import shard, RowParallelWeight, OutputParallelWeight
    layout = os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output')
    if layout not in ('intermediate', 'output'):
        raise ValueError('TP linear layout must be intermediate or output')
    if dim == 1 and layout == 'output':
        return OutputParallelWeight(shard(w, 0, rank, world), world)
    local = shard(w, dim, rank, world)
    return RowParallelWeight(local) if dim == 1 else local


# ----------------------------------------------------------------------------- weights
class LayerWeights:
    """Everything of one backbone layer except the routed experts, dequantized to bf16 on `device`."""

    def __init__(self, get, layer: int, args: Args, device: str):
        p = f"layers.{layer}."
        self.layer = layer
        self.args = args
        dev = device

        def bf(name):
            return get(p + name).to(dev).to(torch.bfloat16)

        def f32(name):
            return get(p + name).to(dev).to(torch.float32)

        fp4_groups = dense_fp4_groups()

        def fp8lin(name):
            w, sc = get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev)
            if FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1":
                # kept in the stored format: half the bytes of bf16, and fp4 again if enabled
                return maybe_fp4(FP8Weight(w, sc), name, fp4_groups)
            return dequant_fp8_block(w, sc)

        self.attn_norm = bf("attn_norm.weight")
        self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink")
        self.q_norm = bf("attn.q_norm.weight")
        self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a")
        self.wq_b = fp8lin("attn.wq_b")
        self.wkv = fp8lin("attn.wkv")
        self.wo_a = make_wo_a(get(p + "attn.wo_a.weight").to(dev), get(p + "attn.wo_a.scale").to(dev), args, fp4_groups)
        self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn")
        self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base")
        self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale")
        self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight")
        self.gate_bias = f32("ffn.gate.bias")
        # Image tokens route through a SEPARATE router bias (inference/model.py:819). Absent on a
        # text-only checkpoint, so this stays None and the vision branch never runs.
        try:
            self.gate_bias_vl = f32("ffn.gate.bias_vl")
        except Exception:  # noqa: BLE001
            self.gate_bias_vl = None
        # Tensor-parallel shared expert: w1/w3 are column-parallel (split the intermediate dim,
        # i.e. their OUTPUT rows), w2 is row-parallel (split its INPUT columns to match). Each rank
        # then computes a PARTIAL [T, dim] that the caller must all-reduce -- see Model.moe.
        # Off by default; the dense path is replicated under EP2 and read in full on both ranks.
        self.sh_w1 = _tp_shard(fp8lin("ffn.shared_experts.w1"), 0)
        self.sh_w2 = _tp_shard(fp8lin("ffn.shared_experts.w2"), 1)
        self.sh_w3 = _tp_shard(fp8lin("ffn.shared_experts.w3"), 0)
        self.ratio = args.compress_ratios[layer]
        self.is_kv_source = layer in args.kv_source_layers
        if self.is_kv_source:
            self.comp_norm = bf("attn.compressor.norm.weight")
            if self.ratio > 1:
                self.comp_wkv = f32("attn.compressor.wkv.weight")
                self.comp_wgate = f32("attn.compressor.wgate.weight")
            else:
                self.comp_wkv = bf("attn.compressor.wkv.weight")


def make_wo_a(weight, scale, args, groups=None):
    """The attention output LoRA. `convert.py` dequantizes wo_a to bf16 and everything downstream
    kept it that way: [8, 1024, 4096] bf16 = 67 MB per layer, 2.7 GB read per decode step. With
    DSV41_WOA_FP8=1 (default) it stays in the stored fp8 format and the grouped Triton GEMM reads
    half the bytes; DSV41_WOA_FP8=0 restores the dequantized-bf16 tensor and the einsum.

    With "wo_a" in DSV41_DENSE_FP4 it is re-quantized once more, to the packed FP4 format, and the
    grouped FP4 kernel reads 0.5307x of the fp8 bytes (17.8 MB instead of 33.6 per layer)."""
    if FP8GroupedWeight is not None and os.environ.get("DSV41_WOA_FP8", "1") == "1":
        w = FP8GroupedWeight(weight, scale, args.o_groups, args.o_lora_rank)
        if "wo_a" in (dense_fp4_groups() if groups is None else groups):
            out = quantize_fp8_grouped_to_fp4(w)
            del w
            return out
        return w
    return dequant_fp8_block(weight, scale).view(args.o_groups, args.o_lora_rank, -1)


def wo_a_proj(o: torch.Tensor, w, tiled: bool = False) -> torch.Tensor:
    """o bf16 [T, G, K] -> [T, G, R]; einsum("sgd,grd->sgr", o, wo_a) or its grouped GEMM.

    Three stored formats: a bf16 [G, R, K] tensor (the einsum), an FP8GroupedWeight (the fp8
    grouped kernel) or an FP4GroupedWeight (the fp4 grouped kernel, at every T; with
    DSV41_FP4_DENSE_PREFILL=dequant prefill falls back to a transient dequant + einsum).
    Both Triton kernels are row-count- and row-offset-invariant by construction, so they never need
    the MM_TILE row tiling the bf16 einsum needs for chunk invariance.
    """
    if FP4GroupedWeight is not None and isinstance(w, FP4GroupedWeight):
        if o.size(0) <= 16 or os.environ.get("DSV41_FP4_DENSE_PREFILL", "kernel") == "kernel":
            return fp4_grouped_linear(o, w)
        return torch.einsum("sgd,grd->sgr", o.to(torch.bfloat16), w.dequant())
    if FP8GroupedWeight is not None and isinstance(w, FP8GroupedWeight):
        return fp8_grouped_linear(o, w)
    fn = (lambda t: torch.einsum("sgd,grd->sgr", t, w))
    return tiled_rows(fn, o) if tiled else fn(o)


class EngramWeights:
    def __init__(self, get, layer: int, device: str):
        p = f"layers.{layer}.engram."
        w, sc = get(p + "wkv.weight").to(device), get(p + "wkv.scale").to(device)
        self.wkv = FP8Weight(w, sc) if (FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1") else dequant_fp8_block(w, sc)
        self.q_weight = get(p + "q_weight").to(device).float()
        self.k_weight = get(p + "k_weight").to(device).float()


# ----------------------------------------------------------------------------- ops
MM_TILE = 0  # 0 = plain GEMMs. >0 = run every activation GEMM in fixed-size row tiles; see mm().
HC_MM_TILE = int(os.environ.get("DSV41_HC_MM_TILE", "32"))
# Memoize FP8Weight.dequant() on the weight object. The dequant is exact and deterministic, so a
# cached bf16 copy is bit-identical to recomputing it, and it is otherwise re-run for every dense
# projection of every layer of every prefill chunk (measured 40-58% of each such call). Gated off
# by default: on the 90 GB / keep 0.61 TP profile it held ~10 GiB resident (MemAvailable 15.1 ->
# 5.1 GiB), which is below the configured KEEP_FREE_GB floor, and its prefill benefit was within
# the run-to-run spread of chunk 2048 alone. Only enable when the box has the headroom.
DENSE_DEQUANT_CACHE = os.environ.get("DSV41_DENSE_DEQUANT_CACHE", "0") == "1"
if HC_MM_TILE not in (16, 32):
    raise ValueError("DSV41_HC_MM_TILE must be 16 or 32")

# DSV41_HC_KERNEL: run the Hyper-Connection mix projection on the split-K fp32 kernel
# (tools/fp32_skinny.py) instead of the padded cuBLAS GEMM. OFF by default; =1 tries it.
# It is faster per STEP but not per SECOND: the gate A/B (docs/decode-fp32-experiments.md 2026-10-01,
# results/hc-kernel-b/) has the step 1.9-2.9 ms faster and acceptance 0.07-0.10 tokens/step lower, so
# tok/s is a wash (explain went backwards) and every workload's tokens change. That is a numerics
# change (~1e-6), not a speedup, and it is why b79092a removed the earlier wiring. Do not flip this
# without the pair A/B and the quality probes.
# 2026-10-08: that one-prompt acceptance drop was noise. Paired over 22 prompts, together with
# staged attention and the BF16 router: -7.30 +- 0.50 ms/step, acceptance +0.003 +- 0.026
# (RESULTS.md 2026-10-08). Judge acceptance over many prompts, never one.
# DSV41_HC_PREC picks the tl.dot input precision: "ieee" is true fp32 (19.6 us cold), "tf32x3" is
# the 3-pass tensor-core emulation (13.4 us, ~2.5e-6 vs cuBLAS). See tools/bench_fp32_skinny.py.
HC_KERNEL = os.environ.get("DSV41_HC_KERNEL", "0") == "1"
HC_PREC = os.environ.get("DSV41_HC_PREC", "tf32x3")
try:
    from fp32_skinny import (BLOCK_MP as _SKINNY_MAX_M, skinny_linear as _skinny_linear,
                             wins as _skinny_wins)
except Exception:  # noqa: BLE001
    _skinny_linear = _skinny_wins = None
    _SKINNY_MAX_M = 0


def hc_kernel_ok(w, m: int) -> bool:
    """Whether the HC projection for weight `w` with `m` activation rows may use the split-K kernel.
    The row bound is the kernel's BLOCK_MP, not an assert inside it: DSV41_BLOCK can widen the verify
    block past it and must fall back to cuBLAS rather than take the pair out of step mid-request."""
    return (HC_KERNEL and _skinny_linear is not None and m <= _SKINNY_MAX_M
            and torch.is_tensor(w) and w.dtype == torch.float32 and w.dim() == 2
            and _skinny_wins(w.size(0)))


def hc_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The HC mix projection: the split-K kernel when configured and the rows fit, else mm()'s
    padded cuBLAS GEMM. Called from Model._hc_mixes and engine/decode_lean.py::hc_mixes."""
    if x.dim() == 2 and x.size(1) == w.size(1) and hc_kernel_ok(w, x.size(0)):
        return _skinny_linear(x, w, prec=HC_PREC)
    return mm(x, w)


# --------------------------------------------------------------------------- the LM head's format
# DSV41_HEAD_FMT picks the stored format of `head.weight` ([129280, 5120], the one weight that is
# read in full on every decode step):
#   bf16  (default) -- the checkpoint's own dtype, 1.324 GB per read, a cuBLAS GEMM
#   fp8             -- e4m3 codes + one UE8M0 scale per 32x32 block, 0.663 GB, `_fp8_linear_kernel`
#   fp4             -- E2M1 codes + one UE8M0 scale per 32 K weights of a row, 0.351 GB,
#                      `_fp4_linear_kernel`
# The quantization happens once, on the GPU, as the weight is loaded. DSV41_HEAD_FP32=1 (the
# reference's fp32 head) stays available and is checked first; the two cannot be combined.
_HEAD_FMTS = ("bf16", "fp8", "fp4")
# DSV41_DRAFT_HEAD_FMT: a second, cheaper copy of head.weight that ONLY the DSpark drafter reads
# (`off` disables; `fp8` is the default). The verifier keeps the main head, so greedy output is
# unchanged -- verification still accepts the target's argmax -- but the draft distribution moves,
# and with it the accepted length, so a pair being measured should compare accept_len_mean and
# tok/s, not just ms/step. Only built from a plain bf16/fp32 head and only when the requested
# format is strictly smaller than the verifier's; see make_draft_head.
_DRAFT_HEAD_FMTS = ("off",) + _HEAD_FMTS
# stored bytes per weight, so a draft format is only built when it is actually cheaper
_HEAD_FORMAT_RANK = {"fp4": 0, "fp8": 1, "bf16": 2, "fp32": 3}
_HEAD_DTYPE_RANK = {torch.float32: 3, torch.bfloat16: 2}


def head_fmt() -> str:
    v = os.environ.get("DSV41_HEAD_FMT", "bf16").strip().lower() or "bf16"
    if v not in _HEAD_FMTS:
        raise ValueError(f"DSV41_HEAD_FMT: {v!r}; use {' / '.join(_HEAD_FMTS)}")
    return v


def draft_head_fmt() -> str:
    v = os.environ.get("DSV41_DRAFT_HEAD_FMT", "fp8").strip().lower() or "fp8"
    if v not in _DRAFT_HEAD_FMTS:
        raise ValueError(f"DSV41_DRAFT_HEAD_FMT: {v!r}; use {' / '.join(_DRAFT_HEAD_FMTS)}")
    return v


def _head_in_format(t: torch.Tensor, fmt: str):
    if fmt == "bf16":
        return t.to(torch.bfloat16)
    if fmt == "fp8":
        if quantize_to_fp8 is None:
            raise RuntimeError("head format fp8 but tools/fp8_linear.py could not be imported")
        return quantize_to_fp8(t.to(torch.bfloat16))
    if fmt == "fp4":
        if quantize_to_fp4 is None:
            raise RuntimeError("head format fp4 but tools/fp4_linear.py could not be imported")
        return quantize_to_fp4(t.to(torch.bfloat16))
    raise ValueError(fmt)


def make_head(t: torch.Tensor):
    """`head.weight` in the format DSV41_HEAD_FMT asks for: a bf16 (or fp32) tensor, an FP8Weight
    or an FP4Weight. Every consumer goes through `head_logits` or `dense`, both of which dispatch
    on the object, so nothing else has to know which one it got."""
    from engine.native_head import validate_config, make_packed_head
    if validate_config() == 'packed':
        return make_packed_head(t.to(torch.bfloat16).contiguous())
    if os.environ.get("DSV41_HEAD_FP32", "0") == "1":
        assert head_fmt() == "bf16", "DSV41_HEAD_FP32=1 and DSV41_HEAD_FMT are mutually exclusive"
        return t.float()
    return _head_in_format(t, head_fmt())


def make_draft_head(head):
    """The drafter's own copy of the LM head, or None to reuse `head` (the verifier's).

    A quantized or TP-split main head has no cheap bf16 source to copy from, and a draft format
    that is not strictly smaller than the main one would only add memory, so both return None.
    DSV41_HEAD_FP32=1 is the named reference path and is left alone."""
    fmt = draft_head_fmt()
    if fmt == "off" or not isinstance(head, torch.Tensor):
        return None
    if os.environ.get("DSV41_HEAD_FP32", "0") == "1":
        return None
    if _HEAD_FORMAT_RANK.get(fmt, 9) >= _HEAD_DTYPE_RANK.get(head.dtype, 9):
        return None
    return _head_in_format(head, fmt)


def _head_blocked(x: torch.Tensor, head, rows: int = 16384) -> torch.Tensor:
    """Prefill path for a quantized head: dequantize `rows` vocabulary rows at a time and hand each
    block to cuBLAS.

    Both kernels are bandwidth-shaped -- one fp32 accumulator per output, a 128-wide N tile and a
    64-row M tile at prefill M -- which is what a 6-row call wants and not what a 512-row one wants:
    measured on this weight at M = 512 they run at 24-27 TFLOPs against cuBLAS's 80. Dequantizing the whole head
    to bf16 first would fix that and cost a 1.324 GB transient -- 70 expert slots' worth of arena --
    so it is done a block at a time instead, 166 MB at 16,384 rows, and the extra DRAM traffic is
    one read plus one write of the head per call."""
    N = head.shape[0]
    out = torch.empty(x.size(0), N, dtype=torch.float32, device=x.device)
    xb = x.to(torch.bfloat16)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        if FP4Weight is not None and isinstance(head, FP4Weight):
            blk = dequant_fp4_packed(head.w[r0:r1], head.s[r0:r1])
        else:
            sc = torch.exp2(head.s[r0 // 32:(r1 + 31) // 32].float() - 127.0)
            sc = sc.repeat_interleave(32, 0)[: r1 - r0].repeat_interleave(32, 1)[:, : head.K]
            blk = (head.w[r0:r1].float() * sc).to(torch.bfloat16)
            del sc
        out[:, r0:r1] = F.linear(xb, blk).float()
        del blk
    return out


def head_logits(x: torch.Tensor, head) -> torch.Tensor:
    """LM-head logits in fp32. A bf16 head runs a bf16 GEMM (fp32 accumulate, bf16 logits) and is
    what the fast decode path always did; an fp32 head (DSV41_HEAD_FP32=1) is the reference's math.

    A quantized head (DSV41_HEAD_FMT) runs its own Triton kernel at decode-sized M and the blocked
    dequant + cuBLAS above it; DSV41_HEAD_PREFILL=kernel runs the kernel at every M instead."""
    if hasattr(head, 'tp_logits'):
        return head.tp_logits(x)
    if hasattr(head, 'native_logits'):
        return head.native_logits(x)
    quant = ((FP8Weight is not None and isinstance(head, FP8Weight))
             or (FP4Weight is not None and isinstance(head, FP4Weight)))
    if quant:
        M = x.numel() // x.shape[-1]
        if M > decode_rows.limit() and os.environ.get("DSV41_HEAD_PREFILL", "blocked") != "kernel":
            return _head_blocked(x.reshape(-1, x.shape[-1]), head).view(*x.shape[:-1], head.shape[0])
        fn = fp8_linear if isinstance(head, FP8Weight) else fp4_linear
        return fn(x.to(torch.bfloat16), head).float()
    if head.dtype == torch.float32:
        return mm(x.float(), head)
    return mm(x.to(torch.bfloat16), head).float()


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """F.linear(x, w), padding decode-sized calls to a fixed row count.

    cuBLAS picks its tile shape AND its split-K count from M, so F.linear(x[:m], w) is generally
    NOT the first m rows of F.linear(x, w) -- for the small-N projections here (wkv N=512, the
    router gate N=384, hc_fn N=24) the two differ by ~1e-4 relative, which is enough to round a
    bf16 activation to a different ulp and to flip a borderline router top-k. Padding M=1..16 to
    MM_TILE makes sequential decode and a speculative verify block identical. Larger calls are
    prefill and stay as one GEMM; splitting a 2048-row projection into 128 launches is both
    unnecessary for speculative correctness and substantially slower. Set MM_TILE from
    engine/model.py; 0 keeps plain behaviour for tools/expert_trace.py and the stored trace.
    The small FP32 HC projection uses HC_MM_TILE (default 32) instead; both single-token
    and speculative calls use that same shape. See docs/decode-fp32-experiments.md.
    """
    if hasattr(w, 'tp_linear'):
        return w.tp_linear(x)
    if hasattr(w, 'native_logits'):
        return w.native_logits(x).to(torch.bfloat16)
    if (FP8Weight is not None and isinstance(w, FP8Weight)) or (FP4Weight is not None and isinstance(w, FP4Weight)):
        return dense(x, w)  # quantized-weight kernel; row-invariant by construction, not row-tiled
    B = MM_TILE
    # HC's 24x20480 FP32 projection underutilizes GB10 at M=16. M=32 selects
    # a faster full-FP32 cuBLAS algorithm, but changes summation order. Always
    # use the same padded shape for single-token and speculative decode.
    # Larger prefill calls, router precision and all other projections stay unchanged.
    if (B > 0 and x.ndim == 2 and x.size(0) <= B and w.dtype == torch.float32
            and tuple(w.shape) == (24, 20480)):
        B = HC_MM_TILE
    if B <= 0 or x.ndim != 2 or x.size(0) >= B:
        return F.linear(x, w)
    m = x.size(0)
    n = (m + B - 1) // B * B
    if n != m:
        x = torch.cat([x, x.new_zeros(n - m, x.size(1))])
    out = torch.empty(n, w.size(0), dtype=torch.promote_types(x.dtype, w.dtype), device=x.device)
    for i in range(0, n, B):
        out[i:i + B] = F.linear(x[i:i + B], w)
    return out[:m]


def tiled_rows(fn, *xs: torch.Tensor):
    """Pad decode-sized calls to MM_TILE rows before applying `fn` (see mm()).

    Torch picks the block/vector configuration of a last-dim reduction from the number of rows, so
    x.square().mean(-1) is not row-count-invariant either (it differs for M <= ~14 against a large
    M). Decode uses exactly MM_TILE rows while prefill stays a single wide operation. `fn` may
    return a tensor or a tuple of tensors; zero padding rows are dropped.
    """
    B = MM_TILE
    m = xs[0].size(0)
    if B <= 0 or m >= B:
        return fn(*xs)
    n = (m + B - 1) // B * B
    if n != m:
        xs = tuple(torch.cat([x, x.new_zeros(n - m, *x.shape[1:])]) for x in xs)
    outs = [fn(*[x[i:i + B] for x in xs]) for i in range(0, n, B)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[k] for o in outs])[:m] for k in range(len(outs[0])))
    return torch.cat(outs)[:m]


def rms_rsqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """rsqrt(mean(x^2, -1, keepdim=True) + eps), row-count-invariant."""
    return tiled_rows(lambda t: torch.rsqrt(t.square().mean(-1, keepdim=True) + eps), x)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * rms_rsqrt(xf, eps)
    return (w.float() * xf).to(dtype)


def dense(x: torch.Tensor, w) -> torch.Tensor:
    """x @ w^T where w is a bf16 tensor, an FP8Weight (stored-format fp8 + ue8m0 block scales) or an
    FP4Weight (E2M1 codes + one ue8m0 scale per 32 K weights).
    FP8Weight: the Triton kernel for decode-sized M, otherwise a transient bf16 dequant + cuBLAS.
    FP4Weight: the Triton kernel at every M (BLOCK_M 16 / 64); DSV41_FP4_DENSE_PREFILL=dequant
    restores the transient-dequant + cuBLAS shape of the fp8 path for M > 16."""
    if hasattr(w, 'native_logits'):
        return w.native_logits(x).to(torch.bfloat16)
    if FP4Weight is not None and isinstance(w, FP4Weight):
        if x.numel() // x.shape[-1] <= 16 or os.environ.get("DSV41_FP4_DENSE_PREFILL", "kernel") == "kernel":
            return fp4_linear(x, w)
        return F.linear(x.to(torch.bfloat16), w.dequant())
    if FP8Weight is not None and isinstance(w, FP8Weight):
        if x.numel() // x.shape[-1] <= decode_rows.limit():
            return fp8_linear(x, w)
        return F.linear(x.to(torch.bfloat16),
                        w.dequant_cached() if DENSE_DEQUANT_CACHE else w.dequant())
    return F.linear(x, w)


# DSV41_FP8_ACT_QDQ_FUSED=1: for decode-sized bf16 input to an fp8 weight, quantize the activation
# inside the GEMM (fp8_linear act_qdq) instead of ~13 torch kernels per call. Meant to be
# bit-identical (tools/test_fp8_act_qdq.py). Read at call time so the A/B bench can toggle it;
# the EP2 boot guard pins it.
ACT_QDQ_FUSED = os.environ.get("DSV41_FP8_ACT_QDQ_FUSED", "1") == "1"
_ACT_QDQ_REF = act_qdq_fp8   # the fused path reproduces THIS function; engine/model.py can swap it out


def _act_qdq_fusable(x: torch.Tensor, w) -> bool:
    if not (ACT_QDQ_FUSED and act_qdq_fp8 is _ACT_QDQ_REF and FP8Weight is not None):
        return False
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.numel() // x.shape[-1] > decode_rows.limit():
        return False
    if type(w) is FP8Weight:
        return x.shape[-1] == w.K
    # TP wrappers that take the unquantized activation and quantize inside their fp8 GEMM
    return type(getattr(w, "local", None)) is FP8Weight and hasattr(w, "tp_linear_act_qdq")


def qlinear(x: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    """Quantized-weight linear: fake-quantize the activation to fp8 (as the kernels do), bf16 GEMM."""
    if _act_qdq_fusable(x, w_bf16):
        if type(w_bf16) is FP8Weight:
            return fp8_linear(x, w_bf16, act_qdq=True)
        return w_bf16.tp_linear_act_qdq(x)
    return mm(act_qdq_fp8(x), w_bf16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc: int, iters: int, eps: float):
    """Port of kernel.hc_split_sinkhorn_kernel. mixes: [n, (2+hc)*hc] fp32."""
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(x: torch.Tensor, hc_fn, hc_scale, hc_base, args: Args):
    """x: [s, hc, d] -> (pre [s,hc], post [s,hc], comb [s,hc,hc]); normalized over the flattened stream."""
    xf = x.flatten(1).float()
    rsqrt = rms_rsqrt(xf, args.norm_eps)
    mixes = mm(xf, hc_fn) * rsqrt
    return tiled_rows(lambda t: hc_split_sinkhorn(t, hc_scale, hc_base, args.hc_mult,
                                                  args.hc_sinkhorn_iters, args.hc_eps), mixes)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
    x: [s, d], residual: [s, hc, d], post: [s, hc], comb: [s, hc, hc] -> [s, hc, d].

    Reference (`model.py:Block.hc_post`): `post.unsqueeze(-1) * x.unsqueeze(-2)
    + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` -- the sum runs over the FIRST
    index of comb, i.e. y[j] = post[j] * x + sum_i comb[i, j] * residual[i]. (An earlier version
    of this port summed over the second index, comb @ residual: a transposed mixing matrix that
    left the model coherent but measurably worse -- teacher-forced NLL and stuttering generation.)
    """
    mixed = torch.einsum("sij,sid->sjd", comb.float(), residual.float())
    y = post.unsqueeze(-1) * x.float().unsqueeze(1) + mixed
    return y.type_as(x)


def router(x: torch.Tensor, w: LayerWeights, args: Args):
    """Gate.forward for text tokens: sqrt(softplus(scores)); bias only steers selection."""
    scores = mm(x.float(), w.gate_w)
    scores = F.softplus(scores).sqrt()
    indices = (scores + w.gate_bias).topk(args.n_activated_experts, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * args.route_scale
    return weights, indices, scores


def expert_ffn(x: torch.Tensor, w1, w2, w3, limit: float, weights: torch.Tensor | None = None,
               w13=None) -> torch.Tensor:
    """`w13`: optional fp8_linear.concat_rows(w1, w3) merge. One activation quantization and one
    launch instead of two; column-wise the GEMM is the same, and the slices below feed the same
    elementwise ops, so the output is meant to be identical (tools/test_fp8_decode_tiles.py)."""
    dtype = x.dtype
    if w13 is not None:
        gu = qlinear(x, w13)
        gate = gu[..., :w1.N].float()
        up = gu[..., w1.N:].float()
    else:
        gate = qlinear(x, w1).float()
        up = qlinear(x, w3).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return qlinear(h.to(dtype), w2)


class ExpertLoader:
    """Loads one routed expert's three FP4 matrices from an open safetensors shard and dequantizes."""

    def __init__(self, get, layer: int, device: str):
        self.get, self.layer, self.device = get, layer, device

    def __call__(self, e: int):
        p = f"layers.{self.layer}.ffn.experts.{e}."
        d = self.device
        w1 = dequant_fp4_packed(self.get(p + "w1.weight").to(d), self.get(p + "w1.scale").to(d))
        w2 = dequant_fp4_packed(self.get(p + "w2.weight").to(d), self.get(p + "w2.scale").to(d))
        w3 = dequant_fp4_packed(self.get(p + "w3.weight").to(d), self.get(p + "w3.scale").to(d))
        return w1, w2, w3


# ----------------------------------------------------------------------------- attention (prefill, T <= 512)
class SeqState:
    """Per-sequence state carried across layers: hc residual stream, shifted pre_mix, shared compressed KV."""

    def __init__(self, h: torch.Tensor, pre_mix: torch.Tensor):
        self.h = h  # [T, hc, d] bf16
        self.pre_mix = pre_mix  # [T, hc] fp32
        self.compress_kv: torch.Tensor | None = None  # [T//ratio, head_dim] bf16, RoPE'd + FP4-qdq'd
        self.compress_ratio: int = 0


# Training only (dream/train.py): recompute the float32 [T, heads, n] score tensors in backward
# instead of keeping them. They grow with T^2 and were most of a thinking-length lesson's
# activations. Only `_attend` is recomputed: it is pure, whereas attention() also writes
# st.compress_kv, which a recompute in backward would overwrite under a later layer.
ATTN_CHECKPOINT = False


def _attend(q, kv_all, mask, sink, scale):
    scores = torch.einsum("thd,nd->thn", q.float(), kv_all.float()) * scale
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    m = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(scores - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - m)
    return torch.einsum("thn,nd->thd", p / denom, kv_all.float()).to(torch.bfloat16)


def attention(x: torch.Tensor, w: LayerWeights, st: SeqState, args: Args) -> torch.Tensor:
    """x: [T, d] (post attn_norm). Exact for T <= index_topk (see module docstring)."""
    T = x.size(0)
    dev = x.device
    rd = args.rope_head_dim
    ratio = w.ratio
    if ratio:
        freqs = precompute_freqs_cis(rd, T, args.original_seq_len, args.compress_rope_theta,
                                     args.rope_factor, args.beta_fast, args.beta_slow, str(dev))
    else:
        freqs = precompute_freqs_cis(rd, T, 0, args.rope_theta, args.rope_factor, args.beta_fast,
                                     args.beta_slow, str(dev))

    qr = rmsnorm(qlinear(x, w.wq_a), w.q_norm, args.norm_eps)
    q = qlinear(qr, w.wq_b).view(T, args.n_heads, args.head_dim)
    q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], freqs)], dim=-1)

    # sliding-window KV (fp8-quantized after RoPE, over the whole vector incl. the RoPE tail)
    kv = rmsnorm(qlinear(x, w.wkv), w.kv_norm, args.norm_eps)
    kv = torch.cat([kv[:, :-rd], apply_rotary(kv[:, -rd:], freqs)], dim=-1)
    kv = act_qdq_fp8(kv)
    pos = torch.arange(T, device=dev)
    win_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - args.window_size)  # [T, T]

    kv_all, mask = kv, win_mask
    if ratio:
        if w.is_kv_source:
            if ratio > 1:
                xf = x.float()
                ckv, score = F.linear(xf, w.comp_wkv), F.linear(xf, w.comp_wgate)
                cutoff = T - T % ratio
                ckv, score = ckv[:cutoff].unflatten(0, (-1, ratio)), score[:cutoff].unflatten(0, (-1, ratio))
                latent = (ckv * score.softmax(dim=1)).sum(dim=1)
                latent = rmsnorm(latent.to(torch.bfloat16), w.comp_norm, args.norm_eps)
            else:
                latent = rmsnorm(F.linear(x, w.comp_wkv), w.comp_norm, args.norm_eps)
            n_c = latent.size(0)
            cf = freqs[: T - T % ratio : ratio]
            latent = torch.cat([latent[:, :-rd], apply_rotary(latent[:, -rd:], cf)], dim=-1)
            latent = fp4_qdq(latent, 16, "e4m3")
            st.compress_kv, st.compress_ratio = latent, ratio
        assert st.compress_kv is not None and st.compress_ratio == ratio, (w.layer, st.compress_ratio, ratio)
        ckv = st.compress_kv
        n_c = ckv.size(0)
        compress_lens = (pos + 1) // ratio  # groups completed before/at the query
        c_mask = torch.arange(n_c, device=dev)[None, :] < compress_lens[:, None]
        kv_all = torch.cat([kv, ckv], dim=0)
        mask = torch.cat([win_mask, c_mask], dim=1)

    if ATTN_CHECKPOINT and torch.is_grad_enabled():
        o = torch.utils.checkpoint.checkpoint(_attend, q, kv_all, mask, w.attn_sink, args.head_dim ** -0.5,
                                              use_reentrant=False)
    else:
        o = _attend(q, kv_all, mask, w.attn_sink, args.head_dim ** -0.5)
    o = torch.cat([o[..., :-rd], apply_rotary(o[..., -rd:], freqs, inverse=True)], dim=-1)

    o = o.reshape(T, args.o_groups, -1)
    o = wo_a_proj(o, w.wo_a)
    return qlinear(o.flatten(1), w.wo_b)


# ----------------------------------------------------------------------------- engram
def engram_forward(h: torch.Tensor, rows: torch.Tensor, ew: EngramWeights, args: Args,
                   token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """h: [T, hc, d]; rows: [T, 24, 256] float32 dequantized table rows (already scaled).

    `token_mask`: [T] bool, False for positions that take no part in an n-gram (image spans).
    Those get their gate zeroed, so the engram contributes nothing there -- the same thing the
    reference does at inference/model.py:364. The hashes for those positions are DEAD anyway;
    zeroing the gate is what stops a dead row from being written into the stream."""
    T = h.size(0)
    kv = qlinear(rows.reshape(T, -1).to(torch.bfloat16), ew.wkv)
    return engram_combine(h, kv, ew, args, token_mask)


def engram_combine(h: torch.Tensor, kv: torch.Tensor, ew: EngramWeights, args: Args,
                   token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """engram_forward after its wkv projection `kv` [T, (hc + 1) * dim]: everything here is per
    token row, so batched decode projects every sequence's rows at once and combines per sequence."""
    T = h.size(0)
    key, value = kv.split([args.hc_mult * args.dim, args.dim], dim=-1)
    key = key.float().view(T, args.hc_mult, args.dim)
    weight = ew.q_weight * ew.k_weight
    hf, eps = h.float(), args.norm_eps

    def _gate(hh, kk):
        rstd = rms_rsqrt(hh, eps) * rms_rsqrt(kk, eps)
        dot = (hh * weight * kk).sum(-1, keepdim=True) * rstd * args.dim ** -0.5
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))

    gate = tiled_rows(_gate, hf, key)  # [T, hc, 1]
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.view(-1, 1, 1), 0)
    return (hf + gate * value.float().unsqueeze(1)).to(h.dtype)


# ----------------------------------------------------------------------------- block
def block_forward(st: SeqState, w: LayerWeights, experts: ExpertLoader, args: Args, expert_cache: dict,
                  record=None, record_norms=None):
    """One backbone block over one sequence. `record(indices, weights, scores)` receives the router output.
    `expert_cache` maps expert id -> (w1,w2,w3) for experts already dequantized in this layer."""
    x = st.h
    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, args)
    y = hc_pre(x, st.pre_mix)
    y = rmsnorm(y, w.attn_norm, args.norm_eps)
    y = attention(y, w, st, args)
    x = hc_post(y, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, args)
    y = hc_pre(x, attn_pre)
    y = rmsnorm(y, w.ffn_norm, args.norm_eps)

    weights, indices, scores = router(y, w, args)
    if record is not None:
        record(indices, weights, scores)
    out = torch.zeros_like(y, dtype=torch.float32)
    norms = torch.zeros_like(weights, dtype=torch.float32) if record_norms is not None else None
    for e in torch.unique(indices).tolist():
        if e not in expert_cache:
            expert_cache[e] = experts(e)
        w1, w2, w3 = expert_cache[e]
        idx, top = torch.where(indices == e)
        contrib = expert_ffn(y[idx], w1, w2, w3, args.swiglu_limit, weights[idx, top, None]).float()
        out[idx] += contrib
        if norms is not None:
            norms[idx, top] = contrib.norm(dim=-1) / weights[idx, top].clamp_min(1e-20)
    if record_norms is not None:
        record_norms(norms)
    out += expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, args.swiglu_limit).float()
    y = out.to(y.dtype)

    st.h = hc_post(y, residual, ffn_post, ffn_comb)
    st.pre_mix = ffn_pre
