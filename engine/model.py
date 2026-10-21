"""
model.py -- DeepSeek-V4.1-Flash text model for one-box serving: chunked prefill, multi-token
decode blocks (DSpark verification) and cache rollback, batch size 1.

Ported from the reference `inference/model.py`; the tilelang kernels are replaced by torch ops
and the routed experts by `engine.experts.ExpertStore` (+ the Triton FP4 grouped-MoE kernel in
`tools/fp4_moe.py`). Activation quantization is optional; the window KV and indexer's Q/K remain
in bf16. Compressed KV values are quantize/dequantized to FP4-E4M3 before being stored in bf16.
Position semantics are the reference's: a chunk of T tokens at absolute start position S.
Any (S, T) with T <= 512 works, which is what chunked prefill and 6-token verify blocks need.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
import os
import sys
import time

import torch
import torch.nn.functional as F
from engine.collective_rails import forward_phase, prefill_phase
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import v41_ref as R  # noqa: E402

# Window ring slots. Must exceed window_size + the longest chunk a single forward sees, because
# `attention` gathers a query's window out of the ring AFTER writing the whole chunk into it
# (128 + 2048 here). 4096 slots x 512 dims x bf16 x 40 layers = 167 MB.
RING = int(os.environ.get("DSV41_RING", 4096))
# Longest prefill chunk. Bigger chunks are strictly cheaper on this recipe: a prefill chunk streams
# nearly every expert of every layer through the transient ring whatever its length (a 512-token
# chunk already touches ~370 of 384), so the NVMe traffic of a prompt is ~chunks x layers x 384
# experts and quadrupling the chunk quarters it. The ceiling is activation memory: at T=2048 the
# gathered window+compressed KV of one layer is ~2.7 GB.
MAX_CHUNK = int(os.environ.get("DSV41_PREFILL_CHUNK", 2048))
# Record which experts the router wanted but pruning removed (Model._record_prune_miss).
# Resolved with the rest of adaptation (engine/adapt_config.py): on whenever the DSV41_ADAPT_*
# knobs are used, else the legacy DSV41_PRUNE_MISS.
from engine.adapt_config import CFG as _ADAPT  # noqa: E402
PRUNE_MISS = _ADAPT.record
# Record decode-sized blocks with one Triton launch instead of ~20 torch kernels
# (engine/prune_miss_fused.py). Database-only: routing and tokens are unchanged either way.
# Read at call time so the A/B bench can toggle it; the boot guard pins it.
PRUNE_MISS_FUSED = os.environ.get("DSV41_PRUNE_MISS_FUSED", "1") == "1"
from engine import expert_topp as _ETP_MOD  # noqa: E402
_EXPERT_TOPP = _ETP_MOD.config()  # DSV41_EXPERT_TOPP: (topp, min_k) or None
# How the demand database counts evidence. "slot" (default) adds one per (token, expert) routing
# decision, so a request's vote is its token count: a 10k-token prompt outvotes a three-sentence
# story in proportion to their lengths, and short benchmark probes are effectively invisible.
# "request" normalizes each request's per-layer routed distribution to sum 1 and ages the database
# once per request, so every request is one vote. The unit changes with the mode -- per-layer
# routing slots vs requests -- so DSV41_PRUNE_PRIOR and DSV41_PRUNE_HALFLIFE must be set to match,
# and the persisted DB is tagged with its unit so a mismatched one is ignored rather than misread.
PRUNE_UNIT_REQUEST = _ADAPT.request_unit
# Storage dtype of the prefill indexer's score buffer (see Model._indexer). DSV41_INDEX_SCORE_BF16=0
# restores fp32 for an A/B.
SCORE_DTYPE = (torch.bfloat16 if os.environ.get("DSV41_INDEX_SCORE_BF16", "1") == "1"
               else torch.float32)
# Compressed-KV QDQ reproduces the reference's FP4/E4M3 cache values in BF16 storage.
# Window KV deliberately stays BF16: the previous act_qdq_fp8 call was an identity under
# act_quant=False, so it did not establish FP8-cache parity in serving. Do not claim otherwise.
KV_CACHE_QDQ = os.environ.get("DSV41_KV_CACHE_QDQ", "1") != "0"
PACKED_KV = os.environ.get("DSV41_PACKED_KV", "0") == "1"
# Stream the prefill indexer's score buffer one query tile at a time (Model._indexer_stream)
# instead of materializing the whole [chunk, context] matrix. The score never feeds arithmetic --
# it is masked_fill -> block amax -> topk -- and the query axis is independent, so the same einsum
# tiles can be reduced per query tile. At 512k the monolithic buffer is 2.1 GB of bf16 (4.2 GB
# before DSV41_INDEX_SCORE_BF16); the tiled one is [ATTN_TILE, n_pad] = 67 MB. DSV41_INDEX_STREAM=0
# restores the monolithic path, which stays the default until test_prefix_invariance passes on the
# streaming one.
INDEX_STREAM = os.environ.get("DSV41_INDEX_STREAM", "0") == "1"
# DSV41_INDEX_FUSED=1: prefill-sized score matrices in one kernel (tools/indexer_prefill.py) instead
# of the tiled einsum/relu/weight/head-sum loop below. The heads are summed in a fixed order that
# torch's reduction does not share: a ranking-only numerics change, pinned by the boot guard.
INDEX_FUSED = os.environ.get("DSV41_INDEX_FUSED", "0") == "1"
try:
    from indexer_prefill import index_scores as _index_scores  # noqa: E402
except Exception:  # noqa: BLE001
    _index_scores = None
# Dense tensor parallelism (DSV41_TP_DENSE): world size, or 1 when off. Read once -- both ranks
# must agree, and a rank that shards while its peer does not desyncs on the first all-reduce.
TP_DENSE_WORLD = (int(os.environ.get("WORLD_SIZE", 1))
                  if os.environ.get("DSV41_TP_DENSE", "0") == "1" else 1)

# Chunk invariance requires every GEMM to give the same row whatever the batch length M. cuBLAS
# picks split-K kernels for small M and, with this flag on, reduces the K-splits in bf16, so
# F.linear(x[:6], w) != F.linear(x, w)[:6] by ~2.4e-3 for the N=512 wkv projection -- which the
# attention softmax then amplifies ~2x per layer. fp32 reduction cuts that to ~9e-5.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Sequential decode (M=1) and speculative verification (M<=16) must issue the same GEMM shape or
# cuBLAS switches tiling/split-K and changes verifier logits. Larger calls are prefill and run as
# one wide GEMM; see v41_ref.mm().
MM_TILE = 16
# The attention softmax is a *batched* GEMM (one independent problem per token), which is already
# offset-invariant, so it can use a bigger tile.
ATTN_TILE = 64
# DSV41_ATTN_TIMING=1: split a prefill layer into GPU-timeline phases using CUDA events (see
# _mark). Cheap enough to leave on, and unlike the host-timer version it replaced it does not
# reattribute the work: that one synchronised, so the first phase in the function swallowed the
# whole queued backlog and reported q_proj at 72 % of attention. Several prefill rewrites were
# proposed off that bad number and measured at 1.0x or worse before being written.
ATTN_TIMING = os.environ.get("DSV41_ATTN_TIMING", "0") == "1"
# Single source of truth for the fused prefill attention switch (see _use_fused). It was read in
# two places with DIFFERENT defaults -- "1" here and "0" in v41_engine's /health reporter -- so
# with the variable unset the engine used the fused path and /health said it did not.
PREFILL_FUSED_ATTN = os.environ.get("DSV41_PREFILL_FUSED_ATTN", "1") == "1"
# DSV41_PREFILL_ATTN_INDEXED=1: the fused prefill attention reads its keys in place -- window rows
# from the ring, compressed rows from the (packed) cache by the indexer's selection -- instead of
# from the gathered [T, 128, d] and [T, 512, d] copies (tools/prefill_attn.py). Same bits; the
# 2,048-row compressed copy alone was 1 GB written and read back per layer. Read at call time.
PREFILL_ATTN_INDEXED = os.environ.get("DSV41_PREFILL_ATTN_INDEXED", "0") == "1"
try:
    from prefill_attn import prefill_attention_indexed as _prefill_attn_indexed  # noqa: E402
except Exception:  # noqa: BLE001
    _prefill_attn_indexed = None
ATTN_PHASES: dict = {}
_PHASE_MARKS: list = []   # [(name, cuda_event)] in stream order; read once, at report time


def _use_fused(mtp_extra, T: int) -> bool:
    """Whether this attention call takes the fused kernel (tools/decode_attn.py).

    Decided in one place because the key tensors are BUILT differently for it: the fused path takes
    the window and the compressed rows as two base pointers and walks them as one key axis, so it
    never wants the concatenated [T, 640, d] tensor at all. At a 2,048-token chunk that tensor is
    ~1.5 GB written and re-read per layer.

    T > 16 keeps the decode-sized calls (verify blocks, drafts) on the torch path, where the split
    tuning and the graph capture live. mtp_extra is the DSpark draft, whose second key block is a
    stride-0 broadcast; left alone for now.

    Back on by default. It was defaulted off three minutes after the prefix cache landed, with no
    message; the evident worry is that this kernel rounds differently from the torch path (~1e-2
    relative, bf16 level) and a cache that resumes a prefill has to agree with itself. But that is
    the wrong comparison: what has to match is a RESUMED prefill against a COLD one under the same
    kernel, not this kernel against the other one. tools/test_prefix_invariance.py checks exactly
    that -- resuming at 2259 tokens, deliberately not a multiple of the 2048 chunk size, so the two
    runs tile the same tokens differently -- and it passes here, with the cache demonstrably used
    (2259 tokens reused, byte-identical output). Prefill 544 -> 1047 tok/s on a 4513-token prompt,
    and tools/generation_gate.py is 6/6.
    """
    return mtp_extra is None and T > 16 and _prefill_attn is not None and PREFILL_FUSED_ATTN


def _mark(name):
    """Record a CUDA event on the current stream. NOT a host timestamp.

    The first version of this used torch.cuda.synchronize() + perf_counter and it lied: a sync
    drains everything queued, so whichever phase is first in the function absorbs the whole
    backlog. That is how q_proj came out at 72 % of attention, and why merely enabling the flag
    moved attn_s from 8.93 s to 25.91 s -- the same work measured three times larger. Host
    timestamps cannot attribute asynchronous work: without a sync they time the launch, with a
    blocking op they time the wait for unrelated kernels queued earlier.

    A CUDA event is a timestamp ON the stream, between kernels. The gap between consecutive events
    is the GPU time of the work queued between them. A few microseconds to record, so the ~700 of
    them in a prefill are free.
    """
    if ATTN_TIMING:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _PHASE_MARKS.append((name, e))


def phase_report():
    """{phase: total GPU ms}. One synchronise, here, after everything has already run."""
    if not ATTN_TIMING or len(_PHASE_MARKS) < 2:
        return None
    torch.cuda.synchronize()
    out: dict = {}
    for (n0, e0), (n1, e1) in zip(_PHASE_MARKS, _PHASE_MARKS[1:]):
        if n1.endswith("_begin"):
            # The gap from the last attention mark to the next layer's region is the rest of the
            # block: output projection, the HC residual mixes, and anything not marked below.
            out["layer_rest"] = out.get("layer_rest", 0.0) + e0.elapsed_time(e1)
            continue
        out[n1] = out.get(n1, 0.0) + e0.elapsed_time(e1)
    _PHASE_MARKS.clear()
    return {k: round(v, 1) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def _aph(name, t0):
    """Call-site compatible with the old host-timer helper; records an event instead."""
    _mark(name)
    return t0
KEY_BLOCK = 512  # score tile along the compressed-key axis; independent of index_topk
R.MM_TILE = MM_TILE

# Fused sinked-softmax attention for prefill-sized calls (T > 16). Same kernel the decode path
# uses (tools/decode_attn.py). It once measured attn_s 3.16 -> 13.3 s on the live EP2 prefill and
# was defaulted off for it; both causes were in how it was CALLED, not in the kernel -- a constant
# SPLIT tuned for a 6-token verify block, and a caller that built the [T, 640, d] concatenation the
# kernel exists to avoid. Fixed, and default ON: prefill 544 -> 1047 tok/s on a 4,513-token prompt.
# Set DSV41_PREFILL_FUSED_ATTN=0 to fall back to the torch tile path.
# Independent of DSV41_FUSED_ATTN (decode graphs).
try:
    from decode_attn import decode_attention as _prefill_attn  # noqa: E402
except Exception:  # noqa: BLE001
    _prefill_attn = None

# Fused Hyper-Connection coefficients. engine/hc_sinkhorn.py replaces ~160 tiny torch ops per call
# with one launch, and the decode path has used it all along -- prefill did not, so it ran the
# reference torch version twice per layer (attention and FFN mixes), i.e. ~160 launches x 2 x 40
# layers x every chunk. That is what `layer_rest` was.
#
# It is also the better shape for chunk invariance: one program per token row, so a row's result
# cannot depend on how many rows are in the call, which is exactly what v41_ref.tiled_rows exists
# to fake for the torch version. DSV41_HC_FUSED=0 restores the torch path.
try:
    from engine.hc_sinkhorn import hc_split_sinkhorn as _hc_sinkhorn_fused  # noqa: E402
except Exception:  # noqa: BLE001
    _hc_sinkhorn_fused = None
HC_FUSED = os.environ.get("DSV41_HC_FUSED", "1") == "1" and _hc_sinkhorn_fused is not None

# Fused hyper-connection stream ops (engine/hc_ops.py). The torch versions materialise large fp32
# temporaries over the hc-wide residual -- [T, 4, 5120] is 84 MB at a 2,048-token chunk -- and the
# four HC sites of a layer measured 33 % of prefill on the GPU timeline, more than the MoE kernel,
# while doing almost no arithmetic. Fused: hc_post 9.6x, hc_pre+rmsnorm 6.6x.
try:
    from engine.hc_ops import hc_post as _hc_post_fused, hc_pre_rmsnorm as _hc_pre_rn_fused  # noqa: E402
except Exception:  # noqa: BLE001
    _hc_post_fused = _hc_pre_rn_fused = None
HC_OPS = os.environ.get("DSV41_HC_OPS", "1") == "1" and _hc_post_fused is not None

# DSV41_HC_PREFILL_FUSED=1: prefill-sized HC mix fronts (projection + RMS scale) in one pass over the
# BF16 residual (tools/hc_prefill.py) instead of x.float() + cuBLAS FP32 SIMT + square + mean.
# ~1e-7 relative from the summation order: a numerics switch, pinned by the boot guard.
HC_PREFILL_FUSED = os.environ.get("DSV41_HC_PREFILL_FUSED", "0") == "1"
try:
    from hc_prefill import hc_front as _hc_front_prefill  # noqa: E402
except Exception:  # noqa: BLE001
    _hc_front_prefill = None


# ----------------------------------------------------------------------------- weights
class IndexerWeights:
    def __init__(self, get, p: str, owns_k: bool, device: str):
        w, sc = get(p + "indexer.wq_b.weight").to(device), get(p + "indexer.wq_b.scale").to(device)
        self.wq_b = R.FP8Weight(w, sc) if (R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1") else R.dequant_fp8_block(w, sc)
        self.weights_proj = get(p + "indexer.weights_proj.weight").to(device).to(torch.bfloat16)
        self.owns_k = owns_k
        if owns_k:
            self.wk = get(p + "indexer.wk.weight").to(device).to(torch.bfloat16)
            self.k_norm = get(p + "indexer.k_norm.weight").to(device).to(torch.bfloat16)


class Weights:
    """All non-routed-expert weights on the GPU, bf16 (fp32 where the reference uses fp32)."""

    def __init__(self, model_dir: str, index: dict, args: R.Args, device: str, log=print, act_quant: bool = False,
                 n_layers: int | None = None, load_mtp: bool = True, engram_dir: str | None = None):
        self.args, self.device = args, device
        n_load = args.n_layers if n_layers is None else n_layers
        wm = index["weight_map"]
        handles = {}

        def get(name):
            f = wm[name]
            if f not in handles:
                path = os.path.join(model_dir, f)
                if not os.path.exists(path) and ".engram." in name:
                    # engram shard not (yet) on disk: the small non-table tensors fetched by tools/engram_rows.py
                    L = name.split(".")[1]
                    path = os.path.join(engram_dir or "engram_rows", f"layer{L}_weights.safetensors")
                handles[f] = safe_open(path, "pt", device="cpu")
            return handles[f].get_tensor(name)

        # DSV41_ABLIT_WOB: an abliteration overlay for layers 10-35 attn.wo_b (engine/ablit.py),
        # substituted here so every consumer -- and the DENSE_FP4 re-quantization -- sees it.
        from engine import ablit
        get = ablit.loader(get)

        t0 = time.time()
        if os.environ.get('DSV41_TP_EMBED', '0') == '1':
            from engine.tensor_parallel import FeatureParallelEmbedding, shard
            rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
            if world != 2 or not 0 <= rank < world:
                raise ValueError('embedding TP requires two ranks')
            self.embed = FeatureParallelEmbedding(
                shard(get('embed.weight'), 1, rank, world).to(device).to(torch.bfloat16), world)
        else:
            self.embed = get("embed.weight").to(device).to(torch.bfloat16)
        # bf16 (the stored dtype) unless DSV41_HEAD_FP32=1. The reference keeps the LM head in fp32
        # ("so the logits come out in fp32 directly"); the fast decode path already ran a bf16 copy
        # (fp32 accumulate, logits rounded to bf16), so with a bf16 head here the two paths use the
        # same weights and the 2.65 GB fp32 copy disappears (= ~140 more expert slots).
        # ... and, with DSV41_HEAD_FMT, in fp8 or fp4 instead: the head is read in full on every
        # decode step, so its stored format is worth as much as a dense projection group's.
        head = get('head.weight')
        from engine.tensor_parallel import draft_head_bytes, make_tp_draft_head
        from engine import draft_vocab
        self.draft_vocab_digest = None
        if os.environ.get('DSV41_TP_HEAD', '0') == '1':
            from engine.tensor_parallel import VocabParallelHead
            rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
            if world != 2 or head.shape[0] % world:
                raise ValueError('vocabulary TP requires two equal shards')
            shard = head.chunk(world, dim=0)[rank].to(device)
            self.head = VocabParallelHead(R.make_head(shard), world)
            # DSV41_DRAFT_VOCAB (engine/draft_vocab.py): the drafter's head over listed ids only,
            # built from the full head.weight because a rank's run of listed ids is not its
            # verifier half. DSV41_BENCH_DRAFT_VOCAB_AB=1 also keeps the full-vocabulary draft head
            # (`draft_head_full`) so a driver can A/B the two in one process; never set in serving.
            listed = draft_vocab.load(head.shape[0], world) if load_mtp else None
            if listed is not None:
                from engine.tensor_parallel import make_listed_draft_head, tp_draft_head_enabled
                if not tp_draft_head_enabled():
                    raise ValueError('DSV41_DRAFT_VOCAB needs DSV41_TP_DRAFT_HEAD=1')
                self.draft_vocab_digest = draft_vocab.digest(listed)
                self.draft_head = make_listed_draft_head(head, listed, rank, world, device)
                if os.environ.get('DSV41_BENCH_DRAFT_VOCAB_AB', '0') == '1':
                    self.draft_head_full = make_tp_draft_head(self.head, load_mtp=load_mtp, source=shard)
            else:
                self.draft_head = make_tp_draft_head(self.head, load_mtp=load_mtp, source=shard)
            # Benchmark-only (DSV41_BENCH_DRAFT_HEAD_SRC): keep this rank's bf16 vocabulary shard so
            # a driver can rebuild the draft head in another format and A/B the two inside one
            # process. The head format is a load-time choice otherwise, and comparing two of them
            # across processes costs a run-to-run drift larger than the effect. 662 MB a rank;
            # never set in serving.
            if os.environ.get('DSV41_BENCH_DRAFT_HEAD_SRC', '0') == '1':
                self.draft_head_src = shard.clone()
            del shard
        else:
            if draft_vocab.enabled():
                raise ValueError('DSV41_DRAFT_VOCAB needs the TP vocabulary head (DSV41_TP_HEAD=1)')
            self.head = R.make_head(head.to(device))
            # DSV41_DRAFT_HEAD_FMT: the verifier keeps `head`; only the drafter reads the cheaper
            # copy. None means "reuse the verifier's" (off, already quantized, or FP32 reference).
            # Gated on load_mtp: diagnostics that never draft should not pay for a second head.
            self.draft_head = R.make_draft_head(self.head) if load_mtp else None
        # This is extra resident storage, allocated before the engine prices its expert arena.
        # It is reported separately so draft speed is not bought with hidden model memory.
        self.draft_head_bytes = draft_head_bytes(self.draft_head)
        del head
        self.norm = get("norm.weight").to(device).to(torch.bfloat16)
        self.layers = []
        self.indexers = {}
        self.engram = {}
        for L in range(n_load):
            self.layers.append(R.LayerWeights(get, L, args, device))
            if os.environ.get('DSV41_TP_ATTN', '0') == '1':
                from engine.tensor_parallel import shard_attention
                shard_attention(self.layers[-1], args, int(os.environ.get('RANK', '0')),
                                int(os.environ.get('WORLD_SIZE', '1')))
            if L in args.index_source_layers:
                self.indexers[L] = IndexerWeights(get, f"layers.{L}.attn.", L in args.kv_source_layers, device)
            if L in args.engram_layer_ids:
                self.engram[L] = R.EngramWeights(get, L, device)
            if L % 10 == 9:
                log(f"weights: layer {L} loaded ({time.time() - t0:.0f}s)")
            for f in list(handles):
                if f.endswith(f"{L + 3:05d}-of-00048.safetensors"):
                    del handles[f]
        # DSpark blocks
        self.mtp = []
        for k in range(3 if load_mtp else 0):
            self.mtp.append(MTPWeights(get, k, args, device))
            if os.environ.get('DSV41_TP_DRAFT_ATTN', '0') == '1':
                from engine.tensor_parallel import shard_attention
                if os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output') != 'output':
                    raise ValueError('draft attention TP requires disjoint output layout')
                shard_attention(self.mtp[-1], args, int(os.environ.get('RANK', '0')),
                                int(os.environ.get('WORLD_SIZE', '1')))
        self.dspark_experts = None  # filled by the engine (arena of 3 x 128 experts)
        log(f"weights: all non-expert weights on GPU in {time.time() - t0:.0f}s")


class MTPWeights:
    """One DSpark block (`mtp.k.*`): a Block with a 128-expert MoE, plus stage-specific heads."""

    def __init__(self, get, k: int, args: R.Args, device: str):
        p = f"mtp.{k}."
        self.k = k

        class _A(R.Args):
            pass

        a = R.Args(**{f: getattr(args, f) for f in R.Args.__dataclass_fields__})
        a.n_routed_experts = 128
        a.n_activated_experts = 3
        self.args = a
        # LayerWeights expects "layers.{L}." prefix; build the same fields by hand
        dev = device

        def bf(name):
            return get(p + name).to(dev).to(torch.bfloat16)

        def f32(name):
            return get(p + name).to(dev).to(torch.float32)

        fp4_groups = R.dense_fp4_groups()

        def fp8lin(name):
            w, sc = get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev)
            if R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1":
                return R.maybe_fp4(R.FP8Weight(w, sc), name, fp4_groups)
            return R.dequant_fp8_block(w, sc)

        self.attn_norm = bf("attn_norm.weight"); self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink"); self.q_norm = bf("attn.q_norm.weight"); self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a"); self.wq_b = fp8lin("attn.wq_b"); self.wkv = fp8lin("attn.wkv")
        self.wo_a = R.make_wo_a(get(p + "attn.wo_a.weight").to(dev), get(p + "attn.wo_a.scale").to(dev), args, fp4_groups); self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn"); self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base"); self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale"); self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight"); self.gate_bias = f32("ffn.gate.bias")
        # Image tokens route through a SEPARATE router bias (inference/model.py:819). Absent on a
        # text-only checkpoint, so this stays None and the vision branch never runs.
        try:
            self.gate_bias_vl = f32("ffn.gate.bias_vl")
        except Exception:  # noqa: BLE001
            self.gate_bias_vl = None
        self.sh_w1 = fp8lin("ffn.shared_experts.w1"); self.sh_w2 = fp8lin("ffn.shared_experts.w2"); self.sh_w3 = fp8lin("ffn.shared_experts.w3")
        self.ratio = 0
        self.is_kv_source = False
        self.layer = args.n_layers + k
        if k == 0:
            self.main_proj = fp8lin("main_proj")
            self.main_norm = bf("main_norm.weight")
        if k == 2:
            self.norm = bf("norm.weight")
            self.markov_embed = bf("markov_head.embed.weight")
            # fp32 once, for the same reason as the LM head: this one is applied once per drafted
            # token, i.e. five times per DSpark step.
            self.markov_head = get(p + "markov_head.head.weight").to(dev).to(torch.bfloat16)
            self.conf_proj = get(p + "confidence_head.proj.weight").to(dev).float()


# ----------------------------------------------------------------------------- caches
class Caches:
    def __init__(self, args: R.Args, max_seq: int, device: str):
        self.args, self.max_seq, self.device = args, max_seq, device
        d = args.head_dim
        self.win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(args.n_layers)]
        self.mtp_win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(3)]
        self.ckv = {}
        self.ik = {}
        self.pending = {}  # ratio-2 sources: (kv fp32 [512], score fp32 [512]) of an unpaired position, or None
        if PACKED_KV and (not KV_CACHE_QDQ or d % 128):
            raise ValueError('packed KV requires FP4 QDQ and head_dim divisible by 128')
        for L in args.kv_source_layers:
            r = args.compress_ratios[L]
            n = max_seq // r + 1
            self.ckv[L] = torch.zeros(n, d * 9 // 128 if PACKED_KV else d,
                                     dtype=torch.int64 if PACKED_KV else torch.bfloat16,
                                     device=device)
            self.ik[L] = torch.zeros(n, args.index_head_dim, dtype=torch.bfloat16, device=device)
            self.pending[L] = None
        self.len = 0  # number of valid positions
        # per-chunk memory for rollback of the compressor state
        self._chunk_inputs = {}  # L -> (S, kv [T,512] fp32, score [T,512] fp32, pending_before)

    def rollback(self, n: int):
        """Discard everything at positions >= n.

        Only the compressor carries state across positions, so only `pending` has to be restored:
        the window ring and the compressed/index caches are append-only and every slot at or after
        n is rewritten by the next forward before anything can read it.
        """
        assert n <= self.len
        for L, (S, kv, sc, before) in self._chunk_inputs.items():
            r = self.args.compress_ratios[L]
            if r == 1:
                continue
            if n % r == 0:
                self.pending[L] = None  # n positions = n/r whole groups, nothing left over
                continue
            p = n - 1  # the position left unpaired at n
            if p >= S:
                self.pending[L] = (kv[p - S], sc[p - S])
            elif p == S - 1:
                self.pending[L] = before  # exactly back to the start of the last chunk
            else:
                raise ValueError(f"rollback({n}) reaches before the last chunk (start {S}); the "
                                 f"compressor input for position {p} is no longer kept")
        self.len = n


class Shared:
    def __init__(self):
        self.ckv = None
        self.ik = None
        self.ratio = 0
        self.topk = None  # [T, k] absolute compressed positions or -1
        self.candidates = None  # [T, n_c] bool


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, W: Weights, store, caches: Caches, moe_fn, act_quant: bool = False):
        self.W, self.store, self.c, self.moe_fn = W, store, caches, moe_fn
        self.args = W.args
        self.dev = W.device
        a = self.args
        self.freqs_c = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, a.original_seq_len, a.compress_rope_theta,
                                              a.rope_factor, a.beta_fast, a.beta_slow, self.dev)
        self.freqs_w = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, 0, a.rope_theta, a.rope_factor,
                                              a.beta_fast, a.beta_slow, self.dev)
        # Positions and fixed sliding-window offsets are used in every layer.  Keeping one device
        # copy removes dozens of small allocations/launches per prefill chunk without changing a
        # single value handed to the model.
        self._positions = torch.arange(caches.max_seq + 8, device=self.dev)
        self._window_offsets = torch.arange(a.window_size - 1, -1, -1, device=self.dev)
        self.tap = None  # optional diagnostic hook: callable(name, L, tensor)
        # Persistent because fastdecode captures compressed-KV QDQ in a CUDA graph.  Copying the
        # tiny lookup table from CPU inside capture is illegal even though the arithmetic itself
        # is graph-safe.
        self.fp4_grid = R.FP4_GRID.to(self.dev)
        self.engram_rows = None  # callable (layer, hashes [T,24]) -> [T,24,256] float32
        self.vision = None       # engine.vision.VisionTower, set by the engine when enabled
        self.image_mask = self.engram_mask = None
        self._want_counts = self._want_mass = self._miss_tot = None  # DSV41_PRUNE_MISS accumulators
        self._req_counts = self._req_mass = None  # also accessed when demand logging is disabled
        self._want_phase = self._miss_phase = None   # same, split prefill vs decode
        self.hash_state = None  # reference NgramHashState
        if not act_quant:
            R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)
        self.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "ep_s": 0.0, "ep_calls": 0,
                      "tokens": 0}
        self.begin_prompt()

    def _tap(self, name, L, t):
        if self.tap is not None:
            self.tap(name, L, t)

    # ------------------------------------------------------------------ attention
    def _hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """(pre, post, comb) for one hyper-connection site. Same arithmetic as v41_ref.hc_mixes;
        only the Sinkhorn balancing runs fused. R.mm stays for the projection because chunk
        invariance depends on its fixed-row tiling."""
        a = self.args
        if (HC_PREFILL_FUSED and _hc_front_prefill is not None and x.size(0) > 16
                and x.dtype == torch.bfloat16 and torch.is_tensor(hc_fn) and hc_fn.dtype == torch.float32):
            mixes = _hc_front_prefill(x.flatten(1), hc_fn, a.norm_eps)
        else:
            xf = x.flatten(1).float()
            mixes = R.hc_linear(xf, hc_fn) * R.rms_rsqrt(xf, a.norm_eps)
        if HC_FUSED:
            return _hc_sinkhorn_fused(mixes, hc_scale, hc_base, a.hc_mult,
                                      a.hc_sinkhorn_iters, a.hc_eps)
        return R.tiled_rows(lambda t: R.hc_split_sinkhorn(t, hc_scale, hc_base, a.hc_mult,
                                                          a.hc_sinkhorn_iters, a.hc_eps), mixes)

    def _window_positions(self, pos: torch.Tensor, width: int | None = None):
        """Absolute positions each query may see, oldest first; -1 if none.

        Reference prefill uses ``min(sequence_length, window_size)`` columns.  Padding a short
        prompt to all 128 columns changes the FP32 attention GEMM's reduction shape enough to
        perturb its output and downstream router ties, even though the extra columns are masked.
        Decode deliberately keeps the full ring width, matching the reference decode path.
        """
        w = self.args.window_size if width is None else int(width)
        p = pos[:, None] - self._window_offsets[-w:][None, :]
        return torch.where(p >= 0, p, torch.full_like(p, -1))

    def attention(self, x: torch.Tensor, w, L: int, S: int, sh: Shared, ring: torch.Tensor,
                  freqs: torch.Tensor, mtp_extra=None, win_lo: int = 0, prefill: bool = False):
        """x: [T, d] normed input. Returns [T, d]. `ring` is this layer's window KV ring.

        `win_lo` is the first position whose window KV this ring actually holds. It is 0 everywhere
        except in the decoder replay (SWA Bounded Replay, tech report 3.2.2), where the decoder
        layers have only seen the last 128 prompt tokens and a query near the start of the replay
        would otherwise gather whatever the ring happens to hold below it.
        """
        a = self.args
        T = x.size(0)
        pos = self._positions[S:S + T]
        rd = a.rope_head_dim
        fq = freqs[S:S + T]

        self._tap("attn_x", L, x)
        _mark("attn_begin")
        _t = 0.0
        qr = R.rmsnorm(R.qlinear(x, w.wq_a), w.q_norm, a.norm_eps)
        q = R.qlinear(qr, w.wq_b).view(T, getattr(w, 'tp_heads', a.n_heads), a.head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], fq)], dim=-1)
        _t = _aph("q_proj", _t)
        self._tap("q", L, q)

        kv = R.rmsnorm(R.qlinear(x, w.wkv), w.kv_norm, a.norm_eps)
        kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], fq)], dim=-1)
        _t = _aph("kv_proj", _t)
        self._tap("kv_new", L, kv)
        if mtp_extra is None:
            # gather the window BEFORE writing (a chunk may overwrite slots older queries still need)
            # Only prefill presents the short reference shape (min(seq, window) columns).  Decode
            # presents the full window ring, as the reference decode and FastDecoder both do;
            # narrowing it here is what silently moved the eager path off the graph path.
            win_width = min(a.window_size, S + T) if prefill else None
            wpos = self._window_positions(pos, win_width)
            ring[pos % RING] = kv
            wmask = wpos >= win_lo if win_lo else wpos >= 0
            # The indexed kernel reads window and compressed rows where they live (see
            # PREFILL_ATTN_INDEXED); taps and the torch/compact paths still want the copies.
            indexed = (prefill and PREFILL_ATTN_INDEXED and _prefill_attn_indexed is not None
                       and self.tap is None and _use_fused(mtp_extra, T)
                       and S + T > a.window_size
                       and wpos.size(1) % int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_N", "32")) == 0)
            wkv = None if indexed else ring[wpos.clamp_min(0) % RING]  # [T, 128, d]
            self._tap("win_kv", L, wkv); self._tap("win_mask", L, wmask)
            kv_all, mask, ckv_rows, cidx = wkv, wmask, None, None
            _t = _aph("window_gather", _t)
            if w.ratio:
                ckv_rows, cmask = self._compressed(x, qr, w, L, S, T, pos, sh, rows=not indexed)
                if indexed:
                    cidx, ckv_rows = ckv_rows, None
                _t = _aph("compressed", _t)
                self._tap("ckv_rows", L, ckv_rows); self._tap("c_mask", L, cmask)
                mask = torch.cat([wmask, cmask], dim=1)   # [T, 640] of bool: 3.8 kB, always built
                if not _use_fused(mtp_extra, T):
                    kv_all = torch.cat([wkv, ckv_rows], dim=1)   # only the torch path needs it
        else:
            # DSpark draft attention: window from the main stream's ring (positions <= S-1) + all draft kvs
            main_last = mtp_extra  # position of the last main token in the ring
            wpos = main_last - self._window_offsets
            wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))  # [128]
            wkv = ring[wpos.clamp_min(0) % RING][None].expand(T, -1, -1)
            kv_all = torch.cat([wkv, kv[None].expand(T, -1, -1)], dim=1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1), torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)

        # The softmax runs on fixed-size token tiles for the same reason the GEMMs do: the batched
        # score/PV products are not invariant to the number of query rows in the call.
        # fp32 PV product, like tools/v41_ref: rounding the probabilities to bf16 first is a
        # cliff that turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip
        # a borderline router top-k and make the MoE output depend on the chunk length.
        _t = time.perf_counter() if ATTN_TIMING else _t
        compact_window = prefill and mtp_extra is None and S + T <= a.window_size
        if compact_window:
            # The reference prefill kernel sees one compact [sequence, d] KV matrix while the
            # sequence still fits in the window.  Expressing the same values as a padded
            # [query, 128, d] gather is mathematically equivalent, but selects a different GEMM
            # reduction shape; the tiny FP32 difference is amplified by wo_a/wo_b and flips MoE
            # router ties in the first layer.  Keep this exact path bounded to <=128 tokens so
            # long-prefill performance continues to use the fused/gathered implementation.
            compact_kv = ring[:S + T]
            compact_mask = torch.arange(S + T, device=self.dev)[None, :] <= pos[:, None]
            if w.ratio:
                n_c = (S + T) // w.ratio
                from engine.packed_kv import gather
                compact_kv = torch.cat([compact_kv, gather(
                    sh.ckv, torch.arange(n_c, device=self.dev))], dim=0)
                compact_mask = torch.cat([
                    compact_mask,
                    torch.arange(n_c, device=self.dev)[None, :] <
                    ((pos + 1) // w.ratio)[:, None],
                ], dim=1)
            scores = torch.einsum("thd,nd->thn", q.float(), compact_kv.float()) * a.head_dim ** -0.5
            scores.masked_fill_(~compact_mask[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - mx)
            o = torch.einsum("thn,nd->thd", p / denom, compact_kv.float()).to(torch.bfloat16)
        elif mtp_extra is None and indexed:
            # 32 heads a program: every local head of a token in one program, so its keys are
            # loaded and dequantised once. 7.35 vs 9.86 ms at 16 (2,048 rows); every tile shape
            # swept gave the same bits (results/prefill-trace-20261008/tune_pattn.py).
            bh = min(int(os.environ.get("DSV41_PREFILL_ATTN_INDEXED_BH", "32")), q.size(1))
            o = _prefill_attn_indexed(q, ring, wpos, sh.ckv if w.ratio else None, cidx, mask,
                                      w.attn_sink, a.head_dim ** -0.5, block_h=bh,
                                      block_n=int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_N", "32")),
                                      num_warps=4, num_stages=1)
        elif _use_fused(mtp_extra, T):
            # Same math as _softmax_attn (sinked softmax, fp32 scores, split-precision PV), fused:
            # no [T, H, N] score tensor, no concatenated key tensor. Measured 3.6x faster than the
            # torch path at T=512 once decode_attn picks SPLIT from the available parallelism --
            # the old constant SPLIT=2 is tuned for a 6-token verify block and is a slowdown here.
            o = _prefill_attn(q, wkv, ckv_rows, mask, w.attn_sink, a.head_dim ** -0.5,
                              block_h=int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_H", "16")),
                              block_n=int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_N", "32")),
                              split=1)
        else:
            o = self._softmax_attn(q, kv_all, mask, w.attn_sink)
        self._tap("attn_core", L, o)
        _t = _aph("softmax_attn", _t)
        o = torch.cat([o[..., :-rd], R.apply_rotary(o[..., -rd:], fq, inverse=True)], dim=-1)
        o = o.reshape(T, getattr(w, 'tp_groups', a.o_groups), -1)
        # grouped output projection: "sgd,grd->sgr" is a GEMM with M = number of tokens, so it too
        # has to run on fixed-size token tiles (it differs most visibly at a 1-token chunk).
        o = R.wo_a_proj(o, w.wo_a, tiled=True)
        self._tap("attn_wo_a", L, o)
        out = R.qlinear(o.flatten(1), w.wo_b)
        _mark("o_proj")      # wo_a (grouped) + wo_b; was being charged to the moe gap
        self._tap("attn_out", L, out)
        return out

    def _softmax_attn(self, q, kv_all, mask, sink):
        """Sinked softmax attention over [T, n, d] KV, in fixed-size query tiles.

        Padding rows are all-masked: their scores are -inf, so m clamps to -1e30, p is 0 and the
        sink term makes the denominator +inf -- 0/inf = 0, no NaN.
        """
        T = q.size(0)
        scale = self.args.head_dim ** -0.5
        B = ATTN_TILE if ATTN_TILE > 0 else T

        def tile(qt, kvt, mt):
            scores = torch.einsum("thd,tnd->thn", qt, kvt) * scale
            scores = scores.masked_fill(~mt[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
            return torch.einsum("thn,tnd->thd", p / denom, kvt)

        outs = []
        for i in range(0, T, B):
            j = min(i + B, T)
            qt, kvt, mt = q[i:j].float(), kv_all[i:j].float(), mask[i:j]
            n = j - i
            if n < B:  # pad the last tile so every call sees exactly B query rows
                qt = torch.cat([qt, qt.new_zeros(B - n, *qt.shape[1:])])
                kvt = torch.cat([kvt, kvt.new_zeros(B - n, *kvt.shape[1:])])
                mt = torch.cat([mt, mt.new_zeros(B - n, mt.size(1))])
            outs.append(tile(qt, kvt, mt)[:n])
        return torch.cat(outs).to(torch.bfloat16)

    def _compressed(self, x, qr, w, L, S, T, pos, sh: Shared, rows: bool = True):
        """Produce/read the shared compressed KV for this chunk; run/reuse the indexer; return the
        gathered rows [T, k, d] (or with rows=False the selected positions [T, k], -1 = none) and
        their mask [T, k]."""
        a = self.args
        r = w.ratio
        c = self.c
        rd = a.rope_head_dim
        if w.is_kv_source:
            # latent for every position of the chunk (plus the pending unpaired one)
            if r > 1:
                xf = x.float()
                kvl, sc = R.mm(xf, w.comp_wkv), R.mm(xf, w.comp_wgate)
                before = c.pending[L]
                c._chunk_inputs[L] = (S, kvl, sc, before)
                if before is not None:
                    kvl = torch.cat([before[0][None], kvl]); sc = torch.cat([before[1][None], sc])
                    first = S - 1
                else:
                    first = S
                n_tok = kvl.size(0)
                cut = n_tok - n_tok % r
                if n_tok % r:
                    c.pending[L] = (kvl[-1], sc[-1])
                else:
                    c.pending[L] = None
                if cut > 0:
                    g_kv = kvl[:cut].unflatten(0, (-1, r)); g_sc = sc[:cut].unflatten(0, (-1, r))
                    latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                    latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                    j0 = first // r
                else:
                    latent, j0 = None, first // r
            else:
                latent = R.rmsnorm(R.mm(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = S
            if latent is not None:
                nj = latent.size(0)
                jpos = self._positions[j0:j0 + nj] * r
                fj = self.freqs_c[jpos]
                if L in self.W.indexers:  # index key from the pre-RoPE latent
                    iw = self.W.indexers[L]
                    k = R.rmsnorm(R.mm(latent, iw.wk), iw.k_norm, a.norm_eps)
                    k = torch.cat([k[:, :-rd], R.apply_rotary(k[:, -rd:], fj)], dim=-1)
                    c.ik[L][j0:j0 + nj] = k
                lat = torch.cat([latent[:, :-rd], R.apply_rotary(latent[:, -rd:], fj)], dim=-1)
                if PACKED_KV:
                    from engine.packed_kv import write
                    write(c.ckv[L], lat.contiguous(), j0)
                else:
                    if KV_CACHE_QDQ:
                        lat = R.fp4_qdq(lat, 16, "e4m3", self.fp4_grid)
                    c.ckv[L][j0:j0 + nj] = lat
                if self.tap is not None:
                    if PACKED_KV:
                        from engine.packed_kv import gather
                        lat = gather(c.ckv[L], torch.arange(j0, j0 + nj, device=self.dev))
                    self._tap("latent", L, (j0, lat))
            sh.ckv, sh.ik, sh.ratio = c.ckv[L], c.ik[L], r
        assert sh.ratio == r, (L, sh.ratio, r)
        compress_lens = (pos + 1) // r  # visible compressed positions per query
        n_c = int((S + T) // r)
        if L in self.W.indexers:
            sh.topk = self._indexer(x, qr, L, pos, compress_lens, n_c, sh)
        idx = sh.topk  # [T, k] absolute compressed positions, -1 = none
        self._tap("topk", L, idx); self._tap("n_c", L, n_c)
        if not rows:
            return idx, idx >= 0
        from engine.packed_kv import gather
        return gather(sh.ckv, idx.clamp_min(0)), idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, n_c, sh: Shared):
        a = self.args
        iw = self.W.indexers[L]
        T = x.size(0)
        rd = a.rope_head_dim
        if n_c == 0:
            return self._pad_topk(torch.full((T, 0), -1, dtype=torch.int64, device=self.dev))
        q = R.qlinear(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], self.freqs_c[pos])], dim=-1)
        wts = (R.mm(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5))  # [T, H]
        # fixed [MM_TILE queries x KEY_BLOCK keys] score tiles: n_c grows with the chunk, so a
        # single GEMM over sh.ik[:n_c] would be a different shape in every chunk. Columns past
        # n_c are masked out below, so the padding cannot be selected.
        NB = KEY_BLOCK
        n_pad = max(NB, (n_c + NB - 1) // NB * NB)
        k = sh.ik[:n_pad]
        if k.size(0) < n_pad:
            k = torch.cat([k, k.new_zeros(n_pad - k.size(0), k.size(1))])
        B = ATTN_TILE if ATTN_TILE > 0 else T
        if INDEX_STREAM:
            # Same key/query tiles, reduced one query tile at a time; see _indexer_stream.
            # DSV41_INDEX_STREAM=0 restores the monolithic [T, n_pad] buffer below.
            return self._indexer_stream(q, wts, k, n_pad, min(a.index_topk, n_c), L,
                                        compress_lens, sh, B, NB, T)
        # The score buffer is a RANKING quantity -- masked_fill, amax over blocks, topk -- and
        # never feeds arithmetic, so it only has to preserve ORDER, not value. The per-head sum
        # below is still accumulated in fp32; only the store is narrowed. That halves the largest
        # transient in prefill: [chunk, context] fp32 is 4 GB at 256k and 8 GB at 512k, which is
        # what caps the context, not the KV cache (972 MB at 256k).
        # Measured on simulated scores: bf16 keeps 98.3% of the top-512 selection and 99.995% of
        # its score mass. fp8 was tried and fails twice over -- 75% overlap, and topk/amax/
        # masked_fill are all unimplemented for float8_e4m3fn, which are the only three operations
        # this buffer exists for.
        if INDEX_FUSED and _index_scores is not None and T > 16:
            score = _index_scores(q.contiguous(), k.contiguous(), wts.contiguous(), SCORE_DTYPE)
        else:
            score = torch.empty(T, n_pad, dtype=SCORE_DTYPE, device=self.dev)
            for i in range(0, T, B):
                j = min(i + B, T)
                qt, wt = q[i:j], wts[i:j]
                if j - i < B:
                    qt = torch.cat([qt, qt.new_zeros(B - (j - i), *qt.shape[1:])])
                    wt = torch.cat([wt, wt.new_zeros(B - (j - i), wt.size(1))])
                for jb in range(0, n_pad, NB):
                    sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])  # bf16
                    sc = sc.float().relu_() * wt[:, :, None]
                    score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i].to(score.dtype)
        cpos = self._positions[:n_pad]
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        is_cand_src = L == a.candidate_source_layer
        if is_cand_src:
            sh.candidates = self._select_candidates(score, compress_lens, a.candidate_topk_blocks, a.candidate_block_size)
        elif 0 <= a.candidate_source_layer < L and sh.candidates is not None:
            score = score.masked_fill(~sh.candidates, float("-inf"))
        k_ = min(a.index_topk, n_c)
        idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
        idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
        return self._pad_topk(idx)

    def _indexer_stream(self, q, wts, k, n_pad, k_, L, compress_lens, sh, B, NB, T):
        """_indexer's math with the score reduced one query tile at a time.

        The query axis is independent at every step this buffer takes part in (einsum over keys,
        masked_fill, topk), so processing a query tile of ATTN_TILE rows yields the same per-row
        result as the whole [T, n_pad] matrix while keeping only [ATTN_TILE, n_pad] live. At 512k
        that is 67 MB instead of 2.1 GB, which is the transient Model._indexer's comment names as
        the thing that caps the context. Compute is unchanged: the same einsum tiles, the same
        total topk/amax work. DSV41_INDEX_STREAM=0 restores the monolithic buffer.
        """
        a = self.args
        cpos = self._positions[:n_pad]
        is_cand_src = L == a.candidate_source_layer
        later = 0 <= a.candidate_source_layer < L and sh.candidates is not None
        if is_cand_src:
            # Filled tile by tile; the decoder indexers read it whole, exactly as before.
            sh.candidates = torch.zeros(T, n_pad, dtype=torch.bool, device=self.dev)
        idx_out = torch.full((T, k_), -1, dtype=torch.int64, device=self.dev)
        for i in range(0, T, B):
            j = min(i + B, T)
            bq = j - i
            qt, wt = q[i:j], wts[i:j]
            cl = compress_lens[i:j]
            if bq < B:
                # Same zero padding the monolithic path does for its last query tile; the padded
                # rows are scored and then dropped.
                pad = B - bq
                qt = torch.cat([qt, qt.new_zeros(pad, *qt.shape[1:])])
                wt = torch.cat([wt, wt.new_zeros(pad, wt.size(1))])
                cl = torch.cat([cl, cl.new_zeros(pad)])
            score = torch.empty(B, n_pad, dtype=SCORE_DTYPE, device=self.dev)
            for jb in range(0, n_pad, NB):
                sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])  # bf16
                sc = sc.float().relu_() * wt[:, :, None]
                score[:, jb:jb + NB] = sc.sum(dim=1).to(score.dtype)
            score.masked_fill_(cpos[None, :] >= cl[:, None], float("-inf"))
            if is_cand_src:
                keep = self._select_candidates(score, cl, a.candidate_topk_blocks,
                                               a.candidate_block_size)
                sh.candidates[i:j] = keep[:bq]
            elif later:
                # Only the real rows carry a candidate mask; the zero-padded tail is discarded.
                score[:bq].masked_fill_(~sh.candidates[i:j], float("-inf"))
            tv = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
            tv = torch.where(tv < cl[:, None], tv, torch.full_like(tv, -1))
            idx_out[i:j] = tv[:bq]
        return self._pad_topk(idx_out)

    def _pad_topk(self, idx: torch.Tensor) -> torch.Tensor:
        """Always hand back exactly index_topk columns, the tail filled with -1 (= masked off).

        The number of compressed rows a chunk can reach, min(index_topk, (S+T)//ratio), grows with
        the chunk, so without this the concatenated KV of the attention softmax would be a
        different width in a short chunk than in a long one. The extra columns are fully masked
        and change nothing mathematically, but a different N makes cuBLAS pick a different kernel
        for the score GEMM, and the resulting ulp differences flip router decisions downstream.
        """
        pad = self.args.index_topk - idx.size(1)
        return idx if pad <= 0 else F.pad(idx, (0, pad), value=-1)

    @staticmethod
    def _select_candidates(logits, compress_lens, topk_blocks, block_size):
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = scores.size(-1)
        last = ((compress_lens - 1) // block_size)[:, None]
        scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device)[None, :] == last, float("inf"))
        top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block_size, dim=-1)[..., :width]

    # ------------------------------------------------------------------ blocks
    MISS_POLICIES = ("substitute", "skip", "skip_raw", "random")

    def _apply_miss_policy(self, policy, unmasked, scores, indices, weights, keep, k):
        """Eval-only (tools/bench_miss_policy_tp.py): what a routed pick the resident set lacks
        becomes. Serving never sets `miss_policy`, so it keeps `substitute`: the masked top-k above,
        where the next-ranked resident expert fills the slot with weight from its own score.

        A slot is a substitute when the unmasked router did not choose that expert. Its expert id is
        resident by construction, so every policy keeps the slot table valid and changes only an
        identity (`random`) or a weight (`skip*`). Rows with no miss come out bit-identical:
          skip      substitutes weigh 0; the resident picks renormalize to the full route scale
          skip_raw  substitutes weigh 0; the resident picks keep the weights the unpruned router
                    gives them, so the routed output is short by the missing experts' share
          random    each substitute's expert becomes a random other resident one, same weight
        """
        if policy not in self.MISS_POLICIES:
            raise ValueError(f"miss policy {policy!r}")
        want = unmasked.topk(k, dim=-1)[1]
        sub = ~(indices[:, :, None] == want[:, None, :]).any(-1)
        stats = getattr(self, "miss_stats", None)
        if stats is not None:  # device counters: substitute slots, all slots (no host sync)
            stats[0] += sub.sum()
            stats[1] += sub.numel()
        if policy == "random":
            r = torch.rand(unmasked.shape, generator=self.miss_gen, device=unmasked.device)
            r = r.masked_fill(~keep, -1.0).scatter(1, indices, -1.0)
            pick = r.topk(k, dim=-1)[1]  # k distinct residents a row, none already selected
            nth = (sub.to(torch.int64).cumsum(-1) - 1).clamp(min=0)
            return torch.where(sub, pick.gather(1, nth), indices), weights
        kept = scores.gather(1, indices).masked_fill(sub, 0.0)
        denom = (kept.sum(dim=-1, keepdim=True) if policy == "skip"
                 else scores.gather(1, want).sum(dim=-1, keepdim=True))
        return indices, kept / (denom + 1e-20) * self.args.route_scale

    def _apply_expert_topp(self, rule, weights):
        """Eval-only (tools/bench_miss_policy_tp.py `topp*` arms): TensorFold's lossy routed-expert
        pruning (its families/deepseek_v41/cuda/prune.py). Per row, by weight share f descending (ties:
        the lower slot), keep the shortest prefix whose mass before it is < `topp`, never fewer than
        `min_k`; a dropped slot weighs 0 (its id stays, so the slot table is valid; a serving version
        would also skip its read). `orig` keeps the kept weights as the router gave them, so the
        routed output shrinks by the dropped share; `kept` renormalizes them to the route scale.
        Rows that keep every slot are bit-identical."""
        topp, min_k, renorm = rule
        f = weights / weights.sum(dim=-1, keepdim=True)
        order = torch.argsort(-f, dim=-1, stable=True)
        fs = f.gather(1, order)
        keep_sorted = (fs.cumsum(dim=-1) - fs) < topp
        keep_sorted[:, :min_k] = True
        keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
        stats = getattr(self, "topp_stats", None)
        if stats is not None:  # device counters: kept slots, all slots
            stats[0] += keep.sum()
            stats[1] += keep.numel()
        nw = torch.where(keep, weights, torch.zeros_like(weights))
        if renorm == "kept":
            nw = torch.where(keep.all(dim=-1, keepdim=True), weights,
                             nw / (nw.sum(dim=-1, keepdim=True) + 1e-20) * self.args.route_scale)
        else:
            nw = torch.where(keep.all(dim=-1, keepdim=True), weights, nw)
        return nw

    def _record_prune_miss(self, logits, scores, keep_mask, L: int, k: int, decode: bool = False):
        """Record what the router WANTED, per (layer, expert) -- reachable or not.

        Recording only misses would make the database self-erasing: promote an expert and it
        stops being missed, so the evidence that justified promoting it disappears and the next
        adaptation demotes it again. Demand is stable under adaptation, and the miss set is just
        demand restricted to the currently-pruned experts, so nothing is lost.

        `logits` is still UNMASKED here. Everything is fixed-shape (the decode path records from
        inside a captured CUDA graph) and stays on the GPU until a request ends.
        """
        if getattr(self, '_prefix_replay_only', False):
            return  # Cache preparation must not count the same request a second time.
        probe = getattr(self, '_probe_demand', None)
        if probe is not None:
            probe.record(logits, scores, L, k, self._moe_start)
            return  # A discovery probe is provisional, never a history vote.
        d = 1 if decode else 0
        if PRUNE_MISS_FUSED:
            from engine import prune_miss_fused as PMF
            if PMF.supported(logits, scores, keep_mask):
                self.alloc_prune_miss(logits.size(-1), logits.device)
                PMF.record(logits, scores, keep_mask, k, self._rec_counts[L], self._rec_mass[L],
                           self._want_phase[d][L], self._miss_tot[L], self._miss_phase[d],
                           self._miss_mass_tot[L])
                return
        want = logits.topk(k, dim=-1)[1]                       # [T, k] an unpruned router's picks
        self.alloc_prune_miss(logits.size(-1), logits.device)
        flat = want.reshape(-1)
        ones = torch.ones_like(flat, dtype=torch.float64)
        sc = scores.gather(1, want).double().reshape(-1)
        self._rec_counts[L].scatter_add_(0, flat, ones)
        # score mass: being denied a high-scoring expert costs more than a marginal one
        self._rec_mass[L].scatter_add_(0, flat, sc)
        # Split by phase. The prompt shows the router ~18x more expert choices than the
        # generation does, so a combined history is ~95% prefill -- yet output quality depends on
        # DECODE routing. Whether one predicts the other decides both how to weight this database
        # and whether a per-request preload between prefill and decode is worth anything.
        self._want_phase[d][L].scatter_add_(0, flat, ones)
        miss = (~keep_mask[want]).double()
        self._miss_tot[L, 0] += miss.sum()
        self._miss_tot[L, 1] += miss.numel()
        self._miss_phase[d, 0] += miss.sum()
        self._miss_phase[d, 1] += miss.numel()
        self._miss_mass_tot[L, 0] += (miss.reshape(-1) * sc).sum()
        self._miss_mass_tot[L, 1] += sc.sum()

    def alloc_prune_miss(self, n_experts: int, device):
        """Allocate the accumulators eagerly. The decode path records inside a captured CUDA
        graph, so the tensors must already exist (and keep their addresses) before capture --
        every op in the recorder is fixed-shape for the same reason."""
        if self._want_counts is None:
            z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=device)  # noqa: E731
            self._want_counts = z(self.args.n_layers, n_experts)
            self._want_mass = z(self.args.n_layers, n_experts)
            self._miss_tot = z(self.args.n_layers, 2)
            self._miss_mass_tot = z(self.args.n_layers, 2)
            self._want_phase = [z(self.args.n_layers, n_experts), z(self.args.n_layers, n_experts)]
            self._miss_phase = z(2, 2)
            # Where _record_prune_miss writes. In request mode the live request accumulates here
            # raw and is folded into _want_counts normalized at request end; in slot mode it IS the
            # database. The captured decode graph holds the chosen tensor's address, so this is
            # fixed for the process lifetime.
            self._req_counts = z(self.args.n_layers, n_experts)
            self._req_mass = z(self.args.n_layers, n_experts)
            self._rec_counts = self._req_counts if PRUNE_UNIT_REQUEST else self._want_counts
            self._rec_mass = self._req_mass if PRUNE_UNIT_REQUEST else self._want_mass

    def decay_demand(self, factor: float):
        """Scale recorded demand down, in place, so old evidence fades.

        A purely cumulative history is wrong for a server whose workload switches -- news, then
        research, then coding all average into a compromise resident set that fits none of them.
        Decaying turns the database into an exponential moving average: the current mode wins,
        and a mode that stops being used fades rather than holding slots forever.
        """
        if self._want_counts is None or not (0.0 < factor < 1.0):
            return
        self._want_counts *= factor
        self._want_mass *= factor

    def reset_request_demand(self):
        """Zero the per-request accumulator at request start (request-unit mode)."""
        if PRUNE_UNIT_REQUEST and self._req_counts is not None:
            self._req_counts.zero_()
            self._req_mass.zero_()

    def flush_request_demand(self, halflife_requests: float) -> int:
        """Fold this request in as ONE vote, then age the database by one request.

        Per layer the request's routed counts are normalized to a distribution, then
        db = db * 0.5**(1/H) + distribution. Every request contributes a unit of mass however many
        tokens it moved, which is what makes a three-sentence prompt count as much as a 10k-token
        one. Returns the number of layers that saw any demand. No-op in slot mode, where
        _record_prune_miss writes straight into the database and decay_demand ages it on idle ticks.
        """
        if not PRUNE_UNIT_REQUEST or self._req_counts is None:
            return 0
        decay = (0.5 ** (1.0 / halflife_requests)
                 if halflife_requests and halflife_requests > 0 else 1.0)
        self._want_counts.mul_(decay)
        self._want_mass.mul_(decay)
        # Normalize score votes on device; a Python `if score_total > 0` per layer would add
        # forty blocking device readbacks at every prefill boundary and request end.
        score_tot = self._req_mass.sum(dim=1, keepdim=True)
        self._want_mass.add_(self._req_mass / score_tot.clamp_min(1e-20))
        tot = self._req_counts.sum(dim=1)
        folded = 0
        for L in torch.nonzero(tot > 0).flatten().tolist():
            self._want_counts[L].add_(self._req_counts[L] / tot[L])
            # Scores need their OWN request accumulator and normalization. Previously raw
            # scores went straight into history, then flush decayed the new observation too
            # and never normalized it. That history cannot seed score-based ranking.
            folded += 1
        self._req_counts.zero_()
        self._req_mass.zero_()
        return folded

    def load_demand(self, counts, mass):
        """Seed the accumulators from a persisted database so it accumulates across restarts."""
        self.alloc_prune_miss(counts.shape[1], self.dev)
        self._want_counts += torch.as_tensor(counts, dtype=torch.float64, device=self.dev)
        self._want_mass += torch.as_tensor(mass, dtype=torch.float64, device=self.dev)

    def prune_miss_report(self):
        """(summary, demand counts [L, E], demand mass [L, E]) or None if nothing recorded."""
        if self._want_counts is None:
            return None
        tot = self._miss_tot.cpu()
        counts = self._want_counts.cpu()
        mass = self._want_mass.cpu()
        score_missed, score_total = self.miss_mass_snapshot()
        missed, slots = float(tot[:, 0].sum()), float(tot[:, 1].sum())
        per_layer = (tot[:, 0] / tot[:, 1].clamp_min(1)).tolist()
        worst = sorted(range(len(per_layer)), key=lambda L: -per_layer[L])[:5]
        ph = {}
        if self._miss_phase is not None:
            mp = self._miss_phase.cpu()
            wp = [x.cpu() for x in self._want_phase]
            for i, name in ((0, "prefill"), (1, "decode")):
                n_slots = float(mp[i, 1])
                if n_slots:
                    ph[name] = {"miss_rate": round(float(mp[i, 0]) / n_slots, 4),
                                "slots": int(n_slots),
                                "distinct_experts": int((wp[i] > 0).sum())}
            if len(ph) == 2:
                # do the two phases want the SAME experts? cosine over the per-layer demand
                a, b = wp[0].flatten(), wp[1].flatten()
                den = float(a.norm() * b.norm())
                ph["prefill_decode_cosine"] = round(float((a * b).sum()) / den, 4) if den else None
        return {
            "phase": ph,
            "miss_rate": round(missed / max(slots, 1.0), 4),
            "score_miss_rate": round(score_missed / max(score_total, 1e-20), 6),
            "missed_slots": int(missed), "total_slots": int(slots),
            "distinct_experts_wanted": int((counts > 0).sum()),
            "worst_layers": [{"layer": L, "miss_rate": round(per_layer[L], 4)} for L in worst],
        }, counts, mass

    def miss_snapshot(self):
        """(missed, total) routing slots so far -- for a PER-REQUEST delta.

        The accumulators are lifetime by design (they feed the demand database), so a cumulative
        miss rate in a per-request log line stops moving after a few hundred requests and tells
        you nothing about the prompt you just ran.
        """
        if self._miss_tot is None:
            return None
        tot = self._miss_tot.sum(dim=0)
        return float(tot[0]), float(tot[1])

    def stage_decode_miss(self):
        """Queue the decode counter pair's copy into pinned memory on the current stream, without
        waiting; the next decode_miss_snapshot() returns it. The greedy loop calls this right
        after a verify block and before it queues the next draft (DSV41_EARLY_DRAFT), then waits
        for the copy with the block's own readback. The drafter records no misses, so these are
        the values a synchronous read after the draft returns -- without the host waiting for
        the drafter, which a .tolist() on the stream did at the top of every step."""
        if self._miss_phase is None:
            return
        src = self._miss_phase[1]
        pin = getattr(self, "_miss_pin", None)
        if pin is None or pin.shape != src.shape or pin.dtype != src.dtype:
            pin = self._miss_pin = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        pin.copy_(src, non_blocking=True)
        self._miss_staged = True

    def decode_miss_snapshot(self):
        """Existing decode-only counters: one tiny readback, no new per-layer work."""
        if self._miss_phase is None:
            return None
        if getattr(self, "_miss_staged", False):
            self._miss_staged = False
            return tuple(self._miss_pin.tolist())
        return tuple(self._miss_phase[1].tolist())

    def miss_mass_snapshot(self):
        """Cumulative pre-mask router score denied by pruning / total requested score."""
        if self._want_counts is None:
            return None
        return tuple(self._miss_mass_tot.sum(dim=0).tolist())

    def reset_prune_miss(self):
        self._want_counts = self._want_mass = self._miss_tot = None
        self._miss_staged = False

    def moe(self, y: torch.Tensor, w, L: int, prefill: bool, store, arena, n_experts: int):
        a = self.args
        moe_timer = (getattr(self, "prefill_moe_timing", None)
                     if prefill and y.size(0) > 16 and n_experts != 128 else None)
        if moe_timer is not None:
            moe_timer.start(L, y.size(0))
        self._tap("moe_in", L, y)
        scores = F.softplus(R.mm(y.float(), w.gate_w)).sqrt()
        k = 3 if n_experts == 128 else a.n_activated_experts
        # The VL bias picks different experts for image tokens; it does not scale them (weights
        # still come from the raw scores below), so only the topk sees it.
        img = getattr(self, "image_mask", None)
        if img is not None and w.gate_bias_vl is not None and img.shape[0] == y.shape[0]:
            logits = scores + torch.where(img.unsqueeze(-1), w.gate_bias_vl, w.gate_bias)
        else:
            logits = scores + w.gate_bias
        pm = getattr(self, "prune_mask", None)
        streaming = n_experts != 128 and L in getattr(self, 'stream_layers', ())
        focus = getattr(self, 'user_prompt', None)
        focus_rows = (focus.row_mask(self._moe_start, y.size(0))
                      if prefill and n_experts != 128 and focus is not None else None)
        streaming = streaming or focus_rows is not None
        rescued = []
        if pm is not None and n_experts != 128 and L in pm:
            focus_keep = (focus.streaming_keep(L, logits, scores, focus_rows, pm[L], k, store)
                          if focus_rows is not None else None)
            effective_keep = (torch.where(focus_rows[:, None], focus_keep[None, :], pm[L][None, :])
                              if focus_rows is not None else
                              self.stream_keep[L] if streaming else pm[L])
            critical = getattr(self, 'critical_stream', None)
            if prefill and critical is not None and critical.enabled:
                rescued = critical.plan(L, logits, scores, pm[L], k, store.ep)
                if rescued:
                    effective_keep = pm[L].clone()
                    effective_keep[rescued] = True
            # What the router WANTED before pruning masked it. The prune set is ranked from a
            # traced corpus (coding + general), so on traffic unlike that corpus the router keeps
            # asking for experts that are not there -- and nothing downstream can see it, because
            # masking happens before topk and the second-choice expert looks like a normal pick.
            # DSV41_PRUNE_MISS=1 records, per (layer, expert), how often a pruned expert was the
            # router's actual preference, plus the score mass that preference carried. That is the
            # same shape as the coverage histogram the prune set is built from, so a run over real
            # traffic can re-rank it. Off by default: it costs a second topk per layer.
            if PRUNE_MISS:
                # In mixed-row streaming, this reports *resident* demand/misses;
                # the focus collector separately reports the rows served in full.
                self._record_prune_miss(logits, scores,
                                        pm[L] if focus_rows is not None else effective_keep, L, k)
            # expert pruning experiment: the router may only pick surviving experts (REAP-style drop)
            unmasked = logits
            logits = logits.masked_fill(~effective_keep, float("-inf"))
            hot = getattr(self, "hot_eval", None)  # eval-only (bench_miss_policy_tp `hot:`/`exact`)
            if hot is not None and prefill and not streaming and not rescued:
                row_keep, rescued = hot.widen(L, unmasked, scores, effective_keep, k)
                if rescued:  # the transient-ring path below loads them; residents stay put
                    logits = unmasked.masked_fill(~row_keep, float("-inf"))
        indices = logits.topk(k, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        policy = getattr(self, "miss_policy", "substitute")
        if (policy != "substitute" and pm is not None and n_experts != 128 and L in pm
                and not streaming and not rescued):
            indices, weights = self._apply_miss_policy(policy, unmasked, scores, indices, weights,
                                                       effective_keep, k)
        topp = getattr(self, "expert_topp", None)  # eval-only rule, any phase (bench_miss_policy_tp)
        if topp is None and not prefill and _EXPERT_TOPP is not None:
            # DSV41_EXPERT_TOPP on the eager decode path: the graphed path (FastDecoder) also skips
            # the dropped experts' reads; here they compute with weight 0, which adds exact zeros.
            topp = (_EXPERT_TOPP[0], _EXPERT_TOPP[1], "orig")
        if topp is not None and n_experts != 128 and not streaming and not rescued:
            weights = self._apply_expert_topp(topp, weights)
        if focus_rows is not None:
            focus.record(L, indices, weights / a.route_scale, focus_rows)
        self._tap("route_idx", L, indices); self._tap("route_w", L, weights)
        t0 = time.perf_counter()
        # Dreaming FP8 tier (engine/dream_overlay.py::load_fp8_tp): an expert the overlay serves as
        # FP8 still routes and resolves exactly as before -- its MXFP4 copy is in the arena -- and
        # the FP8 kernels overwrite its pairs' rows inside moe_fn (tools/fp4_moe.py::moe_forward).
        side = getattr(self, "dream_fp8", None)
        dream = (side[L], indices) if side and n_experts != 128 and L in side else None
        # All-resident configurations carry a device slot table (built by the engine from the
        # store's LRU). Using it here turns routing into one GPU gather instead of a host round-trip
        # plus a Python pass over every (layer, expert) pair in the chunk. The table is only valid
        # while nothing is evicted, which is exactly the pruned all-resident case; anything else
        # takes the host path.
        #
        # EP2: the table is built from THIS rank's LRU, which holds only the experts this rank
        # owns, so a non-owned expert would gather -1 rather than the null slot. v41_engine.py
        # refuses to build it under expert parallel for exactly that reason; the assert is the
        # backstop, because the failure mode is silent wrong slots, not an error.
        lut = getattr(self, "slot_lut", None)
        sector_stream = streaming and focus is not None and getattr(focus, 'dynamic', False)
        if sector_stream:
            try:
                from engine.global_residency import streaming_moe
                routed = streaming_moe(self, y, indices, weights, L, store, arena, prefill=prefill)
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)  # a failed TP load/plan must not leave its peer serving
        elif lut is not None and n_experts != 128 and not rescued and not streaming:
            # EP2-safe: build_lut fills non-owned entries with the null slot rather than -1, and
            # validates that once at construction -- checking it here would be a device->host sync
            # per layer per chunk, which is the cost this path exists to remove.
            slots = lut[L][indices]
            self.stats["hits"] = self.stats.get("hits", 0) + indices.numel()
        else:
            try:
                # Streaming decode also uses the transient ring. Never evict a resident
                # whose address is baked into another layer's captured graph.
                slots = store.resolve(L, indices, prefill or streaming)
            except Exception:
                if rescued or streaming:
                    # After the common plan, a one-rank load failure must stop the pair.
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
                raise
        # EP2 combine: an ExpertStore with a null slot is by construction an ep-enabled main model
        # store (the DSpark drafter runs on a FixedStore and never has one), and its `routed` holds
        # only THIS rank's owned subset -- zeros everywhere else. One fp32 all-reduce makes both
        # ranks hold the full routed sum, which is what keeps the two decode loops computing the
        # same tokens (engine/dist.py). shared is added AFTER the combine, and only here, so it is
        # counted once and its single-node numerics are untouched.
        #
        # The rounding point matters as much as the sum. The kernel accumulates the k-sum in fp32
        # and rounds to bf16 on the way out; under EP2 that would round THIS HALF, the peer's half
        # separately, and leave their fp32 total carrying both errors. So EP2 takes the half in
        # fp32, all-reduces, and rounds once, exactly where the single-box path rounds: what
        # remains is purely the REGROUPED fp32 addition engine/dist.py documents.
        if getattr(store, "null_slot", None) is not None and not sector_stream:
            route = getattr(self, "prefill_routes", {}).get(L) if prefill else None
            if streaming:
                route = None  # static compact IDs do not include nonresident experts
            route_args = {}
            if rescued:
                # Static compact IDs omit pruned experts. Rebuild only this call's routing;
                # the resident LUT, weights and captured decode graphs stay intact.
                slot_map, inverse = torch.unique(slots, sorted=True, return_inverse=True)
                route_args.update(routing_ids=inverse, routing_slot_map=slot_map)
                route = None  # the static route[0][expert_id] lookup does not apply here
            if moe_timer is not None:
                route_args['stage_mark'] = moe_timer.mark
            if route is not None and os.environ.get("DSV41_PREFILL_FIXED_ROUTING", "1") == "1":
                route_args.update(routing_ids=route[0][indices], routing_slot_map=route[1])
            if moe_timer is not None:
                moe_timer.mark('router_and_lookup')
            # slots_repeat + null_slot: every non-owned expert of this call shares ONE zero slot.
            # At decode size the router gives those pairs temporary unique keys, then maps them
            # back to the skipped null slot, so real experts retain the tuned BM=16. Without the
            # special route, ~18 null pairs overflow one 16-row block; using BM=64 avoids the
            # corruption but makes every real expert slower. See tools/fp4_moe.py.
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit, out_dtype=torch.float32,
                                 slots_repeat=True, prefill=prefill,
                                 null_slot=(store.null_slot if (not prefill or
                                            os.environ.get("DSV41_PREFILL_SKIP_NULL", "1") == "1") else -1),
                                 dream=dream, **route_args)
            # Split the combine off on the GPU timeline. The host timer below (ep_s) measures the
            # LAUNCH -- dist.all_reduce is async on CUDA -- which is why it reports ~0.25 ms for a
            # collective that G2 measured at 3.5 ms for this payload. The gap between these two
            # marks is the real cost, and it includes waiting for the peer to arrive: with 40
            # rendezvous per chunk, that wait is the prime suspect for the GPU going up and down
            # during prefill.
            _mark("moe_kernel")
            tc = time.perf_counter()
            if not getattr(store.ep, 'tensor_parallel', False):
                store.ep.combine(routed)
                _mark("ep_combine")
            self.stats["ep_s"] += time.perf_counter() - tc   # pure network time; moe_s includes it
            self.stats["ep_calls"] += 1                      # ep_s/ep_calls = per-collective cost
        elif not sector_stream:
            # TP without a null slot (EXPERT_FORMAT=exl3): the same static per-layer compact routes as
            # the branch above, so prefill routing needs no unique() and no host sync per layer.
            route = (getattr(self, "prefill_routes", {}).get(L)
                     if prefill and not streaming and not rescued else None)
            route_args = {}
            if route is not None and os.environ.get("DSV41_PREFILL_FIXED_ROUTING", "1") == "1":
                route_args = dict(routing_ids=route[0][indices], routing_slot_map=route[1])
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit, prefill=prefill, dream=dream,
                                 **route_args).float()
        _mark("moe")         # router + slot resolve + routed experts; combine may overlap below
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        # A TP shared expert's RowParallelWeight reduces before BF16 rounding inside
        # expert_ffn. Unsharded DSpark shared experts must not be reduced a second time.
        _mark("shared_expert")   # the dense FFN every token passes through
        self._tap("moe_routed", L, routed); self._tap("moe_shared", L, shared)
        out = routed + shared
        self.stats["moe_s"] += time.perf_counter() - t0
        return out.to(y.dtype)

    def block(self, h, pre_mix, w, L, S, sh, ring, freqs, prefill, store, arena, n_experts, mtp_extra=None,
              win_lo: int = 0):
        a = self.args
        residual = h
        attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        y = (_hc_pre_rn_fused(h, pre_mix, w.attn_norm, a.norm_eps) if HC_OPS
             else R.rmsnorm(R.hc_pre(h, pre_mix), w.attn_norm, a.norm_eps))
        _mark("hc_attn_mix")
        t0 = time.perf_counter()
        y = self.attention(y, w, L, S, sh, ring, freqs, mtp_extra, win_lo=win_lo, prefill=prefill)
        self.stats["attn_s"] += time.perf_counter() - t0
        h = (_hc_post_fused(y, residual, attn_post, attn_comb) if HC_OPS
             else R.hc_post(y, residual, attn_post, attn_comb))
        _mark("hc_post_attn")
        graphs = getattr(self, 'prefill_graphs', None)
        if graphs is not None and graphs.eligible(h, L, prefill, n_experts):
            # Intermediate outputs feed the next attention block immediately. Final
            # encoder/decoder outputs must outlive staging for replay/prefix caches.
            retain = L in (a.candidate_source_layer, a.n_layers - 1)
            return graphs.run(h, attn_pre, w, L, store, arena, n_experts, retain=retain)
        self._moe_start = S  # Absolute row position, including bounded decoder replay.
        return self._ffn(h, attn_pre, w, L, prefill, store, arena, n_experts)

    def _ffn(self, h, attn_pre, w, L, prefill, store, arena, n_experts):
        """Position-independent FFN, shared verbatim by eager and graphed prefill."""
        a = self.args
        residual = h
        ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        y = (_hc_pre_rn_fused(h, attn_pre, w.ffn_norm, a.norm_eps) if HC_OPS
             else R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps))
        # Everything above since o_proj is hyper-connection bookkeeping on the 4x-wide residual
        # stream, not MoE -- it was being charged to the moe_kernel gap.
        _mark("hc_ffn_mix")
        y = self.moe(y, w, L, prefill, store, arena, n_experts)
        h = (_hc_post_fused(y, residual, ffn_post, ffn_comb) if HC_OPS
             else R.hc_post(y, residual, ffn_post, ffn_comb))
        _mark("hc_post_ffn")
        return h, ffn_pre

    # ------------------------------------------------------------------ SWA bounded replay
    def begin_prompt(self):
        """Drop whatever the previous prompt left in the replay buffer."""
        self._rep = {"h": [], "pre_mix": [], "topk": [], "cand": []}
        self._rep_end = 0

    def _rep_keep(self, h, pre_mix, sh: Shared, S: int, T: int):
        """Remember the last `window_size` encoder outputs of the prompt so far.

        Only the tail is ever needed, so each chunk contributes at most `window_size` rows and the
        buffer is trimmed as soon as it has more than that.
        """
        w = self.args.window_size
        n = min(w, T)
        sl = slice(T - n, T)
        r = self._rep
        r["h"].append(h[sl]); r["pre_mix"].append(pre_mix[sl])
        # layers 21..23 reuse layer 20's top-k and layers 24..39 search inside layer 20's candidate
        # pool, and both are computed per query -- so they belong to the queries, not to the caches,
        # and the replay has to carry them across from the encoder pass instead of recomputing them.
        r["topk"].append(sh.topk[sl])
        r["cand"].append(sh.candidates[sl] if sh.candidates is not None else None)
        self._rep_end = S + T
        while sum(t.size(0) for t in r["h"]) - r["h"][0].size(0) >= w:
            for k in r:
                r[k].pop(0)

    def _rep_tail(self):
        w = self.args.window_size
        r = self._rep
        h = torch.cat(r["h"])[-w:]
        pre_mix = torch.cat(r["pre_mix"])[-w:]
        topk = torch.cat(r["topk"])[-w:]
        cands = r["cand"]
        cand = None
        if cands and cands[0] is not None:
            width = max(c.size(1) for c in cands)
            # older chunks scored fewer compressed columns; a query can only ever see columns below
            # its own position, all of which are inside its own chunk's width, so padding the rest
            # with False changes nothing that is reachable.
            cand = torch.cat([c if c.size(1) == width else F.pad(c, (0, width - c.size(1)), value=False)
                              for c in cands])[-w:]
        return h, pre_mix, topk, cand, self._rep_end - h.size(0)

    @torch.inference_mode()
    @prefill_phase
    def decoder_replay(self, need_logits: bool = True):
        """Decoder SWA Bounded Replay (tech report 2.2 / 3.2.2).

        Under CED the decoder's global KV is projected from the last encoder layer's hidden state,
        which the encoder pass has already written for every prompt position. The only thing the
        decoder layers still owe the first decode steps is their own sliding-window KV -- so they
        are run over the last `window_size` prompt tokens only, with SWA truncated to that segment,
        instead of over the whole prompt. The prompt's final logits come from this pass.
        """
        # Replay runs the decoder half over the window tail, which is text by construction and a
        # different length from the chunk that set these. A stale image_mask here would apply the
        # VL routing bias to ordinary tokens whenever the lengths happened to agree.
        self.image_mask = self.engram_mask = None
        a = self.args
        h, pre_mix, topk, cand, S = self._rep_tail()
        T = h.size(0)
        sh = Shared()
        src = a.candidate_source_layer
        sh.ckv, sh.ik, sh.ratio = self.c.ckv[src], self.c.ik[src], self.args.compress_ratios[src]
        sh.topk, sh.candidates = topk, cand
        main_hiddens = []
        for L in range(src + 1, len(self.W.layers)):
            w = self.W.layers[L]
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, True, self.store,
                                    self.store.arena, a.n_routed_experts, win_lo=S)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.last_h, self.last_pre_mix = h, pre_mix
        logits = None
        if need_logits:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        self.stats["replay_tokens"] = self.stats.get("replay_tokens", 0) + T
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None), S

    @torch.inference_mode()
    @forward_phase
    def forward(self, ids: torch.Tensor, S: int, prefill: bool, need_logits: bool = True,
                encoder_only: bool = False, hashes: torch.Tensor | None = None, get_rows=None,
                token_types: torch.Tensor | None = None, images=None):
        """ids: [T] token ids at positions S..S+T-1. Returns (logits [T, V] fp32 or None, main_hidden [T, 15360]).
        Caches must be valid for positions < S (self.c.len == S).

        ``hashes`` / ``get_rows`` let the caller drive the engram reads across chunk boundaries
        (see EngramReadAhead): the hashes depend on the token ids alone, so a prefill loop can hash
        the whole prompt once and have chunk k+1's rows already in flight while chunk k runs. When
        they are None this falls back to hashing here and prefetching within the chunk.

        ``encoder_only`` stops after the candidate-source layer (the last layer that writes global
        KV, layer 20 here): everything above it is replayed once over the prompt tail by
        `decoder_replay`. It returns (None, None) -- there are no logits and no DSpark hidden
        states below layer 37.
        """
        a = self.args
        assert self.c.len == S, (self.c.len, S)
        T = ids.size(0)
        # An image span must be spliced whole, so a span longer than MAX_CHUNK gets a chunk of
        # its own (V41Engine._prefill_spans). Everything downstream is shape-generic; the cost is
        # the indexer's [T, context] score buffer, which is why ordinary chunks stay at MAX_CHUNK.
        assert T <= MAX_CHUNK or images, (T, MAX_CHUNK)
        # Vision masks for this chunk. TEXT is -1, so `types >= 0` is the whole image span --
        # delimiters included. image_mask selects the router's VL bias (Model.moe); its complement
        # shuts the engram off, both in the hasher (an n-gram must not span an image) and in the
        # engram forward itself (a dead row must not be written into the stream).
        self.image_mask = None if token_types is None else (token_types >= 0)
        self.engram_mask = None if self.image_mask is None else ~self.image_mask
        if hashes is None:
            t0 = time.perf_counter()
            hashes = (self.hash_state(ids[None], S, None if self.engram_mask is None
                                      else self.engram_mask[None])[0]
                      if self.hash_state is not None else None)  # [T, 2, 24]
            self.stats["engram_s"] += time.perf_counter() - t0
        e = self.W.embed[ids]
        if images:
            # before the hc_mult expand: the splice writes plain [T, dim] embedding rows
            assert self.vision is not None, "prompt has images but no vision tower is loaded"
            e = e.clone()          # W.embed is the shared table; never scribble on it
            self.vision.splice(e, images)
        h = e.unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(T, a.hc_mult, device=self.dev)
        pre_mix[:, 0] = 1.0
        sh = Shared()
        main_hiddens = []
        n_layers = len(self.W.layers)
        last = a.candidate_source_layer if encoder_only else n_layers - 1
        prefetch = getattr(self, "engram_prefetch", None)
        if get_rows is not None:
            context = nullcontext(get_rows)   # caller owns the reads (cross-chunk read-ahead)
        else:
            context = (prefetch(hashes) if prefill and hashes is not None and prefetch is not None
                       and os.environ.get("DSV41_PREFILL_ENGRAM", "1") == "1"
                       else nullcontext(self.engram_rows))
        with context as get_rows:
            for L in range(last + 1):
                w = self.W.layers[L]
                if L in self.W.engram:
                    t0 = time.perf_counter()
                    li = list(a.engram_layer_ids).index(L)
                    rows = get_rows(L, hashes[:, li, :])
                    self._tap("engram_rows", L, rows)
                    # image spans take no part in an n-gram and get no engram contribution
                    h = R.engram_forward(h, rows, self.W.engram[L], a,
                                         token_mask=getattr(self, "engram_mask", None))
                    self._tap("engram_out", L, h)
                    self.stats["engram_s"] += time.perf_counter() - t0
                    _mark("engram")
                if L in a.dspark_target_layer_ids:
                    main_hiddens.append(h.float().mean(dim=1))
                freqs = self.freqs_c if w.ratio else self.freqs_w
                h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                        self.store.arena, a.n_routed_experts)
                self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.c.len = S + T
        self.stats["tokens"] += T
        if encoder_only:
            self._rep_keep(h, pre_mix, sh, S, T)
            return None, None
        logits = None
        self.last_h, self.last_pre_mix = h, pre_mix
        if need_logits and n_layers == a.n_layers:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None)

    # ------------------------------------------------------------------ DSpark
    @torch.inference_mode()
    def dspark_seed(self, main_hidden: torch.Tensor, S: int):
        """Write the drafter's window KV for main positions S..S+M-1 from their main hiddens [M, 15360]."""
        a = self.args
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(R.qlinear(main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        M = main_x.size(0)
        pos = self._positions[S:S + M]
        rd = a.rope_head_dim
        for k, w in enumerate(self.W.mtp):
            kv = R.rmsnorm(R.qlinear(main_x, w.wkv), w.kv_norm, a.norm_eps)
            kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], self.freqs_w[S:S + M])], dim=-1)
            self.c.mtp_win[k][pos % RING] = kv

    @torch.inference_mode()
    def dspark_draft(self, tok: int, last_main_pos: int, temperature: float):
        """Draft block: returns (draft ids [B], draft probs [B, V] fp32 at the given temperature,
        confidence [B]) for B = the DSpark block size. Queries sit at last_main_pos+1 .. +B."""
        a = self.args
        from engine.fastdecode import T_DRAFT as B   # DSV41_BLOCK, default the checkpoint's 5
        ids = torch.full((B,), 128799, dtype=torch.long, device=self.dev)
        ids[0] = tok
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(B, a.hc_mult, device=self.dev); pre_mix[:, 0] = 1.0
        S = last_main_pos + 1
        sh = Shared()
        for k, w in enumerate(self.W.mtp):
            h, pre_mix = self.block(h, pre_mix, w, a.n_layers + k, S, sh, self.c.mtp_win[k], self.freqs_w, False,
                                    self.W.dspark_store, self.W.dspark_arena, 128, mtp_extra=last_main_pos)
        w = self.W.mtp[2]
        x = R.hc_pre(h, pre_mix)
        # The reference feeds the UN-normed hc_pre output to the confidence head and the normed one
        # to the LM head (inference/model.py::DSparkBlock.forward_head), so keep both.
        x_pre = x
        x = R.rmsnorm(x, w.norm, a.norm_eps)
        logits = R.head_logits(x, self.W.head)  # [B, V] fp32
        out = torch.empty(B + 1, dtype=torch.long, device=self.dev)
        out[0] = tok
        probs = []
        embeds = []
        for i in range(B):
            e = w.markov_embed[out[i]]  # [256]
            bias = F.linear(e.to(torch.bfloat16)[None], w.markov_head).float()[0]  # [V]
            lg = logits[i] + bias
            if temperature <= 0:
                p = torch.zeros_like(lg); p[lg.argmax()] = 1.0
                nxt = lg.argmax()
            else:
                p = torch.softmax(lg / temperature, dim=-1)
                nxt = torch.multinomial(p, 1)[0]
            out[i + 1] = nxt
            probs.append(p)
            embeds.append(e.float())
        # DSparkConfidenceHead returns the raw projection (no sigmoid); adaptive verification is off
        # in this engine, so it is reported, not acted on.
        conf = (torch.cat([x_pre.float(), torch.stack(embeds)], dim=-1) @ w.conf_proj.T).squeeze(-1)
        return out[1:], torch.stack(probs), conf
