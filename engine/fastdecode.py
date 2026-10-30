"""
fastdecode.py -- CUDA-graph decode path for the 6-token DSpark verify step (and the 5-token draft).

Why: profiled with every expert resident, one verify step of `Model.forward` costs ~436 ms, of which
~360 ms is CPU launch overhead (~10k tiny ops) and the GPU side is slowed by fp32 GEMMs (the fp32 LM
head alone: 37 ms at 72 GB/s). This module runs the same math with static buffers so it can be
captured into CUDA graphs, uses bf16 GEMMs (fp32 accumulation) everywhere the checkpoint stores bf16
or fp8, one fused Triton kernel for the Hyper-Connection Sinkhorn coefficients, and fixed-length
(masked) indexer scoring instead of context-length-dependent slices.

Semantics are those of engine/model.py (window ring, shared compressed KV, Hierarchical Sparse
Indexer with the layer-20 candidate pool, engram rows, mHC single-pass shift, DSpark draft with the
Markov head). Per backbone layer there are two graphs: A = attention + HC + router (ends with the
expert ids), then the host resolves expert slots (LRU / NVMe), then B = MoE + shared expert + HC
residual. With every routed expert resident (`--prune-keep`) the host step is a LUT lookup and the
whole layer could be one graph; that is left for later.

Numerics vs `Model.forward`: operations whose CUDA implementation depends on the activation row
count use the same fixed row shapes as the eager path (16, or 32 for HC by default). This is required for speculative decode:
the verifier must produce the target model's logits, not merely a close approximation to them.
`engine/test_fastdecode.py` measures the gap.
"""

from __future__ import annotations

import os
from collections import OrderedDict
import time

import torch
import torch.nn.functional as F

from engine import model as M
from engine.model import HC_OPS as _HC_OPS, _hc_post_fused, _hc_pre_rn_fused
from engine import comm as _comm
from engine import l2pf
from engine import expert_topp as _ETP
from engine.perf_config import DecodeExperiments
import v41_ref as R

try:
    from decode_attn import decode_attention  # tools/decode_attn.py (Triton)
except Exception:  # noqa: BLE001
    decode_attention = None

# One fused Triton kernel for the sinked softmax attention instead of two fp32 SIMT batched GEMMs
# and the elementwise passes around them. DSV41_FUSED_ATTN=0 restores the torch path.
# Default OFF since 2026-09-11: with the dense projections in fp4 and the fp8 head, greedy
# decoding through this kernel diverges from the same decode without it and can fall into a
# repetition loop (NOTES 2026-09-11 20:30, engine/test_spec_lossless.py). Each piece is fine
# alone; together they cross the precision the verify step needs. DSV41_FUSED_ATTN=1 re-enables.
FUSED_ATTN = os.environ.get("DSV41_FUSED_ATTN", "0") == "1" and decode_attention is not None
# Capture whole runs of layers into one graph instead of one graph per layer (resident mode only).
# DSV41_GRAPH_SEGMENTS=0 restores one graph per layer.
GRAPH_SEGMENTS = os.environ.get("DSV41_GRAPH_SEGMENTS", "1") == "1"
# Lean step: the same switch engine/v41_engine.py reads. It removes the per-step host work that is
# not the graph replay itself -- the `torch.arange` and the `repeat` of the embedding rebuilt every
# step, the constant pre_mix, the second `prepare_pending_buffers()` (only the capture run needs it)
# and the self-copy of the block ids. DSV41_LEAN_STEP=0 restores the original sequence exactly; the
# math is identical either way (the buffers get the same values).
LEAN_STEP = os.environ.get("DSV41_LEAN_STEP", "1") == "1"
# Context-bucketed sparse-index scoring.  The fallback is intentionally retained for controlled
# A/Bs and emergency rollback; it restores the old full-max-seq graph shapes exactly.
INDEX_BUCKETS = os.environ.get("DSV41_INDEX_BUCKETS", "1") == "1"
# Experimental fork/join of the replicated shared expert and routed TP work.
# Keep off until full-engine timings establish a gain, not just more GPU activity.
SHARED_OVERLAP = os.environ.get("DSV41_DECODE_SHARED_OVERLAP", "0") == "1"
# Prune-miss accounting (DSV41_PRUNE_MISS) on a side stream. It only feeds the demand database,
# never routing, so moving it off the router's critical path cannot change a token: 12.6 us a
# layer on the main stream in the 2026-10-21 profile. Joined after the MoE when one graph holds
# both halves of a layer, else at the end of the attention half.
PRUNE_MISS_SIDE = os.environ.get("DSV41_PRUNE_MISS_SIDE", "1") == "1"
# Experimental: one fp8 launch for projections that read the same activation -- attention
# wq_a||wkv and shared-expert w1||w3 -- instead of two launches and two identical activation
# quantizations each. Column-wise the GEMM is unchanged, so the output is meant to be
# bit-identical; the stacked weight replaces the originals' storage (they become views), so no
# memory is added. Read at call time so the A/B bench can toggle it; the boot guard pins it.
MERGED_PROJ = os.environ.get("DSV41_DECODE_MERGED_PROJ", "1") == "1"
# Fewer launches for the same bits (engine/decode_lean.py, docs/decode-launches.md): rmsnorm and
# the HC coefficients keep their row-padded torch reductions but get static padding buffers and one
# fused elementwise tail, and the layer writes straight into its static buffers instead of copying
# fresh results into them. DSV41_DECODE_LEAN=0 restores the previous spelling; the boot guard pins it.
LEAN = os.environ.get("DSV41_DECODE_LEAN", "1") == "1"
# Within LEAN: RoPE as one kernel. It must reproduce nvcc's FMA contraction of torch's complex
# multiply (decode_lean.ROPE_RE/IM), which tools/test_decode_lean.py searches for.
LEAN_ROPE = os.environ.get("DSV41_DECODE_LEAN_ROPE", "1") == "1"
# Within LEAN: the torch decode attention's softmax elementwise ops as two kernels around torch's sum.
LEAN_SOFTMAX = os.environ.get("DSV41_DECODE_LEAN_SOFTMAX", "1") == "1"
# Within LEAN: routed and shared experts share their two TP all-gathers (intermediates, then the
# summed outputs) instead of four, and hc_post reads the gathered output in place.
LEAN_MOE = os.environ.get("DSV41_DECODE_LEAN_MOE", "1") == "1"
# Within LEAN: the torch verify attention gathers its keys straight into fp32 (_attention_lean).
LEAN_ATTN = os.environ.get("DSV41_DECODE_LEAN_ATTN", "1") == "1"
# EXPERT_FORMAT=exl3: LEAN_MOE's two shared gathers for the EXL3 routed experts too (_moe_merged_exl3),
# over the RoCE one-shot path instead of moe_forward's two NCCL ring gathers. Exact (same kernels,
# same fp32 sum order); changes the collectives both ranks run, so the boot guard pins it.
EXL3_MERGED = os.environ.get("DSV41_EXL3_MERGED", "1") == "1"
# The drafter's Markov chain bias head, quantized (engine/fastdecode.py FastDecoder.__init__). The
# draft-only head format (DSV41_DRAFT_HEAD_FMT) covers the vocabulary projection; this one covers
# the five sequential GEMVs that chain the draft positions, which is a separate 66 MB bf16 read.
# Proposals only, so greedy output is unchanged -- but acceptance still moves, so it is opt-in.
MARKOV_HEAD_FMT = os.environ.get("DSV41_MARKOV_HEAD_FMT", "off").strip().lower() or "off"
if MARKOV_HEAD_FMT not in ("off", "bf16", "fp8", "fp4"):
    raise ValueError("DSV41_MARKOV_HEAD_FMT must be off, bf16, fp8 or fp4")


def merge_decode_projections(weights) -> int:
    """Attach `_wqkv_a` / `_sh_w13` to every block that can take them; returns the merge count.
    Idempotent. The originals are rebound to row views of the merged storage."""
    from fp8_linear import concat_rows
    merged = 0
    for w in weights:
        for first, second, attr in (("wq_a", "wkv", "_wqkv_a"), ("sh_w1", "sh_w3", "_sh_w13")):
            if getattr(w, attr, None) is not None:
                continue
            a, b = getattr(w, first, None), getattr(w, second, None)
            got = concat_rows(a, b) if a is not None and b is not None else None
            if got is None:
                continue
            both, (va, vb) = got
            setattr(w, first, va); setattr(w, second, vb); setattr(w, attr, both)
            merged += 1
    return merged


# The verify block: one accepted token plus DSV41_BLOCK drafted ones. The checkpoint's DSpark head
# was trained at `dspark_block_size` = 5 (so 6 verify positions), which stays the default; a larger
# block reads the same 40 layers of routed experts for more candidate tokens per step but drafts
# further outside the head's trained horizon, and a smaller one does the reverse. Even target
# widths remain the default. The opt-in odd widths preserve the compressor's unpaired token.
from engine.spec_depth import verify_odd_enabled
VERIFY_ODD = verify_odd_enabled()


def _compressor_group_cut(tokens: int, parity: int, ratio: int = 2) -> int:
    """Complete grouped positions, including a pending position at odd start parity.

    The remaining final position becomes pending; a width-one even-start
    forward makes no compressed row. All values are static for a graph key.
    """
    return (tokens + parity) // ratio * ratio


def _draft_block() -> int:
    v = os.environ.get("DSV41_BLOCK", "").strip()
    if v in ("", "off", "default"):
        return 5
    try:
        b = int(v)
    except ValueError:
        raise ValueError(f"DSV41_BLOCK: {v!r}; use an odd integer 1..15, or leave it unset") from None
    if not 1 <= b <= 15 or b % 2 == 0:
        raise ValueError(f"DSV41_BLOCK: {b}; must be odd and 1..15 so the verify block (b + 1) is even")
    return b


def _dynamic_depths():
    """DSV41_BLOCK_DYNAMIC="3,5": two draft depths the engine switches between per request as
    acceptance changes (engine/spec_depth.py). The drafter always drafts the deeper block (the
    checkpoint was trained at 5) and the verifier checks the first `depth` drafts, so each depth
    has its own verify graphs and nothing else changes. Greedy output is identical at any depth
    schedule -- verification accepts exactly the target model's argmax -- so switching moves
    speed, never text."""
    v = os.environ.get("DSV41_BLOCK_DYNAMIC", "").strip()
    if v in ("", "0", "off"):
        return None
    try:
        depths = tuple(sorted({int(x) for x in v.split(",")}))
    except ValueError:
        raise ValueError(f"DSV41_BLOCK_DYNAMIC: {v!r}; use two odd depths, e.g. 3,5") from None
    # T_VERIFY <= 10 is the graph-capturable routing limit (wider verify block)
    if len(depths) != 2 or any(d > 9 or d < (0 if VERIFY_ODD else 1)
                             or (not VERIFY_ODD and d % 2 == 0) for d in depths):
        allowed = "depths in 0..9" if VERIFY_ODD else "odd depths in 1..9"
        raise ValueError(f"DSV41_BLOCK_DYNAMIC: {v!r}; use two different {allowed}")
    fixed = os.environ.get("DSV41_BLOCK", "").strip()
    if fixed not in ("", "off", "default") and int(fixed) != depths[-1]:
        raise ValueError(f"DSV41_BLOCK={fixed} conflicts with DSV41_BLOCK_DYNAMIC={v}: the drafter "
                         f"drafts the deeper block ({depths[-1]}); unset DSV41_BLOCK")
    return depths


DYNAMIC_DEPTHS = _dynamic_depths()
from engine.spec_depth import confidence_depths
CONFIDENCE_DEPTHS = confidence_depths(DYNAMIC_DEPTHS)
# DSV41_SPEC_CONF=1: the draft graph also evaluates DSpark's confidence head (per-position
# conditional acceptance logits, tech report 2.4.3) into FastDecoder.d_conf, and the decode loop
# records it per step. The confidence depth policy also requires this head, but does not
# retain per-step logits unless observation was explicitly requested.
SPEC_CONF = os.environ.get("DSV41_SPEC_CONF", "0") == "1"
COMPUTE_CONF = SPEC_CONF or CONFIDENCE_DEPTHS is not None
# DSV41_TREE_PROBE=1: record, per draft position, the DSpark drafter's top-2 (d_top2) so the decode
# loop can log it beside the verifier's argmax. Observation only -- it answers one question before a
# tree is built: when the drafter's top-1 misses, how often is the target's token its #2? That rate,
# not the tree mechanism, decides whether a second candidate is worth a second draft chain.
TREE_PROBE = os.environ.get("DSV41_TREE_PROBE", "0") == "1"
# DSV41_DRAFT_NUCLEUS=1: sampled drafts come from the drafter's own top-p nucleus (taken within its
# top DSV41_DRAFT_NUCLEUS_K tokens) instead of its full softmax, and that truncated q is what
# verification divides by. The target is nucleus-truncated at the request's top_p
# (v41_engine.sample_probs), so an untruncated q proposes tail tokens the target gives p = 0 --
# always rejected -- and under-weights the nucleus. Rejection sampling with min(1, p/q) and the
# max(p - q, 0) residual is exact for ANY q it samples from, so this moves acceptance, never the
# output distribution. DeepInfer does the same (models/mtp/dflash.py process_sampling). Greedy
# drafts are unaffected.
DRAFT_NUCLEUS = os.environ.get("DSV41_DRAFT_NUCLEUS", "0") == "1"
DRAFT_NUCLEUS_K = int(os.environ.get("DSV41_DRAFT_NUCLEUS_K", "512"))
_DEPTHS = CONFIDENCE_DEPTHS or DYNAMIC_DEPTHS
T_DRAFT = max(5, _DEPTHS[-1]) if VERIFY_ODD and _DEPTHS else (_DEPTHS[-1] if _DEPTHS else _draft_block())
T_VERIFY = _DEPTHS[-1] + 1 if _DEPTHS else T_DRAFT + 1  # widest configured target block
# Every verify width this process captures graphs and static buffers for.
VERIFY_WIDTHS = tuple(d + 1 for d in _DEPTHS) if _DEPTHS else (T_VERIFY,)
if VERIFY_ODD:
    VERIFY_WIDTHS = tuple(sorted({1, *VERIFY_WIDTHS}))
# FastDecoder attributes that are sized by the verify width; _bind(T) points them at T's set.
_WIDTH_ATTRS = ("ids", "pos", "eg_rows", "slots", "h", "pre_mix", "attn_pre", "ffn_post", "ffn_comb",
                "ffn_pre", "y", "route_idx", "route_w", "hot_scores", "hot_logits", "topk", "candidates",
                "main_hidden", "logits", "candidate_ids",
                "_ar_t", "_premix0", "kvl_buf", "sc_buf")

# The graphed indexer needs a fixed key-axis shape, but that shape does not need to be the
# configured maximum context.  A small floor avoids capturing a new graph every few hundred
# tokens while powers of two keep the number of resident graph variants logarithmic.
INDEX_BUCKET_MIN = 4096


def _index_bucket(used_tokens: int, max_seq: int) -> int:
    """Smallest power-of-two token bucket covering ``used_tokens``, capped at ``max_seq``."""
    if not 0 < used_tokens <= max_seq:
        raise ValueError(f"index bucket needs 0 < used_tokens <= max_seq ({used_tokens}, {max_seq})")
    target = max(used_tokens, min(INDEX_BUCKET_MIN, max_seq))
    bucket = 1 << (target - 1).bit_length()
    return min(bucket, max_seq)


def _index_cache_rows(token_bucket: int, ratio: int, max_rows: int) -> int:
    """Cache-key rows needed by an uncompressed-token bucket at compression ``ratio``."""
    if token_bucket <= 0 or ratio <= 0 or max_rows <= 0:
        raise ValueError((token_bucket, ratio, max_rows))
    return min(max_rows, (token_bucket + ratio - 1) // ratio)


def _lin(x, w):  # bf16 tensor -> cuBLAS; FP8Weight/FP4Weight -> the Triton kernel for that stored format
    return R.dense(x, w)


class FastDecoder:
    def __init__(self, model: M.Model, engine, use_graphs: bool = True):
        self.m = model
        self.eng = engine
        self.a = model.args
        self.dev = model.dev
        self.W = model.W
        self.c = model.c
        self.use_graphs = use_graphs
        self.experiments = DecodeExperiments.from_env()
        if self.experiments.index_candidates and (
                self.a.candidate_source_layer < 0 or self.a.candidate_topk_blocks < 1
                or self.a.candidate_block_size < 1):
            raise ValueError("DSV41_INDEX_CANDIDATES requires hierarchical candidate blocks")
        # Runtime diagnostics are attached only between requests on both ranks.
        # Normal graphs contain no probe events or input snapshots.
        self.decode_probe = None
        self._probe_draft_key = None
        self.stream_layers = getattr(model, 'stream_layers', frozenset())
        # DSV41_HOT_READS (engine/hot_reads.py): every pruned MoE layer splits at its router, and
        # the heaviest missing pick is read into the transient ring before the layer's MoE runs.
        # DSV41_HOT_ASYNC: no split; the reads start after the step and are routable from the next.
        from engine.hot_reads import HotReads, HotRing
        self.hot, self.hot_layers, self._hot_live = None, frozenset(), None
        self.hot_async, self.hot_ring = False, None
        if HotReads.enabled():
            pm = getattr(model, "prune_mask", None) or {}
            self.hot = HotReads(self.a.n_layers, self.a.n_routed_experts, self.a.n_activated_experts)
            self.hot_layers = frozenset(L for L in pm if L not in self.stream_layers)
            if not self.hot_layers:
                raise ValueError("DSV41_HOT_READS needs pruned (resident-only) routing")
            self.hot.layers = sorted(self.hot_layers)
            n = 2 * self.hot.width
            self._hot_dev = torch.zeros(self.a.n_layers, n, dtype=torch.float32, device=self.dev)
            self._hot_host = torch.zeros(self.a.n_layers, n, dtype=torch.float32).pin_memory()
            self.hot_async = self.hot.asynchronous
            if self.hot_async:
                if os.environ.get("DSV41_MAX_CONCURRENCY", "1") != "1":
                    raise ValueError("DSV41_HOT_ASYNC runs with one decode lane (DSV41_MAX_CONCURRENCY=1)")
                self.hot_ring = HotRing(model.store.reserve_transient(self.hot.ring))
                self._hot_pin = torch.zeros(self.hot.ring, 3, dtype=torch.int64).pin_memory()
                self._hot_pin_np = self._hot_pin.numpy()
                self._hot_idx = torch.zeros(self.hot.ring, 3, dtype=torch.int64, device=self.dev)
                # The ring's live experts, every layer in one mask: a hot layer's router ORs its
                # row into the resident mask (graph-captured), so a step sets them with one fill
                # instead of a write a layer, and the resident masks are never touched.
                E = self.a.n_routed_experts
                self._hot_mask = torch.zeros(self.a.n_layers, E, dtype=torch.bool, device=self.dev)
                self._hot_keep = {L: torch.zeros(E, dtype=torch.bool, device=self.dev) for L in self.hot_layers}
                self._hot_side = torch.cuda.Stream(device=self.dev)
                self._hot_ev = torch.cuda.Event()
                self._hot_futs, self._hot_saved = [], None
        # Layers whose decode graph splits at the router, for a host step between the halves.
        self.split_layers = self.stream_layers | (frozenset() if self.hot_async else self.hot_layers)
        self.shared_stream = torch.cuda.Stream(device=self.dev) if SHARED_OVERLAP else None
        self.acct_stream = torch.cuda.Stream(device=self.dev) if PRUNE_MISS_SIDE else None
        self._acct_pending = False
        # DSV41_L2PF_MB: prefetch the next layer's weights during the MoE all-gather (engine/l2pf.py)
        self.l2pf_side = torch.cuda.Stream(device=self.dev) if l2pf.enabled() else None
        self.attn_prefetch_mb = int(os.environ.get("DSV41_L2PF_ATTN_MB", "0"))
        if not 0 <= self.attn_prefetch_mb <= 12:
            raise ValueError("DSV41_L2PF_ATTN_MB must be in 0..12")
        # Cold FP8 decode projections are latency-bound: wq_b 166 us cold vs 46 us from L2 on GB10
        # (8.8/20 MiB wq_a+wkv/wq_b, 11 MiB shared w13). DSV41_L2PF_QKV_MB prefetches this layer's
        # wq_a+wkv then wq_b while HC mixes run; DSV41_L2PF_SH_MB prefetches the shared expert while
        # the FFN HC mixes and router run. Loads only -- outputs are bit-identical.
        self.qkv_prefetch_mb = int(os.environ.get("DSV41_L2PF_QKV_MB", "0"))
        self.sh_prefetch_mb = int(os.environ.get("DSV41_L2PF_SH_MB", "0"))
        if not (0 <= self.qkv_prefetch_mb <= 24 and 0 <= self.sh_prefetch_mb <= 24):
            raise ValueError("DSV41_L2PF_QKV_MB and DSV41_L2PF_SH_MB must be in 0..24")
        self._pf_ev = {}
        # Before any capture: graphs bake in weight addresses. The MTP blocks are merged even when
        # the EXL3 draft matrices replace what the merge feeds: the merged FP8 tensors are then
        # dead (~100 MB a rank), but an A/B against `DSV41_EXL3_DRAFT_DENSE=0` must compare the
        # shipped FP8 spelling, not a slower unmerged one. Skipping the MTP merge once the EXL3
        # draft is kept is a separate change, worth ~100 MB of arena.
        self.merged_proj = (merge_decode_projections(list(self.W.layers) + list(self.W.mtp))
                            if MERGED_PROJ else 0)
        self.lean = None
        staged = os.environ.get("DSV41_ATTN_STAGED", "0")
        if staged not in ("0", "1", "2"):
            raise ValueError("DSV41_ATTN_STAGED must be 0 (off), 1 (FP32 keys), or 2 (BF16 keys)")
        self.staged_attention = int(staged)
        if self.staged_attention and FUSED_ATTN:
            raise ValueError("DSV41_ATTN_STAGED and DSV41_FUSED_ATTN are separate arithmetic paths")
        if LEAN and _HC_OPS and M.HC_FUSED:
            from engine.decode_lean import LeanOps
            self.lean = LeanOps(model.dev, R.MM_TILE, R.HC_MM_TILE)
            self.lean.prepare_router_weights([w.gate_w for w in self.W.layers])
            if self.experiments.draft_router:
                self.lean.prepare_router_weights([w.gate_w for w in self.W.mtp])
        if self.staged_attention and (self.lean is None or not LEAN_SOFTMAX):
            raise ValueError("DSV41_ATTN_STAGED requires decode lean and lean softmax")
        if self.experiments.draft_router and self.lean is None:
            raise ValueError("DSV41_DRAFT_ROUTER_FUSED requires decode lean")
        if self.experiments.hc_post_pre and self.lean is None:
            raise ValueError("DSV41_HC_POST_PRE requires decode lean")
        if self.experiments.attention_direct and (
                self.lean is None or not LEAN_ATTN or not LEAN_ROPE or not LEAN_SOFTMAX
                or FUSED_ATTN or not M.PACKED_KV):
            raise ValueError("DSV41_ATTN_DIRECT requires lean attention/RoPE/softmax and packed KV")
        if self.experiments.engram_h2d and (
                os.environ.get("DSV41_MAX_CONCURRENCY", "1") != "1"
                or os.environ.get("DSV41_ENGRAM_ROW_SPLIT", "0") != "0"):
            raise ValueError("DSV41_ENGRAM_ASYNC_H2D requires one lane and unsplit Engram reads")
        self.engram_upload_stream = (torch.cuda.Stream(device=self.dev)
                                     if self.experiments.engram_h2d else None)
        a = self.a
        dev = self.dev
        self.pend_buf = {L: torch.zeros(2, a.head_dim, dtype=torch.float32, device=dev)
                         for L in a.kv_source_layers if a.compress_ratios[L] > 1}
        # Width-sized static buffers, one set per verify width (graphs bake in their addresses).
        self._width_bufs = {T: self._alloc_width(T) for T in VERIFY_WIDTHS}
        self.width = None
        self._bind(T_VERIFY)
        # draft
        self.d_tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.d_last = torch.zeros(1, dtype=torch.long, device=dev)   # last main position
        self.d_noise = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)  # gumbel noise
        self.d_temp = torch.zeros(1, dtype=torch.float32, device=dev)
        self.d_top_p = torch.ones(1, dtype=torch.float32, device=dev)  # DRAFT_NUCLEUS only
        self.d_out = torch.zeros(T_DRAFT, dtype=torch.long, device=dev)
        self.d_probs = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)
        # (tok, last_main_pos) of a greedy draft already queued by draft_ahead(), or None
        self._ahead = None
        self.d_conf = torch.zeros(T_DRAFT, dtype=torch.float32, device=dev)   # raw logits, SPEC_CONF only
        self.d_top2 = torch.zeros(T_DRAFT, 2, dtype=torch.long, device=dev)   # drafter top-2, TREE_PROBE only
        # Mode-independent draft constants.  They used to be rebuilt by captured fill/arange
        # kernels every replay; only element zero of the ids changes with the accepted token.
        self._d_ids = torch.full((T_DRAFT,), a.dspark_noise_token_id, dtype=torch.long, device=dev)
        self._d_ar = torch.arange(T_DRAFT, device=dev)
        self._d_premix0 = torch.zeros(T_DRAFT, a.hc_mult, dtype=torch.float32, device=dev)
        self._d_premix0[:, 0] = 1.0
        # weights in decode-friendly dtypes (views/copies; small)
        # Preserve the configured head format.  In particular, DSV41_HEAD_FP32 is a numerical
        # choice and the verifier must not silently narrow it to bf16.
        self.head = self.W.head
        # The DSpark drafter reads its own, cheaper head when DSV41_DRAFT_HEAD_FMT built one. The
        # verifier (_final) always reads self.head, so its logits and the accepted tokens are
        # untouched; only the proposed drafts differ.
        self.draft_head = getattr(self.W, "draft_head", None) or self.head
        # Routers read the original FP32 gate_w below. Do not retain the unused BF16
        # copies: 40 backbone + 3 draft gates cost 153.75 MiB per rank without a consumer.
        self.markov_embed_bf16 = self.W.mtp[2].markov_embed.to(torch.bfloat16)
        # Proposal-only shortlist. The target still verifies every emitted token.
        # Kept opt-in: changing proposals can reduce acceptance even with a faster draft pass.
        self.draft_markov_topk = int(os.environ.get('DSV41_DRAFT_MARKOV_TOPK', '0'))
        if self.draft_markov_topk not in (0, 64, 128, 256, 512):
            raise ValueError('DSV41_DRAFT_MARKOV_TOPK must be 0, 64, 128, 256 or 512')
        # TensorFold/vLLM-style candidates: each rank's top-(K/world) of its own vocabulary shard,
        # gathered, instead of gathering the full fp32 logits and taking a global top-K. Greedy
        # drafting only; proposals change, the verified output does not.
        self.draft_markov_local = os.environ.get('DSV41_DRAFT_MARKOV_LOCAL', '0') == '1'
        # Exact-equivalent split of the full Markov chain (greedy): each rank scores its own
        # vocabulary half, so the five 66 MB Markov GEMVs read 33 MB each and the 2.6 MB logits
        # gather becomes five 8-byte gathers. Proposals match the full chain up to GEMV rounding.
        self.draft_markov_tp = os.environ.get('DSV41_DRAFT_MARKOV_TP', '0') == '1'
        self._markov_rank = int(os.environ.get('RANK', '0')) if hasattr(self.draft_head, 'world') else 0
        # The Markov chain's per-position bias is one GEMV against a [129280, 256] matrix, five
        # times a draft pass: 66 MB of bf16 even with DRAFT_MARKOV_TP halving it, and the same
        # weights every read. `_lin` already dispatches a quantized weight to its Triton kernel, so
        # the read shrinks by the format's ratio and nothing else changes. Proposals only: the
        # verifier still accepts exactly the target's argmax, so greedy output is unaffected.
        self.markov_head_bf16 = None
        self.markov_head_local = None
        self.set_markov_fmt(MARKOV_HEAD_FMT)
        self.win_off = torch.arange(a.window_size - 1, -1, -1, device=dev)
        self._index_cpos = torch.arange(self._n_cache(1), device=dev)
        self.graphs = OrderedDict()
        self.graph_limit = int(os.environ.get("DSV41_GRAPHS_MAX", "8"))
        if self.graph_limit < 0:
            raise ValueError("DSV41_GRAPHS_MAX must be >= 0 (0 = unbounded)")
        self.graph_evictions = 0
        self.graph_pool_resets = 0
        self.graph_captures = 0
        self.graph_memory = {}
        # graph key -> the per-step memo its capture filled (_step_memo); kept alive with the graphs
        self._memos = {}
        self._memo = None
        self.draft_graphs = None
        self.pool = None
        # cuBLAS retains a 32 MiB workspace per warm-up stream on this build.
        # Reuse one stream across context/width captures instead of leaking a new
        # workspace into its process-wide stream cache on every graph variant.
        self.capture_warmup_stream = torch.cuda.Stream(device=self.dev)
        # resident mode: (layer, expert) -> arena slot as a device table, so the router's expert ids can be
        # turned into slots inside the graph and the whole layer is ONE graph (no host round-trip per layer)
        self.lut = None
        self.lut_version = -1
        # DSV41_EXPERT_TOPP (engine/expert_topp.py): lossy decode pruning of low-share routed picks.
        # When configured, the slot table gets a sentinel column (expert id n_routed_experts) that a
        # dropped pick is rewritten to, so its weights are never read. `expert_topp` is the live rule
        # (a benchmark may set it to None to compare); the column exists whenever it is configured.
        self.expert_topp = _ETP.config()
        self.topp_sentinel = self.a.n_routed_experts if self.expert_topp is not None else None
        self.stats = {"steps": 0, "graph_s": 0.0, "resolve_s": 0.0, "engram_s": 0.0, "draft_s": 0.0}
        # DSV41_GPU_TIMING=1: CUDA events around each phase of a step. Host timers cannot decompose
        # this path -- every one of them measures the host waiting for, or launching into, an async
        # queue, which is why removing route_s, the host resolve and the pageable engram copy each
        # collapsed its own timer and moved the wall clock not at all. Events sit ON the GPU
        # timeline and measure the work itself. Pairs are recorded per step and only read at the
        # end of the request, so no step pays a synchronise for being measured.
        self.gpu_timing = os.environ.get("DSV41_GPU_TIMING", "0") == "1"
        self._ev_pairs: dict[str, list] = {
            "draft": [], "layers": [], "segments": [], "hash": [], "sample": [],
            "engram_ready_wait": [],
        }
        # DSV41_ROUTE_STATS=1 counts, per backbone layer, how many DISTINCT routed experts the
        # T_VERIFY tokens of a verify block ask for -- the quantity that sets the expert bytes a step
        # has to read, since one expert is read once however many of the block's tokens route to it.
        # Both ops have static shapes and no host round-trip, so they capture into the layer graphs
        # like everything else; with the flag off nothing is allocated and `_layer_a` runs one
        # `if self.rs_hits is not None` per layer.
        if M.PRUNE_MISS:
            # must exist, and keep their addresses, before any graph capture records into them
            self.m.alloc_prune_miss(a.n_routed_experts, dev)
        self.rs_hits = self.rs_uniq = None
        if os.environ.get("DSV41_ROUTE_STATS", "0") == "1":
            self.rs_hits = torch.zeros(a.n_routed_experts, dtype=torch.int32, device=dev)
            self.rs_uniq = torch.zeros(self.m.args.n_layers, dtype=torch.float64, device=dev)
            self.rs_steps = 0

    # ------------------------------------------------------------------ verify widths
    def _alloc_width(self, T: int) -> dict:
        """The static buffers whose first dimension is the verify width T."""
        a, dev = self.a, self.dev
        f32, bf16 = torch.float32, torch.bfloat16
        b = dict(
            # inputs of the graphs
            ids=torch.zeros(T, dtype=torch.long, device=dev),
            pos=torch.zeros(T, dtype=torch.long, device=dev),
            eg_rows={L: torch.zeros(T, a.engram_n_heads * (a.engram_max_ngram_size - 1), a.engram_head_dim,
                                    dtype=f32, device=dev) for L in a.engram_layer_ids},
            slots=torch.zeros(T, a.n_activated_experts, dtype=torch.int32, device=dev),
            # state carried between graphs of one step
            h=torch.zeros(T, a.hc_mult, a.dim, dtype=bf16, device=dev),
            pre_mix=torch.zeros(T, a.hc_mult, dtype=f32, device=dev),
            attn_pre=torch.zeros(T, a.hc_mult, dtype=f32, device=dev),
            ffn_post=torch.zeros(T, a.hc_mult, dtype=f32, device=dev),
            ffn_comb=torch.zeros(T, a.hc_mult, a.hc_mult, dtype=f32, device=dev),
            ffn_pre=torch.zeros(T, a.hc_mult, dtype=f32, device=dev),
            y=torch.zeros(T, a.dim, dtype=bf16, device=dev),
            route_idx=torch.zeros(T, a.n_activated_experts, dtype=torch.long, device=dev),
            route_w=torch.zeros(T, a.n_activated_experts, dtype=f32, device=dev),
            # DSV41_HOT_READS: a hot layer's pre-mask router outputs, for graph B to route again
            hot_scores=torch.zeros(T, a.n_routed_experts, dtype=f32, device=dev),
            hot_logits=torch.zeros(T, a.n_routed_experts, dtype=f32, device=dev),
            topk=torch.full((T, a.index_topk), -1, dtype=torch.long, device=dev),
            candidates=torch.zeros(T, self._n_cache(1), dtype=torch.bool, device=dev),
            candidate_ids=torch.full((T, a.candidate_topk_blocks * a.candidate_block_size
                                     if self.experiments.index_candidates else 0),
                                    -1, dtype=torch.long, device=dev),
            main_hidden=torch.zeros(T, a.dim * len(a.dspark_target_layer_ids), dtype=f32, device=dev),
            logits=torch.zeros(T, a.vocab_size, dtype=f32, device=dev),
            # constants the lean step reuses instead of rebuilding them per step
            _ar_t=torch.arange(T, device=dev),
            _premix0=torch.zeros(T, a.hc_mult, dtype=f32, device=dev),
            # per-step compressor inputs, kept for Caches.rollback
            kvl_buf={L: torch.zeros(T, a.head_dim, dtype=f32, device=dev) for L in self.pend_buf},
            sc_buf={L: torch.zeros(T, a.head_dim, dtype=f32, device=dev) for L in self.pend_buf},
        )
        b["_premix0"][:, 0] = 1.0
        assert set(b) == set(_WIDTH_ATTRS)
        return b

    def _bind(self, T: int):
        """Point the width-sized attributes at T's buffers. Captures and replays of width T both
        run bound to T, so a graph always reads and writes the storage it was captured with."""
        if self.width == T:
            return
        for name, value in self._width_bufs[T].items():
            setattr(self, name, value)
        self.width = T

    # ------------------------------------------------------------------ helpers
    def _n_cache(self, r):
        # rows of the shared compressed cache for ratio r (Caches allocates max_seq // r + 1)
        return self.c.max_seq // r + 1

    def _rope(self, x, fq, inverse=False):
        rd = self.a.rope_head_dim
        if (self.lean is not None and LEAN_ROPE and x.dtype == torch.bfloat16 and fq.dtype == torch.complex64
                and x.dim() in (2, 3) and fq.dim() == 2 and fq.shape[0] == x.shape[0]):
            return self.lean.rope(x, fq, rd, inverse)
        return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq, inverse=inverse)], dim=-1)

    def _hc_mixes(self, x, hc_fn, hc_scale, hc_base, out=None):
        # This projection is numerically load-bearing: an M=6 cuBLAS call does not reduce K in
        # the same order as Model._hc_mixes' fixed padded call. The resulting ~1e-4 change in
        # the 24 HC coefficients is first visible in the post-attention residual and grows until
        # router top-k choices differ.  Calling the model implementation also keeps the RMS
        # reduction and Sinkhorn path identical instead of maintaining a second approximation.
        # The lean spelling runs that same padded GEMM and padded reduction (engine/decode_lean.py).
        # `out=(pre, post, comb)`: write the coefficients into these buffers.
        if self.lean is not None and self.lean.usable_hc(x, hc_fn):
            a = self.a
            return self.lean.hc_mixes(x, hc_fn, hc_scale, hc_base, a.norm_eps, a.hc_mult,
                                      a.hc_sinkhorn_iters, a.hc_eps, out=out)
        res = self.m._hc_mixes(x, hc_fn, hc_scale, hc_base)
        if out is None:
            return res
        for dst, src in zip(out, res):
            dst.copy_(src)
        return out

    def _rmsnorm(self, x, w, eps):
        if self.lean is not None and self.lean.usable_rms(x):
            return self.lean.rmsnorm(x, w, eps)
        return R.rmsnorm(x, w, eps)

    def _norm_w(self, w):
        """The fp32 copy of a norm weight for the fused HC kernels, which would otherwise convert
        the bf16 weight on every call (`w.contiguous().float()` inside hc_ops)."""
        return self.lean._wf32(w) if self.lean is not None else w

    def _step_memo(self, key, fn):
        """Per-step values that depend only on the step's positions (and the indexer's topk), made
        once per verify graph instead of once per layer. `self._memo` is a dict owned by the graph
        key being captured and kept for as long as its graphs live: later segments' graphs read the
        tensors the first segment's capture allocated. None (lanes, eager steps) = no memo."""
        memo = getattr(self, "_memo", None)
        if memo is None:
            return fn()
        v = memo.get(key)
        if v is None:
            v = memo[key] = fn()
        return v

    def _hc_pre_rn(self, h, pre_mix, w, out=None):
        if self.lean is not None:
            if out is None:
                out = torch.empty(h.size(0), h.size(2), dtype=torch.bfloat16, device=h.device)
            return _hc_pre_rn_fused(h, pre_mix, self._norm_w(w), self.a.norm_eps, out=out)
        y = _hc_pre_rn_fused(h, pre_mix, w, self.a.norm_eps)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def _shared_ffn(self, y, w):
        x3 = getattr(w, "_x3sh", None)
        if x3 is not None:
            # EXL3 shared expert (engine/exl3_draft.py). w13 is two groups over one input, so the
            # input is duplicated on the group axis; the output is [T, 2N] with w1's columns first,
            # the order `expert_ffn` slices at `sh_w1.N`. The tail is `expert_ffn`'s, with the
            # elementwise middle as one kernel when the lean ops are available.
            mh = y if y.dtype == torch.bfloat16 else y.to(torch.bfloat16)
            gu = x3.w13(mh[:, None, :].expand(-1, 2, -1))
            if self.lean is not None:
                return x3.w2(self.lean.swiglu(gu, x3.n, self.a.swiglu_limit))
            gate, up = gu[..., :x3.n].float(), gu[..., x3.n:].float()
            limit = self.a.swiglu_limit
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate = torch.clamp(gate, max=limit)
            return x3.w2((F.silu(gate) * up).to(torch.bfloat16))
        w13 = getattr(w, "_sh_w13", None) if MERGED_PROJ else None
        if self.lean is not None and w13 is not None and y.dtype == torch.bfloat16:
            # v41_ref.expert_ffn with its elementwise middle as one kernel (engine/decode_lean.py)
            h = self.lean.swiglu(R.qlinear(y, w13), w.sh_w1.N, self.a.swiglu_limit)
            return R.qlinear(h, w.sh_w2)
        return R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, self.a.swiglu_limit, w13=w13)

    # ------------------------------------------------------------------ attention (decode, T tokens)
    def _attention(self, x, w, L, ring, freqs, pos, sh_state, mtp_last=None):
        a = self.a
        T = x.size(0)
        lean_step = self.lean is not None and mtp_last is None
        fq = (self._step_memo(("fq", freqs.data_ptr()), lambda: freqs[pos]) if lean_step
              else freqs[pos])
        both = getattr(w, "_wqkv_a", None) if MERGED_PROJ else None
        # `_x3` is set by engine/exl3_dense.py on backbone layers and by engine/exl3_draft.py on the
        # draft blocks; the projection path below is the same for both, `mtp_last` only chooses how
        # the keys are gathered. The old `if mtp_last is None` guard predates the draft pack.
        x3 = getattr(w, "_x3", None)
        if x3 is not None:
            # EXL3 decode matrices (engine/exl3_dense.py): two GEMVs, each with its own input rotation
            self._pf_wait("qkv")
            q_lin, kv_lin = x3.wq_a(x), x3.wkv(x)
        elif both is not None:
            self._pf_wait("qkv")
            qkv = R.qlinear(x, both)
            # contiguous: the norms' reductions then see exactly the tensors the split path makes
            q_lin, kv_lin = qkv[:, :w.wq_a.N], qkv[:, w.wq_a.N:]
            if self.lean is None or not self.lean.usable_rms(q_lin):
                q_lin, kv_lin = q_lin.contiguous(), kv_lin.contiguous()
        else:
            q_lin, kv_lin = R.qlinear(x, w.wq_a), R.qlinear(x, w.wkv)
        qr = self._rmsnorm(q_lin, w.q_norm, a.norm_eps)
        if (lean_step and LEAN_ATTN and not FUSED_ATTN and LEAN_ROPE and LEAN_SOFTMAX and ring.is_contiguous()
                and fq.dtype == torch.complex64):
            return self._attention_lean(x, qr, kv_lin, w, L, ring, fq, pos, sh_state)
        self._pf_wait("qkv")
        self._pf_wait("wq_b")
        q_b = x3.wq_b(qr) if x3 is not None else R.qlinear(qr, w.wq_b)
        q = self._rope(q_b.view(T, getattr(w, 'tp_heads', a.n_heads), a.head_dim), fq)
        kv = self._rope(self._rmsnorm(kv_lin, w.kv_norm, a.norm_eps), fq)
        # The key set is two pieces: the window rows and (verify) the CSA2 rows / (draft) the draft
        # keys. The fused kernel takes both as base pointers, so they are never cat'ed; only the
        # torch fallback materialises kv_all. The draft's window is a stride-0 broadcast view.
        if mtp_last is None and lean_step:
            slot_w, slot_r, wmask = self._step_memo("win", lambda: self._window(pos))
            ring[slot_w] = kv
            kv1 = ring[slot_r]
            kv2 = None
            mask = wmask
            if w.ratio:
                kv2, mask = self._compressed(x, qr, w, L, pos, sh_state, wmask=wmask)
        elif mtp_last is None:
            wpos = pos[:, None] - self.win_off[None, :]
            ring[pos % M.RING] = kv
            kv1 = ring[wpos.clamp_min(0) % M.RING]
            kv2 = None
            mask = wpos >= 0
            if w.ratio:
                kv2, cmask = self._compressed(x, qr, w, L, pos, sh_state)
                mask = torch.cat([mask, cmask], dim=1)
        else:
            wpos = mtp_last - self.win_off  # [128]
            kv1 = ring[wpos.clamp_min(0) % M.RING][None].expand(T, -1, -1)
            kv2 = kv[None].expand(T, -1, -1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1),
                              torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)
        scale = a.head_dim ** -0.5
        if FUSED_ATTN:
            o = decode_attention(q, kv1, kv2, mask, w.attn_sink, scale)
        else:
            if self.lean is not None:
                # the fp32 keys once, written piecewise: the same tensor kv_all.float() makes, which
                # the torch spelling below builds twice (one per einsum) after a bf16 cat
                n1 = kv1.shape[1]
                kvf = torch.empty(T, n1 + (0 if kv2 is None else kv2.shape[1]), kv1.shape[2],
                                  dtype=torch.bfloat16 if self.staged_attention == 2 else torch.float32,
                                  device=kv1.device)
                kvf[:, :n1].copy_(kv1)
                if kv2 is not None:
                    kvf[:, n1:].copy_(kv2)
                kvs = (kvf, kvf)
            else:
                kv_all = kv1 if kv2 is None else torch.cat([kv1, kv2], dim=1)
                kvs = (kv_all.float(), None)
            if self.lean is not None and LEAN_SOFTMAX:
                if self.staged_attention:
                    from decode_attn_staged import scores
                    logits = scores(q, kvs[0])
                else:
                    logits = (self._verify_attention_product("thd,tnd->thn", q.float(), kvs[0]) if mtp_last is None
                              else torch.einsum("thd,tnd->thn", q.float(), kvs[0]))
                probs = self.lean.attn_probs(logits,
                                             mask, w.attn_sink, scale)
            else:
                scores = (self._verify_attention_product("thd,tnd->thn", q.float(), kvs[0]) if mtp_last is None
                          else torch.einsum("thd,tnd->thn", q.float(), kvs[0])) * scale
                scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
                mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
                p = torch.exp(scores - mx)
                denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - mx)
                probs = p / denom
            if self.staged_attention and self.lean is not None and LEAN_SOFTMAX:
                from decode_attn_staged import values
                o = values(probs, kvs[1]).to(torch.bfloat16)
            else:
                keys = kvs[1] if kvs[1] is not None else kv_all.float()
                o = (self._verify_attention_product("thn,tnd->thd", probs, keys) if mtp_last is None
                     else torch.einsum("thn,tnd->thd", probs, keys)).to(torch.bfloat16)
        o = self._rope(o, fq, inverse=True).reshape(T, getattr(w, 'tp_groups', a.o_groups), -1)
        if x3 is not None:
            return x3.wo_b(x3.wo_a(o))
        o = R.wo_a_proj(o, w.wo_a, tiled=True)
        return R.qlinear(o.flatten(1), w.wo_b)

    def _attention_lean(self, x, qr, kv_lin, w, L, ring, fq, pos, sh_state):
        """The verify branch of _attention's torch path with its fp32 staging written in place:
        RoPE emits q already widened (`.float()` of the bf16 result), the window and compressed
        keys are gathered straight into the fp32 key block, and the output RoPE rounds the einsum
        result to bf16 itself. Same values at every step; 4 launches fewer per layer."""
        a = self.a
        T = x.size(0)
        self._pf_wait("qkv")
        self._pf_wait("wq_b")
        x3 = getattr(w, "_x3", None)
        q_b = x3.wq_b(qr) if x3 is not None else R.qlinear(qr, w.wq_b)
        o = self._attn_core_lean(x, qr, q_b, kv_lin, w, L, ring, fq, pos, sh_state,
                                 prefetch=bool(self.attn_prefetch_mb and self.l2pf_side is not None))
        if x3 is not None:
            return x3.wo_b(x3.wo_a(o.reshape(T, getattr(w, 'tp_groups', a.o_groups), -1)))
        o = R.wo_a_proj(o.reshape(T, getattr(w, 'tp_groups', a.o_groups), -1), w.wo_a, tiled=True)
        return R.qlinear(o.flatten(1), w.wo_b)

    def _attn_core_lean(self, x, qr, q_b, kv_lin, w, L, ring, fq, pos, sh_state, prefetch=False):
        """The part of _attention_lean that reads and writes this sequence's caches: RoPE, the
        window ring, the compressor/indexer, the key block and the attention itself, from the
        projections `q_b` and `kv_lin` to the inverse-RoPE'd output [T, heads, head_dim] bf16.
        Nothing here reads a large weight except the compressor/indexer's own; batched decode
        (engine/batch_decode.py) runs this per sequence between projections it shares."""
        a = self.a
        T = x.size(0)
        rd = a.rope_head_dim
        x3 = getattr(w, "_x3", None)
        qf = self.lean.rope(q_b.view(T, getattr(w, 'tp_heads', a.n_heads), a.head_dim),
                            fq, rd, out_f32=True)
        kv = self.lean.rope(self._rmsnorm(kv_lin, w.kv_norm, a.norm_eps), fq, rd)
        slot_w, slot_r, wmask = self._step_memo("win", lambda: self._window(pos))
        ring[slot_w] = kv
        cache = idx = None
        mask = wmask
        if w.ratio:
            (cache, idx), mask = self._compressed(x, qr, w, L, pos, sh_state, wmask=wmask, rows=False)
        # This window is attention arithmetic, longer than the RoCE-only site.
        # Prefetch only this layer's output projection; next-layer weights would
        # be evicted by MoE before use. The graph joins before consuming weights.
        if prefetch:
            self.l2pf_side.wait_stream(torch.cuda.current_stream())
            l2pf.touch(self.l2pf_side, [x3.wo_a, x3.wo_b] if x3 is not None else [w.wo_a, w.wo_b],
                        self.attn_prefetch_mb * 1024**2, mode="bulk", pace=0)
        if self.experiments.attention_direct:
            # Reuse each unpacked tile for QK and PV. The old direct tile path
            # re-unpacked it for every PV column tile and lost 236 vs 47.7 us.
            # Online softmax changes reduction order: qualify as numerics.
            from engine.attention_packed import attention_packed
            raw = attention_packed(qf, ring, slot_r, cache, idx, mask,
                                   w.attn_sink, a.head_dim ** -0.5)
        elif self.staged_attention:
            kvf = self.lean.keys_f32(ring, slot_r, cache, idx,
                                   dtype=torch.bfloat16 if self.staged_attention == 2 else torch.float32)
            from decode_attn_staged import attention
            raw = attention(qf, kvf, mask, w.attn_sink, a.head_dim ** -0.5, self.lean)
        else:
            kvf = self.lean.keys_f32(ring, slot_r, cache, idx)
            probs = self.lean.attn_probs(self._verify_attention_product("thd,tnd->thn", qf, kvf), mask, w.attn_sink,
                                         a.head_dim ** -0.5)
            raw = self._verify_attention_product("thn,tnd->thd", probs, kvf)
        o = self.lean.rope(raw, fq, rd, inverse=True)
        if prefetch:
            torch.cuda.current_stream().wait_stream(self.l2pf_side)
        return o

    def _verify_attention_product(self, equation, left, right):
        # CUDA's batch-1 SGEMM accumulates differently from batch>=2 BMM. In
        # results/decode-followup-20261007/bmm-rows.json, broadcast batch-2
        # QK/PV matched T6 bitwise at N=128/1152/3200; native T1 flipped BF16
        # output elements. Stride-0 expand keeps the same keys without a copy.
        # Opt-in verify only: the existing single-row drafter stays unchanged.
        if VERIFY_ODD and left.size(0) == 1:
            return torch.einsum(equation, left.expand(2, -1, -1), right.expand(2, -1, -1))[:1]
        return torch.einsum(equation, left, right)

    def _window(self, pos):
        """(ring slot of each query, ring slots of its window, window mask) -- the same for every
        backbone layer of a step."""
        wpos = pos[:, None] - self.win_off[None, :]
        return pos % M.RING, wpos.clamp_min(0) % M.RING, wpos >= 0

    def _compressed(self, x, qr, w, L, pos, st, wmask=None, rows: bool = True):
        """st: dict with 'parity' (python int, S % 2, fixed per graph), 'ckv', 'ik', 'ratio'.

        Returns (rows, cmask), or with `wmask` (the lean step) (rows, cat([wmask, cmask])): the
        concatenated mask and the clamped indices only change when an indexer layer rewrites
        self.topk, so they are made once per such layer and reused by the layers after it."""
        a = self.a
        r = w.ratio
        c = self.c
        T = x.size(0)
        if w.is_kv_source:
            if r > 1:
                kvl, sc = R.mm(x.float(), w.comp_wkv), R.mm(x.float(), w.comp_wgate)
                # static per-layer copies for Caches.rollback (host patches the tuple after the step)
                self.kvl_buf[L].copy_(kvl); self.sc_buf[L].copy_(sc)
                buf = self.pend_buf[L]  # [2, 512]: kv, score of the pending token (host-maintained)
                if st["parity"] == 1:  # Include the pending position before this block.
                    kvl2 = torch.cat([buf[0][None], kvl]); sc2 = torch.cat([buf[1][None], sc])
                else:
                    kvl2, sc2 = kvl, sc
                cut = _compressor_group_cut(T, st["parity"], r)
                if (T + st["parity"]) % r:
                    buf[0].copy_(kvl[T - 1]); buf[1].copy_(sc[T - 1])
                if cut:
                    g_kv = kvl2[:cut].unflatten(0, (-1, r)); g_sc = sc2[:cut].unflatten(0, (-1, r))
                    latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                    latent = self._rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                else:
                    # Same as Model._compressed: no complete pair yet. Leave
                    # compressed/index history intact and keep this token pending.
                    latent = None
                j0 = (pos[0] - st["parity"]) // r
            else:
                latent = self._rmsnorm(R.mm(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = pos[0]
            if latent is not None:
                nj = latent.size(0)
                jidx = j0 + self._ar_t[:nj]
                fj = self.m.freqs_c[jidx * r]
                if L in self.W.indexers:
                    iw = self.W.indexers[L]
                    k = self._rmsnorm(R.mm(latent, iw.wk), iw.k_norm, a.norm_eps)
                    c.ik[L][jidx] = self._rope(k, fj)
                ckv = self._rope(latent, fj)
                if M.PACKED_KV:
                    from engine.packed_kv import write
                    write(c.ckv[L], ckv.contiguous(), jidx)
                else:
                    if M.KV_CACHE_QDQ:
                        ckv = R.fp4_qdq(ckv, 16, "e4m3", self.m.fp4_grid)
                    c.ckv[L][jidx] = ckv
            st["ckv"], st["ik"], st["ratio"] = c.ckv[L], c.ik[L], r
        from engine.packed_kv import gather
        if wmask is not None:
            if L in self.W.indexers:
                compress_lens = self._step_memo(("clen", r), lambda: (pos + 1) // r)
                self.topk.copy_(self._indexer(x, qr, L, pos, compress_lens, st))
                if getattr(self, "_memo", None) is not None:
                    self._memo.pop("topk", None)
            idx_c, mask = self._step_memo("topk", lambda: (
                self.topk.clamp_min(0), torch.cat([wmask, self.topk >= 0], dim=1)))
            if not rows:
                return (st["ckv"], idx_c), mask  # the caller gathers straight into fp32
            return gather(st["ckv"], idx_c), mask
        compress_lens = (pos + 1) // r
        if L in self.W.indexers:
            self.topk.copy_(self._indexer(x, qr, L, pos, compress_lens, st))
        idx = self.topk
        rows = gather(st["ckv"], idx.clamp_min(0))
        return rows, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, st):
        a = self.a
        iw = self.W.indexers[L]
        T = x.size(0)
        fq = self._step_memo(("fq", self.m.freqs_c.data_ptr()), lambda: self.m.freqs_c[pos])
        q = self._rope(R.qlinear(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim), fq)
        wts = R.mm(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5)
        # CUDA graphs require a fixed N, but scoring all max_seq rows makes every request pay for
        # the configured context capacity.  ``index_bucket`` is fixed for one graph variant and
        # expressed in uncompressed token positions; convert it to this layer's cache rows.  The
        # exact per-query visibility mask below is still authoritative, so bucket padding cannot
        # affect which rows are selected.
        token_bucket = st.get("index_bucket", self.c.max_seq)
        ratio = st["ratio"]
        cache_rows = _index_cache_rows(token_bucket, ratio, st["ik"].size(0))
        k = st["ik"][:cache_rows]
        compact = getattr(self, "experiments", DecodeExperiments()).index_candidates
        capacity = a.candidate_topk_blocks * a.candidate_block_size if compact else 0
        if (compact and 0 <= a.candidate_source_layer < L and cache_rows > capacity):
            from engine.index_candidates import scores, select
            return select(scores(q, k, wts, self.candidate_ids, compress_lens),
                          self.candidate_ids, a.index_topk)
        sc = torch.einsum("thd,nd->thn", q, k)
        score = (sc.float().relu_() * wts[:, :, None]).sum(dim=1)  # [T, N]
        cpos = self._index_cpos[:score.size(1)]
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        if L == a.candidate_source_layer:
            selected = M.Model._select_candidates(score, compress_lens, a.candidate_topk_blocks,
                                                   a.candidate_block_size)
            self.candidates[:, :score.size(1)].copy_(selected)
            if compact and cache_rows > capacity:
                from engine.index_candidates import compact_mask
                self.candidate_ids.copy_(compact_mask(selected, a.candidate_block_size, capacity))
        elif 0 <= a.candidate_source_layer < L:
            score = score.masked_fill(~self.candidates[:, :score.size(1)], float("-inf"))
        # Small context allocations can have fewer cache rows than the serving top-k.
        # Preserve the fixed output width expected by attention and captured graphs.
        idx = score.topk(min(a.index_topk, score.size(1)), dim=-1, sorted=False).indices.sort(dim=-1).values
        idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
        pad = a.index_topk - idx.size(1)
        return F.pad(idx, (0, pad), value=-1) if pad else idx

    # ------------------------------------------------------------------ layer graphs
    def _tap(self, name, L, t):
        f = getattr(self, 'tap', None)
        if f is not None:
            f(name, L, t)

    def _prefetch(self, groups):
        """Touch each (key, tensors, budget_bytes) group on the L2 side stream in order, recording
        one event per group for its consumer to wait on."""
        side = self.l2pf_side
        side.wait_stream(torch.cuda.current_stream())
        for key, tensors, budget in groups:
            if budget > 0 and l2pf.touch(side, tensors, budget):
                ev = torch.cuda.Event()
                ev.record(side)
                self._pf_ev[key] = ev

    def _pf_wait(self, key, stream=None):
        ev = self._pf_ev.pop(key, None)
        if ev is not None:
            (stream or torch.cuda.current_stream()).wait_event(ev)

    def _pf_join(self):
        """Wait for every outstanding prefetch: graph capture must rejoin the side stream."""
        for key in list(self._pf_ev):
            self._pf_wait(key)

    def set_markov_fmt(self, fmt: str):
        """(Re)build the Markov bias head in `fmt` from the bf16 source on `self.W.mtp[2]`.

        `off`/`bf16` share the source tensor. The two shards are quantized from their own rows
        rather than sliced out of a quantized whole: a block scale straddling the vocabulary split
        would be wrong, and the ranks score their halves independently anyway.
        """
        if fmt not in ('off', 'bf16', 'fp8', 'fp4'):
            raise ValueError(f"markov head format {fmt!r} must be off, bf16, fp8 or fp4")
        self.markov_fmt = fmt
        block = self.W.mtp[2]
        src = block.markov_head
        # Every derived head is a pure function of the shared weights, so it is made once per
        # process and kept on the weights: a concurrent lane is one more decoder over them.
        derived = getattr(block, '_markov_derived', None)
        if derived is None:
            derived = {}
            setattr(block, '_markov_derived', derived)

        def once(key, make):
            if key not in derived:
                derived[key] = make()
            return derived[key]

        self.markov_head_bf16 = (src if fmt in ('off', 'bf16')
                                 else once(('head', fmt), lambda: R._head_in_format(src, fmt)))
        listed = getattr(self.draft_head, 'ids_local', None)
        if listed is not None:
            # DSV41_DRAFT_VOCAB: the split chain scores the listed head's own columns, so its Markov
            # rows are those ids, gathered once into contiguous storage (21 MB of bf16 at 40,960 a
            # rank, under the 24 MiB L2 -- whether the five back-to-back reads then hit it is a
            # trace question, not an assumption).
            local = once(('listed', listed.data_ptr()), lambda: src.index_select(0, listed.to(torch.int64)))
            self.markov_head_local = (local if fmt in ('off', 'bf16') else
                                      once(('listed', listed.data_ptr(), fmt), lambda: R._head_in_format(local, fmt)))
        elif self.markov_head_local is not None or hasattr(self.draft_head, 'world'):
            rows = self.draft_head.local.shape[0]
            # off/bf16 keep the strided row view the drafter always passed cuBLAS; only a quantized
            # copy needs its own contiguous storage.
            local = src[self._markov_rank * rows:(self._markov_rank + 1) * rows]
            self.markov_head_local = (local if fmt in ('off', 'bf16') else
                                      once(('shard', self._markov_rank, rows, fmt),
                                           lambda: R._head_in_format(local.contiguous(), fmt)))

    def _layer_a(self, L, sh_state, defer_acct: bool = False):
        """attention + HC + router for backbone layer L, reading self.h/self.pre_mix/self.pos.
        `defer_acct`: the caller joins the prune-miss side stream itself (_layer_ab, after the MoE)."""
        a = self.a
        w = self.W.layers[L]
        h = self.h
        self._tap('h_in', L, h)
        if self.qkv_prefetch_mb and self.l2pf_side is not None and self.lean is not None:
            mb = self.qkv_prefetch_mb * 1024 ** 2
            x3 = getattr(w, "_x3", None)
            if x3 is not None:
                qkv, wq_b = [x3.wq_a, x3.wkv], x3.wq_b
            else:
                qkv = [w._wqkv_a] if getattr(w, "_wqkv_a", None) is not None else [w.wq_a, w.wkv]
                wq_b = w.wq_b
            first = min(mb, sum(l2pf.raw(t).numel() * l2pf.raw(t).element_size() for t in qkv))
            self._prefetch([("qkv", qkv, first), ("wq_b", [wq_b], mb - first)])
        if L in self.W.engram:
            h = R.engram_forward(h, self.eg_rows[L], self.W.engram[L], a)
        if L in a.dspark_target_layer_ids:
            i = list(a.dspark_target_layer_ids).index(L)
            self.main_hidden[:, i * a.dim:(i + 1) * a.dim] = h.float().mean(dim=1)
        residual = h
        attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        # Same fused stream ops the prefill path uses (engine/hc_ops.py). At T=6 the tensors are
        # small, so the win here is launch count rather than bandwidth: these replace ~10 tiny
        # elementwise kernels per site with one, and a graphed step issues ~1,015 such launches.
        y = (self._hc_pre_rn(h, self.pre_mix, w.attn_norm) if _HC_OPS
             else R.rmsnorm(R.hc_pre(h, self.pre_mix), w.attn_norm, a.norm_eps))
        self._tap('attn_x', L, y)
        y = self._attention(y, w, L, self.c.win[L], self.m.freqs_c if w.ratio else self.m.freqs_w, self.pos, sh_state)
        self._tap('attn_out', L, y)
        if self.sh_prefetch_mb and self.l2pf_side is not None and getattr(w, "_sh_w13", None) is not None:
            x3d = getattr(self.eng, "exl3_dense", None) if getattr(w, "_x3", None) is not None else None
            targets = x3d.shared_targets(L) if x3d is not None else [w._sh_w13, w.sh_w2]
            self._prefetch([("shared", targets, self.sh_prefetch_mb * 1024 ** 2)])
        fused_pre = None
        if self.experiments.hc_post_pre:
            from engine.hc_post_pre import post_pre
            h, fused_pre = post_pre(y, residual, attn_post, attn_comb, attn_pre,
                self._norm_w(w.ffn_norm), a.norm_eps, out=self.h, normalized=self.y)
        elif self.lean is not None:
            # written in place: hc_post reads a tile's residual streams before it stores the tile
            h = _hc_post_fused(y, residual, attn_post, attn_comb, out=self.h)
        else:
            h = (_hc_post_fused(y, residual, attn_post, attn_comb) if _HC_OPS
                 else R.hc_post(y, residual, attn_post, attn_comb))
            self.h.copy_(h)
        self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base,
                       out=(self.ffn_pre, self.ffn_post, self.ffn_comb))
        if fused_pre is not None:
            y = fused_pre
        elif _HC_OPS:
            y = self._hc_pre_rn(h, attn_pre, w.ffn_norm, out=self.y)
        else:
            y = R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps)
            self.y.copy_(y)
        # fp32, exactly as Model.moe does it. The gate picks 6 of 384 experts and its scores are
        # full of near-ties, so a bf16 GEMM here (which this path used until 2026-09-11) changes
        # 11 % of the picks at layer 0 -- where the inputs are bit-identical -- and up to 31 %
        # deeper in, which is what made the graphed path disagree with the reference at all.
        pm = getattr(self.m, "prune_mask", None)
        keep = pm[L] if pm is not None and L in pm else None
        if L in self.stream_layers:
            keep = self.m.stream_keep[L]
        if self.hot_async and L in self.hot_layers:
            keep = torch.logical_or(keep, self._hot_mask[L], out=self._hot_keep[L])
        if self.lean is not None and self.lean.usable_router(y, w.gate_w):
            # the same router with its elementwise work fused (engine/decode_lean.py); the gate
            # GEMM, topk and the k-way sum stay torch ops on the same shapes
            record = None
            hot = L in self.hot_layers
            if (keep is not None and M.PRUNE_MISS) or hot:
                # The router's unmasked logits/scores: prune-miss accounting, and a hot layer's
                # capture for graph B and the host (DSV41_HOT_READS)
                def record(lg, sc, _keep=keep, _L=L, _hot=hot):
                    if _keep is not None and M.PRUNE_MISS:
                        self._record_prune_miss_side(lg, sc, _keep, _L)
                    if _hot:
                        self._hot_capture(_L, sc, lg, _keep)
            self.lean.router(y, w.gate_w, w.gate_bias, keep, a.n_activated_experts, a.route_scale,
                             self.route_idx, self.route_w, record=record)
            idx = self.route_idx
        else:
            scores = F.softplus(R.mm(y.float(), self.W.layers[L].gate_w)).sqrt()
            logits = scores + w.gate_bias
            if L in self.hot_layers:
                self._hot_capture(L, scores, logits, keep)
            if keep is not None:
                # same accounting as Model.moe: what the router wanted before pruning masked it.
                # Fixed-shape throughout, and the accumulators are allocated before capture, so this
                # records correctly from inside the graph.
                if M.PRUNE_MISS:
                    self.m._record_prune_miss(logits, scores, keep, L, a.n_activated_experts, decode=True)
                logits = logits.masked_fill(~keep, float("-inf"))
            idx = logits.topk(a.n_activated_experts, dim=-1)[1]
            wts = scores.gather(1, idx)
            wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            self.route_idx.copy_(idx); self.route_w.copy_(wts)
        if self.rs_uniq is not None:
            # one-hot the block's k*T expert ids into a [n_routed_experts] table and count the rows
            # that were hit; `index_fill_` writes 1 however many tokens name the same expert.
            self.rs_hits.zero_()
            self.rs_hits.index_fill_(0, idx.reshape(-1), 1)
            self.rs_uniq[L] += self.rs_hits.sum()
        self._tap('moe_in', L, y); self._tap('route_idx', L, idx); self._tap('topk', L, self.topk)
        if self.expert_topp is not None:
            # After the stats and taps (they describe the router), before any slot lookup.
            _ETP.apply_(self.route_idx, self.route_w, *self.expert_topp, self.topp_sentinel)
        if not defer_acct:
            self._acct_join()

    def _record_prune_miss_side(self, lg, sc, keep, L):
        """Model._record_prune_miss on self.acct_stream. The router allocates `lg`/`sc` fresh each
        call and never writes them after record(); record_stream keeps their storage from being
        recycled before the side kernel has read it."""
        side = self.acct_stream
        if side is None:
            self.m._record_prune_miss(lg, sc, keep, L, self.a.n_activated_experts, decode=True)
            return
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            self.m._record_prune_miss(lg, sc, keep, L, self.a.n_activated_experts, decode=True)
        lg.record_stream(side)
        sc.record_stream(side)
        self._acct_pending = True

    def _acct_join(self):
        if self._acct_pending:
            torch.cuda.current_stream(self.dev).wait_stream(self.acct_stream)
            self._acct_pending = False

    def _routed_experts(self, L=None):
        a = self.a
        store = self.m.store
        tier = (self.m.dream_fp8 or {}).get(L) if getattr(self.m, "dream_fp8", None) else None
        dream = (tier, self.route_idx) if tier is not None else None
        if getattr(store, "null_slot", None) is not None:
            # EP2 (engine/dist.py). This path does NOT go through Model.moe, so the two things
            # that method does for expert parallel have to be done here as well -- and both are
            # load-bearing:
            #   slots_repeat=True + null_slot  every non-owned expert shares one zero slot. The
            #                      decode router gives those pairs temporary unique keys, skips
            #                      their compute, and keeps BM=16 for the real experts. Omitting
            #                      both hooks either overflows build_routing_small or falls back to
            #                      the substantially slower BM=64 path (tools/fp4_moe.py).
            #   fp32 + combine     this rank computed only its half of the k-sum; the all-reduce
            #                      supplies the rest. Rounding to bf16 happens ONCE, after the
            #                      combine, at the same point the single-box path rounds.
            # The all-reduce is captured into this graph -- Gate G1 (scripts/gate_g1_graph_nccl.py)
            # verified NCCL collectives capture and replay correctly on these two boxes.
            out = self.m.moe_fn(self.y, self.slots, self.route_w, store.arena, a.swiglu_limit,
                                out_dtype=torch.float32, slots_repeat=True,
                                null_slot=store.null_slot, dream=dream)
            if not getattr(store.ep, 'tensor_parallel', False):
                store.ep.combine(out)
        else:
            out = self.m.moe_fn(self.y, self.slots, self.route_w, store.arena, a.swiglu_limit,
                                dream=dream).float()
        return out

    def _moe_merged_ok(self, w) -> bool:
        """Whether _moe_merged reproduces _routed_experts + _shared_ffn for layer weights `w`. Fixed
        per process (config, arena and weight types), so both ranks answer the same."""
        ok = getattr(self, "_merged_ok", None)
        if ok is None:
            ok = self._merged_ok = self._merged_check()
        return ok and getattr(w, "_sh_w13", None) is not None and self.lut is not None

    def _merged_check(self) -> bool:
        if self.lean is None or not LEAN_MOE or not MERGED_PROJ:
            return False
        store = self.m.store
        arena = getattr(store, "arena", None)
        if getattr(self.eng, "kernel", None) == "exl3-cuda":
            return self._merged_check_exl3(arena)
        if getattr(store, "null_slot", None) is None or not getattr(arena, "tp_output", False):
            return False
        if getattr(arena, "tp_world", 1) != 2 or getattr(self.eng, "kernel", None) != "triton-fp4":
            return False
        try:
            import fp4_moe as K
        except Exception:  # noqa: BLE001
            return False
        if not K.CUDA_DECODE or K.DOT_SCALED or not isinstance(arena, K.ExpertArena):
            return False
        if R.act_qdq_fp8 is R._ACT_QDQ_REF:
            return False  # act_quant on: the shared w2 input would be quantized before its gather
        from engine.tensor_parallel import OutputParallelWeight
        from fp8_linear import FP8Weight
        w = self.W.layers[0]
        w13 = getattr(w, "_sh_w13", None)
        return (w13 is not None and isinstance(w.sh_w2, OutputParallelWeight)
                and type(w.sh_w2.local) is FP8Weight and w.sh_w1.N == arena.w1.shape[1]
                and w.sh_w2.local.shape[0] == arena.w2.shape[1] and T_VERIFY * self.a.n_activated_experts <= 64)

    def _merged_check_exl3(self, arena) -> bool:
        """The EXL3 twin of the FP4 conditions below: TP2 output layout, FP8 shared expert whose
        column split matches the arena's, decode-sized blocks, activations not fake-quantized."""
        if not EXL3_MERGED or getattr(arena, "tp_world", 1) != 2:
            return False
        try:
            import exl3_moe as X3
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(arena, X3.Exl3Arena) or R.act_qdq_fp8 is R._ACT_QDQ_REF:
            return False
        from engine.tensor_parallel import OutputParallelWeight
        from fp8_linear import FP8Weight
        w = self.W.layers[0]
        return (getattr(w, "_sh_w13", None) is not None and isinstance(w.sh_w2, OutputParallelWeight)
                and type(w.sh_w2.local) is FP8Weight and w.sh_w1.N == arena.inter
                and w.sh_w2.local.shape[0] == arena.down_n and T_VERIFY * self.a.n_activated_experts <= 64)

    def _moe_merged_exl3(self, L, w):
        """_moe_merged for EXL3 routed experts (tools/exl3_moe_cuda.py's decode kernels).

        Same two collectives as the FP4 version. The routed intermediate here is the fp16 ROTATED
        down input (gateup_epilogue's Xd) and the shared one is bf16; both are 16-bit, so they are
        stacked as raw words and gathered as one buffer, then each half is read back in its own
        dtype -- a gather moves bytes, so neither changes. The outputs are summed in fp32 before the
        second gather, routed first, which is what hc_post(x2=) computed after two gathers."""
        import exl3_moe_cuda as XC
        a = self.a
        arena = self.m.store.arena
        x = self.y
        T, k = self.route_idx.shape
        P = T * k
        inter, down_n, world = arena.inter, arena.down_n, arena.tp_world
        side = self.shared_stream
        main = torch.cuda.current_stream(self.dev)
        h_all = torch.empty(P + T, inter, dtype=torch.int16, device=x.device)
        x3d = getattr(self.eng, "exl3_dense", None) if getattr(w, "_x3", None) is not None else None
        if x3d is not None:
            # the shared expert is one more EXL3 expert: slot L of a 5-bit arena, weight 1; its
            # gathered half is the fp16 rotated down input, like the routed rows above it
            sh_arena = x3d.shared_arena
            sh_pick = x3d.shared_pick[L, :T]
            sh_grp = XC.decode_group(sh_pick, T, 1, sh_arena)

        def shared_up():
            if x3d is not None:
                XC.decode_up(x, sh_pick, sh_grp, sh_arena, T, 1, a.swiglu_limit, h_all[P:].view(torch.float16))
                return
            gu = R.qlinear(x, w._sh_w13)
            self.lean.swiglu(gu, w.sh_w1.N, a.swiglu_limit, out=h_all[P:].view(torch.bfloat16))

        if side is not None:
            side.wait_stream(main)
            self._pf_wait("shared", side)
            with torch.cuda.stream(side):
                shared_up()
        else:
            self._pf_wait("shared")
        torch.index_select(self.lut[L], 0, self.route_idx.view(-1), out=self.slots.view(-1))
        pick = self.slots.view(-1)
        grp = XC.decode_group(pick, T, k, arena)
        XC.decode_up(x, pick, grp, arena, T, k, a.swiglu_limit, h_all[:P].view(torch.float16))
        if side is not None:
            main.wait_stream(side)
        else:
            shared_up()
        gathered = torch.empty(world * (P + T), inter, dtype=torch.int16, device=x.device)
        if self.l2pf_side is not None:
            self.l2pf_side.wait_stream(main)
            l2pf.touch(self.l2pf_side, self._l2pf_targets(L), l2pf.budget_bytes())
        _comm.all_gather_fast(gathered, h_all)
        if self.l2pf_side is not None:
            main.wait_stream(self.l2pf_side)
        full = gathered.view(world, P + T, inter).transpose(0, 1).reshape(P + T, world * inter)
        def shared_down():
            if x3d is not None:
                return XC.decode_down(full[P:].view(torch.float16), sh_pick, sh_grp, x3d.ones[:T], sh_arena, T, 1)
            return R.mm(full[P:].view(torch.bfloat16), w.sh_w2.local)

        if side is not None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                local_s = shared_down()
        local = XC.decode_down(full[:P].view(torch.float16), pick, grp, self.route_w, arena, T, k)
        if side is not None:
            main.wait_stream(side)
            local_s.record_stream(main)
        else:
            local_s = shared_down()
        local += local_s
        out = torch.empty(world * T, down_n, dtype=torch.float32, device=x.device)
        _comm.all_gather_fast(out, local)
        _hc_post_fused(out, self.h, self.ffn_post, self.ffn_comb, out=self.h, split_w=down_n)

    def _moe_merged(self, L, w):
        """_routed_experts() + _shared_ffn() + the add, with two all-gathers instead of four.

        Output-layout TP gathers the routed intermediate [P, 1152] and the shared one [T, 1152]
        (both this rank's half of 2304 columns), and later the routed and shared outputs (both
        this rank's 2560 of 5120 hidden columns). Each pair has the same split, so each pair is
        one collective here: the intermediates stacked, the outputs summed locally in fp32 --
        `routed + shared.float()` is elementwise, so summing before the gather gives the bits
        summing after it gave. hc_post reads the gathered [2, T, 2560] as is and rounds to bf16.
        The native routing is moe_forward's split_decode_null path, with its elementwise
        prologue/epilogue as two kernels."""
        import fp4_moe as K
        import fp4_moe_cuda as CUDA
        from engine.collective_rails import group as collective_group
        a = self.a
        store = self.m.store
        arena = store.arena
        null = int(store.null_slot)
        x = self.y
        T, k = self.route_idx.shape
        P = T * k
        inter = arena.w1.shape[1]
        world = arena.tp_world
        side = self.shared_stream
        main = torch.cuda.current_stream(self.dev)
        h_all = torch.empty(P + T, inter, dtype=torch.bfloat16, device=x.device)

        def shared_up():
            gu = R.qlinear(x, w._sh_w13)
            self.lean.swiglu(gu, w.sh_w1.N, a.swiglu_limit, out=h_all[P:])

        if side is not None:
            side.wait_stream(main)
            self._pf_wait("shared", side)  # only the shared FFN waits; routed experts start now
            with torch.cuda.stream(side):
                shared_up()
        else:
            self._pf_wait("shared")
        route = self.lean.route_prep(self.route_idx, self.lut[L], self.slots, null)
        block_slot, block_pair, block_route, NB = CUDA.build_routing_small(route, 16, k)
        self.lean.block_null(block_slot, null)
        CUDA.up(x, arena.w1, arena.s1, arena.w3, arena.s3, h_all[:P], self.route_w.reshape(-1),
                block_slot, block_route, float(a.swiglu_limit), k, inter, K.DIM, NB, null)
        tier = (self.m.dream_fp8 or {}).get(L) if getattr(self.m, "dream_fp8", None) else None
        if tier is not None:
            # the dreaming FP8 tier overwrites its experts' rows (engine/dream_overlay.py)
            CUDA.dream_fp8_up(x, self.route_idx.view(-1), tier, h_all[:P], self.route_w.view(-1),
                              float(a.swiglu_limit), k)
        if side is not None:
            main.wait_stream(side)
        else:
            shared_up()
        gathered = torch.empty(world * (P + T), inter, dtype=torch.bfloat16, device=x.device)
        if self.l2pf_side is not None:
            self.l2pf_side.wait_stream(main)
            l2pf.touch(self.l2pf_side, self._l2pf_targets(L), l2pf.budget_bytes())
        _comm.all_gather_fast(gathered, h_all, group=collective_group())
        if self.l2pf_side is not None:
            main.wait_stream(self.l2pf_side)   # the budget is what keeps this join off the critical path
        full = gathered.view(world, P + T, inter).transpose(0, 1).reshape(P + T, world * inter)
        down_n = arena.w2.shape[1]
        if side is not None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                local_s = R.mm(full[P:], w.sh_w2.local)
        parts = torch.empty(P, down_n, dtype=torch.float32, device=x.device)
        CUDA.down(full[:P], arena.w2, arena.s2, parts, block_slot, block_pair, k, down_n,
                  world * inter, T, NB, null, False, 1)
        if tier is not None:
            CUDA.dream_fp8_down(full[:P], self.route_idx.view(-1), tier, parts, k, T)
        local = parts.view(k, T, down_n).sum(dim=0)
        if side is not None:
            main.wait_stream(side)
            local_s.record_stream(main)
        else:
            local_s = R.mm(full[P:], w.sh_w2.local)
        local += local_s
        out = torch.empty(world * T, down_n, dtype=torch.float32, device=x.device)
        _comm.all_gather_fast(out, local, group=collective_group())
        _hc_post_fused(out, self.h, self.ffn_post, self.ffn_comb, out=self.h, split_w=down_n)

    def _l2pf_targets(self, L):
        """The next backbone layer's attention/HC/shared weights: what the kernels after this gather read."""
        if L + 1 >= len(self.W.layers):
            return []
        nxt = self.W.layers[L + 1]
        x3 = getattr(nxt, "_x3", None)
        if x3 is not None:
            return ([x3.wq_a, x3.wkv, x3.wq_b, x3.wo_b, x3.wo_a, nxt.hc_attn_fn, nxt.hc_ffn_fn]
                    + self.eng.exl3_dense.shared_targets(L + 1))
        names = ("wq_a", "wkv", "wq_b", "wo_b", "wo_a", "hc_attn_fn", "hc_ffn_fn",
                 "sh_w1", "sh_w2", "sh_w3")
        return [getattr(nxt, n) for n in names if getattr(nxt, n, None) is not None]

    def _layer_b(self, L):
        a = self.a
        w = self.W.layers[L]
        if L in self.hot_layers and not self.hot_async:
            # Route again under the keep mask the host may have widened (DSV41_HOT_READS); with no
            # read it equals graph A's routing.
            lg = self.hot_logits.masked_fill(~self.m.prune_mask[L], float("-inf"))
            idx = lg.topk(a.n_activated_experts, dim=-1)[1]
            wts = self.hot_scores.gather(1, idx)
            wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            self.route_idx.copy_(idx); self.route_w.copy_(wts)
            if self.expert_topp is not None:
                _ETP.apply_(self.route_idx, self.route_w, *self.expert_topp, self.topp_sentinel)
            if self.lut is not None:
                torch.index_select(self.lut[L], 0, self.route_idx.view(-1), out=self.slots.view(-1))
        if L not in self.stream_layers and self._moe_merged_ok(w):
            if getattr(self.eng, "kernel", None) == "exl3-cuda":
                self._moe_merged_exl3(L, w)
            else:
                self._moe_merged(L, w)
            self.pre_mix.copy_(self.ffn_pre)
            return
        self._pf_join()
        lean = self.lean is not None
        if self.shared_stream is None:
            out = self._routed_experts(L)
            _sh = self._shared_ffn(self.y, w)
            if not lean:
                _sh = _sh.float()
        else:
            # Both branches read y; neither consumes the other's result. Fork AFTER
            # y is ready and rejoin BEFORE residual update/next layer overwrites it.
            # The explicit fork/join is also required inside CUDA graph capture.
            main = torch.cuda.current_stream(self.dev)
            self.shared_stream.wait_stream(main)
            with torch.cuda.stream(self.shared_stream):
                _sh = self._shared_ffn(self.y, w)
                if not lean:
                    _sh = _sh.float()
            try:
                out = self._routed_experts(L)
            finally:
                main.wait_stream(self.shared_stream)
            # _sh is allocated on the side stream but consumed on main. Keep the
            # allocator from recycling its storage before the addition completes.
            _sh.record_stream(main)
        # shared expert is replicated and added AFTER the combine, so it is counted once
        if lean and out.dtype == torch.float32 and _sh.shape == out.shape:
            # hc_post forms (out + _sh.float()).to(bf16) itself, and writes self.h in place
            _hc_post_fused(out, self.h, self.ffn_post, self.ffn_comb, out=self.h, x2=_sh.contiguous())
        elif lean:
            out += _sh.float()
            _hc_post_fused(out.to(torch.bfloat16), self.h, self.ffn_post, self.ffn_comb, out=self.h)
        else:
            out += _sh
            h = (_hc_post_fused(out.to(torch.bfloat16), self.h, self.ffn_post, self.ffn_comb) if _HC_OPS
                 else R.hc_post(out.to(torch.bfloat16), self.h, self.ffn_post, self.ffn_comb))
            self.h.copy_(h)
        self.pre_mix.copy_(self.ffn_pre)

    def _final(self):
        self.logits.copy_(R.head_logits(self._final_x(), self.head))
        self._seed_draft_rings()

    def _final_x(self):
        """The head's input: the final norm over the collapsed residual streams."""
        return self._rmsnorm(R.hc_pre(self.h, self.pre_mix), self.W.norm, self.a.norm_eps)

    def _seed_draft_rings(self):
        a = self.a
        # seed the drafter rings for all 6 positions (harmless beyond the accepted ones)
        m0 = self.W.mtp[0]
        x3 = getattr(m0, "_x3", None)
        mh = self.main_hidden.to(torch.bfloat16)
        # DSV41_TP_DRAFT_DENSE: this rank's half of the rows, gathered (engine/tensor_parallel.py)
        main_proj = getattr(m0, "_main_proj_tp", None) or m0.main_proj
        proj = (x3.main_proj(mh) if x3 is not None and x3.main_proj is not None
                else R.qlinear(mh, main_proj))
        main_x = self._rmsnorm(proj, m0.main_norm, a.norm_eps)
        fq = self.m.freqs_w[self.pos]
        for k, w in enumerate(self.W.mtp):
            # Same weights as the draft's own attention below: the ring value a draft key is
            # compared against must come from the same projection the draft reads, or the EXL3
            # switch would only move half of the draft's keys.
            kx3 = getattr(w, "_x3", None)
            kv_in = kx3.wkv(main_x) if kx3 is not None else R.qlinear(main_x, w.wkv)
            kv = self._rope(self._rmsnorm(kv_in, w.kv_norm, a.norm_eps), fq)
            self.c.mtp_win[k][self.pos % M.RING] = kv

    def _draft(self, greedy: bool):
        """DSpark draft: 5 positions after d_last; gumbel-max sampling with self.d_noise (temperature in d_temp,
        or argmax in the separately captured greedy graph)."""
        a = self.a
        ids = self._d_ids
        ids[0] = self.d_tok[0]
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = self._d_premix0
        pos = self.d_last[0] + 1 + self._d_ar
        for k, w in enumerate(self.W.mtp):
            residual = h
            attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
            y = (self._hc_pre_rn(h, pre_mix, w.attn_norm) if _HC_OPS
                 else R.rmsnorm(R.hc_pre(h, pre_mix), w.attn_norm, a.norm_eps))
            y = self._attention(y, w, 40 + k, self.c.mtp_win[k], self.m.freqs_w, pos, None, mtp_last=self.d_last[0])
            fused_pre = None
            if self.experiments.hc_post_pre:
                from engine.hc_post_pre import post_pre
                h, fused_pre = post_pre(y, residual, attn_post, attn_comb, attn_pre,
                                        self._norm_w(w.ffn_norm), a.norm_eps)
            else:
                h = (_hc_post_fused(y, residual, attn_post, attn_comb) if _HC_OPS
                     else R.hc_post(y, residual, attn_post, attn_comb))
            residual = h
            ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
            y = (fused_pre if fused_pre is not None else
                 self._hc_pre_rn(h, attn_pre, w.ffn_norm) if _HC_OPS else
                 R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps))
            if self.experiments.draft_router:
                idx = torch.empty(y.shape[0], 3, dtype=torch.long, device=y.device)
                wts = torch.empty(y.shape[0], 3, dtype=torch.float32, device=y.device)
                self.lean.router(y, w.gate_w, w.gate_bias, None, 3, a.route_scale, idx, wts)
            else:
                scores = F.softplus(R.mm(y.float(), self.W.mtp[k].gate_w)).sqrt()  # fp32, as Model.moe
                idx = (scores + w.gate_bias).topk(3, dim=-1)[1]
                wts = scores.gather(1, idx); wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            slots = (idx.to(torch.int32) + k * 128)
            if self._draft_merged_ok(w):
                out = self._draft_moe_merged(w, y, slots, wts)
                h = _hc_post_fused(out, residual, ffn_post, ffn_comb, split_w=out.shape[-1])
                pre_mix = ffn_pre
                continue
            out = self.m.moe_fn(y, slots, wts, self.W.dspark_arena, a.swiglu_limit).float()
            # TP draft arenas gather complete outputs inside moe_fn. Without DSV41_TP_DRAFT_DENSE
            # the shared experts are replicated: another collective here would double their
            # contribution. With it, _shared_ffn gathers through OutputParallelWeight instead.
            sh = self._shared_ffn(y, w)
            if self.lean is not None and sh.shape == out.shape:
                h = _hc_post_fused(out, residual, ffn_post, ffn_comb, x2=sh.contiguous())
            else:
                out += sh.float()
                h = (_hc_post_fused(out.to(torch.bfloat16), residual, ffn_post, ffn_comb) if _HC_OPS
                     else R.hc_post(out.to(torch.bfloat16), residual, ffn_post, ffn_comb))
            pre_mix = ffn_pre
        w = self.W.mtp[2]
        # The confidence head reads the UN-normed hc_pre output (as Model.dspark_draft does).
        x_pre = R.hc_pre(h, pre_mix)
        x = self._rmsnorm(x_pre, w.norm, a.norm_eps)
        split = (greedy and self.draft_markov_tp and not TREE_PROBE and self.draft_markov_topk == 0
                 and hasattr(self.draft_head, 'tp_markov_greedy'))
        shortlist = split or (greedy and self.draft_markov_topk > 0 and not TREE_PROBE)
        local = shortlist and not split and self.draft_markov_local and hasattr(self.draft_head, 'tp_candidates')
        logits = None if (local or split) else R.head_logits(x, self.draft_head)  # [T_DRAFT, V]
        if split:
            proposals, embeds = self.draft_head.tp_markov_greedy(x, self.markov_embed_bf16, self.markov_head_local,
                                                                 self.d_tok[0], _lin, self._markov_rank)
            self.d_out.copy_(proposals)
        elif local:
            from engine.draft_markov import chain_greedy
            ids, values = self.draft_head.tp_candidates(x, self.draft_markov_topk // self.draft_head.world)
            proposals, embeds = chain_greedy(ids, values, self.markov_embed_bf16,
                                             self.markov_head_bf16, self.d_tok[0], _lin)
            self.d_out.copy_(proposals)
        elif shortlist:
            from engine.draft_markov import shortlist_greedy
            proposals, embeds = shortlist_greedy(logits, self.markov_embed_bf16,
                                                self.markov_head_bf16, self.d_tok[0],
                                                self.draft_markov_topk, _lin)
            self.d_out.copy_(proposals)
        prev = self.d_tok[0]
        if not shortlist:
            embeds = []
        for i in range(0 if shortlist else T_DRAFT):
            e = self.markov_embed_bf16[prev.view(1)]  # 1-d index: a 0-d tensor index syncs
            embeds.append(e)
            bias = _lin(e, self.markov_head_bf16).float()[0]
            lg = logits[i] + bias
            if TREE_PROBE:
                self.d_top2[i].copy_(lg.topk(2).indices)
            if greedy:
                # Greedy verification never reads q=d_probs.  Avoid five vocabulary-wide
                # softmax/log/Gumbel passes that the old dynamic torch.where computed anyway.
                nxt = lg.argmax()
            else:
                temp = self.d_temp[0]
                p = torch.softmax(lg / temp.clamp_min(1e-5), dim=-1)
                if DRAFT_NUCLEUS:
                    # fixed shapes (graph-captured): topk is sorted, so the cumulative rule is the
                    # target's own `cum - p < top_p`
                    vals, idx = p.topk(DRAFT_NUCLEUS_K)
                    vals = torch.where(vals.cumsum(0) - vals < self.d_top_p[0], vals, 0.0)
                    p = torch.zeros_like(p).scatter_(0, idx, vals / vals.sum())
                nxt = (torch.log(p.clamp_min(1e-30)) + self.d_noise[i]).argmax()
                self.d_probs[i].copy_(p)
            self.d_out[i] = nxt
            prev = nxt
        if COMPUTE_CONF:
            # embeds[i] is the Markov embedding of the token before draft i, as in the reference
            feat = torch.cat([x_pre.float(), torch.cat(embeds).float()], dim=-1)
            self.d_conf.copy_((feat @ w.conf_proj.T).squeeze(-1))

    def _draft_merged_ok(self, w) -> bool:
        """Whether _draft_moe_merged reproduces moe_fn + _shared_ffn for DSpark block `w`. Fixed
        per process (config, arena and weight types), so both ranks answer the same."""
        ok = getattr(self, "_draft_merged", None)
        if ok is None:
            ok = self._draft_merged = self._draft_merged_check()
        return ok and getattr(w, "_sh_w13", None) is not None and getattr(w, "_x3sh", None) is None

    def _draft_merged_check(self) -> bool:
        if self.lean is None or not LEAN_MOE or not MERGED_PROJ:
            return False
        arena = getattr(self.W, "dspark_arena", None)
        if not getattr(arena, "tp_output", False) or getattr(arena, "tp_world", 1) != 2:
            return False
        try:
            import fp4_moe as K
        except Exception:  # noqa: BLE001
            return False
        if not K.CUDA_DECODE or K.DOT_SCALED or not isinstance(arena, K.ExpertArena):
            return False
        if R.act_qdq_fp8 is R._ACT_QDQ_REF:
            return False  # act_quant on: the shared w2 input would be quantized before its gather
        from engine.tensor_parallel import OutputParallelWeight
        from fp8_linear import FP8Weight
        w = self.W.mtp[0]
        return (isinstance(w.sh_w2, OutputParallelWeight) and type(w.sh_w2.local) is FP8Weight
                and w.sh_w1.N == arena.w1.shape[1] and w.sh_w2.local.shape[0] == arena.w2.shape[1]
                and T_DRAFT * 3 <= 64)

    def _draft_moe_merged(self, w, y, slots, wts):
        return draft_moe_merged(y, slots, wts, self.W.dspark_arena, w._sh_w13, w.sh_w1.N, w.sh_w2.local,
                                self.lean, self.a.swiglu_limit, self.shared_stream)

    def _capture_draft_graphs(self):
        """Capture one graph per sampling mode; neither mode pays for the other's dead branch."""
        if self.draft_graphs is not None:
            return self.draft_graphs
        out = []
        for greedy in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._draft(greedy)
            out.append(graph)
        self.draft_graphs = tuple(out)
        return self.draft_graphs

    def build_lut(self):
        """Device slot table from the store's LRU. Only valid while no expert is evicted/loaded; the
        engine rebuilds it whenever the store reports a miss.

        EP2: this rank's LRU holds only the experts it OWNS, so every other routable expert would
        be left at the -1 fill and gather garbage. The right value for them is the null slot --
        zero-filled weights, an exact 0 contribution, and the peer's all-reduce supplies the real
        one. That is the same substitution ExpertStore.resolve makes on the host path, so the
        graphed and eager paths agree. Experts pruned out of the keep set also land on the null
        slot and never matter: the router masks them to -inf, so they are never routed at all.

        This is what unlocks the segmented capture. `self.lut is not None` gates the branch in
        capture() that records whole runs of layers as one graph -- 41 replays per step become 3,
        with no host round-trip between A and B -- which is where the 40 blocking device->host
        syncs per step go.
        """
        st = self.m.store
        null = getattr(st, "null_slot", None)
        fill = -1 if null is None else int(null)
        lut = torch.full((self.a.n_layers, self.a.n_routed_experts), fill, dtype=torch.int32)
        for (L, e), slot in st.lru.items():
            lut[L, e] = slot
        if self.topp_sentinel is not None:
            # The pruned-pick column: the FP4 kernels skip the null slot's pairs; the EXL3 decode
            # grouping skips negative picks (reading the EXL3 null slot would cost a full read).
            exl3 = getattr(self.eng, "kernel", None) == "exl3-cuda"
            if getattr(self.eng, "expert_format", None) == "exl3" and not exl3:
                raise ValueError("DSV41_EXPERT_TOPP on EXL3 experts needs the EXL3 CUDA decode kernel")
            if not exl3 and null is None:
                raise ValueError("DSV41_EXPERT_TOPP needs the null slot of the TP/EP expert store")
            lut = torch.cat([lut, torch.full((self.a.n_layers, 1), -1 if exl3 else int(null),
                                             dtype=torch.int32)], dim=1)
        # Validate ONCE, on the host tensor, before it goes to the device: a -1 left in the table
        # indexes the arena out of bounds inside a captured graph, where there is no bounds check
        # and the symptom would be corrupted experts rather than an error. Checking per call would
        # be a device->host sync per layer, which is what this table exists to remove.
        # Only ROUTABLE experts have to be mapped. A pruned expert is never gathered -- the router
        # masks it to -inf (Model.moe) -- so its -1 is correct and expected, and demanding a slot
        # for it broke the single-box pruned path: at keep 0.31 that is 264 of 384 experts per
        # layer, 10,560 entries, and the assert fired at startup. It never showed under EP2, where
        # the fill value is the null slot and every entry is mapped either way.
        pm = getattr(self.m, "prune_mask", None)
        if pm:
            routable = torch.zeros_like(lut, dtype=torch.bool)
            for L, msk in pm.items():
                routable[L, :msk.numel()] = msk.to(lut.device)
        else:
            routable = torch.ones_like(lut, dtype=torch.bool)
        if self.topp_sentinel is not None:
            routable[:, self.topp_sentinel] = False   # never routed; it is where pruned picks go
        bad = int(((lut < 0) & routable).sum())
        assert bad == 0, (f"slot table leaves {bad} ROUTABLE (layer, expert) entries unmapped; "
                          f"every expert the router can pick needs a real slot or the null slot")
        self.lut = lut.to(self.dev)
        self.lut_version = st.stats.get("misses", 0)

    def _layer_ab(self, L, sh_state):
        self._layer_a(L, sh_state, defer_acct=True)
        # -1 never occurs while the LUT is valid
        if self._moe_merged_ok(self.W.layers[L]):
            pass  # _moe_merged looks the slots up together with its routing keys
        elif self.lean is not None:
            torch.index_select(self.lut[L], 0, self.route_idx.view(-1), out=self.slots.view(-1))
        else:
            self.slots.copy_(self.lut[L][self.route_idx])
        self._layer_b(L)
        self._acct_join()

    # ------------------------------------------------------------------ gpu timing
    def _ev_begin(self, phase):
        if not self.gpu_timing:
            return None
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self._ev_pairs[phase].append((a, b))
        return b

    def gpu_timing_reset(self):
        for v in self._ev_pairs.values():
            v.clear()

    def gpu_timing_report(self):
        """{phase: total ms on the GPU timeline}. Synchronises once, here, not per step."""
        if not self.gpu_timing or not any(self._ev_pairs.values()):
            return None
        torch.cuda.synchronize()
        out = {}
        for phase, pairs in self._ev_pairs.items():
            # A step that raised between the begin and end events leaves the last pair half
            # recorded, and elapsed_time() then throws from inside the error handler -- which is
            # how a skinny-kernel assert reached the log as "Both events must be recorded",
            # twice, with the real traceback buried under it. Measurement must never be the thing
            # that reports the failure.
            total = 0.0
            done = 0
            for a, b in pairs:
                try:
                    total += a.elapsed_time(b)
                except (ValueError, RuntimeError):
                    continue
                done += 1
            out[phase] = round(total, 1)
            out[phase + "_n"] = done
        return out

    # ------------------------------------------------------------------ route stats
    def route_stats_reset(self):
        """Zero the DSV41_ROUTE_STATS accumulator. Call it after the graphs are captured: capture's
        own warm-up and capture runs execute `_layer_a` and would otherwise be counted."""
        if self.rs_uniq is not None:
            self.rs_uniq.zero_(); self.rs_steps = 0

    def route_stats_report(self):
        """Mean distinct routed experts per layer per verify block, over the steps since the last
        reset: {'steps', 'per_layer' (n_layers floats), 'mean', 'total'}."""
        if self.rs_uniq is None or not self.rs_steps:
            return None
        per = (self.rs_uniq / self.rs_steps).tolist()
        return {"steps": self.rs_steps, "per_layer": [round(v, 4) for v in per],
                "mean": sum(per) / len(per), "total": sum(per)}

    def release_graphs(self):
        """Release every owner of the shared graph pool between completed steps.

        Static model/width buffers and resident LUTs remain untouched. A partial
        LRU cannot return this pool's high-water memory while draft graphs pin it.
        Both ranks call this on the same key sequence under the boot guard.
        """
        torch.cuda.synchronize()
        self.graph_evictions += len(self.graphs)
        self.graphs.clear()
        self._memos.clear()
        self._memo = None
        self.draft_graphs = None
        self.pool = None
        self._probe_draft_key = None
        if self.decode_probe is not None:
            self.decode_probe.clear_captures()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.graph_pool_resets += 1

    def memory_report(self):
        """Read-only allocator counters; no device synchronization on /health."""
        return dict(graph_limit=self.graph_limit, graph_count=len(self.graphs),
                    graph_keys=[list(k) for k in self.graphs],
                    graph_captures=self.graph_captures, graph_evictions=self.graph_evictions,
                    graph_pool_resets=self.graph_pool_resets,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    last_capture=dict(self.graph_memory))

    # ------------------------------------------------------------------ capture
    def capture(self, S_parity: int, index_bucket: int):
        """Capture graphs for one start parity, fixed indexer context bucket and the currently
        bound verify width.

        Only ever for the width of the step being run, never ahead for another width: the
        warm-up forward writes KV at the bound width's positions, which the real step then
        rewrites -- but a WIDER block warmed at the same start would write positions past the
        step, and in the 128-slot window ring those overwrite entries the step's own last query
        still attends to."""
        if not self.use_graphs:
            return
        key = (S_parity, index_bucket, self.width)
        if key in self.graphs:
            self.graphs.move_to_end(key)
            return
        # All verify graphs and draft graphs share one pool. LRU removal alone
        # leaves that pool pinned by the surviving graphs, so rotate the entire
        # generation at the cap. This happens at a new key, never on a cache hit.
        if self.graph_limit and len(self.graphs) >= self.graph_limit:
            self.release_graphs()
        before = torch.cuda.memory_allocated()
        # Compiler warmup is not traffic. Preserve the request's counts AND score evidence,
        # in place, so graph capture cannot bias placement or inflate the reported misses.
        counters = {}
        for name in ('_want_counts', '_want_mass', '_req_counts', '_req_mass',
                     '_miss_tot', '_miss_mass_tot', '_miss_phase', '_want_phase'):
            value = getattr(self.m, name, None)
            for tensor in value if isinstance(value, list) else [value]:
                if tensor is not None:
                    counters[id(tensor)] = tensor
        saved_counters = [(tensor, tensor.clone()) for tensor in counters.values()]
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        st = {"parity": S_parity, "index_bucket": index_bucket,
              "ckv": None, "ik": None, "ratio": 0}
        s = self.capture_warmup_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # warm-up run (allocations, triton compiles) on a scratch copy of the state
            saved = [self.h.clone(), self.pre_mix.clone()]
            self._memo = {} if self.lean is not None else None
            for L in range(self.a.n_layers):
                self._layer_a(L, st); self._layer_b(L)
            self._memo = None
            self._final(); self._draft(True); self._draft(False)
            self.h.copy_(saved[0]); self.pre_mix.copy_(saved[1])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        st = {"parity": S_parity, "index_bucket": index_bucket,
              "ckv": None, "ik": None, "ratio": 0}
        if self.lean is not None:
            self._memo = self._memos[key] = {}
        try:
            probe = self.decode_probe
            if probe is not None and probe.active:
                from engine.decode_probe_hooks import capture_hooks
                had_draft = self.draft_graphs is not None
                with capture_hooks(self, probe, key):
                    self._capture_layers(key, st)
                if not had_draft:
                    self._probe_draft_key = key
            else:
                self._capture_layers(key, st)
            self.graph_captures += 1
            self.graph_memory = dict(last_key=list(key), allocated_before=before,
                                     allocated_after=torch.cuda.memory_allocated(),
                                     reserved_after=torch.cuda.memory_reserved())
        finally:
            self._memo = None
            for tensor, saved in saved_counters:
                tensor.copy_(saved)

    def _capture_layers(self, key, st):
        if self.lut is not None and GRAPH_SEGMENTS:
            # Resident mode: routing is a device LUT lookup, so the ONLY host dependency inside a
            # step is the Engram rows of layers 1 and 14. Capture the layers between those
            # boundaries as single graphs -- 41 replays per step become 3 -- and keep the overlap:
            # a segment is queued asynchronously, so the host blocks on the next boundary's NVMe
            # reads while the GPU is still running the segment before it.
            from engine.layer_stream import graph_bounds
            bounds = graph_bounds(self.a.n_layers, self.a.engram_layer_ids, self.split_layers)
            segs = []
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                if lo in self.split_layers:
                    # Router/attention first, then host loads current picks, then MoE.
                    # Both ranks split at these guarded static boundaries unconditionally.
                    gA = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gA, pool=self.pool):
                        self._layer_a(lo, st)
                    gB = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gB, pool=self.pool):
                        self._layer_b(lo)
                        if hi == self.a.n_layers:
                            self._final()
                    segs.append((lo, gA, gB))
                    continue
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    for L in range(lo, hi):
                        self._layer_ab(L, st)
                    if hi == self.a.n_layers:
                        self._final()  # the head + drafter seeding ride in the last segment
                segs.append((lo, g, None))
            gD = self._capture_draft_graphs()
            self.graphs[key] = (None, None, None, gD, segs)
            torch.cuda.synchronize()
            return
        gA, gB = [], []
        for L in range(self.a.n_layers):
            if self.lut is not None and L not in self.split_layers:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._layer_ab(L, st)
                gA.append(g); gB.append(None)
                continue
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_a(L, st)
            gA.append(g)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_b(L)
            gB.append(g)
        gF = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gF, pool=self.pool):
            self._final()
        gD = self._capture_draft_graphs()
        self.graphs[key] = (gA, gB, gF, gD, None)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ run
    def _resolve(self, L):
        # host: expert ids -> arena slots (loads misses from NVMe)
        idx = self.route_idx
        pruned = None
        if self.topp_sentinel is not None:
            # The store knows no sentinel: resolve the row's first pick (always kept) in its place
            # and give the pruned pairs the null slot after (weight 0 either way).
            pruned = idx == self.topp_sentinel
            idx = torch.where(pruned, idx[:, :1], idx)
        try:
            slots = self.m.store.resolve(L, idx, L in self.stream_layers)
            if pruned is not None and getattr(self.m.store, "null_slot", None) is not None:
                slots = torch.where(pruned.to(slots.device), torch.full_like(slots, int(self.m.store.null_slot)), slots)
        except Exception:
            if L in self.stream_layers:
                import traceback
                traceback.print_exc()
                os._exit(1)  # never let a half-loaded pair continue with different slots
            raise
        self.slots.copy_(slots)

    def _hot_capture(self, L, scores, logits, keep):
        """Graph A of a hot layer: keep the unmasked router outputs for graph B, and put the
        layer's candidates in its pinned host row. DSV41_HOT_ASYNC: the candidates only, on the
        device; the step copies every layer's row to the host once, at its end."""
        if self.hot_async:
            self.hot.candidates(scores, logits, keep, self._hot_dev[L])
            return
        self.hot_scores.copy_(scores); self.hot_logits.copy_(logits)
        self.hot.candidates(scores, logits, keep, self._hot_dev[L])
        self._hot_host[L].copy_(self._hot_dev[L], non_blocking=True)

    def _split_resolve(self, L):
        """Between a split layer's graphs: a stream layer resolves its picks; a hot layer reads
        the planned misses into the transient ring and makes them routable for graph B."""
        if L not in self.hot_layers or self.hot_async:
            self._resolve(L)
            return
        if self.hot.room(L) <= 0:
            self.hot.unread.append(L)        # budget spent: no sync (its row is counted next step)
            return
        torch.cuda.current_stream(self.dev).synchronize()       # graph A wrote the host row
        es = self.hot.plan(L, self._hot_host[L].tolist())
        if not es:
            return
        t0, st = time.perf_counter(), self.m.store.stats
        n0 = st["prefill_misses"]
        slots = self.m.store.resolve(L, torch.tensor([es], dtype=torch.int64), True)[0]
        self.hot.stats["loaded"] += st["prefill_misses"] - n0     # the rest were ring hits
        ids = torch.tensor(es, dtype=torch.long, device=self.dev)
        keep, lut = self.m.prune_mask[L], self.lut[L]
        self._hot_live = (L, ids, lut[ids].clone())
        keep[ids] = True
        lut[ids] = slots.to(device=self.dev, dtype=lut.dtype)
        self.hot.stats["s"] += time.perf_counter() - t0

    def _hot_restore(self):
        """After graph B (stream-ordered): the read experts are not residents."""
        if self._hot_live is None:
            return
        L, ids, old = self._hot_live
        self.m.prune_mask[L][ids] = False
        self.lut[L][ids] = old
        self._hot_live = None

    def _hot_async_begin(self):
        """DSV41_HOT_ASYNC, before the step's graphs: plan from the last step's candidates (its
        logits were read, so its host rows are final), start the reads into the ring, and make the
        ring's live experts routable for this step: their slot-table entries, and `_hot_mask`,
        which the hot layers' routers OR into the resident mask. A handful of launches however
        many are live; a write a layer to the resident masks cost ~3 ms of host time a step.

        A read's slot was last routable in an earlier step, all of whose work is queued ahead of
        `_hot_ev`; the copy waits for that event only, so it overlaps this step's graphs, which no
        longer route to the slot (`take` removed its previous key before the marks below)."""
        t0 = time.perf_counter()
        ring, st = self.hot_ring, self.m.store
        plan = self.hot.plan_async(self._hot_host.tolist(), ring.taken())
        null = getattr(st, "null_slot", None)
        if plan:
            self._hot_ev.record()
            self._hot_side.wait_event(self._hot_ev)
            for key in plan:
                slot = ring.take(key)
                if null is None or st.ep.owns(*key):     # EP2: only the owner holds the expert
                    self._hot_futs.append(st.pool.submit(self._hot_load, key, slot))
        if ring.live:
            items = sorted(ring.live.items())
            n = len(items)
            pin = self._hot_pin_np
            for i, ((L, e), slot) in enumerate(items):
                pin[i, 0], pin[i, 1] = L, e
                pin[i, 2] = slot if null is None or st.ep.owns(L, e) else null
            idx = self._hot_idx[:n]
            idx.copy_(self._hot_pin[:n], non_blocking=True)
            flat = idx[:, 0] * self.lut.shape[1] + idx[:, 1]
            lut = self.lut.view(-1)
            self._hot_saved = (flat, lut.index_select(0, flat))
            lut.index_copy_(0, flat, idx[:, 2].to(lut.dtype))
            self._hot_mask.view(-1).index_fill_(0, idx[:, 0] * self._hot_mask.shape[1] + idx[:, 1], True)
        self.hot.stats["s"] += time.perf_counter() - t0

    def _hot_load(self, key, slot):
        """An io thread's read into a ring slot. With `_hot_side` current, the store's copy waits
        for that stream (the event before this step's graphs), not for the graphs themselves."""
        with torch.cuda.stream(self._hot_side):
            self.m.store._load_into_slot(key, slot)

    def _hot_async_end(self):
        """After the step's graphs are queued: undo the marks (stream-ordered after the graphs),
        queue the candidate rows' copy to the host, and wait for the step's reads, which ran while
        the GPU worked; they are routable from the next step. Nothing is in flight between steps,
        so prefill and adaptation never see a mark or a ring slot being written."""
        t0 = time.perf_counter()
        if self._hot_saved is not None:
            flat, saved = self._hot_saved
            self.lut.view(-1).index_copy_(0, flat, saved)
            self._hot_mask.zero_()     # capture warm-ups between steps must route as residents only
            self._hot_saved = None
        self._hot_host.copy_(self._hot_dev, non_blocking=True)
        futs, self._hot_futs = self._hot_futs, []
        try:
            for f in futs:
                f.result()
        except Exception:
            # One rank without the expert would route the same rows to garbage: stop the pair.
            import traceback
            traceback.print_exc()
            os._exit(1)
        self.hot.stats["loaded"] += len(futs)
        self.hot_ring.promote()
        self.hot.stats["wait_s"] += time.perf_counter() - t0

    def _upload_engram(self, layer, future, finish):
        """Upload/dequantize while the preceding segment runs, then join on main.

        This is called on the decode thread, never from an I/O worker. The
        source table owns pinned staging and its reuse events. The result is
        allocated on the upload stream and retained through its main-stream
        copy; record_stream prevents allocator reuse until that consumer ends.
        """
        side = self.engram_upload_stream if self.experiments.engram_h2d else None
        # This event is queued after the previous graph, before a CPU row wait.
        # Its interval includes only the exposed suffix of that wait/upload,
        # rather than charging all overlapping disk-read time to the device.
        wait = self._ev_begin("engram_ready_wait") if side is not None else None
        raw = future.result()
        if side is None:
            self.eg_rows[layer].copy_(finish(*raw))
            return
        main = torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(side):
            rows = finish(*raw)
            ready = torch.cuda.Event()
            ready.record(side)
        main.wait_event(ready)
        if wait is not None:
            wait.record(main)
        rows.record_stream(main)
        self.eg_rows[layer].copy_(rows)

    def step(self, block_ids: torch.Tensor, S: int, engram_rows: dict):
        """Run the T-token verify block at positions S..S+T-1, T = len(block_ids), one of
        VERIFY_WIDTHS. engram_rows: {L: [T,24,256] fp32}. Returns (logits [T,V] fp32,
        main_hidden [T, 15360] fp32) as views of T's static buffers."""
        a = self.a
        T = block_ids.numel()
        assert T in self._width_bufs, (T, VERIFY_WIDTHS)
        if self.hot is not None and not self.hot_async:
            # the previous step's skipped rows are final by now (its logits were read)
            self.hot.begin_step([self._hot_host[L].tolist() for L in self.hot.unread])
        assert self.c.len == S, (self.c.len, S)
        self._bind(T)
        if LEAN_STEP:
            if block_ids is not self.ids:
                self.ids.copy_(block_ids)
            torch.add(self._ar_t, S, out=self.pos)
        else:
            self.ids.copy_(block_ids)
            self.pos.copy_(S + torch.arange(T, device=self.dev))
        rows_fn = engram_rows if callable(engram_rows) else None
        if rows_fn is None:
            for L, rows in engram_rows.items():
                self.eg_rows[L].copy_(rows)
        if LEAN_STEP:
            # expand, not repeat: copy_ reads the stride-0 view directly, nothing is materialised
            self.h.copy_(self.W.embed[self.ids].unsqueeze(1).expand(-1, a.hc_mult, -1))
            self.pre_mix.copy_(self._premix0)
        else:
            self.h.copy_(self.W.embed[self.ids].unsqueeze(1).repeat(1, a.hc_mult, 1))
            self.pre_mix.zero_(); self.pre_mix[:, 0] = 1.0
        parity = S % 2
        index_bucket = (_index_bucket(S + T, self.c.max_seq)
                        if INDEX_BUCKETS else self.c.max_seq)
        graph_key = (parity, index_bucket, T)
        if graph_key in self.graphs:
            self.graphs.move_to_end(graph_key)
        self.prepare_pending_buffers()
        if not LEAN_STEP or graph_key not in self.graphs:
            pending = ({L: None if value is None else (value[0].clone(), value[1].clone())
                        for L, value in self.c.pending.items()}
                       if self.use_graphs and graph_key not in self.graphs else None)
            self.capture(parity, index_bucket)
            if pending is not None:
                # A rollback boundary may alias kvl_buf/sc_buf, which capture's
                # warm-up overwrites. Keep committed history across a cold graph.
                self.c.pending.update(pending)
            self.prepare_pending_buffers()  # capture's warm-up/capture runs overwrite the buffers
        if self.hot_async:
            self._hot_async_begin()
        t0 = time.perf_counter()
        gA, gB, gF, gD, segs = self.graphs[graph_key] if self.use_graphs else (None,) * 5
        # {layer: Future} -- reads already in flight. With segments, the call waits until the first
        # Engram boundary: the caller's `rows` may block on the hash ids, and the segment before
        # that boundary needs none (v41_engine._engram_rows_async).
        futs = rows_fn() if rows_fn is not None and segs is None else None
        _ev_layers = self._ev_begin("layers")
        if self.use_graphs:
            if segs is not None:
                for lo, g, gB in segs:
                    if futs is None and rows_fn is not None and lo in self.a.engram_layer_ids:
                        futs = rows_fn()
                    if futs is not None and lo in futs:
                        # this boundary's rows were read while the previous segment ran on the GPU
                        t0r = time.perf_counter()
                        fut, finish = futs[lo]
                        self._upload_engram(lo, fut, finish)
                        self.stats["engram_s"] += time.perf_counter() - t0r
                    # Unlike the outer `layers` pair, this pair surrounds only one graph replay.
                    # Summing it excludes CPU waits before Engram boundaries; layers-segments is
                    # therefore the exposed GPU-idle/H2D gap rather than assumed kernel work.
                    _ev_segment = self._ev_begin("segments")
                    g.replay()
                    if gB is not None:
                        self._split_resolve(lo)
                        gB.replay()
                        self._hot_restore()
                    if _ev_segment is not None:
                        _ev_segment.record()
            else:
                for L in range(a.n_layers):
                    if futs is not None and L in futs:
                        # this layer's rows were read while the previous layers ran on the GPU
                        t0r = time.perf_counter()
                        fut, finish = futs[L]
                        self._upload_engram(L, fut, finish)
                        self.stats["engram_s"] += time.perf_counter() - t0r
                    _ev_segment = self._ev_begin("segments")
                    gA[L].replay()
                    if gB[L] is not None:
                        self._split_resolve(L)
                        gB[L].replay()
                        self._hot_restore()
                    if _ev_segment is not None:
                        _ev_segment.record()
                gF.replay()
            if self.decode_probe is not None and self.decode_probe.active:
                self.decode_probe.mark_replay(graph_key, phase="verify")
            if self.lut is not None:
                # bookkeeping the host resolve would have done: LRU touch is irrelevant while resident
                self.m.store.stats["hits"] += int((a.n_layers - len(self.stream_layers)) * self.route_idx.numel())
        else:
            if futs is not None:
                for LL, (f, finish) in futs.items():
                    self._upload_engram(LL, f, finish)
            st = {"parity": parity, "index_bucket": index_bucket,
                  "ckv": None, "ik": None, "ratio": 0}
            for L in range(a.n_layers):
                self._layer_a(L, st); self._split_resolve(L); self._layer_b(L); self._hot_restore()
            self._final()
        if self.hot_async:
            self._hot_async_end()
        # host-side bookkeeping for Caches.rollback (same tuple layout as model.py)
        for L in self.kvl_buf:
            before = self.c.pending.get(L)
            self.c._chunk_inputs[L] = (S, self.kvl_buf[L], self.sc_buf[L], before)
            if (S + T) % 2 == 1:
                self.c.pending[L] = (self.kvl_buf[L][T - 1].clone(),
                                     self.sc_buf[L][T - 1].clone())  # the last token is unpaired
            else:
                self.c.pending[L] = None
        if _ev_layers is not None:
            _ev_layers.record()
        self.c.len = S + T
        self.stats["steps"] += 1
        self.stats["graph_s"] += time.perf_counter() - t0
        if self.rs_uniq is not None:
            self.rs_steps += 1
        return self.logits, self.main_hidden

    def draft_ahead(self, count: torch.Tensor, argmax: torch.Tensor, pos: int) -> bool:
        """Queue the greedy drafter for a verify block's outcome before the host has read it.

        `count` is the block's accepted-draft count and `argmax` the target's per-row argmax, both
        on the GPU: the next draft starts from token argmax[count] at main position pos + count,
        which is what the host would pass to draft() after reading them. Writing d_tok/d_last
        on the device lets the drafter run while the host reads the result, emits tokens and runs
        the per-step control handshake; draft() then returns this draft if its arguments match
        (the caller records them in self._ahead), else drafts again. Greedy graphs only."""
        if not (self.use_graphs and self.draft_graphs is not None):
            return False
        torch.index_select(argmax, 0, count, out=self.d_tok)
        torch.add(count, pos, out=self.d_last)
        self.d_temp.zero_()
        t0 = time.perf_counter()
        self.draft_graphs[0].replay()
        self.stats["draft_s"] += time.perf_counter() - t0
        return True

    def draft(self, tok: int, last_main_pos: int, temperature: float, top_p: float = 1.0):
        ahead, self._ahead = self._ahead, None
        if ahead is not None and temperature <= 0 and ahead == (tok, last_main_pos):
            return self.d_out, self.d_probs
        self.d_tok.fill_(tok); self.d_last.fill_(last_main_pos); self.d_temp.fill_(temperature)
        if DRAFT_NUCLEUS:
            self.d_top_p.fill_(min(max(float(top_p), 1e-6), 1.0))
        greedy = temperature <= 0
        if not greedy:
            u = torch.rand_like(self.d_noise).clamp_min(1e-30)
            self.d_noise.copy_(-torch.log(-torch.log(u)))
        t0 = time.perf_counter()
        _ev_draft = self._ev_begin("draft")
        if self.use_graphs and self.draft_graphs is not None:
            self.draft_graphs[0 if greedy else 1].replay()
            if self.decode_probe is not None and self.decode_probe.active and self._probe_draft_key is not None:
                self.decode_probe.mark_replay(self._probe_draft_key,
                    phase="draft.greedy" if greedy else "draft.sampled")
        else:
            self._draft(greedy)
        if _ev_draft is not None:
            _ev_draft.record()
        self.stats["draft_s"] += time.perf_counter() - t0
        return self.d_out, self.d_probs

    def prepare_pending_buffers(self):
        """Copy Caches.pending (host truth) into the static buffers the graphs read."""
        for L, buf in self.pend_buf.items():
            p = self.c.pending.get(L)
            if p is not None:
                buf[0].copy_(p[0]); buf[1].copy_(p[1])


def draft_moe_merged(y, slots, wts, arena, w13, n: int, w2_local, lean, limit: float, side=None):
    """One DSpark block's routed + shared FFN as FastDecoder._moe_merged does the backbone's: two
    RoCE gathers instead of moe_forward's two NCCL ones, the shared expert split by output rows
    (DSV41_TP_DRAFT_DENSE: `w13` this rank's w1||w3 rows, `n` their width, `w2_local` its w2 rows)
    instead of read whole on both ranks, and on the `side` stream during the routed `up`.
    Returns the gathered fp32 [world*T, down_n] for hc_post's split_w read.

    Same values as moe_fn + _shared_ffn: the routed kernels get moe_forward's native decode
    arguments (null slot -1, its routing builder); each shared output is the same complete
    fp8_linear dot product, on one rank; and hc_post's ROUND of (routed + shared) summed in fp32
    here is its ADD2 of the same two values. tools/test_draft_merged_tp.py compares the bytes."""
    import fp4_moe as K
    import fp4_moe_cuda as CUDA
    from engine.collective_rails import group as collective_group
    T, k = slots.shape
    P = T * k
    inter, down_n, world = arena.w1.shape[1], arena.w2.shape[1], arena.tp_world
    main = torch.cuda.current_stream(y.device)
    h_all = torch.empty(P + T, inter, dtype=torch.bfloat16, device=y.device)

    def shared_up():
        lean.swiglu(R.qlinear(y, w13), n, limit, out=h_all[P:])

    if side is not None:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared_up()
    block_slot, block_pair, block_route, NB = CUDA.build_routing_small(slots.contiguous(), 16, k)
    wgt = wts.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    CUDA.up(y, arena.w1, arena.s1, arena.w3, arena.s3, h_all[:P], wgt, block_slot, block_route,
            float(limit), k, inter, K.DIM, NB, -1)
    if side is not None:
        main.wait_stream(side)
    else:
        shared_up()
    gathered = torch.empty(world * (P + T), inter, dtype=torch.bfloat16, device=y.device)
    _comm.all_gather_fast(gathered, h_all, group=collective_group())
    full = gathered.view(world, P + T, inter).transpose(0, 1).reshape(P + T, world * inter)
    if side is not None:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            local_s = R.mm(full[P:], w2_local)
    parts = torch.empty(P, down_n, dtype=torch.float32, device=y.device)
    CUDA.down(full[:P], arena.w2, arena.s2, parts, block_slot, block_pair, k, down_n,
              world * inter, T, NB, -1, False, 1)
    local = parts.view(k, T, down_n).sum(dim=0)
    if side is not None:
        main.wait_stream(side)
        local_s.record_stream(main)
    else:
        local_s = R.mm(full[P:], w2_local)
    local += local_s
    out = torch.empty(world * T, down_n, dtype=torch.float32, device=y.device)
    _comm.all_gather_fast(out, local, group=collective_group())
    return out
