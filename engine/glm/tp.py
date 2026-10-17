"""GLM-5.3 on two ranks (TP2). Shares engine/muse/tp.py's communication, memory guards and graph
replay; the layers are GLM's (engine/glm/ref.py is the numerical reference).

Per layer, per rank (r of W=2):

  attention  q_a and kv_a replicated (FP8 128-block, engine/glm/fp8_block.py); q_b, kv_b and
             o_proj split by heads (32 of 64). The MLA latent (normed c [512] + roped shared key
             [64]) is cached on both ranks. Attention is absorbed: q_nope through kv_b's key half
             into latent space, softmax over the selected positions, the latent output through
             kv_b's value half, o_proj columns -> partial [T, H] -> all-reduce.
  indexer    replicated on "full" layers (21 of 78): relu(q.k) weighted per head over every
             cached position, causal, top-2048; "shared" layers reuse the last full layer's set.
             Both ranks select identically from identical inputs.
  experts    MXFP4 slots (rows of gate/up, output rows of down, as in Muse). The shared expert is
             a 9th pick with weight 1; a dense MLP (intermediate 12288) is 6 pseudo-experts of
             2048, all picked with weight 1. One kernel path for all three.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.muse.tp import Comm, env, mem_available, rss_anon  # noqa: F401  (tools path set there)
from engine.muse.weights import GROUP, quantize_mxfp4, dequant_mxfp4
from .config import GlmConfig
from .fp8_block import FP8Block, fp8_block_linear
from .ref import LATENT_EPS, rope_interleave
from .weights import GlmCheckpoint, P

VERSION = 1


class Lin:
    """FP8 block-scaled or BF16 weight on the GPU."""

    def __init__(self, w, device):
        if isinstance(w, FP8Block):
            self.q, self.w = w.to(device), None
        else:
            self.q, self.w = None, w.to(device, torch.bfloat16).contiguous()

    def __call__(self, x, out_dtype=torch.bfloat16):
        if self.q is not None:
            return fp8_block_linear(x, self.q, out_dtype=out_dtype)
        y = F.linear(x.to(torch.bfloat16), self.w)
        return y if out_dtype == torch.bfloat16 else y.to(out_dtype)

    @property
    def nbytes(self):
        return self.q.nbytes if self.q is not None else self.w.numel() * 2


# MUSE_FUSED_NORM=1: rms() in one kernel (engine/glm/norm_kernel.py) for 2-D BF16 rows: 12x faster
# at a 1,024-row prefill chunk, a seventh of the kernels in a decode graph; 99.999% identical bits.
FUSED_NORM = env("MUSE_FUSED_NORM", "1") == "1"


def rms(x, weight, eps):
    if FUSED_NORM and x.dim() == 2 and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 \
            and x.stride(-1) == 1:
        from .norm_kernel import rms_norm
        return rms_norm(x, weight, eps)
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


KV_BLOCK = 128          # values per fp32 scale in an FP8 cache row


def kv_write(cache, idx, x):
    """Store rows x [n, d] at idx: a BF16 tensor directly, an (fp8, scale) pair quantized per
    KV_BLOCK values (or per row when the scale has one column)."""
    if isinstance(cache, torch.Tensor):
        cache.index_copy_(0, idx, x)
        return
    q, sc = cache
    n, d = x.shape
    g = x.float().view(n, sc.shape[1], d // sc.shape[1])
    s = (g.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    # FP8 has no index_copy_/gather kernels: move the bytes as uint8 (the same bits)
    q.view(torch.uint8).index_copy_(0, idx, (g / s).to(torch.float8_e4m3fn).view(n, d).view(torch.uint8))
    sc.index_copy_(0, idx, s.view(n, -1))


def kv_read(cache, idx, dtype=torch.bfloat16):
    """Rows at idx (a tensor of any shape, or an int) back in `dtype`."""
    if isinstance(cache, torch.Tensor):
        return cache[idx].to(dtype)
    q, sc = cache
    v, s = q.view(torch.uint8)[idx].view(torch.float8_e4m3fn), sc[idx]
    d, b = v.shape[-1], s.shape[-1]
    # Straight to `dtype`: an fp32 intermediate doubled the gather's traffic (long prefill 1.66x).
    return (v.to(dtype).view(*v.shape[:-1], b, d // b) * s[..., None].to(dtype)).view(v.shape)


def kv_span(cache, S, dtype=torch.bfloat16):
    """Rows [0, S) in `dtype`."""
    if isinstance(cache, torch.Tensor):
        return cache[:S].to(dtype)
    q, sc = cache
    d, b = q.shape[1], sc.shape[1]
    return (q[:S].to(dtype).view(S, b, d // b) * sc[:S, :, None].to(dtype)).view(S, d)


class Arena:
    """This rank's half of resident experts (routed, shared, dense pseudo-experts) in MXFP4."""

    def __init__(self, inter, hidden, slots, rank, world, device):
        self.inter, self.hidden, self.slots, self.rank, self.world = inter, hidden, slots, rank, world
        self.Il, self.Hl = inter // world, hidden // world
        u8 = dict(dtype=torch.uint8, device=device)
        self.w1 = torch.zeros((slots, self.Il, hidden // 2), **u8)
        self.s1 = torch.zeros((slots, self.Il, hidden // GROUP), **u8)
        self.w3 = torch.zeros((slots, self.Il, hidden // 2), **u8)
        self.s3 = torch.zeros((slots, self.Il, hidden // GROUP), **u8)
        self.w2 = torch.zeros((slots, self.Hl, inter // 2), **u8)
        self.s2 = torch.zeros((slots, self.Hl, inter // GROUP), **u8)

    @staticmethod
    def slot_bytes(inter, hidden, world):
        return 2 * (inter // world) * (hidden // 2 + hidden // GROUP) + (hidden // world) * (inter // 2 + inter // GROUP)

    @property
    def bytes_per_slot(self):
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    def put(self, slot, gate, up, down):
        """gate/up: (packed, scale) of this rank's intermediate rows; down: (packed, scale) of
        this rank's output rows over the full intermediate."""
        for (w, s), (dw, ds) in ((gate, (self.w1, self.s1)), (up, (self.w3, self.s3)), (down, (self.w2, self.s2))):
            dw[slot].copy_(w.view(torch.uint8))
            ds[slot].copy_(s.view(torch.uint8))

    def dequant(self, slot):
        return (dequant_mxfp4(self.w1[slot], self.s1[slot]).to(torch.bfloat16),
                dequant_mxfp4(self.w3[slot], self.s3[slot]).to(torch.bfloat16),
                dequant_mxfp4(self.w2[slot], self.s2[slot]).to(torch.bfloat16))


class GlmTP:
    def __init__(self, model_dir: str, *, max_seq: int | None = None, chunk: int | None = None,
                 slots: int | str | None = None, miss: str | None = None, log=print):
        self.comm = Comm()
        self.rank, self.world, self.dev = self.comm.rank, self.comm.world, self.comm.device
        self.log = log if self.rank == 0 else (lambda *_: None)
        need = float(env("MUSE_MIN_FREE_GB", "90"))
        if need and mem_available() < need * 1e9:
            raise RuntimeError(f"only {mem_available() / 1e9:.1f} GB MemAvailable, need {need:.0f}")
        self.cfg = cfg = GlmConfig.load(model_dir)
        self.ck = GlmCheckpoint(model_dir)
        self.max_seq = int(max_seq or env("MUSE_MAX_SEQ", "32768"))
        # MUSE_KV=fp8: the MLA latent (512) and the indexer keys (128) are cached as FP8 E4M3 with
        # one fp32 scale per 128 values, the MTP hidden ring as FP8 with one per row; the RoPE keys
        # (64) stay BF16. ~54 KB a position a rank instead of ~97 (+ 6 KB of ring instead of 12).
        self.kv8 = env("MUSE_KV", "bf16") == "fp8"
        # MUSE_INDEX_KERNEL=1: the indexer's scores in one Triton kernel (engine/glm/index_kernel.py)
        # instead of fp32 einsums over [T, 32, S]; decode scored all max_seq positions that way.
        self.index_kernel = env("MUSE_INDEX_KERNEL", "1") == "1"
        # MUSE_ATTN_KERNEL=1: the sparse attention in one Triton kernel (engine/glm/sparse_attn.py)
        # instead of gathering each 32-row block's selected latents into BF16 copies.
        self.attn_kernel = env("MUSE_ATTN_KERNEL", "1") == "1"
        self.exl3_attn = env("MUSE_EXL3_ATTN", "0") == "1"
        # MUSE_GATHER_MERGE=1: MoE layers gather the routed and the fixed experts' intermediates
        # in one all-gather and their weighted, summed outputs (BF16) in another (mlp()).
        self.gather_merge = env("MUSE_GATHER_MERGE", "1") == "1"
        # MUSE_FIXED_DENSE=1: in prefill the fixed experts (shared, dense layers) as BF16 GEMMs on
        # weights dequantized once (engine/glm/mxfp4_dense.py) -- only where they are all of the
        # call's slots (EXL3 config, dense layers); MXFP4's routed picks share the MoE kernel.
        self.fixed_dense = env("MUSE_FIXED_DENSE", "1") == "1"
        self.chunk = int(chunk or env("MUSE_CHUNK", "1024"))
        self.miss = miss or env("MUSE_MISS", "substitute")
        self.native = env("MUSE_NATIVE_MOE", "1") == "1"
        self.prefill_moe = env("MUSE_PREFILL_MOE", "triton")
        self.use_graphs = env("MUSE_GRAPHS", "1") == "1"
        self.decode_rows, self.graph_rows = 16, {1}
        W, r = self.world, self.rank
        if cfg.heads % W or cfg.vocab % W or cfg.moe_inter % W or cfg.hidden % W:
            raise ValueError("dimensions must split evenly over the ranks")
        self.hl, self.vl = cfg.heads // W, cfg.vocab // W
        self.mtp_on = (env("MUSE_MTP", "1") == "1" and cfg.mtp_layers > 0
                       and self.ck_has_mtp(model_dir, cfg))
        self.mtp_L = cfg.layers
        self.nL = cfg.layers + (1 if self.mtp_on else 0)
        self.moe_layers = [L for L in range(self.nL) if not self.dense(L)]
        self.n_pseudo = cfg.inter // cfg.moe_inter
        fixed = len(self.moe_layers) * cfg.shared_experts + sum(cfg.dense_mlp) * self.n_pseudo
        # MUSE_EXPERT_FORMAT=exl3: the decoder layers' routed experts come from the EXL3 pack at
        # MUSE_EXL3_PACK ({rank} filled in) into their own arena (engine/glm/exl3.py). The shared
        # experts, the dense layers and the MTP layer's experts (4-bit in that checkpoint, a width
        # the routed arena does not hold) stay MXFP4, the MTP layer's all resident. Slot numbers
        # run through both: MXFP4 [0, x3_off), EXL3 from x3_off on.
        self.xfmt = env("MUSE_EXPERT_FORMAT", "mxfp4")
        assert self.xfmt in ("mxfp4", "exl3"), self.xfmt
        self.x3 = self.x3pack = None
        if self.xfmt == "exl3":
            from .exl3 import Exl3Pack
            self.x3pack = Exl3Pack(env("MUSE_EXL3_PACK", "").format(rank=r), cfg.hidden, cfg.moe_inter)
        # MUSE_MTP_FULL=1: the MTP layer's routed experts are all resident (MXFP4, 2.5 GB a rank)
        # and out of the swaps: drafts never substitute, and decode-layer residency no longer
        # trades against the drafter. Always so under exl3 (the arena cannot hold their 4 bits).
        self.mtp_full = self.mtp_on and (self.x3pack is not None or env("MUSE_MTP_FULL", "1") == "1")
        if self.mtp_full:
            fixed += cfg.experts
        self.swap_layers = [L for L in self.moe_layers if not (self.mtp_full and L == self.mtp_L)]
        total = len(self.swap_layers) * cfg.experts
        # Prefill prefetch (MUSE_PF=1): two buffers of MUSE_PF_SLOTS transient slots, one per
        # alternate layer, taken from the resident budget. See _pf_predict.
        self.pf_on = env("MUSE_PF", "0") == "1" and self.miss == "substitute"
        self.pf_n = int(env("MUSE_PF_SLOTS", "32")) if self.pf_on else 0
        self.pf_min_rows = int(env("MUSE_PF_MIN_ROWS", "128"))
        # MUSE_PF_KEEP=1: after its layer, a prefetched expert replaces the coldest resident
        # (lowest long-term score, any layer, never a layer below 2 x top-k) instead of being
        # dropped, so later chunks, the post-prefill adaptation and decode reuse the read.
        self.pf_keep = env("MUSE_PF_KEEP", "0") == "1"
        self.pf_keep_margin = float(env("MUSE_PF_KEEP_MARGIN", "1.5"))
        # Hot misses (MUSE_HOT=N, eager decode): per layer, read up to N missed experts whose
        # router weight share is at least MUSE_HOT_MIN synchronously and compute them exactly.
        # They share the transient slots with the prefill prefetch (never both at once).
        self.hot_n = int(env("MUSE_HOT", "0")) if self.miss == "substitute" else 0
        self.hot_min = float(env("MUSE_HOT_MIN", "0.2"))
        # Graphed decode: the threshold rises above hot_min to keep about MUSE_HOT_TARGET reads a
        # step (an MTP verify step of 3 rows finds a >= 0.2 miss in about half the layers), and
        # never more than twice that. Both ranks count the same reads, so they keep the same one.
        self.hot_target = float(env("MUSE_HOT_TARGET", "4"))
        # MUSE_HOT_MAX: at most this many graphed hot reads a step, whatever the threshold
        # (0: twice MUSE_HOT_TARGET, or no cap without a target). Earlier layers take it first.
        self.hot_max = int(env("MUSE_HOT_MAX", "0"))
        # MUSE_HOT_PACE=g (with a cap): the i-th of n MoE layers may bring the step's reads up to
        # ceil(cap * ((i+1)/n)^g) -- unspent budget carries forward, g=1 spreads it evenly, g>1
        # holds more of it for the last layers. 0: first come, first served.
        self.hot_pace = float(env("MUSE_HOT_PACE", "0"))
        self.hot_pace_lead = int(env("MUSE_HOT_PACE_LEAD", "0"))   # reads a layer may run ahead of the pace
        # MUSE_SUB_WEIGHT: weights of a substituted top-k. renorm (the router's own rule over the
        # picks it ended with); exact (resident picks keep their exact weights, substitutes get
        # their own score over the exact picks' total); transfer (each substitute takes a missed
        # pick's exact weight, heaviest to the best-scored substitute).
        self.sub_weight = env("MUSE_SUB_WEIGHT", "renorm")
        assert self.sub_weight in ("renorm", "exact", "transfer"), self.sub_weight
        dec = [L for L in self.moe_layers if L < cfg.layers]
        self._hot_frac = {L: ((i + 1) / len(dec)) ** self.hot_pace for i, L in enumerate(dec)}
        self._hot_quarter = {L: 4 * i // len(dec) for i, L in enumerate(dec)}
        self.hot_thr = {}
        # MUSE_HOT_KEEP=N: hot reads stay resident instead of being dropped after their layer. Each
        # background period (MUSE_BG=1) rank 0 queues up to N victims after its own plan's -- cold,
        # lowest long-term score first -- and each hot read takes the next one (_hot_keep).
        self.hot_keep = int(env("MUSE_HOT_KEEP", "0")) if self.hot_n else 0
        # MUSE_HOT_KEEP_AUDIT=1 (measurement only, rank 0, syncs per keep): how many kept reads a
        # rank gate -- adapt()'s rule, score beats the victim's by MUSE_ADAPT_MARGIN -- would drop.
        self.hot_audit = env("MUSE_HOT_KEEP_AUDIT", "0") == "1" and self.rank == 0
        self.hot_q = []                 # [(VL, ve, slot)] victims for kept hot reads, weakest last
        self.hot_live = None            # [(e, transient slot)] hot reads of the layer being computed
        self._pre_route = {}            # L -> (x, scores, choice) from _hot_candidate, for _route
        self._warming = False           # warmup(): no adaptation or background step
        transient = cfg.experts if self.miss == "stream" else max(2 * self.pf_n, self.hot_n)
        t0 = time.perf_counter()
        self._load_dense()
        self._alloc_cache()
        per = Arena.slot_bytes(cfg.moe_inter, cfg.hidden, W)
        mx_bytes = 0                       # MXFP4 bytes outside the routed budget (exl3: fixed + MTP)
        if self.x3pack:
            from .exl3 import slot_bytes
            mx_bytes, per = fixed * per, slot_bytes(r, W, cfg.hidden, cfg.moe_inter)
        want = slots if slots is not None else env("MUSE_SLOTS", "auto")
        if want == "auto":
            self.ck.drop_cache()
            free = max(torch.cuda.mem_get_info(self.dev)[0], mem_available())
            reserve = float(env("MUSE_RESERVE_GB", "20")) * 1e9
            share = float(env("MUSE_ARENA_SHARE", "0.88"))
            mine = max(0, int(share * (free - reserve - mx_bytes) // per) - transient - (0 if self.x3pack else fixed))
            want = min(self.comm.gather_objects(mine))
            self.log(f"[glm] auto slots: {want} routed + {fixed} fixed ({free / 1e9:.1f} GB available, "
                     f"{reserve / 1e9:.0f} GB floor, share {share}, {per / 1e6:.1f} MB a slot)")
        self.slots_resident = min(int(want), total)
        self._boot_guard(transient)
        if self.x3pack:
            from .exl3 import make_arena
            self.arena = Arena(cfg.moe_inter, cfg.hidden, fixed, r, W, self.dev)
            self.x3 = make_arena(self.slots_resident + transient, r, W, self.dev, cfg.hidden, cfg.moe_inter)
            self.x3_off = fixed
            self.n_slots = fixed + self.x3.slots
            orig_drop = self.ck.drop_cache
            self.ck.drop_cache = lambda: (orig_drop(), self.x3pack.drop_cache())
        else:
            self.arena = Arena(cfg.moe_inter, cfg.hidden, fixed + self.slots_resident + transient, r, W, self.dev)
            self.x3_off = None
            self.n_slots = self.arena.slots
        self.lut = torch.full((self.nL, cfg.experts), -1, dtype=torch.int32, device=self.dev)
        self.owner: list[tuple[int, int] | None] = [None] * self.n_slots
        self._place_fixed()
        self._place_resident()
        if self.x3 is not None:
            from .exl3 import warm
            warm(self.x3)                  # nvcc build + pointer tables now, not inside a capture
        first_t = fixed + self.slots_resident
        self.transient = list(range(first_t, first_t + transient))
        self.pf_slots = self.transient if self.pf_on else []
        self.hot_stats = {"reads": 0, "layers": 0, "s": 0.0, "kept": 0, "unkept": 0, "quarters": [0, 0, 0, 0]}
        self.pf_pending = {}            # layer -> [(e, slot, future)] submitted by the layer before
        self.pf_live = None             # (layer, [experts]) installed for the layer being computed
        self.pf_stats = {"chunks": 0, "planned": 0, "used_mass": 0.0, "wait_s": 0.0, "reads": 0}
        self.allowed = (self.lut >= 0) if self.miss == "substitute" else None
        self.pos_t = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.graphs: dict[int, tuple] = {}
        self.stats = {"stream_loads": 0, "stream_s": 0.0}
        self.miss_stats = None
        self.miss_hist = None          # [2, 50]: missed picks and their mass by router weight / routed scale
        self.route_dump = None         # list: per MoE layer call (L, ids0, w0, missed, {L+d: ids predicted from this x})
        self.record = None
        self.demand = None             # [layers, E] router mass wanted since the last adapt()
        self.mtp_fill = True
        self.spec_stats = {"rounds": 0, "accepted": 0}
        self.spec_cal = None            # [(draft index, confidence product, accepted)] when a tool asks
        self.mtp_dyn = float(env("MUSE_MTP_DYN", "0"))
        self.mtp_graph_rows = 1
        if self.mtp_on and self.x3 is not None and int(env("MUSE_MTP_DEPTH", "3")) > 7:
            # exl3.routed's graph-safe decode path takes <= 64 picks (8 rows x top-8); a deeper
            # verify step would take the prefill routing (host syncs) inside a graph capture.
            raise ValueError("MUSE_MTP_DEPTH > 7 is not supported with MUSE_EXPERT_FORMAT=exl3")
        if self.mtp_on:
            self.graph_rows = {1, int(env("MUSE_MTP_DEPTH", "3")) + 1}
            self.mtp_graph_rows = max(1, int(env("MUSE_MTP_DEPTH", "3")))
        self.adapt_every = int(env("MUSE_ADAPT_EVERY", "200"))
        self.adapt_prefill_swaps = int(env("MUSE_ADAPT_PREFILL_SWAPS", "2048"))
        self.adapt_decode_swaps = int(env("MUSE_ADAPT_DECODE_SWAPS", "256"))
        self.adapt_margin = float(env("MUSE_ADAPT_MARGIN", "1.5"))
        self.adapt_log = []
        self.adapt_cross_layer = env("MUSE_ADAPT_CROSS_LAYER", "1") == "1"
        # Expert reads that bypass the page cache (O_DIRECT) need no drop_cache() after a swap.
        self.reads_direct = self.ck.direct and (self.x3pack is None or self.x3pack.dfd is not None)
        # Background expert reads during decode (MUSE_BG=1): every MUSE_BG_EVERY decode rounds,
        # rank 0 ranks the non-resident experts by their recent router mass (decayed by
        # MUSE_BG_DECAY a period), picks up to MUSE_BG_READS of them and a cold resident victim
        # for each; both ranks read them on the thread pool while decode goes on and install
        # them at the next period boundary. bg_step() has the rules.
        self.bg_on = env("MUSE_BG", "0") == "1"
        self.bg_every = max(1, int(env("MUSE_BG_EVERY", "2")))
        self.bg_reads = int(env("MUSE_BG_READS", "48"))
        self.bg_decay = float(env("MUSE_BG_DECAY", "0.7"))
        self.bg_protect = int(env("MUSE_BG_PROTECT", "8"))
        self.bg_cold = float(env("MUSE_BG_COLD", "0.01"))   # victims: recent router mass below this
        if env("MUSE_ADAPT", "1") == "1" and self.miss == "substitute":
            self.enable_adaptation(float(env("MUSE_ADAPT_DECAY", "0.9")))
        if env("MUSE_MISS_LOG", "1") == "1" and self.miss == "substitute":
            self.miss_stats = {"picks": 0, "missed": 0, "mass": 0.0, "mass_missed": 0.0}
        torch.cuda.synchronize()
        self.log(f"[glm] rank {r}/{W} ready in {time.perf_counter() - t0:.1f}s: dense {self.dense_bytes / 1e9:.2f} GB, "
                 f"{self.slots_resident} of {total} routed experts resident + {fixed} fixed "
                 f"({(self.arena.bytes_per_slot * self.arena.slots + (self.x3.total_bytes if self.x3 else 0)) / 1e9:.2f} GB"
                 f"{', routed EXL3' if self.x3 else ''}), miss={self.miss}, "
                 f"max_seq {self.max_seq}; {mem_available() / 1e9:.1f} GB available")
        self.reset()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def ck_has_mtp(model_dir, cfg):
        return GlmCheckpoint(model_dir).has(f"{P}layers.{cfg.layers}.eh_proj.weight")

    def full(self, L):
        return L >= self.cfg.layers or self.cfg.indexer_full[L]

    def dense(self, L):
        return L < self.cfg.layers and self.cfg.dense_mlp[L]

    def _boot_guard(self, transient):
        src = Path(__file__).resolve().parent
        digest = hashlib.sha256(b"".join((src / f).read_bytes() for f in
                                         ("tp.py", "ref.py", "weights.py", "config.py", "fp8_block.py",
                                          "index_kernel.py", "exl3.py", "sparse_attn.py",
                                          "norm_kernel.py", "direct_io.py", "mxfp4_dense.py", "exl3_attn.py"))).hexdigest()[:16]
        mine = {"version": VERSION, "code": digest, "miss": self.miss, "native": self.native,
                "prefill_moe": self.prefill_moe, "graphs": self.use_graphs, "max_seq": self.max_seq,
                "chunk": self.chunk, "slots": self.slots_resident, "transient": transient, "mtp": self.mtp_on,
                "head": env("MUSE_HEAD", "bf16"), "depth": env("MUSE_MTP_DEPTH", "3"), "db": env("MUSE_STATS_DB", ""),
                "xfmt": self.xfmt, "kv": env("MUSE_KV", "bf16"), "index_kernel": self.index_kernel, "gather_merge": self.gather_merge, "attn_kernel": self.attn_kernel, "fused_norm": FUSED_NORM, "h_ring": self.h_ring_n, "fixed_dense": self.fixed_dense, "exl3_attn": self.exl3_attn, "draft_head": env("MUSE_DRAFT_HEAD", "same"), "x3pack_dims": self.x3pack.reader.header.get("dims") if self.x3pack else None,
                "pf": [env(k, "") for k in ("MUSE_PF", "MUSE_PF_SLOTS", "MUSE_PF_MIN_ROWS", "MUSE_PF_KEEP", "MUSE_PF_KEEP_MARGIN", "MUSE_HOT", "MUSE_HOT_MIN",
                                            "MUSE_HOT_TARGET", "MUSE_HOT_KEEP", "MUSE_HOT_MAX", "MUSE_HOT_KEEP_AUDIT", "MUSE_HOT_PACE",
                                            "MUSE_HOT_PACE_LEAD", "MUSE_SUB_WEIGHT", "MUSE_MTP_FULL", "MUSE_MISS_LOG",
                                            "MUSE_MTP_DYN")],
                "adapt": [env(k, "") for k in ("MUSE_ADAPT", "MUSE_ADAPT_EVERY", "MUSE_ADAPT_PREFILL_SWAPS",
                                               "MUSE_ADAPT_DECODE_SWAPS", "MUSE_ADAPT_DECAY", "MUSE_ADAPT_CROSS_LAYER",
                                               "MUSE_ADAPT_MARGIN", "MUSE_BG", "MUSE_BG_EVERY", "MUSE_BG_READS",
                                               "MUSE_BG_DECAY", "MUSE_BG_PROTECT", "MUSE_BG_COLD", "MUSE_WARMUP")],
                "fast_comm": self.comm.fast,
                "config": json.dumps({k: v for k, v in self.cfg.extra["raw"].items()
                                      if k != "quantization_config"}, sort_keys=True)}
        allv = self.comm.gather_objects(mine)
        if any(v != allv[0] for v in allv):
            raise RuntimeError(f"glm ranks disagree: {[{k: v.get(k) for k in mine if v.get(k) != allv[0].get(k)} for v in allv]}")

    def _load_dense(self):
        cfg, d, ck, W, r = self.cfg, self.dev, self.ck, self.world, self.rank
        hq, hv = cfg.qk_dim, cfg.nope + cfg.v_dim
        self.embed = ck.get_rows(P + "embed_tokens.weight", r * self.vl, (r + 1) * self.vl).to(d)
        head = ck.get_rows("lm_head.weight", r * self.vl, (r + 1) * self.vl)
        # MUSE_DRAFT_HEAD=mxfp4: the MTP drafts score with a 4-bit copy of the head (engine/glm/
        # mxfp4_dense.gemv, ~0.25 GB a rank): the head is half of a draft's reads. Verification keeps
        # the main head, so this can only move acceptance.
        self.draft_head = None
        if self.mtp_on and env("MUSE_DRAFT_HEAD", "same") == "mxfp4":
            from .mxfp4_dense import quantize
            self.draft_head = quantize(head.to(d))
        if env("MUSE_HEAD", "bf16") == "fp8":
            # Halves the largest single read of a decode step (vocab 154880: ~0.95 GB a rank).
            from .weights import _fp8_quant
            head = FP8Block(*_fp8_quant(head.to(d)))    # (not hq, hs: hq is the q head width below)
        self.head = Lin(head, d)
        self.final_norm = ck.get(P + "norm.weight").to(d)
        self.L = []
        # MUSE_EXL3_ATTN=1: the decoder layers' q_a, q_b, kv_a, o_proj and indexer wq_b from the EXL3
        # checkpoint's 5-bit matrices (engine/glm/exl3_attn.py), mounted at MUSE_EXL3_ATTN_SRC; the
        # MTP layer keeps FP8. Splits follow the FP8 ones: q_b by heads (N), o_proj by heads (K).
        x3a = None
        if self.exl3_attn:
            from .exl3_attn import Exl3Source
            x3a = Exl3Source(env("MUSE_EXL3_ATTN_SRC", "/exl3src"))
        nbytes = self.embed.numel() * 2 + self.head.nbytes + (
            sum(t.numel() for t in self.draft_head) if self.draft_head is not None else 0)
        for L in range(self.nL):
            p = f"{P}layers.{L}."
            a = p + "self_attn."
            kvb = ck.fp8(a + "kv_b_proj.weight", (r * self.hl * hv, (r + 1) * self.hl * hv)).dequant(torch.bfloat16)
            kvb = kvb.view(self.hl, hv, cfg.kv_lora)
            x3l = x3a is not None and L < cfg.layers
            xa = f"model.layers.{L}.self_attn."
            o = None if x3l else ck.fp8(a + "o_proj.weight")
            lw = {
                "ln_in": ck.get(p + "input_layernorm.weight").to(d),
                "ln_post": ck.get(p + "post_attention_layernorm.weight").to(d),
                "q_a": x3a.lin(xa + "q_a_proj", d, 8) if x3l else Lin(ck.fp8(a + "q_a_proj.weight"), d),
                "q_norm": ck.get(a + "q_a_layernorm.weight").to(d),
                "q_b": x3a.lin(xa + "q_b_proj", d, 1, r, W, "n") if x3l else
                Lin(ck.fp8(a + "q_b_proj.weight", (r * self.hl * hq, (r + 1) * self.hl * hq)), d),
                "kv_a": x3a.lin(xa + "kv_a_proj_with_mqa", d, 8, n_out=cfg.kv_lora + cfg.rope_dim) if x3l else
                Lin(ck.fp8(a + "kv_a_proj_with_mqa.weight"), d),
                "kv_norm": ck.get(a + "kv_a_layernorm.weight").to(d),
                # Absorbed MLA: kv_b's key half [h, nope, c] and value half [h, v, c], BF16.
                "w_uk": kvb[:, :cfg.nope].contiguous().to(d),
                "w_uv": kvb[:, cfg.nope:].contiguous().to(d),
                "o": x3a.lin(xa + "o_proj", d, 2, r, W, "k") if x3l else
                Lin(o.cols(r * self.hl * cfg.v_dim, (r + 1) * self.hl * cfg.v_dim), d),
            }
            if L == self.mtp_L:
                lw.update({"enorm": ck.get(p + "enorm.weight").to(d), "hnorm": ck.get(p + "hnorm.weight").to(d),
                           "sh_norm": ck.get(p + "shared_head.norm.weight").to(d),
                           # rows split over the ranks + a gather: half the bytes of a draft step's 151 MB
                           "eh": Lin(ck.get_rows(p + "eh_proj.weight", r * cfg.hidden // W, (r + 1) * cfg.hidden // W), d)})
            if self.full(L):
                i = a + "indexer."
                lw.update({"i_q": x3a.lin(xa + "indexer.wq_b", d, 1) if x3l else Lin(ck.fp8(i + "wq_b.weight"), d),
                           "i_k": Lin(ck.fp8(i + "wk.weight"), d),
                           "i_kn_w": ck.get(i + "k_norm.weight").to(d).float(),
                           "i_kn_b": ck.get(i + "k_norm.bias").to(d).float(),
                           "i_w": ck.get(i + "weights_proj.weight").to(d).float()})
            if not self.dense(L):
                lw["router"] = ck.get(p + "mlp.gate.weight").to(d).float()
                lw["bias"] = ck.get(p + "mlp.gate.e_score_correction_bias").to(d).float()
            nbytes += sum(v.nbytes for v in lw.values() if hasattr(v, "nbytes") and not isinstance(v, torch.Tensor))
            nbytes += (lw["w_uk"].numel() + lw["w_uv"].numel()) * 2
            self.L.append(lw)
            del kvb, o
        self.dense_bytes = nbytes
        if x3a is not None:
            x3a.drop_cache()
        D = cfg.rope_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float, device=d) / D))
        f = torch.arange(self.max_seq, device=d, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((f, f), -1)
        self.cos, self.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        self.ck.drop_cache()

    def _alloc_cache(self):
        cfg, kw = self.cfg, dict(dtype=torch.bfloat16, device=self.dev)
        S, f8 = self.max_seq, dict(dtype=torch.float8_e4m3fn, device=self.dev)
        f32 = dict(dtype=torch.float32, device=self.dev)
        if self.kv8:
            self.c_cache = [(torch.zeros((S, cfg.kv_lora), **f8), torch.zeros((S, cfg.kv_lora // KV_BLOCK), **f32))
                            for _ in range(self.nL)]
            self.i_cache = {L: (torch.zeros((S, cfg.index_dim), **f8), torch.zeros((S, cfg.index_dim // KV_BLOCK), **f32))
                            for L in range(self.nL) if self.full(L)}
        else:
            self.c_cache = [torch.zeros((S, cfg.kv_lora), **kw) for _ in range(self.nL)]
            self.i_cache = {L: torch.zeros((S, cfg.index_dim), **kw) for L in range(self.nL) if self.full(L)}
        self.r_cache = [torch.zeros((S, cfg.rope_dim), **kw) for _ in range(self.nL)]
        # Post-norm target hidden of every position: the MTP layer's input when a follow-up
        # turn reuses the cache and its last reused position gets a new next token. It covers
        # max_seq (~12 KB a position, 400 MB at 32K): a ring of one chunk refused reuse whenever
        # the previous turn ran a chunk past the shared prefix -- every chat turn whose answer
        # (reasoning included) passed 1,024 tokens re-prefilled the whole conversation.
        # MUSE_H_RING (default 16,384) bounds it: a reuse boundary older than the ring skips the
        # MTP redo -- that one draft position keeps its old entry, which can only cost
        # acceptance (the target verifies every draft). At 128K the full ring was 0.8 GB a rank.
        self.h_ring_n = min(self.max_seq, int(env("MUSE_H_RING", "16384")))
        if self.kv8:
            self.h_ring = (torch.zeros((self.h_ring_n, cfg.hidden), **f8), torch.zeros((self.h_ring_n, 1), **f32))
        else:
            self.h_ring = torch.zeros((self.h_ring_n, cfg.hidden), **kw)

    # ------------------------------------------------------------------ experts
    def _expert_parts(self, L, e, urgent=False):
        """(gate, up, down) of this rank for routed expert e (or "shared") of layer L; the six
        tensors are read together (get_rows_many)."""
        if self.x3pack and e != "shared" and L != self.mtp_L:
            return self.x3pack.read(L, e, urgent)
        r, Il, Hl = self.rank, self.arena.Il, self.arena.Hl
        b = f"{P}layers.{L}.mlp." + ("shared_experts." if e == "shared" else f"experts.{e}.")
        specs = []
        for part, a, z in (("gate_proj", r * Il, (r + 1) * Il), ("up_proj", r * Il, (r + 1) * Il),
                           ("down_proj", r * Hl, (r + 1) * Hl)):
            specs += [(b + part + ".weight", a, z), (b + part + ".weight_scale", a, z)]
        t = self.ck.get_rows_many(specs, urgent)
        return (t[0], t[1]), (t[2], t[3]), (t[4], t[5])

    def _read_urgent(self, L, experts):
        """Hot-miss reads: a decode step is waiting, so neither the expert pool nor the piece
        pool is shared with background reads (those can hold 48 experts in the queue)."""
        if len(experts) == 1:                    # no pool hop for the usual single read
            return [self._expert_parts(L, experts[0], True)]
        if not hasattr(self, "_hot_pool"):
            from concurrent.futures import ThreadPoolExecutor
            self._hot_pool = ThreadPoolExecutor(max_workers=4)
        return list(self._hot_pool.map(lambda e: self._expert_parts(L, e, True), experts))

    def _pseudo_parts(self, L, j):
        """Dense MLP of layer L as pseudo-expert j: intermediate columns [j*I, (j+1)*I)."""
        cfg, r, Il, Hl = self.cfg, self.rank, self.arena.Il, self.arena.Hl
        I = cfg.moe_inter
        b = f"{P}layers.{L}.mlp."
        lo = j * I + r * Il
        gate = self.ck.mx_rows(b + "gate_proj.weight", lo, lo + Il)
        up = self.ck.mx_rows(b + "up_proj.weight", lo, lo + Il)
        w, s = self.ck.mx_rows(b + "down_proj.weight", r * Hl, (r + 1) * Hl)
        down = (w[:, j * I // 2:(j + 1) * I // 2].contiguous(), s[:, j * I // GROUP:(j + 1) * I // GROUP].contiguous())
        return gate, up, down

    def _place_fixed(self):
        cfg = self.cfg
        slot = 0
        self.fixed: dict[int, torch.Tensor] = {}
        for L in range(self.nL):
            if self.dense(L):
                ids = []
                for j in range(self.n_pseudo):
                    self.arena.put(slot, *self._pseudo_parts(L, j))
                    ids.append(slot)
                    slot += 1
            else:
                self.arena.put(slot, *self._expert_parts(L, "shared"))
                ids = [slot]
                slot += 1
            self.fixed[L] = torch.tensor(ids, dtype=torch.int32, device=self.dev)
        if self.mtp_full:
            L = self.mtp_L
            for e, parts in zip(range(cfg.experts), self._read_many(L, list(range(cfg.experts)))):
                self.arena.put(slot, *parts)
                self.owner[slot] = (L, e)
                self.lut[L, e] = slot
                slot += 1
        self.first_routed = slot
        self.ck.drop_cache()

    def _db_resident(self):
        """Cross-layer top-N by the stats DB's accumulated router mass (rank 0 reads, broadcasts)."""
        plan = None
        if self.rank == 0 and self.db_mass is not None and self.db_tokens > 0:
            layers = torch.tensor(self.swap_layers)
            m = self.db_mass[layers].clone()
            E, k2 = self.cfg.experts, 2 * self.cfg.topk
            plan = {int(L): [] for L in layers}
            # Every layer first gets its 2 x top-k best, then the rest goes by global mass.
            for li, L in enumerate(self.swap_layers):
                top = torch.argsort(m[li], descending=True)[:k2].tolist()
                plan[L] += top
                m[li, top] = -1.0
            left = self.slots_resident - sum(len(v) for v in plan.values())
            for idx in torch.argsort(m.reshape(-1), descending=True)[:max(0, left)].tolist():
                li, e = divmod(idx, E)
                plan[self.swap_layers[li]].append(e)
            plan = {L: sorted(v) for L, v in plan.items()}
        return self.comm.broadcast_obj(plan)

    def _load_db(self):
        self.db_path = env("MUSE_STATS_DB", "")
        self.db_mass, self.db_tokens = None, 0
        if self.rank == 0 and self.db_path and Path(self.db_path).exists():
            db = torch.load(self.db_path, map_location="cpu", weights_only=True)
            if tuple(db["mass"].shape) == (self.nL, self.cfg.experts):
                self.db_mass, self.db_tokens = db["mass"].double(), int(db["tokens"])
        if self.rank == 0 and self.db_path and self.db_mass is None:
            self.db_mass = torch.zeros((self.nL, self.cfg.experts), dtype=torch.float64)

    def flush_db(self):
        """Fold the demand since the last adapt() into the DB and save it (rank 0; the server calls
        this on SIGTERM, so a stop does not lose the last request's routing)."""
        if self.rank != 0 or self.demand is None or getattr(self, "db_mass", None) is None:
            return
        self.db_mass += self.demand.double().cpu()
        self.db_tokens += int(self.demand_tokens)
        self.demand_tokens = 0
        self.demand.zero_()
        self.save_db()

    def save_db(self):
        if self.rank != 0 or not self.db_path or self.db_mass is None:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path + ".tmp"
        torch.save({"version": 1, "mass": self.db_mass, "tokens": self.db_tokens}, tmp)
        os.replace(tmp, self.db_path)

    def _place_resident(self, resident=None):
        cfg = self.cfg
        self._load_db()
        if resident is None and not env("MUSE_RESIDENT", ""):
            resident = self._db_resident()
            if resident:
                self.log(f"[glm] resident set from the stats DB ({self.db_tokens} tokens of routing)")
        if resident is None and env("MUSE_RESIDENT", ""):
            raw = json.loads(Path(env("MUSE_RESIDENT", "")).read_text())
            raw = raw.get("resident", raw)
            resident = {int(k): [int(e) for e in v] for k, v in raw.items()}
        if resident is not None:
            # A trace of prefill alone never routes through the MTP layer: give layers the file
            # leaves out the remaining slots by the placeholder ranking, at least top-k each.
            absent = [L for L in self.swap_layers if L not in resident]
            left = self.slots_resident - sum(len(v) for v in resident.values())
            if absent:
                per = left // len(absent)
                if per < cfg.topk:
                    raise ValueError(f"{left} slots left for {len(absent)} layers missing from the resident set")
                for L in absent:
                    order = torch.argsort(self.L[L]["bias"].cpu(), stable=True).tolist()
                    resident[L] = sorted(order[:per])
                self.log(f"[glm] resident set lacks layers {absent}: {per} placeholder experts each")
        if resident is None:
            total = len(self.swap_layers) * cfg.experts
            if self.slots_resident >= total:
                resident = {L: list(range(cfg.experts)) for L in self.swap_layers}
            else:
                per = self.slots_resident // len(self.swap_layers)
                if per < cfg.topk:
                    raise ValueError(f"{self.slots_resident} slots give {per} experts a layer, fewer than "
                                     f"top-{cfg.topk}: substitution could not fill the routing")
                extra = self.slots_resident - per * len(self.swap_layers)
                resident = {}
                for k, L in enumerate(self.swap_layers):
                    n = per + (1 if k < extra else 0)
                    order = torch.argsort(self.L[L]["bias"].cpu(), stable=True).tolist()
                    resident[L] = sorted(order[:n])
        slot = self.first_routed
        t0 = time.perf_counter()
        for L in self.swap_layers:
            experts = resident.get(L, [])
            if slot + len(experts) > self.first_routed + self.slots_resident:
                raise ValueError("resident set larger than the slots")
            for e, parts in zip(experts, self._read_many(L, experts)):
                self._load(L, e, slot, parts)
                slot += 1
            self.ck.drop_cache()
        self.log(f"[glm] placed {slot - self.first_routed} routed experts in {time.perf_counter() - t0:.1f}s; "
                 f"{mem_available() / 1e9:.1f} GB available, {rss_anon() / 1e9:.1f} GB anon")

    def _read_many(self, L, experts):
        """Read several experts' parts at once: preadv releases the GIL, so a thread pool keeps
        the NVMe queue full (one at a time ran at ~0.7 GB/s)."""
        if not hasattr(self, "_pool"):
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=int(env("MUSE_READ_THREADS", "12")))
        return list(self._pool.map(lambda e: self._expert_parts(L, e), experts))

    def _put(self, slot, parts):
        if self.x3 is not None and slot >= self.x3_off:
            rec, bits = parts
            self.x3.load_slot(slot - self.x3_off, rec, bits)
        else:
            self.arena.put(slot, *parts)

    def _copy_slot(self, dst, src):
        if self.x3 is not None and dst >= self.x3_off:
            from .exl3 import copy_slot
            copy_slot(self.x3, dst - self.x3_off, src - self.x3_off)
        else:
            A = self.arena
            for t in (A.w1, A.s1, A.w3, A.s3, A.w2, A.s2):
                t[dst].copy_(t[src])

    def _load(self, L, e, slot, parts=None):
        self._put(slot, parts or self._expert_parts(L, e))
        old = self.owner[slot]
        if old is not None and int(self.lut[old[0], old[1]]) == slot:
            self.lut[old[0], old[1]] = -1
        self.owner[slot] = (L, e)
        self.lut[L, e] = slot

    def reset(self):
        self.pos = 0
        self.tokens: list[int] = []

    def enable_adaptation(self, decay: float = 0.9):
        self.demand_tokens = 0
        self.adapt_epoch = 0
        self.slot_epoch = [-10] * self.n_slots              # adaptation that last loaded a slot
        self.adapt_totals = {"swaps": 0, "churn": 0}
        self.demand = torch.zeros((self.nL, self.cfg.experts), dtype=torch.float32, device=self.dev)
        self.score = torch.zeros_like(self.demand)        # decayed history the swaps rank by
        self.decay = decay
        self.bg_score = torch.zeros_like(self.demand)     # short-term decayed router mass (background reads)
        self.bg_last = torch.zeros_like(self.demand)      # demand at the last background period
        self.bg_pending = []                              # [(L, e, VL, ve, future)] read, not installed
        self.bg_period = 0
        self.bg_until = torch.zeros((self.nL, self.cfg.experts), dtype=torch.long, device=self.dev)  # no eviction before
        self.pf_kept = torch.zeros((self.nL, self.cfg.experts), dtype=torch.bool, device=self.dev)   # this prefill's promotions
        self._bg_layers = torch.tensor(self.swap_layers, device=self.dev)
        self.bg_stats = {"periods": 0, "planned": 0, "installed": 0, "stale": 0, "wait_s": 0.0, "install_s": 0.0}

    @torch.no_grad()
    def adapt(self, max_swaps: int = 4096, margin: float | None = None):
        """Swap toward recent demand between requests. Per MoE layer: the non-resident experts
        with the most decayed router mass replace the resident ones with the least, while the
        newcomer's score beats the victim's by ``margin``. Rank 0 decides; both ranks load the
        same experts into the same slots. Returns (swaps, seconds)."""
        t0 = time.perf_counter()
        margin = self.adapt_margin if margin is None else margin
        self.adapt_epoch += 1
        churn = 0
        if self.rank == 0 and getattr(self, "db_mass", None) is not None:
            self.db_mass += self.demand.double().cpu()
            self.db_tokens += int(self.demand_tokens)
        self.demand_tokens = 0
        self.score.mul_(self.decay).add_(self.demand)
        self.demand.zero_()
        if getattr(self, "bg_last", None) is not None:
            self.bg_last.zero_()                          # demand restarts from zero
        plan = None
        if self.rank == 0:
            sc = self.score.cpu()
            lut = self.lut.cpu()
            layers = torch.tensor(self.swap_layers)
            res = lut[layers] >= 0                                   # [moe layers, E]
            s = sc[layers]
            if self.adapt_cross_layer:
                # One ranking over all layers: router mass is comparable across layers (each
                # token's top-k weights sum to the routed scale), and layers whose routing is
                # diffuse should get slots from layers whose routing is concentrated.
                cand = torch.argsort(s.masked_fill(res, -1.0).reshape(-1), descending=True)
                vict = torch.argsort(s.masked_fill(~res, float("inf")).reshape(-1))
                left = res.sum(1).clone()
                plan, vi = [], 0
                E = self.cfg.experts
                for ci in cand.tolist():
                    if len(plan) >= max_swaps:
                        break
                    cl, ce = divmod(ci, E)
                    if res[cl, ce] or s[cl, ce] <= 0:
                        break
                    while vi < vict.numel():
                        vl, ve = divmod(int(vict[vi]), E)
                        vi += 1
                        if res[vl, ve] and left[vl] > 2 * self.cfg.topk:
                            break
                    else:
                        break
                    if s[cl, ce] <= s[vl, ve] * margin:
                        break
                    left[vl] -= 1
                    left[cl] += 1
                    plan.append((int(layers[cl]), ce, int(layers[vl]), ve))
            else:
                plan = []
                for li, L in enumerate(self.swap_layers):
                    cand = torch.argsort(s[li].masked_fill(res[li], -1.0), descending=True).tolist()
                    vict = torch.argsort(s[li].masked_fill(~res[li], float("inf"))).tolist()
                    for c, v in zip(cand, vict):
                        if res[li, c] or not res[li, v] or s[li, c] <= s[li, v] * margin or s[li, c] <= 0:
                            break
                        plan.append((L, c, L, v))
                plan.sort(key=lambda t: -float(sc[t[0], t[1]] - sc[t[2], t[3]]))
                plan = plan[:max_swaps]
        plan = self.comm.broadcast_obj(plan)
        # The lut changes on a host copy and goes back once (per swap, a device read and two
        # device writes); reads and installs go in waves, so the pooled read buffers are reused
        # instead of 256 records (1.8 GB) being held at once.
        lut, moves = self.lut.cpu(), []
        for L, c, VL, v in plan:
            slot = int(lut[VL, v])
            # Churn: evicting what one of the last two adaptations just brought in.
            churn += self.adapt_epoch - self.slot_epoch[slot] <= 2
            self.slot_epoch[slot] = self.adapt_epoch
            lut[VL, v] = -1
            lut[L, c] = slot
            moves.append((L, c, slot))
        if moves:
            if not hasattr(self, "_pool"):
                self._read_many(self.swap_layers[0], [])
            for i in range(0, len(moves), 16):
                wave = moves[i:i + 16]
                for (L, c, slot), pr in zip(wave, self._pool.map(lambda m: self._expert_parts(m[0], m[1]), wave)):
                    self._put(slot, pr)
                    self.owner[slot] = (L, c)
            self.lut.copy_(lut)
        if self.allowed is not None:
            self.allowed.copy_(self.lut >= 0)
        if not self.reads_direct:
            self.ck.drop_cache()                  # direct reads leave nothing in the page cache
        self.save_db()
        torch.cuda.synchronize()
        self.adapt_totals["swaps"] += len(plan)
        self.adapt_totals["churn"] += churn
        self.last_churn = churn
        return len(plan), time.perf_counter() - t0

    @torch.no_grad()
    def bg_step(self):
        """One background-read period boundary; both ranks call it at the same decode round.

        1. Install the previous period's reads: each rank waits for its own (normally done), then
           an item goes in only if its victim still holds the slot and the expert is still not
           resident -- decided on the shared lut, so both ranks agree.
        2. Rank 0 plans the next reads: candidates are non-resident experts by bg_score (router
           mass of the recent periods, decayed); victims are resident experts whose recent mass is
           below MUSE_BG_COLD (decay never reaches zero), not filled by a background read in the
           last MUSE_BG_PROTECT periods, the lowest long-term score first, never taking a layer
           below 2 x top-k. Broadcast, then both ranks
           submit the reads and return to decoding."""
        st = self.bg_stats
        self.bg_period += 1
        st["periods"] += 1
        items = self.bg_pending
        if items:
            t0 = time.perf_counter()
            parts = [f.result() for *_, f in items]
            t1 = time.perf_counter()
            st["wait_s"] += t1 - t0
            torch.cuda.synchronize()
            lut = self.lut.cpu()
            for (L, e, VL, ve, _), pr in zip(items, parts):
                slot = int(lut[VL, ve])
                if slot < 0 or int(lut[L, e]) >= 0 or self.owner[slot] != (VL, ve):
                    st["stale"] += 1
                    continue
                lut[VL, ve] = -1
                lut[L, e] = slot
                self.lut[VL, ve] = -1
                self.owner[slot] = None
                self._load(L, e, slot, pr)
                self.bg_until[L, e] = self.bg_period + self.bg_protect
                st["installed"] += 1
            if self.allowed is not None:
                self.allowed.copy_(self.lut >= 0)
            if st["installed"] // 512 != (st["installed"] - len(parts)) // 512:
                self.ck.drop_cache()                  # page cache shares the GPU's pool on GB10
            self.bg_pending = []
            st["install_s"] += time.perf_counter() - t1
        window = self.demand - self.bg_last
        self.bg_last.copy_(self.demand)
        self.bg_score.mul_(self.bg_decay).add_(window)
        plan, hot_q = None, None
        if self.rank == 0 and (self.bg_reads > 0 or self.hot_keep):
            # One pass on the GPU, one copy back: candidates, victims, per-layer counts.
            lay = self._bg_layers
            res = self.lut[lay] >= 0
            short = self.bg_score[lay]
            cold = res & (short < self.bg_cold) & (self.bg_until[lay] <= self.bg_period)
            cs = torch.topk(short.masked_fill(res, -1.0).reshape(-1), self.bg_reads)
            nv = min(4 * (self.bg_reads + self.hot_keep), cold.numel())
            vs = torch.topk((self.score + self.demand)[lay].masked_fill(~cold, float("inf")).reshape(-1), nv,
                            largest=False)
            c_val, c_idx, v_val, v_idx, v_slot, left = (
                t.cpu().tolist() for t in (cs.values, cs.indices, vs.values, vs.indices,
                                           self.lut[lay].reshape(-1)[vs.indices], res.sum(1)))
            E, floor = self.cfg.experts, 2 * self.cfg.topk
            plan, vi = [], 0
            for cv, ci in zip(c_val, c_idx):
                if cv <= 0:
                    break
                cl, ce = divmod(ci, E)
                while vi < nv and v_val[vi] != float("inf"):
                    vl, ve = divmod(v_idx[vi], E)
                    vi += 1
                    if left[vl] > floor:
                        break
                else:
                    break
                left[vl] -= 1
                left[cl] += 1
                plan.append((self.swap_layers[cl], ce, self.swap_layers[vl], ve))
            hot_q = []
            while len(hot_q) < self.hot_keep and vi < nv and v_val[vi] != float("inf"):
                vl, ve = divmod(v_idx[vi], E)
                if left[vl] > floor:
                    left[vl] -= 1
                    hot_q.append((self.swap_layers[vl], ve, v_slot[vi]))
                vi += 1
        plan, hot_q = self.comm.broadcast_obj((plan, hot_q))
        self.hot_q = hot_q[::-1] if hot_q else []
        if not hasattr(self, "_pool"):
            self._read_many(self.swap_layers[0], [])
        self.bg_pending = [(L, e, VL, ve, self._pool.submit(self._expert_parts, L, e)) for L, e, VL, ve in plan]
        st["planned"] += len(plan)

    # ------------------------------------------------------------------ attention
    def _indexer(self, L, x, q_resid, positions, cos, sin, S):
        """Top-k cached positions for each row (both ranks compute the same)."""
        cfg, lw = self.cfg, self.L[L]
        T = x.shape[0]
        q = lw["i_q"](q_resid).view(T, cfg.index_heads, cfg.index_dim)
        k = F.layer_norm(lw["i_k"](x, out_dtype=torch.float32), (cfg.index_dim,), lw["i_kn_w"], lw["i_kn_b"], 1e-6)
        k = k.to(torch.bfloat16)[:, None]
        rd = cfg.rope_dim
        q = torch.cat([rope_interleave(q[..., :rd], cos, sin), q[..., rd:]], -1)
        k = torch.cat([rope_interleave(k[..., :rd], cos, sin), k[..., rd:]], -1)[:, 0]
        kv_write(self.i_cache[L], positions, k)
        w = F.linear(x.float(), lw["i_w"]) * cfg.index_heads ** -0.5                 # [T, h]
        if self.index_kernel:
            # engine/glm/index_kernel.py: same scores without the fp32 [T, h, S] intermediates.
            # Rows in chunks to bound the [rows, S] scores (prefill at long contexts).
            from .index_kernel import index_scores
            step = max(1, (32 << 20) // max(S, 1))
            return torch.cat([index_scores(q[t0:t0 + step], w[t0:t0 + step], self.i_cache[L],
                                           positions[t0:t0 + step], S, cfg.index_dim ** -0.5)
                              .topk(min(cfg.index_topk, S), -1).indices for t0 in range(0, T, step)])
        K = kv_span(self.i_cache[L], S, torch.float32)
        kpos = torch.arange(S, device=x.device)
        out = []
        step = max(1, 4096 * 1024 // max(S, 1))
        for t0 in range(0, T, step):
            qs = q[t0:t0 + step]
            sc = F.relu(torch.einsum("thd,sd->ths", qs.float(), K.float()) * cfg.index_dim ** -0.5)
            score = torch.einsum("th,ths->ts", w[t0:t0 + step], sc)
            score = score.masked_fill(kpos[None] > positions[t0:t0 + step, None], float("-inf"))
            out.append(score.topk(min(cfg.index_topk, S), -1).indices)
        return torch.cat(out)

    def attention(self, L, x, topk_prev):
        cfg, lw = self.cfg, self.L[L]
        T = x.shape[0]
        decode = T <= self.decode_rows
        positions = (self.pos_t + torch.arange(T, device=x.device)) if decode else \
            torch.arange(self.pos, self.pos + T, device=x.device)
        S = self.max_seq if decode else self.pos + T
        cos, sin = self.cos.index_select(0, positions), self.sin.index_select(0, positions)
        q_resid = rms(lw["q_a"](x), lw["q_norm"], LATENT_EPS)
        q = lw["q_b"](q_resid).view(T, self.hl, cfg.qk_dim)
        q_nope, q_rope = q[..., :cfg.nope], rope_interleave(q[..., cfg.nope:], cos, sin)
        ckv = lw["kv_a"](x)
        c = rms(ckv[:, :cfg.kv_lora], lw["kv_norm"], LATENT_EPS)
        kr = rope_interleave(ckv[:, None, cfg.kv_lora:], cos, sin)[:, 0]
        kv_write(self.c_cache[L], positions, c)
        self.r_cache[L].index_copy_(0, positions, kr)
        topk = self._indexer(L, x, q_resid, positions, cos, sin, S) if self.full(L) else topk_prev
        q_lat = torch.einsum("thn,hnc->thc", q_nope, lw["w_uk"])                  # [T, h, kv_lora]
        scale = cfg.qk_dim ** -0.5
        C, R = self.c_cache[L], self.r_cache[L]
        if not decode and S <= cfg.index_topk and not self.attn_kernel:
            C = kv_span(C, S)
            # Every cached position is selected: dense causal attention over [0, S).
            sc = (torch.einsum("thc,sc->ths", q_lat, C) + torch.einsum("thr,sr->ths", q_rope, R[:S])).float() * scale
            kpos = torch.arange(S, device=x.device)
            sc = sc.masked_fill(kpos[None, None] > positions[:, None, None], float("-inf"))
            p = torch.softmax(sc, -1).to(torch.bfloat16)
            o_lat = torch.einsum("ths,sc->thc", p, C)
        elif self.attn_kernel:
            # engine/glm/sparse_attn.py: the selected latents read where they live, fp32 scores,
            # online softmax. 1,024 rows: 4.5 ms a layer against the loop's 81; 4-8 rows 0.17 ms.
            # Within the first index_topk positions the indexer's picks are every position, so the
            # dense case goes here too (its [T, heads, S] fp32 scores made 2,048-row chunks slower).
            from .sparse_attn import sparse_mla
            o_lat = sparse_mla(q_lat, q_rope, C, R, topk, positions, scale)
        else:
            outs = []
            for t0 in range(0, T, 32):
                sel = topk[t0:t0 + 32].long()
                cg, rg = kv_read(C, sel), R[sel]                                   # [t, k, c|r]
                sc = (torch.einsum("thc,tkc->thk", q_lat[t0:t0 + 32], cg)
                      + torch.einsum("thr,tkr->thk", q_rope[t0:t0 + 32], rg)).float() * scale
                valid = sel <= positions[t0:t0 + 32, None]
                sc = sc.masked_fill(~valid[:, None], float("-inf"))
                p = torch.softmax(sc, -1).to(torch.bfloat16)
                outs.append(torch.einsum("thk,tkc->thc", p, cg))
            o_lat = torch.cat(outs)
        o = torch.einsum("thc,hvc->thv", o_lat, lw["w_uv"]).reshape(T, self.hl * cfg.v_dim)
        partial = lw["o"](o, out_dtype=torch.float32)
        return self.comm.all_reduce(partial).to(torch.bfloat16), topk

    # ------------------------------------------------------------------ MoE
    def request_snapshot(self):
        """Rank 0, for the server's per-request line: routing and hot-read counters so far."""
        ms = self.miss_totals() or {}
        h = self.hot_stats
        return {"picks": ms.get("picks", 0), "missed": ms.get("missed", 0), "mass": ms.get("mass", 0.0),
                "mass_missed": ms.get("mass_missed", 0.0), "hot": h["reads"], "kept": h.get("kept", 0),
                "swaps": self.adapt_totals["swaps"] if self.demand is not None else 0, "tokens": len(self.tokens),
                "rounds": self.spec_stats.get("rounds", 0), "accepted": self.spec_stats.get("accepted", 0),
                "drafted": self.spec_stats.get("drafted", 0)}

    @staticmethod
    def request_stats(s0, s1, s2):
        """Prefill (s0 -> first token s1) and decode (s1 -> end s2): share of routed weight and of
        picks that ran on a substitute, hot reads (kept), adaptation swaps."""
        def miss(a, b):
            m = b["mass"] - a["mass"]
            return round((b["mass_missed"] - a["mass_missed"]) / m, 3) if m > 0 else None
        p = s2["picks"] - s1["picks"]
        return {"prefill_miss": miss(s0, s1), "decode_miss": miss(s1, s2),
                "decode_miss_picks": round((s2["missed"] - s1["missed"]) / p, 3) if p > 0 else None,
                "hot_reads": s2["hot"] - s1["hot"], "hot_kept": s2["kept"] - s1["kept"],
                "hot_per_tok": round((s2["hot"] - s1["hot"]) / max(1, s2.get("tokens", 0) - s1.get("tokens", 0)), 2),
                "swaps": s2["swaps"] - s0["swaps"],
                "drafted_per_round": round((s2["drafted"] - s1["drafted"]) / max(1, s2["rounds"] - s1["rounds"]), 2),
                "accepted_per_round": round((s2["accepted"] - s1["accepted"]) / max(1, s2["rounds"] - s1["rounds"]), 2)}

    def miss_totals(self):
        """miss_stats with the graph replays' device counts folded in."""
        ms = self.miss_stats
        acc = ms.get("dev") if ms else None
        if acc is not None:
            p, m, a, b = acc.tolist()
            acc.zero_()
            ms["picks"] += int(p)
            ms["missed"] += int(m)
            ms["mass"] += a
            ms["mass_missed"] += b
        return ms

    def _route(self, L, x):
        """(slots int32 [T, k'], weights fp32 [T, k']) including the shared expert or the
        dense layer's pseudo-experts."""
        cfg, lw = self.cfg, self.L[L]
        T = x.shape[0]
        fixed = self.fixed[L][None].expand(T, -1)
        ones = torch.ones(fixed.shape, dtype=torch.float32, device=x.device)
        if self.dense(L):
            return fixed.contiguous(), ones
        from .ref import route, route_pick, route_scores
        allowed = self.allowed[L] if self.allowed is not None else None
        # One router matmul serves every pick below; under split graphs the attention segment's
        # hot candidate already ran it on this same x (_hot_candidate).
        pre = self._pre_route.pop(L, None)
        sc = pre[1:] if pre is not None and pre[0] is x else route_scores(cfg, x, lw["router"], lw["bias"])
        w, ids = route_pick(cfg, *sc, allowed)
        w0 = ids0 = None
        if allowed is not None and (self.demand is not None or self.miss_stats is not None
                                    or self.route_dump is not None or self.miss_hist is not None):
            w0, ids0 = route_pick(cfg, *sc, None)
        if allowed is not None and self.sub_weight != "renorm":
            w = self._sub_weights(cfg, lw, x, ids)
        if self.record is not None:
            self.record.append((L, ids.cpu(), w.cpu()))
        if self.demand is not None and allowed is not None:
            self.demand[L].index_add_(0, ids0.reshape(-1), w0.reshape(-1))
        if self.miss_stats is not None and allowed is not None:
            missed = ~allowed[ids0]
            ms = self.miss_stats
            # Counted on the device only (no host sync a layer, and graph replays count too);
            # miss_totals() folds them into the dict. The first call is never under capture
            # (prefill and the capture's warm-up run eagerly first).
            if "dev" not in ms:
                ms["dev"] = torch.zeros(4, dtype=torch.float64, device=x.device)
            acc = ms["dev"]
            acc[0].add_(ids0.numel())
            acc[1].add_(missed.sum())
            acc[2].add_(w0.sum())
            acc[3].add_((w0 * missed).sum())
        if self.route_dump is not None and allowed is not None:
            ahead = {}
            for d in (1, 2, 4, 8):
                N = L + d
                if N < self.nL - 1 and not self.dense(N):        # not into the MTP layer
                    ahead[N] = route(cfg, x, self.L[N]["router"], self.L[N]["bias"], None)[1].to(torch.int16)
            self.route_dump.append((L, ids0.to(torch.int16), w0.half(), ~allowed[ids0], ahead))
        if self.miss_hist is not None and allowed is not None:
            m = (~allowed[ids0]).float().reshape(-1)
            wn = (w0 / cfg.routed_scaling).reshape(-1)
            b = (wn * self.miss_hist.shape[1]).long().clamp_(0, self.miss_hist.shape[1] - 1)
            self.miss_hist[0].index_add_(0, b, m)
            self.miss_hist[1].index_add_(0, b, wn * m)
        if self.miss == "stream":
            self._stream_misses(L, ids)
        slots = self.lut[L][ids]
        # A pick with no slot (a layer holding fewer experts than top-k, or a stream miss not yet
        # placed) must not reach the kernels as -1 -- they skip it and leave its output rows
        # unwritten. It becomes a zero-weight pick of the layer's fixed slot.
        missing = slots < 0
        w = torch.where(missing, torch.zeros_like(w), w)
        slots = torch.where(missing, fixed[:, :1].expand_as(slots), slots)
        return torch.cat([slots, fixed], 1).contiguous(), torch.cat([w, ones], 1)

    def _sub_weights(self, cfg, lw, x, ids):
        """MUSE_SUB_WEIGHT=exact|transfer for the substituted picks ``ids`` (graph-safe)."""
        from .ref import route
        sc = F.linear(x.float(), lw["router"].float()).sigmoid()            # [T, E]
        w0, ids0 = route(cfg, x, lw["router"], lw["bias"], None)
        if self.sub_weight == "exact":
            w = sc.gather(1, ids)
            denom = sc.gather(1, ids0).sum(-1, keepdim=True) if cfg.norm_topk else 1.0
            return w / (denom + 1e-20) * cfg.routed_scaling
        match = ids[:, :, None] == ids0[:, None, :]                          # [T, k, k]
        resident = (match.float() * w0[:, None, :]).sum(-1)                  # exact weight where kept
        is_sub = ~match.any(-1)
        missed_w = torch.where(match.any(1), torch.zeros_like(w0), w0)       # exact picks not chosen
        missed_sorted = missed_w.sort(-1, descending=True).values
        choice = (sc + lw["bias"].float()).gather(1, ids).masked_fill(~is_sub, float("-inf"))
        order = choice.argsort(-1, descending=True)                          # substitutes first, best first
        moved = torch.zeros_like(w0).scatter_(1, order, missed_sorted)
        return resident + moved * is_sub

    def _stream_misses(self, L, ids):
        need = ids.unique()
        missing = need[self.lut[L][need] < 0].tolist()
        if len(missing) > len(self.transient):
            raise RuntimeError(f"{len(missing)} misses exceed {len(self.transient)} transient slots")
        t0 = time.perf_counter()
        for (e, slot), parts in zip(zip(missing, self.transient), self._read_many(L, missing)):
            self._load(L, e, slot, parts)
        self._streamed = (L, missing)
        if self.stats["stream_loads"] % 2048 < len(missing):
            self.ck.drop_cache()
        self.stats["stream_loads"] += len(missing)
        self.stats["stream_s"] += time.perf_counter() - t0

    def _experts_native(self, x, slots):
        h, rt = self._native_up(x, slots)
        return self._native_down(self.comm.cat_last(h), rt)

    def _native_up(self, x, slots):
        import fp4_moe_cuda as CUDA
        A, cfg = self.arena, self.cfg
        T, k = slots.shape
        P_ = T * k
        bs, bp, br, nb = CUDA.build_routing_small(slots.reshape(-1).contiguous(), 16, k)
        h = torch.empty((P_, A.Il), dtype=torch.bfloat16, device=x.device)
        ones = torch.ones(P_, dtype=torch.float32, device=x.device)
        CUDA.up(x, A.w1, A.s1, A.w3, A.s3, h, ones, bs, br, float("inf"), k, A.Il, cfg.hidden, nb, -1)
        return h, (bs, bp, nb, T, k)

    def _native_down(self, full, rt):
        import fp4_moe_cuda as CUDA
        A, cfg = self.arena, self.cfg
        bs, bp, nb, T, k = rt
        parts = torch.empty((T * k, A.Hl), dtype=torch.float32, device=full.device)
        CUDA.down(full, A.w2, A.s2, parts, bs, bp, k, A.Hl, cfg.moe_inter, T, nb, -1, False, 1)
        return parts.view(k, T, A.Hl)

    def _gather_pair(self, a, b):
        """(cat_last(a), cat_last(b)) for two 2-D tensors of any dtypes in one all-gather."""
        W, na = self.world, a.numel() * a.element_size()
        g = self.comm.all_gather(torch.cat([a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)]))
        out = []
        for t, part in ((a, g[:, :na]), (b, g[:, na:])):
            v = part.contiguous().view(t.dtype).view(W, *t.shape)
            out.append(v.movedim(0, -2).reshape(*t.shape[:-1], W * t.shape[-1]))
        return out

    def _experts_triton(self, x, slots):
        h, rt = self._triton_up(x, slots)
        return self._triton_down(self.comm.cat_last(h), rt)

    def _triton_up(self, x, slots):
        import fp4_moe as K4
        A, cfg = self.arena, self.cfg
        T, k = slots.shape
        P_ = T * k
        BM = K4._pick_bm(P_)
        uniq, inv = torch.unique(slots, return_inverse=True)
        bs, bp, nb = K4.build_routing(inv.to(torch.int32).view_as(slots), int(uniq.numel()), BM)
        bs = torch.where(bs >= 0, uniq.to(torch.int32)[bs.clamp_min(0)], bs)
        bn1, nw1, ns1 = K4._UP_CFG[BM]
        h = torch.empty((P_, A.Il), dtype=torch.bfloat16, device=x.device)
        ones = torch.ones(P_, dtype=torch.float32, device=x.device)
        K4._moe_up_kernel[(nb, A.Il // bn1)](x, A.w1, A.s1, A.w3, A.s3, h, ones, bs, bp, x.stride(0), h.stride(0),
                                             float("inf"), TOPK=k, N=A.Il, K=cfg.hidden, BM=BM, BN=bn1,
                                             SCALED=False, NULL_SLOT=-1, num_warps=nw1, num_stages=ns1)
        return h, (bs, bp, nb, T, k, BM)

    def _triton_down(self, full, rt):
        import fp4_moe as K4
        A, cfg = self.arena, self.cfg
        bs, bp, nb, T, k, BM = rt
        bn2, nw2, ns2 = K4._DOWN_CFG[BM]
        parts = torch.empty((T * k, A.Hl), dtype=torch.float32, device=full.device)
        K4._moe_down_kernel[(nb, A.Hl // bn2)](full, A.w2, A.s2, parts, bs, bp, full.stride(0), parts.stride(0),
                                               TOPK=k, N=A.Hl, K=cfg.moe_inter, BM=BM, BN=bn2, NTOK=T,
                                               SCALED=False, NULL_SLOT=-1, PARTIAL=False, PARTS_WORLD=1,
                                               num_warps=nw2, num_stages=ns2)
        return parts.view(k, T, A.Hl)

    def _pf_next(self, L):
        N = L + 1
        return N if N < self.cfg.layers and not self.dense(N) else None

    def _pf_predict(self, L, x):
        """Prefill: run the next layer's router on this layer's MLP input, sum the router mass
        over the chunk's rows, and read the heaviest non-resident experts (up to MUSE_PF_SLOTS)
        into the next layer's transient buffer while this layer and the next attention compute.
        Rank 0 decides (the same experts must land on both ranks); a misprediction only costs the
        read -- the next layer's router still decides, over resident + prefetched experts."""
        N = self._pf_next(L)
        if N is None:
            return
        from .ref import route
        plan = None
        if self.rank == 0:
            w, ids = route(self.cfg, x, self.L[N]["router"], self.L[N]["bias"], None)
            mass = torch.zeros(self.cfg.experts, dtype=torch.float32, device=x.device)
            mass.index_add_(0, ids.reshape(-1), w.reshape(-1))
            mass.masked_fill_(self.lut[N] >= 0, 0.0)
            top = torch.topk(mass, self.pf_n)
            es = [int(e) for e, m in zip(top.indices.tolist(), top.values.tolist()) if m > 0]
            plan = [(e, None) for e in es]
            if self.pf_keep and es and self.demand is not None:
                # Kept only if it ranks in: its long-term score + this request's demand + the
                # predicted mass beats the victim's by the adaptation margin (MUSE_PF_KEEP_MARGIN).
                cs = (self.score[N] + self.demand[N] + mass)[es].tolist()
                plan = [(e, v) for e, v in zip(es, self._pf_victims(cs))]
        plan = self.comm.broadcast_obj(plan)
        if not hasattr(self, "_pool"):
            self._read_many(L, [])
        buf = self.pf_slots[(N % 2) * self.pf_n:(N % 2 + 1) * self.pf_n]
        self.pf_pending[N] = [(e, slot, v, self._pool.submit(self._expert_parts, N, e))
                              for (e, v), slot in zip(plan, buf)]
        self.pf_stats["planned"] += len(plan)

    def _pf_victims(self, cand_scores):
        """Rank 0: n resident experts to give up for promoted prefetches -- lowest long-term score
        (decayed history + this request's demand so far), not promoted in this prefill, each layer
        keeping more than 2 x top-k. None where there is no such expert."""
        lay = self._bg_layers
        slots = self.lut[lay]
        res = slots >= 0
        # Not this prefill's promotions, and not the layer being computed: its prefetched experts
        # sit in transient slots (the highest indices) until _pf_release promotes them.
        ok = res & ~self.pf_kept[lay] & (slots < self.transient[0])
        n = len(cand_scores)
        k = min(4 * n, int(ok.sum()))
        if k == 0:
            return [None] * n
        vs = torch.topk((self.score + self.demand)[lay].masked_fill(~ok, float("inf")).reshape(-1), k, largest=False)
        idx, vval = vs.indices.tolist(), vs.values.tolist()
        left = (res & (slots < self.transient[0])).sum(1).tolist()
        E, floor, margin = self.cfg.experts, 2 * self.cfg.topk, self.pf_keep_margin
        out, vi = [None] * n, 0
        # Strongest candidates take the weakest victims while they beat them by the margin.
        for ci in sorted(range(n), key=lambda i: -cand_scores[i]):
            while vi < len(idx):
                vl, ve = divmod(idx[vi], E)
                if left[vl] > floor:
                    break
                vi += 1
            if vi >= len(idx) or cand_scores[ci] <= vval[vi] * margin:
                break
            left[vl] -= 1
            out[ci] = (self.swap_layers[vl], ve)
            self.pf_kept[self.swap_layers[vl], ve] = True        # not chosen twice
            vi += 1
        return out

    def _pf_install(self, L):
        items = self.pf_pending.pop(L, None)
        if not items:
            return
        t0 = time.perf_counter()
        parts = [f.result() for *_, f in items]
        self.pf_stats["wait_s"] += time.perf_counter() - t0
        for (e, slot, _, _), pr in zip(items, parts):
            self._put(slot, pr)
            self.lut[L, e] = slot
        es = [e for e, *_ in items]
        self.allowed[L, es] = True
        self.pf_live = (L, es)
        self.pf_promote = [(e, slot, v) for e, slot, v, _ in items if v is not None]
        self.pf_stats["reads"] += len(es)
        if self.pf_stats["reads"] // 512 != (self.pf_stats["reads"] - len(es)) // 512:
            self.ck.drop_cache()

    def _pf_release(self):
        if self.pf_live:
            L, es = self.pf_live
            self.lut[L, es] = -1
            self.allowed[L, es] = False
            self.pf_live = None
            if self.hot_live:
                self._hot_keep(L, self.hot_live)
                self.hot_live = None
        items, self.pf_promote = getattr(self, "pf_promote", None), None
        if not items:
            return
        # Promote: copy each prefetched expert from its transient slot over its victim's slot
        # (stream-ordered after this layer's MLP), then repoint the lut. Same plan on both ranks.
        vslots = self.lut[[v[0] for _, _, v in items], [v[1] for _, _, v in items]].tolist()
        for (e, slot, (VL, ve)), vs in zip(items, vslots):
            if vs < 0 or self.owner[vs] != (VL, ve):
                continue
            self._copy_slot(vs, slot)
            self.lut[VL, ve] = -1
            self.owner[vs] = (L, e)
            self.lut[L, e] = vs
            self.allowed[VL, ve] = False
            self.allowed[L, e] = True
            self.pf_kept[L, e] = True
            self.pf_stats["promoted"] = self.pf_stats.get("promoted", 0) + 1

    def _hot_keep(self, L, items):
        """MUSE_HOT_KEEP: copy each hot read from its transient slot over the next queued victim that
        still holds its slot (stream-ordered after this layer's MLP), repoint the lut, and protect it
        from background eviction like a background install. The queue, the hot reads and owner are
        the same on both ranks, so both keep the same experts in the same slots."""
        q, st = self.hot_q, self.hot_stats
        for e, src in items:
            while q:
                VL, ve, vs = q.pop()
                if self.owner[vs] == (VL, ve):
                    break
            else:
                st["unkept"] += 1
                continue
            if self.hot_audit:
                sc = self.score + self.demand
                c, v = float(sc[L, e]), float(sc[VL, ve])
                res = sc[self._bg_layers][self.lut[self._bg_layers] >= 0]
                st["gate_drop"] = st.get("gate_drop", 0) + (c <= v * self.adapt_margin)
                st["gate_below_victim"] = st.get("gate_below_victim", 0) + (c <= v)
                st["pct_sum"] = st.get("pct_sum", 0.0) + float((res < c).float().mean())   # share of residents it outranks
            self._copy_slot(vs, src)
            self.lut[VL, ve] = -1
            self.allowed[VL, ve] = False
            self.owner[vs] = (L, e)
            self.lut[L, e] = vs
            self.allowed[L, e] = True
            self.bg_until[L, e] = self.bg_period + self.bg_protect
            st["kept"] += 1

    def _hot_stream(self, L, x):
        """Decode: read this layer's heaviest missed picks (weight share >= MUSE_HOT_MIN, up to
        MUSE_HOT) now and allow them for this layer only. Both ranks route the same replicated
        x with the same weights, so they choose the same experts (as _route already relies on)."""
        from .ref import route
        cfg = self.cfg
        w0, ids0 = route(cfg, x, self.L[L]["router"], self.L[L]["bias"], None)
        share = (w0 / cfg.routed_scaling) * (~self.allowed[L][ids0])
        vals, idx = torch.topk(share.reshape(-1), min(share.numel(), 4 * self.hot_n))
        picks = ids0.reshape(-1)[idx].tolist()
        es = []
        for v, e in zip(vals.tolist(), picks):
            if v < self.hot_min or len(es) >= self.hot_n:
                break
            if e not in es:
                es.append(e)
        if not es:
            return False
        t0 = time.perf_counter()
        for e, slot, pr in zip(es, self.transient, self._read_urgent(L, es)):
            self._put(slot, pr)
            self.lut[L, e] = slot
        self.allowed[L, es] = True
        self.pf_live = (L, es)
        if self.hot_keep:
            self.hot_live = list(zip(es, self.transient))
        st = self.hot_stats
        st["reads"] += len(es)
        st["layers"] += 1
        st["s"] += time.perf_counter() - t0
        if st["reads"] // 512 != (st["reads"] - len(es)) // 512:
            self.ck.drop_cache()
        return True

    def mlp(self, L, x):
        self._streamed = None
        pf = self.pf_on and x.shape[0] >= self.pf_min_rows
        if pf:
            self._pf_install(L)
            self._pf_predict(L, x)
        elif (self.hot_n and x.shape[0] <= self.decode_rows and not self.dense(L)
              and not torch.cuda.is_current_stream_capturing()):
            pf = self._hot_stream(L, x)
        slots, w = self._route(L, x)
        x = x.contiguous()
        y = out = None
        x3 = self.x3 is not None and not self.dense(L) and L != self.mtp_L
        if x3:
            from .exl3 import routed, routed_down, routed_up
            k = self.cfg.topk
            rs = slots[:, :k] - self.x3_off
            bad = rs < 0                      # a slotless pick: zero weight on any loaded slot
            wr = torch.where(bad, torch.zeros_like(w[:, :k]), w[:, :k])
            slots, w = slots[:, k:].contiguous(), w[:, k:]
        if self.gather_merge:
            # One all-gather each way. The routed EXL3 and the fixed experts' intermediates go
            # together; every pick's output rows are weighted and summed on the rank that owns
            # their columns and gathered once as BF16 -- the gather only concatenates, so this is
            # per element the old arithmetic (the sum over picks, + the routed part, the BF16
            # cast), at a fraction of the bytes: MXFP4's 9 picks went over as fp32 [9, T, H/2]
            # (332 KB at 3 rows, past the RoCE gather's 256 KB, so NCCL every MoE layer).
            native = self.native and slots.numel() <= 512
            # Prefill: every row picks the same fixed slots, a dense matrix of thousands of rows.
            # Only where the call's slots are all fixed: an EXL3 layer (its routed picks went to
            # routed_up) or a dense layer -- never MXFP4's routed MoE layers or the MTP layer.
            dense = (self.fixed_dense and not native and slots.shape[0] >= 256
                     and (x3 or self.dense(L)))
            if dense:
                from .mxfp4_dense import fixed_down, fixed_up
                fs = self.fixed[L].tolist()
                h, rt = fixed_up(x, self.arena, fs), None
            else:
                h, rt = (self._native_up if native else self._triton_up)(x, slots)
            local = None
            if x3:
                xd, ctx = routed_up(x, rs.clamp_min(0), wr, self.x3)
                xd_full, h_full = self._gather_pair(xd, h)
                local = routed_down(xd_full.contiguous(), ctx, self.x3)
            else:
                h_full = self.comm.cat_last(h)
            if dense:
                parts = fixed_down(h_full, self.arena, fs, self.world)
            else:
                parts = (self._native_down if native else self._triton_down)(h_full, rt)
            z = (parts * w.t()[:, :, None]).sum(0)
            out = self.comm.cat_last((z if local is None else z + local).to(torch.bfloat16))
        elif x3:
            y = routed(x, rs.clamp_min(0), wr, self.x3, self.comm)
        if out is None:
            T, k = slots.shape
            if self.native and T * k <= 512:
                parts = self._experts_native(x, slots)
            else:
                parts = self._experts_triton(x, slots)
        if self._streamed:
            sl, missing = self._streamed
            for e in missing:
                self.lut[sl, e] = -1
        if pf:
            self._pf_release()
        if out is not None:
            return out
        z = (self.comm.cat_last(parts) * w.t()[:, :, None]).sum(0)
        return (z if y is None else z + y).to(torch.bfloat16)

    # ------------------------------------------------------------------ model
    def _layers(self, h):
        topk = None
        for L in range(self.cfg.layers):
            a, topk = self.attention(L, rms(h, self.L[L]["ln_in"], self.cfg.rms_eps), topk)
            h = h + a
            h = h + self.mlp(L, rms(h, self.L[L]["ln_post"], self.cfg.rms_eps))
        return h

    def embed_ids(self, ids):
        local = ids - self.rank * self.vl
        inside = (local >= 0) & (local < self.vl)
        e = self.embed[local.clamp(0, self.vl - 1)] * inside[:, None].to(torch.bfloat16)
        return self.comm.all_reduce(e.float()).to(torch.bfloat16)

    def logits(self, h):
        """(logits, post-norm hidden)."""
        hn = rms(h, self.final_norm, self.cfg.rms_eps)
        return self.comm.cat_last(self.head(hn, out_dtype=torch.float32)), hn

    def _step(self, ids):
        return self.logits(self._layers(self.embed_ids(ids)))

    def _replay(self, ids):
        T = ids.numel()
        entry = self.graphs.get(T)
        if entry is None:
            static_ids = ids.clone()
            side = self._side_stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._step(static_ids)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out, static_h = self._step(static_ids)
            entry = self.graphs[T] = (graph, static_ids, static_out, static_h)
            torch.cuda.empty_cache()
            self.log(f"[glm] captured decode graph for {T} rows")
        graph, static_ids, static_out, static_h = entry
        static_ids.copy_(ids)
        graph.replay()
        return static_out.clone(), static_h.clone()

    def _hot_candidate(self, L, x):
        """Graph-safe: this layer's heaviest missed picks (weight share, expert) into the pinned
        host buffer, read by _replay_split between this layer's two graphs."""
        from .ref import route_pick, route_scores
        cfg, n = self.cfg, self.hot_n
        sc = route_scores(cfg, x, self.L[L]["router"], self.L[L]["bias"])
        self._pre_route[L] = (x, *sc)          # the MLP segment's _route reuses the matmul
        w0, ids0 = route_pick(cfg, *sc, None)
        share = (w0 / cfg.routed_scaling) * (~self.allowed[L][ids0])
        k = min(share.numel(), 2 * n)
        v, i = share.reshape(-1).topk(k)
        self._hot_dev[L, :k] = v
        self._hot_dev[L, 2 * n:2 * n + k] = ids0.reshape(-1)[i].float()
        self._hot_host[L].copy_(self._hot_dev[L], non_blocking=True)
        return sc

    def _side_stream(self):
        """The stream every capture's eager warm-up runs on. The caching allocator keeps blocks per
        stream, so a fresh stream per capture left each warm-up's blocks stranded (a dozen
        captures under dynamic MTP depth); empty_cache() after a capture returns them."""
        if getattr(self, "_side", None) is None:
            self._side = torch.cuda.Stream()
        return self._side

    def _graph_pool(self):
        """One capture pool for every split set and MTP graph. They never replay concurrently,
        each replay rewrites every boundary tensor it reads (in segment order, from its inputs),
        and the outputs a caller reads are kept alive by their entries -- so one graph's
        temporaries may land on another's: dynamic MTP depth captures a split set per verify
        row count, ~230 MB each in pools of their own."""
        if getattr(self, "_graph_pool_h", None) is None:
            self._graph_pool_h = torch.cuda.graph_pool_handle()
        return self._graph_pool_h

    def _capture_split(self, ids):
        """Decode for T rows as per-layer graphs: [embed], then per layer (attention + the hot
        candidate) and (MLP), then [head]. Captured in replay order into one memory pool, every
        boundary tensor kept alive by the entry, so later graphs read earlier graphs' outputs."""
        cfg = self.cfg
        if not hasattr(self, "_hot_dev"):
            self._hot_dev = torch.zeros((self.nL, 4 * self.hot_n), dtype=torch.float32, device=self.dev)
            self._hot_host = torch.zeros((self.nL, 4 * self.hot_n), dtype=torch.float32).pin_memory()
        st_ids = ids.clone()
        side = self._side_stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step(st_ids)
        torch.cuda.current_stream().wait_stream(side)
        pool = self._graph_pool()
        segs, keep = [], []

        def cap(fn):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = fn()
            keep.append(out)
            return g, out

        g, h = cap(lambda: self.embed_ids(st_ids))
        segs.append(("embed", -1, g))
        topk = None
        for L in range(cfg.layers):
            def attn(L=L, h=h, topk=topk):
                a, tk = self.attention(L, rms(h, self.L[L]["ln_in"], cfg.rms_eps), topk)
                h1 = h + a
                x = rms(h1, self.L[L]["ln_post"], cfg.rms_eps)
                sc = self._hot_candidate(L, x) if not self.dense(L) else None
                return h1, x, tk, sc            # sc: kept alive for the MLP graph that reads it
            g, (h1, x, topk, _) = cap(attn)
            segs.append(("attn", L, g))
            g, h = cap(lambda L=L, h1=h1, x=x: h1 + self.mlp(L, x))
            segs.append(("mlp", L, g))
        g, (st_out, st_h) = cap(lambda: self.logits(h))
        segs.append(("head", -1, g))
        torch.cuda.empty_cache()
        self.log(f"[glm] captured split decode graphs for {ids.numel()} rows ({len(segs)} segments)")
        return segs, st_ids, st_out, st_h, keep

    def _replay_split(self, ids):
        T = ids.numel()
        entry = self.graphs.get(("split", T))
        if entry is None:
            entry = self.graphs[("split", T)] = self._capture_split(ids)
        segs, st_ids, st_out, st_h, _ = entry
        st_ids.copy_(ids)
        n, st = self.hot_n, self.hot_stats
        thr = self.hot_thr.get(T, self.hot_min)
        cap = self.hot_max or (max(1, int(2 * self.hot_target)) if self.hot_target > 0 else 1 << 30)
        reads = 0
        for kind, L, g in segs:
            g.replay()
            if kind == "mlp" and self.pf_live:
                self._pf_release()                 # stream-ordered after the layer's experts
            if kind != "attn" or self.dense(L):
                continue
            # The candidate's pinned copy is the attention graph's last op; both ranks read the
            # same values (replicated x), so they read the same experts.
            torch.cuda.current_stream().synchronize()
            row = self._hot_host[L].tolist()
            room = cap - reads
            if self.hot_pace and cap < 1 << 30:
                room = min(room, -int(-cap * self._hot_frac[L] // 1) + self.hot_pace_lead - reads)
            es = []
            for v, e in zip(row[:2 * n], row[2 * n:]):
                if v < thr or len(es) >= min(n, room):
                    break
                if int(e) not in es:
                    es.append(int(e))
            if not es:
                continue
            reads += len(es)
            st["quarters"][self._hot_quarter[L]] += len(es)
            t0 = time.perf_counter()
            for e, slot, pr in zip(es, self.transient, self._read_urgent(L, es)):
                self._put(slot, pr)
                self.lut[L, e] = slot
            self.allowed[L, es] = True
            self.pf_live = (L, es)
            if self.hot_keep:
                self.hot_live = list(zip(es, self.transient))
            st["reads"] += len(es)
            st["layers"] += 1
            st["s"] += time.perf_counter() - t0
            if st["reads"] // 512 != (st["reads"] - len(es)) // 512:
                self.ck.drop_cache()
        if reads >= cap:
            st["capped"] = st.get("capped", 0) + 1      # later layers' misses went unread
        if self.hot_target > 0:
            # Multiplicative step toward the target; float math on identical counts on both ranks.
            thr *= 1.1 ** ((reads - self.hot_target) / self.hot_target)
            self.hot_thr[T] = min(1.0, max(self.hot_min, thr))
        st["steps"] = st.get("steps", 0) + 1
        st[f"thr_{T}"] = round(self.hot_thr.get(T, self.hot_min), 3)
        return st_out.clone(), st_h.clone()

    def _mtp_core(self, next_ids, h_prev, logits=True):
        """The MTP layer at positions pos_t.. (device) -- graph-capturable. logits=False: the
        cache fill only (the hidden is returned, the head and its gather skipped)."""
        cfg, L = self.cfg, self.mtp_L
        lw = self.L[L]
        T = next_ids.numel()
        positions = self.pos_t + torch.arange(T, device=self.dev)
        e = self.embed_ids(next_ids)
        e = torch.where(positions[:, None] == 0, torch.zeros_like(e), e)
        x = self.comm.cat_last(lw["eh"](torch.cat([rms(e, lw["enorm"], cfg.rms_eps),
                                                    rms(h_prev.to(e.dtype), lw["hnorm"], cfg.rms_eps)], -1)))
        a, _ = self.attention(L, rms(x, lw["ln_in"], cfg.rms_eps), None)
        h = x + a
        h = h + self.mlp(L, rms(h, lw["ln_post"], cfg.rms_eps))
        hn = rms(h, lw["sh_norm"], cfg.rms_eps)
        if not logits:
            return hn, None
        if self.draft_head is not None:
            from .mxfp4_dense import gemv
            return hn, self.comm.cat_last(gemv(hn, *self.draft_head))
        return hn, self.comm.cat_last(self.head(hn, out_dtype=torch.float32))

    @torch.no_grad()
    def mtp_forward(self, next_ids, h_prev, pos0, logits=True):
        """The MTP layer at positions pos0..: (token i+1, post-norm hidden at i) -> (its
        post-norm hidden, logits for token i+2, or None with logits=False). Writes the MTP
        layer's caches. Calls of up to depth rows replay a captured graph: the one-row draft
        chain, and the cache fills of accepted drafts (no head: a 0.95 GB read a call that
        nothing used, and in prefill a [chunk, vocab] gather)."""
        next_ids = torch.as_tensor(next_ids, dtype=torch.long, device=self.dev).reshape(-1)
        T = next_ids.numel()
        saved, saved_t = self.pos, self.pos_t.clone()
        self.pos = pos0
        self.pos_t.fill_(pos0)
        if T <= self.mtp_graph_rows and self.use_graphs and self.miss == "substitute":
            key = ("mtp", T, logits)
            entry = self.graphs.get(key)
            if entry is None:
                st_ids, st_h = next_ids.clone(), h_prev.to(torch.bfloat16).clone()
                side = self._side_stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._mtp_core(st_ids, st_h, logits)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._graph_pool()):
                    st_hn, st_z = self._mtp_core(st_ids, st_h, logits)
                entry = self.graphs[key] = (graph, st_ids, st_h, st_hn, st_z)
                torch.cuda.empty_cache()
                self.log(f"[glm] captured MTP graph for {T} row(s){'' if logits else ', cache fill'}")
            graph, st_ids, st_h, st_hn, st_z = entry
            st_ids.copy_(next_ids)
            st_h.copy_(h_prev)
            graph.replay()
            out = st_hn.clone(), (st_z.clone() if logits else None)
        else:
            out = self._mtp_core(next_ids, h_prev, logits)
        self.pos = saved
        self.pos_t.copy_(saved_t)
        return out

    def reusable_prefix(self, prompt):
        n, limit = 0, min(len(prompt) - 1, len(self.tokens))
        while n < limit and prompt[n] == self.tokens[n]:
            n += 1
        return n

    def prefill(self, prompt):
        if getattr(self, "pf_kept", None) is not None:
            self.pf_kept.zero_()
        n = self.reusable_prefix(prompt)
        if n:
            old_len = len(self.tokens)
            self.pos, self.tokens = n, self.tokens[:n]
            # Position n-1 still in the ring: verify steps also wrote up to depth rows past the
            # last kept token, so leave that much margin before a slot could have been reused.
            if self.mtp_on and n - 1 >= old_len - self.h_ring_n + 16:
                # Position n-1's MTP entry was made with the old next token; redo it.
                h = kv_read(self.h_ring, (n - 1) % self.h_ring_n)[None]
                self.mtp_forward(torch.tensor([prompt[n]], device=self.dev), h, n - 1, logits=False)
        else:
            self.reset()
        self.last_reuse = n
        return self.forward(prompt[n:], all_logits=False)

    @staticmethod
    def sample_probs(z, temperature, top_p, top_k):
        z = z.float() / max(temperature, 1e-6)
        if 0 < top_k < z.numel():
            vals, idx = z.topk(top_k)
        else:
            vals, idx = z.sort(descending=True)
        pr = torch.softmax(vals, -1)
        if top_p < 1.0:
            keep = pr.cumsum(0) - pr < top_p
            pr = pr * keep
            pr = pr / pr.sum()
        return torch.zeros_like(z).scatter_(0, idx, pr)

    def sample(self, z, temperature, top_p, top_k, gen):
        if temperature <= 0:
            return int(z.float().argmax())
        return int(torch.multinomial(self.sample_probs(z, temperature, top_p, top_k), 1, generator=gen))

    def _nvme_awake(self, on: bool):
        """MUSE_NVME_KEEPALIVE_MS (default 50): while a request generates, a 4 KB O_DIRECT read that
        often keeps the expert drive out of its power-saving states. Hot reads come a few a round
        and rounds are ~190 ms apart; the Samsung drive here sleeps after ~100-200 ms idle and the
        next read then takes 10.3 ms instead of 1.2 (hot reads: p50 1.3, p90 10.3 ms). Reads only,
        on this rank's own drive; nothing numeric."""
        ms = float(env("MUSE_NVME_KEEPALIVE_MS", "50"))
        if not on or not ms:
            ev = getattr(self, "_awake", None)
            if ev is not None:
                ev.set()
                self._awake = None
            return
        if getattr(self, "_awake", None) is not None:
            return
        fd = self.x3pack.dfd if self.x3pack is not None and self.x3pack.dfd is not None else \
            next(iter(self.ck._dfds.values()), None) if getattr(self.ck, "direct", False) else None
        if fd is None:
            return
        import mmap
        import threading
        ev, buf, off = threading.Event(), mmap.mmap(-1, 4096), self.x3pack.data0 if self.x3pack is not None else 0

        def loop():
            while not ev.wait(ms / 1000):
                try:
                    os.preadv(fd, [buf], off)
                except OSError:
                    return
        threading.Thread(target=loop, daemon=True, name="nvme-awake").start()
        self._awake = ev

    @torch.no_grad()
    def stream(self, prompt, max_new, *, temperature=0.0, top_p=1.0, top_k=0, seed=0,
               stop=None, should_stop=None, depth=None):
        self._nvme_awake(True)
        try:
            yield from self._stream(prompt, max_new, temperature=temperature, top_p=top_p, top_k=top_k,
                                    seed=seed, stop=stop, should_stop=should_stop, depth=depth)
        finally:
            self._nvme_awake(False)

    def _stream(self, prompt, max_new, *, temperature=0.0, top_p=1.0, top_k=0, seed=0,
                stop=None, should_stop=None, depth=None):
        """Generate; with the MTP layer, speculative rounds of ``depth`` drafts. Rank 0 decides
        every token and broadcasts; both ranks run the same calls."""
        stop = self.cfg.eos if stop is None else stop
        depth = int(env("MUSE_MTP_DEPTH", "3")) if depth is None else depth
        # MUSE_MTP_DYN=tau: depth is the most drafts a round; a draft is kept only while the
        # product of the drafts' own probabilities (the MTP's softmax at the drafted token) stays
        # at or above tau, so the verify step takes 1 + kept rows. 0: always `depth` drafts.
        dyn = self.mtp_dyn
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.mtp_fill = True
        logits = self.prefill(list(prompt))
        if self.demand is not None and not self._warming:
            # After prefill: swap toward what this prompt routed to (decode benefits; the prompt's
            # KV is already computed with the substituted experts).
            n, sec = self.adapt(self.adapt_prefill_swaps)
            self.adapt_log.append(("prefill", n, round(sec, 2), self.last_churn))
        spec = self.mtp_on and depth > 0
        self.mtp_fill = not spec
        if spec:
            self.graph_rows |= set(range(1, depth + 2)) if dyn else {depth + 1}
        emitted = 0

        def decide(v, n=1):
            """Rank 0's int(s) to both ranks as a small tensor over the fast gather: broadcast_obj
            pickles through gloo's TCP pair, ~1.9 ms a call at 2-3 calls a round."""
            vals = ([int(v)] if n == 1 else [int(a) for a in v]) if self.rank == 0 else [0] * n
            out = self.comm.all_gather(torch.tensor(vals, dtype=torch.long, device=self.dev))[0].tolist()
            return out[0] if n == 1 else out

        anchor = decide(self.sample(logits[-1], temperature, top_p, top_k, gen) if self.rank == 0 else None)
        h_last = self.last_hidden[-1:]
        next_adapt = self.adapt_every
        rounds = 0
        while True:
            halt = bool(decide(bool(should_stop and self.rank == 0 and should_stop())))
            if halt:
                return
            if self.demand is not None and self.adapt_every and emitted >= next_adapt and not self._warming:
                # Both ranks reach this at the same token count (emitted is decided on rank 0).
                n, sec = self.adapt(self.adapt_decode_swaps)
                self.adapt_log.append((f"decode@{emitted}", n, round(sec, 2), self.last_churn))
                next_adapt += self.adapt_every
            rounds += 1
            if self.bg_on and self.demand is not None and rounds % self.bg_every == 0 and not self._warming:
                self.bg_step()                    # same round on both ranks
            yield anchor
            emitted += 1
            if anchor in stop or emitted >= max_new:
                self.mtp_fill = True
                self.forward([anchor], all_logits=False)
                return
            P = self.pos
            if not spec:
                logits = self.forward([anchor], all_logits=False)
                anchor = decide(self.sample(logits[-1], temperature, top_p, top_k, gen) if self.rank == 0 else None)
                continue
            # Drafts: the MTP layer at P-1 (anchor, target hidden at P-1), then on its own output.
            # Under MUSE_MTP_DYN rank 0 decides whether each draft is kept (and which it is), so
            # both ranks run the same MTP calls and verify the same rows.
            drafts, qs, confs, conf = [], [], [], 1.0
            hm, z = self.mtp_forward(torch.tensor([anchor], device=self.dev), h_last, P - 1)
            for k in range(depth):
                q, p = None, 1.0
                if temperature <= 0:
                    if dyn:
                        zl = z[-1].float()
                        p, d = torch.stack([(zl.max() - torch.logsumexp(zl, 0)).exp(), zl.argmax().float()]).tolist()
                        d = int(d)
                    else:
                        d = int(z[-1].float().argmax())
                else:
                    q = self.sample_probs(z[-1], temperature, top_p, top_k)
                    d = int(torch.multinomial(q, 1, generator=gen)) if self.rank == 0 else 0
                    if dyn and self.rank == 0:
                        p = float(q[d])
                    if not dyn:
                        d = decide(d)
                if dyn:
                    conf *= p
                    keep, d = decide([conf >= dyn, d], 2)
                    if not keep:
                        break
                qs.append(q)
                drafts.append(d)
                confs.append(conf)
                if k + 1 < depth:
                    hm, z = self.mtp_forward(torch.tensor([d], device=self.dev), hm, P + k)
            nd = len(drafts)
            rows = [anchor] + drafts
            tl = self.forward(rows)                                  # [nd+1, V]
            th = self.last_hidden                                    # [nd+1, H]
            decision = None
            if self.rank == 0:
                n_acc, nxt = nd, None
                for k in range(nd):
                    if temperature <= 0:
                        t = int(tl[k].float().argmax())
                        if t != drafts[k]:
                            n_acc, nxt = k, t
                            break
                    else:
                        pk = self.sample_probs(tl[k], temperature, top_p, top_k)
                        dk = drafts[k]
                        ratio = float(pk[dk] / qs[k][dk].clamp_min(1e-20))
                        if float(torch.rand((), generator=gen, device=self.dev)) >= min(1.0, ratio):
                            resid = (pk - qs[k]).clamp_min(0)
                            resid = resid if float(resid.sum()) > 0 else pk
                            n_acc, nxt = k, int(torch.multinomial(resid / resid.sum(), 1, generator=gen))
                            break
                if nxt is None:
                    nxt = self.sample(tl[nd], temperature, top_p, top_k, gen)
                decision = (n_acc, nxt)
            n_acc, nxt = decide(decision, 2)
            self.spec_stats["rounds"] += 1
            self.spec_stats["accepted"] += n_acc
            self.spec_stats["drafted"] = self.spec_stats.get("drafted", 0) + nd
            if self.spec_cal is not None and dyn:
                self.spec_cal.extend((k, c, k < n_acc) for k, c in enumerate(confs))
            self.pos = P + 1 + n_acc
            self.tokens = self.tokens[:self.pos]
            # The MTP cache for positions P..P+n_acc-1 gets the accepted tokens as next tokens
            # (position P+n_acc is redone with the next anchor at the start of the next round).
            if n_acc:
                self.mtp_forward(torch.tensor(drafts[:n_acc], device=self.dev), th[:n_acc], P, logits=False)
            h_last = th[n_acc:n_acc + 1]
            for d in drafts[:n_acc]:
                yield d
                emitted += 1
                if d in stop or emitted >= max_new:
                    return
            anchor = nxt

    def warmup(self, tokens: int = 48):
        """MUSE_WARMUP (default 1): capture the decode graphs before the first request. Dynamic MTP
        depth captures a split set per verify row count and MTP graphs per fill size the first time
        each occurs, so the first request after boot ran ~40% slower. Two short greedy generations
        on synthetic tokens -- a repeating pattern (confident drafts, deep verify steps) and an
        irregular run (shallow ones). The graphs must be captured with the demand and miss
        counters live (a capture bakes in what runs), so those stay on and are zeroed after;
        adaptation and the background step are skipped, so residency and the stats DB are
        untouched. Both ranks run the same calls."""
        if not tokens or not self.use_graphs or env("MUSE_WARMUP", "1") != "1":
            return
        t0 = time.perf_counter()
        self._warming = True
        try:
            for prompt in ([1000 + (i % 12) for i in range(192)], [(i * 7919) % 50000 + 1000 for i in range(64)]):
                for _ in self.stream(prompt, tokens):
                    pass
        finally:
            self._warming = False
            if self.demand is not None:
                self.demand.zero_()
                self.demand_tokens = 0
            if self.miss_stats is not None:
                self.miss_totals()
                for k in ("picks", "missed", "mass", "mass_missed"):
                    self.miss_stats[k] = 0 if k in ("picks", "missed") else 0.0
            self.reset()
        self.log(f"[glm] warm-up: {len(self.graphs)} graphs captured in {time.perf_counter() - t0:.1f}s")

    def generate(self, prompt, max_new, **kw):
        return list(self.stream(list(prompt), max_new, **kw))

    @torch.no_grad()
    def forward(self, ids, *, all_logits: bool = True):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev).reshape(-1)
        if self.pos + ids.numel() > self.max_seq:
            raise ValueError(f"context {self.pos + ids.numel()} exceeds max_seq {self.max_seq}")
        T = ids.numel()
        self.tokens.extend(ids.tolist())
        if self.demand is not None:
            self.demand_tokens += T
        if T <= self.decode_rows:
            self.pos_t.fill_(self.pos)
            if self.use_graphs and T in self.graph_rows and self.miss == "substitute":
                out, hn = self._replay_split(ids) if self.hot_n else self._replay(ids)
            else:
                out, hn = self._step(ids)
            self._after_rows(ids, hn, self.pos)
            self.pos += T
            return out if all_logits else out[-1:]
        outs = []
        for c0 in range(0, T, self.chunk):
            part = ids[c0:c0 + self.chunk]
            self.pos_t.fill_(self.pos)
            h = self._layers(self.embed_ids(part))
            if all_logits:
                z, hn = self.logits(h)
                outs.append(z)
            else:
                # Only the prompt's last row is sampled from: the head over every row was a
                # [chunk, vocab] fp32 product and gather nobody read (1.4 s of a 12K prompt at
                # 2,048-row chunks, 2.5 GB of transient at 4,096).
                hn = rms(h, self.final_norm, self.cfg.rms_eps)
                if c0 + self.chunk >= T:
                    outs.append(self.comm.cat_last(self.head(hn[-1:], out_dtype=torch.float32)))
            self._after_rows(part, hn, self.pos, next_ids=ids[c0 + 1:c0 + self.chunk + 1])
            self.pos += part.numel()
        return torch.cat(outs)

    def _after_rows(self, ids, hn, pos0, next_ids=None):
        """Keep the rows' post-norm hidden; in prefill also run the MTP layer over the rows
        whose next token is known (fills its cache; the last row waits for the sampled token)."""
        T = ids.numel()
        idx = (torch.arange(pos0, pos0 + T, device=self.dev)) % self.h_ring_n
        kv_write(self.h_ring, idx, hn)
        self.last_hidden = hn
        if self.mtp_on and self.mtp_fill and next_ids is not None and next_ids.numel():
            n = min(T, next_ids.numel())
            self.mtp_forward(next_ids[:n], hn[:n], pos0, logits=False)
