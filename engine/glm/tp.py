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


def rms(x, weight, eps):
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


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
        total = len(self.moe_layers) * cfg.experts
        transient = cfg.experts if self.miss == "stream" else 0
        t0 = time.perf_counter()
        self._load_dense()
        self._alloc_cache()
        per = Arena.slot_bytes(cfg.moe_inter, cfg.hidden, W)
        want = slots if slots is not None else env("MUSE_SLOTS", "auto")
        if want == "auto":
            self.ck.drop_cache()
            free = max(torch.cuda.mem_get_info(self.dev)[0], mem_available())
            reserve = float(env("MUSE_RESERVE_GB", "20")) * 1e9
            share = float(env("MUSE_ARENA_SHARE", "0.88"))
            mine = max(0, int(share * (free - reserve) // per) - transient - fixed)
            want = min(self.comm.gather_objects(mine))
            self.log(f"[glm] auto slots: {want} routed + {fixed} fixed ({free / 1e9:.1f} GB available, "
                     f"{reserve / 1e9:.0f} GB floor, share {share}, {per / 1e6:.1f} MB a slot)")
        self.slots_resident = min(int(want), total)
        self._boot_guard(transient)
        self.arena = Arena(cfg.moe_inter, cfg.hidden, fixed + self.slots_resident + transient, r, W, self.dev)
        self.lut = torch.full((self.nL, cfg.experts), -1, dtype=torch.int32, device=self.dev)
        self.owner: list[tuple[int, int] | None] = [None] * self.arena.slots
        self._place_fixed()
        self._place_resident()
        first_t = fixed + self.slots_resident
        self.transient = list(range(first_t, first_t + transient))
        self.allowed = (self.lut >= 0) if self.miss == "substitute" else None
        self.pos_t = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.graphs: dict[int, tuple] = {}
        self.stats = {"stream_loads": 0, "stream_s": 0.0}
        self.miss_stats = None
        self.record = None
        self.mtp_fill = True
        self.spec_stats = {"rounds": 0, "accepted": 0}
        if self.mtp_on:
            self.graph_rows = {1, int(env("MUSE_MTP_DEPTH", "3")) + 1}
        torch.cuda.synchronize()
        self.log(f"[glm] rank {r}/{W} ready in {time.perf_counter() - t0:.1f}s: dense {self.dense_bytes / 1e9:.2f} GB, "
                 f"{self.slots_resident} of {total} routed experts resident + {fixed} fixed "
                 f"({self.arena.bytes_per_slot * self.arena.slots / 1e9:.2f} GB), miss={self.miss}, "
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
                                         ("tp.py", "ref.py", "weights.py", "config.py", "fp8_block.py"))).hexdigest()[:16]
        mine = {"version": VERSION, "code": digest, "miss": self.miss, "native": self.native,
                "prefill_moe": self.prefill_moe, "graphs": self.use_graphs, "max_seq": self.max_seq,
                "chunk": self.chunk, "slots": self.slots_resident, "transient": transient, "mtp": self.mtp_on,
                "head": env("MUSE_HEAD", "bf16"), "depth": env("MUSE_MTP_DEPTH", "3"),
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
        if env("MUSE_HEAD", "bf16") == "fp8":
            # Halves the largest single read of a decode step (vocab 154880: ~0.95 GB a rank).
            from .weights import _fp8_quant
            hq, hs = _fp8_quant(head.to(d))
            head = FP8Block(hq, hs)
        self.head = Lin(head, d)
        self.final_norm = ck.get(P + "norm.weight").to(d)
        self.L = []
        nbytes = self.embed.numel() * 2 + self.head.nbytes
        for L in range(self.nL):
            p = f"{P}layers.{L}."
            a = p + "self_attn."
            kvb = ck.fp8(a + "kv_b_proj.weight", (r * self.hl * hv, (r + 1) * self.hl * hv)).dequant(torch.bfloat16)
            kvb = kvb.view(self.hl, hv, cfg.kv_lora)
            o = ck.fp8(a + "o_proj.weight")
            lw = {
                "ln_in": ck.get(p + "input_layernorm.weight").to(d),
                "ln_post": ck.get(p + "post_attention_layernorm.weight").to(d),
                "q_a": Lin(ck.fp8(a + "q_a_proj.weight"), d),
                "q_norm": ck.get(a + "q_a_layernorm.weight").to(d),
                "q_b": Lin(ck.fp8(a + "q_b_proj.weight", (r * self.hl * hq, (r + 1) * self.hl * hq)), d),
                "kv_a": Lin(ck.fp8(a + "kv_a_proj_with_mqa.weight"), d),
                "kv_norm": ck.get(a + "kv_a_layernorm.weight").to(d),
                # Absorbed MLA: kv_b's key half [h, nope, c] and value half [h, v, c], BF16.
                "w_uk": kvb[:, :cfg.nope].contiguous().to(d),
                "w_uv": kvb[:, cfg.nope:].contiguous().to(d),
                "o": Lin(o.cols(r * self.hl * cfg.v_dim, (r + 1) * self.hl * cfg.v_dim), d),
            }
            if L == self.mtp_L:
                lw.update({"enorm": ck.get(p + "enorm.weight").to(d), "hnorm": ck.get(p + "hnorm.weight").to(d),
                           "sh_norm": ck.get(p + "shared_head.norm.weight").to(d),
                           # rows split over the ranks + a gather: half the bytes of a draft step's 151 MB
                           "eh": Lin(ck.get_rows(p + "eh_proj.weight", r * cfg.hidden // W, (r + 1) * cfg.hidden // W), d)})
            if self.full(L):
                i = a + "indexer."
                lw.update({"i_q": Lin(ck.fp8(i + "wq_b.weight"), d), "i_k": Lin(ck.fp8(i + "wk.weight"), d),
                           "i_kn_w": ck.get(i + "k_norm.weight").to(d).float(),
                           "i_kn_b": ck.get(i + "k_norm.bias").to(d).float(),
                           "i_w": ck.get(i + "weights_proj.weight").to(d).float()})
            if not self.dense(L):
                lw["router"] = ck.get(p + "mlp.gate.weight").to(d).float()
                lw["bias"] = ck.get(p + "mlp.gate.e_score_correction_bias").to(d).float()
            nbytes += sum(v.nbytes for v in lw.values() if isinstance(v, Lin))
            nbytes += (lw["w_uk"].numel() + lw["w_uv"].numel()) * 2
            self.L.append(lw)
            del kvb, o
        self.dense_bytes = nbytes
        D = cfg.rope_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float, device=d) / D))
        f = torch.arange(self.max_seq, device=d, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((f, f), -1)
        self.cos, self.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        self.ck.drop_cache()

    def _alloc_cache(self):
        cfg, kw = self.cfg, dict(dtype=torch.bfloat16, device=self.dev)
        self.c_cache = [torch.zeros((self.max_seq, cfg.kv_lora), **kw) for _ in range(self.nL)]
        self.r_cache = [torch.zeros((self.max_seq, cfg.rope_dim), **kw) for _ in range(self.nL)]
        self.i_cache = {L: torch.zeros((self.max_seq, cfg.index_dim), **kw)
                        for L in range(self.nL) if self.full(L)}
        # Post-norm target hidden of recent positions: the MTP layer's input when a follow-up
        # turn reuses the cache and its last reused position gets a new next token.
        self.h_ring_n = self.chunk + 64
        self.h_ring = torch.zeros((self.h_ring_n, cfg.hidden), **kw)

    # ------------------------------------------------------------------ experts
    def _expert_parts(self, L, e):
        """(gate, up, down) of this rank for routed expert e (or "shared") of layer L."""
        r, Il, Hl = self.rank, self.arena.Il, self.arena.Hl
        b = f"{P}layers.{L}.mlp." + ("shared_experts." if e == "shared" else f"experts.{e}.")
        return (self.ck.mx_rows(b + "gate_proj.weight", r * Il, (r + 1) * Il),
                self.ck.mx_rows(b + "up_proj.weight", r * Il, (r + 1) * Il),
                self.ck.mx_rows(b + "down_proj.weight", r * Hl, (r + 1) * Hl))

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
        self.first_routed = slot
        self.ck.drop_cache()

    def _place_resident(self, resident=None):
        cfg = self.cfg
        if resident is None and env("MUSE_RESIDENT", ""):
            raw = json.loads(Path(env("MUSE_RESIDENT", "")).read_text())
            raw = raw.get("resident", raw)
            resident = {int(k): [int(e) for e in v] for k, v in raw.items()}
        if resident is None:
            total = len(self.moe_layers) * cfg.experts
            if self.slots_resident >= total:
                resident = {L: list(range(cfg.experts)) for L in self.moe_layers}
            else:
                per = self.slots_resident // len(self.moe_layers)
                if per < cfg.topk:
                    raise ValueError(f"{self.slots_resident} slots give {per} experts a layer, fewer than "
                                     f"top-{cfg.topk}: substitution could not fill the routing")
                extra = self.slots_resident - per * len(self.moe_layers)
                resident = {}
                for k, L in enumerate(self.moe_layers):
                    n = per + (1 if k < extra else 0)
                    order = torch.argsort(self.L[L]["bias"].cpu(), stable=True).tolist()
                    resident[L] = sorted(order[:n])
        slot = self.first_routed
        t0 = time.perf_counter()
        for L in self.moe_layers:
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

    def _load(self, L, e, slot, parts=None):
        self.arena.put(slot, *(parts or self._expert_parts(L, e)))
        old = self.owner[slot]
        if old is not None and int(self.lut[old[0], old[1]]) == slot:
            self.lut[old[0], old[1]] = -1
        self.owner[slot] = (L, e)
        self.lut[L, e] = slot

    def reset(self):
        self.pos = 0
        self.tokens: list[int] = []

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
        self.i_cache[L].index_copy_(0, positions, k)
        K = self.i_cache[L][:S]
        w = F.linear(x.float(), lw["i_w"]) * cfg.index_heads ** -0.5                 # [T, h]
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
        self.c_cache[L].index_copy_(0, positions, c)
        self.r_cache[L].index_copy_(0, positions, kr)
        topk = self._indexer(L, x, q_resid, positions, cos, sin, S) if self.full(L) else topk_prev
        q_lat = torch.einsum("thn,hnc->thc", q_nope, lw["w_uk"])                  # [T, h, kv_lora]
        scale = cfg.qk_dim ** -0.5
        C, R = self.c_cache[L], self.r_cache[L]
        if not decode and S <= cfg.index_topk:
            # Every cached position is selected: dense causal attention over [0, S).
            sc = (torch.einsum("thc,sc->ths", q_lat, C[:S]) + torch.einsum("thr,sr->ths", q_rope, R[:S])).float() * scale
            kpos = torch.arange(S, device=x.device)
            sc = sc.masked_fill(kpos[None, None] > positions[:, None, None], float("-inf"))
            p = torch.softmax(sc, -1).to(torch.bfloat16)
            o_lat = torch.einsum("ths,sc->thc", p, C[:S])
        else:
            outs = []
            for t0 in range(0, T, 32):
                sel = topk[t0:t0 + 32].long()
                cg, rg = C[sel], R[sel]                                            # [t, k, c|r]
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
    def _route(self, L, x):
        """(slots int32 [T, k'], weights fp32 [T, k']) including the shared expert or the
        dense layer's pseudo-experts."""
        cfg, lw = self.cfg, self.L[L]
        T = x.shape[0]
        fixed = self.fixed[L][None].expand(T, -1)
        ones = torch.ones(fixed.shape, dtype=torch.float32, device=x.device)
        if self.dense(L):
            return fixed.contiguous(), ones
        from .ref import route
        allowed = self.allowed[L] if self.allowed is not None else None
        w, ids = route(cfg, x, lw["router"], lw["bias"], allowed)
        if self.record is not None:
            self.record.append((L, ids.cpu(), w.cpu()))
        if self.miss_stats is not None and allowed is not None:
            w0, ids0 = route(cfg, x, lw["router"], lw["bias"], None)
            missed = ~allowed[ids0]
            ms = self.miss_stats
            ms["picks"] += ids0.numel()
            ms["missed"] += int(missed.sum())
            ms["mass"] += float(w0.sum())
            ms["mass_missed"] += float((w0 * missed).sum())
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
        import fp4_moe_cuda as CUDA
        A, cfg = self.arena, self.cfg
        T, k = slots.shape
        P_ = T * k
        bs, bp, br, nb = CUDA.build_routing_small(slots.reshape(-1).contiguous(), 16, k)
        h = torch.empty((P_, A.Il), dtype=torch.bfloat16, device=x.device)
        ones = torch.ones(P_, dtype=torch.float32, device=x.device)
        CUDA.up(x, A.w1, A.s1, A.w3, A.s3, h, ones, bs, br, float("inf"), k, A.Il, cfg.hidden, nb, -1)
        full = self.comm.cat_last(h)
        parts = torch.empty((P_, A.Hl), dtype=torch.float32, device=x.device)
        CUDA.down(full, A.w2, A.s2, parts, bs, bp, k, A.Hl, cfg.moe_inter, T, nb, -1, False, 1)
        return parts.view(k, T, A.Hl)

    def _experts_triton(self, x, slots):
        import fp4_moe as K4
        A, cfg = self.arena, self.cfg
        T, k = slots.shape
        P_ = T * k
        BM = K4._pick_bm(P_)
        uniq, inv = torch.unique(slots, return_inverse=True)
        bs, bp, nb = K4.build_routing(inv.to(torch.int32).view_as(slots), int(uniq.numel()), BM)
        bs = torch.where(bs >= 0, uniq.to(torch.int32)[bs.clamp_min(0)], bs)
        bn1, nw1, ns1 = K4._UP_CFG[BM]
        bn2, nw2, ns2 = K4._DOWN_CFG[BM]
        h = torch.empty((P_, A.Il), dtype=torch.bfloat16, device=x.device)
        ones = torch.ones(P_, dtype=torch.float32, device=x.device)
        K4._moe_up_kernel[(nb, A.Il // bn1)](x, A.w1, A.s1, A.w3, A.s3, h, ones, bs, bp, x.stride(0), h.stride(0),
                                             float("inf"), TOPK=k, N=A.Il, K=cfg.hidden, BM=BM, BN=bn1,
                                             SCALED=False, NULL_SLOT=-1, num_warps=nw1, num_stages=ns1)
        full = self.comm.cat_last(h)
        parts = torch.empty((P_, A.Hl), dtype=torch.float32, device=x.device)
        K4._moe_down_kernel[(nb, A.Hl // bn2)](full, A.w2, A.s2, parts, bs, bp, full.stride(0), parts.stride(0),
                                               TOPK=k, N=A.Hl, K=cfg.moe_inter, BM=BM, BN=bn2, NTOK=T,
                                               SCALED=False, NULL_SLOT=-1, PARTIAL=False, PARTS_WORLD=1,
                                               num_warps=nw2, num_stages=ns2)
        return parts.view(k, T, A.Hl)

    def mlp(self, L, x):
        self._streamed = None
        slots, w = self._route(L, x)
        T, k = slots.shape
        x = x.contiguous()
        if self.native and T * k <= 512:
            parts = self._experts_native(x, slots)
        else:
            parts = self._experts_triton(x, slots)
        if self._streamed:
            sl, missing = self._streamed
            for e in missing:
                self.lut[sl, e] = -1
        y = (self.comm.cat_last(parts) * w.t()[:, :, None]).sum(0)
        return y.to(torch.bfloat16)

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
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._step(static_ids)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out, static_h = self._step(static_ids)
            entry = self.graphs[T] = (graph, static_ids, static_out, static_h)
            self.log(f"[glm] captured decode graph for {T} rows")
        graph, static_ids, static_out, static_h = entry
        static_ids.copy_(ids)
        graph.replay()
        return static_out.clone(), static_h.clone()

    def _mtp_core(self, next_ids, h_prev):
        """The MTP layer at positions pos_t.. (device) -- graph-capturable for one row."""
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
        return hn, self.comm.cat_last(self.head(hn, out_dtype=torch.float32))

    @torch.no_grad()
    def mtp_forward(self, next_ids, h_prev, pos0):
        """The MTP layer at positions pos0..: (token i+1, post-norm hidden at i) -> (its
        post-norm hidden, logits for token i+2). Writes the MTP layer's caches. One-row calls
        (the draft chain) replay a captured graph."""
        next_ids = torch.as_tensor(next_ids, dtype=torch.long, device=self.dev).reshape(-1)
        T = next_ids.numel()
        saved, saved_t = self.pos, self.pos_t.clone()
        self.pos = pos0
        self.pos_t.fill_(pos0)
        if T == 1 and self.use_graphs and self.miss == "substitute":
            entry = self.graphs.get("mtp")
            if entry is None:
                st_ids, st_h = next_ids.clone(), h_prev.to(torch.bfloat16).clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._mtp_core(st_ids, st_h)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    st_hn, st_z = self._mtp_core(st_ids, st_h)
                entry = self.graphs["mtp"] = (graph, st_ids, st_h, st_hn, st_z)
                self.log("[glm] captured MTP graph")
            graph, st_ids, st_h, st_hn, st_z = entry
            st_ids.copy_(next_ids)
            st_h.copy_(h_prev)
            graph.replay()
            out = st_hn.clone(), st_z.clone()
        else:
            out = self._mtp_core(next_ids, h_prev)
        self.pos = saved
        self.pos_t.copy_(saved_t)
        return out

    def reusable_prefix(self, prompt):
        n, limit = 0, min(len(prompt) - 1, len(self.tokens))
        while n < limit and prompt[n] == self.tokens[n]:
            n += 1
        if n == 0 or self.pos - n > self.chunk:
            return 0
        return n

    def prefill(self, prompt):
        n = self.reusable_prefix(prompt)
        if n:
            self.pos, self.tokens = n, self.tokens[:n]
            if self.mtp_on:
                # Position n-1's MTP entry was made with the old next token; redo it.
                h = self.h_ring[(n - 1) % self.h_ring_n][None]
                self.mtp_forward(torch.tensor([prompt[n]], device=self.dev), h, n - 1)
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

    @torch.no_grad()
    def stream(self, prompt, max_new, *, temperature=0.0, top_p=1.0, top_k=0, seed=0,
               stop=None, should_stop=None, depth=None):
        """Generate; with the MTP layer, speculative rounds of ``depth`` drafts. Rank 0 decides
        every token and broadcasts; both ranks run the same calls."""
        stop = self.cfg.eos if stop is None else stop
        depth = int(env("MUSE_MTP_DEPTH", "3")) if depth is None else depth
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.mtp_fill = True
        logits = self.prefill(list(prompt))
        spec = self.mtp_on and depth > 0
        self.mtp_fill = not spec
        if spec:
            self.graph_rows.add(depth + 1)
        emitted = 0

        def decide(x):
            return self.comm.broadcast_obj(x)

        anchor = decide(self.sample(logits[-1], temperature, top_p, top_k, gen) if self.rank == 0 else None)
        h_last = self.last_hidden[-1:]
        while True:
            halt = decide(bool(should_stop and self.rank == 0 and should_stop()))
            if halt:
                return
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
            drafts, qs = [], []
            hm, z = self.mtp_forward(torch.tensor([anchor], device=self.dev), h_last, P - 1)
            for k in range(depth):
                if temperature <= 0:
                    d = int(z[-1].float().argmax())
                    qs.append(None)
                else:
                    q = self.sample_probs(z[-1], temperature, top_p, top_k)
                    d = decide(int(torch.multinomial(q, 1, generator=gen)) if self.rank == 0 else None)
                    qs.append(q)
                drafts.append(d)
                if k + 1 < depth:
                    hm, z = self.mtp_forward(torch.tensor([d], device=self.dev), hm, P + k)
            rows = [anchor] + drafts
            tl = self.forward(rows)                                  # [depth+1, V]
            th = self.last_hidden                                    # [depth+1, H]
            decision = None
            if self.rank == 0:
                n_acc, nxt = depth, None
                for k in range(depth):
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
                    nxt = self.sample(tl[depth], temperature, top_p, top_k, gen)
                decision = (n_acc, nxt)
            n_acc, nxt = decide(decision)
            self.spec_stats["rounds"] += 1
            self.spec_stats["accepted"] += n_acc
            self.pos = P + 1 + n_acc
            self.tokens = self.tokens[:self.pos]
            # The MTP cache for positions P..P+n_acc-1 gets the accepted tokens as next tokens
            # (position P+n_acc is redone with the next anchor at the start of the next round).
            if n_acc:
                self.mtp_forward(torch.tensor(drafts[:n_acc], device=self.dev), th[:n_acc], P)
            h_last = th[n_acc:n_acc + 1]
            for d in drafts[:n_acc]:
                yield d
                emitted += 1
                if d in stop or emitted >= max_new:
                    return
            anchor = nxt

    def generate(self, prompt, max_new, **kw):
        return list(self.stream(list(prompt), max_new, **kw))

    @torch.no_grad()
    def forward(self, ids, *, all_logits: bool = True):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev).reshape(-1)
        if self.pos + ids.numel() > self.max_seq:
            raise ValueError(f"context {self.pos + ids.numel()} exceeds max_seq {self.max_seq}")
        T = ids.numel()
        self.tokens.extend(ids.tolist())
        if T <= self.decode_rows:
            self.pos_t.fill_(self.pos)
            if self.use_graphs and T in self.graph_rows and self.miss == "substitute":
                out, hn = self._replay(ids)
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
            z, hn = self.logits(h)
            self._after_rows(part, hn, self.pos, next_ids=ids[c0 + 1:c0 + self.chunk + 1])
            self.pos += part.numel()
            if all_logits:
                outs.append(z)
            elif c0 + self.chunk >= T:
                outs.append(z[-1:])
        return torch.cat(outs)

    def _after_rows(self, ids, hn, pos0, next_ids=None):
        """Keep the rows' post-norm hidden; in prefill also run the MTP layer over the rows
        whose next token is known (fills its cache; the last row waits for the sampled token)."""
        T = ids.numel()
        idx = (torch.arange(pos0, pos0 + T, device=self.dev)) % self.h_ring_n
        self.h_ring.index_copy_(0, idx, hn)
        self.last_hidden = hn
        if self.mtp_on and self.mtp_fill and next_ids is not None and next_ids.numel():
            n = min(T, next_ids.numel())
            self.mtp_forward(next_ids[:n], hn[:n], pos0)
