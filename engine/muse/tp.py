"""Muse Spark text model on two ranks (TP2): dense weights split by heads/rows, experts by rows.

Per layer, per rank (r of W=2):

  attention  q/gate_proj rows of this rank's 64 query heads, k/v of its 8 KV heads, o_proj
             columns of those heads -> partial [T, H], all-reduce. KV cache holds the local heads:
             a ring of ``window + chunk`` positions on sliding layers, the full context on the
             16 NoPE full-attention layers.
  router     replicated FP32 [E, H] on both ranks; both ranks compute the same routing from the
             same residual stream, so expert placement never needs a broadcast.
  latent     pre_expert_proj rows split -> all-gather [T, M] -> pre_expert_norm.
  experts    MXFP4 arena slots hold this rank's half of every resident expert: gate/up rows
             (intermediate I/2) and down rows (output M/2). up -> all-gather h [P, I] -> down ->
             all-gather per-pair outputs [k, T, M] -> post_expert_norm per pair, router weight,
             sum -> post_expert_proj rows split -> all-gather [T, H].
  head       vocab rows split; softcapped logits all-gathered.

Every rank holds the same residual stream bit for bit: each reduction is an all-reduce of two
partials (a + b == b + a) or an all-gather. Sampling is rank 0's and broadcast.

Decode MoE uses the native kernels of tools/fp4_moe_cuda.py (up to 512 pairs a call); prefill
dequantizes each used expert's slot to BF16 and multiplies (correct, not fast). Expert misses:
``substitute`` masks non-resident experts in the router (the policy V4.1 serves with);
``stream`` loads them into transient slots for the step (exact, slow; for traces and quality
references).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from .config import MuseConfig  # noqa: E402
from .ref import norm_centered, norm_input_scaled, norm_plain, residual_gate, route  # noqa: E402
from .weights import GROUP, Checkpoint, PREFIX, dequant_mxfp4, quantize_mxfp4  # noqa: E402

VERSION = 1


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Comm:
    """Rank bookkeeping and the four collectives the model uses."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        self.fast = False
        if self.world > 1 and not dist.is_initialized():
            from engine.dist import _apply_default_net_env
            _apply_default_net_env()
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl", rank=self.rank, world_size=self.world,
                timeout=datetime.timedelta(seconds=float(env("DSV41_DIST_TIMEOUT_S", "600"))))
            if env("MUSE_COMM", "roce") == "roce":
                from engine import comm
                os.environ.setdefault("DSV41_COMM_BACKEND", "roce")
                comm.init(self.device)
                self.fast = comm._COMM is not None

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum over ranks. Decode-sized tensors go as an all-gather plus a local sum: the fast
        RoCE gather instead of an NCCL all-reduce, and the sum runs rank 0 + rank 1 in the same
        order on both ranks, so both hold the same bits."""
        if self.world == 1:
            return t
        if self.fast and t.numel() * t.element_size() <= 256 * 1024:
            return self.all_gather(t).sum(0)
        dist.all_reduce(t)
        return t

    def all_gather(self, t: torch.Tensor) -> torch.Tensor:
        """[...] -> [W, ...]."""
        t = t.contiguous()
        if self.world == 1:
            return t[None]
        out = torch.empty((self.world, *t.shape), dtype=t.dtype, device=t.device)
        if self.fast:
            from engine import comm
            comm.all_gather_fast(out, t)
        else:
            dist.all_gather_into_tensor(out, t)
        return out

    def broadcast_obj(self, obj):
        if self.world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def gather_objects(self, obj) -> list:
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def cat_last(self, t: torch.Tensor) -> torch.Tensor:
        """All-gather and concatenate rank shards along the last dimension."""
        g = self.all_gather(t)
        return g.movedim(0, -2).reshape(*t.shape[:-1], self.world * t.shape[-1])


class Dense:
    """One projection in BF16 or FP8 (tools/fp8_linear.py, 32x32 E8M0 block scales)."""

    def __init__(self, w: torch.Tensor, fmt: str):
        self.fmt = fmt
        if fmt == "fp8":
            from fp8_linear import quantize_to_fp8
            self.q = quantize_to_fp8(w)
            self.w = None
        else:
            self.w = w.to(torch.bfloat16).contiguous()
        self.out_features = w.shape[0]

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        if self.fmt == "fp8":
            from fp8_linear import fp8_linear
            return fp8_linear(x.contiguous(), self.q, out_dtype=out_dtype)
        y = F.linear(x, self.w)
        return y if out_dtype == torch.bfloat16 else y.to(out_dtype)

    @property
    def nbytes(self) -> int:
        if self.fmt == "fp8":
            return self.q.w.numel() + self.q.s.numel()
        return self.w.numel() * 2


class Arena:
    """This rank's half of resident experts in MXFP4, plus the (layer, expert) -> slot table."""

    def __init__(self, cfg: MuseConfig, slots: int, rank: int, world: int, device):
        I, M = cfg.moe_inter, cfg.moe_hidden
        self.cfg, self.slots, self.rank, self.world = cfg, slots, rank, world
        self.Il, self.Ml = I // world, M // world
        u8 = dict(dtype=torch.uint8, device=device)
        self.w1 = torch.zeros((slots, self.Il, M // 2), **u8)
        self.s1 = torch.zeros((slots, self.Il, M // GROUP), **u8)
        self.w3 = torch.zeros((slots, self.Il, M // 2), **u8)
        self.s3 = torch.zeros((slots, self.Il, M // GROUP), **u8)
        self.w2 = torch.zeros((slots, self.Ml, I // 2), **u8)
        self.s2 = torch.zeros((slots, self.Ml, I // GROUP), **u8)
        self.lut = torch.full((cfg.layers, cfg.experts), -1, dtype=torch.int32, device=device)
        self.owner: list[tuple[int, int] | None] = [None] * slots

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    @staticmethod
    def slot_bytes(cfg: MuseConfig, world: int) -> int:
        I, M = cfg.moe_inter, cfg.moe_hidden
        return (2 * (I // world) * (M // 2 + M // GROUP)) + (M // world) * (I // 2 + I // GROUP)

    def load(self, ck: Checkpoint, L: int, e: int, slot: int) -> None:
        """Read this rank's rows of expert (L, e) and place them in ``slot``."""
        cfg, r = self.cfg, self.rank
        i0, i1 = r * self.Il, (r + 1) * self.Il
        m0, m1 = r * self.Ml, (r + 1) * self.Ml
        base = f"{PREFIX}layers.{L}.mlp.experts."
        fmt = ck.expert_layer_format(L)
        if fmt == "mxfp4":
            parts = {}
            for p, (a, b) in (("gate", (i0, i1)), ("up", (i0, i1)), ("down", (m0, m1))):
                parts[p] = (ck.get_rows(f"{base}{e}.{p}_proj.weight", a, b),
                            ck.get_rows(f"{base}{e}.{p}_proj.weight_scale", a, b))
        else:
            # BF16 experts (layers 0 and 61 in both 4-bit releases): quantize this rank's rows
            # on the GPU (a CPU quantizer takes seconds per matrix). Rows split before
            # quantizing give the same codes as splitting after: groups run along K.
            base_bf16 = f"{PREFIX}layers.{L}.mlp.experts."
            if ck.expert_layer_format(L) == "bf16-split":
                g = ck.get_rows(f"{base_bf16}{e}.gate_proj.weight", i0, i1)
                u = ck.get_rows(f"{base_bf16}{e}.up_proj.weight", i0, i1)
                dn = ck.get_rows(f"{base_bf16}{e}.down_proj.weight", m0, m1)
            else:
                gate, up, down = ck.expert_float(L, e)
                g, u, dn = gate[i0:i1], up[i0:i1], down[m0:m1]
            dev = self.w1.device
            parts = {"gate": quantize_mxfp4(g.to(dev)), "up": quantize_mxfp4(u.to(dev)),
                     "down": quantize_mxfp4(dn.to(dev))}
        for (w, s), (dw, ds) in ((parts["gate"], (self.w1, self.s1)),
                                 (parts["up"], (self.w3, self.s3)),
                                 (parts["down"], (self.w2, self.s2))):
            dw[slot].copy_(w.view(torch.uint8))
            ds[slot].copy_(s.view(torch.uint8))
        old = self.owner[slot]
        if old is not None and self.lut[old[0], old[1]].item() == slot:
            self.lut[old[0], old[1]] = -1
        self.owner[slot] = (L, e)
        self.lut[L, e] = slot

    def dequant(self, slot: int):
        """(gate [I/2, M], up [I/2, M], down [M/2, I]) BF16 of this rank's half (prefill path)."""
        return (dequant_mxfp4(self.w1[slot], self.s1[slot]).to(torch.bfloat16),
                dequant_mxfp4(self.w3[slot], self.s3[slot]).to(torch.bfloat16),
                dequant_mxfp4(self.w2[slot], self.s2[slot]).to(torch.bfloat16))


class MuseTP:
    def __init__(self, model_dir: str, *, max_seq: int | None = None, chunk: int | None = None,
                 dense: str | None = None, slots: int | None = None, miss: str | None = None,
                 resident: dict[int, list[int]] | None = None, log=print):
        self.comm = Comm()
        self.rank, self.world, self.dev = self.comm.rank, self.comm.world, self.comm.device
        self.log = log if self.rank == 0 else (lambda *_: None)
        self.cfg = cfg = MuseConfig.load(model_dir)
        self.ck = Checkpoint(model_dir)
        self.max_seq = int(max_seq or env("MUSE_MAX_SEQ", "32768"))
        self.chunk = int(chunk or env("MUSE_CHUNK", "1024"))
        self.dense_fmt = dense or env("MUSE_DENSE", "bf16")
        self.miss = miss or env("MUSE_MISS", "substitute")
        self.native = env("MUSE_NATIVE_MOE", "1") == "1"
        self.use_graphs = env("MUSE_GRAPHS", "1") == "1"
        self.prefill_moe = env("MUSE_PREFILL_MOE", "triton")
        self.decode_rows = 16          # steps of at most this many rows take the device-position path
        self.graph_rows = {1}          # step sizes replayed from captured graphs
        if self.miss not in ("substitute", "stream"):
            raise ValueError("MUSE_MISS must be substitute or stream")
        W, r = self.world, self.rank
        if cfg.heads % W or cfg.kv_heads % W or cfg.moe_inter % W or cfg.moe_hidden % W \
                or cfg.hidden % W or cfg.vocab % W:
            raise ValueError("model dimensions must split evenly over the ranks")
        self.hl, self.kvl = cfg.heads // W, cfg.kv_heads // W
        self.vl = cfg.vocab // W
        total = cfg.layers * cfg.experts
        transient = cfg.experts if self.miss == "stream" else 0
        want = slots or env("MUSE_SLOTS", "auto")
        t0 = time.perf_counter()
        self._load_dense()
        self._alloc_cache()
        if want == "auto":
            # Whatever is free after dense weights and caches, less a reserve for activations,
            # graph pools and the host; the smaller rank decides for both.
            # Unified memory: the page cache (a model download, the checkpoint just read) is
            # reclaimable but cudaMemGetInfo counts it as used. MemAvailable does not.
            free = torch.cuda.mem_get_info(self.dev)[0]
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable:"):
                        free = max(free, int(line.split()[1]) * 1024)
            except OSError:
                pass
            reserve = float(env("MUSE_RESERVE_GB", "12")) * 1e9
            per = Arena.slot_bytes(cfg, W)
            mine = max(0, int((free - reserve) // per) - transient)
            want = min(self.comm.gather_objects(mine))
            self.log(f"[muse] auto slots: {want} ({free / 1e9:.1f} GB free, {reserve / 1e9:.0f} GB reserve,"
                     f" {per / 1e6:.1f} MB a slot)")
        self.slots_resident = min(int(want), total)
        self._boot_guard(transient)
        self.arena = Arena(cfg, self.slots_resident + transient, r, W, self.dev)
        self.transient = list(range(self.slots_resident, self.slots_resident + transient))
        self._place_resident(resident)
        self.allowed = (self.arena.lut >= 0) if self.miss == "substitute" else None
        self.ones = torch.ones(512, dtype=torch.float32, device=self.dev)
        self.pos_t = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.graphs: dict[int, tuple] = {}
        self.record: list | None = None
        self.miss_stats: dict | None = None
        self.stats = {"stream_loads": 0, "stream_s": 0.0}
        torch.cuda.synchronize()
        self.log(f"[muse] rank {r}/{W} ready in {time.perf_counter() - t0:.1f}s: dense {self.dense_fmt}"
                 f" {self.dense_bytes / 1e9:.2f} GB, {self.slots_resident} resident slots of {total}"
                 f" ({self.arena.bytes_per_slot * self.arena.slots / 1e9:.2f} GB), miss={self.miss},"
                 f" max_seq {self.max_seq}, chunk {self.chunk}")
        self.reset()

    # ------------------------------------------------------------------ setup
    def _boot_guard(self, transient: int) -> None:
        """Both ranks must compute the same way; a mismatch would not raise, it would diverge."""
        src = Path(__file__).resolve().parent
        digest = hashlib.sha256(b"".join((src / f).read_bytes() for f in
                                         ("tp.py", "ref.py", "weights.py", "config.py"))).hexdigest()[:16]
        mine = {"version": VERSION, "code": digest, "dense": self.dense_fmt, "miss": self.miss,
                "native": self.native, "graphs": self.use_graphs, "prefill_moe": self.prefill_moe, "max_seq": self.max_seq, "chunk": self.chunk,
                "slots": self.slots_resident, "transient": transient, "fast_comm": self.comm.fast,
                "resident": hashlib.sha256(Path(env("MUSE_RESIDENT", "")).read_bytes()).hexdigest()[:16]
                if env("MUSE_RESIDENT", "") else "",
                "config": json.dumps(self.cfg.extra["raw"].get("text_config", {}), sort_keys=True)}
        allv = self.comm.gather_objects(mine)
        if any(v != allv[0] for v in allv):
            diff = {k: [v.get(k) for v in allv] for k in mine if any(v.get(k) != allv[0].get(k) for v in allv)}
            raise RuntimeError(f"muse ranks disagree: {diff}")

    def _rows(self, name: str, n_total: int) -> torch.Tensor:
        n = n_total // self.world
        return self.ck.get_rows(name, self.rank * n, (self.rank + 1) * n)

    def _load_dense(self) -> None:
        cfg, d, fmt = self.cfg, self.dev, self.dense_fmt
        H, D, W, r = cfg.hidden, cfg.head_dim, self.world, self.rank

        def g(name):
            return self.ck.get(name).to(d)

        self.embed = self._rows(PREFIX + "embed_tokens.weight", cfg.vocab).to(d, torch.bfloat16)
        self.head = Dense(self._rows("lm_head.weight", cfg.vocab).to(d), fmt)
        self.final_norm = g(PREFIX + "norm.weight")
        self.L = []
        nbytes = self.embed.numel() * 2 + self.head.nbytes
        for L in range(cfg.layers):
            p = f"{PREFIX}layers.{L}."
            a, m = p + "self_attn.", p + "mlp."
            o = self.ck.get(a + "o_proj.weight")
            hd = self.hl * D
            layer = {
                "q": Dense(self._rows(a + "q_proj.weight", cfg.heads * D).to(d), fmt),
                "k": Dense(self._rows(a + "k_proj.weight", cfg.kv_heads * D).to(d), fmt),
                "v": Dense(self._rows(a + "v_proj.weight", cfg.kv_heads * D).to(d), fmt),
                "g": Dense(self._rows(a + "gate_proj.weight", cfg.heads * D).to(d), fmt),
                "o": Dense(o[:, r * hd:(r + 1) * hd].contiguous().to(d), fmt),
                "ln_in": g(p + "input_layernorm.weight"),
                "ln_ff": g(p + "pre_feedforward_layernorm.weight"),
                "ln_post_ff": g(p + "post_feedforward_layernorm.weight"),
                "gate_attn": g(p + "post_attention_residual_gate.gate").float(),
                "gate_ff": g(p + "post_feedforward_residual_gate.gate").float(),
                "router": g(m + "gate.weight").float(),
                "bias": g(m + "gate.e_score_correction_bias").float(),
                "pre": Dense(self._rows(m + "pre_expert_proj.weight", cfg.moe_hidden).to(d), fmt),
                "pre_norm": g(m + "pre_expert_norm.weight"),
                "post_norm": g(m + "experts.post_expert_norm.weight"),
                "post": Dense(self._rows(m + "post_expert_proj.weight", H).to(d), fmt),
            }
            nbytes += sum(v.nbytes for v in layer.values() if isinstance(v, Dense))
            self.L.append(layer)
        self.dense_bytes = nbytes
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float, device=d) / D))
        freqs = torch.arange(self.max_seq, device=d, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((freqs, freqs), -1)
        self.cos, self.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def _place_resident(self, resident) -> None:
        """Fill the resident slots. Default: all experts if they fit, else per layer the experts
        with the lowest load-balancing bias (aux-loss-free routing raises the bias of experts the
        router under-uses, so a low bias marks a naturally popular expert). A routing trace
        replaces that guess (``resident`` = {layer: [experts]})."""
        cfg = self.cfg
        if resident is None and env("MUSE_RESIDENT", ""):
            raw = json.loads(Path(env("MUSE_RESIDENT", "")).read_text())
            raw = raw.get("resident", raw)
            resident = {int(k): [int(e) for e in v] for k, v in raw.items()}
            n = sum(len(v) for v in resident.values())
            if n > self.slots_resident:
                # Keep the file's order within each layer and trim the layers evenly.
                raise ValueError(f"MUSE_RESIDENT lists {n} experts, {self.slots_resident} slots")
        if resident is None:
            if self.slots_resident >= cfg.layers * cfg.experts:
                resident = {L: list(range(cfg.experts)) for L in range(cfg.layers)}
            else:
                per = self.slots_resident // cfg.layers
                extra = self.slots_resident - per * cfg.layers
                resident = {}
                for L in range(cfg.layers):
                    n = per + (1 if L < extra else 0)
                    order = torch.argsort(self.L[L]["bias"].cpu(), stable=True).tolist()
                    resident[L] = sorted(order[:n])
        slot = 0
        t0 = time.perf_counter()
        for L in range(cfg.layers):
            for e in resident.get(L, []):
                if slot >= self.slots_resident:
                    raise ValueError("resident set larger than MUSE_SLOTS")
                self.arena.load(self.ck, L, e, slot)
                slot += 1
        self.log(f"[muse] placed {slot} experts in {time.perf_counter() - t0:.1f}s")

    def _alloc_cache(self) -> None:
        cfg = self.cfg
        self.ring = cfg.sliding_window + self.chunk
        kw = dict(dtype=torch.bfloat16, device=self.dev)
        self.kc, self.vc = [], []
        for L in range(cfg.layers):
            n = self.ring if cfg.sliding[L] else self.max_seq
            self.kc.append(torch.zeros((n, self.kvl, cfg.head_dim), **kw))
            self.vc.append(torch.zeros((n, self.kvl, cfg.head_dim), **kw))

    def reset(self) -> None:
        self.pos = 0
        self.tokens: list[int] = []

    def reusable_prefix(self, prompt: list[int]) -> int:
        """Tokens of ``prompt`` whose KV the cache already holds from the previous request.

        Full layers keep every position, but a sliding layer's ring only holds the last
        ``window + chunk`` positions, so rolling back is possible by at most ``chunk`` positions
        (the ring's slack beyond the window). An agent turn that appends to the conversation
        reuses everything; an edit further back recomputes. At least one prompt token is always
        recomputed, for its logits."""
        n = 0
        limit = min(len(prompt) - 1, len(self.tokens))
        while n < limit and prompt[n] == self.tokens[n]:
            n += 1
        if n == 0 or self.pos - n > self.chunk:
            return 0
        return n

    def prefill(self, prompt: list[int]) -> torch.Tensor:
        """Logits of the prompt's last token, reusing the cached prefix when it matches."""
        n = self.reusable_prefix(prompt)
        if n:
            self.pos = n
            self.tokens = self.tokens[:n]
        else:
            self.reset()
        self.last_reuse = n
        return self.forward(prompt[n:], all_logits=False)

    # ------------------------------------------------------------------ layers
    def attention(self, L: int, x: torch.Tensor) -> torch.Tensor:
        cfg, lw = self.cfg, self.L[L]
        T, D = x.shape[0], cfg.head_dim
        q = lw["q"](x).view(T, self.hl, D)
        k = lw["k"](x).view(T, self.kvl, D)
        v = lw["v"](x).view(T, self.kvl, D)
        q = norm_plain(q, cfg.rms_eps) * cfg.qk_scale
        k = norm_plain(k, cfg.rms_eps)
        if T <= self.decode_rows:
            return self._attention_decode(L, x, q, k, v)
        p0 = self.pos
        if cfg.rope[L]:
            cos, sin = self.cos[p0:p0 + T, None], self.sin[p0:p0 + T, None]
            q = q * cos + torch.cat((-q[..., D // 2:], q[..., :D // 2]), -1) * sin
            k = k * cos + torch.cat((-k[..., D // 2:], k[..., :D // 2]), -1) * sin
        positions = torch.arange(p0, p0 + T, device=x.device)
        if cfg.sliding[L]:
            idx = positions % self.ring
            self.kc[L][idx] = k
            self.vc[L][idx] = v
            lo = max(0, p0 + 1 - cfg.sliding_window)
            kpos = torch.arange(lo, p0 + T, device=x.device)
            K, V = self.kc[L][kpos % self.ring], self.vc[L][kpos % self.ring]
            mask = (kpos[None] <= positions[:, None]) & (kpos[None] > positions[:, None] - cfg.sliding_window)
        else:
            self.kc[L][p0:p0 + T] = k
            self.vc[L][p0:p0 + T] = v
            K, V = self.kc[L][:p0 + T], self.vc[L][:p0 + T]
            kpos = torch.arange(0, p0 + T, device=x.device)
            mask = kpos[None] <= positions[:, None]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
            attn_mask=mask[None, None], scale=D ** -0.5, enable_gqa=True)[0].transpose(0, 1)
        out = norm_plain(out, cfg.rms_eps).reshape(T, self.hl * D)
        out = out * torch.sigmoid(lw["g"](x))
        partial = lw["o"](out, out_dtype=torch.float32)
        return self.comm.all_reduce(partial).to(torch.bfloat16)

    def _attention_decode(self, L, x, q, k, v) -> torch.Tensor:
        """Decode rows at positions pos_t..pos_t+T-1, all on the device (graph-capturable)."""
        from .attn_decode import decode_attention
        cfg, lw = self.cfg, self.L[L]
        T, D = x.shape[0], cfg.head_dim
        positions = self.pos_t + torch.arange(T, device=x.device)
        if cfg.rope[L]:
            cos = self.cos.index_select(0, positions)[:, None]
            sin = self.sin.index_select(0, positions)[:, None]
            q = q * cos + torch.cat((-q[..., D // 2:], q[..., :D // 2]), -1) * sin
            k = k * cos + torch.cat((-k[..., D // 2:], k[..., :D // 2]), -1) * sin
        idx = positions % self.ring if cfg.sliding[L] else positions
        self.kc[L].index_copy_(0, idx, k)
        self.vc[L].index_copy_(0, idx, v)
        out = decode_attention(q.contiguous(), self.kc[L], self.vc[L], self.pos_t,
                               sliding=cfg.sliding[L], ring=self.ring, window=cfg.sliding_window,
                               scale=D ** -0.5)
        out = norm_plain(out, cfg.rms_eps).reshape(T, self.hl * D)
        out = out * torch.sigmoid(lw["g"](x))
        partial = lw["o"](out, out_dtype=torch.float32)
        return self.comm.all_reduce(partial).to(torch.bfloat16)

    def _stream_misses(self, L: int, ids: torch.Tensor) -> list[int]:
        """Load the non-resident experts of this step into transient slots (both ranks pick the
        same slots from the same ids). Returns the experts placed, to unmap afterwards."""
        lut = self.arena.lut[L]
        need = ids.unique()
        missing = need[lut[need] < 0].tolist()
        if len(missing) > len(self.transient):
            raise RuntimeError(f"{len(missing)} misses exceed {len(self.transient)} transient slots")
        t0 = time.perf_counter()
        for e, slot in zip(missing, self.transient):
            self.arena.load(self.ck, L, e, slot)
        self.stats["stream_loads"] += len(missing)
        self.stats["stream_s"] += time.perf_counter() - t0
        return missing

    def moe_native(self, L: int, z: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Per-pair expert outputs [k, T, M/W] (BF16-rounded, fp32) from the decode kernels."""
        import fp4_moe_cuda as CUDA
        cfg, A = self.cfg, self.arena
        T, k = slots.shape
        P = T * k
        flat = slots.reshape(-1).contiguous()
        bs, bp, br, nb = CUDA.build_routing_small(flat, 16, k)
        h = torch.empty((P, A.Il), dtype=torch.bfloat16, device=z.device)
        CUDA.up(z, A.w1, A.s1, A.w3, A.s3, h, self.ones[:P], bs, br, float("inf"), k, A.Il,
                cfg.moe_hidden, nb, -1)
        full = self.comm.cat_last(h)
        parts = torch.empty((P, A.Ml), dtype=torch.float32, device=z.device)
        CUDA.down(full, A.w2, A.s2, parts, bs, bp, k, A.Ml, cfg.moe_inter, T, nb, -1, False, 1)
        return parts.view(k, T, A.Ml)

    def moe_triton(self, L: int, z: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Same contract as moe_native for prefill sizes: tools/fp4_moe.py's grouped kernels
        (their N/K are compile-time arguments; only V4.1's moe_forward wrapper fixes them), with
        the call's slots relabelled compactly first, as V4.1's prefill does."""
        import fp4_moe as K4
        cfg, A = self.cfg, self.arena
        T, k = slots.shape
        P = T * k
        BM = K4._pick_bm(P)
        uniq, inv = torch.unique(slots, return_inverse=True)
        compact = inv.to(torch.int32).view_as(slots)
        bs, bp, nb = K4.build_routing(compact, int(uniq.numel()), BM)
        bs = torch.where(bs >= 0, uniq.to(torch.int32)[bs.clamp_min(0)], bs)
        bn1, nw1, ns1 = K4._UP_CFG[BM]
        bn2, nw2, ns2 = K4._DOWN_CFG[BM]
        h = torch.empty((P, A.Il), dtype=torch.bfloat16, device=z.device)
        ones = torch.ones(P, dtype=torch.float32, device=z.device)
        K4._moe_up_kernel[(nb, A.Il // bn1)](
            z, A.w1, A.s1, A.w3, A.s3, h, ones, bs, bp, z.stride(0), h.stride(0), float("inf"),
            TOPK=k, N=A.Il, K=cfg.moe_hidden, BM=BM, BN=bn1, SCALED=False, NULL_SLOT=-1,
            num_warps=nw1, num_stages=ns1)
        full = self.comm.cat_last(h)
        parts = torch.empty((P, A.Ml), dtype=torch.float32, device=z.device)
        K4._moe_down_kernel[(nb, A.Ml // bn2)](
            full, A.w2, A.s2, parts, bs, bp, full.stride(0), parts.stride(0),
            TOPK=k, N=A.Ml, K=cfg.moe_inter, BM=BM, BN=bn2, NTOK=T, SCALED=False, NULL_SLOT=-1,
            PARTIAL=False, PARTS_WORLD=1, num_warps=nw2, num_stages=ns2)
        return parts.view(k, T, A.Ml)

    def moe_dequant(self, L: int, z: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Same contract as moe_native, one BF16 matmul per used expert (prefill sizes)."""
        A = self.arena
        T, k = slots.shape
        flat = slots.reshape(-1)
        tok = torch.arange(T * k, device=z.device) // k
        h = torch.empty((T * k, A.Il), dtype=torch.bfloat16, device=z.device)
        used = flat.unique().tolist()
        groups = {s: (flat == s).nonzero(as_tuple=True)[0] for s in used}
        weights = {}
        for s in used:
            g, u, dn = A.dequant(s)
            weights[s] = dn
            xs = z[tok[groups[s]]]
            h[groups[s]] = F.silu(F.linear(xs, g)) * F.linear(xs, u)
        full = self.comm.cat_last(h)
        parts = torch.empty((T * k, A.Ml), dtype=torch.float32, device=z.device)
        for s in used:
            parts[groups[s]] = F.linear(full[groups[s]], weights[s]).float()
        return parts.view(T, k, A.Ml).transpose(0, 1)

    def moe(self, L: int, x: torch.Tensor) -> torch.Tensor:
        cfg, lw = self.cfg, self.L[L]
        allowed = self.allowed[L] if self.allowed is not None else None
        w, ids = route(cfg, x, lw["router"], lw["bias"], allowed)
        if self.record is not None:
            self.record.append((L, ids.cpu(), w.cpu()))
        if self.miss_stats is not None and allowed is not None:
            # What the unmasked router would have picked, and how much of it was substituted.
            w0, ids0 = route(cfg, x, lw["router"], lw["bias"], None)
            missed = ~allowed[ids0]
            self.miss_stats["picks"] += ids0.numel()
            self.miss_stats["missed"] += int(missed.sum())
            self.miss_stats["mass_missed"] += float((w0 * missed).sum())
            self.miss_stats["mass"] += float(w0.sum())
        streamed = self._stream_misses(L, ids) if self.miss == "stream" else []
        slots = self.arena.lut[L][ids]
        z = self.comm.cat_last(lw["pre"](x))
        z = norm_centered(z, lw["pre_norm"], cfg.rms_eps).contiguous()
        T = x.shape[0]
        if self.native and T * cfg.topk <= 512:
            parts = self.moe_native(L, z, slots)
        elif self.prefill_moe == "triton":
            parts = self.moe_triton(L, z, slots)
        else:
            parts = self.moe_dequant(L, z, slots)
        for e in streamed:
            self.arena.lut[L, e] = -1
        d = self.comm.cat_last(parts).to(torch.bfloat16)                 # [k, T, M]
        d = norm_input_scaled(d, lw["post_norm"], cfg.post_norm_eps)
        s = (d.float() * w.t()[:, :, None]).sum(0).to(torch.bfloat16)
        return self.comm.cat_last(lw["post"](s))

    def layer(self, L: int, h: torch.Tensor) -> torch.Tensor:
        cfg, lw = self.cfg, self.L[L]
        x = norm_centered(h, lw["ln_in"], cfg.rms_eps)
        a = norm_plain(self.attention(L, x), cfg.post_norm_eps)
        h = residual_gate(lw["gate_attn"], cfg.residual_gate_temperature, h, a)
        x = norm_centered(h, lw["ln_ff"], cfg.rms_eps)
        f = norm_input_scaled(self.moe(L, x), lw["ln_post_ff"], cfg.post_norm_eps)
        return residual_gate(lw["gate_ff"], cfg.residual_gate_temperature, h, f)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        local = ids - self.rank * self.vl
        inside = (local >= 0) & (local < self.vl)
        e = self.embed[local.clamp(0, self.vl - 1)] * inside[:, None].to(torch.bfloat16)
        e = self.comm.all_reduce(e.float()).to(torch.bfloat16)
        return norm_plain(e, self.cfg.rms_eps)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = norm_plain(h, cfg.rms_eps, self.final_norm)
        z = self.head(h, out_dtype=torch.float32)
        z = torch.tanh(z * (cfg.output_multiplier / cfg.softcap)) * cfg.softcap
        return self.comm.cat_last(z)

    # ------------------------------------------------------------------ driver
    @torch.no_grad()
    def forward(self, ids, *, all_logits: bool = True, hidden_out: list | None = None) -> torch.Tensor:
        """Append ``ids`` to the sequence in chunks; logits [T, V] (or of the last token only)."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev).reshape(-1)
        if self.pos + ids.numel() > self.max_seq:
            raise ValueError(f"context {self.pos + ids.numel()} exceeds max_seq {self.max_seq}")
        T = ids.numel()
        self.tokens.extend(ids.tolist())
        if T <= self.decode_rows:
            self.pos_t.fill_(self.pos)
            # Graphs only for the step sizes decoding repeats (one row; verify blocks later).
            # A short prompt suffix after prefix reuse runs eagerly instead of capturing a
            # graph for its one-off row count.
            if (self.use_graphs and T in self.graph_rows and self.miss == "substitute"
                    and hidden_out is None and self.record is None):
                out = self._replay(ids)
            else:
                out = self._step(ids, hidden_out)
            self.pos += T
            return out if all_logits else out[-1:]
        outs = []
        for c0 in range(0, ids.numel(), self.chunk):
            part = ids[c0:c0 + self.chunk]
            self.pos_t.fill_(self.pos)       # a chunk of <= decode_rows tokens takes the decode path
            h = self.embed_ids(part)
            for L in range(self.cfg.layers):
                h = self.layer(L, h)
                if hidden_out is not None:
                    hidden_out.append(h)
            self.pos += part.numel()
            if all_logits:
                outs.append(self.logits(h))
            elif c0 + self.chunk >= ids.numel():
                outs.append(self.logits(h[-1:]))
        return torch.cat(outs)

    def _step(self, ids: torch.Tensor, hidden_out: list | None = None) -> torch.Tensor:
        """One decode-size step at pos_t (no host position use inside)."""
        h = self.embed_ids(ids)
        for L in range(self.cfg.layers):
            h = self.layer(L, h)
            if hidden_out is not None:
                hidden_out.append(h)
        return self.logits(h)

    def _replay(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.numel()
        entry = self.graphs.get(T)
        if entry is None:
            static_ids = ids.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):          # warm-up: Triton/JIT compiles, lazy NCCL setup
                    self._step(static_ids)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._step(static_ids)
            entry = self.graphs[T] = (graph, static_ids, static_out)
            self.log(f"[muse] captured decode graph for {T} rows")
        graph, static_ids, static_out = entry
        static_ids.copy_(ids)
        graph.replay()
        return static_out.clone()

    def sample(self, z: torch.Tensor, temperature: float, top_p: float, top_k: int,
               gen: torch.Generator) -> int:
        z = z.float()
        if temperature <= 0:
            return int(z.argmax())
        z = z / temperature
        if 0 < top_k < z.numel():
            vals, idx = z.topk(top_k)
        else:
            vals, idx = z.sort(descending=True)
        pr = torch.softmax(vals, -1)
        if top_p < 1.0:
            keep = pr.cumsum(0) - pr < top_p
            pr = pr * keep
        return int(idx[torch.multinomial(pr, 1, generator=gen)])

    @torch.no_grad()
    def stream(self, prompt: list[int], max_new: int, *, temperature: float = 0.0,
               top_p: float = 1.0, top_k: int = 0, seed: int = 0,
               stop: tuple[int, ...] | None = None, should_stop=None):
        """Yield generated tokens. Both ranks iterate it to the end: rank 0 samples and decides
        whether to stop (``should_stop()``, e.g. a closed client), and broadcasts both."""
        stop = self.cfg.eos if stop is None else stop
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        logits = self.prefill(list(prompt))
        for _ in range(max_new):
            msg = None
            if self.rank == 0:
                msg = (self.sample(logits[-1], temperature, top_p, top_k, gen),
                       bool(should_stop and should_stop()))
            tok, halt = self.comm.broadcast_obj(msg)
            if halt:
                return
            yield tok
            if tok in stop:
                return
            logits = self.forward([tok], all_logits=False)

    def generate(self, prompt, max_new: int, **kw) -> list[int]:
        return list(self.stream(list(prompt), max_new, **kw))
