"""Pure-torch Muse Spark text model: the numerical reference for the TP2 engine.

Follows transformers' modeling_muse_spark.py (5.20.0) operation for operation, in a chosen compute
dtype. engine/muse/test_ref.py checks it against transformers itself (fp32, random tiny config);
the engine is then checked against this, on the same weights. It is a single-process model with no
kernels of its own: slow, but every rounding point is visible here.

Expert weights are fetched per (layer, expert) through ``expert_fn`` and cached, so a checkpoint
whose experts do not fit can still be run (slowly) by a caller that evicts the cache.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .config import MuseConfig


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """MuseSparkRMSNorm._norm: x * (mean(x^2) + eps)^-0.5, in fp32."""
    x = x.float()
    return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)


def norm_plain(x, eps, weight=None):
    """MuseSparkRMSNorm (with_scale = weight is not None)."""
    out = rms(x, eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def norm_centered(x, weight, eps):
    """MuseSparkTextCenteredRMSNorm: normalize, then scale by (1 + w)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def norm_input_scaled(x, weight, eps):
    """MuseSparkTextInputScaledRMSNorm: scale by (1 + w), round to x's dtype, then normalize."""
    scaled = (x.float() * (1.0 + weight.float())).to(x.dtype).float()
    out = scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)
    return out.to(x.dtype)


def residual_gate(gate, temperature, residual, h):
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return (alpha * residual + beta * h).to(residual.dtype)


def rope_tables(cfg: MuseConfig, positions: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float, device=positions.device) / d))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def apply_rope(x, cos, sin):
    """x [T, heads, D]; cos/sin [T, D]."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def route(cfg: MuseConfig, x, weight, bias, allowed=None):
    """MuseSparkTopkRouter. Returns (weights fp32 [T, k], ids int64 [T, k]).

    ``allowed`` (bool [E]) masks experts that are not resident: their choice score becomes -inf,
    so a missing pick becomes the next-ranked allowed expert and the k weights renormalize over
    what was picked (the engine's substitution policy)."""
    logits = F.linear(x.float(), weight.float()) * cfg.output_multiplier
    scores = torch.sigmoid(logits)
    choice = scores + bias.float()
    if allowed is not None:
        choice = choice.masked_fill(~allowed.to(choice.device), float("-inf"))
    ids = torch.topk(choice, k=cfg.topk, dim=-1, sorted=False)[1]
    w = scores.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-15) * cfg.routed_scaling
    return w, ids


class MuseRef:
    def __init__(self, cfg: MuseConfig, get: Callable[[str], torch.Tensor],
                 expert_fn: Callable[[int, int], tuple], *, device="cpu", dtype=torch.float32):
        """``get(name)`` returns a dense tensor by its checkpoint name (``layers.{L}.self_attn...``,
        ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``); ``expert_fn(L, e)`` returns
        (gate [I,H], up [I,H], down [H,I]) as float tensors."""
        self.cfg, self.device, self.dtype = cfg, torch.device(device), dtype
        self._get, self._expert_fn = get, expert_fn
        self._dense: dict[str, torch.Tensor] = {}
        self._experts: dict[tuple[int, int], tuple] = {}
        self.allowed: dict[int, torch.Tensor] = {}        # layer -> bool [E]; absent = all
        self.record_routes = False
        self.routes: list[tuple[int, torch.Tensor]] = []
        self.reset()

    def reset(self):
        self.k: list[torch.Tensor | None] = [None] * self.cfg.layers
        self.v: list[torch.Tensor | None] = [None] * self.cfg.layers
        self.pos = 0

    def w(self, name: str, fp32=False) -> torch.Tensor:
        t = self._dense.get(name)
        if t is None:
            src = self._get(name)
            keep32 = fp32 or src.dtype == torch.float32
            t = self._dense[name] = src.to(self.device, torch.float32 if keep32 else self.dtype)
        return t

    def expert(self, L: int, e: int):
        t = self._experts.get((L, e))
        if t is None:
            t = self._experts[(L, e)] = tuple(x.to(self.device, self.dtype) for x in self._expert_fn(L, e))
        return t

    def drop_experts(self):
        self._experts.clear()

    def attention(self, L: int, x: torch.Tensor) -> torch.Tensor:
        cfg, T = self.cfg, x.shape[0]
        a = f"layers.{L}.self_attn."
        q = F.linear(x, self.w(a + "q_proj.weight")).view(T, cfg.heads, cfg.head_dim)
        k = F.linear(x, self.w(a + "k_proj.weight")).view(T, cfg.kv_heads, cfg.head_dim)
        v = F.linear(x, self.w(a + "v_proj.weight")).view(T, cfg.kv_heads, cfg.head_dim)
        q = norm_plain(q, cfg.rms_eps) * cfg.qk_scale
        k = norm_plain(k, cfg.rms_eps)
        positions = torch.arange(self.pos, self.pos + T, device=x.device)
        if cfg.rope[L]:
            cos, sin = rope_tables(cfg, positions, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        self.k[L] = k if self.k[L] is None else torch.cat([self.k[L], k])
        self.v[L] = v if self.v[L] is None else torch.cat([self.v[L], v])
        K, V = self.k[L], self.v[L]
        S = K.shape[0]
        kpos = torch.arange(S, device=x.device)
        mask = kpos[None, :] <= positions[:, None]
        if cfg.sliding[L]:
            mask &= kpos[None, :] > positions[:, None] - cfg.sliding_window
        Kr = K.repeat_interleave(cfg.kv_groups, 1)
        Vr = V.repeat_interleave(cfg.kv_groups, 1)
        scores = torch.einsum("thd,shd->hts", q, Kr) * (cfg.head_dim ** -0.5)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, -1, dtype=torch.float32).to(q.dtype)
        out = torch.einsum("hts,shd->thd", probs, Vr)
        out = norm_plain(out, cfg.rms_eps).reshape(T, -1)
        out = out * torch.sigmoid(F.linear(x, self.w(a + "gate_proj.weight")))
        return F.linear(out, self.w(a + "o_proj.weight"))

    def moe(self, L: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        m = f"layers.{L}.mlp."
        weights, ids = route(cfg, x, self.w(m + "gate.weight", fp32=True),
                             self.w(m + "gate.e_score_correction_bias", fp32=True),
                             self.allowed.get(L))
        if self.record_routes:
            self.routes.append((L, ids.cpu()))
        z = F.linear(x, self.w(m + "pre_expert_proj.weight"))
        z = norm_centered(z, self.w(m + "pre_expert_norm.weight"), cfg.rms_eps)
        acc = torch.zeros(z.shape, dtype=torch.float32, device=z.device)
        post_w = self.w(m + "experts.post_expert_norm.weight")
        for e in ids.unique().tolist():
            tok, slot = (ids == e).nonzero(as_tuple=True)
            gate, up, down = self.expert(L, e)
            h = z[tok]
            act = F.silu(F.linear(h, gate)) * F.linear(h, up)
            d = norm_input_scaled(F.linear(act, down), post_w, cfg.post_norm_eps)
            acc.index_add_(0, tok, d.float() * weights[tok, slot, None])
        return F.linear(acc.to(z.dtype), self.w(m + "post_expert_proj.weight"))

    def layer(self, L: int, h: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{L}."
        x = norm_centered(h, self.w(p + "input_layernorm.weight"), cfg.rms_eps)
        a = norm_plain(self.attention(L, x), cfg.post_norm_eps)
        h = residual_gate(self.w(p + "post_attention_residual_gate.gate", fp32=True),
                          cfg.residual_gate_temperature, h, a)
        x = norm_centered(h, self.w(p + "pre_feedforward_layernorm.weight"), cfg.rms_eps)
        f = norm_input_scaled(self.moe(L, x), self.w(p + "post_feedforward_layernorm.weight"),
                              cfg.post_norm_eps)
        return residual_gate(self.w(p + "post_feedforward_residual_gate.gate", fp32=True),
                             cfg.residual_gate_temperature, h, f)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.w("embed_tokens.weight")[ids.to(self.device)]
        return norm_plain(e, self.cfg.rms_eps)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = norm_plain(h, cfg.rms_eps, self.w("norm.weight"))
        z = F.linear(h, self.w("lm_head.weight"))
        z = z * cfg.output_multiplier / cfg.softcap
        return torch.tanh(z) * cfg.softcap

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, hidden_out: list | None = None) -> torch.Tensor:
        """Append ``ids`` (1D) to the sequence; returns logits [T, vocab] (fp32 math, input dtype)."""
        h = self.embed(ids)
        for L in range(self.cfg.layers):
            h = self.layer(L, h)
            if hidden_out is not None:
                hidden_out.append(h)
        self.pos += ids.numel()
        return self.logits(h)


def from_checkpoint(model_dir, *, device="cpu", dtype=torch.float32) -> MuseRef:
    from .config import MuseConfig
    from .weights import Checkpoint, PREFIX
    cfg = MuseConfig.load(model_dir)
    ck = Checkpoint(model_dir)

    def get(name):
        return ck.get(name if name == "lm_head.weight" else PREFIX + name)

    return MuseRef(cfg, get, ck.expert_float, device=device, dtype=dtype)


def nll(model: MuseRef, ids: torch.Tensor) -> float:
    """Mean next-token NLL (nats) of ``ids`` under ``model`` (resets its cache)."""
    model.reset()
    logits = model.forward(ids[:-1]).float()
    return float(F.cross_entropy(logits, ids[1:].to(logits.device)))


__all__ = ["MuseRef", "from_checkpoint", "route", "nll"]
