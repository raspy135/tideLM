"""Pure-torch GLM-5.3 (glm_moe_dsa) text model: the numerical reference for the TP2 engine.

Follows transformers 5.20's modeling_glm_moe_dsa.py operation for operation: MLA attention with
interleaved RoPE on the rope slice, the DSA indexer (relu(q.k) weighted per head, causal, top-k)
on "full" layers and its top-k reused on "shared" ones, DeepSeek MoE (sigmoid scores plus the
correction bias for choice, group limit, normalized top-k times the routed scale) plus a shared
expert, dense MLPs in the first layers. engine/glm/test_ref.py checks it against transformers.

Weights come through ``get(name)`` (names without the leading ``model.``) and
``expert_fn(L, e) -> (gate, up, down)`` so any checkpoint layout can feed it.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .config import GlmConfig


LATENT_EPS = 1e-6


def rms_norm(x, weight, eps):
    """GlmMoeDsaRMSNorm: normalize in fp32, cast back, then scale (weight * x in x's dtype)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * out.to(x.dtype)


def rope_tables(cfg: GlmConfig, positions: torch.Tensor, dtype):
    d = cfg.rope_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float, device=positions.device) / d))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rope_interleave(x, cos, sin):
    """apply_rotary_pos_emb_interleave for one tensor; x [T, heads, d], cos/sin [T, d]."""
    c = cos[:, : cos.shape[-1] // 2][:, None]
    s = sin[:, : sin.shape[-1] // 2][:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def route(cfg: GlmConfig, x, weight, bias, allowed=None):
    """GlmMoeDsaTopkRouter -> (weights fp32 [T, k], ids [T, k]). ``allowed`` (bool [E]) masks
    non-resident experts out of the choice (substitution)."""
    return route_pick(cfg, *route_scores(cfg, x, weight, bias), allowed)


def route_logits(x, weight):
    """The router matmul, fp32 logits."""
    if weight.dtype == torch.bfloat16:
        # GLM_ROUTER_BF16: BF16 weights and (BF16) activations, fp32 sums and logits.
        x2 = x.reshape(-1, weight.shape[1]).to(torch.bfloat16)
        return torch.mm(x2, weight.t(), out_dtype=torch.float32).view(*x.shape[:-1], weight.shape[0])
    return F.linear(x.float(), weight.float())


def route_scores(cfg: GlmConfig, x, weight, bias):
    """The router matmul and group limit: (scores, choice) for route_pick. One call serves the
    masked and the unmasked pick of the same rows."""
    logits = route_logits(x, weight)
    scores = logits.sigmoid()
    choice = scores + bias.float()
    if cfg.n_group > 1:
        E = cfg.experts
        g = choice.view(-1, cfg.n_group, E // cfg.n_group).topk(2, -1)[0].sum(-1)
        gi = torch.topk(g, cfg.topk_group, -1, sorted=False)[1]
        gm = torch.zeros_like(g).scatter_(1, gi, 1)
        sm = gm.unsqueeze(-1).expand(-1, cfg.n_group, E // cfg.n_group).reshape(-1, E)
        choice = choice.masked_fill(~sm.bool(), float("-inf"))
    return scores, choice


def route_pick(cfg: GlmConfig, scores, choice, allowed=None):
    if allowed is not None:
        choice = choice.masked_fill(~allowed.to(choice.device), float("-inf"))
    ids = torch.topk(choice, cfg.topk, -1, sorted=False)[1]
    w = scores.gather(1, ids)
    if cfg.norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * cfg.routed_scaling, ids


class GlmRef:
    def __init__(self, cfg: GlmConfig, get: Callable[[str], torch.Tensor],
                 expert_fn: Callable[[int, int], tuple], *, device="cpu", dtype=torch.float32):
        self.cfg, self.device, self.dtype = cfg, torch.device(device), dtype
        self._get, self._expert_fn = get, expert_fn
        self._dense: dict[str, torch.Tensor] = {}
        self._experts: dict[tuple[int, int], tuple] = {}
        self.allowed: dict[int, torch.Tensor] = {}
        self.reset()

    def full(self, L):
        return L >= self.cfg.layers or self.cfg.indexer_full[L]

    def dense(self, L):
        return L < self.cfg.layers and self.cfg.dense_mlp[L]

    def reset(self):
        n = self.cfg.layers + self.cfg.mtp_layers
        self.kv: list[torch.Tensor | None] = [None] * n       # normed latent [S, kv_lora]
        self.kr: list[torch.Tensor | None] = [None] * n       # roped shared key [S, rope_dim]
        self.ik: list[torch.Tensor | None] = [None] * n       # indexer keys [S, index_dim]
        self.pos = 0

    def w(self, name, keep=False):
        t = self._dense.get(name)
        if t is None:
            src = self._get(name)
            # Only tensors marked ``keep`` (router, correction bias, indexer weights) stay in
            # their stored dtype; dequantized FP8/MXFP4 weights arrive fp32 and take the model's.
            t = self._dense[name] = src.to(self.device, src.dtype if keep else self.dtype)
        return t

    def expert(self, L, e):
        t = self._experts.get((L, e))
        if t is None:
            t = self._experts[(L, e)] = tuple(x.to(self.device, self.dtype) for x in self._expert_fn(L, e))
        return t

    def indexer(self, L, x, q_resid, cos, sin, positions):
        cfg = self.cfg
        p = f"layers.{L}.self_attn.indexer."
        T = x.shape[0]
        q = F.linear(q_resid, self.w(p + "wq_b.weight")).view(T, cfg.index_heads, cfg.index_dim)
        k = F.layer_norm(F.linear(x, self.w(p + "wk.weight")), (cfg.index_dim,),
                         self.w(p + "k_norm.weight"), self.w(p + "k_norm.bias"), 1e-6)[:, None]
        r = cfg.rope_dim
        q = torch.cat([rope_interleave(q[..., :r], cos, sin), q[..., r:]], -1)
        k = torch.cat([rope_interleave(k[..., :r], cos, sin), k[..., r:]], -1)[:, 0]
        self.ik[L] = k if self.ik[L] is None else torch.cat([self.ik[L], k])
        K = self.ik[L]
        scores = F.relu(torch.einsum("thd,sd->ths", q.float(), K.float()) * cfg.index_dim ** -0.5)
        wproj = self.w(p + "weights_proj.weight", keep=True).float()   # fp32 module in transformers
        weights = F.linear(x.float(), wproj) * cfg.index_heads ** -0.5
        idx = torch.einsum("th,ths->ts", weights, scores)
        kpos = torch.arange(K.shape[0], device=x.device)
        idx = idx.masked_fill(kpos[None] > positions[:, None], float("-inf"))
        return idx.topk(min(cfg.index_topk, K.shape[0]), -1).indices

    def attention(self, L, x, topk_prev):
        cfg = self.cfg
        a = f"layers.{L}.self_attn."
        T = x.shape[0]
        positions = torch.arange(self.pos, self.pos + T, device=x.device)
        cos, sin = rope_tables(cfg, positions, x.dtype)
        # q_a/kv_a norms are built without an eps in transformers: the class default 1e-6, not
        # the config's rms_norm_eps (1e-5). Caught by a float64 comparison (7.5e-5 in q_resid).
        q_resid = rms_norm(F.linear(x, self.w(a + "q_a_proj.weight")), self.w(a + "q_a_layernorm.weight"), LATENT_EPS)
        q = F.linear(q_resid, self.w(a + "q_b_proj.weight")).view(T, cfg.heads, cfg.qk_dim)
        q_pass, q_rot = q[..., :cfg.nope], q[..., cfg.nope:]
        ckv = F.linear(x, self.w(a + "kv_a_proj_with_mqa.weight"))
        kv_pass, k_rot = ckv[..., :cfg.kv_lora], ckv[..., cfg.kv_lora:]
        kv_pass = rms_norm(kv_pass, self.w(a + "kv_a_layernorm.weight"), LATENT_EPS)
        q_rot = rope_interleave(q_rot, cos, sin)
        k_rot = rope_interleave(k_rot[:, None], cos, sin)[:, 0]
        self.kv[L] = kv_pass if self.kv[L] is None else torch.cat([self.kv[L], kv_pass])
        self.kr[L] = k_rot if self.kr[L] is None else torch.cat([self.kr[L], k_rot])
        if self.full(L):
            topk = self.indexer(L, x, q_resid, cos, sin, positions)
        else:
            topk = topk_prev
        S = self.kv[L].shape[0]
        kvb = F.linear(self.kv[L], self.w(a + "kv_b_proj.weight")).view(S, cfg.heads, cfg.nope + cfg.v_dim)
        k_nope, v = kvb[..., :cfg.nope], kvb[..., cfg.nope:]
        k = torch.cat([k_nope, self.kr[L][:, None].expand(S, cfg.heads, cfg.rope_dim)], -1)
        qs = torch.cat([q_pass, q_rot], -1)
        scores = torch.einsum("thd,shd->hts", qs, k) * cfg.qk_dim ** -0.5
        kpos = torch.arange(S, device=x.device)
        allowed = kpos[None] <= positions[:, None]
        sel = torch.zeros(T, S, dtype=torch.bool, device=x.device).scatter_(1, topk.long(), True)
        allowed = allowed & sel
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        probs = torch.softmax(scores, -1, dtype=torch.float32).to(qs.dtype)
        out = torch.einsum("hts,shd->thd", probs, v).reshape(T, -1)
        return F.linear(out, self.w(a + "o_proj.weight")), topk

    def mlp(self, prefix, x):
        g = F.linear(x, self.w(prefix + "gate_proj.weight"))
        u = F.linear(x, self.w(prefix + "up_proj.weight"))
        return F.linear(F.silu(g) * u, self.w(prefix + "down_proj.weight"))

    def moe(self, L, x):
        cfg = self.cfg
        m = f"layers.{L}.mlp."
        w, ids = route(cfg, x, self.w(m + "gate.weight", keep=True),
                       self.w(m + "gate.e_score_correction_bias", keep=True), self.allowed.get(L))
        acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in ids.unique().tolist():
            tok, slot = (ids == e).nonzero(as_tuple=True)
            gate, up, down = self.expert(L, e)
            h = x[tok]
            y = F.linear(F.silu(F.linear(h, gate)) * F.linear(h, up), down)
            acc.index_add_(0, tok, y.float() * w[tok, slot, None])
        return acc.to(x.dtype) + self.mlp(m + "shared_experts.", x)

    def layer(self, L, h, topk_prev):
        cfg = self.cfg
        p = f"layers.{L}."
        a, topk = self.attention(L, rms_norm(h, self.w(p + "input_layernorm.weight"), cfg.rms_eps), topk_prev)
        h = h + a
        x = rms_norm(h, self.w(p + "post_attention_layernorm.weight"), cfg.rms_eps)
        f = self.mlp(p + "mlp.", x) if self.dense(L) else self.moe(L, x)
        return h + f, topk

    @torch.no_grad()
    def mtp(self, next_ids: torch.Tensor, h_prev: torch.Tensor, pos0: int):
        """The MTP layer (DeepSeek's, as vLLM's deepseek_mtp runs it) at positions pos0..: token
        i+1's embedding (zero at position 0) and the target's post-norm hidden at i ->
        eh_proj(cat(enorm(e), hnorm(h))) -> a decoder layer with its own caches -> shared norm
        -> lm_head. Returns (normed hidden for the next step, logits predicting token i+2)."""
        cfg = self.cfg
        L = cfg.layers
        T = next_ids.numel()
        positions = torch.arange(pos0, pos0 + T, device=self.device)
        e = self.w("embed_tokens.weight")[next_ids.to(self.device)]
        e = torch.where(positions[:, None] == 0, torch.zeros_like(e), e)
        p = f"layers.{L}."
        x = F.linear(torch.cat([rms_norm(e, self.w(p + "enorm.weight"), cfg.rms_eps),
                                rms_norm(h_prev.to(e.dtype), self.w(p + "hnorm.weight"), cfg.rms_eps)], -1),
                     self.w(p + "eh_proj.weight"))
        saved, self.pos = self.pos, pos0
        h, _ = self.layer(L, x, None)
        self.pos = saved
        hn = rms_norm(h, self.w(p + "shared_head.norm.weight"), cfg.rms_eps)
        return hn, F.linear(hn, self.w("lm_head.weight"))

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, hidden_out: list | None = None) -> torch.Tensor:
        h = self.w("embed_tokens.weight")[ids.to(self.device)]
        topk = None
        for L in range(self.cfg.layers):
            h, topk = self.layer(L, h, topk)
            if hidden_out is not None:
                hidden_out.append(h)
        self.pos += ids.numel()
        h = rms_norm(h, self.w("norm.weight"), self.cfg.rms_eps)
        self.last_hidden = h                  # post-norm, what the MTP layer takes
        return F.linear(h, self.w("lm_head.weight"))
