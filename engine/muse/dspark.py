# Portions of the drafting algorithm (context-KV precompute, anchored block with mask tokens,
# sequential Markov bias) follow vLLM's qwen3_dflash.py / qwen3_dspark.py and the
# speculators DFlash/DSpark training code:
#   SPDX-License-Identifier: Apache-2.0
#   SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#   Copyright (c) the speculators contributors (vllm-project/speculators), Apache-2.0
# Re-implemented here for this engine's TP2 layout; no code is imported from either project.
"""DSpark drafter for Muse Spark (Inferact/Muse-Spark-Open-DSpark), split over two ranks.

What the checkpoint and the reference code fix (docs/muse-spark-plan.md has the evidence):

* Context: the target's hidden states at 5 tapped layers, concatenated [T, 5*8192] -> ``fc`` ->
  ``hidden_norm``; each draft layer projects that to K/V (qk-normed, RoPE at the token's
  position) once per target position and keeps the last ``window`` of them.
* Block: slot 0 is the anchor (the last token the target chose, not yet run through it), slots
  1..6 the mask token; target embedding rows *without* Muse's embedding norm; positions p..p+6.
  Slot k predicts the token at p+k+1 ("sample from anchor"): 7 drafts per pass.
* A draft query sees context positions [p - window, p) and the block causally.
* Layer: the Muse decoder layer with a dense SwiGLU FFN (tensor names map one to one: the
  pre_qkv/pre_ffn/post_ffn norms, o_gate_proj, add_attn/add_ffn residual gates). Every stored
  norm gain sits near 0 (hidden_norm, pre_*, post_ffn, and norm at 0.18): zero-centred, (1 + w).
* Markov head: bias_k = W2 @ W1[token at p+k] added to slot k's logits, sampled left to right.
* Confidence head: sigmoid(proj([normed hidden, W1[prev]])), the predicted acceptance.

Not fixed by anything published, so they are settings (DraftOptions), chosen by measured
acceptance: the attention scale, the tap convention (a layer's input or output), whether the
block embedding is normed, and whether the Markov bias joins the raw logits or the processed
ones. Verification makes the output the target's own whatever these are; they move only
acceptance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .ref import norm_centered, norm_input_scaled, norm_plain, residual_gate


@dataclass
class DraftOptions:
    qk_scale: float = 5.4730064863875      # multiplier on normed q (the target's value)
    attn_scale: str = "sqrt"               # "sqrt": D^-0.5, "linear": 1/D
    tap: str = "input"                     # tapped hidden = layer L's "input" or "output"
    embed_norm: bool = False               # Muse's weightless embedding norm on block tokens
    bias_on: str = "raw"                   # Markov bias on "raw" lm_head logits or "processed"
    post_ffn: str = "input_scaled"         # post-FFN norm: "input_scaled" (Muse) or "centered"
    residual_temp: float = 0.3

    @classmethod
    def from_env(cls) -> "DraftOptions":
        o = cls()
        for k in o.__dataclass_fields__:
            v = os.environ.get(f"GLM_DRAFT_{k.upper()}")
            if v is not None:
                t = type(getattr(o, k))
                setattr(o, k, (v == "1") if t is bool else t(v))
        return o


class DSpark:
    def __init__(self, eng, path: str, fmt: str | None = None, opts: DraftOptions | None = None):
        from .tp import Dense
        from .weights import Checkpoint
        self.eng, self.comm = eng, eng.comm
        self.opts = opts or DraftOptions.from_env()
        W, r, d = eng.world, eng.rank, eng.dev
        cfg = json.loads((Path(path) / "config.json").read_text())
        self.cfg = cfg
        self.block = int(cfg["block_size"])
        self.taps = [int(x) for x in cfg["target_layer_ids"]]
        self.window = int(cfg["sliding_window"])
        self.mask_id = int(cfg["mask_token_id"])
        self.H = int(cfg["hidden_size"])
        self.heads, self.kv_heads, self.D = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
        self.hl, self.kvl = self.heads // W, self.kv_heads // W
        self.eps = float(cfg.get("rms_norm_eps", 1e-5))
        self.post_eps = float(cfg.get("rms_post_norm_eps", 1e-8))
        self.softcap = float(cfg.get("output_soft_cap_temp", 20.0))
        metap = cfg.get("metap") or {}
        self.mult = (float(metap.get("base_width", 256)) / self.H) ** 0.5 if metap.get("use_metap") else 1.0
        fmt = fmt or os.environ.get("GLM_DRAFT_DENSE", eng.dense_fmt)
        h = Checkpoint(path, detect=False)        # preadv reads (see Checkpoint._read)

        def rows(name, total):
            n = total // W
            return h.get_rows(name, r * n, (r + 1) * n).to(d)

        def cols(name, total):
            n = total // W
            return h.get(name)[:, r * n:(r + 1) * n].contiguous().to(d)

        def full(name):
            return h.get(name).to(d)

        self.fc = Dense(rows("fc.weight", self.H), fmt)
        self.hidden_norm = full("hidden_norm.weight")
        self.final_norm = full("norm.weight")
        self.layers = []
        n_layers = int(cfg["num_hidden_layers"])
        inter = int(cfg["intermediate_size"])
        for i in range(n_layers):
            p = f"layers.{i}."
            self.layers.append({
                "ln_qkv": full(p + "self_attn.pre_qkv_proj_norm.weight"),
                "q": Dense(rows(p + "self_attn.q_proj.weight", self.heads * self.D), fmt),
                "k": Dense(rows(p + "self_attn.k_proj.weight", self.kv_heads * self.D), fmt),
                "v": Dense(rows(p + "self_attn.v_proj.weight", self.kv_heads * self.D), fmt),
                "g": Dense(rows(p + "self_attn.o_gate_proj.weight", self.heads * self.D), fmt),
                "o": Dense(cols(p + "self_attn.o_proj.weight", self.heads * self.D), fmt),
                "gate_attn": full(p + "add_attn.gate").float(),
                "ln_ff": full(p + "pre_ffn_norm.weight"),
                "w1": Dense(rows(p + "feed_forward.gate_proj.weight", inter), fmt),
                "w3": Dense(rows(p + "feed_forward.up_proj.weight", inter), fmt),
                "w2": Dense(cols(p + "feed_forward.down_proj.weight", inter), fmt),
                "ln_post_ff": full(p + "post_ffn_norm.weight"),
                "gate_ff": full(p + "add_ffn.gate").float(),
            })
        # The README calls the head the frozen target head; share it when it is bit-identical.
        own = rows("lm_head.weight", eng.cfg.vocab)
        tgt = eng.ck.get_rows("lm_head.weight", r * eng.vl, (r + 1) * eng.vl).to(d)
        same = bool(torch.equal(own, tgt))
        same = all(self.comm.gather_objects(same))
        self.head = eng.head if same else Dense(own, fmt)
        self.shared_head = same
        del own, tgt
        self.markov_w1 = full("markov_head.markov_w1.weight")                 # [V, R], replicated
        h.drop_cache()
        self.markov_w2 = Dense(rows("markov_head.markov_w2.weight", eng.cfg.vocab), "bf16")
        self.conf_w = full("confidence_head.proj.weight").float()[0]
        self.conf_b = float(h.get("confidence_head.proj.bias").float()[0])
        # Context K/V ring per draft layer: window plus the largest update (a prefill chunk).
        self.ring = self.window + eng.chunk
        kw = dict(dtype=torch.bfloat16, device=d)
        self.kc = [torch.zeros((self.ring, self.kvl, self.D), **kw) for _ in self.layers]
        self.vc = [torch.zeros((self.ring, self.kvl, self.D), **kw) for _ in self.layers]
        inv = 1.0 / (float(cfg.get("rope_theta", 500000.0))
                     ** (torch.arange(0, self.D, 2, dtype=torch.float, device=d) / self.D))
        self.inv_freq = inv
        self.scale = self.D ** -0.5 if self.opts.attn_scale == "sqrt" else 1.0 / self.D
        self.oracle = None

    # ------------------------------------------------------------------ helpers
    def _rope(self, x, positions):
        f = positions.float()[:, None] * self.inv_freq[None]
        emb = torch.cat((f, f), -1)
        cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
        D = self.D
        return x * cos + torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1) * sin

    def _kv(self, lw, x, positions):
        T = x.shape[0]
        k = norm_plain(lw["k"](x).view(T, self.kvl, self.D), self.eps)
        v = lw["v"](x).view(T, self.kvl, self.D)
        return self._rope(k, positions), v

    # ------------------------------------------------------------------ context
    @torch.no_grad()
    def update_context(self, taps: torch.Tensor, pos0: int) -> None:
        """taps [T, n_taps, H] at positions pos0..pos0+T-1 -> each layer's context K/V."""
        T = taps.shape[0]
        x = self.comm.cat_last(self.fc(taps.reshape(T, -1).to(torch.bfloat16)))
        x = norm_centered(x, self.hidden_norm, self.eps)
        positions = torch.arange(pos0, pos0 + T, device=x.device)
        idx = positions % self.ring
        for lw, kc, vc in zip(self.layers, self.kc, self.vc):
            k, v = self._kv(lw, x, positions)
            kc[idx] = k
            vc[idx] = v

    # ------------------------------------------------------------------ drafting
    @torch.no_grad()
    def _block_hidden(self, anchor: int, p: int) -> torch.Tensor:
        B, D = self.block, self.D
        ids = torch.full((B,), self.mask_id, dtype=torch.long, device=self.eng.dev)
        ids[0] = anchor
        eng = self.eng
        local = ids - eng.rank * eng.vl
        inside = (local >= 0) & (local < eng.vl)
        x = eng.embed[local.clamp(0, eng.vl - 1)] * inside[:, None].to(torch.bfloat16)
        x = self.comm.all_reduce(x.float()).to(torch.bfloat16)
        if self.opts.embed_norm:
            x = norm_plain(x, eng.cfg.rms_eps)
        positions = torch.arange(p, p + B, device=x.device)
        lo = max(0, p - self.window)
        cpos = torch.arange(lo, p, device=x.device)
        causal = torch.ones(B, B, dtype=torch.bool, device=x.device).tril()
        mask = torch.cat([torch.ones(B, cpos.numel(), dtype=torch.bool, device=x.device), causal], 1)
        for lw, kc, vc in zip(self.layers, self.kc, self.vc):
            res = x
            hx = norm_centered(x, lw["ln_qkv"], self.eps)
            q = norm_plain(lw["q"](hx).view(B, self.hl, D), self.eps) * self.opts.qk_scale
            q = self._rope(q, positions)
            k, v = self._kv(lw, hx, positions)
            K = torch.cat([kc[cpos % self.ring], k])
            V = torch.cat([vc[cpos % self.ring], v])
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                attn_mask=mask[None, None], scale=self.scale, enable_gqa=True)[0].transpose(0, 1)
            out = norm_plain(out, self.eps).reshape(B, self.hl * D)
            out = out * torch.sigmoid(lw["g"](hx))
            a = self.comm.all_reduce(lw["o"](out, out_dtype=torch.float32)).to(torch.bfloat16)
            a = norm_plain(a, self.post_eps)
            x = residual_gate(lw["gate_attn"], self.opts.residual_temp, res, a)
            res = x
            hx = norm_centered(x, lw["ln_ff"], self.eps)
            f = lw["w2"](F.silu(lw["w1"](hx)) * lw["w3"](hx), out_dtype=torch.float32)
            f = self.comm.all_reduce(f).to(torch.bfloat16)
            if self.opts.post_ffn == "input_scaled":
                f = norm_input_scaled(f, lw["ln_post_ff"], self.post_eps)
            else:
                f = norm_centered(f, lw["ln_post_ff"], self.post_eps)
            x = residual_gate(lw["gate_ff"], self.opts.residual_temp, res, f)
        return norm_centered(x, self.final_norm, self.eps)

    def _process(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z * (self.mult / self.softcap)) * self.softcap

    @torch.no_grad()
    def propose(self, anchor: int, p: int, *, temperature: float = 0.0, top_k: int = 0,
                top_p: float = 1.0, gen: torch.Generator | None = None):
        """Draft ``block`` tokens after ``anchor`` (at position p). Returns (tokens, q, conf):
        q [B, V] the draft distributions sampled from (None when greedy), conf [B]."""
        if self.oracle is not None:            # tests: drafts from a known continuation
            toks = self.oracle(anchor, p)
            return toks, None, torch.zeros(len(toks))
        hn = self._block_hidden(anchor, p)
        z = self.head(hn, out_dtype=torch.float32)                            # [B, V/W] raw
        if self.opts.bias_on == "processed":
            z = self._process(z)
        eng, B = self.eng, self.block
        toks: list[int] = []
        qs, confs = [], []
        prev = anchor
        for k in range(B):
            emb = self.markov_w1[prev]                                           # [R]
            zk = z[k] + self.markov_w2(emb[None], out_dtype=torch.float32)[0]
            if self.opts.bias_on == "raw":
                zk = self._process(zk)
            feats = torch.cat([hn[k].float(), emb.float()])
            confs.append(torch.sigmoid(feats @ self.conf_w + self.conf_b))
            if temperature <= 0:
                # Global argmax from each rank's half: gather (value, id), lowest id on ties.
                v, i = zk.max(0)
                pair = torch.stack([v, (i + eng.rank * eng.vl).float()])
                g = self.comm.all_gather(pair)
                best = int(torch.argmax(g[:, 0]))
                tok = int(g[best, 1])
                qs.append(None)
            else:
                full = self.comm.cat_last(zk[None])[0]
                q = eng.sample_probs(full, temperature, top_p, top_k)
                tok = None
                if eng.rank == 0:
                    tok = int(torch.multinomial(q, 1, generator=gen))
                tok = self.comm.broadcast_obj(tok)
                qs.append(q)
            toks.append(tok)
            prev = tok
        conf = torch.stack(confs).cpu()
        return toks, (torch.stack(qs) if temperature > 0 else None), conf
