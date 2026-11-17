"""inkling_ref.py: Inkling 2's text layers in plain PyTorch against the Rust engine's dump, layer by layer.

The engine (glm_run with TIDELM_INK_DUMP=PREFIX, GLM_INK_DENSE=bf16) writes the prompt's tokens
and the residual stream [T][4608] after every layer. This runs the same tokens through a
transcription of transformers' modeling_inkling2.py (Inkling2SlidingDecoderLayer,
Inkling2FullDecoderLayer and what they call: the rounding points are the reference's, the
attention is its eager one), with the checkpoint's BF16 weights under their native names and the
routed experts dequantized from the EXL3 checkpoint (tools/exl3_ref.py, the expert's
W = diag(suh) H W_q H diag(svh)), and prints each layer's relative error and the rows' cosine. With
--residents it routes over the engine's residents only (PREFIX.residents.json), as the engine does,
and prints the last rows' top tokens. A check of the engine's arithmetic, not a quality measure.

    TIDELM_INK_DUMP=/tmp/ink GLM_INK_DENSE=bf16 glm_run ... --max-tokens 1
    python -I tools/inkling_ref.py --dense DIR --experts CHECKPOINT --dump /tmp/ink --layers 6

It is a transcription and not the transformers classes because Inkling 2 is in no transformers
release yet (2026-11-16). Needs torch (CUDA when there is one) and safetensors.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402

P = "model.llm."
BF = torch.bfloat16


class Ckpt:
    """Tensors by name out of a checkpoint directory's shards."""

    def __init__(self, d: str, dev: str):
        from safetensors import safe_open
        self.dir, self.dev, self._open, self._files = d, dev, safe_open, {}
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def __call__(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        if shard not in self._files:
            self._files[shard] = self._open(os.path.join(self.dir, shard), framework="pt", device=self.dev)
        return self._files[shard].get_tensor(name)


def rms_norm(x, w, eps):
    """Inkling2RMSNorm: fp32, the weight applied in fp32, back to x's dtype."""
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (w * x).to(dt)


def unit_rms(x, eps=1e-6):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def l2_clip(x, thr, scale=None):
    dt = x.dtype
    x = x.float()
    if scale is not None:
        x = x * scale.float()
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return (x * (thr / (n + 1e-6)).clamp(max=1.0)).to(dt)


def sconv(x, w):
    """Inkling2ShortConvolution over x [T][C], w [C][1][4]: (conv + x) / sqrt(taps + 1)."""
    dt, t = x.dtype, x.shape[0]
    x = x.float()
    k = w.shape[-1]
    y = F.conv1d(x.T[None], w.float(), None, padding=k - 1, groups=x.shape[1])[0, :, :t].T
    return ((y + x) / math.sqrt(k + 1)).to(dt)


def dyn_sconv(x, dyn_in, static_w, weight_proj, heads):
    """Inkling2DynamicShortConvolution: taps = static [lag][C] + a per-token, per-head projection."""
    dt, (t, c) = x.dtype, x.shape
    k = static_w.shape[0]
    dyn = F.linear(dyn_in, weight_proj).float().view(t, heads, 1, k)
    w = static_w.float().T.view(heads, -1, k) + dyn                     # [T][heads][C / heads][k]
    x = x.float()
    ctx = F.pad(x.T, (k - 1, 0))                                         # [C][T + k - 1]
    win = ctx.unfold(-1, k, 1).flip(-1).transpose(0, 1)                  # [T][C][k], lag-indexed
    y = (win.reshape(t, heads, -1, k) * w).sum(-1).reshape(t, c)
    return ((y + x) / math.sqrt(2 * k + 1)).to(dt)


def mix(blocks, cur, query):
    """Inkling2BlockAttentionResidual: [T][nq][H] fp32."""
    s = unit_rms(torch.stack([b.float() for b in blocks] + [cur.float()], dim=-2))   # [T][n][H]
    h = s.shape[-1]
    q = query.float().view(-1, h).expand(s.shape[0], -1, -1)
    return F.scaled_dot_product_attention(q, s, s, scale=h ** -0.5)


class Experts:
    """A layer's routed experts, dequantized from the EXL3 checkpoint when first hit."""

    def __init__(self, ck: Ckpt, layer: int, codebook: str):
        self.ck, self.layer, self.codebook = ck, layer, codebook

    def weights(self, e: int):
        w = []
        for part in ("gate_proj", "up_proj", "down_proj"):
            p = f"{P}layers.{self.layer}.mlp.experts.{e}.{part}."
            tr = self.ck(p + "trellis")
            wk = R.dequantize(tr, self.ck(p + "suh"), self.ck(p + "svh"), R.bits_of_shape(tuple(tr.shape)), self.codebook)
            w.append(wk.T.to(BF).contiguous())                           # [N][K] for F.linear
        return w


def moe(x, router_w, bias, experts: Experts, topk, scale, caps, allowed):
    """Inkling2SparseMoeBlock on x [T][H] bf16."""
    logits = F.linear(x.float(), router_w.float())
    scores = torch.sigmoid(logits)
    sel = scores + bias.float()
    if allowed is not None:
        # the engine's router: a non-resident expert is never chosen; the chosen ones' weights are
        # normalised among themselves
        sel = torch.where(allowed, sel, torch.full_like(sel, -1e30))
    idx = sel.topk(topk, dim=-1).indices
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * scale
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in torch.unique(idx).tolist():
        tok, pos = torch.where(idx == e)
        g_w, u_w, d_w = experts.weights(e)
        g, u = F.linear(x[tok], g_w).float(), F.linear(x[tok], u_w).float()
        act = (caps[0] * torch.tanh(g / caps[0]) * torch.sigmoid(g) * caps[1] * torch.tanh(u / caps[1])).to(BF)
        out.index_add_(0, tok, (F.linear(act, d_w) * wts[tok, pos, None]).float())
    return out.to(x.dtype)


def attention(ck, cfg, l, full, xq, xk, xv, pos):
    """Inkling2SlidingAttention / Inkling2FullAttention, eager, no cache: x* [T][H] bf16 -> [T][H]."""
    a = f"{P}layers.{l}.attn."
    t = xq.shape[0]
    hq, hk, d, dr = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["d_rel"]
    q = F.linear(xq, ck(a + "wq_du.weight"))
    k = F.linear(xk, ck(a + "wk_dv.weight"))
    v = F.linear(xv, ck(a + "wv_dv.weight"))
    if full:
        k = dyn_sconv(k, xk, ck(a + "k_sconv.static_w"), ck(a + "k_sconv.weight_proj"), hk)
        v = dyn_sconv(v, xv, ck(a + "v_sconv.static_w"), ck(a + "v_sconv.weight_proj"), hk)
    else:
        k, v = sconv(k, ck(a + "k_sconv.weight")), sconv(v, ck(a + "v_sconv.weight"))
    r = F.linear(xq, ck(a + "wr_du.weight"))
    gate = F.linear(xq, ck(a + "gate_attn.weight"))
    sinks = F.linear(xq, ck(a + "attn_sink_proj.weight"))                # [T][hq]
    q = l2_clip(q.view(t, hq, d), cfg["qk_l2_clip_threshold"], ck(a + "q_l2_clip.scale"))
    k = l2_clip(k.view(t, hk, d), cfg["qk_l2_clip_threshold"])
    v = v.view(t, hk, d)
    v = unit_rms(v.float(), cfg["rms_norm_eps"]).to(v.dtype)
    r = l2_clip(r.view(t, hq, dr), cfg["rel_l2_clip_threshold"], ck(a + "rel_l2_clip.scale"))
    if full:
        tau = 1.0 + cfg["log_scaling_alpha"] * torch.log(((pos + 1).float() / cfg["log_scaling_n_floor"]).clamp(min=1.0))
        tau = tau[:, None, None]
        q = (q.float() * tau).to(q.dtype)
        r = (r.float() * tau).to(r.dtype)
    proj = ck(a + "rel_logits_proj.proj")                                # [d_rel][extent]
    extent = proj.shape[1]
    rel = (r @ proj).transpose(0, 1)                                     # [hq][T][extent]
    dist = pos[:, None] - pos[None, :]                                   # [Tq][Tk]
    bias = rel.gather(-1, dist.clamp(0, extent - 1)[None].expand(hq, -1, -1))
    bias = bias.masked_fill(((dist < 0) | (dist >= extent))[None], 0.0)
    rep = hq // hk
    ke = k.transpose(0, 1).repeat_interleave(rep, dim=0)                 # [hq][T][d]
    ve = v.transpose(0, 1).repeat_interleave(rep, dim=0)
    w = torch.matmul(q.transpose(0, 1), ke.transpose(1, 2)) * (d ** -0.5) + bias
    mask = dist < 0
    if not full:
        mask = mask | (dist >= cfg["sliding_window_size"])
    w = w.masked_fill(mask[None], float("-inf"))
    w = F.softmax(torch.cat([w, sinks.T[:, :, None]], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
    o = torch.matmul(w, ve).transpose(0, 1).reshape(t, -1)
    o = (o.float() * torch.sigmoid(gate.float())).to(o.dtype)
    return F.linear(o, ck(a + "wo_ud.weight"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dense", required=True, help="the non-expert checkpoint directory (the engine's GLM_MODEL_DIR)")
    ap.add_argument("--experts", required=True, help="the EXL3 checkpoint directory (its expert shards and index)")
    ap.add_argument("--dump", required=True, help="the engine's TIDELM_INK_DUMP prefix")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--rows", action="store_true", help="each row's relative error, layer by layer")
    ap.add_argument("--residents", action="store_true",
                    help="route over the engine's residents only (PREFIX.residents.json), as the engine does")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    top = json.load(open(os.path.join(a.dense, "config.json")))
    cfg = top["text_config"]
    codebook = top["quantization_config"]["codebook"]
    ck, cx = Ckpt(a.dense, dev), Ckpt(a.experts, dev)
    h, n_layers, block = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["n_layers_per_attn_res_block"]
    R.set_dims(h, cfg["intermediate_size"])
    K = min(a.layers, n_layers)
    local = set(cfg["local_layer_ids"])
    eps = cfg["rms_norm_eps"]

    passes = sorted(int(re.search(r"\.p(\d+)\.tok\.bin$", f).group(1)) for f in glob.glob(a.dump + ".p*.tok.bin"))
    toks = [np.fromfile(f"{a.dump}.p{p}.tok.bin", dtype=np.int32) for p in passes]
    if passes != list(np.cumsum([0] + [len(t) for t in toks])[:-1]):
        raise SystemExit(f"passes at {passes} do not tile the sequence")
    tok = torch.from_numpy(np.concatenate(toks).astype(np.int64)).to(dev)
    T = tok.shape[0]
    pos = torch.arange(T, device=dev)
    allowed = None
    if a.residents:
        allowed = torch.zeros(n_layers, cfg["n_routed_experts"], dtype=torch.bool, device=dev)
        for l, x in json.load(open(a.dump + ".residents.json")):
            allowed[l, x] = True

    outs = {}
    with torch.no_grad():
        x = rms_norm(ck(P + "embed.weight")[tok], ck(P + "embed_norm.weight").float(), eps)
        blocks: list[torch.Tensor] = []
        for l in range(K):
            p = f"{P}layers.{l}."
            full = l not in local
            x_in = x
            if l == 0:
                qi = ki = vi = x_in
            else:
                m = mix(blocks, x_in, ck(f"{P}attn_res.attn_queries.{l - 1}"))
                qi, ki, vi = (m[:, 0], m[:, 1], m[:, 2]) if full else (m[:, 0], m[:, 0], m[:, 0])
            if full:
                xq = rms_norm(qi, ck(p + "q_input_norm.weight").float(), eps).to(BF)
                xk = rms_norm(ki, ck(p + "k_input_norm.weight").float(), eps).to(BF)
                xv = rms_norm(vi, ck(p + "v_input_norm.weight").float(), eps).to(BF)
            else:
                xq = xk = xv = rms_norm(qi, ck(p + "attn_norm.weight").float(), eps).to(BF)
            att = attention(ck, cfg, l, full, xq, xk, xv, pos)
            att = sconv(att, ck(p + "attn_sconv.weight"))
            res = att.float()
            if l % block == 0:
                blocks = blocks + [x_in]
            else:
                res = x_in.float() + res
            m = mix(blocks, res, ck(f"{P}attn_res.mlp_queries.{l}"))[:, 0]
            xm = rms_norm(m, ck(p + "mlp_norm.weight").float(), eps).to(BF)
            y = moe(xm, ck(p + "mlp.gate.weight"), ck(p + "mlp.gate.bias"), Experts(cx, l, codebook), cfg["num_experts_per_tok"],
                    cfg["route_scale"], (cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"]), None if allowed is None else allowed[l])
            y = sconv(y, ck(p + "mlp_sconv.weight"))
            x = (res + y.float()).to(BF)
            outs[l] = x
            print(f"layer {l} ({'full' if full else 'window'}) done", flush=True)
        fin = None
        if K == n_layers:
            m = mix(blocks, x, ck(P + "attn_res.output_query"))[:, 0]
            fin = rms_norm(m, ck(P + "norm.weight").float(), eps).to(BF)

    print(f"{T} tokens in {len(passes)} passes, {K} layers")
    for p0, tk in zip(passes, toks):
        n, line = len(tk), []
        for l in range(K):
            path = f"{a.dump}.p{p0}.x{l}.bin"
            if not os.path.exists(path):
                continue
            eng = torch.from_numpy(np.fromfile(path, dtype=np.uint16).view(np.int16).astype(np.int32) << 16).view(torch.float32)
            eng = eng.reshape(n, -1).to(dev)
            ref = outs[l][p0:p0 + n].float()
            rel = ((eng - ref).norm() / ref.norm()).item()
            cos = F.cosine_similarity(eng, ref, dim=-1).min().item()
            line.append(f"L{l} {rel:.1e}/{cos:.4f}")
            if a.rows:
                rr = (eng - ref).norm(dim=-1) / ref.norm(dim=-1)
                print(f"  L{l} rows' rel err: " + " ".join(f"{v:.3f}" for v in rr.tolist()))
        print(f"pass at {p0:5d} ({n:4d} rows): " + "  ".join(line))
    if fin is not None:
        logits = F.linear(fin, ck(P + "unembed.weight")).float()
        logits = (cfg["final_logit_softcapping"] * torch.tanh(logits / cfg["final_logit_softcapping"]))[:, : cfg["unpadded_vocab_size"]]
        top5 = logits.topk(5, dim=-1).indices
        for r in range(max(0, T - 3), T):
            print(f"row {r}: the reference's top-5 ids {top5[r].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
