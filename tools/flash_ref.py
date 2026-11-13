"""flash_ref.py: GLM-5.3-Flash's first layers in transformers (modeling_glm5_next) against the Rust
engine's dump, layer by layer.

The engine (glm_run with TIDELM_FLASH_DUMP=PREFIX, GLM_FLASH_DENSE=bf16, the checked MoE layers'
experts all resident) writes the prompt's tokens and the mHC streams [T][4][4096] after every
layer. This runs the same tokens through transformers' Glm5NextTextModel cut to --layers layers,
with the checkpoint's BF16 weights and the routed experts dequantized from the engine's own pack
(tools/exl3_ref.py, the expert's W = diag(suh) H W_q H diag(svh)), and prints each layer's streams'
relative error and the rows' cosine. A check of the engine's arithmetic, not a quality measure.

    TIDELM_FLASH_DUMP=/tmp/fl GLM_FLASH_DENSE=bf16 glm_run ... --max-tokens 1
    python -I tools/flash_ref.py --dense DIR --pack PACK --dump /tmp/fl --layers 6

Needs torch with CUDA and a transformers with glm5_next (tooling, not serving).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402

P = "model.language_model."


class PackExperts(torch.nn.Module):
    """Glm5NextTextExperts' forward on experts dequantized from a --world 1 pack when first hit."""

    def __init__(self, reader, layer, n_experts, limit, dev):
        super().__init__()
        self.reader, self.layer, self.num_experts, self.swiglu_limit, self.dev = reader, layer, n_experts, limit, dev
        self.cache = {}

    def weights(self, e):
        if e not in self.cache:
            rec = self.reader.read_expert(self.layer, e)
            bits = self.reader.records[(self.layer, e)]["bits"]
            bits = bits if isinstance(bits, list) else [bits] * 3
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)
            w = []
            for tag, b in zip(("1", "3", "2"), bits):
                wk = R.dequantize(t(rec["t" + tag]), t(rec["suh" + tag]), t(rec["svh" + tag]), float(b), self.reader.codebook)
                w.append(wk.T.to(torch.bfloat16).contiguous())       # [N][K] for F.linear
            self.cache[e] = (torch.cat([w[0], w[1]], 0), w[2])
        return self.cache[e]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            if e < 0 or e >= self.num_experts:
                continue
            tok, pos = torch.where(top_k_index == e)
            gu, down = self.weights(e)
            g, u = F.linear(hidden_states[tok], gu).chunk(2, dim=-1)
            g = g.clamp(min=None, max=self.swiglu_limit)
            u = u.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = F.linear(F.silu(g) * u, down) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dense", required=True, help="the non-expert checkpoint (glm_fetch_dense.py's directory)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--dump", required=True, help="the engine's TIDELM_FLASH_DUMP prefix")
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args(argv)
    from safetensors import safe_open
    from transformers.models.glm5_next import configuration_glm5_next as C
    from transformers.models.glm5_next import modeling_glm5_next as M

    dev = "cuda"
    top = json.load(open(os.path.join(a.dense, "config.json")))
    tc = dict(top["text_config"])
    K = a.layers
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        tc[k] = tc[k][:K]
    tc["num_hidden_layers"] = K
    tc["num_nextn_predict_layers"] = 0
    cfg = C.Glm5NextTextConfig(**tc)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = M.Glm5NextTextModel(cfg)
    reader = R.PackReader(a.pack)
    for l, layer in enumerate(model.layers):
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts = PackExperts(reader, l, cfg.n_routed_experts, cfg.swiglu_limit, dev)
    model = model.to_empty(device=dev).to(torch.bfloat16)
    # the checkpoint's tensors for the kept layers (names without the model prefix)
    idx = json.load(open(os.path.join(a.dense, "model.safetensors.index.json")))["weight_map"]
    sd, files = {}, {}
    for name, shard in idx.items():
        if not name.startswith(P):
            continue
        k = name[len(P):]
        if k.startswith("layers."):
            if int(k.split(".")[1]) >= K or ".mlp.experts." in k:
                continue
        elif not (k.startswith("embed_tokens") or k.startswith("norm")):
            continue
        if shard not in files:
            files[shard] = safe_open(os.path.join(a.dense, shard), framework="pt", device=dev)
        sd[k] = files[shard].get_tensor(name)
    # the checkpoint's names as transformers' modules hold them (its load-time renaming)
    ren = {}
    convs = {}
    for k, v in sd.items():
        n = k
        for site in ("attn", "ffn"):
            for f in ("fn", "base", "scale"):
                n = n.replace(f".hc_{site}_{f}", f".{site}_hc.{f}")
        for f in ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"):
            n = n.replace(f".self_attn.{f}", f".self_attn.forget_gate.{f}")
        for c in ("q", "k", "v"):
            if n.endswith(f".self_attn.{c}_conv1d.weight"):
                convs.setdefault(n[: -len(f"{c}_conv1d.weight")], {})[c] = v
                n = None
                break
        if n is not None:
            ren[n] = v
    for base, cs in convs.items():
        ren[base + "conv1d.weight"] = torch.cat([cs["q"], cs["k"], cs["v"]], 0)
    sd = ren
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the checkpoint's fp32 tensors (A_log, dt_bias, the router bias, mHC base / scale) stay fp32
    for k, v in sd.items():
        if v.dtype == torch.float32:
            mod, _, attr = k.rpartition(".")
            m = model.get_submodule(mod)
            if attr in m._parameters:
                m._parameters[attr] = torch.nn.Parameter(v.clone(), requires_grad=False)
            else:
                m._buffers[attr] = v.clone()
    missing = [m for m in missing if ".experts." not in m]
    if missing or unexpected:
        print("missing:", missing[:10], "unexpected:", unexpected[:10])
    # buffers HF initialises in post_init that to_empty left uninitialised (none expected beyond weights)
    model.eval()
    tok = np.fromfile(a.dump + ".tok.bin", dtype=np.int32)
    T = len(tok)
    outs = {}
    for l, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, i, o, l=l: outs.__setitem__(l, o[0].detach()))
    with torch.no_grad():
        model(input_ids=torch.from_numpy(tok.astype(np.int64))[None].to(dev), use_cache=False)
    print(f"{T} tokens, {K} layers ({', '.join(t[:3] for t in cfg.layer_types)})")
    for l in range(K):
        path = f"{a.dump}.x{l}.bin"
        if not os.path.exists(path):
            print(f"layer {l}: no engine dump")
            continue
        eng = torch.from_numpy(np.fromfile(path, dtype=np.uint16).view(np.int16).astype(np.int32) << 16).view(torch.float32)
        eng = eng.reshape(T, 4, -1).to(dev)
        ref = outs[l][0].float()                     # [T, 4, H]
        rel = ((eng - ref).norm() / ref.norm()).item()
        cos = F.cosine_similarity(eng.reshape(T, -1), ref.reshape(T, -1), dim=-1)
        print(f"layer {l:2d} ({cfg.layer_types[l][:6]}, {cfg.mlp_layer_types[l]}): rel err {rel:.3e}, "
              f"row cosine min {cos.min().item():.5f} mean {cos.mean().item():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
