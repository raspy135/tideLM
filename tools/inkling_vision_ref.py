"""inkling_vision_ref.py: Inkling 2's vision tower in plain PyTorch against the Rust engine's rows.

`inkling_vision_check` (the Rust tower, rust/crates/tidelm-glm/src/inkling_vision.rs) writes the
patches it encoded and the feature rows it got; this reads the same patches, runs them through a
transcription of transformers' Inkling2VisionModel (modeling_inkling2.py: Inkling2VisionEmbeddings,
Inkling2VisionRotaryEmbedding, Inkling2VisionEncoderLayer, Inkling2VisionMerger, and vision_utils'
position ids and interpolation taps; bf16, the rounding points the reference's) with the
checkpoint's model.visual.* weights under their native names, and prints the relative error and the
rows' cosine. With --dump it compares the sequence after the embeddings, ln_pre, every layer and
the merger's first half too (the engine's TIDELM_VISION_DUMP=PREFIX files).

    inkling_vision_check --model-dir DIR --rows 3 --cols 5 --patches-out P.bin --out O.bin
    python -I tools/inkling_vision_ref.py --dense DIR --patches P.bin --rows 3 --cols 5 --engine O.bin

It is a transcription and not the transformers classes because Inkling 2 is in no transformers
release yet (2026-11-16): what it checks is the engine's arithmetic against this reading of the
reference, so a point both read the same wrong way (the RoPE base, which the checkpoint's config
leaves to transformers' default of 10000; the patch layout the caller is told to send) passes here.
Needs torch (CUDA when there is one) and safetensors.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

P = "model.visual."
BF = torch.bfloat16


class Ckpt:
    """Tensors by name out of a checkpoint directory's shards."""

    def __init__(self, d: str, dev: str):
        from safetensors import safe_open
        self.dir, self.dev, self._open, self._files = d, dev, safe_open, {}
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def __call__(self, name: str) -> torch.Tensor:
        shard = self.map[P + name]
        if shard not in self._files:
            self._files[shard] = self._open(os.path.join(self.dir, shard), framework="pt", device=self.dev)
        return self._files[shard].get_tensor(P + name)


def read_bf16(path: str, width: int) -> torch.Tensor:
    """A file of bf16 [rows][width] as fp32."""
    bits = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(bits.view(np.int16).astype(np.int32) << 16).view(torch.float32).reshape(-1, width)


def axis_taps_weights(index, size, side):
    """vision_utils._interpolation_axis_taps_weights: bilinear, align_corners False, border padding."""
    index = index.to(torch.float32)
    src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - distance).clamp(min=0)


def block_major(h, w, merge, dev):
    """Each patch's (row, col) on the h x w grid in the order the tower takes them: merge-block-major."""
    within = torch.arange(h * w, device=dev)
    blocks_w = w // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col


def interpolation(h, w, side, merge, dev):
    """vision_utils.get_vision_interpolation_indices_and_weights for one image of one frame."""
    row, col = block_major(h, w, merge, dev)
    heights, widths = torch.full_like(row, h), torch.full_like(col, w)
    h_taps, h_weights = axis_taps_weights(row, heights, side)
    w_taps, w_weights = axis_taps_weights(col, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def position_ids(h, w, merge, dev):
    """vision_utils.get_vision_position_ids for one image of one frame: [h * w][2], (h, w) a patch."""
    hpos, wpos = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    hpos = hpos.reshape(shape).transpose(1, 2).flatten()
    wpos = wpos.reshape(shape).transpose(1, 2).flatten()
    return torch.stack([hpos, wpos], dim=-1)


def rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(q, k, cos, sin):
    dt = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((q * cos) + (rotate_pairwise(q) * sin)).to(dt), ((k * cos) + (rotate_pairwise(k) * sin)).to(dt)


@torch.no_grad()
def tower(W, patches, rows, cols, cfg, attn="sdpa", theta=10000.0):
    """patches bf16 [rows * cols * merge^2][patch^2 * 3], what Inkling2VisionEmbeddings.forward's
    reshape(-1, merge, patch, merge, patch, C).transpose(2, 3) makes of pixel_values. Returns the
    feature rows [rows * cols][text hidden] and the intermediates by the engine's dump names."""
    dev, acts = patches.device, {}
    merge = cfg.get("spatial_merge_size", cfg.get("adapter_merge_size", 2))
    heads = cfg.get("num_attention_heads", 16)
    layers = cfg.get("num_hidden_layers", 23)
    eps = cfg.get("layer_norm_eps", 1e-5)
    if cfg.get("hidden_act", "gelu") != "gelu" or cfg.get("projector_bias", False):
        raise SystemExit("vision_config: hidden_act / projector_bias are not the ones transcribed here")
    e = "backbone.embeddings."
    table = W(e + "position_embedding")                       # nn.Embedding(side^2, hidden).weight
    side = int(round(table.shape[0] ** 0.5))
    d = table.shape[1]
    hd = d // heads
    h, w = rows * merge, cols * merge

    # Inkling2VisionEmbeddings
    x = F.linear(patches, W(e + "patch_embedding.weight"))
    indices, weights = interpolation(h, w, side, merge, dev)
    pos = (table[indices] * weights[:, :, None]).sum(1)      # bf16 * fp32: fp32
    x = x + pos.to(x.dtype)
    cls = (W(e + "class_embedding") + W(e + "cls_position_embedding")).to(x.dtype)
    x = torch.cat([cls[None], x])                             # one image: the class token, then its patches
    acts["embed"] = x
    x = F.layer_norm(x, (d,), W("backbone.ln_pre.weight"), W("backbone.ln_pre.bias"), eps)
    acts["pre"] = x
    s = x.shape[0]

    # patches are 1-indexed (row, col) and the class token sits at (0, 0); Inkling2VisionRotaryEmbedding
    pid = torch.cat([torch.zeros(1, 2, dtype=torch.long, device=dev), position_ids(h, w, merge, dev) + 1])
    spatial = hd // 2
    inv_freq = (1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))).to(dev)
    freqs = pid[..., None].float() * inv_freq                 # [s][2][hd / 4]
    recompose = lambda f: torch.cat([f[:, 1], f[:, 0]], dim=-1).repeat_interleave(2, dim=-1)   # width, then height
    cos, sin = recompose(freqs.cos()), recompose(freqs.sin())

    for i in range(layers):
        l = f"backbone.layers.{i}."
        a = l + "self_attn."
        # Inkling2VisionEncoderLayer / Inkling2VisionAttention
        y = F.layer_norm(x, (d,), W(l + "ln_1.weight"), W(l + "ln_1.bias"), eps)
        q = F.linear(y, W(a + "q_proj.weight"), W(a + "q_proj.bias")).view(s, heads, hd)
        k = F.linear(y, W(a + "k_proj.weight"), W(a + "k_proj.bias")).view(s, heads, hd)
        v = F.linear(y, W(a + "v_proj.weight"), W(a + "v_proj.bias")).view(s, heads, hd)
        q, k = apply_rope(q, k, cos, sin)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        if attn == "eager":                                   # eager_attention_forward
            p = torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5
            p = F.softmax(p, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(p, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        o = o.transpose(1, 2).contiguous().reshape(s, -1)
        o = F.linear(o, W(a + "out_proj.weight"), W(a + "out_proj.bias"))
        x = x + W(l + "ls_1.gamma") * o
        y = F.layer_norm(x, (d,), W(l + "ln_2.weight"), W(l + "ln_2.bias"), eps)
        y = F.linear(y, W(l + "mlp.c_fc.weight"), W(l + "mlp.c_fc.bias"))
        y = F.linear(F.gelu(y), W(l + "mlp.c_proj.weight"), W(l + "mlp.c_proj.bias"))
        x = x + W(l + "ls_2.gamma") * y
        acts[f"b{i}"] = x

    # Inkling2VisionModel.forward's tail and Inkling2VisionMerger: no final norm, the class token dropped
    y = F.gelu(F.linear(x[1:].reshape(-1, merge ** 2 * d), W("proj_in.weight")))
    acts["merge"] = y
    return F.linear(y, W("proj_out.weight")), acts


def compare(name, eng, ref):
    ref = ref.float().cpu().reshape(-1, ref.shape[-1])
    if eng.shape != ref.shape:
        print(f"  {name}: shapes differ, engine {tuple(eng.shape)} reference {tuple(ref.shape)}")
        return None
    rel = ((eng - ref).norm() / ref.norm()).item()
    cos = F.cosine_similarity(eng, ref, dim=-1)
    print(f"  {name}: {ref.shape[0]} rows x {ref.shape[1]}, rel err {rel:.3e}, row cosine min {cos.min().item():.5f} "
          f"mean {cos.mean().item():.5f}, max |diff| {(eng - ref).abs().max().item():.3e} (max |ref| {ref.abs().max().item():.3e})")
    return cos.min().item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dense", required=True, help="the checkpoint directory (config.json, model.visual.* BF16)")
    ap.add_argument("--patches", required=True, help="bf16 [rows * cols * 4][1200]: inkling_vision_check's --patches-out")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--engine", required=True, help="bf16 [rows * cols][4608]: inkling_vision_check's --out")
    ap.add_argument("--dump", help="the engine's TIDELM_VISION_DUMP prefix: compare the intermediates too")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--theta", type=float, default=10000.0, help="RoPE base (transformers' default_theta)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--floor", type=float, default=0.99, help="fail under this row cosine")
    a = ap.parse_args(argv)

    cfg = json.load(open(os.path.join(a.dense, "config.json")))["vision_config"]
    W = Ckpt(a.dense, a.device)
    kin = W("backbone.embeddings.patch_embedding.weight").shape[1]
    patches = read_bf16(a.patches, kin).to(a.device, BF)
    merge = cfg.get("spatial_merge_size", cfg.get("adapter_merge_size", 2))
    if patches.shape[0] != a.rows * a.cols * merge ** 2:
        print(f"{a.patches}: {patches.shape[0]} patches, {a.rows} x {a.cols} tokens need {a.rows * a.cols * merge ** 2}")
        return 1
    ref, acts = tower(W, patches, a.rows, a.cols, cfg, attn=a.attn, theta=a.theta)
    print(f"{a.rows} x {a.cols} tokens, {patches.shape[0]} patches, {a.device}, attention {a.attn}, theta {a.theta:g}")
    if a.dump:
        for name, t in acts.items():
            f = f"{a.dump}.{name}.bin"
            if os.path.exists(f):
                compare(name, read_bf16(f, t.shape[-1]), t)
    low = compare("tower", read_bf16(a.engine, ref.shape[-1]), ref)
    if low is None or not low >= a.floor:
        print(f"FAIL: row cosine under {a.floor}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
