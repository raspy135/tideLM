"""GLM-5.3 checkpoints in AMD's Quark layout (amd/GLM-5.3-Quark-MXFP4-AttnFP8), read lazily.

Read from the release's shard headers (2026-10-15):

* attention q_a/q_b/kv_a/kv_b/o and the indexer's wq_b/wk: F8_E4M3 with an F32
  ``weight_scale`` per 128x128 block (engine/glm/fp8_block.py);
* routed experts, shared experts and the 3 dense MLPs: MXFP4, U8 [N, K/2] (low nibble first)
  + E8M0 ``weight_scale`` [N, K/32] -- the layout tools/fp4_moe_cuda.cu reads;
* router (BF16 weight, F32 correction bias), norms, indexer weights_proj, embedding and
  lm_head: BF16;
* layer 78 is the MTP layer (eh_proj, enorm, hnorm, shared_head.norm, a full decoder layer).

The raw reader is engine/muse/weights.Checkpoint (preadv; safetensors slices leaked host memory
on this box, docs/gotchas.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from engine.muse.weights import Checkpoint, ExpertMX, dequant_mxfp4, quantize_mxfp4  # noqa: F401
from .config import GlmConfig
from .fp8_block import FP8Block, SB

P = "model."


class GlmCheckpoint(Checkpoint):
    def __init__(self, model_dir: str | Path):
        super().__init__(model_dir, detect=False)
        self.format = "glm-quark"

    def fp8(self, name: str, rows: tuple[int, int] | None = None) -> FP8Block:
        """An FP8 block-scaled weight (optionally a 128-aligned row range)."""
        if rows is None:
            return FP8Block(self.get(name), self.get(name + "_scale"))
        a, b = rows
        return FP8Block(self.get_rows(name, a, b), self.get_rows(name + "_scale", a // SB, (b + SB - 1) // SB))

    def fp8_float(self, name: str) -> torch.Tensor:
        return self.fp8(name).dequant(torch.float32)

    def mx_rows(self, name: str, a: int, b: int, urgent: bool = False):
        return self.get_rows(name, a, b, urgent), self.get_rows(name + "_scale", a, b, urgent)

    def mx_float(self, name: str) -> torch.Tensor:
        return dequant_mxfp4(self.get(name), self.get(name + "_scale"))

    def has(self, name: str) -> bool:
        return name in self.map


def reference_getter(ck: GlmCheckpoint, cfg: GlmConfig):
    """``get``/``expert_fn`` for engine/glm/ref.GlmRef over this checkpoint (dequantized)."""
    def get(name):
        full = name if name == "lm_head.weight" else P + name
        if ck.has(full + "_scale"):
            if ck.get(full).dtype == torch.float8_e4m3fn:
                return ck.fp8_float(full)
            return ck.mx_float(full)
        return ck.get(full)

    def expert_fn(L, e):
        b = f"{P}layers.{L}.mlp.experts.{e}."
        return tuple(ck.mx_float(b + f"{p}_proj.weight") for p in ("gate", "up", "down"))
    return get, expert_fn


def _fp8_quant(w: torch.Tensor):
    N, K = w.shape
    nb, kb = (N + SB - 1) // SB, K // SB
    pad = torch.zeros(nb * SB, K, dtype=torch.float32, device=w.device)
    pad[:N] = w.float()
    amax = pad.view(nb, SB, kb, SB).abs().amax((1, 3)).clamp_min(1e-12)
    s = amax / 448.0
    q = (pad.view(nb, SB, kb, SB) / s[:, None, :, None]).view(nb * SB, K)[:N].to(torch.float8_e4m3fn)
    return q.cpu(), s.cpu()


def write_synthetic(model_dir: str | Path, raw: dict, *, seed: int = 0, std: float = 0.02,
                    bias_std: float = 0.05, device: str = "cpu") -> None:
    """A random checkpoint in AMD's Quark layout (tests), with the MTP layer when the config has
    num_nextn_predict_layers (layer index = num_hidden_layers, a full-indexer MoE layer)."""
    from safetensors.torch import save_file
    cfg = GlmConfig.from_dict(raw)
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, s=std):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    t: dict[str, torch.Tensor] = {}

    def fp8(name, w):
        q, s = _fp8_quant(w.to(device))
        t[name], t[name + "_scale"] = q, s

    def mx(name, w):
        q, s = quantize_mxfp4(w.to(device))
        t[name], t[name + "_scale"] = q.cpu(), s.cpu()

    H = cfg.hidden
    t[P + "embed_tokens.weight"] = rnd(cfg.vocab, H, s=1.0)
    t[P + "norm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
    t["lm_head.weight"] = rnd(cfg.vocab, H)
    for L in range(cfg.layers + cfg.mtp_layers):
        mtp = L >= cfg.layers
        p = f"{P}layers.{L}."
        a = p + "self_attn."
        if mtp:
            t[p + "enorm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
            t[p + "hnorm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
            t[p + "shared_head.norm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
            t[p + "eh_proj.weight"] = rnd(H, 2 * H)
        t[p + "input_layernorm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
        t[p + "post_attention_layernorm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
        fp8(a + "q_a_proj.weight", rnd(cfg.q_lora, H))
        t[a + "q_a_layernorm.weight"] = (1.0 + torch.randn(cfg.q_lora, generator=g) * 0.1).to(torch.bfloat16)
        fp8(a + "q_b_proj.weight", rnd(cfg.heads * cfg.qk_dim, cfg.q_lora))
        fp8(a + "kv_a_proj_with_mqa.weight", rnd(cfg.kv_lora + cfg.rope_dim, H))
        t[a + "kv_a_layernorm.weight"] = (1.0 + torch.randn(cfg.kv_lora, generator=g) * 0.1).to(torch.bfloat16)
        fp8(a + "kv_b_proj.weight", rnd(cfg.heads * (cfg.nope + cfg.v_dim), cfg.kv_lora))
        fp8(a + "o_proj.weight", rnd(H, cfg.heads * cfg.v_dim))
        if mtp or cfg.indexer_full[L]:
            i = a + "indexer."
            fp8(i + "wq_b.weight", rnd(cfg.index_heads * cfg.index_dim, cfg.q_lora))
            fp8(i + "wk.weight", rnd(cfg.index_dim, H))
            t[i + "k_norm.weight"] = (1.0 + torch.randn(cfg.index_dim, generator=g) * 0.1).to(torch.bfloat16)
            t[i + "k_norm.bias"] = rnd(cfg.index_dim, s=0.1)
            t[i + "weights_proj.weight"] = rnd(cfg.index_heads, H)
        m = p + "mlp."
        if not mtp and cfg.dense_mlp[L]:
            for n, shape in (("gate", (cfg.inter, H)), ("up", (cfg.inter, H)), ("down", (H, cfg.inter))):
                mx(m + f"{n}_proj.weight", rnd(*shape))
            continue
        t[m + "gate.weight"] = rnd(cfg.experts, H)
        t[m + "gate.e_score_correction_bias"] = (torch.randn(cfg.experts, generator=g) * bias_std).float()
        si = cfg.moe_inter * cfg.shared_experts
        for n, shape in (("gate", (si, H)), ("up", (si, H)), ("down", (H, si))):
            mx(m + f"shared_experts.{n}_proj.weight", rnd(*shape))
        for e in range(cfg.experts):
            for n, shape in (("gate", (cfg.moe_inter, H)), ("up", (cfg.moe_inter, H)), ("down", (H, cfg.moe_inter))):
                mx(m + f"experts.{e}.{n}_proj.weight", rnd(*shape))
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(t, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(raw, indent=1))
