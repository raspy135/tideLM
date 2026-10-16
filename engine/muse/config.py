"""The text half of a Muse Spark `config.json`, as the engine reads it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MuseConfig:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    experts: int
    topk: int
    moe_hidden: int
    moe_inter: int
    sliding_window: int
    sliding: tuple[bool, ...]          # per layer: sliding-window attention (else full)
    rope: tuple[bool, ...]             # per layer: RoPE applied (full layers are NoPE)
    rope_theta: float
    rms_eps: float
    post_norm_eps: float
    qk_scale: float
    output_multiplier: float
    softcap: float
    residual_gate_temperature: float
    routed_scaling: float
    bos: int = 200000
    eos: tuple[int, ...] = (200001,)
    pad: int = 200018
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def kv_groups(self) -> int:
        return self.heads // self.kv_heads

    @classmethod
    def from_dict(cls, raw: dict) -> "MuseConfig":
        t = raw.get("text_config", raw)
        layers = int(t["num_hidden_layers"])
        types = list(t["layer_types"])
        mlp = list(t.get("mlp_layer_types") or ["sparse"] * layers)
        if any(m != "sparse" for m in mlp):
            raise ValueError("dense MLP layers are not supported: every Muse Spark layer is MoE")
        if int(t.get("num_shared_experts", 0) or 0):
            raise ValueError("shared experts are not supported (Muse Spark has none)")
        thetas = list(t.get("layer_rope_theta") or [t["rope_parameters"]["rope_theta"]] * layers)
        rope = t.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rope.get('rope_type')} is not supported")
        nonzero = {float(x) for x in thetas if x}
        if len(nonzero) > 1:
            raise ValueError("per-layer RoPE thetas other than one value or 0 are not supported")
        eos = t.get("eos_token_id", 200001)
        return cls(
            hidden=int(t["hidden_size"]), layers=layers,
            heads=int(t["num_attention_heads"]), kv_heads=int(t["num_key_value_heads"]),
            head_dim=int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]),
            vocab=int(t["vocab_size"]), experts=int(t["num_local_experts"]),
            topk=int(t["num_experts_per_tok"]),
            moe_hidden=int(t.get("moe_hidden_size") or t["moe_intermediate_size"]),
            moe_inter=int(t["moe_intermediate_size"]),
            sliding_window=int(t["sliding_window"]),
            sliding=tuple(x == "sliding_attention" for x in types),
            rope=tuple(bool(x) for x in thetas),
            rope_theta=float(nonzero.pop() if nonzero else rope.get("rope_theta", 500000.0)),
            rms_eps=float(t.get("rms_norm_eps", 1e-5)),
            post_norm_eps=float(t.get("post_norm_eps", 1e-8)),
            qk_scale=float(t.get("qk_scale_factor", 5.4730064863875)),
            output_multiplier=float(t.get("output_multiplier", 0.1767766952966369)),
            softcap=float(t.get("final_logit_softcapping", 20.0)),
            residual_gate_temperature=float(t.get("residual_gate_temperature", 0.3)),
            routed_scaling=float(t.get("routed_scaling_factor", 1.0)),
            bos=int(t.get("bos_token_id", 200000)),
            eos=tuple(eos) if isinstance(eos, list) else (int(eos),),
            pad=int(t.get("pad_token_id", 200018)),
            extra={"raw": raw},
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "MuseConfig":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))
