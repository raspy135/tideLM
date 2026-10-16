"""The parts of a GLM-5.3 (glm_moe_dsa) config.json the engine reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GlmConfig:
    hidden: int
    layers: int
    heads: int
    vocab: int
    q_lora: int
    kv_lora: int
    nope: int
    rope_dim: int
    v_dim: int
    index_heads: int
    index_dim: int
    index_topk: int
    indexer_full: tuple[bool, ...]      # per layer: runs its own indexer (else reuses the last)
    dense_mlp: tuple[bool, ...]         # per layer: dense MLP (else MoE)
    inter: int                          # dense MLP intermediate
    experts: int
    topk: int
    moe_inter: int
    shared_experts: int
    n_group: int
    topk_group: int
    routed_scaling: float
    norm_topk: bool
    rope_theta: float
    rms_eps: float
    mtp_layers: int
    eos: tuple[int, ...]
    pad: int
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def qk_dim(self) -> int:
        return self.nope + self.rope_dim

    @classmethod
    def from_dict(cls, raw: dict) -> "GlmConfig":
        t = raw.get("text_config", raw)
        L = int(t["num_hidden_layers"])
        types = t.get("indexer_types")
        if types is None:
            pattern = t.get("index_topk_pattern")
            if pattern is not None:
                types = [{"F": "full", "S": "shared"}[c] for c in pattern]
            else:
                freq = max(int(t.get("index_topk_freq", 1)), 1)
                off = int(t.get("index_skip_topk_offset", 2))
                types = ["full" if (max(i - off + 1, 0) % freq) == 0 else "shared" for i in range(L)]
        mlp = t.get("mlp_layer_types")
        if mlp is None:
            n = min(int(t.get("first_k_dense_replace", 1)), L)
            mlp = ["dense"] * n + ["sparse"] * (L - n)
        rope = t.get("rope_parameters") or {"rope_theta": t.get("rope_theta", 10000.0), "rope_type": "default"}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rope.get('rope_type')} is not supported")
        if not t.get("rope_interleave", True) or not t.get("indexer_rope_interleave", True):
            raise ValueError("only interleaved RoPE is implemented (GLM-5.3's layout)")
        if types[0] != "full":
            raise ValueError("layer 0 must run its own indexer")
        eos = t.get("eos_token_id", [])
        return cls(
            hidden=int(t["hidden_size"]), layers=L, heads=int(t["num_attention_heads"]),
            vocab=int(t["vocab_size"]), q_lora=int(t["q_lora_rank"]), kv_lora=int(t["kv_lora_rank"]),
            nope=int(t["qk_nope_head_dim"]), rope_dim=int(t["qk_rope_head_dim"]),
            v_dim=int(t["v_head_dim"]), index_heads=int(t["index_n_heads"]),
            index_dim=int(t["index_head_dim"]), index_topk=int(t["index_topk"]),
            indexer_full=tuple(x == "full" for x in types), dense_mlp=tuple(x == "dense" for x in mlp),
            inter=int(t["intermediate_size"]),
            experts=int(t.get("n_routed_experts") or t.get("num_local_experts")),
            topk=int(t["num_experts_per_tok"]), moe_inter=int(t["moe_intermediate_size"]),
            shared_experts=int(t.get("n_shared_experts", 1)), n_group=int(t.get("n_group", 1)),
            topk_group=int(t.get("topk_group", 1)), routed_scaling=float(t["routed_scaling_factor"]),
            norm_topk=bool(t.get("norm_topk_prob", True)), rope_theta=float(rope["rope_theta"]),
            rms_eps=float(t.get("rms_norm_eps", 1e-5)),
            mtp_layers=int(t.get("num_nextn_predict_layers", 0)),
            eos=tuple(eos) if isinstance(eos, list) else (int(eos),),
            pad=int(t.get("pad_token_id", 0) or 0), extra={"raw": raw})

    @classmethod
    def load(cls, model_dir: str | Path) -> "GlmConfig":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))
