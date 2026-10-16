"""engine/glm/ref.py against transformers' own glm_moe_dsa code (CPU, fp32, random tiny config).
Needs transformers with models.glm_moe_dsa (5.15+); skipped otherwise."""

import copy
import unittest

import torch

from engine.glm.config import GlmConfig
from engine.glm.ref import GlmRef


def tiny_raw():
    return {
        "hidden_size": 64, "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 4,
        "vocab_size": 128, "q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "index_n_heads": 2, "index_head_dim": 16,
        "index_topk": 5, "index_topk_pattern": "FFSSF", "first_k_dense_replace": 1,
        "intermediate_size": 96, "n_routed_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32, "n_shared_experts": 1, "n_group": 2, "topk_group": 1,
        "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rope_interleave": True,
        "indexer_rope_interleave": True, "rms_norm_eps": 1e-5, "num_nextn_predict_layers": 0,
        "max_position_embeddings": 4096, "pad_token_id": 0, "eos_token_id": [1],
        "tie_word_embeddings": False, "attention_bias": False, "hidden_act": "silu",
    }


def hf_available():
    try:
        import transformers.models.glm_moe_dsa  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(hf_available(), "transformers with models.glm_moe_dsa not installed")
class TransformersParityTests(unittest.TestCase):
    def test_prefill_and_decode_match_transformers(self):
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        torch.manual_seed(0)
        raw = tiny_raw()
        kw = copy.deepcopy(raw)
        hf_cfg = GlmMoeDsaConfig(**kw)
        hf_cfg._attn_implementation = "eager"
        model = GlmMoeDsaForCausalLM(hf_cfg).float().eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 1:
                    p.copy_(1.0 + torch.randn_like(p) * 0.2)
                elif "gate.weight" in name or "experts" in name:
                    p.copy_(torch.randn_like(p) * 0.2)
            for layer in model.model.layers:
                if hasattr(layer.mlp, "gate"):
                    layer.mlp.gate.e_score_correction_bias.copy_(torch.randn(8) * 0.1)
        sd = {k.removeprefix("model."): v.detach().clone() for k, v in model.state_dict().items()}
        cfg = GlmConfig.from_dict(raw)
        self.assertEqual(cfg.indexer_full, (True, True, False, False, True))

        def expert_fn(L, e):
            gu = sd[f"layers.{L}.mlp.experts.gate_up_proj"][e]
            return gu[:32], gu[32:], sd[f"layers.{L}.mlp.experts.down_proj"][e]

        ours = GlmRef(cfg, lambda n: sd[n], expert_fn, dtype=torch.float32)
        ids = torch.randint(2, 128, (14,), generator=torch.Generator().manual_seed(3))

        def rel(x, y):
            return float((x - y).norm() / y.norm())
        # Prefill against a transformers prefill.
        with torch.no_grad():
            want = model(input_ids=ids[None]).logits[0]
        got = ours.forward(ids)
        self.assertLess(rel(got, want), 1e-5, rel(got, want))
        self.assertTrue(torch.equal(got.argmax(-1), want.argmax(-1)))
        # Step-by-step decode against transformers' own cached decode. (Not against a prefill:
        # the indexer's relu leaves exact-zero ties that topk breaks differently for 13 and 14
        # keys, so a prefill and a decode may legitimately pick different keys.)
        from transformers import DynamicCache
        ours.reset()
        cache = DynamicCache(config=hf_cfg)
        with torch.no_grad():
            want = [model(input_ids=ids[None, :9], past_key_values=cache, use_cache=True).logits[0]]
            for t in range(9, 14):
                want.append(model(input_ids=ids[None, t:t + 1], past_key_values=cache,
                                  use_cache=True).logits[0])
        want = torch.cat(want)
        got = [ours.forward(ids[:9])] + [ours.forward(ids[t:t + 1]) for t in range(9, 14)]
        got = torch.cat(got)
        self.assertLess(rel(got, want), 1e-5, rel(got, want))
        self.assertTrue(torch.equal(got.argmax(-1), want.argmax(-1)))


if __name__ == "__main__":
    unittest.main()
