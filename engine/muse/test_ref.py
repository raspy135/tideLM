"""engine/muse/ref.py against transformers' own Muse Spark code, and the MXFP4 helpers. CPU only.

The transformers comparison needs a transformers with ``models.muse_spark`` (5.20+); it is
skipped otherwise. Run (repo root):  python -m unittest engine.muse.test_ref
"""

import copy
import tempfile
import unittest

import torch

from engine.muse.config import MuseConfig
from engine.muse.ref import MuseRef, from_checkpoint, route
from engine.muse.weights import (Checkpoint, dequant_mxfp4, quantize_mxfp4, write_synthetic)


def tiny_raw(layers=4, window=6):
    types = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"][:layers]
    return {"text_config": {
        "hidden_size": 256, "num_hidden_layers": layers, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 512,
        "num_local_experts": 8, "num_experts_per_tok": 2, "moe_hidden_size": 128,
        "moe_intermediate_size": 128, "intermediate_size": 512, "sliding_window": window,
        "layer_types": types, "mlp_layer_types": ["sparse"] * layers,
        "layer_rope_theta": [10000.0 if t == "sliding_attention" else 0 for t in types],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8, "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.1767766952966369, "final_logit_softcapping": 20.0,
        "residual_gate_temperature": 0.3, "routed_scaling_factor": 1.0,
        "num_shared_experts": 0, "hidden_act": "silu", "attention_bias": False,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "max_position_embeddings": 4096,
    }}


def hf_available():
    try:
        import transformers.models.muse_spark  # noqa: F401
        return True
    except Exception:
        return False


class MXFP4Tests(unittest.TestCase):
    def test_grid_values_round_trip_exactly(self):
        grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6, 6] * 2)
        w = (grid * 2.0 ** -3).view(1, 32)
        packed, scale = quantize_mxfp4(w)
        self.assertTrue(torch.equal(dequant_mxfp4(packed, scale), w))

    def test_low_nibble_is_the_even_element(self):
        w = torch.zeros(1, 32)
        w[0, 0], w[0, 1] = 1.0, -6.0
        packed, scale = quantize_mxfp4(w)
        # amax 6 -> exponent floor(log2 6) - 2 = 0, so codes are the raw E2M1 values.
        self.assertEqual(int(scale[0, 0]), 127)
        self.assertEqual(int(packed[0, 0]) & 15, 2)        # +1.0
        self.assertEqual(int(packed[0, 0]) >> 4, 8 | 7)    # -6.0

    def test_random_error_is_fp4_sized(self):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(64, 256, generator=g)
        d = dequant_mxfp4(*quantize_mxfp4(w))
        rel = float((d - w).norm() / w.norm())
        self.assertLess(rel, 0.15)
        self.assertGreater(rel, 0.03)


class CheckpointTests(unittest.TestCase):
    def test_synthetic_mxfp4_round_trips_through_the_reader(self):
        raw = tiny_raw()
        with tempfile.TemporaryDirectory() as d:
            write_synthetic(d, raw, bf16_expert_layers=(0,))
            ck = Checkpoint(d)
            self.assertEqual(ck.format, "mxfp4")
            self.assertEqual(ck.expert_layer_format(0), "bf16-split")
            self.assertEqual(ck.expert_layer_format(1), "mxfp4")
            mx = ck.expert_mx(1, 3)
            gate, up, down = ck.expert_float(1, 3)
            self.assertTrue(torch.equal(dequant_mxfp4(mx.w1, mx.s1), gate))
            self.assertEqual(tuple(mx.w2.shape), (128, 64))
            # A BF16 layer comes back quantized for the engine and exact for the reference.
            mx0 = ck.expert_mx(0, 0)
            g0 = ck.expert_float(0, 0)[0]
            rel = float((dequant_mxfp4(mx0.w1, mx0.s1) - g0).norm() / g0.norm())
            self.assertLess(rel, 0.15)
            model = from_checkpoint(d)
            logits = model.forward(torch.tensor([1, 5, 9, 4]))
            self.assertEqual(tuple(logits.shape), (4, 512))
            self.assertTrue(torch.isfinite(logits).all())


class RouteTests(unittest.TestCase):
    def test_masked_experts_are_substituted_and_weights_renormalize(self):
        cfg = MuseConfig.from_dict(tiny_raw())
        g = torch.Generator().manual_seed(1)
        x = torch.randn(5, 256, generator=g)
        W = torch.randn(8, 256, generator=g) * 0.02
        bias = torch.zeros(8)
        w, ids = route(cfg, x, W, bias)
        banned = ids[0, 0].item()
        allowed = torch.ones(8, dtype=torch.bool)
        allowed[banned] = False
        w2, ids2 = route(cfg, x, W, bias, allowed)
        self.assertFalse((ids2 == banned).any())
        self.assertTrue(torch.allclose(w2.sum(-1), torch.ones(5)))


@unittest.skipUnless(hf_available(), "transformers with models.muse_spark not installed")
class TransformersParityTests(unittest.TestCase):
    def build(self, seed=0):
        from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
        from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextModel
        torch.manual_seed(seed)
        raw = tiny_raw()
        tc = copy.deepcopy(raw["text_config"])
        tc.pop("hidden_act", None)
        hf_cfg = MuseSparkTextConfig(**tc)
        hf_cfg._attn_implementation = "eager"
        model = MuseSparkTextModel(hf_cfg).float().eval()
        # The initializer zeroes the norms and gates; randomize them so every term matters.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 1:
                    p.copy_(torch.randn_like(p) * 0.3)
            for layer in model.layers:
                layer.mlp.gate.e_score_correction_bias.copy_(torch.randn(8) * 0.05)
        head = torch.randn(512, 256) * 0.05
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state["lm_head.weight"] = head
        cfg = MuseConfig.from_dict(raw)

        def expert_fn(L, e):
            gu = state[f"layers.{L}.mlp.experts.gate_up_proj"][e]
            return gu[:128], gu[128:], state[f"layers.{L}.mlp.experts.down_proj"][e]

        ours = MuseRef(cfg, lambda n: state[n], expert_fn, dtype=torch.float32)
        return model, head, ours, cfg

    def hf_logits(self, model, head, cfg, ids):
        with torch.no_grad():
            h = model(input_ids=ids[None]).last_hidden_state[0]
            z = h @ head.T * cfg.output_multiplier / cfg.softcap
            return torch.tanh(z) * cfg.softcap

    def test_prefill_and_incremental_decode_match_transformers(self):
        model, head, ours, cfg = self.build()
        ids = torch.randint(3, 512, (15,), generator=torch.Generator().manual_seed(2))
        want = self.hf_logits(model, head, cfg, ids)
        got = [ours.forward(ids[:10])]
        for t in range(10, 15):
            got.append(ours.forward(ids[t:t + 1]))
        got = torch.cat(got)
        err = float((got - want).abs().max())
        self.assertLess(err, 2e-4, err)
        self.assertTrue(torch.equal(got.argmax(-1), want.argmax(-1)))


if __name__ == "__main__":
    unittest.main()
