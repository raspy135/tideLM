"""Two-rank check of engine/muse/tp.py against engine/muse/ref.py on a synthetic checkpoint.

Real per-layer dimensions (hidden 8192, 128/16 heads x 64, latent experts 4096, top-8) with 4
layers, 16 experts, a 128-token sliding window and a 32K vocabulary, so every kernel runs at the
shapes the real model gives it. Run with tools/muse_pair_run.sh (serving stopped). Rank 0 prints
the comparisons; any failed check exits non-zero on both ranks.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.muse.ref import from_checkpoint  # noqa: E402
from engine.muse.tp import MuseTP  # noqa: E402
from engine.muse.weights import write_synthetic  # noqa: E402

LAYERS = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"]
RAW = {"text_config": {
    "hidden_size": 8192, "num_hidden_layers": 4, "num_attention_heads": 128,
    "num_key_value_heads": 16, "head_dim": 64, "vocab_size": 32768,
    "num_local_experts": 16, "num_experts_per_tok": 8, "moe_hidden_size": 4096,
    "moe_intermediate_size": 4096, "intermediate_size": 16384, "sliding_window": 128,
    "layer_types": LAYERS, "mlp_layer_types": ["sparse"] * 4,
    "layer_rope_theta": [500000.0 if t == "sliding_attention" else 0 for t in LAYERS],
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    # The real 5.47 with random q/k makes attention one-hot and the comparison chaotic.
    "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8, "qk_scale_factor": 1.0,
    "output_multiplier": 0.1767766952966369, "final_logit_softcapping": 20.0,
    "residual_gate_temperature": 0.3, "routed_scaling_factor": 1.0, "num_shared_experts": 0,
    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
}}
MODEL = Path(os.environ.get("MUSE_SYNTH_DIR", "/app/results/muse-synth-v2"))
FAILED = []


def say(*a):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, flush=True)


def check(name, ok, detail=""):
    say(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    rel = float((a - b).norm() / b.norm())
    agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    return {"rel": round(rel, 5), "argmax": round(agree, 4), "max_abs": round(float((a - b).abs().max()), 4)}


def time_decode(eng, steps=64) -> float:
    g = torch.Generator().manual_seed(3)
    eng.reset()
    eng.forward(torch.randint(3, 32768, (600,), generator=g))
    for t in range(4):
        eng.forward([5])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(steps):
        eng.forward([7 + t])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def main():
    rank = int(os.environ.get("RANK", 0))
    if not (MODEL / "model.safetensors").exists():
        t0 = time.time()
        write_synthetic(MODEL, RAW, seed=0, device="cuda", bias_std=2.0)
        print(f"[rank {rank}] wrote synthetic checkpoint in {time.time() - t0:.0f}s", flush=True)
    g = torch.Generator().manual_seed(7)
    prompt = torch.randint(3, 32768, (600,), generator=g)
    follow = torch.randint(3, 32768, (8,), generator=g)
    ids = torch.cat([prompt, follow])

    def run(engine) -> torch.Tensor:
        engine.reset()
        out = [engine.forward(prompt)]
        for t in follow.tolist():
            out.append(engine.forward([t]))
        return torch.cat(out).cpu()

    # A: everything resident, native decode kernels, prefill in 256-token chunks.
    eng = MuseTP(str(MODEL), max_seq=2048, chunk=256, dense="bf16", miss="substitute")
    layers_eng = []
    eng.forward(prompt[:200], hidden_out=layers_eng)
    layers_eng = [h.cpu() for h in layers_eng]
    full = run(eng)
    eng.use_graphs = False
    eager = run(eng)
    eng.use_graphs = True
    eng.native = False
    eng.prefill_moe = "dequant"
    full_dequant = run(eng)
    eng.native = True

    def time_prefill(n=1024):
        eng.reset()
        x = torch.randint(3, 32768, (n,), generator=torch.Generator().manual_seed(9))
        eng.forward(x[:64])
        eng.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.forward(x, all_logits=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000
    prefill_dequant_ms = time_prefill()
    eng.prefill_moe = "triton"
    prefill_triton_ms = time_prefill()
    step_ms = time_decode(eng)
    t0 = time.perf_counter()
    # Prefix reuse: an appended turn and an edit near the end must equal a fresh prefill.
    p1 = prompt[:400].tolist()
    eng.reset()
    gen = eng.generate(p1, 6)
    p2 = p1 + gen + follow.tolist()
    reuse_appended = eng.prefill(p2).cpu()
    n_appended = eng.last_reuse
    eng.reset()
    fresh_appended = eng.forward(p2, all_logits=False).cpu()
    p3 = p2[:-20] + [11, 12, 13]
    eng.prefill(p2)
    reuse_edit = eng.prefill(p3).cpu()
    n_edit = eng.last_reuse
    eng.reset()
    fresh_edit = eng.forward(p3, all_logits=False).cpu()
    eng.reset()
    toks = eng.generate(prompt[:64], 32)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    del eng
    gc.collect(); torch.cuda.empty_cache()

    # B: half the experts resident, substitution (router masks the rest).
    half = 4 * 16 // 2
    eng = MuseTP(str(MODEL), max_seq=2048, chunk=256, dense="bf16", slots=half, miss="substitute")
    allowed = (eng.arena.lut >= 0).cpu()
    sub = run(eng)
    del eng
    gc.collect(); torch.cuda.empty_cache()

    # C: half resident, misses streamed into transient slots: must equal A.
    eng = MuseTP(str(MODEL), max_seq=2048, chunk=256, dense="bf16", slots=half, miss="stream")
    stream = run(eng)
    loads = eng.stats["stream_loads"]
    del eng
    gc.collect(); torch.cuda.empty_cache()

    # D: FP8 dense weights, everything resident.
    eng = MuseTP(str(MODEL), max_seq=2048, chunk=256, dense="fp8", miss="substitute")
    fp8 = run(eng)
    step_ms_fp8 = time_decode(eng)
    del eng
    gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        ref = from_checkpoint(str(MODEL), device="cuda", dtype=torch.bfloat16)
        want = ref.forward(ids.cuda()).cpu()
        ref.reset()
        ref.allowed = {L: allowed[L] for L in range(4)}
        want_sub = ref.forward(ids.cuda()).cpu()
        ref32 = from_checkpoint(str(MODEL), device="cuda", dtype=torch.float32)
        layers_ref = []
        ref32.forward(prompt[:200].cuda(), hidden_out=layers_ref)
        ref32.reset()
        want32 = ref32.forward(ids.cuda()).cpu()
        ref.reset()
        layers_ref16 = []
        ref.allowed = {}
        ref.forward(prompt[:200].cuda(), hidden_out=layers_ref16)
        per_layer = [compare(a.cpu(), b.cpu())["rel"] for a, b in zip(layers_eng, layers_ref)]
        per_layer16 = [compare(a.cpu(), b.cpu())["rel"] for a, b in zip(layers_ref16, layers_ref)]
        say(f"per-layer hidden rel error vs fp32 ref: engine {per_layer} / bf16 ref {per_layer16}")
        check("layer 0 hidden state within 2x bf16 noise", per_layer[0] <= max(2 * per_layer16[0], 5e-3))
        r = {
            "ref_bf16_vs_ref_fp32": compare(want, want32),
            "engine_vs_ref_bf16": compare(full, want),
            "engine_vs_ref_fp32": compare(full, want32),
            "engine_decode_steps_vs_ref_fp32": compare(full[600:], want32[600:]),
            "native_vs_dequant_decode": compare(full[600:], full_dequant[600:]),
            "triton_vs_dequant_prefill": compare(full[:600], full_dequant[:600]),
            "graph_vs_eager_decode": compare(full[600:], eager[600:]),
            "substitute_vs_masked_ref": compare(sub, want_sub),
            "substitute_vs_full": compare(sub, full),
            "stream_vs_full": compare(stream, full),
            "fp8_vs_ref_fp32": compare(fp8, want32),
        }
        say(json.dumps(r, indent=1))
        noise = r["ref_bf16_vs_ref_fp32"]["rel"]
        check("engine within 2x the reference's own bf16 noise",
              r["engine_vs_ref_fp32"]["rel"] <= max(2 * noise, 0.01), f"{r['engine_vs_ref_fp32']} noise {noise}")
        check("decode steps match", r["engine_decode_steps_vs_ref_fp32"]["rel"] <= max(2 * noise, 0.01))
        ra, re_ = compare(reuse_appended, fresh_appended), compare(reuse_edit, fresh_edit)
        check("prefix reuse (appended turn) equals a fresh prefill",
              n_appended == len(p1) + len(gen) and ra["rel"] <= max(2 * noise, 0.01), f"reused {n_appended}: {ra}")
        check("prefix reuse (edit near the end) equals a fresh prefill",
              n_edit == len(p2) - 20 and re_["rel"] <= max(2 * noise, 0.01), f"reused {n_edit}: {re_}")
        check("graph replay equals eager decode", r["graph_vs_eager_decode"]["rel"] == 0.0,
              f"{r['graph_vs_eager_decode']}")
        say(f"decode step with graphs, 4 layers: bf16 dense {step_ms:.2f} ms, fp8 dense {step_ms_fp8:.2f} ms")
        check("triton prefill MoE agrees with dequant", r["triton_vs_dequant_prefill"]["rel"] <= max(2 * noise, 0.01),
              f"{r['triton_vs_dequant_prefill']}")
        say(f"prefill 1024 tokens (4 layers, chunk 256): dequant MoE {prefill_dequant_ms:.0f} ms, "
            f"triton MoE {prefill_triton_ms:.0f} ms")
        check("native and dequant MoE agree", r["native_vs_dequant_decode"]["rel"] <= max(2 * noise, 0.01))
        check("substitution matches the masked reference",
              r["substitute_vs_masked_ref"]["rel"] <= max(2 * noise, 0.01))
        check("substitution actually changed the output", r["substitute_vs_full"]["rel"] > 0.01)
        check("streamed misses reproduce full residency", r["stream_vs_full"]["rel"] < 1e-3,
              f"{loads} expert loads")
        # FP8 kernels against a reference holding the same FP8-rounded dense weights: isolates
        # kernel correctness from the cost of requantizing BF16 to FP8 (reported, not judged).
        from fp8_linear import quantize_to_fp8
        from engine.muse.ref import MuseRef
        from engine.muse.weights import Checkpoint, PREFIX
        ck = Checkpoint(str(MODEL))
        fp8_names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "pre_expert_proj",
                     "post_expert_proj", "lm_head")

        def get8(name):
            t = ck.get(name if name == "lm_head.weight" else PREFIX + name)
            if t.dim() == 2 and any(k in name for k in fp8_names):
                return quantize_to_fp8(t.cuda()).dequant().float().cpu()
            return t
        ref8 = MuseRef(ref32.cfg, get8, ck.expert_float, device="cuda", dtype=torch.float32)
        want8 = ref8.forward(ids.cuda()).cpu()
        r["fp8_vs_fp8_ref_fp32"] = compare(fp8, want8)
        r["fp8_ref_vs_ref_fp32 (requant cost)"] = compare(want8, want32)
        say(json.dumps({k: r[k] for k in list(r)[-2:]}))
        check("fp8 kernels match an fp8-weight reference", r["fp8_vs_fp8_ref_fp32"]["rel"] <= max(2 * noise, 0.01),
              f"{r['fp8_vs_fp8_ref_fp32']}")
        say(f"greedy 32 tokens (eager, 4 layers) in {gen_s:.2f}s: {toks[:8]}...")
    flags = [FAILED]
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        import torch.distributed as dist
        flags = [None, None]
        dist.all_gather_object(flags, FAILED)
    if any(flags):
        say(f"FAILED: {flags}")
        sys.exit(1)
    say("ALL PASS")


if __name__ == "__main__":
    main()
