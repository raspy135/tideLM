"""Two-rank check of engine/glm/tp.py against engine/glm/ref.py on a synthetic checkpoint in
AMD's Quark layout, at GLM-5.3's per-layer shapes (hidden 6144, 64 MLA heads, latent 512, indexer
32x128, experts 2048 wide, a 12288 dense MLP) with 4 layers: dense+full indexer, MoE+full,
MoE+shared, MoE+full; 16 experts, top-8, indexer top-64 so a 600-token prompt is sparse.
Run: tools/muse_pair_run.sh glm_pair_test.py (serving stopped)."""

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import torch  # noqa: E402

from engine.glm.config import GlmConfig  # noqa: E402
from engine.glm.ref import GlmRef  # noqa: E402
from engine.glm.tp import GlmTP  # noqa: E402
from engine.glm.weights import GlmCheckpoint, reference_getter, write_synthetic  # noqa: E402

RAW = {
    "hidden_size": 6144, "num_hidden_layers": 4, "num_attention_heads": 64, "num_key_value_heads": 64,
    "vocab_size": 32768, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128,
    "index_topk": 64, "index_topk_pattern": "FFSF", "first_k_dense_replace": 1,
    "intermediate_size": 12288, "n_routed_experts": 16, "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
    "routed_scaling_factor": 2.5, "norm_topk_prob": True,
    "rope_parameters": {"rope_theta": 8000000.0, "rope_type": "default"}, "rope_interleave": True,
    "indexer_rope_interleave": True, "rms_norm_eps": 1e-5, "num_nextn_predict_layers": 0,
    "pad_token_id": 0, "eos_token_id": [1], "max_position_embeddings": 65536,
}
MODEL = Path(os.environ.get("GLM_SYNTH_DIR", "/app/results/glm-synth-v1"))
FAILED = []


def say(*a):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, flush=True)


def check(name, ok, detail=""):
    say(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def compare(a, b):
    a, b = a.float(), b.float()
    return {"rel": round(float((a - b).norm() / b.norm()), 5),
            "argmax": round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 4)}


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

    def run(eng):
        eng.reset()
        out = [eng.forward(prompt)]
        for t in follow.tolist():
            out.append(eng.forward([t]))
        return torch.cat(out).cpu()

    def time_decode(eng, n=32):
        eng.reset()
        eng.forward(prompt)
        for t in range(3):
            eng.forward([5 + t])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(n):
            eng.forward([9 + t])
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    eng = GlmTP(str(MODEL), max_seq=2048, chunk=256, slots=16 * 3)
    full = run(eng)
    eng.use_graphs = False
    eager = run(eng)
    eng.use_graphs = True
    step_ms = time_decode(eng)
    del eng
    gc.collect(); torch.cuda.empty_cache()

    eng = GlmTP(str(MODEL), max_seq=2048, chunk=256, slots=24)
    allowed = (eng.lut >= 0).cpu()
    sub = run(eng)
    del eng
    gc.collect(); torch.cuda.empty_cache()

    eng = GlmTP(str(MODEL), max_seq=2048, chunk=256, slots=24, miss="stream")
    stream = run(eng)
    loads = eng.stats["stream_loads"]
    del eng
    gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        cfg = GlmConfig.from_dict(RAW)
        get, ef = reference_getter(GlmCheckpoint(str(MODEL)), cfg)
        r32 = GlmRef(cfg, get, ef, device="cuda", dtype=torch.float32)
        want32 = r32.forward(ids.cuda()).cpu()
        r16 = GlmRef(cfg, get, ef, device="cuda", dtype=torch.bfloat16)
        want16 = r16.forward(ids.cuda()).cpu()
        r16.reset()
        r16.allowed = {L: allowed[L] for L in range(cfg.layers)}
        want_sub = r16.forward(ids.cuda()).cpu()
        noise = compare(want16, want32)["rel"]
        r = {"bf16_ref_vs_fp32_ref": compare(want16, want32), "engine_vs_fp32_ref": compare(full, want32),
             "engine_decode_vs_fp32_ref": compare(full[600:], want32[600:]),
             "graph_vs_eager": compare(full[600:], eager[600:]), "substitute_vs_masked_ref": compare(sub, want_sub),
             "substitute_vs_full": compare(sub, full), "stream_vs_full": compare(stream, full)}
        say(json.dumps(r, indent=1))
        tol = max(2 * noise, 0.01)
        check("engine within 2x the reference's bf16 noise", r["engine_vs_fp32_ref"]["rel"] <= tol, f"noise {noise}")
        check("decode rows match", r["engine_decode_vs_fp32_ref"]["rel"] <= tol)
        check("graph replay equals eager decode", r["graph_vs_eager"]["rel"] == 0.0)
        check("substitution matches the masked reference", r["substitute_vs_masked_ref"]["rel"] <= tol)
        check("substitution changed the output", r["substitute_vs_full"]["rel"] > 0.01)
        check("streamed misses reproduce full residency", r["stream_vs_full"]["rel"] < 1e-3, f"{loads} loads")
        say(f"graphed decode step, 4 layers: {step_ms:.2f} ms")
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
