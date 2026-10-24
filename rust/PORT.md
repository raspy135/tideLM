# Rust port of the DS v4.1 engine

Branch `rust-port`, workspace `rust/`. Goal: **no Python in the prefill/decode pipeline.**
Python stays for build tooling (Triton AOT, pack builders) and offline tools.

The port exists for performance, so it inherits the repo's rule in its strongest form:
no stage ships on a claim. Every milestone below has a measured gate against the Python
engine on the same box and workload, and a stage that loses decode tok/s does not merge
until the number says otherwise. Decode tok/s is the primary metric; prefill is
secondary. Negative results get recorded in `docs/gotchas.md` like everything else.

User-set goals, in order:
1. No Python in prefill/decode at runtime.
2. No performance regression — the port is *for* performance.
3. Concurrency on par with mainstream LLM engines (continuous batching), not the
   current `DSV41_MAX_CONCURRENCY=1` lanes.
4. Bigger CUDA graphs than the Python engine captures.

## What the engine is (inventory, 2026-10-24, shipped DS profile)

Mapped from `configs/ds41-mxfp4.env.example`: TP2 across two GB10 boxes, FP4 experts
with the lossless scale pack, RoCE one-shot gathers under 256 KB, `BLOCK_DYNAMIC=3,5`,
`DYNAMIC_EXPERTS=1` at `PRUNE_KEEP=0.61`, prefill graphs off.

Three native tiers already exist, and they decide the FFI strategy:

1. **Pure C ABI, already Rust-callable** (done, see `tidelm-kernels`):
   `tools/fp4_moe_cuda.cu` (the entire decode MoE: `route_small`, `up_scales`/`up_v2`,
   `down_scales`/`down_v3`) and `engine/engram_gather.cpp` (CPU row gather). Stream-first
   args, `cudaGetLastError` return, graph-capturable, no torch ABI anywhere.
2. **~30 Triton/Gluon kernels on the step path** (`engine/decode_lean.py`,
   `hc_sinkhorn/hc_ops`, `tools/fp8_linear.py`, `fp32_skinny.py`, `router_bf16.py`,
   `decode_attn_staged.py`, `packed_kv.py`, `expert_topp.py`, `head_packed_gluon.py`,
   plus the prefill set in `tools/fp4_moe.py`, `prefill_attn.py`, …): AOT-compile at
   build time to cubins, load with `cuModuleLoadData`. During bring-up, load the exact
   `.cubin` artifacts out of the Triton cache that the Python engine JIT-built, so the
   object code is bit-identical by construction.
3. **torch ops still on the step path**, each needing a Rust-side replacement:
   router `topk` + k-sum, indexer einsum/relu/topk/sort, fp32 cuBLAS `mm` (compressor,
   indexer), group softmax, LUT `index_select`, embedding gather, verify
   argmax/eq/cumprod/sum, draft argmax/softmax/Gumbel, NgramHashState (the checkpoint's
   own `inference/engram.py`, eager every step), engram row dequant. cuBLASLt for the
   GEMMs; the rest are small custom kernels with parity tests against torch output.

The one torch-coupled native piece is RoCE (`tools/roce/roce.cpp`: pybind + c10 stream +
ibverbs busy-spin proxy). It needs a de-torched shim (raw stream argument) — the kernel
(`roce.cu`) is already plain CUDA. There is **no exllamav3 dependency**; EXL3 is a
self-contained format implementation and is off in the DS profile.

Also Python-only today: tokenizer (`transformers` → the `tokenizers` Rust crate, parity-
checked on corpus), `image_processor` (vision, later), safetensors header parsing +
O_DIRECT expert reads (`engine/experts.py` → Rust, straightforward), the scale pack
reader (codec already ported), prefix persistence.

### The step today (what the Rust loop must beat)

Per verify step at width T ∈ {4, 6}: Gloo control broadcast → draft graph replay (or the
pre-queued `draft_ahead` result) → eager hash → async engram NVMe reads → **three graph
replays** ([L0], [L1–13], [L14–39 + head + draft-ring seed]) with engram H2D uploads at
the two boundaries → eager accept (argmax/eq/cumprod/sum, one 7-int readback) → host
rollback of compressor state → next draft pre-queued from device values. Under TP2,
~180 small all-gathers per step ride RoCE; the two engram boundaries are NVMe-bound
host waits. Prefill is eager (graphs off in the profile).

## Crates

| Crate | Status | Contents |
|---|---|---|
| `tidelm-core` | done | GPU-free logic, tests mirror the Python tests: sampling + penalties, batched spec verifier, residency planning (keep masks, demand blend, global plan), spec depth policies, UE8M0 scale codec, NumPy's PCG64 + the engram n-gram hash |
| `tidelm-text` | done | tokenizer (HF `tokenizers`, itself Rust) and the engram compressed token map — sha256-identical to the checkpoint's Python |
| `tidelm-kernels` | done | `fp4_moe_cuda.cu` + `engram_gather.cpp` built from their in-tree sources with the Python JIT's flags, plus **`cuda/ds41.cu`: the engine's own decode-path kernels** (skinny FP8/BF16/FP32 GEMMs, norms, HC, RoPE, router, attention, compressor, indexer, engram, graph-safe step plumbing); 18 GPU tests against CPU transcriptions of the Python |
| `tidelm-cuda` | done | streams, events, device/pinned buffers, async copies, graph capture/replay |
| `tidelm-model` | done | safetensors headers, O_DIRECT span reads (parallel aligned pieces), expert run grouping — byte-exact vs buffered IO on the real checkpoint |
| `tidelm-engine` | in progress | forward pass (streaming and resident), indexer, DSpark spec decode, rollback, one-graph steps; CLIs `tidelm-generate`, `tidelm-parity` |
| `tidelm-server` | later | HTTP front, same API as `server/app.py` |

Kernel strategy, as built: the Triton/Gluon kernels were **not** AOT-compiled. Every op
on the step path is a hand-written CUDA kernel in `ds41.cu` (the decode MoE stays the
repo's own `fp4_moe_cuda.cu`), so no Python is needed even at build time. Each reproduces
its Python counterpart's rounding points and differs only in summation order; the FP8
skinny GEMM streams 214-266 GB/s on the decode shapes against a 246 GB/s D2D copy
(`bench_skinny`), at least on par with the 211-230 GB/s the repo records for Triton.

## Milestones and gates

- **M1 — single-rank greedy decode, eager.** Weights + arena warm start, engram, all
  step math in Rust, no graphs yet. *Gate: layer parity against tools/v41_ref.py with the
  compressed-KV QDQ off on both sides (bf16-level per row), plus teacher-forced NLL with it
  on — a correctness gate, tok/s recorded but expected to lose to the graphed Python
  engine.* Token-for-token parity across implementations is NOT the gate: see below.

  Status 2026-10-24: the streaming engine (weights from NVMe each step, ~1 GB) passes
  layer parity — QDQ off, all 6 rows x 40 layers at 5e-3..3e-2 rel L2, logits corr
  0.99994, top-10 10/10 (rust/tools/ref_forward.py + compare_dumps.py). With QDQ on, the
  compressed latents that agree to 5e-3 come out of the FP4/E4M3 quantizer 5-13% apart:
  a 0.5% change in a group's amax flips its E4M3 scale (3-bit mantissa, 12.5% steps) and
  requantizes all 16 values, which cascades into router flips (50 vs 8). That is a property
  of the model's FP4 cache, not of either implementation — it applies to the Python engine
  against its own reference too — so cross-implementation token identity is unattainable
  at near-ties and must not be used as a gate.
- **M2 — graphed decode + speculation.** Graph capture of the step, DSpark draft +
  verify, depth policy live. *Gate: ≥ Python decode tok/s on the README workloads
  (code and prose separately — the depth policy history says they diverge), identical
  acceptance. A regression here blocks the branch.*

  Status 2026-10-24, measured on what fits next to a live server:
  - DSpark speculation is lossless in streaming mode on the full model — `def
    fibonacci(n):`, 24 tokens identical with and without speculation, 3.43 tokens/step
    (the Python engine records ~3.56). Exact, not approximate: every kernel is
    row-invariant.
  - The indexer lifts the context cap; the all-visible path is kept where it is exact.
  - Resident mode (weights, embedding, head, routable experts on the device; routing via
    a slot LUT) is bit-identical to streaming; with `--graphs` each backbone step is one
    CUDA graph, bit-identical to both (`tidelm-parity`, 4-layer truncation: 9 steps x
    129,280 logits). On that truncation graphs take 17.1 -> 15.3 ms/step.
  - Not yet measured: anything at full scale. The resident full model needs the box
    to itself (the GLM server holds 103 GB), and single-box residency needs a pruned keep
    set; the real deployment is TP2 (M4).
- **M3 — bigger graphs** (user goal). Today a step is 4 graph replays + eager
  embedding/hash + ~10 host-launched accept ops. Merge order, each step A/B'd on
  `boundary_gap_ms_per_step` and tok/s:
  1. accept/verify decision into the tail of segment 3 (keeps the single 7-int
     readback, deletes the host-launched tail),
  2. the greedy draft into segment 3 (EARLY_DRAFT already proves the dependency
     order works),
  3. embedding + ngram hash into segment 1 (hash inputs are the drafted block —
     available at graph launch),
  4. CUDA graph conditional nodes to collapse the T=4/T=6 graph pairs, halving
     capture count and warm-up.
  The L1/L14 engram boundaries stay: they wait on NVMe, and a graph cannot hide a
  host wait — the win there is prefetch depth, not capture width.
- **M4 — TP2.** NCCL from Rust, control message over NCCL (replacing Gloo), RoCE shim,
  full boot config guard (typed struct, broadcast, any differing field aborts — the
  repo's silent-pair-divergence rule). *Gate: two-rank token parity and tok/s vs the
  Python TP2 profile; the `bench_accept_ab_tp.py` workload.*
- **M5 — prefill.** Chunked eager prefill, Triton prefill MoE AOT, indexed attention,
  engram read-ahead, prefix restore. *Gate: prefill tok/s vs Python; secondary.*
- **M6 — concurrency** (user goal). Continuous batching: admission at step boundaries,
  per-lane KV/compressor/penalties/depth state, decode graphs keyed additionally by
  batch bucket {1, 2, 4, 8}, swaps only at scheduler boundaries (the
  `apply_global_swaps` rule). The Lane split happens in M1's data model so this stage
  is a scheduler, not a rewrite. *Gate: aggregate tok/s at N streams vs N sequential
  runs, plus p50/p95 per-stream latency.*
- **M7 — server + adaptation.** Port or front: decided by measurement, the pipeline is
  already Python-free by M2.

## Rules carried over from the Python engine

- **The pair fails silently.** Anything that changes numerics or control flow per rank
  goes in the boot guard; every new collective is reached by both ranks
  unconditionally. Core already breaks all ties deterministically (NumPy argsort order
  was unspecified; the Rust port fixes value-desc/id-asc and documents it).
- **Measure before claiming.** The A/B harness is the Python engine itself over HTTP vs
  the Rust binary on the same prompts; numbers go next to the change that produced
  them. `tools/generation_gate.py` remains the floor test for output sanity.
- **Kernel identity.** The boot guard carries source digests of the .cu files (embedded
  at build time) exactly as the Python guard does, so a Rust rank and a Python rank
  can never silently serve different kernels during the transition.
