# Plan: EXL3 routed experts with adaptive residency

Status (2026-10-08): P0-P5 done -- EXL3 is a serving profile (`.env.example.exl3`, packs from
`scripts/exl3-packs.sh`), decode 31.0 / 64.4 / 46.7 tok/s and prefill at FP4 parity (README).
P6 (two-tier arena) and exact-size slots for the 2-bit layers remain. Written for an agent working in this
repository; read `CLAUDE.md` and the
last `RESULTS.md` sections first.

## Progress (2026-10-08)

- **P0 packs**: built on node 0, `~/models/exl3-packs/exl3-experts-r{0,1}of2.bin`, 98.24 GB each,
  source manifest sha256 `464d2dc4d3edbc48`; rank 1 copied to `spark2`. Header sha is per rank.
- **P1 done** (commit `EXL3 P1`): `tools/exl3_format.py` (vendored oracle), `tools/exl3_ref.py`,
  `tools/pack_exl3_experts.py`, `tools/test_exl3_ref.py`. unpack is bit-exact vs the oracle on
  layers 0/18/39; rank slices exact; pack round-trips.
- **P2 core done**: `tools/exl3_moe.py` (`Exl3Arena` + reference MoE, TP gather path) and
  `tools/exl3_store.py` (pack reader behind `ExpertStore`'s own LRU/ring/lease), with
  `tools/test_exl3_moe.py` and `tools/test_exl3_store.py`. Engine wiring is in and default-off:
  `EXPERT_FORMAT=exl3`, boot-guard pack sha, health, argparse, `.env.example`.
- **P2 proven end to end (2026-10-08)**: the pair booted with `EXPERT_FORMAT=exl3` (packs on both
  nodes), warm-started **9,400 experts / 60.1 GB read in 12 s (~5 GB/s)** into 13,522 slots (88 %
  of all routed experts), and served a 2-token completion. The reference MoE is the serving path,
  so that request took ~76 s -- a correctness arm, not a speed one.
- **Adaptive residency works on the exl3 arm (2026-10-08)**: global cross-layer swaps
  (`DSV41_DYNAMIC_EXPERTS`, `DSV41_RESIDENT_EXPERTS`) and predictive prefill (`mode=shadow`) run
  over the pack. A request adapted **205 sector transfers across layers (196 cross-layer) in
  0.27 s**, residents 204-273 per layer, swapping EXL3 experts between the arena and NVMe. The
  swap path needed a TP fix (there is no EP null slot, so a swapped-out expert gets a -1
  placeholder it can never be routed to) and the earlier auto-disable had to be removed. Its
  *speed* is still unmeasurable on the reference MoE; the feature is present and correct now.
- **P3 done and serving (2026-10-08)**: JIT CUDA decode kernel (`tools/exl3_moe_cuda.{cu,py}`),
  ported from TF's ExLlamaV3-derived kernels. vs the FP4 Triton kernel: 0.97x at T=1, 0.57-0.87x
  above it; vs the fp64 reference 575-931 ms. TP2, mixed 3-bit/2-bit slots, CUDA-graph capturable.
- **P4 done (2026-10-08)**: same kernels with `build_routing` grouping (BM=64, slots compacted
  first -- without compaction a 2,530-token prefill took 62 s; compacted it is 6.8 s). Prefill 866
  tok/s vs FP4's ~1,423 (0.61x); T=2048 is 58 ms/layer against FP4's 34.6. A dedicated prefill GEMM
  is owed for parity.
- **Decode is at FP4 parity and not MoE-bound.** Same prompts, same harness: prose 23.83 (EXL3) vs
  23.78 (FP4); the step accounting is moe_compute 4.7%, engram 25%, engram_wait 27%, attn 1.2%,
  unaccounted 40%. The 35 tok/s prose goal is an engine-wide Engram/draft target, not an EXL3 one.
- **Re-enabled on exl3 (graphs are back)**: variable spec depth, predictive prefill, adaptive
  residency. Still FP4-only: critical prefill, layer streaming, the `_moe_merged` fusions.
- **P7 done, off by default (2026-10-11)**: the three DSpark draft blocks' attention + shared-expert
  matrices from their own 4-bit pack, `DSV41_EXL3_DRAFT_DENSE` — `tools/pack_exl3_draft.py`,
  `engine/exl3_draft.py`, `tools/test_exl3_draft.py`. Two paired 24-prompt gates pool to **−1.05 ± 0.37 ms/step**,
  24/24 bit-identical outputs, acceptance unchanged. The draft graph's span went 7.05 → 5.66 ms a
  pass, from 10.40 on 2026-10-07. Measured and rejected there: the drafter's two *heads* at fp4
  (+0.29 ± 1.03 ms) — they read at the box's plain-read ceiling, so a smaller format buys nothing.
  See `docs/exl3-draft.md`.
- **Decode tuning (2026-10-08, RESULTS.md)**: EXL3's routed MoE joined the merged decode collectives
  (`DSV41_EXL3_MERGED`, exact, ~-2 ms/step), the grouping kernel went parallel (48 -> 11 us a
  layer), draft-ahead + early verify (exact, -2.03 +- 0.49 ms/step paired), and the live resident
  budget was re-derived for EXL3 slots (9,574 -> 13,500, no speed cost). Live prose/HTML/code
  29-31 / 61-63 / 45 tok/s (FP4: 26.0 / 51.5 / 38.7). The routed kernel runs at 185-213 GB/s
  L2-cold, ~the plain-read ceiling. EXL3 dense attention + shared experts were built, measured and
  turned off: -1.3 ms/step but -0.07 accepted tokens/step, net zero, top-1 0.976 vs FP8.
- **Prefill at FP4 parity (2026-10-08)**: `exl3m_prefill` (one trellis decode per tile for a
  block's rows, warps split N) + the static per-layer routes FP4 uses. Same 9 prompts, same
  harness: EXL3 5.103 s vs FP4 5.061 s; live 8K 1,533 tok/s, 32K 1,528 (FP4 live 1,423-1,471).
- **Pending, needs serving stopped + both nodes**: nothing for P2. P3/P4 kernels and the paired
  A/B measurements remain. Measure kernels with the `tools/bench_*` microbenchmarks, not by
  rebooting the engine; `tools/generation_gate.py` is a floor test only.

## Goal

Add `EXPERT_FORMAT=exl3`: the **routed experts** (MoE layers 0–39, 384 each) come from the local
EXL3 pack instead of the native MXFP4 checkpoint. Everything else stays as it is: native FP8
dense/attention, BF16 head, Engram, the native DSpark drafter, native shared experts, vision,
prefix cache, adaptive residency, TP2.

Why: an EXL3 expert is ~6.7 MB per rank (4.45 MB in the 2-bit layers), against 9.4 MB for FP4.
The same 90.2 GB arena then holds 88–92% of all routed experts instead of 62.5%, and ~98–102 GB
holds all of them. Adaptive
loading still covers whatever does not fit, so memory stays a dial (TTS on, smaller arena; TTS
off, everything resident).

**Performance is the point.**  The target is TensorFold-parity throughput, not just the
coverage/miss-rate win.  Two standing consequences: (1) the pack layout and arena slots are
designed for the fast grouped EXL3 kernels (contiguous per-rank trellis tiles, no reshape at
load), and the reference MoE is a correctness gate only; (2) the kernel port prioritises TF's
ExLlamaV3-derived `decode.cuh` + `experts_grouped.cuh` (+ `experts.cu`, all permissively
licensed).  The one fast piece that must **not** be copied is `x3ld.cu`/`loads.py`
(GLM-patch-0580 lineage, AGPL risk): its load-path scheduling is ours to write, and it is the
main known performance risk versus TensorFold.

Out of scope: EXL3 dense/attention/head; a new TP layout; changing decode/speculation.
Optional final phase: a two-tier arena (hot experts native FP4, the rest EXL3).

## Verified facts (2026-10-08)

Checkpoint: `/home/ryan/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (**node 0 only**; node 1 does
not have it). exllamav3 v1.4.2, codebook `mul1`, `out_scales=always`, avg 2.90 bpw (`--hq`),
quantized from the **base** `deepseek-ai/DeepSeek-V4.1-Flash` (README, MIT). The abliteration
overlay (`DSV41_ABLIT_WOB`) touches only attention, so it still composes.

TensorFold's prepared packs (`~/models/dsv41-tensorfold/prepared/model/*/rank*/data.bin`, 101.8 GB
each, `X3Stack`/`mul1`, exactly the per-rank slice we want) belong to **its own**
`dsv41-uncensored-2.9bpw`, not to this base checkpoint. Provenance was settled 2026-10-08: TF's
`engine/kernels/exl3/experts.py` docstring names that checkpoint, and the manifest's source key
is the base `config.json` sha256 only because the uncensored overlay changes attention alone. So
build our own pack from this directory; do not point the engine at TF's.

Measured 2026-10-08: the base pack is **98.24 GB per rank** + 0.87 MB header, **6.670 MB** per
3-bit expert and **4.458 MB** per 2-bit expert at the TP-output slice. Node 0's root had 723 GB
free at that point (it was 141 GB earlier in the day -- re-check before writing a pack). Verify
provenance from this directory's `config.json`/README, not TensorFold's docs. The HF-cache entry
`models--Mia-AiLab--...` is an empty stub; this directory is the only copy of the base pack.

Per routed expert, per projection (`layers.L.ffn.experts.E.{w1,w3,w2}`):

| Tensor | w1 / w3 (gate/up, 5120 → 2304) | w2 (down, 2304 → 5120) |
| --- | --- | --- |
| `trellis` int16 | [320, 144, 16·bits] (K tiles, N tiles, words) | [144, 320, 16·bits] |
| `suh` fp16 (input scale) | [5120] | [2304] |
| `svh` fp16 (output scale) | [2304] | [5120] |
| `mul1` int32 scalar | codebook marker 0x83DCD12D | same |

Bits: 3 for layers 0–17 and 23–39, **2 for layers 18–22**. A 3-bit expert is 13.3 MB in full,
6.65 MB per rank; a 2-bit one 8.9 / 4.45 MB. All routed experts at TP2: ~98 GB per rank.
(The pack's dense layers, router and DSpark are EXL3 or fp16 too; we use none of them.)

Memory per rank, routed experts only:

| Arena | FP4 today | EXL3, 3-bit-sized slots (v1) | EXL3, per-bit-width pools |
| --- | ---: | ---: | ---: |
| 90.2 GB (TTS-sized profile) | 9,574 (62.5%) | ~13,560 (88%) | ~14,100 (92%) |
| ~98 GB (TTS off) | ~10,400 (68%) | ~14,700 (96%) | 15,360 (100%) |
| ~102 GB | — | 15,360 (100%) | 15,360 (100%) |

Our TP layout is `DSV41_TP_EXPERT_LAYOUT=output`. Each rank holds w1/w3 output columns for half
the intermediate width (1,152). The full intermediate activation is all-gathered, and w2 produces
half of the 5,120 outputs. EXL3's Hadamard rotations act on **blocks of 128** along K and N, and
1,152 = 9 × 128 and 2,560 = 20 × 128. So a rank's matrix is an exact slice:

- w1/w3: trellis N tiles `[rank*72, +72)`, `svh[rank*1152 : +1152]`, `suh` whole;
- w2: trellis N tiles `[rank*160, +160)`, `svh[rank*2560 : +2560]`, `suh` whole (K = 2,304 full).

TensorFold uses the *intermediate* layout instead (w2 split on K, fp32 partials summed); its
`engine/kernels/exl3/experts.py` docstring states the split rule and its test checks it.

## P3/P4 targets (measured 2026-10-08, serving image)

`tools/bench_exl3_decode.py`, one layer, TP-world-1 shapes, real EXL3 experts, synthetic routing:

| T | P | FP4 Triton kernel | EXL3 reference |
| --- | --- | ---: | ---: |
| 1 | 6 | **0.444 ms** | 584.24 ms |
| 6 | 36 | 1.101 ms | 932.83 ms |
| 128 | 768 | 1.962 ms | 950.10 ms |
| 512 | 3072 | 6.747 ms | 943.76 ms |
| 2048 | 12288 | **38.340 ms** | 986.78 ms |

An EXL3 slot is 0.708x an FP4 slot at world 1 (13.32 vs 18.80 MB). The reference is flat ~950 ms
because it materializes full fp64 weight matrices (`dequant_slot` = 115 ms/expert), which the
fused kernel must not do. Bandwidth-scaled, prefill should land near **~27 ms** at T=2048; decode
near the FP4 **0.44 ms** at T=1. Those are the P3/P4 bars. Re-run these probes and the engine's
own `DSV41_PREFILL_MOE_TIMING` up/down envelopes once the kernel exists.

## Reference material and licenses

- **Bit-exact numpy reference:** `~/git/deepseek-v41-tensorfold-spark/vendor/TensorFold/src/tensorfold/cuda/exl3/format.py`
  (`codebook`, `stream_ends`/`unpack`, `rotate`, `dequantize`, `forward`), plus
  `vendor/TensorFold/docs/recipes/exl3.md` for the bit arithmetic. The weight is
  `diag(suh) · H_K · W_q · H_N · diag(svh)`; the kernels rotate the input, multiply by `W_q`, then
  rotate the output.
- **Grouped CUDA expert kernels:** `vendor/TensorFold/src/tensorfold/cuda/exl3/{experts.cu,
  experts_grouped.cuh, decode.cuh, experts_cb*.cu, experts_prefill.cu, experts.cpp}`, and the
  DSV41 TP assembly in `~/git/deepseek-v41-tensorfold-spark/engine/kernels/exl3/{experts.py,
  loads.py, x3ld.cu, prefill.py}`.
- **Licenses:** TensorFold is Apache-2.0 from 0.6.0 (MIT before); exllamav3 is MIT (Turboderp);
  the TF DSV41 repo is Apache-2.0. Keep their SPDX/copyright headers on anything adapted, and add a
  NOTICE / third-party entry. **Do not copy** code whose lineage is the Mia's AI Lab GLM kit after
  2026-09-07 (AGPL-3.0). TF's `NOTICE` names the "fat" expert kernels in `exl3_fast.cu` as adapted
  from that kit; check each file's header before porting.

## Design

### 1. Pack tool (offline): `tools/pack_exl3_experts.py`

Write one file per rank, e.g. `exl3-experts-r{rank}of2.bin` beside the model or on the data
volume. It holds a JSON header (layer → bits, `(layer, expert)` → byte offset, source SHA-256) and
each expert's **rank slice stored contiguously**: w1, w3, w2 trellis slices, then suh/svh. Records
are 4096-aligned so `O_DIRECT` loads one expert in one `preadv` (`engine/experts.py` already does
this for FP4 via `ShardFile` spans). Never `mmap` the pack in serving: page-faulted reads stalled
the GPU through reclaim/compaction (2026-10-08).

Disk (2026-10-08, after cleanup): node 0 has ~470 GB free, node 1 ~1 TB; a rank pack is ~98 GB.
Check space again before writing. Node 1 needs either the source checkpoint copied
over (fast over the 200 Gb link) or the rank-1 pack written on node 0 and copied.

### 2. Arena: `Exl3Arena` (`tools/exl3_moe.py`)

Interface parallel to `tools/fp4_moe.py::ExpertArena`: `slots`, `load_slot(slot, ...)`,
`bytes_per_slot`, plus a zero **null slot** (non-owned / non-resident pairs must contribute exactly
0, as `NULL_SLOT` does for FP4). Version 1 sizes every slot for 3 bits; a 2-bit expert wastes a
third of its slot (≤ ~4 GB if layers 18–22 are fully resident). Per-bit-width pools can come
later. Slot tensors: `t1, t3` [S, 320, 72, 48]; `t2` [S, 144, 160, 48]; `suh1, suh3` [S, 5120];
`svh1, svh3` [S, 1152]; `suh2` [S, 2304]; `svh2` [S, 2560] (fp16); `bits` [S] int8.

### 3. Store and residency (`engine/experts.py`, `engine/v41_engine.py`)

Leave `ExpertStore` logic alone: LRU, transient ring, slot LUT, compact prefill routes, global
residency and sectors, swaps, urgent loads. Add an EXL3 reader that maps `(layer, expert)` to a
pack record, and make `EXPERT_BYTES` per format. In `V41Engine`, extend
`expert_format in ("fp4", "cb3")` to accept `exl3`. Build the arena (TP-wired, unlike CB3, which
is single-node), and dispatch `moe_fn` on the arena type as the CB3 branch does. Keep the DSpark
draft arena FP4. Disable FP4-only paths for this format: `_moe_merged` in `engine/fastdecode.py`
(native FP4 + shared-expert fusion), prefill replicas, and `DSV41_FP4_PREFILL_DOT_SCALED`.

Boot guard (`V41Engine` cfg dict): `expert_format`, the pack's header SHA-256, the bits map,
kernel choices. Health: the same fields.

### 4. MoE compute

Same contract as `fp4_moe.moe_forward`: x bf16 [T, 5120]; slots/weights [T, 6]; routing via
`build_routing` (block_slot, block_pair); fp32 out, combined exactly where the FP4 path combines.
SwiGLU limit 10, routing weight applied as today. Reuse the output-layout gathers.

**a. Reference path first** (`moe_forward_exl3_ref`): decode the touched experts' trellis to
fp16 (port `format.py` to torch/Triton), apply the rotations, then a BF16 grouped matmul. It's slow,
but establishes correctness end to end, the way `engine/moe_fallback.py` did for FP4.

**b. Decode kernel** (P ≤ 64 pairs; used inside CUDA graphs). Per (expert block, N tile): decode
the 16×16 trellis tiles in registers. For mul1: take the 16-bit window at each value's
`stream_ends` position, multiply by 0x83DCD12D, byte-sum, then apply the fp16 scale and bias.
Multiply with the input rotated `x' = H128(x ⊙ suh)` (per matrix: w1 and w3 have different `suh`),
then rotate the output `H128(y') ⊙ svh`. Static shapes, no host sync, graph-capturable.

**c. Prefill kernel** (P > 64): grouped GEMM over `build_routing` blocks. Dequantize the tile
to fp16/bf16 in registers, MMA with fp32 accumulation, rotations as above.

Route choice: first try porting TensorFold's `experts.cu`/`experts_grouped.cuh` as a torch CUDA
extension built into the image (as `fp4_moe_cuda` is), adapting its call interface to our
slot/routing tensors. Fall back to Triton if the port fights the build. Either way, data-dependent
sizes (tokens, pairs, resident counts) must be **runtime arguments with `do_not_specialize`**,
never `tl.constexpr` (2026-10-08). Keep row-invariant arithmetic (a row's result
must not depend on the chunk), so the prefix cache's resume == cold invariant holds. Rank partials
are added in rank order.

### 5. Optional: two-tier arena

FP4 pool for the hottest experts (native precision where it matters most), EXL3 pool for the
next tier, NVMe/skip for the rest. The residency planner ranks by demand, as now, and assigns a
tier. A MoE call splits pairs by tier: two launches, summed in fixed order. Do this only after
pure EXL3 is measured.

## Phases and acceptance criteria

| Phase | Deliverable | Done when |
| --- | --- | --- |
| P0 | Checkpoint on both nodes, disk plan, NOTICE entry | Both nodes read the same SHA-256 source |
| P1 | Torch reference decoder + pack tool + unit tests | Dequantized fp16 values equal `format.py` bit for bit on real tensors from layers 0, 18 (2-bit) and 39; each rank slice equals the slice of the full decode; pack round-trips |
| P2 | `Exl3Arena`, store reader, `EXPERT_FORMAT=exl3`, reference MoE | Both ranks boot, the guard agrees, generation runs; reference-path MoE matches `format.forward` to fp32 tolerance on random inputs |
| P3 | Decode kernel in graphs | Matches the reference path (stated tolerance) at T = 1–6; decode A/B done (below) |
| P4 | Prefill kernel | Matches the reference; row-invariance unit test; prefill A/B done |
| P5 | Serving profile, `.env.example`, README/RESULTS | Live numbers recorded; rollback is `EXPERT_FORMAT=fp4` |
| P6 (optional) | Two-tier arena | Measured against pure EXL3 and pure FP4 |

## Measurement protocol

Follow the practice in RESULTS.md 2026-10-08:

- Use disposable two-node gates (`results/*/run_gate.sh` pattern, serving stopped), one engine
  process, paired arms with rotated order, at least 9 prompts (random 8K + natural ~8K), mean ±
  standard error. Drivers: `tools/bench_prefill_trace_tp.py` (`--arms`, logit capture),
  `tools/bench_accept_ab_tp.py` (22-prompt decode acceptance/ms-per-step), and
  `tools/bench_prefill_expert_tp.py` with `tools/prefill_expert_probes.py` (36 objective probes).
- EXL3 vs FP4 is a different-weights comparison, not a kernel-order one. Report logit KL and top-1
  agreement against FP4, alongside today's chunk-size noise floor (mean KL ~0.03–0.04), plus the
  probes. Report **routed-miss rate**: the point of EXL3 is fewer misses. Compare at the same
  memory budget (EXL3 ~92% resident vs FP4 62.5% at 90.2 GB) and at full residency.
- Also run serving-like (`DSV41_BENCH_SERVING_LIKE=1`, swaps on, 512K allocation), then live HTTP
  `bench/bench.py`: random 8K/32K prefill and prose/HTML/code decode. Frozen-map gates hid two
  serving-only costs today (compiles, adaptation).
- `tools/generation_gate.py` is a floor test only.

## Pitfalls already paid for

- A `tl.constexpr` that follows request shape compiles inside requests. Count variants per kernel
  name in `.triton-cache`, and compile the bounded ones at boot (`fp4_moe.warm_routing`).
- Page-cache churn stalls GPU kernels on this box: `O_DIRECT` for anything streamed, no mmap.
- Both ranks must reach every collective; anything numeric goes in the boot guard.
- One prompt never decides an arithmetic question; use the paired many-prompt drivers.
- Unified memory: the arena competes with everything else (`KEEP_FREE_GB`, the 2.5 GB
  MemAvailable watchdog, adaptive prefill rows). Size the EXL3 arena with them in mind.
- `V41Engine` must be constructed outside an outer `torch.inference_mode()` (arena loader
  threads).

## Files

New: `tools/pack_exl3_experts.py`, `tools/exl3_moe.py` (+ CUDA extension sources if ported),
`tools/test_exl3_*.py`. Changed: `engine/experts.py` (reader, bytes per format),
`engine/v41_engine.py` (format, arena, dispatch, guard, health), `engine/fastdecode.py` (merged
path gating), `engine/model.py` (only if dispatch needs it), `.env.example`, docs.

## Decisions for the owner before P5

- Arena size for the EXL3 profile: 90.2 GB (TTS-compatible) or ~98 GB (TTS off, all resident).
- Pure EXL3 first, or go straight to the two-tier arena.
- Whether quality at matched memory (EXL3 92% vs FP4 62.5% resident) is the deciding comparison.
