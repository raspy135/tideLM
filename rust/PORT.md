# Rust port of the DS v4.1 and GLM-5.3 engines

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
| `tidelm-engine` | in progress | forward pass (streaming and resident), indexer, DSpark spec decode, rollback, one-graph steps, lanes + continuous batching; CLIs `tidelm-generate`, `tidelm-parity`, `tidelm-batch` |
| `tidelm-server` | in progress | HTTP front over the online scheduler: `server/app.py`'s chat routes for text messages (`/v1/chat/completions` streamed or not, `/v1/models`, `/health`, `/v1/debug/prompt`); `tidelm_text::chat` is the V4.1 prompt encoder for text turns |

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
  - Full scale, 2026-10-24, one box (GLM server stopped): all 40 layers resident with the
    4,600 highest-demand experts (30%, `--keep-db` over the prune demand database, the rest
    pruned), greedy, graphs, bench.py's prose and code prompts, 256 tokens, every arm
    token-identical to plain greedy (`tidelm-generate --sweep-depth`):

    | arm | prose tok/s | code tok/s | ms/step |
    |---|---:|---:|---:|
    | plain | 13.4-13.6 | 13.4-13.5 | 74 |
    | fixed depth 3 | 16.7-17.5 | 18.4-19.2 | 124-130 |
    | fixed depth 5 | 14.6-15.3 | 17.0-17.5 | 146-153 |
    | window 3/5 (the Python default) | 17.0 | 19.1-19.2 | 127 |
    | confidence, any width 1..5 | 18.1-18.2 | 18.9-19.1 | 108-121 |

    A verify row costs 8-16 ms (the experts it routes to); the engram overlap took 9% off a
    speculative step. Not comparable to the README rows (two boxes, 68% resident, hot reads,
    sampling): that comparison needs the pair.
  - Depth: the oracle verify depth is +7-23% over the best fixed depth on the retained
    confidence traces; DSpark's head gets about a third of it, and neither online
    recalibration nor acceptance history adds anything (offline replays). Beyond that needs
    a target-side signal (e.g. progressive verification: decide after part of the stack).
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
- **M4 — TP2.** Status 2026-10-24: tensor parallel runs, bit-identical to one box. The Python
  profile's output layout (DSV41_TP_LINEAR_LAYOUT=output): attention heads split (wq_b rows,
  sinks, wo_a groups), wo_b, the shared expert, every routed expert (w1/w3 intermediate rows,
  w2 output rows - each box holds half of every resident expert), the engram projection and
  the vocab head by output rows; every sharded output is all-gathered whole, so each element is
  the one-box dot product and nothing is summed across the pair. Four gathers a layer (wo_a's
  groups, the attention output, the shared + routed intermediates in one, the MoE output),
  about 163 a step, on the RoCE one-shot gather (`tidelm_comm::roce`: tools/roce/roce.cu as is,
  its proxy torch-free behind a C ABI; 12-25 us a gather inside a step graph against ~150 us
  for an NCCL all-reduce there, examples/pair_step.rs). The small projections, routing and
  caches are computed identically on both ranks; the drafter runs whole on each. `tidelm-comm`
  also links the engine image's NCCL 2.28.3 (~/.local/nccl) for the bring-up handshake. Rank 0
  mirrors every engine call to rank 1 over the control channel after a byte-for-byte boot
  config check. 4-layer truncation: the pair's logits digests equal one box's, plain and
  speculative. (An expert-parallel first cut ran before this and was replaced: its per-layer
  all-reduce cost ~150 us on NCCL and its ranks waited on whichever held the hot experts.)

  Full model on the pair, 10,000 demand-ranked experts resident (65%, the rest pruned),
  greedy, graphs, bench.py's prompts, 256 tokens, after the small-kernel and split-key
  attention work (every arm token-identical to plain greedy):

  | arm | prose tok/s | code tok/s |
  |---|---:|---:|
  | plain | 23.2 (43.2 ms) | 19.9 (50.2 ms) |
  | confidence 1..5 | 26.1-27.1 | 38.9-39.9 |
  | window 3/5 (Python's default) | 25.9-26.7 | 38.2-39.2 |
  | depth 3 | 26.7 | 39.2 |

  The README protocol itself (bench/bench.py: the same "[req N]" prompts as the README runs,
  1,024 tokens with EOS ignored, temperature 0.6 / top_p 0.95, decode rate after the first
  token, median of three after one warmup; `tidelm-generate --bench prose,code`), same pair
  and resident set, 2026-10-24:

  | arm | prose tok/s (runs) | code tok/s (runs) |
  |---|---:|---:|
  | Python, README row (2026-10-22) | 25.59 (26.40 / 25.59 / 25.39) | 43.31 (43.31 / 42.31 / 43.79) |
  | window 3/5, Python's policy | 26.27 (29.21 / 26.27 / 25.44) | 41.44 (40.07 / 45.30 / 41.44) |
  | confidence 1..5 | 28.90 (28.72 / 29.16 / 28.90) | 42.66 (42.66 / 43.28 / 41.13) |
  | window 3/5, drafter split + listed | 28.10 (31.09 / 28.10 / 27.57) | 48.58 (48.58 / 48.83 / 47.11) |
  | confidence, drafter split + listed | **31.92** (31.92 / 30.55 / 33.02) | **49.73** (49.26 / 49.80 / 49.73) |

  The drafter rows are the same prompts and seed: prose run 2 under the window policy took the
  same 524 steps and 499 accepts in both (its sampled text unchanged) and went 26.27 -> 28.10.

  A verify step is shorter than Python's (code at depth 3-5: ~74 ms against ~88 implied by
  43.3 tok/s at 3.83 tokens/step), but on code it accepts less: 3.1-3.6 tokens a step against
  Python's 3.7-3.9. The two differ in what the target is: Python's 10,452 residents adapt to
  each request (its warmup run re-ranks them) and its decode drops the low-share routed picks
  (DSV41_EXPERT_TOPP=0.90); here 10,000 demand-ranked experts are fixed at boot. Python's
  drafter is also cheaper per step (see M7's list).
  Original plan: NCCL from Rust, control message over NCCL (replacing Gloo), RoCE shim,
  full boot config guard (typed struct, broadcast, any differing field aborts — the
  repo's silent-pair-divergence rule). *Gate: two-rank token parity and tok/s vs the
  Python TP2 profile; the `bench_accept_ab_tp.py` workload.*
- **M5 — prefill.** Chunked eager prefill, Triton prefill MoE AOT, indexed attention,
  engram read-ahead, prefix restore. *Gate: prefill tok/s vs Python; secondary.*

  Status 2026-10-24: a step with no decoding lane takes up to TIDELM_PREFILL_ROWS = 2,048
  prompt rows, run eager (TMAX = 64 still bounds graphed steps and head rows). The 8K-token
  first token on the pair, bench.py's random8k workload (same labels and seed as the README
  row, median of three after a warmup), as each piece went in:

  | change | 8K first token |
  |---|---:|
  | 64-row steps (M7's row below) | 27.3 s |
  | 2,048-row steps: M-tiled dense GEMMs, ds_route_large, one-pass prompt-row attention | 12.76 s |
  | NCCL all-gather for an eager step's > 1 MB shards (the RoCE one-shot moved them at ~5 GB/s) | 12.41 s |
  | grouped-GEMM routed experts (64-pair blocks, FP4 decoded to bf16 in shared memory) | 10.20 s |
  | HC mixes in one FMA launch, tensor-core indexer scores, engram next-chunk prefetch | 8.24 s |
  | grouped GEMM's weights fetched 64 bytes a row, 32-head attention blocks, key prefetch | **6.76 s** (6.71 / 6.76 / 6.76) |
  | Python, README row | 7.3 s |

  Decode after the prompt is unchanged throughout (66-68 ms a step in every run). Rank 0's
  GPU time at the last row was 96.7% busy. The bounds that set the numbers: the grouped GEMM
  streams FP4 weights (2.4 GB a layer a step), and 32-byte runs a row read at 124 GB/s on
  this bus against 215-228 for 64-128 (a read-only walk of the same layout), so it fetches
  two K steps at once (19.4 -> 13.9 ms a layer at 2,048 rows; its tensor-core work alone is
  6.7 ms); attention's gathered keys were loaded between barriers, one block an SM (14.7 ->
  6.87 ms with 32 heads a block and the next tile's keys in registers; Python's kernel 7.35).

  Numerics: a prompt row's bits no longer match a 64-row step's once its step reaches 512 rows
  (the grouped GEMM's dots are bf16 x bf16, the decode kernels' f16 x f16) or runs eager (the
  indexer's per-head dots accumulate in the mma's order; its head sum keeps warp_sum's tree).
  Every prefill kernel is pair-invariant and the pair still equals one box bit for bit
  (4 layers, 2,048 + 1,211 rows). Two facts in a 9,019-token prompt (start, middle) come back
  exactly. Not comparable in one respect: Python's prefill can read missing experts (hot
  reads); here the 10,000 residents are all there is, so no prefill pays for a miss.
- **M6 — concurrency** (user goal). Continuous batching: admission at step boundaries,
  per-lane KV/compressor/penalties/depth state, decode graphs keyed additionally by
  batch bucket {1, 2, 4, 8}, swaps only at scheduler boundaries (the
  `apply_global_swaps` rule). The Lane split happens in M1's data model so this stage
  is a scheduler, not a rewrite. *Gate: aggregate tok/s at N streams vs N sequential
  runs, plus p50/p95 per-stream latency.*

  Status 2026-10-24: lanes and the scheduler are in (`forward_batch`, `decode::generate_batch`).
  Every cache is lane-major; a step stacks up to 64 rows from any lanes, padded to a row
  bucket (1, 2, 4, 8, 16, then multiples of 8) with filler rows in a scratch lane that route
  to no expert, so graphs are keyed by bucket, not by batch composition. Decode rows go
  first, prompt chunks fill the rest (chunked prefill); drafts, rollbacks and seeding are
  batched across lanes. Measured (`tidelm-batch --check`, 4-layer 16-expert truncation):
  - batched == alone bit for bit while steps stay at <= 16 rows: 8 requests over 4 lanes
    plain, 6 over 2 lanes speculative (eager and graphs) — identical logits digests.
  - above 16 rows the GEMMs take the tensor-core kernel, which rounds differently: prefill
    logits move by 7e-3..2.3e-2 rel L2 with the FP4 cache QDQ off, but by up to 1.4e-1
    with it on (`tidelm-parity --mma`; the QDQ amplification of M1), enough to flip a 0.375
    logit margin on a truncated model. Not batch-invariant past 16 rows, like other engines.
  - aggregate tok/s, resident + graphs, warm: plain 67 (1 lane) -> 184 (4) -> 244 (8) ->
    293 (16); speculative 27 -> 39 (2) -> 51 (4) -> 34 (8). The 8-lane speculative loss is
    the bm16 MoE kernels going compute-bound (7.2 + 5.8 ms per layer at 384 pairs, ~52 GB/s
    effective): they run each pair on CUDA cores, and the Python engine leaves them for
    tensor-core kernels above 64 pairs. A tensor-core FP4 MoE for > 16-row steps is next.
- **M7 — server + adaptation.** Port or front: decided by measurement, the pipeline is
  already Python-free by M2.

  Status 2026-10-24: `tidelm-server` serves the engine over HTTP with no Python
  (`decode::Scheduler`: continuous batching that takes requests at any time, a depth policy
  per lane, tokens streamed per step; `generate_batch` now runs on it, and every batch check
  still passes). bench/bench.py run unchanged against it, on the pair (10,000 experts
  resident, drafter split and listed, confidence policy, one lane - Python's
  MAX_CONCURRENCY=1), with the README runs' labels and seed:

  | workload | Rust over HTTP (runs) | Python README row |
  |---|---:|---:|
  | prose, decode | **31.04** tok/s (31.04 / 30.42 / 31.23) | 25.59 |
  | code, decode | **51.03** tok/s (51.03 / 49.47 / 51.16) | 43.31 |
  | prose / code, first token | 0.22 / 0.24 s | 1.22 / 1.25 s |
  | 8K-token prompt, first token | **27.3 s** (27.36 / 27.24 / 27.28) | 7.3 s |
  | 8K-token prompt, decode | 38.6 tok/s | 32.65 |

  The 8K first token is 3.7x slower: prefill steps carry at most TMAX = 64 rows, so an 8K
  prompt is 128 steps and each re-reads every dense weight and every expert its rows touch.
  Python prefills 2,048 rows a chunk.

  Sampling is in (2026-10-24): `ds_nucleus` computes each row's temperature + top-p nucleus on
  the device (a log-probability histogram picks a candidate window, which is sorted and cut by
  the `cum - p < top_p` rule; 39-42 us a row, rows in parallel, bench_nucleus), and only the
  nucleus comes back; a nucleus over 2,048 tokens is recomputed on the host (none in the README
  runs). The drafter draws inside its graph from its own nucleus at DRAFT_NUCLEUS_K = 512
  (fastdecode.py's DSV41_DRAFT_NUCLEUS), with uniforms from the host, and verification is
  `tidelm_core::spec::verify_sparse` - verify_probabilities' decisions bit for bit over sparse
  rows. Every sampled speculative run checks itself: the acceptances it saw against the sum of
  min(p, q) over the positions it tested agree within a standard deviation in every run so far.
  Batched sampled requests draw from per-lane streams and match their alone runs token for token
  on the FMA path.

  The drafter is split over the pair and lists 81,920 ids (2026-10-24): its blocks shard like
  the backbone's (heads, wo_a groups, wo_b, the shared and routed experts by output rows, every
  output gathered whole), and its FP8 head and markov bias head score only the ids of the
  Python profile's DSV41_DRAFT_VOCAB list (the v1 ranking, built in; TIDELM_DRAFT_VOCAB=all
  turns it off), each rank its contiguous half. The pair's drafts stay bit-identical to one
  box's. A draft step on the pair (tidelm-batch, 4-layer backbone under the whole drafter,
  depth 5):

  | drafter | all ids | listed |
  |---|---:|---:|
  | whole on each rank | 8.9 ms | 7.4 ms |
  | split (TIDELM_TP_DRAFT, default) | 5.7 ms | **4.7 ms** |

  Sampled drafts: 9.1 -> 4.8 ms. Python's drafter, split and listed, is 4.58 ms a pass in its
  traces.

  DSV41_EXPERT_TOPP is ported as TIDELM_EXPERT_TOPP (decode rows keep their routed picks up to
  that share of their weight, at least 3; a dropped pick routes to the arena's null slot, which
  reads no weights and adds an exact 0), off by default because it did not pay: at 0.90 the
  README protocol's medians moved +1.6 / -1.0% on prose and -2.1 / -3.4% on code (window /
  confidence), and code's acceptance fell from 3.79 / 3.72 / 3.37 to 3.45 / 3.61 / 3.29 tokens
  a step under the window policy - the drafter predicts the unpruned target.

  Adaptive expert loading is in (2026-10-24), every stage the serving profile enables, with its
  settings (`tidelm-server --adapt-db HISTORY.npz --trace coverage.json --budget N`; rank 0 reads
  the history and sends rank 1 the boot residents during the join):
  - demand: each backbone router also takes its unmasked top-6 and records per lane the picks'
    counts and scores and per phase the misses (`ds_router_rec`); the history is the Python
    engine's `DSV41_PRUNE_DB` file (np.savez, request units, score history v2), read and
    written in place by `tidelm_core::npz`, blended with the routing trace (rank "sum", prior 4,
    half-life 10 folds, miss gate 0.0101733 - adapt_config.py at sensitivity high);
  - boot: the same residents as the Python engine's from the same files (budget 10,000: the
    masks are byte-identical), and the planner gives the same 512 moves in the same order for
    the same pending demand (examples/adapt_boot.rs against the engine's own functions in the
    `tidelm:local` container);
  - pre-prefill: the probe (probe_prefill.SmallProbe): below 500 tokens the prompt is routed
    once with the current residents (two-pass), above it a 64-token excerpt (select_spans:
    head, two suffix windows, tail) keeping each row's engram n-grams from its place in the
    prompt; at >= 1.02% predicted misses the residents are re-fitted before the real prefill;
  - post-prefill: the prompt folds into the history at the prefill -> decode boundary and a
    prompt of >= 32 tokens missed at >= the gate re-fits them;
  - post-decode: every 600 output tokens a stretch missed at >= the gate re-fits them to the
    history previewed with the answer so far; each request's end folds and saves it and plans;
  - hot load: async hot reads (hot_reads.py at HOT_MIN 0.3, HOT_MAX 2, ring 128 past the
    residents): the heaviest missing picks of a decode step are read during the next and
    routable from the one after (marks on the LUT and keep mask around the step's graphs);
    both ranks plan alone from the same table, and rank 0 sends a digest of that state with
    every step so a divergence stops the pair rather than routing garbage.
  Moves read each rank's share on the io workers: 9-243 experts in 0.10-0.32 s a pass.

  The pair, bench.py's README prompts (sampled 0.6, 1,024 tokens, median of 3), adaptation +
  probe + hot reads on from the Python server's history, against the Python README rows (same
  stages on):

  | workload | Rust | Python |
  |---|---:|---:|
  | prose decode | 30.63 tok/s (decode misses 6.0 / 4.9 / 4.3%) | 25.59 |
  | code decode | 50.65 tok/s (5.6 / 4.4 / 4.0%) | 43.31 |
  | prose / code first token | 0.71 / 0.73 s | 1.22 / 1.25 s |
  | 8K first token | 7.31 s (6.76 with adaptation off) | 7.3 s |

  Hot reads: 972 reads over 3,401 decode steps, 1.7 s waited in all; prose decode 32.36 tok/s
  with them and 32.37 without, in back-to-back runs. Rust's layers 21-39 vote with every prompt
  row (it computes them all; Python's SWA decoder replay routes only the last 128 there).

  The prefix cache is in (2026-10-24), both tiers as the serving profile runs them
  (`--prefix-cache 1`, 16 snapshots, `--prefix-disk DIR`, 20 GB, a save at most every 300 s,
  dedup). A lane's ckv / ik rows stay where they are; a snapshot holds what a later step
  overwrites: the last min(n, 128) positions' window rows of all 40 layers and the drafter's
  3 rings (Rust has no SWA replay to rebuild them, nor a prefill -> decode replay to reseed the
  drafter), and the ratio-2 compressors' unpaired row. Boundaries are taken before each 2,048-row
  chunk, the full one at P - 1 (the last prompt token runs again for its logits); a restore
  scatters the rows back to their ring slots and prefills the rest. Disk bundles (rank-R dirs,
  one id across the pair, ckv / ik rows once, FNV checksum, namespace = binary + model + layout)
  load on both ranks or neither. A request's probe and boundary pass count new tokens from the
  restore, an exact re-send as none (the Python engine's full hit). The pair, max-seq 524,288:
  a 9,019-token prompt 13.09 s cold, a follow-up 2.65 s with 8,192 restored; a 3,288-token
  re-send 4.44 -> 0.37 s (Python: 14,336 tokens 20.43 -> 0.35 s); a boundary restore and a disk
  restore after a restart give the cold answer. Bundles: 15.8 MB at 3,288 tokens, 40.5 MB at
  9,019. Two changes it needed: reset_lane keeps the lane's engram n-gram history (replacing it
  left a restored prefix's entries empty, and the suffix's tokens differed from a cold run's),
  and the probe's excerpt runs in a scratch lane past the request lanes, where a ragged step's
  filler rows also sit (its ring's last slot), so an excerpt never touches a lane's restored
  rows.

  Image inputs are in (2026-10-24), the profile's DSV41_VISION=1 / VISION_MODE=peer (`--vision 0`
  turns them off). The server decodes and preprocesses as the checkpoint's image_processor.py
  does (crates/tidelm-image: PNG, JPEG through libjpeg-turbo - Pillow's library - GIF's first
  frame, WebP; Pillow's integer bicubic in ImageOps.pad; the bf16 normalization), and expands
  each placeholder into its span. Rank 1 holds the tower (0.97 GB) and sends rank 0 the
  aligner rows over the control channel; both splice them, with the three learned delimiters,
  into the embedding, route the span with ffn.gate.bias_vl in all 40 layers (the reference's
  choice: the Python engine's decoder replay routes up to 125 tail positions of a span with the
  text bias in layers 21-39), and keep engram off it (dead n-gram positions, gate skipped). A
  step with image rows runs eager. The prefix cache keys an image's positions by a digest of its
  patches, so a prompt reuses a span only with the same image. An http(s) URL is fetched (30 s,
  64 MB); a local path, which the Python reader opens, is refused.
  - preprocessing: 12 images (PNG RGB / RGBA / palette / gray / 16-bit gray, JPEG 4:2:0 / 4:2:2 /
    4:4:4 / gray, GIF, WebP; upscaled, padded, tall) give Pillow's patches bit for bit
    (tests/pillow holds 10 of them)
  - the tower (vision.cu: a flash attention for the 16 x 64 heads, 2D RoPE, the epilogues;
    ds41's tensor-core GEMMs), against an fp32 run of the same weights on a 640 x 480 image:
    4.53% relative error and 0.99898 mean row cosine, where PyTorch's bf16 tower has 4.75% and
    0.99887 (the two bf16 towers differ from each other by 2.7%: summation order, grown over 32
    blocks of large activations; the patch embedding is bit-equal). 0.115 s at 1,610 patches
    (Python 0.055) and 0.487 s at 8,418 (Python 0.471)
  - the pair: the 640 x 480 image's text ("TIDE 42 vision"), shapes and colors read back;
    first token 0.84 s for its 227-token prompt, 1.59 s for a 1920 x 1080 image's 989; the same
    request again restores 226 of 227 tokens (0.06 s); the same grid with other pixels restores
    none. The first image prompt missed 30% of its expert picks with residents fitted to text
    (the boundary pass then moved 311 experts).

  CPU drafting (2026-10-24, measured, not built). The CPU cores pull 128-145 GB/s of the
  LPDDR5X the GPU decodes from (10 X925 / 10 A725 cores, a 512 MB read): a CPU bandwidth load
  beside a greedy decode took it from 32.2 to 26.9 tok/s (the 10 A725 cores) and 28.3 (7 X925),
  so a CPU drafter that streams weights costs the GPU what it reads. A prompt-lookup (copy)
  drafter does not: 30 ns a lookup on one core over a 2M-token 8-gram index. What decides it is
  the verify width a long copy needs - the pair's verify step by rows on copy text (2,585 tokens
  of context, `tidelm-server --bench-width`): 1 / 2 / 4 / 6 rows 43.1 / 51.0 / 65.8 / 84.3 ms,
  8 / 12 / 16 / 17 96.3 / 132.5 / 149.1 / 153.1, 24 / 32 / 33 176.5 / 205.8 / 217.4, against a
  4.88 ms DSpark pass. DSpark already accepts 5.2 of its 6-token cap a step on a file rewrite,
  so a copy only pays past it; replayed exactly over greedy runs the server logged (x_debug,
  rust/tools/copy_draft_replay.py), copies of up to 16 / 32 tokens after a 6-token match:
  file rewrite 61.8 -> 107.0 / 139.5 tok/s, tool call re-run 63.2 -> 89.4 / 93.2, quoting code
  53.3 -> 67.6 / 74.1, fresh code 55.5 -> 53.8 / 53.5 (a few early-failing copies), prose
  unchanged (no copy fires). The pricing reproduces the logged DSpark runs within ~5%.

  The copy drafter is in (copy.rs, --copy-draft 1 by default, x_copy_draft per request): the rule
  the replay chose - copy only from the prompt (files, tool results, earlier turns), after a
  6-token match, at least 6 tokens, the next copy 2 x the last one's accepted + 2 within 8..32.
  Its first rule (copies from anywhere, fixed 32) lost 3% on fresh code: repeats inside the answer
  diverge part way at 30 rows of verify. Measured on the pair, greedy: a file rewrite 58.6 ->
  131.9 tok/s (50 copy steps accepting 1,482 of 1,536 tokens; the output equal to the copy-off
  run's), a re-run tool call 60.9 -> 100.4 (equal), quoting code 50.6 -> 75.1; back to back on one
  boot, fresh code 57.4 -> 58.9 and prose 32.3 -> 32.5 tok/s, no copy step firing on either.

  The decoder SWA bounded replay is in (2026-10-24, --swa-replay 1, the profile's
  DSV41_SWA_REPLAY=1; tech report 2.2 / 3.2.2, engine/model.py::decoder_replay): prompt rows run
  the encoder half (layers 0..=20, everything that writes global KV, no head); every step that
  runs prompt rows keeps their state entering layer 20 for the last 128 positions of a lane; the
  prompt's end replays those rows through layers 20-39 and the head, the decoder layers' windows
  floored at the replay's first position (a per-lane floor the attention kernels read, which
  decode keeps until the window passes it). Layer 20 runs again in the replay to remake its rows'
  index top-k and candidate pool - the Python engine carries both across - and the drafter seeds
  from the replayed rows. Snapshots and disk bundles carry the kept rows and the floor (bundle
  format 2); restoring a prompt's own snapshot at length - 1 (an exact re-send) runs its last row
  through every layer against the restored decoder windows instead of replaying, which is the same
  computation. Measured: a 14-token prompt's greedy output equal to the full prefill's (86 of 86
  tokens), a 6,032-token needle prompt 6.03 -> 3.79 s to its first token with the same answer;
  the README protocol 8K first token 9.30 -> 4.81 s. An exact re-send of 8,028 tokens 0.29 s, a
  follow-up turn after it 0.72 s.

  Still missing, in order of what the numbers above say (prefill at large chunks is in: M5,
  8K first token 27.3 -> 6.76 s):
  1. With speculation the pair's logits digest no longer equals one box's (4 layers, prose
     prompt, `--spec`: e7aff077 vs 79e83f92); already so at 3f6fa2d, after the drafter split
     claimed bit-identity. The two ranks agree with each other; where the pair departs from
     one box is not yet found.
  2. The lossless packed verify head (DSV41_HEAD_KERNEL=packed, 2.94 -> 2.25 ms/step in
     Python), the sharded head's argmax without gathering logits (`ds_argmax_bf16_pairs` /
     `_combine` are in, unused), the server's Anthropic routes. (Tool calls are in: the checkpoint's encoding.py ported whole, DSML
     calls parsed strictly then tolerantly, as server/app.py does; its tool grammar is off in the profile.
     Images are in: below.)

## GLM-5.3 (`crates/tidelm-glm`, 2026-10-25)

The second engine on the same stack: engine/glm/tp.py's TP2 forward for the native EXL3
checkpoint (GLM-5.3-EXL3-3.0bpw), served by `tidelm-server` when `--model-dir`'s config.json has
`model_type: glm_moe_dsa` (`crates/tidelm-server/src/glm_main.rs`), launched by
`scripts/rust-up.sh` from a GLM profile (`GLM_*` keys); `scripts/tide-up.sh` picks the Rust engine
for every profile it serves unless the profile sets `TIDE_ENGINE=python`.

What is ported, by the DS port's feature list:
- **TP2 pair**: RoCE one-shot gathers for decode, NCCL for prefill, rank 0 mirroring every call;
  the boot guard compares config, binary, pack source and the serving flags both ranks act on.
- **Forward** (`forward.rs`): absorbed MLA with FP8 kv_b, the DSA indexer (tensor-core scores,
  ordered top-k, picks reused by "shared" layers), the KV split by position parity with the
  split decode's two gathers and cross-rank merge, prefill on gathered spans; EXL3 everywhere
  (tools/exl3_moe_cuda.cu compiled as-is; prefill's dense matrices unpacked for cuBLAS, 17-64
  rows through the decode GEMV), the MTP layer on its own whole cache with its 4-bit arena.
- **CUDA graphs** for decode passes (1-8 rows) and MTP passes, captured on first use, split
  after every MoE layer's exact routing when hot reads are on; a side stream inside them.
- **MTP speculation** (`serve.rs`): up to 7 drafts under the calibrated gate (mtp_cal.py's
  table) or the flat one, exact acceptance greedy or sampled (each rank's top-1024 logits and
  log-sum-exp; the whole row when they cannot prove the nucleus).
- **Copy rounds**: lookup_draft.ExactDraftCache v2 (GLM_LOOKUP), not the DS copy drafter's rule.
- **Adaptive residency** (`adapt.rs`): tp.adapt after the prefill and every 200 tokens, the
  two-pass prefill, bg_step's victims, synchronous hot reads kept over victims; the history in
  its own .npz (read at first boot from the Python server's db.pt).
- **Prefix cache**: the live sequence's prefix in RAM, and the disk store (`prefix.rs`).
- **Server**: the OpenAI routes, the Anthropic Messages route, tool calls (GLM's XML), the GLM
  template and parser (`tidelm-text/src/glm.rs`, fixture-tested against glm_app.py).

Not ported, and why: concurrency (the Python GLM server serves one request at a time too; two
lanes would halve the 12,500 residents' share of memory per request), vision (GLM-5.3 has
none), the bounded replay (GLM has no sliding window), GLM_MISS=stream, MXFP4 experts, FP4 KV.

Kernels (`tidelm-kernels/cuda/glm.cu`, `tests/glm_gpu.rs` against CPU transcriptions): the EXL3
unpack equals the numpy oracle's golden values on the real checkpoint; both EXL3 paths match
fp64 at 2.1e-4 (Python's routed bar 3.5e-4); MLA attention 1.8e-3 in every mode (Python's kernel
1.7e-3 against fp32).

Where a decode pass goes (one row, graphs, the pair; nsys with node tracing): the EXL3 grouped
GEMV ~60% (the same kernel as Python's), the RoCE gathers ~8% (five a layer: the split
attention's two, o_proj's sum, the MoE's two), the router's fp32 GEMV ~4%.

The GEMV is at the DRAM roof (2026-10-25, cold reads; `glm_bench_moe` on random 3-bit experts at
the rank shapes, `glm_bench_exl3` on layer 40): routed experts at one row and 8 experts 212 GB/s
gate+up and 190-205 down, 212-220 at 2-8 rows; o_proj 31 MB 208-217, the head 229-242, q_b 10.5 MB
147-156 (58-67 us in the decode pass), q_a 7.9 MB 145; only the 2-4 MB matrices fall short (kv_a
94, shared down 107). TensorFold's GLM-5.3-Flash kit measures the same kernel on this hardware at
207 / 204 GB/s for one row and its small dense matrices at 94-95
(`glm53-tensorfold-spark/docs/DECODE-KERNELS.md`); its persistent-grid rewrite (0440) ran slower.
This section first said "q_b 95 GB/s, latency-bound when cold": that was the bench's own artifact.
Its L2 flush was a memset, and the timed call wrote the dirty lines back as it read (105-112 us
then, 63-72 after a read flush).

Measured and not taken: split-K counts other than tp.py's (dense within ~5% of the best setting
except kv_a, 31 vs 26 us at 16 splits on the side stream; routed within 3% at every row count,
TensorFold's 4-way gate/up split included), trellis prefetch depth 2, 4 n-tiles a warp;
prefetching prefill attention's key tiles. The next levers: the ~40% of a decode pass outside the
GEMV, and prefill attention (5.1 of an 8K prefill's 19.4 s: its 32-key tiles are bound by
shared-memory work; Triton's 64-key tiles were ~1.8x faster per pick).

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
