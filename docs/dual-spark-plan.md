# Plan: DeepSeek-V4.1-Flash on TWO DGX Sparks

> Status: **plan, with Phase 0 measured.** Gate G0 (the cross-node collective latency this
> whole design rests on) has run and **passed** — see "Gate G0 — RESULT" below. Everything
> downstream of it is still proposal. Every tok/s figure marked
> *(est.)* is arithmetic extrapolated from the single-box numbers in RESULTS.md and
> docs/architecture.md, and must be treated as a hypothesis until Phase 0 and the Phase 4
> benchmarks run.

The single-box recipe exists because one 121 GiB box cannot hold the model:
**288.8 GB of FP4 routed experts against ~85–90 GB of expert budget = ~1.3 bpw average**,
which kills every all-resident scheme (docs/architecture.md). Two Sparks pool ~242 GiB
visible. The first question the plan answers is what that buys, arithmetically, before any
code is written.

---

## 0. What dual Spark buys — the arithmetic

Assumptions per box (from the repo's own measurements): ~121 GiB visible unified,
~18.5 GB always-resident non-expert weights, `keep_free` floor ≥ 8–10 GB, local NVMe
~5.5 GB/s at depth (O_DIRECT), KV ~890 B/token (negligible), activations ~3 GB.

Non-expert weights (~18.5 GB incl. the 7.9 GB DSpark drafter) **must be replicated on
both nodes** in the chosen design (Expert-Parallel), because every token passes through
attention and the drafter on every layer on every node.

| # | Mode | Resident need (dual total) | Fits in ~242 GB? | Quality | Speed *(est.)* |
|---|---|---|---|---|---|
| A | **EP2, pruned all-resident, `PRUNE_KEEP≈0.55–0.60`, native NVFP4** | 0.60×288.8 = 173.3 experts + 37 replicated + ~5 KV/act ≈ **215 GB** | **borderline** (~4 GB margin — real EP2 fit is keep≈0.55; TP2-sharded fits keep-0.60 with headroom, see "Mode A+") | strictly better than the shipped single-box 0.40-CB3 row (more kept, no requantization) | **~19–21 tok/s** (est., parity with single-box — dense-weight reads, not residency, are the wall; see caveat below) |
| B | **EP2, full-quality streaming, arena ~85–90 GB/node** | ~175 arena + 37 ≈ **212–222 GB** | **yes** | **the checkpoint's own NVFP4, nothing pruned** | ~6–10 tok/s vs 2.6–4 today: 63.7 % of all experts resident (hit-rate curve ⇒ ~0.93–0.95) **and** the remaining miss bytes split across **two NVMe drives** |
| C | EP2, full CB3 all-resident (every expert, 3-bit) | 222.2 + 37 ≈ **259 GB** | **no** (~37 GB short) | 3-bit, all routable | — |
| D | PP2 (20 layers/node), full CB3 all-resident | 222.2 + ~21 ≈ **243 GB** | **no**, ~21 GB short even before the keep-free floor | 3-bit | — and the CB3 kernel is 4x too slow today (LIMITATIONS v0.2.0-wip) |
| E | 2× independent single-box servers + round-robin proxy | none (no shared state) | trivial | per current defaults | unchanged tok/s, **2× concurrent users** |

**Headline:** even two Sparks cannot make the *full* model resident (that is TP4-on-4-Spark
territory — and confirms the public vLLM build's "TP2 does not fit either way",
NOTES.md §0.5, which refers to all-resident tensor parallel). Dual Spark buys instead:

1. **Mode A** — a pruned all-resident config at **keep≈0.55–0.60 in native NVFP4**: better
   quality than the shipped keep-0.40/CB3 recipe, with *no* CB3 repack and *no* dependency
   on the 4x-slow CB3 kernel. This is the recommended flagship deliverable.
   **The honest caveat:** its speed is *not* what residency buys. Decode is bounded by the
   bytes a step moves, and a step reads ~18.5 GB of always-resident weights per token
   (FP8 dense/attention/shared/LM-head — all read in full every token; the NVFP4 experts
   the same way) at the box's ~273 GB/s ceiling — a floor of ~12–14 tok/s at batch 1 even
   with the whole model resident. The single-box 19 tok/s row is already at/near that
   floor, and EP2 cannot shard those weights (chunk-invariance machinery + mHC residual
   structure), so Mode A realistically lands **~19–21**: a **quality** win (0.40→~0.6
   kept, no requantization), not a speed win. Only the TP2-sharded variant ("Mode A+",
   below) lifts the ceiling — by halving the replicated dense reads.
2. **Mode B** — the **full-NVFP4-quality streaming mode at roughly 2x** single-box speed.
   This is where dual genuinely helps bandwidth: streaming is NVMe-bound, and EP2 splits
   the cold-expert reads across **two** drives (each node reads misses only for experts
   *it owns*) — 2× NVMe bandwidth is real, unlike Mode A's memory-bandwidth wall.
   The 2× also assumes the cross-node collective tax stays small, which Phase 0 measures;
   at full quality expect ~5–8 tok/s, still far from Mode A's floor — **pruning is the
   quality tax that pays for speed; residency alone does not.**

Modes A and B are the same engine; they differ by `PRUNE_KEEP` and arena policy, exactly as
on one box. The checkpoint's experts are **NVFP4/MXFP4** (e2m1 codes + UE8M0 scale per 32
values, 4.25 bits/weight — "FP4" in this repo, "MXFP4" in the SGLang/vLLM docs; same
format); "native" there means *never requantized*, not a format claim. The dense
projections are FP8 (e4m3 + UE8M0 32×32 block scales) and are read **in full by every node
every token** under replicated-attention designs — that read is the speed wall; see below.

### The bandwidth law (validated against two published fleets)

Decode tok/s at batch 1 is set by the bytes each *node* moves per token divided by its
~273 GB/s. Bytes = full dense reads + routed-expert reads + (sharded designs) NCCL. This
model reproduces **both** public datasets:

| | this repo, single box | MiaAI-Lab 3× Spark (SGLang TP3/EP3) |
|---|---|---|
| dense FP8+head bytes per node per token | ~16 GB → ~59 ms | ~5.3 GB → ~17 ms (their profile: "dense FP8 projections ~17 ms") |
| routed NVFP4 per node per token | 6×40×18.8 MB = 4.5 GB | ÷3 ranks ≈ 1.5 GB (~27 ms/step, "at memory bandwidth") |
| NCCL | 0 | ~13–16 ms (104 collectives, 50–100 µs each) |
| model ⇒ tok/s | ~70 ms/token ⇒ ~14 est. (measured 19 with DSpark absorption) | ~26 ms/token ⇒ **37.9 measured** ✓ |

Consequence: **whatever a design replicates, it pays for on every token** — and dual Spark
buys bandwidth only for the bytes it actually splits (routed experts, NVMe streaming), not
for replicated dense reads.

### The chosen parallelization: EP2 replicated-attention (default) vs TP2-sharded ("Mode A+")

| | **EP2** (this plan's default) | **TP2-sharded** ("Mode A+", much heavier) |
|---|---|---|
| what's replicated | attention, HC, dense, LM head, DSpark, shared expert, KV, sampling — **everything but the routed experts** | nothing: heads 64→32, o_groups 8→4, vocab 129,280→64,640, draft experts 128→64 — **all divide cleanly by 2** (unlike TP3, which forced MiaAI's `tp3_pad.py` to pad heads 64→96, groups 8→12, vocab, and draft experts, leaving rank 2's attention shard entirely padding) |
| new collectives per decode step | 40 small MoE-sum all-reduces | ~100+ (every split GEMM combines) ≈ 10–15 ms/step at their measured per-collective tax |
| keep-0.60 memory | 0.6×288.8/2 + 18.5 ≈ **105 GB/node vs ~99 usable → real EP2 fit is keep≈0.55** | experts 80.7 + dense ~8 + head/embed/DSpark ~4 + KV ≈ **~96 GiB/rank — fits with ~25 GiB headroom** |
| Mode A speed *(est.)* | **~19–21** — the replicated-dense wall; dual does not move it | **~22–28** — dense bytes halved per node; the only dual design that beats the wall |
| Mode B speed *(est.)* | ~5–8 (2× NVMe streaming minus the network tax) | n/a for streaming today (EP is how this engine streams; a TP2-dense + EP-expert hybrid is a Phase-5 conversation) |
| engine surgery | moderate: one combine point; everything else stays bit-identical to the single-box engine | heavy: shard attention/HC/mixes/GEMMs, re-derive the chunk-invariance guarantees under sharding, real NCCL-per-layer plumbing |

**Decision point (before Phase 2):** ship EP2 (fast to correct: full-quality Mode B +
keep≈0.55 pruned Mode A on exactly two Sparks), then optionally revisit TP2-sharding as a
second iteration for pruned-mode speed. PP2 remains rejected: at batch-1 decode it adds
inter-node hops without cutting single-stream latency and still can't unlock residency
(§0 table, row D).

Why not off-the-shelf TP2 (SGLang/vLLM)? Those stacks **cannot fit 2 Sparks at all**: they
never prune or stream (305 GiB ÷ 2 = 145 GiB/rank > 121.7 — MiaAI's own table, and the
public vLLM 4×-Spark build's "TP2 does not fit either way"). On exactly two Sparks this
repo's resident-hot-set + pruning + NVMe-streaming design is the only thing that holds the
model — EP2 extends it directly; TP2-sharding this engine is the harder, faster future.

**EP2 (expert parallel, world=2)** keeps *everything* — attention, Engram, router, HC,
DSpark drafter, shared experts, KV, sampling — **bit-identically replicated on both nodes**,
and splits only the 15,360 routed experts:

```
every token, every layer L:
  both nodes compute the router identically  -> same top-6 indices (deterministic)
  node r resolves + computes only the owned experts (ownership: e % 2 == r)
  dist.all_reduce(routed_partial, SUM, fp32) -> combine
  + shared expert (replicated, added locally, NOT inside the all-reduce)
```

Traffic per verify block (6 rows × 5120 × fp32 ≈ 123 KB) × 40 layers ≈ **5 MB/token-block**
over a ~24.5 GB/s aggregate link (measured; each half is behind PCIe Gen5 x4)
— latency-bound, not bandwidth-bound. Budget **~2–4 ms per generated
token** for the ~40 small cross-node collectives (measure in Phase 0; if each round trip is
>100 µs the estimate needs revisiting).

### Partial TP2 — RESULT: **no gain** (measured 2026-09-13, `DSV41_TP_DENSE=1`)

The "~22–28 vs ~19–21" row above is an estimate from halved dense bytes, and the cheapest
piece of it was built and measured first: `DSV41_TP_DENSE=1` shards only the **shared
experts** (column-parallel w1/w3, row-parallel w2, one extra all-reduce per layer), 0.71 GB
of the 8.06 GB replicated per step. 8k/512, three runs each:

| | ttft | tpot | decode | accept_len | step |
|---|---|---|---|---|---|
| EP2 (TP off) | 14919 ms | 51.13 ms | **19.56 tok/s** | 3.56 | ~182 ms |
| + dense TP | 19808 ms | 69.48 ms | **14.39 tok/s** | 2.87 | ~199 ms |

Slower, not faster — and the premise was wrong, not just the size of the effect. **Decode is
not bandwidth-bound on this engine**: GPU utilization during decode fluctuates 70–90% instead of
pinning at the ceiling, so the step is losing time to gaps in the serial chain, not to reads.
Under speculation the shared-expert GEMM is `[6,5120]×[5120,2304]`, skinny enough to be launch-
and latency-bound, so halving N does not halve its time, while 40 more collectives per step
lengthen the very chain that is already leaving the GPU idle. Halving bytes cannot buy anything
until the gaps are closed. (Caveat: the runs are an hour apart with a
decaying demand DB and the per-run spread is wide, so read this as "slightly slower", not
"26% slower". The `accept_len` drop is the part noise does not explain — summing partials moves
the numerics the drafter sees.)

This does not disprove full TP2, but it removes its cheapest evidence: attention is the same
skinny shape and would add another 40 collectives for its 5.41 GB. Revisit only if the
per-collective cost drops. The flag stays, default off — correct, and the place attention
sharding would build from.

---

## 1. Prerequisites & Phase 0 gates (no engine code)

Hardware/OS facts to confirm before anything else:

* [ ] Two GB10 boxes, same DGX OS / driver (580.173.02+) / CUDA 13, both aarch64.
* [ ] **Each node gets its own full 510 GB checkpoint copy on local NVMe.** O_DIRECT does
      not work over NFS/overlay (env.example), and per-node streaming only pays if each
      node reads its *own* owned experts from its *own* disk. Disk budget per node:
      ≥ 600 GB free (weights + logs + scratch).
* [ ] Interconnect: ConnectX-7 200GbE direct-attach (QSFP DAC) or via switch; RoCE
      configured (`ibstat` shows PORT_ACTIVE); MTU/jumbo frames.
* [ ] `torch.distributed` + NCCL works **between the two boxes** with the cu130 aarch64
      wheels (torch 2.13.0+cu130). NCCL IB verbs enabled: `NCCL_IB_HCA=mlx5`,
      `NCCL_SOCKET_IFNAME` set to the 200G link.

**Gate G0 — measure, don't assume** (half a day, `nccl-tests` or a 10-line torch script):

| quantity | why | kill criterion |
|---|---|---|
| all_reduce(123 KB fp32) 2-rank latency, e.g. 1,000 reps | ~40 of these per verify step | if per-collective > 150 µs, EP2 adds > 240 ms/token → stop and reconsider |
| ping / RDMA write RTT | floor for all network design | — |
| NCCL over socket fallback (RoCE off) | worst case if RoCE is fought by the OS | sanity only |
| NVMe O_DIRECT read rate per box (existing tooling in engine/experts.py path) | Mode B's parallel-streaming claim needs ~2.5 GB/s *per node* sustained | — |

### Gate G0 — RESULT: **PASS** (measured 2026-09-12, `scripts/run_g0.sh`)

| quantity | measured | criterion | verdict |
|---|---|---|---|
| all_reduce(123 KB fp32), 2-rank NCCL/RoCE, n=200 | **48.5 / 59.6 / 60.1 µs median** (3 runs; mean 51–83, p95 55–198) | > 150 µs ⇒ stop | **PASS** |
| ⇒ projected collective tax | **1.9–2.4 ms/step**, **0.6–0.8 ms/token** at acc=3 | budget 2–4 ms/token (§0) | inside budget |
| **same gate inside the serving image** (`G0_DOCKER=1`) | **58.9 / 61.7 / 68.6 µs median**; mean 84–102, **p95 185–444 µs** | > 150 µs ⇒ stop | **PASS** — but see the tail note |
| same all_reduce over gloo/CPU (fallback) | ~1.9 ms median | sanity only | ~30× slower — RoCE is doing real work |
| ICMP RTT over the CX7 link | 0.12–1.37 ms | — | — |
| NVMe O_DIRECT per box | **not yet measured** | Mode B needs ≥ 2.5 GB/s/node | **open** |

The EP2 collective tax is therefore ~1 % of a ~70 ms decode step: the plan's central
latency risk (§7, row 1) is retired, and Phase 1's numbers stand.

**Container vs native — the medians match, the tail does not.** Containerised medians sit
within noise of native (58.9–68.6 vs 48.5–60.1 µs), so RoCE is genuinely working through
`--device /dev/infiniband` and not quietly falling back to TCP (that would read as ~1.9 ms,
not ~60 µs). But the containerised **p95 is 404–444 µs against 55–198 µs native**, and a
decode step fires 40 of these: at p95 ≈ 420 µs, the ~2 collectives per step that land in the
tail add ~0.8 ms/step on top of the ~2.4 ms median cost. That still fits the 2–4 ms/token
budget, but it spends a visible slice of it, and it is **unexplained** — candidates are
docker's cgroup CPU scheduling against the NCCL proxy thread, or simply a noisier box during
those runs. Worth re-measuring alongside the Phase 4 benchmarks before any dual tok/s number
is quoted from a container; not worth blocking Phase 1 on.

**Two bring-up bugs cost most of this gate, both worth keeping written down:**

1. **Launch order.** Rank 0 is the TCPStore *server*; the first launcher started rank 1 first,
   so the peer burned its connect budget against a port nothing had bound yet and reported
   "TCPStore timeout" — which reads like a dead fabric and is a dead launcher.
   `scripts/run_g0.sh` now starts rank 0, waits for the port to actually accept, then rank 1.
2. **GID indices are not the same on the two boxes.** The RoCEv2/IPv4 entry for the CX7 port is
   index **5** on 10.0.0.1 and index **6** on 10.0.0.2 (whose index 5 is an empty slot). A single
   `NCCL_IB_GID_INDEX=5` for both ranks — which is what the single-box vLLM/SGLang configs
   appeared to license — is correct on rank 0 and points rank 1 at nothing, and NCCL does *not*
   fall back to sockets for it: `ibv_modify_qp failed with 61 ... local GID index 5, local
   GID ::`. Each rank now derives its own with `scripts/roce_gid.sh`, wired into both
   `run_g0.sh` and `start.sh`. **Never pin one GID index for both boxes.**

**Gate G1 — pick the deliverable.** Decide with the owner: flagship = Mode A (fast,
slightly-lossy, all-resident keep≈0.55–0.60 NVFP4), Mode B (full quality, medium speed), or both
(same code, different flags — recommended, mirrors the single-box story).

Cheap parallel workstream: **Mode E first** (two standalone servers + a stateless
OpenAI-compatible round-robin proxy in `server/` or a tiny new `tools/proxy.py`). It
validates both boxes, the network, and the operational scripts end-to-end with *zero*
engine surgery, and is useful on its own. Do it inside Phase 1.

---

## 2. Phase 1 — Distributed plumbing (skeleton, tiny arena)

Goal: two processes on two boxes, `world_size=2`, serving a **2-layer debug config with a
small arena**, producing the same logits as the single-node engine within fp32-reordering
tolerance. No performance target yet.

Files and specific changes:

1. **`engine/v41_engine.py`** (`V41Engine.__init__`, ~L232–360)
   * New args: `--rank`, `--world-size` (or read `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from the
     environment, torchrun-style); `dist.init_process_group("nccl")` guarded by
     `world_size > 1` so every single-box path stays untouched.
   * Ownership: `owned = (expert_id % world_size == rank)` per layer. Interleaved (`% 2`)
     rather than contiguous halves so the trace-ranked hot set splits ~evenly across nodes
     by construction (the coverage.json ranking is global; verify the split is balanced and
     log it).
   * Arena sizing: unchanged logic per node (`mem_get_info` ∪ `MemAvailable`,
     `keep_free_gb` floor — **do not lower the floor**, the box hard-resets at negative
     MemAvailable); the auto-sizer now sizes a *half-width* expert set.
2. **`engine/experts.py`** (`ExpertStore`, L113–445)
   * `resolve()` (L368): non-owned expert ids return a sentinel; **no local read, no LRU
     entry, no transient-ring slot consumed** for them.
   * `warm_start()` (L425): filter `ranked_keys` to owned experts before filling slots.
   * `hit_rate()` (L441): per-rank counters + a combined-view structure that rank 0 can
     serve in `/health`.
3. **`engine/model.py`** (`moe()`, L482–504)
   * Compute `routed` over owned slots only (the `tools/fp4_moe.py` grouped kernel already
     groups by slot — confirm it tolerates a *missing* k in the (token,k) list; pad-to-no-op
     may be needed), accumulate in fp32, then `dist.all_reduce(routed, SUM)` and add
     `shared` locally afterwards, so the shared expert is counted once and its numerics are
     untouched.
   * **Numerics contract:** the single-node kernel writes one row per `(k, token)` and sums
     the k experts in fixed order. The distributed sum changes only the *grouping* of the
     6-term addition (`{owned} + {remote}` instead of one 6-sum). Acceptance target:
     teacher-forced NLL within **±0.01 nats** of single-node (same bar as the existing
     fp4-dense rows in RESULTS.md), and `engine/test_spec_lossless.py` must still report
     64-of-64 identical greedy tokens.
4. **Generation loop / control flow** (`V41Engine.generate`, `server/app.py::generate`)
   * One HTTP server (rank 0). Both ranks run the *same* decode loop; rank 0 broadcasts the
     sampled token id (and prompt token ids at prefill start, and rollback counts) via a
     tiny NCCL broadcast before each step. Sampling RNG stays rank-0-authoritative; rank 1
     never samples. Caches are replicated (attention is replicated) so rollbacks replay
     identically.
   * DSpark drafter: fully replicated on both nodes (7.9 GB, must stay resident — the draft
     block's `indices` are then known on both nodes before the verify forward, which is
     also what lets both nodes prefetch their owned experts — keep this property).

## 3. Phase 2 — Correctness gates before any speed claim

* [ ] 2-node parity run: fixed prompt, `SPEC=0`, greedy — token-identical to single-node
      through ≥ 64 tokens (chunk-invariance bar from docs/architecture.md).
* [ ] `engine/test_spec_lossless.py` distributed variant: speculative vs autoregressive
      identical **on the dual setup** (losslessness must survive the all-reduce).
* [ ] Teacher-forced held-out loss on the existing corpus (`corpus/`,
      `tools/expert_trace.py` path) at equal arena/keep settings vs single-box: Δnats
      ≤ 0.01 attributed to distributed grouping; anything larger is a bug, not a tradeoff.
* [ ] Crash/resume story: kill rank 1 mid-generation ⇒ rank 0 returns a clean 5xx and the
      launcher restarts the pair (no half-dead arena; mirrors the single-box
      "restart is the only recovery" limitation).

## 4. Phase 3 — Operational: launching, guarding, stopping

* **`start.sh`**: new env `WORLD_SIZE=2`, `PEER=host2` (ssh target), `MASTER_ADDR=<rank0
  200G-interface ip>`. Rank 0 starts locally (nohup, as today); rank 1 started via
  `ssh $PEER 'cd <repo> && ...'` with the *same* env, `--no-wait`; the /health wait covers
  both (rank 0's health implies rank 1 joined — process group init is collective). The
  `MIN_FREE_GIB` guard must run **on the peer too** (ssh `grep MemAvailable /proc/meminfo`)
  before launching anything.
* **`stop.sh`**: SIGTERM rank 0, ssh SIGTERM rank 1, wait for memory to return on both.
* **`env.example`**: document `WORLD_SIZE`, `PEER`, `MASTER_ADDR`, `MASTER_PORT`,
  `NCCL_*` (with a "leave NCCL_* alone unless RoCE is broken" note); single-node defaults
  must keep working unchanged (`WORLD_SIZE` unset = today's behavior — every existing
  number in RESULTS.md stays reproducible).
* **`server/app.py`**: `/health` gains `world`, `rank`, per-rank arena GB/slots/%,
  NVMe GB/token summed across ranks; every completion's `x_engine_stats` gains
  `peer_hit_rate` (bytes served by the peer's NVMe vs local) and `net_ms` (fraction of
  step time in collectives). The bench's rule — *"a tok/s number without the hit rate and
  the GB that produced it is an anecdote"* — extends to: **without the network share**.
* **Traces/warm start**: rerun `tools/expert_trace.py`/`expert_stats.py` once (routing is
  unchanged by EP2, so the existing coverage.json ranking stays valid; only the
  owned-filter changes what each node fills). No re-trace needed unless keep-sets change.
* **Container path**: **built and gated** (reversing this plan's earlier "defer"). Two
  things changed the calculus. First, neither Spark can run the engine natively right now:
  `python3.12-dev` is absent on both, so Triton cannot JIT its CUDA shim (`cuda_utils.c`
  includes `Python.h`) and the first MoE call dies — the image carries `python3-dev` and
  simply does not have the problem. Second, EP2 punishes version skew: mismatched ranks do
  not disagree about an answer, they wedge on a mismatched collective, and the peer was in
  fact found running stale `engine/*.py`. One image on both boxes removes that entire class.
  - `scripts/dual-build.sh` builds **once** and ships the result with `docker save | docker
    load`, asserting the image ids match on both boxes. Two independent `docker build`s are
    not guaranteed to produce the same image (mutable base tag, apt and PyPI both move), and
    that is precisely the skew EP2 cannot absorb.
  - `scripts/dual-up.sh` / `dual-down.sh` replace `start.sh` / `stop.sh` for the pair. Compose
    is *not* used: it does not span hosts, and a compose+env file per box is two more copies
    of a config that must not drift. One flag list is built and applied to both ranks; only
    RANK, the bound port, and each box's own GID index differ.
  - Verbs need `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` and
    `--network host`. Omit any of them and NCCL does not complain — it drops to TCP and the
    collective goes ~60 µs → ~1.9 ms. `G0_DOCKER=1 scripts/run_g0.sh` is the check.
  - `/models` must stay a bind mount onto real local NVMe: the engine reads experts with
    O_DIRECT, which overlayfs will not serve.

## 5. Phase 4 — Benchmarks and the two shipped configs

New `start.sh` profiles (documented in README/env.example):

* **`dual-pruned`** (Mode A): `WORLD_SIZE=2 PRUNE_KEEP=0.55 PRUNE_SELECT=uniform
  TRANSIENT_SLOTS=... KEEP_FREE_GB=10` (EP2; keep-0.60 only under TP2-sharded Mode A+)
  — all-resident, no decode-time streaming.
  Target: **≥ 19 tok/s — a parity bar, not an improvement target** (the wall is per-token
  dense-weight reads at the ~273 GB/s memory-bandwidth ceiling; see §0's Mode A caveat;
  this row's win is quality). Report loss vs the *full unpruned* model on both corpora,
  and vs the single-box 0.40 row (the actual product comparison).
* **`dual-streaming`** (Mode B): `WORLD_SIZE=2` with `PRUNE_KEEP` empty, arena auto.
  Target: ≥ 2× the single-box streaming config (≥ 5.5 tok/s) with reported hit rate
  ~0.93+ and per-node NVMe GB/token; acceptance length unchanged (spec stays lossless).
* `bench/bench.py --workload code|prose --runs 2+` per README's four rules; fill in the
  currently-missing `prose` row at the same time. TTFT with a 1.8k prompt: each node's
  transient ring serves only owned prefill misses, so expect ~2× bytes-per-node reduction
  on the prefill path — measure, don't promise.
* RESULTS.md gains a dated dual section; LIMITATIONS.md drops "no tensor parallelism, no
  second machine" from the README's claims and gains honest dual caveats (collective
  latency share, 2× disk requirement, batch-1 lock still there).

## 6. Phase 5 — Stretch (only after 0–4 land)

1. **Peer-RAM tier:** evicted experts from node A's LRU move into a small region of node
   B's memory (200GbE ≈ 24.5 GB/s measured aggregate, vs local NVMe ≈ 2.5 GB/s effective at decode read sizes ⇒
   a remote-RAM hit is ~10× cheaper than an NVMe miss). A three-tier
   LRU: local arena → peer arena → NVMe. This is the main lever that could push **Mode B
   toward pruned-mode speeds at full quality**.
2. **Overlap**: prefetch the DSpark block's likely owned experts one layer ahead
   (indices are known on both nodes after drafting — see Phase 1.4); single-box LIMITATIONS
   already flags that nothing overlaps reads with compute today, and EP2 halves the bytes
   worth overlapping.
3. **CB3 kernel speedup** (per-lane PTX decoder, LIMITATIONS v0.2.0-wip): unlocks cheaper
   tiers (e.g. keep-0.75 as FP4-hot/CB3-cold two-tier within the same 242 GB).
4. **Concurrency**: the server still serialises requests on one lock; with two boxes' worth
   of arena, micro-batching is the throughput story.

## 7. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| Cross-node collective latency > budget on ConnectX-7 (no NVLink between Sparks) | medium | Gate G0 measures it *before* any engine work; kill criterion in §1 |
| NCCL/RoCE doesn't work out of the box on DGX OS aarch64 | medium | socket-ifname fallback path; Mode E still works without RDMA |
| fp32 re-grouping changes numerics more than ±0.01 nats | low | Phase 2 parity gates; shared-expert stays outside the all-reduce |
| Arena + replication arithmetic overshoots 121 GB/node (hard reset, no OOM) | medium | keep `keep_free_gb` floors, refuse-to-start guards on **both** nodes, start with tiny `ARENA_GB` in Phase 1 |
| Second 510 GB checkpoint copy is an extra disk + download day | certain | `scripts/download-model.sh` on both; document |
| CB3 kernel slowness poisons Mode A if we'd relied on it | resolved | **Mode A is NVFP4-all-resident keep≈0.55–0.60** — no CB3 dependency |
| Repo is WIP: image untested, batch-1, one bench row | accepted | dual plan touches only the native path; every gate is measurement-first |

## 8. Suggested order of work (recap)

```
Phase 0   hardware gates: RoCE + NCCL latency + per-node NVMe        [~0.5 day]
Phase 1   Mode E: two standalone servers + proxy                     [~0.5 day]
Phase 2   EP2 skeleton on 2-layer debug config (rank-owned arena)    [2–4 days]
Phase 3   correctness gates (lossless spec, NLL parity, chunk tests) [1 day]
Phase 4   operations: start.sh/stop.sh/env/health/bench stats        [1–2 days]
Phase 5   dual configs: keep≈0.55 NVFP4 + full streaming, benchmarks  [1–2 days]
Phase 6   docs: README dual section, RESULTS/LIMITATIONS updates     [0.5 day]
Stretch   peer-RAM tier, prefetch overlap, CB3 kernel, concurrency
```

Open decisions for the owner: (1) confirm both boxes + direct-attach CX7 link exist today;
(2) Mode A vs A+B as the headline (recommend both, one codebase); (3) appetite for
`torch.distributed` surgery — if zero, stop at Mode E.
