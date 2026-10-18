# Experimental prefill replicas (default off)

> **Removed 2026-10-17.** `engine/prefill_replicas.py`, its tests and `DSV41_PREFILL_REPLICA_GB` /
> `_BUDGET_MS` are gone (git history has them); a nonzero `DSV41_PREFILL_REPLICA_GB` now refuses to
> boot. It was EP2-only and never enabled in a profile; serving is TP. This page stays as the record.

This is whole-expert EP relocation for prefill, not TP. Original weights and decode lookup
tables remain resident and unchanged. It is not yet a demonstrated performance improvement.

## Lifecycle

1. `DSV41_PREFILL_REPLICA_GB` reserves dedicated tail slots in the main packed FP4 arena,
   additional to the original arena budget. Maximum 1 decimal GB per rank (~53 experts).
   These slots are excluded from both the LRU and transient ring. Memory safety checks include them.
2. With two or more chunks, the first chunk records actual post-pruning expert assignments.
   This is separate from the existing *wanted/unpruned* demand statistics; those are unchanged.
3. Rank 0 plans whole-expert moves that reduce predicted maximum assignment counts per layer.
   Destination capacity is global per rank, not per layer. The plan is broadcast to both ranks.
4. Each destination loads replicas from its local checkpoint using the existing I/O pools and
   staging buffers. Waves contain at most eight moves globally. Both ranks check completion/failure
   after every wave, including a rank that has no loads. A failure activates no replicas.
5. Private prefill LUTs route moved experts only to the destination; originals are masked to the
   null slot for that computation. The router's expert selection and weights are unchanged.
6. Tables are cleared before decoder replay, adaptation, and decode, and again on request cleanup.
   Returning to the original routing requires no weight reload. No mid-decode adaptation is added.

`DSV41_PREFILL_REPLICA_BUDGET_MS=500` is a **soft admission budget**, including planning/loading.
The next wave is not started if elapsed time plus the preceding wave duration exceeds it. Active
disk reads cannot be cancelled safely; a wave, status checks, or table creation may overrun the
budget. Reported `load_ms` and `budget_exceeded` expose that, rather than claiming a hard bound.
The planner predicts assignment counts, not elapsed-time savings. A positive count improvement
does not prove that loading replicas pays for itself.

## Scope and safety

- Supported initial scope: EP2, software FP4, pruned all-resident device-LUT serving.
- Flags/capacity/budget are in the boot-time cross-rank guard. Every new status collective is
  reached by both ranks; load errors are coordinated before any routing activation.
- Prefix reuse is disabled while this experiment is enabled. Regrouping routed FP32 sums can
  change rounding and downstream routing; do not reuse a baseline prefix under another grouping.
- Originals are never evicted for replicas. Existing swap cadence and decode graph LUT addresses
  remain unchanged. Replica slots are reused for fresh plans on subsequent requests.
- No token text or IDs are logged; statistics describe capacity, timing, and predicted work.

## Validation status

CPU unit tests cover planner limits, identical execution ownership, compact-route/LUT agreement,
original-state restoration, invalid plans, budget exhaustion, and load failure. A two-process
CPU/Gloo test exercises the actual collective sequence and coordinated fallback when rank 0's
load fails. Nine tests including chunk timing passed. A synthetic 40x384 Poisson routing histogram
was planned in ~4.9 ms; that is a planner microbenchmark, not an engine speedup.

Run CPU checks without interrupting serving:

```bash
.venv/bin/python -m unittest engine.test_prefill_replicas engine.test_prefill_replicas_dist
```

Before enabling serving, stop the service and run the real-weight GPU regrouping check:

```bash
MODEL_DIR=/home/ryan/models/DeepSeek-V4.1-Flash DSV41_FP4_DOT_SCALED=0 \
  .venv/bin/python tools/test_fp4_replicas.py
```

The GPU test was run and **failed the strict numerical gate at T=2048**. With real layer-0
weights, relocation preserved final BF16 output at T=1/6/63/512 (FP32 maximum differences
0 / 0 / 5.96e-8 / 2.38e-7), but a 2.38e-7 FP32 difference crossed a BF16 rounding boundary
at T=2048. Restoring original routing remained bit-exact in every case. This demonstrates
regrouping sensitivity, not a measured full-model quality failure. Serving activation and speed
benchmarking were initially withheld. The user subsequently authorized evaluating generated
quality and speed despite this tiny numerical difference; that measured trial is recorded below.

The initial GPU harness incorrectly placed real replica slots above the null sentinel, producing
a large false failure in small-batch routing. The harness now matches production: null is last,
above every real and replica slot. Do not attribute that initial failure to production routing.

Broader model quality still requires validation. Short nesting prompts alone do not exercise replicas:
use a passing long-context control with at least two prefill chunks, and inspect `loaded` to
ensure the path actually activated. Keep the feature off until those checks pass. Do not apply
it to decode or split weights into TP shards as part of this experiment.

## Serving trial: no net speedup on captured input

The first enabled startup caught an input-validation bug: the constructor's optional world-size
argument can be None while EPDistributed resolves WORLD_SIZE=2 from the environment. Validation
now uses `self.ep.world`. Both nodes then ran image
`cbfc721c7f174cbb17ab1bf04a9ec5b0095e5bca473a176ed866021b3a58e4bd` with replica GB=1,
budget=500 ms, chunk=2048, ring=4096, normal adaptive keep=.59/spec ON and profiling ON.

`bench/replica_quality.py` supplies 6,082-token synthetic context followed by depth-8 or depth-10
nesting. Both controls passed off and on. On: 66 then 64 replicas were successfully loaded globally,
in 285.104 / 236.548 ms (planning, I/O, coordination and table preparation). Both were within
the budget and activated, so these checks exercised the feature rather than merely testing short
prompts that bypass it. These two passing controls do not establish broad quality parity.

Saved private 14,396-token real input, output capped at 32 tokens, no prefix reuse:

| mode | prefill seconds | tok/s | replicas loaded, pair total | setup ms |
|---|---:|---:|---:|---:|
| off, quiet baseline after build/transfer | 16.849 | 854.39 | 0 | 0 |
| on, first replay | 18.250 | 788.83 | 47 | 198.217 |
| on, second replay | 17.488 | 823.19 | 50 | 210.435 |

47/50 replicas represent 0.884/0.940 GB of expert payload globally, not per node. Capacity reserved
was 53 slots (~0.996 GB) on each node, whether used or not. Each plan loaded all proposed replicas;
there was no need to consume the full 500 ms. Setup time is not pure disk-copy time.

On the second replay, rank 0/1 expert-routing GPU envelopes were 5.965/6.161 s, and combine/wait
envelopes 1.578/2.058 s. Assignment-count predictions improved (~139,203 -> 129,024 summed
per-layer maxima), but end-to-end prefill did not. Adaptive generations/settings changed between
requests and startup occurred between arms, so this is not an isolated attribution of slowdown.
Replicas were disabled again rather than retain extra memory without a demonstrated benefit.
Do not claim memory pressure caused the difference: no OOM occurred during this serving trial,
and concurrent memory-pressure telemetry was not collected.

## README replay after disabling replicas

**Invalid as a correctness-preserving speed baseline:** the subsequent investigation below
found stale fixed-prefill routing after adaptive swaps. Do not use these numbers to decide
replica value or claim a warmup optimization until that bug is fixed and remeasured.

After a subsequent user-requested enabled trial with normal arena=87 GB plus replicas=1 GB,
the user requested another restart with replicas off. Normal arena was restored to 88 GB;
adaptive loading and the same diagnostic timers remained enabled. The server reported zero
replica slots. This repository's README was submitted with a three-bullet summary question:
7,751 prompt tokens, temperature=0, seed=42, thinking off, output capped at 256 tokens.
All runs below processed the full prompt without prefix reuse; an unrelated short request
evicted the prefix before the off repeat. Outputs reached the token cap, so this is a speed
comparison, not a quality gate.

| mode | prefill seconds | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| on, earlier README run, arena=87+1 GB | 15.610 | 496.55 | 12.72 |
| off, first README after restart, arena=88 GB | 14.524 | 533.66 | 13.91 |
| off, warmed repeat | 11.453 | 676.78 | 16.98 |

The on run loaded 52 replicas globally in 222.767 ms. Expert/routing elapsed GPU envelopes
on rank 0/1 were 8.551/5.569 s on, 5.918/4.484 s off-first, and 3.191/5.289 s off-warmed.
Combine/wait envelopes were respectively 0.540/4.790, 2.860/3.041, and 3.112/0.823 s.
Thus imbalance remains without replicas and the slower expert/routing rank can flip on the
same prompt. These are elapsed phase envelopes, not kernel-busy measurements or proof of
token-assignment imbalance. Compilation and host dispatch stalls have not been isolated.

The warmed off run is 36% higher throughput than the earlier on run, but this is not a matched
warmed on/off comparison: restart, cache warmup, arena allocation and adaptive state differ.
Do not attribute that entire difference to replicas. Replicas remain off at the user's request.

## Follow-up diagnosis: adaptation leaves fixed prefill routing stale

Three additional replicas-off README replays (same 7,751 tokens, 256-token cap, zero prefix
reuse) measured 9.098 / 8.523 / 8.530 s, or 851.93 / 909.41 / 908.71 tok/s. Decode measured
17.98 / 17.24 / 17.13 tok/s. Each reported zero prefill expert misses and zero NVMe GB;
the requested-but-pruned routing miss rate fell 12.64% -> 12.03% -> 11.49%. Those counters
do not detect the bug below. `bench/readme_timing.py` records these observations with
per-rank phase and adaptation logs; it deliberately leaves adaptive loading enabled.

Code inspection found that startup builds `model.prefill_routes` (expert ID -> compact ID
-> arena slot), while `V41Engine.apply_swaps` updates only the pruning mask, arena/LRU and
`fast.lut`. It does not update that separate fixed-prefill mapping. With fixed routing on,
`Model.moe` passes this stale mapping to `moe_forward`, which uses it instead of the updated
slot LUT for large prefill calls. A promoted expert still maps to the null slot on its owner;
the other rank also maps it to null. Consequently its contribution disappears from prefill.

A CPU reproduction calling the actual `apply_swaps` on two mock ranks confirmed this:
swap 0->8 on rank 0 and 1->9 on rank 1; each promoted expert executes on exactly one rank
according to the updated decode LUT, but on **zero ranks** according to fixed-prefill routing.
No GPU context or serving-state mutation was needed to reproduce it. This establishes the
correctness bug, not the fraction of measured speedup it explains. Replica-enabled private
routing tables are rebuilt from current LRU state, so on/off comparisons can also differ in
how this bug manifests. All speed comparisons after adaptation need revalidation.

The required fix is to move each evicted expert's compact routing ID to the promoted expert
on its owner, map the evicted expert to null, and preserve the slot map and tensor addresses.
Test repeated promotions/evictions on both ranks against the decode LUT before redeployment.
Live stack sampling was unavailable without additional privileges; a full GPU timeline was
not collected. Further profiling should follow correctness repair, not interpret less valid
expert work as better performance.

### Repair

`apply_swaps` now transfers the outgoing expert's compact prefill ID to the incoming expert
on the owning rank and sends the outgoing ID to the final/null entry. The slot map, tensor
addresses and shapes remain unchanged. The peer's mappings stay null. This adds two device
scalar assignments per locally owned swap, not per token; expert loading and swap cadence
are unchanged. A boot configuration version field identifies the repaired behavior.

`engine.test_prefill_swap_routes` checks four successive two-rank swap generations, including
replacing previously promoted experts and restoring original experts. Every expert's prefill
mapping must equal the decode LUT; every kept expert must have exactly one owner, every pruned
expert zero owners, and the slot maps and tensor addresses must stay unchanged. The test fails
on the pre-fix `apply_swaps` and passes on the repair. The no-fixed-routing case also passes.
The 11-test combined swap-routing/replica/planning/coordination/chunk-timing suite passes, as
does `tools/test_swap_apply.py` (its mock needed the existing expert-generation counter).

The repaired image `c8a10691145e3049d355811e9601ae03f7a9f85183ba5771043242f93042a44f`
was built once, transferred, and verified running on both EP2 nodes after one restart.
Replicas/capture remain off, normal arena=88 GB, adaptive loading remains on. Full README
prefill after restart measured 11.864 s / 653.33 tok/s, then 9.046 s / 856.80 tok/s warmed
(7,751 tokens each, zero prefix reuse). Decode measured 17.48 / 20.65 tok/s. These are new
corrected-path observations, not a matched attribution of speed change to the repair.
Logs confirmed 2 then 14 expert swaps after those README runs and another swap before the
nesting controls. Both 6,082-token depth-8 and depth-10 nesting controls then passed exact JSON
grading, with no prefix reuse. Thus serving validation was after actual adaptation rather
than only freshly initialized tables. This bounded check does not establish broad quality parity.
