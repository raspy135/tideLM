# Urgent expert loading during decode

> **Removed 2026-10-18.** Its trigger and loading were validated as safe (below) but never shown
> to improve output, and each pass paused decode ~0.2 s. Asynchronous hot reads
> (`DSV41_HOT_READS=1 DSV41_HOT_ASYNC=1`, docs/hot-reads.md) read the experts a decode step misses
> most while the next step runs, at no measured throughput cost, with a measured held-out NLL gain.
> `DSV41_ADAPT_URGENT=1` now refuses to boot; `0` is accepted. The periodic 600-token decode
> adaptation is unchanged. This page is kept as the record of what was measured.

`DSV41_ADAPT_URGENT=1` adds a rolling check to the existing decode adaptation path:

- More than 10% missing expert selections over approximately the last 30 output tokens.
- At most 64 expert swaps per urgent pass, using the normal gain-ranked planner
  with the current request's demand previewed, without adding another historical vote.
- At least 150 output tokens between attempts, including attempts whose plan is empty.
- Rank 0 decides and sends flag 2 through the existing per-step control broadcast;
  both ranks execute the same plan. Periodic adaptation uses flag 1.

The default is on when the two-knob adaptation configuration and decode adaptation
are enabled. It remains off with legacy-only settings, frozen placement, sensitivity
off, or `DSV41_ADAPT_DECODE_TOKENS=0`. Set `DSV41_ADAPT_URGENT=0` and restart both
nodes to disable only the urgent monitor; the usual 600-token checks remain.
The resolved trigger, window, cap and cooldown are included in the boot guard.

## What is measured

The router already records a decode-only `(missing selections, total selections)`
counter pair. Rank 0 reads those 16 bytes once per completed verification burst.
Monitoring adds no expert loads or per-layer recording kernels. The rolling
history is bounded, as are the last 64 diagnostic checks returned in request stats.

The first output token comes from prefill. Its routing is excluded. Verification
bursts can produce several output tokens, so the window begins at the nearest
completed-burst boundary at or before 30 tokens ago. It can be slightly wider
than 30; it is not a lifetime or whole-request average. The counts include verified
draft positions, including rejected drafts, as the existing routed-miss metric does.
They are a residency-pressure signal, not a measured quality score.

Graph warmup executes extra forwards. When new decode graphs appear, the urgent
window restarts and waits for another full window rather than treating warmup
routes as generated tokens. A loading attempt also restarts the window and the
regular periodic interval, avoiding an immediate second pass on stale evidence.

The planner still respects its historical prior, positive-gain requirement and
fixed resident-slot budget. An urgent trigger may find no beneficial swaps.
It previews the current request's demand, not exclusively the last 30 tokens.
Routing-weight counters exist but are not currently a rolling missing-weight
metric; this change does not modify their kernels or substitute an unqualified
quality proxy. Higher quality is not guaranteed by a lower selection-miss rate.

`x_engine_stats.decode_adaptation` reports recent checks and loading passes,
including token positions, miss rates, expert counts and elapsed loading time.
Health configuration exposes `urgent_adaptation`. Past output/KV stays intact;
later tokens can change because they have access to a different resident set.

## Validation

CPU tests cover strict threshold semantics, trailing-window expiry, speculative
burst boundaries, cooldown, missing counters, graph warmup, the 64-swap cap and
configuration guards. Existing planner and swap-application invariants are also
checked. `tools/bench_urgent_adapt_tp.py` separately measures monitor-only overhead
with fixed speculative depth, forces the trigger while exercising real loading,
and runs natural-miss workloads. Forced-counter runs are explicitly safety tests,
not evidence of natural miss rates or a quality improvement.

The initial two-node gate passed 23 requests with identical token sequences on
both ranks. Monitor-only on/off output was also identical with fixed depth 3.
Natural Python triggered at tokens 32 and 185, natural prose at 37, and natural
story at 35 and 189. Each pass loaded 64 experts. Natural-pass wall times were
0.187-0.200 seconds; the first forced-counter pass took 0.302 seconds and later
forced passes 0.191-0.194 seconds. The 150-token cooldown and resident masks agreed
on both nodes, including after real swaps and reverse restoration between tests.

Evidence: `results/urgent-adapt-20261003/`. Its initial 128-token warmups left some
later Engram rows cold in the first 512-token baseline. Those first timings are
retained, but must not be used to claim a monitor speedup. A separate fully warmed
prose comparison is used to estimate monitoring overhead.

These checks establish trigger/load safety, not a measured quality gain. The
urgent planner changes future routing, so adaptation-on output is not expected
to match adaptation-off output. Loading pauses also count against throughput:
at 25-40 output tokens/s, a 0.20-second pass every 150 tokens would add roughly
3-5% to wall time if the trigger stays persistently high. A response with no
trigger pays only monitoring cost.

The follow-up fully warmed prose gate used two complete 512-token warmups before
off/monitor/monitor/off timing, fixed verification depth 3, and identical 512-token
outputs (235 steps) in every run. Off runs were 24.13 and 24.84 tok/s; monitor runs
were 24.66 and 24.78 tok/s. Medians: **24.485 vs 24.720 tok/s**. The apparent +1.0%
is smaller than the baseline run-to-run spread and is not claimed as a speedup.
No noticeable monitoring penalty was established on this workload. Both ranks
passed all six requests. Evidence: `results/urgent-overhead-20261003/`.

## Deployment

Both nodes use `deepseek-v41-flash-spark:urgent-adapt`, image
`sha256:c893217dfa1cc72992605663226f1658145d0494935467e17d2c56ca9afaea0a`.
The head and peer serving configuration explicitly set `DSV41_ADAPT_URGENT=1`.
Confidence scheduling, 2048/4096 MiB Engram caches, 524288 context and port 8000
are retained; prefill graphs remain disabled. Normal start/stop scripts apply.

Live API checks passed for greedy and sampled requests after deployment. Greedy
triggered at output token 57 after graph warmup, loading 64 experts in 0.198 s;
sampled triggered at token 31, loading 64 in 0.187 s. Both enforced the cooldown.
The last observed windows were below 10%, but these changed-output trajectories
are not a controlled quality or causal miss-rate comparison. Health reported the
feature enabled and both running image IDs matched. Saved verification:
`results/urgent-overhead-20261003/service-enabled.json`.
