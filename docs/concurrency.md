# Concurrent requests (`DSV41_MAX_CONCURRENCY`)

Default serving is one request at a time. `DSV41_MAX_CONCURRENCY=N` (2..8, TP2 only) serves up to N
requests at once:

* **Requests queue** (`server/concurrency.py`). Each is admitted to the smallest free lane whose
  context holds `prompt + max_tokens + 8`; lane 0 keeps `MAX_SEQ`, lanes 1..N-1 allocate
  `DSV41_LANE_MAX_SEQ` (default 64K). A long request waits for lane 0 without holding back later
  requests that fit a free short lane. Up to 32 wait; past that the server answers 503.
* **Prefill is sequential.** Admission runs a lane's whole prefill at once; the other lanes pause
  at their verify boundary until it finishes.
* **Decode is batched weight by weight** (`engine/batch_decode.py`). Every round, all lanes at a
  verify boundary take one step together: each weight-bearing operation runs once over every
  lane's rows, so a weight is read once per round instead of once per request. A round with one
  ready lane runs that lane's own single-request graphs instead (below).

The adaptive features run with lanes in flight (see "Adaptive features" below): dynamic expert
allocation, probe prefill, dynamic speculative depth, decode-time and urgent adaptation, the RAM
prefix cache. Still refused at boot: latest-user and layer streaming, critical prefill, prefill
graphs and replicas, lookup drafts and draft bypass. Confidence depth is allowed but has not run
concurrently; the disk prefix cache is wired as before but was not exercised.
`DSV41_MAX_CONCURRENCY=1` constructs none of it.

## One round

Per backbone layer, in stream order (`BatchDecoder._program`):

| graph | runs | what |
| --- | --- | --- |
| E | once | Engram `wkv` projection (layers 1, 14) |
| P1 | per lane | previous layer's MoE residual, Engram combine, HC attention mixes, norm |
| A | once | `wq_a`‖`wkv`, `q_norm` (per lane, row-wise), `wq_b` |
| C | per lane | RoPE, window ring, compressor, indexer, key block, attention -- keyed by (start parity, index bucket) |
| O | once | `wo_a`, `wo_b` |
| P3 | per lane | attention residual, HC FFN mixes, norm, router (expert-demand counters) |
| M | once | routed experts (one routing over every lane's pairs: an expert two lanes picked is read once) + shared expert |

then a per-lane final norm, one LM head over all rows, and per-lane drafter-ring seeding. The
per-lane pieces are `FastDecoder`'s own code at its own 6-row shapes; the shared pieces call the
same kernels with N x 6 rows, on decode tiles widened to 32/64 rows inside `decode_rows.wide()`
(`tools/decode_rows.py`). Outside that context nothing changes, so single-request decode and
prefill keep their shapes.

Rules the design depends on (each is also a comment at its site):

* **Idle lanes.** Shared graphs always cover all N lanes; an idle lane's rows hold stale values.
  No shared op mixes rows, and M gives an idle lane's expert pairs routing key -1, which both expert
  kernels skip -- an idle lane reads no expert.
* **Side effects are per lane.** KV caches, compressor state, drafter rings and the shared demand
  database are touched only by per-lane graphs, replayed only for lanes with a request.
* **Each lane's C graphs have a private graph pool.** C keeps per-step values (positions, window
  slots, indexer rows) in pool memory across layers; in a pool shared with graphs replayed in
  between, the allocator could hand that memory to another graph's scratch.
* **One ready lane runs solo.** The batched step is bit-identical to `FastDecoder.step`, but with
  one of four lanes active it costs 87.7 ms against 71.0: it carries all four lanes' rows and
  ~6 graph launches a layer. A round with one ready lane therefore replays that lane's own graphs;
  the switch changes no bits.

## Adaptive features with lanes in flight

Each of these was refused above concurrency 1 until 2026-10-14; none needed a lane-specific
mechanism beyond two:

* **Rank 0's per-step decisions reach rank 1.** A lane's `ScheduledPeer.control` is the real Gloo
  broadcast (three ints) that single-request decode uses: it carries the speculative depth and the
  decode-adaptation flag. Both ranks reach it inside the same scheduler action, so it pairs up.
* **Swaps edit shared state in place, at a boundary.** Within-layer and cross-layer
  (`apply_global_swaps`) swaps rewrite the slot table, the keep masks and the prefill route maps
  in place -- the objects every lane's graphs read -- and run only while every other lane waits
  at its verify boundary: during one lane's prefill (probe prefill, inline adaptation), during
  one lane's advance (decode-time, urgent), or between requests. Engine attributes a swap
  *reassigns* (`_swap_baseline`, `resident_layer_counts`) are carried between the primary and the
  lane that ran it.

Dynamic depth puts lanes at different widths in one round: a depth-3 lane's 4-row block runs in a
6-row slot whose last two rows repeat its last token, are routed to no expert, are not recorded
as demand, and are rolled back like rejected drafts. Each lane has its own depth policy (a shared
one was reset by every new request).

Batched rounds are left out of each policy's step times (`_policy_step_s`; see "Open" below):
a round's wall time is set by every lane in it. The policy still learns acceptance from them, and
decides from its solo step times.

Rejected 2026-10-14: capping a lane at depth 3 while another lane decodes (`DSV41_BATCH_DEPTH`).
The reasoning was that depth 5's two extra rows a lane add experts the other lane does not share.
Measured on the live server (table under "Speed"), it cost the code pair: alone, both Python
requests ran mostly at depth 5 at 4.3-4.65 tokens a step; capped, 3.4-3.6, and the pair came out
at 0.88-0.92x of one after another. Prose lost nothing (acceptance ~2 at either depth). Reverted;
each lane keeps its own policy in batched rounds.

Measured 2026-10-14, TP2, arena 90.2 GB, four lanes:

* **Gate** (`tools/gate_batch_decode.py`, run i): dynamic experts and dynamic depth 3/5 on, swaps
  frozen (`DSV41_ADAPT_SENSITIVITY=off`, so probe prefill off) -- four concurrent greedy requests
  token-identical to serial, batched logits bitwise equal to single-request steps with lanes at
  mixed depths, sampling isolation (depth pinned for that check: a timing-chosen depth changes a
  sampled request's random draws), cancellation and reuse exact. Serial 37.9 tok/s, together
  41.8 (1.10x; 1.20x in the previous run, h).
* **HTTP, the full `.env`** (`tools/serve_bench_tp.py`, run b): probe prefill, global swaps,
  decode-time and urgent adaptation all live; four prompts x 256 tokens. 37 global sector-transfer
  passes (36 across layers in the first), applied identically on both ranks; urgent decode
  adaptation 5 times mid-generation with other lanes in flight (64 slots, ~280 ms each); depth
  switches 3 <-> 5; /health ok throughout. Serial 29.2 tok/s, together 33.9 cold, 38.7 and 37.4
  warm (~1.3x).

Open, cause found 2026-10-14 (fix not yet measured): with dynamic depth, one request alone on the
runtime ran at 42.8-43.4 tok/s against 49.0-52.0 on the single-request path (same prompt), where at
fixed depth the two were within 1.5%. Each lane's policy timed its steps across the batched rounds
before it, where a round's wall time is set by every lane in it: the gate log has a lone request
deciding with depth 3 estimated at 17.0 tok/s while it ran at ~50, and lanes in a batched round
estimating depth 3 at 11-17 tok/s against depth 5 at 32-56 (a solo-timed depth 5 against a
batched-timed depth 3). A depth the policy stops choosing keeps such a stale time. Batched rounds
no longer train the policy.

## Correctness

`tools/gate_batch_decode.py` (TP2, frozen expert map, static residency, no prefix caches, fixed
depth 5, 4 lanes, four different prompts, 2026-10-14; passed in runs d, e and f; the
adaptive-feature runs are above):

* Each lane's batched-step logits equal that lane's single-request `FastDecoder.step` at the
  same state **bitwise** (max |diff| 0.0, all four lanes).
* Four greedy requests decoded together equal each decoded alone on the normal single-request
  path, **token for token** (4 x 192 tokens).
* Seeded sampling is unchanged by interleaving; cancelling one lane leaves the others exact; reused
  lanes are exact; the expert map is unchanged.

`tools/test_decode_rows.py` checks the kernel claims underneath: every row of a 12-60 row FP8
dense, fused-act-quant, grouped (`wo_a`) or packed-head call equals the same row in a 6-row call
on the 16-row tile, under `DSV41_FP8_DECODE_BLOCK_N` 32/auto and `_BLOCK_K` 128/256; the 65-512
pair router equals a stable sort; four lanes' experts in one call equal four calls.

## Speed

All TP2 native FP4 with the live profile's kernels (FP8 dense, packed head, staged attention,
BF16 router), arena 84 GB, keep 0.57, static residency, fixed depth 5, lane 0 at 64K and lanes
1-3 at 32K, 2026-10-14.

**Verify step** (`tools/bench_batch_decode_tp.py`: one state, every lane at its first verify
boundary with a different prompt, median of 5, CUDA events):

| lanes | k serial single-request steps | one batched step | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 71.0 ms | 87.7 ms | 0.81 |
| 2 | 143.8 | 122.2 | 1.18 |
| 3 | 210.2 | 157.9 | 1.33 |
| 4 | 277.5 | 199.4 | 1.39 |

A repeat run gave 1.388 at four lanes (289.5 / 208.5 ms); absolute times moved ~4% between runs.

GPU time of the batched step by graph, one lane active vs four:

| | A | O | H | E | C | P1+P3 | M | sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 lane | 6.6 | 11.0 | 2.8 | 1.4 | 5.5 | 5.1 | 47.9 | 80.7 ms |
| 4 lanes | 7.0 | 11.8 | 3.0 | 1.4 | 25.1 | 18.3 | 125.5 | 193.3 ms |

The shared dense work (A, O, H, E: ~22 ms) does not grow with lanes -- that part works as
intended. What limits the gain is M: four different prompts route to largely different experts,
so the expert bytes grow with the union of their picks (~3.1x one lane's for four lanes here).
Identical or same-domain prompts share more. C and P1/P3 are per lane by design: many small
kernels at ~5-6 ms a lane, not profiled further.

**End to end** (the gate's run f: four greedy requests x 192 tokens, `ignore_eos`):

| | wall | aggregate |
| --- | ---: | ---: |
| one after another, single-request path | 24.05 s | 31.9 tok/s |
| all four together | 19.49 s | 39.4 tok/s (1.23x) |
| one request alone through the runtime (solo path), same prompt | 40.4 tok/s | vs 41.0 single-request |

The together run took 77 rounds: 196 ms of batched-verify GPU time and ~57 ms of everything else
per round. That remainder is each lane's own work between verify steps -- its draft graph (~5 ms),
the hash-id and acceptance syncs, Engram reads, the accept loop -- run lane after lane.

**HTTP** (`tools/serve_bench_tp.py` + `tools/bench_concurrency_http.py`: the real server and
scheduler, static residency with request-unit swaps and the RAM prefix cache (run a, before the
adaptive features were allowed), four prompts x 128 tokens,
`ignore_eos`, arena 90.2 GB, keep 0.61): one after another 21.28 s (24.1 tok/s); together 18.13 s
cold (graphs captured), 16.45 s warm (31.1 tok/s, 1.29x). `/health` reports the lanes, their
contexts, the active set, the queue depth and the batch graph counts.

**Two streams on the live 512K profile after keeping graphs, with batched depth capped at 3 (since
reverted)**
(2026-10-14, `tools/bench_concurrency_http.py --tokens 512 --rounds 2` against the running server,
`DSV41_MAX_CONCURRENCY=2`, arena 97.4 - 1.89 GB, full `.env`; results/serve-bench-20261014-live-new):

| pair | one after another | together (two rounds) | tokens/step alone -> together |
| --- | ---: | ---: | --- |
| Python + Python (prompts 2, 8) | 44.5 tok/s | 39.1, 41.0 (0.88-0.92x) | 4.30, 4.65 -> 3.39-3.59 |
| Python + story (prompts 2, 3) | 29.8 | 28.5, 30.9 (0.96-1.04x) | 3.99, 1.92 -> 3.35-3.38, 1.99-2.05 |

Graph keeping worked: 3 batched captures (0.57 s) for the whole run, no core or shared release;
lowest MemAvailable 8.09 / 5.37 GiB (rank 0 / 1). Two streams together still did not beat one
after another. A two-lane round took ~160 ms against ~92 ms for one stream's solo step, so the
round costs ~1.75x a solo step -- the verify-step bench's 122 / 71 ms (1.72x) plus each lane's
draft and accept work run one after another. And the depth cap cost the code pair: alone, both
Python requests ran mostly at depth 5 (4.3-4.65 tokens a step); capped at 3, 3.4-3.6. Prose
lost nothing (its acceptance is ~2 at either depth).

## Memory

Measured per extra lane at 32K context (`tools/bench_batch_decode_tp.py`):

| | MB |
| --- | ---: |
| caches: window ring (`DSV41_RING` 4096 x 43 layers x 512 bf16), fixed | 180 |
| caches: compressed KV + indexer keys (~1.36 KB a token) | 45 |
| model: RoPE tables (~0.5 KB a token) | 17 |
| decoder buffers (after sharing the BF16 router gates and Markov rows across lanes) | ~20 |

Three extra lanes at 32K allocated 0.75 GB (1.28 GB before the BF16 router and Markov copies were
shared: those were 168 + 21 MB a lane of identical data). Batched graphs added ~0.25 GB reserved.
Estimated from the breakdown: ~0.35 GB a lane at 64K, ~1.2 GB at 512K.

**Lanes come out of the arena.** The profiles size `ARENA_GB` so one request settles with a few GB
free; the engine's boot check only caps an arena larger than what is available at load, which a
tuned arena is not, and the lanes are built after it. So with `DSV41_MAX_CONCURRENCY > 1` the
engine takes `serving.lane_reserve_bytes` out of an explicit arena and logs it: extra lanes (ring,
KV, RoPE, buffers, a graph allowance that grows with the lane's context), lane 0's
attention-core scratch at its context, and the shared batched graphs. 1.89 GB for two lanes on
the 512K profile (lane 1 at 64K); `DSV41_LANE_RESERVE_GB` replaces the per-lane graph allowance.

**Graph memory is bounded, not released per request.** A lane keeps one context bucket of
attention-core graphs (`BatchDecoder._drop_stale_core`: capturing a new bucket drops the old one,
both parities of a bucket stay); a short lane's own single-request graphs are bounded by
`DSV41_GRAPHS_MAX` like lane 0's; the shared batched graphs (~0.25 GB) stay, unless
`DSV41_BATCH_IDLE_STEPS` > 0 releases them after that many single-lane steps in a row; the batched
warm-up's scratch is returned after every capture. All of it is inside the lane reserve. Each
release is decided from state both ranks share (the lane's position and bucket) -- a graph only one
rank dropped would be recaptured with an eager warm-up that runs the step's collectives one extra
time, and the pair would hang.

The first version of this policy, which the measurements below were taken with, released a lane's
graphs when its request ended and the batched graphs after 128 solo steps. That was safe but cost
every concurrent pair a recapture: 1.1-1.4 s a pair on the 2026-10-14 serving trace, ~8% of a pair
of 200-token answers.

Why: on 2026-10-14 a two-lane session on the 512K / 98.4 GB profile (two clients, then one client
with a growing conversation) took rank 1 to the memory watchdog after ~17 minutes (MemAvailable
2.4 GB < 2.5 GB floor). Without either fix, nothing was ever released and nothing was reserved.

Measured after the fix (`tools/serve_memory_tp.py`: two clients, then one conversation growing to
86K tokens over five turns, then long and short together; lowest MemAvailable per phase):

| | rank 0 | rank 1 |
| --- | ---: | ---: |
| concurrency 1, arena 97.4 GB | 5.95 GiB | 3.46 GiB |
| concurrency 2, arena 97.4 - 1.89 GB | 6.74 GiB | 6.08 GiB |

Boot-time MemAvailable differed by ~2-4 GiB between the two runs, so compare drops from boot to
the low: concurrency added ~1.5 GiB on rank 0 and ~1.2 GiB on rank 1, inside the 1.76 GiB
reserve. Note rank 1 at concurrency 1: 3.46 GiB, ~1 GB above the watchdog floor -- the profile
itself has little room left for anything else (a disk-prefix save stages its bundle in host
memory first).

## Negative results and defects found

* **Idle lanes as null-slot pairs.** Routing an idle lane's pairs to the null slot with unique keys
  (the convention for live pairs) made the down kernel write zeros for every one of them: one of
  four lanes active cost 56.2 ms in M; key -1 (skipped) brought it to 47.9.
* **`DSV41_ROCE_MAX_KB=1024`** to keep the 387 KB MoE gather on RoCE: every graph got slower,
  199.4 -> 253.2 ms at four lanes. Keep 256.
* **64-row decode tiles** run at 1.5-2x the 6-row time (cold L2, `wq_a`‖`wkv` 62 -> 139 us), against
  1.05-1.4x for 32-row tiles: past five lanes at depth 5 the dense calls lose much of their sharing.
  Depth 3 (`DSV41_BLOCK=3`, 4 rows a lane) fits eight lanes in 32 rows; not measured.
* **Two defects in the lane code this replaced**, both silent or rank-1-only: every lane shared one
  verify-block buffer (`_vbufs`), and rank 1 chose a root-only verify because the cooperative
  control value is never broadcast (`KeyError: 2` on rank 1's first decode step here). The old
  two-lane runtime has the same code paths.

## Not done

* Drafts run per lane, one after another (~5 ms each); a batched draft pass and queueing every
  lane's draft before any host sync would attack the ~57 ms a round outside verify.
* C, P1 and P3 are per lane; batching them needs stacked caches and row-invariant HC/router GEMMs.
* Prefill is not interleaved with decode: a long prompt stalls every decoding lane until it
  finishes (as asked: prefill is sequential).
* Batched rounds decide depth from solo step costs. A policy that timed batched rounds on their
  own (a second set of costs) would weigh depth 5's extra experts; not built.
* Measured at four lanes on short prompts only; long contexts and 8 lanes are unmeasured.
  Periodic decode adaptation (every 600 tokens) did not trigger in the 256-token HTTP run.

## Running it

```bash
# profile: DSV41_MAX_CONCURRENCY=4, DSV41_LANE_MAX_SEQ=65536, and room for the lanes in
#          ARENA_GB/MAX_SEQ; the adaptive settings stay as they are
python -m unittest tools.test_decode_rows server.test_concurrency      # kernels + scheduler
GATE_IMAGE=<image> GATE_LOG_DIR=results/<dir> GATE_SOURCE_ROOT=<snapshot on both hosts> \
  GATE_ENV="DSV41_MAX_CONCURRENCY=4" GATE_TIMEOUT=2400s \
  bash tools/run_two_node_gate.sh gate_batch_decode.py --out /app/results/<dir> \
  --arena-gb 90.2                          # also: bench_batch_decode_tp.py, serve_bench_tp.py
```

## History: the two-lane prototype (2026-09-15)

The first implementation (`engine/batch2.py`, removed) batched only the routed and shared experts
of two lanes and ran everything else per lane. Its kernel-level gate measured 1.08-1.75x for two
identical prompts and its HTTP check 1.19x for two different ones, with an earlier version of the
surrounding engine; as of 2026-10-14 its code had the two defects above. Not comparable with the
tables here.
