# Adaptive expert loading


The engine records the router's choices before pruning, blends that demand with a
routing trace, and replaces less-used resident experts. Swaps can happen after
prefill, every 600 tokens during long answers, and at the end of a request, subject to
the thresholds below. They do not retroactively recompute tokens that were already
processed.

Two settings control it; the engine derives the rest and logs what it chose at startup:

| Setting | Default in `.env.example.fp4` / `.exl3` | Meaning |
| --- | --- | --- |
| `DSV41_ADAPT_SENSITIVITY` | `high` | How far one request moves the resident set. See the levels below. |
| `DSV41_ADAPT_PRIOR` | `4` | Weight of the shipped routing trace, in votes (below). Lower lets this server's own traffic dominate sooner. |

A request votes twice: its prompt's demand at the prefill→decode boundary, its answer's at the
end (both engines). Each vote ages the history by one step, so a half-life in votes is about half
as many requests.

| Sensitivity | Demand half-life | Newest vote's share of observed demand | Miss gate (prefill and decode passes) |
| --- | ---: | ---: | ---: |
| `off` | never (ranking frozen) | — | — |
| `low` | 40 votes (~20 requests) | 1.7% | 4% |
| `medium` | 20 votes (~10 requests) | 3.4% | 2% |
| `high` | 10 votes (~5 requests) | 6.7% | 1% |
| `max` | 5 votes (~2.5 requests) | 12.9% | 0.5% |

A number in `[0, 0.5)` sets the newest vote's share directly. Higher sensitivity swaps
more experts per request (roughly 2x from `medium` to `high`). Judge it by the `routed-miss`
rate in the request logs, not by swap counts: too high a setting chases each prompt and makes
misses worse.

Fixed by the engine:
- demand counted per request, not per routing slot;
- swaps at the end of each request, and at the prefill→decode boundary when the prompt has at
  least 32 new tokens and misses at least the gate above;
- during long answers, a pass every 600 output tokens when that stretch misses at least the
  same gate (~0.2 s each, about 1% of decode time at ~20 tok/s). It plans with the answer's
  demand so far without adding an extra vote, and logs `decode adaptation at output token N`.
  `DSV41_ADAPT_DECODE_TOKENS` changes the interval; `0` turns only these passes off. Added
  2026-09-24 and unit-tested; its effect on miss rate in serving is not yet measured;
- at most 512 swaps per pass;
- a gain floor of 0.005 of the global mean score with dynamic allocation.

`DSV41_PRUNE_DB` (`results/prune_demand_req_score.npz` in the template) is the demand history, kept across
restarts. Keep it private and out of Git. With neither knob set, the legacy `DSV41_PRUNE_*`
settings are read exactly as before. With a knob set they are ignored and named in the log,
except that `DSV41_PRUNE_SWAP=0` and `DSV41_PRUNE_SWAP_PREFILL=0` still switch swapping off
(benchmarks use them to freeze placement).
`medium` reproduces the previous request-weighted profile exactly.

`DSV41_PRUNE_METRIC=score` ranks observed demand by the sum of the positive router
scores for its original, unmasked top-k choices. `frequency` remains the engine fallback; the example configuration selects `score`.
Score history is normalized per layer at each request-demand fold and aged alongside
counts; the same metric is used at startup and for prefill, decode, and idle swaps.
The shipped trace contains frequencies, so it remains the cold-start prior in score
mode. This measures router preference, not the magnitude of an expert's output.

Score mode defaults to `results/prune_demand_req_score.npz`. If `DSV41_PRUNE_DB` is
already set, give it a separate path explicitly. Old request databases contain
unnormalized scores and are rejected by score mode; frequency mode can still read
their counts. New databases carry score-history version 2. Both ranks check the
metric and version at boot, and rank 0 broadcasts history and swap plans.
`/health` reports `prune_metric`; request statistics report both selection
`miss_rate` and `score_miss_rate` under `prune_miss_request`. A lower score-weighted
miss rate still needs an answer-quality check.

Experimental `DSV41_PRUNE_LAYER_COUNTS` sets explicit per-layer resident counts
while preserving the total budget. `DSV41_STREAM_LAYERS` instead temporarily
loads current missing router picks before MoE in selected layers, including
decode, without evicting residents. Both are unset by default. See
[immediate layer streaming](layer-stream.md) for capacity and graph constraints.

Experimental `DSV41_USER_PROMPT_STREAM=1` streams experts for the latest actual
user text during prefill, then promotes and protects its strongest used experts
until the next request. Earlier messages,
structured attachments and tool results do not contribute to this temporary
priority signal. Decode remains resident. This text-only TP2 prototype requires
one streaming discovery prefill followed by one resident prefill after admission,
so the chosen experts affect the first answer token as well. It requires
concurrency 1, `PRUNE_MISS=1`, disabled prefix/response caches and prefill graphs,
and enough transient slots for every cold expert in one layer. Plain-text
documents bundled into the question need a client boundary.

`DSV41_USER_PROMPT_MAX_LOADS=100` limits temporary streaming cold expert loads
across one request. Default `0` is unlimited. Each visited layer selects cold
experts by score mass over its selected user rows; hits already in the transient
cache on both ranks are free. The selected cold set fits one transient batch,
preventing eviction/reloads within that call. Once the budget is spent, later
calls use ordinary resident routing. This is an online budget, not a global
importance ranking across future layers or tokens.
Normal prefill/decode/idle resident adaptation retains its existing rules
and does not spend this budget. The cap is inactive with `USER_PROMPT_STREAM=0`,
as in the current production pilot. Resident prefill makes no discovery expert
reads and avoids the extra priority rebuild. Uncapped full discovery can be very
slow on a long last-user field.

`DSV41_DYNAMIC_EXPERTS=1` shares the same resident arena sectors across all
layers. Startup fills the global budget from normalized demand, then latest-user
admission and later adaptive swaps can transfer a sector between layers.
Unset fixed layer counts and discounts. The only minimum is the router's top-k
(six residents per layer), while the total resident count and arena size stay
fixed. The global startup budget is the arena's: every LRU slot less 16, capped at
all routed experts (`engine/expert_budget.py::auto_budget`), so it follows `ARENA_GB`
and `EXPERT_FORMAT`. `DSV41_RESIDENT_EXPERTS=<n>` pins an exact count instead (for
experiments); it still needs enough arena space on both ranks and does not resize memory.
Discovery batches complete token rows so their cold experts fit the
transient ring; both TP ranks receive the same batch plan. Decode lookup tables
and masks update in place; changed layers' eager prefill directories are rebuilt.
This experimental mode supports ordinary resident prefill with
`USER_PROMPT_STREAM=0`, also with concurrent requests (docs/concurrency.md). It requires adaptive pruned residency,
`PRUNE_MISS=1`, resident LUTs, no replicas/fixed profiles/layer streaming, and
response caching and prefill graphs disabled. With prompt streaming off,
`DSV41_PREFIX_CACHE=1` can reuse RAM prefixes and `DSV41_PREFIX_DISK=1` disk ones (since
2026-10-14; with `DSV41_PREFIX_DISK_STRICT=1` a cross-layer swap refreshes the routing signature
bundles match on). These retain historical encoder
state after expert swaps, as in the original adaptive policy; they do not
recompute old tokens under the new expert selection. Latest-user streaming
still requires all prefix caching off. Global startup and ordinary demand-based
cross-layer adaptation do not require prompt streaming.

`TRACE_STATS` chooses the initial `coverage.json`; when unset, the launcher looks
under `results/trace-*/stats/`. Keep the demand database private and out of Git.
To freeze expert placement for an A/B test, set `DSV41_ADAPT_SENSITIVITY=off`.

Optional [fixed topic profiles](expert-profiles.md) use the author's 39-topic
statistics to select residents by frequency or gate-weighted expert-output norms.
The `maxmin` ranker repeatedly helps the least-covered topic, rather than letting
one topic dominate the budget. This changes the resident expert IDs while keeping
the native weights and inference kernels. These profiles require all adaptation
and swap triggers off; their admission priorities cannot be mixed with the
router-score history. Defaults preserve the existing adaptive trace path.

## Learned fresh-start distribution

The engine-only example uses a 9,800-resident budget and seeds demand from
`profiles/learned-experts-v1.npz`, captured from the live TP2 working set on
2026-10-06. It includes the exact expert IDs per layer and aggregate request
counts/router-score mass, without prompts, responses, KV state or weights.
The companion JSON records the snapshot, layer counts and checksum.

This seed is used only when no local demand DB exists, with dynamic allocation,
request-normalized score history and adaptation enabled. Existing DBs always
win, including incompatible files handled by the normal recovery path. At the
original 9,400 budget the exact resident map is restored; other budgets use the
seed history with the normal global selector. Subsequent traffic updates the
new user's own history through the usual persistence path. Rank 0 broadcasts
the seed and ranking; the boot guard includes its checksum and protocol version.
Set `DSV41_EXPERT_SEED=0` to start from the original trace, or to a seed file's
path to start from another one. This is a useful starting point, not a claim
that one person's distribution suits every workload.

The v1 seed was learned on the original V4.1-Flash. `tools/export_expert_seed.py`
packages a running pair's map and demand DB in the same format (run it in the
rank-0 container while the server is idle); a 1017 seed made that way goes in
`profiles/` and the 1017 profile names it with `DSV41_EXPERT_SEED=<path>`.

### GLM-5.3

GLM's history is `GLM_STATS_DB`: router mass per (layer, expert) and a token
count, added to after every adaptation and read only at boot, to place the
resident set. Without it, a fresh install placed the same number of experts in
every layer by the router's load-balancing bias. Now a missing DB starts from
`profiles/glm53-experts-v1.pt` (2,339,364 tokens: a trace of the routing corpora,
`tools/glm_train_db.py` over them, and live serving through 2026-10-19; the
`.json` records where it came from and its checksum). The first save writes it to
`GLM_STATS_DB`, and the user's own traffic accumulates on top. An existing DB
always wins; `GLM_STATS_SEED=0` starts empty, and a path starts from another
seed. The history does not decay, so for a long time a new install's boot
placement leans on the seed. Within a session, adaptation follows its own
decayed demand regardless.

What it is worth on the first request after a fresh boot (2026-10-19, EXL3
profile at 320K, 12,000 residents, adaptation on; `tools/glm_quality.py --mode
eval --set tokyo --ref x3tk26 --rows 3`, 900 positions, scored against exact
routing; GLM_STATS_DB missing at boot in both arms):

| boot placement | KL to exact | routing weight missed | hot reads | top-1 agree |
| --- | ---: | ---: | ---: | ---: |
| router bias (`GLM_STATS_SEED=0`) | 0.104 | 12.6% | 1,247 | 93.6% |
| bundled seed | 0.093 | 9.9% | 1,058 | 93.3% |

One prompt set; top-1 differs by 2 of 900 positions, within noise.
