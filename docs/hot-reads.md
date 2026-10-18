# Hot reads: the heaviest missing experts, read at decode (DSV41_HOT_READS)

The pruned decode router can only pick resident experts: a non-resident pick is replaced by the
next resident one (`substitute`, docs/gotchas.md "Skipping a missing expert costs more than
substituting it"). Hot reads read the missing picks that carry the most routed weight from NVMe
and make them routable. Ported from the GLM-5.3 engine (`GLM_HOT`), where reading one heavy
missing pick a layer cut KL to the exact model 0.369 -> 0.135. Code: `engine/hot_reads.py`,
`FastDecoder._split_resolve` (sync) and `_hot_async_begin` / `_hot_async_end` (async).

| setting | default | meaning |
| --- | ---: | --- |
| `DSV41_HOT_READS` | 0 | 1 = on (pruned decode only) |
| `DSV41_HOT_ASYNC` | 0 | 1 = read after the step, routable from the next one (below) |
| `DSV41_HOT_MIN` | 0.2 | a candidate's share of its row's unpruned top-6 weight (the average pick is 1/6) |
| `DSV41_HOT_MAX` | 16 | reads a verify step |
| `DSV41_HOT_RING` | 32 | async: ring slots, taken from the transient ring (TRANSIENT_SLOTS >= 8 + ring) |
| `DSV41_HOT_PER_LAYER` | 1 | reads a layer a step |

All of them are in the boot guard (`hot_reads`). Async refuses `DSV41_MAX_CONCURRENCY` > 1.

## Two modes

**Sync** splits every pruned layer's decode graph at its router. Between the halves the host
reads the layer's candidates, takes the heaviest within a per-step budget paced over the layers,
resolves it into the transient ring, widens the keep mask and slot table, and graph B routes again.
It is exact for the current token, and it costs a host sync per hot layer plus the reads on the
critical path.

**Async** keeps the graphs whole. Each hot layer's router leaves its candidates on the device; the
step copies them to the host at its end. The next step plans from them (the heaviest `cap` across
all hot layers), reads them into its own ring on the io threads while its graphs run, and waits for
them after queueing the graphs. A read becomes routable from the step after that and stays
routable until the ring wraps over its slot. Marks are applied before a step's graphs and undone
after them, so nothing outside `FastDecoder.step` ever sees one.

## Quality (DeepSeek-V4.1-Flash-1017, teacher-forced)

`tools/bench_miss_policy_tp.py` on `corpus/heldout_eval.jsonl` (109 sequences, 51,778 scored
positions), serving's residency (fp4, arena 99.3 GB with a 200-slot transient ring = the 97.4 GB
profile's LRU), demand DB seeded from the old model's. Hot reads are emulated in prefill
(`hot_reads.PrefillEmulation`): blocks of 4 rows stand for a verify step, each with decode's paced
budget; `lag`/`hold` model the async read. Difference from `substitute`, nats, SE clustered by
sequence:

| arm | ring ~ | reads / token | coding | general |
| --- | ---: | ---: | ---: | ---: |
| substitute (no reads) | | 0 | 1.4979 | 2.2786 |
| sync, 0.3 / 2 a step | | 0.42 | -0.0092 +- 0.0022 | -0.0317 +- 0.0042 |
| lag 1, hold 16 steps, 0.3 / 2 | | 0.38 | -0.0056 +- 0.0017 | -0.0720 +- 0.0112 |
| async (lag 2), hold 16, 0.3 / 2 | 32 | 0.39 | -0.0034 +- 0.0015 | -0.0526 +- 0.0078 |
| async, hold 40, 0.3 / 2 | 64 | 0.36 | -0.0053 +- 0.0018 | -0.0981 +- 0.0203 |
| **async, hold 80, 0.3 / 2** | 128 | 0.34 | -0.0074 +- 0.0020 | -0.1379 +- 0.0282 |
| async, hold 16, 0.2 / 4 | 64 | 0.93 | -0.0051 +- 0.0022 | -0.0915 +- 0.0149 |
| async, hold 32, 0.25 / 3 | 64 | 0.62 | -0.0063 +- 0.0021 | -0.1271 +- 0.0279 |
| async, hold 64, 0.25 / 3 | 150 | 0.58 | -0.0084 +- 0.0025 | -0.1756 +- 0.0362 |
| exact (every missing pick) | | 43.5 | -0.0145 +- 0.0116 | -0.3546 +- 0.0627 |

"ring ~" is the ring that gives that hold at that arm's reads a step (hold x reads a 4-row block).
The emulation draws the ring from the bench's 200 transient slots, not from the residents; in
serving the ring comes out of the LRU (128 slots = 1.2 GB, 1.3% of the residents).

Three things follow. Pruning on 1017 costs prose ten times what it costs code (0.355 vs 0.015
nats), with residency ranked by the old model's demand history, which is code-heavy. The sync
read is worth more on code, where heavy misses are scattered one-offs; the async read more on
prose, where the same missing experts recur and a read serves every step it stays in the ring.
And holding is cheaper than reading: at 0.3 / 2, quadrupling the hold (16 -> 80 steps) gained
more on both corpora than doubling the reads (0.2 / 4), at fewer reads a token. Reads cost
decode time (below); a held slot costs only a resident.

## Speed (two-node gate, frozen map, concurrency 1, arena 92 GB)

`tools/generation_gate.py`, six prompts, 1,200 tokens each; token-weighted throughput:

| run | tok/s | vs off | reads | read time |
| --- | ---: | ---: | ---: | ---: |
| old model, off | 39.79 | | | |
| old model, sync 0.3 / 2 (first version) | 33.23 | -16.5% | 4,553 | 10.8 s |
| old model, sync 0.3 / 2 (no sync once the budget is spent) | 37.29 | -6.3% | 3,539 | 8.8 s |
| 1017, off | 41.08 | | | |
| 1017, sync 0.3 / 2 | 37.61 | -8.4% | 3,614 (2,040 from NVMe) | 7.7 s |
| 1017, async 0.3 / 2, ring 32 | 41.13 | +0.1% | 2,875 | 0.41 s waited |
| 1017, async 0.2 / 4, ring 64 (marks a layer at a time) | 39.30 | -4.3% | 6,822 | 0.54 s waited |
| 1017, async 0.2 / 4, ring 64 (one mask) | 39.22 | -4.5% | 6,822 | 0.15 s waited |
| **1017, async 0.3 / 2, ring 128** | **41.21** | **+0.3%** | 2,081 | 0.14 s waited |
| 1017, async 0.25 / 3, ring 160 | 39.74 | -3.3% | 3,693 | 0.13 s waited |

The sync mode's cost is the reads themselves (~2.5 ms each, 9.4 MB a rank) plus a host sync per
hot layer; prose pays the most per token because it accepts ~2 drafts a step against code's ~4.5.
44% of its reads were ring hits even with the 8-slot transient ring.

The async mode is free up to ~1.4 reads a step and not above it: 1.8 reads a step cost 3.3%, 3.4
cost 4.5%, with the host's own time (planning, marks, waiting for the reads) at ~0.7 ms a step
in all three. What costs is not on the host thread's clock -- the io threads' Python (each read
is ~5 chunked preads and a copy, under the GIL) or the extra distinct experts a step routes to;
not profiled. A longer ring also lowers the read rate: at 0.3 / 2, ring 128 read 2,081 times
against ring 32's 2,875, because the ring already holds what later steps want.

Live serving, same gate over HTTP, each model on its own serving configuration (adaptation,
probe prefill, prefix cache on; the gate's six requests are the first after boot):

| serving | tok/s | gate |
| --- | ---: | ---: |
| old model, image 20261011-topp, `.env` (concurrency 2, urgent adaptation) | 36.55 | 6/6 |
| 1017, image 20261018-1017-hot, `.env.1017` (concurrency 1, async hot reads, ring 128) | 38.52 | 6/6 |

1017 serves 6% below its frozen-map gate (41.21): its demand history is the old model's, so the
600-token decode adaptation pass fires in every long answer (0.5-0.8 s; 7-14% decode misses at
first) until the ranking has learned 1017's traffic.

## Serving setting

`DSV41_HOT_READS=1 DSV41_HOT_ASYNC=1 DSV41_HOT_MIN=0.3 DSV41_HOT_MAX=2 DSV41_HOT_RING=128
TRANSIENT_SLOTS=136` (`.env.1017`): no measured decode cost, 6/6 on the generation gate (the
frozen-map 1017 runs without hot reads failed its arithmetic prompt), and in the emulation
-0.0074 nats on code and -0.138 on prose. Stronger settings measured better (0.25 / 3 with a
150-slot ring: -0.0084 / -0.176) but cost 3% of decode.
