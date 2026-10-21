# Dynamic speculative depth

`DSV41_BLOCK_DYNAMIC=3,5` lets the engine choose the draft depth per request, as acceptance
changes. Code off by default; both `.env.example.*` profiles enable it. It is in the EP2 boot guard, and it
cannot be combined with `DSV41_BLOCK`. Concurrent lanes (docs/concurrency.md) receive rank 0's depth through their
control broadcast and each keep their own policy.

## Why

The best fixed depth depends on how predictable the text is. The measurements below come from
TP2 (served config), greedy, 512 tokens, two samples per depth in the order 3, 5, 5, 3
(`tools/bench_decode_block_tp.py`, `results/decode-block-20260923-2223/`). The last column is
tokens per step at depth 3 → depth 5.

| Workload | Depth 1 | Depth 3 | Depth 5 | Tokens per step (3 → 5) |
| --- | ---: | ---: | ---: | --- |
| python | 22.4 tok/s | 35.8–36.1 | 44.3–44.6 | 3.87 → 5.65 |
| html | 21.6 | 31.9–32.2 | 34.0–35.0 | 3.38 → 4.29 |
| explain | 18.2 | 20.3–20.5 | 18.3–18.5 | 2.17 → 2.31 |
| story | 17.6 | 18.5–18.9 | 16.8 | 1.97 → 2.07 |
| story, temperature 0.7 | — | 16.7–17.4 | 14.2–14.5 | 1.96 → 1.92 |
| step time | 88 ms | ~106 ms | ~124 ms | |

- Depth 5 makes every step ~16% slower, and pays that back only where the drafter is right.
- Depth 1's step is 18% faster, but its 2-token cap loses on every workload. It was measured
  with a single sample and rejected.
- Depth 7 was not measured. It lies past the drafter's trained horizon of 5; see
  `docs/gotchas.md` on the wider verify block.

## How

1. The drafter always drafts the deeper block. The checkpoint was trained at 5.
2. The verifier checks the first `depth` drafts. Each verify width has its own static buffers
   and CUDA graphs (`FastDecoder._bind`), and the graph key is (parity, bucket, width).
3. A width's graphs are captured only at the first step that uses it. Capturing a wider width
   ahead of time would be wrong: its warm-up forward writes KV past the step, and in the
   128-slot window ring that overwrites entries the current step still attends to.
4. `engine/spec_depth.py` holds the policy. It re-decides at most once per
   `DSV41_BLOCK_DYNAMIC_TOKENS` (default 60) output tokens and needs a 3% win to switch.
   - **At depth 5**, it knows the depth-3 alternative exactly: `min(accepted, 3) + 1` tokens per
     step.
   - **At depth 3**, it estimates depth 5 from saturated steps, using the extra acceptance
     learned in this request.
   - It compares tokens per second, using step times measured on this process.
   - A request starts at `DSV41_BLOCK_DYNAMIC_START` (default 3).
5. Rank 0 decides. `EPDistributed.control()` now always broadcasts `[keep_going, depth]`,
   including on abort paths, so both ranks verify the same width.

Greedy output is identical under any depth schedule, because verification accepts exactly the
target model's argmax. With sampling the output distribution is unchanged, but the random stream
depends on the schedule.

## Full-engine check, 2026-09-23

`tools/bench_decode_dynamic_tp.py`, TP2, results in `results/decode-dynamic-20260923-2300/`.
This run includes the decode projection defaults (`docs/decode-projection-fusion.md`). For each
greedy workload, token-identical output was checked on both ranks across four modes:
alternating every step, pinned 3, pinned 5, and adaptive. All matched.

| Workload | Pinned 3 | Pinned 5 | Adaptive | Adaptive: steps at 3 / 5 |
| --- | ---: | ---: | ---: | --- |
| python | 35.5 | 44.5 | 43.1 | 15 / 81, 1 switch |
| html | 31.0 | 35.0 | 34.3 | 17 / 108, 1 switch |
| explain | 20.4 | 18.2 | 20.6 | 235 / 0 |
| story | 19.1 | 16.8 | 19.1 | 228 / 0 |
| story, temperature 0.7 | 18.1 | — | 19.4 | 250 / 0 (both at depth 3) |

- Adaptive gets about 97% of depth 5 on code. The gap is the first ~60 tokens, before the first
  decision.
- It never leaves depth 3 on prose.
- Drafting 5 and verifying 3 did not lower depth-3 acceptance compared with true fixed-3 runs:
  3.36 vs 3.38, 3.87 vs 3.87, 2.18 vs 2.17.
- Peak allocated memory was 103.147 GB, versus 103.138 GB without the feature. Both widths
  share one graph pool.

**Not measured:** thinking-mode traces (the policy should keep them at 3 if they accept like
prose), long contexts, and requests that switch between code and prose several times.

## Idea, not built: per-step depth from the DSpark confidence head (2026-09-29)

DSpark has a confidence head that predicts the conditional acceptance of each draft position
(V4.1 tech report 2.4.3; DeepSeek's scheduler turns it into a per-step verify length).
TensorFold's Qwen3.8 Flash Next CUDA engine also uses confidence to shorten drafts
(`--mtp-confidence`), but its score is the sampled draft token's temperature-1
probability, not DSpark's learned acceptance score. This engine chooses depth
from a recent window of acceptance, so the question was whether the head is
good enough to choose it per step.

`DSV41_SPEC_CONF=1` (off by default) makes the draft graph compute the head's logits and
records `(depth, leading accepts, logits[5])` per step in `last_stats["spec_conf"]`. It is
observation only: nothing reads it back, and it adds no collective. Collect with
`tools/bench_spec_conf_tp.py` (verification pinned at depth 5, so no draft's outcome is hidden),
then replay the policies with `tools/analyze_spec_conf.py`. Data: `results/spec-conf-on/`,
image `e47eb4e9152b`.

The head is informative. Its rank AUC against the real conditional acceptance is 0.86-0.97 on
code and 0.61-0.84 on prose. Replaying the logged steps with fixed step costs of 88/106/124 ms
for depths 1/3/5 gives these gains over the better fixed depth per workload. They are rates of
a model, not engine measurements:

| Workload (steps) | Confidence rule, depths 3/5 | Confidence rule, depths 1/3/5 | Perfect predictor, 1/3/5 |
| --- | ---: | ---: | ---: |
| python (99) | +0.6% | +0.7% | +4.8% |
| html (106) | +2.9% | +3.6% | +6.9% |
| explain (221) | +1.0% | +3.2% | +15% |
| story (245) | -0.1% | +5.5% | +17% |
| story, temperature 0.7 (253) | +0.3% | +8.5% | +18% |
| prose with code (181) | +5.1% | +6.0% | +19% |

- With the existing 3/5 graphs, the modeled gain looked too small to justify the extra
  collective a per-step width needs; its cost had not yet been measured. Rank 0 broadcasts
  adaptive depth before drafting; a confidence-based depth is only known after it.
- Most of the value needs depth 1, which means width-2 graphs as well. The confidence rule gets
  about a third of the perfect-predictor ceiling on prose. Tuning a logit offset on the same
  data barely helps, so the head's ranking accuracy is the limit, not its calibration.
- Reading the logits can share the wait for the draft graph already required by the n-gram
  hash D2H. A separate readback and control message still add host overhead.
- Greedy and seeded sampled output were token-identical with the flag on and off, and on both
  ranks. The on/off decode rates differed by 5-16% in the flag-on run's favour on all six
  prompts. That is run-to-run variation, not an effect of the flag, so its overhead remains
  unmeasured.
- One prompt per workload. A prototype still has to beat adaptive depth in alternating
  full-engine runs, net of the per-step broadcast.

## TensorFold-style cutoff replay, 2026-10-03

The Qwen TensorFold/vLLM comparison prompted a fresh check of a simple cutoff.
The mechanism does not transfer literally: Qwen executes sequential MTP passes,
so an early cutoff avoids subsequent draft passes. DSpark computes its five
draft positions together (with a small sequential Markov-head loop); their
backbone cost is already paid when confidence is available. Here the saving
would come from fewer verifier rows, including fewer routed expert reads.

`tools/analyze_spec_conf.py --cutoff .7` now compares a prefix cutoff with the
existing cost-aware confidence rule. It stops before the first conditional
confidence below the threshold, rounds down to an available depth, and floors
at the smallest available depth. Confidence later in the block cannot restart
the prefix. The following replays use the existing September 29 traces and
historical 88/106/124 ms step costs for depths 1/3/5:

| Workload | Simple 70% cutoff | Cost-aware confidence |
| --- | ---: | ---: |
| HTML | -2.2% | +3.6% |
| Python | -0.6% | +0.7% |
| Explanation | -7.8% | +3.2% |
| Story | -0.7% | +5.5% |
| Story, temperature 0.7 | -0.4% | +8.5% |
| Prose mixed with code | +0.1% | +6.0% |

These are modeled changes against the best fixed depth on each trace, **not
measured gains against the running adaptive policy**. They exclude added
control/synchronization cost, confidence-head overhead, and graph capture.
Changing depth also changes which future positions become step boundaries, so
replaying the original steps is only a screening model, not a full rollout.

With only existing depths 3/5, the 70% cutoff is neutral on pure prose, -0.3%
on HTML, +0.6% on Python and +4.1% on mixed text. Lowering the cutoff to 30%
with depths 1/3/5 improves some cases (story +3.9%, sampled story +6.9%), but
still loses slightly on explanation and Python. A single threshold is not
automatically a better policy; the cost-aware rule estimates expected tokens
from cumulative acceptance confidence and divides by each width's step cost.

Retained results: `results/spec-confidence-cutoff-20261003/`, with 3/5 and
1/3/5 replays at 70%, plus 1/3/5 at 30%. No serving policy or defaults were
changed, and the Qwen TensorFold service remained running. The candidate for
an engine experiment is cost-aware confidence per step, with rank-0 decisions
broadcast to both ranks and width-2 graph qualification before enabling depth 1.

## Opt-in confidence policy, 2026-10-03

`DSV41_BLOCK_CONFIDENCE=1` enables a greedy-only experiment using depths 1/3/5.
Leave `DSV41_BLOCK_DYNAMIC=3,5` in place, and leave `DSV41_BLOCK` unset or set it
to 5. Speculation and the graphed fast path are required (above concurrency 1 it is allowed but not
yet exercised). The
flag defaults to 0; the serving `.env` is unchanged.

For each draft, rank 0 calculates
`expected_tokens(d) = 1 + sum(k=1..d, product(j=1..k, sigmoid(conf[j])))`
and chooses the depth maximizing `expected_tokens(d) / step_seconds(d)`.
Measured times are medians of the last 32 observations (minimum until three
samples), excluding graph-capture steps. The initial 88/106/124 ms priors come
from the earlier TP2 measurements. Unseen widths scale their priors by the
median observed/prior ratio rather than assuming that this machine still has
the historical absolute latency. Timing history survives requests; counters reset.
No forced exploration or per-step switch logging is added. Invalid confidence
falls back to depth 3.

Both ranks compute the head, but only rank 0 reads and uses it. Both then enter
a second three-integer control broadcast, after drafting and before building
the verify block. Reusing the stop protocol lets `release_peer()` abort a worker
waiting there. The feature and observation flags are in the boot config guard.
Width-2/4/6 graphs are captured only at first actual use: capturing ahead can
overwrite the KV ring. `last_stats.spec_depth` reports the selected policy,
counts per depth and observed/estimated costs. `DSV41_SPEC_CONF=1` separately
retains per-step logits; it is not required for the policy itself.

Sampled requests retain the existing adaptive 3/5 controller. Looking ahead
across all five confidence scores can select prefix length based on sampled
prefix tokens; preserving the sampler's distribution requires more than
reusing its acceptance formula. That extension is deliberately unqualified.

`tools/bench_confidence_depth_tp.py` checks greedy equality and rank agreement
while alternating 1/3/5, compares pinned widths and both controllers, and checks
seeded sampled fallback with a pinned legacy schedule. It then runs 512-token
HTML/Python/explanation requests in adaptive/confidence/confidence/adaptive order.
Both benchmark arms share the compiled confidence head; the confidence arm alone
pays its pre-verify readback and second broadcast. Thus this measures policy and
synchronization effects, but not the head's incremental compute overhead.

### Two-Spark result

The disposable TP2 run passed on both ranks. In the following table, each cell
is the median of two measured requests, with greedy temperature 0 and a maximum
of 512 new tokens. HTML stopped naturally at 453 tokens, Python at 511, and
explanation reached 512. Context capacity was 524288; these were short prompts,
not long-context or thinking-mode measurements.

| Workload | Existing adaptive 3/5 | Confidence 1/3/5 | Change |
| --- | ---: | ---: | ---: |
| HTML | 41.28 tok/s | 42.52 tok/s | +3.0% |
| Python | 52.25 tok/s | 53.66 tok/s | +2.7% |
| Explanation | 24.61 tok/s | 25.38 tok/s | +3.2% |

All four measured outputs per workload were token-identical, and both ranks
agreed across all 26 qualification and measured requests. Alternating widths
and fixed depths 1/3/5 also matched on the 256-token HTML qualification. The
128-token sampled fallback matched with the legacy policy pinned to 3. The
Python output parses and its five generated unit tests pass; this is still not
a general model-quality evaluation. The policy/control CPU suite passed 22 tests.

Confidence used depths 1/3/5 on 12/29/70 HTML steps, 0/5/87 Python steps, and
approximately 125/122/2 explanation steps. It therefore saves verification rows
on prose, while immediately choosing depth 5 on predictable code instead of
waiting for the legacy 60-token acceptance window. Peak PyTorch allocation on
rank 0 was 101.78 GB, with all six parity/width graph variants present.

This is a small preliminary gain, not grounds to replace the default controller.
Only one prompt per workload was tested, with two repetitions. Both arms share
the confidence-head computation, whose incremental overhead remains unisolated.
Sampling, long-context thinking and throughput under multiple clients are not
qualified for confidence scheduling. No thresholds were tuned to these outputs.

Retained evidence: `results/confidence-depth-20261003/summary.json`, both
`confidence-rank*.json` reports, both rank logs, decoded outputs, and
`generated-python-check.txt`. The summary records hashes of the exact engine
snapshot used for the gate. Serving `.env` remains unchanged and the wrapper
restores the Qwen TensorFold service after the test.

### Enabled as the recipe default

After reviewing these results, the user selected confidence scheduling as the
DS recipe default on 2026-10-03. Both environment templates and the local
serving configuration now set `DSV41_BLOCK_CONFIDENCE=1`, alongside adaptive
3/5 for sampled requests. The earlier opt-in/default-off wording describes the
experiment before this decision. The low-level engine switch still defaults to
0 when absent, so standalone fixed-width gates remain explicit and compatible.
Set `DSV41_BLOCK_CONFIDENCE=0` to return to adaptive 3/5 for all requests.

The installed DS runtime is `deepseek-v41-flash-spark:confidence-depth` on both
Sparks, built from the tested CUDA runtime plus the current engine/tools/server
source. The head launcher's `.env` selects this image and forwards identical
settings to both ranks. The user subsequently requested switching the active
service from Qwen TensorFold to DS4.1 with this default enabled.

## Per-step confidence at all temperatures, 2026-10-04

`DSV41_BLOCK_CONFIDENCE=1` now chooses verification depth 1/3/5 every step for
both greedy and sampled requests. The API temperature default remains 1.0;
clients do not need to select greedy decoding. Set the flag to 0 to restore
the older acceptance-window controller.

Greedy keeps the existing whole-block expected-token/cost comparison.
Sampling instead uses a proposal-prefix stopping rule: include depth 1,
then decide whether to extend to 3, then whether to extend to 5. At each
boundary d, the chooser reads only confidences through index d. DSpark's
confidence at that index depends on the token before the next proposal, so
all proposals influencing the decision are already included. The next two
confidences are estimated from the boundary score when comparing expected
tokens per step against measured cost. The score's accuracy affects which
width is economical, not the sampler's acceptance/rejection correction.

Inclusion of a proposal is decided without consulting that proposal or its
suffix. Its conditional proposal probability q therefore remains valid for
the existing `min(1, p/q)` acceptance rule and positive-residual correction.
Do not simply use the greedy whole-block argmax at positive temperature:
it can condition inclusion on a token that is then excluded. Rounding a
token-dependent cutoff down to an available graph width has the same risk.
Even invalid-confidence fallback checks must stop at the included prefix.

`engine/test_confidence_depth.py` enumerates all five-token proposals and
acceptance/rejection outcomes of a two-state Markov model, then checks the
joint distribution of the first three output tokens. It matches ordinary
target sampling to twelve decimal places at temperatures 0.1/0.6/1.0/2.0
and top_p 0.5/0.95/1.0. As a negative control, substituting the greedy
whole-block selector fails six of the twelve cases. The combined policy,
control, prefetch, and sampled-verifier CPU/GPU suite passes 42 tests.

Rank 0 chooses each step's depth and both ranks unconditionally enter the
existing second control broadcast. The sampling-policy version is in the
boot config guard. `spec_depth.selection` reports `lookahead` for greedy
and `prefix` for sampling. Different schedules can consume different random
draws; matching seeds across unequal widths is not a distribution test.
No sampled throughput or broad quality improvement is claimed.

The two-Spark qualification completed 26 requests with no rank disagreement:
64-token greedy HTML at alternating/fixed depths 1/3/5 and both policies;
fixed-depth-3 old/new seeded comparisons at all four positive temperatures;
and variable sampled requests at each of the twelve temperature/top_p pairs.
Fixed-width seeded output matched at every temperature. Variable sampled
requests selected depth 1 on 304 steps and depth 3 on 162; this short story
workload did not select 5. Peak PyTorch allocation was 102.261 GB on rank 0.
Placement and prefix reuse were frozen for qualification. Retained reports
and both logs: `results/confidence-all-temp-20261004/`. These are short-context
correctness checks, not long-context or thinking-mode performance measurements.

The installed `deepseek-v41-flash-spark:confidence-all-temp` image is identical
on both Sparks. Its engine files match the qualification image; the final
layer also refreshes the sampled-verifier test harness. A live request that
omitted temperature (default 1.0) returned the integers 1 through 8 exactly,
used the `prefix` policy at depth 5 on two steps, and retained
`dense_fp4=off`. See `http-smoke.json` and `summary.json` in the same directory.

## Default-off decode experiments (2026-10-04)

Three experiments preserve the target's attention/head precision and participate
in the EP2 boot guard. Lookup and bypass require single-request graphed speculation.
The TP draft-head flag controls allocation separately; the fast drafter uses the
copy when present, and callers that omit MTP loading do not allocate it.

* `DSV41_LOOKUP_DRAFT_ENABLED=1` restores request-local continuation copying.
  `DSV41_LOOKUP_DRAFT_NGRAM=16` alone no longer enables it. Rank 0 chooses a
  continuation and depth from settled history before drafting, broadcasts that
  proposal, and skips DSpark. At positive temperature its proposal probabilities
  are delta rows, consumed by the existing acceptance/residual verifier after
  target grammar/penalty handling. The confidence broadcast is still entered by
  both ranks; copied steps do not contaminate DSpark cost estimates.

  Re-checked 2026-10-21 on the shipped profile (`bench.py`, 3 runs a workload, two fresh boots
  with lookup on, one off; `scratchpad` A/B, not kept). It works: on the new `edit` workload (re-emit
  a 130-line module with three changes) 89% of steps copy and 97.5-99% of copied tokens are
  accepted, and prose never copies (0-1 hits in ~500 steps). It is **not a gain**, because a copy
  is only as deep as the depth policy's 3 or 5, while DSpark already reads continuations out of
  the context well:

  | workload | accept/step off | on | tok/s off | on |
  | --- | ---: | ---: | ---: | ---: |
  | edit | 5.62 | 4.48 | 59.9 | 54.9 / 55.5 |
  | code | 3.84 | 3.56 / 3.61 | 43.8 | 43.0 / 43.1 |
  | prose | 1.99 | 1.97 / 2.00 | 26.5 | 26.2 / 26.6 |

  Copied steps averaged 3.6 tokens (801 / 223) against DSpark's 5.6 on the same text. The only way
  this pays is a deeper copy than DSpark can draft: a copy-only verify graph at T = 10 (9 copied;
  `build_routing_small` caps P at 64 pairs). That costs ~34% a step (the T=10 entry in
  `gotchas.md`) and is worth it only where copies stay ~97% accepted, so it would help edit-style
  replies and nothing else. Default stays off.
* `DSV41_TP_DRAFT_HEAD=1 DSV41_DRAFT_HEAD_FMT=fp8` builds a separate quantized
  local vocabulary shard. It uses the existing vocabulary all-gather; the BF16
  verifier shard is untouched. The actual 64640-by-5120 checkpoint shard adds
  331,280,000 bytes per rank, allocated before arena sizing. A local five-row
  graph replay microbenchmark, 25 ABBA quartets, measured 2.888 ms BF16 versus
  1.502 ms FP8. This excludes all-gather, acceptance and the rest of the engine.
* `DSV41_DRAFT_BYPASS=1` selects occasional ordinary target steps using only
  completed-step cost/acceptance. It skips DSpark, runs the two-row graph on root
  plus a discarded dummy, samples only root logits, and rolls back after root.
  Ratio-2 compression currently prevents a genuine width-one graph. A bypass
  selected from current proposal tokens would be invalid for sampling; this
  policy decides before drawing any current proposals.

The lookup CPU screen used exact-16 suffixes, 512 append/propose calls and three
repetitions: periodic 16K/96K prompts took about 23 us per hit; random misses took
2-3 us. Initial 96K indexing took 32-71 ms. Dense delta rows at vocabulary
129280 occupy 0.52/1.55/2.59 MB for depths 1/3/5. These are host overhead figures,
not model-throughput measurements.

`engine/test_lookup_sampling.py` enumerates joint first-three-token distributions
under four temperatures, three nucleus thresholds, masked/adjusted targets and
four fixed/history-dependent width rules (96 combinations).
`engine/test_decode_experiments.py` executes the actual control/commit branches
without loading weights. `tools/test_tp_draft_head_cuda.py` checks real two-rank
all-gathers and changing-input graph replay. `tools/bench_decode_ideas_tp.py`
reuses one loaded engine for bounded rollout qualification followed by warmed
head-format ABBA tests. Evidence is retained under
`results/decode-ideas-20261004/`. Production switches remain off while these
experiments are measured.

### Warmed two-Spark screen

The combined gate completed on both ranks with no output-hash disagreement.
Lookup qualification comprised 41 requests per rank: periodic and stale-context
greedy equality, forced rejection of a deliberately corrupted copied token, and
seeded repeat/rank checks at temperatures 0.1/0.6/1.0/2.0 with top_p 0.5/0.95/1.0.
Bypass qualification comprised 68 requests per rank: greedy off/forced-on/alternating/
adaptive equality on explanation and Python, plus pinned seeded repeats and
adaptive rank agreement across the same twelve sampling settings. The draft-only
head matched greedy Python output and agreed across ranks on four sampled rollouts.
The combined CPU/GPU suite passed 75 tests in 1.230 s; separate head tests and its
real two-rank all-gather gate also passed. These checks do not establish broad
language quality or sampled distribution equality from seeds alone.

After graph/prompt warmup, each comparison ran
baseline/candidate/candidate/baseline, two measured requests per arm. Warmups
generated 32 tokens for copy/bypass and 64 for the head tests, shorter than the
timed outputs, so later Engram rows can retain cold-read effects. All measured
requests used top_p=0.95:

| Experiment / workload | Temperature / output tokens | Baseline median tok/s | Candidate median tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| Continuation copy / periodic text | 1.0 / 96 | 44.715 | 47.950 | +7.23% |
| Forced draft bypass / explanation | 1.0 / 96 | 22.540 | 15.530 | -31.10% |
| FP8 draft head / HTML | 0 / 256 | 51.515 | 51.975 | +0.89% |
| FP8 draft head / Python | 0 / 256 | 54.920 | 56.265 | +2.45% |
| FP8 draft head / explanation | 1.0 / 256 | 24.605 | 27.440 | +11.52% |

These medians use the reported rounded decode rates. Copying is a favorable
repeated-text case with depth pinned to 3: all 24 lookups and 72 copied proposals
per run succeeded, skipping DSpark on every step. All four copy outputs matched.
This is evidence for repeated continuations, not ordinary prose or miss-heavy
requests. Forced bypass saved the draft pass but emitted only one token per step;
95 bypass steps lost to the baseline's average 1.88 tokens per step. It remains off.

Both greedy head comparisons emitted identical target tokens in every arm. HTML
acceptance fell from 5.08 to 4.98 tokens per step, leaving its small throughput
change within the run spread. Python acceptance remained 5.59. The sampled
explanation changed proposal/RNG schedules and text; FP8 runs ranged from 25.96
to 28.92 tok/s with acceptance 2.17 to 2.46, versus baseline acceptance 2.10.
The larger median gain therefore includes different stochastic trajectories and
confidence schedules; it needs representative repeated trials before promotion.

The target head stayed BF16 and attention stayed FP8 (`dense_fp4=off`). Placement,
prefix reuse and urgency were frozen. Both head arms retained the extra
331,280,000-byte FP8 shard per rank and the same 90.1 GB arena/0.61 keep fraction;
this comparison does not price a reduction in expert residency. Peak PyTorch
allocation was 102.726 GB on each rank. `INDEX_TOPK=512` explicitly matched the
existing serving configuration despite the host tree's new default of 1024.
Context capacity was 524288, but these were short prompts, without long-context,
thinking-mode or concurrent-client measurements.

The experiments were not deployed. All three switches remain off. The original
`confidence-all-temp` pair was restored and passed health plus a default-temperature
HTTP request returning exactly `1 2 3 4`; extra attention quantization and the
draft head remain off, with confidence depth enabled.
Exact samples, both rank reports and the summary are retained under
`results/decode-ideas-20261004/model-gate/` and
`results/decode-ideas-20261004/summary.json`.
