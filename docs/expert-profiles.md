# Fixed topic expert profiles

> **Removed 2026-10-17.** The code (`engine/expert_profiles.py`'s topic loader and maxmin ranker,
> `DSV41_EXPERT_TOPICS`, `DSV41_PRUNE_SOURCE=saliency`, `DSV41_PRUNE_RANK=maxmin`) is gone; git
> history has it. This page stays as the record of what was measured and why it was not adopted.

The author published statistics for 39 topics, each containing 40 layers of 384
expert frequencies and saliency values. This port uses that database to choose
the resident expert IDs; it does not download, convert, or change expert weights.
The TP2 output shards, native FP4 arithmetic, arena and transient slots are unchanged.

Saliency here is the accumulated `gate_weight * ||expert_output||` over traced
tokens. Frequency counts picks without their output magnitude. Neither measures
the causal effect of omitting an expert on an answer. Quality must be tested.

The initial small comparison did not qualify a new default. Frequency/maxmin
scored 10/14 versus the adaptive baseline's 8/14, then 9/14 versus 12/14 on a
separate set, including one format failure. Saliency/maxmin scored 8/14 on the
first set. The normal service keeps adaptive selection. See the 2026-10-06
topic-profile comparison in [RESULTS.md](../RESULTS.md) for protocol and costs.

`maxmin` normalizes each selected topic separately in each layer. It gives the
next slot to the topic with the lowest currently covered mass, admitting that
topic's highest-ranked unselected expert. Shared experts help every topic that
uses them. At keep 0.61 this fills 235 slots per layer, 9,400 total. It prioritizes
the least-covered topic at each admission; it is a greedy heuristic, not a proof
of an optimal expert set. `sum` and `max` are also available.

## Reproduce the database

```bash
python3 tools/fetch_expert_profile.py
rsync -a results/keepsets/upstream-topics/ \
  ryan@10.0.0.2:/home/ryan/git/deepseek-v41-flash-spark/results/keepsets/upstream-topics/
```

The helper fetches only the 13,048,828-byte statistics file, pinned to
[upstream commit 45a0caff](https://github.com/0xBakeer/deepseek-v41-flash-spark/tree/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f).
It reuses an existing matching file and refuses a checksum mismatch. Its adjacent
provenance file records the MIT source and checksum:
`eb5214a78791a1e8cc0db51353f6ea7931f4f2d18142776f90784e01e67f16d3`.
Substitute your peer/path when deploying elsewhere. Put the same file on both
nodes; no model weights are fetched by this helper.

## Launch a fixed profile

Build an image containing `engine/expert_profiles.py` and the updated engine on
both nodes. After stopping the current pair, a one-off launch can use:

```bash
TRACE_STATS=/app/results/keepsets/upstream-topics/coverage.json \
DSV41_PRUNE_SOURCE=saliency DSV41_PRUNE_RANK=maxmin DSV41_EXPERT_TOPICS='' \
DSV41_PRUNE_ADAPT=0 DSV41_ADAPT_SENSITIVITY=off \
DSV41_PRUNE_SWAP=0 DSV41_PRUNE_SWAP_PREFILL=0 \
DSV41_ADAPT_DECODE_TOKENS=0 DSV41_ADAPT_URGENT=0 \
PRUNE_KEEP=0.61 PRUNE_SELECT=uniform \
scripts/dual-up.sh
```

An empty topic list uses all topics from the file, sorted for reproducibility.
For a specialized workload, `DSV41_EXPERT_TOPICS=japanese,python,academic` selects
those topics; measure that choice on separate requests before using it broadly.
The loader rejects missing saliency arrays, invalid numbers, unknown topics and
incomplete layer/expert shapes. It never silently substitutes frequency data.

All nondefault profiles are fixed. The engine rejects demand-history blending
or any swap trigger because raw router-score units differ from saliency and
maxmin admission priorities. Demand is still recorded in the configured history
file for diagnostics. Use a separate `DSV41_PRUNE_DB` during experiments to keep
production history intact. `maxmin` also rejects global selection and explicit
layer budgets; leave `DSV41_PRUNE_LAYER_COUNTS` unset.
The existing default `counts`/`sum`, with no explicit topics, retains the old
coding/general trace and adaptive history behavior.

## TP2 consistency and verification

Both nodes independently validate the database. The existing rank-0 score
broadcast still determines both keep sets; no collective was added. The boot
configuration guard now includes profile version, source, ranker, resolved
topics, database SHA256 and the SHA256 of actual initial expert IDs. A same-size
but different-ID mask is detected. `/health` reports these fields plus per-topic
mean/minimum layer coverage. That coverage is a trace proxy, not an answer score.
`initial_keep_sha256` describes startup placement, including in legacy adaptive
mode where later swaps can change the current IDs.

Nine profile tests cover validation, zero-mass topics, budget preservation,
normalization, mixed-unit rejection, legacy defaults and ID fingerprints. The
ported maxmin scores match the pinned author's implementation exactly for both
families on all 39 topics, 40 layers and 384 experts. The focused suite passed
60 tests. Artifacts and the frozen comparison protocol are in
`results/expert-profile-20261006/`; measured quality is recorded in `RESULTS.md`.
