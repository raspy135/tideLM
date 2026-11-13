# The hot-read predictor: what was built, why it lost, and what a real one needs

`GLM_HOT_PREDICT` ranks the async hot reads by each (layer, expert)'s **missed share accumulated
over the pass's rows and decayed over passes** — a tally of past misses. It was asked to be an
expert-loading predictor, and what arrived predicts the next pass's demand from history alone:
no token, no context, no forward computation. Its own measurements rejected it. This file is the
record, the ceiling any predictor faces at the current read budget, and the staged proposal that
follows from both. The tooling the branch added (trace, oracle, sim, eval) is kept — it is the
part worth keeping, together with `GLM_HOT_VICTIM=slow`.

## What was built

`HotPredict` (adapt.rs) folds every decode pass's replicated routing table into per-(layer,
expert) accumulators (`acc <- DECAY * acc + missed shares`), and the async plan takes up to
`PER_LAYER` experts a layer with `acc >= MIN` (tier 1), behind the engine's own heaviest-miss
pick (tier 0, `REACT=1`). A read resets its accumulator. `GLM_HOT_VICTIM=slow` ranks cold victims
by slowly decayed demand (`BG_SLOW`, half-life ~70 passes) instead of the long-run score.
`TIDELM_GLM_HOT_TRACE` records every pass's routing and reads; `TIDELM_GLM_HOT_ORACLE` replays an
earlier trace's next-pass heavy misses as a perfect predictor. The tools: `tools/glm_hot_sim.py`
(policy replay), `tools/glm_hot_eval.py` (read quality, history-feature recall), and
`tools/glm_route_predict.py` — **written to measure the predictors the ask wanted, never run: it
needs token ids, and every trace on disk is GHT1. The working tree records GHT2 (ids + first
position a pass); no trace has been taken with it.**

## The measured record (heldout_quick, 10,151 tokens, paired against the rule arm A)

`tools/glm_nll_pair.py`, `~/tidelm-rust-state/hot-predict-20261108/`:

| arm | paired NLL | blown windows | reads/token | waited |
| --- | ---: | ---: | ---: | ---: |
| A: the rule (reactive heaviest miss) | — | 0 | 0.57 | 15.7 s |
| V: A + slow victims | **-0.0031 ± 0.0026** | 0 | 0.52 | 14.7 s |
| P: tally alone (0.9,1.0,1) | +0.0747 ± 0.0075 | 1 | 0.48 | 13.7 s |
| P3: another tally setting | **mean NaN** (a p=0 token) | — | 0.48 | 13.7 s |
| H: hybrid REACT=1 (0.9,1.0,1,1) | **+0.1222 ± 0.0092** | 2 | 0.73 | 17.3 s |
| Hchk: H re-run | +0.1227 ± 0.0092 | 2 | 0.73 | 17.1 s |
| HV: hybrid + slow victims | +0.1197 ± 0.0092 | 2 | 0.66 | 16.1 s |

Slow victims on heldout_eval (47,164 tokens): **-0.0036 ± 0.0016**. The predictor arms are the
only ones that lose; slow victims is the only change that wins.

Why the tally loses, from its own trace (`glm_hot_eval.py --skip 16`, H.trace):

- **Its reads are good predictions of the wrong thing.** Predicted reads hit@1 70% (the rule's
  34%) but served 0.008 picks ≥ 0.3 each in 16 passes against the rule's 0.151: they predict the
  recurring *light* picks, which the residency and the ring already cover. oracle@1 26.9% of the
  best same-count plan.
- **They take the rule's slots.** A rule read's life fell 1098 → 438 passes. 43.1% of the next
  pass's heavy-miss share was experts the engine itself had evicted for a read; 27.6% of all
  missed share. The blown windows are this churn reaching a text that needed one expert
  catastrophically (one next token 16.5 nats; P3's arm hit p = 0 and a NaN mean).
- **Heavy misses are not a history phenomenon.** Of the heavy-miss share (≥ 0.3, 1,511 picks),
  28.3% had been picked heavily in the last 16 passes; the tally ranks it top-1 among a layer's
  non-residents 29-37%, top-4 44-49%. The heavy pick that costs NLL is mostly a **first-time**
  heavy pick: only 0.5% of the heavy misses left behind had one in the 16 passes before.
- **Total missed weight is not the quality.** P3 missed 10.32% by score against A's 11.88% and
  produced a NaN mean. The damage concentrates in a row's heaviest miss (the sim's `max^3`
  column, and the NLL above).

## The ceiling for any next-pass predictor (sim, V.trace, skip 100, recent victims)

`tools/glm_hot_sim.py`, fixed routing, sim victims — ranking and budget are what matter here:

| policy | reads/pass | miss score | heavy ≥ 0.3 /row | max^3 /row |
| --- | ---: | ---: | ---: | ---: |
| the rule (async, lag 1) | 1.75 | 12.03% | 0.075 | 0.0692 |
| previous pass, capped 2 | 1.93 | 11.88% | 0.187 | 0.0842 |
| oracle, summed share, 2 | 1.93 | 11.00% | 0.153 | 0.0809 |
| **oracle, share^3, 2** | 1.93 | 11.27% | **0.058** | 0.0686 |
| oracle, share^3, 4 | 3.86 | 9.42% | **0.020** | 0.0535 |
| oracle, 16 | 15.99 | 6.00% | 0.007 | 0.0388 |
| the rule, sync (in-pass) | 1.75 | 11.98% | 0.001 | 0.0600 |

Two conclusions. **Summed-share prediction is the wrong objective at any accuracy** — a perfect
oracle of the next pass's summed demand is *worse* on the heavy metric than the reactive rule at
the same budget; share^3 ranking (the quality metric's own shape) is what turns prediction into
quality. And **at the current budget (~2 reads a pass) even a perfect share^3 oracle only edges
the rule** (0.058 vs 0.075; the engine's recorded runs are 0.070-0.080): the wins beyond that
come from reads (4 a pass: 0.020) or from blocking in-pass (0.001, which costs decode time — the
DS measurements: -6% to -16% tok/s). Prediction's real margins at this budget are therefore:

1. **The wait.** Reads are planned after a pass and waited on at the next pass's start:
   15.7 s of a 317 s run (5%). A predictor that reads *before* the miss — from the anchor and the
   drafts, which are known before the pass runs — converts wait into hidden time.
2. **The anchor row's novel heavy pick.** The rule is structurally one pass late: the row that
   hurts is row 0 of the next pass, whose token id is known at acceptance. The tally cannot see
   it coming; a token-conditioned predictor can.
3. **The victim side.** 43% of heavy-miss share was self-evicted. A forecast term in the victim
   ranking (and in `bg_step`'s demand fold) keeps the experts the next pass will want — the
   collapse class is exactly this.
4. **Prefill.** The probe pays a second pass for discovery; a routing model over the suffix's
   token ids pays ~nothing (docs/predictive-prefill.md is the DS precedent).
5. **The first occurrence, inline — built as `GLM_HOT_RESCUE`, off until its A/B.** Async
   covers a heavy pick's second and later passes; its first pass is always substituted. A
   blocking rescue read for the extreme tail catches it inline. Trigger rates on A.trace
   (heldout_quick, skip 16): share >= 0.3 hits 18.6% of passes (0.25 reads a pass), >= 0.4
   4.5% (0.06), >= 0.5 1.1% (0.02); the p99 missed pick is share 0.50 — the blown-window
   class. But the damage ceiling is small: by band, the per-(layer,row) heaviest-miss damage
   (share^3) is 78.2% below 0.2, 11.4% in [0.2, 0.3), 5.3% in [0.3, 0.4), 5.1% at >= 0.4.
   Replayed on the trace (glm_hot_sim's run loop, fixed routing, sim victims), the rescue arm
   on top of the async rule: >= 0.4 — cube 0.0726 -> 0.0707 a row (-2.6%), miss by score
   unchanged, 0.05 sync reads a pass on 3.9% of passes; >= 0.3 — cube 0.0678 (-6.6%), 0.26
   sync reads on 16.6% of passes. The sim cannot model the rescued row's corrected hidden
   state reaching later layers and the head, so these are lower bounds on quality; the NLL
   A/B is the gate. The blocker is the structure, not the reads: a mid-pass rescue needs the
   router output on the host before that layer's MoE runs — segment-split execution with host
   points at every MoE layer, every pass, because a first occurrence cannot be flagged before
   the pass. DS blamed exactly that structure for most of sync's -6% to -16% (the "no sync
   once the budget is spent" fix recovered 10 of the 16.5 points).

   Implemented (adapt.rs `RESCUE_STEP`, `Hot.rescue`; needs `GLM_HOT_ASYNC=1`): with
   `GLM_HOT_RESCUE=TH` a decode pass runs as its segments (the sync path's graphs, their
   per-layer candidates copy now captured when rescue is on) and at each MoE layer the
   heaviest **missed** pick with share >= TH is read into the transient slot before that
   layer's MoE, computed exactly, then kept over a queued victim as any hot read (`hot_release`)
   or dropped. The async plan after the pass is unchanged and skips kept rescue experts (they
   are resident); a dropped rescue's recurrence is covered one pass late, as the rule would
   have it. No pacing (a first occurrence can be any layer), cap `RESCUE_STEP` = 2, threshold
   in the boot guard (`--hot-rescue` / `GLM_HOT_RESCUE`, both ranks must agree), inline reads
   counted separately (`hot_rescue`, printed by `--bench-nll` and in the per-request log). The
   rescue reads are not in
   `TIDELM_GLM_HOT_TRACE` (its read list is the async plan's), so a rescue arm's trace shows
   residency-replay disagreements in `glm_hot_eval.py` — the bench NLL A/B, not the trace
   replay, qualifies this arm.

   **Qualification** (the structure question is what it measures): `--bench-nll` arms on
   heldout_quick and heldout_eval — A (async rule), A + rescue 0.4, A + rescue 0.3, and the
   old sync arm for the structure reference — paired with `glm_nll_pair.py` (blown windows
   are the signal: the predictor arms' whole loss was 1-2 of them), reads/token and
   `hot_wait_s` from the summaries, then `--bench-ab` tok/s. Ship `0.4` if the pair is <= 0
   NLL at <= +1% round time; the `0.3` arm carries 3.7x the stalls for +2.6 points of cube
   and is expected to lose on time. The alternative that needs no stall stays Stage 1.

   **On GLM-5.3-Flash TP=1 (2026-11-13) 0.4 failed the NLL half and passed the time half**:
   heldout_quick + heldout_eval, two boots from one state file, NLL +0.0048 +- 0.0015 paired
   (coding +0.0120), against a noise floor of +0.0006 +- 0.0012 (TIDELM_GLM_MLA_HALF=0 through the
   same harness); passes +0.0 to +0.2%, 0.10 rescues a pass. Off there; docs/glm-flash.md has the
   table. GLM-5.3 itself is still unmeasured.

   `RUST_ARGS="--bench-nll corpus/heldout_quick.jsonl ..."` runs one arm (RUST_ARGS is the
   launcher's passthrough for both ranks).

   **First boot found a latent bug in the segment path** (2026-11-08): `Hot::host`, the
   per-layer candidates buffer the host point copies into, was sized `VERIFY_ROWS` (8) rows
   while the graphed pass runs up to `MAX_ROWS` (16); a 9-16-row pass (two lanes, or rows
   9-16) copied past the pinned allocation and `cudaMemcpyAsync` failed with invalid argument
   at boot warm-up. The async path never copies per layer, which is why it never showed. Sized
   `MAX_ROWS` now — the sync arm (`GLM_HOT_ASYNC=0`) boots for the first time in this tree, so
   its old numbers cannot be assumed to describe it.

## Proposal

Staged, each gated by a number. Nothing goes in the profiles before its gate.

### Stage 0 — run, 2026-11-08: the signal is the token's, and it is not enough

GHT2 trace taken (`--bench-nll` heldout_quick, shipped arm, 1,277 passes of 8 rows, 50 MB at
`~/tidelm-rust-state/route-predict-20261108/A.trace`), `tools/glm_route_predict.py --skip 16`,
1,352 scored passes, 1,013 heavy picks (share >= 0.3). Recall of the next pass's heavy-miss
weight, from information the engine holds before the pass:

| predictor | B=2 | B=4 | B=8 | B=16 |
| --- | ---: | ---: | ---: | ---: |
| previous pass | 4.4% | 7.0% | 10.9% | 15.7% |
| tally (GLM_HOT_PREDICT's shape) | 4.4% | 6.2% | 10.1% | 15.3% |
| context (routing decayed 0.98 a token) | 1.7% | 2.3% | 3.5% | 5.7% |
| same token (within the sequence) | 0.4% | 0.7% | 1.2% | 1.6% |
| **token elsewhere (per-token table, leave-one-sequence-out)** | **11.1%** | **16.0%** | **23.1%** | **31.6%** |

Three things this settles. **Routing is a token property, not a history one**: the per-token
table is 2.5x the previous pass and the tally at B=2 and the only predictor that scales with
budget. Local context is nearly worthless (1.7% at B=2); "same token inside this sequence" is
worse still, so an MTP draft's own earlier occurrences in the same text carry almost nothing.
The table generalizes across sequences (leave-one-sequence-out, honest for unseen prompts), which
is what a serving predictor needs. **The absolute power is small**: 11.1% of heavy-miss weight at
2 reads a pass is ~0.04 share a read. **And it does not survive a heavy threshold**: replayed
causally in `glm_hot_sim`'s loop (table folded from earlier passes only, ranked among
non-residents), a mean-share table replacing the rule gives cube 0.1333 a row against the rule's
0.0692 — its top picks are light experts — and as an *added* tier at expected share >= 0.3 or
0.4 it fires 0.01 times a pass and changes nothing (0.0745 -> 0.0741). Storing each (token,
layer, expert)'s maximum share instead of the mean does not help: a heavy pick is token *and*
context determined, so no token reliably carries one.

**Verdict: the per-read prefetch predictor is closed at this read budget.** Perfect next-pass
knowledge moves the damage metric ~1% at 2 reads a pass (0.0686 vs 0.0692); it needs 4 reads a
pass to pay (0.0535, and 0.020 heavy misses a row against the rule's 0.075), and those reads cost
round time. A trained head would have to beat a *perfect* oracle's information, and the features
that would let it (the drafter's hidden state, context) are the ones this trace cannot measure —
so the honest next step is not a bigger model here, it is a different target: the residency
planner (bulk moves, no per-read churn, where 43% of heavy-miss share is self-evicted) and
prefill discovery (where 100s of ms per pass, not 2 reads, is the budget). A hidden-conditioned
P(heavy) classifier stays the right shape *if* either of those needs a forecast per token —
measured against the same oracle ceiling, not against the tally.

### Stage 1 — the anchor/draft-informed lookup, no training

The engine knows the next pass's rows before it runs them: the anchor id at acceptance, the
draft ids as the MTP passes return them (all before the verify pass). Keep an online per-token
routing table — folded from the same replicated table the tally folds, keyed by token id instead
of nothing (host-side, a bounded LRU or hash of recent ids; the drafts' ids are looked up, not
scanned) — plus the lane's recent-context routing vector (75 x 256 f32 a lane, decayed a token:
nearly free, and the strongest single feature in the trace's own recall columns). Plan a small
*separate* read tier from the drafts at the first MTP pass:

- **Its own budget** (the hybrid's measured failure was predicted reads eating reactive slots);
  reactive reads keep theirs and are issued first — they still have the longer window.
- **Slow-class victims only**, and a forecast term in the victim ranking so a predicted read
  cannot evict what the forecast itself wants (the 43% number).
- **Heavy-only threshold**: read a predicted expert only when its expected share clears the
  heavy bar. The tally's reads were 70% accurate at light picks and worth less than nothing.

**Gate.** `glm_nll_pair.py` ≤ 0 against the A arm on both corpora at ≤ A's reads/token + 0.15,
zero blown windows, `hot_wait_s` not worse by > 1 s per 3,000 passes; then the generation gate.
The sim gets the policy first (`glm_hot_sim.py` on a GHT2 trace: it replays any plan without an
engine run).

### Stage 2 — the MTP-technology predictor: trained route heads

The ask, done properly: a small model that predicts routing from *computed context*, like the
MTP head predicts tokens. Per MoE layer, a head `g_l` maps what the drafter already has — the
anchor's kept hidden (`hlast`), the draft rows' MTP hidden (`mhn`), the draft token embedding,
the MTP layer's own routing for those rows, the lane's context vector — to 256 expert logits.
Full heads are 75 x 6144 x 256 fp16 ≈ 236 MB; factorized (rank 64) ≈ 64 MB. Inference is one
batched GEMM over ≤ 8 rows (< 1 ms), run beside the draft pass; top-k copied back with the table
copy that already happens.

Training data is what the traces already hold: teacher-forced `--bench-nll` runs record every
pass's actual per-layer top-8 and shares (GHT2 adds the ids and positions; the hiddens are
replayed offline with the engine's own MTP ops). Distill from the target's own routers. **Loss
shaped like the damage** — per-row heaviest^3 or a heavy-pick classification objective, not
summed share: the oracle table above says a summed-share-trained predictor reproduces the tally's
failure at perfect accuracy. Train offline (`tools/glm_route_train.py`, the dream/mtp_cal
pattern), ship a small file beside the state DB, pin its digest in the boot guard (the pair fails
silently; the predictor's inputs are replicated on both ranks, so plans agree without a message,
and any new collective stays unconditional).

**Gate.** Offline first: the head's top-2-a-layer heavy recall must beat Stage 1's lookup by
≥ 10 points before it is wired; then the same NLL/wait/generation gates.

### Stage 3 — wire it where the ceiling says it pays

In measured-leverage order: the read tier (Stage 1/2 above); the victim forecast term and a
predicted-demand vote in `bg_step`/`adapt` (bulk moves, no per-read churn — this is where the
43% self-eviction and the blown windows are attacked); session-preload's conversation votes
generalized by the same model; and prefill discovery from the suffix's ids (replacing the probe's
second pass for short suffixes — `DSV41_PROBE_TWO_PASS_BELOW`'s path — with the per-token table
or the heads, probe kept for long suffixes).

## What not to repeat (all measured)

Summed-share ranking, at any accuracy. Predicted reads taking reactive reads' slots. A
predictor-only arm. Ranking the quality by total missed weight. Full-bar sync blocking reads
(the numbers above); the high-bar rescue is the measured-arm version, not a return of it. And:
the harness for the forward-looking predictors must run before the next predictor is built —
Stage 0 exists because `glm_route_predict.py` was written, correct, and never given a GHT2
trace.
