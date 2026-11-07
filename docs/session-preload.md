# Session-demand preloading

Status (2026-11-06): implemented, **off by default** (`GLM_PRELOAD=0`), and **not measured in
serving**. What is measured is the traffic that motivated it and the planning logic (unit
tests, the serving log, the histories' own curves); what is not is every serving claim. This
page keeps the two apart, and records what stage 0 killed so it is not retried.

## What it is

A request that continues a conversation -- a prefix-cache hit with a fresh suffix -- loads
the experts that conversation already demanded, per (layer, expert), the accumulated votes of
its earlier turns, **before** its suffix prefills. Today the suffix prefills against stale
placement on substitutes, and the correction arrives after the prefill: two-pass re-runs
only suffixes of 4..=512 tokens (a full extra prefill), and the boundary pass loads its
swaps after the prefill, inside the measured prefill window. Preloading moves the loads the
boundary pass would pay for to before the compute they exist for; the boundary pass that
follows then plans a smaller set, because the preloaded experts are already resident.

## Why: the measured traffic (2026-11-06)

26 h of the serving log, one request line each:

- 48 requests; **47 prefix-cache hits**, median 17,860 cached tokens (max 92,378).
- Fresh suffixes: **median 840 tokens, max 8,441**, prefilled at median 307 tok/s
  (the window the boundary pass reads inside), decode 21.6 tok/s median.
- Only 2 of the 47 continuations came from the disk prefix store; the rest continue the
  lane's own sequence, so no disk identity keys them.
- The median suffix is longer than two-pass: it computes entirely on substitutes, corrected
  only for decode onward. Each turn adds ~840 substituted positions to the session's context.

## The design

- **Key.** `prefix::conv_key`: sha256 of the conversation's opening 256 tokens, as a u64.
  Every turn of an agent session resubmits the whole conversation, so the opening identifies
  it across turns and exists at turn 1, whose demand is the largest single turn.
  Conversations that share an opening share a key and their votes merge -- conservative:
  more known demand, never less. Tested: `a_conversation_keys_by_its_opening_across_turns`.
- **Votes.** Both phase votes of each request (prompt and answer), per-layer normalized as
  the DS history's fold (sum to one a layer) and accumulated without decay, in memory only,
  bounded at 32 conversations by last use. Nothing is persisted: a restart loses them, and
  the first turns after a restart do not preload; accumulation rebuilds over turns. Tested:
  `conv_votes_fold_per_layer_normalized_and_accumulate`.
- **Plan** (rank 0 only, as every plan here). Candidates are only what the conversation's
  votes name, non-resident, ranked by accumulated share; tested in
  `preload_candidates_rank_the_conversations_non_resident_demand`, which also proves
  residents are excluded whatever their mass. Victims are the globally weakest residents the
  conversation did **not** name, with `plan_swaps`' per-layer floor and width classes (a
  newcomer takes a slot of its own class). No margin check: conversation shares are
  per-layer normalized and `res.score` is decayed demand, not one scale -- the protections
  are the cap, the floor, and that every candidate is demand this conversation voiced. A
  conversation's own residents are never its victims.
- **Transport.** The boundary pass's own: the plan mirrored over the control channel
  (`OP_ADAPT`), applied by both ranks through the same load path (the 12 parallel
  O_DIRECT readers). No new collective, no new message type, and the setting sits in the
  boot guard (`guard_extra`), so a disagreeing pair refuses to start.
- **Nothing is folded.** No live demand is read and no history is written by the preload;
  the request's own votes still fold at their usual sites. The histories keep the only
  persistence.
- **No pin.** The design sketch had a request-duration pin; the swap machinery's default
  persistence (until a later plan outcompetes) makes one unnecessary, and the not-demanded
  victim filter is the stronger protection. Preloaded experts are ordinary residents.

## What the logs say

- At the preload: `[glm] preload request N: K experts of its conversation's demand, M MB
  read, T s`.
- The request line: `preload K (M MB, T s)` -- per-request deltas, the same accounting as
  hot reads and swaps.

## Open: measure before believing

1. **TTFT.** "Wall time roughly flat" rests on the preload's reads being the boundary pass's
   reads reordered. Measure prefill tok/s on follow-ups before/after; if the reads are
   additive rather than reordered, the TTFT claim is wrong and the lever is a pure quality
   trade at a TTFT cost -- decide it on that number.
2. **Quality.** Miss rate and KL to exact on replays, before/after.
3. **The join.** What share of a suffix's miss mass its conversation had already voiced --
   the stage-0 measurement this lever presupposes. The log's preload counts bound it from
   above; the offline join over the conversation log sizes it exactly, and is still owed.
4. **Kill criteria.** The join comes back low (turns are genuinely novel demand); the
   conversation's demand is diffuse beyond the cap; or the later plans evict the loads
   immediately (the boundary and decode plans re-plan from live demand, which includes this
   request's). Each is a number; record it either way.

## Phase 2 live evidence (first ~day of traffic, 2026-11-06/07)

The serving restart carried `GLM_PRELOAD=512`. The window below is the first
~day of mixed traffic: one resumed ~300K-token work session (its own turns),
plus deliberate short-prompt probes. Request ids are global across lanes.

**The accidental A/B.** A probe prompt (Japanese, eight specialist domains,
thinking off) was fired twice in error; the engine ran both fully. Same text
-> same conversation key, so the second request is a genuine turn 2 carrying
the first request's folded votes. The before/after is the engine's own line:

| same conversation | turn 1 (req 8) | turn 2 (req 10) |
|---|---|---|
| prompt-phase miss | 15.1% | **9.9%** |
| decode miss | 12.4% | 10.6% |
| accepted / step | 0.95 | 1.27 |
| boundary swaps | 94 (margin-refused) | 824 installs |
| preload | 0 (no votes yet) | **512 experts, 2,987 MB, 0.30 s** |

Reading: the turn-1 boundary pass took only 94 experts -- its margin rule
refused the niche demand. The preload's planner (no margin, floor/class/
not-demanded victims only) took 512 experts in 0.30 s of reads, and the
prompt-phase miss fell 5.2 points; acceptance rose 34%. The lever is real
where its class occurs.

**Wall clock did not improve at this size.** The 221-token prefill ran
1.31 s (168 tok/s) after a 0.89 s (250 tok/s) turn 1, plus 0.30 s of
preload reads: a wash, with read contention confounding the prefill rate.
The lever's wall-clock value scales with the fresh-suffix size (a
7,511-token suffix at ~237 tok/s is ~32 s; 5 miss points there is
seconds), not with a short prompt. The mechanism is demonstrated; the
TTFT claim is not, and open item 1 stays open.

**Gating behaved as designed.** Single fires of fresh niche prompts (Bulgarian,
Japanese turn 1) show 14.4-15.8% prompt miss and no action: no votes exist.
A hand-forged replay (assistant content spliced by hand) matched 7 tokens
and inherited nothing -- the conversation key hashes the rendered opening,
which diverges for forged structures. Correct: votes are not stealable.

**The real session never fired.** Every continuation turn of the resumed
work session (requests 6, 7, 9, 11, 12, 14, 15, 20...) produced an action
line of 0 experts: its conversations' voiced demand was already resident,
loaded by those conversations' own earlier boundary/decode adaptation.
This is the "already covered" class, and it dominated the real traffic in
this window.

**Cost note.** The firing loaded 2,987 MB of niche experts that the work
traffic does not want, displacing the globally weakest not-demanded
residents; they persist until outcompeted. The demo perturbed placement
by ~3 GB; the arena re-plans as work traffic resumes.

Interim verdict: mechanism validated (miss collapse and acceptance gain on
the firing class; correct refusals everywhere else); market unproven (zero
firings on real continuations in the first day). Keep harvesting passively;
the join (open item 3) and the class frequency decide keep-or-rip.

## What stage 0 killed (do not retry unknowingly)

- **The ring-scale market is thin.** The next 512 experts outside the resident cut hold
  ~2.2% of demand mass in the live history, the next 128 ~0.55% (GLM, 2026-11-06). A warm
  tier, ring reload priority, or any read-budget lever caps there: dead.
- **A trained forecaster or draft-oracle prefetch** buys the same thin slice at thousands of
  reads: dead. The two histories' top sets churn (Jaccard 0.669 between the aging rules),
  but the mass just outside any cut is too small to chase.
- **The diffuse tail is a memory question.** 12-17% of demand mass sits beyond +2,048 ranks
  of the cut, spread over thousands of experts. No load lever reads that; it moves only with
  the slots/context allocation.
- **DS's own-traffic tail was thin.** The Nov-3-era demand history: the 10,000 residents
  covered 97.9% of accumulated demand mass; the next 512 outside the cut ~0.5%. The
  hot-read gains measured on public held-out prose describe public demand, not this traffic.
