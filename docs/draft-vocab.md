# Listed draft vocabulary: `DSV41_DRAFT_VOCAB`

Status (2026-10-11): built, measured, on in the two-node example profiles (code default off). At 40,960 listed ids a rank, paired over the
24-prompt gate with a warm-up before every measured run: **-1.06 +- 0.37 ms/step, +0.56 +- 0.22
tok/s, acceptance -0.0013 +- 0.0013**, 24/24 greedy outputs bit-identical. The draft graph goes
6.07 -> 4.58 ms a pass in rank-0 traces.

## The lever

The drafter's two heads were 2.26 ms of a 5.66 ms draft pass after `DSV41_EXL3_DRAFT_DENSE`
(docs/exl3-draft.md): the fp8 vocabulary head (331 MB a rank, 1.47 ms) and the Markov bias head
(33 MB of bf16 read five times, 0.79 ms). A smaller *format* did not make
them faster (2026-10-11). This changes their *row count* instead: the draft scores only a
listed vocabulary, through the same kernels at the same bytes per row.

The draft only proposes and the verifier accepts exactly the target's argmax, so a token the draft
cannot propose costs acceptance at that position, never a token of output. The question is
entirely bytes saved against tokens unlisted.

## How it is wired

* `engine/draft_vocab.py` reads a ranking file (ids, most frequent first), takes its first
  `2 x DSV41_DRAFT_VOCAB_ROWS` ids, sorts them, and gives rank 0 the lower run and rank 1 the upper.
  Contiguous runs keep every rank-0 id below every rank-1 id, so "rank 0 wins a cross-rank tie" and
  "the lower column wins" are still "the lower id wins", as in a full-vocabulary argmax.
* The runs are **not** the verifier's vocabulary halves. TensorFold lists a fixed quota per half
  (`draft_head.per_rank`); on this tokenizer the common tokens sit mostly below id 64,640, and an
  equal quota wasted rank 1's rows: held-out UltraChat missed 2.13% per half against 1.17% global
  at 24,576 a rank. Rank 0's run ends at id 50,707 with the v1 list at 40,960 a rank.
* `engine/tensor_parallel.ListedVocabHead` is the base `VocabParallelHead` with a column-to-id map:
  the split greedy chain and the candidate exchange are unchanged apart from that map (two hooks,
  `_column_id` and `_gathered_ids`; the plain head's path computes the same tensors as before).
  `tp_logits` scatters into a [T, V] row of -inf, so the sampling draft, the global shortlist and
  the tree probe see the shape they always did; an unlisted token gets q = 0 and rejection sampling
  stays exact.
* `engine/model.py` builds the head from the full `head.weight`, which every rank loads before
  taking its verifier shard; `FastDecoder.set_markov_fmt` gathers the Markov rows for the same ids.
* The boot guard pins the list's digest (`draft_vocab`): two ranks drafting from different lists
  emit different drafts with nothing in the log.

Requires `DSV41_TP_HEAD=1`, `DSV41_TP_DRAFT_HEAD=1` and a draft head format of fp8, fp4 or bf16.
The split chain (`DSV41_DRAFT_MARKOV_TP=1`) is where the Markov saving lands; without it the greedy
draft still reads the full Markov head.

## The list

`engine/draft_vocab_ranking_v1.txt`; its header records the sources and weights.
`tools/build_draft_vocab.py` writes one. Held-out miss rate (target token not listed):

| rows a rank | accept-ab 24 replies | UltraChat (held out) | docs en | ja | zh | ko | ru |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1.46% | 0.26% | 0.07% | 0.19% | 0.64% | 0.08% | 0.26% |
| 40,960 | 0.28% | 0.04% | 0.01% | 0.07% | 0.43% | 0.04% | 0.21% |

Two traps found on the way, both worth knowing before building another list:

1. **The lowest ids are not the common ones on this tokenizer.** TensorFold fills a short list
   with each half's lowest unlisted ids; ordering by id alone covered 87.5% of held-out English docs
   at 24,576 a rank, against 99.5% ordered by counts.
2. **The A/B driver's prompts recur all over results/.** `prefill-trace-20261008` held 35 of its
   42 replies, `exl3-tune-20261008` 8, the generation gates one each. A ranking built with them in
   showed 0.59% missed on the A/B replies at 32,768; with them removed it is 1.46%.
   `tools/build_draft_vocab.py --exclude-replies-of` drops them.

## What it costs and saves

Single GPU, the real `head.weight`, L2 flushed before each timed call (median of 50):

| per draft pass | full vocabulary | 40,960 listed | delta |
| --- | ---: | ---: | ---: |
| draft head, fp8, 5 rows | 1.575 ms | 1.030 ms | -0.55 ms |
| Markov bias, 5 x M=1 bf16 GEMV | 0.76 ms | 0.27 ms | -0.49 ms |

The head scales with its rows (0.63x the rows, 0.65x the time). The Markov chain does better than
its rows (0.63x the rows, 0.35x the time), which is what the later of five back-to-back reads
hitting the 24 MiB L2 would look like: 21 MB of listed rows fit it, the 33 MB half never could.
(Inferred from the timings; no L2 counters were read.) Memory: the listed fp8 head
is 210 MB a rank against 331 MB; the Markov rows add a 21 MB copy.

## Measured in the engine

**Draft graph, rank-0 traces** (`tools/bench_round_breakdown_tp.py --bursts 12`, one process per
arm, `tools/decode_graph_families.py --draft 11`; `results/draft-vocab-20261011/trace-{full,listed}/`):

| draft pass, ms | full vocabulary | 40,960 listed |
| --- | ---: | ---: |
| **span** | **6.07** | **4.58** |
| fp8 draft head (`_fp8_linear_kernel`) | 1.54 | 0.94 |
| Markov bias (bf16 GEMV x5) | 0.80 | 0.27 |
| communication | 0.48 | 0.36 |
| other | 1.77 | 1.61 |
| routed experts | 1.18 | 1.14 |
| kernel nodes | 335 | 330 |

The Markov GEMVs also changed cuBLAS kernel (`32x32_128x1` to `16x16_128x1` tiles) at the smaller
N. Communication probably falls because the five per-position exchanges wait less for the peer's
GEMV (not separated in the trace). The draft is on the critical path in both traces: the next
kernel starts ~2 us after the draft's last, and the host had queued it ~2.2 ms earlier.

**Paired A/B** (`tools/bench_accept_ab_tp.py --a vocab_full --b vocab_listed --warm-each`, one
process loaded with `DSV41_DRAFT_VOCAB_ROWS=40960 DSV41_BENCH_DRAFT_VOCAB_AB=1`, frozen map, depth
3, `DSV41_EXL3_DRAFT_DENSE=1`; `results/draft-vocab-20261011/ab-40960-warm/`):

| | delta +- SE |
| --- | ---: |
| ms/step | **-1.06 +- 0.37** (70.58 -> 69.52) |
| accepted tokens/step | -0.0013 +- 0.0013 (one prompt, 2.16 -> 2.13) |
| tok/s | **+0.56 +- 0.22** |
| greedy outputs | 24 / 24 identical |

The same A/B without `--warm-each` read **-0.02 +- 0.70** (`ab-40960/`). That run is not a second
opinion; it is the driver's first-run penalty (2026-10-11, "the first run of each
prompt"): split by which arm ran first, it gives -1.56 and +1.68, i.e. an arm effect of +0.06 with
a +1.62 ms penalty on whichever arm went first, and a per-prompt SD of 3.3 ms against 1.74 with the
warm-up.

## Not measured

* Rows between 32,768 and 40,960: 32,768 leaves 1.46% of the A/B replies unlisted, enough to cost
  about what the rows save, so it was not run.
* A ranking built from this model's own replies. v1 is built from other corpora; replies from the
  target itself are the right source and `tools/build_draft_vocab.py` takes them.
* Sampled decoding: `tp_logits` scatters into -inf, which keeps rejection sampling exact, but no
  temperature > 0 A/B was run. The sampled draft also still reads the full Markov head.
* A fallback to the full head for text outside the list (TensorFold's `TF_DSV41_DRAFT_VOCAB_FALLBACK`).
  Chinese docs miss 0.43% at 40,960; a language the list does not cover would lose acceptance, not
  output.
