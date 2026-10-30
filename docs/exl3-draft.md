# EXL3 draft projections: `DSV41_EXL3_DRAFT_DENSE`

Status (2026-10-11): built, measured, off by default. Two paired 24-prompt gates of one
configuration pool to **−1.05 ± 0.37 ms/step** (−1.4% of a step) with **24/24 bit-identical greedy
outputs** and acceptance unchanged (−0.0070 ± 0.0128 accepted tokens/step). The win is modest because
two earlier levers had already taken most of the draft's weight bytes.

## Why the drafter is worth touching at all

The DSpark draft only produces proposals. The verifier accepts exactly the target's argmax, so
greedy output is identical under **any** drafter (docs/decode-dynamic-depth.md, "How"). A worse
drafter costs throughput, never a token. That makes the draft blocks the one place in this engine
where a weight format is purely a speed question — everywhere else (routed experts, dense attention,
the LM head) a format change is a quality change and needs the nats gate.

It is still opt-in: acceptance moves both ways, and a format change is a different-weights
comparison, not a kernel-order one.

## What a draft pass actually costs

Rank-0 kernel traces from `tools/bench_round_breakdown_tp.py`, per draft pass (3 blocks), analysed
with `tools/decode_graph_families.py`:

| | 2026-10-07 | 2026-10-11, off | 2026-10-11, on |
| --- | ---: | ---: | ---: |
| **span** | **10.40 ms** | **7.05 ms** | **5.66 ms** |
| drafter LM head | 2.95 (bf16) | 1.47 (fp8) | 1.47 (fp8) |
| Markov bias head | 1.56 (bf16, 5×) | 0.79 (bf16, 5×) | 0.79 (bf16, 5×) |
| draft block dense | 1.95 (fp8) | 3.09 (fp8) | 0.93 (EXL3) |
| routed experts (fp4) | 1.20 | 1.12 | 1.12 |
| communication | 0.72 | 0.46 | 0.46 |
| fp32 / mHC / norms / other | 0.94 | 1.24 | 0.85 |

Two things this table says that the coarse phase timers could not:

1. **The two heads were 43% of the old draft pass.** The 2026-10-07 trace's "4.51 ms of BF16
   matmuls" was one 2.95 ms call reading a 662 MB bf16 LM head (grid `[8,505]`, 505×128 = 64,640 =
   vocabulary/2 at 224 GB/s) plus five 0.31 ms calls reading a 66 MB Markov head. Nothing to do with
   the draft blocks.
2. **The draft's own projections were only 1.95 ms**, and are the only part a pack can shrink.

The 2026-10-07 → 2026-10-11 drop from 10.40 to 7.05 ms is not this change; it is the 2026-10-08
levers (`DSV41_TP_DRAFT_HEAD` + `DSV41_DRAFT_MARKOV_TP` −1.79 ms/step, `DSV41_TP_DRAFT_ATTN`
−1.67 ms/step), which attacked exactly the two head rows.

## The pack

`tools/pack_exl3_draft.py` writes `exl3-draft-r{rank}of2.safetensors` (~195 MB a rank) from the same
EXL3 checkpoint the expert packs come from. Every draft matrix in it is **4-bit** against 8 for the
FP8 weights the draft reads otherwise, so the draft's dense bytes halve.

It is a **separate file** from `exl3-dense-r{rank}of2.safetensors` (2.1 GB, backbone attention and
shared experts) on purpose: the draft is a speed-only lever, so enabling it must not force the
backbone pack to be resident and must not change the backbone's numerics. `DSV41_EXL3_DRAFT_DENSE`
therefore works under `EXPERT_FORMAT=fp4` too, and the boot guard checks the two packs share one
source digest when both are loaded.

Slices follow `engine/tensor_parallel.shard_attention` exactly, which is what `DSV41_TP_DRAFT_ATTN=1`
applies to `mtp.*`: `wq_a`/`wkv` whole, `wq_b` on heads, `wo_a` this rank's 4 of 8 groups, `wo_b`
through `OutputParallel`, and — unlike the backbone — the draft's shared experts **whole**, because
the draft's shared expert is not TP-split (`engine/fastdecode.py` `_draft`: another collective there
would double its contribution).

`tools/test_exl3_draft.py` checks three separate claims: the kernel against `exl3_ref.dequantize`,
the packed rank slice against the **rank slice of the full source matrix's decode** (0.00e+00 — a
wrong N offset would otherwise produce a plausible-looking drafter), and the shared expert's
`w1‖w3` group order. Plus row invariance, because the graph is captured once and replayed at a fixed
`T`.

## Wiring

* `engine/exl3_draft.py` — loader, `Exl3Draft`, `attach(mtp)`; `_x3` for `_attention`/`_final`,
  `_x3sh` for `_shared_ffn`.
* `engine/fastdecode.py` — the `if mtp_last is None else None` guard on `x3` is gone (it predates
  the draft pack; `_x3` is only set on the blocks we chose to attach it to). `_shared_ffn` gained an
  EXL3 branch, and `_final`'s drafter-ring seed now reads `_x3.wkv`, because a ring value a draft key
  is compared against must come from the same projection the draft reads — otherwise the change
  would move only half of the draft's keys.
* `engine/v41_engine.py` — construction, attach, boot-guard and health entries. Requires
  `DSV41_TP_DRAFT_ATTN=1`, `DSV41_TP_LINEAR_LAYOUT=output`, and the EXL3 JIT kernel.

`merge_decode_projections` still merges the MTP blocks, so ~100 MB a rank of merged FP8 tensors are
dead when the flag is on. Deliberate: the off arm of an A/B has to be the *shipped* FP8 spelling, not
a slower unmerged one. Skipping the MTP merge once this is kept is a separate change worth ~100 MB of
arena.

## Measured, and rejected

**The two heads at fp4 (2026-10-11).** `DSV41_MARKOV_HEAD_FMT` and `DSV41_DRAFT_HEAD_FMT=fp4`, paired
in one process through `DSV41_BENCH_DRAFT_HEAD_SRC`:
**+0.29 ± 1.03 ms/step** (no effect) and −0.018 ± 0.014 accepted tokens/step (slightly worse).

This is the useful part of the result. The drafter LM head already reads at ~225 GB/s, i.e. at the
box's plain-read ceiling, so it can only be won by reading fewer bytes — and fp4 *does* read half as
many (331 MB → 165 MB) without being faster, so the FP4 Triton kernel's cost per byte at M = 5/6 eats
the whole saving. The Markov head is the same story from the other side: it is five sequential M = 1
GEMVs against 33 MB, inherently serial, and `_lin` already dispatches a quantized weight to its
kernel. **Draft bytes are not the lever any more; the heads are at the memory ceiling and the rest is
launch count.** The next step for the draft is fusing its elementwise/mHC/norm chain (335 kernel
nodes a pass against TensorFold's 147), not a smaller format.

**Follow-up, same day: fewer rows did what a smaller format did not.** The fp4 arm above changed
the kernel along with the bytes. Keeping the fp8 and cuBLAS kernels and reading fewer *rows* --
the heads over a listed vocabulary of 40,960 ids a rank (`DSV41_DRAFT_VOCAB`, docs/draft-vocab.md)
-- took the draft head 1.54 -> 0.94 ms and the Markov GEMVs 0.80 -> 0.27 ms a pass (more than the
rows alone, consistent with 21 MB of listed Markov rows fitting the L2 that the 33 MB half could
not), and the step **-1.06 +- 0.37 ms**,
acceptance -0.0013 +- 0.0013. "At the memory ceiling" was right about the format; it was not the
end of the bytes lever. Separately, the +0.29 +- 1.03 above carries the A/B driver's first-run
penalty (2026-10-11); it is unbiased but its error bar is roughly twice what
`--warm-each` gives.

## Where the remaining gap is

TensorFold's draft graph is 3.716 ms for the same three blocks (its `docs/DECODE.md` §2; measured
on our pair in `results/round-breakdown-20261007/`). Against our 5.66 ms:

| draft graph, ms a pass | ours | TensorFold |
| --- | ---: | ---: |
| draft block dense | 0.93 (EXL3 4-bit) | 1.68 (EXL3) |
| routed experts | 1.12 | 1.08 |
| drafter LM head + Markov bias | **2.26** | ~0.26 (`_site` + `_chain` + `_finish_k`) |
| communication | 0.46 | 0.17 |
| mHC / attention core / norms | ~0.85 | ~0.41 (`mhc_boundary`, `attn_kernel`, `_rms`) |

We are ahead on the blocks and level on the experts; the whole remaining difference is the drafter's
two heads and the launch count of the surrounding elementwise chain. The heads are at this box's
plain-read ceiling and were measured not to respond to a smaller format (above), so the draft side is
down to fusing its 335 kernel nodes — the same exercise `DSV41_DECODE_LEAN` did for the verify path,
which the draft does not use (`lean_step` is gated on `mtp_last is None`).

## Not the same lever: the verifier's head

`DSV41_HEAD_FMT=fp8` was measured in the 2026-09-11 addendum (`RESULTS.md`) at **2.96 ms against
bf16's 5.68 ms** for +0.0004 coding / +0.0014 general nats, and the addendum made it the default.
The profile later moved to lossless `DSV41_HEAD_KERNEL=packed` instead (+2% decode, bit-identical),
and the two are mutually exclusive. That is ~2.7 ms/step of the *verify* step — four times the whole
remaining draft gap — sitting behind a quality decision, not an engineering one. Measure it before
assuming either profile is the right default.
