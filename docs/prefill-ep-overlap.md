# Prefill EP overlap

> **Removed 2026-10-17.** The overlap path, `EPDistributed.combine_async` / `finish_combine`, their
> tests and `tools/bench_ep_overlap_engine.py` are gone (git history has them); a nonzero
> `DSV41_PREFILL_EP_OVERLAP` now refuses to boot. Measured below: no reliable gain. This page stays
> as the record -- including the join race, which matters to anyone who tries it a third time.

Default off: `DSV41_PREFILL_EP_OVERLAP=0`. The opt-in path overlaps the routed expert
all-reduce with the independent shared expert, then adds the two FP32 outputs and rounds
as before. Decode is unchanged. The flag and completion-join version are in the EP boot guard.

## Original race and repair

`EPDistributed.combine_async` launches `all_reduce(..., async_op=True)` from a user-created
CUDA stream. ProcessGroupNCCL actually runs the collective on an internal NCCL stream.
The old consumer waited only on the user-created stream and ignored the returned Work.
That does not order reading the reduced tensor after the NCCL operation. Stale or partially
reduced routed values can feed later routers, consistent with the previously observed
late out-of-bounds failure (the exact original device assertion was not traced).

After queueing shared-expert work, the caller now invokes `Work.block_current_stream()`
through `EPDistributed.finish_combine`. This establishes the real collective completion
dependency on the consuming stream and returns immediately, including when the process
uses `TORCH_NCCL_BLOCKING_WAIT=1`. The partial tensor remains live through the join and
addition. The communication and reduction order, output dtype, and shared-expert arithmetic
are unchanged. The explicit producer-stream dependency before launch is retained.

This follows the installed PyTorch ProcessGroupNCCL header's requirement to synchronize
the consuming stream with the Work, and the public distributed Work API:
https://docs.pytorch.org/docs/stable/distributed

## Tests

`engine.test_ep_overlap` checks that completion is joined through Work, not by waiting
on the launch stream. `tools/test_ep_overlap_cuda.py` is a two-node test for use with
serving stopped: delay rank 1 before launch, add an independent shared tensor, consume
the result, then verify exact sums. It tests token sizes 1/16/128/512/2048 three times
each, including allocator reuse. The legacy control must exhibit a race; the repaired
path must have zero mismatches. It never uses intentionally incorrect control values
as indices, avoiding a full-model device assertion.

The pre-deployment overlap-off baseline at PRUNE_KEEP=.60, arena=88 GB, adaptation on,
replicas off measured 782.45 / 821.28 tok/s on two 7,751-token README requests with no
prefix reuse. Both 6,082-token depth-8/depth-10 nesting controls passed. Adaptive state
changes between requests, so serving on/off timings alone are not a tightly controlled
causal performance estimate.

## Results (2026-09-15)

Image `0b615614f838ec9466e14626eed4bbd2b7b4379729ecfd04bf3aace81522847e` was built once
and verified on both nodes. The delayed-peer CUDA test produced **28 mismatches / 30
rank-trials** with the legacy join and **0 / 30** with the repaired join. Both 6,082-token
nesting controls passed with overlap enabled and had the same answer hashes as off.
The enabled serving README series measured 662.43 / 729.73 / 828.03 tok/s, insufficient
to distinguish a benefit from the earlier off baseline and the known warmup variability.

`tools/bench_ep_overlap_engine.py` therefore loads the full engine once per node, disables
swaps and prefix caching, suppresses demand DB writes, and broadcasts each off/on choice
before generating. It verifies the pruning-mask hash and expert-generation counter stay
unchanged. This standalone test uses a raw-token README prompt (7,696 tokens, 16 output
tokens), not the 7,751-token HTTP chat-template prompt. First off/on calls warm each engine;
the following four calls are the measured pairs. A second load reverses their order:

| experiment | measured modes in order | prefill seconds in order |
|---|---|---|
| off then on | off, on, off, on | 10.705, 9.845, 8.730, 7.623 |
| on then off | on, off, on, off | 9.007, 8.006, 7.950, 7.845 |

All four warmed outputs matched exactly within each experiment (token hashes). The first
cold overlap-off output differed from warmed outputs in both experiments, while the later
off and on outputs matched. This is an unresolved cold-start reproducibility observation,
not evidence that overlap itself changes the warmed output, nor proof of broad quality parity.

The apparent forward-order gain did not survive reverse ordering: the final warm pair was
968.08 tok/s on versus 980.95 off. Continued warmup dominates this small sample. **No reliable
end-to-end speed gain is established.** Keep the repair available but leave the serving flag
off; adaptation remains on, replicas and detailed instrumentation off. No extra memory is
required by the repair beyond the already existing stream/collective resources.

The standalone full-engine containers stalled during process-group/graph teardown after
publishing their completed results and were explicitly stopped to reclaim memory. The
harness now handshakes after assertions, flushes its results, and exits without that graceful
teardown; this exit-path adjustment has only been syntax checked, not rerun through another
model load. It does not alter the measured forward path or serving code.
