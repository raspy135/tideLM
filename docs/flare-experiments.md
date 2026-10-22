# Flare-inspired decode experiments (2026-10-21)

Baseline: `87b336a`. The tracked tree was clean before work began; existing local
`bench/results/2026-10-21/` and `output/` artifacts are not part of these changes.
The implementation branch is `codex/flare-perf-experiments`.

All experiments default off. No serving profile has changed, no build or GPU
benchmark has run, and there is no measured speedup. GPU use remains paused at
the user's request. The shared rank guard compares the flags and new numerical
kernel sources; health reports them as `decode_experiments`. Prefix persistence
already hashes engine sources and DSV41 settings, so different arms cannot reuse
each other's disk prefixes.

## Runtime paths

| Switch | Change | Required qualification |
| --- | --- | --- |
| `DSV41_INDEX_CANDIDATES=1` | Layer 20's existing block mask becomes a fixed-capacity, ascending list. Later indexers score that list when the context bucket exceeds its capacity. BF16 query/key storage and serving top-k are retained. | Fused score reduction can change near-ties: compare selected keys, held-out logits/NLL, long-context retrieval, output structure, and acceptance. |
| `DSV41_HC_POST_PRE=1` | Attention's residual update and the lagged FFN pre/RMSNorm share one kernel, in the backbone and draft. Both BF16 boundaries and 1024-column norm tiles are retained. | In-place residual and normalized output must pass the bytewise component test, then full-model quality/acceptance checks. |
| `DSV41_DRAFT_ROUTER_FUSED=1` | Reuse the existing fused router prologue and normalization for DSpark's top-3 gate. Gate scratch is keyed by expert count as well as query width. | Greedy target equality, sampled rank agreement, and measured acceptance. The existing BF16 router option also applies to draft gates in this arm. |
| `DSV41_ATTN_DIRECT=1` | One unpacked key tile feeds QK and PV in a program retaining the full output accumulator. No full gathered-key or probability buffer. | GPU resource limits, all-masked rows, graph replay, numerical error, held-out quality, and full-round latency. Online softmax changes reduction order. |
| `DSV41_ENGRAM_ASYNC_H2D=1` | Upload and dequantize on a side stream while the preceding segment runs. Main waits on a ready event and retains the result through its consuming copy. | Real pinned-staging reuse, graph consumers, repeated rank parity, and full-round time. One lane and unsplit reads only. |

The Engram GPU-timing field `engram_ready_wait_ms_per_step` is the local exposed
row-wait/upload interval after the preceding graph. It is not the sum of disk
reads, not a measurement of the peer's propagated stalls, and is not subtracted
from depth-policy costs. Head-sharded Engram projections and paired readiness
cost calibration remain separate follow-ups; no collective was added here.

### Keep the negative attention result

The previous direct-cache tile implementation lost **236 us versus 47.7 us**
for BF16 staging at T=4, H=32, N=1152 (`docs/gotchas.md`, October 7). It unpacked
the same keys separately for every 32-column PV tile. The new experiment reuses
each tile for QK and the full 512-column PV accumulator. Key partitions supply
more blocks at small query widths; a merge combines their partial accumulators.
It deliberately does
not expose the rejected kernel as a new serving option. The new reuse schedule
may still lose on occupancy or arithmetic; no gain is claimed without a number.

## Validation and measurement

CPU checks cover candidate membership, source-to-decoder handoff through the
actual indexer, causality/padding, flag validation, rank-guard inputs, and the
existing speculative policy/sampling checks. Syntax and shell checks pass.
`tools/test_perf_decode_cuda.py` is gated by `DSV41_RUN_CUDA_TESTS=1`; it is not
run implicitly by CPU discovery. Component GPU qualification must precede a model
benchmark. None of those GPU tests has run yet.

An existing `engine.test_decode_experiments` fixture fails with 21 subtest errors
in five tests: its extracted function omits `top_p` and its fake engine omits
`_root_width`. The exact same 21 errors reproduce from baseline `87b336a` in a
temporary directory. This change does not repair or suppress them.

Inspect a bounded benchmark plan without importing torch or accessing CUDA:

```sh
python3 tools/bench_perf_decode_tp.py --plan --experiment index_candidates \
  --prompt-tokens 32768 --workload code --depth 3 --pairs 1
```

After GPU use is released and component tests pass, run the disposable two-rank
gate with `GATE_PROFILE=.env.ds41-mxfp4`, an immutable `GATE_IMAGE`, and a matching
`GATE_SOURCE_ROOT` on both nodes. Do not sync edits into a running service.
Choose `GATE_TIMEOUT` for the selected prefill and generation workload; a full
matrix can exceed the gate's default 900 seconds.

`bench_perf_decode_tp.py` freezes the resident map, disables hot expert reads and
prefix reuse, pins depth, warms both complete output trajectories, then measures
off/on/on/off. Separate graph catalogs preserve warm captures. A timed capture,
incomplete output, changed resident map, or cross-rank hash disagreement fails
the run. Draft-only and delivery-only arms additionally require matching greedy
output. Reports retain per-arm configuration, prompt/output hashes, timings,
acceptance, and GPU phase timings. This isolates mechanics; any winner must then
be measured again under the shipped adaptive/hot-read profile and sampled traffic.

Generation-gate passes alone do not establish quality. Keep Python parsing,
structural answers, held-out loss, long-context retrieval, and request settings
explicit for target-arithmetic changes.

## Remaining residency and workload work

`engine/scale_blocks.py` implements a standalone lossless scale codec and
`scale_blocks_cuda.py` provides explicit CUDA expansion qualification. It is
TideLM's `tidelm-scale-blocks-v1` format, not a reader for b12x CSF. Adjacent
exponent palettes, one-bit codes, exact byte exceptions, and raw fallbacks
compress the existing bytes without changing quantization. Incompressible
planes remain raw. Offsets and exceptions are validated before device address
arithmetic. CPU tests cover every byte value, partial blocks, noncontiguous
inputs, mixed raw/palette blocks, and malformed records. GPU expansion has not
run; its test includes poisoning and reusing caller-owned scratch.

The CPU-only auditor reads scales and weight sizes, never model weights into a
GPU. On local `DeepSeek-V4.1-Flash-1017`, layers 0/20/39 and experts 0/1, output
TP2 slices, every scale byte round-tripped:

| TP half | Native scale bytes | Encoded bytes including offsets | Saving relative to sampled weights plus scales |
| --- | ---: | ---: | ---: |
| rank 0 | 3,317,760 | 1,142,828 | 3.8561% |
| rank 1 | 3,317,760 | 1,142,638 | 3.8565% |

These are six sampled experts per rank, not an all-expert compression result,
an arena-memory measurement, or a throughput measurement. Each sample includes
8,847,360 packed weight bytes and 552,960 native scale bytes per rank. Reports
retain sample hashes under `results/flare-perf-20261021/scale-rank{0,1}.json`.
Reproduce without a model load:

```sh
python3 tools/audit_scale_compression.py --model-dir /path/to/DeepSeek-V4.1-Flash-1017 \
  --layers 0,20,39 --experts 0,1 --tp-rank 0
```

Lossless scale storage needs a separate format/arena integration, not just an
encoder: native kernels index scale planes by global arena slot. Keeping those
planes alongside compressed copies would increase memory. Selected-slot scratch,
slot replacement, async hot-ring lifetime, and both prefill/decode consumers must
be integrated and qualified before claiming additional resident experts.

The earlier deeper-copy verification and checkpoint-compatible EXL3 draft pack
ideas remain queued. No old-checkpoint quantization pack is enabled for 1017.
Drafter training/calibration and the temperature-floor idea remain out of scope.

References inspected read-only: Flare checkout `692c020`, especially
`ignition/components/mxfp4_indexer.py`, `mhc.py`, `compressed_mla.py`, and
`readiness.py`. This implementation retains TideLM's adaptive arena and arithmetic
choices rather than importing Flare's runtime or its default A8 expert path.
