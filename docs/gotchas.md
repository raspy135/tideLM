# Gotchas

Every one of these was hit for real, or is written down because the code had to be shaped
around it. Sources: [`NOTES.md`](../NOTES.md) and [`LIMITATIONS.md`](../LIMITATIONS.md).

## The box hard-resets when host memory goes negative

There is no OOM killer moment to catch and no log line to find afterwards. On unified
memory the GPU allocator and the page cache draw on the same pool; push `MemAvailable`
towards zero and the machine goes down. Everything below follows from that:

* The engine keeps a hard `keep_free_gb` floor (20 GB) under `MemAvailable` when sizing the
  arena, and the auto factor was lowered from 0.88 to 0.82 after a warm start peaked at
  111 GiB of 121 with 10 GiB left.
* `start.sh` and the container entrypoint **refuse to start** below `MIN_FREE_GIB` (90),
  naming the processes that hold the pool.
* `stop.sh` and `./run.sh stop` block until the pool is actually back. Pinned and
  page-cache-backed memory is reclaimed lazily; starting the next server before that lands
  is the classic way to wedge the box.
* `compose.yaml` uses `restart: on-failure:1`, not `unless-stopped`. A load that fails
  usually fails because the pool is spoken for, and restarting into the same condition
  turns "the server did not start" into "the power button is the only way back".

## Do not set a memory limit on the container

`--memory` / `mem_limit` on this hardware is a cap on **GPU** allocations too, and the
arena auto-sizer will silently shrink to fit it — giving a small hot set and a permanently
NVMe-bound server with no error anywhere. There is no cgroup knob that means "host RAM but
not GPU" here. Leave it unset.

## `torch.cuda.mem_get_info()` lies on GB10

It counts the page cache as used. Right after the checkpoint download it reported **32.0 GB
free on a box with 99.9 GiB `MemAvailable`** — which would have handed the engine a 23 GB
arena, about 7% of the routed experts, and a server that streams every token. The engine
now takes the larger of `mem_get_info()` and `/proc/meminfo`'s `MemAvailable` and logs
both. If you write anything that sizes a buffer from free memory on this box, do the same.

## The transient ring must hold a full layer

Prefill touches nearly every expert of a layer (370–381 of 384 measured at layer 0), so
prefill misses go to a small ring of transient slots instead of through the LRU — otherwise
every prompt evicts the hot set. The ring's size is not a tuning knob with a soft floor:
below 384 slots it **wraps inside a single layer** and the engine computes that layer with
the wrong experts. No exception, no warning — just an 0.88 relative error in the output,
which is exactly what the first smoke test produced. Default 400. Do not lower it.

## An expert is two reads, not six and not one

The six tensors of an expert (`w1/w2/w3` × weight and scale) are not adjacent in the shard:
all the scales sit near the front, all the weights far behind, but within each group an
expert's three tensors are contiguous. Read it as six `preadv`s and you pay six `O_DIRECT`
round trips — and at a large arena a decode step misses only about one expert per layer, so
nothing else is in flight to hide the latency. The rate collapses from ~4.7 GB/s to
~0.6 GB/s. `ShardFile.expert_runs` groups them into exactly two runs: 1.1 MB of scales,
17.7 MB of weights.

## `O_DIRECT` alignment, and the last tensor of a shard

`O_DIRECT` reads must start and end on 4 KiB boundaries and land in an aligned buffer, so
every read is widened outward to `ALIGN` and the wanted bytes are sliced out afterwards. Two
consequences that bite:

* The staging buffers are pinned and over-allocated by `8 × ALIGN` for exactly that
  widening.
* The **aligned tail can run past EOF** on the last tensor of a shard, and the short read
  that comes back is not an error — the loop tops up until it has what it asked for and
  accepts a short final read. Code that treats a short `preadv` as a failure will break here
  only on some shards, which is the worst way to find out.

And the whole path needs a real filesystem: `O_DIRECT` does not work on overlayfs, so the
checkpoint must be a **bind mount**, never a `COPY` into the image or a network filesystem.

## Engram rows are too small for `O_DIRECT`

48 random 264-byte reads per token. `O_DIRECT` on those is all overhead: the engram reader
uses **buffered** `preadv` in a thread pool with `POSIX_FADV_RANDOM` and a small
process-local row cache, and it is not the bottleneck (3,792 rows in 0.65 s on a 64-token
generation). Do not "optimise" it into the `O_DIRECT` path.

## The CDN rate-limits single-range requests

`tools/engram_rows.py` fetches only the n-gram rows a corpus needs, straight out of the two
101 GB shards on the Hub, without downloading them. Single-range requests get HTTP 429 at
around 400/s. **Multipart** range requests are the way: 500 rows per request, ~314 requests
per layer, 10–11 s for a 10,760-token corpus. If you rewrite that fetcher, keep the
multipart batching or you will be rate-limited into hours.

## cuBLAS makes the same GEMM give different bits

Two separate mechanisms, both of which had to be disarmed for the engine to be chunk-invariant:

* **split-K reduction in bf16.** `F.linear(x[:6], w) != F.linear(x, w)[:6]` by ~2.4e-3 on
  the N=512 `wkv` projection, and the attention softmax amplifies that ~2× per layer.
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` cuts it to
  ~9e-5.
* **tiling chosen from M.** cuBLAS picks its tiling and split-K from the number of rows, and
  for several shapes even a row's *offset inside* the tile changes its last bits — 8 and 16
  are offset-invariant, 32/64/128 are not. So every activation GEMM and last-dim reduction
  runs in fixed 16-row tiles. It costs about +10%: a 512-token prefill chunk issues 32 GEMM
  launches per projection instead of one.

## Bit-exactness stops at 512 tokens

Beyond ~1024 tokens of context the indexer's top-512 stops keeping every visible compressed
position, and which 512 it keeps is decided by scores computed against a key cache whose
*length* differs between a chunked and a single-chunk run. Ties then break differently and
the two runs can select different positions. The engine is close but not identical past that
point — and so is the reference implementation, for exactly the same reason. Do not write a
long-context regression test that asserts equality.

Related: `Caches.rollback(n)` only restores the compressor's pending token if `n` lies inside
the last forwarded chunk (or exactly at its start). Further back it raises rather than
silently producing a wrong latent. That covers the speculative-decoding use and nothing else.

## 16-byte tiles cap at ~130 GB/s on GB10

The FP4 MoE kernel is bandwidth-bound on the 18.8 MB every active expert weighs, so tile
width is not a micro-optimisation. Narrow (16-byte) tiles top out around 130 GB/s on this
part; the 64-byte-wide tiles the kernel uses reach 193–197 GB/s effective at decode sizes,
against a ~210 GB/s practical copy ceiling on a nominally 273 GB/s box. If you retune
`BM`/`BN`, watch the effective bandwidth, not the occupancy.

## A native CUDA spelling is not automatically faster than Triton

`tools/fp4_moe_cuda.cu` is a CUDA C++ decode implementation of the same packed-FP4
operation (`DSV41_FP4_CUDA=1`). It uses the hardware E2M1 conversion, coalesced 64-byte row loads,
the same per-32 UE8M0 boundary, BF16 linear-output rounding, a one-kernel stable router, and is
CUDA-graph capturable. After the tuning below, native CUDA with relaxed reduction was selected as
the default on 2026-09-22. The initial negative measurements are retained here as history.

Real layer-0 weights, 32 experts, FP32 routed output, 20 order-balanced A/B/B/A measurements on
GB10 (another resident process made the absolute bandwidth lower than an isolated run):

| tokens / distinct experts | Triton | native CUDA | Triton/native |
|---|---:|---:|---:|
| 1 / 6 | 0.774 ms | 0.847 ms | 1.09x |
| 6 / 30 | 5.453 ms | 9.365 ms | 1.72x |
| 8 / 32 | 5.613 ms | 9.594 ms | 1.71x |

The T=6 relative output error against Triton was `7.73e-7` (`2.29e-6` at T=8), and graph replay
passed. A rows-per-warp sweep over 1, 2, 4, 8 and 16 made one row the least-slow arm; none won.
Triton's tensor-core tile reuses each activation across many output rows, while the CUDA-core
row kernel saves padded-pair math but pays scalar reductions and repeated activation traffic.
These eager timings also include host launch
gaps and different router implementations, so they do not isolate the source of the difference.

A later source-level pass made power-of-two index arithmetic explicit, hoisted weight/scale row
bases, and moved the up kernel's `pair / topk` out of the dot-product loop into packed router
metadata. With the competing worker stopped, two fresh runs of 100 order-balanced measurements
showed a real crossover rather than a general win:

| tokens / distinct experts | Triton range | native CUDA range | result |
|---|---:|---:|---:|
| 1 / 6 | 0.801-0.803 ms | 0.745-0.746 ms | CUDA 1.074-1.077x faster |
| 6 / 30 | 3.028-3.034 ms | 3.775-3.960 ms | CUDA 1.245-1.305x slower |
| 8 / 32 | 3.245-3.300 ms | 4.418-4.661 ms | CUDA 1.339-1.436x slower |

The output errors remained zero at T=1, `7.73e-7` at T=6, and `2.29e-6` at T=8. This is enough
to retain the CUDA arm for single-token investigation, but not to enable it globally: decode call
size varies, and the larger calls still lose. At that stage `DSV41_FP4_CUDA=0` remained the default;
the subsequent CUDA-core tuning below supersedes that decision.

### CUDA-core tuning with graph replay (2026-09-22)

The next review moved the routed-pair loop outside K, keeping only one pair's accumulators and
activation base pointer live. The previous version kept sixteen pairs' state and checked sixteen
validity branches per K tile, despite the usual 1-2 pairs per expert. The new version rereads weights
for each pair, but the smaller register footprint wins on these sparse decode routes. Activation
loads are now one aligned 64-bit load per lane instead of four scalar BF16 loads. The ordered
four-subgroup reduction drops its redundant lane-0 broadcast. These changes preserve the old CUDA
arithmetic order. Up register usage initially fell from 64 to 40 with the pair-loop change.

Use graph replay to exclude host launch gaps. The earlier 7.5% T=1 eager win was **not** proof
that source-level shifts/hoisting accelerated the kernel: the pre-tuning CUDA version was slower
than Triton at T=1 under graph replay (0.725 vs 0.697 ms in the first four-arm comparison).

Two 100-repetition, alternating-order graph runs on GB10 with the serving worker stopped, real
layer-0 weights in a 32-expert arena, top-k=6, FP32 routed output:

| tokens / distinct experts | Triton (ms) | previous CUDA (ms) | tuned CUDA (ms) | relaxed CUDA (ms) |
|---|---:|---:|---:|---:|
| 1 / 6 | 0.697-0.702 | 0.725-0.726 | 0.556-0.559 | 0.534-0.537 |
| 6 / 30 | 2.875-2.890 | 3.679-3.763 | 2.799-2.803 | 2.685-2.727 |
| 8 / 32 | 3.108-3.150 | 4.659-4.866 | 3.099-3.126 | 2.896-2.967 |

The tuned original-order mode was bit-identical to the previous CUDA output in all three cases. The
`DSV41_FP4_CUDA_RELAXED=1` combines even/odd partials before subgroup reduction and accumulates
each subgroup across K before the final warp reduction. It retains FP32 accumulators and the BF16
linear-output boundaries, but changes summation order: relative errors against Triton were
`4.59e-8`, `2.54e-5`, and `2.17e-4` respectively. These are kernel errors, not language-quality
measurements. Following these results, `DSV41_FP4_CUDA=1` and `DSV41_FP4_CUDA_RELAXED=1` were
selected as defaults at the user's request. Explicit `DSV41_FP4_CUDA=0` restores Triton;
`DSV41_FP4_CUDA_RELAXED=0` retains native CUDA with the original summation order. Prefill and
unsupported shapes still use Triton. Full-engine timing and generation-quality gates remain
outstanding; changing the defaults does not expand the measured validation scope.
The relaxed flag and CUDA source digest are included in the EP2 boot-time agreement guard.

Rejected candidates, compared against the vector-load, eight-warp version in graph replay:

* Whole activation-row staging into 10 KiB shared memory per CTA: 0.606/3.034/3.541 ms versus
  0.560/2.783/3.110 ms at T=1/6/8 (8-14% slower), with identical results.
* Four, sixteen, and thirty-two warps per CTA: no consistent improvement over eight; T=8 was
  3.225/3.265/3.494 ms versus about 3.10-3.14 ms with eight.

Six GPU regression tests cover top-k=1/2/3/4/6, up to sixteen pairs on one expert, null experts,
partial output and scatter layouts, output-row tails, graph replay, and long router runs. Native
routing now splits runs longer than sixteen into additional blocks instead of dropping pairs.
Both reduction modes pass. Compute Sanitizer memcheck on the default mode reports zero errors.
No two-rank serving run or full-model generation quality evaluation was performed for this tuning.

Reproduce current arms with `MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_moe_cuda.py
--graph --compare-relaxed --iters 100`. `--baseline-source PATH` adds a saved ABI-compatible CUDA
source as the fourth arm; the previous source SHA256 was
`1f33165b6759be39368eb77605a3ff9013445c22daf730037fc3a54e55d7b0c0`.

### Casting consolidation and predecoded weights: negative results (2026-09-22)

The follow-up tested whether fewer conversions, two-pair reuse, or extra weight memory help the
**already tuned relaxed CUDA default**. Each row below is a separate 100-repetition alternating
CUDA-graph A/B run on GB10, with the serving worker stopped, real layer-0 weights in a 32-expert
arena, top-k=6, and FP32 routed output. Entries are baseline -> candidate milliseconds; compare
within each entry, not across rows (clocks drift). No candidate was promoted.

| candidate | T=1 / 6 experts | T=6 / 30 experts | T=8 / 32 experts |
|---|---:|---:|---:|
| Pair FP32->FP16 activation conversions with half2 | 0.535 -> 0.536 | 2.683 -> 2.685 | 2.913 -> 2.886 |
| Convert each activation tensor once before up/down | 0.534 -> 0.542 | 2.720 -> 2.729 | 2.917 -> 2.917 |
| Share weight decode across two routed pairs | 0.536 -> 0.567 | 2.712 -> 2.852 | 2.894 -> 3.108 |
| Combine four FP4 values' unpacking in one asm block | 0.532 -> 0.532 | 2.722 -> 2.731 | 2.899 -> 2.934 |
| Predecode FP4 codes to FP16; retain group scales | 0.536 -> 1.878 | 2.762 -> 11.383 | 2.879 -> 13.969 |
| Predecode and fold group scales into FP16 weights | 0.535 -> 1.822 | 2.819 -> 11.018 | 3.051 -> 14.082 |

Every candidate was bit-identical to its CUDA baseline on these three real-weight inputs. Relative
errors against Triton remained 4.59e-8 / 2.54e-5 / 2.17e-4; this is not a general guarantee for
prescaling other weights or a language-quality evaluation. All candidates keep FP32 accumulation.

The packed conversion changes show no consistent useful win. A second 100-repetition packed
activation run gave 0.537 -> 0.534 / 2.719 -> 2.691 / 2.894 -> 2.904 ms: the roughly 1% advantage
moved between workloads, and T=8 changed from winning to losing. A second precasting run gave
0.534 -> 0.543 / 2.695 -> 2.701 / 2.882 -> 2.884 ms, confirming no win. Precasting removes repeated
BF16->FP16 conversion from the row loops but adds two conversion launches and scratch buffers;
those costs are included in its times. Two-pair reuse raises up-kernel registers from 40 to 54
(no spills) and loses 5-7% on these sparse routes. It might behave differently on denser routing;
these results do not prove every possible two-pair implementation loses.

FP16 weight expansion grows just the weight payload from 566,231,040 to 2,264,924,160 bytes for
32 experts, excluding scales and the original FP4 copy retained for A/B. Expansion is outside the
timed region, making these optimistic steady-state cache numbers. Even eliminating scale loads
and multiplies does not offset the extra weight traffic: these variants lose 3.4-4.9x. Freeing memory
by reducing context length therefore does not make this particular FP16 CUDA-core cache faster.
This is not a measurement of a different tensor-core GEMM implementation.

The serving CUDA source was restored byte-for-byte (SHA256
`f28eb41f84b0e7b607335ca7837055a0850d9972116da1074e0410089875eede`), keeping both CUDA defaults
enabled. All six GPU regression tests pass in both reduction modes after restoration. No engine settings, serving
allocations, or quality assumptions changed.

Reproduce with the worker stopped and `MODEL_DIR` pointing at the checkpoint:

```sh
python tools/bench_fp4_casting.py packed --iters 100
python tools/bench_fp4_casting.py precast --iters 100
python tools/bench_fp4_casting.py pair2 --iters 100
python tools/bench_fp4_casting.py decode4 --iters 100
python tools/bench_fp4_predecoded.py --iters 100
python tools/bench_fp4_predecoded.py --iters 100 --scaled
```

The benchmark-only patches under `tools/experiments/` are applied to temporary sources with
zero patch fuzz. They are not production switches; patch failures after kernel changes should be
reviewed rather than silently adapting a stale experiment.

### Aggressive half2 arithmetic: no useful speed/accuracy tradeoff (2026-09-22)

The next experiment changes the arithmetic, not just the casts. `half2_group` keeps decoded
FP4 values and rounded activations in packed FP16, uses `__hmul2`/`__hfma2` for the local
products, and performs the three width-8 shuffle/add stages in packed FP16. Only then are
the even/odd partials widened to FP32, combined, scaled, and accumulated across K. No
unscaled partial crosses a 32-element scale boundary. The less aggressive `half2_local`
widens immediately after the local two-term packed sums and retains FP32 subgroup reduction.
Both are benchmark-only patches; neither changes the serving source or boot configuration.

Two 100-repetition alternating graph runs of `half2_group`, real layer-0 weights, top-k=6,
32-expert arena, stopped serving worker, relaxed baseline, FP32 output:

| tokens / distinct experts | baseline -> candidate, run 1 (ms) | run 2 (ms) | candidate relative L2 error vs Triton |
|---|---:|---:|---:|
| 1 / 6 | 0.5361 -> 0.5329 | 0.5362 -> 0.5342 | 0.002738 |
| 6 / 30 | 2.7086 -> 2.7160 | 2.7028 -> 2.7134 | 0.002511 |
| 8 / 32 | 2.9155 -> 2.9156 | 2.8880 -> 2.8836 | 0.002660 |

That is within about +/-0.6% of baseline, with substantially greater numerical error. The
baseline errors remain 4.59e-8 / 2.54e-5 / 2.17e-4. The CUDA 13.0 sm_121a disassembly confirms
actual HMUL2/HFMA2 and packed reduction instructions; up registers fall from 40 to 38 with
no spills. Fewer instructions alone do not establish a speedup on this workload.

`half2_local`'s first 100-repetition run gave 0.5353 -> 0.5340 / 2.7427 -> 2.7101 /
2.9165 -> 2.8721 ms, with errors 0.002147 / 0.001805 / 0.001867. These 0.3-1.5% improvements
are small and all three errors still exceed the existing 1e-3 kernel gate. This is not a
language-quality measurement, and the gate was not loosened to promote either candidate.

The aggressive candidate also fails the synthetic parity checks (seven routing/shape subcases,
relative error 0.0030-0.0033) and the mixed-null parity assertion (0.00318). Graph replay,
down scatter/tail layout, and both routing tests pass. These accuracy failures are distinct
from the following deliberate range stress, not evidence of a routing bug.

The range stress uses a down projection with K=128, all FP4 weights +6, scales 1/128, and
constant BF16 activations. At activation 1024 the exact and baseline result is 6144, while
`half2_group` produces infinity before the scale can bring it back into range. `half2_local`
remains finite there, but overflows at activation 8192 (baseline 49152). These deliberately
large inputs are not a measured serving activation distribution; they demonstrate lost
range, not observed full-model collapse. Neither candidate has a full-model quality test.

Reproduce without changing the serving kernel:

```sh
MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_casting.py half2_group --iters 100 --stress
MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_casting.py half2_local --iters 100 --stress
```

For these approximate candidates the benchmark reports `kernel_error_gate_passed=False` but
continues through all workloads. Its process exit status is **not** an accuracy gate. The
production CUDA source retains the SHA256 recorded above, and its six GPU tests still pass.

### Batching the sampled verifier can waste work after early rejection

The opt-in `DSV41_BATCHED_VERIFY=1` prototype removes per-proposal scalar readbacks and
batches probability calculation. At three drafts/top_p=0.95, synthetic mixed-acceptance
sampler latency improved 0.855 -> 0.719 ms, but first-rejection latency regressed
0.344 -> 0.720 ms because unused target rows were sorted. The absolute saving is small,
seeded outputs change, and no end-to-end speedup is established. It stays default-off;
see [sampling tests and measurements](spec-sampling.md) before enabling or extending it.

### Mapped host weights avoid a copy but can slow repeated reads

GB10's shared physical DRAM does not make the current pinned staging buffer and CUDA arena the
same allocation. `ExpertStore._read_leased` reads SSD data into pinned host memory, then
`_load_into_slot` copies it into the arena. NVIDIA also documents that ordinary `cudaMalloc`
allocations cannot be coherently accessed by the CPU/I/O complex on Spark; see the
[Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).
A zero-copy design needs an appropriate mapped allocation and ownership/synchronization protocol.

`tools/bench_fp4_mapped.py` compares the same native kernel with device weights and pinned CPU
weights obtained through `cudaHostGetDevicePointer`. A 60-repetition alternating graph run,
32 real experts, original-order tuned reduction (`DSV41_FP4_CUDA_RELAXED=0`), gave:

| tokens / distinct experts | device weights | mapped weights |
|---|---:|---:|
| 1 / 6 | 0.565 ms | 0.588 ms |
| 6 / 30 | 2.871 ms | 3.493 ms |
| 8 / 32 | 3.198 ms | 4.434 ms |

Outputs were bit-identical, but mapped reads were 4%, 22%, and 39% slower. This excludes the
initial copy and SSD I/O: it rejects assuming mapped memory is free for a hot resident arena,
not the possibility of saving latency on one-use cold experts. Measure load-plus-compute and
buffer lifetime before adopting it. The serving expert allocator/loader was not changed.

## `persistent_topk` does not fit in GB10's shared memory

This one is inherited, and it is the reason no upstream engine runs this model on one
Spark. The sparse-attention indexer's `persistent_topk` kernel wants 128 KB of shared memory
per block; GB10 has 99 KB. The four-Spark vLLM build works around it with
`top_k_per_row_decode` instead, plus block size 64/128 for the sparse SWA and indexer caches.
Anyone porting a datacenter kernel to this box should expect to meet the same wall — and
`DeepSelect`, DeepSeek's own top-k kernel, is `sm_100a`/`sm_103a` only.

## Triton has to be able to compile for `sm_121a`

The MoE kernel is JIT-compiled on the first call and emits inline PTX
(`cvt.rn.f16x2.e2m1x2`), so `ptxas` inside the container must know the target. Triton wheels
bundle their own `ptxas`, and that copy has shipped behind the driver before — the CUDA-12.8
one in the triton 3.5 wheels could not name `sm_121` at all. The image is therefore built on
`nvidia/cuda:13.0.2-devel-ubuntu24.04` with `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
If a kernel launch dies with something about `gpu-name` or an unknown target, that variable
is the first thing to check.

If Triton cannot compile at all the engine falls back to a dequantise-then-GEMM path — it
is correct and very slow, and it will not announce itself except in `engine_config.kernel`
on `/health`. Check that field before believing a slow row.

## The warm start is not optional (but it is silent)

Without a `coverage.json` the arena is filled in `(layer, expert)` index order, which is a
measurably worse hot set than the traced one, and the only sign is a lower `expert_hit_rate`
in the benchmark. The container's entrypoint auto-discovers the newest
`results/trace-*/stats/coverage.json` and logs which one it used, or says it is falling back
to index order. Read that line.

## First token can be minutes, and startup can be tens of minutes

The socket is bound only after the warm start has read tens of GB off NVMe (measured:
~63 s for the 19 GB of non-expert weights, then 73.2 GB of experts in 16 s), and a cold
prompt misses almost every expert. Health waits are 20 minutes native, 45 in the container,
and any client timeout has to be set accordingly. Nothing is wrong at minute 6.

## Chunk counts lie under speculative decoding

Several accepted tokens arrive per SSE chunk. Use `usage.completion_tokens`; counting chunks
under-reports decode speed by 3–5×. See [benchmarking](benchmarking.md).

## "The GPU is idle" is usually the accounting, and then it is usually the engram

Two separate traps, hit in that order while chasing a decode that showed 70–90% GPU utilization.

**First, `decode_accounting` measured the wrong path.** Its `attn`/`moe` timers live in the eager
forward; with CUDA graphs on, decode replays a captured graph and those lines never execute, so
every number in that dict is the *prefill* figure and the whole decode step lands in
`unaccounted` — which read 46–75% and looks exactly like an idle GPU. Fixed, but the lesson
generalizes: a host wall-clock timer around a graph replay measures queueing, not work. The
numbers that can be trusted are `gpu_timing` (`DSV41_GPU_TIMING=1`, CUDA events on the GPU
timeline) and the `[step timing]` table (`DSV41_STEP_TIMING=1`).

**Then the engram read really was the gap.** `_gather_rows` sized its thread tasks
`max(1024, len(ids) // nt + 1)`. A decode verify block asks for `T_VERIFY × 24 ≈ 144` unique
rows, and 144 ≤ 1024, so every decode gather ran as ONE task — serially, one synchronous page
fault at a time, 83 ms of a 196 ms step. The rows were always correct, so nothing ever pointed at
it. Sizing tasks from the row count took the same gather from 93–105 ms to 4.5–6.0 ms.

What did NOT help, both measured rather than reasoned about:

* **Dropping the EP row split at decode.** Halving 144 rows is worth about a millisecond and
  costs a collective; removing it moved the step time not at all. (Kept anyway — it is size-gated
  now, and at prefill volumes the split is worth ~100 ms.)
* **Pinned staging for the H2D.** It works and it cuts the engram wait 58 → 11 ms/step, and the
  decode rate does not move: the host stops blocking in `step` and starts blocking in `verify`
  instead. Its original note said exactly this, and it is still true for a new reason — once the
  gather is fixed, the GPU is genuinely the bottleneck (`layers_ms_per_step` ≈ 150 of a ~167 ms
  step). Leave `DSV41_ENGRAM_PINNED` off unless something changes upstream of it.
  Re-measured 2026-10-08 after a CUPTI trace showed 0.8–1.5 ms GPU gaps at both Engram graph
  boundaries: same loaded TP2 process, frozen map, fixed depth 3, ABBA ×2, 192 tokens, outputs
  bit-identical on both ranks. Prose 85.41 → 84.99 ms/step, code 85.99 → 86.14 ms/step; run
  ranges overlap. The traced gaps were mostly profiler cost: CUPTI made each segment's
  `cudaGraphLaunch` take 0.55–1.05 ms of host time, and the traced round ran 3.5 ms slower
  than unprofiled steps from the same run. Do not size host gaps from a kernel trace.
  (`results/engram-pinned-20261008/`, `tools/bench_engram_pinned_tp.py`.)

The residual utilization dip is therefore real but small. Anything further has to come from the
GPU side: fewer bytes per step, or more accepted tokens per step (`accept_len` is ~2.5 of 6).

## Prefill halved when the fused attention kernel was defaulted off

### Live-profile regression after the nesting investigation (2026-09-14)

The local serving `.env` still had `DSV41_PREFILL_FUSED_ATTN=0`, despite the engine and
`env.example` defaulting to 1. Restoring it to 1 retains the short (<=128-token) reference-shaped
prefill branch and the corrected expert rounding; it is not enabling fused **decode** attention.

Replayed an existing local 14,393-token model-ready capture with EP2, native FP4,
keep=0.59, spec ON, demand logging and adaptation ON. Prompt/completion content was not printed
or committed. GPU phase timings were enabled. Every full-prefill row reused **zero** prefix tokens:

| configuration | prefill seconds | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| fused prefill OFF | 46.721 | 308.06 | not meaningful: saved request capped output at 2 tokens |
| fused prefill ON, first replay | 23.380 | 615.61 | 14.92 (239 tokens, acceptance 3.16) |
| fused prefill ON, subsequent replay after prefix eviction | 24.909 | 577.83 | 17.15 (221 tokens, acceptance 3.17) |

The unfused softmax phase alone cost 14.587 s, versus 1.210 s with fusion. EP-combine event
time fell from 14.697 to 5.159 s, then varied to 8.238 s; this includes waiting for the peer,
not just network transfer. Adaptive expert generations changed, and completion lengths differed:
these are live-profile observations, not an isolated throughput guarantee. The previously cited
warm 4.5k synthetic-code test with demand logging OFF understated the user's experienced loss.
An unfused cached-prefix decode-only replay produced 181 tokens at 14.82 tok/s, acceptance 2.97;
its cached prefill rate is deliberately not reported as prefill throughput.

Normal-profile nesting after restoration passed 4/4. `tools/test_prefix_invariance.py --tries 1`
was **inconclusive**: adaptation moved expert generation 8 -> 12 during the comparison. This is
not a new prefix-parity pass or a demonstrated parity failure. The historical 800–1000 tok/s
prefill range had not yet been reproduced at this point. Later warmed runs below reached it,
but do not establish consistent speed recovery.

Follow-up: the original 14,393-token capture reached 907.64 tok/s after restoring unrestricted
CPU affinity. A P-core-only trial was not retained: adaptation changed between runs, so the
668.29 -> 689.48 -> 907.64 sequence does not isolate affinity. A new real 14,396-token capture
replayed twice at 588.57 and 862.06 tok/s (24.459 and 16.700 s), both with zero prefix reuse and
2048-token chunks. EP-combine GPU envelopes were 6.993 and 0.516 s; MoE envelopes were 7.352
and 6.184 s. Adaptation remained enabled (expert generations 3 and 5). These are eight equal
chunk boundaries in both runs, not evidence that chunk dispatch costs are equal.

`DSV41_PREFILL_TIMING=1` records per-chunk host-call duration, host gaps, GPU stream envelopes
and GPU gaps on **both** ranks; JSON logs contain positions/timings only, no token IDs or text.
Events are resolved after generation, never synchronized at chunk boundaries. Host duration
includes existing blocking operations; GPU envelopes include idle/collective waits, not just
kernel execution. Do not subtract host and GPU durations as "launch overhead", or subtract
timestamps across ranks. With `DSV41_ATTN_TIMING=1`, both ranks also log their aggregate phases.
These diagnostics locate stalls; a CPU/GPU timeline may still be needed to prove dispatch starvation.

Instrumented EP2 replay of the 14,396-token capture (32 output tokens, zero prefix reuse) measured
667.61 then 762.84 prefill tok/s (21.564 / 18.872 s). Both ranks recorded eight chunks. Host
gaps between calls were 8–22 microseconds; CUDA-stream gaps were 1–4 microseconds. Thus an
expensive pause *between* chunks is not supported by these runs. Dispatch stalls *inside* a
chunk are not ruled out. First-chunk GPU envelopes were ~5.3 then ~3.7 s; most later full chunks
were 2.2–3.0 s. Rank-0/rank-1 aggregate combine envelopes were 2.316/3.633 s in run one and
3.014/2.123 s in run two: the greater waiter switched ranks. The first replay also includes
post-restart warm-up effects; do not treat the difference as an isolated code speedup.

Doubling chunks to 4096 (with RING=8192 to preserve history) was not retained. Same real input,
zero prefix reuse, normal adaptive EP2, 32-token output cap: first new-shape run 61.153 s
(235.41 tok/s), then warmed runs 21.473 and 20.969 s (670.42 / 686.55 tok/s). Four chunks were
confirmed. New-shape first/last chunks dominated the cold run; do not use it as steady-state
throughput. Warm performance did not beat the preceding 2048-token profiles (667.61 / 762.84),
and host MemAvailable fell to ~2.8 GB during the trial. Nesting passed 4/4, but those short
probes do not prove long-context chunk-size parity. Adaptive generations changed, so this is
not an isolated causal comparison. Restore chunk=2048, ring=4096 rather than retain a larger
working set without a demonstrated benefit. The public default remains unchanged.

HC final-output-store cleanup: keep the FP32 normalization scratch and reduction order, but
store the final result directly to BF16 for T>=512 instead of an FP32 store plus torch cast.
`python -m engine.test_hc_output_store` compares against the old store/cast path: bit-identical
at T=1/6/60/64/128/512/2048/2108. Two alternating timing sweeps measured T=2048 at
0.850/0.844 ms old versus 0.674/0.675 ms direct, T=512 at 0.179/0.192 versus 0.157/0.181 ms.
Smaller batches retain the old default (T=128 showed no consistent gain). This is only an
estimated ~0.095 s across 560 full-chunk calls on a 14k-token prompt, not a measured end-to-end
speedup. It is a minor cleanup, not an explanation or solution for the multi-second regression;
do not prioritize more restarts/tuning around this sub-millisecond site.

Rejected global schedule change: `tools/tune_fp4_prefill.py --confirm` alternated software-FP4
down-projection tuples (128,8,3) and (128,4,2) using fixed serving-style routing. FP32 outputs
were bit-identical. Candidate improved T=512 (~22.1–22.5 vs 23.4–23.8 ms), but not consistently
T=2048 (~33.3–33.9 vs 32.8–34.2 ms); retain the current default. Wider up tiles (BN=128)
were substantially slower in the initial sweep. Do not infer whole-engine gains from that sweep.

Capture remains the existing `DSV41_CAPTURE_NEXT` mechanism in `server/app.py`, now taking a
count: it captures the next N requests and disarms itself (`captured_request.pt` for N=1, else
`captured_request_0.pt`, `_1.pt`, ...), so two requests that should share a prefix can be diffed
token-for-token without a second capture mechanism. `bench/replay_capture.py` still replays
the N=1 filename, overrides the old diagnostic output cap, and writes only timings, usage, and
content hashes to a mode-0600 result file. It rejects vision/grammar captures rather than
silently replaying a different workload.

Upstream inspection at `45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f` found that the headline
24–36 decode tok/s rows used CB3 experts, `DENSE_FP4=attn,wo_a`, an FP8 head, and mostly high-
acceptance code. Its newer core changes focus on contribution-based expert ranking and routing
modes, not a replacement fast-decode kernel. Those settings are not a like-for-like native-FP4,
BF16-dense EP2 comparison. No merge or quantization rollback was performed.
See [upstream measurements](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/RESULTS.md#43-shipped-configuration).

### Earlier investigation

`DSV41_PREFILL_FUSED_ATTN` went from `"1"` to `"0"` in `692d5c1`, three minutes after `8ccf9a3`
landed the prefix cache. Neither commit has a message. Prefill went 1045–1135 → 544 tok/s on the
same 4,513-token prompt, and `softmax_attn` went back to dominating attention, which is ~73% of
prefill.

The apparent reasoning: the fused kernel rounds differently from the torch path (~1e-2 relative,
bf16 level), and a cache that resumes a prefill has to agree with itself. But that compares the
wrong two things. Nothing requires the fused path to match the torch path. What must match is a
**resumed prefill against a cold one under the same kernel** — and the fused kernel is internally
chunk-invariant: each query row is its own program and the key axis is reduced in fixed `BLOCK_N`
tiles, neither of which depends on the chunk length. (`split` must stay 1 at prefill; it is
derived from available parallelism, which *does* depend on T.)

`tools/test_prefix_invariance.py` is the test that was missing. It resumes at 2,259 tokens —
deliberately not a multiple of the 2,048 chunk size, so the two runs tile the same tokens
differently — and asserts both halves: that the output is identical, **and** that the cache was
actually used (`prefix_cached_tokens == 2259`). Equality alone proves nothing, since a cache that
silently stopped working leaves both runs cold and agreeing.

Back on by default: prefill 544 → 1047 tok/s, `generation_gate.py` 6/6, cache used and
byte-identical. Note that `DSV41_PREFILL_ATTN_BLOCK_H`/`_BLOCK_N` in `.env` were tuned in
`df74991` while this path was switched off, so those values measured nothing — they are read per
call at the `_prefill_attn` call site and are worth re-tuning now that the path is live.

## A wider verify block makes utilization better and decode slower

`DSV41_BLOCK=9` (T_VERIFY=10 instead of 6) was tried because the dense weights are read once per
step however many token rows ride along — measured, at the shapes decode uses:

```
wq_b  M=6: 0.193 ms   M=16: 0.200 ms      MoE  T=6, 30 experts: 2.824 ms
w1    M=6: 0.044      M=16: 0.044              T=64, 32 experts: 3.389 ms
```

So rows look free, and GPU utilization does visibly improve with the wider block. Decode got **27%
slower**:

```
              step      accept   decode
T=6           162.7 ms   2.95    18.10 tok/s
T=10          217.4 ms   2.87    13.24 tok/s
```

Two reasons. **Rows are free for dense weights but not for routed experts** — more tokens route to
more *distinct* experts, and that read grows nearly linearly with T until it saturates:

```
T=6    21.13 distinct experts/layer    7.94 GB/step
T=10   29.85 distinct experts/layer   11.23 GB/step
```

Total bytes 14.86 → 18.1 GB. And **acceptance did not move** (2.95 → 2.87): the DSpark head is
trained at `dspark_block_size=5`, so drafts 6–9 are outside its horizon and are essentially never
accepted. Four more candidate positions, paid for, none collected.

This is the clearest case in the repo of utilization and throughput pointing opposite ways. Fuller
tiles are not the goal; tokens per byte read is.

Note also that the documented `DSV41_BLOCK` range (odd, 1–15) is wrong. The graph-capturable
routing path (`build_routing_small`) handles P ≤ 64 pairs and P = T_VERIFY × 6, so T_VERIFY ≤ 10
caps it at 9; above that the MoE falls into a `torch.unique` path that cannot be captured at all.

## Low decode utilization is mostly skinny work, but EP null routing made it worse

The GPU timeline on the live FP4/EP2 configuration is nearly full: 140.9 ms of backbone plus
14.7 ms of DSpark drafting in a 162.6 ms decode loop. A 70–90% utilization reading therefore does
not imply 10–30% host-idle time. The work consists mostly of small-row dense projections and FP4
routed-expert kernels, interleaved with one collective per layer; occupancy and arithmetic-unit
utilization are low even while the device is continuously executing.

There was still one exact EP2 loss. Every non-owned route mapped to a shared zero expert, so about
half of a 36-pair verify block collided on one arena slot. The safe implementation raised the MoE
tile from the single-device-tuned BM=16 to BM=64 for *all* experts. Passing the known null slot and
giving each null pair a temporary unique routing key lets the kernels skip those pairs and restores
BM=16 for the real experts. On the focused GB10 kernel test the call moved 1.58 → 1.03 ms; at full
model scale, with essentially matched demand (~21.8 distinct experts/layer), backbone time moved
from roughly 150 to 140.9 ms/step. The mathematical result is unchanged; the EP2 oracle test keeps
the same 0.16% FP4-kernel error against dequantized GEMM.

After that change only about 7 ms/step is outside the GPU timeline. The next single-request gains
must reduce executed bytes/work or improve DSpark acceptance. Dense TP and a wider verify block
have both been measured in the wrong direction; top-k 5 is faster but exceeds the prose loss gate,
and the BF16 LM head remains intentionally unquantized.

## The demand half-life has to be read against your traffic, not as a constant

`DSV41_PRUNE_HALFLIFE` is the EWMA half-life of the expert-demand history, in *routing slots*, and
the decay applied is `0.5 ** (grown / half)`. The number only means something next to how many
slots a request actually produces: a 4.5k-token prompt here generates **~588,000**.

`.env` carried `2e6` against a code default of `2e7`, i.e. a factor of ten more aggressive. At
`2e6` each request decays the history to `0.5 ** (588000/2e6) = 0.815` — the whole history
half-lives every ~3.4 requests. The ranking genuinely moved that much, the planner dutifully
swapped it, and the result was ~100 experts moved per request (near the `DSV41_PRUNE_SWAP_MAX=128`
cap) at a **0.9% miss rate**, costing ~1 GB of NVMe and ~0.2 s every request to fix nothing.

Restoring `2e7` (0.98 per request, stable over ~34) on the same workload:

```
                swaps/request                        routed-miss
2e6    89, 116, 96, 92, 72, 71                       2.3 → 0.9%
2e7    38, 50, 44, 43, 41, 41, 34 … 21, 18, 20, 16   0.9 → 0.4%
```

Churn fell ~5x **and** the miss rate halved. That direction is the useful part: a steadier history
is not a staler one here, it is a more accurate one — the twitchy version was chasing each prompt
in turn and never settling on the set that serves all of them.

The symptom to watch for is the swap count sitting near `DSV41_PRUNE_SWAP_MAX` while the miss rate
is already low. That combination always means the ranking is churning, never that there is real
work to do; reach for the half-life before `DSV41_PRUNE_SWAP_MIN_GAIN`, because the gain threshold
only suppresses the symptom.

## Structural corruption: test expert arithmetic, not just engine switches

**Correction to 01ed3e2 (2026-09-14):** the earlier conclusion below that the checkpoint's
MXFP4 scale grid caused nesting corruption was not established by those tests. They all kept
the custom expert arithmetic. Dequantizing the *same* packed expert weights to BF16 and using
BF16-output matmuls passed all four nesting probes (1.000 versus 0.750 for the custom kernel
in the single-box handoff). That fallback is diagnostic only: host-side `unique().tolist()`
makes it uncapturable in CUDA graphs, and dequantizing each expert per call is much slower.

The fused kernel omitted the BF16 output boundary of the gate/up linear before FP32
clamp/SwiGLU. It also omitted each down projection's BF16 output boundary; conversely the
EP2 path rounded the routed aggregate *before* adding the shared expert. These are engine
differences, not missing expert weights or a different checkpoint. Preserve FP32 accumulation,
round each linear where the reference rounds, then sum routed and shared results in FP32
before the final BF16 cast. "More precision" at a different boundary is not reference parity.

The handoff's gate/up correction made software decode (`DOT_SCALED=0`) pass 1.000, while
`DOT_SCALED=1` remained at 0.750. Do not interpret a pass at one greedy rounding boundary as a
quality guarantee. The corrected standalone test disables reduced-precision BF16 reduction,
as the engine already does: at T=512, relative error is 2.88e-4 (software) / 1.40e-4 (scaled),
not the previously reported ~4.5e-3. Both pass a 1e-3 tolerance; the scaled path having *lower*
random-tensor error did not predict its nesting score.

Fresh-process margin check, full experts (`PRUNE_KEEP=1`, no adaptive pruning, speculation
off), same canonical spaced depth-8 answer, one-token decode after prefill:

| expert arithmetic | final `}}}}` logit | competing `}}` logit | target-minus-competitor |
|---|---:|---:|---:|
| software decode + BF16 boundaries | 32.00 | 31.75 | +0.25 |
| dot_scaled + BF16 boundaries | 32.00 | 32.25 | -0.25 |
| BF16-dequant control | 33.00 | 32.75 | +0.25 |

The control itself is borderline. This verifies a rounding-sensitive decision, **not** a
comfortable-margin or general-quality fix. `MARGIN=1 SPEC=0 PRUNE_KEEP=1` with
`tools/nesting_arm.py` reproduces the check; run each arithmetic arm in a fresh process because
activation-quantization monkeypatching is process-global. The canonical target is not the only
valid tokenization; use actual generation alongside margins when evaluating a candidate.

Rejected speed-preserving attempt: resetting dot_scaled's accumulator every 128 K and adding
the partials outside MMA. Depth-8 generation passed via an earlier switch to *compact* JSON,
but the same spaced-prefix closing margin worsened to -1.0; T=512 MoE time rose from 8.8 to
10.6 ms. This is exactly why a pass alone is insufficient. Keep the original scaled kernel as
an opt-in arm; the serving profile selects software decode, without claiming that its +0.25
margin is robust. Initial layer-0/32-expert microbenchmarks measured T=512 at 15.1 ms software
versus 8.8 ms scaled; these are kernel timings, not whole-engine prefill throughput.

Second rejected speed-preserving attempt: explicit scaled **prefill** plus software **decode**
(including prefill dispatch during decoder replay). Full-expert single-node depth-8 generation
failed with an extra closing brace; the forced closing margin worsened to **-0.75**
(target 31.50, competitor 32.25). The experimental dispatch was removed, not enabled in serving.

Matched EP2 timing, both with corrected rounding and cycle breaker OFF: arena 88 GB,
keep=0.59, spec ON, MAX_SEQ=262144, adaptation/demand logging OFF. Three deterministic
cache-busted 4514-token prompts, 64-token completions per arm; all prompt hashes and generated
texts matched and every run reported zero prefix-cached tokens. Warm runs (1 and 2):

| arithmetic | prefill tok/s, run 1 / 2 | decode tok/s, run 1 / 2 |
|---|---:|---:|
| software (passing configuration) | 348.52 / 491.77 | 30.99 / 30.87 |
| scaled (comparison arm) | 431.07 / 520.37 | 31.54 / 31.77 |

Software was 19.2% / 5.5% slower in warm prefill and 1.7% / 2.8% slower in decode.
Run 0 includes cold/JIT/cache effects (software prefill 164.64, scaled 289.15 tok/s) and is
not a steady-state comparison. Warm results also vary substantially: do not claim a universal
percentage or "no speed regression." This compares two arithmetic settings of the corrected
engine, not a matched historical pre-fix build. Command: `bench/prefill_ab.py OUT --runs 3
--cache-bust --max-tokens 64`. No full quality benchmark was run at keep=1.

Final normal-profile deployment (EP2, keep=0.59, spec ON, arena 88 GB, adaptive settings
restored) passed the arithmetic floor smoke test: 10/10, 503 tokens, 26.0 decode tok/s.
`generation_gate.py --only arithmetic --max-tokens 600` checks basic operation, not general
quality; the full-expert nesting result above is the separate quality evidence.

Single-node regression checks with software decode and these rounding boundaries:
`tools/test_fp4_moe.py` passes at 1e-3 for both BF16 and FP32 routed output, with bit-identical
repeated runs and prefixes at call sizes 1, 3, 6, 7, 17, 64, 148, 256, and 300.
`engine/test_fastdecode.py` passes exact equality for logits, hidden state, and drafts at both
tested parities (keep=0.25, transient slots=16). `engine/test_spec_lossless.py --max-tokens 64`
passes both prompts: all 64 tokens identical with speculation off versus on, using
`results/trace-union/stats/coverage.json`, keep=0.25, and 16 transient slots. These are bounded
regression checks, not long-output quality certification.

Other retained fixes: HC pre-mix must round to BF16 before RMSNorm, HC post-mix adds the branch
after the residual mix, and short reference-shaped attention applies only to **prefill**.
Applying the compact attention path to decode broke FastDecoder/eager agreement (relative
logit error 0.104, top-1 agreement 0.83); the prefill gate restores the same decode shape.
The removed window-KV FP8 QDQ call had been an identity with `act_quant=False`: it was not
evidence of reference FP8-cache parity. Compressed-KV QDQ is separate and remains graph-safe.

### Historical negative tests (before the expert rounding correction)

**HTTP comparison confound found 2026-09-14:** these served nesting failures also had a
default-on `Penalties` cycle breaker. Direct `eng.generate()` probes did not construct it.
The breaker bans the correct `n` token at answer position 13 in canonical depth-8 JSON.
Thus the historical HTTP failures below do not independently establish kernel/checkpoint
corruption. With corrected software-decode kernels, EP2, full experts, and speculation ON,
changing **only** `DSV41_CYCLE_BREAK=1` to `0` raised the four-probe nesting score from
0.458 (1.000 / 0.833 / 0 / 0) to **1.000 (4/4)**. Both nodes used identical image
`6c5a316bafb4187d86a27471f5c1ca4e69a53ccd0cecae510056f601431e865d`.
This is a bounded quality check, not a guarantee for arbitrary prompts. The breaker is now
opt-in, covered by `engine/test_penalties.py`, and exposed in health and the EP boot guard.

Additional eliminated suspects: the single-node depth-8 run still passed with serving's
MAX_SEQ=262144 and PREFILL_FUSED_ATTN=0 (same +0.25 closing margin); prompt IDs and deployed
engine source hashes matched. `tools/test_fp4_ep_sum.py` with 30 real layer-0 experts measured
full versus parity-split FP32 sums at T=1/6/63/512: maximum differences 0 / 2.98e-8 / 2.38e-7 /
4.77e-7, and identical BF16 results after adding the shared term. This does not prove exact
full-model EP parity, but no extra arithmetic change was needed for the four served probes.

Symptom, found comparing this engine against MiaAI's EXL3 2.9 bpw build of the same checkpoint on a
nesting probe (`llm_benchmark/quality_quant2.py --only nesting`): asked to emit `{"n": ...}` nested
N deep, it instead produces, deterministically,

```
depth=4   {"n": {"n": {"n": {"n": 44}}}}                          ok
depth=8   {"n": {"n": {"n": {"n": {"n": {{"n": {"n": 48}}}}}}}}}   doubled brace
depth=10  {"n": {"n": {... {"n": "n": {...  "nn": 50}}...        dropped quote, merged key
```

Depth 4 is always right, 6 marginal, 8+ broken, and the corruption is character duplication and a
lost quote -- not a semantic substitution. It reproduced on every engine configuration then tried:

| hypothesis | test | result |
|---|---|---|
| checkpoint corrupt | sha256 of all 48 shards vs the Hub's LFS metadata | all match |
| wrong base revision | ours `dba1be0a` vs the quant's source `fb2764a5` | one encoding commit apart; weights unchanged |
| pruning | `PRUNE_KEEP=1.0` (full model, streaming) | still fails |
| speculative decoding | `SPEC=0` | still fails |
| `tl.dot_scaled` MoE | `DSV41_FP4_DOT_SCALED=0` | still fails |
| Decoder SWA Bounded Replay | documented bit-exact for ≤128-token prompts; this prompt is ~50 | not reachable |
| chat template | official `encode_messages(thinking_mode="chat")`; `reasoning_effort` is inert outside thinking mode | standard |
| detokenizer | canonical nesting strings through `IncrementalDetokenizer`, 1- and 6-token bursts | byte-exact |
| prefix cache | cold (evict), warm, cold again | identical corruption |

The same weights under a per-row codebook requant (EXL3 2.9 bpw, vLLM) answer all four depths
exactly. That comparison changes both quantization and runtime; it cannot isolate the scale grid
as the cause. The same-weight BF16-dequant experiment above is the more relevant control.

**Do not generalize it.** The battery's macro is decided by `nesting` + `constraint` +
`char_count`; `counter`, `json_strict`, `verbatim`, `copy_transform` and `escape_json` saturate at
1.00 for both models. Neither a four-probe nesting pass nor teacher-forced loss establishes
whole-model quality parity.

Requantization alternatives investigated at the time (not established fixes for this bug):

* **NVFP4** (E2M1 + E4M3 per 16) is the small change -- only the routed experts differ, and e4m3
  scales carry 3 mantissa bits instead of a power of two. It is **not reachable natively here yet**:
  `tl.dot_scaled`'s shape validator accepts the layout (`is_fp8e4nv()` selects block 16) but the
  sm_121a backend asserts in `TritonGPUAccelerateMatmul` (`type.getElementType().isIntOrIndex()`).
  MXFP4 (e8m0, block 32) lowers fine. So NVFP4 needs a software MoE path (decode ~free, it is
  byte-bound; prefill pays) or a newer Triton.
* **Size runs the other way.** MXFP4 is 18.80 MB/expert (4.25 bits/weight), NVFP4 19.91 MB
  (4.5 b/w, +5.9%), CB3 14.45 MB. At one arena NVFP4 costs ~4% of residency, itself a quality cost
  on a model where residency is the dominant lever. The bet is whether E4M3-per-16 beats
  UE8M0-per-32 by more than 4% fewer experts costs.
* Public "NVFP4" checkpoints are MXFP4→NVFP4 casts of this same checkpoint (`base_model:quantized`),
  so they redistribute the same error; there is no higher-precision source to download.

Keep these rejected alternatives as history, not as a reason to rule out engine arithmetic.

## The dense fp4 and fp8 head buy ~5% decode and cost structure

Prompted by the byte budget: per decode step a rank reads ~16 GB of replicated dense against ~6 GB
of split routed experts, so the dense is the *bigger per-step read*, and the fork keeps it at
checkpoint precision on purpose. Measured anyway, on the served pair (`generation_gate.py` 6/6 in
every arm; ms/step is `1000/tok_s * accept_len`, so it does not move with acceptance):

| arm | dense_fp4 | head | ms/step | nesting (3 reps) | copy_transform (3 reps) |
|---|---|---|---|---|---|
| B | off | bf16 | 180.4 | 0.458 | 1.00 |
| C | off | fp8 | **171.0** | 0.483 | **0.50** |
| A | attn,wo_a | fp8 | 170.9 | **0.278** | — |

Two results, both negative:

* **All of the ~5% is the head, and it costs exact-string fidelity.** C is 5.2% faster and
  nesting-neutral, but `copy_transform` "level radar civic" comes back "lavev radar civic" in 3/3
  reps where B is correct 3/3. The single-box measurement had the fp8 head at +0.0014 nats and
  byte-identical greedy output; under EP2 that margin is gone on a palindrome reversal, which is a
  near-tie on the last characters. The head stays bf16.
* **Attention/`wo_a` fp4 is free of speed and full of cost.** A is not faster than C
  (170.9 vs 171.0) despite removing another 2.4 GB/step, because the attention projections at M=6
  are launch/latency-bound, not byte-bound -- the same reason the dense-TP experiment moved
  nothing. And it drops nesting depth=6 from 0.83 to 0.11. It stays off.

The lesson matches the entry above: on this model the levers that move are residency and the expert
bytes, not the dense ones, and "it looks right" is not the quality these near-tie probes measure.

**Confirmed on the decode-lean build, 2026-10-01, and the speed gap is now completely gone.** Six
two-node `bench_decode_kernels_tp.py` gate runs, arms alternating, native FP4 experts,
`DSV41_TP_ATTN=1`, draft head off, 256 greedy tokens: `off` python 49.2/48.4 tok/s, `attn,wo_a`
48.8/48.6, `attn` 49.8, `wo_a` 49.3 -- every arm inside the 8-10% spread, and the profiled
dense-projection family stayed at ~30 ms/step in all six even though the fp4 weights are half the
bytes. The aggregate hides a bimodal kernel: at M=6 the fp4 kernel reaches 199-217 GB/s on wide-N
(`wq_b`, `w2`, 1.6x faster) but collapses to 33-67 GB/s on narrow-N (`wkv`, `wq_a`, `w1`) against
fp8's 120-204 GB/s on the same shapes. That is a parallelism bug -- `BLOCK_N=32` starves `wkv` to 16
CTAs for 48 SMs and a sweep of block/warps/stages does not fix it. A split-K with a fixed-order
reduction (`DSV41_FP4_DENSE_SPLIT`) is now built: `wq_a` 66 -> 143 GB/s, `w1` 113 -> 175, `wo_b`
105 -> 158, and the same end-to-end pair run gives `attn,wo_a` ~7 % fewer ms/step than `off`
(python 106 -> 98) where before the fix it was at parity. `wkv` (N=512) still only reaches parity.
The quality cost this entry records is unchanged, so the default stays `off`. Full tables:
`decode-launches.md`. (`DSV41_DENSE_FP4` could not even load under TP
attention before this date -- `FP4Weight` had no `.shard()` and `shard_attention` had no
`FP4GroupedWeight` branch; both added.)

`dense_fp4` and `head_fmt` are now in the EP2 boot config guard: they are load-time numerics
choices read from the environment, so a pair split across them would diverge silently.

## A per-request timestamp in the prompt defeats the exact-match prefix cache

Six consecutive requests from a local app on 2026-09-15, captured with the (now count-taking)
`DSV41_CAPTURE_NEXT`. The two "identical" prompts were token-for-token identical
(`captured_request_0.pt == captured_request_1.pt`, LCP 14,336), and the server reported
`prefix=14336/14336 (100%)`, prefill 20.43 s -> 0.35 s. The cache itself works.

The misses were caused by the prompt template embedding `[User current time: 2026-09-15 07:5X]`
at token index 14,331 -- five tokens from the end of the prompt. The next request on the next
minute changes that digit, so the prompts share only 14,331 of 14,336 tokens, and
`_restore_prefix` -- which requires the previous prompt to be an exact prefix of the new one --
drops all 14,331. Measured `prefix=0/14336` and 16-23 s prefills on the re-sent prompts. The
continuations within a minute (`_0 -> _1`, `_3 -> _4 -> _5`) still hit, because there the cached
prompt really is a prefix.

The cheap fix is app-side: a timestamp belongs in the history once, not re-stamped on every
request, and a growing conversation then extends the cached prompt normally. The engine-side
one is `DSV41_PREFIX_SNAPSHOTS=N`: keep the N most recent chunk-boundary prefix snapshots and,
on a full miss, resume at the longest one whose tokens are still a real prefix. With N=8 on the
pair, the timestamped re-send that used to be `prefix=0/14336` and 20.94 s (684.54 tok/s) came
back `prefix=12288/14337 (85.7%)` and 4.58 s (3131.84 tok/s): the 14,331-token LCP rounded down
to the 2,048-token snapshot granularity. It is off by default; each snapshot clones the 128-row
window and replay tail (~6 MB at window 128 / head_dim 512 / 21 window layers), and it is in the
EP2 boot config guard because the restored length decides how many prefill collectives a request
issues. Filled under `DSV41_PREFIX_SNAPSHOTS=8` in the local `.env`.

## A gate that forwards nothing still passes, and serving does not quantize activations

Two traps from 2026-09-23 ([decode projection fusion](decode-projection-fusion.md)):

* **Check the gate's reported config, not `.env`.** `run_two_node_gate.sh` builds its
  `-e DSV41_*` flags from `env`. When the filter command (`rg`) existed only as an interactive
  shell function, nothing was forwarded. Both ranks booted on code defaults (EP2,
  `DSV41_BLOCK=5`), agreed with each other, and passed. The boot guard compares ranks with each
  other, not with `.env`. The gate now uses `grep` and refuses to start when nothing is
  forwarded. Still read `tp_experts` / `draft_tokens` from the result's `config` before
  believing a number.
* **`qlinear` costs no quantization in serving.** With `act_quant=False` (the default),
  `R.act_qdq_fp8` is replaced by a bf16 cast. Removing its ~13 kernels saved 15–20 µs per call
  in a microbenchmark and exactly nothing in the engine. Count a cost only on the path serving
  actually takes.

## A 200GbE port is two ~100Gb/s logical rails, and the GID index moves across reboots

Two interconnect facts that both read as fabric faults until they are written down.

**PCIe Gen5 x4 is the ceiling per logical rail, not per 200G port.** `ib_write_bw` between
the two Sparks measures **111.85 Gb/s on either rail alone** (4 QPs) while `ethtool` reports
`Speed: 200000Mb/s` and `/sys/class/infiniband/rocep1s0f1/ports/1/rate` reads
`200 Gb/sec (2X NDR)`, `PORT_ACTIVE`. One QSFP port exposes two netdev/RoCE pairs backed by
independent PCIe Gen5 x4 links. Both CX7 functions sit at `32.0 GT/s`, width `4`, max width
`4`; x4 is the wired width of each half. PCIe 5.0 x4 is `32 GT/s x 128/130 / 8` ≈ 15.7 GB/s
theoretical, consistent with the ~14.0 GB/s observed on one rail.

The rejected conclusion was that the physical port therefore tops out at one rail. Running
both devices concurrently measured **98.02 + 98.02 = 196.04 Gb/s** (**24.5 GB/s**) aggregate.
Workload: `ib_write_bw` 6.20, 8 MiB messages, 4 QPs per HCA, local-to-peer writes;
5 seconds per isolated rail and 10 seconds with both running concurrently.
More QPs do not help one rail, but assigning IPs to both logical netdevs and using both HCAs
does. Check the two PCIe paths with:

```bash
for d in 0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1; do p=/sys/bus/pci/devices/$d; \
  echo "$d $(cat $p/current_link_width)/$(cat $p/max_link_width) @ $(cat $p/current_link_speed)"; done
```

**GID index 6 on the peer was one boot, not a property.** After rebooting 10.0.0.2, the
RoCEv2 GID for its CX7 port moved from index 6 to index 5 — now the same as 10.0.0.1. The
index is just where the kernel packed the IPv4 GID in that boot's GID table, so it is not
stable across reboots, and the two boxes do not differ by design. Always derive it
(`scripts/roce_gid.sh`); never pin `NCCL_IB_GID_INDEX` or a perftest `-x` across a reboot.
A stale index fails as `ibv_modify_qp failed with 61 ... local GID ::`, not as a slow link.
For dual rail, do not set the scalar `NCCL_IB_GID_INDEX`; constrain NCCL's per-HCA dynamic
selection with `NCCL_IB_ADDR_RANGE=10.0.0.0/23`, `NCCL_IB_ADDR_FAMILY=AF_INET`, and
`NCCL_IB_ROCE_VERSION_NUM=2`.

**NCCL result, native torch 2.13 / NCCL 2.29.7, Gate G2:** one HCA delivered 12.68 GB/s bus
bandwidth at the 42 MB prefill payload (3.31 ms); two HCAs delivered 19.00–20.54 GB/s
(2.21–2.05 ms), a **1.50–1.62x** gain. The 123 KB decode payload measured 63 us on one rail
and 62/76 us in two dual-rail runs: no established decode win, because that path is latency-
bound. Keep the second rail for prefill throughput; do not claim it accelerates decode without
an end-to-end engine measurement.

**The original G2 bus-bandwidth column was 2x too high.** All-reduce bus bandwidth is
`2*(world-1)/world * bytes/time`; for world=2 the factor is **1**, not 2. The old code
counted send+receive as bus bandwidth, reporting 25.36 and 38.00–41.07 GB/s. Its timings
and speedup ratios remain valid; the column and the numbers above are corrected.

**Prefill-only dual rail (2026-09-25):** `DSV41_PREFILL_DUAL_RAIL=1` with
`NCCL_IB_MERGE_NICS=0`, the two explicit HCAs, dynamic GIDs, and `NCCL_CROSS_NIC=0`.
`engine/collective_rails.py` initializes a separate prefill group, exports NCCL's discovered
graph, filters its channels to the first rail for the default/decode group, and warms both
before arena allocation. This is guarded for NCCL 2.29 and one visible GPU per node. It
rejects unsupported graphs or conflicting graph/network overrides. Do not substitute
environment changes between calls: NCCL's HCA list and NETDEVS_POLICY are process-cached.
The first attempted one-channel shortcut did select one rail, but small-message latency
was poor (~427 us in that probe); it was rejected. Keep multiple channels on the decode rail.

TP routing covers dense/attention gathers, expert gather/reduce/scatter, embeddings, head,
and Engram row-split reductions. Explicit prefill scopes restore on return/exception; decode
and draft graphs capture only the default group. The setting is checked via Gloo BEFORE
optional group creation on both ranks and again in V41Engine's boot-time config guard.

`tools/bench_collective_rails.py` passed actual TP embedding, all-reduce, all-gather,
reduce-scatter, and CUDA graph replay with changed inputs after prefill. Per-HCA counters
confirmed decode RX writes `[1600, 0]` and large prefill `[6400, 6400]` on both nodes.
For 100 timed 42 MB all-reduces, rank 0 measured **3.34 / 3.15 ms single rail** around the
**2.19 ms dual-rail** run (12.56–13.32 vs 19.20 GB/s); 123 KB decode measured 54 / 68 us.
The counter gate needs a barrier after BOTH ranks read counters, otherwise the faster
rank's next prefill warmup can contaminate its peer's decode sample.
Raw local measurement logs are in `results/collective-rails-20260925/`.

**Full TP engine check:** `tools/bench_prefill_rails_tp.py` alternated single / dual /
dual / single prefill on one engine load, with decode always on the default single-rail
group. Experts, dense layers, attention, embeddings, head, and draft all used the serving
TP configuration; prefix reuse and ranking adaptation were disabled. The 2,191-token
Python-function prompt generated 64 greedy tokens with speculative CUDA-graph decode.
All four runs on both ranks had bit-identical final prefill logits and generated tokens.
After the first run of each path warmed up, dual prefill took **2.212 s (990.46 tok/s)**
versus single **2.636 s (831.04 tok/s)**: **16.1% less prefill time** on this one prompt,
not a broad throughput claim. Decode measured **31.90 vs 32.27 tok/s**, respectively;
no decode speedup is claimed. The initial single/dual warmup prefills were 4.750/3.840 s
and are not used for the comparison. Logs and per-rank JSON are under
`results/prefill-rails-tp-20260925-v2/` on the respective nodes. Reproduce with an
identical image on both nodes and serving stopped (create
`results/prefill-rails-tp-check/` in both checkouts first):

```bash
GATE_IMAGE=<immutable-image-id> GATE_LOG_DIR=results/prefill-rails-tp-check \
  bash tools/run_two_node_gate.sh bench_prefill_rails_tp.py \
  --out /app/results/prefill-rails-tp-check
```

## FP8 attention exposed a typed-load failure in L2 prefetch (2026-10-04)

With `DSV41_DENSE_FP4=off` and `DSV41_L2PF_MB=2`, both ranks started and
`/health` returned 200, but the first decode graph capture failed in
`engine/l2pf.py`: `cannot cast int32[constexpr[4096]] to fp8e4nv`. The masked
prefetch load tried to cast `other=0` into the weight's FP8 dtype. Startup
health alone therefore missed the failure; the server marked the pair out of
step after the request failed.

Prefetch now views the same storage as uint8 and budgets bytes directly. It
does not decode or alter weights. The byte-load version and prefetch budget
are checked in the EP2 boot config guard. `tools/test_l2pf_cuda.py` passed
three GPU tests in 0.639 s on the serving runtime: seven storage dtypes and
masked byte tails, wrapper/budget handling, and FP8 graph capture/replay.
Both ranks must restart after the original failure.

## Owning-copy expert swap staging loses its read-only screening test (2026-10-04)

An asynchronous reader could overlap swap disk reads with installation while
preserving the agreed plan and application boundary. The first prototype retained
owning CPU copies from `ExpertStore.read_expert` in a bounded four-expert window,
with two reader workers. The production baseline passes its leased pinned bytes
directly to the installer, avoiding those extra copies. Merely making the reads
concurrent therefore adds work before there is anything to overlap.

On the head Spark, 32 real packed checkpoint experts across layers, warm working
set without a global page-cache flush, three samples per arm in alternating order:

| Simulated install delay per expert | Serial leased reader, median | Owning-copy staged reader, median | Staged change |
| --- | ---: | ---: | ---: |
| 0 ms | 77.53 ms | 104.23 ms | +34.4% time |
| 0.25 ms | 89.34 ms | 107.39 ms | +20.2% time |
| 1 ms | 113.99 ms | 110.67 ms | -2.9% time |

The install intervals are CPU sleeps, **not measured GPU copies**. This runner
loads no inference model and executes no GPU copy, collective, expert eviction or
routing update. The small apparent win at 1 ms is within the staged-arm spread
(100.88–113.66 ms); it does not justify a serving integration or throughput claim.
The initial staged expert matched the independently reread checkpoint bytes, and
the peak retained payload was exactly the configured 75,202,560-byte budget,
returning to zero after completion. Existing store I/O buffers are outside that
incremental payload budget. PyTorch was 2.13.0+cu130 with 20 CPU intra-op threads.

The prototype remains a standalone experiment in `engine/swap_staging.py`;
`V41Engine.apply_swaps` and the production ExpertStore load path are unchanged.
Ten CPU tests cover exact file-byte ownership, fixed plan order/ownership,
initial and later read failures, cancellation, stale generation/plan rejection,
borrowed record lifetime, and the memory bound across superseded plans.
A future zero-copy lease design would need to retain source buffers through
installation and prove it cannot starve other ExpertStore readers. Staging a plan
across decode steps would also require a new application boundary, which can change
routing trajectories; this experiment does not claim to preserve that behavior.

Evidence: `results/swap-staging-micro-20261004/rank0.json`. Reproduce the screening
test with the pair stopped:

```bash
.venv/bin/python tools/bench_swap_staging.py \
  --model-dir /home/ryan/models/DeepSeek-V4.1-Flash \
  --count 32 --rounds 3 --window 4 --workers 2 \
  --out results/swap-staging-micro-20261004/rank0.json
python3 tools/test_swap_staging.py -v
```

## Skipping DSpark can save a pass and lose throughput (2026-10-04)

The default-off draft-bypass prototype uses the existing two-row graph on the
root token plus a discarded dummy. Only the root logits produce an ordinary
target sample; the dummy is rolled back. This avoids DSpark but still reads
the main model's weights and emits only one token per step.

On the disposable two-Spark gate, sampled sky/sunset explanation at temperature
1.0/top_p=0.95, 96 output tokens, warmed off/on/on/off with two samples per arm:
the normal confidence-controlled DSpark path measured **22.29/22.79 tok/s**;
forced bypass measured **15.53/15.53 tok/s**. Medians were **22.54 -> 15.53 tok/s,
-31.10%**. Normal steps yielded 1.88 tokens on average, while bypass needed 95
one-token steps. The sampled texts differ because their proposal/RNG schedules
change; this is a throughput screen, not a quality comparison. It does not show
that an adaptive policy could never use bypass on a worse-acceptance workload.

The prototype passed 68 short qualification requests per rank: greedy equality
for off/forced-on/alternating/adaptive modes on two prompts, plus seeded-repeat
and rank checks across four temperatures and three top_p settings. All 20
measured combined-gate request hashes also agreed across ranks. Target head
BF16 and attention FP8 stayed fixed, with no extra attention quantization;
expert placement, prefix reuse and urgency were frozen, and `INDEX_TOPK=512`
matched serving. The context was sized for 524288, but prompts were short.

Keep `DSV41_DRAFT_BYPASS=0`. Source and tests remain for investigation; no
experiment was deployed. Retain the ordinary speculative path whenever its
extra accepted tokens repay the draft cost. A genuine width-one graph would
also need compressor work; the current ratio-2 path requires a dummy row.
Evidence: both reports in `results/decode-ideas-20261004/model-gate/` and
`results/decode-ideas-20261004/summary.json`.

## Sampled confidence depth can get stuck behind an expensive intermediate width (2026-10-04)

`results/code.json` records the HTTP code benchmark at temperature 0.6/top_p 0.95,
1024 output tokens and sparse-attention width 1024. Confidence policy version 2
used depth 1 on 1538 of 1569 measured steps (98.0%); mean accepted block lengths
were 1.90/2.13/1.86, at 24.52/24.45/23.79 tok/s. The two depth-1-only runs had
roughly 90%/86% draft-token acceptance, so the near-two block counter does not
establish a weak drafter.

The retained depth-3 estimate was 155.4 ms while depth 1 cost 73.4-75.2 ms and
depth 5 cost 107.3-113.1 ms. The sampled prefix rule first asks whether extending
1 to 3 pays, and stops without considering 5 if it does not. When depth 3 costs
more than twice depth 1, even perfect predicted acceptance cannot justify that
first extension: the maximum expected-token ratio is only 4/2. Replaying the
saved costs with near-certain confidence therefore still selects 1. These costs
persist across requests, and an unselected width receives no new measurements.

The mixed-width run included 31 depth-5 steps. Even allowing every one of its
449 depth-1 steps to accept its draft, the rounded overall mean implies at least
about 3.94 tokens per depth-5 step. The drafter is capable of longer blocks here.
The source of the anomalous depth-3 timing is not yet established. Restore the
previously qualified acceptance-based 3/5 controller for a baseline comparison;
do not dismiss this scheduling problem as sampling temperature alone. A fix that
skips an expensive intermediate width must still use only the already-included
proposal prefix, with exact sampled-distribution tests and a new boot-guard version.

## Router-score history must use the same units as request counts

The original request-unit recorder accumulated raw router scores directly into
`_want_mass`, then decayed that column at request flush without adding a normalized
request distribution. Its counts were a request EWMA; its scores were not. Feeding
that score column into placement would silently give long prompts more influence
and decay the newest evidence immediately. Do not retrofit score ranking onto an
old request database by simply reading its `mass` field.

Score-history version 2 records scores into a private request buffer, normalizes
each layer at demand fold, and then adds the observation to aged history. Graph
warmup must restore these buffers as well as counts and miss counters; compiler
warmup is not workload evidence. The fused GPU recorder is checked against the
torch recorder and CUDA graph replay. Frequency mode preserves legacy request
counts but discards the incompatible score column; score mode starts fresh.

`DSV41_PRUNE_METRIC=score` measures the raw positive router score of the unmasked
top-k picks, excluding selection bias. It does not measure expert output norms
or establish that a rarer expert is essential to a task. Selection misses and
score-weighted misses are both reported. Retention is still a quality tradeoff;
neither miss statistic substitutes for held-out answer quality.

### A router miss is not a quality loss; selective rescue needs better evidence

2026-10-05: on identical saved history, frequency and score masks differed by 257
of 9,400 resident experts. Frequency missed 13.36% of wanted slots; score missed
13.99%. All 14 held-out MMLU-Pro answer letters were identical (9/14 correct).
Do not present a lower missing rate as a quality improvement.

A bounded prefill rescue prototype used calibrated gate-weight times expected
expert-output norm, emphasizing the largest per-token share rather than frequency.
Its small 0–3-layer profile rescued nine layer/expert events with five NVMe expert
reads, but the same answers remained 9/14. It stays off. Norm magnitude is not
causal answer importance; sparse/unseen experts and late-layer sensitivity need
more evidence. Prefix persistence/caching and prefill graphs are unsupported
while it is enabled. See [critical-prefill.md](critical-prefill.md).

Confidence depth gave +15.0% code decode throughput with identical outputs in two
128-token trials, but prose lost 2.4%; do not infer a universal speedup.

The calibration helper previously failed for `engram_rows.py --layers 1` because
it indexed all configured Engram layers into a subset result dict, and imported
the HTTP requests library even with local shards. Both are corrected; the pilot
read local rows with networking disabled and fetched no model files.

## Layer position alone is not a retention score

On the 2026-10-06 frozen 14-question diagnostic, giving the first and last five
layers 330 experts each at the same 9,400-slot budget recovered a pruning-sensitive
math answer but broke a previously correct biology answer. Both uniform and this
edge allocation scored 9/14; unrestricted native routing scored 11/14. The early/
late hypothesis is worth isolating, but this measured edge allocation is not a
quality win. See RESULTS.md and `results/expert-tuning-20261005/` for the protocol.

## A zero decode step counter does not prove decode was skipped

The generation loop exits on an emitted stop token before incrementing `steps`.
An answer-letter request can therefore report `steps=0` after running a complete
verification graph. Use completion length, decode timing, graph/resolve activity,
and a multi-token generation when qualifying a new decode path.

## A passing layer subset can miss new pruning-sensitive cases

Streaming 3–7 and 15–19 recovered two selected failures (11/14 versus 9/14), but
on separate questions it was identical to resident routing (9/14). Full routing
scored 12/14 on the separate set, recovering three other cases. The subset was
therefore insufficient outside its selection cases. Do not promote a post-hoc
layer policy based on those two recoveries, or interpret a harmless individual
layer ablation as proof that several layers can be pruned together.

The same subset made a short 39-token counting request take 2.184 s versus
1.219 s resident, medians of two trials after warmup. Immediate loading fixes the
timing of expert rescue; it does not remove its host synchronization and IO cost.
All experimental streaming flags remain disabled in the normal service.

## The author's broad topic profiles are not a demonstrated quality default

2026-10-06: the pinned 39-topic database and maxmin ranker were ported exactly,
with native FP4 TP2, the same 9,400 experts / 235 per layer and 90.1 GB arena.
Fixed frequency/maxmin changed 3,085 resident IDs versus a frozen current-history
baseline, gained business and law answers (10/14 versus 8/14), but scored 9/14
versus 12/14 on separate questions. It lost engineering and health answers and
ignored the one-letter instruction on chemistry, which counts as a failed
response. The partial test and remaining-only continuation are retained; no
question was retried or given a larger token limit. Saliency/maxmin changed
3,036 IDs and tied the first baseline at 8/14, gaining business but losing biology.
Do not select a default from the first-set gain or assume saliency is superior.

Frequency/maxmin's first-set raw miss rate was **27.19%**, versus **13.89%** for
the baseline, despite its two extra correct answers. Lower miss rate and greater
output magnitude are both incomplete proxies for answer quality. The traces
measure frequency or accumulated contribution norms, not causal task importance.

`maxmin` returns synthetic admission priorities above 1 for selected experts;
rejected experts receive scores below 1. Feeding those numbers into the adaptive
router-score blender or swap planner mixes incompatible units and can destroy
the topic balance. Fixed profiles therefore reject demand blending and every
swap trigger, set the adaptive prior to unavailable in memory, and guard the
source/ranker/topics/database/actual initial expert IDs across TP2 ranks.
Defaults retain the original adaptive path. See [expert-profiles.md](expert-profiles.md).

Exploratory 128-token timing also did not show a universal gain: frequency/maxmin
code request median was 3.996 s versus 3.234 s baseline, while prose was 5.894 s
versus 6.895 s (two measured trials after warmup). Outputs differed and were
token-limited; these are throughput probes, not passing code-quality tests.
Evidence: `results/expert-profile-20261006/summary.json` and `RESULTS.md`.

## Mia's 2-bit layer allocation is a clue, not a proven pruning budget

2026-10-06: local EXL3 metadata assigns all 1,152 routed matrices per layer in
18–22 two bits, with three bits in the other 35 backbone layers. Tested one
preselected **0.8 priority factor** for those five layers at the same 9,400-slot
TP2 budget: 193 residents in 18–22 and 241 elsewhere, versus 235 everywhere.
The existing score-history seed and ranking within each layer stayed identical;
all swaps and prefix reuse were off. Multiplying layer scores under uniform
235-per-layer selection would not change any expert IDs, so this was a budget
redistribution using the existing guarded layer-count mechanism.

The candidate scored **7/14 versus 8/14** on fresh seed 20261010, losing law
1501. On five prior pruning-sensitive probes it scored **3/5 versus 2/5**,
recovering history 5003. Both totals were 10/19; every response was valid.
Engineering 12071 changed between wrong answers. The current-history baseline
already recovered math and business compared with older snapshots, so old
baseline scores cannot be reused for this comparison. Retain the uniform default.
This rejects this particular discount under the predeclared rule, not every
possible bit-informed allocation. No weights changed, no streaming was enabled,
and no model data was downloaded. Records: `results/mia-layer-priority-20261006/`.

## Extract the actual last user before chat templating

The checkpoint encoder merges tool results into user-format content blocks and
treats mid-conversation system messages as user-like for template purposes.
Searching the rendered prompt for the last User delimiter therefore includes
inputs that are not the user's typed question. Read the original request's last
`role=user` instead, before any role normalization or tool merging.

`server/latest_user.py` selects only explicit text in that one message, excludes
older/system/developer/assistant/tool/reminder/search messages and structured
attachments, and never falls back to an older question when the latest user
message is empty or attachment-only. It neither mutates model input nor logs or
persists user text. Tests cover identity, nontext blocks, native tool-result
blocks, empty inputs, Unicode/spacing and nonmutation. Plain-text documents
bundled into the same text field need a client-provided boundary; their origin
cannot be inferred reliably from message roles alone.

`DSV41_USER_PROMPT_STREAM=1` now carries verified token ranges in the shared TP2
request payload. Private markers identify the original fields in a temporary
rendering; removing them must reproduce the actual input exactly. Token offsets
exclude tokens mixed with template text. Markers never reach the model. The
conversation remains available for attention, while only those user rows use
full MoE routing during prefill and contribute to the temporary priority buffer.
The normal bounded decoder replay remains enabled: upper-layer evidence covers
only the retained sliding-window tail on a prompt longer than that window.

At prefill completion rank 0 broadcasts an unconditional admission plan on both
ranks, including empty/no-user requests. Normalized gate-score mass ranks used
experts ahead of historical demand. Admissions preserve every layer quota;
overflow is resolved by score with expert ID as a deterministic tie break.
Admitted experts are protected from all adaptive eviction paths through the
response and idle maintenance, until the next request replaces the protection.
The temporary scores are not inserted into the persistent demand database.
Decode uses the existing resident graphs and LUTs; no decode streaming is added.
Admission must also precede the state used to sample the first answer. Version
1 admitted after prefill logits had already been computed: its first token did
not benefit from the selected residents. It scored 10/14 on the small paired
set versus 11/14 for both frozen controls. Refreshing only the upper decoder
tail would leave encoder KV and the assistant cue computed under the old mask.
Version 2 therefore rebuilds the prompt once in resident mode after admission,
with priority collection and ordinary demand recording paused. This applies
the new keep set throughout the first-answer computation. It costs an extra
resident prefill and requires bounded replay; it does not stream decode or
count the request twice. `first_logits_refreshed` and `refresh_s` expose it.
Regular history/miss telemetry covers discovery and decode; the resident
rebuild does not add another vote. Mixed-row discovery reports resident
coverage, so a reported miss is not necessarily a dropped user prefill
contribution. See the separate `user_prompt` scope/row counters.

This prototype requires text-only inputs, concurrency 1, native FP4 output TP,
resident LUTs, `PRUNE_MISS=1`, no replicas/calibrated rescue/layer streaming,
and prefix/cache-response/prefill graphs disabled. The transient ring must fit
all cold experts in one layer, including a long prompt selecting every expert.
At the same 9,400 resident budget, the 5% discount allocates 225 to layers 18–22,
237 to layers 0–14 and 236 to the other layers. A 160-slot ring fits inside the
90.1 GB arena (9,423 resident-capable slots). Boot guards cover the flag and
policy version. Streaming, cold NVMe reads and resident promotions add latency;
quality and latency measurements belong to the paired trial, not the helper.

The 2026-10-06 light paired trial (seed 20261011, 14 direct-letter MMLU-Pro
questions, same immutable score-history seed and 90.1 GB arena) scored 11/14
with uniform quotas, 11/14 with the 5% discount alone (identical answers), and
12/14 with version 2 prompt admission plus discount. The final policy recovered
business q671 and psychology q2367, but lost engineering q11754. All responses
were valid. Version 1's 10/14 result and the benchmark's initial incorrect
all-layers/all-tokens scope assertion are retained alongside the corrected
window-aware check; no question was retried to replace an answer.

This is a small net gain, not proof of general quality improvement. Mean short
question wall times were 1.53 s baseline and 8.66 s final policy, including one
cold graph warmup per arm. A separate three-run counting workload (one warmup,
two measured) averaged 1.47 s baseline vs 2.89 s final; prefill averaged 0.408
vs 1.869 s, decode 0.985 vs 0.952 s. Japanese conversion, counting, seven Python
RLE cases, and exact no-admission rebuild logits at 6 and 426 tokens passed.
The feature stays an explicit pilot: it trades prompt latency for retention,
turns prefix/response caching off, and does not guarantee that prefill score
mass identifies the most important experts for every later decode token.
Protocol, all responses, rejected version, timings and launch settings are in
`results/user-prompt-priority-20261006/`.

## Fixed layer quotas constrain an otherwise shared expert arena

The store already has sectors (`arena` slots) and directories (`lru`,
`slot_key`, and the GPU LUT). Each TP2 sector stores a 9,400,320-byte half
expert. Fixed quotas came from selection and the within-layer swap planner,
not a separate physical arena per layer. Compact prefill maps also assumed
unchanging per-layer cardinalities: updating only the decode LUT cannot safely
grow a layer's resident set. Cross-layer transfers rebuild those eager prefill
maps and update the decode LUT and masks in place. Total occupancy is preserved.

`DSV41_DYNAMIC_EXPERTS=1` uses global normalized demand for startup and global
normalized latest-user score for admission. Ordinary adaptation can also trade
sectors across layers. There are no fixed layer counts or discounts; only the
router's top-k minimum of six residents per layer. The 9,400 resident budget is
fixed, not the expert identities. A 100-load request budget covers every resident
transfer, including prefill/decode/end/idle adaptation. Protection resets only
at the next request. Transient discovery reads are uncapped and counted
separately; the cap cannot promise at most 100 total weight reads.

The old full-layer transient-capacity check would have imposed a 224-resident
minimum with a 160-slot ring, undermining dynamic allocation. Streaming now
batches complete token rows so each batch's unique cold experts fit the ring.
Rank 0 broadcasts spans on both ranks unconditionally, including a single-batch
call. Token rows keep every top-k contribution and its sum order. Splitting
the expert sum instead would add rounding points. A six-token real-weight TP2
gate with six residents, eight transients and three batches reproduced an
all-resident MoE exactly on both ranks (max absolute delta 0), leaving the
resident directory untouched. This validates the tested kernel/shape; it is not
a quality guarantee for global score-based placement.

The same 14-question exploratory set scored 12/14 with uncapped fixed prompt
admission, 11/14 with fixed quotas and 100 admissions, and 10/14 with dynamic
allocation and 100 admissions. Dynamic also removes the 5% discount, so these
are whole-policy comparisons. Mean times were 8.659 / 6.338 / 6.998 s, including
cold graph warmup. The cap lost psychology q2367; dynamic additionally lost
math q8468. Every answer was valid. Dynamic counting, Japanese conversion,
seven Python RLE cases and no-admission exact rebuild checks passed. Global
normalized router mass does not measure an expert's causal importance across
layers. Preserve these negative results rather than assuming flexibility
automatically improves quality. The user explicitly requested keeping dynamic
mode enabled despite the regression; it remains a pilot, not a proven upgrade.

The isolated gate could not initialize another CUDA context alongside the live
90.1 GB arena (OOM at `torch.cuda.set_device`). It passed during the scheduled
restart with the serving pair stopped. Do not infer that the apparent free pool
can accommodate another PyTorch/CUDA process. Artifacts, including that failed
coexistence attempt, are in `results/dynamic-experts-20261006/`; fixed-cap data
are in `results/user-prompt-cap-100-20261006/`.

## A resident replacement cap does not bound full discovery

The 2026-10-06 custom-harness incident selected 9,847 latest-user tokens from
15,434 total tokens and remained in discovery after more than 113 seconds.
The role selector had not selected the entire conversation. However, a chunk
containing any selected rows entered the host streaming helper, and cold reads
for discovery were uncapped. A cap of 100 resident promotions did not cap that
work. Preserve this failure rather than interpreting the cap as an I/O budget.
Only aggregate incident metadata was saved.

Prompt streaming is now off by the user's explicit request. Dynamic sector
allocation is independent of that flag. `UserPrompt.remaining_loads` applies
the nonzero resident budget even with its streaming/priority feature disabled,
so normal prefill/decode/end/idle adaptation stays bounded. There is no discovery
or extra resident priority rebuild in this mode. Two synthetic counting checks
passed with zero discovery/resolves and 100 total resident replacements each;
10,014 tokens took 40.465 s of prefill and 41.326 s wall time. The cold 38-token
check included graph warmup, so its timing is not steady throughput. Ordinary
routed pruning misses still exist and differ from I/O miss counters. Results
and settings are in `results/dynamic-resident-20261006/`.

## Dynamic allocation does not require RAM prefix caching off

The blanket dynamic-mode restriction was inherited from latest-user streaming,
which requires fresh discovery and a post-admission rebuild. With streaming off,
RAM prefix snapshots contain encoder tensors rather than arena slot pointers.
Cross-layer compact-map replacement does not invalidate their storage. Dynamic
policy version 4 permits `DSV41_PREFIX_CACHE=1` in resident mode. Streaming still
requires it off; disk/response caching and prefill graphs remain disabled in
this prototype.

This reuses historical KV across expert selection changes, just as the original
adaptive policy does. It does not recompute old tokens under today's keep mask
or guarantee equality with a fresh prefill. A snapshot restore must not restore
old sector directories. The two-rank CPU test checks that property after a
cross-layer transfer. Live synthetic requests reused 2,289/2,289 tokens and
2,289/2,329 tokens while generation advanced on every request. Exact-repeat
prefill was 0.381 s; the initial 13.548 s prefill included graph warmup. All
returned the expected count and used at most 100 replacements, with streaming
off and total residency still 9,400. Results are in
`results/dynamic-prefix-20261006/`.

The live arena directory itself is in memory on each node, not a map file copied
between hosts. Rank 0 broadcasts startup demand and replacement plans. Both TP2
ranks build/update matching logical maps and load their half weights locally.
Only rank 0 writes the demand database; startup broadcasts its ranking and
history, so a stale or absent peer database is not a separate authority.

## The 100-load limit belongs to streaming, not ordinary adaptation

The user corrected the meaning of the requested limit: at most 100 temporary
streaming cold expert loads, not 100 resident replacements. Versions 3–5 of
the prompt policy incorrectly clamped `plan_swaps` and both swap executors with
a request-wide resident budget. Prefill could consume it completely, leaving
urgent decode adaptation with an empty plan even when its miss trigger fired.
The earlier capped-resident measurements remain historical results of that
wrong policy, not validation of the intended streaming cap.

Prompt policy version 6 removes that coupling. Generic prefill/decode/urgent/
end/idle adaptation uses its original thresholds and per-pass limits. Only
latest-user streaming consumes `DSV41_USER_PROMPT_MAX_LOADS`; it is inactive
with `USER_PROMPT_STREAM=0`. A CPU regression applies 101 real directory
transfers after exhausting the stream budget, and a controlled urgent decode
test still produces and applies a nonempty plan.

For bounded streaming, rank 0 selects cold experts by score mass among the
current user rows' original top-k. Both ranks exchange transient inventories;
only a hit present on both ranks is free. Each selected cold set fits the ring,
so it cannot be evicted and re-read within the same call. The plan and logical
cold-load count are broadcast, and the same budget is consumed on both ranks.
After exhaustion, later calls return to resident routing without store resolves.
This is an online budget, so early layers may spend it; it does not forecast
importance over later layers. The CPU model-plus-real-resolver test verifies
one actual cold load per rank at cap1 followed by resident-only routing.

Prompt streaming remains off in production. RAM prefix caching and fully
dynamic allocation remain enabled. `resident_loads_used` is telemetry only;
`resident_load_cap` is null. The streaming limit and counters have distinct
`stream_load_*` fields. Urgent loading still requires its normal rolling miss
threshold and cooldown, so lack of an urgent log by itself is not a failure.


### TTS startup beside the native TP2 expert arena

On 2026-10-06, starting Qwen3-TTS after the 88.7 GB native arena was resident
failed while initializing CUDA (`cudaMemGetInfo` out of memory), even with
roughly 12–13 GiB MemAvailable. Reclaimable host memory did not guarantee that
a fresh CUDA context could start. Starting/warming TTS first, then the TP2 pair,
worked with TTS max-seq-len 3072, arena 88.7 GB and 32 transient slots. All 9,400
resident experts were retained; a concurrent short text/audio smoke check
passed. Use that order when restarting both. Do not infer that the reported
available memory guarantees arbitrary simultaneous long requests. See
`results/tts-headroom-20261006/` for the failed startup and successful check.


### Short TTS coexistence smoke did not establish long-session headroom

The 2026-10-06 88.7 GB/9,400-resident engine plus TTS max-seq-len 3072 passed
short simultaneous requests, but rank 0 later exited via the memory watchdog:
22:32:16 UTC, MemAvailable 2.4 GB below the 2.5 GB floor for three seconds.
The active request had 20,855 prompt tokens. A prior 20,637-token turn had fully
prefilled after the one-question math probe replaced the conversational prefix
cache; this is an observed cache side effect, not proof of the allocation that
caused the watchdog exit. Rank 1 remained waiting and TTS remained live.
Do not present startup order or this small arena reduction as a validated fix
for sustained long-context coexistence. No change to the watchdog floor was
made. Failure logs: `results/memory-watchdog-20261006/`.


The follow-up reduced prefill chunks to 1024 without reducing resident experts.
A copied-history test of 20,775-token full prefill plus a cached 20,801-token
extension/319-token decode and concurrent TTS passed, with main-node
MemAvailable at least 6.948 GB (0.25 s sampling). This is bounded text-only
evidence, not a guarantee for larger or multimodal sessions. Production history
was preserved. See `results/memory-chunk-20261006/`.

### Predictive prefill needs a contextual bank, and long prompts need signed features

The aggregate expert-demand DB cannot tell which prompt caused which routing
pattern. It cannot initialize a prompt-to-demand predictor; shadow collection must
pair new prompt features with prefill observations. A lexical match is only a
heuristic and must be evaluated against actual promotions before enabling apply.

An unsigned token-hash histogram becomes close to uniform on long diverse prompts:
unrelated 200k contexts can look similar. The predictor uses signed hashing and
separate full-context/suffix/bigram blocks; a regression test covers disjoint long
vocabularies. Do not replace this with positive bag counts without rechecking that
failure. Repeated identical prompts replace their bank entries rather than adding
spurious independent neighbors.

Prediction validation must run after `self.ep` exists. The first live test failed
at boot when the TP2 requirement check was placed alongside predictor construction,
before distributed initialization. The check now sits with the dynamic-residency
validation, while all control fields still join the normal configuration guard.

## Dynamic draft length makes mean accepted block length a policy metric

`accept_len_mean` is mean leading accepted drafts plus one, before the final
output-length clamp. It is not a probability. A five-draft step can yield six
tokens; a three-draft step can yield four. `DepthPolicy` compares their measured
tokens/second, so slowing the wider step can lower the reported average simply
by making the controller verify fewer drafts. Compare temperature, generated
length, depth mix and per-depth timings before diagnosing a drafter regression.

On 2026-10-06 the old/new code headline was 3.72 versus 2.82, but temperature
changed 0.6 to 0, output length changed 512 to 256, and the five-draft share fell
60.8% to 23.1%. Fixed-five tests with the current native-attention overlay still
yielded 3.85/3.78. Removing the overlay did not recover acceptance (3.36/3.70);
do not assume abliteration explains this result. Reducing index_topk 1024 to 512
with stock weights preserved both answers and acceptance exactly in those two
short tests, while reducing mean step time 4.23%. These are workload-specific
observations, not quality claims. Protocol and full results are in
[RESULTS.md](../RESULTS.md#2026-10-06--investigating-the-apparent-speculative-acceptance-drop).

### October 6: a lower acceptance length can be the faster policy

The frozen-map dynamic-residency gate compared the same greedy tokens at
verification depths 3 and 5. Prose accepted 1.88 versus 1.96 tokens/step, but
ran at 20.11 versus 17.93 tok/s. Automatic depth stayed shallow and reached
20.27. Code's automatic 32.26 tok/s was close to fixed depth 5's 32.75.
Do not force depth 5 merely to restore an old acceptance-length headline.
`tools/bench_dynamic_residency_depth_tp.py` retains the exact-output check and
alternating-order measurement; artifacts are under
`results/dynamic-speed-investigation-20261006/`.

Current dynamic serving subsequently measured 32.01 tok/s on the same
512-token sampled code protocol as the native-attention rollback (31.52).
The original shorter greedy README protocol still measured 27.98 on that
same restored engine. Comparing it directly with the historical 37.34
FP4-attention / 512-token sampled result overstates evidence of a code
regression. Keep precision, request settings, expert map and cache state
explicit; timing variation and adaptive depth also change the mean acceptance
length. No depth-policy rewrite or allocator rollback was warranted by this
screen. The existing host Engram cache also differed by only 0.4% in its
matched ABBA comparison; disabling it was not justified as a major speed fix.


### October 7: graph warm-up streams retain cuBLAS workspaces

`FastDecoder.capture` used to create a new CUDA stream for each context/parity/
verification-width capture. On the current Torch/GB10 build, an isolated
16x5120 by 5120x384 FP32 GEMM probe allocated another **32 MiB per new stream**;
six calls on the same stream added **zero** bytes. cuBLAS's process-wide
workspace cache outlives these short-lived Python stream objects. Reuse the
single `capture_warmup_stream`; do not recreate it during graph-cache rotation.
This explains a measured source of growth, not necessarily the entire earlier
55k-context OOM. Probe: `results/tensorfold-port-20261007/workspace-probe.json`.

The verification graphs and draft graphs also share a CUDA graph pool. Evicting
individual graph keys does not release every owner of that pool. At
`DSV41_GRAPHS_MAX` (default 8, 0 = unbounded), release all verify graphs, their
memos, draft graphs and pool handle after synchronization, then recapture on
demand. `empty_cache` belongs at that infrequent boundary, never every token.
Static model/width buffers and expert LUTs remain valid. `/health` reports
allocated, reserved and peak allocated bytes, captures and pool rotations.

### October 7: bulk prefetch needs the right overlap window

A 15-us simulated communication-window microbenchmark favored a 4 MiB bulk
prefetch over the existing touch kernel (20.5 vs 25.0 us for wait plus read),
but pacing at 150 GB/s made it slower (33.6 us). This engine's short RoCE window
cannot blindly inherit TensorFold's larger prefetch budget. The initial
full-engine screen found no clear benefit from replacing the existing 2 MiB
site with bulk 2 or 4 MiB; retain the negative result and measure alternative
placement beside attention separately. A warm L2 microbenchmark alone is
insufficient evidence for deployment.

### October 7: chunk comparison must distinguish answers from forced post-EOS tokens

The 11,063-token chunk-512/chunk-256 probe returned the same complete answer
about weather, train arrivals and library hours. Its fixed 32-token benchmark
continued *after EOS*, where token streams diverged. Record both the full
stream and normal-answer comparisons; do not label this an answer-quality
failure, or claim complete token equivalence. Long-context sparse-index ties
can depend on chunk extent (see the earlier bit-exactness limitation).
128-token chunks additionally encounter the special compact-attention path at
the beginning of a prompt; the adaptive default minimum remains 256 pending
separate qualification. Memory adaptation plans once per request, using both
nodes' reports before Engram read-ahead. An indivisible image span can exceed
the chosen rows; this is surfaced in telemetry. The memory estimate is not an
allocation guarantee, and the watchdog must remain enabled.

### October 7: classify shared experts before comparing dense-kernel totals

The first TensorFold breakdown classified our FP8 shared-expert projections as
dense, but TensorFold's EXL3 expert kernel family includes shared experts. In the
four-row capture this misplaced 7.272 ms up + 2.983 ms down per window. Corrected
dense/head kernel sums are **21.752 vs 9.564 ms**, not 32.007 vs 9.564. Shared
expert kernels overlap routed kernels: neither the original nor corrected
component sums are additive phase latency. The trace-specific classifier and
480-up/480-down count assertion live in
`results/round-breakdown-20261007/reclassify_shared.py`; raw results remain intact.

The actual EXL3 metadata has 5-bit attention projections, except layer 0's
6-bit wq_a/wkv, and a 6-bit vocabulary head. A 2.9-bpw pack average is not the dense layer precision. Our
native counterparts are FP8 and BF16 respectively, so do not attribute their
entire time ratio to kernel quality, or promise a 3x lossless implementation gain.

### October 7: screen native FP8 tweaks before spending full-model test time

`results/dense-decode-opt-20261007/` retains cold-weight microbenchmarks (16
distinct weight copies, decode rows 1/4/6) and experiments. Computing activation
QDQ once helped narrow standalone projections, but hurt wide ones. Independent
128-K partial dots followed by ordered FP32 reduction did **not** reproduce the
original accumulated dot bits (maximum observed difference 2.07e-5 without QDQ);
an ordered final sum does not restore the original MMA accumulation order.

Bit-derived activation/weight scales passed every finite BF16 value and all
256 scale codes, then real layer 0/10/20/30/39 tests at 1/4/6/16 rows, with exact
FP32 outputs and row invariance. Five-stage pipelining initially appeared 14%
faster for wo_b in a schedule sweep, but the controlled production-shape test
showed only 103.4 -> 99.9 us at four rows. Fused qkv was 47.3 -> 45.6 us;
wide wq_b did not improve. The decode pipeline switch stays **off**: these small
results do not justify a full-model campaign or a large speedup claim.

### October 7: staged attention improves the core, but changes accumulation

`DSV41_ATTN_STAGED` remains opt-in: 0 retains the existing FP32 cuBLAS GEMMs,
1 uses tensor cores around the **existing full-matrix FP32 softmax**, and 2 also
stages the keys in BF16 instead of widening them into FP32 scratch. These keys
already contain BF16 values; bitwise comparison after widening found no changed
key bits, including signed zeros. Projection weights remain native FP8. PV uses
TF32x3 with FP32 probabilities and output, not a single BF16 probability product.

This is **not bit-exact attention**. BF16-typed keys also change the compiler's
TF32x3 lowering slightly (about 6.5e-8 relative output difference versus FP32
key storage in the screen). Three exact structural answers survived both full
model A/Bs, but prose/code token streams changed. Do not call three probes a
general quality gate, or attribute changed speculative acceptance entirely to
kernel speed. Mode 2's A/B/A measured 23.76 versus 21.145 tok/s prose and 30.80
versus 29.56 code. The depth policy's final smoothed step estimates fell about
2-3%, but total decode wall time divided by steps was 91.33 vs 92.17 ms for
prose and 93.97 vs 93.64 ms for code: there is no demonstrated large round-time
gain. Both ranks agreed, the expert map was frozen, and repeat baseline token
hashes matched.

Rejected variants are retained in `tools/decode_attn_staged.py` and the micro
drivers. Direct packed-cache tile loads were **236 us versus 47.7 us** for
BF16 staging at T=4, H=32, N=1152: repeated tile-local unpacking outweighed the
saved intermediate buffer. Splitting PV four ways did not improve long-key
latency. Three BF16 probability components were no faster and increased FP64
relative error to 3.72e-6 at 3200 keys, versus 6.96e-7 for TF32x3. None is
used by the engine. Artifacts: `results/attention-core-opt-20261007/`.

## Batch-one attention and odd verification (2026-10-07)

Adding odd compressor groups is not enough to make width-one verification
match a wider target block. Native `torch.einsum` QK/PV at batch one selects
different FP32 accumulation from batched calls. In the 128/1,152/3,200-key
screen, this changed 2/9/4 final BF16 elements; broadcasting both inputs to
batch two, then keeping row zero, restored bit-identical scores and values.
Widths two through five already matched width six in that screen. See
`tools/bench_bmm_rows.py` and `results/decode-followup-20261007/bmm-rows.json`.
The initial model check caught a 0.875 maximum logit difference at width one
despite exact CPU compressor/rollback tests; it was rejected before generation.

Odd widths must update host pending state from **end parity `(S + T) % 2`**,
not just the starting parity. The old condition is equivalent only for even
widths. Preserve a pending tuple before cold graph capture: it can alias the
static compressor buffers that warm-up overwrites.

The corrected TP2 test matched full logits/hidden prefixes at all widths 1–6
and both parities after a 4,114/4,115-token prefix. This qualification used
native attention/router and a frozen learned expert map. The short exact
generation comparison found code depth four 9% faster than five, but native
depth three was already as fast in a separate control. Do not turn that into
a 9% gain over the default. Prose depth two beat three by only 1.6%.

The default eight-graph cap cannot hold both start parities across all six
widths in one context bucket. Crossing it rotates the whole pool, including
draft graphs. Use a sufficient cap for qualification and track captures/pool
resets; do not interpret recurring cold captures as steady-state kernel cost.

## Ordered-chain pair batching (2026-10-07)

`FP4_V2_PAIR_BATCH=2` stages two tokens' partials and runs their identical
ordered chains together. Eighty-four intermediate, final and mixed-null bit
checks passed. Cold real TP-output weights improved favorable low-U cells
4–11%, but rows4/U14 slowed 609.3 -> 620.0 us and rows6/U20 slowed
866.4 -> 884.3 us. Shared memory rises 20,736 -> 31,232 bytes/CTA for up
and 10,368 -> 15,616 for down. Keep the compile default at one; even repeated
favorable cells imply only a few milliseconds across 40 layers. The retained
source and `tools/bench_fp4_pairbatch.py` prevent repeating this campaign.

## BF16-source router: component win is not output equivalence (2026-10-07)

All 40 main gates originated as BF16. A tensor-core BF16-source/FP32-accumulate
router cut cold projection latency from 60–61 to 19–20 us, and 9,600 synthetic
routing cases preserved selected experts/order. Its split accumulation still
changes logits: both prose/code output hashes changed in the full-model test.
Three exact structural answers survived, which is only a quality floor.
Following-baseline prose wall time per step improved 94.78 -> 91.83 ms;
code worsened 93.58 -> 95.35 ms. `DSV41_ROUTER_BF16` stays off, adds 150 MiB
per rank when enabled, and rejects genuinely FP32 gates that would narrow.
Artifacts: `results/decode-followup-20261007/tp-v2/router-rank*.json`.

## Lossless head storage still needs an arithmetic gate (2026-10-07)

The native BF16 head compresses losslessly by keeping sign/mantissa bytes,
four-bit exponent deltas per 128 values and rare full-exponent escape groups.
The TP2 shard shrank 661,913,600 -> 508,819,072 bytes; every original bit was
checked. A straightforward byte-derived Triton dot still changed 72/77 of
258,560 logits in two actual-weight cases, despite identical decoded weights.
The compiler chose **kWidth=4**, while native BF16 loads selected **kWidth=2**,
changing the K-fragment summation grouping. Tiny rounded random tests missed it.

The repeated K16 cancellation pattern `[2**25, 1, -2**25, 1, 0, ...]` gives
**640** with native geometry and guarded cuBLAS, but **0** with the inferred
byte-dot geometry. Explicit Gluon `DotOperandLayout(..., k_width=2)` preserves
the native raw FP32 result in this regression. The test checks raw accumulators,
odd widths, vocabulary tails and captured replay, as well as final BF16 logits.

Row-major packed storage measured **4.624 ms** versus native **2.882 ms** at M4;
tiled storage with inferred byte-dot reached **2.956 ms** but still changed
logits. Explicit-layout tiled Gluon BN32 measured **2.25–2.28 ms** at widths1–6
against production **2.94–3.08 ms**, exact over 5.43 million checked logits.
BN64 was slower, so no larger tuning campaign was run. Reconstruction accuracy
does not establish universal cuBLAS accumulation equivalence. Keep the opt-in
separate from the default and preserve the negative controls.

`DSV41_HEAD_KERNEL=packed` requires native BF16 head storage and draft-head
reuse (`DSV41_DRAFT_HEAD_FMT=off` or `bf16`). Packing runs before capture and
retains no original GPU tensor in serving. The benchmark adapters intentionally
retain original control weights and must never be used for serving. Larger
prefill calls unpack at most 16,384 vocabulary rows at a time; initial JIT and
this extra traffic can increase prefill latency. The feature changes neither
expert allocation nor the unconditional TP vocabulary gather. Both ranks must
agree on the kernel and version in the boot guard.

Artifacts: `results/head-native-20261008/`, `results/head-packed-asm-20261007/`
and `results/head-packed-integration-20261008/`; `tools/test_native_head.py`
keeps the cancellation failure and integration regressions executable.

## Native expert rewrites need a byte floor and the exact reduction tree (2026-10-07)

The corrected four-row profile's 44.55 ms expert kernel sum contains 34.30 ms
of native routed FP4 kernels and 10.26 ms of overlapping FP8 shared-expert
work. TensorFold's 22.19 ms includes both kinds too. The ratio is not an
independently recoverable native-kernel speedup: their weight formats, routes
and trajectories differ, and the component sums include overlapping streams.

The cold actual-weight TP-output screen in
`results/fp4-pairbatch-20261007/screen.json` uses four independent copies of
32 layer-0 experts, rotating disjoint expert sets in captured calls. Each
native expert occupies **9,400,320 bytes per rank**:

```
up:   2 * (1152 * 5120 / 2 + 1152 * 5120 / 32) = 6,266,880 bytes
down:      2560 * 2304 / 2 + 2560 * 2304 / 32  = 3,133,440 bytes
effective GB/s = distinct_real_experts * 9,400,320 / total_microseconds / 1000
```

This is minimum weight payload per call divided by measured kernel time,
including scale bytes. It is not a DRAM-counter measurement and excludes
activation/output traffic. Baseline cells, before rejected pair batching:

| Tokens / distinct experts | Up + down, us | Minimum payload rate, GB/s |
| --- | ---: | ---: |
| 4 / 6 | 437.3 | 129.0 |
| 4 / 14 | 609.3 | 216.0 |
| 4 / 20 | 852.6 | 220.5 |
| 6 / 6 | 573.7 | 98.3 |
| 6 / 20 | 866.4 | 217.0 |
| 6 / 26 | 1,207.2 | 202.5 |

The v2 kernel already loads a complete packed weight row into registers once,
before processing that expert's routed pairs. A tensor-core rewrite cannot
save repeated DRAM reads that this implementation has already removed.
Low-U repeated-expert cells still have a useful compute gap. Using the new
head's roughly 223 GB/s payload rate as a reference, six experts take 252.9 us
just to read their native weight payload: the 4/6 and 6/6 component ceilings
would be 1.73x and 2.27x. That reference is not a hardware guarantee; high-U
cells already approach it, and the low-U fraction in a real request matters.

Lossless FP4 entropy coding also has much less room than BF16-head packing.
A CPU-only histogram of expert zero at layers 0/20/39, all w1/w2/w3, measured
**3.884–3.896 bits per nibble** including signed zero. Ideal zero-order coding
would save only 2.6–2.9% of code bytes before headers/decompression; scales
are additional. This does not bound predictive or conditional compression.
Reproduce with `tools/inspect_fp4_entropy.py`; counts and source hash are in
`results/fp4-pairbatch-20261007/symbol-entropy.json`.

Ordinary FP16/BF16 tensor-core GEMM changes the deployed reduction tree even
with unchanged weights. `group_partial` forms two independent two-product
FP32 FMA chains per four-K virtual lane, adds even/odd chains, then performs
an explicit eight-leaf tree. Four separate scaled chains accumulate groups
`i, i+4, ...` and are combined in order. Pre-scaling weights or carrying an
MMA accumulator through all K changes those boundaries. Existing
`DOT_SCALED` results do not price the cold TP-output low-U cells and retain
structural-quality negatives; they are not a qualified exact replacement.

An unqualified exact-TC prototype can instead give each real member eight
sparse virtual MMA rows, compute its even and odd two-product dots separately,
then reproduce the old tree/chains explicitly. It keeps the packed weights
and avoids an expanded GPU weight cache, but increases padded TC arithmetic
substantially. First prove raw per-group FP32 equality against the CUDA
reference, including BF16-to-FP16 conversion, subnormals and cancellation.
Only a passing proof justifies a small cold low-U screen plus a high-U control;
no full-model campaign or production option is justified by this audit alone.

## Sparse native FP4 MMA: exact products work, but this layout is slower (2026-10-07)

The benchmark-only `tools/fp4_group_tc.py` tested the proposed eight virtual
rows per real member against the extracted serving `group_partial`, including
its BF16-to-FP16 activation conversion. The even/odd two-product MMA spelling
failed **854/7,488** mixed signed-logrange raw FP32 outputs, although the
18,304 uniform-code and 4,096 byte-pattern boundary checks passed. PTX retained
the explicit `add.rn` tree: preserving the tree alone does not preserve the
rounding inside an FP16 MMA. An isolated example with `x0=1`,
`x2=1.5 * 2^-23`, `w0=w2=1`, all other activations zero, gives CUDA
`1 + 2^-22` versus MMA `1 + 2^-23`.

Four separate one-product MMAs, followed by RN FP32 `p0+p2`, `p1+p3`, their
sum and the original eight-leaf tree, passed **35,776 finite raw FP32 checks**.
These cover every BF16 code that stays finite after FP16 conversion, all 16
FP4 codes, all 256 packed-byte patterns, subnormals, signed zero, cancellation,
mixed exponent ranges and the isolated alignment counterexample. A separate
32-output overflow/NaN report also matched on this device; it does not extend
the finite exactness claim to arbitrary NaN payloads or future hardware.

The exact spelling still loses the isolated helper screen. With 256 groups,
N1152, one 32-K group, captured 32 calls and six balanced quartets on GB10:

| Members | Extracted CUDA SIMT, us | Four-product sparse MMA, us | MMA / SIMT |
| ---: | ---: | ---: | ---: |
| 1 | 8.063 | 109.108 | 13.53x |
| 2 | 15.220 | 108.539 | 7.13x |

This is a hot one-group arithmetic/layout screen, **not cold actual-weight
MoE timing**. It excludes scale decoding, four ordered full-K chains, SiLU,
routing and shared experts. Padding four MMAs plus conversion/gather work
overwhelms reuse here; no full-K extension or model campaign followed. A
different relayout could improve it, so this rejects the measured spelling,
not the possibility of an exact tensor-core algorithm.

Reproduce the positive gate and helper cost with
`tools/test_fp4_group_tc.py --single-products --timing --out <report.json>`.
Omit `--single-products` to retain the numerical failure. Artifacts and
qualified source/PTX snapshots: `results/fp4-group-tc-20261007/`. Production
expert kernels, configuration and defaults were not changed by this proof.
## Runtime decode snapshots need storage independent of captured graph pools

The initial runtime probe retained each private input/output Tensor but allocated
its storage inside the shared capture pool. The 33-token/96-completion full-engine
TP2 canary produced only 432/488 exact local replays on **both** ranks: the same
40 verify sites and 16 greedy-draft sites failed. The small shared-pool synthetic
fixture passed both versions and did not reproduce the full-engine failure.
An external 64 MiB snapshot slab, with disjoint slices reused until graph eviction,
made all 488/488 comparisons exact on both ranks. Retaining a Python Tensor alone
is insufficient qualification for snapshots across this engine's graph variants.
The guard withheld all rejected latency comparisons; do not bypass it.

Reuse the existing capture warm-up stream for isolation. cuBLAS retains workspace
per stream on this build; creating one stream per case can consume gigabytes while
profiling hundreds of small operations. A standalone CUDA test has roughly 100 MiB
of tracked allocation including library workspaces, despite less than 4 MiB of
explicit test/snapshot/flush storage. Starting another CUDA context alongside the
loaded engine can fail; the runtime probe uses the engine's existing context.

## HTTP backpressure can starve the TP2 keepalive

On October 7, generation finished at 01:11:23 UTC, but the final SSE write did
not return until 01:26:25 (`OSError: No route to host`). The request still held
the engine lock, so the 30-second heartbeat could not acquire it. Rank 1's
600-second idle Gloo receive expired at 01:21:23; the subsequent dashboard
command exposed the broken pair rather than causing it.

Bound HTTP socket I/O to 30 seconds, including the final statistics chunk, and
explicitly close an active streaming generator when its consumer fails. This
limits time blocked in a write; it does not cap generation or prefill runtime.
Real socket backpressure tests cover active chat/completion streams, final
statistics and non-stream responses without loading the model.

A heartbeat failure must latch the pair fault before releasing the request
lock. `/health` returns 503 for a faulted pair. A failed headless broadcast must
exit nonzero immediately: graceful distributed teardown after a failed Gloo
group can hang and leave Docker reporting a container as running. Explicit
shutdown still performs normal cleanup. The dashboard retains its saved report
but disables diagnostic actions until health recovers.

## Decode probe intervals are not intrinsic kernel durations

The initial saved 33-token/48-output TP2 diagnostic reads one last CUDA-event
interval per capture key and phase, not a mean over `replay_count`. At L17,
key `[1,4096,4]`, the output projection wrapper measured 0.180/1.551 ms on
rank 0/1. Its local FP8 leaf was 0.127/0.119 ms; the TP communication interval
was 0.013/1.388 ms. A projection wrapper can include a peer rendezvous, child
snapshot/event work and concurrent GPU resource contention. Even a local leaf's
live event interval can include device scheduling delays. Exact private replay
outputs qualify correctness, not the fidelity of instrumented live durations.

The quick screen used only two calls and two repeats, with one eager warmup.
Among 478 matched FP8/grouped leaves, the first warm sample exceeded the second
at 472 sites on rank 0 and 477 on rank 1. Median first/second ratios were
1.85/1.81. With two samples, the reported median averages that startup-biased
sample with the later one: L15 `wo_a` rank 0 was 0.538/0.036 ms, reported as
0.287 ms. The warm graph is not replay-primed before each measured warm replay.
Do not use that saved screen to claim pure kernel cost or available speedup.
Before making such claims, qualify instrumentation with a small serial/fork-join
calibration and collect warmed repeated samples with their spread. Its 16 MiB
flush is controlled cache pressure, not a guarantee of fully cold DRAM weights.

## A desktop session on one node stalls the pair (2026-10-07)

Symptom: the decode probe compares nodes and one random cheap kernel (`rope`,
`lean.rmsnorm`, `residual`, `attention.softmax`, a 0.1 ms projection) takes
0.5-1.5 ms on node 0. The peer then shows the same amount in its next
collective. Example: L23 `projection#2` measured 1.256 ms on node 0 vs 0.126 ms on node 1, and
node 1 waited 1.130 ms in the following `wo_b` all-reduce. Across the three
saved reports in `results/decode-probe-20261008/`, 29 of 32 leaf-op gaps
larger than 0.3 ms were on node 0, and the affected op changed between runs.

The probe's events are recorded inside the CUDA graph, so this is GPU-side.
Host launch gaps and the GIL cannot appear between two in-graph events. Node 0
runs a logged-in GNOME desktop on the same GB10: Xorg, gnome-shell, Chromium
(which was rendering the decode dashboard itself) and an Electron app's GPU
process. Node 1 has no display connected, and its only graphical clients are
the GDM greeter's Xorg and gnome-shell (18 + 6 MiB). Those clients cost nothing
measurable without a display. Graphics work is time-sliced against the engine's
compute context.

`tools/gpu_timeslice_gaps.cu`, engine idle, 5 s windows:

| node | gaps > 50 us | gaps > 0.5 ms | max | GPU time lost |
|---|---|---|---|---|
| 0, desktop busy | 52-59/s | 73-74 per 5 s | 1.29 ms | 2.4-2.6% |
| 0, desktop quieter | 8-12/s | 5 per 3-5 s | 1.21-1.49 ms | 0.24-0.48% |
| 1, no display, greeter only | 0-0.2/s | 0 | 0.19 ms | 0.00% |

The lost share depends on what is on screen. The ~1.0-1.5 ms gap length did
not change. Under TP2 lockstep every stall on one node is paid by both. No
engine-side setting addresses this: stream priority does not cross contexts,
and MPS does not cover graphics. Run the serving nodes with no logged-in
graphical session and view the dashboards from another machine. Rerun the gap
probe on both nodes before attributing a cross-node imbalance to the engine.

## One prompt cannot reject an arithmetic change on acceptance (2026-10-08)

Any rounding change reshuffles greedy near-ties. The text diverges after a few tokens, and that
prompt's accepted length moves by up to ±0.3 in either direction. `DSV41_HC_KERNEL` was rejected
for a 0.07–0.10 drop on one prompt, and `DSV41_ATTN_STAGED` / `DSV41_ROUTER_BF16` were screened the
same way. Paired over 22 prompts, the three switches together measured −7.30 ± 0.50 ms/step,
acceptance +0.003 ± 0.026 and +9.0% tok/s (RESULTS.md 2026-10-08). Measure step time per operation
(token-independent) and acceptance as a paired mean over many prompts with its standard error:
`tools/bench_accept_ab_tp.py`.

## Cold FP8 projections are latency-bound, and L2 prefetch did not recover it (2026-10-08)

On GB10 (24 MiB L2), the FP8 decode projections run far below DRAM bandwidth when their weights
are cold. With four rows, after a 256 MiB flush: wq_b 166 µs cold vs 46 µs from L2, wq_a+wkv
87 vs 35, wo_b 171 vs 91, shared w13 103 vs 40. The routed expert kernels reach ~220 GB/s.
`DSV41_L2PF_QKV_MB` (this layer's wq_a+wkv then wq_b, issued while the HC mixes run) and
`DSV41_L2PF_SH_MB` (the shared expert, issued while the FFN mixes and router run) test whether
TensorFold-style prefetch recovers it. Outputs were identical in every configuration
(`results/l2pf-trace-20261008/`, `results/l2pf-bulk-20261008/`):

* `touch` (`evict_last`) with consumer waits made it worse: 66.8 → 69.4–73.1 ms per verify
  forward. Dense time fell by up to 3.1 ms, but projections waited up to 5.5 ms for the touch
  kernels. The sticky lines also slowed the MoE and the draft (+3.4 / +2.2 ms at 12 MiB).
* `touchn` (normal eviction, waited): 70.6 ms.
* `bulk` hints, not waited: dense time fell 1.1–1.9 ms, but the forward stayed at 66.8–67.9 ms
  against 66.4 / 68.3 ms for two unprefetched runs. The ~60 µs windows cannot land 12–24 MiB, and
  the rest competes with the projections and MoE for DRAM.

Both switches stay at 0. The headroom is in the cold FP8 kernel itself (memory-level
parallelism), not in prefetch scheduling.

## Markov candidate shortlists cost acceptance on this drafter (2026-10-08)

Greedy DSpark with the Markov bias applied only to a candidate shortlist, as TensorFold and
vLLM do. Draft-only changes leave the target output untouched; all 24 outputs were identical
in every arm, so these paired acceptance numbers are exact:

* global top-128 (`DSV41_DRAFT_MARKOV_TOPK=128`): −0.10 ± 0.03 accepted, −0.3 ms/step, −1.2 tok/s.
* per-rank top-256 with only candidates gathered (`DSV41_DRAFT_MARKOV_LOCAL=1`, K=512):
  −1.79 ± 0.43 ms/step and −0.044 ± 0.014 accepted, net +0.2 ± 0.3 tok/s.

`DSV41_DRAFT_MARKOV_TP=1` keeps the full chain and splits it by vocabulary half instead: each rank
biases its own 64,640 rows and exchanges one (max, id) pair per position. Proposals match the
full-vocabulary chain (the CPU check is in RESULTS.md 2026-10-08).

Measurement note: re-capturing graphs after a runtime configuration switch, with a 24-token
warmup on another prompt, left the first measured run ~1.9 ms/step slower. Alternating arm order
cancels this in the mean but inflates the error. `tools/bench_accept_ab_tp.py` now warms up on
the prompt about to be measured.

## A wider K tile is not split-K; BF16 input is not FP32 input (2026-10-08)

`_fp8_linear_kernel` passes its accumulator into every `tl.dot`, so the MMA k-steps run in the
same order whether the K tile is 128 or 256. Every tile, warp, stage and K-tile combination swept
on the decode shapes was bit-identical, and `BLOCK_N=32`/`BLOCK_K=256` saved 8–15 µs per
projection on cold weights. Ordered split-K across programs is different: it changes the bits
(see October 7 above).

The reverse surprise: feeding the split-K HC projection BF16 activations instead of their exact
FP32 upcast changed its output at ~1e-7. The values are identical, but the compiled MMA lowering
is not. Treat a dtype change at a `tl.dot` operand as a numerics change and measure it.

## Page-faulted Engram reads stall the GPU through the kernel's memory management (2026-10-08)

Cold prefill on random text was 358–545 tok/s at 8K, and the same prompt again ran 19.6 → 11.4 s.
Only the MoE phase changed (12.9 → 5.3 s of GPU time). CUPTI traces showed why: single
`_moe_up_kernel` calls of 20–176 ms against a typical 3–10 ms, the stall *inside* the kernel, on
one rank at a time. The other rank then sits in the TP all-gather. Paired per call, the
collectives' real transfer was 0.29 s of an 8K prefill and the waiting 8.9 s.

The trigger is the host, not the GPU work. Each node runs with ~1 GB free, and the memmap row
path page-faults two 4 KB pages per 264-byte row: ~1.5M faults for a 42K prompt. That forced
reclaim and compaction: 700–1,500 compaction stalls and 0.35–0.83M migrated pages per prompt.
Unrelated churn (`cat` of other model files) slowed a hot-row prefill 2.7× by itself, so it is
the migration/compaction, not the Engram code. The arena is ordinary `cudaMalloc` memory, which
on GB10 is still system RAM behind the SMMU. Kernels with the largest footprint (the experts)
stall.

Fix: `DSV41_ENGRAM_DIRECT=1` — `O_DIRECT` reads of the covering 512-byte sectors in the native
workers, for every gather size. Nothing enters the page cache and nothing is mapped. The bytes
are identical (30K rows per table, edges included), and a cold 24K-row gather is 0.13–0.16 s
against 0.38–0.84 s. Paired gate, same seeded prompts: cold 8K 358–545 → 782–933 tok/s, 32K
530 → 925, first tokens equal. TensorFold arrived at the same design ("never pinned or
page-cached").

Direct reads for *decode-sized* gathers were measured too, and lost: 22 paired prompts gave
+0.89 ± 0.43 ms/step with identical outputs. A 150-row gather pays the `O_DIRECT` round trip
where the memmap often hits the page cache. `DSV41_ENGRAM_DIRECT=prefill` (the serving setting)
reads directly only above 384 rows; decode keeps the memmap and the native gather.

Not every stall is gone. A repeated prompt under direct reads still had a 7.3 s MoE phase, so
other memory activity on the box can do the same. Bigger prefill chunks only pay once the stalls
are gone: with them, 512 vs 2,048 rows was within prompt-to-prompt noise; without them, the
2,048-row expert kernels take 2.1 s of an 8K prompt against ~4 s at 512.

## Prefill kernels: what was exact and what was not (2026-10-08)

Three prefill rewrites, each behind a switch in the boot guard:

- **`DSV41_PREFILL_ATTN_INDEXED`** (exact). Attention reads window rows from the ring and
  compressed rows from the packed cache by index, instead of from gathered [T, 128, d] and
  [T, 512, d] copies (the second one is 1 GB per layer at 2,048 rows). Dequantization uses the
  packed gather's formula, and tiles keep `decode_attention`'s boundaries. Bit-identical in the
  unit test and on 5 prompts' prefill logits and 64-token decodes. Every block shape swept also
  gave the same bits; 32 heads per program is fastest (7.35 vs 15.3 ms).
- **`DSV41_HC_PREFILL_FUSED`** (numerics). One pass for the HC mix front, instead of
  `x.float()` + cuBLAS FP32 SIMT GEMM (N = 24) + square + mean: 0.57 vs 5.45 ms. Triton's
  `input_precision="ieee"` was **not** IEEE-accurate here: 5e-6 relative against FP64, against
  torch's 6e-7, and slower. `tf32x3` gives 6.6e-7–1.1e-6.
- **`DSV41_INDEX_FUSED`** (ranking numerics). One kernel for the indexer's score matrix: 24×
  faster at 4K keys, 99.99% identical scores. The heads are summed in a fixed order.

How big "numerics" is: a 1e-6 change in the HC mixes moves 8K-prompt logits by up to ~3, KL up
to 0.28. Most of that is chaotic amplification through MoE routing near-ties, not precision.
The engine already does this to itself. Changing only the chunk size (1,024 vs 2,048 rows, same
code), as adaptive prefill sizing does under memory pressure, gave mean KL 0.026 (max 0.074).
The three switches together gave mean KL 0.096. Top-1 changed on 1 of 8 prompts in each, and
64-token continuations diverged on 4 of 6 natural prompts in each. Judge such a change against
that floor, and not by whether one prompt's tokens match.

## A data-dependent tl.constexpr is a compile per value, inside the request (2026-10-08)

The FP4 MoE routing kernels took the pair count (`_route_counts` `P`), the layer's resident
count (`_route_kernel` `NS`) and the chunk's token count (`_moe_down_kernel` `NTOK`) as
`tl.constexpr`; so did the packed-KV `_gather` (`N`). Every new value is a fresh compile. The
shared cache held 5,382 / 1,905 / 3,123 / 1,416 variants of them. In serving, the last chunk of
almost every prompt has a new length, and every expert swap changes some layer's resident count,
so requests kept compiling with the GPU idle. A serving-like gate traced 3.7 s of a 10 s 8K
prefill in five 0.5–0.9 s gaps before `_route_kernel`. The frozen-map gate never saw it: no swaps,
and its seeded prompts had compiled their shapes on an earlier run.

They are runtime arguments now, also with `do_not_specialize`, since Triton otherwise compiles per
divisible-by-16 / == 1 pattern. The outputs are bit-identical (unit check: both routing paths,
T = 70–2,048, gather up to 1M rows). The remaining constexprs are powers of two: 60 routing
variants, compiled at boot by `fp4_moe.warm_routing` (10 s cold, 1.1 s from the disk cache).

Rule: a kernel argument that follows request shape (tokens, pairs, resident experts) must not be
`tl.constexpr`. Count variants per kernel name in `.triton-cache` to find the next one.

Also measured here: 4,096-row prefill chunks were slower than 2,048 (8K 6.7 vs 5.7–5.9 s, 32K
24.6 vs 23.1 s), and their first two prompts took ~50 s while the allocator grew.

## Prefill-only scaled FP4 saves part of the expert time, not the whole component (2026-10-08)

`DSV41_FP4_PREFILL_DOT_SCALED=1` is an opt-in prompt/replay arm independent of
`DSV41_FP4_DOT_SCALED`. It can coexist with `DSV41_FP4_CUDA=1`: decode and drafting use
exactly their old kernels. Pass `prefill=True` explicitly through the model/arena dispatcher,
including bounded decoder replay and streamed expert batches. Do not infer the phase from
batch size: a tiny prefill and a verification block can have the same row count. The setting
must agree in the boot guard; prefill replicas remain restricted to software arithmetic.

On the October 8 profile, a frozen-map TP2 screen passed 36/36 objective checks in both arms,
including the earlier nesting depths, executable Python, strict tool calls and retrieval up
to 32K. The real-weight standalone scaled error was <=9.80e-5 relative L2 to BF16-dequant
matmuls. These results do not undo the September full-expert nesting failure or establish
universal quality equivalence. See the complete protocol in RESULTS.md.

Four approximately 8K random prompts measured **6.164 → 5.579 s** mean total prefill;
up/down GPU event envelopes were **2.400 → 1.642 s**. The measured saving is 0.585 s,
not the requested/estimated 2 seconds. One pair was 0.079 s slower. A component that takes
roughly two seconds cannot save that entire amount unless its cost is eliminated; even the
illustrative 2x expert speedup was not reached (measured 1.46x). Do not confuse component
cost, total latency, and time saved, or transfer the 1.865-second **32K** saving to **8K**.

A subsequent 66-cell tile screen preserved scaled outputs but only showed modest isolated
component wins at TP shapes, with different schedules winning at 128 and 2,048 rows. Its
synthetic routes and microbenchmark times do not establish extra end-to-end savings. Those
tiles remain unpromoted; the default-off arithmetic arm and all measurements are retained.
Serving was restored to the original `prefill-v4` image/configuration.

**Promoted after review (same day).** Two things were missing for a decision, and both are in
now. First, a logit-level comparison against the engine's own noise. Over 9 paired prompts (3
random 8K, 6 natural), scaled vs software prefill gave mean KL 0.119 (max 0.37) with top-1
unchanged 9/9. Changing only the chunk size (1,024 vs 2,048 rows) gave 0.042 (max 0.22), also
9/9. That is the same ~3x ratio accepted for the HC/indexer kernels. Second, the tile winners in
an engine run: `DSV41_FP4_PREFILL_SCALED_TILES` (BM32; up BN128/4 warps/3 stages, down
BN128/4/4) reproduced the scaled arm's tokens on 9/9 prompts. Paired against the software arm:
scaled -0.448 ± 0.219 s, scaled + tiles -0.792 ± 0.143 s per 8K prefill. Decode, 22 paired
prompts, old vs new prefill arithmetic: ms/step +0.27 ± 0.49, acceptance -0.030 ± 0.027.
Serving runs both switches since `prefill-v5`.

Harness note: construct `V41Engine` outside an outer `torch.inference_mode()` context.
Expert arenas are populated by loader threads, and inference tensors allocated in the main
thread cannot be mutated by those threads outside that context. Generation/model methods
already enter inference mode where appropriate. Also disable the shadow predictor when
freezing swaps; its validation intentionally requires request-unit adaptation.

## TensorFold's EXL3 packs are its "uncensored" model, and its fast loader is AGPL-risky (2026-10-08)

`~/models/dsv41-tensorfold/prepared/model/*/rank{0,1}/data.bin` (101.8 GB each, `X3Stack`
`trellis`/`mul1`, exactly the per-rank layout an EXL3 engine wants, and present on **both**
nodes) are TensorFold's preparation of `dsv41-uncensored-2.9bpw`, not of the base
`DeepSeek-V4.1-Flash-EXL3-2.9bpw`. The manifest's source key is the sha256 of the base
`config.json`, which the uncensored overlay reuses because it changes attention only -- so a
matching config hash does **not** mean the routed experts match. TF's own
`engine/kernels/exl3/experts.py` docstring names the checkpoint. Build our own pack; do not read
these. Measured: the base pack is 98.24 GB per rank (header 0.87 MB), 6.670 MB per 3-bit expert
and 4.458 MB per 2-bit expert at the TP-output slice.

Porting boundary: TF DSV41's `x3ld.cu` and `loads.py` are "our GLM patch 0580" -- Mia's AI Lab
GLM-kit lineage, AGPL-3.0 after 2026-09-07 -- and the `experts.py` that front-runs them. Do not
copy. The math (`decode.cuh`, `experts_grouped.cuh`, `experts.cu`) is ExLlamaV3/TensorFold,
permissive; `format.py` is vendored as the oracle. Their fast load path is the one file we cannot
port, so it is the known performance risk versus TensorFold.

## EXPERT_FORMAT=exl3 inherited FP4's resident budget; a quarter of the arena sat empty (2026-10-08)

`DSV41_RESIDENT_EXPERTS` is an absolute count. The live `.env` kept FP4's 9,574 when the format
became EXL3, whose 6.67 MB slots give the same 90.2 GB arena 13,514 LRU slots: the expert map
showed 9,574 resident, exactly the FP4 number, and ~26 GB of allocated slots held nothing.
Raising it to 13,500 (87% of all routed experts routable instead of 62%) cost no measurable
decode speed: distinct experts a verify step moved 21.1 -> 21.5 a layer (python) and
17.2 -> 16.0 (explain). Re-derive the budget from `lru_slots` in `/health` whenever the
format or `ARENA_GB` changes.

## One boot per arm cannot resolve a 2 ms decode change (2026-10-08)

The early-verify/draft A/B as three separate gate boots (on, off, on) gave explain 64.5 / 68.7 /
67.5 ms per step: the two "on" boots differed by 3 ms, as much as the effect. The same change in
`tools/bench_accept_ab_tp.py` -- one boot, 24 prompts, arms switched at runtime and interleaved
per prompt, depth pinned -- measured -2.03 ± 0.49 ms. Prefer runtime switches and the paired
harness for anything under ~5%; the single-boot `run.py` live bench moves ±5% on prose
between boots by acceptance alone.

## A per-step GPU read at the loop top undoes host/GPU overlap (2026-10-08)

`DSV41_EARLY_DRAFT` queues the next draft before the host reads the verify result, so the host's
bookkeeping, token emission and `control()` run while the drafter does. With urgent adaptation on
(live; the gate freezes the map), `_decode_adapt_due` read the decode miss counters with
`.tolist()` at the top of every step -- a stream sync that waited for the queued draft. Live
wall time per step went UP (64-65 -> 68 ms prose) while the depth policy's own timer, which no
longer saw the draft, went down. The counters are now staged into pinned memory with the verify
readback (`Model.stage_decode_miss`). Any new device read in the decode loop needs the same
treatment, and "policy step_ms" is not wall time once work moves across the loop boundary --
compare `decode_s / steps`.

## EXL3 dense attention: a third of the microbenchmark's saving, and the drafter disagrees (2026-10-08)

`DSV41_EXL3_DENSE=1` (default off) serves decode's wq_a/wkv/wq_b/wo_a/wo_b and shared expert from
the EXL3 checkpoint's 5-bit matrices. L2-cold the grouped kernel beats the FP8 GEMVs on every shape
(273 vs 394 µs a layer). In the engine it saved 1.30 ± 0.53 ms a step: each matrix is three
launches (input rotation, GEMV, output rotation) and wq_a/wkv stop being one merged GEMV. And
acceptance fell 0.071 ± 0.024 tokens a step: DSpark was trained against the FP8 target, and the
changed target agrees with FP8 greedy only 97.6% teacher-forced. Net -0.22 ± 0.53 tok/s. Do not
retry this as "fewer bytes = faster" without (a) the rotations fused into the GEMV and (b) a plan
for the acceptance loss; the 2.4% top-1 change is a quality question of its own.

The one reusable finding: the abliteration overlay (`DSV41_ABLIT_WOB`) is rank-1 against the base
wo_b in every layer 10-35 (σ1 6.8-9.5, then ≤ 0.62 and flat at FP8 rounding), so any re-quantized
wo_b can carry it as u (v · x) instead of a second matrix.

## A decode kernel reused for prefill re-reads the weights every 16 rows (2026-10-08)

EXL3's first prefill path fed `build_routing`'s 64-row blocks to the decode kernel, which gives
each 16-row m-tile its own program and splits K across warps. Each m-tile program decoded and read
the expert's whole N slice again, and the K split needed a shared-memory reduction: 52.8 ms a layer
at 2,048 rows vs 25.5 for `exl3m_prefill` (one decode per tile for the block's rows, warps on N).
Bigger blocks were not the fix -- BM 64 with 4 m-tiles was slower than BM 32 (accumulator
registers cut occupancy) -- splitting N instead of K was. The same branch also called
`torch.unique` with a host sync on every layer because the FP4-only static routes were skipped
for an arena without a null slot; a -1 null entry makes them work for TP EXL3 (bit-identical).

## dual-up.sh runs the image's code; dual-dev.sh runs the working tree (2026-10-08)

A restart with `scripts/dual-up.sh` brought the pair back healthy but unusable: `/health` said
`"kernel": "exl3-ref"` and a request took minutes. The `exl3-ref` image predates the EXL3 CUDA
kernel, and only `scripts/dual-dev.sh` bind-mounts `engine/ tools/ server/` from the checkout --
every EXL3 measurement since the kernel landed ran that way. Check `kernel` (and `DSV41_DEV_SOURCE`
in `docker inspect`) after any restart, and rebuild the image (`scripts/dual-build.sh`) whenever
the serving code should survive a plain `dual-up.sh`.

## Small computed routing probes have a fixed cost and lose context (2026-10-09)

Do not equate a token fraction with a time fraction. On the native-FP4 TP2
computed-probe gate, a 64-token probe of three 8K retrieval prompts took median
489.8 ms including planning (9.29% of ordinary prefill), yet promotion precision
was only 40.43% and recall 23.99%. A cached 964-token suffix paid 436.2 ms for its
64-token probe, 49.16% of remaining prefill. Bigger probes improve selection
agreement in the small screen but cost more; neither answer quality nor selection
equivalence has been established. Keep the backend experimental, and evaluate
cached workloads separately from cold long prompts. See the 2026-10-09 results.

The probe uses the same router weights but different contextual hidden states:
excerpts are joined, positions change, and omitted context no longer contributes
attention. Real routing on this smaller sequence is still an estimate of routing
on the full sequence. Preserve original-source Engram n-grams at each excerpt
boundary; hashing the joined sequence invents n-grams. Cached instruction-head
rows must not vote as if newly prefilled. Keep probe counters, KV, replay state and
hash writes separate from the actual request, and bypass prefill FFN graphs whose
captured recorder pointers still refer to the real request buffers.

Shadow promotion evaluation is rank-0-only. The first gate reached its final
summary and then crashed on rank 1 when it assumed that evaluation fields were
available on both ranks. Drivers must tolerate absent rank-local telemetry and
both ranks must still reach the final completion collective.

## Topic-shift coverage gains do not establish quality or latency gains (2026-10-09)

The CS/Python/news -> synthetic medical gate is more favorable to a computed
probe than the mixed-history retrieval screen. Across three 8K medical documents,
64-token early apply reduces median first-prefill count misses from 17.03% to
9.35%, with ~280 early expert loads. The off-demand counterfactual still falls
to 9.53%, so this is improved resident coverage, not only changed hidden states.
The 64-token compute/planning ratio is 9.54%, but another 929 ms is spent loading;
total prefill increases from 6.430 s to 7.797 s. The 256-token probe cuts misses
to 6.85% but exceeds the 8K compute target at 12.97%. All arms answer the three
questions correctly. Keep coverage, graded answers and latency as separate
measurements; lower misses alone do not prove better medical reasoning.

Timing is one sample per case/arm with warmed kernels and shared file/Engram
caches. First-use 64-token prediction/planning took 1.790 s separately. Skipping
probes during stable-topic traffic does not automatically eliminate that first
probe cost. A topic-change trigger also cannot be inferred from this gate's
history-demand TV: Python continuation is 0.608 and medical cases 0.615–0.625.
The intended trigger remains unimplemented. See `RESULTS.md` and
`results/topic-shift-20261009/` for exact workloads and matched-state checks.

## Full suffix discovery must restore the actual cached context (2026-10-09)

For 1–499 new tokens, starting a full pass in an empty scratch cache would lose
the conversation context and become another approximation. The two-pass path
starts at the actual restored prefix, protects its bounded window rings, pending
compressor input and replay containers, then restores that boundary before real
prefill. Compressed/index rows below prefix//ratio must stay unchanged, including
odd prefix lengths where an unpaired old token completes a group with the first
new token. The gate covers odd/even boundaries and new suffixes crossing a chunk
boundary, and hashes every valid compressed/index prefix row. Counting only
new decoder rows also changes demand: ordinary bounded replay includes cached
tail queries, so full discovery must collect them too.

At 99.80% prefix reuse, the 64-token medical turn's full discovery takes about
0.6 s and early loading another 1.1 s for 326 experts. Its instrumented total
prefill is 3.107 s vs 0.589 s off, including 72 later correction loads. A full
discovery match does not make expert I/O free or guarantee identical demand after
changing the map (apply TV 0.114). The short path intentionally exceeds the
excerpt 10% target; telemetry marks that target not applicable. Preserve the
earlier excerpt-negative results when using this short-input exception.

The initial two-pass gate repeated the rank-1 metadata mistake: `finish` evaluates
shadow only on rank 0. It passed cached-prefix isolation and exact routing first,
then crashed on the peer's missing TV key. Broadcast rank-0 evaluation to both
ranks before gate assertions; never assume every rank has evaluation fields.
The corrected gate passed. Logs remain in `results/two-pass-prefill-20261009/`.


### Moving the checkout while the pair is running (2026-10-09)

Changing a project's source folder does not change existing containers' bind mounts
or the image already running. Here the checkout was renamed while both ranks still
mounted its original `results` path. The running process could read the bound history,
but after stopping, `docker cp` and a helper using `--volumes-from` could no longer find
it at that original path. The latest demand DB was recovered from the renamed checkout
and copied byte-for-byte into both new checkouts before restart. Preserve live demand
history before removing or renaming a serving checkout; validate both image code labels
and the host paths of `results` and `.triton-cache` after boot.

## A typed `<｜deepseek_image｜>` is text, not an image (2026-10-10)

Reported by the DeepSeek Harness as
`400 cannot encode messages: Message content contains image special token '<｜deepseek_image｜>'`.
The harness (and any agent that reads this repo, its logs, or the checkpoint's
`encoding/encoding.py`) puts the literal into a tool result, and the checkpoint's reference
encoder refuses the whole request when it sees it in `content` or `reasoning_content`. The
refusal is right about the danger -- the tokenizer reads that byte string as image id 129264,
an image with no pixels behind it -- but it takes every *later* turn of that conversation with
it, because the offending message is now permanent history.

The server escapes the literal before the encoder runs: `escape_image_tokens` replaces it with
`<` + U+200B + `｜deepseek_image｜>`, which tokenises as ordinary BPE and can never be id 129264.
That covers message text, `reasoning_content`, tool results, tool-call arguments, tool schemas
and a `response_format` schema, and `/v1/completions` escapes raw text as well. Tool-call
arguments are decoded from their JSON string and re-encoded, because a Python client's
`json.dumps` writes the fullwidth bars as `\uff5c` and a raw substring replace would miss them.
Real image blocks never pass through the escape: the encoder inserts their placeholder after
that point, once the pixels exist, so a genuine image is still the only way the id reaches the
model. The price is eight ordinary tokens where the model used to read one reserved id; that is
the point, because the id had no image behind it.

Related, not fixed: `_reject_images` counts an `input_image` block as an image (so it is not
refused for lack of a vision tower), but the checkpoint encoder's `_process_image_blocks` only
knows `image` and `image_url`. An `input_image` block therefore renders as
`[Unsupported input_image]` and its pixels are dropped silently. No client in the DSH incident
sent one; a Responses-API client would need this normalised to `image_url` first.

## The drafter's heads are at the memory ceiling, not limited by their format (2026-10-11)

The obvious next move after `DSV41_EXL3_DRAFT_DENSE` (`docs/exl3-draft.md`) is to quantize the
drafter's other two weight matrices: its LM head (fp8, 331 MB a draft pass, one call) and its Markov
bias head (bf16, 33 MB read five times, 165 MB total). Both were measured through
`DSV41_MARKOV_HEAD_FMT` and `DSV41_DRAFT_HEAD_FMT=fp4`, with the two formats switched inside one
process so run-to-run drift could not fake an effect.

**They do not help: +0.29 ± 1.03 ms/step**, and acceptance −0.018 ± 0.014 accepted tokens/step.

The reason is worth keeping. The fp8 draft head's single `_fp8_linear_kernel` call is grid
`[2020, 1, 1]` × 32 = 64,640 = vocabulary/2, and it draws 331 MB in 1.47 ms — **225 GB/s, which is
this box's plain-read ceiling.** fp4 halves the bytes to 165 MB and is not faster, so the FP4 Triton
kernel's cost per byte at M = 5/6 consumes the entire saving. A GEMV at M = 1 against a 33 MB matrix
is the same story. Do not reach for a smaller format here without measuring it: on this box, at these
shapes, "fewer bytes" and "less time" have come apart for the draft's heads while they still hold for
the routed experts and the dense projections.

The draft pass is 335 kernel nodes against TensorFold's 147, so the remaining work on it is fusing
its elementwise/mHC/norm chain, not a format change. (For scale: the *verifier's* head is a different
lever entirely — `HEAD_FMT=fp8` was measured at 2.96 ms against bf16's 5.68 ms for +0.0004 nats, and
the profile currently runs lossless `HEAD_KERNEL=packed` instead. That is a quality decision, and it
is four times the draft's remaining gap.)

## `dual-dev.sh` overrides the caller's settings with `.env` (2026-10-11)

`scripts/dual-up.sh` keeps a caller's `EXPERT_FORMAT=... ARENA_GB=... scripts/dual-up.sh` over
`.env`. `scripts/dual-dev.sh` sources `.env` itself before exec'ing `dual-up.sh`, so the caller's
value is already gone by the time `dual-up.sh` captures it: `EXPERT_FORMAT=exl3 ARENA_GB=92.3
scripts/dual-dev.sh` booted the `.env`'s FP4 at 98.4 GB, healthy and silent. Variables `.env` does
not set (an experiment switch) pass through, which is why it went unnoticed. Until it is fixed,
sync engine/tools/server to the peer and call `DSV41_DEV_SOURCE=1 scripts/dual-up.sh` directly, and
check `/health` `engine_config` before benchmarking a profile.

## Draft heads: fewer rows worked where a smaller format did not (2026-10-11)

Follow-up to the entry above. The fp4 arm changed the kernel together with the bytes. Reading
fewer *rows* through the same fp8 and cuBLAS kernels -- the drafter's heads over 40,960 listed ids a
rank instead of all 64,640 (`DSV41_DRAFT_VOCAB`, docs/draft-vocab.md) -- took the draft graph 6.07 ->
4.58 ms a pass and the step **-1.06 +- 0.37 ms** (+0.56 +- 0.22 tok/s), acceptance -0.0013 +- 0.0013,
24/24 outputs identical. The Markov GEMVs fell more than their rows (0.80 -> 0.27 ms for 0.63x the
rows), consistent with the 21 MB of listed rows staying in L2 across the five back-to-back reads.

Three things that went wrong on the way and will go wrong again:

* **Per-half quotas waste rank 1.** The common tokens of this tokenizer sit mostly below id 64,640;
  an equal list per vocabulary half missed 2.13% of held-out English against 1.17% for the same
  number of ids chosen globally. The list is split into contiguous runs instead, which also keeps
  the lower-id tie rule.
* **Low ids are not common ids here.** Ordering by id covered 87.5% of held-out English docs at
  24,576 a rank, against 99.5% ordered by corpus counts. Do not fill a short list from the lowest ids.
* **The A/B prompts' replies are all over results/.** `prefill-trace-20261008` held 35 of 42,
  `exl3-tune-20261008` 8, each generation gate one. A ranking that included them reported 0.59%
  unlisted at 32,768 a rank; the honest figure is 1.46%. `tools/build_draft_vocab.py
  --exclude-replies-of` exists for this.

## The A/B driver's first run of each prompt is ~1.5 ms/step slow (2026-10-11)

`tools/bench_accept_ab_tp.py` warms up only when the arm changes. With per-prompt alternating order
the arm never changes between prompt i's second run and prompt i+1's first, so the first run of
every prompt captures that prompt's graph variants (parities, index buckets) inside its measured
window. Splitting each A/B's paired deltas by which arm ran first:

| A/B | first-run penalty | arm effect, order removed |
| --- | ---: | ---: |
| `exl3-draft-ab/ab` (x3d_off -> x3d_on) | +1.64 ms | -0.82 ms |
| `head-fp8/ab` (h_packed -> h_fp8) | +1.05 ms | +1.99 ms |
| `draft-vocab-20261011/ab-40960` (vocab_full -> vocab_listed) | +1.62 ms | +0.06 ms |

Alternation cancels it in the mean, so past results are unbiased, but it dominates the error bar:
the draft-vocab A/B read -0.02 +- 0.70 ms/step without a warm-up per run and -1.06 +- 0.37 with one
(per-prompt SD 3.3 -> 1.74 ms, penalty -0.23). `--warm-each` warms up on the prompt before every
measured run; it doubles the decode time of a gate and roughly quarters the variance. A null result
from this driver without it, at an expected effect near 1 ms, is not a measured null.

## Skipping a missing expert costs more than substituting it (2026-10-11)

The router masks non-resident experts before its top-k, so a missing pick becomes the next-ranked
resident expert. Skipping that slot instead would save the substitute's expert read, so it was
measured: teacher-forced NLL on `corpus/heldout_eval.jsonl` (109 sequences, 51,778 scored
positions), serving's resident set frozen (fp4, 10,467 slots), four rules switched in one process
(`Model._apply_miss_policy`, eval-only; `tools/bench_miss_policy_tp.py`). 11.0% of routed slots
missed on this corpus. Difference from `substitute`, nats, SE clustered by sequence:

| rule | coding | general | sequences worse |
| --- | ---: | ---: | ---: |
| substitute (serving) | 1.5132 | 2.1211 | -- |
| skip, renormalize | **+0.0212 +- 0.0044** | **+0.0333 +- 0.0070** | 81 / 109 |
| skip, unpruned weights | +0.0126 +- 0.0058 | **+0.0254 +- 0.0058** | 72 / 109 |
| random resident, same weight | +0.0117 +- 0.0033 | **+0.0158 +- 0.0039** | 70 / 109 |

Both skips fail the +0.015-nat budget on prose, and renormalizing is the worst rule. The substitute
is not noise: a random resident expert in its slot, at the same weight, is 3.5-4 SE worse. But the
random one still beats an empty slot on prose, so part of a substitute's value is only that the
slot holds *an* expert output at that weight. Repeating `substitute` reproduced it bit for bit.

Scale: prefill on this corpus misses 11%; decode on the code benchmark missed 3.2-3.9%, so the
decode cost of skipping would be roughly a third of the table -- and so would its saving (~2-5%
of expert reads, est. ~1 ms/step, needing a zero slot in TP mode). Not pursued.

## Top-p expert pruning: 0.90 is free, TensorFold's production 0.85-renormalized is not (2026-10-11)

TensorFold's lossy lever (`TF_DSV41_EXPERT_TOPP=0.85 MIN_K=3 RENORM=kept`, +5% decode there): per
token and layer, keep the shortest prefix of the top-6 by weight share that reaches top-p (at least
3) and drop the rest. Measured here before building it (`Model._apply_expert_topp`, eval-only;
`tools/bench_miss_policy_tp.py`, same held-out NLL harness and frozen serving residency as the entry
above, every scored position pruned):

| rule | routed slots kept | coding | general | sequences worse |
| --- | ---: | ---: | ---: | ---: |
| unpruned (substitute) | 100% | 1.5068 | 2.1560 | -- |
| top-p 0.90, router weights | 91.8% | -0.0020 +- 0.0018 | -0.0004 +- 0.0034 | 47 / 109 |
| top-p 0.85, router weights | 80.9% | +0.0138 +- 0.0032 | +0.0106 +- 0.0047 | 73 / 109 |
| **top-p 0.85, renormalized (TensorFold prod)** | 81.2% | **+0.0211 +- 0.0036** | **+0.0214 +- 0.0044** | 83 / 109 |

0.90 drops 8% of picks with no measurable loss; 0.85 drops 19% inside the +0.015 budget (code at
+0.014 is close to it); renormalizing the survivors fails on both corpora. That is the third time
today renormalization lost to keeping the router's own weights (skip vs skip_raw above; TensorFold's
own docstring cites Training-Free Halving for the same) -- TensorFold chose `kept` on a speed number
that bytes cannot explain, since both rules drop the same experts. Speed is not measured yet: the
saving is in distinct experts a verify block reads, which falls less than the picks do.

## At FP4's residency the French translation prompt fails, top-p or not (2026-10-11)

`tools/bench_accept_ab_tp.py`'s prompt 18 ("Translate into French: The meeting has been moved to
Thursday afternoon because of the holiday.") comes out mixed-language in every FP4 gate run on
record -- "La réunionión a été verplaatst pentru a dilluns délután porque van de vakantie"
(2026-10-08 base), "Le rendez-vous a été moved to Thursday afternoon..." (2026-10-11) -- and correct
in both EXL3 arms of the same day ("La réunion a été déplacée à jeudi après-midi à cause du jour
férié."). The gates run FP4 at ~62% residency and EXL3 at ~90%; the demand history is English- and
code-heavy, so the experts French needs lose their slots and substitutes produce other languages
(Japanese drifting into Chinese in serving is the same thing). Serving heals it: adaptation loads
what such requests ask for, and the text improves as a conversation goes on. The gates freeze the
map (`DSV41_PRUNE_SWAP=0`, decode adaptation off), so there it never heals and stays visible in
every run -- which makes the prompt a useful canary for residency or ranking changes. Throughput
gates do not notice (acceptance is fine); read the text.

## Ranking residency by top-p share does not beat the raw router score (2026-10-11)

With `DSV41_EXPERT_TOPP=0.90` dropping the low-share picks in decode, the question was whether the
demand database should stop paying for them: today `DSV41_PRUNE_METRIC=score` adds each unpruned
top-6 pick's raw router score, so a pick carrying 3% of a token's weight votes nearly as much as one
carrying 40%. `tools/bench_rank_metric_tp.py` on the EXL3 all-weight profile (every expert resident,
so any resident set is emulated by the router mask alone): demand recorded over both trace corpora
(152 documents, 26K tokens, one request-unit vote each), one resident set per metric at FP4's
9,793-expert budget via `global_residency.initial_selection`, held-out NLL
(`corpus/heldout_eval.jsonl`) with decode's top-p 0.90 on every position. Paired against `score`:

| metric (per pick) | experts differing | miss rate | share missed | coding | general |
| --- | ---: | ---: | ---: | ---: | ---: |
| all resident (floor) | -- | 0 | 0 | 1.5336 (-0.006) | 1.9669 (-0.234) |
| **score** (raw score; serving) | 0 | 13.1% | 11.1% | 1.5394 | 2.2009 |
| frequency (1) | 438 | 13.1% | 11.8% | +0.0172 +- 0.0036 | +0.0344 +- 0.0065 |
| kept (1 if top-p keeps it) | 326 | 13.1% | 11.5% | +0.0121 +- 0.0033 | +0.0329 +- 0.0076 |
| share (w / sum w) | 168 | 13.2% | 11.3% | +0.0010 +- 0.0027 | +0.0116 +- 0.0037 |
| share_kept (share if kept) | 202 | 13.2% | 11.2% | -0.0029 +- 0.0024 | +0.0025 +- 0.0039 |

Without top-p in scoring (what a prefill computes), share_kept vs score is -0.0017 +- 0.0022 /
+0.0009 +- 0.0041. The reference arm re-scored bit-identically at the end.

Read: weighting is what matters -- the two count metrics lose 0.012-0.034 nats at the same miss
rate. Dropping the picks top-p discards helps a little within a family (share_kept vs share, kept
vs frequency), but normalizing each token to one vote hurts (share vs score, 3 SE on general): the
raw score's magnitude is itself information, since a token the router is sure about outvotes one it
is not. share_kept only gets back to a tie with score, which does not pay for a new metric and a
fresh database. Untested: raw score restricted to the kept picks (`score_kept`), which keeps both
effects. The residency cost itself dwarfs every metric difference on general text (+0.234 for the
best set vs +0.006 on code): 26K tokens of mostly English/code training demand leave whole topics
unranked, which is the gap serving's adaptation exists to close.

## Concurrent lanes shared one verify-block buffer: `copy.copy` shares `_vbufs` (2026-10-14)

`make_engine_lane` gives each lane a `copy.copy` of the engine and then replaces the per-request
state. It replaced `_blk`/`_vout`/`_vhost`, but the decode loop reads `self._vbufs[width]`, a dict
of those triples per verify width that the shallow copy shares. The lean greedy path builds the
verify block in place in that buffer and yields it, so every lane's `VerifyStep.block` is the same
tensor: the last lane to draft overwrites the others' blocks before the batched step reads them.
Found by reading and fixed before the batched runtime first ran, so its effect was not observed;
nothing would raise. Each lane now gets its own `_vbufs`. Any new per-request buffer on the engine
has to be added to `engine/serving.py:make_engine_lane`, or every lane shares it.

The opposite case: state an expert swap *reassigns* lands only on the engine view that ran the
swap. A swap inside a lane's prefill set that lane's `_swap_baseline`, and the primary engine's
between-requests pass kept its stale one and re-planned early. Such attributes are carried between
the primary and the active lane in `engine/serving.py:_SWAP_STATE`.

## Wide decode tiles: BLOCK_K 256 at 64 rows does not fit shared memory (2026-10-14)

Batched decode (`engine/batch_decode.py`) runs the FP8 decode kernel at 32/64-row tiles. With the
live `DSV41_FP8_DECODE_BLOCK_K=256`, a 64-row tile stages ~40 KB a pipeline stage, 115 KB over
three, against GB10's 99 KB: the launch fails with `OutOfResources`. Wide tiles drop to K 128 when a
stage would pass 32 KB (`fp8_linear._wide_block_k`). That is bit-identical, because each output
still accumulates over K in the same order: `tools/test_decode_rows.py` compares every row of a
24-60 row call against 6-row calls on the 16-row tile at K 256, under BLOCK_N 32/auto and K
128/256. The 16-row tile never changes.

## Concurrent lanes: rank 1 verified a root-only block (2026-10-14)

`_decode_step_value` returns a placeholder 0 on rank 1 and relies on `EPDistributed.control` to
broadcast rank 0's choice. A concurrent lane's `ScheduledPeer.control` has no broadcast -- the
scheduler owns termination -- and kept each rank's own value, so rank 1 took mode 0 ("root-only",
a 2-row verify) while rank 0 verified the full block. With fixed depth there is no 2-row buffer
and rank 1 died with `KeyError: 2` on its first decode step; had the width existed, the two ranks
would have replayed different graphs. A lane's `control` is now the real broadcast: both ranks
reach it inside the same scheduler action, so it pairs up, and it carries rank 0's dynamic depth
and decode-adaptation flag to rank 1. (A first fix had each rank compute the fixed depth itself;
it could not carry a depth the policy chose, and was replaced.)

Related: `copy.copy(engine)` also shared the `DepthPolicy` object between lanes, so a new
request's `reset_request()` reset the depth of the requests already running and every lane trained
one set of step-cost and acceptance statistics. Each lane now deep-copies its policies.

## Raising `DSV41_ROCE_MAX_KB` to carry batched gathers made every graph slower (2026-10-14)

Batched decode's first MoE all-gather is (pairs + rows) x 1152 bf16 a rank: 387 KB at four lanes,
past the 256 KiB RoCE slot, so it falls back to NCCL inside the graph. Raising the slot to 1024 KiB
to keep it on RoCE was measured with `tools/bench_batch_decode_tp.py` (TP2, keep 0.57, same build,
two consecutive runs, one arm each): the four-lane batched step went 199.4 -> 253.2 ms, and graphs with no collective
at all slowed too (attention core 25.1 -> 32.4 ms, per-lane HC/router 18.3 -> 22.1 ms). The MoE
graph itself did not get faster (125.5 -> 126.9). Keep 256; the NCCL fallback for that one gather
is not where the time goes.

## Concurrency's memory came out of the steady-state margin (2026-10-14)

A two-lane session on the 512K / 98.4 GB profile took rank 1 to the memory watchdog after ~17
minutes: two clients, then one client with a growing conversation. Two causes. The arena check
at boot (`cap = MemAvailable - KEEP_FREE_GB`) only bites when the requested arena exceeds what is
available at load, and a hand-tuned `ARENA_GB` does not -- it is tuned so one request *settles*
with a few GB free -- so the lanes, built afterwards, came straight out of that margin. And no
graph was ever released: the batched graphs, each lane's attention-core pool (growing with the
context bucket) and lane 1's own single-request graphs all stayed after their last use. Rank 1 is
the tight node: at concurrency 1 the same session bottoms out at 3.46 GiB there, ~1 GB above the
2.5 GB floor. Fix and measurements: docs/concurrency.md, "Memory". A first version of the fix only
subtracted the reserve from the cap, which this profile never reaches; it would not have helped.

## Batched rounds trained the depth policy (2026-10-14)

With dynamic depth, a request alone on the concurrent runtime ran 15-18% slower than on the
single-request path (42.8-43.4 against 49.0-52.0 tok/s, same prompt); at fixed depth the two were
within 1.5%. The policy times each step from the top of the loop to its accept, and on the runtime
that includes the batched round the lane waited in -- a round's wall time, set by every lane in it,
so depth 3 and depth 5 read alike and both read ~3x slow. The gate log shows it: lanes in a batched
round compared a batched-timed depth 3 (11-17 tok/s) with a solo-timed depth 5 (32-56) and all went
to 5; the lone request afterwards estimated depth 3 at 17.0 tok/s while running at ~50. A depth the
policy stops choosing keeps its stale time indefinitely. Fix: `_policy_step_s` returns None for a
step verified in a batched round. Capping batched rounds at depth 3 as well was measured and
rejected: it cost a code pair ~20% of its tokens a step (docs/concurrency.md).

## Releasing a lane's graphs at every request end (2026-10-14)

The first memory fix for concurrency released a lane's attention-core graphs when its request
ended and the batched graphs after 128 solo steps. Memory-safe, but every concurrent pair then
recaptured: 1.1-1.4 s a pair on the serving trace, ~8% of a pair of 200-token answers, and part of
why two streams together came out 14-16% *below* one after another there. The bound that matters
is per bucket, not per request: a lane now keeps one context bucket of core graphs and drops it
only when it captures another (`BatchDecoder._drop_stale_core`).


## safetensors slices kept every Muse expert in host memory (2026-10-15)

The Muse engine's first real load (engine/muse/) read each expert's rank half with
`safe_open(...).get_slice(name)[a:b]` and copied it into a preallocated GPU slot. Host anonymous
memory then grew by the bytes read: +15 MB a MXFP4 slot, +53 MB a BF16 expert, never returned
(measured: 128 experts +2.0 GB; 1,000 experts +16.5 GB; MemAvailable fell in step). The first
load was OOM-killed at 40 GB anon; the second was stopped by `scripts/mem-watchdog.sh --kill` at
3.5 GiB MemAvailable on rank 1. The same calls in a small standalone loop did not retain
anything, so the trigger is not obvious; it was not chased further. `Checkpoint._read` now parses
the shard header once and `preadv`s the row range into a fresh tensor: 128 experts +0.0 GB, and
a full 5,098-slot load holds 1.2 GB anon with MemAvailable flat at 28 GB through placement. Any
loader on this box should be checked the same way (log RssAnon per layer) before it fills an
arena.

Also from that load: size arenas the V4.1 way (a 20 GB floor under MemAvailable, then a share of
the rest), drop the page cache of files read once (`posix_fadvise(DONTNEED)`), and refuse to start
below ~90 GB available. GPU allocations themselves count once (8 GiB `torch.zeros` = 8.9 GB off
MemAvailable, no further growth when written).

## GLM-5.3 at 39% routed residency: adaptation frequency does not move the miss floor (2026-10-16)

Question: with the decode swap cap at 256, is the adaptation interval what limits the miss rate?
Workload: tools/glm_adapt_test.py, the one ISO-8601 coding prompt, 600 tokens x 3 rounds, eager,
7,524 of 19,456 routed experts resident, placement from the trained stats DB (328,554 tokens),
decay 0.9, margin 1.5. Results in results/glm-freq/every-{64,200,600}/.

| MUSE_ADAPT_EVERY | round 0 decode mass missed | round 2 | round 2 swaps (s) | churn r0/r1/r2 |
| --- | --- | --- | --- | --- |
| 64 | 32.7% | 21.0% | 835 (9.7) | 0 / 0 / 0 |
| 200 | 37.6% | 20.9% | 324 (3.5) | 39 / 13 / 3 |
| 600 (prefill only) | 41.7% | 21.6% | 588 (2.6) | 0 / 492 / 200 |

Frequency matters only inside a request the residency has not seen: round 0 drifts from 29% to
47% by the end of a 600-token answer without decode adaptations, and stays ~33% at every 64. Once
the set has settled, all three land at ~21% -- the floor is capacity, not cadence or swap count.
The earlier 1,024-swap cap settled at the same ~19-20%. The stats DB is not the fix either: a
placement covers ~67% of the mass of the routing it was built from and ~58% of another corpus's
(DB placement vs the trace-v3 set; they share 5,010 of ~7,500 experts), so a fixed 39% set
misses about a third in-sample. The trace curve put ~10% mass missed at ~70% residency. Default
stays 200: same floor as 64 at a third of the swap time.

## GLM-5.3 decode misses: router weight does not single out the costly ones; background streaming (2026-10-16)

Same workload as above (every 200, round 2 settled: 23.7% decode mass missed, 172 missed picks a
token). `glm_adapt_test.py --hist`: missed picks by router weight (share of the token's routed
mass; uniform top-8 = 0.125). Streaming every miss at or above a weight, synchronously:

| weight >= | streams / token | mass left missed |
| --- | --- | --- |
| 0.30 | 0.65 | 23.4% |
| 0.20 | 4.4 | 22.2% |
| 0.16 | 10 | 20.9% |
| 0.12 | 39 | 15.7% |
| 0.10 | 83 | 9.3% |

GLM's sigmoid top-8 weights are flat; the misses are ordinary picks (0.06-0.12). A cold read costs
~15 ms for one expert (rank half, ~10 MB), ~30 ms for eight at once (~3.8 ms each; `--read-bench`),
against a ~110 ms decode token, so synchronous streaming of 80 a token is out.

Background streaming, simulated on a decode routing dump (`--dump`, tools/glm_bg_stream_sim.py):
- reactive (a miss queues a read, heaviest first, B a token; it lands next token in an LRU of C
  transient slots): C=1024/B=40 21.6% -> 15.5%; C=2048/B=40 -> 12.4%. The slots are not free --
  they come out of the resident set -- so the net is smaller than shown. Short-term reuse is low:
  only 14% of missed mass was picked by the same layer the token before, 43% within 16 tokens.
- pre-gating (layer N's router on layer N-d's input): recall of missed mass 66% at d=1, 56% d=2,
  44% d=4, 31% d=8. A layer is ~1.4 ms and a read 15 ms, so d=1-2 leaves no time to read, and at
  d=8 recall is a third. It also costs ~135 reads a token. Not usable for decode.

## GLM-5.3 background expert reads and prefill prefetch: what they buy (2026-10-16)

Decode (MUSE_BG): every 2 rounds rank 0 ranks non-resident experts by recent router mass (decay
0.7 a period), reads up to 48 on the thread pool, installs them over cold residents at the next
boundary. First version never planned after the first request: victims needed recent mass == 0,
and a decayed score never reaches zero (4 reads in 1,200 tokens). Victims now need mass below
MUSE_BG_COLD (0.01). glm_adapt_test, ISO prompt, eager, decode mass missed round 0 / 1 / 2:
37.6 / 22.7 / 20.9% off, 24.2 / 20.4 / 19.8% on; reads waited 0.02 s in 600 tokens; ~14K
installs a round (the budget is always spent -- it churns, and the settled gain is ~1 point).
Speed, graphed, MTP depth 2, glm_probe 3 prompts x 400 tokens: 13.25 off, 12.03 at 24 reads,
11.97 at 48; the off run's text was visibly broken ("the premise is flawed -- the sky isn't
blue", "#!/usr/bin bash"), both on runs clean.

Prefill (MUSE_PF): at each MLP the next layer's router runs on this layer's input; the chunk's
heaviest non-resident experts (by summed mass) are read into one of two transient buffers while
this layer and the next attention compute, and are allowed for that layer only. 3,321-token
prompt (engine/glm/ref.py), round 0: prefill mass missed 30.8% off; 18.4% with 32 a layer but
TTFT 24.5 -> 43.6 s (18 s waiting: a 1,024-row layer is ~57 ms, 32 reads ~120 ms); 23.4% with
16 a layer, TTFT 30.5 s, 3.4 s waiting. MUSE_READ_THREADS 12 -> 24 did not raise read speed (8
experts ~31 ms either way, ~2.6 GB/s a rank): the buffered read path is the limit, not threads.

## GLM-5.3 quality against the exact model: the heaviest miss matters more than its mass (2026-10-16)

tools/glm_quality.py: three Tokyo-sightseeing chat prompts, 300 answer tokens each (generated once
with background reads on), scored position by position against an exact run (MUSE_MISS=stream,
teacher-forced prefill). Eval feeds the answer as decode steps, eager, DB snapshot at 372K tokens.
NLL is not comparable here (the text is the background-reads model's own greedy output); top-1
agreement and KL(exact || this) over exact's top-20 are.

| config | mass missed | top-1 = exact | KL | sync reads a token |
| --- | --- | --- | --- | --- |
| substitute only | 22.2% | 82.3% | 0.369 | -- |
| + background reads (48 / 2 rounds) | 16.4% | 89.2% | 0.231 | -- |
| + hot 1 (weight share >= 0.2) | 15.8% | 91.4% | 0.135 | 2.5 (45 ms) |
| + hot 2 (>= 0.15) | 14.7% | 91.0% | 0.122 | 8.1 (129 ms) |

Reading the single heaviest miss of a layer synchronously cut KL by 42% while moving mass missed
by 0.6 points: mass missed understates what a top pick costs. A second hot read buys little for 3x
the reads. One synchronous expert read is ~18 ms (six preadv calls in series on one thread).

## Hot-miss reads under graphs: what they cost, and the read path that made them affordable (2026-10-16)

Reads: one cold rank-half expert (six tensors) took 8.8 ms through get_rows one tensor after
another, each tensor's 256 KB pieces in parallel; all six tensors' pieces submitted together
(Checkpoint.get_rows_many) take 3.8 ms. The GPU copy is not the cost (0.2 ms pageable or pinned).
Hot reads have their own expert and piece pools: on the shared ones they queued behind up to 48
background reads (9.7 ms a hot read in eager decode with background reads at 48).

Graphs: decode with MUSE_HOT runs as per-layer graphs ([attention + hot candidate], [MLP]) with a
host sync between; with no hot reads (MUSE_HOT_MIN=9) the samples are identical to the single
graph at fixed MUSE_SLOTS (auto slots differ boot to boot -- an equality test must pin them),
3-4% slower. A fixed 0.2 threshold read in about half the layers of an MTP verify step (3 rows):
4.85 tok/s. The threshold now tracks MUSE_HOT_TARGET reads a step. glm_probe, MTP depth 2,
7,000 slots, background reads 8: 12.50 tok/s without hot reads, 10.63 at target 4 (threshold
settles at 0.37 for 3 rows, 0.29 for 1), 8.98 at target 8.

Quality at 7,000 slots (glm_quality, eager, fixed 0.2): hot 1 + background 48: KL 0.167, top-1
89.1%; hot 1 + background 8: 0.178, 90.7%, with 3,535 background installs instead of 21,324;
hot 2 (>= 0.15) + background 8: 0.147, 91.2% at 12 reads a token. Background 96 + hot 1 was worse
than 48 (KL 0.179 vs 0.139 at auto slots) while missing less mass (15.6% vs 16.0%): eviction
churn costs more than the extra coverage buys, and mass missed alone would have picked it.

## Prefill prefetch costs more TTFT than it buys; keeping prefetched experts churns (2026-10-16)

glm_adapt_test, engine/glm/ref.py as a 3,321-token prompt, 2 rounds x 300 tokens, eager, 7,000
slots, background reads 8, hot reads 1. Round 0 / round 1:

| config | TTFT s | prefill mass missed | decode mass missed |
| --- | --- | --- | --- |
| no prefetch | 23.7 / 20.1 | 33.0 / 28.3% | 36.2 / 35.1% |
| no prefetch, 512 post-prefill swaps | 22.0 / 20.1 | 33.0 / 28.9% | 36.3 / 35.8% |
| prefetch 16 a layer | 32.8 / 29.0 | 25.0 / 21.9% | 36.6 / 39.5% |
| prefetch 24 | 36.0 / 32.4 | 22.3 / 19.9% | 36.6 / 34.6% |
| prefetch 24, keep every read (MUSE_PF_KEEP, no margin) | 51.4 / 50.5 | 25.7 / 26.5% | 39.7 / 37.8% |
| prefetch 24, keep if it ranks in (margin 1.5) | 34.9 / 31.4 | 24.2 / 20.2% | 38.4 / 35.6% |

Prefetch waits on reads for under 2 s; the rest of its ~9 s is the per-layer machinery (rank 0
prediction and host sync, a broadcast, slot writes: ~27 ms a layer-chunk) and buys nothing in
decode. Keeping every prefetched expert replaced ~7,000 residents a prompt -- nearly the whole
set, evicting what later chunks needed -- and the post-prefill adaptation swapped 2,048 back.
Gated on ranking it promoted 4,017, then 416 on the repeat, with the same miss as dropping them.
Prefetch is off by default; post-prefill swaps 512.

Also measured here: 256 KB read pieces for every read (not just urgent ones) made the prefill
7 s slower (32.8 vs 39.8 s TTFT with prefetch 16) -- hundreds of small Python tasks a layer
contend for the GIL with an eager prefill. Bulk reads use whole tensors (MUSE_READ_BULK_CHUNK_KB=0).

Post-prefill swaps on short prompts (glm_quality, the three Tokyo prompts, eager, 7,000 slots,
hot 1, background 8), all 900 positions / first (fresh-topic) answer: 512 swaps top-1 88.8% /
85.3%, KL 0.189 / 0.276; 1,024 89.7% / 88.0%, 0.186 / 0.269; 2,048 90.7% / 88.0%, 0.178 / 0.287.
The fresh-topic answer is no better at 2,048 (its KL is the worst of the three); the overall gap
is ~6% KL. A 2,048-swap adaptation costs ~4-6 s of TTFT on a short prompt: 512 stays.

## Hot reads: fixed 0.2, kept resident, a per-step cap paced over the layers; one-row evals flatter the controller (2026-10-16)

glm_quality on the three Tokyo prompts (900 positions), 7,000 slots, graphed, KL to exact /
top-1; glm_probe tok/s at MTP depth 2 (400 tokens, same DB). `--rows 3` feeds the answer in
3-token blocks, so decode runs the 3-row verify graphs and their per-step thresholds and caps as
serving does; `--rows 1` steps one token at a time.

| config | rows 3 | rows 1 | tok/s |
| --- | --- | --- | --- |
| target-4 controller (old default), bg 8, not kept | 0.299 / 86.4% | 0.200 / 89.2% | 10.7 |
| 0.2, no cap, bg 8, not kept | -- | 0.178 / 90.7% (eager) | 7.64 |
| 0.2, no cap, bg 8, kept | -- | 0.152 / 90.2% | 7.9 |
| 0.2, no cap, bg 8, kept, 2 a layer | -- | 0.161 / 90.7% | 7.5 |
| 0.2, no cap, no bg, kept | 0.172 / 90.3% | -- | 8.24 |
| 0.2, cap 10 first come, no bg, kept | 0.235 / 88.7% | 0.164 / 89.6% | 9.19 |
| 0.25, cap 10, no bg, kept | -- | 0.257 / 87.0% | 10.39 |

- One-token steps rarely find 4 heavy misses, so the controller sat at its 0.2 floor there (0.200);
  in 3-row steps it rose (0.25 at the end of this eval, 0.37 in the probe) and the same config
  scored 0.299. Judge per-step policies with `--rows 3`.
- Keeping a hot read (MUSE_HOT_KEEP: copied over a cold, lowest-ranked resident after its layer)
  cut reads 29% and KL 0.178 -> 0.152 at one row. A second read a layer bought nothing.
- 0.2 -> 0.25 halved the reads (2,552 -> 1,064) and gave back most of the gain (KL 0.164 -> 0.257):
  misses at 0.20-0.25 of a token's routed weight matter.
- A first-come cap lets earlier layers spend it and later layers' misses go unread. Cap 10 hit
  in 39% of 3-row steps and cost KL 0.172 -> 0.235 for 24% fewer reads (10% speed). Pacing it
  over the layers (MUSE_HOT_PACE=g: by the i-th of n MoE layers, at most ceil(cap*((i+1)/n)^g))
  recovers most of that -- 3-row eval, reads by quarter of the MoE layers:

  | cap 10 unless noted | KL / top-1 | reads | by quarter | tok/s |
  | --- | --- | --- | --- | --- |
  | no cap | 0.172 / 90.3% | 2,534 | 596 / 871 / 557 / 441 | 8.24 |
  | first come | 0.235 / 88.7% | 2,043 | 581 / 778 / 359 / 250 | 9.19 |
  | g=0.5 (front-loaded) | 0.220 / 89.0% | 2,024 | 540 / 674 / 410 / 327 | -- |
  | **g=1 (even)** | 0.212 / 89.9% | 1,862 | 425 / 566 / 432 / 367 | 9.29 |
  | g=2 (back-loaded) | 0.231 / 89.1% | 1,500 | 234 / 376 / 412 / 407 | -- |
  | g=1, may run 2 ahead (MUSE_HOT_PACE_LEAD=2) | 0.210 / 89.2% | 2,013 | 529 / 671 / 451 / 292 | -- |
  | g=1, cap 12 | 0.189 / 89.2% | 2,022 | 466 / 617 / 470 / 398 | 9.03 |
  | **g=1, cap 16** (default) | 0.171 / 90.9% | 2,224 | 502 / 737 / 501 / 415 | 8.95 |

  Heavy misses are densest in the second quarter and sparsest in the last; holding budget for
  the last layers (g=2) starves the middle and leaves it unspent (5 reads a step against 10).
  Even pacing at cap 12 reads what first-come cap 10 did, at its speed, for KL 0.235 -> 0.189;
  at cap 16 it matches no cap (0.171 vs 0.172) on 12% fewer reads, at cap 12's speed.
- A lower threshold is worse, not just slower: 0.15 paced at cap 12 scored 0.248 (3,251 reads)
  and at cap 16 0.194 (3,955) against 0.2's 0.189 / 0.171. The 0.15-0.2 misses take budget from
  heavier ones and their keeps evict residents.
- A rank gate on keeping (adapt()'s rule: the newcomer's score must beat the victim's by 1.5)
  would have dropped 817 of 1,969 kept reads, 418 of them below the victim outright; kept experts
  outrank 29% of residents on average. Not built: the single-topic eval cannot show what it
  protects (long-run residents for the next topic), and kept experts that go cold are the first
  victims anyway since their long-term score is low.
- First request after boot is still the weak one: "浅浅浅..." at 200 tokens on a fresh boot with
  this config, typos on the second, clean by the fourth.

## Substitute weights: the router's renormalization beats keeping exact weights (2026-10-16)

A substituted top-k is renormalized over the picks it ended with, so a substitute's lower score
also inflates the resident picks' weights. Two alternatives (MUSE_SUB_WEIGHT; glm_quality,
Tokyo, 3-row steps, hot 0.2 paced cap 12, KL to exact / top-1): renorm 0.189 / 89.2%; `exact`
(resident picks keep their exact weights, substitutes their own score over the exact total)
0.217 / 88.8%; `transfer` (each substitute takes a missed pick's exact weight) 0.217 / 89.3%.
Both lose by the same margin; renorm stays. With V4.1's skip results above, the pattern holds:
leaning on the experts that are computed exactly beats preserving the router's exact numbers.

## Residency is the biggest quality lever left on MXFP4; the 20 GB floor was costing ~1,500 slots (2026-10-16)

glm_quality, Tokyo, 3-row steps, hot 0.2 paced cap 16, kept, no background reads, max_seq 8192:

| routed slots | KL / top-1 | miss mass | hot reads |
| --- | --- | --- | --- |
| 7,000 (36%) | 0.171 / 90.9% | 20.3% | 2,224 |
| 8,500 (44%) | 0.132 / 91.2% | 16.0% | 1,826 |
| 9,394 (48%, 6 GB floor, share 0.97) | 0.113 / 92.3% | 13.6% | 1,598 |

Fewer misses also mean fewer hot reads, so the speed cost of more slots is negative. The stats
DB's static coverage agrees (global top-N by router mass): 7,100 -> 37.9% of mass missed, 8,000
-> 33.4%, 10,200 -> 23.7%, 15,300 -> 6.9%.

Serving (max_seq 32768) has less room than the eval: the KV caches are 2.7 GB larger and a live
pair loses another 4-5 GB of MemAvailable after its first long prefill (allocator reserve, not
growing with context: a 28K-token context after a 20K one added nothing). At 8,900 slots a pair
started at 7.1 GB available on rank 0 and fell to 5.1 GB on a 5.6K prompt -- too close on GB10,
where running out wedges the box instead of raising an OOM. At 8,600: ready at 10.9 GB, minimum
6.0 / 8.2 GB (rank 0 / 1) through 20K and 28K prompts, with an hf download holding ~4 GB on rank
0 and rank 1's polkitd (5.3 GB) and bluetoothd (1.1 GB) leaking. Slots are the minimum over the
ranks, so a leak on either node costs the pair. Watch with
`NAME0=glm-serve-rank0 NAME1=glm-serve-rank1 PEER=ryan@10.0.0.2 scripts/mem-watchdog.sh --kill`.

## EXL3 routed experts: same exact-model NLL as MXFP4 at 71% of the bytes (2026-10-16)

bertholomus/GLM-5.3-EXL3-3.0bpw (exllamav3 1.5.3 from BF16, routed experts 3-bit mul1, MTP
4-bit) packed per rank with tools/glm_pack_exl3.py: 139.1 GB a rank, 7.1 MB a slot against
MXFP4's 10.0. Layout check (cosine of the dequantized rank slices against the Quark MXFP4 ones):
0.981-0.984 at 3 bits, 0.991 for the 4-bit MTP experts, on both ranks -- a wrong slice or
orientation would be ~0. Engine path (engine/glm/exl3.py, V4.1's CUDA kernels, GLM's Comm for the
two gathers) against a dequantized fp32 reference: rel err 3.5e-4 at T = 1, 3, 256.

Teacher-forced NLL on text neither checkpoint wrote (glm_quality --set natural: 12 of
corpus/heldout_eval.jsonl, 64-token prompt, 300 scored, 3-row steps), every expert exact
(MUSE_MISS=stream): MXFP4 1.6141, EXL3 1.6107. Serving config on MXFP4 (8,600 slots, hot 0.2
paced cap 16, kept): 1.6697, +0.056 over its exact, KL 0.126, 23.8% of routed mass missed --
code and docs miss more than the Tokyo set. The EXL3 serving run (11,756 slots, same memory) was
stopped for serving before it finished. Caveat from the user, and fair: the corpus is mostly
AI-written docs, close to what quantizers calibrate on, so an average NLL there cannot show the
rare errors a 3-bit format makes; judge it in use.

First live EXL3 requests: 8.66 tok/s fresh after boot, 9.99 on the second (MXFP4 at 8,600: 6.6-8.6).

## 128K context: an FP8 KV cache costs +0.002-0.005 NLL and buys ~900 EXL3 slots (2026-10-16)

The KV cache is preallocated for max_seq and both ranks hold all of it (the MLA latent is not
split by TP): 96.6 KB a position a rank in BF16 -- 3.2 GB at 32K, 12.7 GB at 128K, 50.7 GB at
512K -- plus the MTP hidden ring (12 KB a position). Decode's indexer also scans max_seq every
step (CUDA graphs need a fixed S).

MUSE_KV=fp8 caches the latent and the indexer keys as E4M3 with an fp32 scale per 128 values and
the MTP ring with one per row; RoPE keys stay BF16 (~54 KB a position). FP8 has no index_copy_ or
gather kernels: kv_write/kv_read move the bytes as uint8. EXL3 routed experts, 10,750 slots:

| | BF16 KV | FP8 KV |
| --- | --- | --- |
| natural set, every expert exact, NLL | 1.6107 | 1.6131 |
| long set (6K/12K/18K prompts, 300 scored, 3-row, served config) NLL | 0.8746 | 0.8795 |
| long set wall time | 298 s | 373 s (494 s dequantizing through fp32) |
| 128K, MemAvailable at ready (rank 0) | 10.9 GB | 17.5 GB |
| 128K, probe tok/s (accepted/round) | 8.94 (1.70) | 8.25 (1.62); per round 301 -> 318 ms |

Serving at 128K, FP8 KV, 11,650 slots (61%): minimum MemAvailable 5.1 / 6.6 GB (rank 0 / 1)
through a 5.6K and a 20K prompt. Rank 0 binds: its engine process holds ~1.5 GB more (the HTTP
server), so freeing rank 1 (polkitd 3.4 GB + bluetoothd 1.1 GB restarted) did not raise slots.

## An idle GLM pair took itself down at the process-group timeout (2026-10-16)

muse_app's follower (rank 1) waits for the next command inside a gloo broadcast, and gloo bounds
every wait by the process group's timeout (DSV41_DIST_TIMEOUT_S, 1,800 s from scripts/glm-up.sh).
The 128K server sat idle for 34 min after its last request: rank 1 exited with "Timed out waiting
1800000ms for recv operation", rank 0 kept answering /health, and the next request crashed it on
the closed pair ("Connection closed by peer"). V4.1's server had already hit this and sends an
idle ping (`_ep_heartbeat`); the GLM/Muse serve loop never got one. Raising the timeout only moves
the cliff, since an idle server waits unboundedly by nature.

engine_loop now waits for a job for at most MUSE_PING_S (default a quarter of the timeout, 450 s)
and broadcasts {"op": "ping"} when none came; the follower skips it. The ping goes out from the
engine thread itself, so unlike V4.1's heartbeat thread it needs no lock to stay out of the gap
between a request's broadcast and its generate.

Side effect worth knowing: with GLM gone from its model list, Open WebUI fell back to an Ollama
model (qwen3.5:122b, 87 GB on rank 0, keep-alive 3 h). Restarting the pair while that is loaded
overcommits rank 0; `ollama stop` it first.

## Decode speed: where an MTP round went, and what took it from 8.6 to 16 tok/s (2026-10-16)

tools/glm_decode_prof.py times every split-graph segment with CUDA events (no added syncs) and the
host phases with wall clocks, then optionally prints a torch.profiler kernel table. Workload: the
serving config (EXL3, 11,650 slots, 128K max_seq, FP8 KV, hot 0.2 paced cap 16 kept), a 6,027-token
prompt (held-out docs + "summarize section by section"), 400 measured decode tokens, greedy.

| step | tok/s | ms a round | accepted / round |
| --- | --- | --- | --- |
| serving config as it was (depth 2) | 8.61 | 309 | 1.66 |
| + indexer kernel, one router matmul a layer, tensor decisions, headless MTP fills | 12.28 | 221 | 1.71 |
| + O_DIRECT expert reads | 14.25 | 190 | 1.71 |
| same, MTP depth 3 | 15.58 | 213 | 2.32 |
| + one all-gather each way in the MoE (bit-identical text) | 15.68 | 212 | 2.32 |
| + dynamic depth: max 5, MUSE_MTP_DYN=0.8 | 16.23 | 202 | 2.27 of 2.66 drafted |
| dynamic, max 5, tau 0.9 | 16.82 | 206 | 2.44 of 2.70 |
| **dynamic, max 7, tau 0.9** | **17.59** | 227 | 2.98 of 3.42 |
| dynamic, max 5, tau 0.95 | 16.26 | 192 | 2.12 of 2.26 |

What the first profile showed (309 ms): the 21 indexer layers' attention took 87 ms against 52 ms
for the other 57 layers. Decode scored every one of max_seq = 131,072 positions with fp32 einsums
over [rows, 32 heads, S] whatever the context (a fixed S for the graph), plus the FP8 key cache
dequantized to fp32 each step: 3.25 ms a layer at a 6K context. engine/glm/index_kernel.py does the
scores in one Triton kernel (BF16 tensor-core dots, fp32 accumulation, the row scale after the
dot; blocks past the last row's position only write -inf): 0.10 ms with the top-k, 0.014 without.
Same scores to 1e-5 relative, top-2048 sets > 99.9% equal (engine/glm/test_index_kernel.py).

Smaller items in that step: _route ran the fp32 router matmul up to four times a layer (the
pick, demand, miss stats, and the hot candidate in the attention graph) -- now once, the masked
and unmasked picks from the same scores (bit-identical); stream()'s per-round decisions went
through broadcast_obj (gloo pickling, ~1.9 ms each) -- now a small int tensor on the RoCE gather;
the MTP cache fills (accepted drafts, prefill chunks) computed the head and gathered [rows, vocab]
logits nobody read (~10 ms eager a round) -- now headless and graphed up to depth rows.

Expert reads: Exl3Pack read buffered, 256 KB pieces on 16 threads, 3.1 ms an expert; one O_DIRECT
preadv of the whole 7.1 MB record takes 0.83 ms (8.5 GB/s), eight at once 9.5 GB/s. Fresh buffers
cost 3.7 ms of page faults a read, so aligned buffers are pooled and lent (exl3._Lease: back to
the pool only when the record's last view dies). Hot reads 22 -> 9 ms a round, adaptation's
synchronous swaps 20 -> 5 ms amortized.

MTP depth, fixed (same workload, O_DIRECT not yet in): depth 1 10.65 tok/s (0.88 accepted), 2
12.28 (1.71), 3 13.33 (2.32), 4 12.89 (2.72), 5 12.51 (2.89). An extra verify row costs 30-45 ms
(the MoE reads more experts, one more draft, more hot reads); a draft pays only above ~0.55
acceptance. The MTP's own probability predicts it: with the product of the drafts' probabilities
>= 0.9, 80-97% are accepted; 0.7-0.9 about half; 0.5-0.7 a third. MUSE_MTP_DYN=tau keeps drafting
while that product is >= tau (rank 0 decides each step and broadcasts it, so both ranks verify the
same rows): 0.3 12.84 and 0.5 12.68 (both before O_DIRECT, too permissive), 0.7 15.36, 0.8 16.23.
Each verify row count captures its own split set; capture inside a measured window cost ~20 ms a
round in the first dynamic runs (the profiler now warms 200 tokens and reports late captures).

Quality of the indexer kernel (the only change above that is not bit-identical or byte-identical):
glm_quality --set long (6K/12K/18K prompts, 900 positions, 3-row steps), kernel on 0.8866, off
0.8826 NLL; paired per position +0.0040 +- 0.0068 (0.6 SE). For scale, the same FP8 config run
twice differed by -0.0015 +- 0.0067 and BF16 vs FP8 KV by -0.0049 +- 0.0068: two runs share 1-2 of
900 positions bit for bit (a flipped near-tie changes routing, hot reads, everything after), so
this eval resolves about +-0.013 and the kernel is inside it.

Memory of dynamic depth: each verify row count captures its own split set (158 graphs). One
capture pool for all sets and MTP graphs saved 0.3 GB; one warm-up stream plus empty_cache() after
a capture another 0.5 GB (a fresh stream per capture stranded each warm-up's cached blocks). What
is left is ~0.23 GB a set -- CUDA's own executable graphs, not tensors: max 7 holds 9.11 GB
available after the run against fixed depth 3's 10.32.

Not worth it: tuning the decode FP8 GEMV tiles. A sweep over BN / warps / stages and a
deterministic split-K (partials summed in order, so both ranks keep identical bits) found 670-1054
GB/s on the small projections -- L2-resident in a microbenchmark, not in a forward -- and only
o_proj (50 MB) is honest: 214 -> 223 GB/s. Split-K helps kv_a (N=576, 13 -> 8 us) and the
indexer key (N=128, 13 -> 6 us): ~1 ms a forward.

Prefill (tools/glm_prefill_prof.py, same 6K prompt, chunk 1024): 42 s, 143 tok/s, of which
attention is 30 s of GPU time and the MLP 10 s. Past 2,048 positions the attention goes sparse:
per 32-row block it gathers each row's 2,048 selected latents into BF16 copies ([32, 2048, 512])
and runs einsums, ~5 s a chunk. The indexer is 0.26 s, the MTP fill 0.7 s, the post-prefill
adaptation 0.8 s.

## Prefill attention: one kernel for the sparse MLA, 42 -> 19.6 s on a 6K prompt (2026-10-17)

Past 2,048 cached positions GlmTP.attention gathered each 32-row block's selected latents into
BF16 copies (kv_read(C, sel): [32, 2048, 512], 64 MB a block, plus the RoPE keys), then ran bf16
einsums and a softmax over [32, heads, 2048] -- 32 blocks a 1,024-row chunk on every layer, 30 of
the 6K prompt's 42 s. engine/glm/sparse_attn.py (MUSE_ATTN_KERNEL=1) gives a program BLOCK_H heads
of one row, loads the selected rows by index (FP8 widened and scaled in BF16 as kv_read does),
scores latent + RoPE parts with fp32 accumulation and folds them into an online softmax.

| shape (one layer) | einsum path | kernel | rel. error vs fp32 |
| --- | --- | --- | --- |
| 1,024 prefill rows | 81.1 ms | 4.5 ms (32 heads x 64 keys, 8 warps) | 1.6e-2 -> 1.8e-3 |
| 4 decode rows | 0.25 ms | 0.17 ms (16 x 64) | same |
| 8 decode rows | 0.46 ms | 0.17 ms | same |

The einsum path rounds its scores to BF16 twice; the kernel keeps them fp32, hence 9x closer to
an fp32 reference. Long-set NLL on 0.8815, off 0.8881, paired -0.0066 +- 0.0073. The first try
needed 104 KB of shared memory (limit 99 KB on GB10): the per-128 scales were gathered as a full
[rows, 512] fp32 tile; loading [rows, 4] and broadcasting through a reshape fixed it.

Prefill, 6K prompt: 143 -> 307 tok/s (attention 29.9 -> 7.4 s of GPU time; the MLP's 10.2 s is
now the larger part). Decode did not move: a decode layer's attention is ~0.94 ms either way, the
projections and the all-reduce being most of it. Decode tok/s under dynamic MTP depth also swings
with the sample's acceptance (2.18 vs 2.98 accepted a round on two runs of the same prompt after a
numerics change), so judge decode changes on several prompts, not one.

## Prefill after the attention kernel: norms, projection tiles, the unread head (2026-10-17)

tools/glm_prefill_prof.py, 12K-token prompt, EXL3 serving config (repeat runs vary by ~1.5 s):

| step | 12K prefill | tok/s |
| --- | --- | --- |
| attention kernel, chunk 1024 | 37.05 s | 324 |
| + head only for the sampled row | 36.61 s | 328 |
| same, chunk 2048 | 35.69 s | 337 |
| + fused RMSNorm, prefill FP8 tiles, first chunk through the kernel | 29.83 s (32.74 on another run) | 403 |
| same without the fused norm | 31.96 s | 376 |

- forward() ran the head over every prompt row (a [chunk, vocab] fp32 product and gather) and kept
  the last: 0.44 s of a 12K prompt at 1,024-row chunks, 1.4 s at 2,048, 2.5 GB of transient at 4,096.
- rms() as seven eager kernels: 918 us at 1,024 x 6,144; engine/glm/norm_kernel.py 74 us, 99.999%
  identical bits (max 1.35 BF16 ulp where not). Decode graphs carry a seventh of the kernels.
- fp8_block_linear's prefill tile (64 x 128, 8 warps, 3 stages) ran o_proj (K = 8,192) at 26.5
  TFLOPS; 128 x 128 / 8 warps / 2 stages 51; the other projections 64 x 128 / 4 / 2 (50-62 -> 60-77).
  Per 1,024-row chunk forward 405 -> 238 ms. cuBLAS on BF16 weights is faster still (170 ms) but
  needs either ~5 GB of resident BF16 copies a rank or a dequantization per call (1.5-6.5 ms each).
- Within the first index_topk positions the indexer selects every position, so the dense einsum
  case ([T, heads, S] fp32 scores) went to the sparse kernel too; it made 2,048-row chunks slower.
- Chunk 4,096 dipped to 2.03 GiB on rank 0 (watchdog stop) before the head fix; 2,048 holds 5.8.
- The EXL3 prefill MoE is at its kernel's limit: a sweep of every compiled (row block, tile) gave
  the default (32 rows, 4x2x2) within 2% at GLM's shapes; 2,048-row chunks cost 20% less a token
  than 1,024 (28.8 vs 18.1 ms a layer). Reconstructing experts to FP16 for tensor-core GEMMs (a
  published GLM-5.2 EXL3 setup does it past 128 rows an expert) would move ~19 GB a layer at 4,096
  rows -- ~87 ms at GB10's bandwidth against ~40 for the fused kernel.
- sparse_mla's tile depends on the cache: 32 heads x 64 keys x 8 warps over a BF16 latent cache
  needs 168 KB of shared memory (GB10 has 99), so MXFP4's BF16-KV config failed at its first
  prefill. The wrapper now picks per cache type (TILES) and the test runs both.

Quality, long set, 3-row steps: 0.8863 with all of the above against 0.8815 (attention kernel only),
paired +0.0048 +- 0.0065. MXFP4 (8,344 slots, BF16 KV, dynamic depth 7 / 0.9): glm_probe 9.8 tok/s.

## MoE output gathers and MXFP4's expert reads (2026-10-17)

- mlp() gathered every pick's fp32 output rows across the ranks and weighted and summed them after
  the gather. The gather only concatenates (each column belongs to one rank), so the sum over
  picks, the routed part and the BF16 cast can run first, per element the same arithmetic: the same
  text, by hash, on EXL3 (6K-prompt decode) and MXFP4 (glm_probe, MUSE_GATHER_MERGE 0 vs 1). MXFP4
  had gathered [9 picks, rows, 3,072] fp32 -- 332 KB at 3 rows, past the RoCE gather's 256 KB, so
  NCCL on every MoE layer: 9.90 -> 10.08 tok/s. EXL3 12K prefill: gathers 2.88 -> 1.55 s of GPU
  time, the total inside the run-to-run spread.
- GlmCheckpoint.get_rows_many (MXFP4 hot reads, swaps, placement) reads with O_DIRECT
  (MUSE_CK_DIRECT=1): a rank-half expert is six ranges (weight + scale of gate, up, down) split by
  the other rank's rows, read in parallel into one pooled aligned buffer (engine/glm/direct_io.py,
  shared with Exl3Pack). Cold expert 4.78 -> 3.09 ms through the reader (the bare reads 1.7-2.1 ms;
  the rest is per-call Python). glm_probe, MXFP4 8,344 slots, dynamic depth 7 / 0.9: hot-read time
  16.5 -> 9.4 s, 10.05 -> 11.70 tok/s, identical reads and acceptance. Bytes checked against
  buffered reads on 33 experts of both ranks.

## FP8 LM head, and deferring adaptation's swaps (2026-10-17)

MUSE_HEAD=fp8 (128-block FP8 of the BF16 head, quantized at load) had never run: `hq, hs =
_fp8_quant(head)` rebound `hq`, the q head width the next lines use for q_b's rows, and boot
failed in a CPU multiply on FP8. Fixed, and measured -- 6K prompt, dynamic depth 7 / 0.9: 191 ms a
round against ~200 (each MTP draft 6.2 -> 4.4 ms; four head reads a round), 17.55 tok/s;
0.48 GB a rank back (~67 EXL3 slots). Long-set NLL +0.0013 +- 0.0011 (paired, 900 positions):
every position moves a little, by ~0.15% in perplexity. Taken for both configs.

Deferring decode adaptations to the background period (read in the background, installed two
periods later by bg_step's own rule) moved 6.0 ms a round out of adapt() and 7.5 into bg_step:
after O_DIRECT the reads were not the cost, the installs were (nine copies an expert, and per
swap a device read of the lut plus two device writes). It also held all 256 pending records --
1.8 GB of host memory, on GB10 the GPU's -- for ~4 rounds, and faulted ~250 fresh buffers. Not
kept. Instead adapt() changes the lut on a host copy and copies it back once, and reads and
installs in waves of 16 so the pool's buffers are reused; drop_cache() after a swap is skipped
when expert reads are direct (it opened all 141 shards to fadvise them).
