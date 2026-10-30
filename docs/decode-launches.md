# Decode kernel launches

`DSV41_DECODE_LEAN=1` (default) cuts the kernels a TP2 decode step launches from **8,457 to
4,115**, bit-identically: same logits path, same token hashes on every measured workload. It is in
the EP2 boot guard with its two sub-switches, `DSV41_DECODE_LEAN_ROPE` and
`DSV41_DECODE_LEAN_SOFTMAX`. `0` on both ranks restores the previous spelling.

## Why launches

A graphed decode step was 8,457 kernels (2026-09-30, current build, python prompt), and
7,600 of them were torch elementwise/copy/reduce kernels of 1-8 us. TensorFold's GLM-5.3 round on
the same pair of Sparks is 1,853 kernels (its `docs/THEORY-2.md`), about 41 per layer against our
180. Inside a CUDA graph a tiny kernel still costs its own duration (~1-2 us minimum on GB10) plus
a sub-microsecond gap, so the cost is per kernel, not per byte. Measured here: the first round
removed 2,072 kernels and 4.6 ms of kernel time, **~2.2 us per kernel removed**.

## How to see a layer's program

`tools/bench_decode_kernels_tp.py` (run through `run_two_node_gate.sh` with a
`GATE_SOURCE_ROOT` snapshot) generates three greedy workloads twice, hashes the output token ids,
then profiles 8 bursts and writes the ordered kernel sequence. `tools/decode_layer_program.py
<dir>/kernels-seq-rank0.tsv` prints one layer: everything between two consecutive routed-expert
`moe_down` kernels. An ordinary layer went 180 -> 134 -> 80 kernels; compressor/indexer layers
are 30-120 more.

## What changed, and why each is exact

The rule: every op whose result depends on reduction order or on the row count stays the same
torch op on the same shape. Only what surrounds it moves.

| Site | Before | After | How |
| --- | ---: | ---: | --- |
| rmsnorm (104 / step) | 12 | 4 | static zero-padded [16, D] fp32 buffer; torch `square().mean()` on it; one kernel for `+eps`, `rsqrt`, the two multiplies and the cast; fp32 weight made once |
| HC coefficients (86 / step) | 15 (+3 copies) | 6 | same padded buffer feeds the padded cuBLAS GEMM (`torch.mm(..., out=)`) and the padded mean; `rsqrt` and `mixes * rsqrt` move into the Sinkhorn kernel, which writes the layer's static buffers |
| hc_post / hc_pre+norm | + copies | in place | `hc_post(out=self.h)` (a program reads its tile's residual streams before writing that tile), `hc_pre_rmsnorm(out=self.y)` direct bf16 store |
| RoPE (~4 / layer) | 4-5 | 1 | one kernel; torch's complex multiply is `re = fma(a, c, -(b*d))`, `im = fma(b, c, a*d)` in this build (below) |
| router | ~23 | ~10 | padded gate GEMM from a static buffer; `softplus`/`sqrt`/`+bias`/prune mask as one kernel; `topk(out=route_idx)`; torch gather + sum; `+1e-20`/divide/scale as one kernel; `index_select(out=slots)` |
| attention softmax | ~12 | 3 | `*scale`, mask, `amax` (exact in any order), `clamp`, `exp` in one kernel; torch `sum`; sink term and division in one kernel |
| attention keys | cat + 2x `.float()` | 1-2 copies | written once into an fp32 buffer |
| shared expert | 7 elementwise | 1 | SwiGLU tail: torch's silu is `x / (1 + exp(-x))` with libdevice `exp` and an IEEE division |
| routed + shared | `.float()`, `+=`, `.to(bf16)` | 0 | `hc_post(x2=)` forms `(routed + shared.float()).to(bf16)` in-kernel |
| per-step index math | per layer | per step | window slots/mask, `freqs[pos]`, compress lengths and the indexer's clamp/mask memoized per captured graph key |

`tools/test_decode_lean.py` (14 tests, real layer-3 weights from `MODEL_DIR`) checks each against
the torch spelling with `torch.equal`, eager and under graph replay.

## Two traps for anyone writing the next one

- **Triton flushes denormals in libdevice by default** (`enable_reflect_ftz=True`); torch's nvcc
  build does not. The first router and softmax kernels differed by 1.2e-38: `exp` of a very
  negative logit is a denormal in torch and 0 in Triton, and `sqrt(softplus(x))` turns that into
  1e-19 vs 0. Every lean kernel that calls libdevice launches with `enable_reflect_ftz=False`
  (and `enable_fp_fusion=False`). Triton's plain `/`, `tl.exp` and `tl.sqrt` are approximate; use
  `libdevice.div_rn`, `libdevice.exp`, `libdevice.sqrt_rn`.
- **FMA contraction is part of torch's numerics.** Checking RoPE at bf16 output accepted three
  contractions of the imaginary part, because bf16 rounding hides fp32 differences. At fp32, only
  one of nine matches: 0 of 4.2M products differ, every other pairing 24-33 %
  (`test_rope_fp32_contraction`). Compare before the final rounding.

## Measured (TP2 pair, frozen ranking, greedy, 256 tokens)

Arms alternate; run-to-run spread of one build is 8-10 %, so single runs decide nothing.

Round 2 (everything above), order lean, off, lean, off:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step | kernel ms / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| lean | 43.37 | 44.99 | 22.50 | 4,115 | 119.6 |
| off | 39.61 | 43.29 | 21.45 | 8,457 | 123.8 |
| lean | 45.14 | 46.66 | 23.20 | 4,115 | 114.1 |
| off | 41.83 | 43.32 | 21.63 | 8,457 | 123.2 |
| **mean change** | **+8.7 %** | **+5.8 %** | **+6.1 %** | **-51 %** | |

Per step that is -9.1 ms (html), -6.2 ms (python), -5.8 ms (explain), from the arms' mean
ms/step (104.8 vs 113.9, 107.1 vs 113.2, 95.4 vs 101.2). Round 1 alone (rmsnorm, HC, buffers;
6,385 kernels) measured 42.23 / 42.08 html against 40.30 off. The profiled `kernel ms / step`
includes profiler overhead and is only comparable between arms.

Token hashes (`df46ab5e`, `73a30f4e`, `285bf77a`) are identical in every arm. Results:
`results/kernels-20260930/`.

## Round 3: shared collectives (`DSV41_DECODE_LEAN_MOE`)

Output-layout TP gathered the routed intermediate [P, 1152] and the shared-expert intermediate
[T, 1152] separately, then the routed and shared outputs separately: four all-gathers and four
reorder copies per layer. Both pairs have the same column split, so `_moe_merged` stacks the two
intermediates into one gather, sums the two outputs locally in fp32 before one gather
(`routed + shared.float()` is elementwise, so the order of sum and gather does not matter), and
`hc_post(split_w=2560)` reads the gathered [2, T, 2560] as is. The native routing prologue and
epilogue (null-slot keys, block remap) became two kernels. `tools/test_decode_lean.py` covers the
pieces; the two-node run covers the collectives:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step | NCCL / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| merged | 43.99 | 45.42 | 22.69 | 3,715 | 170 (7.9 ms) |
| separate | 41.70 | 44.72 | 22.18 | 4,115 | 250 (8.9 ms) |
| merged | 43.90 | 45.22 | 22.77 | 3,715 | |
| separate | 43.62 | 45.20 | 22.55 | 4,115 | |

Token hashes identical. The gain is ~1 ms per step and within the noise of the second pair:
80 fewer collectives saved only ~1 ms of NCCL time, because the remaining ones wait longer. The
collective cost here is latency and rank skew, not bytes -- merging more of them will not buy much.

## Attention staging (`DSV41_DECODE_LEAN_ATTN`): exact, fewer launches, no measurable gain

`_attention_lean` has RoPE emit q already widened to fp32, gathers the window and compressed keys
straight into the fp32 key block (`keys_f32`: the packed-KV decode reproduced bit for bit,
negative zero included), and has the output RoPE round the einsum result itself. 158 fewer
kernels per step, same token hashes -- and no speed change:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step |
| --- | ---: | ---: | ---: | ---: |
| on | 44.42 | 46.01 | 22.81 | 3,557 |
| off | 45.18 | 46.83 | 23.16 | 3,715 |
| on | 45.92 | 47.44 | 23.67 | 3,557 |
| off | 44.66 | 46.92 | 23.26 | 3,715 |

The removed kernels were copies, and writing the gathered keys as fp32 moves the same bytes the
copies did. Kept on because it is exact and costs nothing; do not count it as a speedup.

## Routed-expert kernels: v2 (`DSV41_FP4_CUDA_V2`, default on)

Distinct routed experts per layer per verify step (`DSV41_ROUTE_STATS=1`, 256 greedy tokens):
python 21.7, html 20.3, explain 16.5 (the last mostly at depth 3). At TP2 a slot is 6.27 MB for
`up` (w1, w3, scales) and 3.13 MB for `down`, so python reads ~136 + 68 MB per layer.

`tools/bench_fp4_moe_decode.py` (one GPU, L2-cold calls, graph replay) put v1 at 150-185 GB/s at
these U against a 205-225 GB/s plain read of the same bytes; below U = 14 it fell to 80-160. v1
gives each lane 2 bytes of a 64-byte warp tile per K step and walks the row again for each routed
pair. v2 gives each lane whole 32-value scale groups (one 16-byte load per matrix), keeps the row
in registers across the pairs, and replays v1's relaxed reduction exactly: each group's 8 virtual
lane partials with v1's fma pattern and shuffle tree, then each subgroup's running sum as a
sequential chain through shared memory. `tools/test_fp4_moe_v2.py`: bit-identical on TP and
single-box shapes, T = 4 and 6, U = 6..36, with and without null-slot pairs. (Whether v1's
`p *= scale; acc += p` was contracted into an fma does not matter: UE8M0 scales are powers of two.)

| U | up v1 -> v2 (us) | GB/s | down v1 -> v2 (us) | GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 554 -> 409 | 159 -> 214 | 312 -> 299 | 140 -> 147 |
| 20 | 680 -> 599 | 184 -> 209 | 364 -> 303 | 172 -> 207 |
| 26 | 820 -> 760 | 199 -> 214 | 417 -> 391 | 195 -> 209 |

In the engine (alternating, token hashes identical):

| Arm | html tok/s | python tok/s | explain tok/s | up / down per layer |
| --- | ---: | ---: | ---: | --- |
| v2 | 45.87 | 48.22 | 23.99 | 709 / 355 us |
| v1 | 44.55 | 45.98 | 22.98 | 842 / 380 us |
| v2 | 46.75 | 48.18 | 24.00 | 730 / 355 us |
| v1 | 45.88 | 47.68 | 23.73 | 770 / 360 us |

+2.4 to +2.9 % (-2.5 to -3 ms per step), less than the microbenchmark promised, and the reason is
the ceiling: in the engine the shared expert's w1||w3 (~12 MB) streams on the side stream during
`moe_up`, so `moe_up`'s window moves ~148 MB in ~710 us, ~208 GB/s. **The routed experts are now
at the DRAM ceiling in situ.** What is left of their ~46 ms per step is bytes -- ~22 distinct
experts x 9.4 MB x 40 layers -- and only fewer or smaller expert reads will move it, not a kernel.
v2's own weak spot, many pairs on few experts (U < 14), is not where serving runs; batching the
chain replay across pairs would fix it and was not done.

## What is left in a layer (~66 kernels)

- the kept cores: 2 x (padded GEMM + split-K reduce + square + mean) for HC, square + mean per
  norm, topk and the k-way sum in the router, the softmax sum;
- three all-gathers (MoE intermediates, MoE outputs, and the `wo_b` input/output pair), two
  reorder copies;
- the matmuls themselves.

Further exact trims exist -- the ring write folded into the kv RoPE, the HC and router padding
copies written by the kernel before them, `hc_post` reading the `wo_b` gather in place -- and add
up to well under 1 ms per step. The time is elsewhere now (per step, python prompt, profiled):
routed experts ~50 ms (the native `moe_up`/`moe_down` at an estimated 165-180 GB/s against a
~235 GB/s streaming ceiling), dense fp8 projections ~30 ms, fp32 GEMMs ~10 ms (HC, router gate,
torch attention), the bf16 LM head ~7.7 ms (read twice per step), NCCL ~7.9 ms.

## Dense FP4 (`DSV41_DENSE_FP4`): a starved kernel, fixed with split-K

Measured 2026-10-01 on the round-3/v2 build, TP2, native FP4 experts, `DSV41_TP_ATTN=1`,
`DSV41_DRAFT_HEAD_FMT=off` (to isolate the variable), greedy, 256 tokens.

### The first A/B was flat

Six `bench_decode_kernels_tp.py` gate runs before the fix, arms alternating:

| dense_fp4 | run | python tok/s | python ms/step | html tok/s | explain tok/s | profiled dense ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1 | 49.2 | 101.6 | 44.8 | 22.1 | 29.5 |
| attn,wo_a | 1 | 48.8 | 102.4 | 45.4 | 22.1 | 30.2 |
| attn,wo_a | 2 | 48.6 | 102.8 | 45.5 | 22.1 | 29.5 |
| off | 2 | 48.4 | 103.4 | 43.3 | 21.5 | 30.9 |

The profiled dense family stayed at ~30 ms/step even though the fp4 weights are ~half the bytes.

### Why: the kernel is starved on narrow N

L2-cold achieved bandwidth at M=6, real weights (`tools/test_fp4_linear.py` method):

| weight | N x K | fp8 GB/s | fp4 GB/s | fp4 time / fp8 |
| --- | --- | ---: | ---: | ---: |
| wq_b | 32768x1280 | 224-231 | 199-217 | 0.62 |
| shared w2 | 5120x2304 | 214-218 | 162-174 | 0.67-0.70 |
| wo_b | 5120x8192 | 197 | 105-111 | 0.94-1.00 |
| shared w1 | 2304x5120 | 160-204 | 113 | 0.96 |
| wq_a | 1280x5120 | 185-190 | 67 | 1.47 |
| wkv | 512x5120 | 120-131 | 33 | 1.81-2.11 |

Wide N reaches the fp8 kernel's bandwidth (the 0.53 byte ratio delivered). Narrow N collapses to
33-67 GB/s: `BLOCK_N=32` gives `wkv` 16 CTAs for 48 SMs with a 40-iteration serial K loop of 8
dependent dots, and a sweep of `BLOCK_N` x `num_warps` x `num_stages` does not lift it. That is a
parallelism bug, not a property of 4 bits. `attn` is mostly the narrow shapes, so before the fix its
wide-N win cancelled their loss; `wo_a`'s grouped kernel has a G=8 axis and was already fine.

### Fix: split-K with a fixed-order combine

`DSV41_FP4_DENSE_SPLIT` (`auto`, default; an integer forces S; 1 restores the unsplit kernel). At
M <= 16 `pick_split_k` targets ~8 CTAs per SM, capped at 8 splits and at KQ/4 so each slice keeps
real work. `_fp4_linear_split_kernel` partitions the K quads into contiguous slices
`[s*KQ/S, (s+1)*KQ/S)` and writes fp32 partials; `_fp4_combine_kernel` adds the planes in split
order. Splitting changes the fp32 summation order, so a decode call and a >16-row prefill call can
differ in the last bits -- but `pick_split_k` is constant over M <= 16, so the invariance the engine
relies on (sequential M=1 vs the M=6 verify block) holds. `tools/test_fp4_linear.py` checks
split-vs-unsplit (~1e-7 relative, fp32 re-association only) and decode row invariance.

Bandwidth after (best S, L2-cold, 3 reps each):

| weight | fp8 GB/s | fp4 before | fp4 split-K | time fp4/fp8 |
| --- | ---: | ---: | ---: | ---: |
| wkv | 125 | 35 | 61 | 1.09 |
| wq_a | 157-190 | 66 | 143 | 0.58 |
| shared w1 | 205 | 113 | 175 | 0.62 |
| shared w2 | 218 | 162-174 | 171 | 0.68 |
| wo_b | 197 | 105-111 | 158 | 0.66 |
| wq_b | 229 | 199-217 | 205-210 | 0.58 |

Every slim shape is now 1.5-1.7x faster than fp8; `wkv` (N=512) only reaches parity and is the one
shape a wider split would need to fix.

### End to end (`attn,wo_a`), off run bracketed between the two fp4 runs

| arm | python ms/step | html ms/step | explain ms/step | dense ms/step | kernels/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 106.1 | 103.9 | 91.9 | 30.6 | 3536 |
| attn,wo_a | 97.6 | 95.3 | 86.1 | 24.6 | 3735 |
| attn,wo_a | 99.9 | 97.5 | 87.4 | 25.6 | 3735 |
| all | 98.4 | 96.3 | 87.3 | 20.6 | 4565 |
| shared | 102.8 | 101.1 | 91.8 | 28.4 | 4366 |

`attn,wo_a` is ~5-8 ms/step faster (~7 %) than the contemporaneous `off`; the pre-fix fp4 arms were
at parity (python 102.4/102.8 ms). `all` drives the dense family lower (20.6 ms) but adds ~800
combine launches, which land in the profiler's "small" family and cancel the gain, and the shared
experts are the quality-sensitive group. `attn,wo_a` is the setting worth having; the quality caveat
of the nesting-depth probe still applies, so the default stays `off`.

### Two code fixes it needed

- `DSV41_DENSE_FP4` could not even load under TP attention before this: `shard_attention` called
  `.shard()` on `wq_b`/`wo_b`, which `FP4Weight` did not implement, and it had no
  `FP4GroupedWeight` branch for `wo_a`. Added (`tools/fp4_linear.py`, `engine/tensor_parallel.py`).
- `bench_decode_kernels_tp.py` gained the missing `os.makedirs(args.out, exist_ok=True)`: the gate
  creates `GATE_LOG_DIR` on rank 0 only, so a fresh results directory made rank 1 fail on the final
  write after every measurement had already run.

## The decode all-gather goes over RoCE (`DSV41_COMM_BACKEND=roce`): −5.5 ms/step, bits identical

The step's ~170 decode-sized all-gathers were NCCL. Captured in a CUDA graph on this pair NCCL's
all-gather costs 44-70 us whatever the payload, because its kernel hands the send to NCCL's network
proxy thread; tools/bench_roce_gather.py measures a one-shot host-staged RoCE all-gather at 12.7 (16
KiB) / 15.5 (48 KiB) / 19.9 (96 KiB) us, bits equal. `tools/roce/` is that transport (ported from
TensorFold 0230 / b12x "RoCEnante", Apache-2.0); engine/comm.py routes a contiguous decode shard of
at most DSV41_ROCE_MAX_KB through it and everything else through torch.distributed, and comm.check()
turns a run-time timeout into a diagnosis.

Gate A/B, image 136c549d, off/on arms, all other settings identical (dense_fp4=attn,wo_a, hc_kernel=0,
draft_head_fmt=off). **Token hashes are identical in every arm** (`bad8285e` html, `91d2b6e0` python,
`ee1b74b4` explain) -- the transport moves the same bytes:

| arm | html ms/step (tok/s) | python | explain |
| --- | ---: | ---: | ---: |
| nccl (2 runs) | 94.6 / 96.3 (46.1) | 96.3 / 96.3 (50.9) | 86.9 / 86.9 (23.3) |
| roce (2 runs) | 91.7 / 88.1 (48.9) | 93.1 / 88.5 (54.1) | 83.9 / 79.6 (24.8) |
| roce + `DSV41_L2PF_MB=2` | 89.4 (49.2) | 88.8 (55.3) | 80.6 (25.1) |

- **roce vs nccl: -5.5 ms/step, +6.2 % tok/s** (5.15-5.55 ms across the three workloads), the
  projection from the microbench.
- **L2PF on top: another ~0.5-1.1 ms.** `DSV41_L2PF_MB=2` reads the next layer's weights on a side
  stream during the MoE all-gather; the 2 MB budget is what fits the ~15 us RoCE window
  (tools/bench_l2_prefetch.py has the rule -- more than the window and the join delays the read).
  It competes with RoCE for the same window, so it is smaller than it was on NCCL.
- Results: `results/roce-20261001/`. Off switch: `DSV41_COMM_BACKEND=nccl`.

Two bugs the run found, both in the wiring rather than the transport: the peer's v41_ref.py has to
carry the boot-guard keys (a stale tree crashes rank 1 on a missing attribute and rank 0 as a
connection reset, which reads like a fabric failure), and `import roce` resolves `tools/roce/` as a
namespace package unless `tools/roce` itself is on sys.path.
