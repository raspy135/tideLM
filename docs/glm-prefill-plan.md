# GLM-5.3 prefill: the measured budget, what is done, and what is left

Status: two changes landed and measured with microbenchmarks (2026-10-23); the end-to-end engine run
is deliberately deferred to the end of the work. Two more items were dropped with measurements
rather than opinions. Two Sparks' serving profile: native EXL3, 12,000 slots, 256K FP8 split KV,
`GLM_CHUNK=4096`, `GLM_PREFILL_REDUCE=bf16`, `GLM_DENSE_CAUSAL=1`.

An 8,020-token prompt takes 18.5 s to the first token (`bench/bench.py`), 17.77 s of it in the
prefill itself. The question this answers: how much of that is a defect, and what is the cheapest
honest ordering of fixes.

## What the microbenchmarks say

Two harnesses, both runnable on one GPU with no engine:

```bash
# MoE: one layer's real experts from the rank pack, every tile the launcher instantiates
docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app \
  -e DSV41_EXL3_PACK=/packs/glm-exl3-experts-r0of2.bin --entrypoint python tidelm:local \
  /app/tools/bench_exl3_prefill.py --layer 20 --rows 4096 --experts 160 --topk 8 --iters 3 \
  --tiles "grouped,4x2x2,4x4x2,2x4x2,smem:4x1x8x2,smem:8x1x8x2,smem:8x1x8x3,smem:8x2x8x2,smem:4x1x16x2,smem:8x1x16x2,smem:16x1x8x2"

# attention: GLM's serving shapes, synthetic FP8 latent cache, no checkpoint
docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app --entrypoint python \
  tidelm:local /app/tools/bench_glm_prefill_attn.py --rows 1024 --keys 2048 --check
```

MoE, one layer, one 4,096-row chunk, this rank (top-8, 160 experts, ~205 rows an expert), ms:

| kernel setting | group | rot_in | gate_up | epi | down | combine | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grouped` (decode kernel) | 3.6 | 6.2 | 82.0 | 1.5 | 42.7 | 4.5 | 140.5 |
| `4x2x2` (tile) | 3.4 | 4.9 | 45.3 | 1.8 | 22.6 | 4.1 | 82.0 |
| `smem:4x1x8x2` (**shipped**) | 4.5 | 4.8 | 24.4 | 1.6 | 12.1 | 4.1 | **51.5** |
| `smem:4x1x16x2` | 3.1 | 4.4 | 24.1 | 1.5 | 12.6 | 4.1 | 49.8 |
| `smem:8x1x8x3` (PF 3) | 3.6 | 4.9 | 24.8 | 1.5 | 13.6 | 5.4 | 53.7 |
| `smem:8x2x8x2` (MTW 2) | 3.3 | 4.7 | 47.7 | 2.8 | 24.5 | 4.1 | 87.1 |

A single pass through that list is not enough to order the top two — the first setting in a
process runs several percent slow. Interleaved A/B/A/B on the same boot, 4 iterations each:

| setting | total, ms a layer |
| --- | ---: |
| `smem:4x1x8x2` (shipped) | 50.2, 49.0 |
| `smem:4x1x16x2` | 53.2, 52.8 |

So the shipped tile is the right one, the single-pass 3.3% was run order, and all of these settings
are bitwise equal to each other (the bench's `vs first` column reads `bitwise` for every `smem` and
`tile` variant; the 8.4e-05 it prints against `grouped` is that kernel's split-K sum order).

Attention, 1,024 rows x 2,048 keys x 64 heads (292 GFLOP):

| tile | ms | TFLOPS | note |
| --- | ---: | ---: | --- |
| `block_h 32, block_n 64, 8 warps, 1 stage` (**shipped**) | 8.27 | **35.3** | the fastest that fits |
| `16, 64, 8, 1` | 14.75 | 19.8 | bitwise equal, half the rows an mma |
| `16, 32, 4, 1` | 12.04 | 24.3 | 40 KB of smem, still 1 stage |
| `32, 128, 8, 1` | — | — | 102,400 B of smem against GB10's 101,376 |
| anything with `stages >= 2` | — | — | 177 KB: the tile is materialised in BF16 |

### Four things this settles

1. **The MoE GEMMs run at 34 TFLOPS, not 19.** `gate_up` 24.4 ms for 0.83 TFLOP and `down` 12.1 ms
   for 0.41 TFLOP. The 19 TFLOPS in the earlier roofline note came from dividing the whole MoE
   span by its FLOPs and charging the glue to the GEMM.
2. **29% of the MoE is glue, and it is bandwidth-wasteful by construction.** `rot_in` 4.8 +
   `down_combine` 4.1 + `group` 4.5 + `gateup_epi` 1.6 = 15.0 ms of 51.5. Two of those write
   ~800 MB each a layer: `rot_in` rotates every token's activations 8 times (once per pick) into
   `[T*8, 6144]` fp16, and `down_combine` round-trips `[T*8, 3072]` fp32 twice to reduce it.
3. **Neither prefetch depth nor the tile setting is the constraint.** `PF=3` (53.7) is slower than
   `PF=2` (54.9) at the same tile, `MTW=2` is nearly 2x worse, and an interleaved re-run puts the
   shipped tile ahead of the rest of the sweep. The cp.async rewrite buys less than the earlier
   version of this document assumed, and a config change buys nothing at all.
4. **Attention is near its floor for its kernel shape.** 35.3 TFLOPS, and the shipped tile is the
   fastest that fits in shared memory. `stages >= 2` is impossible only because the gathered tile
   is dequantized to BF16 before use — 80 KB a stage against 99 KB of budget. Keeping the staged
   tile in FP8 (40 KB a stage) would allow 2 stages or `block_n 128`, and that is the only cheap
   lever left in that kernel.

### Revised budget for the 8K prefill

| component | measured | at the rate the bench shows | floor at 60 TFLOPS |
| --- | ---: | --- | ---: |
| MoE (75 layers x 2 chunks) | 9.78 s | 34 TFLOPS GEMM (6.9 s) + 29% glue (2.8 s) | 3.1 s |
| attention, both chunks | 6.50 s | projections 65 TFLOPS (3.3 s) + sparse 35 (2.7 s) + 0.5 | 1.8 s |
| o_proj reduce | 0.53 s | tuned | — |
| embed, norms, head, MTP fill | 0.96 s | unmeasured residual | — |
| **total** | **17.77 s** | | **≈6-8 s** |

So the honest near-term target was first written as **13-15 s** (five bit-exact items), with
**11-12 s** if the expert GEMM went from 34 to 50 TFLOPS. The 8-9 s of the version before assumed
the GEMM was at 19 TFLOPS and the glue was free; both were wrong — and the 50-TFLOPS path was then
*also* measured wrong (2026-10-23, "Item 5, built and probed" below): the GEMM's halves already
overlap and its consume side is the floor. With items 2/3/5 landed (-0.64 s measured, -0.28 s
predicted from the ca microbenchmark) the bit-exact book is ~**17.9 s**, and item 4 (numerics,
KL eval) is the one remaining measured lever at 0.6-1.0 s.

## Done, with the number that settled each

**1. The MoE's down combine wrote a 402 MB buffer nobody reads.** `down_combine_kernel` kept its
per-pick values in shared memory (`part[32][32]`) and *also* stored them to a `y` buffer that the
caller allocated and never read (`tools/exl3_moe_cuda.py::_down` returned `local`, not `y`). The
store and the buffer are gone (kernel, C entry, argtypes, both call sites). `down_combine`
4.19/4.32 -> **2.06/2.08** ms a layer, and 402 MB of prefill transient per layer-chunk with it.

**2. `build_routing`'s `precount` path.** `engine/glm/exl3.py::routed_up` compacted the arena's
12,500 slot ids down to the ~160 present (`torch.unique`, 0.089 ms — rightly kept: on the full id
space this grouping costs **2.2 s** a layer) and then ran `build_routing` without `precount`, which
makes every one of its programs scan all P pairs. With `precount=True`: 3.63 -> **2.86** ms.

Verified partition-identical, not assumed: same NB (672), same 598 used blocks, every one of the
32,768 pairs placed exactly once, no block holding a foreign slot, none overfull. `block_pair`
differs in order only, and the kernel stores each member's output at
`rows_sh = row*topk + k` — its own pair index — so the order inside a block cannot move a result.
(The unused tail of `block_pair` is uninitialised and `ucount` is the upper bound, but
`block_slot`'s tail is `-1`, so those blocks return before reading it.)

**Together: 49.5 -> 46.5 ms a layer, -6%**, three repeats in one process: 46.37 / 46.79 / 47.59
against 49.0 / 50.2 for the same tile before.

### Same-day engine A/B, 8,020-token prefill

`tools/muse_pair_run.sh glm_prefill_prof.py --warmup --reps 4 --long-tokens 8000`, one boot an arm,
identical env apart from the two changes, `GLM_PREFIX_DISK=0` (otherwise reps 2+ restore from the disk
store and "prefill" in about a second) and both arms started from a copy of the served demand
history (`699f3726`). Raw logs: `results/glm-prefill-20261023-base/` (reverted) and
`results/glm-prefill-20261023-8k/` (applied).

| arm | mlp, s (reps 1-3) | attention, s (control) | prefill total, s |
| --- | --- | --- | --- |
| changes reverted | 10.23 / 10.31 / 10.61 (med 10.31) | 6.85 / 6.88 / 6.98 (med 6.88) | 18.77 / 18.88 / 19.18 (med 18.88) |
| changes applied | 9.85 / 9.88 / 9.83 (med 9.85) | 6.83 / 6.69 / 6.71 (med 6.71) | 18.40 / 18.24 / 18.13 (med 18.24) |
| delta | **-0.46 s (-4.5%)** | -0.17 s | **-0.64 s (-3.4%)** |

The microbenchmark predicted -0.45 s and the engine gave -0.46 s, which is the useful part: the
per-layer number transfers 1:1 once multiplied by the 150 layer-chunks, so the remaining items can
be costed from the bench alone. The untouched `attention` column moved -0.17 s between the two arms
and the arm that ran *later* was the slower one, so the noise floor on a component here is about
+-0.2 s and the `mlp` delta is ~2.3x that.

`bench/bench.py` on the same build (prose / code / fresh 8K, 3 runs after one warmup, temperature
0.6, seed 42, `--ignore-eos`, 12,500 residents): **prose 15.70 tok/s, code 20.73 tok/s, 8K TTFT
19.07 s**, against the README row's 15.2 / 20.3 / 18.5 measured on 2026-10-22 at 12,000 residents.
That instrument cannot resolve this change either — its 8K TTFT sits above the bare prefill (18.24 s
today) by the HTTP path and a different prompt salt, and the README's 18.5 s is a different boot.
Treat the 19.07 s as "the same within the spread", not as a regression; an A/B through `bench.py`
would need several boots a side (`docs/gotchas.md`, "one pair of boots is not enough").

## Dropped with evidence, so nobody spends the day on them

**Rotating each token once instead of once per pick — impossible.** It looked like 805 MB of
redundant writes (`rot_in` 4.8 ms), and it is not redundant: the EXL3 input scale is per *expert*
(`rot_in_kernel`: `suh = suh0 + e * K`, `e = pick[p]`), so a token's rotated input genuinely
differs for each of its 8 picks. That also means the buffer's size is inherent, and the item is
off the list for good.

**Pipelining the sparse-attention kernel by shrinking the tile — measured worse.** The only
configuration that could take a second stage is a smaller `BLOCK_N`, and small tiles lose badly
at these shapes (1,024 rows x 2,048 keys, 292 GFLOP):

| tile | ms | TFLOPS |
| --- | ---: | ---: |
| `32 x 64, 8 warps, 1 stage` (shipped) | 8.11 | **36.0** |
| `16 x 32, 8 warps, 2 stages` | 19.01 | 15.4 |
| `32 x 32, 8 warps, 1 stage` | 11.74 | 24.9 |
| `16 x 16, 8 warps, 3 stages` | 28.01 | 10.4 |
| `32 x 16, 8 warps, 3 stages` | 46.22 | 6.3 |

The per-tile costs (index gather, softmax, PV dot) need `BLOCK_N >= 64` to amortize, and at 64 keys
the BF16 tile is 80 KB of the 99 KB budget, so a second stage cannot fit. The one design that
could still win keeps `BLOCK_N = 64` and splits the dequantization into 128-column slices (the
`acc0`/`acc1` structure `_sparse_mla_fp4_kernel` already uses), so the live BF16 operand is 16 KB
a slice: 4 QK dots + 4 PV dots per tile, 2 stages. That changes the mma chain and the
online-softmax boundaries, so it is a numerics decision needing the KL eval, worth ~0.6-1.0 s. It
is the largest remaining item and it is *not* done.

## What is still on the list

| # | item | measured basis | expected | status |
| --- | --- | --- | ---: | --- |
| 1 | `rot_in` per token | — | 0 | impossible (per-expert `suh`) |
| 2 | routing `precount` | 3.63 -> 2.86 ms | 0.1 s | **done** |
| 3 | dead `y` write in `down_combine` | 4.3 -> 2.07 ms | 0.35 s | **done** |
| 4 | FP8-staged sparse attention (slices, 2 stages) | 36 -> ~55 TFLOPS | 0.6-1.0 s | not done, numerics eval |
| 5 | `cp.async` in the expert GEMM | -1.9 +- 0.3 ms a layer | **0.3 s, not 2.3 s** | **done — and the 2.3 s premise measured wrong (below)** |

## Item 5, built and probed (2026-10-23): the 34 -> 50 TFLOPS path does not exist in this kernel

The earlier version of this section inferred from the traffic/FLOP budget (5.5 GB = 27.5 ms, 1.24
TFLOP = 20.7 ms, measured 46.5) that **bytes and mma were not overlapping at all**. Two probe
kernels (`exl3m_prefill_smem_mode`, bench specs `nomma:` and `notrellis:`) measured the halves
directly, 4,096 rows x top-8, layer 20, this rank's shapes:

| arm | gate_up | down | GEMM total |
| --- | ---: | ---: | ---: |
| full kernel (`smem:4x1x8x2`) | 21.8 | 11.0 | 32.8 ms |
| `nomma:` trellis loads + decode + smem stores only | 9.95 | 5.26 | 15.2 ms |
| `notrellis:` A loads + mma on a constant tile | 19.75 | 9.08 | 28.8 ms |

32.8 << 15.2 + 28.8: **the halves already overlap**; the serial-sum reading of the budget was
wrong. The kernel's floor is the consume side — A loads + mma + the decoded-tile smem reads —
at 28.8 of the 32.8 ms, so perfect load staging could recover at most ~4 ms a layer, not ~19.
(The probes run at 48/96 registers against the real kernel's 128, so they flatter each half a
little; the conclusion survives that.) `ptxas` on the shipped tile: 128 registers = exactly 4
blocks/SM of 12 — the accumulators alone are 64 registers a thread, which is why every wider
tile (MTW=2 at 198 regs, NTB=16 at 224-240) loses on occupancy.

The `cp.async` staging was built anyway (`smem_tiles_ca`, `GLM_EXL3_PREFILL=ca:WxMTWxNTBxS`): raw
trellis words staged global -> shared S stages deep in place of the 2-deep register pipeline, same
decode order and mma chain, **bitwise equal to `smem` in every run and through the engine call
shape**. Interleaved A/B at 4,096 rows: smem 43.0-44.8, ca 41.2-42.8 ms a layer, S = 4/6/12 within
noise — **-1.9 +- 0.3 ms a layer (-4.3% of the MoE), ~0.28 s of the 8K prefill**. The example
profile now selects `ca:4x1x8x4`.

Also measured, so nobody retries them:

* **Prefetching the A operands one step ahead in registers** (into the 4 registers the removed pf
  pipeline freed): no change at all (ca 41.2-42.7 ms both ways). The per-step A loads are already
  hidden by L2 and the step's own mma slack. Removed.
* **`smem:8x1x4x2`** (KS = 2, half the steps and barriers) was already in
  `results/glm-tune/bench-exl3-prefill-l20.txt`: 69.5 ms at these shapes. Dead.

What a faster expert GEMM now requires is a structurally different consume — fewer decoded-tile
smem reads per mma or more blocks per SM — and the measured room is the gap from 32.8 ms to the
28.8 ms consume floor plus whatever a restructuring buys inside that floor. TensorFold's
producer/consumer split (0.74-0.85x, could not launch on GB10) remains rejected separately.

## Also rejected, for the record

* **The `cp.async` rewrite of the expert kernel**: built and measured 2026-10-23 — see "Item 5,
  built and probed" above. Kept (bit-identical, -1.9 ms a layer), but the 50-TFLOPS estimate died
  with the probes: the consume side, not the load side, is the floor.
* **A producer/consumer warp split.** TensorFold built it for this model family (patch 0330,
  `exl3_tc.cu`, bit-identical) and measured **0.74-0.85x** at 4,096-8,192 rows; GB10 could not
  launch its 17-warp configuration. Do not spend the day on it.
* **Experts reconstructed to FP16 and handed to cuBLAS**: ~19 GB a layer at 4,096 rows, ~87 ms at
  bandwidth against ~40 for the fused kernel (2026-10-17).
* **Bigger routing blocks on their own**: 64-row blocks bought nothing at 2,048 rows and 10% at
  4,096; 128 and 256 rows are slower, `MTW=2` is nearly 2x slower (measured again above).
* **`GLM_PF` prefill prefetch**: miss 33% -> 25%, TTFT 23.7 -> 32.8 s. Stays off.
* **Chunk 8,192**: reserved peak is already 107.6 GB at 4,096 and the watchdog floor is 3 GiB.
  Revisit only if the per-block trellis decode amortization starts to matter, which at 34 TFLOPS in
  the GEMM it does not.

## Measurement protocol for this work

The two harnesses are the whole loop: `bench_exl3_prefill.py` prints per-stage milliseconds and
`vs first` equality for every tile in one process (2 min including 160 real experts), and
`bench_glm_prefill_attn.py` sweeps attention tiles with `--check` against the einsum path. Change
code, re-run one of them, read the stage column that moved. No engine boot per arm, no peer skew in
the number.

One trap, found the hard way here: **the first setting a bench process runs is several percent
slow**, so a single pass through a config list mis-orders the top candidates (this document's first
version recommended a tile that a repeat showed to be 4 ms worse). Put the incumbent first and
interleave: `--tiles "smem:4x1x8x2,smem:4x1x16x2,smem:4x1x8x2,smem:4x1x16x2"`. The same applies
to the engine arms — `docs/gotchas.md`'s "one pair of boots is not enough" is the same effect with
more zeros on it.

## The deferred engine measurement

Run this **once, after the remaining items land**, not per change: a 0.45 s prefill change cannot be
resolved through a column that drifts 0.4 s between boots.

```bash
set -a; . ./.env.glm53-exl3; set +a
GLM_ENV="$(grep -E '^(GLM|DSV41)_[A-Z0-9_]+=' .env.glm53-exl3 | tr '\n' ' ')" GLM_SYNC=1 \
  tools/muse_pair_run.sh glm_prefill_prof.py --warmup --reps 4
```

`GLM_SYNC=1` is required: the boot guard compares a digest of `engine/glm/*.py` across the ranks,
and this work changed `engine/glm/exl3.py`. What to read: the `mlp` and `attention` component
columns against the 2026-10-20 record (MoE 9.78 s, attention 6.50 s), the median of the three
measured reps, and that the run completes at all — the prefill path only reaches the changed C ABI
at P > 64. If the MoE column lands anywhere in 9.3-9.8 s the two changes are invisible at that
granularity, as predicted; the microbenchmark above is the measurement.

**Closed 2026-10-23:** `GlmTP._boot_guard` digests `engine/glm/*.py` but did **not** pin
`tools/exl3_moe_cuda.cu`, which this work edits — two ranks with a stale rsync would run different
kernels and emit different tokens with nothing in the log. The guard now carries
`exl3_moe_cuda.source_digest()` the way `V41Engine` does (`x3_moe_cuda` key, try/except so an
MXFP4-only boot still starts), and pins `DSV41_EXL3_PREFILL_CA_TILE` beside the other prefill
kernel settings.

## Recording

Each item lands with its number in `docs/gotchas.md` (or beside the code if it is a setting), and
the negative results above — `PF=3`, `MTW=2`, `stages>=2` not fitting, the tile sweep landing
within 3.3% — are worth keeping precisely because they look plausible enough to be tried again.
