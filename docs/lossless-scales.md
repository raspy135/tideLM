# Lossless DeepSeek expert scales

`DSV41_SCALE_PACK` stores native MXFP4 expert scales in a compact, random-access
layout. Kernels reconstruct each original scale byte as they read it; weights
and arithmetic stay unchanged. This currently supports single-request native
FP4, output TP2, CUDA v2 decode, including the two-row down kernel and Triton
prefill. DSpark's draft experts keep their native layout.

Packing is **disabled** in the shipped template and private serving profile.
The slower serving runs below were the expert-load path, not the codec: every
packed load hashed both scale versions and read its record through the page
cache. With that fixed, a paired serving A/B measured packed at parity or better
(see "Why the first serving runs were slower"). Enabling it is a profile
decision; it needs the packs built first. Build the image, then pack on the head node:

```bash
docker run --rm --entrypoint python \
  -v /home/you/models:/models tidelm:local \
  /app/tools/pack_expert_scales.py \
  --model-dir /models/DeepSeek-V4.1-Flash-1017 \
  --out-dir /models/dsv41-scale-packs
```

This is a CPU job. It reads every expert's scales, verifies every byte survives
the round trip, and writes both ranks' packs. Allow about 25 GB of temporary
disk space while packing. Copy `ds41-scales-r1of2.bin` to the same directory on
the worker; the head needs `ds41-scales-r0of2.bin`. Keep the native checkpoint.
Set this in `.env.ds41-mxfp4`, using the **container** path under the model
directory's mounted parent:

```bash
DSV41_SCALE_PACK=/models/dsv41-scale-packs/ds41-scales-r{rank}of2.bin
```

Restart both ranks. The startup log reports the smaller bytes per slot and the
pack fingerprint. Both packs must come from the same packing run. Repack after
changing checkpoint weights; loading checks the checkpoint manifest, original
scale bytes and encoded record checksums. A pack from the original V4.1 weights
must not be used with 1017. Unset `DSV41_SCALE_PACK` to restore native scales.

On the full 1017 checkpoint, each rank's pack is 5.169 GB. Scale planes shrink
from 184,320 to 112,128 bytes; total expert slots shrink from 9,400,320 to
9,183,744 bytes (**2.30%**). Fixed strides use the worst exception count across
all experts and both ranks, keeping slots replaceable during hot reads and
adaptation. The earlier six-expert sample suggested 3.86% total savings; it was
not representative of that worst case.

This changes arena capacity, rather than producing free memory outside the
arena: with the same `ARENA_GB`, more experts can remain resident. Candidate
indexing is a separate compute optimization and does not add resident capacity.
Loading still reads the native scales (they arrive with the expert's one weight
read), then reads the pack record with O_DIRECT; do not expect an NVMe traffic
reduction. A record's native-byte and record checksums run on its first load in a
process; later loads of the same expert skip them.

Qualification: `engine/test_scale_layout.py`, `engine/test_scale_pack.py`, and
`DSV41_RUN_CUDA_TESTS=1 python tools/test_scale_pack_cuda.py`. GPU tests compare
native and packed up/down intermediates and final outputs bytewise, including
replacement, captured replay, both TP slices and scaled/unscaled prefill.

The full-engine fixed-residency check also matches every output token across
native/packed arms and both ranks: 10,208 residents, hot reads/adaptation/prefix
reuse off, depth 3, 128 greedy output tokens, one warmup and two measured repeats
each for prose and code. Raw outputs and timings are in
`results/scale-pack-20261021/{native,packed}-rank{0,1}.json`. These short separate
boots establish output identity; they do not establish a speed gain.

| Workload | Native median ms/step | Packed median ms/step |
| --- | ---: | ---: |
| Prose | 74.246 | 75.828 |
| Code | 74.014 | 73.797 |

At the same 97.4 GB budget, normal adaptive serving boots with **10,452**
residents instead of **10,208**. Its `bench/bench.py` reports are kept as
`results/scale-pack-20261021/http-*.json`: three measured runs after one warmup,
temperature 0.6, seed 42, 32K context allocation, 1024 output tokens for
prose/code and 256 for random 8K. These use a different prompt salt and resident
set from the earlier README run; compare kernel overhead with the fixed-output
test above. The README's DS chart retains the measured native-scale profile,
with candidate indexing on. Unsetting the pack path restores native scales.

The optional packed-profile run measured 23.0142 tok/s prose, 39.4160 tok/s
code, 29.8773 tok/s random 8K, and 9.016 s 8K TTFT. These numbers do not improve
the earlier native-scale measurements; retain packing for its resident capacity
rather than claiming a throughput gain. The packing audit took about 263 seconds plus final pack
writes on the development head node.

A compression-on repeat used the original chart's prompt salts
(`flare-perf-http-20261021-prose` / `-code`), temperature 0.6, sampling seed 42,
top-p 0.95, 1024 output tokens, one warmup and three measured requests:

| Workload | Packed median tok/s | Median TTFT |
| --- | ---: | ---: |
| Prose | 23.6656 | 1.393 s |
| Code | 41.3714 | 1.833 s |

It held 10,452 experts at 97.4 GB, with adaptive placement and hot reads on,
32K context allocation. These are on-only measurements, not a fresh paired
regression estimate against the historical 25.9777 / 44.2217 tok/s native runs.
The interrupted native arm's partial logs are retained but do not constitute
the planned three-run comparison. Reports and exact settings are in
`results/scale-pack-20261021/matched-http/`.

## Why the first serving runs were slower (2026-10-22)

**Not the kernels.** `tools/bench_scale_pack_kernels.py` (48 real experts, rank-0
shapes, L2-cold graph replays, outputs compared bytewise) times the packed decode
kernels *faster* than native: up -3.8 to -4.5%, two-row down -2.2 to -5.7%,
one-row down -2.0 to +0.2% at 1/2/4 tokens. Only 0.05% of those experts' scales
take the exception path. gpt's templating left the native kernels unchanged:
native timings before and after the commit agree within 3%.

**The load path.** Every packed `load_slot` hashed the native scale planes
(553 KB, 0.23 ms), read the record through the page cache (0.89 ms cold for
336 KB) and hashed it (0.14 ms): 1.26-1.52 ms a load against native's 0.32.
Serving reads 2-5 GB of experts a request (adaptation swaps, hot reads), 250-500
loads. Now the record is read with
O_DIRECT through a per-thread aligned bounce buffer (0.20 ms) and each record's
checks run once per process: 0.83-0.93 ms a load in steady state.

**Paired serving A/B**, same image, dev-mounted source, seed history `45ef7129`,
the chart's prompt salts, 32K, 97.4 GB, `auto` residents (native 10,208, packed
10,452), three measured runs after a warmup:

| Workload | Native | Packed + fix | Packed, before the fix (above) |
| --- | ---: | ---: | ---: |
| Prose | 25.61 tok/s | **25.82** | 23.67 |
| Code | 40.62 tok/s | **44.81** | 41.37 |

The prose runs before and after the fix have identical acceptance (1.99 / 1.97 /
1.98) and identical NVMe traffic (4.65 / 3.77 / 2.85 GB): the same text and the
same loads, 9.1% faster. Against native, prose acceptance was equal and packed
was 0.8% faster; code acceptance differed between the arms (native 3.32-3.77,
packed 3.53-4.18), so its +10% is not a packing measurement. Prose TTFT 1.22 s
native, 1.30 packed. Reports: `results/scale-pack-ab-20261022/`.
