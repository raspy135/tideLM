# Lossless DeepSeek expert scales

`DSV41_SCALE_PACK` stores native MXFP4 expert scales in a compact, random-access
layout. Kernels reconstruct each original scale byte as they read it; weights
and arithmetic stay unchanged. This currently supports single-request native
FP4, output TP2, CUDA v2 decode, including the two-row down kernel and Triton
prefill. DSpark's draft experts keep their native layout.

Build the image first, then pack the checkpoint once on the head node:

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
Loading still reads native scales to validate their bytes, then reads the pack;
do not expect an NVMe traffic reduction.

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
test above. The README's DS chart remains the default native-scale profile,
with candidate indexing on; scale packing is optional.
