# Paged context caches (Rust DeepSeek engine)

The compressed caches a lane holds (each kv-source layer's packed rows and index keys, 1,360 B a
token) used to be allocated per lane at `MAX_SEQ`, plus a scratch lane that only ever held filler
rows and probe excerpts. Two 512K lanes reserved 3 x 713 MB a rank whatever the requests used, and
one lane could not reach 1M without every lane paying for 1M.

They now come out of one pool of physical chunks (`rust/crates/tidelm-engine/src/kv.rs`), mapped
into the lanes' address ranges as the lanes grow.

## How

CUDA virtual memory (`rust/crates/tidelm-cuda/src/vmm.rs`, the driver's cuMem* calls through
`cudaGetDriverEntryPointByVersion`). Each cache keeps the layout every kernel addresses,
`[lane slots][rows][width]` with lanes a fixed stride apart, but as a reserved address range: a
lane's part is backed by real 2 MB chunks as far as its rows reach, and everything else - every
lane's tail, the scratch lane past its filler rows - maps one shared dummy chunk. The indexer
scores a power-of-two bucket of keys and masks the ones past each row's count, so reads past a
lane's end are normal; they read the dummy and never fault.

Why not page tables in the kernels (vLLM's PagedAttention): six kernels address these caches, and
an indirection in each would have to keep every one bit-identical and as fast. With the GPU's own
page tables nothing in the kernels or their CUDA graphs changes. A chunk is coarser than the 4K
tokens a page discussed beforehand - 7,280 tokens of a ratio-1 layer's packed rows, 16,383 of a
ratio-2 layer's keys - at no cost the decode step can see.

`Engine::prepare` maps the chunks a step writes into before its device work (both ranks run the
same steps, so their maps never differ); a disk-tier restore maps before it uploads. Mapping waits
for the device first, since a kernel in flight may be reading the range: once per ~7K tokens a
cache, measured at 35-110 us a chunk on GB10.

## Memory

| Config (a rank) | Fixed lanes | Paged |
| --- | ---: | ---: |
| 2 lanes x 512K (the shipped profile) | 2.14 GB | 1.48 GB (704 chunks) |
| `MAX_SEQ=1048576`, `RUST_KV_TOKENS=1572864` | 4.28 GB | 2.19 GB (1,044 chunks) |

Boot memory with the shipped profile: 103.22 -> 102.60 GiB device a rank. The second row lets one
lane reach 1M while the other holds 512K, for ~50 MB more than the shipped profile's caches (RoPE
tables for 1M positions add 0.27 GB on top).

## When the pool is overcommitted

`RUST_KV_TOKENS` under `RUST_LANES x MAX_SEQ` (it must be at least `MAX_SEQ`, so one lane always
fits, and the prefix disk tier must be on). The scheduler (`decode.rs`, `kv_room`):

1. admits a queued request, or brings a parked one back, only if its rows fit and every running
   request can still grow 4,096 tokens (`KV_HEADROOM`);
2. before a step whose pages are not free, takes idle lanes' cached prefixes back, largest first;
3. if that is not enough, swaps the newest request out: a snapshot at its exact length, a disk
   bundle of its rows, its chunks back to the pool, the request parked (vLLM's preemption order,
   but kept rather than recomputed: recomputing a 1M-token prompt is ~10 minutes of prefill);
4. brings it back once it fits, from the bundle, with its pending token: that first step verifies
   the token alone, which seeds the drafter (a restore brings no drafter state).

Rejected: taking pages from the expert arena when the pool runs out - it would trade quality away
silently.

The headroom rule is measured, not assumed: without it a swapped-out request saw its own freed
chunks as room, came straight back, made one token and was swapped out again - six cycles in one
run (two requests, 26K and 57K tokens, a pool of 65,536).

## Measured

One box, fixed residents (`--keep-first 100`, no adaptation, no hot reads, so two boots emit the
same tokens), greedy:

- **Bit-identity, default pool** (two lanes of 64K): prose, code, a 21,845-token prompt and its
  prefix-cache restore (20,480 cached) gave the same text on the paged build twice and on the
  build before it. The one exception was that build's own first boot on the long prompt (one token,
  `-` against `*`): the unpaged build is not boot-to-boot identical there either.
- **Swapping is exact** (two lanes of 64K, a pool of 65,536 tokens, plain decoding so a step is at
  most two rows and its bits do not depend on what shares it): request A (26,294 prompt tokens,
  6,000 out) and B (48,608, 4,000 out) overlapped; B was swapped out once at 51,337 tokens (39
  chunks back) and brought back once A ended. Both texts are the ones A and B give run one after
  the other: all 6,000 and 4,000 tokens, B's ~1,270 after its restore included.
- With speculative decoding the overlapped and the one-at-a-time runs differ for reasons that
  predate paging: a step above 16 rows (B's copy-drafted blocks are ~27) runs other kernels, so a
  lane's tokens depend on what shares its steps. The same overlap without any swap diverged at the
  same character as the run with one.
