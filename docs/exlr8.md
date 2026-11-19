# exlr8 on the pair: GLM-5.3 `bit260` (2026-11-11)

[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4):
every routed expert stored at 2, 3 and 4 bits in independent 128-channel units, a profile
(manifest) naming one width a unit; the trellis is EXL3's with one change - every codebook value
is rounded to FP8 E4M3, and the encoder searched over the rounded values. `bit260` is the profile
served here: gate / up at K2 in 45 layers and K3 in 30, every down at K3, the shared expert K4.

## What runs

Nothing in the engine's structure changed. The units regroup, tile for tile, into the records
and matrices the EXL3 kernels already read; what is new is the codebook.

| piece | where |
| --- | --- |
| the codebook: mcg, then the E4M3 rounding - exact in 5 instructions a pair (`CB_MCG_E4M3`, 3) or in 2 (`CB_MCG_E4M3_FAST`, 4) | `tools/exl3_moe_cuda.cu` `cb_pair`, `e4m3_round2`; the prefill unpack `glm_exl3_unpack_cb` |
| fetch a profile by HTTP Range into a sparse mirror (bit260: 237 of 838 GB) | `tools/exlr8_fetch.py` |
| the routed experts as an EXL3-layout pack, a rank (120.4 GB) | `tools/glm_pack_exlr8.py`; `arena.rs` takes codebook `mcg-e4m3` |
| exlr8's 4-bit o_proj, q_b, dense MLPs and shared experts as an EXL3-layout checkpoint | `tools/glm_exlr8_dense.py`; `weights.rs` takes the codebook from a `.mcg_e4m3` marker |
| reference decode, checks | `tools/exlr8_ref.py`, `exlr8_check.py`, `test_glm_pack_exlr8.py`, `exlr8_fixture.py`, `glm_gpu.rs` `exlr8_codebook_matches_the_reference` |

Left as the EXL3 checkpoint's (bertholomus 3.0bpw): what exlr8 keeps at 8 bits - q_a, kv_a, the
indexer's wq_b (MXFP8), kv_b, the embedding and the head (FP8) - and five int4 matrices. At 5-6
bits they are fewer bytes a pass than exlr8's; a pure-exlr8 engine would be slower there.

A profile is a boot-time choice: one pack a profile, from the same mirror. `GLM_SLOTS` is a memory
budget (3-bit records), so a profile of smaller records holds more experts in it.

## Checks

- 36 units' xxhash64 against the index; the tiles against `tools/exl3_ref.py` plus torch's
  `float8_e4m3fn` cast for all 65,536 states; the three widths against each other (K3 / K4
  correlation 0.987, K2 / K4 0.955).
- The kernel's rounding against the reference on every fp16 value in [-4, 4] (CPU emulation), and
  the decode GEMV on a real record: relative error 2.1e-4 (the same bytes as plain mcg: 2.7e-2).
- Pack records and converted dense matrices decode to the reference; the dense ones correlate
  0.99 with the EXL3 checkpoint's decode of the same weights.

## Serving it

`configs/glm53-exlr8.env.example` was the profile (dropped 2026-11-17 for `home`, below, which was
dropped 2026-11-18 for `bit322`, further below: `configs/glm53-exlr8-bit322.env.example`), and the
README's quick start the setup: fetch
(`tools/exlr8_fetch.py`), pack a rank (`tools/glm_pack_exlr8.py`), write the dense checkpoint
(`tools/glm_exlr8_dense.py`). Against the EXL3 profile three keys differ - `GLM_EXL3_DIR`,
`GLM_EXL3_PACK`, `GLM_MODEL_DIR` - and `GLM_EXLR8_ROUND=exact` asks for the format's own rounding
instead of the two-instruction one (below).

Carried over and exercised on it: adaptive residency (396 replayed requests: 7,226 async hot
reads, 46,237 swaps, session preloads), MTP speculation, copy rounds, the disk prefix store across
a restart (a 6,792-token prompt saved, then restored in 0.09 s) and a second lane beside it. The
store's namespace now carries the dense checkpoint and the rounding for an exlr8 pack - one pack
is served with either - and an EXL3 pack's is as it was. Not exercised: the abliteration overlay,
contexts past 14K tokens, two lanes under sustained load.

## Measured against the served EXL3 (Mia's 2.75 bpw experts + bertholomus dense)

Same binary, same memory budget (`GLM_SLOTS=12200`: 86.8 GB of experts a rank - 12,980 EXL3
experts, 14,133 exlr8), one boot an arm, the arms alternating.

**bench/bench.py** (the README's protocol; `results/rust-glm53-exlr8-20261111`; medians of three):

| | EXL3 2.75 bpw | exlr8 bit260 | |
| --- | ---: | ---: | ---: |
| prose, 1,024 tokens | 20.50 tok/s (21.51 / 20.50 / 19.85) | 22.34 (22.34 / 22.40 / 22.07) | +8.9% |
| code, 1,024 tokens | 26.21 (26.38 / 25.38 / 26.21) | 28.20 (28.20 / 28.34 / 27.44) | +7.6% |
| random 8K, first token | 17.28 s | 17.88 s | +3.5% |
| random 8K, its 256 tokens | 20.19 tok/s (24.02 / 20.19 / 18.96) | 20.49 (19.45 / 20.49 / 20.88) | inside the spread |
| tokens a step: prose / code / 8K | 2.17 / 2.96 / 1.98 | 2.03 / 3.05 / 1.98 | |

**Verify pass** (`--bench-width`, 2,500 tokens of context, fixed residents, ms):

| rows | EXL3 | exlr8 bit260 | |
| ---: | ---: | ---: | ---: |
| 1 | 60.61 (60.86 / 60.36) | 56.16 (56.07 / 56.25) | -7.3% |
| 4 | 110.67 (110.76 / 110.57) | 103.75 (103.79 / 103.71) | -6.3% |

and with the format's exact rounding (`GLM_EXLR8_ROUND=exact`), three boots each:

| rows | EXL3 | exlr8 bit260, exact | |
| ---: | ---: | ---: | ---: |
| 1 | 59.88 (59.71 / 59.83 / 60.09) | 57.49 (57.64 / 57.33 / 57.50) | -4.0% |
| 2 | 79.04 | 75.59 | -4.4% |
| 4 | 110.29 | 103.79 | -5.9% |
| 8 | 163.65 | 156.90 | -4.1% |

The MTP draft pass is the same (2.44 ms). A 2,500-token prefill is 2% slower (4.68 -> 4.79 s):
the prefill tile is bound by the decode arithmetic and exlr8's codebook is the heavier one.

**Held-out NLL** (`--bench-nll`, heldout_quick, 10,151 tokens, no adaptation, no hot reads, paired
by token): EXL3 1.8793; exlr8 experts 1.8534 (-0.026 +- 0.006); with exlr8's dense 1.8549
(-0.024 +- 0.005). At equal memory: the 1,153 more residents are part of it.

**Serving** (the logged-turn replay of `docs/residency-speed.md`; hot reads, adaptation, two
lanes): paired rounds, tokens a round +0.8 +- 1.1% and +1.3 +- 1.2% (two runs): not different.
Six runs of up to 768 tokens from the logged prompts, one boot each: 23.29 -> 25.21 tok/s (115.6 ->
110.3 ms a step); the texts part ways, so that is one sample, not the size of the gain. Two boots
of one arm replay token for token (25.56 / 25.66 tok/s).

## Where the time went, and what it took

Fewer bytes alone bought nothing at one row. The pack's residents are 8% fewer bytes a pick
(6.14 against 6.69 MB) and the 4-bit dense 0.83 GB less a pass, and the first build measured
1 row +0.7%, 2 rows -1.6%, 4 rows -2.8%, 8 rows -2.6% (five and four alternating boots). The
decode kernels are bound by their arithmetic wherever their bytes come from L2 or their warps are
few (the notes' "EXL3 decode is not DRAM-bound"), and exlr8's codebook is the heavier one: mcg's
9 instructions a pair plus the rounding.

| change | measured |
| --- | --- |
| the exact rounding, 14 -> 7 -> 5 instructions a pair | cold 2-bit gate+up, 1 row, split 1: 174 -> 147 -> 137 us (plain mcg 133) |
| split-K for the routed experts' decode GEMVs on an exlr8 pack (gate+up 4 at one row, 2 above; down 2 at one row) | in-boot `--bench-ab split`: 1 / 2 / 4 / 8 rows -1.5 / -0.9 / -0.6 / -0.3% (an EXL3 pack: -0.4 / -0.4 / -0.8 / -0.4%, left at split 1 - its sums' order) |
| o_proj (exlr8, 4-bit) at split 8 instead of 2 | in-boot `--bench-ab osplit`, two boots: 1 row -2.2 / -2.3%, 4 rows -1.4 / -1.7% |
| the rounding in two instructions (`GLM_EXLR8_ROUND=fast`, the default): half an E4M3 step added to the fp16 bits, the low mantissa bits cleared | in-boot `--bench-ab round`: 1 row 60.25 -> 59.06 ms (-2.0%), 4 rows 111.11 -> 109.25 (-1.7%); NLL in the same boot 2.3197 -> 2.3196 on 3,477 tokens (-0.0002 +- 0.0042) |
| q_b at split 4 | +0.4 / -0.6%: left |
| 4-bit dense, timed alone (`glm_bench_chain`) | o_proj 149.5 -> 128 us a layer, q_b 55.4 -> 47.3; the shared expert's gate+up 51.4 -> 53.2 and down 29.8 -> 28.8 (latency-bound: no gain) |

The two-instruction rounding is not the format's: ties go away from zero and a value under 2^-6
keeps three mantissa bits instead of the 2^-9 grid, so 1,078 of the 65,536 states decode to a
neighbouring E4M3 value - an rms of 0.8% of the codebook's, against the 16-29% by which two widths
of one unit differ. (No rounding at all is 2.7%, and 98% of the states.)

Built and left off, or not built, with the reason:

- **One input rotation and one output rotation a row** (the reference's order of operations:
  exlr8's sides are the layer's signs times one scale a 128-block; `exl3m_rot_in_shared`,
  `exl3m_down_combine_shared`, knob `rot1`): equal to the per-pick kernels to fp32 rounding (NLL
  2.3197 -> 2.3183, -0.0014 +- 0.0040) and no faster - 1 row 59.62 -> 59.90 ms, 4 rows 108.93 ->
  108.87. The per-pick kernels are 768 short blocks side by side; a warp a block looping over the
  row's picks does less work in the same time.
- **The format's own split** (a rank owns a unit's gate, up and down; one MoE exchange a layer
  instead of two): not built. The gather it removes is 17 us at one row, and the engine already
  reads the down rows into L2 during it - the bytes move into the down kernel.
- **A codebook table** (65,536 states -> an E4M3 byte): not built. A warp's 32 lookups land in 32
  cache lines; the arithmetic is 11-14 instructions a pair.

Measurement traps met: a boot that runs `--bench-nll` before `--bench-width` times its passes
~3 ms (1 row) to ~7 ms (4 rows) slower than one that does not; and for a quarter of an hour
(14:45-15:00) boots of either pack timed passes over an 8 ms spread (1 row: min 55.4, max 64.1)
where the alternating sets before and after hold within 1.5 ms. The in-boot A/Bs (ABBA rounds)
are free of both.

## Disk

The mirror is 237 GB of experts and 8.8 GB of dense files (the head; not read once the packs
exist), a rank's pack 120.4 GB (each node), the dense checkpoint 6.9 GB beside the base's 15 GB
(each node; the base's shards are hard links).

## `home` (3.25 bpw, 2026-11-12)

The release's `home` manifest: each routed expert whole at one width - in every decoder layer 192
at K3 and 64 at K4 (the widths of GLM-5.3-EXL3-TR3-3.25bpw's tier bitmap), the MTP layer's 256 at
K3 - and the shared experts as `bit260` has them (K4; K3 in the MTP layer). Profile:
`configs/glm53-exlr8-home.env.example` (since 2026-11-18 `configs/glm53-exlr8-bit322.env.example`), which differs from `bit260`'s in `GLM_EXL3_PACK`.

Nothing in the engine changed for it. A pack's records carry their widths and the arena makes a
pool of slots a width triple (`class_arena`), so 4/4/4 records are a class beside 3/3/3 as 2/2/3
is in `bit260`; the kernels read a slot's widths from its tables. It is the first 4-bit routed
record served from an exlr8 pack. Checked: the pack against the reference decode
(`tools/test_glm_pack_exlr8.py`, ten records of layer 10 on rank 0, three of them 4/4/4: max
|pack - reference| 6.7e-8), every unit against the release's xxhash64 as the packer read it, and
in serving that it boots, answers six prompts of a graded set correctly and has carried a
morning of agent traffic with hot reads, swaps and preloads. Not checked: the GPU kernels' decode
of a 4-bit record with this codebook against the reference (the fixture test above ran on
`bit260`'s widths).

The dense checkpoint is `bit260`'s: the two manifests name the same shared-expert widths in all
76 layers, and nothing else of the manifest reaches `glm_exlr8_dense.py`.

The fetch needed one fix. `home` cuts the release's K3 files at other offsets than `bit260` did
(192 of a layer's experts where `bit260` took every down), and the journal was compared piece by
piece, so a `home` fetch into
`bit260`'s mirror planned 292.9 of its 299.5 GB again. The plan now subtracts the byte ranges
the journal holds (`missing`): 172.1 GB, fetched in 54 minutes beside the serving engine.

| | `bit260` | `home` |
| --- | ---: | ---: |
| a rank's pack | 120.4 GB | 149.8 GB |
| records | 5.5 MB (2/2/3), 7.1 MB (3/3/3) | 7.1 MB (3/3/3), 9.5 MB (4/4/4) |
| resident at `GLM_SLOTS=12400` (88.3 GB a rank), bundled history | 14,361 (74.8%) | 11,430 (59.5%): 8,505 3-bit, 2,925 4-bit |
| the release's KL to BF16, held-out, every expert loaded: mean / median | 0.192 / 0.0222 | 0.149 / 0.0125 |
| bench/bench.py prose / code, tok/s | 22.34 / 28.20 | 20.38 / 23.91 |
| random 8K: first token, its 256 tokens | 17.88 s, 20.49 tok/s | 17.18 s, 19.69 tok/s |
| decode picks missed in those runs: prose, code, 8K | 2.1-2.6%, 6.3-8.2%, 13.0-15.3% | 6.3-7.9%, 13.1-16.8%, 21.7-23.3% |

The speed rows are a day and a build apart (`results/rust-glm53-exlr8-home-20261112`: `home` ran
the image of main plus uncommitted hot-read work with its knobs unset, at 12,400; `bit260` a host
build of this branch at 12,200), so the gap - prose -8.8%, code -15.2% - is an estimate, not an
A/B. Its direction has two causes that are not in doubt: more bytes a pick, and 2,900 fewer
residents, which here shows as two to three times the missed picks on prose and code.

Residency is the cost that matters. The routing history (the served one, the bundled seed
included) has no dead experts to give up - none of the 19,456 has zero mass, the least used holds
a tenth of an even share of its layer, and half of all picks need 7,125 experts, 90% need 16,076
- so the 2,900 that `home` leaves on disk are
experts traffic asks for. Whether 3.25 bpw at 59.5% resident answers better than 2.6 bpw at
74.8% is the open question, and it is not measured: no held-out NLL, no verify pass, no graded
comparison was run for `home`.

One property of mixed widths to know: the two pools are sized at boot from the history (8,505 /
2,925 from the bundled one; 8,413 / 2,994 and 8,408 / 2,998 from a used one) and a swap, a
preload or a kept hot read replaces a resident of its own width (`plan_swaps`), so memory does
not move between the widths until a restart re-plans them.

Disk for `home`: 299.5 GB of the release's experts (172.1 GB more in a mirror that holds
`bit260`), a rank's pack 149.8 GB (each node).

## `bit322` (3.22 bpw, 2026-11-18)

**The release changed under the profile.** 0.6 (2026-11-18) re-encoded every routed and shared expert
of layers 3-78 in the **mul1** codebook (from the same Hessians), stored the embedding and head in BF16,
gave the weights a new set ID and removed the `home` manifest (`bit325` had lower KL at its size); 0.7
the same day replaced the width profiles with a block sweep's (`bit200` ... `bit400`, `bit325` gone for
`bit322`). Nothing of the 0.5 mirror is reusable: the profile is a new fetch, pack and dense directory.

**The port.** A trellis tensor now carries its codebook in its name (`trellis.gate.mul1`; bare is mcg),
and the dense trellis matrices stay mcg while the experts are mul1. mul1 rounded to E4M3 is kernel
codebook 5 (6 with the two-instruction rounding): `cb_pair` computes mul1's fp16 fma as for EXL3 and
then the same `e4m3_round2` (mul1's values are under 3.46, inside its range), `exl3_unpack_kernel`
likewise, and `exl3::exlr8_round` switches a matrix between its exact and fast rounding keeping its
hash, since one boot now holds both. The pack's header says `mul1-e4m3`; the dense directory marks a
mul1 shared expert `<name>.mul1_e4m3`. `tools/exlr8_ref.py`, the fetch, the packer, the dense writer
and the checks read the codebook from the tensor names and refuse a file in none they know.

Checks: `tools/exlr8_check.py` on layer 40 (units at K2/K3/K4) against the publisher's loader
(`kindlingai/exlr8` `loader/exlr8.py`): tiles identical, the codebook identical in values and bytes
to torch's E4M3 cast of `exl3_ref`'s mul1 table, offsets and checksums as the index says, the engine's
operation order against x W within 1e-4. Both ranks' packs against the reference: 12 records each,
spread over the layers, within 9e-8. The decode kernels over a pack record
(`exlr8_codebook_matches_the_reference`, `EXLR8_FIXTURE_CB=mul1`): 2.1e-4 from the reference; the
same tiles read as plain mul1 2.7e-2, with the two-instruction rounding 2.6e-3. That rounding moves 845
of mul1's 65,536 values to a neighbouring E4M3 value (rms 0.24% of the codebook; mcg: 1,078, 0.79%,
measured NLL-neutral), so it stays the default.

**`bit340` first.** At `GLM_SLOTS=12400` it held 10,946 experts (estimated 10,964 beforehand from the
routing history - the same estimate gave `home` 11,417 against its measured 11,406-11,430) and missed
25-30% of its picks in serving. Its user found it hallucinating within the hour, and it was deleted
without a measurement.

**`bit322`.** 11,544 experts resident at the same slots (2/2/3 2,402, 3/3/3 3,676, 3/3/4 1,539, 4/4/3
773, 4/4/4 3,154). Held-out NLL, `--bench-nll` on heldout_quick (10,151 tokens) with fixed residents,
no adaptation and no hot reads, `bit260`'s arm settings (`GLM_SLOTS=12200`):

| arm | residents | NLL | miss (scored passes) |
| --- | ---: | ---: | ---: |
| `bit260` (0.5 weights) | 14,133 | **1.8549** | - |
| EXL3 2.75 bpw | 12,980 | 1.8793 | - |
| `bit322` (0.7) | 11,359 | 1.9073 | 29.05% |

In step with the residents, against the release's KL (which loads every expert: `bit322` 0.149 mean on
its held-out text at every position, the old `bit260` 0.192): here a missed pick is covered by a
substitute, and the experts a width costs are the larger effect. Estimated from the routing history at
12,400 slots, the lower profiles of 0.7 would hold `bit262` ~14,107 (82.4% of the routed mass),
`bit271` ~13,689, `bit280` ~13,252, `bit300` ~12,421 - not measured; its user keeps `bit322`.

Speed (`bench/bench.py`, `results/rust-glm53-exlr8-bit322-20261118`, inline rescue at 0.3): prose
20.35, code 25.36 tok/s, 8K first token 17.09 s, its decode 19.78 - `home`'s rows the same within
noise but code, 4-6% faster. Inline rescue at 0.3 instead of 0.4 read 0.6-1.0 picks a pass inline for
~0.15, at that speed; its NLL A/B is still not run.

Disk for `bit322`: 297 GB of the release's experts and 8.2 GB of dense files (the head; 44 GB more
than `bit340`'s mirror), a rank's pack 148.6 GB, the dense directory 21 GB (each node).
