# GLM-5.3-Flash on one Spark (branch `glm-flash`, started 2026-11-12)

Goal: serve [Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-3.5bpw)
(rev `6f0cbf4a`, 156.3 GB) at TP=1 on one DGX Spark with adaptive residency, tuned to the level of
the GLM-5.3 and DS engines from the start (micro benches per kernel, full runs only to confirm).
TP=2 stays possible (world-generic code, packs derived from the one-box pack) but is not tuned:
at TP=2 the model is fully resident and other engines serve that case.

## The model (glm5_next) against GLM-5.3 (glm_moe_dsa)

| | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| hidden / layers | 6144 / 78 + MTP | 4096 / 45 + MTP (layer 45) |
| attention | MLA + DSA every layer (RoPE 64) | 34 KDA layers; MLA + DSA on 3, 7, ..., 43, no RoPE anywhere |
| MLA head | nope 192 + rope 64, kv_b 448 rows a head | nope 256, kv_b 512 rows a head |
| indexer | 32 x 128, top-2048 tokens | 32 x 128 over 4:1 pooled keys (gate + ape), top-512 pools + 0-3 tail tokens (<= 2051) |
| residual | plain | mHC: 4 streams, Sinkhorn 20 (DS-V4's), the MTP layer plain |
| MoE | 256 x top-8, inter 2048 | 288 x top-8, inter 2048, swiglu limit 10 |
| checkpoint | EXL3 everywhere | routed experts EXL3 mcg (3/4-bit), everything else BF16 |

Tokenizer: byte-identical to GLM-5.3's. Chat template: an older revision of the same family.

Sizes (safetensors headers): routed 133.3 GB (12,096 experts, ~11.0 MB), MTP experts 3.6 GB
(4-bit), non-expert BF16 18.2 GB without the vision tower (1.1 GB, not served).

## Decisions

- Non-expert weights in FP8 (the user's call, 2026-11-12): e4m3 with 32 x 32 UE8M0 block scales
  (ds41.cu's format and GEMVs, converted at load), BF16 kept loadable as the reference. To measure:
  held-out NLL BF16 vs FP8 in one boot.
- One GLM-family engine with a layer kind (MLA-DSA, KDA, nope-DSA), not a third crate: the MoE,
  residency, lanes, MTP, prefix and serving code is shared by construction. New code in its own
  modules (flash.cu, kda/pool kernels; Rust modules for KDA and mHC); GLM-5.3's kernels became
  templates whose GLM-5.3 instances compile to the same SASS.
- KDA rollback: double-buffered state a lane, rejected drafts undone by replaying the kept rows
  (bit-exact, no per-row snapshots). Rollback reaches back one pass; prefix reuse past that needs
  saved states (serving work).

## Status

Done (commits on `glm-flash`):
- tools/glm_pack_exl3.py `--repo` (packs straight from the Hub), `--from-pack` (TP=2 packs from the
  one-box pack), `--resume`, `--verify` (04652b3).
- cuda/flash.cu: KDA conv / chain / replay, key pools, ring, expansion; tests/flash_gpu.rs (2f90c4f).
- glm.cu templates: absorbs at nope 256, glm_mla_nope, router over 288, BF16 kv_b quantizer (512662e).

Fetch (2026-11-12): `~/tidelm-rust-state/flash/fetch.sh` -> `~/models/GLM-5.3-Flash-3.5bpw-dense`
(non-expert tensors, glm_fetch_dense.py) and `~/models/glm-flash-packs/glm-flash-35-r0of1.bin`.

- The engine (677c822): one GLM-family Engine, a layer kind each, Lin (EXL3 / FP8 / BF16), mHC via
  ds41.cu, KDA with rollback, nope-DSA with pools, Flash's MTP layer on GLM's MTP path.
- tools/flash_ref.py against transformers' glm5_next (4af5c62 and after), engine at BF16:
  - 24-token prompt, layers 0-5 (MoE layers wholly resident): rel. err. 0.6-0.9%, row cosine >= 0.99984;
  - a 3,259-token prompt (two prefill chunks, pool selection active, 815 pools) and 8 decode steps:
    prefill 0.6-1.2%, decode 0.7-1.8% (one decode row's sparse layer 5.3% / cosine 0.9986, as the worst
    prefill rows: the FP8 pooled keys moving a pick near the cutoff);
  - all 45 layers with transformers routing over the engine's residents: 0.4-2% through the stack, the
    same next tokens (top-5 agree; one near-tie).
- Greedy MTP (depth 3) and plain decode agree token for token over the 48 tokens compared.

Next: glm_main.rs / rust-up.sh for one box and a profile, GLM-5.3's byte-for-byte check on the pair,
held-out NLL BF16 vs FP8, then tuning.

## Residency needs a history

The first runs placed residents uniformly (no routing history): 42% of each token's picks masked,
and the output degraded from the first token on - transformers' glm5_next routing over the same
residents degraded identically, so it is the placement, not the engine. A history from prefills of
150K tokens of the repo's calibration sources (`glm_run --history-out`, 188 s; the exact router's
weight per expert, whatever is resident): half a layer's routed weight is in 56 of its 288 experts,
the top 58% hold 85% of it, the top 67% 90%. With it the same prompt reads right; with synchronous
hot reads the first tokens too (a short reasoning, then the answer).

| run (one box, FP8, graphs, 26-token prompt) | residents | tok/s | MTP accepted a round |
|---|---:|---:|---:|
| plain, uniform placement | 7,040 (58%) | 16.8 | - |
| MTP 3, uniform placement | 7,040 | 26.2 | 1.58 |
| MTP 3, history v0 | 6,895 | 23.9 | 1.68 |
| MTP 3, history v0, sync hot reads (1,018 reads, 1.33 s) | 6,895 | 20.5 | 1.70 |

## Measurements

KDA (flash_gpu kda_bench: 34 layers back to back, own state each, 64 heads, split 2):
1 / 4 / 8 rows 1.54 / 1.81 / 2.47 ms a pass; one row is the state's 8 MB a layer at ~195 GB/s.
Many rows 3.1 us a row (an 8K prefill ~0.9 s: the chunked WY form is the fix).

FP8 GEMV (flash_gpu fp8_gemv_bench, cold, best split), 1 / 4 / 8 rows:

| matrix | MB | us | GB/s at 1 row |
|---|---:|---|---:|
| KDA q\|k\|v\|f_a\|g_a\|b 24896 x 4096 | 102.0 | 460 / 466 / 493 | 222 |
| KDA o_proj 4096 x 8192 | 33.6 | 159 / 174 / 186 | 211 |
| MLA q_a\|kv_a\|indexer 2336 x 4096 | 9.6 | 86 / 83 / 94 | 111 |
| MLA q_b\|wq_b 20480 x 1536 | 31.5 | 156 / 157 / 161 | 201 |
| MLA o_proj 4096 x 16384 | 67.1 | 311 / 333 / 349 | 216 |
| shared gate\|up 4096 x 4096 | 16.8 | 109 / 124 / 123 | 154 |
| shared down 4096 x 2048 | 8.4 | 72 / 77 / 74 | 116 |
| dense gate\|up 24576 x 4096 | 100.7 | 476 / 481 / 518 | 212 |
| head 154880 x 4096 | 634.4 | 2570 / 2686 / 2699 | 247 |

From these, a 1-row pass: ~42 ms of non-expert reads (KDA 21, shared experts 7.6, MLA 6.1, head
2.6, dense 2.1, router + mHC ~1.3) + ~18 ms of routed experts (3.7 GB) + 1.5 ms KDA state: ~16 tok/s
before speculation. The small matrices (8-17 MB at 110-154 GB/s) are the GEMV lever left.

## Vision (2026-11-12)

Image parts of chat / messages requests (GLM_VISION=1). The checkpoint's chat template has no image
form, so the renderer writes `<|begin_of_image|><|image|><|end_of_image|>` where it would write its
"unable to process" reminder, and `<|image|>` becomes the image's tokens (one a 2 x 2 patch window).

- Preprocessing (tidelm-image glm5next): transformers' Glm5NextImageProcessor bit for bit (torch's
  int16-weight antialiased bicubic, smart_resize, zero padding, fused normalize) - identical patches
  for six images from 20 x 10 to 4000 x 3000, RGBA and JPEG included.
- The tower (flash_vision.rs): each block matches transformers' from the same input to 1-3e-3; over the
  24 blocks the two drift as transformers' own bf16 tower drifts from its fp32 one (the ViT amplifies
  bf16 rounding). 78 ms for 1,656 patches, 1.13 s for 14,868.
- Dynamic load (lend.rs): the tower's 1.13 GB and scratch are lent expert slots for the request - the
  least-wanted class's last ~150-250 slots leave routing, the tower loads into them, the images encode
  into the lane's feature buffer (8 KB a token, allocated for the request, freed after its prefill;
  refused below 6 GiB MemAvailable after it), the slots are refilled from the pack before the prefill.
  Text-only traffic keeps the memory for experts. An encode costs ~1.0-1.1 s for a small image (mostly
  the load and the refill), 2.3 s for a large one; 5.4 s for 4000 x 3000 + 1920 x 1080 (10,622 rows).
- GLM_VISION_MAX_TOKENS (8000, the processor's max_image_tokens) caps an image, not a request. Until
  2026-11-12 it also sized a boot-time buffer for the whole request, so a chat's second large image
  (the first one still in its history) failed: "8976 image tokens in a request" in served traffic.
- Image state is a lane's own: a prompt prefilling in parts beside a decoding lane keeps its features
  while the other lane starts a request (one shared buffer had let any start, with images or without,
  clear the spans - the rest of the image rows would have read the placeholder's embedding).
  Checked: a 29,441-token prompt with its images at the end, the other lane finishing and starting a
  request during its 53 s prefill: both images read right, the text quoted exactly.
- The prefix reaches past an image when the lane's sequence holds the same image (position, grid, a
  128-bit hash of its patches) there; the images before the reused prefix are not encoded again.

Served examples: the check image's shapes and text read exactly; the repo's expert-map screenshot read
in detail (first token 7.9 s for 3,738 tokens); two images in a request; a second turn answering from
an image in its history.

Open: keeping the tower lent across consecutive image requests (the load is most of an encode), video.

## Chat follow-ups: the KDA state at the prompt's end (2026-11-12)

KDA's state goes back only within its last pass, so until now a prompt reused a lane's sequence
only when it extended it to that pass. A chat follow-up does not: the template renders an earlier
turn's reasoning as `<think></think>` (chat_template.jinja: kept only after the last user message,
or with clear_thinking false), so the prompt leaves the lane's sequence where the last answer began.
In served traffic follow-ups of 93-107K tokens prefilled whole, 132-155 s each; tool loops, which
keep the reasoning, reused theirs (91,567 of 91,647 cached).

Now each prompt's prefill snapshots the lane's KDA state, conv rows and sparse-layer rings two rows
before its end (a chunk's edge there, or a replay of a short pass's saved rows; the DSA rows are
positional and stay), committed when the lane decodes; a later set_pos to it restores it. Two rows
back covers a follow-up (it shares the prompt to its end, then the MTP row is refilled) and a
regenerate. Two snapshot slots a lane (the committed one, the one the prefill is taking: a two-pass
prefill goes back to the committed one between its passes): 0.3 GB a lane.

glm_run --snap-check (24K-token prompt, residency fixed): the last 40 rows prefilled again from the
snapshot give the original logits bit for bit; a follow-up from it equals a fresh prefill split at
the same row bit for bit; a snapshot replayed inside a 10-row decode pass equals the in-pass rollback
bit for bit. (A fresh prefill in one call differs from one split at that row by 17% in logits, the
same argmax: the engine's chunk-edge sensitivity, snapshot or not.)

Served (the profile, Open WebUI's shape - history without reasoning, greedy):

| turn | prompt | reused | first token |
|---|---:|---:|---:|
| 1 | 24,151 | 0 | 34.5 s |
| 2, a follow-up (turn 1 reasoned) | 24,268 | 24,149 | 1.19 s |
| 2 again, a regenerate | 24,268 | 24,266 | 0.39 s |
| 3, a follow-up | 24,354 | 24,266 | 0.67 s |

Fixed with it: an emptied lane (kv_shrink: after the boot's warm-up, every lane; make_room, when
the pool is short) dropped its positions but not its KDA state, and set_pos(0) at the same position
did nothing - so the first prompt on each lane after a boot, and one on a lane the pool emptied,
started from the old sequence's state. Not measured on the old build; --snap-check now checks that an
emptied lane prefills as a fresh one does, bit for bit. And rank 1 now rolls KDA back with rank 0
(its set_pos only truncated: a Flash pair, never run, would have diverged at its first rejected draft).

## Residency and MTP depth (2026-11-12)

GLM_SLOTS 9000 -> 9700: 8,345 experts resident (69.0%, was 64.1%), the box at 7.6 GiB MemAvailable
idle, 7.25 after bench.py (GLM-5.3's profiles: 6.5-7.2; the KV pool is allocated whole at boot).

MTP depth, bench.py (temperature 0.6, seed 42, a warm-up and 3 runs), one boot an arm from the same
state file, 9700 slots:

| | prose tok/s | code tok/s | tokens a step, prose / code |
|---|---:|---:|---:|
| depth 3, no gate (the profile until now) | 20.27 (20.15-20.57) | 28.40 (27.67-28.64) | 2.17 / 3.17 |
| depth 7, calibrated gate 0.36 (GLM-5.3's) | **22.12** (22.07-22.28) | 28.15 (27.83-28.17) | 2.01 / 2.96 |
| depth 5, calibrated gate 0.36 | 21.51 (21.41-22.34) | 27.81 (26.99-28.41) | 1.96 / 2.91 |

Depth 7 with the gate is the profile's now: prose +9.1%, code within its spread. Fewer tokens a step
in cheaper rounds - the gate leaves out the rows that would be rejected.

## TP=2 (2026-11-12, first run)

configs/glm53-flash-35-tp2.env.example: each rank's pack re-sliced from the one-box pack
(`glm_pack_exl3.py --from-pack ... --world 2 --rank R`, 68.6 GB a rank, ~1 min each at 2.2-2.7 GB/s;
rank 1's copied to the peer at ~490 MB/s), the dense dir on both nodes. All 12,096 routed experts
resident (66.8 GB a rank), dense 5.76 GB a rank, KV 2.24 GB (whole on each rank: GLM-5.3's position
split is refused for Flash), device 74.5 GiB, ~38 GiB MemAvailable a box; residency, hot reads and
adaptation off, no vision (one box's only). Boot 44 s.

First requests (greedy, single runs - not bench.py): a 129-token poem at 33.5 tok/s, a 219-token
function at 53.2; a 24,146-token prompt prefilled at 1,101 tok/s (TP=1: 700), decoded at 48.9; its
follow-up's first token 0.42 s from the snapshot (24,144 reused), a regenerate 0.08 s. Not tuned.

Found with it: the DS-rule routing history (residency::DsHistory, written at each request's end, read
by nothing in serving) had a fixed name beside the state db, glm-ds-history.npz - GLM-5.3's. Flash
opened it, found another shape, started empty and saved its own over GLM-5.3's (unrecoverable;
GLM-5.3's next boot starts it anew). Flash's is now its state db's own (<state db>.ds-history.npz).

### TP=2 at 1M a lane (2026-11-12)

The model's own context (max_position_embeddings 1,048,576; no RoPE) for each of two lanes at once:
GLM_MAX_SEQ=1048576 and GLM_KV_TOKENS=2097152, a new key (the KV pool in positions over all lanes;
unset, one lane's max_seq shared as before). KV 14.42 GB a rank, device 86.9 GiB, MemAvailable
26-28 GiB a box. One box keeps 327,680: there the same memory is ~1,300 expert slots. bench.py was
not rerun (its rows were measured at 327,680, short prompts).

Long prompts on the pair (2026-11-13, greedy, reasoning low): the repo's own text (repeated past its
length) with two facts planted at 3% and 97% of it, both asked for at the end - sized by
/v1/debug/prompt, one request each:

| prompt | prefill | first token | decode | both facts |
|---:|---:|---:|---:|---|
| 343,326 | 337 s, 1,018 tok/s | 337.6 s | 43.3 tok/s | yes |
| 1,008,546 | 1,230 s, 820 tok/s | 1,231 s | 42.3 tok/s | yes |

Device memory did not move (86.9 GiB, the pool allocated at boot); MemAvailable 25 GiB after.

Found with it: the pass's decoder lane offsets took the split cache's lane stride (dec_rows, padded
from s_loc + 1) where the caches are whole (mtp_rows, padded from max_seq). Rows pad to 524,288, so
the two agree at 262,144 and 327,680 and differ at 1,048,576 (1,572,864 against 1,048,576): lane 1
read and wrote the wrong rows and served garbage ("## ## ##" from the first token), lane 0 was right
(glm_run's one lane gave the 327,680 tokens exactly). The offset is now the caches' own stride; the
configurations served before get the same value. Checked: two concurrent requests and a third, at 1M
on the pair, all answer right.

### TP=1 prefill by length (2026-11-13)

Fresh prompts of the repo's text (a nonce first: no prefix reuse), greedy, reasoning low, one
request at a time on the TP=1 profile: 23,181-25,239 tokens 709-713 tok/s (three runs), 30,212 /
30,327 tokens 713 / 720 tok/s, 305,387 tokens 690 tok/s (442.5 s) - prefill falls 4% over 10x the
length (34 of 45 layers are KDA; the 11 sparse layers' indexer is the part that grows). Prompt misses
fell with length (18.9% at 25K, 13.9% at 305K: the hot reads catch up), which hides some of it.
Decode after the 305K prompt 28.0 tok/s.

### Inline rescue and seed v1 (2026-11-13)

A user reported a thinking loop the author had not met. A new install's residency comes from the
bundled seed; v0 (150K tokens of the repo's sources, prefill only) placed a 69% residency over
74.6% of the author's served traffic's routed weight, against 84.2% placed by the traffic itself
(the traffic's file still holds v0, so the gap to a stranger's traffic is wider). Two changes:

- Seed v1 (profiles/glm53-flash-history-v1.npz): the development box's state file after a day of
  serving - 2.9M routed rows (chat, bench.py, 438K prompt tokens of long-prompt tests, v0). Read only
  while RUST_GLM_STATE_DB's file does not exist.
- GLM_HOT_RESCUE=0.4 on the TP=1 profile, then off again (below; docs/hot-predict.md: built for
  GLM-5.3 and never qualified there). Flash runs it through the shared decode_segment path (layer_pre and final_norm dispatch to
  Flash's own); the transient slot is its own class, last in the arena.

Rescue, two boots from one frozen state file, `--bench-nll` over heldout_quick + heldout_eval (57,315
tokens, 8-row passes, no adaptation in the bench), then `--bench-width --bench-ab rescue` in the
second boot (the knob switches between the boot's threshold and off):

| | off | rescue 0.4 |
|---|---:|---:|
| NLL | 1.7121 | 1.7169 (+0.0048 +- 0.0015 paired; coding +0.0120, general -0.0034) |
| top-1 | 62.96% | 62.92% |
| blown windows | | 0 / 0 (median 250-window +0.0035) |
| miss (by score) | 11.48% (7.17%) | 11.42% (7.13%) |
| hot reads, waited | 15,635, 65.6 s | 15,581 incl. 728 rescues, 61.4 s |
| run | 1,095 s | 1,069 s |
| decode pass 1 / 4 / 8 / 16 rows | 62.93 / 105.75 / 142.33 / 198.68 ms | +0.1 / +0.2 / +0.0 / +0.2% |

The segment structure costs nothing measurable here (DS's sync arm lost 6-16% to it). The NLL moved
the wrong way for a change that computes misses exactly; the two boots' hot reads diverge from the
first rescued token on, which the paired error does not see - so a noise-floor arm: the off boot
again with TIDELM_GLM_MLA_HALF=0 (decode attention's split partials in fp32: numerics-neutral, and
the hot reads diverge from it the same way). NLL 1.7127, +0.0006 +- 0.0012 paired (coding +0.0012,
general -0.0001, median window +0.0005), miss 11.46% (7.16%), 1,038 s. Rescue's +0.0048 (coding
+0.0120) is 8x that: the loss is its own. Shipped at 0.4 for the hour this took (2c7df05), then
back to off, and on again the same day by the author's choice with every other async profile
(docs/hot-predict.md) - no new measurement behind that. Cause not found; next to look at: the same harness with GLM_HOT_RESCUE=1.0 (the
segment path with no rescue read - whether the per-segment graphs alone differ from the whole
pass on Flash), then whether keeping rescued experts over the queued victims costs the code
texts their residents.

## Review and decode tuning (2026-11-13, branch `flash-opt`)

Three read-only reviews of the released engine (KDA and its lane state, the sparse layers and mHC,
the serving layer) and a decode profile. The head box kept serving; everything was measured on the
second Spark as a one-box engine (the one-box pack copied there, the TP=1 example profile with
vision off, the frozen routing state and text of the rescue A/B: `--bench-width`, 2.5K context).

### Where a decode pass goes

The released build: 60.7 / 73.7 / 98.7 / 131.1 ms for a verify pass of 1 / 2 / 4 / 8 rows, an MTP
draft pass 3.6-3.7 ms. nsys with graph nodes traced, a pass's span attributed by completion order;
one KDA + MoE layer of a 1-row pass, 1,231 us:

| piece | us | bytes read, rate |
|---|---:|---|
| KDA q \| k \| v \| f_a \| g_a \| b | 422 | 102 MB, 242 GB/s |
| f_b, g_b, the conv and its state | 21 | |
| the KDA chain | 27.5 | 8.4 MB of state read and written |
| o_proj | 161.5 | 33.6 MB, 208 GB/s |
| the MLP's mHC site (seven launches) | 18 | |
| router and shared gate \| up (side stream) | 103 | 19.2 MB, 186 GB/s |
| routed gate + up | 230 | ~60 MB |
| epilogue, the one-box copies | 8 | |
| routed and shared down, the next proj's prefetch | 200 | 46 MB, 230 GB/s |
| combine, output | 7.5 | |
| the next layer's attention site (seven launches) | 30 | |

About 90% of the pass is weight bytes at 190-240 GB/s; the launches between them are ~120 us a
layer. An MTP draft pass is its head: 2.71 of 3.66 ms (634 MB).

### Decode: two changes, bit for bit

| in-boot A/B, ABBA rounds | 1 row | 2 rows | 3 rows | 4 rows | 8 rows |
|---|---:|---:|---:|---:|---:|
| fused mHC sites alone (3 rounds) | -1.8% | -0.3% | | -0.5% | 0.0% |
| fused mHC sites, prefetch on (5) | 0.0% | -1.0% | | -0.8% | |
| prefetch, fused sites on (5) | -1.6% | -0.8% | | -0.6% | |
| **both** (5; 61.11 -> 60.39, 75.32 -> 74.31, 87.36 -> 86.39, 99.53 -> 98.39 ms) | **-1.2%** | **-1.3%** | **-1.1%** | **-1.1%** | |

- An mHC site's seven launches (the last site's hc_post, the stream's rsqrt, the mix GEMV and its
  reduce, Sinkhorn, hc_pre, the RMSNorm) as two, `flash_hc_site` (flash.cu): the GEMV's grid, which
  computes the post on the way in and the rsqrt's partial sums beside its own, then one block a row
  for the rest. Same sums in the same order: `hc_site_is_the_seven_launches` compares stream, pre,
  post, comb and the normed input bit for bit at 1-16 rows (one unrolled product had to be an explicit
  FMA: the old kernel's runtime loop rounded it the other way in ~1 element of 10^4). 90 sites back
  to back, 33.8 -> 17.6 us a site at one row. `TIDELM_FLASH_HC_FUSED=0` keeps the seven; in the boot
  guard. One block a row for the whole site was ruled out on paper: the mix weights' 0.8 MB through
  one SM.
- L2 prefetches in a KDA layer's two idle windows (`TIDELM_FLASH_PF`): o_proj's first 6 MB during
  f_b / g_b / the conv, the shared gate | up's first 3 MB during the MLP's site. Other budgets
  (3 / 2, 6 / 6, 12 / 3, 12 / 6 MB, a boot each) measured the same within +-0.4%.

The two overlap: a site's time was already partly the prefetch branch's, so their gains do not add.
A greedy 160-token decode with both off and both on gives the same ids (glm_run, 24.85 and 25.04
tok/s). bench.py was not rerun: 1.2% is under what it resolves between boots.

Measured and not taken: on one box the pair's collectives are copies (two in the MoE, its output's,
o_proj's f32 -> bf16); reading each source in place removes 177 launches a pass and measured
+0.1 / +0.2 / +0.4 / -0.2% at 1 / 2 / 4 / 8 rows. They are 2-3 us each and the prefetch branch uses
the memory meanwhile.

The larger lever, the MTP head (2.71 of a 3.66 ms draft pass), is the next section's.

### Fixed

- **Lane admission** (serve.rs `pick_lane`). A queued prompt went to the free lane sharing the most
  tokens with it. Every chat prompt shares the template's first tokens with every lane, so any
  unrelated request - a client's title request after a chat's first turn - took the lane of the
  conversation instead of the empty lane beside it, and on this model kept nothing of it (KDA goes
  back to a snapshot or to 0). The conversation's next turn prefilled from 0. Now the lane with the
  largest 3 x reusable - held: what the prompt can really keep, against half of what it would destroy.
  Served, a 4.7K-token chat with a title request after each turn:

  | | released | now |
  |---|---|---|
  | turn 2 (4,807 tokens) | 0 reused, first token 7.26 s | 4,733 reused, 1.05 s |
  | turn 3 (4,869) | 0 reused, 6.9 s | 4,805 reused, 0.51 s |

  At 100K tokens the released build's turn is the 132-155 s this document's chat section set out to
  remove.
- **A non-streamed request whose client has gone** decoded to its stop or max_tokens (the profile's
  32,768): the closed-client check ran only when no token arrived for a second. Now a second apart
  between tokens too: a `stream: false` client leaving after 3 s is cancelled at 67 tokens.
- **The indexer's ring** (flash.cu ring_kernel): every row of a pass wrote slot `position % 64`, so a
  pass of more than 64 rows wrote each slot from several blocks of one launch, the last to land
  staying. The slot is read when the lane's next pass starts off a multiple of 4 (a pool's tokens
  before the pass: the snapshot split, a follow-up's chunk edges, the MTP fills); a pair's ranks
  each settled it on their own. Now a lane's last 64 rows only
  (`ring_of_a_long_pass_is_its_last_64_rows`). How often the wrong row won was not measured.
- **Staging** (forward.rs stage): the decoder's and the MTP layer's passes share pinned staging
  buffers that the stream reads when it reaches the copy. A prompt's chunks are queued without a
  wait, so a chunk's MTP fill could read ids the next chunk had already staged. Counted on a 9,000-token
  prompt: none at 2,048-row chunks (their launches block), one in each short prefill split at its
  snapshot (the 2-row chunk after it rewrote the fill's first two ids). Drafts only. Each stage now
  waits for the last one's copies.
- **Snapshots**: taken after the chunk's MTP fill (the snapshot holds the MTP layer's ring, which a
  restore then put back without that chunk's rows); a rollback keeps its pass as the lane's rollback
  (`last` was dropped, so a request ending on a rejected draft left only the prompt-end snapshot,
  and a tool loop's next turn prefilled the whole answer again). `--snap-check` gained the rollback
  inside a rollback's kept rows: bit for bit.
- **Image requests**: expert reads still on their way into slots (a background adaptation's, the
  async hot reads') are waited for before those slots are lent to the tower.
- **The KV pool between two lanes**: a round checked each lane's growth against the free chunks on
  its own, so two lanes crossing a chunk edge in one round could both pass for the same chunks, and
  the mapping that then failed ended the process. The check now counts what the round's earlier
  lanes will map; the lane that does not fit ends with `length`. Not reproduced (it needs the pool
  nearly full): by reading.
- GLM_PREFIX_DISK on this model built the store and saved every request's rows with nothing able to
  restore them: off with a boot line. GLM_ABLIT edited the MLPs' writes and not the attentions'.

### Found, not changed

One box is the audience (the user, 2026-11-13: the pair has better recipes elsewhere): nothing here
was rerun at TP=2, and the pair-only items below stay as notes.

- The Flash profiles do not set GLM_LOOP_BREAK (rust-up's default is 0; GLM-5.3's profiles set 12).
- Of two lanes prefilling in parts only the lower-numbered advances until it is done.
- A pair whose rank 1 fails at boot (a missing pack) is reported after 600 s, by rank 0's log.
- GLM_MTP_DEPTH is not checked against GLM_LANES (a shared verify past 33 rows exits).
- GLM_FLASH_DENSE=bf16 still quantizes kv_b, the latent cache and the pooled keys to FP8: a BF16 /
  FP8 NLL pair isolates the linear layers only.

## A listed draft vocabulary (2026-11-13)

An MTP draft pass is its head: 634 MB over 154,880 ids, 2.71 of 3.66 ms, and a round runs 2-3 of
them. The draft only proposes - the verify pass scores every id - so the draft passes now score a
list of the tokenizer's common ids through their own head (the listed rows of lm_head, in the dense
weights' format), the DS engine's idea (docs/draft-vocab.md) on one box. A token outside the list
cannot be drafted: that round's chain ends there, and no output token changes (greedy: the same ids
with the list on and off, checked in glm_run and in the served runs below; sampled: an unlisted
token has q = 0 and the rejection sampling stays exact).

`GLM_DRAFT_VOCAB` (unset: the built-in ranking, engine/glm_draft_vocab_ranking_v1.txt; `all`: no
list; a path: another ranking) and `GLM_DRAFT_VOCAB_ROWS` (57,344). One box only; a pair loads no
draft head. Its cost is memory: 235 MB of FP8 beside the weights (the box idled at 7.6 GiB
MemAvailable; no expert slot was given up for it).

### The list

This tokenizer spreads text over far more ids than the DS one, and by language: the lowest 32,768
ids miss 9-43% of held-out tokens, and a ranking over 18 language groups still leaves 5.9% of
English chat and 8.9% of Chinese unlisted at 40,960 ids (1.3% and 2.1% at 81,920). So v1 ranks
English chat (x3), agent traces and code (x2 each), English Wikipedia and Japanese (x1 each) -
69M tokens of public text and of GLM-5.3's own answers to public prompts, every 10th text held
out - and leaves the rest to the fallback below (tools/build_glm_draft_vocab.py; the file's header
has the sources). Held-out tokens unlisted, and the draft pass:

| listed ids | chat | agent | code | en wiki | ja | zh | ko | es | de | draft pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40,960 | 2.10% | 0.74% | 1.44% | 2.80% | 1.39% | 33.9% | 70.3% | 25.2% | 31.1% | 1.65 ms |
| 49,152 | 1.36% | 0.41% | 0.85% | 1.55% | 0.93% | 32.4% | 70.2% | 22.5% | 27.4% | 1.81 ms |
| **57,344** | 0.82% | 0.21% | 0.51% | 0.76% | 0.58% | 30.2% | 67.7% | 19.5% | 23.7% | **1.94 ms** |
| 65,536 | 0.48% | 0.06% | 0.29% | 0.34% | 0.35% | 28.7% | 64.1% | 16.8% | 19.7% | 2.10 ms |
| all 154,880 | | | | | | | | | | 3.64 ms |

(`--bench-ab draft_vocab`, three rounds a size.) A list by 32-id blocks of the head, which would
keep the full head's scale blocks and its bits, was ruled out on the counts: 7.5% of chat unlisted
at 32,768 ids against 2.7% by id.

### A lane's language decides

Text the list leaves out is 20-70% unlisted, and a drafter that cannot propose a fifth of the
tokens loses most rounds. Each lane therefore keeps the share of its recent tokens outside the list
(an average over ~64 tokens, started from its prompt's last 256), and a round drafts over the list
only while every drafting lane's share is at most 4% (serve.rs DRAFT_UNLISTED_MAX; the break-even
is ~7% for prose and ~4.5% for code, from the README runs' rounds). Past it the round's draft
passes score every id, as before. The request's log line counts both.

### Measured

Served on the second Spark, greedy, residency fixed (no hot reads, no adaptation: the two boots
decode the same tokens), eight prompts of 320 tokens, each twice:

| | list off | list on | | rounds over the list |
|---|---:|---:|---:|---|
| prose (two prompts) | 23.94, 20.36 tok/s | 24.96, 21.59 | +4.3%, +6.0% | all but 1 |
| code (two) | 31.76, 29.28 | 34.26, 30.92 | +7.9%, +5.6% | all |
| an agent's turn | 25.82 | 26.58 | +2.9% | 236 of 265 |
| Japanese | 27.57 | 27.54 | 0.0% | all |
| Chinese | 22.85 | 22.78 | -0.3% | none |
| Spanish | 23.61 | 24.44 | +3.5% | 180 of 288 |
| all 5,067 tokens | 25.20 | 26.12 | +3.7% | |

Two runs of one prompt in one boot differ by up to 10% (the calibrated gate learns between them:
Japanese with the list off ran at 29.00 and 26.14), so a single row here resolves a few percent.
Chinese is the fallback working: no round over the list, the speed it had.

bench.py, the README's protocol (sampled at 0.6, three runs after a warm-up): on the second Spark,
two boots from one state file, prose 22.28 -> 24.08 tok/s (+8.1%) and code 27.15 -> 28.63 (+5.5%)
with the list; then on the serving box, the profile as shipped, prose 23.29 and code 28.91 against
the 20261112 run's 22.58 and 27.56 (results/rust-glm53-flash-35-tp1-20261113: the README's row). The
8K first token is prefill and did not move (12.25 s against 12.00, inside the runs' spread).

Found with it: `--bench-width`'s MTP draft pass scores a row with no finite logit (its kept hidden is
not a decode's) - its draft is argmax's "none", list or no list. It times the pass; the id mapping
now passes that value through instead of indexing with it.
