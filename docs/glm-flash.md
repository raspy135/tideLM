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
