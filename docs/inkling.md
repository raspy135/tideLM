# Inkling-2-Small on two Sparks (started 2026-11-16)

Goal: serve [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B parameters, 17B active; Apache-2.0) on two DGX Sparks at TP=2 with adaptive residency, from
[0xSero/Inkling-2-Small-EXL3-4bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw) (rev
`996dc404`: the routed experts in EXL3 mcg at a uniform 4 bits, everything else the source's BF16 - the
layout of the GLM-5.3-Flash checkpoints this engine already reads). Profile:
`configs/inkling2-small-40-tp2.env.example`.

Status: text, image input and the model's drafter work on the pair.

## The model (inkling2) against the GLM families

| | GLM-5.3 | GLM-5.3-Flash | Inkling 2 Small |
|---|---|---|---|
| hidden / layers | 6144 / 78 + MTP | 4096 / 45 + MTP | 4608 / 48 |
| attention | MLA + DSA | KDA; nope-DSA every 4th | grouped-query (48 query heads, 8 key / value heads x 128): a 1,024-key window on 40 layers, the whole context on every 6th |
| positions | RoPE | none | a relative bias a head (a 16-wide state mixed with a learned 16 x 1,024 table), a sink logit, a sigmoid gate on the output; the full layers scale q past position 128,000 |
| residual | plain | mHC, 4 streams | the block attention residual: every layer's inputs are a softmax mix of the saved block states (a layer l with l % 6 == 0 saves its input and restarts the residual) |
| recurrent state | none | KDA's, a conv | four 4-tap convolutions a layer (keys, values, the attention's output, the MoE's output) |
| MoE | 256 x top-8, inter 2048, a shared expert | 288 x top-8, a shared expert | 384 x top-8, inter 2304, no shared expert, softcapped SwiGLU (4 / 25) |
| head | EXL3 | BF16 | BF16, logits softcapped at 30, 200,058 of 201,024 rows are tokens |
| drafter | one MTP layer | one MTP layer | DSpark (5 layers, block 8, an RNN Markov head): not ported |

Sizes: 18,432 routed experts, 15.9 MB each at 4 bits (293 GB); non-expert BF16 14 GB with the vision
tower. The official checkpoint is NVFP4 in layers 0-39 and MXFP8 in 40-47 (373 GB of experts: about
48% would be resident here); the model was trained in NVFP4, so EXL3 re-quantizes a 4-bit model.

Reference: transformers' `modeling_inkling2.py` (on its main branch; in no release on 2026-11-16).
It has no drafter; SGLang's `dspark.py` does.

## How it sits in the engine

A third family of `tidelm-glm` beside GLM-5.3 and Flash (`Arch::Inkling2`, `AttnKind::{Swa, Full}`):
the MoE, residency, hot reads, lanes, graphs and serving code are the same code.

- Weights (`inkling.rs`): native names (`model.llm.*`). One fused projection a window layer (q | k | v
  | r | gate | sink), three a full layer (its q, k and v inputs are separate mixes). FP8 with 32 x 32
  block scales (`GLM_INK_DENSE=fp8`, the default) or BF16. The embedding is stored normed.
- TP=2: query heads and key / value heads by rank (24 and 4), o_proj by its input columns and summed;
  the mixes, norms, router and the two hidden-wide convolutions run whole on both ranks.
- State (`inkling_fwd.rs`): by position. The convolutions' inputs in rings of 64 positions, a window
  layer's keys and values in a ring of next_pow2(chunk + 1024 + 33), the full layers' in the paged
  cache (2 KB a position a layer a rank). A verify pass's rejected rows are not undone: the position
  goes back and the next pass writes over them. A set_pos further back than 60 positions restores a
  prompt-end snapshot (the window and the convolutions' last three inputs), taken as Flash takes
  KDA's. A plain rewind never goes below where the rings were last restored (the lane's floor).
- The disk prefix store (`GLM_PREFIX_DISK=1`): a segment is the full layers' rows of its positions
  and, as its tail, the lane's snapshot at its end (87 MB a rank), so a prompt is saved up to its
  snapshot (two tokens before its end) and a restore ends at a segment's end, never inside one
  (`Store::whole_only`). Checked 2026-11-16: an 8,629-token prompt saved in 0.15 s; after a server
  restart its follow-up turn restored [0, 8627) in 0.07 s (first token 0.98 s against 9.9 s) and
  answered from the document.
- Kernels (`cuda/inkling.cu`): the convolutions, the mix + RMSNorm, the head vectors (clips, unit RMS,
  log scaling), decode attention (six query heads a key head, online softmax in 4 or 64 key splits),
  prefill attention around cuBLAS (scores, a softmax kernel, weights x values), the logit softcap.
  `glm.cu` gained a 384-expert router instance, the EXL3 file an `act_mode 2` (the softcaps) in
  `exl3m_gateup`; Inkling's prefill takes the fat tile without the fused epilogue, which is the plain
  SwiGLU's.
- Text (`tidelm-text/src/inkling.rs`): the chat template (no generation prompt: the model writes its
  own `<|message_model|>` messages), ids, the answer router (thinking, text, tool-call messages).
  Its 95 golden prompts were rendered from the checkpoint's Jinja template.

- Images (`inkling_vision.rs`, `tidelm-image/src/inkling2.rs`, `flash_image.rs`): the tower (23
  pre-norm layers with LayerScale, 1024 wide, axial RoPE, a class token, a 2 x 2 merger to 4608) is
  resident on the pair's last rank only (0.65 GB of weights, 1.02 GB with scratch for 4,096 tokens):
  rank 0 sends a request's patches (`OP_INK_IMAGES`), the last rank encodes, a gather gives every
  rank the same feature rows, which replace the placeholder's embedding in the prefill. An image is
  a token a 40 x 40 px cell at its own size (alpha over a checkerboard, padded with -1), scaled down
  only past `GLM_VISION_MAX_TOKENS` (ours: the reference never resizes). A prompt with images is not
  saved to the disk store, and restores only what lies before its first image.

- The drafter (`inkling_draft.rs`, `cuda/inkling_draft.cu`; spec from SGLang's branch
  `inkling-2-small-support` @ `ea3742849fa4`): "DSpark", 5 dense layers over a block of 8 rows
  [anchor, 7 masks] that attend the block and context keys / values made from 8 taps of the target (the
  residual entering layers 5, 11, .., 41 and the final normed hidden, mixed by softmax(fuse_weight)),
  then a 7-step recurrent head whose correction goes through the unembedding: 7 drafts a round. Here
  it runs whole on the pair's last rank (2.74 GB of weights in FP8 with its own head copy; not split
  like the MTP layers of the GLM families: rank 0 keeps its memory, and the taps are already on every
  rank), the taps captured in every target pass there and written by position into rings (a verify
  pass's rejected rows are overwritten before they are read); rank 0 asks for a round (`OP_INK_DRAFT`)
  and a gather brings the 7 tokens and their probabilities back. Drafts are the drafter's top choices
  (for a sampled request: accepted with the target's probability of the token, the residual drawn
  otherwise - exact); the serve loop's gate and verify are the GLM families'. Its global layer keeps
  16,384 positions of context where the reference keeps all.

## Checks (2026-11-16)

- `tidelm-kernels/tests/inkling_gpu.rs`: each kernel against a CPU transcription of the reference, a
  sequence in two passes against one (the rings), prefill against decode, the two cuBLAS calls: 8 / 8.
- `tools/inkling_ref.py` (a PyTorch transcription of the reference, run in the serving image)
  against `TIDELM_INK_DUMP`, 27 tokens, layers 0-5, BF16, the engine's residents: rows agree to
  0.4-0.7% a layer. Outlier rows (5-27%) are one routed expert flipped at a near-tie, their error
  decaying over the next three rows (the convolutions); that run had 17% of the experts resident.
- `glm_run --check`: one-row decode, eight-row decode and prefill give the same predictions, eager and
  under graphs.
- The tower against `tools/inkling_vision_ref.py` (a PyTorch transcription; a synthetic 3 x 5-token
  image): 0.4% after the first block, 1.5% after the 23rd, 2.3% at the output, row cosine 0.9986
  minimum and 0.9997 mean. Through the model: a drawn image's shapes, colours and positions, the
  text of four quadrants of a 650 x 410 image (not a multiple of the cell) with their colours, and a
  follow-up turn that reuses the image's rows (not encoded again) all came back right. A 96-token
  image encodes in 0.02 s.
- The drafter against `tools/inkling_draft_ref.py` (a PyTorch transcription of the spec; fabricated
  taps, 40 positions of context, BF16): every stage within 1.5% (row cosine >= 0.9994), the 7 draft
  tokens the same, a near-tie included. Serving (FP8): 2.4 of 2.9-3.3 drafts accepted a round on two
  greedy chat requests, 24-25 -> 40-42 tok/s; temperature 0.6: 0.95 of 2.0, 26.8 tok/s.
- The other families: GLM-5.3-Flash 4 bpw on one box, the build before against this one, 10,151
  held-out tokens: the per-token NLL dumps are the same bytes. The GLM-5.3 pair (exlr8 `home`), a
  smoke test only: two short greedy chats on this build and the one before, each from its own boot
  with adaptation on. The 9-token answer is the same tokens; the 292- / 284-token one is correct code
  on both and parts at token 26 (a docstring's wording). Not a parity check: the first boot rewrote
  the history the second placed its residents from, and a missed expert changes the numbers. A frozen
  history with adaptation and hot reads off is still to do.

## Measurements (2026-11-16, the pair, 12,016 experts a rank; served at 12,091 with the drafter)

- Routing: a history of 116K prefilled tokens of `corpus/sources` puts 94.6% of the routed weight in
  the top 65.2% of the experts (86.5% in 48.8%, 73.5% in 32.6%). GLM-5.3's top 59% hold 72%.
- Held-out NLL, `corpus/heldout_quick.jsonl` as raw text (12,617 tokens): 1.1567 with BF16 dense
  weights, 1.1587 with FP8 (+0.0020 +- 0.0014 paired); coding 0.95 / top-1 76.4%, general 1.64 / 59.7%;
  2.38% of the routed weight missed, 1.13 hot reads a pass.
- Decode, plain (no drafter): 18-19 tok/s with BF16 dense weights and eager passes, 24-25 with FP8
  and graphs (two chat requests; not the README protocol).
- Uniform placement (no history) at 49% resident: 51% missed, and the model loops from the first
  sentence. The placement, not the engine: a history fixes it.
- Prefill deep in a context (the pair, live traffic): full layers' attention through cuBLAS ran
  39-65 tok/s for a turn's ~2,000 new tokens at 118-163K (40-60 s a turn) and stalled the other
  lane's decode to 3-4 tok/s meanwhile; its row blocks were sized to the score buffer, a few rows,
  each re-reading every key for each head. `ink_pf_flash` (fused, tensor cores) instead: 2,427
  tokens at 110K in 4.45 s (546 tok/s), a fresh 60K prompt at 895 tok/s (292 before), and 1.4-2.4 s
  for the 8 layers of a 2,300-row turn at 150K in the kernel's own timing (11-24 TFLOP/s).

## Open

The README protocol (bench/bench.py), the GLM-5.3 pair's parity on this build (beyond the smoke test
above), a history from chat traffic (tools/routing_session.py: decode routing, which a prefilled seed
never sees), sampled drafts (the
reference draws its drafts at the request's temperature and verifies against that distribution: more
accepted than a greedy draft's), the drafter's global window and a TP=2 split of it against one rank's,
measured; decode attention past ~100K (the verify rows each read every key: about +33 ms a step at 140K
against 20K; a lane's rows could share one read, as the drafter's attention does), FP8 key / value
caches, prefill tuning (the fused expert epilogue, batched GEMMs, the window layers' attention, the
fused kernel's loads), the tower against transformers' own classes once a release has them (the reference
here is a transcription, with RoPE theta 10000 and exact GELU taken from the config class's defaults),
image prompts in the disk store (their key would need the images' hashes).
