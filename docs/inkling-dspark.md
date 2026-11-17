# Inkling-2-Small drafter (`mtp.safetensors`, "DSpark") — implementation spec

Written 2026-11-16 from source reading, before the port: every numeric claim below is what the reference
code computes. Implemented by `rust/crates/tidelm-glm/src/inkling_draft.rs` and
`rust/crates/tidelm-kernels/cuda/inkling_draft.cu`; `tools/inkling_draft_ref.py` is section 8 in PyTorch, and
`inkling_draft_check` compares the two (2026-11-16: every stage within 1.5%, row cosine >= 0.9994, the 7
draft tokens equal). Section 9's questions were settled as: Q5 the norm of the unrounded sum, the residual
stored rounded (SGLang's `forward_native`); Q6 alpha rounded to bf16; Q7 query head h reads key / value head
h / 6, neox rotation halves. How the engine serves it (one rank, its context rings) is in docs/inkling.md.

## Sources and how they are cited

A tag's file is its repository path with `/` written `__` (`sglang__python__sglang__srt__models__dflash.py` is
SGLang's `python/sglang/srt/models/dflash.py`); line numbers are at the commits named here.
The reference is **SGLang branch `inkling-2-small-support`, commit `ea3742849fa4`** (2026-11-15). SGLang `main`
does not contain the Inkling 2 code (only the cookbook page). An independent re-implementation,
**`0xSero/inkling2-exl3` @ `d98291637e55`**, was read as a cross-check; where the two differ it is said so.

| Tag | File |
|---|---|
| MTP | `sglang__python__sglang__srt__models__inkling2_mtp.py` — the drafter model + Markov head (**the wiring**) |
| MTPK | `sglang__python__sglang__kernels__ops__inkling2__mtp.py` — its three Triton kernels |
| MTPW | `sglang__python__sglang__srt__speculative__inkling2_mtp_worker.py` — worker subclass |
| CFG2 | `sglang__python__sglang__srt__configs__inkling2.py` — `mtp_config` -> draft config |
| CFG1 | `sglang__python__sglang__srt__configs__inkling.py` |
| INK2 | `sglang__python__sglang__srt__models__inkling2.py` — target model, feature capture |
| ARES | `sglang__python__sglang__srt__models__inkling_common__inkling2_attn_res.py` |
| DFL | `sglang__python__sglang__srt__models__dflash.py` — `DFlashDraftModel`, the drafter's backbone |
| DFU | `sglang__python__sglang__srt__speculative__dflash_utils.py` |
| DSP | `sglang__python__sglang__srt__models__dspark.py` |
| DRAFT | `sglang__python__sglang__srt__speculative__dspark_components__dspark_draft.py` |
| VER | `...dspark_components__dspark_verify.py` |
| WRK | `...dspark_components__dspark_worker_v2.py` |
| KVI | `...dspark_components__dspark_kv_inject.py` |
| PLN | `...dspark_components__dspark_planner.py` |
| DCFG | `...dspark_components__dspark_config.py` |
| MS | `sglang__python__sglang__kernels__ops__speculative__dspark__markov_sample.py` |
| ACC | `sglang__python__sglang__kernels__ops__speculative__dspark__dspark_accept.py` |
| RS | `sglang__python__sglang__kernels__ops__speculative__reject_sampling.py` |
| DFK | `sglang__python__sglang__kernels__ops__speculative__dflash.py` |
| FAB | `sglang__python__sglang__srt__layers__attention__flashattention_backend.py` |
| LN | `sglang__python__sglang__srt__layers__layernorm.py` |
| ROPE / ROPEU | `sglang__python__sglang__srt__layers__rotary_embedding__base.py` / `...__utils.py` |
| QKNR | `sglang__python__sglang__kernels__ops__attention__table_qk_norm_rope.py` |
| ACT | `sglang__python__sglang__srt__layers__activation.py` |
| AUX | `sglang__python__sglang__srt__layers__aux_hidden_states.py` |
| ENV | `sglang__python__sglang__srt__environ.py` |
| SPI | `sglang__python__sglang__srt__speculative__spec_info.py` |
| X-DRAFT / X-MARKOV / X-ENG | `0xSero-inkling2-exl3__overlay__exllamav3__modules__inkling2_{draft,markov,engine}.py` |
| HF | transformers' `modeling_inkling2.py` (the target model) |
| CKPT | the header of the checkpoint's `mtp.safetensors` and its `config.json` |

Notation: `bf16(x)` = round to bfloat16 (then continue in fp32). `P` = number of committed positions that
already have target features = absolute position of the anchor token. `D = 4608`, `V = 201024` (padded vocab),
`Vu = 200058` (unpadded), `r = 256`.

---

## 1. Overview

**Where it is wired.** SGLang selects `Inkling2MTPWorker` (a `DSparkWorkerV2` subclass) when the target's
architecture is `Inkling2ForConditionalGeneration` and the algorithm is DSPARK [SPI 334-343]. The draft model class
is `Inkling2ForConditionalGenerationMTP(DSparkDraftModel)`; it loads only `mtp.safetensors`
(`allow_patterns_overrides = ["mtp.safetensors"]`) [MTP 103-106]. Its config is synthesized from the target's
`mtp_config` by `Inkling2MTPConfig.from_target` [CFG2 123-210; DCFG 240-248]. The target loader skips `mtp.*`
tensors [INK2 457].

**One draft round.**
- Inputs: the anchor token id (the last committed token, which the target has *not* yet processed); its absolute
  position `P`; the per-layer context K/V cache built from target features of positions `< P`; and `f_rnn`, one
  4608-vector derived from the target features of position `P-1`.
- Work: **one** drafter forward over a block of **8 rows** = `[anchor, mask x 7]` at positions `P..P+7`
  (5 dense layers, all 8 rows in parallel, bidirectional inside the block) [MTP 261-277; DRAFT 602-663];
  one unembed GEMM for rows 1..7; then a 7-step recurrent "Markov head" that corrects each row's logits using the
  previously drafted token [DRAFT 263-308].
- Output: **7 draft tokens** `d1..d7` (`gamma = block_size - 1 = 7`) [MTP 122-124; CFG2 201], and, when sampling,
  the 7 rows of corrected logits (needed for rejection sampling).
- The anchor row's own output hidden is discarded (`sample_from_anchor = False`): hidden rows `1..7` predict
  positions `P+1..P+7` [MTP 124; DRAFT 95-113, 698-712]. Row `j` (a mask slot at position `P+j`) predicts the token
  *at its own position*.

**"Semi-autoregressive".** The transformer is non-autoregressive: all 7 mask rows are computed in one pass and never
see drafted tokens. Token-to-token dependence inside the block is restored only by the cheap head: step `i` reads
the embedding (`markov_w1`) of the token drafted at step `i-1` plus a 256-wide RNN state, and adds a logit
correction. So: 1 transformer pass + 7 sequential head steps, each a 9728->768 linear and a 256->V GEMV
[MTP 72-97; DSP 153-155].

**Size.** 3.42 GB BF16 [CKPT]. Per layer: q 6144x4608, k/v/k_ctx/v_ctx 1024x4608 each, o 4608x6144,
gate/up 18432x4608, down 4608x18432 (~660 MB/layer, the MLP is 510 MB of it). Head: `markov_w1` 201024x256
(103 MB), `rnn_proj` 768x9728, `hidden_out` 4608x256.

**What the drafter does *not* have** (all present in the target; verified absent from DFL/MTP and from the tensor
list): no relative-position bias, no attention sink, no gate, no short convolutions, no block-attention-residual,
no `log_scaling`, no L2 clip, no value norm, no softcaps in the MLP, no confidence head (`confidence_head = None`
[MTP 130]), no own embedding or unembedding. It is a plain Qwen3-style pre-norm transformer with per-head QK
RMSNorm and RoPE [DFL 271-405, 927-1023].

---

## 2. Inputs from the target model

### 2.1 Which hidden states (`target_layer_ids = [5,11,17,23,29,35,41,48]`)

8 "taps", each a 4608-vector **per position**, captured for **every row the target processes** (all prompt rows at
prefill, all 8 rows at verify) [INK2 259-279; WRK 826-830; VER 319-324 `CaptureHiddenMode.FULL`].

- Tap `t` for id `k < 48`: `state.partial` **at the entry of layer `k`**, i.e. *before* layer `k` runs
  [INK2 271-274: capture happens before `layer(state, ...)`]. `partial` is the running residual of the *current
  6-layer attention-residual block* [ARES 147-156]: at a block boundary (`layer_idx % 6 == 0`) it is reset to
  `attn_out + mlp_out`, otherwise `partial += attn_out + mlp_out`. So tap id 5 = sum of the (post-sconv) attention
  and MLP outputs of layers 0..4 — it does **not** include the embedding and is **not** normed; id 11 = same for
  layers 6..10; ... id 41 = layers 36..40.
  In HF terms this is exactly the variable `hidden_states` passed into `self.layers[k]` in
  `Inkling2TextModel.forward` [HF 1293-1307], i.e. the return value of layer `k-1` [HF 805, 868].
- Tap for id 48 (`== num_hidden_layers`, a sentinel): the **final normed hidden**,
  `norm(final_input(block_states, hidden_states))` — the vector the unembed is applied to
  [INK2 275-277; CFG2 170-180 "followed by the final-norm sentinel"; INK2 388-391 "the post-final-norm hidden
  state"; HF 1309-1310].
- Each tap is cast to the unembed dtype (BF16) [INK2 273, 277] and the 8 taps are packed tap-major per row into
  `[rows, 8*4608]`, viewed as `[rows, 8, 4608]` in the order of `target_layer_ids` [AUX 29-90; MTP 298-300].
- Cross-check: X-ENG 447-468 documents the same thing: "inputs of layers 5..41, final-norm output".

### 2.2 Fusion (`fuse_weight [6, 8]`)

- `6 = num_draft_layers + 1`, `8 = number of taps` [MTP 136-139].
- `alpha = softmax(fuse_weight.float(), dim=-1)` (over the 8 taps), then **cast back to BF16** and kept
  [MTP 411-417]. Row `l` (0..4) is drafter layer `l`'s mixture; row 5 (`-1`) is the Markov head's mixture
  [MTP 306, 241].
  *Discrepancy:* X-DRAFT 157 keeps `alpha` in fp32 (no BF16 rounding). SGLang is the reference; the difference is
  one BF16 rounding of 48 scalars.
- `mix_rmsnorm(targets[R,8,D], alpha[8], w[D], eps)` [MTPK 7-34], all in fp32:
  `m = sum_t float(targets[:,t]) * float(alpha[t])` (taps accumulated in order 0..7);
  `out = bf16( m * rsqrt(sum(m*m)/D + eps) * float(w) )`. `eps = 1e-6` [MTP 121, 160-161].
  There is **no linear projection** (the generic DFlash `fc` / `hidden_norm` are set to `None` [MTP 128] and
  `project_target_hidden` raises [MTP 279-282]); the mixture is a convex combination of taps followed by a weighted
  RMSNorm.
- The norm weight `w` is `layers.l.dflare_norm.weight` for layer `l`, `markov_head.dflare_norm.weight` for the head.

### 2.3 What is derived and cached between rounds

Two things, nothing else:

**(a) Per-layer context K/V** for committed positions [MTP 284-345]. For each drafter layer `l` and each row with
target features at position `p`:
```
c  = mix_rmsnorm(feats, alpha[l], layers[l].dflare_norm.weight, 1e-6)      # [D] bf16
k  = bf16(c @ k_proj_ctx[l].T)  -> 8 heads x 128
v  = bf16(c @ v_proj_ctx[l].T)  -> 8 heads x 128                            # no norm, no scale, no RoPE
k  = RMSNorm_128(k per head; weight k_norm_ctx[l], eps 1e-6)                # bf16
k  = RoPE(k, position p)                                                     # bf16
store (k, v) at position p of layer l's cache
```
These are written (i) for **all prompt rows** at prefill [WRK 875-881; KVI 65-145], and (ii) after each verify,
for the **committed rows only** (`commit_len` rows starting at the anchor's position) [VER 389-434; MTP 313-330].
They overwrite whatever the block forward left in those slots (see 4.3).

Which positions are actually read later: layer 2 (global) reads **all** committed positions `0..P-1`; layers
0,1,3,4 (local) read only the last 1024 committed positions `max(0,P-1024)..P-1` (section 4.3). So local layers
need only a ring of >= 1024 + 8 positions; the global layer needs K/V for the whole context
(8 heads x 128 x 2 x 2 B = 4096 B/token in BF16). X-DRAFT 173-188 does exactly this (ring 1088 for local layers,
paged int8+scale for layer 2).

**(b) `f_rnn`**, one `[D]` BF16 vector per sequence [MTP 235-259]:
`f_rnn = mix_rmsnorm(feats_of_last_committed_row, alpha[5], markov_head.dflare_norm.weight, 1e-6)`.
- After prefill: the last prompt row [MTPW 80-89].
- After a verify: verify row `commit_len - 1` (the last committed row — the row whose target logits produced the
  next anchor) [MTPW 17-47; MTP 250-259].
So `f_rnn` is always the features of position `P-1`.

The RNN state is **not** carried across rounds (zeros at every block start, [MTP 76-85]). No drafter self-attention
K/V survives a round.

---

## 3. The drafter's input sequence

- Block = 8 rows [MTP 209-218]:
  - row 0: `E(anchor)` = the **target's normed embedding**: `embed_norm(embed[anchor])`
    [MTPW 57-58 -> INK2 398-399 -> INK2 245-251; HF 1215-1221 `Inkling2NormedEmbedding`]. Tensors:
    `model.llm.embed.weight` BF16 [201024,4608], `model.llm.embed_norm.weight` BF16 [4608] [CKPT].
  - rows 1..7: the raw `mask_embed` parameter `[4608]` (cast to the embedding dtype; **not** passed through
    `embed_norm`). The config's `mask_token_id = 0` is a placeholder that is never embedded [CFG2 203; MTP 212-217].
  - `forward()` always rebuilds the embeddings itself and ignores any `input_embeds` passed in [MTP 269].
- Positions: absolute, `P + j` for row `j` (`positions_2d = prefix_lens + arange(8)`; `prefix_lens` = committed
  length = `P`) [DRAFT 617-625; PLN 839-840]. The anchor sits at `P`, the masks at `P+1..P+7`.
- RoPE [DFL 324-351; ROPE 238-271; ROPEU 65-91]: `theta = 1e6` (`rope_theta`), rotary over the **full head dim**
  (128, 64 frequencies), **neox layout**: `inv_freq[i] = theta^(-2i/128)`, `i = 0..63`; with
  `x1 = x[..., :64]`, `x2 = x[..., 64:]`: `o1 = x1*cos - x2*sin`, `o2 = x2*cos + x1*sin`, angle
  `= pos * inv_freq[i]`. Applied to q and k only (never v), after the per-head norm. cos/sin tables are fp32 on
  CUDA [ROPE 120-125]. No position offset (`position_start = 0`).
- `block_size = 8` is fixed; the worker refuses any other value [MTP 163-165].

---

## 4. Each drafter layer

### 4.1 Layer skeleton [DFL 962-1023, 1223-1278]

Standard pre-norm residual stream, with the next norm fused onto each residual add:
```
res = x0 (the 8 input embeddings);  xn = RMSNorm(res; layers[0].input_layernorm)
for l in 0..4:
    a   = SelfAttn_l(xn)
    res = res + a ;  xn = RMSNorm(res; layers[l].post_attention_layernorm)
    m   = MLP_l(xn)
    res = res + m ;  xn = RMSNorm(res; layers[l+1].input_layernorm  if l < 4 else  norm)
hidden = xn            # [8, 4608] bf16, final-normed
```
`RMSNorm(x; w) = x * rsqrt(mean(x^2) + 1e-6) * w`, computed in fp32, output BF16 [LN 971-1020]. In the fused
add+norm the residual is stored as `bf16(res + delta)` while the norm is computed from the **unrounded fp32 sum**
[LN 980-1010, `forward_native`]. *Not verified for the CUDA kernel* (`fused_add_rmsnorm`, source not fetched);
X-DRAFT 327-335 instead norms the BF16-rounded sum. See open question Q5.

### 4.2 Projections and per-head norms [DFL 446-494; QKNR 57-87]

```
q = xn @ q_proj.T   -> [8, 48 heads, 128]
k = xn @ k_proj.T   -> [8,  8 heads, 128]
v = xn @ v_proj.T   -> [8,  8 heads, 128]          # used as is: no norm, no scale (v_scale is None)
q = RoPE(bf16(RMSNorm_128(q; q_norm)), pos)         # RMSNorm over each head's 128 dims, weight [128], eps 1e-6
k = RoPE(bf16(RMSNorm_128(k; k_norm)), pos)
```
- No biases anywhere (`q_bias`/`o_bias` must be false [MTP 114-117]; the DFlash attention bias defaults false).
- `q_norm` / `k_norm` / `k_norm_ctx` are ordinary weighted RMSNorms of size `head_dim` shared across heads
  [DFL 319-322; MTP 160], **not** L2 norms and not the target's "L2 clip".
- Rounding in the fused kernel: the normed value is rounded to BF16, the rotation is done in fp32 with fp32
  cos/sin, the result is rounded to BF16 [QKNR 70-87].

### 4.3 Attention: one softmax over [context K/V | the block's own K/V]

There is a **single attention**, not two. The context enters only as extra key/value rows in the same cache; the
block's own K/V (from `k_proj`/`v_proj`, 4.2) are written into the cache at positions `P..P+7` and then all 8 query
rows attend over the cache [FAB 1139-1182: `cache_seqlens = seq_lens + 8`].

- Keys/values at positions `< P` come from the **context path** (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`,
  section 2.3). Keys/values at positions `P..P+7` come from the **block path** (`k_proj`, `v_proj`, `k_norm`).
  Queries always come from `q_proj`/`q_norm` of the block rows. Context rows are never queries.
- **Not causal**: every block row sees all 8 block rows, including later ones
  (`is_causal = False` -> `AttentionType.ENCODER_ONLY`) [CFG2 192; DFL 229-263; FAB 1682-1686].
- Key range, the same for all 8 query rows of a block [FAB 1828-1875 with `sliding_window_size = 1024 - 1`
  from DFU 423-440; MTP 146]:
  - layer 2 (`full_attention`): positions `[0, P+8)`.
  - layers 0,1,3,4 (`local_layer_ids`, "block-anchored window"): positions `[max(0, P-1024), P+8)`, i.e. the
    **1024 committed positions before the anchor plus the 8 block rows**. The window is anchored at the block
    start, not at each query's own position
    (`start = cache_seqlens - query_len - (window_left + 1)` = `P + 8 - 8 - 1024`).
  - Cross-check: X-DRAFT 55-59 `lo = max(0, pa - 1024)`, `hi = pa + 7`.
- Score: `s = (q . k) * 128^-0.5` [DFL 353, 396-405]; plain softmax over the key range; no bias terms, no sink, no
  logit cap, no temperature / `log_scaling`.
- GQA: 48 query heads over 8 KV heads; query head `h` uses KV head `h // 6`. *The FA4 kernel was not read; this is
  the standard packing and X-DRAFT 47-53 (`hq = hk*G + g`) does the same.*
- Output: `o[8, 48*128] @ o_proj.T -> [8, 4608]`, no bias [DFL 311-317, 492-494].
- The block's own K/V are transient: after verify, committed positions are overwritten by context K/V (2.3) and
  uncommitted ones are overwritten by the next block before being read.

### 4.4 MLP [DFL 724-780; ACT 141-143]

`m = (silu(xn @ gate_proj.T) * (xn @ up_proj.T)) @ down_proj.T`, intermediate 18432, `hidden_act = "silu"`
[CFG2 190], no biases, **no softcaps**, not routed (dense). In SGLang gate/up are one merged GEMM
(`gate_up_proj`, first half gate) — an implementation detail.

### 4.5 Dtypes

Checkpoint and parameters must be BF16 [MTP 112-113, 374-377; CFG2 151-155]. Every GEMM is a BF16 GEMM with BF16
output; norms/RoPE/softmax compute in fp32 and round to BF16 at the points stated. The FA4 attention kernel's
internal softmax dtype was not read (X-DRAFT 80-95 uses fp32 scores and accumulators, BF16 P.V dot inputs).

---

## 5. Output head

### 5.1 Final norm and base logits

- `hidden = RMSNorm(res_final; norm.weight)` (the last fused add+norm in 4.1), take rows 1..7 -> `h[7, D]`
  [DFL 1264-1273; DRAFT 95-113].
- `base = bf16(h @ U.T)`, `U` = the **target's unembed** (`model.llm.unembed.weight`, BF16 [201024, 4608]; shared
  module, must be unquantized) [MTP 167-171, 200-207; DSP 124-142]. No scaling
  (`logits_mup_width_multiplier` is `None` for this model: not in `config.json` [CFG1 187; WRK 262-264]).
  The softcap and the vocabulary crop are **not** applied here; they are applied after the Markov bias (5.3).

### 5.2 Markov head `rnn_hidden` [MTP 31-100; MTPK 60-160]

Parameters: `markov_w1` = embedding `[V, 256]` of the *previous* token; `rnn_proj` = Linear(9728 -> 768, with
bias); `hidden_out` = Linear(256 -> 4608, with bias); `dflare_norm` (used only for `f_rnn`, 2.3b).

- `9728 = 256 + 256 + 4608 + 4608`: `z = [state | markov_w1[prev] | h_i | f_rnn]`, in that order [MTPK 138-139].
  `h_i` is the same final-normed drafter hidden used for `base[i]`; `f_rnn` is constant over the block.
- `768 = 3 x 256`: `raw = bf16(z @ rnn_proj.W.T + rnn_proj.b)`, chunked in order `[gate_raw | cand_raw | out_raw]`
  [MTPK 72-74].
- Cell [MTPK 76-81], each op rounded to BF16:
  ```
  gate   = bf16(1 / (1 + exp(-gate_raw)))
  state' = bf16( bf16(gate * state) + bf16( bf16(1 - gate) * bf16(tanh(cand_raw)) ) )
  latent = bf16(tanh(out_raw))
  ```
  A leaky/GRU-like single-gate cell. Note `latent` comes from `out_raw`, which was computed from the **old**
  state; `state'` only affects the next step. Initial state = zeros [MTP 76-85].
- **There is no `markov_w2`.** The head's output is a correction in *hidden space*, pushed through the target
  unembed: mathematically `logits_i = U @ (h_i + hidden_out.W @ latent + hidden_out.b)`. SGLang pre-folds this
  once at load, in fp64, rounded once to BF16 [MTP 46-67]:
  `fold_w = bf16(U @ hidden_out.W)` `[V, 256]`, `fold_b = bf16(U @ hidden_out.b)` `[V]`,
  and each step computes `bias = bf16(fold_b + latent @ fold_w.T)` [MTP 87-97]. `fold_w` plays the role of
  `markov_w2`. X-DRAFT 165-172 folds identically.

### 5.3 Step logits, softcap, vocabulary crop [MS 36-96; MTP 134-135]

```
logit = bf16(float(base[i]) + float(bias))
logit = bf16(30 * tanh(logit / 30))          # final_logit_softcapping = 30, applied to the corrected sum
logit[Vu:] = -inf                             # Vu = unpadded_vocab_size = 200058
```
(`step_softcap = 30.0`, `step_vocab_limit = 200058`: CFG2 185-189 with CFG1 128-139, 186.)

### 5.4 Order of sampling within a block [DRAFT 263-308]

Strictly left to right: `prev = anchor`; for `i = 0..6`: compute `(state', latent)` from
`(state, prev, h[i], f_rnn)`, form `logit_i`, pick `d_{i+1}`, set `prev = d_{i+1}`, `state = state'`.

Token choice per step [MS 100-146; DRAFT 147-177]:
- greedy request (`top_k <= 1`; SGLang maps temperature 0 to this): `argmax(logit_i)`, lowest index on ties.
- otherwise: Gumbel-max at the **request temperature** `T` (clamped >= 1e-5): `argmax(logit_i / T + G)`.
  top-k / top-p are *not* applied to the draft distribution. The 7 rows of `logit_i` are kept; the draft
  distribution used by the verifier is `q_i = softmax(logit_i / T)`.
- An env switch (`SGLANG_DSPARK_USE_GREEDY_DRAFT`, default off [ENV 571]) drafts by argmax even for sampled
  requests and treats `q` as one-hot.

---

## 6. The draft / verify loop

State per sequence: committed length `P`, the anchor token ("bonus" of the previous step), context K/V, `f_rnn`.

1. **Prefill** [WRK 810-891; MTPW 80-89]: target runs the prompt with feature capture on all rows; context K/V are
   written for all prompt positions; `f_rnn` from the last prompt row; the target's first sampled token becomes the
   anchor; `P = prompt length`.
2. **Draft** (sections 3-5): 7 tokens `d1..d7`.
3. **Verify** [WRK 1110-1114, 1173-1179; VER 306-358]: the target runs **8 rows** `[anchor, d1..d7]` at positions
   `P..P+7` in one forward (default mode `SGLANG_RAGGED_VERIFY_MODE = "static"`: always all 8 [ENV 1809]) and must
   return, **per row**: the next-token logits (softcap 30, `>= Vu` masked to `-inf` [INK2 184-196]) and the 8 taps
   (section 2.1). Row `j`'s logits predict position `P+j+1`.
4. **Accept**:
   - Greedy requests — exact match [ACC 807-832; DFK 399-452]: `correct_len` = number of leading `j` in `1..7`
     with `d_j == argmax(target_logits[j-1])`; `bonus = argmax(target_logits[correct_len])`.
   - Sampled requests — standard speculative (rejection) sampling [VER 1350-1438; ACC 108-229; RS classic kernel]:
     `p_j` = target distribution at row `j-1` (temperature, plus top-k/top-p if requested), `q_j` from 5.4;
     accept `d_j` while `u_j * q_j(d_j) < p_j(d_j)`; on the first rejection sample the bonus from
     `normalize(max(p - q, 0))` of that row; if all 7 are accepted sample the bonus from `p` of row 7.
     (An optional "block-wise rejection" variant and block verification exist behind flags that are off by default
     [ENV 624-626]; not needed.)
   - Mixed batches pick per row by the greedy mask.
5. **Commit** [ACC 928-942; VER 1152-1234]: `commit_len = correct_len + 1` (the anchor + accepted drafts);
   `P += commit_len`; tokens emitted this round = `d1..d_correct_len, bonus` (1..8 tokens); new anchor = `bonus`.
6. **Drafter state after partial acceptance** [VER 389-434; MTPW 17-47]: write context K/V (2.3a) from the taps of
   verify rows `0..commit_len-1` at positions `old P .. old P + commit_len - 1`; set `f_rnn` from verify row
   `commit_len - 1`. Rows `>= commit_len` are dropped (target KV and features alike). Nothing else: no RNN state,
   no block K/V, no hidden is reused.

**No confidence head, no dynamic block length** in the reference configuration: the model sets
`confidence_head = None` [MTP 130], static mode builds no estimator [WRK 558-563; DCFG 39-52], the block is always
8 rows and all 7 drafts are verified. (X-ENG/README add their own adaptive verify width from the corrected-softmax
probability of each drafted token; that is an engine extension, not part of the checkpoint.)

Requirements the SGLang worker asserts, for orientation: draft sliding window + 1 <= target sliding window
[MTPW 71-76]; no PD-disaggregation [MTPW 52-53].

---

## 7. Tensor table (`mtp.safetensors`, all BF16) [CKPT]

`N` = 0..4.

| Name | Shape | Role | Used in |
|---|---|---|---|
| `fuse_weight` | [6, 8] | tap-mixture logits; softmax over dim 1; rows 0-4 = layer N, row 5 = Markov head | 2.2 |
| `mask_embed` | [4608] | input embedding of the 7 mask rows (un-normed) | 3 |
| `norm.weight` | [4608] | final RMSNorm before the unembed | 4.1 / 5.1 |
| `layers.N.input_layernorm.weight` | [4608] | pre-attention RMSNorm | 4.1 |
| `layers.N.post_attention_layernorm.weight` | [4608] | pre-MLP RMSNorm | 4.1 |
| `layers.N.dflare_norm.weight` | [4608] | RMSNorm weight on the mixed target features feeding layer N's context K/V | 2.2, 2.3a |
| `layers.N.self_attn.q_proj.weight` | [6144, 4608] | block rows -> 48 query heads | 4.2 |
| `layers.N.self_attn.k_proj.weight` | [1024, 4608] | block rows -> 8 key heads | 4.2 |
| `layers.N.self_attn.v_proj.weight` | [1024, 4608] | block rows -> 8 value heads | 4.2 |
| `layers.N.self_attn.o_proj.weight` | [4608, 6144] | attention output projection | 4.3 |
| `layers.N.self_attn.q_norm.weight` | [128] | per-head RMSNorm weight for q | 4.2 |
| `layers.N.self_attn.k_norm.weight` | [128] | per-head RMSNorm weight for block-row k | 4.2 |
| `layers.N.self_attn.k_proj_ctx.weight` | [1024, 4608] | mixed target features -> 8 context key heads | 2.3a |
| `layers.N.self_attn.v_proj_ctx.weight` | [1024, 4608] | mixed target features -> 8 context value heads | 2.3a |
| `layers.N.self_attn.k_norm_ctx.weight` | [128] | per-head RMSNorm weight for context k | 2.3a |
| `layers.N.mlp.gate_proj.weight` | [18432, 4608] | SwiGLU gate | 4.4 |
| `layers.N.mlp.up_proj.weight` | [18432, 4608] | SwiGLU up | 4.4 |
| `layers.N.mlp.down_proj.weight` | [4608, 18432] | SwiGLU down | 4.4 |
| `markov_head.markov_w1.weight` | [201024, 256] | embedding of the previously drafted token | 5.2 |
| `markov_head.rnn_proj.weight` / `.bias` | [768, 9728] / [768] | `[state\|w1[prev]\|h\|f_rnn]` -> `[gate\|cand\|out]` | 5.2 |
| `markov_head.hidden_out.weight` / `.bias` | [4608, 256] / [4608] | latent -> hidden-space logit correction (folded with the unembed) | 5.2 |
| `markov_head.dflare_norm.weight` | [4608] | RMSNorm weight on the mixed target features giving `f_rnn` | 2.3b |

84 tensors total (5 x 15 + 9); the SGLang loader rejects any unrecognized or missing name [MTP 347-410].
From the target: `model.llm.embed.weight`, `model.llm.embed_norm.weight`, `model.llm.unembed.weight`.

---

## 8. Reference algorithm (numpy-style)

```python
EPS = 1e-6; D = 4608; R = 256; VU = 200058; W = 1024; LOCAL = {0, 1, 3, 4}

def rms(x, w):                      # fp32 math, BF16 out; x[..., n], w[n]
    xf = f32(x)
    return bf16(xf * rsqrt(mean(xf * xf, -1, keepdims=True) + EPS) * f32(w))

def rope(x, pos):                   # x[rows, heads, 128] bf16, pos[rows]; neox, theta 1e6
    inv = 1e6 ** (-arange(0, 128, 2, f32) / 128)          # [64]
    ang = f32(pos)[:, None] * inv[None]                    # [rows, 64]
    c, s = cos(ang)[:, None], sin(ang)[:, None]
    x1, x2 = f32(x[..., :64]), f32(x[..., 64:])
    return bf16(concat([x1 * c - x2 * s, x2 * c + x1 * s], -1))

# ---- load time
alpha  = bf16(softmax(f32(fuse_weight), -1))               # [6, 8]   (SGLang rounds; exl3 does not)
fold_w = bf16(f64(U) @ f64(hidden_out.W))                  # [V, 256]
fold_b = bf16(f64(U) @ f64(hidden_out.b))                  # [V]

def mix_rms(feats, a, w):                                  # feats[rows, 8, D] bf16
    m = zeros(rows, D, f32)
    for t in range(8): m += f32(feats[:, t]) * f32(a[t])
    return bf16(m * rsqrt(sum(m * m, -1, keepdims=True) / D + EPS) * f32(w))

# ---- whenever the target produced features for committed rows (prefill rows; verify rows 0..commit_len-1)
def store_context(seq, feats, pos):                        # feats[rows, 8, D], pos[rows] absolute
    for l in range(5):
        c = mix_rms(feats, alpha[l], L[l].dflare_norm)
        k = bf16(c @ L[l].k_proj_ctx.T).reshape(-1, 8, 128)
        v = bf16(c @ L[l].v_proj_ctx.T).reshape(-1, 8, 128)
        k = rope(rms(k, L[l].k_norm_ctx), pos)
        seq.K[l][pos], seq.V[l][pos] = k, v                # local layers: only the last 1024(+8) are ever read
    seq.f_rnn = mix_rms(feats[-1:], alpha[5], markov_head.dflare_norm)[0]     # last committed row

# ---- one draft round
def draft(seq, anchor, P, temperature=None):
    x = stack([embed_norm(embed[anchor])] + [mask_embed] * 7)                 # [8, D] bf16
    pos = P + arange(8)
    res = x; xn = rms(res, L[0].input_layernorm)
    for l in range(5):
        q = bf16(xn @ L[l].q_proj.T).reshape(8, 48, 128)
        k = bf16(xn @ L[l].k_proj.T).reshape(8, 8, 128)
        v = bf16(xn @ L[l].v_proj.T).reshape(8, 8, 128)
        q = rope(rms(q, L[l].q_norm), pos)
        k = rope(rms(k, L[l].k_norm), pos)
        lo = max(0, P - W) if l in LOCAL else 0
        K = concat([seq.K[l][lo:P], k]); Vv = concat([seq.V[l][lo:P], v])    # [n, 8, 128]
        o = zeros(8, 48, 128)
        for h in range(48):
            g = h // 6
            s = (f32(q[:, h]) @ f32(K[:, g]).T) * 128 ** -0.5                # [8, n]; no mask, no bias
            o[:, h] = softmax(s, -1) @ f32(Vv[:, g])
        a = bf16(bf16(o).reshape(8, 6144) @ L[l].o_proj.T)
        s1 = f32(res) + f32(a); res = bf16(s1)
        xn = rms(s1, L[l].post_attention_layernorm)                           # norm of the unrounded sum (Q5)
        m = bf16(bf16(silu(f32(xn @ L[l].gate_proj.T)) * f32(xn @ L[l].up_proj.T)) @ L[l].down_proj.T)
        s2 = f32(res) + f32(m); res = bf16(s2)
        xn = rms(s2, L[l + 1].input_layernorm if l < 4 else norm)
    h = xn[1:8]                                                               # [7, D]
    base = bf16(h @ U.T)                                                      # [7, V]

    state = zeros(R); prev = anchor; drafts = []; step_logits = []
    for i in range(7):
        z   = concat([state, markov_w1[prev], h[i], seq.f_rnn])               # [9728] bf16
        raw = bf16(z @ rnn_proj.W.T + rnn_proj.b)                             # [768]
        g_raw, c_raw, o_raw = raw[:256], raw[256:512], raw[512:]
        gate   = bf16(1 / (1 + exp(-g_raw)))
        new    = bf16(bf16(gate * state) + bf16(bf16(1 - gate) * bf16(tanh(c_raw))))
        latent = bf16(tanh(o_raw))
        bias   = bf16(fold_b + latent @ fold_w.T)                             # [V]
        y = bf16(f32(base[i]) + f32(bias))
        y = bf16(30 * tanh(y / 30))
        y[VU:] = -inf
        prev = argmax(y) if temperature is None else argmax(y / temperature + gumbel(V))
        state = new
        drafts.append(prev); step_logits.append(y)
    return drafts, step_logits            # q_i = softmax(step_logits[i] / temperature) for rejection sampling

# ---- round driver (greedy)
drafts, _ = draft(seq, anchor, P)
logits, feats = target_verify([anchor] + drafts, positions=P + arange(8))     # 8 rows: logits + [8, 8, D] taps
pred = argmax(logits, -1)
c = 0
while c < 7 and drafts[c] == pred[c]: c += 1
emit(drafts[:c] + [pred[c]])
store_context(seq, feats[:c + 1], P + arange(c + 1))                          # also sets f_rnn from row c
P += c + 1; anchor = pred[c]                                                  # and truncate target state to P
```

### Numerical details that matter for matching the reference

1. `eps = 1e-6` for every norm (layer norms, per-head q/k/k_ctx norms, both `dflare_norm` uses), added to the
   mean square before `rsqrt`.
2. Attention scale `128^-0.5 = 0.08838834764831845`; MLP activation plain SiLU; no softcap inside the drafter.
3. Softcap 30 is applied to `bf16(base + bias)`, **after** the Markov correction, then rounded to BF16 again;
   columns `>= 200058` become `-inf` after the softcap.
4. RoPE: theta 1e6, full 128-dim, neox halves, absolute positions; on q and on both key paths; BF16 rounding
   after the norm and again after the rotation; cos/sin in fp32.
5. `alpha` is rounded to BF16 after the softmax (SGLang); the mix itself accumulates in fp32, taps in order.
6. Mixed features are stored BF16 before the `*_proj_ctx` GEMM and before the RNN.
7. RNN: `raw` is BF16 (output of a BF16 linear); `sigmoid` as `1/(1+exp(-x))`; every intermediate
   (`gate`, `tanh(cand)`, `1-gate`, both products, their sum, `latent`) is rounded to BF16 [MTPK 76-81].
8. `fold_w`, `fold_b` computed in fp64 from the BF16 unembed and `hidden_out`, rounded once.
9. Argmax tie-break: lowest token id [MS 139-146].
10. `markov_w1` is indexed by the **previous** token: the anchor at step 0.
11. Tap dtype BF16; taps for ids < 48 are un-normed block partials, tap 48 is post-final-norm.
12. There is no `log_scaling` anywhere in the drafter.

Not reproducible bit-for-bit without the vendor kernels: accumulation order inside BF16 GEMMs and inside the
attention kernel. Expect agreement of logits to BF16 noise, and identical argmax tokens except at near-ties.

---

## 9. Open questions for the porter (most blocking first)

**Q1. No golden vectors.** Nothing here was executed. Before trusting a port, a plain-PyTorch reference of
section 8 should be run against the real weights and compared with the engine (the spec is complete enough to
write that reference: every tensor in the checkpoint has a defined use and every operation is specified). The
only external sanity number available is 0xSero's reported acceptance length of 2.50-2.72 tokens/round on their
EXL3 build of this model (their README, lines 112 and 179) — a different engine and workload, usable only as an
order-of-magnitude check, not a target.

**Q2. Feature mismatch with a quantized target.** The drafter was trained on features of the full-precision
target; here taps come from a 4 bpw EXL3 target. All 8 taps (7 un-normed block partials + the final normed
hidden) feed every layer and the RNN. Whether acceptance holds is an empirical question this spec cannot answer.

**Q3. Context K/V persistence vs prefix cache / paged KV / swap-to-disk.** Drafting at position `P` needs, per
layer, context K/V for the last 1024 committed positions (layers 0,1,3,4) and for *all* committed positions
(layer 2, 4096 B/token in BF16). These can only be produced from target taps at the time the target processes a
row. If the engine reuses a cached prefix without re-running the target, the drafter's layer-2 context for that
prefix must have been stored with it, or the drafter runs with missing context (behaviour untested; the reference
always has it: its draft pool shares slot ids with the target pool [KVI 106-108]). Decide: persist layer-2 draft
K/V alongside target KV, or accept degraded drafts after a cache hit, or measure a truncated layer-2 context.

**Q4. Prefill tap volume.** Capturing 8 x 4608 BF16 per prompt row is 73.7 KB/row transient. It can be reduced
row-chunk by row-chunk into the 5 context K/V sets immediately (20.5 KB/row if all layers are kept for all rows;
only 4.1 KB/row persistent + a 1032-row ring x 4 layers if local layers keep only their window). The reference
does the reduction inside the prefill step [WRK 852-881].

**Q5. Norm input after a residual add: rounded or unrounded sum.** SGLang's `forward_native` norms the fp32 sum
and stores the residual as BF16 [LN 980-1010]; the CUDA `fused_add_rmsnorm` kernel was not read. 0xSero norms the
BF16-rounded sum. The difference is below BF16 resolution of the output in almost all elements; pick one and do
not expect bit equality with either.

**Q6. `alpha` rounding** (BF16 in SGLang, fp32 in 0xSero) — same class as Q5.

**Q7. GQA head mapping and attention-kernel precision** come from FA4, whose source was not read. `h // 6` is the
standard layout and is what 0xSero implements; if acceptance is near zero, this and the RoPE layout (neox halves
vs interleaved pairs) are the first two things to flip in the PyTorch reference.

**Q8. Sampling.** For non-greedy requests the reference drafts by Gumbel-max at the request temperature and
verifies by rejection sampling against `q = softmax(step_logits / T)`; this requires keeping 7 x 201024 corrected
logits per sequence per round. A greedy-only first port (exact-match accept) avoids this. Drafting by argmax
while the target samples is also valid with `q` one-hot (accept with probability `p(d)`).

**Q9. Verify width.** The reference always verifies all 8 rows. A shorter verify (fewer drafts) is an engine
choice and does not change the drafter: the block forward must still run all 8 rows (mask rows attend to each
other bidirectionally, so truncating the block changes every row's hidden).

**Q10. Other references not consulted.** 0xSero's file header says its numerics follow
`vllm.models.inkling2.dspark`, and its NOTICE cites NVIDIA `hyperflare-inference` (branch `testing/inkling2`).
Neither was fetched; they would be a third and fourth opinion on Q5-Q7 if needed. The SGLang code is on a
development branch and may change before it is merged.
