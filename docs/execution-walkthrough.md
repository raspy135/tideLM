# Execution walkthrough: what runs, in what order, and how long

A guided tour of the prefill and decode paths, written for someone who is strong in
Python/CUDA-adjacent systems but new to transformer inference. Every claim about
*order* is read from the source, with `file:line`. Every number is either marked
*measured* (taken from an artifact in `results/` or a doc) or *estimated* (arithmetic
from a measured envelope). Read
[architecture.md](architecture.md) for why the design is shaped this way.

---

## 0. Vocabulary in one screen (skip if you know inference)

| term | what it means here |
|---|---|
| **token** | one vocabulary id; `vocab_size=129280`. The model turns a sequence of ids into a sequence of next-token distributions. |
| **prefill** | run the whole prompt through the model once, in parallel. Shape is `[T_prompt, model]`. Compute-bound. Produces the KV cache and the first logits. |
| **decode** | produce one (or a few) tokens at a time, reusing the KV cache. Shape is `[T_verify≈6, model]`. Memory-bandwidth-bound. |
| **KV cache** | per-layer key/value tensors for every position already processed, so decode doesn't recompute the past. Here it is MLA-style: one 512-dim latent per token, not full K and V. |
| **MoE** | Mixture of Experts. Each token is routed to the top-k (k=6) of 384 small FFNs ("experts") per layer. Only those experts run. The weights are the 288 GB problem. |
| **router / gate** | the tiny linear + softmax that picks the top-6 experts and their weights. |
| **GEMM** | matrix multiply (`C = A @ B`). "fp8/fp4/bf16 GEMM" = the stored dtype of the weights. |
| **kernel launch** | one CPU→GPU work submission. Hundreds per token. Launch overhead matters at T=1. |
| **CUDA graph** | a recorded sequence of kernels replayed with one launch. The graphed decode path replays ~3 graphs per step instead of ~1000 launches. |
| **TP2 / EP2** | two DGX Sparks. **TP** splits a tensor across ranks and all-reduces the result; **EP** gives each rank half the experts. This fork runs both. |
| **DSpark** | a resident 3-block drafter that proposes 5 tokens; the main model verifies them in one pass. Speculative decoding: lossless, faster when drafts are accepted. |
| **Engram** | an n-gram lookup table on layers 1 and 14. 24 rows/token/layer, keyed by token ids alone (no forward pass needed to know the addresses). |

Two model stacks live in one process:

* **main**: `n_layers=40` backbone layers, indices `0..39`.
* **drafter**: 3 MTP blocks, indices `40..42`, own 128-expert MoE each, always resident.

Layer constants (from `inference/config.json`): `dim=5120`, `hc_mult=4`
(hyper-connection stream width), `n_heads=64`, `head_dim=512`, `window_size=128`,
`n_routed_experts=384`, `n_activated_experts=6`, `compress_ratios = [0,0, 2×18, 1×20, 0,0,0]`,
`kv_source_layers=(2,8,14,20)`, `engram_layer_ids=(1,14)`,
`dspark_target_layer_ids=(37,38,39)`, `candidate_source_layer=20`.

Size of the job, for intuition:

```
T=1 decode, per emitted token:
  40 layers × (attention + 6 routed experts + 1 shared expert)
  + 3 drafter blocks × (attention + 3 of 128 experts)
  ≈ 1000 CUDA kernel launches, ~100 ms of GPU work
```

---

## 1. The two phases at a glance

```
                          PREFILL                              DECODE (per verify step)
input shape               [T_prompt, ...]                      [6, ...]  (1 tok + 5 drafts)
chunking                  DSV41_PREFILL_CHUNK=1024 (default 2048)  none
layers run                0..20 encoder, then 21..39 replay   0..39 every step
KV writes                 every position                       6 positions, rolling window
bottleneck                GPU compute + EP2 grouping            memory bandwidth + NVMe misses
measured envelope         ~460-950 tok/s                       ~19-25 tok/s
per-token wall            ~1.1-2.2 ms (fast) / ~10-20 ms (cold) ~37-51 ms
```

The single most important structural fact: **prefill only runs the encoder half of
the prompt.** Under CED (Cross-layer Encoder-Decoder, `swa_replay=True`, the default),
layers 0..20 write all the global compressed KV the decoder needs; the decoder's only
per-position state is its 128-token sliding window, which is replayed once over the
prompt's tail. The code for this is `Model.forward(..., encoder_only=True)` at
`engine/model.py:1215` and `Model.decoder_replay` at `engine/model.py:1176`.

---

## 2. Boot (once, before the port opens — no request yet)

`V41Engine.__init__` (`engine/v41_engine.py:454`) then `server/app.py:main`.

| order | what | where | measured time |
|---|---|---|---|
| 1 | read `model.safetensors.index.json`, `config.json` | `v41_engine.py:475` | ms |
| 2 | init NCCL/Gloo process group | `dist.py::EPDistributed` | seconds |
| 3 | size + allocate expert arena | `v41_engine.py:610-666` | — |
| 4 | build keep mask, broadcast to peer, boot guard | `v41_engine.py:787-905` | — |
| 5 | load 19 GB of non-expert weights | `model.Weights.__init__` | ~63 s (doc) |
| 6 | load all 384 DSpark experts (7.2 GB) | `v41_engine.py:666` | — |
| 7 | **warm start the LRU** from routing trace | `store.warm_start` `experts.py:497` | ~16 s / 73 GB at 4.6 GB/s |
| 8 | build device slot LUT (all-resident) | `FastDecoder.build_lut` `fastdecode.py:535` | ~1 s |
| 9 | capture CUDA graphs (warm-up + capture) | `FastDecoder.capture` `fastdecode.py:638` | seconds |

The arena sizing is deliberately paranoid (`v41_engine.py:610`): `torch.cuda.mem_get_info()`
undercounts free memory after a download because it counts page cache, so the engine takes
`max(mem_get_info, /proc/meminfo MemAvailable)` and keeps a hard `keep_free_gb` floor.
Getting this wrong gives a 23 GB arena and a permanently NVMe-bound server with no error.

**Graph capture** is where the interesting boot cost is. `capture(S_parity, index_bucket)`
does one warm-up pass and then records. With the device LUT it records **whole segments
of layers**, not one graph per layer:

```
bounds = {0, 1, 14, 40}  (0, each engram layer boundary, n_layers)
→ 3 segment graphs: [0,1), [1,14), [14,40)
41 replays/step become 3.
```

Plus two drafter graphs (greedy and sampled) and one graph per `(S%2, index bucket)`.
The `S%2` parity is required because the ratio-2 key compressor groups positions in pairs
around one pending slot (`fastdecode.py:71`). Context buckets are powers of two
(`_index_bucket`, `fastdecode.py:93`) so the number of resident graph variants is
logarithmic rather than one per token.

---

## 3. Request lifecycle

```
HTTP POST /v1/chat/completions
  server/app.py Handler._chat / _completions
    State.generate(prompt_ids, sampling, ...)              server/app.py:611
      engine.generate(prompt_ids, ...)                     v41_engine.py:1201   ⏎ generator
        engine._generate(...)                              v41_engine.py:1219
          engine._decode_loop(...)                         v41_engine.py:1810
            [ PREFILL ] ─────────────────────────────► first token
            [ DECODE  ] ──── yields bursts of tokens ─► ...
      IncrementalDetokenizer.push(ids)                     server/app.py:356
      OutputRouter.feed(text)                              server/app.py:398
    SSE / JSON to client
```

`generate` holds `self.lock` and is a **generator**: each `yield` is a burst of one or
more settled token ids. The server detokenizes and applies stop strings/tool-call parsing
*outside* the engine. The engine is single-sequence by default
(`DSV41_MAX_CONCURRENCY=1`).

`_reset()` (`v41_engine.py:1009`) zeroes every counter at the top of each request, because
the stats epilogue runs in a `finally` and an aborted request still has to report its own
work rather than the previous request's.

---

## 4. Prefill, in execution order

Driver: `V41Engine._decode_loop` (`v41_engine.py:1810`), lines 1828–1945.

### 4.1 Before the first chunk

```
m.begin_prompt()                     # empties the SWA replay buffer (only if prefix_start==0)
hashes_t = m.hash_state(ids[None], 0, mask)          # GPU: n-gram hashes for the WHOLE prompt
ra = self._engram_readahead(...)                      # CPU thread pool: start NVMe reads NOW
```

`hash_state` (`engine/engram.py:396::make_hash_state`) maps token ids through a compressed
vocab (NFKC+lowercase, 99092 ids) and XORs previous ids under per-`(layer,lookback)`
multipliers. **The addresses depend on ids alone**, so the engine hashes the entire prompt
up front (`v41_engine.py:1863-1865`) and lets chunk *k+1*'s rows read off NVMe while chunk
*k* is on the GPU. Hashing per-chunk is bit-identical but costs ~1 s of idle GPU per chunk.

`_prefill_spans` (`v41_engine.py:1755`) computes chunk boundaries: `MAX_CHUNK` tokens,
except no image span may straddle one.

### 4.2 Per chunk (CED path, the default)

For each span `(s, e)`:

```
_fwd(s, chunk, need_logits=False, encoder_only=True)   # v41_engine.py:1867
  Model.forward(chunk, s, prefill=True, encoder_only=True)   engine/model.py:1215
```

Inside `Model.forward` (`model.py:1238-1294`):

1. `hashes = m.hash_state(...)` — only if the caller didn't pass them (here it did).
2. `e = self.W.embed[ids]` — embedding gather.
3. `h = e.unsqueeze(1).repeat(1, hc_mult, 1)` — expand to the 4-wide hyper-connection stream.
4. loop `L = 0 .. 20`:

   a. **Engram** (`L in (1,14)`): `rows = get_rows(L, hashes[:, li, :])` →
      `R.engram_forward(h, rows, ...)` (`v41_ref.py:791`).
      The read is already in flight; this line blocks on the future if it isn't.
      Row = 256 B FP8 + 8 B UE8M0 scale, 24 rows/token/layer.

   b. **`Model.block(h, pre_mix, w, L, S, ...)`** (`model.py:1104`):

      ```
      _hc_mixes  →  hc_split_sinkhorn  (rms_rsqrt + mm(hc_fn) + 20 Sinkhorn iters)
      _hc_pre_rn_fused(h, pre_mix, attn_norm)      # hc_ops.py:99 Triton kernel
      Model.attention(y, w, L, ...)                # model.py:484
          q  = rmsnorm(qlinear(x, wq_a)) ; q = qlinear(q, wq_b)     # fp8 Triton
          q[..., -64:] = apply_rotary(...)                          # RoPE
          kv = rmsnorm(qlinear(x, wkv)) ; kv = apply_rotary(...)
          ring[pos % 128] = kv ; wkv = ring[wpos % 128]              # window gather
          if w.ratio:                                               # layers 2..39
              Model._compressed(...)                                # model.py:627
                  # source layers (2,8,14,20): write global KV
                  # indexer layers (2,8,14,20): Model._indexer → top-512 scores
                  # others: reuse sh.ckv / sh.topk
          o = _prefill_attn(...) or _softmax_attn(...)              # sinked softmax
          o = wo_a_proj(o, w.wo_a, tiled=True) ; out = qlinear(o, wo_b)  # grouped + fp8
      _hc_post_fused(y, residual, attn_post, attn_comb)  # hc_ops.py:59
      _hc_mixes ; _hc_pre_rn_fused(h, attn_pre, ffn_norm)
      Model.moe(y, w, L, prefill=True, ...)        # model.py:971
          scores  = softplus(mm(y.float(), gate_w)).sqrt()   # fp32, near-ties matter
          logits  = scores + gate_bias
          indices = logits.topk(6)                            # top-6 of 384
          weights = normalize(scores.gather(indices)) * route_scale
          slots   = store.resolve(L, indices, prefill=True)   # experts.py:420
          routed  = moe_fn(y, slots, weights, arena, ...)     # fp4_moe.py:558
              _moe_up_kernel   (grid (pair-block, n-block))   # gate+up+SwiGLU
              _moe_down_kernel (scatter-add fp32)             # down proj
          shared  = expert_ffn(y, sh_w1, sh_w2, sh_w3)        # dense, every token
          out     = routed + shared
      _hc_post_fused(out, residual, ffn_post, ffn_comb)
      ```

   c. if `L == 20`: `_rep_keep(h, pre_mix, sh, S, T)` caches the last 128 rows for replay.

5. `self.c.len = S + T`.

Then per-chunk bookkeeping in `_decode_loop`: `_snapshot_prefix` (optional), `replicas.finish_probe`.

### 4.3 After the last chunk

```
self._save_prefix(prompt)                     # in-memory prefix snapshot
self.prefix_disk.save(prompt)                 # optional, persistent
logits, mh, s_rep = m.decoder_replay()        # engine/model.py:1176
    # layers 21..39 over the last 128 tokens + Model.head
m.dspark_seed(mh, s_rep)                      # write drafter window KV (model.py:1305)
self.maintain_inline(P - prefix_start)        # adaptation swaps, if enabled
tok = sample_probs(logits[-1], temperature, top_p) → multinomial/argmax
yield [tok]
```

**Why replay is only 128 tokens:** the decoder's global KV is projected from layer 20's
hidden state (CED), so the only thing the decoder layers owe the first decode steps is their
own 128-token sliding window. Replay runs them over that tail, not the whole prompt.

### 4.4 Prefill phase budget (measured)

From `results/prefill-before.json`, `results/prefill-stream-1-2-4.json` (EP2, 4509-token
prompt, 4 chunks of 1024/2048, 19 output tokens). These are host wall-clock counters from
`last_stats`, so they include waits; do not add across rows.

| phase | slow run (before) | fast run (stream) | per prompt token (fast) |
|---|---:|---:|---:|
| total `prefill_s` | 9.916 s | 5.545 s | **1.23 ms** |
| `prefill_tok_s` | 455 | 813 | — |
| `moe_s` (40 layers × chunks) | 6.99 s | 0.37 s | 0.08 ms |
| `attn_s` | 0.30 s | 2.62 s | 0.58 ms |
| `engram_s` (host, 94,973 rows) | 2.58 s | 1.53 s | 0.34 ms |
| `ep_s` / 82 calls | 0.01 s / 118 µs | 0.02 s / 284 µs | — |
| `engram_read_s` | 2.29 s | 0.28 s | 0.06 ms |

The `moe_s` vs `attn_s` swap between the two runs is the point: the slow run is dominated by
MoE host grouping/dispatch (`prefill-moe-timing.md` measured 4.15 s of host grouping on a cold
request versus 44 ms of GPU grouping envelope), the fast run is attention-bound. Neither is
stable across runs. The README's 7,709-token runs measured 489 / 576 / 954 tok/s
(15.8 / 13.4 / 8.1 s) with zero prefix reuse.

The MoE subphases (`DSV41_PREFILL_MOE_TIMING=1`, `engine/prefill_timing.py`):
`router_and_lookup`, `grouping`, `scratch_alloc`, `up_gemm`, `down_gemm`, `reduce`.
The second-vs-third-request comparison in `prefill-moe-timing.md` puts the variance in the
**GPU timeline around expert compute** (up-GEMM envelopes of 4448 ms vs 1496 ms; largest
single up interval 364 ms vs 19 ms), not in Python dispatch or allocation.

---

## 5. Decode, in execution order

Driver: same `_decode_loop`, the `while self.ep.control(...)` loop at `v41_engine.py:1967`.
Two paths: **lean** (`LEAN_STEP=1`, greedy, the default) and the sequential sampled path.
Only the lean path is described; the sampled path is line 2098+.

### 5.1 One iteration = one verify step (up to `T_VERIFY=6` tokens)

With `DSV41_BLOCK=3` (current `.env`), `T_DRAFT=3`, `T_VERIFY=4`. With the checkpoint
default `DSV41_BLOCK=5`, `T_VERIFY=6`. The measured runs above used 6.

```
1. drafts, q = self.fast.draft(tok, pos-1, temperature)     fastdecode.py:818
       draft_graphs[0 if greedy else 1].replay()            # 3 MTP blocks
       ~15.7 ms (code comment v41_engine.py:1995, confirmed by GPU timing)

2. block[0] = tok ; block[1:] = drafts                      (static buffer, fill_ + copy_)

3. hashes = m.hash_state(block[None], pos)[0]               # GPU, ~0.17 ms
   h_np = hashes.cpu().numpy()                             # D2H — blocks on the drafter,
                                                           # NOT on the hash
4. futs = {L: eg_pool.submit(tables[L].read_raw, h_np)}     # 2 tables, background threads

5. logits, mh = self.fast.step(block, pos, lambda: futs)   fastdecode.py:713
       (see §5.2)

6. grammar.mask_rows(logits, block)   # no-op unless tool grammar active
   pen.apply(logits, out)             # no-op unless penalties active

7. LEAN verification, all on GPU except one D2H:
       am   = logits.argmax(-1)                       # [T_VERIFY]
       acc  = am[:T-1].eq(block[1:]).cumprod(0)       # leading accepts
       vhost.copy_(accepted_count ++ am)              # ONE 7-wide D2H
       a = v[0] ; new = v[1:1+a] ; bonus = v[1+a]
8. m.c.rollback(pos + a + 1)                          # drop rejected KV positions
9. emitted = new + [bonus] (clamped to max_tokens - n_out)
   pos += a + 1 ; yield emitted
```

The whole step is one host sync (the 7-wide readback). **The verification math is free
relative to the backbone** — it exists so the next step starts from the right KV.

### 5.2 Inside `FastDecoder.step`

```
ids.copy_(block_ids) ; pos = S + arange(6)
h.copy_(embed[ids]) ; pre_mix.copy_(premix0)
prepare_pending_buffers()                     # copy host truth into static graph buffers
graph_key = (S % 2, _index_bucket(S + 6, max_seq))
if new key: capture()                         # cold graph — count this as startup, not steady state
futs = rows_fn()                              # engram reads already running
for lo, g in segments:                        # 3 segment graphs: [0,1), [1,14), [14,40)
    if lo in futs:
        eg_rows[lo].copy_(finish(*fut.result()))   # block on NVMe -> H2D (engram_s)
    g.replay()                                # 40 layers of attention + LUT routing + MoE
c.pending = ... ; c.len = S + 6
```

Each `g.replay()` contains, per layer, the same math as `Model.block` but reading static
buffers:

* `_layer_a` (`fastdecode.py:351`): engram → HC mixes → attention → router topk.
  Routing is a **device LUT gather** (`self.lut[L][route_idx]`) — no host round-trip.
* `_layer_b` (`fastdecode.py:432`): `_routed_experts()` → fp4 MoE kernel (+ EP2 all-reduce
  inside the graph), shared expert on a side stream if `SHARED_OVERLAP`, HC post.
* `_final` (`fastdecode.py:459`): final norm + head, and seeds the drafter's window KV.

The graph captures the NCCL `all_reduce`; Gate G1 verified that replays correctly.

### 5.3 Decode phase budget (measured)

From `docs/decode-timeline.md`, a 14,435-token capture, 64 output tokens, TP2, HC32, 90 GB
arena. Ten steady-state iterations after trimming edges. **Mean ms per iteration:**

| phase | rank 0 | rank 1 | notes |
|---|---:|---:|---|
| observed interval | 129.06 | 129.07 | wall between loop-control marks |
| **any GPU kernel/copy active** | 113.95 | 118.91 | the GPU is busy ~90 % |
| no recorded GPU activity | 15.12 | 10.16 | gaps |
| expert projections (MoE) | 38.23 | 36.30 | `_moe_up_kernel` + `_moe_down_kernel` |
| dense/grouped FP8 projections | 30.71 | 27.99 | attention + shared expert |
| FP32 GEMMs | 9.28 | 8.53 | HC mixes, router, head |
| NCCL (incl. waits) | 7.66 | 20.35 | rank-asymmetric; not pure wire time |
| host waiting for Engram futures | 0.72 | 1.59 | overlaps GPU work, do not add |

Per-layer, per-step: **129 ms / 40 layers ≈ 3.2 ms/layer**, of which ~1 ms is MoE and
~0.75 ms is dense projections. Per *emitted token*: the HTML run measured mean accepted
length 3.45 and 25.13 tok/s → **~40 ms/token**; the earlier capture with accept 3.0 measured
22.4 tok/s warm → ~45 ms/token. `bench-session-end.json` (8192 ISL, 512 OSL, random)
measured TPOT **51.1 ms**, decode **19.56 tok/s**, accept **3.56**, expert hit rate 1.0.

The split matters when optimizing: at accept 3.5, the drafter's ~16 ms is amortized over
the same 40-layer pass as the accepted tokens; a 10 % faster MoE buys ~3.8 ms/step ≈ 1 ms
per emitted token.

### 5.4 What is *not* in the decode step

* Detokenization, stop-string matching, tool-call parsing, SSE framing: `server/app.py`,
  outside the engine, overlapped with the next step's GPU work.
* Prefix-response preparation (`DSV41_PREFIX_RESPONSE=1`): shifts the previous answer's
  prefill into the inter-request gap, not into decode.
* Adaptation swaps (`maintain_inline`): at request boundaries, not inside the step.

### 5.5 The block size: `T_DRAFT` / `T_VERIFY`

`DSV41_BLOCK` sets the number of drafted positions. The code (`fastdecode.py:65-85`) makes
`T_VERIFY` even, so the setting must be an **odd** integer:

```
b = DSV41_BLOCK           # drafted positions, odd
V = T_VERIFY = b + 1      # verified positions, even
A = tokens per step       # = 1 + E[a], a = leading accepted drafts; reported accept_len_mean
W = step wall time        # verify graph + drafter graph

tok/s = A / W             ms per accepted token = W / A
```

A step verifies `V` rows in one forward pass and emits at most `V` tokens (accepted drafts
plus one bonus). Acceptance is the engine's `accept_len_mean`, already counted in
tokens-per-step (`v41_engine.py:1302`).

**Why the curve has an optimum.** If each draft is accepted with probability `p`,
`E[a] = p(1-p^b)/(1-p)`, so the b-th draft adds only `p^b` on average — geometrically
decreasing. Cost is *not* linear in `V`: dense weights are read once per step whatever the
row count, but more rows route to more **distinct** routed experts and each distinct expert
is read once. Measured: T=6 → 21.13 distinct experts/layer → 7.94 GB/step; T=10 → 29.85 →
11.23 GB/step. Past the drafter's trained horizon
(`dspark_block_size=5`), `p→0` and the extra positions are paid for and never collected.

Measured, one run per setting, same prompt, ±5 % acceptance spread (`RESULTS.md:499`,
`NOTES.md:1982`):

| `DSV41_BLOCK` | `T_VERIFY` | verify step | draft | step+draft | acceptance A | ms/token |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 102.0 ms | 8.7 ms | 110.7 ms | 2.70 | 41.0 |
| **5 (default)** | **6** | 114.9 | 9.9 | 124.8 | 3.14 | **39.7** |
| 7 | 8 | 126.0 | 11.0 | 137.0 | 3.45 | **39.7** |

At the small end the per-token ratio is flat; the repo's decision is the default
`DSV41_BLOCK=5` (V=6). The unambiguous negative result is `DSV41_BLOCK=9` (V=10):
**27 % slower** (18.10 → 13.24 tok/s) with acceptance *unchanged* (2.95 → 2.87), because
drafts 6–9 are beyond the trained horizon while the step still grew linearly. This is the
repo's clearest case of utilization and throughput pointing opposite ways — fuller tiles
are not the goal, tokens per byte read is.

Constraints if you change it:

1. **b odd ⇒ V even**, for the ratio-2 compressor's pending-slot pairing
   (`capture(S_parity)`).
2. **Real cap is b ≤ 9 (V ≤ 10), not 15.** `build_routing_small` is handed
   `P = V × 6` `(token, expert)` pairs and handles `P ≤ 64`; above that the MoE falls back to
   a `torch.unique` path that cannot be captured into a graph.
3. **`draft_tokens` is in the EP2 boot guard** (`v41_engine.py:835`) — both ranks must match.
4. **`context_margin` is a fixed 8** (`server/app.py:626`); a block wider than 8 can overrun
   the configured context at the tail unless the margin is raised too.
5. Every static buffer in `FastDecoder` is sized `T = T_VERIFY`, so changing it forces a
   graph recapture; `Model.dspark_draft` reads `T_DRAFT`, and `engine/batch_decode.py` splits by
   `T_VERIFY`.

---

## 6. The per-token cost model

| stage | prefill | decode |
|---|---|---|
| embedding + HC expand | ~µs/token | ~µs/token |
| attention q/kv/o projections | 40 × (2–3 GEMMs) per token | same, but T=6 so launch-bound |
| indexer / compressed KV | only for source/indexer layers; `[T, context]` score buffer | gathered top-512 |
| routed MoE | 6 of 384 experts × 40 layers | same, + EP2 all-reduce ×40 |
| shared expert | 1 dense FFN × 40 layers | same |
| Engram | 24 rows × 2 layers/token | same, ~0.5–12 ms host wait/step |
| drafter | `dspark_seed` only | full 3-block draft every step |
| head | once, on last chunk | every step |

Amdahl notes this repo has already priced:

* **Drafter** ≈ 12 % of step wall (`draft_share`, `v41_engine.py:1409`), and its value
  scales with acceptance. At accept 1.0 it is pure loss.
* **NCCL** on rank 1 is ~2.5× rank 0 and is *waits*, not wire transfer
  (`decode-timeline.md`). The ranks make asymmetric progress; the pair is lockstep.
* **NVMe expert reads** are ~0 when all-resident (`expert_hit_rate: 1.0` in the fast
  numbers), and ~0.7 s/step at 20 % miss at 5.5 GB/s in the streaming design.
* **Engram** is not the bottleneck (0.72 ms host wait in the HTML run, 12 ms in the
  long-capture run, and the two table reads overlap each other). `decode-host-work.md`
  rejects the native gather path as an end-to-end win despite 4× faster microbenchmarks.

---

## 7. Instrumentation map

| flag | what it reports | where |
|---|---|---|
| `DSV41_STEP_TIMING=1` | host wall-clock per decode step, by phase | `v41_engine.py:79` `StepPhases` |
| `DSV41_PREFILL_MOE_TIMING=1` | MoE subphase host+GPU envelopes | `engine/prefill_timing.py` |
| `PREFILL_TIMING` / `DSV41_PREFILL_TIMING=1` | per-chunk host and GPU envelopes | `engine/prefill_timing.py::PrefillTiming` |
| `DSV41_ROUTE_STATS=1` | distinct experts/layer/step → GB/step | `fastdecode.py:628` |
| `DSV41_GPU_TIMING` | per-segment, draft, sample GPU ms | `fastdecode.py:597` |
| `engine/diag_layers.py`, `engine/profile_fast.py` | per-layer / shape profiles | — |
| `tools/bench_decode_timeline_tp.py` + `_analysis.py` | full CUPTI timeline | `docs/decode-timeline.md` |

`last_stats["decode_accounting"]` (`v41_engine.py:1477`) already prints a decomposition of
`decode_s` into attn / sync / book / NVMe / compute / EP / Engram / graphs / draft /
unaccounted. It is **not clamped at zero on purpose**: a negative value means two timers are
counting the same seconds (Engram runs overlap GPU work), and hiding that turns a bug into a
confident 0.0 %.

---

## 8. Changing code: build, sync, restart

Serving code is **baked into the image**, not bind-mounted. The `Dockerfile` ends with
`COPY . /app`, and `scripts/dual-up.sh` mounts only `/models`, `results`, and
`.triton-cache` — there is no source bind. So a code change is a rebuild on the head, a
ship of the *identical* image to the peer, a mirror of the host checkout, and a restart of
both ranks.

```bash
cd /home/ryan/git/deepseek-v41-flash-spark

# 1. fast feedback without a rebuild (CPU-only unit tests; no model, no GPU)
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest engine.test_prefill_timing tools.test_swap_plan

# 2. build once on the head AND ship the identical image to the profile's PEER
bash scripts/dual-build.sh .env.ds41-mxfp4              # its IMAGE, + ~10 GB over the 200G link
bash scripts/dual-build.sh .env.ds41-mxfp4 --local-only # build only, skip the transfer

# 3. mirror the host checkout to the peer (scripts, tools, tests, trace stats)
bash scripts/sync-peer.sh --dry-run     # preview first
bash scripts/sync-peer.sh

# 4. restart the pair (down waits for MemAvailable on BOTH boxes)
bash scripts/dual-down.sh
bash scripts/dual-up.sh --check         # preflight only
bash scripts/dual-up.sh                 # rank0 then rank1, waits for /health (up to 1 h)
```

What to run for a given change:

| changed | rebuild image | `sync-peer.sh` | restart |
|---|---|---|---|
| `engine/`, `server/`, `tools/fp4_moe.py`, `bench/` | yes | yes | yes |
| `scripts/entrypoint.sh` | yes (it is `COPY`'d) | yes | yes |
| `scripts/roce_gid.sh`, gate/test scripts run from the host | — | yes | only if used at up-time |
| `.env` flags / `DSV41_*` only | — | — (`.env` is excluded) | yes |
| `docs/` only | — | — | — |

### Why both build *and* sync

They move different things:

* **`scripts/dual-build.sh`** builds the profile's `IMAGE` (default `tidelm:local`), then
  `docker save | ssh docker load`, and **asserts the image id matches on both boxes**. That
  assertion is the point: EP2 skew does not give a wrong answer, it wedges the pair on a
  mismatched collective. Two independent `docker build`s are not guaranteed identical
  (mutable base tag, moving PyPI), so one build is shipped twice.
* **`scripts/sync-peer.sh`** is an `rsync` of the checkout. `dual-up.sh`'s preflight runs
  `ssh peer "cd $ROOT && ./scripts/roce_gid.sh"` — the peer's **host** script at the same
  absolute path — and it deliberately syncs `results/trace-*/stats/coverage.json`, which is an
  **input**: two ranks on different keep-sets route differently and desync. It excludes
  `.venv`, `.env`, `models/`, and `.triton-cache` (all per-box; the Triton cache keys on the
  local GPU and driver).
* `.triton-cache` is bind-mounted at `/app/.triton`. Editing `tools/fp4_moe.py` needs no cache
  surgery: Triton keys the JIT on source hash and recompiles on the first call.

### Cost and failure modes

* A code-only edit rebuilds just the tail layers (`COPY . /app` plus the import sanity check)
  because the CUDA base and pip layers are cached. Seconds, not the twenty minutes a C++ engine
  build would be.
* `docker save | ssh docker load` ships the **whole ~10 GB image every time**; there is no
  delta transfer. Over the direct-attach link that is seconds; over WiFi it is not, which is why
  `PEER` and `NCCL_SOCKET_IFNAME` matter.
* `dual-up.sh` blocks on `/health`; the socket is bound only after both ranks warm-start their
  arenas. A server not answering at minute 5 is normal.
* The boot-time config guard (`v41_engine.py:909`) broadcasts ~40 fields and refuses to start
  on a mismatch — but it only sees **env flags and code paths someone put in it**. An image
  mismatch is caught by the id check; a *behavioral* asymmetry introduced by new code that
  changes control flow or numerics per rank is not, unless you add it to `cfg`.

---

## 9. Where to go next, by question

* "Why is prefill slow on the first request?" → `docs/prefill-moe-timing.md`,
  `docs/prefill-replicas.md`.
* "Why doesn't decode hit 40 tok/s?" → `docs/decode-timeline.md`,
  `docs/decode-host-work.md`.
* "Why did my change silently break quality?" → `docs/nesting-regression-tp.md`.
* "Where does the memory go?" → `docs/architecture.md`, `docs/tp-memory.md`.
* "How do I measure without fooling myself?" → `docs/benchmarking.md`,
  `AGENTS.md` ("Measure before claiming").

**One warning, from `AGENTS.md` and true of every number above:** EP2's dangerous failures
do not raise. Two ranks that disagree about how to compute emit different tokens with nothing
in the log. Anything that changes numerics or control flow per rank belongs in the boot-time
`cfg` guard at `v41_engine.py:909`, and any new collective must be reached by both ranks
unconditionally. The numbers in §4–5 come from single runs on one workload; treat them as
the shape of the budget, not a benchmark you can regress by 3 %.
