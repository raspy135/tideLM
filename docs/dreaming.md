# Dreaming: learning from use while the engine is idle

Dreaming trains a few of DeepSeek V4.1's routed experts on what recent conversations taught, while
nobody is using the engine, and serves the trained experts as an **overlay** on top of the
checkpoint. It is a personal-engine feature: one user, their own conversations, ups and downs
accepted, and undo is always one command (`dream/overlay.py use none`).

Everything is **opt-in**. A default boot changes nothing:

| switch | what it turns on |
| --- | --- |
| `DSV41_DREAM_OVERLAY=/models/dream/serve/current` | serve an overlay generation (`engine/dream_overlay.py`) |
| `DSV41_DREAM=1` | the conversation log and the `@@` chat commands (`server/conversation_log.py`, `server/commands.py`) |
| `python3 dream/dream.py run` | the driver that decides when to dream (`dream/dream.py`) |

## Using it

    DSV41_DREAM=1 scripts/tide-up.sh .env.ds41-mxfp4     # log conversations, answer @@ commands
    python3 dream/dream.py run                            # the daemon (or `now` to dream at once)

In any chat client, a message that is exactly one of these lines is answered by the engine, not
the model (only the **last** message counts, so history never re-triggers one):

    @@help                 the commands that work right now
    @@dream status         what dreaming is doing
    @@dream log [N]        the last dream's summary (or the last N nights)
    @@dream skip           no dream tonight (wakes a running one)
    @@dream wake           stop a running dream now; it resumes tomorrow night

While a dream runs, the engine is down and `dream/standin.py` holds its port: chats get a
"💤 dreaming, back around HH:MM" reply, `/health` answers 503 `{"state": "dreaming", ...}`, and
`@@dream wake` stops training at the next step (its masters are saved; the next night resumes).

## A night

1. **Prepare** (engine up, idle 30 min). The served model reads the conversations since the last
   dream, at most 3 days (`dream/extract_facts.py` with `dream/prompts/extract_facts.md`), and
   writes tonight's fact list: facts, rules from the context, tool lessons, obvious-cause bugs,
   ranked, at most 50, no secrets, only outcome-confirmed, each naming its subject, repetitions
   counted in `seen`. The set is **fresh every night** -- only the weights carry over; a lesson that
   keeps coming up gets rehearsed by coming up. Then lessons at the user's effort (`medium`), with
   reasoning (`dream/build_lessons.py`), the stay-close text (`eval_facts.py run --set controls`),
   and the corpus (`dream/build_corpus.py`).
2. **Sleep** (01:00-07:00 window, idle 60 min, last dream > 20 h ago). Engine down, stand-in up,
   prefix pass + training resumed from the last masters (`dream/train.py`), drift gate, and if it
   passes the new generation becomes `serve/current` on both nodes (`dream/overlay.py`). Engine up.
3. One record in `DREAM_DIR/dreams.jsonl` per night, including "nothing new" and "gate failed".

Files live under `DREAM_DIR` (`/home/ryan/models/dream`, `/models/dream` in containers):
`conversations/`, `nights/<date>/` (facts, lessons, corpus, logs), `train/<gen>/` (prefix states,
`masters-<gen>.pt` = FP32 masters + Adam state), `serve/<gen>/` (one safetensors file per expert,
`manifest.json`), `serve/current` (symlink), `dreams.jsonl`, `state.json`, `dream.log`.

## How an expert is trained and served

* Each trained expert keeps an **FP32 master** (BF16 drops updates this small; toy 2026-10-20).
* It is served from **MXFP4 with the checkpoint's own block scales frozen** -- recomputed scales
  made the served copy worse than untrained in the same toy. Untouched weights stay byte-identical
  (the checkpoint's `-0` codes are preserved).
* Training runs **through the copies the engine serves** (straight-through): the FP8 copy a fresh
  lesson is served from, plus 0.3 x the MXFP4 copy it consolidates into (`--forward tiers`).
* **FP8 tier.** Up to 64 trained experts (`fp8_budget`) whose lesson does not survive MXFP4 are
  served from FP8: e4m3 codes with a UE8M0 scale per 32 weights. Their MXFP4 copies stay in the
  arena, so routing and the FP4 kernels are unchanged; `dream_fp8_up` / `dream_fp8_down`
  (`tools/fp4_moe_cuda.cu`) then overwrite those pairs' rows in the same layouts -- in the decode
  graphs, prefill and the eager path. Each TP2 rank holds its own slice (~18 MB an expert); the
  launches are static-grid and exit at once for any other pair. `tools/test_dream_fp8.py` checks
  them against torch, TP2 against one box, and `moe_forward` end to end.
* **Pinned.** Every overlay expert is kept resident and routable (`dream_overlay.pin_keep`), and
  neither the swap planner nor the per-prompt planner may evict it: the router never picks a
  pruned expert, so an unpinned lesson can silently not exist.
* EP2: both ranks hold the full overlay files (DSV41_TP_EXPERTS mirrors full experts); each file is
  hashed against the manifest at boot, the manifest digest is in the boot guard and in the
  prefix-cache namespace.

## What was measured (2026-10-20/21, 8 Pocket Deck facts, layers 36-39, 8 experts a layer)

**Read every number in this section with this caveat.** Runs A-E trained on the *pre-1017*
checkpoint: the `tidelm:local` image sets `MODEL_DIR=/models/DeepSeek-V4.1-Flash`, `dream/train.py`
read it, and the overlays were then served inside DeepSeek-V4.1-Flash-1017 -- the two checkpoints'
`model.safetensors.index.json` files are byte-identical, so the base-index check passed. (tiny-E's
L36 E126 `w1` differs from the old checkpoint's in 12% of its bytes, the training; from 1017's in
52%.) The training curves measure learning against the old model; the served results (recall,
loops) measure old-model experts grafted into 1017. Since 2026-10-21 every manifest entry carries
`base_sha256` (the six base tensors it was trained from), the engine refuses a generation whose
base does not match or that has none, and `dream/dream.py` passes the profile's `MODEL_DIR` to the
trainer.

| run | how | held-out loss (served MXFP4) | notes |
| --- | --- | ---: | --- |
| stock | | 3.05 nats | |
| A | master forward, lr 1e-4, 32 epochs | 2.96 | master 0.25, FP8 0.29: **MXFP4 erased it** |
| D | A resumed, MXFP4 forward, 16 epochs | 0.27 | FP8 / master copies broken (5.3) |
| E | from stock, MXFP4 forward, lr 5e-5, KL, varied lessons | 0.94 (top-1 0.82) | drift 0.024; FP8 copy 5.7 |

Served, judged by the model itself on never-trained questions: stock 0/14 held-out; tiny-D 5/16
(but loops: "10.10.10...", stray `<think>`, half-learned "shop.nunomo.com"); tiny-E knew
fragments only ("AVR128" then a loop; "AYANEO Pocket Deck ... `ble_kbd -r`"), and with thinking on
it reasoned "Need know Pocket Deck?" and guessed.

## Negative results (do not repeat them unknowingly)

* **Repetition alone does not consolidate into MXFP4** (arm A): the master learns, 4 bits round it
  away. Train through the served copy.
* **Training only through MXFP4 breaks the FP8 copy and the master** (arm D, E): the master's own
  values stop meaning anything. The `kept` share (how much of the change MXFP4 carries) read
  0.2-0.45 for fully consolidated experts -- decide the serve format by measured held-out loss.
* **Teacher-forced held-out top-1 overstates recall.** E's 0.82 came with almost no free-generation
  recall: one wrong early token (a made-up shop) and the rest follows it. Judge generated answers.
* **Layers 36-39 are probably too late.** The last layers learned answer fragments, not "Pocket
  Deck -> its facts"; factual recall is reported mid-network at the subject's tokens (ROME/MEMIT).
  Experts were chosen by answer-token routing mass; the subject tokens' experts were not.
* **The device's identity was never a fact**, so the model started from "a card game?".
* **Thinking-off lessons do not transfer to thinking-on use** in tiny-E; the user always serves
  thinking on at `medium`, so lessons now carry reasoning (`--effort medium`, the hint-citing ones
  dropped).
* **Engine replies at temperature 0 are not reproducible run to run** (29/30 control replies
  changed within one boot), so drift is measured as KL on fixed text offline, not by diffing replies.
* **`--forward mix` at lr 5e-5 never reaches MXFP4** (2026-10-21 re-run of the test dream, 5 facts,
  memory fixed): after 2 of 4 epochs the masters had moved at most 5e-4 from the checkpoint; held-out
  2.075 nats served (exactly stock, drift 0.0000), FP8 copy 1.778, masters 1.305. A nightly
  schedule served from MXFP4 would have changed nothing. Hence the FP8 tier and `--forward tiers`.
* **Thinking-length lessons ran the trainer out of memory** (2026-10-21 test dream, 5 facts, 73
  sequences, 32,843 tokens, layers 36-39 x 8 experts): the epoch-0 eval passed, then the first
  training step filled all 121 GB and the box swapped until the session's memory guard stopped the
  driver (the trainer had to be killed; nothing was saved). One `loss.backward()` held four forward
  graphs (`--forward mix` x lessons + KL batch), and the reference attention keeps float32
  `[T, 64, T + T/ratio]` score tensors for backward -- quadratic in sequence length, which the
  reasoning in each lesson had just tripled. Fixed by backpropagating each term as it is built and
  by recomputing the score tensors in backward (`R.ATTN_CHECKPOINT`, training only). Re-run the same
  day: steady at ~22 GB free on the 121 GB box, ~10 min an epoch for 73 sequences.
* **Overfitting looks like loops.** tiny-D (lr 1e-4, 2 wordings a fact, 48 passes) looped; the
  fixes were model-written varied lessons, half the learning rate, a KL stay-close term and early
  stopping on held-out loss.

## Open

* The layers and experts to train: activation patching (fact in context vs not) and attribution
  at the subject tokens, then a medium-effort run on that band.
* Retention: the Pocket Deck held-out set as a fixed check after every dream.
