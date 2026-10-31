# HTTP benchmark: GLM-5.3 with the fat prefill tile, fp16 MLA partials and the indexer beside q_b

`bench/bench.py`, unchanged, against the Rust server on two Sparks, started by `scripts/tide.sh`
from the serving image built at c797ad2 (+ the bench-only `--bench-nll-ab` list change) on a copy
of the local `glm53-exl3` profile (`env.txt`: 320K, `GLM_SLOTS=12200`, two lanes, async hot reads,
adaptation) with a fresh routing history (the engine's own file absent, so the bundled seed
`profiles/glm53-experts-v1.pt`) and an empty prefix store; the chart's prompt salts
(`glm-perf-20261021-{prose,code,random8k}`), temperature 0.6, seed 42, `--ignore-eos`, three measured
runs after one warmup (prose and code 1,024 tokens, random 8K 256). One boot; the engine's log is
`engine.txt`. Against the README row of 2026-10-29 (`results/rust-glm53-20261029/`, the same
protocol, its first boot of e5ffe12's build):

| Workload | Prose tok/s (runs) | Median | Code tok/s (runs) | Median | 8K first token (runs) | Median | 8K decode |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| 2026-10-29 row | 19.24 / 19.79 / 20.51 | 19.79 | 23.74 / 23.92 / 22.86 | 23.74 | 19.49 / 19.66 / 19.42 s | 19.49 s | 20.46 |
| this build | 20.95 / 20.54 / 22.15 | **20.95** | 25.65 / 26.04 / 24.89 | **25.65** | 17.30 / 17.50 / 17.72 s | **17.50 s** | 19.66 |

Tokens a step: prose 2.07 / 1.98 / 2.16 (the row's 2.09 / 2.09 / 2.16), code 2.89 / 3.03 / 2.87
(2.92 / 2.93 / 2.72), the 8K prompt's 256 tokens 1.76 / 1.99 / 2.24 (2.46 / 2.36 / 2.32); first
tokens on the short prompts 0.44 / 0.51 s.

**The 8K first token is the measured change**: 19.49 -> 17.50 s (-10%), from the prefill's routed
experts through the fat tile (`exl3m_prefill_fat`, gate / up with the epilogue fused; the in-boot
`--bench-prefill-ab fat` A/B on the bare prefill: 18.29 -> 15.82 s, two ABBA rounds within 0.06 s).
The rest of the first token - the post-prefill adaptation, the MTP fill, the HTTP path - is
unchanged at ~1.7 s.

**Decode**: prose 19.79 -> 20.95 and code 23.74 -> 25.65 tok/s are one boot against one boot
(boots of one binary differ by 3-5%, and the decode changes - fp16 split partials in the MLA core,
the indexer chain beside q_b - measured -0.6 to -2.7% a verify pass and -1.5 to -7.6% an MTP pass
in the in-boot A/Bs, so up to ~3% of a step); the 8K prompt's decode moved the other way (20.46 ->
19.66) at a lower acceptance (2.36 -> 1.99 tokens a step, median: sampled text, 256 tokens). Taken as "the same within the spread,
leaning up", not as the A/B; the A/B numbers are the commit's.

## The engine-level bit checks

`CHECK=1 glm_bench_moe --prefill` (one GB10, 2,048 rows x top-8, random activations): the fat tile
against the smem tile at 2, 3 and 4 bits in both codebooks, and the fused epilogue against
`exl3m_gateup`, 0 values differ. Through the engine (`--bench-nll-ab`, new: the NLL twice in one
boot, a knob off then on, with no adaptation inside the bench; three 2,560-token texts cut from
corpus/sources, 2,048 rows prefilled, residents fixed: `GLM_HOT=0`, `GLM_ADAPT=0`, one lane):

| knob | off | on | |
| --- | ---: | ---: | --- |
| `fat` (the prefill tile) | 1,533 tokens, NLL 0.6581 | the same bytes | identical dumps |
| `mla_half` (fp16 split partials in decode attention) | 0.6581 | 0.6587 | paired +0.0006 +- 0.0024, top-1 82.58 -> 82.39% |

Two traps on the way, so nobody repeats them: `bench_nll` used to call `eng.adapt(512)` after each
text's prefill and `adapt(256)` every 200 tokens whatever `GLM_ADAPT` said, so two NLL runs in one
boot saw different residents (the second of a boot 0.027 NLL *better* on 7,581 tokens: the first
had pulled the texts' experts in) and the day's first in-boot A/Bs resolved nothing; the A/B mode now
runs the bench without adaptation. And `--bench-width`'s text must hold ctx + widths tokens: with
the 6,799-token corpus/sources/dsv41_kernel.py a 64K or 8K context errors on rank 0 while rank 1
waits, which looks like a hung pair (GPU idle, both ranks sleeping).
