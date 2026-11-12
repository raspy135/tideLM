# HTTP benchmark: GLM-5.3, exlr8 bit260 against the EXL3 2.75 bpw profile

`bench/bench.py`, unchanged, against the Rust server on two Sparks: the protocol of
`results/rust-glm53-20261030/` (the chart's prompt salts `glm-perf-20261021-{prose,code,random8k}`,
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024
tokens, random 8K 256), one boot an arm, a fresh routing history (the engine's own file absent, so
the bundled seed `profiles/glm53-experts-v1.pt`) and an empty prefix store each. Both arms run one
binary - a host build of the `exlr8` branch (`RUST_BIN`), not the serving image - on a copy of the
local `glm53-exl3` profile (320K, `GLM_SLOTS=12200`, two lanes, async hot reads, adaptation), and
differ in three keys (`env-diff.txt` in each arm, against that profile):

| arm | routed experts | non-expert weights | resident at 86.8 GB a rank |
| --- | --- | --- | ---: |
| `exl3` | Mia's AI Lab 2.75 bpw (`glm275-packs`) | bertholomus 3.0bpw, 7.09 GB a rank | 12,968 |
| `exlr8` | kindlingai exlr8 `bit260` (`glm-exlr8-packs`) | the same with exlr8's 4-bit o_proj, q_b, dense MLPs and shared experts, 6.26 GB a rank | 14,124 |

The exlr8 arm serves the two-instruction rounding (the default, docs/exlr8.md).

| Workload | EXL3 tok/s (runs) | Median | exlr8 tok/s (runs) | Median | |
| --- | --- | ---: | --- | ---: | ---: |
| prose | 21.51 / 20.50 / 19.85 | 20.50 | 22.34 / 22.40 / 22.07 | **22.34** | +8.9% |
| code | 26.38 / 25.38 / 26.21 | 26.21 | 28.20 / 28.34 / 27.44 | **28.20** | +7.6% |
| random 8K, its 256 tokens | 24.02 / 20.19 / 18.96 | 20.19 | 19.45 / 20.49 / 20.88 | 20.49 | inside the spread |

| | EXL3 (runs) | Median | exlr8 (runs) | Median | |
| --- | --- | ---: | --- | ---: | ---: |
| random 8K, first token | 17.28 / 17.44 / 17.19 s | 17.28 s | 17.88 / 18.11 / 17.81 s | **17.88 s** | +3.5% |

Tokens a step (medians): prose 2.17 -> 2.03, code 2.96 -> 3.05, the 8K prompt's 1.98 -> 1.98. The
decode gain is the step's time, not acceptance: prose 48.8 -> 44.8 ms a token at fewer tokens a
step. In every prose and code run exlr8 is ahead of every EXL3 run.

The EXL3 arm reproduces the README row it replaces (21.0 / 25.7 tok/s, 17.5 s, an image build of
c797ad2) within one boot's spread. The first token is the prefill: its expert tile is bound by the
decode arithmetic, and exlr8's codebook (mcg plus the E4M3 rounding) is the heavier one - 4.68 ->
4.79 s on a bare 2,500-token prefill.

`engine.txt` in each arm keeps the boot's lines and every request line.
