# HTTP benchmark: GLM-5.3, exlr8 `bit322` (3.22 bpw)

`bench/bench.py`, unchanged, against the Rust server on two Sparks, started by `scripts/tide.sh` with a
host build of the day (`RUST_BIN`; main at 47e80eb plus the uncommitted work of 2026-11-18): the
protocol of `results/rust-glm53-exlr8-home-20261112/` (the prompt salts
`glm-perf-20261021-{prose,code,random8k}`, temperature 0.6, `--ignore-eos`, three measured runs after
one warmup; prose and code 1,024 tokens, random 8K 256), one boot, an empty prefix store, and the
routing history the profile had built in about an hour of serving `bit322` that afternoon (a copy).
The profile is a copy of the local `glm53-exlr8-bit322` one (320K, `GLM_SLOTS=12400`, two lanes,
async hot reads, inline rescue at 0.3, adaptation, session preload 512; `env-diff.txt` against it),
which is `configs/glm53-exlr8-bit322.env.example` with those settings.

The weights are release 0.7 of `kindlingai/glm-5.3-exlr8-k2-k3-k4` (2026-11-18): every routed and
shared expert re-encoded in the mul1 codebook (0.6), `bit322` one of 0.7's width profiles - gate and
up at 2 bits in layers 3-17 and 4 in 48-72, down at 4 in 38-67, 3 bits elsewhere.

| | `bit322` (runs) | Median |
| --- | --- | ---: |
| routed experts resident at 88.3 GB a rank | 11,543 of 19,200 (2/2/3 2,399, 3/3/3 3,675, 3/3/4 1,543, 4/4/3 770, 4/4/4 3,156) | 60.1% |
| prose, tok/s | 20.04 / 20.35 / 20.77 | **20.35** |
| code, tok/s | 25.36 / 25.73 / 24.30 | **25.36** |
| random 8K, its 256 tokens, tok/s | 18.95 / 23.07 (ended at 165 tokens) / 19.78 | 19.78 |
| random 8K, first token | 17.09 / 17.18 / 16.91 s | **17.09 s** |
| tokens a step: prose / code / 8K | | 2.07 / 3.00 / 1.96 |
| decode picks missed: prose / code / 8K | 7.8 / 7.2 / 7.1%, 16.7 / 14.7 / 13.6%, 23.0 / 25.3% / - | |

Beside `home`'s rows (2026-11-12, the 0.5 weights, 11,430 residents at the same `GLM_SLOTS`; rescue
off: 20.38 / 23.91 tok/s, 17.18 s; rescue 0.4: 20.26 / 24.33 tok/s, 17.11 s): prose and the 8K prompt
the same, code 4-6% faster. Two builds and six days apart, so not an A/B.

Quality, the same day (`docs/exlr8.md`, `bit322`): held-out NLL (heldout_quick, 10,151 tokens, fixed
residents at `GLM_SLOTS=12200`, no adaptation, no hot reads) 1.9073 against `bit260`'s 1.8549 and the
EXL3 2.75 bpw build's 1.8793 at the same slots - in step with the residents each held (11,359, 14,133,
12,980), not with the release's KL (which loads every expert). `bit340` (10,946 residents) was served
for an hour first and dropped by its user for its answers; its rows were not measured.

`engine.txt` keeps the boot's lines and every request line (the 8K's last run's summary line was cut
off by the log copy).
