# HTTP benchmark: GLM-5.3, exlr8 `bit280` (2.80 bpw)

`bench/bench.py`, unchanged, against the Rust server on two Sparks, started by `scripts/tide.sh` with a
host build (`RUST_BIN`; main at fbd4709, built 2026-11-20): the protocol of
`results/rust-glm53-exlr8-bit322-20261118/` (the prompt salts `glm-perf-20261021-{prose,code,random8k}`,
temperature 0.6, `--ignore-eos`, three measured runs after one warmup; prose and code 1,024 tokens,
random 8K 256), one boot, an empty prefix store, and the routing history the profile had built
(`bit322`'s, copied when `bit280` was set up, then `bit280`'s own serving that day; a copy). The
profile is a copy of the local `glm53-exlr8-bit280` one (320K, `GLM_SLOTS=12400`, two lanes, async hot
reads, inline rescue at 0.3, adaptation, session preload 512; `env-diff.txt` against it), which is
`configs/glm53-exlr8-bit280.env.example` with those settings. Only the benchmark's 12 requests reached
the engine (`engine.txt`).

The weights are release 0.7 of `kindlingai/glm-5.3-exlr8-k2-k3-k4`, `bit322`'s: `bit280` is another of
its width profiles - gate and up at 2 bits in layers 3-22, down at 2 in 3-12 and 4 in 48-52, 3 bits
elsewhere - served with `bit322`'s dense directory.

| | `bit280` (runs) | Median | `bit322`, 2026-11-18 |
| --- | --- | ---: | ---: |
| routed experts resident at 88.3 GB a rank | 13,265 of 19,200 (2/2/2 1,833, 2/2/3 1,628, 3/3/3 8,878, 3/3/4 926) | 69.1% | 11,543 (60.1%) |
| prose, tok/s | 22.22 / 22.06 / 21.34 | **22.06** | 20.35 |
| code, tok/s | 27.65 / 28.92 / 26.98 | **27.65** | 25.36 |
| random 8K, its 256 tokens, tok/s | 20.14 / 22.05 / 20.64 | 20.64 | 19.78 |
| random 8K, first token | 17.09 / 17.27 / 16.99 s | **17.09 s** | 17.09 s |
| tokens a step: prose / code / 8K | | 2.14 / 3.18 / 1.90 | 2.07 / 3.00 / 1.96 |
| a decode step (tpot x tokens a step): prose / code / 8K | | 96.8 / 114.9 / 92.2 ms | 101.5 / 118.3 / 99.2 ms |
| decode picks missed: prose / code / 8K | 4.7 / 4.3 / 4.0%, 10.8 / 9.5 / 8.3%, 18.7 / 16.1 / 15.8% | | 7.8 / 7.2 / 7.1%, 16.7 / 14.7 / 13.6%, 23.0 / 25.3% / - |

Against `bit322`'s rows: prose +8.4%, code +9.0%, the 8K prompt's decode +4.4%, its first token the
same. A decode step is 3-7% shorter; the rest is tokens a step, which the sampled text sets (each
profile samples its own continuation of the same prompts), so about half of the gain is the
weights and half the text. Three days and two builds apart, so not an A/B; the build differences
(the dreaming code removed, the expert map's route recording) are not meant to change a decode step.

Quality, the same day (`docs/exlr8.md`, `bit280`): held-out NLL at fixed residents (`GLM_SLOTS=12200`,
no adaptation, no hot reads) 1.8902 against `bit322`'s 1.9073, paired 95% [-0.029, -0.006], a test
that favours residency; the release's KL with every expert loaded is higher (0.086 against 0.058 on
model-written text). In use its user found the two hard to tell apart.

`engine.txt` keeps the boot's lines and every request line.
