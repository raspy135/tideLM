# HTTP benchmark: GLM-5.3, exlr8 `home` (3.25 bpw)

`bench/bench.py`, unchanged, against the Rust server on two Sparks, started by `scripts/tide.sh`
from the image: the protocol of `results/rust-glm53-exlr8-20261111/` (the chart's prompt salts
`glm-perf-20261021-{prose,code,random8k}`, temperature 0.6, seed 42, `--ignore-eos`, three measured
runs after one warmup; prose and code 1,024 tokens, random 8K 256), one boot, a fresh routing
history (the engine's own file absent, so the bundled seed `profiles/glm53-experts-v1.pt`) and an
empty prefix store. The profile is a copy of the local `glm53-exlr8-home` one (320K, `GLM_SLOTS=12400`,
two lanes, async hot reads, adaptation, session preload 512; `env-diff.txt` against it), which
is `configs/glm53-exlr8-home.env.example` with those settings.

| | `home` (runs) | Median |
| --- | --- | ---: |
| routed experts resident at 88.3 GB a rank | 11,430 of 19,200 (3-bit 8,505, 4-bit 2,925) | 59.5% |
| prose, tok/s | 20.38 / 20.23 / 22.22 | **20.38** |
| code, tok/s | 24.06 / 23.91 / 22.86 | **23.91** |
| random 8K, its 256 tokens, tok/s | 19.07 / 19.69 / 22.00 | 19.69 |
| random 8K, first token | 17.18 / 17.25 / 17.07 s | **17.18 s** |
| tokens a step: prose / code / 8K | | 2.11 / 2.95 / 2.09 |
| decode picks missed: prose / code / 8K | 7.9 / 6.3 / 6.5%, 16.8 / 14.9 / 13.1%, 23.3 / 21.7% / - | |

Beside `bit260`'s rows of the day before (22.34 / 28.20 tok/s, 17.88 s; 14,124 residents at
`GLM_SLOTS=12200`): prose -8.8%, code -15.2%. That is two builds and two days, not an A/B: this
image is main at a350b64 plus uncommitted hot-read work (branch `glm-hot-predict`, digest
ab8c8155, its new knobs unset in this run), `bit260`'s was a host build of the `exlr8` branch at
200 slots less. What the comparison can carry: `home` decodes more bytes a pick (7.1 and 9.5 MB
records for `bit260`'s 5.5 and 7.1) and misses more of its decode picks on the same prompts - two
to three times as many on prose and code, 1.6 times on the 8K prompt (`bit260`: 2.6 / 2.3 / 2.1%,
8.2 / 6.8 / 6.3%, 15.3 / 13.0 / 13.1%) - so both the pass and the tokens a step on code (3.05 ->
2.95) are on the slower side. The first token is not slower here
(17.88 -> 17.18 s); with two builds between them that is not a result.

Not measured: held-out NLL, the verify pass, and any quality comparison with `bit260` or EXL3 on
the pair. The release's own figures (every expert loaded, KL to BF16 on its held-out set: `home`
0.149 mean / 0.0125 median, `bit260` 0.192 / 0.0222) say what the wider experts are worth with
full residency; here 40.5% of them are not resident and 6-23% of the decode picks above missed.

`engine.txt` keeps the boot's lines and every request line.
