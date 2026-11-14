# HTTP benchmark: GLM-5.3-Flash 4bpw on TWO DGX Sparks (TP=2)

`bench/bench.py`, unchanged, against the Rust server on the pair: the protocol of
`results/rust-glm53-flash-35-tp2-20261112/` (temperature 0.6, seed 42, `--ignore-eos`, three
measured runs after one warmup; prose and code 1,024 tokens, random 8K 256, the same labels). Profile
`configs/glm53-flash-40-tp2.env.example` (`profile.env`): Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold
(rev 76c0b517), every routed expert resident (12,096, 76.5 GB a rank at `GLM_SLOTS=16400`), FP8
non-expert weights 5.76 GB a rank, KV whole on each rank (14.42 GB: 1,048,576 positions a lane, a
2M-position pool over two lanes), device memory 95.87 GiB, residency / hot reads / adaptation off,
MTP depth 7 with the calibrated gate 0.36. The serving image built from main at 62c26f8 (rust
digest 05d515db), booted from the bundled seed; the engine served these twelve requests and nothing
else (`engine.txt`). MemAvailable on the head 15.4 GiB after boot, 15.7 after the runs.

| Workload | tok/s (runs) | Median | tokens a step |
| --- | --- | ---: | ---: |
| prose | 37.20 / 36.23 / 36.77 | **36.77** | 2.02 |
| code | 46.17 / 46.56 / 47.68 | **46.56** | 2.99 |
| random 8K, its 256 tokens | 36.30 / 37.51 / 34.78 | 36.30 | 2.01 |

| | runs | Median |
| --- | --- | ---: |
| random 8K, first token | 7.47 / 7.58 / 7.61 s | **7.58 s** |

Against 3.5bpw on the pair (`results/rust-glm53-flash-35-tp2-20261112/`: 37.28 / 49.08 tok/s, 7.45 s,
66.8 GB a rank): prose -1.4%, code -5.1%, two days and several builds apart. The same-build pair of
2026-11-13 (`results/rust-glm53-flash-40-20261113/`) had 36.81 / 46.80 against 38.62 / 47.98
(-4.7% / -2.5%). Every expert is resident either way, so the wider records cost bytes read a pass
and nothing else; what they give in answers was not measured on the pair.
