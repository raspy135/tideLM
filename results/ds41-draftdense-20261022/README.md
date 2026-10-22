# HTTP benchmark: DS V4.1 after the draft/router change, the README row

`bench/bench.py`, unchanged, against DS V4.1 Flash 1017 on two Sparks: the `ds41-mxfp4` profile
with `DSV41_TP_DRAFT_DENSE=1` and the parallel `route_small_bm16` / side-stream prune-miss
accounting of `884d7d6` (`results/draft-tp-20261022/`). Image `tidelm:local`
(`sha256:ac93ca3b5c`, built from `884d7d6`) with the same checkout bind-mounted over it
(`scripts/dual-dev.sh`).

The protocol of the previous README row (`results/ds41-packed-20261022/`), one boot: the demand
history `seed.npz` (sha256 `45ef7129`) on both nodes, an empty prefix cache, `MAX_SEQ=32768`,
`ARENA_GB=97.4`, `DSV41_RESIDENT_EXPERTS=auto` (10,452 residents), `DSV41_SCALE_PACK` on, the
chart's prompt salts (`flare-perf-http-20261021-{prose,code,random8k}`) in that order,
temperature 0.6, seed 42, `--ignore-eos`, three measured runs after one warmup. `env.txt` is the
profile (token redacted); `engine.txt` the boot's resident lines; `memory.csv` MemAvailable on
both nodes every 5 s.

| Workload | Output tokens | tok/s (runs) | Median tok/s | TTFT (runs) | Median TTFT |
| --- | ---: | --- | ---: | --- | ---: |
| Prose | 1024 | 26.21 / 25.42 / 25.51 | 25.51 | 1.79 / 1.27 / 1.14 s | 1.267 s |
| Code | 1024 | 44.74 / 43.94 / 43.45 | 43.94 | 2.10 / 1.21 / 1.16 s | 1.211 s |
| Random 8K | 256 | 36.35 / 37.34 / 35.69 | 36.35 | 9.45 / 10.81 / 6.89 s | 9.465 s |

Against the previous row (25.60 / 25.55 / 25.29 prose, 39.87 / 41.11 / 44.16 code):

- Prose emits the same text: acceptance 1.99 / 1.97 / 1.98, NVMe 4.65 / 3.77 / 2.85 GB and
  515 / 518 / 517 steps, all identical. Its median did not move (25.55 -> 25.51; means 25.48 ->
  25.71). The greedy gate's -2 ms a step (~2.6%) does not stand out of serving's run-to-run spread.
- Code's acceptance moved again (3.88 / 3.67 / 4.06 -> 4.10 / 3.85 / 4.00), so its 41.11 -> 43.94
  is mostly this boot's acceptance, as in the 2026-10-22 A/B (40.62 native, 44.81 packed).
- Random 8K's time to first token spans 6.9-10.8 s within this boot (6.96-11.17 s before). The
  slow runs carry the long Engram phases (`engram_s` 5.5 s against 4.3-4.6 on the fast ones);
  prefill code did not change, and the fastest runs agree (6.96 -> 6.89 s).

Minimum MemAvailable: 7.61 / 7.53 GB (previous row 7.78 / 8.40) at the 32K allocation. Rank 1
started higher (10.56 against 10.14 GB) and fell during the random 8K phase, where the previous
boot's rose; the change frees 53 MB a rank of draft weights and adds no large allocation. Watch
rank 1 at the profile's 512K, which bottomed ~1 GB above the 2.5 GB floor.
