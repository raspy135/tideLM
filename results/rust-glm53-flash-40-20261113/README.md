# GLM-5.3-Flash: EXL3 4bpw against 3.5bpw, one box and two (2026-11-13)

The comparison written up in docs/glm-flash.md ("The 4 bpw checkpoint"): both checkpoints on the
head box, one host build of main, one frozen copy of the served routing history, the profiles
differing in the pack and the dense directory only (`profile-tp1.env`, `profile-tp2.env`: the 4bpw
arms'; local paths replaced).

- `b35-*`, `b40-*`: bench.py's README protocol on one box, 3.5bpw and 4bpw (`*-engine.txt`: the
  server's log). `b35t2-*`, `b40t2-*`: the same on the pair.
- `nll35.txt`, `nll40.txt`: `--bench-nll` on one box (heldout quick + eval, 57,315 tokens);
  `nll-pair.txt`: `tools/glm_nll_pair.py` over the two per-token dumps (1 MB each, not kept here).
- `nll35-am.txt`: 3.5bpw again from the morning's state file (1.7513: the state is not what moved
  the morning's 1.7169). `q-old.txt`, `q-new.txt`: the released binary e676437 and the day's build
  on heldout_quick (1.9125 both, paired difference 0.0000).

| | 3.5bpw | 4bpw |
|---|---:|---:|
| one box: resident | 8,341 (69.0%) | 7,282 (60.2%) |
| one box: held-out NLL | 1.7510 | 1.7592 (+0.0082 +- 0.0027) |
| one box: prose / code, tok/s | 23.43 / 27.48 | 22.17 / 26.97 |
| two boxes: prose / code, tok/s | 38.62 / 47.98 | 36.81 / 46.80 |

The shipped 4bpw profiles' README rows are a rerun the next day on the serving image:
`results/rust-glm53-flash-40-tp1-20261114/`, `results/rust-glm53-flash-40-tp2-20261114/`.
