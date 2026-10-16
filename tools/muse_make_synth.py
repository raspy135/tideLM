"""Write a synthetic Muse checkpoint with the real vocabulary on this node (both ranks run it),
for exercising the server end to end: tools/muse_pair_run.sh muse_make_synth.py <dir> <tokdir>."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from engine.muse.weights import write_synthetic  # noqa: E402
sys.path.insert(0, "/app/tools")
from muse_pair_test import RAW  # noqa: E402

out, tokdir = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(json.dumps(RAW))
raw["text_config"].update(vocab_size=202048, bos_token_id=200000, eos_token_id=200001, pad_token_id=200018)
if not (out / "model.safetensors").exists():
    write_synthetic(out, raw, seed=0, device="cuda", bias_std=2.0)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(tokdir / f, out / f)
print(f"[rank {os.environ.get('RANK')}] synthetic checkpoint with real vocabulary at {out}", flush=True)
