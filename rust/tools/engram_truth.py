import hashlib, json, sys
sys.path.insert(0, "/m/inference")
import numpy as np, torch
import engram as E
from transformers import AutoTokenizer

cfg = json.load(open("/m/inference/config.json"))
class A:
    engram_layer_ids = tuple(cfg["engram_layer_ids"]); engram_max_ngram_size = cfg["engram_max_ngram_size"]
    engram_n_heads = cfg["engram_n_heads"]; engram_vocab_size = cfg["engram_vocab_size"]
    engram_num_embeddings = tuple(cfg["engram_num_embeddings"]); engram_head_dim = cfg["engram_head_dim"]
    engram_pad_id = cfg["engram_pad_id"]; engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
    max_batch_size = 1; max_seq_len = 64
tok = AutoTokenizer.from_pretrained("/m")
layout = E.EngramLayout.from_args(A)
st = E.NgramHashState(A, layout, tok)
tm = st.token_map.numpy().astype(np.int32)
ids = tok.encode("The capital of France is Paris. The capital of Japan is", add_special_tokens=False)
h = st(torch.tensor([ids]), 0)[0]  # [L, 2, 24]
# a second call continuing the sequence, to pin the incremental cache semantics
more = [ids[-1], 1234, 99]
h2 = st(torch.tensor([more[1:]]), len(ids))[0]
out = {
    "numpy": np.__version__,
    "vocab_len": len(tok), "compressed_vocab": int(tm.max()) + 1, "pad_id": int(st.pad_id),
    "token_map_sha256": hashlib.sha256(tm.tobytes()).hexdigest(),
    "token_map_head": tm[:300].tolist(), "token_map_tail": tm[-300:].tolist(),
    "primes": st.primes.tolist(), "offsets": st.offsets.tolist(), "multipliers": st.multipliers.tolist(),
    "rng_check": {str(s): np.random.default_rng(s).integers(0, 46545386853491, size=6, dtype=np.int64).tolist()
                  for s in (0, 10007, 140098, 123456789)},
    "prompt_ids": ids, "hashes": h.tolist(), "cont_ids": more[1:], "cont_hashes": h2.tolist(),
    "decode_samples": {str(i): tok.backend_tokenizer.decode([i], skip_special_tokens=False)
                       for i in (0, 1, 2, 100, 1000, 128799, 129279)},
}
json.dump(out, open("/out/engram_truth.json", "w"))
print("ok", out["compressed_vocab"], out["token_map_sha256"][:16], len(ids))
