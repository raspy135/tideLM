"""Reference hidden states for the Rust engine's parity check, on CPU.

Runs tools/v41_ref.py (the pure-torch forward the Python engine was validated against)
over a short prompt with the ENGINE's semantics -- activations not fake-quantized
(engine/model.py sets act_qdq_fp8 to a bf16 cast unless act_quant), compressed KV
FP4-QDQ'd, engram rows from the table -- and writes the hc stream after every layer plus
the last row's bf16 logits:

    python3 rust/tools/ref_forward.py OUT_DIR "The capital of France is"

Intended for a throwaway container (no GPU needed); peak memory is about one layer,
because the expert cache keeps a single dequantized expert at a time.
"""
from __future__ import annotations

import json
import os
import struct
import sys

os.environ.setdefault("DSV41_DENSE_FP8", "0")   # bf16 dequant on CPU instead of the Triton path
os.environ.setdefault("DSV41_WOA_FP8", "0")

MODEL = os.environ.get("MODEL_DIR", "/m")
REPO = os.environ.get("REPO_DIR", "/repo")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(MODEL, "inference"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import v41_ref as R  # noqa: E402

R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)  # engine semantics (act_quant=False)
if os.environ.get("REF_NO_QDQ") == "1":  # diagnostic: compressed KV without FP4 QDQ
    R.fp4_qdq = lambda x, *a, **k: x
torch.set_num_threads(max(1, os.cpu_count() // 2))


class OneExpert(dict):
    """block_forward's expert cache, holding one expert: each is used once per layer."""

    def __setitem__(self, k, v):
        self.clear()
        super().__setitem__(k, v)


def main():
    out_dir, prompt = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(MODEL, f), "pt", device="cpu")
        return handles[f].get_tensor(name)

    args = R.Args.from_json(os.path.join(MODEL, "inference", "config.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = [0] + tok.encode(prompt, add_special_tokens=False)
    T = len(ids)

    # engram hashes: the checkpoint's own NgramHashState (engine/engram.py::make_hash_state)
    import engram as E
    cfg = json.load(open(os.path.join(MODEL, "inference", "config.json")))

    class A:
        engram_layer_ids = tuple(cfg["engram_layer_ids"]); engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]; engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"]); engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]; engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1; max_seq_len = 64
    hashes = E.NgramHashState(A, E.EngramLayout.from_args(A), tok)(torch.tensor([ids]), 0)[0]  # [T, 2, 24]

    def engram_rows(layer, h):
        """[T, 24] hash ids -> [T, 24, 256] fp32, as engine/engram.py::EngramTable.rows."""
        f = os.path.join(MODEL, index[f"layers.{layer}.engram.embed.weight"])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        wo = base + hdr[f"layers.{layer}.engram.embed.weight"]["data_offsets"][0]
        so = base + hdr[f"layers.{layer}.engram.embed.scale"]["data_offsets"][0]
        out = np.empty((h.numel(), 264), np.uint8)
        with open(f, "rb") as fh:
            for i, r in enumerate(h.reshape(-1).tolist()):
                fh.seek(wo + r * 256); out[i, :256] = np.frombuffer(fh.read(256), np.uint8)
                fh.seek(so + r * 8); out[i, 256:] = np.frombuffer(fh.read(8), np.uint8)
        raw = torch.from_numpy(out)
        vals = raw[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw[:, 256:].float() - 127.0)
        return (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2).view(T, 24, 256)

    embed = get("embed.weight")
    h = embed[torch.tensor(ids)].to(torch.bfloat16).unsqueeze(1).repeat(1, args.hc_mult, 1)
    del embed
    pre_mix = torch.zeros(T, args.hc_mult); pre_mix[:, 0] = 1.0
    st = R.SeqState(h, pre_mix)
    routes = {}
    for L in range(args.n_layers):
        w = R.LayerWeights(get, L, args, "cpu")
        if L in args.engram_layer_ids:
            li = list(args.engram_layer_ids).index(L)
            st.h = R.engram_forward(st.h, engram_rows(L, hashes[:, li, :]), R.EngramWeights(get, L, "cpu"), args)
        R.block_forward(st, w, R.ExpertLoader(get, L, "cpu"), args, OneExpert(),
                        record=lambda i, wt, sc, L=L: routes.__setitem__(L, i.tolist()))
        if L in args.kv_source_layers:
            st.compress_kv.float().numpy().tofile(os.path.join(out_dir, f"ckv{L:02d}.f32"))
        st.h.float().numpy().tofile(os.path.join(out_dir, f"h{L:02d}.f32"))
        print(f"layer {L} done", flush=True)
        del w
    x = R.rmsnorm(R.hc_pre(st.h, st.pre_mix), get("norm.weight").to(torch.bfloat16), args.norm_eps)
    head = get("head.weight").to(torch.bfloat16)
    logits = torch.nn.functional.linear(x[-1:].to(torch.bfloat16), head).float()[0]
    logits.numpy().tofile(os.path.join(out_dir, "logits.f32"))
    json.dump({"ids": ids, "argmax": int(logits.argmax()), "routes": routes},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("argmax", int(logits.argmax()), repr(tok.decode([int(logits.argmax())])))


if __name__ == "__main__":
    main()
