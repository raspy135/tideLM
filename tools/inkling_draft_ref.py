"""inkling_draft_ref.py: Inkling 2's drafter (mtp.safetensors, "DSpark") in plain PyTorch against the Rust engine's dumps.

`inkling_draft_check` (rust/crates/tidelm-glm/src/bin; the drafter is src/inkling_draft.rs and
tidelm-kernels' cuda/inkling_draft.cu) fabricates the target's 8 hidden-state taps for a short
committed context, stores the drafter's context from them, runs one draft round and writes what it
computed on the way. This reads the same taps and anchor, runs a transcription of the drafter's
specification (docs/inkling-dspark.md section 8: SGLang's
inkling-2-small-support branch as read there; BF16 weights, the rounding points the spec's) with
mtp.safetensors and the target's embed, embed_norm and unembed, and prints each stage's relative
error and row cosine against the engine's files, and whether the 7 draft tokens match.

    inkling_draft_check --model-dir DIR --out OUT
    python -I tools/inkling_draft_ref.py --dense DIR --out OUT

What it checks is the engine's arithmetic against this reading of the specification: nothing here
ran SGLang, so a point both read the same wrong way passes. --gqa and --rope flip the two readings
the specification names as the first suspects (its Q7) if acceptance is ever near zero. The engine
may serve its matrices as FP8 (inkling_draft_check --fp8); this stays BF16, so the errors printed
are then the quantization's. Needs torch (CUDA when there is one; the unembed is 1.85 GB) and
safetensors.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

BF = torch.bfloat16
EPS = 1e-6
D = 4608
RANK = 256
WINDOW = 1024
LOCAL = {0, 1, 3, 4}
LAYERS = 5
HQ, HK, HD = 48, 8, 128
BLOCK = 8
THETA = 1e6
P = "model.llm."


class Ckpt:
    """The target's tensors by name out of the checkpoint directory's shards."""

    def __init__(self, d: str, dev: str):
        from safetensors import safe_open
        self.dir, self.dev, self._open, self._files = d, dev, safe_open, {}
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def handle(self, name: str):
        shard = self.map[name]
        if shard not in self._files:
            self._files[shard] = self._open(os.path.join(self.dir, shard), framework="pt", device=self.dev)
        return self._files[shard]

    def __call__(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def row(self, name: str, i: int) -> torch.Tensor:
        """One row of a matrix, without the matrix."""
        return self.handle(name).get_slice(name)[i:i + 1][0]


class Mtp:
    """mtp.safetensors' tensors by name (the file is not in the weight index)."""

    def __init__(self, d: str, dev: str):
        from safetensors import safe_open
        self.f = safe_open(os.path.join(d, "mtp.safetensors"), framework="pt", device=dev)

    def __call__(self, name: str) -> torch.Tensor:
        return self.f.get_tensor(name)


def read_bf16(path: str, width: int) -> torch.Tensor:
    """A file of bf16 [rows][width] as fp32."""
    bits = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(bits.view(np.int16).astype(np.int32) << 16).view(torch.float32).reshape(-1, width)


def read_f32(path: str, width: int) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(-1, width)


def rms(x, w):
    """Weighted RMSNorm over the last dim: fp32 math, one rounding to bf16."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(BF)


def rope(x, pos, layout="neox"):
    """x bf16 [rows][heads][128] at absolute positions pos [rows]: theta 1e6 over the whole head,
    cos / sin in fp32 from fp32 angles (the reference's table), one rounding."""
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float, device=x.device) / HD))
    ang = pos.to(torch.float)[:, None] * inv[None]
    c, s = ang.cos()[:, None], ang.sin()[:, None]
    if layout == "neox":
        x1, x2 = x[..., :HD // 2].float(), x[..., HD // 2:].float()
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(BF)
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2).to(BF)


def mix_rms(feats, a, w):
    """feats bf16 [rows][8][D], a bf16 [8]: the taps mixed in fp32 in order, RMSNorm, one rounding."""
    m = torch.zeros(feats.shape[0], D, dtype=torch.float32, device=feats.device)
    for t in range(feats.shape[1]):
        m += feats[:, t].float() * a[t].float()
    return (m * torch.rsqrt((m * m).sum(-1, keepdim=True) / D + EPS) * w.float()).to(BF)


class Report:
    def __init__(self, prefix: str):
        self.prefix, self.low, self.missing = prefix, 1.0, []

    def __call__(self, name: str, ref: torch.Tensor, file: str | None = None, f32: bool = False, rows=None, width=None):
        """The engine's PREFIX.<file>.bin against ref [..][n]: the file is [rows][width] (width: n
        unless given - then its first n columns are compared), rows a slice of its rows."""
        path = f"{self.prefix}.{file or name}.bin"
        if not os.path.exists(path):
            self.missing.append(path)
            print(f"  {name}: {path} is missing")
            return
        ref = ref.float().cpu().reshape(-1, ref.shape[-1])
        eng = (read_f32 if f32 else read_bf16)(path, width or ref.shape[-1])[:, :ref.shape[-1]]
        if rows is not None:
            eng = eng[rows]
        if eng.shape != ref.shape:
            self.low = float("nan")
            print(f"  {name}: shapes differ, engine {tuple(eng.shape)} reference {tuple(ref.shape)}")
            return
        rel = ((eng - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        cos = F.cosine_similarity(eng, ref, dim=-1)
        differ = (eng != ref).sum().item()
        self.low = min(self.low, cos.min().item())
        print(f"  {name}: {ref.shape[0]} x {ref.shape[1]}, rel err {rel:.3e}, row cosine min {cos.min().item():.6f}, "
              f"max |diff| {(eng - ref).abs().max().item():.3e} (max |ref| {ref.abs().max().item():.3e}), "
              f"{differ} of {ref.numel()} values differ")


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dense", required=True, help="the checkpoint directory: config.json, mtp.safetensors, the target's shards")
    ap.add_argument("--out", required=True, help="inkling_draft_check's --out directory (taps.bin, draft.json, e.*.bin)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gqa", choices=["blocks", "interleaved"], default="blocks",
                    help="query head h reads key head h // 6 (the specification) or h %% 8")
    ap.add_argument("--rope", choices=["neox", "pairs"], default="neox", help="RoPE's halves (the specification) or interleaved pairs")
    ap.add_argument("--floor", type=float, default=0.99, help="fail under this row cosine at any stage")
    a = ap.parse_args(argv)
    dev = a.device

    j = json.load(open(os.path.join(a.out, "draft.json")))
    anchor, p, lo, start, ctx, first, decoy = (j[k] for k in ("anchor", "p", "lo", "start", "ctx", "first", "decoy"))
    cfg = json.load(open(os.path.join(a.dense, "config.json")))
    vu, cap = cfg["text_config"]["unpadded_vocab_size"], cfg["text_config"]["final_logit_softcapping"]
    M, T = Mtp(a.dense, dev), Ckpt(a.dense, dev)
    rep = Report(os.path.join(a.out, "e"))
    print(f"anchor {anchor} at position {p}, {ctx} context positions from {start}, {dev}, the engine's matrices "
          f"{'FP8' if j['fp8'] else 'BF16'}, rings {j['local_ring']} / {j['global_ring']}")

    # ---- load time
    alpha = torch.softmax(M("fuse_weight").float(), -1).to(BF)              # [6][8]; SGLang rounds it
    rep("alpha", alpha)
    U = T(P + "unembed.weight")                                            # [V][D] bf16
    V = U.shape[0]
    how, hob = M("markov_head.hidden_out.weight").double(), M("markov_head.hidden_out.bias").double()
    fold_w = torch.empty(V, RANK, dtype=BF, device=dev)
    fold_b = torch.empty(V, dtype=BF, device=dev)
    for r0 in range(0, V, 4096):                                            # Inkling2MTPHead.refresh_fold
        head = U[r0:r0 + 4096].double()
        fold_w[r0:r0 + 4096] = head @ how
        fold_b[r0:r0 + 4096] = head @ hob
    del how, hob, head
    rep("fold_w (first rows)", fold_w[:4096], "fold_w")
    rep("fold_b", fold_b[None], "fold_b")

    # ---- the context: every committed position's features -> each layer's keys / values, f_rnn
    bits = np.fromfile(os.path.join(a.out, "taps.bin"), dtype=np.uint16)
    taps = torch.from_numpy(bits.view(np.int16).copy()).view(BF).reshape(8, ctx, D).to(dev)
    feats = taps.transpose(0, 1).contiguous()                              # [ctx][8][D]
    pos_ctx = start + torch.arange(ctx, device=dev)
    # the engine's passes: `first` rows, then the rest behind `decoy` rows of another lane
    passes = [(0, slice(0, first), slice(0, first)), (1, slice(decoy, decoy + ctx - first), slice(first, ctx))]
    K, Vv = [], []
    for l in range(LAYERS):
        s = f"layers.{l}.self_attn."
        c = mix_rms(feats, alpha[l], M(f"layers.{l}.dflare_norm.weight"))
        for ps, rows, mine in passes:
            rep(f"context features, layer {l}, pass {ps}", c[mine], f"c{ps}.{l}", rows=rows)
        k = F.linear(c, M(s + "k_proj_ctx.weight")).reshape(-1, HK, HD)
        v = F.linear(c, M(s + "v_proj_ctx.weight")).reshape(-1, HK, HD)
        K.append(rope(rms(k, M(s + "k_norm_ctx.weight")), pos_ctx, a.rope))
        Vv.append(v)
    f_all = mix_rms(feats, alpha[LAYERS], M("markov_head.dflare_norm.weight"))
    for ps, rows, mine in passes:
        rep(f"f_rnn features, pass {ps}", f_all[mine], f"c{ps}.{LAYERS}", rows=rows)
    f_rnn = f_all[-1]                                                      # position p - 1's
    del feats, taps

    # ---- one draft round
    ex = T.row(P + "embed.weight", anchor).to(dev)
    x = torch.stack([rms(ex, T(P + "embed_norm.weight"))] + [M("mask_embed").to(BF)] * (BLOCK - 1))   # [8][D]
    rep("block input", x, "x0")
    pos = p + torch.arange(BLOCK, device=dev)
    res = x
    xn = rms(res, M("layers.0.input_layernorm.weight"))
    for l in range(LAYERS):
        s = f"layers.{l}.self_attn."
        q = F.linear(xn, M(s + "q_proj.weight")).reshape(BLOCK, HQ, HD)
        k = F.linear(xn, M(s + "k_proj.weight")).reshape(BLOCK, HK, HD)
        v = F.linear(xn, M(s + "v_proj.weight")).reshape(BLOCK, HK, HD)
        q = rope(rms(q, M(s + "q_norm.weight")), pos, a.rope)
        k = rope(rms(k, M(s + "k_norm.weight")), pos, a.rope)
        rep(f"layer {l} queries", q.reshape(BLOCK, -1), f"q{l}")
        # the engine's rings: the window before the block, or what the global ring holds beside it
        klo = max(lo, p - WINDOW) if l in LOCAL else max(lo, p + BLOCK - j["global_ring"])
        Kc = torch.cat([K[l][klo - start:], k])                             # [n][8][128]
        Vc = torch.cat([Vv[l][klo - start:], v])
        rep(f"layer {l} keys | values, {Kc.shape[0] - BLOCK} context + {BLOCK} block rows", torch.cat([Kc, Vc], -1).reshape(Kc.shape[0], -1),
            f"kv{l}")
        qf = q.float()
        if a.gqa == "blocks":
            qg = qf.reshape(BLOCK, HK, HQ // HK, HD)                        # query head h -> key head h // 6
        else:
            qg = qf.reshape(BLOCK, HQ // HK, HK, HD).transpose(1, 2)        # h -> h % 8
        sc = torch.einsum("tgid,ngd->tgin", qg, Kc.float()) * HD ** -0.5    # no mask, no bias
        o = torch.einsum("tgin,ngd->tgid", torch.softmax(sc, -1), Vc.float())
        if a.gqa != "blocks":
            o = o.transpose(1, 2)
        o = o.reshape(BLOCK, HQ * HD).to(BF)
        rep(f"layer {l} attention heads", o, f"o{l}")
        at = F.linear(o, M(s + "o_proj.weight"))
        s1 = res.float() + at.float()
        res = s1.to(BF)
        xn = rms(s1, M(f"layers.{l}.post_attention_layernorm.weight"))     # the norm of the unrounded sum (Q5)
        m = f"layers.{l}.mlp."
        act = (F.silu(F.linear(xn, M(m + "gate_proj.weight")).float()) * F.linear(xn, M(m + "up_proj.weight")).float()).to(BF)
        mo = F.linear(act, M(m + "down_proj.weight"))
        s2 = res.float() + mo.float()
        res = s2.to(BF)
        xn = rms(s2, M(f"layers.{l + 1}.input_layernorm.weight") if l + 1 < LAYERS else M("norm.weight"))
        rep(f"layer {l} residual", res, f"res{l}")
        del q, k, v, Kc, Vc, sc, o, at, act, mo
    rep("final hidden", xn, "hidden")
    h = xn[1:]                                                             # [7][D]
    base = F.linear(h, U)                                                  # [7][V] bf16
    rep("base logits", base[:, :vu], "base", width=V)
    rep("head input [h | f_rnn]", torch.cat([h, f_rnn[None].expand(BLOCK - 1, -1)], -1), "zx")

    w1 = M("markov_head.markov_w1.weight")
    rw, rb = M("markov_head.rnn_proj.weight"), M("markov_head.rnn_proj.bias")
    fwf, fbf = fold_w.float(), fold_b.float()

    def head(forced):
        """The 7 steps; forced: the previous tokens to feed instead of this chain's own."""
        state = torch.zeros(RANK, dtype=BF, device=dev)
        prev, out = anchor, []
        for i in range(BLOCK - 1):
            z = torch.cat([state, w1[prev], h[i], f_rnn])                  # [9728] bf16
            raw = F.linear(z[None], rw, rb)[0].float()
            g_raw, c_raw, o_raw = raw[:RANK], raw[RANK:2 * RANK], raw[2 * RANK:]
            gate = (1 / (1 + torch.exp(-g_raw))).to(BF).float()
            kept = (gate * state.float()).to(BF).float()
            blend = ((1 - gate).to(BF).float() * torch.tanh(c_raw).to(BF).float()).to(BF).float()
            state = (kept + blend).to(BF)
            latent = torch.tanh(o_raw).to(BF)
            bias = (fbf + latent.float() @ fwf.T).to(BF)                    # [V]
            y = (base[i].float() + bias.float()).to(BF)
            y = (cap * torch.tanh(y.float() / cap)).to(BF).float()
            y[vu:] = -float("inf")
            tok = int(torch.argmax(y))                                      # the lowest id on ties
            out.append((tok, float(torch.softmax(y, -1)[tok]), state, latent, y))
            prev = tok if forced is None else forced[i]
        return out

    eng_toks, eng_probs = j["tokens"], j["probs"]
    free = head(None)
    ref_toks = [t for t, *_ in free]
    same = ref_toks == eng_toks
    # the stages of each step, on the engine's own previous tokens when the chains part
    steps = free if same else head(eng_toks)
    for i, (tok, prob, state, latent, y) in enumerate(steps):
        rep(f"step {i} state", state[None], f"state{i}")
        rep(f"step {i} latent", latent[None], f"latent{i}")
        rep(f"step {i} logits", y[None, :vu], f"y{i}", f32=True, width=V)
        top = torch.topk(y, 5)
        print(f"    reference top 5: {[(int(t), round(float(v), 4)) for v, t in zip(top.values, top.indices)]}")
        print(f"    engine    top 5: {[(e['id'], round(e['y'], 4)) for e in j['top5'][i]]}")
        print(f"    token {tok} p {prob:.4f} | engine {eng_toks[i]} p {eng_probs[i]:.4f}")
    print(f"draft tokens: reference {ref_toks}, engine {eng_toks}: {'match' if same else 'DIFFER'}")

    ok = True
    if not same:
        # a near-tie (the engine's token within two bf16 steps of the reference's best) is rounding
        i = next(n for n in range(BLOCK - 1) if ref_toks[n] != eng_toks[n])
        y = free[i][4]
        gap = float(y[ref_toks[i]] - y[eng_toks[i]])
        step = 2.0 ** (np.floor(np.log2(max(abs(float(y[ref_toks[i]])), 1e-3))) - 7)
        print(f"  first difference at step {i}: the reference's logits put its token {gap:.4f} above the engine's (a bf16 step there is {step:.4f})")
        ok = gap <= 2 * step
    if rep.missing:
        print(f"FAIL: {len(rep.missing)} engine files missing")
        ok = False
    if not rep.low >= a.floor:
        print(f"FAIL: a stage's row cosine is {rep.low:.6f}, under {a.floor}")
        ok = False
    if not ok and not same:
        print("FAIL: the draft tokens differ past rounding")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
