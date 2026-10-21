#!/usr/bin/env python3
"""Dreaming trainer: teach a few routed experts of the top layers, write an overlay generation.

Runs in the serving image with the pair stopped (it needs the box's memory), on one GPU, no EP.

    # 1. once per corpus and cut: everything below the trained layers, with tools/v41_ref.py
    python dream/train.py prefix --corpus dream/results/corpus.jsonl --run RUN --first 36
    # 2. any number of times: train layers 36-39 + head on the "train" sequences
    python dream/train.py train --run RUN --first 36 --experts 8 --lr 1e-4 --epochs 4 --gen NAME

Why only the top layers: the layers below the first trained one never change, so the state they
hand up (the four hyper-connection copies of the residual stream, the mixing vector, and the
compressed KV layer 20 produces for 20-39) is computed once by the prefix pass and is exact for
every later run. Backward then covers only layers FIRST..39 and the head, in plain PyTorch
(tools/v41_ref.py with straight-through activation fake-quant), on sequences of ~100 tokens.

Each trained expert keeps an FP32 master (RUN/masters-GEN.pt, with its Adam state). The forward
uses the master; held-out evaluation and the overlay use the master re-quantized to MXFP4 with the
checkpoint's own block scales -- recomputed scales re-round whole 32-weight blocks and made a toy
model worse than untrained (2026-10-20), and a BF16 master drops updates below ~2^-9 of a weight.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
# dense weights as plain bf16 tensors: the fp8/fp4 Triton GEMMs have no backward
os.environ["DSV41_DENSE_FP8"] = "0"
os.environ["DSV41_WOA_FP8"] = "0"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import v41_ref as R  # noqa: E402
from engine import dream_overlay as D  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/DeepSeek-V4.1-Flash-1017")
DREAM_DIR = Path(os.environ.get("DREAM_DIR", "/models/dream"))
DEV = "cuda"
GRID_MID = (R.FP4_GRID[1:] + R.FP4_GRID[:-1]) / 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ straight-through fake-quant
# The activation fake-quant rounds through a float8 cast (act_qdq_fp8) or bucketize (fp4_qdq):
# no gradient. Forward stays exactly the serving numerics; backward passes the gradient through.
_qdq8, _qdq4 = R.act_qdq_fp8, R.fp4_qdq


def _ste_qdq8(x, block=32):
    xb = x.to(torch.bfloat16)
    return xb + (_qdq8(x.detach(), block) - xb).detach()


def _ste_qdq4(x, group, scale_dtype, grid=None):
    return x + (_qdq4(x.detach(), group, scale_dtype, grid) - x).detach()


R.act_qdq_fp8 = _ste_qdq8
R.fp4_qdq = _ste_qdq4


# ------------------------------------------------------------------ MXFP4 with frozen scales
def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    x = packed.view(torch.uint8)
    return torch.stack([x & 0x0F, x >> 4], dim=-1).flatten(-2)


def quantize_fp4(w: torch.Tensor, scale_u8: torch.Tensor, like: torch.Tensor | None = None,
                 group: int = 32) -> torch.Tensor:
    """fp32 [N, K] -> packed codes int8 [N, K/2] on the checkpoint's own UE8M0 scales [N, K/32].
    The inverse of R.dequant_fp4_packed: codes 0-7 are +{0..6}, 8-15 the negatives, low nibble = even.

    `like`: the checkpoint's packed codes. It stores about 6% of weights as -0 (code 8), which
    dequantizes to the same 0.0 as code 0, so the sign is gone by the time a master exists. A weight
    that is zero before and after keeps its original code, so an untouched weight stays
    byte-identical and the overlay's byte diff counts only real changes."""
    s = R.e8m0_to_float(scale_u8).repeat_interleave(group, 1)
    v = (w.float() / s).clamp(-6.0, 6.0)
    mag = torch.bucketize(v.abs(), GRID_MID.to(w.device))
    code = (mag + 8 * ((v < 0) & (mag > 0))).to(torch.uint8)
    if like is not None:
        orig = unpack_fp4(like)
        code = torch.where((mag == 0) & ((orig & 7) == 0), orig, code)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.int8)


def requant(w: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """What serving will compute with: master -> MXFP4 (frozen scales) -> bf16."""
    return R.dequant_fp4_packed(quantize_fp4(w, scale_u8), scale_u8)


def quantize_fp8(w: torch.Tensor, group: int = 32):
    """Master -> (e4m3 codes [N, K], UE8M0 scale bytes [N, K/32]): one power-of-two scale per 32
    weights of a row, rounded up so the group's amax fits e4m3's 448 -- the same rule as act_qdq_fp8."""
    x = w.float().reshape(w.size(0), -1, group)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    codes = (x / torch.exp2(e)).to(torch.float8_e4m3fn).reshape(w.shape)
    return codes, (e.squeeze(-1) + 127).to(torch.uint8)


def fp8_qdq(w: torch.Tensor) -> torch.Tensor:
    """What an FP8 overlay expert computes with (the engine's FP8 tier, tools/fp4_moe_cuda.cu
    dream_fp8_*). Exact for the checkpoint's own MXFP4 values: an untrained expert is stock."""
    codes, sc = quantize_fp8(w)
    s = R.e8m0_to_float(sc).repeat_interleave(32, 1)
    return (codes.float() * s).to(torch.bfloat16)


# ------------------------------------------------------------------ corpus
def load_corpus(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    seqs = []
    for line in open(path):
        d = json.loads(line)
        ids = tok.encode(d["text"], add_special_tokens=False)
        n = len(d["prompt_ids"])
        if ids[:n] != d["prompt_ids"]:
            raise ValueError(f"{d['id']}: tokenizing the text does not reproduce the served prompt ids")
        seqs.append({**d, "ids": ids, "answer_start": n})
    return seqs


# ------------------------------------------------------------------ prefix pass
def cmd_prefix(a):
    log(f"checkpoint: {MODEL_DIR}")
    run = Path(a.run)
    (run / "engram").mkdir(parents=True, exist_ok=True)
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({index[k] for k in index if ".engram.embed." in k})
    py = sys.executable
    subprocess.run([py, str(ROOT / "tools" / "engram_rows.py"), "--model-dir", MODEL_DIR, "--corpus", a.corpus,
                    "--out", str(run / "engram")]
                   + sum((["--local-shard", os.path.join(MODEL_DIR, s)] for s in shards), []), check=True)
    subprocess.run([py, str(ROOT / "tools" / "expert_trace.py"), "--model-dir", MODEL_DIR, "--corpus", a.corpus,
                    "--engram-dir", str(run / "engram"), "--out", str(run / "prefix"),
                    "--layers", f"0-{a.first - 1}", "--max-len", str(a.max_len)], check=True)
    log(f"prefix state: {run / 'prefix' / 'state' / f'after_layer{a.first - 1}.pt'}")


# ------------------------------------------------------------------ the trainable top
class Experts:
    """One layer's 384 routed experts, packed FP4 on the GPU (7.2 GB); trained ones have FP32 masters."""

    def __init__(self, get, layer):
        self.layer = layer
        p = lambda e, n: get(f"layers.{layer}.ffn.experts.{e}.{n}")  # noqa: E731
        self.packed = {n: torch.stack([p(e, n) for e in range(384)]).to(DEV) for n in D.NAMES}
        self.masters: dict[int, torch.nn.ParameterList] = {}
        self.parent_serve: dict[int, str] = {}  # e -> "fp4" | "fp8": how the parent generation serves it

    def dequant(self, e):
        return [R.dequant_fp4_packed(self.packed[f"{m}.weight"][e], self.packed[f"{m}.scale"][e])
                for m in ("w1", "w2", "w3")]

    def train(self, e, init=None):
        """Give expert e an FP32 master: the checkpoint's weights, or `init` (a saved master, so a
        dream continues where the last one stopped)."""
        if init is not None:
            self.masters[e] = torch.nn.ParameterList([torch.nn.Parameter(w.to(DEV).float()) for w in init])
            return
        self.masters[e] = torch.nn.ParameterList([torch.nn.Parameter(w.float()) for w in self.dequant(e)])
        # The quantizer must be the exact inverse of the checkpoint's dequant, or every trained
        # expert drifts on export even with a zero update. Checked on real weights, every run.
        for m, w in zip(("w1", "w2", "w3"), self.masters[e]):
            assert torch.equal(requant(w.detach(), self.packed[f"{m}.scale"][e]), w.detach().to(torch.bfloat16)), \
                (self.layer, e, m)

    def weights(self, e, served):
        """served: "fp4" / "fp8" = the expert as the engine serves that tier, False = the masters
        themselves (bf16 forward, differentiable), "ste" / "ste8" = forward through the MXFP4 / FP8
        copy with the gradient straight to the master."""
        if e not in self.masters:
            return None
        ws = list(self.masters[e])
        if served == "parent":
            if e not in self.parent_serve:
                return None  # new this dream: the parent served the checkpoint's expert
            served = self.parent_serve[e]
        if served == "fp4":
            return [requant(w.detach(), self.packed[f"{m}.scale"][e]) for m, w in zip(("w1", "w2", "w3"), ws)]
        if served == "fp8":
            return [fp8_qdq(w.detach()) for w in ws]
        if served == "ste":
            # Forward through what serving will read, gradient straight to the master: the loss
            # stays high until the SERVED expert has learned, so repeated sessions keep pushing
            # until weights cross the MXFP4 grid instead of stopping once the master has learned.
            return [w.to(torch.bfloat16) + (requant(w.detach(), self.packed[f"{m}.scale"][e])
                                            - w.detach().to(torch.bfloat16))
                    for m, w in zip(("w1", "w2", "w3"), ws)]
        if served == "ste8":
            return [w.to(torch.bfloat16) + (fp8_qdq(w.detach()) - w.detach().to(torch.bfloat16)) for w in ws]
        return [w.to(torch.bfloat16) for w in ws]


def frozen_ffn(y, rw, w1p, s1, w2p, s2, w3p, s3, limit):
    w1, w2, w3 = (R.dequant_fp4_packed(w, s) for w, s in ((w1p, s1), (w2p, s2), (w3p, s3)))
    return R.expert_ffn(y, w1, w2, w3, limit, rw).float()


def layer_forward(states, w, ex: Experts, args, served: bool, record=None):
    """One block over a batch of sequences: attention per sequence, the MoE over all their rows at
    once (each expert is dequantized once per batch, not once per sequence)."""
    parts = []
    for st in states:
        x = st.h
        residual = x
        attn_pre, attn_post, attn_comb = R.hc_mixes(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, args)
        y = R.rmsnorm(R.hc_pre(x, st.pre_mix), w.attn_norm, args.norm_eps)
        x = R.hc_post(R.attention(y, w, st, args), residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = R.hc_mixes(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, args)
        y = R.rmsnorm(R.hc_pre(x, attn_pre), w.ffn_norm, args.norm_eps)
        parts.append((x, y, ffn_pre, ffn_post, ffn_comb))
    y = torch.cat([p[1] for p in parts])
    rw, idx, _ = R.router(y, w, args)
    if record is not None:
        record(rw.detach(), idx)
    out = torch.zeros_like(y, dtype=torch.float32)
    for e in torch.unique(idx).tolist():
        rows, top = torch.where(idx == e)
        ws = ex.weights(e, served)
        if ws is not None:
            contrib = R.expert_ffn(y[rows], *ws, args.swiglu_limit, rw[rows, top, None]).float()
        else:
            pk = ex.packed
            # recomputed in backward: saving 300+ dequantized bf16 experts a layer would not fit
            contrib = checkpoint(frozen_ffn, y[rows], rw[rows, top, None],
                                 pk["w1.weight"][e], pk["w1.scale"][e], pk["w2.weight"][e], pk["w2.scale"][e],
                                 pk["w3.weight"][e], pk["w3.scale"][e], args.swiglu_limit, use_reentrant=False)
        out = out.index_add(0, rows, contrib)
    out = out + R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, args.swiglu_limit).float()
    out = out.to(y.dtype)
    off = 0
    for st, (x, yy, ffn_pre, ffn_post, ffn_comb) in zip(states, parts):
        n = yy.size(0)
        st.h = R.hc_post(out[off:off + n], x, ffn_post, ffn_comb)
        st.pre_mix = ffn_pre
        off += n


def answer_logits(st, seq, norm_w, head):
    """Logits at the rows that predict the answer tokens (+EOS), and those tokens."""
    s = seq["answer_start"]
    h = R.rmsnorm(R.hc_pre(st.h[s - 1:-1], st.pre_mix[s - 1:-1]), norm_w, R_ARGS.norm_eps)
    return F.linear(h, head).float(), torch.tensor(seq["ids"][s:], device=DEV)


def answer_nll(st, seq, norm_w, head):
    """Mean NLL and top-1 over the answer tokens (+EOS) of one sequence."""
    logits, tgt = answer_logits(st, seq, norm_w, head)
    return F.cross_entropy(logits, tgt), (logits.argmax(-1) == tgt).float().mean()


def kl_to_stock(st, seq, norm_w, head, teacher_lp):
    """KL(stock || this model) per answer token, averaged: how far the reply distribution moved on
    text the trained facts have nothing to do with. Stock's log-probs were taken before training."""
    logits, _ = answer_logits(st, seq, norm_w, head)
    lp = F.log_softmax(logits, -1)
    t = teacher_lp.float()
    return (t.exp() * (t - lp)).sum(-1).mean()


def fresh_states(prefix, batch):
    out = []
    for i in batch:
        p = prefix[i]
        st = R.SeqState(p["h"].to(DEV), p["pre_mix"].to(DEV))
        st.compress_kv = None if p["compress_kv"] is None else p["compress_kv"].to(DEV)
        st.compress_ratio = p["compress_ratio"]
        out.append(st)
    return out


def run_states(prefix, batch, layers, experts, args, served, record=None):
    states = fresh_states(prefix, batch)
    for L, w in layers.items():
        layer_forward(states, w, experts[L], args, served,
                      None if record is None else (lambda rw, idx, L=L: record(L, states, batch, rw, idx)))
    return states


def run_batch(seqs, prefix, batch, layers, experts, args, norm_w, head, served, record=None):
    states = run_states(prefix, batch, layers, experts, args, served, record)
    nll, acc = zip(*(answer_nll(st, seqs[i], norm_w, head) for st, i in zip(states, batch)))
    return torch.stack(nll), torch.stack(acc)


def run_kl(seqs, prefix, batch, layers, experts, args, norm_w, head, served, teacher):
    states = run_states(prefix, batch, layers, experts, args, served)
    return torch.stack([kl_to_stock(st, seqs[i], norm_w, head, teacher[i]) for st, i in zip(states, batch)])


@torch.no_grad()
def evaluate_kl(seqs, prefix, ids, layers, experts, args, norm_w, head, bs, teacher, served="fp4"):
    if not ids:
        return float("nan")
    return torch.cat([run_kl(seqs, prefix, ids[k:k + bs], layers, experts, args, norm_w, head, served, teacher)
                      for k in range(0, len(ids), bs)]).mean().item()


@torch.no_grad()
def evaluate(seqs, prefix, ids, layers, experts, args, norm_w, head, bs, served="fp4"):
    nll, acc = [], []
    for k in range(0, len(ids), bs):
        n, a_ = run_batch(seqs, prefix, ids[k:k + bs], layers, experts, args, norm_w, head, served=served)
        nll.append(n)
        acc.append(a_)
    return torch.cat(nll).mean().item(), torch.cat(acc).mean().item()


# SIGTERM (`docker stop`, sent by dream/dream.py when the user types `@@dream wake`): finish the
# current step, save the masters and Adam state as masters-GEN.partial.pt -- the same layout --resume
# reads -- and exit 3, so tonight's progress becomes tomorrow night's starting point. Nothing is
# exported or served from an interrupted run. The driver gives `docker stop` a long grace period:
# the save is ~13 GB for 32 experts.
_STOP = {"requested": False}


def _on_term(signum, frame):
    _STOP["requested"] = True
    log("SIGTERM: stopping after this step and saving the masters for the next dream")


def save_partial(run, gen, chosen, experts, opt, top, a, curve):
    path = run / f"masters-{gen}.partial.pt"
    torch.save({"chosen": chosen, "lr": a.lr, "epochs": a.epochs, "forward": a.forward, "parent": a.resume,
                "interrupted": True,
                "masters": {f"{L}.{e}": [p.detach().cpu() for p in experts[L].masters[e]]
                            for L in top for e in chosen[L]},
                "opt": {f"{L}.{e}": [{k: v.detach().cpu() for k, v in opt.state[p].items()}
                                     for p in experts[L].masters[e]]
                        for L in top for e in chosen[L]},
                "curve": curve}, path)
    log(f"interrupted: wrote {path}; resume from it with --resume")


def cmd_train(a):
    R.ATTN_CHECKPOINT = True  # see v41_ref: the score tensors are recomputed in backward
    import signal
    signal.signal(signal.SIGTERM, _on_term)
    global R_ARGS
    torch.manual_seed(0)
    random.seed(0)
    run = Path(a.run)
    seqs = load_corpus(a.corpus)
    st_path = run / "prefix" / "state" / f"after_layer{a.first - 1}.pt"
    prefix = torch.load(st_path, map_location=DEV)
    assert prefix["layer"] == a.first - 1
    prefix = prefix["states"]
    assert len(prefix) == len(seqs), "prefix state was made from a different corpus"
    train_ids = [i for i, s in enumerate(seqs) if s["category"] == "train"]
    held_ids = [i for i, s in enumerate(seqs) if s["category"] == "heldout"]
    gtrain_ids = [i for i, s in enumerate(seqs) if s["category"] == "general-train"]
    geval_ids = [i for i, s in enumerate(seqs) if s["category"] == "general-eval"]

    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    args = R_ARGS = R.Args.from_json(os.path.join(MODEL_DIR, "inference", "config.json"))
    sys.path.insert(0, str(ROOT / "tools"))
    from expert_trace import ShardGetter
    get = ShardGetter(MODEL_DIR, index)
    top = range(a.first, args.n_layers)
    layers = {L: R.LayerWeights(get, L, args, DEV) for L in top}
    experts = {L: Experts(get, L) for L in top}
    norm_w = get("norm.weight").to(DEV).to(torch.bfloat16)
    head = get("head.weight").to(DEV).to(torch.bfloat16)
    log(f"loaded layers {a.first}-{args.n_layers - 1} + head; "
        f"{len(train_ids)} train / {len(held_ids)} held-out sequences")

    # stock's own distribution on the general sequences, before any expert is trained: the target
    # of the stay-close penalty and the reference of the drift gate
    def reference(served):
        out = {}
        with torch.no_grad():
            gids = gtrain_ids + geval_ids
            for k in range(0, len(gids), a.batch):
                batch = gids[k:k + a.batch]
                for st, i in zip(run_states(prefix, batch, layers, experts, args, served), batch):
                    out[i] = F.log_softmax(answer_logits(st, seqs[i], norm_w, head)[0], -1).to(torch.bfloat16)
        return out

    teacher_stock = reference("fp4")  # no expert has a master yet: this is the checkpoint
    if gtrain_ids or geval_ids:
        log(f"stock reference on {len(gtrain_ids)} general-train / {len(geval_ids)} general-eval sequences")

    # choose the experts: routed weight mass on ANSWER tokens of the training sequences
    mass = {L: torch.zeros(384, device=DEV) for L in top}

    def record(L, states, batch, rw, idx):
        off = 0
        for st, i in zip(states, batch):
            n = st.h.size(0)
            s = seqs[i]["answer_start"] - 1  # the rows that predict answer tokens
            mass[L].index_add_(0, idx[off + s:off + n - 1].flatten(), rw[off + s:off + n - 1].flatten().float())
            off += n

    with torch.no_grad():
        for k in range(0, len(train_ids), a.batch):
            run_batch(seqs, prefix, train_ids[k:k + a.batch], layers, experts, args, norm_w, head,
                      served="fp4", record=record)
    prev = torch.load(a.resume, map_location="cpu") if a.resume else {"chosen": {}, "masters": {}, "opt": {}}
    chosen = {}
    for L in top:
        kept = list(prev["chosen"].get(L, []))
        fresh = [e for e in mass[L].argsort(descending=True).tolist() if e not in kept]
        chosen[L] = kept + fresh[:max(0, a.experts - len(kept))]
        share = mass[L][chosen[L]].sum() / mass[L].sum()
        log(f"layer {L}: training experts {chosen[L]} ({len(kept)} resumed; "
            f"{100 * share:.1f}% of answer-token routing mass)")
        for e in chosen[L]:
            experts[L].train(e, prev["masters"].get(f"{L}.{e}"))
    # The stay-close target. Training sets are fresh every dream (the user's design, 2026-10-20), so
    # last week's lessons are not in tonight's set; what protects them is penalizing drift from the
    # generation being SERVED, not from stock. Its experts serve as the parent manifest says.
    teacher = teacher_stock
    if a.resume:
        man = json.loads((DREAM_DIR / "serve" / a.parent / D.MANIFEST).read_text()) if a.parent else {"experts": []}
        for x in man["experts"]:
            if x["layer"] in experts:
                experts[x["layer"]].parent_serve[x["expert"]] = x.get("serve", "fp4")
        teacher = reference("parent")
        log(f"stay-close target: the parent generation {a.parent} "
            f"({sum(len(ex.parent_serve) for ex in experts.values())} of its experts in these layers)")
    params = [p for L in top for e in chosen[L] for p in experts[L].masters[e]]
    opt = torch.optim.Adam(params, lr=a.lr)
    # Adam's moments per expert, so a resumed expert keeps its step count and running averages
    # instead of restarting with a cold optimizer (whose first steps are full-size in any direction)
    for L in top:
        for e in chosen[L]:
            for p, st in zip(experts[L].masters[e], prev["opt"].get(f"{L}.{e}", [])):
                opt.state[p] = {k: v.to(DEV) for k, v in st.items()}

    h0, acc0 = evaluate(seqs, prefix, held_ids, layers, experts, args, norm_w, head, a.batch)
    t0, tacc0 = evaluate(seqs, prefix, train_ids, layers, experts, args, norm_w, head, a.batch)
    d0 = evaluate_kl(seqs, prefix, geval_ids, layers, experts, args, norm_w, head, a.batch, teacher)
    log(f"epoch 0: train nll {t0:.3f} top1 {tacc0:.3f} | held-out nll {h0:.3f} top1 {acc0:.3f} "
        f"| drift (general-eval KL to stock) {d0:.4f}")
    curve = [{"epoch": 0, "train_nll": t0, "train_top1": tacc0, "heldout_nll": h0, "heldout_top1": acc0,
              "drift": d0}]
    # early stopping: the masters (and optimizer state) of the best held-out epoch are what is
    # exported, not the last epoch's -- tiny-D was trained 16 epochs past its best and looped
    best = {"heldout_nll": h0, "epoch": 0, "masters": None, "opt": None}
    serve_fp8 = a.fp8_budget >= sum(len(chosen[L]) for L in top)
    for ep in range(1, a.epochs + 1):
        order = train_ids[:]
        random.shuffle(order)
        tot = 0.0
        for k in range(0, len(order), a.batch):
            batch = order[k:k + a.batch]
            # forward "mix": the master view keeps the FP32 master (and its FP8 copy) meaningful, so a
            # new lesson works in FP8 after one dream; the beta-weighted MXFP4 view pushes weights
            # across the 4-bit grid a little each dream, so a lesson that keeps coming back ends up
            # holding in FP4. Arm A (master only) never reached FP4; arm D (FP4 only) broke FP8.
            # forward "tiers" (the default since the FP8 kernel, 2026-10-21): through the FP8 copy the
            # engine serves a fresh lesson from, plus the beta-weighted MXFP4 view so a lesson that
            # keeps coming back crosses the 4-bit grid. 2026-10-21's "mix" test moved the master by
            # at most 5e-4 in 4 epochs: FP8 held-out 2.075 -> 1.778, MXFP4 not at all.
            views = {"master": [(False, 1.0)], "fp4": [("ste", 1.0)], "mix": [(False, 1.0), ("ste", a.beta)],
                     "tiers": [("ste8", 1.0), ("ste", a.beta)]}[a.forward]
            gb = random.sample(gtrain_ids, min(a.batch, len(gtrain_ids))) if gtrain_ids and a.kl_weight > 0 else []
            # Each term is backpropagated as soon as it is built: the gradient is the same sum, but
            # only one forward graph is alive at a time. One loss.backward() over all of them held
            # four (2 views x lessons + KL) and, with thinking-length sequences, ran the box out of
            # its 121 GB on the first step of 2026-10-21's test dream.
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for view, wt in views:
                nll, _ = run_batch(seqs, prefix, batch, layers, experts, args, norm_w, head, served=view)
                term = wt * nll.mean()
                term.backward()
                loss += float(term)
                del nll, term
                if gb:
                    term = wt * a.kl_weight * run_kl(seqs, prefix, gb, layers, experts, args, norm_w,
                                                     head, view, teacher).mean()
                    term.backward()
                    loss += float(term)
                    del term
            opt.step()
            tot += loss * len(batch)
            if _STOP["requested"]:
                save_partial(run, a.gen, chosen, experts, opt, top, a, curve)
                sys.exit(3)
        if ep % a.eval_every and ep != a.epochs:
            log(f"epoch {ep}: step loss {tot / len(order):.3f}")
            continue
        h, acc = evaluate(seqs, prefix, held_ids, layers, experts, args, norm_w, head, a.batch)
        t, tacc = evaluate(seqs, prefix, train_ids, layers, experts, args, norm_w, head, a.batch)
        # The same held-out set through the unquantized masters: the gap to the served numbers is
        # what MXFP4 (frozen scales) takes away -- a master that learns while the served model does
        # not means the updates are below the grid, not that training failed.
        hm, accm = evaluate(seqs, prefix, held_ids, layers, experts, args, norm_w, head, a.batch, served=False)
        h8, acc8 = evaluate(seqs, prefix, held_ids, layers, experts, args, norm_w, head, a.batch, served="fp8")
        dr = evaluate_kl(seqs, prefix, geval_ids, layers, experts, args, norm_w, head, a.batch, teacher)
        dr8 = evaluate_kl(seqs, prefix, geval_ids, layers, experts, args, norm_w, head, a.batch, teacher,
                          served="fp8")
        drs = dr if teacher is teacher_stock else evaluate_kl(seqs, prefix, geval_ids, layers, experts, args,
                                                               norm_w, head, a.batch, teacher_stock)
        moved = max((p.detach() - experts[L].dequant(e)[j].float()).abs().max().item()
                    for L in top for e in chosen[L] for j, p in enumerate(experts[L].masters[e]))
        log(f"epoch {ep}: step loss {tot / len(order):.3f} | served: train nll {t:.3f} top1 {tacc:.3f} "
            f"| held-out nll {h:.3f} top1 {acc:.3f} | if fp8: {h8:.3f} top1 {acc8:.3f} "
            f"| masters: held-out nll {hm:.3f} top1 {accm:.3f} "
            f"| drift {dr:.4f} fp8 {dr8:.4f} (vs stock {drs:.4f}) | max |master - checkpoint| {moved:.4g}")
        curve.append({"epoch": ep, "train_nll": t, "train_top1": tacc, "heldout_nll": h, "heldout_top1": acc,
                      "master_heldout_nll": hm, "master_heldout_top1": accm,
                      "fp8_heldout_nll": h8, "fp8_heldout_top1": acc8, "drift": dr, "drift_fp8": dr8,
                      "drift_stock": drs, "max_delta": moved})
        # Early stopping on the copy that will be served: FP8 when the budget covers every trained
        # expert (each one not consolidated into MXFP4 then serves FP8), MXFP4 otherwise.
        hs = h8 if serve_fp8 else h
        if hs < best["heldout_nll"]:
            best = {"heldout_nll": hs, "epoch": ep,
                    "masters": {(L, e): [p.detach().clone() for p in experts[L].masters[e]]
                                for L in top for e in chosen[L]},
                    "opt": {(L, e): [{k: v.detach().clone() for k, v in opt.state[p].items()}
                                     for p in experts[L].masters[e]] for L in top for e in chosen[L]}}

    if best["masters"] is not None and best["epoch"] != a.epochs:
        for (L, e), ws in best["masters"].items():
            for p, w, st in zip(experts[L].masters[e], ws, best["opt"][(L, e)]):
                p.data.copy_(w)
                opt.state[p] = st
        log(f"early stop: exporting epoch {best['epoch']} (held-out nll {best['heldout_nll']:.3f}), not {a.epochs}")
    final = next(c for c in curve if c["epoch"] == best["epoch"]) if best["masters"] is not None else curve[-1]

    # save the masters and their optimizer state (the next dream resumes from them)
    torch.save({"chosen": chosen, "lr": a.lr, "epochs": a.epochs, "forward": a.forward, "parent": a.resume,
                "masters": {f"{L}.{e}": [p.detach().cpu() for p in experts[L].masters[e]]
                            for L in top for e in chosen[L]},
                "opt": {f"{L}.{e}": [{k: v.detach().cpu() for k, v in opt.state[p].items()}
                                     for p in experts[L].masters[e]]
                        for L in top for e in chosen[L]},
                "curve": curve}, run / f"masters-{a.gen}.pt")

    # Every trained expert is written twice, MXFP4 and FP8. Which one it is SERVED as:
    #   kept = how much of the master's learned change its MXFP4 copy carries,
    #          <fp4(m) - c, m - c> / |m - c|^2 over its three matrices (c = the checkpoint).
    #   kept >= --consolidated: the lesson survives 4 bits -> MXFP4, no FP8 slot needed.
    #   otherwise the largest learned changes get the FP8 budget; the rest serve MXFP4 anyway
    #   (whatever survives) and keep training until they consolidate.
    gen = DREAM_DIR / "serve" / a.gen
    gen.mkdir(parents=True)
    wm = index["weight_map"]
    entries, changed = [], 0
    for L in top:
        for e in chosen[L]:
            # the base the masters were trained from: the engine refuses the expert on any other
            base = D.base_sha256(MODEL_DIR, wm, L, e)
            t = D.read_expert(MODEL_DIR, wm, L, e)
            t8 = {}
            dot = norm = 0.0
            for m, p, c in zip(("w1", "w2", "w3"), experts[L].masters[e], experts[L].dequant(e)):
                w = p.detach()
                sc = experts[L].packed[f"{m}.scale"][e]
                codes = quantize_fp4(w, sc, like=experts[L].packed[f"{m}.weight"][e])
                d = w - c.float()
                dot += float(((R.dequant_fp4_packed(codes, sc).float() - c.float()) * d).sum())
                norm += float((d * d).sum())
                dtype, shape, _ = t[f"{m}.weight"]
                raw = codes.cpu().numpy().tobytes()
                old = experts[L].packed[f"{m}.weight"][e].cpu().numpy().tobytes()
                changed += sum(x != y for x, y in zip(raw, old))
                t[f"{m}.weight"] = (dtype, shape, raw)
                c8, s8 = quantize_fp8(w)
                t8[f"{m}.weight"] = ("F8_E4M3", list(w.shape), c8.view(torch.uint8).cpu().numpy().tobytes())
                t8[f"{m}.scale"] = ("F8_E8M0", list(s8.shape), s8.cpu().numpy().tobytes())
            f, f8 = D.file_name(L, e), D.file_name(L, e).replace(".safetensors", ".fp8.safetensors")
            entries.append({"layer": L, "expert": e, "file": f, "sha256": D.write_expert(str(gen / f), L, e, t),
                            "fp8_file": f8, "fp8_sha256": D.write_expert(str(gen / f8), L, e, t8),
                            "base_sha256": base, "kept": dot / norm if norm > 0 else 1.0,
                            "learned": norm ** 0.5})

    # Decided by measured loss first. Training against MXFP4 leaves a master whose own values mean
    # nothing -- arm D (2026-10-20): MXFP4 held-out 0.27 nats, its FP8 copy 5.3, worse than the
    # untrained 3.05 -- and `kept` reads 0.2-0.45 for experts that are fully consolidated. So when
    # the MXFP4 generation is at least as good as FP8 on held-out, everything serves MXFP4; `kept`
    # only ranks the FP8 budget for runs trained on the master, where it is meaningful.
    last = final
    if last.get("fp8_heldout_nll") is not None and last["heldout_nll"] <= last["fp8_heldout_nll"] + a.fp4_margin:
        pending = []
        log(f"MXFP4 held-out {last['heldout_nll']:.3f} vs FP8 {last['fp8_heldout_nll']:.3f}: "
            f"all {len(entries)} experts consolidated, serving MXFP4")
    else:
        pending = sorted((x for x in entries if x["kept"] < a.consolidated), key=lambda x: -x["learned"])
    fp8 = {(x["layer"], x["expert"]) for x in pending[:a.fp8_budget]}
    for x in entries:
        x["serve"] = "fp8" if (x["layer"], x["expert"]) in fp8 else "fp4"
    n_cons = len(entries) - len(pending)
    # The drift gate judges what will be served: the FP8 copies' drift when any expert serves FP8
    # (the larger of the two when the generation mixes tiers), the MXFP4 drift otherwise.
    dr4, dr8 = final.get("drift"), final.get("drift_fp8")
    drift = dr4 if not fp8 or dr8 is None else (dr8 if len(fp8) == len(entries) else max(dr4, dr8))
    gate = {"drift": drift, "max_drift": a.max_drift, "tier": "fp8" if fp8 else "fp4",
            "pass": drift is None or drift != drift or drift <= a.max_drift}
    log(f"drift gate ({gate['tier']}): general-eval KL {drift} vs max {a.max_drift}: "
        f"{'PASS' if gate['pass'] else 'FAIL -- do not serve this generation'}")
    man = {"version": D.VERSION, "model_dir": MODEL_DIR,
           "base_index_sha256": D.sha256_file(os.path.join(MODEL_DIR, "model.safetensors.index.json")),
           "parent": a.parent, "note": a.note or f"train {run.name}: layers {a.first}+, {a.experts}/layer, "
                                                 f"lr {a.lr}, {a.epochs} epochs, forward {a.forward}",
           "fp8_budget": a.fp8_budget, "consolidated_at": a.consolidated, "exported_epoch": best["epoch"],
           "drift_gate": gate, "experts": entries, "curve": curve}
    (gen / D.MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    ks = sorted(x["kept"] for x in entries)
    log(f"wrote {gen}: {len(entries)} experts ({n_cons} consolidated into MXFP4, {len(fp8)} serve FP8, "
        f"{len(entries) - n_cons - len(fp8)} over budget); MXFP4 keeps {ks[0]:.2f} / {ks[len(ks) // 2]:.2f} / "
        f"{ks[-1]:.2f} (min/median/max) of the learned change; {changed} packed FP4 bytes differ from the checkpoint")


def cmd_export_fp8(a):
    """A new generation that serves every expert of an existing one from FP8 made from its masters.

    For runs whose generation predates the FP8 copies (arm A) or chose MXFP4 for all of them: the
    MXFP4 files are copied unchanged (the engine still loads them into the arena), and each expert
    gets an FP8 file quantized from its FP32 master and serve=fp8 in the new manifest."""
    import shutil
    src = DREAM_DIR / "serve" / a.source
    man = json.loads((src / D.MANIFEST).read_text())
    masters = torch.load(a.masters, map_location="cpu")["masters"]
    dst = DREAM_DIR / "serve" / a.gen
    dst.mkdir(parents=True)
    # The source's base hashes carry over; computing them here would label masters trained on
    # another checkpoint as this one's (tiny-A was trained on the pre-1017 weights).
    if any(not x.get("base_sha256") for x in man["experts"]):
        raise SystemExit(f"{a.source} has no base hashes (made before 2026-10-21): retrain it instead")
    for x in man["experts"]:
        L, e = x["layer"], x["expert"]
        shutil.copyfile(src / x["file"], dst / x["file"])
        t8 = {}
        for m, w in zip(("w1", "w2", "w3"), masters[f"{L}.{e}"]):
            c8, s8 = quantize_fp8(w.float())
            t8[f"{m}.weight"] = ("F8_E4M3", list(w.shape), c8.view(torch.uint8).numpy().tobytes())
            t8[f"{m}.scale"] = ("F8_E8M0", list(s8.shape), s8.numpy().tobytes())
        x["fp8_file"] = D.file_name(L, e).replace(".safetensors", ".fp8.safetensors")
        x["fp8_sha256"] = D.write_expert(str(dst / x["fp8_file"]), L, e, t8)
        x["serve"] = "fp8"
    man["note"] = f"{a.source} with every expert served FP8 from {Path(a.masters).name}"
    (dst / D.MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    log(f"wrote {dst}: {len(man['experts'])} experts, all serve=fp8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("prefix")
    x.add_argument("--corpus", required=True)
    x.add_argument("--run", required=True)
    x.add_argument("--first", type=int, required=True, help="first trained layer; the pass covers 0..first-1")
    x.add_argument("--max-len", type=int, default=4096,
                   help="longest sequence the trace accepts (medium-effort lessons with reasoning reach ~1.3K)")
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--first", type=int, required=True)
    t.add_argument("--experts", type=int, default=8, help="trained experts per layer")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--forward", default="tiers", choices=("master", "fp4", "mix", "tiers"),
                   help="training forward: the masters, the MXFP4 they serve as (straight-through), "
                        "master + beta * MXFP4, or FP8 + beta * MXFP4 (both straight-through)")
    t.add_argument("--beta", type=float, default=0.3, help="--forward mix / tiers: weight of the MXFP4 view")
    t.add_argument("--eval-every", type=int, default=1, help="held-out evaluation every N epochs (and the last)")
    t.add_argument("--resume", help="masters-*.pt of an earlier dream: continue its experts and optimizer")
    t.add_argument("--fp8-budget", type=int, default=64, help="trained experts that may serve as FP8")
    t.add_argument("--consolidated", type=float, default=0.8,
                   help="share of the learned change the MXFP4 copy must carry to serve as MXFP4")
    t.add_argument("--kl-weight", type=float, default=1.0,
                   help="weight of the stay-close penalty (KL to stock on general-train sequences)")
    t.add_argument("--max-drift", type=float, default=0.05,
                   help="drift gate: max general-eval KL to stock (nats/token) for a servable generation")
    t.add_argument("--fp4-margin", type=float, default=0.05,
                   help="MXFP4 counts as consolidated when its held-out nll is within this of FP8's")
    t.add_argument("--gen", required=True, help="overlay generation name to write under serve/")
    t.add_argument("--parent")
    t.add_argument("--note", default="")
    f = sub.add_parser("export-fp8", help="serve an existing generation's experts as FP8 from its masters")
    f.add_argument("--source", required=True, help="generation under serve/ whose MXFP4 files are reused")
    f.add_argument("--masters", required=True, help="the masters-*.pt that generation was written from")
    f.add_argument("--gen", required=True)
    a = p.parse_args()
    {"prefix": cmd_prefix, "train": cmd_train, "export-fp8": cmd_export_fp8}[a.cmd](a)


if __name__ == "__main__":
    main()
