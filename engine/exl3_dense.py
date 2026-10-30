"""DSV41_EXL3_DENSE=1: the decode graph's attention and shared-expert matrices in EXL3.

The routed experts already come from the EXL3 pack; their dense neighbours were still FP8, ~86 MB
a rank per layer, a quarter of a decode step's bytes (results/exl3-tune-20261008). The EXL3
checkpoint stores the same matrices at 5 bits (one layer's wq_a/wkv at 6, the shared experts at 4-5),
and the routed experts' grouped kernel computes a T-row GEMV from them L2-cold at 0.6-0.8x the FP8
kernel's time (tools/bench_exl3_dense_decode.py).

Scope: the decode graph only (engine/fastdecode.py reads `w._x3`). Prefill keeps the FP8 weights, so
the prefix cache's bit-invariance (resume == cold) is untouched, and the drafter stays FP8. This is a
numerics change -- EXL3 reconstructs the base weights with its own error, on fp16 activations without
the FP8 activation fake-quant -- so it is opt-in and pinned in the boot guard with the pack digest.

MEASURED AND OFF (2026-10-08, RESULTS.md): paired over 24 prompts it saved
1.30 +- 0.53 ms a step but cost 0.071 +- 0.024 accepted tokens a step (the drafter was trained on
the FP8 target; teacher-forced top-1 vs FP8 is 0.976), net -0.22 +- 0.53 tok/s. Kept for the next
attempt: fused rotations and a drafter that matches the target are the preconditions.

Pack: tools/pack_exl3_dense.py, one safetensors per rank (~2.1 GB), already sliced the way
engine/tensor_parallel.py shards FP8. The abliteration overlay (DSV41_ABLIT_WOB) is a rank-1 change
of wo_b in layers 10-35; the pack carries that term and Exl3OutputParallel adds it.
"""
from __future__ import annotations

import ctypes
import json
import os

import torch

V = ctypes.c_void_p
# Split-K per shape: tools/bench_exl3_dense_decode.py, rows 4, L2-cold (2026-10-08).
SPLITS = {"wq_a": 8, "wkv": 16, "wq_b": 1, "wo_a": 2, "wo_b": 4}
MAX_ROWS = 16


def enabled() -> bool:
    v = os.environ.get("DSV41_EXL3_DENSE", "0")
    if v not in ("0", "1"):
        raise ValueError("DSV41_EXL3_DENSE must be 0 or 1")
    return v == "1"


def pack_path(model_dir: str, rank: int, world: int) -> str:
    env = os.environ.get("DSV41_EXL3_DENSE_PACK")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(model_dir)), "exl3-packs",
                        f"exl3-dense-r{rank}of{world}.safetensors")


def _xc():
    import exl3_moe_cuda as XC
    return XC


class _Members:
    """Static grouping tables for a dense call: uids 0..G-1, each reading its own T rows (t*G + g).
    Built for every row count up front -- a graph capture cannot upload host data."""

    def __init__(self, device):
        self.t = {}
        # G=2 is the draft's shared expert: w1||w3 as two groups over one input (engine/exl3_draft).
        for G in (1, 2, 4):
            uids = torch.arange(G, dtype=torch.int32, device=device)
            count = torch.tensor([G], dtype=torch.int32, device=device)
            for T in range(1, MAX_ROWS + 1):
                m = torch.full((G, 16), -1, dtype=torch.int32)
                for g in range(G):
                    m[g, :T] = torch.arange(T, dtype=torch.int32) * 32 + g
                self.t[(G, T)] = (uids, count, m.to(device))

    def get(self, G, T):
        return self.t[(G, T)]


class Exl3Matrix:
    """One rank's EXL3 matrix, or G same-shape groups that each read their own input rows (wo_a).

    `w` is the trellis storage (all groups, contiguous) so engine/l2pf.py can prefetch it."""

    def __init__(self, trellis: list[torch.Tensor], suh: torch.Tensor, svh: torch.Tensor, bits: list[int],
                 split: int, members: _Members):
        self.G = len(trellis)
        self.K, self.N = int(suh.shape[-1]), int(svh.shape[-1])
        dev = suh.device
        self.w = torch.stack([t.contiguous() for t in trellis]) if self.G > 1 else trellis[0].contiguous()
        self.suh = suh.reshape(self.G, self.K).contiguous()
        self.svh = svh.reshape(self.G, self.N).contiguous()
        stride = self.w[0].numel() * self.w.element_size() if self.G > 1 else 0
        base = self.w.data_ptr()
        self.ptr = torch.tensor([base + g * stride for g in range(self.G)], dtype=torch.int64, device=dev)
        self.k2 = torch.tensor([2 * b for b in bits], dtype=torch.int32, device=dev)
        self.hi = 10 if max(bits) <= 5 else 16
        kt = self.K // 16
        if kt % (split * 4) or self.N % 128 or self.K % 128:
            raise ValueError(f"EXL3 dense shape K={self.K} N={self.N} does not take split {split}")
        self.split = split
        self.members = members

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.suh, self.svh))

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        """x bf16 [T, G*K] (or [T, G, K]) -> [T, G*N] in out_dtype."""
        XC = _xc()
        T = x.shape[0]
        assert 0 < T <= MAX_ROWS, T
        G, K, N, S = self.G, self.K, self.N, self.split
        rows = T * G
        xr = x.reshape(rows, K)
        if xr.stride(1) != 1 or xr.stride(0) != K:
            xr = xr.contiguous()
        dev = x.device
        lib, st = XC._lib(), XC._stream()
        xh = torch.empty((rows, K), dtype=torch.float16, device=dev)
        lib.exl3m_dense_rot(V(xr.data_ptr()), K, V(self.suh.data_ptr()), V(xh.data_ptr()), rows, K, G,
                            1 if xr.dtype == torch.bfloat16 else 0, st)
        uids, count, members = self.members.get(G, T)
        z = torch.empty((S, rows, N), dtype=torch.float32, device=dev)
        lib.exl3m_grouped(V(xh.data_ptr()), V(xh.data_ptr()), V(self.ptr.data_ptr()), V(self.ptr.data_ptr()),
                          V(self.k2.data_ptr()), V(self.k2.data_ptr()), V(uids.data_ptr()), V(count.data_ptr()),
                          V(members.data_ptr()), V(z.data_ptr()), K, N, rows, S, 16, G, G, 1, 8, 4, 1,
                          2, self.hi, XC.CB_MUL1, st)
        out = torch.empty((rows, N), dtype=out_dtype, device=dev)
        lib.exl3m_dense_out(V(z.data_ptr()), V(self.svh.data_ptr()), V(out.data_ptr()), rows, N, S, G,
                            1 if out_dtype == torch.bfloat16 else 0, st)
        return out.view(T, G * N)


class Exl3OutputParallel:
    """engine/tensor_parallel.OutputParallelWeight with an EXL3 local matrix: gather the input
    features, compute this rank's output rows, gather those. Optional rank-1 term (u, v): the
    abliteration overlay's change of wo_b, added in fp32 before the bf16 rounding."""

    def __init__(self, local: Exl3Matrix, world: int, ablit=None):
        self.local, self.world = local, world
        self.ablit = ablit

    @property
    def bytes(self) -> int:
        extra = sum(t.numel() * t.element_size() for t in self.ablit) if self.ablit else 0
        return self.local.bytes + extra

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from engine import comm
        from engine.collective_rails import group as collective_group
        shape = x.shape
        flat = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
        rows, width = flat.shape
        gathered = torch.empty((self.world * rows, width), dtype=flat.dtype, device=flat.device)
        comm.all_gather_fast(gathered, flat, group=collective_group())
        full = gathered.view(self.world, rows, width).transpose(0, 1).reshape(rows, self.world * width)
        if self.ablit is None:
            local = self.local(full)
        else:
            u, v = self.ablit
            y = self.local(full, out_dtype=torch.float32)
            y.addmm_((full.float() @ v)[:, None], u[None, :])
            local = y.to(torch.bfloat16)
        output = torch.empty((self.world * rows, local.shape[-1]), dtype=local.dtype, device=local.device)
        comm.all_gather_fast(output, local, group=collective_group())
        return output.view(self.world, rows, local.shape[-1]).transpose(0, 1).reshape(*shape[:-1], -1)


class Exl3Layer:
    """One backbone layer's EXL3 decode matrices; `shared_slot` indexes the shared-expert arena."""

    def __init__(self, wq_a, wkv, wq_b, wo_a, wo_b, shared_slot):
        self.wq_a, self.wkv, self.wq_b, self.wo_a, self.wo_b = wq_a, wkv, wq_b, wo_a, wo_b
        self.shared_slot = shared_slot


class Exl3Dense:
    """Everything the pack holds, on the GPU: per-layer matrices plus a 40-slot shared-expert arena
    (tools/exl3_moe.Exl3Arena at 5-bit slots), which the routed-expert pipeline drives."""

    def __init__(self, path: str, device, rank: int, world: int, n_layers: int):
        from safetensors import safe_open
        import exl3_moe as X3
        import exl3_ref as R
        self.path = path
        f = safe_open(path, "pt")
        meta = f.metadata() or {}
        if meta.get("format") != "dsv41-exl3-dense-v1":
            raise ValueError(f"{path}: not an EXL3 dense pack")
        if (int(meta["rank"]), int(meta["world"])) != (rank, world):
            raise ValueError(f"{path}: pack is rank {meta['rank']}/{meta['world']}, engine is {rank}/{world}")
        self.meta = meta
        self.source_sha256 = meta["source_sha256"]
        self.ablit_meta = json.loads(meta.get("ablit") or "null")
        bits = json.loads(meta["bits"])
        self.members = _Members(device)
        self.shared_arena = X3.Exl3Arena(n_layers, device, tp_rank=rank, tp_world=world, bits=5.0)
        # gate/up and down widths can differ within a layer here; exl3_moe_cuda reads bits2_gpu for w2
        self.shared_arena.bits2_gpu = self.shared_arena.bits_gpu.clone()
        get = (lambda k: f.get_tensor(k).to(device))
        self.layers = []
        for L in range(n_layers):
            p = f"layers.{L}"

            def mat(key, split, groups=None):
                keys = [key] if groups is None else [f"{key}.{g}" for g in range(groups)]
                return Exl3Matrix([get(k + ".trellis") for k in keys],
                                  torch.stack([get(k + ".suh") for k in keys]),
                                  torch.stack([get(k + ".svh") for k in keys]),
                                  [int(bits[k]) for k in keys], split, self.members)

            ablit = None
            if f"{p}.attn.wo_b.ablit_u" in f.keys():
                ablit = (get(f"{p}.attn.wo_b.ablit_u").float(), get(f"{p}.attn.wo_b.ablit_v").float())
            wo_b = Exl3OutputParallel(mat(f"{p}.attn.wo_b", SPLITS["wo_b"]), world, ablit)
            layer = Exl3Layer(mat(f"{p}.attn.wq_a", SPLITS["wq_a"]), mat(f"{p}.attn.wkv", SPLITS["wkv"]),
                              mat(f"{p}.attn.wq_b", SPLITS["wq_b"]), mat(f"{p}.attn.wo_a", SPLITS["wo_a"], 4),
                              wo_b, L)
            sb = int(bits[f"{p}.shared.w1"])
            rec = {"t1": f.get_tensor(f"{p}.shared.w1.trellis"), "t3": f.get_tensor(f"{p}.shared.w3.trellis"),
                   "t2": f.get_tensor(f"{p}.shared.w2.trellis"),
                   "suh1": f.get_tensor(f"{p}.shared.w1.suh"), "suh3": f.get_tensor(f"{p}.shared.w3.suh"),
                   "suh2": f.get_tensor(f"{p}.shared.w2.suh"),
                   "svh1": f.get_tensor(f"{p}.shared.w1.svh"), "svh3": f.get_tensor(f"{p}.shared.w3.svh"),
                   "svh2": f.get_tensor(f"{p}.shared.w2.svh")}
            if int(bits[f"{p}.shared.w3"]) != sb:
                raise ValueError(f"{p}: shared w1/w3 differ in bits; one launch decodes both at one width")
            self.shared_arena.load_slot(L, rec, sb)
            self.shared_arena.bits2_gpu[L] = 2 * int(bits[f"{p}.shared.w2"])
            self.layers.append(layer)
        # [n_layers, MAX_ROWS] slot ids, so a decode call's pick is a static row view
        self.shared_pick = (torch.arange(n_layers, dtype=torch.int32, device=device)[:, None]
                            .expand(n_layers, MAX_ROWS).contiguous())
        self.ones = torch.ones((MAX_ROWS, 1), dtype=torch.float32, device=device)
        _xc().warm(self.shared_arena)
        torch.cuda.synchronize(device)
        self.bytes = (sum(m.bytes for l in self.layers for m in (l.wq_a, l.wkv, l.wq_b, l.wo_a, l.wo_b))
                      + self.shared_arena.bytes_per_slot * n_layers)
        del R

    def shared_targets(self, L):
        """The shared expert's trellis for L2 prefetch (engine/l2pf.py takes tensors)."""
        a = self.shared_arena
        return [a.t1[L], a.t3[L], a.t2[L]]

    def attach(self, layers):
        for L, w in enumerate(layers):
            w._x3 = self.layers[L]
