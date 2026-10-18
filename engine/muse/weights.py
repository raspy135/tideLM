"""Muse Spark checkpoints: the three published layouts, read lazily from safetensors.

Formats (read from the gated repos' shard headers, 2026-10-15):

* ``bf16``  meta-models/Muse-Spark-Open-816B-A42B. Experts fused per layer:
  ``mlp.experts.gate_up_proj`` [E, 2I, H] (rows 0..I-1 are gate, I..2I-1 up) and
  ``mlp.experts.down_proj`` [E, H, I].
* ``mxfp4`` amd/...-Quark-MXFP4. Per expert ``mlp.experts.{e}.{gate,up,down}_proj.weight`` U8
  [N, K/2] (two E2M1 values a byte, LOW nibble = even K) plus ``.weight_scale`` U8 [N, K/32]
  (E8M0, bias 127). This is the layout tools/fp4_moe_cuda.cu reads (checked against the BF16
  release on layer 1 expert 0 gate: 11.4% relative error low-nibble-first, 141% high-first).
  Layer 0 (and possibly 61) experts are left BF16 per expert, [N, K].
* ``nvfp4`` meta-models/...-NVFP4. Fused U8 [E, 2I, H/2] with FP8 E4M3 scales per 16 values
  and an FP32 ``weight_scale_2`` per expert (and per half for gate_up). Layers 0 and 61 BF16.
  Read here for reference only: the engine's kernels take MXFP4.

Everything else (attention, latent expert projections, norms, router, head) is BF16/FP32 in all
three. Expert tensors come back in one engine form, ``ExpertMX`` (MXFP4 w1/s1/w3/s3/w2/s2,
CPU uint8); BF16 sources are quantized with ``quantize_mxfp4`` on the way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

import threading

GROUP = 32
_OPEN_LOCK = threading.Lock()


_READ_CHUNK = int(os.environ.get("GLM_READ_CHUNK_KB", "256")) * 1024
# Bulk (background, prefetch, placement) reads: whole tensors unless GLM_READ_BULK_CHUNK_KB --
# many small pieces are many Python tasks contending for the GIL with an eager prefill.
_BULK_CHUNK = int(os.environ.get("GLM_READ_BULK_CHUNK_KB", "0")) * 1024
_CHUNK_POOLS = {}


def _chunk_pool(urgent=False):
    """Piece readers; urgent reads (a decode step waiting on them) get their own, so they do not
    queue behind background reads."""
    pool = _CHUNK_POOLS.get(urgent)
    if pool is None:
        from concurrent.futures import ThreadPoolExecutor
        with _OPEN_LOCK:
            pool = _CHUNK_POOLS.get(urgent)
            if pool is None:
                n = 16 if urgent else int(os.environ.get("GLM_READ_CHUNK_THREADS", "32"))
                pool = _CHUNK_POOLS[urgent] = ThreadPoolExecutor(max_workers=n)
    return pool


def _pread_full(fd, view, off):
    """preadv until the view is full or the file ends; returns the bytes read."""
    done = 0
    while done < len(view):
        got = os.preadv(fd, [view[done:]], off + done)
        if got <= 0:
            break
        done += got
    return done
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
PREFIX = "model.language_model."


@dataclass
class ExpertMX:
    """One expert in MXFP4: w1 gate [I, H/2], w3 up [I, H/2], w2 down [H, I/2] + E8M0 scales."""
    w1: torch.Tensor
    s1: torch.Tensor
    w3: torch.Tensor
    s3: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """U8 [N, K/2] + E8M0 U8 [N, K/32] -> fp32 [N, K]. Low nibble is the even K element."""
    packed = packed.view(torch.uint8)
    lut = E2M1.to(packed.device)
    lo, hi = lut[(packed & 15).long()], lut[(packed >> 4).long()]
    n = packed.shape[0]
    values = torch.stack([lo, hi], -1).reshape(n, -1)
    exp = scale.view(torch.uint8).float() - 127.0
    return values * torch.pow(2.0, exp).repeat_interleave(GROUP, 1)


def quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32/bf16 [N, K] -> (U8 [N, K/2], E8M0 U8 [N, K/32]), OCP MX rules.

    Shared exponent floor(log2(amax)) - 2 (E2M1's largest exponent), elements rounded to the
    nearest E2M1 value (ties to the even code), clamped to +-6. Used for synthetic checkpoints
    and for experts a release left BF16; real MXFP4 experts are taken as published.
    """
    w = weight.float()
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    g = w.view(n, k // GROUP, GROUP)
    amax = g.abs().amax(-1, keepdim=True)
    exp = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(2.0 ** -126))) - 2.0,
                      torch.full_like(amax, -127.0))
    exp = exp.clamp(-127.0, 127.0)
    x = (g / torch.pow(2.0, exp)).clamp(-6.0, 6.0)
    grid = E2M1[:8].to(w.device)                      # magnitudes 0..6, codes 0..7
    mag = x.abs().unsqueeze(-1)
    dist = (mag - grid).abs()
    # Ties: argmin picks the lower code; prefer the even code (round-half-to-even on the mantissa).
    best = dist.argmin(-1)
    tie = (dist.gather(-1, (best + 1).clamp(max=7).unsqueeze(-1)).squeeze(-1)
           == dist.gather(-1, best.unsqueeze(-1)).squeeze(-1)) & (best < 7)
    best = torch.where(tie & (best % 2 == 1), best + 1, best)
    code = best | ((x < 0) & (best > 0)).long() << 3
    code = code.view(n, k).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    scale = (exp.view(n, k // GROUP) + 127.0).to(torch.uint8).contiguous()
    return packed, scale


def dequant_nvfp4(packed: torch.Tensor, scale_fp8: torch.Tensor, scale2: float) -> torch.Tensor:
    """NVFP4 U8 [N, K/2] + E4M3 [N, K/16] + global fp32 -> fp32 [N, K] (reference only)."""
    lut = E2M1.to(packed.device)
    p = packed.view(torch.uint8)
    lo, hi = lut[(p & 15).long()], lut[(p >> 4).long()]
    values = torch.stack([lo, hi], -1).reshape(p.shape[0], -1)
    return values * scale_fp8.float().repeat_interleave(16, 1) * float(scale2)


class Checkpoint:
    """Lazy tensor access over a sharded safetensors checkpoint (index or single file).
    ``detect=False`` skips the Muse expert-layout check (the drafter checkpoint)."""

    def __init__(self, model_dir: str | Path, detect: bool = True):
        from safetensors import safe_open
        self.dir = Path(model_dir)
        self._open = safe_open
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            files = sorted(self.dir.glob("*.safetensors"))
            self.map = {}
            for f in files:
                with safe_open(str(f), "pt") as h:
                    for k in h.keys():
                        self.map[k] = f.name
        self._handles: dict[str, object] = {}
        self.format = self._detect() if detect else None

    def _handle(self, name: str):
        f = self.map[name]
        h = self._handles.get(f)
        if h is None:
            h = self._handles[f] = self._open(str(self.dir / f), "pt")
        return h

    def drop_cache(self) -> None:
        """Ask the kernel to drop this checkpoint's pages from the page cache. On GB10 the page
        cache and GPU allocations share one pool and reclaim is lazy (docs/gotchas.md: the box
        can hard-reset near zero MemAvailable); weights read once must not linger there."""
        import os
        for f in set(self._handles) | set(getattr(self, "_headers", {})):
            try:
                fd = os.open(str(self.dir / f), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass

    def has(self, name: str) -> bool:
        return name in self.map

    _DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
               "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64, "I32": torch.int32,
               "F8_E4M3": torch.float8_e4m3fn}

    def _meta(self, name: str):
        """(file path, absolute byte offset, dtype, shape) from the shard's header."""
        f = self.map[name]
        hdr = self._headers.get(f) if hasattr(self, "_headers") else None
        if hdr is None:
            with _OPEN_LOCK:
                return self._meta_locked(name, f)
        base, meta = hdr
        m = meta[name]
        return self.dir / f, base + m["data_offsets"][0], self._DTYPES[m["dtype"]], m["shape"]

    def _meta_locked(self, name: str, f: str):
        hdr = self._headers.get(f) if hasattr(self, "_headers") else None
        if hdr is None:
            import struct
            if not hasattr(self, "_headers"):
                self._headers = {}
            with open(self.dir / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = (8 + n, json.loads(fh.read(n)))
            self._headers[f] = hdr
        base, meta = hdr
        m = meta[name]
        return self.dir / f, base + m["data_offsets"][0], self._DTYPES[m["dtype"]], m["shape"]

    def _read(self, name: str, start: int | None = None, end: int | None = None,
              urgent: bool = False) -> torch.Tensor:
        """Rows [start, end) of a tensor read with preadv into a fresh tensor. Not through
        safetensors' slices: on this box those kept every expert's bytes in host memory after the
        copy to the GPU (+15 MB a slot, +53 MB a BF16 expert), which over a full arena is the
        out-of-memory kill seen on the first real load."""
        import os
        path, off, dtype, shape = self._meta(name)
        shape = list(shape)
        row = 1
        for d in shape[1:]:
            row *= d
        itemsize = torch.empty((), dtype=dtype).element_size()
        if start is not None:
            off += start * row * itemsize
            shape[0] = end - start
        out = torch.empty(shape, dtype=dtype)
        buf = memoryview(out.view(torch.uint8).numpy()).cast("B") if out.numel() else memoryview(b"")
        fd = self._fds.get(path) if hasattr(self, "_fds") else None
        if fd is None:
            with _OPEN_LOCK:               # reads run from a thread pool (positional preadv is safe)
                if not hasattr(self, "_fds"):
                    self._fds = {}
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY)
        chunk = _READ_CHUNK if urgent else _BULK_CHUNK
        if chunk and len(buf) > 2 * chunk:
            # One preadv streams ~1 GB/s from a cold file; 256 KB pieces in parallel keep the NVMe
            # queue deep: a cold rank-half GLM expert (9.6 MB, six tensors) 10.5 -> 3.2 ms.
            # Its own pool: the callers already run on a thread pool, and waiting on pieces
            # queued behind them in the same pool could deadlock.
            pieces = [(c, min(chunk, len(buf) - c)) for c in range(0, len(buf), chunk)]
            for got, (c, m) in zip(_chunk_pool(urgent).map(lambda p: _pread_full(fd, buf[p[0]:p[0] + p[1]], off + p[0]),
                                                      pieces), pieces):
                if got != m:
                    raise IOError(f"short read of {name}")
            return out
        if _pread_full(fd, buf, off) != len(buf):
            raise IOError(f"short read of {name}")
        return out

    def get(self, name: str) -> torch.Tensor:
        return self._read(name)

    def get_rows_many(self, specs, urgent: bool = False) -> list[torch.Tensor]:
        """Several row ranges [(name, start, end)] read at once: every tensor's 256 KB pieces go to
        the piece pool together, instead of one tensor after another (a cold rank-half GLM expert,
        six tensors: 8.2 ms through get_rows in turn)."""
        outs, jobs = [], []
        for name, start, end in specs:
            path, off, dtype, shape = self._meta(name)
            shape = list(shape)
            row = 1
            for d in shape[1:]:
                row *= d
            itemsize = torch.empty((), dtype=dtype).element_size()
            off += start * row * itemsize
            shape[0] = end - start
            out = torch.empty(shape, dtype=dtype)
            outs.append(out)
            if not out.numel():
                continue
            buf = memoryview(out.view(torch.uint8).numpy()).cast("B")
            fd = self._fd(path)
            chunk = (_READ_CHUNK if urgent else _BULK_CHUNK) or len(buf)
            jobs += [(name, fd, buf[c:c + min(chunk, len(buf) - c)], off + c) for c in range(0, len(buf), chunk)]
        for (name, _, view, _), got in zip(jobs, _chunk_pool(urgent).map(lambda j: _pread_full(j[1], j[2], j[3]), jobs)):
            if got != len(view):
                raise IOError(f"short read of {name}")
        return outs

    def _fd(self, path):
        import os
        fd = self._fds.get(path) if hasattr(self, "_fds") else None
        if fd is None:
            with _OPEN_LOCK:
                if not hasattr(self, "_fds"):
                    self._fds = {}
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY)
        return fd

    def get_rows(self, name: str, start: int, end: int, urgent: bool = False) -> torch.Tensor:
        """Rows [start, end) of a 2D+ tensor without reading the rest."""
        return self._read(name, start, end, urgent)

    def _detect(self) -> str:
        if any(k.endswith("experts.gate_up_proj_weight_scale") for k in self.map):
            return "nvfp4"
        if any(".experts.0.gate_proj.weight_scale" in k for k in self.map):
            return "mxfp4"
        if any(k.endswith("experts.gate_up_proj") for k in self.map):
            return "bf16"
        raise ValueError(f"{self.dir}: unrecognized Muse expert layout")

    # ------------------------------------------------------------------ dense
    def dense(self, layer: int, name: str) -> torch.Tensor:
        return self.get(f"{PREFIX}layers.{layer}.{name}")

    def top(self, name: str) -> torch.Tensor:
        return self.get(name if name == "lm_head.weight" else PREFIX + name)

    # ------------------------------------------------------------------ experts
    def expert_layer_format(self, layer: int) -> str:
        base = f"{PREFIX}layers.{layer}.mlp.experts."
        if self.format == "mxfp4":
            return "mxfp4" if self.has(base + "0.gate_proj.weight_scale") else "bf16-split"
        if self.format == "nvfp4":
            return "nvfp4" if self.has(base + "gate_up_proj_weight_scale") else "bf16"
        return "bf16"

    def expert_float(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate [I,H], up [I,H], down [H,I]) fp32, from whatever the checkpoint stores."""
        base = f"{PREFIX}layers.{layer}.mlp.experts."
        fmt = self.expert_layer_format(layer)
        if fmt == "mxfp4":
            mx = self.expert_mx(layer, e)
            return (dequant_mxfp4(mx.w1, mx.s1), dequant_mxfp4(mx.w3, mx.s3),
                    dequant_mxfp4(mx.w2, mx.s2))
        if fmt == "bf16-split":
            return tuple(self.get(f"{base}{e}.{p}_proj.weight").float()
                         for p in ("gate", "up", "down"))
        if fmt == "nvfp4":
            gu = self.get_rows(base + "gate_up_proj", e, e + 1)[0]
            gus = self.get_rows(base + "gate_up_proj_weight_scale", e, e + 1)[0]
            g2 = self.get_rows(base + "gate_up_proj_weight_scale_2", e, e + 1)[0]
            d = self.get_rows(base + "down_proj", e, e + 1)[0]
            ds = self.get_rows(base + "down_proj_weight_scale", e, e + 1)[0]
            d2 = self.get_rows(base + "down_proj_weight_scale_2", e, e + 1)[0]
            inter = gu.shape[0] // 2
            gate = dequant_nvfp4(gu[:inter], gus[:inter], float(g2[0]))
            up = dequant_nvfp4(gu[inter:], gus[inter:], float(g2[1]))
            return gate, up, dequant_nvfp4(d, ds, float(d2))
        gu = self.get_rows(base + "gate_up_proj", e, e + 1)[0].float()
        down = self.get_rows(base + "down_proj", e, e + 1)[0].float()
        inter = gu.shape[0] // 2
        return gu[:inter], gu[inter:], down

    def expert_mx(self, layer: int, e: int) -> ExpertMX:
        """One expert in the engine's MXFP4 form (quantized here when the release is not MXFP4)."""
        base = f"{PREFIX}layers.{layer}.mlp.experts."
        if self.expert_layer_format(layer) == "mxfp4":
            t = {p: (self.get(f"{base}{e}.{p}_proj.weight"),
                     self.get(f"{base}{e}.{p}_proj.weight_scale")) for p in ("gate", "up", "down")}
            return ExpertMX(t["gate"][0], t["gate"][1], t["up"][0], t["up"][1],
                            t["down"][0], t["down"][1])
        gate, up, down = self.expert_float(layer, e)
        (w1, s1), (w3, s3), (w2, s2) = (quantize_mxfp4(x) for x in (gate, up, down))
        return ExpertMX(w1, s1, w3, s3, w2, s2)


def write_synthetic(model_dir: str | Path, raw_config: dict, *, seed: int = 0,
                    expert_format: str = "mxfp4", std: float = 0.02,
                    bf16_expert_layers: tuple[int, ...] = (), device: str = "cpu",
                    bias_std: float = 0.05) -> None:
    """A random checkpoint in a release's layout, for tests (expert_format mxfp4 or bf16).

    Weights are N(0, std) like the model's initializer; norms and gates get small random values
    rather than the initializer's zeros so every term of the forward pass is exercised. A large
    ``bias_std`` makes routing decisive: random weights otherwise leave near-ties that flip under
    BF16 rounding and turn a numerical comparison into a comparison of chaos.
    """
    from safetensors.torch import save_file
    from .config import MuseConfig
    cfg = MuseConfig.from_dict(raw_config)
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, s=std):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    t: dict[str, torch.Tensor] = {}
    H, I, M = cfg.hidden, cfg.moe_inter, cfg.moe_hidden
    t[PREFIX + "embed_tokens.weight"] = rnd(cfg.vocab, H, s=1.0)
    t[PREFIX + "norm.weight"] = (1.0 + torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
    t["lm_head.weight"] = rnd(cfg.vocab, H)
    for L in range(cfg.layers):
        p = f"{PREFIX}layers.{L}."
        a = p + "self_attn."
        t[a + "q_proj.weight"] = rnd(cfg.heads * cfg.head_dim, H)
        t[a + "k_proj.weight"] = rnd(cfg.kv_heads * cfg.head_dim, H)
        t[a + "v_proj.weight"] = rnd(cfg.kv_heads * cfg.head_dim, H)
        t[a + "o_proj.weight"] = rnd(H, cfg.heads * cfg.head_dim)
        t[a + "gate_proj.weight"] = rnd(cfg.heads * cfg.head_dim, H)
        for n in ("input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            t[p + n + ".weight"] = rnd(H, s=0.1)
        t[p + "post_attention_residual_gate.gate"] = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
        t[p + "post_feedforward_residual_gate.gate"] = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
        m = p + "mlp."
        t[m + "gate.weight"] = (torch.randn(cfg.experts, H, generator=g) * std).float()
        t[m + "gate.e_score_correction_bias"] = (torch.randn(cfg.experts, generator=g) * bias_std).float()
        t[m + "pre_expert_proj.weight"] = rnd(M, H)
        t[m + "post_expert_proj.weight"] = rnd(H, M)
        t[m + "pre_expert_norm.weight"] = rnd(M, s=0.1)
        t[m + "experts.post_expert_norm.weight"] = rnd(M, s=0.1)
        if expert_format == "bf16":
            t[m + "experts.gate_up_proj"] = rnd(cfg.experts, 2 * I, M)
            t[m + "experts.down_proj"] = rnd(cfg.experts, M, I)
        else:
            for e in range(cfg.experts):
                for name, shape in (("gate", (I, M)), ("up", (I, M)), ("down", (M, I))):
                    w = rnd(*shape)
                    if L in bf16_expert_layers:
                        t[f"{m}experts.{e}.{name}_proj.weight"] = w
                    else:
                        packed, scale = quantize_mxfp4(w.to(device))
                        t[f"{m}experts.{e}.{name}_proj.weight"] = packed.cpu()
                        t[f"{m}experts.{e}.{name}_proj.weight_scale"] = scale.cpu()
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(t, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(raw_config, indent=1))


def write_synthetic_dspark(out_dir: str | Path, target_raw: dict, *, seed: int = 1,
                           layers: int = 2, taps=(0, 1, 2, 3), std: float = 0.02,
                           device: str = "cpu") -> None:
    """A random DSpark drafter in the published layout, for a synthetic target (tests only)."""
    from safetensors.torch import save_file
    from .config import MuseConfig
    cfg = MuseConfig.from_dict(target_raw)
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, s=std):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    H, D = cfg.hidden, cfg.head_dim
    inter = 2 * H
    t = {"fc.weight": rnd(H, H * len(taps)), "hidden_norm.weight": rnd(H, s=0.01),
         "norm.weight": rnd(H, s=0.1), "lm_head.weight": rnd(cfg.vocab, H),
         "markov_head.markov_w1.weight": rnd(cfg.vocab, 512, s=0.01),
         "markov_head.markov_w2.weight": rnd(cfg.vocab, 512, s=0.01),
         "confidence_head.proj.weight": rnd(1, H + 512), "confidence_head.proj.bias": rnd(1),
         "d2t": torch.zeros(cfg.vocab, dtype=torch.long)}
    for i in range(layers):
        p = f"layers.{i}."
        t.update({p + "self_attn.pre_qkv_proj_norm.weight": rnd(H, s=0.02),
                  p + "self_attn.q_proj.weight": rnd(cfg.heads * D, H),
                  p + "self_attn.k_proj.weight": rnd(cfg.kv_heads * D, H),
                  p + "self_attn.v_proj.weight": rnd(cfg.kv_heads * D, H),
                  p + "self_attn.o_gate_proj.weight": rnd(cfg.heads * D, H),
                  p + "self_attn.o_proj.weight": rnd(H, cfg.heads * D),
                  p + "add_attn.gate": rnd(H, s=0.03), p + "add_ffn.gate": rnd(H, s=0.03),
                  p + "pre_ffn_norm.weight": rnd(H, s=0.02), p + "post_ffn_norm.weight": rnd(H, s=0.01),
                  p + "feed_forward.gate_proj.weight": rnd(inter, H),
                  p + "feed_forward.up_proj.weight": rnd(inter, H),
                  p + "feed_forward.down_proj.weight": rnd(H, inter)})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(t, str(out / "model.safetensors"))
    conf = {"model_type": "muse_spark_dspark", "hidden_size": H, "num_hidden_layers": layers,
            "num_attention_heads": cfg.heads, "num_key_value_heads": cfg.kv_heads, "head_dim": D,
            "intermediate_size": inter, "rms_norm_eps": 1e-5, "rms_post_norm_eps": 1e-8,
            "rope_theta": cfg.rope_theta, "sliding_window": 2048, "block_size": 7,
            "mask_token_id": cfg.pad, "target_layer_ids": list(taps), "markov_rank": 512,
            "output_soft_cap_temp": 20.0, "metap": {"use_metap": True, "base_width": 256.0},
            "vocab_size": cfg.vocab}
    (out / "config.json").write_text(json.dumps(conf, indent=1))
