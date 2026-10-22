"""Decode-sized spellings of rmsnorm and the hyper-connection coefficients with fewer launches.

Both are meant to be bit-identical to the torch spellings they replace (v41_ref.rmsnorm and
Model._hc_mixes at M <= 16). What sets their bits are two row-count-dependent torch ops, and those
stay torch ops on the same shapes:

  * the RMS reduction `t.square().mean(-1)` over a [MM_TILE, D] fp32 tensor (v41_ref.tiled_rows),
  * the HC projection, one cuBLAS fp32 GEMM over [HC_MM_TILE, 20480] rows (v41_ref.mm).

What goes is everything around them. The torch spelling re-creates the zero padding on every call
(`new_zeros` + `cat`), slices the result back out (`cat` + copy), converts the weight to fp32, and
runs `+ eps`, `rsqrt` and two or three multiplies as separate kernels: 12 launches per rmsnorm and
15 per HC site (18 with the three copies into the decoder's static buffers). Here the padding lives
in a static buffer whose tail rows are never written, and the elementwise tail is one Triton kernel:
4 launches per rmsnorm, 6 per HC site.

The Triton tail reproduces the torch elementwise ops one for one: an fp32 add of eps, `rsqrt`, and
IEEE multiplies in the same order, compiled without FMA contraction. tools/test_decode_lean.py
checks torch.equal against the torch spellings on real shapes, in and out of CUDA graphs.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:  # the HC projection's kernel gate/dispatch (tools/fp32_skinny.py via v41_ref)
    import v41_ref as _R
except Exception:  # noqa: BLE001
    _R = None


@triton.jit
def _rms_apply_kernel(X, MEAN, W, OUT, D, EPS, BD: tl.constexpr, OUT_BF16: tl.constexpr):
    """out[t, :] = (w * (x[t, :] * rsqrt(mean[t] + eps))) as v41_ref.rmsnorm computes it."""
    t = tl.program_id(0)
    db = tl.program_id(1) * BD + tl.arange(0, BD)
    m = db < D
    rs = tl.math.rsqrt(tl.load(MEAN + t) + EPS)
    x = tl.load(X + t * D + db, mask=m, other=0.0)
    y = tl.load(W + db, mask=m, other=0.0) * (x * rs)
    if OUT_BF16:
        tl.store(OUT + t * D + db, y.to(tl.bfloat16), mask=m)
    else:
        tl.store(OUT + t * D + db, y, mask=m)


@triton.jit
def _rms_fused_kernel(X, W, OUT, D, stride_x, EPS, BD: tl.constexpr, OUT_BF16: tl.constexpr):
    """rmsnorm in one launch: mean(x^2) over the row in fp32, then _rms_apply_kernel's arithmetic.
    Only the summation order of the mean differs from the pad/square/mean/apply spelling."""
    t = tl.program_id(0)
    acc = tl.zeros((BD,), dtype=tl.float32)
    for d0 in range(0, D, BD):
        db = d0 + tl.arange(0, BD)
        x = tl.load(X + t * stride_x + db, mask=db < D, other=0.0).to(tl.float32)
        acc += x * x
    rs = tl.math.rsqrt(tl.sum(acc, 0) / D + EPS)
    for d0 in range(0, D, BD):
        db = d0 + tl.arange(0, BD)
        m = db < D
        x = tl.load(X + t * stride_x + db, mask=m, other=0.0).to(tl.float32)
        y = tl.load(W + db, mask=m, other=0.0) * (x * rs)
        if OUT_BF16:
            tl.store(OUT + t * D + db, y.to(tl.bfloat16), mask=m)
        else:
            tl.store(OUT + t * D + db, y, mask=m)


@triton.jit
def _hc_rms_kernel(MM, MEAN, S, B, PRE, POST, COMB, EPS_RMS,
                   iters: tl.constexpr, eps: tl.constexpr, HC: tl.constexpr):
    """engine/hc_sinkhorn._hc_kernel with its input formed in-kernel: mixes = mm * rsqrt(mean + eps).
    The body below the first three lines is that kernel's, unchanged."""
    row = tl.program_id(0)
    MIX: tl.constexpr = (2 + HC) * HC
    rs = tl.math.rsqrt(tl.load(MEAN + row) + EPS_RMS)
    j = tl.arange(0, HC)
    s0 = tl.load(S + 0)
    s1 = tl.load(S + 1)
    s2 = tl.load(S + 2)
    m_pre = tl.load(MM + row * MIX + j) * rs
    b_pre = tl.load(B + j)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + eps
    tl.store(PRE + row * HC + j, pre)
    m_post = tl.load(MM + row * MIX + HC + j) * rs
    b_post = tl.load(B + HC + j)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
    tl.store(POST + row * HC + j, post)
    r = tl.arange(0, HC)[:, None]
    c = tl.arange(0, HC)[None, :]
    off = 2 * HC + r * HC + c
    comb = (tl.load(MM + row * MIX + off) * rs) * s2 + tl.load(B + off)
    mx = tl.max(comb, axis=1)[:, None]
    e = tl.exp(comb - mx)
    comb = e / tl.sum(e, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in range(iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    tl.store(COMB + row * HC * HC + r * HC + c, comb)


@triton.jit
def _swiglu_kernel(GU, H, N, LIMIT, BN: tl.constexpr, CLAMP: tl.constexpr):
    """h[t, :] = bf16(silu(min(gate, L)) * clamp(up, -L, L)), gate = gu[t, :N], up = gu[t, N:2N], as
    v41_ref.expert_ffn computes it: fp32 from bf16, torch's silu x / (1 + exp(-x)) with libdevice's
    exp and an IEEE division (Triton's `/` and tl.exp are approximate), then one fp32 multiply."""
    t = tl.program_id(0)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    m = j < N
    g = tl.load(GU + t * 2 * N + j, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + t * 2 * N + N + j, mask=m, other=0.0).to(tl.float32)
    if CLAMP:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    s = libdevice.div_rn(g, 1.0 + libdevice.exp(-g))
    tl.store(H + t * N + j, (s * u).to(tl.bfloat16), mask=m)


@triton.jit
def _rope_kernel(X, F, OUT, H, D: tl.constexpr, RD: tl.constexpr, BD: tl.constexpr,
                 CONJ: tl.constexpr, RE: tl.constexpr, IM: tl.constexpr,
                 IN_F32: tl.constexpr = False, OUT_F32: tl.constexpr = False):
    """One row of cat([x[:D-RD], rotate(x[D-RD:])]) as v41_ref.apply_rotary makes it: fp32 pairs
    times the complex factor, rounded to bf16. RE / IM pick the multiply's FMA contraction
    (0: none, 1: fma on the first product, 2: fma on the second) -- whichever reproduces torch's
    c10::complex<float> operator* as nvcc compiled it (tools/test_decode_lean.py finds it)."""
    row = tl.program_id(0)
    t = row // H
    d = tl.arange(0, BD)
    keep = d < D - RD
    # IN_F32: x is an fp32 tensor that the torch spelling rounds to bf16 before rotating.
    # OUT_F32: store the bf16 result widened to fp32 (what `.float()` of it would give).
    x = tl.load(X + row * D + d, mask=keep, other=0.0).to(tl.bfloat16)
    if OUT_F32:
        tl.store(OUT + row * D + d, x.to(tl.float32), mask=keep)
    else:
        tl.store(OUT + row * D + d, x, mask=keep)
    i = tl.arange(0, RD // 2)
    a = tl.load(X + row * D + (D - RD) + 2 * i).to(tl.bfloat16).to(tl.float32)
    b = tl.load(X + row * D + (D - RD) + 2 * i + 1).to(tl.bfloat16).to(tl.float32)
    c = tl.load(F + t * RD + 2 * i)
    dd = tl.load(F + t * RD + 2 * i + 1)
    if CONJ:
        dd = -dd
    if RE == 0:
        re = a * c - b * dd
    elif RE == 1:
        re = tl.fma(a, c, -(b * dd))
    else:
        re = tl.fma(-b, dd, a * c)
    if IM == 0:
        im = a * dd + b * c
    elif IM == 1:
        im = tl.fma(a, dd, b * c)
    else:
        im = tl.fma(b, c, a * dd)
    re = re.to(tl.bfloat16)
    im = im.to(tl.bfloat16)
    if OUT_F32:
        re = re.to(tl.float32)
        im = im.to(tl.float32)
    tl.store(OUT + row * D + (D - RD) + 2 * i, re)
    tl.store(OUT + row * D + (D - RD) + 2 * i + 1, im)


@triton.jit
def _router_pre_kernel(G, BIAS, KEEP, SCORES, LOGITS, MASKED, E, HAS_KEEP: tl.constexpr, BE: tl.constexpr):
    """scores = sqrt(softplus(g)), logits = scores + bias, masked = where(keep, logits, -inf), as
    Model's router computes them: torch's softplus is `x > 20 ? x : log1p(exp(x))` (beta 1) with
    libdevice's log1p/exp, and its sqrt is the IEEE one."""
    t = tl.program_id(0)
    e = tl.arange(0, BE)
    m = e < E
    g = tl.load(G + t * E + e, mask=m, other=0.0)
    sp = tl.where(g > 20.0, g, libdevice.log1p(libdevice.exp(g)))
    sc = libdevice.sqrt_rn(sp)
    lg = sc + tl.load(BIAS + e, mask=m, other=0.0)
    tl.store(SCORES + t * E + e, sc, mask=m)
    tl.store(LOGITS + t * E + e, lg, mask=m)
    if HAS_KEEP:
        keep = tl.load(KEEP + e, mask=m, other=0) != 0
        lg = tl.where(keep, lg, float("-inf"))
    tl.store(MASKED + t * E + e, lg, mask=m)


@triton.jit
def _router_post_kernel(WTS, SUM, OUT, SCALE, K: tl.constexpr, KB: tl.constexpr):
    """route_w = wts / (sum + 1e-20) * route_scale, one IEEE division and one multiply."""
    t = tl.program_id(0)
    k = tl.arange(0, KB)
    m = k < K
    w = tl.load(WTS + t * K + k, mask=m, other=0.0)
    s = tl.load(SUM + t) + 1e-20
    tl.store(OUT + t * K + k, libdevice.div_rn(w, s) * SCALE, mask=m)


@triton.jit
def _router_gather_post_kernel(SCORES, IDX, OUT, E, SCALE):
    """Exact six-way torch CUDA reduction: ((w0+w4)+w2)+((w1+w5)+w3).

    Keep cuBLAS and torch.topk unchanged (including their near-ties). This only
    removes the gathered tensor and two launches; random/adversarial and graph
    tests must be rerun if PyTorch changes its small-row reduction geometry.
    """
    t = tl.program_id(0)
    w0 = tl.load(SCORES + t * E + tl.load(IDX + t * 6))
    w1 = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + 1))
    w2 = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + 2))
    w3 = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + 3))
    w4 = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + 4))
    w5 = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + 5))
    denom = ((w0 + w4) + w2) + ((w1 + w5) + w3) + 1e-20
    for j in tl.static_range(6):
        w = tl.load(SCORES + t * E + tl.load(IDX + t * 6 + j))
        tl.store(OUT + t * 6 + j, libdevice.div_rn(w, denom) * SCALE)


@triton.jit
def _softmax_pre_kernel(S, MASK, P, MX, H, N, s_t, s_h, s_n, SCALE, BN: tl.constexpr):
    """x = where(mask, s * scale, -inf); mx = max(amax(x), -1e30); p = exp(x - mx) -- the torch
    decode attention's elementwise ops; amax is exact in any order, exp is libdevice's."""
    row = tl.program_id(0)
    t = row // H
    h = row % H
    n = tl.arange(0, BN)
    m = n < N
    x = tl.load(S + t * s_t + h * s_h + n * s_n, mask=m, other=0.0) * SCALE
    keep = tl.load(MASK + t * N + n, mask=m, other=0) != 0
    x = tl.where(keep, x, float("-inf"))
    mx = tl.maximum(tl.max(tl.where(m, x, float("-inf")), axis=0), -1e30)
    tl.store(P + row * N + n, libdevice.exp(x - mx), mask=m)
    tl.store(MX + row, mx)


@triton.jit
def _softmax_post_kernel(P, PSUM, MX, SINK, OUT, H, N, BN: tl.constexpr):
    """out = p / (psum + exp(sink - mx)), one IEEE division per element."""
    row = tl.program_id(0)
    h = row % H
    n = tl.arange(0, BN)
    m = n < N
    denom = tl.load(PSUM + row) + libdevice.exp(tl.load(SINK + h) - tl.load(MX + row))
    p = tl.load(P + row * N + n, mask=m, other=0.0)
    tl.store(OUT + row * N + n, libdevice.div_rn(p, denom), mask=m)


@triton.jit
def _route_prep_kernel(IDX, LUT, SLOTS, ROUTE, P, NULL, PB: tl.constexpr):
    """slots = lut[idx] (int32), and the routing keys moe_forward's split_decode_null path makes:
    route = slots, except a null-slot pair gets the unique key null + 1 + pair."""
    p = tl.arange(0, PB)
    m = p < P
    s = tl.load(LUT + tl.load(IDX + p, mask=m, other=0), mask=m, other=0)
    tl.store(SLOTS + p, s, mask=m)
    tl.store(ROUTE + p, tl.where(s == NULL, NULL + 1 + p, s), mask=m)


@triton.jit
def _route_prep_masked_kernel(IDX, LUT, ACTIVE, SLOTS, ROUTE, P, K, NULL, PB: tl.constexpr):
    """_route_prep_kernel for a batched step whose idle lanes' rows hold stale routes. A pair of an
    inactive token row (ACTIVE[p // K] == 0) gets routing key -1: the router packs those into
    16-pair blocks of slot -1 that both expert kernels skip outright, so an idle lane reads no
    expert and costs no work. Its partial-output rows are left unwritten -- they belong to idle
    rows only, which nothing reads. Giving idle pairs unique null keys instead (the null-slot
    convention for live pairs) made the down kernel write zeros for every one of them across every
    output row: 3.02 vs 2.59 ms a layer (native 6 rows: 2.63) with one of four lanes active, on a
    single-GPU 96-slot arena, 2026-10-14. tools/test_decode_rows.py checks the skip is exact."""
    p = tl.arange(0, PB)
    m = p < P
    live = tl.load(ACTIVE + p // K, mask=m, other=0) != 0
    s = tl.load(LUT + tl.load(IDX + p, mask=m, other=0), mask=m, other=0)
    s = tl.where(live, s, NULL)
    tl.store(SLOTS + p, s, mask=m)
    tl.store(ROUTE + p, tl.where(live, tl.where(s == NULL, NULL + 1 + p, s), -1), mask=m)


@triton.jit
def _block_null_kernel(BS, NB, NULL, NBB: tl.constexpr):
    """block_slot = where(block_slot > null, null, block_slot), in place."""
    b = tl.arange(0, NBB)
    m = b < NB
    v = tl.load(BS + b, mask=m, other=0)
    tl.store(BS + b, tl.where(v > NULL, NULL, v), mask=m)


@triton.jit
def _ring_gather_f32_kernel(RING, SLOT, OUT, N1, NTOT, D: tl.constexpr):
    """out[t, n, :] = ring[slot[t, n], :].float() for n < N1, in a [T, NTOT, D] fp32 buffer."""
    r = tl.program_id(0)
    t = r // N1
    n = r % N1
    d = tl.arange(0, D)
    src = tl.load(SLOT + r)
    tl.store(OUT + (t * NTOT + n) * D + d, tl.load(RING + src * D + d).to(tl.float32))


@triton.jit
def _packed_gather_f32_kernel(Cache, Ids, Out, N, N2, N1, NTOT, D: tl.constexpr,
                              STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    """engine/packed_kv._gather with fp32 output rows placed at out[t, N1 + j, :] of a
    [T, NTOT, D] buffer (row = t * N2 + j). The value is built exactly as there -- magnitude times
    the e4m3 scale, rounded to bf16, sign set on the bf16 bits -- and then widened."""
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = group // (D // 16), group % (D // 16)
    idx = tl.load(Ids + row, row < N, 0)
    word = tl.load(Cache + idx * STRIDE + col, row < N, 0).to(tl.uint64)
    j = tl.arange(0, 16)
    code = ((word[:, None] >> (j[None, :] * 4)) & 15).to(tl.int32)
    mag = code & 7
    value = tl.where(mag < 2, mag * .5,
                    tl.where(mag < 4, mag * .5,
                             tl.where(mag < 6, mag - 2., 2. * mag - 8.)))
    sw = tl.load(Cache + idx * STRIDE + D // 16 + col // 8, row < N, 0).to(tl.uint64)
    sb = ((sw >> ((col % 8) * 8)) & 255).to(tl.uint8)
    scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    bits = (value * scale[:, None]).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits = bits | ((code >= 8).to(tl.uint16) << 15)
    v = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    t = row // N2
    dst = (t * NTOT + N1 + row % N2) * D + col * 16
    tl.store(Out + dst[:, None] + j[None, :], v, row[:, None] < N)


# The FMA contraction of torch's complex multiply (c10::complex<float> operator* as nvcc compiled
# it in this torch build): re = fma(a, c, -(b*d)), im = fma(b, c, a*d). Measured on 4.2M fp32
# products, 0 mismatches; every other pairing mismatches 24-33 % of them
# (tools/test_decode_lean.py test_rope_fp32_contraction). A torch upgrade must re-run that test.
ROPE_RE, ROPE_IM = 1, 2


# id(gate fp32 weight) -> (that weight, its BF16 copy); the weight is held so the id stays its own.
_ROUTER_BF16: dict = {}


class LeanOps:
    """Static buffers plus the two decode ops. One instance per FastDecoder.

    A buffer is keyed by (purpose, width, rows): each verify width, the draft and every norm width
    has its own, so the zero rows past a call's T are never written by anybody. Buffers are
    created on first use, which is the capture warm-up (outside any graph), and then keep their
    addresses for the graphs that bake them in."""

    def __init__(self, device, mm_tile: int, hc_mm_tile: int):
        self.dev = device
        self.mm_tile = int(mm_tile)
        self.hc_mm_tile = int(hc_mm_tile)
        self._bufs: dict = {}
        self._wf: dict = {}
        self.fused_router_tail = os.environ.get("DSV41_ROUTER_FUSED_TAIL", "0") == "1"
        value = os.environ.get("DSV41_ROUTER_BF16", "0")
        if value not in ("0", "1"):
            raise ValueError("DSV41_ROUTER_BF16 must be 0 or 1")
        self.router_bf16 = value == "1"
        self.router_weights = {}
        # Launch-count fusions at fp32 rounding level (2026-10-08): one-pass RMSNorm (4 launches
        # -> 1; q/kv norm outputs bit-identical in the screen) and the HC mixes' mean(x^2) folded
        # into the split-K projection (pad copy, square and mean removed; pre/post/comb differ
        # by <= 2.4e-7). 10.3 -> 4.1 us per q_norm, 20.5 -> 13.4 us per HC mix on GB10.
        self.rms_fused = os.environ.get("DSV41_LEAN_RMS_FUSED", "0") == "1"
        self.hc_front_fused = os.environ.get("DSV41_HC_FRONT_FUSED", "0") == "1"

    def prepare_router_weights(self, weights):
        """Validate/retain original BF16 gate values before any graph capture.

        Opt-in arithmetic experiment: 60–61 -> 19 us per cold gate on GB10,
        but its FP32 rounding is different and is not a quality-equivalence claim.
        """
        if not self.router_bf16:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("BF16 router weights must be prepared outside capture")
        for w in weights:
            if id(w) in self.router_weights:
                continue
            # One copy per gate for the process: every decoder over these weights (a concurrent
            # lane is one) reads the same values, and 43 copies are 168 MB a decoder.
            bf = _ROUTER_BF16.get(id(w))
            if bf is None or bf[0] is not w:
                value = w.to(torch.bfloat16)
                if not torch.equal(w, value.float()):
                    raise ValueError("BF16 router requires exactly BF16-representable checkpoint gates")
                bf = _ROUTER_BF16[id(w)] = (w, value)
            self.router_weights[id(w)] = bf[1]

    def _buf(self, key, shape, dtype=torch.float32):
        b = self._bufs.get(key)
        if b is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"decode_lean buffer {key} first requested inside a graph capture")
            b = self._bufs[key] = torch.zeros(shape, dtype=dtype, device=self.dev)
        return b

    def _wf32(self, w):
        """w.float(), made once. A norm weight never changes after load."""
        k = (w.data_ptr(), tuple(w.shape), w.dtype)
        f = self._wf.get(k)
        if f is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("decode_lean fp32 weight copy first requested inside a graph capture")
            f = self._wf[k] = w.detach().float().contiguous()
        return f

    def usable_rms(self, x) -> bool:
        return (self.mm_tile > 0 and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
                and 0 < x.numel() // x.shape[-1] < self.mm_tile)

    def rmsnorm(self, x, w, eps: float):
        """v41_ref.rmsnorm(x, w, eps) for fewer than MM_TILE rows: 4 launches instead of 12."""
        D = x.shape[-1]
        T = x.numel() // D
        if self.rms_fused:
            x2 = x.reshape(T, D)
            if x2.stride(1) != 1:
                x2 = x2.contiguous()
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            BD = 1024 if D >= 1024 else triton.next_power_of_2(D)
            _rms_fused_kernel[(T,)](x2, self._wf32(w), out, D, x2.stride(0), float(eps), BD=BD,
                                    OUT_BF16=x.dtype == torch.bfloat16, num_warps=4,
                                    enable_fp_fusion=False, enable_reflect_ftz=False)
            return out
        pad = self._buf(("rms", D, T), (self.mm_tile, D))
        pad[:T].copy_(x.reshape(T, D))
        mean = pad.square().mean(-1, keepdim=True)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        BD = 1024 if D >= 1024 else triton.next_power_of_2(D)
        _rms_apply_kernel[(T, triton.cdiv(D, BD))](
            pad, mean, self._wf32(w), out, D, float(eps), BD=BD,
            OUT_BF16=x.dtype == torch.bfloat16, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def swiglu(self, gu, n: int, limit: float, out=None):
        """The elementwise middle of v41_ref.expert_ffn with a merged w1||w3 output `gu` [T, 2n]
        bf16: 1 launch instead of 7. `out`: a contiguous bf16 [T, n] to write into."""
        T = gu.numel() // gu.shape[-1]
        assert gu.dtype == torch.bfloat16 and gu.shape[-1] == 2 * n and gu.is_contiguous()
        h = torch.empty(*gu.shape[:-1], n, dtype=torch.bfloat16, device=gu.device) if out is None else out
        assert h.is_contiguous() and h.numel() == T * n
        BN = 1024
        _swiglu_kernel[(T, triton.cdiv(n, BN))](gu, h, n, float(limit), BN=BN, CLAMP=limit > 0,
                                                num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return h

    def rope(self, x, fq, rd: int, inverse: bool = False, re: int | None = None, im: int | None = None,
             out_f32: bool = False):
        """torch.cat([x[..., :-rd], v41_ref.apply_rotary(x[..., -rd:], fq, inverse)], -1) for x of
        shape [T, D] or [T, H, D] and fq complex64 [T, rd/2]: 1 launch. An fp32 x is rounded to
        bf16 first (the torch spelling's `.to(bf16)` before it); `out_f32` returns the bf16
        result as fp32 (its `.float()` after it)."""
        assert x.dtype in (torch.bfloat16, torch.float32) and fq.dtype == torch.complex64
        x = x.contiguous()
        D = x.shape[-1]
        T = x.shape[0]
        H = x.numel() // (T * D)
        f = torch.view_as_real(fq.contiguous())
        assert f.shape == (T, rd // 2, 2), (f.shape, T, rd)
        out = torch.empty(x.shape, dtype=torch.float32 if out_f32 else torch.bfloat16, device=x.device)
        _rope_kernel[(T * H,)](x, f, out, H, D=D, RD=rd, BD=triton.next_power_of_2(D), CONJ=inverse,
                               RE=ROPE_RE if re is None else re, IM=ROPE_IM if im is None else im,
                               IN_F32=x.dtype == torch.float32, OUT_F32=out_f32,
                               num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def usable_router(self, y, gate_w) -> bool:
        return (self.mm_tile > 0 and y.dtype == torch.bfloat16 and y.dim() == 2
                and 0 < y.size(0) < self.mm_tile and gate_w.dtype == torch.float32 and gate_w.dim() == 2
                and gate_w.shape[0] <= 1024)

    def router(self, y, gate_w, gate_bias, keep, k: int, route_scale: float, idx_out, w_out,
               record=None):
        """Model's decode router for `y` [T, dim] bf16 with the fp32 gate: top-k expert ids into
        `idx_out`, normalized weights into `w_out`. `keep` is the prune mask (or None) and
        `record(logits, scores)` the prune-miss accounting, fed the unmasked logits as before.
        The default padded gate GEMM and topk stay unchanged. The optional six-way tail
        reproduces torch's reduction order in one kernel; other k values keep torch.sum."""
        T = y.size(0)
        E = gate_w.shape[0]
        if self.router_bf16:
            from router_bf16 import project
            # BF16 is the checkpoint's value format, not a new quantization.
            # Tensor-core FP32 accumulation still changes rounding versus cuBLAS.
            g = project(y, self.router_weights[id(gate_w)], split=8)
        else:
            pad = self._buf(("router", T), (self.mm_tile, gate_w.shape[1]))
            pad[:T].copy_(y)
            # Draft gates have 128 experts; the target has 384. Odd-width
            # verification can have the same T as the draft, so T alone aliases.
            g = self._buf(("router_g", T, E), (self.mm_tile, E))
            torch.mm(pad, gate_w.t(), out=g)
        scores = torch.empty(T, E, dtype=torch.float32, device=y.device)
        logits = torch.empty_like(scores)
        masked = torch.empty_like(scores)
        keep_u8 = keep.view(torch.uint8) if keep is not None else None
        _router_pre_kernel[(T,)](g, gate_bias, keep_u8 if keep is not None else g, scores, logits, masked, E,
                                 HAS_KEEP=keep is not None, BE=triton.next_power_of_2(E), num_warps=4,
                                 enable_fp_fusion=False, enable_reflect_ftz=False)
        if record is not None:
            record(logits, scores)
        vals = self._buf(("router_v", T, k), (T, k))
        torch.topk(masked, k, dim=-1, out=(vals, idx_out))
        if self.fused_router_tail and k == 6:
            _router_gather_post_kernel[(T,)](scores, idx_out, w_out, E, float(route_scale),
                                             num_warps=1, enable_fp_fusion=False, enable_reflect_ftz=False)
        else:
            wts = scores.gather(1, idx_out)
            _router_post_kernel[(T,)](wts, wts.sum(dim=-1, keepdim=True), w_out, float(route_scale), K=k,
                                      KB=triton.next_power_of_2(k), num_warps=1, enable_fp_fusion=False, enable_reflect_ftz=False)
        return scores

    def attn_probs(self, s, mask, sink, scale: float):
        """The torch decode attention between its two einsums: scores s [T, H, N] fp32 (any
        strides) -> probabilities [T, H, N]. 3 launches (pre, torch's sum, post) instead of ~12."""
        T, H, N = s.shape
        assert s.dtype == torch.float32 and mask.shape == (T, N) and sink.shape == (H,)
        p = torch.empty(T, H, N, dtype=torch.float32, device=s.device)
        mx = torch.empty(T, H, 1, dtype=torch.float32, device=s.device)
        BN = triton.next_power_of_2(N)
        msk = mask if mask.is_contiguous() else mask.contiguous()
        _softmax_pre_kernel[(T * H,)](s, msk.view(torch.uint8), p, mx, H, N, s.stride(0), s.stride(1),
                                      s.stride(2), float(scale), BN=BN, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        psum = p.sum(-1, keepdim=True)
        out = torch.empty_like(p)
        _softmax_post_kernel[(T * H,)](p, psum, mx, sink, out, H, N, BN=BN, num_warps=4,
                                       enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def route_prep(self, idx, lut_row, slots_out, null: int):
        """(slots, route keys) for moe_forward's native decode path; 1 launch instead of ~6."""
        P = idx.numel()
        route = torch.empty(P, dtype=torch.int32, device=idx.device)
        _route_prep_kernel[(1,)](idx, lut_row, slots_out, route, P, int(null),
                                 PB=triton.next_power_of_2(P), num_warps=1)
        return route.view(idx.shape)

    def route_prep_masked(self, idx, lut_row, active, slots_out, null: int):
        """route_prep with idle token rows (active[t] == 0, uint8 [T]) skipped: key -1, slot null."""
        P = idx.numel()
        route = torch.empty(P, dtype=torch.int32, device=idx.device)
        _route_prep_masked_kernel[(1,)](idx, lut_row, active, slots_out, route, P, idx.shape[-1],
                                        int(null), PB=triton.next_power_of_2(P), num_warps=4)
        return route.view(idx.shape)

    def block_null(self, block_slot, null: int):
        n = block_slot.numel()
        _block_null_kernel[(1,)](block_slot, n, int(null), NBB=triton.next_power_of_2(n), num_warps=1)
        return block_slot

    def keys_f32(self, ring, slot_r, cache=None, idx=None, *, dtype=torch.float32):
        """The fp32 key block of the torch decode attention, [T, N1 (+ N2), D]: window rows
        ring[slot_r] and (optionally) the compressed rows gather(cache, idx), written in place
        instead of gathered as bf16, concatenated and converted. 1-2 launches instead of 4.
        Experimental tensor-core attention can request BF16 staging: all values were
        BF16 before widening, so this only changes the scratch buffer's storage."""
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("key staging supports FP32 or BF16")
        T, n1 = slot_r.shape
        D = ring.shape[-1]
        n2 = 0 if idx is None else idx.shape[1]
        out = torch.empty(T, n1 + n2, D, dtype=dtype, device=ring.device)
        assert ring.dtype == torch.bfloat16 and ring.is_contiguous() and slot_r.is_contiguous()
        _ring_gather_f32_kernel[(T * n1,)](ring, slot_r, out, n1, n1 + n2, D=D, num_warps=4)
        if idx is not None:
            if cache.dtype == torch.int64:
                idx = idx.contiguous()
                d = cache.shape[1] * 128 // 9
                assert d == D
                _packed_gather_f32_kernel[(triton.cdiv(idx.numel() * (D // 16), 128),)](
                    cache, idx, out, idx.numel(), n2, n1, n1 + n2, D=D, STRIDE=cache.shape[1],
                    BLOCK=128, num_warps=4)
            else:
                out[:, n1:].copy_(cache[idx])
        return out

    def usable_hc(self, x, hc_fn) -> bool:
        return (self.mm_tile > 0 and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 3
                and hc_fn.dtype == torch.float32 and tuple(hc_fn.shape) == (24, 20480)
                and x.shape[1] * x.shape[2] == 20480
                and 0 < x.size(0) < min(self.mm_tile, self.hc_mm_tile))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base, norm_eps: float, hc: int, iters: int, eps: float,
                 out=None):
        """Model._hc_mixes(x, ...) with HC_FUSED, for fewer than MM_TILE rows: 6 launches instead of
        15, and none for the copies when `out=(pre, post, comb)` names the destination buffers."""
        T = x.size(0)
        if (self.hc_front_fused and _R is not None and _R.hc_kernel_ok(hc_fn, T)
                and x.is_contiguous()):
            from fp32_skinny import skinny_linear_meansq
            # Same tiles and K order as hc_linear on the fp32 pad, and the bf16 upcast is exact, but
            # bf16 input changes the compiled MMA lowering: the projection differs at ~1e-7 and the
            # row mean of squares (same pass, no pad/square/mean kernels) at fp32 rounding.
            mm, mean = skinny_linear_meansq(x.reshape(T, -1), hc_fn, prec=_R.HC_PREC)
        else:
            mm = mean = None
        if mm is not None:
            if out is None:
                pre = torch.empty(T, hc, dtype=torch.float32, device=x.device)
                post = torch.empty(T, hc, dtype=torch.float32, device=x.device)
                comb = torch.empty(T, hc, hc, dtype=torch.float32, device=x.device)
            else:
                pre, post, comb = out
            _hc_rms_kernel[(T,)](mm, mean, hc_scale.contiguous(), hc_base.contiguous(), pre, post, comb,
                                 float(norm_eps), iters=iters, eps=eps, HC=hc)
            return pre, post, comb
        rows = max(self.mm_tile, self.hc_mm_tile)
        pad = self._buf(("hc", T), (rows, hc_fn.shape[1]))
        pad[:T].copy_(x.reshape(T, -1))
        if _R is not None and _R.hc_kernel_ok(hc_fn, T):
            # split-K kernel: a fresh output, and no padded rows -- it is row-invariant on its own
            mm = _R.hc_linear(pad[:T], hc_fn)
        else:
            mm = self._buf(("hc_mm", T), (self.hc_mm_tile, hc_fn.shape[0]))
            torch.mm(pad[:self.hc_mm_tile], hc_fn.t(), out=mm)
        mean = pad[:self.mm_tile].square().mean(-1, keepdim=True)
        if out is None:
            pre = torch.empty(T, hc, dtype=torch.float32, device=x.device)
            post = torch.empty(T, hc, dtype=torch.float32, device=x.device)
            comb = torch.empty(T, hc, hc, dtype=torch.float32, device=x.device)
        else:
            pre, post, comb = out
        _hc_rms_kernel[(T,)](mm, mean, hc_scale.contiguous(), hc_base.contiguous(), pre, post, comb,
                             float(norm_eps), iters=iters, eps=eps, HC=hc)
        return pre, post, comb
