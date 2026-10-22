"""Byte-exact standalone expansion of validated TideLM scale records."""
import triton
import triton.language as tl


@triton.jit
def _expand(PAYLOAD, OFFSETS, OUT, N: tl.constexpr):
    block = tl.program_id(0)
    i = tl.arange(0, 64)
    first = tl.load(OFFSETS + block)
    kind = tl.load(PAYLOAD + first)
    live = block * 64 + i < N
    if kind == 0:
        values = tl.load(PAYLOAD + first + 1 + i, live, other=0)
    else:
        base = tl.load(PAYLOAD + first + 1).to(tl.int32)
        codes = tl.load(PAYLOAD + first + 2 + i // 8).to(tl.int32)
        bitmap = tl.load(PAYLOAD + first + 10 + i // 8).to(tl.int32)
        escaped = ((bitmap >> (i % 8)) & 1) != 0
        ordinal = tl.cumsum(escaped.to(tl.int32)) - 1
        exception = tl.load(PAYLOAD + first + 18 + ordinal, escaped & live, other=0)
        values = tl.where(escaped, exception, base + ((codes >> (i % 8)) & 1))
    tl.store(OUT + block * 64 + i, values, live)


def expand(payload, offsets, out):
    if out.numel():
        _expand[(triton.cdiv(out.numel(), 64),)](payload, offsets, out, out.numel(), num_warps=4)
