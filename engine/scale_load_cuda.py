"""Random-access lossless scale reconstruction inside the existing MoE kernels."""
import triton
import triton.language as tl


@triton.jit
def load_scale(S, slot, index, SIZE: tl.constexpr, STRIDE: tl.constexpr):
    blocks: tl.constexpr = (SIZE + 63) // 64
    codes0: tl.constexpr = (blocks + 7) // 8 * 8
    escaped0: tl.constexpr = codes0 + blocks * 8
    offsets0: tl.constexpr = escaped0 + blocks * 8
    payload0: tl.constexpr = (offsets0 + blocks * 4 + 7) // 8 * 8
    record = S + slot.to(tl.int64) * STRIDE
    block, bit = index // 64, index % 64
    codes = tl.load((record + codes0).to(tl.pointer_type(tl.uint64)) + block)
    bitmap = tl.load((record + escaped0).to(tl.pointer_type(tl.uint64)) + block)
    escaped = (bitmap >> bit) & 1
    preceding = bitmap & ((tl.full((), 1, tl.uint64) << bit) - 1)
    ordinal = tl.inline_asm_elementwise('popc.b64 $0, $1;', '=r,l', [preceding],
                                       dtype=tl.int32, is_pure=True, pack=1)
    offset = tl.load((record + offsets0).to(tl.pointer_type(tl.int32)) + block)
    exception = tl.load(record + payload0 + offset + ordinal, escaped != 0, other=0)
    base = tl.load(record + block)
    return tl.where(escaped != 0, exception, base.to(tl.uint32) + ((codes >> bit) & 1)).to(tl.uint8)


@triton.jit
def _expand(PACKED, OUT, SIZE: tl.constexpr, STRIDE: tl.constexpr, BS: tl.constexpr):
    index = tl.program_id(0) * BS + tl.arange(0, BS)
    slot = tl.program_id(1)
    # Reading the final valid byte for masked lanes keeps metadata reads in bounds.
    value = load_scale(PACKED, slot, tl.minimum(index, SIZE-1), SIZE, STRIDE)
    tl.store(OUT + slot * SIZE + index, value, index < SIZE)


def expand(packed, size):
    import torch
    if packed.ndim != 2 or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError('expected contiguous packed scale slots')
    output = torch.empty((packed.shape[0], size), dtype=torch.uint8, device=packed.device)
    _expand[(triton.cdiv(size, 256), packed.shape[0])](packed, output, size, packed.shape[1], 256)
    return output
