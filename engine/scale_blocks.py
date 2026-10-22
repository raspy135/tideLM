"""Lossless UE8M0 scale block codec for residency experiments.

This is TideLM's format, not Flare/b12x CSF. A 64-byte block uses a palette
of two adjacent exponent bytes, one-bit codes, and indexed byte exceptions.
Incompressible blocks are stored raw; an incompressible plane stays entirely
raw so the encoded numeric payload never exceeds the original plane.

The codec and standalone CUDA expansion are qualification components. ExpertArena
still owns native scale planes; enabling this codec does not yet save arena RAM.
"""
from dataclasses import dataclass
import numpy as np

FORMAT = 'tidelm-scale-blocks-v1'
BLOCK = 64


@dataclass(frozen=True)
class ScaleBlocks:
    shape: tuple
    payload: bytes
    offsets: bytes = b''

    @property
    def nbytes(self):
        return len(self.payload) + len(self.offsets)

    @property
    def raw_bytes(self):
        return int(np.prod(self.shape))


def encode(scales):
    source = np.asarray(scales)
    if source.dtype != np.uint8:
        raise ValueError('scales must contain native UE8M0 bytes, not converted floats')
    flat = np.ascontiguousarray(source).reshape(-1)
    chunks, offsets = [], [0]
    for first in range(0, flat.size, BLOCK):
        values = flat[first:first + BLOCK]
        hist = np.bincount(values, minlength=256)
        # Deterministic ties select the lower adjacent palette. Do not wrap 255.
        base = int(np.argmax(hist[:-1] + hist[1:]))
        exception = (values != base) & (values != base + 1)
        codes = np.zeros(BLOCK, np.uint8)
        escapes = np.zeros(BLOCK, np.uint8)
        codes[:len(values)] = values == base + 1
        escapes[:len(values)] = exception
        palette = (bytes((1, base)) + np.packbits(codes, bitorder='little').tobytes()
                   + np.packbits(escapes, bitorder='little').tobytes() + values[exception].tobytes())
        raw = b'\x00' + values.tobytes()
        record = palette if len(palette) < len(raw) else raw
        chunks.append(record)
        offsets.append(offsets[-1] + len(record))
    payload = b''.join(chunks)
    if len(payload) + 4 * len(offsets) >= flat.nbytes:
        return ScaleBlocks(tuple(source.shape), flat.tobytes())
    if offsets[-1] > np.iinfo(np.uint32).max:
        raise ValueError('encode one expert scale plane at a time (offsets exceed uint32)')
    return ScaleBlocks(tuple(source.shape), payload, np.asarray(offsets, dtype='<u4').tobytes())


def decode(encoded):
    size = encoded.raw_bytes
    if not encoded.offsets:
        if len(encoded.payload) != size:
            raise ValueError('raw scale payload length differs from shape')
        return np.frombuffer(encoded.payload, np.uint8).copy().reshape(encoded.shape)
    if len(encoded.offsets) % 4:
        raise ValueError('truncated scale offset table')
    offsets = np.frombuffer(encoded.offsets, '<u4').astype(np.int64)
    blocks = (size + BLOCK - 1) // BLOCK
    if (len(offsets) != blocks + 1 or offsets[0] != 0 or offsets[-1] != len(encoded.payload)
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError('invalid scale block offsets')
    result = np.empty(size, np.uint8)
    for block in range(blocks):
        record = encoded.payload[offsets[block]:offsets[block + 1]]
        first = block * BLOCK
        count = min(BLOCK, size - first)
        if record[0] == 0:
            if len(record) != count + 1:
                raise ValueError('invalid raw scale block')
            result[first:first + count] = np.frombuffer(record, np.uint8, offset=1)
        elif record[0] == 1:
            if len(record) < 18 or record[1] == 255:
                raise ValueError('invalid palette scale block')
            codes = np.unpackbits(np.frombuffer(record[2:10], np.uint8), bitorder='little')
            escapes = np.unpackbits(np.frombuffer(record[10:18], np.uint8), bitorder='little').astype(bool)
            if escapes[count:].any() or len(record) != 18 + int(escapes[:count].sum()):
                raise ValueError('invalid scale exceptions')
            values = (codes[:count] + record[1]).astype(np.uint8)
            values[escapes[:count]] = np.frombuffer(record[18:], np.uint8)
            result[first:first + count] = values
        else:
            raise ValueError('unknown scale block kind')
    return result.reshape(encoded.shape)


def expand_cuda(encoded, out=None):
    """Upload and decode one plane. Caller-owned out supports scratch poisoning tests.

    This allocation-owning helper is for qualification, not captured decode.
    A serving integration needs persistent payloads and selected-slot expansion.
    """
    import torch
    from engine.scale_blocks_cuda import expand
    # Validate offsets/exceptions before any device-side address arithmetic.
    decode(encoded)
    if out is None:
        out = torch.empty(encoded.shape, dtype=torch.uint8, device='cuda')
    if (tuple(out.shape) != encoded.shape or out.dtype != torch.uint8
            or out.device.type != 'cuda' or not out.is_contiguous()):
        raise ValueError('scale destination must be contiguous uint8 with the original shape')
    payload = torch.from_numpy(np.frombuffer(encoded.payload, np.uint8).copy()).to(out.device)
    if not encoded.offsets:
        out.reshape(-1).copy_(payload)
    elif encoded.raw_bytes:
        offsets = torch.from_numpy(np.frombuffer(encoded.offsets, '<u4').astype(np.int64)).to(out.device)
        expand(payload, offsets, out)
    return out
