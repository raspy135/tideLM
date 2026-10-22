"""Lossless random-access UE8M0 scales: adjacent exponents plus exact exceptions.

Every 64-byte block has a base, 64 code bits, 64 exception bits and an exception
offset. Fixed slot strides use the largest exception payload in a checkpoint
pack, so every audited expert remains replaceable without allocation in graphs.
"""
from dataclasses import dataclass
import numpy as np

FORMAT = 'tidelm-expert-scales-v1'
BLOCK = 64


def align(value, boundary):
    return (value + boundary - 1) // boundary * boundary


def layout(size, exceptions):
    if size < 1 or not 0 <= exceptions <= size:
        raise ValueError('invalid scale plane size or exception capacity')
    blocks = (size + BLOCK - 1) // BLOCK
    codes = align(blocks, 8)
    escaped = codes + blocks * 8
    offsets = escaped + blocks * 8
    payload = align(offsets + blocks * 4, 8)
    return dict(size=size, capacity=exceptions, blocks=blocks, codes=codes,
                escaped=escaped, offsets=offsets, payload=payload,
                stride=align(payload + exceptions, 256))


@dataclass
class Plane:
    size: int
    bases: np.ndarray
    codes: np.ndarray
    escaped: np.ndarray
    offsets: np.ndarray
    exceptions: np.ndarray

    def pack(self, capacity):
        if len(self.exceptions) > capacity:
            raise ValueError('scale exceptions exceed the audited slot capacity')
        spec = layout(self.size, capacity)
        out = np.zeros(spec['stride'], np.uint8)
        out[:spec['blocks']] = self.bases
        for key, array in (('codes', self.codes), ('escaped', self.escaped), ('offsets', self.offsets)):
            raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
            out[spec[key]:spec[key] + len(raw)] = raw
        out[spec['payload']:spec['payload'] + len(self.exceptions)] = self.exceptions
        return out


def encode(source):
    source = np.asarray(source)
    if source.dtype != np.uint8 or not source.size:
        raise ValueError('expected a nonempty native UE8M0 byte plane')
    flat = np.ascontiguousarray(source).reshape(-1)
    blocks = (len(flat) + BLOCK - 1) // BLOCK
    values = np.zeros((blocks, BLOCK), np.uint8)
    values.reshape(-1)[:len(flat)] = flat
    valid = np.arange(blocks * BLOCK).reshape(blocks, BLOCK) < len(flat)
    bins = np.arange(blocks, dtype=np.int64)[:, None] * 256 + values
    hist = np.bincount(bins[valid], minlength=blocks * 256).reshape(blocks, 256)
    bases = (hist[:, :-1] + hist[:, 1:]).argmax(1).astype(np.uint8)
    codes = values == bases[:, None] + np.uint8(1)
    escaped = (values != bases[:, None]) & ~codes & valid
    counts = escaped.sum(1, dtype=np.int32)
    offsets = np.cumsum(counts, dtype=np.int32) - counts
    bits = lambda x: np.packbits(x, axis=1, bitorder='little').copy().view('<u8').reshape(-1)
    return Plane(len(flat), bases, bits(codes), bits(escaped), offsets.astype('<i4'), values[escaped].copy())


def decode(record, spec):
    if spec != layout(spec['size'], spec['capacity']):
        raise ValueError('invalid scale layout')
    data = np.asarray(record)
    if data.dtype != np.uint8 or data.ndim != 1 or len(data) != spec['stride']:
        raise ValueError('invalid packed scale record')
    blocks = spec['blocks']
    codes = np.unpackbits(data[spec['codes']:spec['codes'] + blocks * 8].reshape(blocks, 8),
                          axis=1, bitorder='little')
    escaped = np.unpackbits(data[spec['escaped']:spec['escaped'] + blocks * 8].reshape(blocks, 8),
                            axis=1, bitorder='little').astype(bool)
    valid = np.arange(blocks * BLOCK).reshape(blocks, BLOCK) < spec['size']
    if escaped[~valid].any() or (data[:blocks] == 255).any():
        raise ValueError('invalid scale bitmap or base')
    counts = escaped.sum(1, dtype=np.int64)
    offsets = data[spec['offsets']:spec['offsets'] + blocks * 4].view('<i4')
    if not np.array_equal(offsets, counts.cumsum() - counts) or counts.sum() > spec['capacity']:
        raise ValueError('invalid scale exception offsets')
    values = data[:blocks, None] + codes
    values[escaped] = data[spec['payload']:spec['payload'] + counts.sum()]
    return values.reshape(-1)[:spec['size']]
