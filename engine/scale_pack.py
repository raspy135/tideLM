"""Validated fixed-stride scale packs; expert weights remain in the native checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from .scale_layout import FORMAT, align, layout


def source_manifest(directory):
    directory = Path(directory)
    digest = hashlib.sha256()
    for name in ('config.json', 'model.safetensors.index.json'):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    for path in sorted(directory.glob('*.safetensors')):
        with path.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            digest.update(path.name.encode())
            digest.update(handle.read(length))
    return digest.hexdigest()


class ScalePack:
    def __init__(self, path, model_dir, rank, world):
        self.path = str(path)
        self.fd = os.open(path, os.O_RDONLY)
        length = struct.unpack('<Q', os.pread(self.fd, 8, 0))[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError('invalid scale pack header length')
        self.header = header = json.loads(os.pread(self.fd, length, 8))
        if header['format'] != FORMAT or (header['rank'], header['world']) != (rank, world):
            raise ValueError('scale pack format or TP rank mismatch')
        if header['source_manifest'] != source_manifest(model_dir):
            raise ValueError('scale pack belongs to a different checkpoint; rebuild it')
        self.spec = layout(header['scale_size'], header['exception_capacity'])
        if self.spec['stride'] >= self.spec['size']:
            raise ValueError('scale pack does not save memory')
        self.stride = self.spec['stride']
        self.record_bytes = self.stride * 3
        self.keys = {tuple(key): i for i, key in enumerate(header['keys'])}
        count = len(self.keys)
        if count != len(header['keys']) or any(len(header[k]) != count for k in ('native_sha256', 'record_sha256')):
            raise ValueError('invalid scale pack directory')
        self.data0 = align(8 + length, 4096)
        if os.fstat(self.fd).st_size != self.data0 + count * self.record_bytes:
            raise ValueError('truncated or oversized scale pack')

    def boot_fields(self):
        return dict(format=FORMAT, source_manifest=self.header['source_manifest'],
                    source_scale_sha256=self.header['source_scale_sha256'],
                    spec=self.spec, keys=len(self.keys))

    def read_into(self, key, native_planes, destinations):
        index = self.keys[tuple(key)]
        # Prove that the scales being replaced are the bytes audited by this
        # pack. A finetune can change values without changing shard headers.
        digest = hashlib.sha256()
        for plane in native_planes:
            digest.update(memoryview(np.ascontiguousarray(plane)))
        if digest.hexdigest() != self.header['native_sha256'][index]:
            raise ValueError(f'scale bytes changed for expert {key}; rebuild the scale pack')
        views = [memoryview(p)[:self.stride] for p in destinations]
        try:
            got = os.preadv(self.fd, views, self.data0 + index * self.record_bytes)
            if got != self.record_bytes:
                raise IOError(f'short scale pack read for expert {key}')
            digest = hashlib.sha256()
            for view in views:
                digest.update(view)
            if digest.hexdigest() != self.header['record_sha256'][index]:
                raise ValueError(f'corrupt scale record for expert {key}')
        finally:
            for view in views:
                view.release()
