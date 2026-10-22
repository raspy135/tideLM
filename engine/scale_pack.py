"""Validated fixed-stride scale packs; expert weights remain in the native checkpoint."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import threading

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
        # Records are read on the expert-load path (hot reads, adaptation swaps). A buffered read
        # went through the page cache -- on GB10 the GPU's memory -- at 0.89 ms for a cold 336 KB
        # record, and the two SHA-256 checks added 0.36 ms: a packed load_slot took 1.26-1.52 ms
        # against native's 0.32 (2026-10-22). So: O_DIRECT like the expert weights, through a
        # per-thread aligned bounce buffer (records are not 4096-aligned), and each record's
        # checks once per process. Filesystems without O_DIRECT (tmpfs) keep the buffered read.
        try:
            self.dfd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError:
            self.dfd = None
        self._tls = threading.local()
        self._verified = set()

    def boot_fields(self):
        return dict(format=FORMAT, source_manifest=self.header['source_manifest'],
                    source_scale_sha256=self.header['source_scale_sha256'],
                    spec=self.spec, keys=len(self.keys))

    def _bounce(self):
        buf = getattr(self._tls, 'buf', None)
        if buf is None:
            # Anonymous mmap: page-aligned, as O_DIRECT requires; one record plus both edges.
            buf = self._tls.buf = mmap.mmap(-1, align(self.record_bytes, 4096) + 4096)
        return buf

    def read_into(self, key, native_planes, destinations):
        index = self.keys[tuple(key)]
        verify = index not in self._verified
        if verify:
            # Prove that the scales being replaced are the bytes audited by this
            # pack. A finetune can change values without changing shard headers.
            digest = hashlib.sha256()
            for plane in native_planes:
                digest.update(memoryview(np.ascontiguousarray(plane)))
            if digest.hexdigest() != self.header['native_sha256'][index]:
                raise ValueError(f'scale bytes changed for expert {key}; rebuild the scale pack')
        offset = self.data0 + index * self.record_bytes
        start = offset - offset % 4096 if self.dfd is not None else offset
        span = align(offset + self.record_bytes, 4096) - start if self.dfd is not None else self.record_bytes
        view = memoryview(self._bounce())
        record = None
        try:
            got = os.preadv(self.fd if self.dfd is None else self.dfd, [view[:span]], start)
            skip = offset - start
            if got < skip + self.record_bytes:          # the aligned tail may pass the file's end
                raise IOError(f'short scale pack read for expert {key}')
            record = np.frombuffer(view, np.uint8, count=self.record_bytes, offset=skip)
            if verify and hashlib.sha256(record).hexdigest() != self.header['record_sha256'][index]:
                raise ValueError(f'corrupt scale record for expert {key}')
            for i, destination in enumerate(destinations):
                destination[:self.stride] = record[i * self.stride:(i + 1) * self.stride]
        finally:
            record = None                               # drop the array's export before release
            view.release()
        if verify:
            self._verified.add(index)
