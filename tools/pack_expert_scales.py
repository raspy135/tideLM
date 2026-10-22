"""Pack every DS native FP4 expert's scales losslessly, for output TP2 (CPU only).

Writes ds41-scales-r{0,1}of2.bin. Both ranks use the same worst-case exception
capacity, so slots stay interchangeable and their memory budgets agree. All
source bytes round-trip before either final pack replaces an existing file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.scale_layout import FORMAT, align, decode, encode, layout
from engine.scale_pack import source_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--layers', default='', help='subset for qualification only; default all')
    ap.add_argument('--experts', default='', help='subset for qualification only; default all')
    args = ap.parse_args()
    mapping = json.loads((args.model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    pattern = re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.scale$')
    keys = sorted({(int(m[1]), int(m[2])) for name in mapping if (m := pattern.match(name))})
    if args.layers:
        wanted = set(map(int, args.layers.split(',')))
        keys = [key for key in keys if key[0] in wanted]
    if args.experts:
        wanted = set(map(int, args.experts.split(',')))
        keys = [key for key in keys if key[1] in wanted]
    if not keys:
        raise ValueError('no native DS expert scales found')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest(args.model_dir)
    headers, handles = {}, {}

    def tensor(name):
        path = args.model_dir / mapping[name]
        if path not in headers:
            fd = os.open(path, os.O_RDONLY)
            handles[path] = fd
            length = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            headers[path] = 8 + length, json.loads(os.pread(fd, length, 8))
        base, header = headers[path]
        meta = header[name]
        first, last = meta['data_offsets']
        raw = os.pread(handles[path], last-first, base+first)
        if len(raw) != last-first or len(raw) != int(np.prod(meta['shape'])):
            raise ValueError(f'invalid native scale plane {name}')
        return np.frombuffer(raw, np.uint8).reshape(meta['shape'])

    started, capacity, size = time.monotonic(), 0, None
    global_digest = hashlib.sha256()
    directories, native_hashes = [[], []], [[], []]
    with tempfile.TemporaryDirectory(prefix='.scales-', dir=args.out_dir) as temporary:
        with open(Path(temporary)/'rank0', 'w+b') as s0, open(Path(temporary)/'rank1', 'w+b') as s1:
            spools = (s0, s1)
            for number, (layer, expert) in enumerate(keys):
                raw = [tensor(f'layers.{layer}.ffn.experts.{expert}.w{p}.scale') for p in (1, 2, 3)]
                for plane in raw:
                    global_digest.update(memoryview(plane))
                for rank, spool in enumerate(spools):
                    spans, digest = [], hashlib.sha256()
                    for plane in raw:
                        if plane.shape[0] % 2:
                            raise ValueError('output TP2 requires even scale rows')
                        rows = plane.shape[0] // 2
                        local = np.ascontiguousarray(plane[rank*rows:(rank+1)*rows])
                        digest.update(memoryview(local))
                        if size is None:
                            size = local.size
                        if size != local.size:
                            raise ValueError('expert scale planes must have one common size')
                        encoded = encode(local)
                        count = len(encoded.exceptions)
                        spec = layout(size, count)
                        packed = encoded.pack(count)
                        np.testing.assert_array_equal(decode(packed, spec), local.reshape(-1))
                        capacity = max(capacity, count)
                        spans.append((spool.tell(), len(packed)))
                        spool.write(memoryview(packed))
                    directories[rank].append(spans)
                    native_hashes[rank].append(digest.hexdigest())
                if (number + 1) % 512 == 0:
                    print(f'[scales] {number+1}/{len(keys)} experts, {time.monotonic()-started:.1f}s', flush=True)
            spec = layout(size, capacity)
            stride = spec['stride']
            if stride >= size:
                raise ValueError(f'worst-case scale record {stride} does not save bytes over {size}; keep native scales')
            for rank, spool in enumerate(spools):
                spool.flush()
                header = dict(format=FORMAT, rank=rank, world=2, keys=keys, scale_size=size,
                              exception_capacity=capacity, source_manifest=manifest,
                              source_scale_sha256=global_digest.hexdigest(), native_sha256=native_hashes[rank],
                              record_sha256=['0'*64 for _ in keys])
                blob = json.dumps(header, separators=(',', ':')).encode()
                data0 = align(8+len(blob), 4096)
                target = args.out_dir / f'ds41-scales-r{rank}of2.bin'
                temporary_target = target.with_suffix('.bin.tmp')
                with temporary_target.open('w+b') as dest:
                    dest.seek(data0)
                    for i, spans in enumerate(directories[rank]):
                        digest = hashlib.sha256()
                        for offset, length in spans:
                            spool.seek(offset)
                            packed = spool.read(length) + bytes(stride-length)
                            digest.update(packed)
                            dest.write(packed)
                        header['record_sha256'][i] = digest.hexdigest()
                    blob = json.dumps(header, separators=(',', ':')).encode()
                    if align(8+len(blob), 4096) != data0:
                        raise RuntimeError('scale directory size changed')
                    dest.seek(0)
                    dest.write(struct.pack('<Q', len(blob)))
                    dest.write(blob)
                    dest.flush()
                    os.fsync(dest.fileno())
                os.replace(temporary_target, target)
                print(f'[scales] rank {rank}: {target}, {target.stat().st_size/1e9:.3f} GB, '
                      f'scales {size} -> {stride} bytes/plane; all bytes round-tripped', flush=True)
    for fd in handles.values():
        os.close(fd)


if __name__ == '__main__':
    main()
