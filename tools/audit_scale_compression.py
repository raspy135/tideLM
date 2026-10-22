"""Read selected expert scale planes and audit lossless compression on CPU only.

Does not load weights, import torch, touch CUDA, write packs, or change serving.
Savings exclude expansion scratch and directory costs in a future arena.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.scale_blocks import encode, decode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--layers', default='0,20,39')
    ap.add_argument('--experts', default='0,1')
    ap.add_argument('--tp-rank', type=int, choices=(0, 1), default=0)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    index = json.loads((args.model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    headers, rows = {}, []

    def tensor(name, read):
        path = args.model_dir / index[name]
        if path not in headers:
            with path.open('rb') as handle:
                length = struct.unpack('<Q', handle.read(8))[0]
                headers[path] = 8 + length, json.loads(handle.read(length))
        first, header = headers[path]
        meta = header[name]
        start, end = meta['data_offsets']
        if not read:
            return end - start
        with path.open('rb') as handle:
            handle.seek(first + start)
            payload = handle.read(end - start)
        if len(payload) != end - start:
            raise ValueError(f'truncated tensor {name}')
        if len(payload) != int(np.prod(meta['shape'])):
            raise ValueError(f'{name} is not a native byte-valued scale plane')
        return np.frombuffer(payload, dtype=np.uint8).reshape(meta['shape'])

    for layer in map(int, args.layers.split(',')):
        for expert in map(int, args.experts.split(',')):
            raw, stored, weights = 0, 0, 0
            digest = hashlib.sha256()
            for projection in ('w1', 'w2', 'w3'):
                prefix = f'layers.{layer}.ffn.experts.{expert}.{projection}'
                plane = tensor(prefix + '.scale', True)
                if plane.shape[0] % 2:
                    raise ValueError('output TP2 requires evenly split scale rows')
                first = args.tp_rank * plane.shape[0] // 2
                local = plane[first:first + plane.shape[0] // 2]
                digest.update(np.ascontiguousarray(local).tobytes())
                packed = encode(local)
                np.testing.assert_array_equal(decode(packed), local)
                raw += local.nbytes
                stored += packed.nbytes
                weights += tensor(prefix + '.weight', False) // 2
            rows.append(dict(layer=layer, expert=expert, tp_rank=args.tp_rank,
                scale_sha256=digest.hexdigest(),
                native_scale_bytes=raw, encoded_scale_bytes=stored, weight_bytes=weights,
                hypothetical_slot_saving_fraction=(raw - stored) / (weights + raw)))
    raw = sum(r['native_scale_bytes'] for r in rows)
    stored = sum(r['encoded_scale_bytes'] for r in rows)
    weights = sum(r['weight_bytes'] for r in rows)
    report = dict(format='tidelm-scale-blocks-v1', checkpoint=args.model_dir.name,
        byte_exact=True, samples=rows,
        native_scale_bytes=raw, encoded_scale_bytes=stored,
        hypothetical_slot_saving_fraction=(raw - stored) / (weights + raw),
        runtime_arena_integration=False, expansion_scratch_included=False)
    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
