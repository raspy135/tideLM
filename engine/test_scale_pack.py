import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from engine.scale_layout import decode
from engine.scale_pack import ScalePack
from tools.pack_expert_scales import main


class ScalePackTest(unittest.TestCase):
    def test_packer_roundtrip_replacement_and_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root/'model', root/'packs'
            model.mkdir()
            planes, header, mapping, offset = {}, {}, {}, 0
            for expert in (0, 1):
                for part in (1, 2, 3):
                    name = f'layers.0.ffn.experts.{expert}.w{part}.scale'
                    values = np.full((32, 64), 126+expert, np.uint8)
                    values[:, ::3] += 1
                    values[3, 7], values[24, 8] = 0, 255
                    planes[name] = values
                    header[name] = dict(dtype='U8', shape=list(values.shape),
                                        data_offsets=[offset, offset+values.size])
                    mapping[name] = 'weights.safetensors'
                    offset += values.size
            blob = json.dumps(header).encode()
            (model/'weights.safetensors').write_bytes(struct.pack('<Q', len(blob)) + blob +
                b''.join(p.tobytes() for p in planes.values()))
            (model/'config.json').write_text('{}')
            (model/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
            with patch.object(sys, 'argv', ['pack', '--model-dir', str(model), '--out-dir', str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            for rank in (0, 1):
                pack = ScalePack(output/f'ds41-scales-r{rank}of2.bin', model, rank, 2)
                for expert in (1, 0, 1):
                    raw = [planes[f'layers.0.ffn.experts.{expert}.w{part}.scale'][rank*16:(rank+1)*16]
                           for part in (1, 2, 3)]
                    destinations = [np.zeros(2048, np.uint8) for _ in raw]
                    pack.read_into((0, expert), raw, destinations)
                    for expected, stored in zip(raw, destinations):
                        np.testing.assert_array_equal(decode(stored[:pack.stride], pack.spec), expected.reshape(-1))
                # The buffered fallback (filesystems without O_DIRECT) reads the same bytes.
                buffered = ScalePack(output/f'ds41-scales-r{rank}of2.bin', model, rank, 2)
                if buffered.dfd is not None:
                    os.close(buffered.dfd)
                    buffered.dfd = None
                same = [np.zeros(2048, np.uint8) for _ in raw]
                buffered.read_into((0, expert), raw, same)
                for a, b in zip(destinations, same):
                    np.testing.assert_array_equal(a[:pack.stride], b[:pack.stride])
                os.close(buffered.fd)
                bad = [p.copy() for p in raw]
                bad[0][0, 0] ^= 1
                # Checks run on a record's first load in a process (the load path's cost,
                # engine/scale_pack.py); a verified record is not hashed again.
                pack.read_into((0, expert), bad, destinations)
                fresh = ScalePack(output/f'ds41-scales-r{rank}of2.bin', model, rank, 2)
                with self.assertRaisesRegex(ValueError, 'scale bytes changed'):
                    fresh.read_into((0, expert), bad, destinations)
                with open(pack.path, 'r+b') as handle:
                    handle.seek(pack.data0)
                    handle.write(b'\xff')
                with self.assertRaisesRegex(ValueError, 'corrupt scale record'):
                    fresh.read_into((0, 0), [planes[f'layers.0.ffn.experts.0.w{part}.scale'][rank*16:(rank+1)*16]
                                           for part in (1, 2, 3)], destinations)
                for p in (pack, fresh):
                    os.close(p.fd)
                    if p.dfd is not None:
                        os.close(p.dfd)


if __name__ == '__main__':
    unittest.main()
