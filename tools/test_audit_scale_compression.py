"""Exercise the on-disk auditor with byte-valued tensor planes and both TP halves."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.audit_scale_compression import main


class AuditTest(unittest.TestCase):
    def test_reads_only_rank_local_scale_slices_and_reports_exact_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tensors = {}
            native = []
            for name in ('w1', 'w2', 'w3'):
                prefix = 'layers.20.ffn.experts.0.' + name
                plane = np.full((128, 16), 126, np.uint8)
                plane[64:] = 253
                plane[:, ::2] += 1
                tensors[prefix + '.scale'] = plane
                tensors[prefix + '.weight'] = np.zeros((128, 256), np.uint8)
                native.append(plane)
            header, payload = {}, bytearray()
            for name, array in tensors.items():
                start = len(payload)
                payload.extend(array.tobytes())
                header[name] = dict(dtype='U8', shape=list(array.shape), data_offsets=[start, len(payload)])
            raw_header = json.dumps(header).encode()
            (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw_header)) + raw_header + payload)
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {key: 'model.safetensors' for key in tensors}}))
            hashes = []
            for rank in (0, 1):
                out = root / f'audit{rank}.json'
                arguments = ['audit', '--model-dir', folder, '--layers', '20', '--experts', '0',
                             '--tp-rank', str(rank), '--out', str(out)]
                with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()):
                    main()
                report = json.loads(out.read_text())
                self.assertTrue(report['byte_exact'])
                self.assertFalse(report['runtime_arena_integration'])
                self.assertEqual(report['native_scale_bytes'], 3 * 64 * 16)
                self.assertLess(report['encoded_scale_bytes'], report['native_scale_bytes'])
                self.assertEqual(report['samples'][0]['weight_bytes'], 3 * 64 * 256)
                hashes.append(report['samples'][0]['scale_sha256'])
            self.assertNotEqual(*hashes)


if __name__ == '__main__':
    unittest.main()
