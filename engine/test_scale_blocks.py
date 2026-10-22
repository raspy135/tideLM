from dataclasses import replace
import unittest
import numpy as np
from engine.scale_blocks import encode, decode


class ScaleBlocksTest(unittest.TestCase):
    def check(self, source):
        encoded = encode(source)
        np.testing.assert_array_equal(decode(encoded), source)
        self.assertLessEqual(encoded.nbytes, source.nbytes)
        return encoded

    def test_all_native_bytes_and_incompressible_fallback(self):
        self.assertFalse(self.check(np.arange(256, dtype=np.uint8)).offsets)
        self.check(np.random.default_rng(7).integers(0, 256, (17, 73), dtype=np.uint8))

    def test_palettes_exceptions_partial_blocks_and_strides(self):
        values = np.full((53, 160), 126, np.uint8)
        values[:, 1::2] = 127
        values.reshape(-1)[::97] = 255
        values.reshape(-1)[::211] = 0
        self.assertTrue(self.check(values).offsets)
        self.check(values[:, ::3])
        self.check(np.full((3, 67), 255, np.uint8))
        self.check(np.zeros((2, 0), np.uint8))

    def test_incompressible_blocks_inside_compressed_plane(self):
        values = np.full(1024, 127, np.uint8)
        values[64:128] = np.arange(64, dtype=np.uint8)
        self.assertTrue(self.check(values).offsets)

    def test_malformed_payloads_rejected_before_gpu_addressing(self):
        encoded = encode(np.full(512, 126, np.uint8))
        for bad in (replace(encoded, offsets=encoded.offsets[:-1]),
                    replace(encoded, payload=encoded.payload[:-1]),
                    replace(encoded, payload=b'\xff' + encoded.payload[1:]),
                    replace(encoded, payload=b'\x01\xff' + encoded.payload[2:])):
            with self.subTest(bad=bad.payload[:2]), self.assertRaises(ValueError):
                decode(bad)
        with self.assertRaises(ValueError):
            encode(np.ones(10, np.float32))


if __name__ == '__main__':
    unittest.main()
