import unittest
import numpy as np

from engine.scale_layout import decode, encode, layout


class ScaleLayoutTest(unittest.TestCase):
    def test_every_byte_and_partial_blocks_roundtrip(self):
        for values in (np.arange(256, dtype=np.uint8), np.array([0, 255, 127], np.uint8),
                       np.zeros(129, np.uint8), np.full(257, 255, np.uint8),
                       np.random.default_rng(93).integers(0, 256, (13, 17), dtype=np.uint8)[:, ::2]):
            with self.subTest(shape=values.shape):
                plane = encode(values)
                spec = layout(values.size, len(plane.exceptions))
                np.testing.assert_array_equal(decode(plane.pack(spec['capacity']), spec), values.reshape(-1))

    def test_exceptions_and_padding_cannot_address_other_records(self):
        plane = encode(np.arange(192, dtype=np.uint8))
        count = len(plane.exceptions)
        with self.assertRaises(ValueError):
            plane.pack(count - 1)
        spec = layout(plane.size, count)
        packed = plane.pack(count)
        packed[spec['offsets']:spec['offsets'] + 4] = 255
        with self.assertRaises(ValueError):
            decode(packed, spec)


if __name__ == '__main__':
    unittest.main()
