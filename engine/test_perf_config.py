import os
import unittest
from unittest.mock import patch
from engine.perf_config import DecodeExperiments, FLAGS


class ConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_and_guarded(self):
        with patch.dict(os.environ, {}, clear=True):
            config = DecodeExperiments.from_env()
        for name in FLAGS:
            self.assertFalse(getattr(config, name))
            self.assertIn('perf_' + name, config.boot_fields())
        self.assertEqual(len(config.boot_fields()['perf_source']), 64)

    def test_each_flag_changes_rank_agreement(self):
        base = DecodeExperiments().boot_fields()
        for name, flag in FLAGS.items():
            with self.subTest(flag=flag), patch.dict(os.environ, {flag: '1'}, clear=True):
                self.assertNotEqual(base, DecodeExperiments.from_env().boot_fields())
                self.assertTrue(getattr(DecodeExperiments.from_env(), name))
            with patch.dict(os.environ, {flag: 'yes'}, clear=True), self.assertRaises(ValueError):
                DecodeExperiments.from_env()


if __name__ == '__main__':
    unittest.main()
