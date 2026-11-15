import argparse
import tempfile
import unittest
from pathlib import Path
from utils.config_utils import parse_training_args


class TrainingConfigurationTests(unittest.TestCase):
    def parse(self, text, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'scene.yaml'
            config.write_text(text)
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', required=True)
            parser.add_argument('--iterations', type=int, default=30000)
            parser.add_argument('--save_iterations', nargs='+', type=int, default=[1, 1000, 30000])
            parser.add_argument('--source_path', default='')
            parser.add_argument('--load_path', default='')
            parser.add_argument('--test_only', action='store_true')
            return parse_training_args(parser, ['--config', str(config)] + list(extra))

    def test_cli_overrides_yaml_and_saves_the_actual_final_iteration(self):
        args = self.parse('iterations: 1200\nsource_path: data_hdr/synthetic/bathroom\n', ['--iterations', '25', '--source_path', 'custom_scene'])
        self.assertEqual(args.iterations, 25)
        self.assertEqual(args.source_path, 'custom_scene')
        self.assertIn(25, args.save_iterations)

    def test_yaml_defaults_are_used_without_overrides(self):
        args = self.parse('iterations: 1200\nsource_path: data_hdr/real/flower\n')
        self.assertEqual(args.iterations, 1200)
        self.assertIn(1200, args.save_iterations)
        self.assertEqual(args.source_path, 'data_hdr/real/flower')

    def test_test_only_requires_a_saved_model(self):
        with self.assertRaises(SystemExit):
            self.parse('iterations: 10\n', ['--test_only'])
        args = self.parse('iterations: 10\n', ['--test_only', '--load_path', 'pretrained_weights/bathroom'])
        self.assertTrue(args.test_only)


if __name__ == '__main__':
    unittest.main()
