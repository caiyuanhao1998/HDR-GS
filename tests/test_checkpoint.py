import io
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from arguments import OptimizationParams
from utils.graphics_utils import BasicPointCloud


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA and the compiled extensions are required')
class GaussianCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scene.gaussian_model import GaussianModel
        cls.model_type = GaussianModel
        parser = ArgumentParser()
        optimization = OptimizationParams(parser)
        cls.options = optimization.extract(parser.parse_args([]))
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        cls.point_cloud = BasicPointCloud(points, np.full_like(points, 0.5), np.zeros_like(points))

    def make_model(self):
        model = self.model_type(3)
        model.create_from_pcd(self.point_cloud, 1.0, W=16)
        model.training_setup(self.options)
        return model

    def step(self, model):
        loss = sum(parameter.square().sum() for group in model.optimizer.param_groups for parameter in group['params'])
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

    def assert_parameters_equal(self, first, second):
        for source_group, restored_group in zip(first.optimizer.param_groups, second.optimizer.param_groups):
            self.assertEqual(source_group['name'], restored_group['name'])
            self.assertEqual(source_group['lr'], restored_group['lr'])
            for source, restored in zip(source_group['params'], restored_group['params']):
                torch.testing.assert_close(source, restored, rtol=0, atol=0)

    def test_checkpoint_preserves_tone_mapper_optimizer_and_next_step(self):
        source = self.make_model()
        self.step(source)
        source.active_sh_degree = 2
        source.xyz_gradient_accum.fill_(0.25)
        source.denom.fill_(3)
        source.max_radii2D.fill_(7)
        checkpoint = io.BytesIO()
        torch.save((source.capture(), 1000), checkpoint)
        checkpoint.seek(0)
        state, iteration = torch.load(checkpoint, map_location='cuda')
        restored = self.make_model()
        restored.restore(state, self.options)
        self.assertEqual(iteration, 1000)
        self.assertEqual(restored.active_sh_degree, 2)
        self.assert_parameters_equal(source, restored)
        for name in ['xyz_gradient_accum', 'denom', 'max_radii2D']:
            torch.testing.assert_close(getattr(source, name), getattr(restored, name), rtol=0, atol=0)
        self.step(source)
        self.step(restored)
        self.assert_parameters_equal(source, restored)

    def test_legacy_checkpoint_requires_matching_saved_tone_mapper(self):
        source = self.make_model()
        self.step(source)
        legacy_state = source.capture()[:12]
        restored = self.make_model()
        with self.assertRaisesRegex(ValueError, 'Legacy checkpoint.*tone-mapper'):
            restored.restore(legacy_state, self.options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone_mapper.pth'
            source.save_tone_mapper(str(path))
            restored.load_tonemapper(str(path))
        restored.restore(legacy_state, self.options)
        self.assert_parameters_equal(source, restored)

    def test_saved_tone_mapper_keeps_original_file_schema_and_outputs(self):
        source = self.make_model()
        self.step(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone_mapper.pth'
            source.save_tone_mapper(str(path))
            state = torch.load(str(path), map_location='cpu')
            self.assertEqual(set(state), {'tone_mapper_width', 'r_weights', 'g_weights', 'b_weights'})
            restored = self.make_model()
            restored.load_tonemapper(str(path))
            restored.training_setup(self.options)
        inputs = torch.linspace(-2, 2, 9, device='cuda').unsqueeze(1)
        for channel in ['r', 'g', 'b']:
            expected = inputs
            actual = inputs
            for layer in getattr(source, 'tone_mapper_' + channel):
                expected = layer(expected)
            for layer in getattr(restored, 'tone_mapper_' + channel):
                actual = layer(actual)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
