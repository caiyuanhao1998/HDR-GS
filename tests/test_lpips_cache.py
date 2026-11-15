import unittest
from unittest.mock import patch

import torch
import lpipsPyTorch


class ExamplePerceptualMetric(torch.nn.Module):
    def __init__(self, net_type, version):
        super().__init__()

    def forward(self, first, second):
        return (first - second).square().mean()


class PerceptualMetricReuseTests(unittest.TestCase):
    def setUp(self):
        lpipsPyTorch._get_criterion.cache_clear()
        self.constructor = patch.object(lpipsPyTorch, 'LPIPS', side_effect=ExamplePerceptualMetric)
        self.factory = self.constructor.start()
        self.addCleanup(self.constructor.stop)
        self.addCleanup(lpipsPyTorch._get_criterion.cache_clear)

    def test_reuses_network_but_evaluates_each_image_pair(self):
        reference = torch.zeros(1, 3, 8, 8)
        first = lpipsPyTorch.lpips(torch.ones_like(reference), reference)
        second = lpipsPyTorch.lpips(torch.full_like(reference, 2), reference)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(first.item(), 1)
        self.assertEqual(second.item(), 4)

    def test_reused_network_preserves_input_gradients(self):
        reference = torch.zeros(1, 3, 8, 8)
        for _ in range(2):
            image = torch.ones_like(reference, requires_grad=True)
            lpipsPyTorch.lpips(image, reference).backward()
            torch.testing.assert_close(image.grad, torch.full_like(image, 2 / image.numel()))
        self.assertEqual(self.factory.call_count, 1)

    def test_network_variants_have_separate_weights(self):
        image = torch.ones(1, 3, 8, 8)
        lpipsPyTorch.lpips(image, image, net_type='alex')
        lpipsPyTorch.lpips(image, image, net_type='vgg')
        lpipsPyTorch.lpips(image, image, net_type='squeeze')
        self.assertEqual(self.factory.call_count, 3)


if __name__ == '__main__':
    unittest.main()
