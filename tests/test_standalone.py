"""CPU integration checks for the extracted repository; no data or GPU needed."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import GMM
import MNIST_utils as MNIST
from RenyiSampler import image_ratio_gradient, mobility


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20260912)

    def test_gmm_geometry_and_score(self):
        GMM.validate_gmm()

    def test_mnist_prior_and_replay(self):
        weights = MNIST.stacked_weights()
        self.assertAlmostEqual(weights.sum().item(), 1, places=6)
        self.assertAlmostEqual(
            weights[torch.arange(1000) % 10 == 7].sum().item(), 0.01, places=6
        )
        first = MNIST.make_composition(10000, np.random.RandomState(20260912))
        second = MNIST.make_composition(10000, np.random.RandomState(20260912))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        digits, modes = first
        torch.testing.assert_close(
            modes, digits[:, 0] * 100 + digits[:, 1] * 10 + digits[:, 2]
        )

    def test_mnist_model_and_ratio_gradient(self):
        x = torch.randn(2, 3, 28, 28)
        with torch.no_grad():
            output = MNIST.UNet()(x, 3)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        model = MNIST.RatioCNN().double()
        x = x.double()
        h, gradient = image_ratio_gradient(model, x, 3)
        direction = torch.randn_like(x)
        direction /= direction.norm()
        with torch.no_grad():
            finite_difference = (
                model(x + 0.01 * direction, 3).sum()
                - model(x - 0.01 * direction, 3).sum()
            ) / 0.02
        torch.testing.assert_close(
            finite_difference, (gradient * direction).sum(), rtol=0.03, atol=1e-5
        )
        a, da, _ = mobility(h, gradient, 1.0, 1)
        torch.testing.assert_close(a, torch.ones_like(a), rtol=0, atol=0)
        torch.testing.assert_close(da, torch.zeros_like(da), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
