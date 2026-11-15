"""Behavioral tests for the probability baseline's default loss."""
from pathlib import Path
import os
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch runtime is required')
class BinaryBaselineLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get('BINARY_LOSS_TEST_PROGRAM')
        path = Path(configured) if configured else (
            Path(__file__).resolve().parents[1] / 'local_tasks/tp_gnn_binary_loss_timing/initial_program.py')
        namespace = {'torch': torch, 'F': F}
        exec(compile(path.read_text(), str(path), 'exec'), namespace)
        cls.loss_function = staticmethod(namespace['compute_binary_partition_loss'])

    def evaluate(self, p, epoch=400, binary_weight=10.0, areas=None):
        features = torch.zeros((len(p), 10), dtype=p.dtype)
        features[:, 0] = torch.linspace(-3, 1, len(p), dtype=p.dtype)
        features[:, 9] = torch.arange(1, len(p)+1, dtype=p.dtype).square()
        areas = torch.ones_like(p) if areas is None else areas
        return self.loss_function(p, p.sum()*0, torch.zeros(3), torch.ones(3),
            features, areas, .5, 1., 200., binary_weight, epoch, 400)

    def test_active_margin_pushes_near_half_probabilities_apart(self):
        p = torch.tensor([.499, .501, .499, .501, .499, .501], requires_grad=True)
        value = self.evaluate(p)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((p.grad[::2] > 0).all())
        self.assertTrue((p.grad[1::2] < 0).all())

    def test_final_objective_prefers_confident_balanced_partition(self):
        uncertain = self.evaluate(torch.full((6,), .5))
        confident = self.evaluate(torch.tensor([.01,.99,.01,.99,.01,.99]))
        self.assertLess(float(confident), float(uncertain))

    def test_soft_balance_uses_expected_area(self):
        fractional = torch.tensor([.6,.6,.6,.6,.6,0.])
        balanced = torch.tensor([.01,.99,.01,.99,.01,.99])
        self.assertAlmostEqual(float(fractional.mean()), .5, places=6)
        self.assertAlmostEqual(float(self.evaluate(fractional,binary_weight=0.)),
                               float(self.evaluate(balanced,binary_weight=0.)), places=10)

    def test_fixed_weights_make_loss_independent_of_features_and_epoch(self):
        p = torch.tensor([.2,.8,.3,.7,.1,.9], requires_grad=True)
        original = self.evaluate(p)
        features = torch.full((len(p), 10), 17.)
        alternative = self.loss_function(p, p.sum()*0, torch.zeros(3), torch.ones(3),
            features, torch.ones_like(p), .5, 1., 200., 10., 10, 400)
        self.assertEqual(float(original), float(alternative))
        alternative.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_constant_features_remain_finite_at_uniform_probabilities(self):
        p = torch.full((6,), .5, requires_grad=True)
        value = self.loss_function(p, p.sum()*0, torch.zeros(3), torch.ones(3),
            torch.zeros((6,10)), torch.ones_like(p), .5, 1., 200., 0., 1, 400)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_endpoint_probabilities_have_finite_loss_and_gradient(self):
        p = torch.tensor([0.,1.,0.,1.,0.,1.], requires_grad=True)
        value = self.evaluate(p, epoch=10)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(p.grad).all())

if __name__ == '__main__':
    unittest.main()
