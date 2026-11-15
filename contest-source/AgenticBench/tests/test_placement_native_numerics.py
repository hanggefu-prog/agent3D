"""Small numerical execution checks; run directly in the maintained Docker image."""
import ast
import logging
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from local_bench.numerical_programs import lower_torch_compat
from local_bench.program_safety import normalized_program
from local_tasks.tp_gnn_loss_patch import SAFE_TORCH_APIS, SAFE_FUNCTIONAL_APIS
from local_tasks import tp_gnn_timing_context as adapter

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(torch is None, 'Torch execution is checked in the maintained Docker image')
class NativeNumerics(unittest.TestCase):
    def compatibility(self):
        namespace = {'torch': torch}
        source = 'def clean(x, **options):\n    return torch.nan_to_num(x, **options)\n'
        exec(compile(lower_torch_compat(source), '<compatibility fixture>', 'exec'), namespace)
        return namespace['clean']

    def test_finite_values_replacements_dtype_and_gradient(self):
        had_api = hasattr(torch, 'nan_to_num')
        clean = self.compatibility()
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([float('nan'), float('inf'), -float('inf'), 2., -3.], dtype=dtype, requires_grad=True)
            y = clean(x, nan=0.5, posinf=4., neginf=-4.)
            self.assertTrue(torch.equal(y, torch.tensor([.5, 4., -4., 2., -3.], dtype=dtype)))
            self.assertEqual(y.dtype, x.dtype)
            self.assertEqual(y.device, x.device)
            y.sum().backward()
            self.assertTrue(torch.equal(x.grad, torch.tensor([0., 0., 0., 1., 1.], dtype=dtype)))
            default = clean(x.detach())
            self.assertEqual(default[1].item(), torch.finfo(dtype).max)
            self.assertEqual(default[2].item(), torch.finfo(dtype).min)
        self.assertEqual(hasattr(torch, 'nan_to_num'), had_api)
        integers = torch.tensor([1, 2], dtype=torch.int64)
        self.assertTrue(torch.equal(clean(integers), integers))
        self.assertEqual(clean(torch.empty(0)).numel(), 0)
        out = torch.empty(2)
        self.assertIs(clean(torch.tensor([float('nan'), 2.]), out=out), out)
        self.assertTrue(torch.equal(out, torch.tensor([0., 2.])))

    def test_failed_real_preconditioner_runs_with_masks_and_unchanged_inputs(self):
        path = ROOT / 'AgenticBench/releases/local-matrix-evolution-20261018/candidates/placement_preconditioner/round-01.py'
        namespace = {'torch': torch, 'logging': logging}
        exec(compile(lower_torch_compat(normalized_program(path.read_text())), str(path), 'exec'), namespace)
        placedb = SimpleNamespace(regions=[0, 1], num_nodes=4, num_movable_nodes=2,
                                  num_filler_nodes=1, filler_start_map=[0, 0, 1])
        data = SimpleNamespace(node2fence_region_map=torch.tensor([0, 1, 2, 1]),
                               pos=[torch.zeros(8)], net_weights=torch.ones(2), node_areas=torch.ones(4))
        original_areas = data.node_areas.clone()
        op = namespace['PreconditionOp'](placedb, data, SimpleNamespace(pws_op=lambda weights: torch.ones(4)))
        for density in (torch.tensor([1., 1.]), torch.tensor([2., 4.])):
            before = density.clone()
            grad = torch.ones(8)
            # The fixed physical node already has zero objective gradient.
            grad[2] = grad[6] = 0
            result = op(grad, density, update_mask=torch.tensor([True, False]),
                        fix_nodes_mask=torch.tensor([True, False]))
            self.assertEqual(result.data_ptr(), grad.data_ptr())
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertTrue(torch.equal(result, torch.zeros_like(result)))
            self.assertTrue(torch.equal(density, before))
        self.assertTrue(torch.equal(data.node_areas, original_areas))
        self.assertTrue(torch.equal(data.net_weights, torch.ones(2)))

    def module_and_args(self, program):
        source = (ROOT / adapter.TP_GNN).read_text()
        patched = adapter._patch_tp_gnn(source, normalized_program(program))
        name = '_tp_numerical_fixture'
        module = types.ModuleType(name)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(patched, str(ROOT / adapter.TP_GNN), 'exec'), module.__dict__)
        n = 12
        features = [[float((i * (j + 1)) % 7) for j in range(10)] for i in range(n)]
        neighbors = [{(i - 1) % n, (i + 1) % n} for i in range(n)]
        config = module.TPGNNConfig(device='cpu', embedding_dim=8, epochs=2, min_epochs=2,
                                   batch_size=6, random_walks=1, walk_length=1, negative_samples=2,
                                   feature_regime='partition-last')
        context = adapter._agentic_loss_graph([[str(i)] for i in range(n)],
                    [[str(i), str((i+1) % n)] for i in range(n)],
                    [1+i/n for i in range(n)], [1]*n, 'partition-last')
        return module, [features, [0]*n, 0, neighbors, config, context]

    def test_new_loss_helpers_execute_inside_native_training(self):
        program = (ROOT / 'AgenticBench/local_tasks/tp_gnn_post_placement_timing/initial_program.py').read_text()
        program = program.replace('    return -(', '''    def clean(values):
        return torch.nan_to_num(values, nan=0.0)
    positive_scores = clean(positive_scores)
    weight = 1.0
    for term in range(3):
        weight = weight + 0.001 * term
    positive_scores = positive_scores * weight
    return -(''')
        module, args = self.module_and_args(program)
        embeddings, report = module.train_random_walk_embeddings(*args)
        self.assertEqual(len(embeddings), 12)
        self.assertTrue(all(math.isfinite(value) for row in embeddings for value in row))

    def test_fully_detached_loss_is_still_rejected_by_native_training(self):
        signature = ast.parse(normalized_program((ROOT / 'AgenticBench/local_tasks/tp_gnn_post_placement_timing/initial_program.py').read_text())).body[0]
        program = 'def compute_tp_gnn_loss(' + ', '.join(arg.arg for arg in signature.args.args) + '):\n    return positive_scores.detach().sum()\n'
        module, args = self.module_and_args(program)
        with self.assertRaisesRegex(ValueError, 'detached the training objective'):
            module.train_random_walk_embeddings(*args)

    def test_documented_module_apis_exist_in_evaluation_runtime(self):
        import torch.nn.functional as functional
        self.assertFalse([name for name in SAFE_TORCH_APIS - {'nan_to_num'} if not hasattr(torch, name)])
        self.assertFalse([name for name in SAFE_FUNCTIONAL_APIS if not hasattr(functional, name)])


if __name__ == '__main__':
    unittest.main()
