"""Offline adapter/algebra checks; not native DREAMPlace or GPU qualification.

The tiny tensor double executes the actual patched optimizer class. It is not
an alternative placement solver and does not claim Torch/kernel equivalence.
"""
import ast
import contextlib
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_bench.model import load_task
from local_bench.verifier import verify_candidate
from local_tasks.placement_parameters_reference_20260923 import program_adapter as dynamic
from local_tasks.placement_parameters_reference_20260923 import nesterov_runtime as runtime
from local_bench.program_bridge import materialize_program

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"


class Tensor:
    """Minimal float vector surface used by native Nesterov backtracking."""
    dtype = "float64"
    device = "cpu"

    def __init__(self, values):
        self.values = list(values) if isinstance(values, (list, tuple)) else [values]
        self.grad = True

    @property
    def data(self):
        return self

    @property
    def shape(self):
        return (len(self.values),)

    def _binary(self, other, op):
        other = other if isinstance(other, Tensor) else Tensor(other)
        n = max(len(self.values), len(other.values))
        assert len(self.values) in (1, n) and len(other.values) in (1, n)
        return Tensor([op(self.values[i % len(self.values)], other.values[i % len(other.values)]) for i in range(n)])

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other): return Tensor(other) - self
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return Tensor(other) / self
    def __gt__(self, other): return self._binary(other, lambda a, b: a > b)
    def __lt__(self, other): return self._binary(other, lambda a, b: a < b)
    def __eq__(self, other): return self._binary(other, lambda a, b: a == b)
    def __and__(self, other): return self._binary(other, lambda a, b: bool(a and b))
    def __or__(self, other): return self._binary(other, lambda a, b: bool(a or b))
    def __bool__(self):
        assert len(self.values) == 1
        return bool(self.values[0])
    def __getitem__(self, key):
        if isinstance(key, Tensor):
            return Tensor([a for a, selected in zip(self.values, key.values) if selected])
        return Tensor(self.values[key])
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            positions = list(range(len(self.values)))[key]
            values = value.values if isinstance(value, Tensor) else [value] * len(positions)
            assert len(positions) == len(values)
            for i, v in zip(positions, values): self.values[i] = v
        else:
            self.values[key] = value.item() if isinstance(value, Tensor) else value
    def detach(self): return self
    def clone(self): return Tensor(self.values)
    def copy_(self, other): self.values[:] = other.values; return self
    def norm(self, p=2): return Tensor(math.sqrt(sum(v * v for v in self.values)))
    def pow(self, n): return Tensor([v ** n for v in self.values])
    def __pow__(self, n): return self.pow(n)
    def sqrt(self): return Tensor([math.sqrt(v) for v in self.values])
    def dot(self, other): return Tensor(sum(a * b for a, b in zip(self.values, other.values)))
    def sum(self): return Tensor(sum(self.values))
    def all(self): return Tensor(all(self.values))
    def any(self): return Tensor(any(self.values))
    def numel(self): return len(self.values)
    def item(self): assert len(self.values) == 1; return self.values[0]
    def masked_fill_(self, mask, value):
        for i, selected in enumerate(mask.values):
            if selected: self.values[i] = value
        return self


class TorchDouble:
    bool = "bool"
    no_grad = staticmethod(contextlib.nullcontext)
    autograd = SimpleNamespace(Variable=lambda tensor, **kwargs: tensor)
    ones = staticmethod(lambda n, **kwargs: Tensor([1.0] * n))
    zeros = staticmethod(lambda n, **kwargs: Tensor([0.0] * n))
    ones_like = staticmethod(lambda tensor, **kwargs: Tensor([1.0] * tensor.numel()))
    zeros_like = staticmethod(lambda tensor, **kwargs: Tensor([0.0] * tensor.numel()))
    sum = staticmethod(lambda tensor: tensor.sum())
    sqrt = staticmethod(lambda tensor: tensor.sqrt())
    isfinite = staticmethod(lambda tensor: Tensor([math.isfinite(v) for v in tensor.values]))
    cat = staticmethod(lambda tensors: Tensor([v for tensor in tensors for v in tensor.values]))
    equal = staticmethod(lambda a, b: a.values == b.values)


class OptimizerDouble:
    def __init__(self, params, defaults):
        self.param_groups = [dict(defaults, params=list(params))]


def optimizer_class(patched):
    text = (ROOT / dynamic.NESTEROV).read_text()
    if patched:
        text = dynamic._patch_optimizer(text)
    cls = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef))
    namespace = {"torch": TorchDouble, "Optimizer": OptimizerDouble, "required": object()}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "native-nesterov-test", "exec"), namespace)
    return namespace[cls.name]


def quadratic(point):
    # Non-isotropic smooth objective exercises backtracking and acceleration.
    weights = Tensor([1.0, 3.0, 2.0, 4.0])
    grad = point * weights
    return (point * grad).sum() * 0.5, grad


def new_optimizer(patched=True, use_bb=False):
    point = Tensor([2.0, -1.0, 1.5, -0.25])
    return point, optimizer_class(patched)([point], lr=0.01, obj_and_grad_fn=quadratic,
                                           constraint_fn=lambda p: None, use_bb=use_bb)


def test_source_baseline_is_exact_native_noop_and_fixed_recipe():
    spec = load_task(TASKS / "placement_preconditioner")
    result = materialize_program(spec, spec.program_template.read_text(), ROOT)
    assert result["source_patch"] == ""
    recipe = result["candidate_files"]["placement.json"]["concurrent_dreamplace"]
    assert recipe["iterations"] == 4000 and recipe["macro_place_flag"] is True
    assert recipe["learning_rate"] == 0.01 and recipe["num_bins"] == 512 and recipe["target_density"] == 0.6
    assert recipe["die_params"] == {}


def test_source_evolution_does_not_replace_native_optimizer_or_schedule():
    spec = load_task(TASKS / "placement_preconditioner")
    initial = spec.program_template.read_text()
    patch = materialize_program(spec, initial.replace("self.alpha = 1.0", "self.alpha = 1.1"), ROOT)["source_patch"]
    assert "BasicPlace.py" not in patch and "PlaceObj.py" in patch
    assert "NonLinearPlace.py" not in patch and "concurrent_dreamplace.py" not in patch
    assert '"optimizer": "adam"' not in patch and "FixedReferenceLR" not in patch


@pytest.mark.parametrize("directory,kind", [
    ("placement_parameters_reference_20260923", "placement_dynamic_parameters"),
    ("placement_initialization", "placement_initialization_source"),
    ("placement_preconditioner", "placement_preconditioner_source"),
])
def test_versioned_tasks_preserve_kind_and_frozen_profile(directory, kind):
    task = load_task(TASKS / directory)
    assert task.kind == kind
    config = yaml.safe_load(task.base_config.read_text())
    assert config["threads"] == 4 and config["placement"]["seed"] == 1000
    policy = yaml.safe_load((TASKS / directory / "task.yaml").read_text())["candidate_policy"]
    constraints = {item["path"]: item for item in policy["value_constraints"]}
    assert constraints["/candidate_files/placement.json/concurrent_dreamplace/iterations"]["enum"] == [4000]
    assert constraints["/candidate_files/placement.json/concurrent_dreamplace/macro_place_flag"]["enum"] == [True]


def test_initial_controller_is_identity_and_patch_is_active_nesterov_only():
    text = Path(dynamic.__file__).with_name("initial_program.py").read_text()
    namespace = {}
    exec(text, namespace)
    assert namespace["adjust_learning_rate"](*([0] * 9)) == (1.0, (1.0, 1.0))
    result = dynamic.materialize(text, ROOT)
    patch = result["source_patch"]
    assert dynamic.NESTEROV in patch and dynamic.NONLINEAR in patch
    assert dynamic.CONCURRENT not in patch and '"optimizer": "adam"' not in patch
    assert "agentic3d_step_scale(alpha_k, g_k.detach())" in patch
    assert "u_kp1 = v_k - alpha_k*(g_k*agentic3d_scale)" in patch
    assert result["candidate_files"]["placement.json"]["concurrent_dreamplace"]["macro_place_flag"] is True


@pytest.mark.parametrize("name", ["placement_parameters_reference_20260923", "placement_preconditioner"])
def test_materialized_candidate_passes_existing_verifier_and_patch_check(name):
    directory = TASKS / name
    spec = load_task(directory)
    initial = spec.program_template.read_text()
    if name == "placement_preconditioner":
        initial = initial.replace("self.alpha = 1.0", "self.alpha = 1.1")
    candidate = materialize_program(spec, initial, ROOT)
    verify_candidate(load_task(directory), candidate)
    subprocess.run(["git", "apply", "--check", "--whitespace=error-all", "-"],
                   input=candidate["source_patch"], text=True, cwd=ROOT,
                   check=True, capture_output=True)


def test_identity_hook_matches_native_all_optimizer_state_over_several_steps():
    point, opt = new_optimizer(True)
    native_point, native = new_optimizer(False)
    calls = []
    opt.agentic3d_step_scale = lambda step, grad: calls.append((step.item(), grad.values[:]))
    for _ in range(4):
        opt.step()
        native.step()
        assert point.values == native_point.values
        for key in ("u_k", "v_k", "g_k", "a_k", "alpha_k", "v_k_1", "g_k_1", "obj_k", "obj_k_1", "v_kp1"):
            assert [v.values for v in opt.param_groups[0][key]] == [v.values for v in native.param_groups[0][key]]
        assert opt.param_groups[0]["obj_eval_count"] == native.param_groups[0]["obj_eval_count"]
    assert len(calls) == 4  # An ignored callback cannot pass this test.


def test_nontrivial_scale_updates_major_reference_and_previous_state_consistently():
    point, opt = new_optimizer()
    scales = Tensor([8.0, 12.5, 8.0, 12.5])
    captures = []
    trials = []
    def constraint(trial):
        trials.append((opt.param_groups[0]["alpha_k"][0].item(), trial.values[:]))
    opt.constraint_fn = constraint
    def hook(step, grad):
        group = opt.param_groups[0]
        captures.append((point.values[:], group["u_k"][0].values[:], group["a_k"][0].item(), step.item(), grad.values[:]))
        return scales
    opt.agentic3d_step_scale = hook
    for _ in range(4):
        before_trials = len(trials)
        opt.step()
        old_v, old_u, old_a, initial_alpha, gradient = captures[-1]
        accepted_alpha, accepted_trial = trials[-1]
        expected_u = [v - accepted_alpha * g * s for v, g, s in zip(old_v, gradient, scales.values)]
        next_a = (1.0 + math.sqrt(4.0 * old_a * old_a + 1.0)) / 2.0
        coef = (old_a - 1.0) / next_a
        expected_v = [u + coef * (u - previous) for u, previous in zip(expected_u, old_u)]
        group = opt.param_groups[0]
        assert group["u_k"][0].values == pytest.approx(expected_u)
        assert point.values == pytest.approx(expected_v)
        assert point.values == accepted_trial
        assert group["v_k_1"][0].values == old_v
        actual_gradient = quadratic(point)[1].values
        assert group["g_k"][0].values == pytest.approx(actual_gradient)
        assert group["g_k_1"][0].values == gradient
        next_alpha = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected_v, old_v))
                               / sum((a - b) ** 2 for a, b in zip(actual_gradient, gradient)))
        assert group["alpha_k"][0].item() == pytest.approx(next_alpha)
        assert group["obj_k"][0].item() == pytest.approx(quadratic(point)[0].item())
        assert group["a_k"][0].item() == next_a
        assert 1 <= len(trials) - before_trials <= 10  # Native backtrack budget retained.
    assert len(captures) == 4
    assert len(trials) > len(captures)  # Multiple trials still call controller once.
    assert opt.param_groups[0]["obj_eval_count"] == len(trials)


def test_bb_and_invalid_coordinate_scales_fail_closed():
    _, opt = new_optimizer(use_bb=True)
    opt.agentic3d_step_scale = lambda step, grad: None
    with pytest.raises(ValueError, match="no-BB backtracking"):
        opt.step()
    for scales in (Tensor([1]), Tensor([1, 1, 0, 1]), Tensor([1, 1, math.nan, 1])):
        _, opt = new_optimizer()
        opt.agentic3d_step_scale = lambda step, grad: scales
        with pytest.raises(ValueError, match="Invalid Nesterov"):
            opt.step()


@pytest.mark.parametrize("decision", [(0, (1, 1)), (math.inf, (1, 1)), (1, (0, 1)), (1, (1, math.nan)), (1, (1,))])
def test_invalid_controller_decisions_rejected(decision):
    with pytest.raises(ValueError):
        runtime._agentic3d_project_step_control(decision, (1, 1))


def test_relative_die_projection_and_identity():
    assert runtime._agentic3d_project_step_control((1, (1, 1)), (1, 1)) == (1, (1, 1))
    shared, die = runtime._agentic3d_project_step_control((2, (100, 0.01)), (1, 1))
    assert shared == 2 and die[0] * die[1] == pytest.approx(1)
    assert die[0] == pytest.approx(2 ** 0.2) and die[1] == pytest.approx(2 ** -0.2)


def test_runtime_invokes_controller_and_preserves_fixed_coordinates(monkeypatch):
    monkeypatch.setattr(runtime, "torch", TorchDouble, raising=False)
    logs = []
    monkeypatch.setattr(runtime, "logging", SimpleNamespace(info=lambda *args: logs.append(args)), raising=False)
    # Three physical nodes: bottom movable, top movable, fixed. Two fillers.
    placedb = SimpleNamespace(regions=[[], []], concurrent_lol_region_names=["bottom", "top"],
                              num_nodes=5, num_movable_nodes=2, num_physical_nodes=3,
                              num_filler_nodes=2, filler_start_map=[0, 1, 2])
    data = SimpleNamespace(node2fence_region_map=Tensor([0, 1, -1]))
    model = SimpleNamespace(data_collections=data, density_weight=Tensor([0.1, 0.2]))
    metric = SimpleNamespace(hpwl=Tensor(100), overflow=Tensor([0.3, 0.4]))
    pos = Tensor([0] * 10)
    gradient = Tensor([1] * 10)
    state, calls = {}, []
    def identity(*args): calls.append(args); return 1, (1, 1)
    assert runtime._agentic3d_nesterov_step_control(identity, placedb, model, pos, metric, 0, state, Tensor(0.2), gradient) is None
    assert len(calls) == 1 and state["calls"] == 1 and len(logs) == 1
    scale = runtime._agentic3d_nesterov_step_control(lambda *args: (2, (1, 1)), placedb, model, pos, metric, 1, state, Tensor(0.2), gradient)
    assert scale.values == [2, 2, 1, 2, 2, 2, 2, 1, 2, 2]
    assert pos.values == [0] * 10 and gradient.values == [1] * 10
    assert state["calls"] == 2


def test_candidate_program_cannot_use_external_effects_or_old_absolute_lr_signature():
    initial = Path(dynamic.__file__).with_name("initial_program.py").read_text()
    with pytest.raises(ValueError, match="signature"):
        dynamic.materialize(initial.replace("native_step_size", "init_learning_rate"), ROOT)
    with pytest.raises(ValueError):
        dynamic.materialize(initial.replace("return 1.0, (1.0, 1.0)", 'return open("secret"), (1, 1)'), ROOT)


def test_old_adam_tasks_are_removed():
    for directory in ("placement_parameters", "placement_init_preconditioner"):
        with pytest.raises(FileNotFoundError):
            load_task(TASKS / directory)
