"""Wider algorithm expression with fixed task interfaces and physical contracts."""
import ast
import math
from pathlib import Path

import pytest

from local_bench.hbt_policy import MATH_ARITY, validate_policy
from local_bench.model import load_task
from local_bench.numerical_programs import lower_torch_compat, validate_helpers
from local_bench.program_bridge import materialize_program
from local_bench.verifier import verify_candidate
from src.hbt.policy import ScalarPolicy, PolicyError
from src.hbt.policy_grammar import NET_FIELDS, MATH_ARITY as NATIVE_MATH_ARITY

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / 'AgenticBench/local_tasks'


def materialize(name, rewrite):
    spec = load_task(TASKS / name)
    program = rewrite(spec.program_template.read_text())
    candidate = materialize_program(spec, program, ROOT)
    verify_candidate(spec, candidate)
    return candidate


def test_initializer_helper_stays_in_its_component():
    candidate = materialize('placement_initialization', lambda source: source.replace(
        '    return init_pos', '    def copy_coordinates(values):\n        return values.copy()\n    return copy_coordinates(init_pos)'))
    assert 'def copy_coordinates' in candidate['source_patch']
    assert 'PlaceObj.py' not in candidate['source_patch']


def test_preconditioner_additional_method_and_torch_compatibility():
    def rewrite(source):
        source = source.replace('    def set_overflow(', '    def finite_scale(self, value):\n        return torch.nan_to_num(value, nan=1.0)\n\n    def set_overflow(')
        return source.replace('        with torch.no_grad():', '        with torch.no_grad():\n            grad.copy_(self.finite_scale(grad))')
    candidate = materialize('placement_preconditioner', rewrite)
    assert '_agentic3d_nan_to_num' in candidate['source_patch']
    assert 'BasicPlace.py' not in candidate['source_patch']


def test_real_failed_preconditioner_now_materializes_compatibility():
    spec = load_task(TASKS / 'placement_preconditioner')
    old = ROOT / 'AgenticBench/releases/local-matrix-evolution-20261018/candidates/placement_preconditioner/round-01.py'
    candidate = materialize_program(spec, old.read_text(), ROOT)
    verify_candidate(spec, candidate)
    assert '+def _agentic3d_nan_to_num' in candidate['source_patch']
    assert '+            weight_ratio = _agentic3d_nan_to_num(' in candidate['source_patch']


def test_nesterov_can_factor_scalar_calculations():
    candidate = materialize('placement_parameters_reference_20260923', lambda source: source.replace(
        '    return ', '    def bounded(value):\n        return min(2.0, max(0.5, value))\n    scale = bounded(1.05)\n    return ', 1))
    assert 'def bounded' in candidate['source_patch']


def test_tp_helpers_reductions_and_intermediate_detach():
    candidate = materialize('tp_gnn_post_placement_timing', lambda source: source.replace('    return -(', '''    def centered(values):
        return values - values.mean()
    weights = torch.nan_to_num(graph_weights.detach(), nan=1.0)
    spread = torch.amax(centered(embeddings))
    scale = 1.0
    for term in range(3):
        scale = scale + 0.001 * term
    return -('''))
    assert '_agentic3d_nan_to_num' in candidate['source_patch']
    assert 'if not loss.requires_grad' in candidate['source_patch']


@pytest.mark.parametrize('expression', ['optimizer', 'config', 'model'])
def test_tp_helper_parameters_do_not_expose_enclosing_training_scope(expression):
    with pytest.raises(ValueError, match='declared inputs'):
        materialize('tp_gnn_post_placement_timing', lambda source: source.replace('    return -(',
            '    def identity(' + expression + '):\n        return ' + expression + '\n    forbidden = ' + expression + '\n    return -('))


@pytest.mark.parametrize('body', [
    'def f(x):\n    return f(x)\nreturn f(1)',
    'def f(x):\n    return g(x)\ndef g(x):\n    return f(x)\nreturn f(1)',
    'def f(x):\n    return x\nalias = f\nreturn alias(1)',
])
def test_helper_recursion_and_aliases_stay_rejected(body):
    function = ast.parse('def run():\n' + '\n'.join('    ' + line for line in body.splitlines())).body[0]
    with pytest.raises(ValueError, match='Recursive|alias'):
        validate_helpers(function)


@pytest.mark.parametrize('body', [
    '    for term in range(source_batch_size):\n        value = term\n',
    '    value = torch.nan_to_num(graph_weights, out=graph_weights)\n',
    '    def load_file(x):\n        return open(x).read()\n    value = load_file("data")\n',
])
def test_tp_relaxation_keeps_budget_and_input_protections(body):
    with pytest.raises(ValueError):
        materialize('tp_gnn_post_placement_timing', lambda source: source.replace('    return -(', body + '    return -('))


def test_compatibility_lowering_is_conditional_and_utf8_correct():
    untouched = 'def f(x):\n    return torch.sum(x)\n'
    assert lower_torch_compat(untouched) == untouched
    source = 'def f(x):\n    "数学"; return torch.nan_to_num(x, nan=2.0)\n'
    lowered = lower_torch_compat(source)
    assert '"数学"; return _agentic3d_nan_to_num(x, nan=2.0)' in lowered
    compile(lowered, '<compat>', 'exec')


@pytest.mark.parametrize('name', sorted(MATH_ARITY))
def test_hbt_extended_math_matches_native_interpreter(name):
    assert MATH_ARITY == NATIVE_MATH_ARITY
    args = [0.5] * MATH_ARITY[name]
    expression = 'math.' + name + '(' + ', '.join(map(str, args)) + ')'
    source = 'def net_priority(context):\n    return ' + expression + '\ndef site_cost(net_context, site_context):\n    return 0\n'
    source = validate_policy(source)
    interpreter = ScalarPolicy({'interface_version': 'hbt-policy-v1', 'source': source})
    actual = interpreter.call('net_priority', context={name: 1. for name in NET_FIELDS})
    assert actual == pytest.approx(getattr(math, name)(*args))


def test_hbt_bad_math_still_fails_without_fallback():
    source = 'def net_priority(context):\n    return math.sqrt(-1)\ndef site_cost(net_context, site_context):\n    return 0\n'
    interpreter = ScalarPolicy({'interface_version': 'hbt-policy-v1', 'source': source})
    with pytest.raises(PolicyError):
        interpreter.call('net_priority', context={name: 1. for name in NET_FIELDS})
    with pytest.raises(ValueError):
        validate_policy(source.replace('math.sqrt(-1)', 'math.__dict__'))


@pytest.mark.parametrize('body', [
    '    for i in range(8):\n        for j in range(8):\n            for k in range(8):\n                value = k\n',
    '    for i in range(1, 8, 0):\n        value = i\n',
    '    def terms(x):\n        for i in range(8):\n            for j in range(8):\n                x = x + i + j\n        return x\n    for k in range(8):\n        value = terms(k)\n',
])
def test_tp_loop_budget_includes_nested_helper_work(body):
    with pytest.raises(ValueError, match='iterations|range step'):
        materialize('tp_gnn_post_placement_timing', lambda source: source.replace('    return -(', body + '    return -('))


def test_candidate_cannot_shadow_lowered_compatibility_function():
    with pytest.raises(ValueError, match='reserved'):
        lower_torch_compat('def f(x):\n    _agentic3d_nan_to_num = 1\n    return torch.nan_to_num(x)\n')
