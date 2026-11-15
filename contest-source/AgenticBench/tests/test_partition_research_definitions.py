"""Definition/interface checks, never EDA or native reference qualification."""
import ast
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from local_bench.cli import main
from local_bench.evaluation import evaluate_task
from local_bench.model import EvaluationResult, load_candidate, load_task, require_executable_task
from local_bench.optimization_feedback import optimization_result
from local_bench.program_bridge import materialize_program
from local_bench.prompting import export_openevolve_bundle
from local_bench.search_api import ProgramEvaluationHarness, TaskEvaluationOptions
from local_bench.search_session import prepare_search_problem
from local_bench.task_inventory import CORE_TASK_DIRECTORIES, EXECUTABLE_TASK_STAGES
from local_bench.verifier import verify_candidate
from local_tasks import tp_gnn_local_task_definitions as definitions

ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "AgenticBench/local_tasks"
TASKS = sorted(definitions.TASKS)


def binary_program(expression):
    return "def compute_binary_partition_loss(" + ", ".join(definitions.BINARY_ARGS) + "):\n    return " + expression + "\n"


def decoder_program(body):
    initial = (TASK_ROOT / definitions.DECODER_TASK / "initial_program.py").read_text()
    source = definitions.validate_program(initial, definitions.DECODER_TASK)
    tree = ast.parse(source)
    loss = ast.get_source_segment(source, tree.body[0])
    return loss + "\n\ndef decode_tp_gnn_embeddings(" + ", ".join(definitions.DECODER_ARGS) + "):\n" + body


@pytest.mark.parametrize("name", TASKS)
def test_new_definitions_are_complete_visible_and_canonical(name, capsys):
    spec = load_task(TASK_ROOT / name)
    assert spec.task_id == spec.kind == name
    assert spec.availability == "available"
    assert name in CORE_TASK_DIRECTORIES and EXECUTABLE_TASK_STAGES[name] == "partition"
    assert len(EXECUTABLE_TASK_STAGES) == 10
    require_executable_task(spec)
    assert spec.start_stage == "partition" and spec.stop_stage == "opt"
    assert spec.upstream_stages == ["input", "floorplan"]
    assert spec.allowed_config_paths == spec.allowed_source_paths == []
    assert spec.max_changed_files == spec.max_added_lines == spec.max_deleted_lines == 0
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    assert candidate == load_candidate(spec.initial_candidate)
    verify_candidate(spec, candidate)
    definition = yaml.safe_load((spec.root / "task.yaml").read_text())["definition"]
    assert definition["objective"]["metric"] == "combined_score"
    assert definition["endpoint"]["opt_passes"] == 1
    assert definition["endpoint"]["directory"] == "60_opt"
    assert "0.6" in definition["objective"]["formula"] and "0.4" in definition["objective"]["formula"]
    assert definition["admission"]["trusted_context_producer"] == "source_connected"
    assert definition["admission"]["trusted_physical_collector"] == "source_connected"
    assert definition["admission"]["same_contract_native_reference"] == "pending_per_case_fresh_qualification"
    assert definition["admission"]["reference_task_id"] == name
    assert definition["admission"]["missing_reference"] == "only_exact_initial_program_may_create_fresh_reference"
    original = load_task(TASK_ROOT / "tp_gnn_post_placement_timing")
    assert spec.frozen_inputs == original.frozen_inputs
    config = yaml.safe_load(spec.base_config.read_text())
    parent_config = yaml.safe_load(original.base_config.read_text())
    if name == definitions.BINARY_TASK:
        assert config["partition"]["config"] == "config/partition/tp_gnn_binary_regularized.json"
        config["partition"]["config"] = parent_config["partition"]["config"]
    # Public matrices override the per-case/seed halo through the shared policy.
    config["placement"].pop("macro_halo_um")
    parent_config["placement"].pop("macro_halo_um")
    assert config == parent_config
    assert main(["list", "--core"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 10
    assert next(row for row in rows if row["id"] == name)["availability"] == "available"


@pytest.mark.parametrize("name", TASKS)
def test_definition_gates_run_before_physical_or_model_dispatch(name, tmp_path):
    # Preserve the generic definition-only guard with a private task fixture.
    # The shipped tasks are now source-connected and must remain available.
    framework = tmp_path / "framework"
    task = framework / "local_tasks" / name
    task.parent.mkdir(parents=True)
    shutil.copytree(TASK_ROOT / name, task)
    definition = yaml.safe_load((task / "task.yaml").read_text())
    definition["availability"] = "definition-only"
    (task / "task.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    spec = load_task(task)
    with pytest.raises(ValueError, match="definition-only"):
        require_executable_task(spec)
    with pytest.raises(ValueError, match="definition-only"):
        evaluate_task(spec.root, spec.initial_candidate, ROOT, tmp_path / "physical", dry_run=True)
    with pytest.raises(ValueError, match="definition-only"):
        export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT, tmp_path / "export")
    for mode in ("full", "interface"):
        options = TaskEvaluationOptions(spec.root, tmp_path / mode, evaluation_mode=mode, open3dbench_root=ROOT)
        with pytest.raises(ValueError, match="definition-only"):
            ProgramEvaluationHarness(options, evaluation_transport=lambda **kwargs: pytest.fail("EDA called"))
    from scripts import run_local_task_smoke as smoke
    with pytest.raises(ValueError, match="definition-only"):
        smoke.run_smoke(SimpleNamespace(framework=framework), task=name,
            output=tmp_path / "smoke", rounds=2,
            provider_factory=lambda **kwargs: pytest.fail("Provider called"),
            transport_factory=lambda **kwargs: pytest.fail("EDA called"))
    assert {path.name for path in tmp_path.iterdir()} == {"framework"}


@pytest.mark.parametrize("name", TASKS)
def test_model_context_and_offline_verify_are_explicitly_nonphysical(name, tmp_path):
    spec = load_task(TASK_ROOT / name)
    options = TaskEvaluationOptions(spec.root, tmp_path / "verify", evaluation_mode="verify", open3dbench_root=ROOT)
    problem = prepare_search_problem(options)
    assert problem.metadata["availability"] == "available"
    assert "0.6" in problem.description and "0.4" in problem.description
    assert str(ROOT) not in problem.description
    harness = ProgramEvaluationHarness(options, evaluation_transport=lambda **kwargs: pytest.fail("EDA called"))
    harness.evaluate_program(spec.program_template.read_text())
    assert harness.records[-1]["status"] == "accepted"
    result = optimization_result(EvaluationResult(metrics=harness.records[-1]["metrics"], artifacts={}), task_id=name)
    assert result.metrics["verifier_only"] == 1
    assert "synthetic protocol values" in result.artifacts["optimization_feedback"]
    assert "No partition training" in result.artifacts["optimization_feedback"]


@pytest.mark.parametrize("name", TASKS)
def test_source_connected_templates_export_without_claiming_native_qualification(name, tmp_path):
    spec = load_task(TASK_ROOT / name)
    bundle = export_openevolve_bundle(spec, load_candidate(spec.initial_candidate), ROOT, tmp_path / "export")
    config = yaml.safe_load(Path(bundle["config"]).read_text())
    system = config["prompt"]["system_message"]
    assert "definition-only" not in system
    assert "same-case reference" in system
    assert "initial program" in system
    assert "does not authorize model calls" in system
    assert "source-connected" in optimization_result(
        EvaluationResult(metrics={"combined_score": 1., "runs_successfully": 1.}, artifacts={}),
        task_id=name).artifacts["optimization_feedback"].lower()


@pytest.mark.parametrize("expression", [
    "torch.exp(probabilities, out=probabilities)",
    'torch.exp(probabilities, **{"out": probabilities})',
    "F.relu(probabilities, True)",
    "F.normalize(probabilities, 2, 0, 1e-12, probabilities)",
    "probabilities.detach().sum()",
    "torch.amax(probabilities)",
    "F.l1_loss(probabilities, probabilities)",
    "config.seed * probabilities.sum()",
    "compute_binary_partition_loss(probabilities)",
    "open('result.txt').read()",
])
def test_binary_loss_rejects_effects_outer_state_and_recursion(expression):
    with pytest.raises(ValueError):
        definitions.validate_program(binary_program(expression), definitions.BINARY_TASK)


@pytest.mark.parametrize("body", [
    "    return np.clip(embeddings, -1, 1, out=embeddings)\n",
    "    return np.sum(embeddings, 1, np.float64, node_areas)\n",
    "    return node_areas.cumsum(0, np.float64, node_areas)\n",
    '    return np.clip(embeddings, -1, 1, **{"out": embeddings})\n',
    "    return np.random.randint(0, 2, len(node_areas))\n",
    "    return reference_global_two_way(embeddings, node_areas, seed+1, max_iterations, balance_target)\n",
    "    return reference_global_two_way(embeddings, node_areas, seed, max_iterations*50, balance_target)\n",
    "    for iteration in range(max_iterations):\n        labels = reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)\n    return labels\n",
    "    labels = reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)\n    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)\n",
    "    for iteration in range(500):\n        labels = node_areas\n    return labels\n",
    "    node_areas[0] = 0\n    return node_areas\n",
    "    seed = 2\n    return node_areas\n",
])
def test_decoder_rejects_mutation_rng_and_budget_bypasses(body):
    with pytest.raises(ValueError):
        definitions.validate_program(decoder_program(body), definitions.DECODER_TASK)


@pytest.mark.parametrize("name", TASKS)
def test_literal_candidates_cannot_bypass_program_validation(name):
    spec = load_task(TASK_ROOT / name)
    initial = load_candidate(spec.initial_candidate)
    filename = definitions.PACKAGES[name][0]
    malicious = copy.deepcopy(initial)
    malicious["candidate_files"][filename]["source"] += "\nimport os\n"
    with pytest.raises(ValueError):
        verify_candidate(spec, malicious)
    malicious = copy.deepcopy(initial)
    malicious["config_overrides"] = {"partition": {"seed": 999}}
    with pytest.raises(ValueError):
        verify_candidate(spec, malicious)
    malicious = copy.deepcopy(initial)
    malicious["candidate_files"][filename]["interface_version"] = "another-task"
    with pytest.raises(ValueError):
        verify_candidate(spec, malicious)


def test_binary_loss_supports_nonlinear_per_net_penalties():
    nonlinear = binary_program("(net_weights * net_cut_probabilities.square()).sum() / net_weights.sum()")
    definitions.validate_program(nonlinear, definitions.BINARY_TASK)


def test_decoder_supports_a_real_linear_memory_area_prefix_alternative():
    np = pytest.importorskip("numpy")
    body = (
        '    order = np.argsort(embeddings[:, 0], kind="stable")\n'
        "    cumulative = np.cumsum(node_areas[order])\n"
        "    count = np.argmin(np.abs(cumulative-balance_target*np.sum(node_areas)))+1\n"
        "    return (np.argsort(order) < count).astype(np.int64)\n"
    )
    source = definitions.validate_program(decoder_program(body), definitions.DECODER_TASK)
    namespace = {"np": np}
    exec(compile(source, "trusted_test_prefix_decoder", "exec"), namespace)
    embeddings = np.array([[0., 1.], [1., 0.], [.2, .8], [.8, .2], [.4, .6], [.6, .4]])
    area = np.ones(6)
    before = embeddings.copy(), area.copy()
    embeddings.setflags(write=False)
    area.setflags(write=False)
    labels = namespace["decode_tp_gnn_embeddings"](embeddings, area, 1001, 50, .5)
    assert labels.tolist() == [1, 0, 1, 0, 1, 0]
    assert area[labels == 1].sum() / area.sum() == .5
    assert np.array_equal(embeddings, before[0]) and np.array_equal(area, before[1])


def test_new_v1_loss_language_is_independent_of_the_legacy_language(monkeypatch):
    from local_tasks import tp_gnn_loss_patch as legacy
    monkeypatch.syspath_prepend(str(ROOT / 'Place-3D'))
    from src.partition.lol.methods.tp_gnn_evolution import validate_evolution_source
    assert definitions._validate_objective.__module__ == 'local_tasks.tp_gnn_evolution_loss_language'
    assert 'amax' in legacy.SAFE_TORCH_APIS
    assert 'detach' not in legacy.FORBIDDEN_TENSOR_ATTRIBUTES
    header = 'def compute_binary_partition_loss(' + ', '.join(definitions.BINARY_ARGS) + '):\n'
    source = header + '    for iteration in range(2):\n        value = probabilities.sum()\n    return value\n'
    with pytest.raises(ValueError):
        definitions.validate_program(source, definitions.BINARY_TASK)
    with pytest.raises(ValueError):
        validate_evolution_source(source, 'tp-gnn-binary-loss-v1')
