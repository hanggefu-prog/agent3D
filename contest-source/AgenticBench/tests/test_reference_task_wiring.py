"""Real task/catalog dispatch contracts; no model, container or EDA calls."""
from pathlib import Path

import pytest
import yaml

from global_bench.model import load_global_task
from global_bench.verifier import load_trusted_verifier
from local_bench.metrics import extract_metrics, placement_metrics
from local_bench.model import load_task
from local_bench.prompting import composed_system_message
from test_reference_placement_evidence import HEADER, row


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = "placement_dynamic_parameters_reference_20260923"


@pytest.mark.parametrize("task,mode,profile", [
    ("lol_partition_placement_timing_one_pass", "timing", "tp_gnn_placement_one_pass"),
    ("lol_partition_placement_cts_thermal_one_pass", "cts-thermal", "tp_gnn_placement_cts_thermal_one_pass"),
])
def test_one_pass_task_loads_exact_registered_verifier_profile(task, mode, profile):
    spec = load_global_task(ROOT / "global_tasks" / task)
    trusted = load_trusted_verifier(spec.verifier.verifier_id)
    command = trusted.entrypoint
    assert command[command.index("--mode") + 1] == mode
    assert command[command.index("--profile") + 1] == spec.objective.evaluator_profile == profile
    assert trusted.timeout_seconds == 21600
    assert spec.budget.global_rounds == spec.budget.expensive_evaluations == 5


@pytest.mark.parametrize("directory", ["placement_parameters_reference_20260923", "placement_initialization", "placement_preconditioner"])
def test_reference_placement_system_prompt_and_search_budget_load(directory):
    spec = load_task(ROOT / "local_tasks" / directory)
    message = composed_system_message(spec)
    assert "placement" in message
    config = yaml.safe_load((spec.root / "config.yaml").read_text())
    assert config["max_iterations"] == 50
    assert config["database"] == {"population_size": 10, "archive_size": 8, "num_islands": 2}
    assert config["diff_based_evolution"] is False


def placement_log(root, control):
    stage = root / "30_placement/stages/02_global_place"
    stage.mkdir(parents=True)
    (stage / "run.log").write_text(control +
        "[INFO] iteration 1, wHPWL 1000, Overflow [0.1, 0.2], MaxDensity [0.6, 0.7]\n")
    return root


def test_task_identity_dispatches_nesterov_without_changing_placement_score(tmp_path):
    spec = load_task(ROOT / "local_tasks/placement_parameters_reference_20260923")
    candidate = placement_log(tmp_path / "candidate", HEADER + row() + row(1, 2, shared="2"))
    # A controller-free archived reference can supply the unchanged HPWL /
    # overflow denominator. Active hook qualification remains separate.
    reference = placement_log(tmp_path / "reference", "")
    result = extract_metrics(spec.kind, candidate, reference, task_id=spec.task_id)
    assert result["combined_score"] == placement_metrics(candidate, reference)["combined_score"] == 1
    assert result["nesterov_control_updates"] == 2
    assert result["nesterov_native_step_hint_initial"] == .2
    assert result["nesterov_shared_multiplier_final"] == 2
    assert result["nesterov_identity_controls_only"] is False
    assert not any(key.startswith("dynamic_lr_") for key in result)


@pytest.mark.parametrize("control", ["", HEADER, "DynamicLR step=0 lr=0.01 hpwl=1000 overflow=0.2\n"])
def test_nesterov_dispatch_rejects_missing_or_wrong_optimizer_trace(tmp_path, control):
    candidate = placement_log(tmp_path, control)
    with pytest.raises(ValueError, match="zero active"):
        extract_metrics("placement_dynamic_parameters", candidate, None, task_id=DYNAMIC)


def test_legacy_dynamic_dispatch_does_not_accept_nesterov_trace(tmp_path):
    candidate = placement_log(tmp_path, HEADER + row())
    with pytest.raises(ValueError, match="DynamicLR"):
        extract_metrics("placement_dynamic_parameters", candidate, None,
                        task_id="placement_dynamic_parameters")
    with pytest.raises(ValueError, match="task kind"):
        extract_metrics("placement_source", candidate, None, task_id=DYNAMIC)
