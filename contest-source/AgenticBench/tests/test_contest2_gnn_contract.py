from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from contest2.scoring import PUBLIC_SPLITS, SEEDS
from local_bench import gnn_evolution_timing as timing
from local_tasks.tp_gnn_local_task_definitions import (
    BINARY_TASK, DECODER_TASK, validate_program,
)


ROOT = Path(__file__).resolve().parents[1]


def test_only_the_selected_function_interfaces_are_editable():
    for task, directory in ((BINARY_TASK, "binary_loss"), (DECODER_TASK, "loss_decoder")):
        source = (ROOT / "agent" / directory / "program.py").read_text()
        source = validate_program(source, task)
        assert source.startswith("def ")
        with pytest.raises(ValueError):
            validate_program("import os\n" + source, task)
        with pytest.raises(ValueError):
            validate_program(source + "\ndef extra():\n    return 1\n", task)
    assert len(PUBLIC_SPLITS["train"]) * len(SEEDS) == 9


@pytest.mark.parametrize("task", [BINARY_TASK, DECODER_TASK])
def test_post_opt_score_uses_document_weights_and_refuses_zero_reference(task):
    candidate = {"wns": -8.0, "tns": -20.0, "partition_config": {"decoder": "fixed"}}
    reference = {"wns": -10.0, "tns": -40.0, "partition_config": {"decoder": "fixed"}}
    with (patch.object(timing, "_opt_snapshot", side_effect=[candidate, reference]),
          patch.object(timing, "_independent", return_value={"partition_constraints_passed": 1.0}),
          patch.object(timing, "sha256", return_value="same"),
          patch("local_bench.observed_diagnostics.assignment_comparison", return_value={}),
          patch.object(timing, "_binary_diagnostics")):
        row = timing.timing_metrics("candidate", "reference", task_id=task)
    assert row["combined_score"] == pytest.approx(100 * (0.4 * 0.2 + 0.6 * 0.5))
    assert row["opt_passes"] == 1.0
    assert row["timing_evaluation_boundary"] == "post-opt-pre-cts"
    reference["wns"] = 0.0
    with (patch.object(timing, "_opt_snapshot", side_effect=[candidate, reference]),
          patch.object(timing, "_independent", return_value={}),
          patch.object(timing, "sha256", return_value="same")):
        with pytest.raises((ValueError, AssertionError), match="zero-deficit"):
            timing.timing_metrics("candidate", "reference", task_id=task)


@pytest.mark.parametrize("task", [BINARY_TASK, DECODER_TASK])
def test_packed_reference_scores_without_native_reference_run(tmp_path, task):
    import hashlib
    import json
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    for name in timing.TIMING_FILES:
        path = candidate / "20_partition/timing_input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    assignment = reference / "20_partition/partition_assignment.csv"
    assignment.parent.mkdir(parents=True)
    assignment.write_text("inst_name,die\ncell_a,top\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1, "task_id": task, "wns": -10.0, "tns": -40.0,
        "partition_config": {"decoder": timing.TASK_DECODERS[task]},
        "timing_sha256": {name: digest(candidate / "20_partition/timing_input" / name)
                          for name in timing.TIMING_FILES},
        "assignment_sha256": digest(assignment),
        "native_stage_sha256": {name: "a" * 64 for name in
                                ("20_partition", "30_placement", "40_pdn", "50_hbt", "60_opt")},
        "independent_partition_metrics": {"partition_constraints_passed": 1},
    }
    (reference / "reference_summary.json").write_text(json.dumps(summary))
    assert timing._packed_reference(reference, task)["wns"] == -10.0
    with (patch.object(timing, "_opt_snapshot", return_value={
            "wns": -8.0, "tns": -20.0,
            "partition_config": summary["partition_config"]}),
          patch.object(timing, "_independent", return_value={"partition_constraints_passed": 1}),
          patch("local_bench.observed_diagnostics.assignment_comparison", return_value={}),
          patch.object(timing, "_binary_diagnostics")):
        row = timing.timing_metrics(candidate, reference, task_id=task)
    assert row["combined_score"] == pytest.approx(38.0)
    (candidate / "20_partition/timing_input/timing_net_weights.csv").write_text("changed")
    with (patch.object(timing, "_opt_snapshot", return_value={
            "wns": -8.0, "tns": -20.0,
            "partition_config": summary["partition_config"]}),
          patch.object(timing, "_independent", return_value={})):
        with pytest.raises(ValueError, match="frozen timing differs"):
            timing.timing_metrics(candidate, reference, task_id=task)


def test_agent_container_makes_evaluator_read_only_and_offline():
    from contest2.agent_container import command
    with TemporaryDirectory() as temp:
        root = Path(temp)
        paths = {name: root / name for name in ("agent", "workspaces", "scratch", "bridge", "framework", "source", "input")}
        for path in paths.values():
            path.mkdir()
        with patch("pathlib.Path.is_socket", return_value=True):
            argv = command(name="o3db-contest-agent-123456789abc", image="sha256:" + "a" * 64,
                agent_dir=paths["agent"], workspaces=paths["workspaces"],
                scratch=paths["scratch"], bridge_dir=paths["bridge"],
                framework=paths["framework"], source=paths["source"],
                training_inputs={"bp_fe-ng45": paths["input"]})
            joined = " ".join(argv)
            assert "--network=none" in joined
            assert "/opt/contest/AgenticBench,readonly" in joined
            assert "/opt/contest/source,readonly" in joined
            assert "/workspace/agent,readonly" not in joined
            assert "/workspace/agent" in joined


def test_two_baselines_have_independent_ten_evaluations_and_fifteen_rmb(tmp_path):
    from contest2.budget import BudgetLedger, BudgetExceeded, BudgetConflict
    books = {baseline: BudgetLedger(tmp_path / baseline / "budget.sqlite",
             model_cost_limit_rmb="15", task_id=baseline, allowed_model="deepseek-flash")
             for baseline in ("binary_loss", "loss_decoder")}
    binary, decoder = books.values()
    binary.reserve_model_call("same-request", "deepseek-flash", "15")
    assert binary.snapshot()["model_cost_available_rmb"] == "0"
    assert decoder.snapshot()["model_cost_available_rmb"] == "15"
    binary.record_model_call("same-request", "deepseek-flash", 100, 100, "15")
    with pytest.raises(BudgetExceeded):
        binary.reserve_model_call("extra", "deepseek-flash", "1")
    decoder.reserve_model_call("same-request", "deepseek-flash", "1")
    decoder.record_model_call("same-request", "deepseek-flash", 100, 100, "0.5")
    assert decoder.snapshot()["model_cost_available_rmb"] == "14.5"
    for baseline, ledger in books.items():
        for index in range(10):
            request = "request-" + str(index)
            if index % 2:
                ledger.reject_submission(request, "a" * 64, "Invalid interface", baseline=baseline)
            else:
                ledger.accept_matrix(request, f"candidate-{index}", "a" * 64, baseline=baseline)
                ledger.finish_evaluation(request, "failed", "native failure")
        assert ledger.snapshot()["evaluations_remaining"] == 0
        ledger.accept_matrix("request-0", "candidate-0", "a" * 64, baseline=baseline)
        with pytest.raises(BudgetExceeded, match="evaluation_limit_reached"):
            ledger.accept_matrix("extra", "candidate-extra", "b" * 64, baseline=baseline)
        resumed = BudgetLedger(ledger.path, model_cost_limit_rmb="15", task_id=baseline,
                               allowed_model="deepseek-flash")
        assert resumed.snapshot() == ledger.snapshot()
    with pytest.raises(BudgetConflict, match="task_id"):
        BudgetLedger(binary.path, model_cost_limit_rmb="15", task_id="loss_decoder",
                     allowed_model="deepseek-flash")
    with pytest.raises(BudgetConflict, match="Baseline differs"):
        binary.reject_submission("wrong", "b" * 64, "Invalid", baseline="loss_decoder")


@pytest.mark.parametrize("task,recipe", [
    (BINARY_TASK, "tp_gnn_binary_regularized.json"),
    (DECODER_TASK, "tp_gnn_fast.json"),
])
def test_matrix_seed_is_bound_into_the_native_gnn_recipe(task, recipe):
    import json
    from grt_bench.flow import bind_method_seeds
    source = ROOT.parent
    config = {"partition": {"algorithm": "tp-gnn-flat", "seed": 1002,
              "config": str(source / "Place-3D/config/partition" / recipe),
              "evolution_program": "/candidate/capsule.json"},
              "placement": {"algorithm": "concurrent-dreamplace", "seed": 1002,
              "config": str(source / "Place-3D/config/placement/concurrent_dreamplace/default.json")}}
    with TemporaryDirectory() as temp:
        folder = Path(temp) / "method_seeds"
        bind_method_seeds(config, {"partition": 1002, "placement": 1002}, folder)
        assert json.loads(Path(config["partition"]["config"]).read_text())["partition"]["seed"] == 1002
        timing.validate_recipe(config, source, task_id=task)
        config["partition"]["seed"] = 1003
        with pytest.raises((ValueError, AssertionError), match="native seed|other than"):
            timing.validate_recipe(config, source, task_id=task)


def test_probability_baseline_program_and_training_contract_stay_aligned():
    import json
    import yaml
    from local_bench.model import load_candidate, load_task
    from local_bench.program_bridge import materialize_program

    spec = load_task(ROOT / "local_tasks" / BINARY_TASK)
    task_program = validate_program(spec.program_template.read_text(), BINARY_TASK)
    agent_program = validate_program((ROOT / "agent/binary_loss/program.py").read_text(), BINARY_TASK)
    assert task_program == agent_program
    assert materialize_program(spec, spec.program_template.read_text(), ROOT.parent) == load_candidate(spec.initial_candidate)

    definition = yaml.safe_load((spec.root / "task.yaml").read_text())["definition"]
    recipe_path = ROOT.parent / definition["fixed_recipe"]
    recipe = json.loads(recipe_path.read_text())["partition"]
    fixed = definition["fixed_training"]
    fields = {"learning_rate": "learning_rate", "updates": "epochs",
              "temperature_initial": "binary_temperature", "temperature_final": "binary_temperature_final",
              "warmup_fraction": "binary_warmup_fraction", "ramp_end_fraction": "binary_ramp_end_fraction",
              "cut_weight": "binary_cut_weight", "area_balance_weight": "binary_area_balance_weight",
              "final_binarization_weight": "binary_binarization_weight",
              "initial_logit_noise_scale": "binary_logit_noise_scale"}
    for field, native in fields.items():
        assert fixed[field] == recipe["tp_gnn_" + native]
    assert fixed["updates"] == recipe["tp_gnn_min_epochs"] == recipe["tp_gnn_min_optimizer_steps"]
    timing._recipe(ROOT.parent, BINARY_TASK)  # The pinned hash still protects the entire recipe.
