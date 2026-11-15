"""Host CLI contracts only: no model, Docker, GPU or EDA calls."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_global_task_smoke as smoke
from scripts.run_scheduled_codesign import REQUIRED_PREFLIGHT_CHECKS

IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def harness(tmp_path, monkeypatch):
    batch = tmp_path / "runs"
    bundle = batch / "inputs/bp_fe"
    bundle.mkdir(parents=True)
    (bundle / "benchmark_input.json").write_text('{"frozen": true}\n')
    receipt = tmp_path / "qualified-workspace.json"
    receipt.write_text(json.dumps({
        "success": True, "exit_code": 0, "timed_out": False,
        "final_message": "OK", "model": smoke.MODEL, "image": IMAGE,
        "qualification": "workspace-terminal-and-resume-v1", "eda_executed": False,
        "checks": {key: True for key in REQUIRED_PREFLIGHT_CHECKS},
    }))
    output = batch / "global/timing"
    calls = SimpleNamespace(runtimes=[], drivers=[], inputs=[])
    options = SimpleNamespace(case="bp_fe", summary_changes={}, driver_error=None, status="completed")

    class Runtime:
        def __init__(self, source, root, image, gpu, cpus, memory_gb, **kwargs):
            self.source, self.batch, self.image = source, root, image
            self.source_commit = "b" * 40
            self.gpus = [int(value) for value in gpu.split(",")]
            self.framework = smoke.ROOT
            self.cpus, self.memory_gb, self.kwargs = cpus, memory_gb, kwargs
            calls.runtimes.append(self)
        def profile(self, case):
            return smoke.ROOT / "local_bench/benchmark_inputs" / (case + "_lol_ng45_tp_gnn_sta")
        def contained(self, path):
            path = Path(path).resolve()
            path.relative_to(self.batch)
            return path

    def validate(bundle, *, task_id):
        calls.inputs.append((bundle, task_id))
        return {"run_dir": str(bundle / "run"),
                "config": str(smoke.ROOT / "local_bench/benchmark_inputs" /
                              (options.case + "_lol_ng45_tp_gnn_sta/config.yaml"))}

    def driver(runtime, case, objective, benchmark, root, binary):
        calls.drivers.append((runtime, case, objective, benchmark, root, binary))
        if options.driver_error:
            raise options.driver_error
        (root / "status.json").write_text(json.dumps({"state": options.status}))
        summary = {"state": "completed", "case": case, "objective": objective,
                   "final_evaluations": runtime.workspace_rounds,
                   "reference_evaluations": 0 if runtime.reference_policy == "ADOPT" else 1,
                   "reference_adoptions": int(runtime.reference_policy == "ADOPT"),
                   "successful_final_evaluations": 4, **options.summary_changes}
        (root / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(smoke, "Runtime", Runtime)
    monkeypatch.setattr(smoke, "validate_benchmark_input", validate)
    monkeypatch.setattr(smoke, "run_workspace_experiment", driver)
    argv = ["--source", str(smoke.ROOT.parent), "--batch-root", str(batch),
            "--benchmark-input", str(bundle), "--output", str(output), "--objective", "timing",
            "--image", IMAGE, "--gpu", "1", "--preflight-result", str(receipt),
            "--codex-bin", "/trusted/codex"]
    return SimpleNamespace(batch=batch, bundle=bundle, output=output, receipt=receipt,
                           calls=calls, options=options, argv=argv)


@pytest.mark.parametrize("objective,kind,passes", [
    ("timing", "tp_gnn_placement_one_pass", 1),
    ("thermal", "tp_gnn_placement_cts_thermal_one_pass", 1),
    ("routed_thermal", "tp_gnn_placement_thermal", 1),
])
def test_cli_dispatches_fixed_endpoint_with_shared_locks_and_no_extra_probe(harness, objective, kind, passes, capsys):
    harness.argv[harness.argv.index("--objective") + 1] = objective
    before = harness.receipt.read_bytes(), harness.receipt.stat().st_mtime_ns
    assert smoke.main(harness.argv) == 0
    runtime = harness.calls.runtimes[0]
    assert runtime.cpus == 16 and runtime.memory_gb == 48
    assert runtime.batch == harness.batch
    assert runtime.kwargs == {"share_gpu": True, "min_gpu_free_mb": 2048, "min_data_free_gb": 64}
    assert harness.calls.inputs == [(harness.bundle, kind)]
    assert harness.calls.drivers == [(runtime, "bp_fe", objective, harness.bundle, harness.output, "/trusted/codex")]
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    assert contract["resources"]["min_data_free_gb"] == 64
    assert contract["case_selection"] == "smoke-default" and contract["case"] == "bp_fe"
    assert contract["candidate_final_budget"] == 5
    assert contract["evaluation_endpoint"]["opt_passes"] == passes
    assert contract["reference_qualification_in_candidate_budget"] is False
    assert contract["reference_policy"] == runtime.reference_policy == "ADOPT"
    assert contract["reference_evaluator_invocations_expected"] == 0
    assert contract["reference_fallback"] == "none"
    assert contract["model_probe_started"] is False
    assert contract["shared_lock_root"] == str(harness.batch / "locks")
    assert contract["model_preflight"]["result"] == str(harness.receipt)
    assert len(contract["input_manifest_sha256"]) == 64
    assert len(contract["model_preflight"]["result_sha256"]) == 64
    assert (harness.receipt.read_bytes(), harness.receipt.stat().st_mtime_ns) == before
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in events] == ["global-smoke-started", "global-smoke-completed"]
    assert events[-1]["successful_final_evaluations"] == 4


@pytest.mark.parametrize("policy", ["ADOPT", "FRESH", "EXTEND"])
def test_explicit_reference_controls_reach_driver_and_smoke_contract(harness, policy):
    store = harness.batch.parent / "shared-reference-store"
    record = store / "records" / ("a" * 64 + ".json")
    extra = ["--reference-policy", policy, "--reference-store", str(store)]
    if policy != "FRESH":
        extra += ["--reference-record", str(record)]
    assert smoke.main([*harness.argv, *extra]) == 0
    runtime = harness.calls.runtimes[-1]
    assert runtime.reference_policy == policy and runtime.reference_store == store
    assert runtime.reference_record == (record if policy != "FRESH" else None)
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    assert contract["reference_policy"] == policy
    assert contract["reference_store"] == str(store)
    assert contract["reference_record"] == (str(record) if policy != "FRESH" else None)
    assert contract["reference_evaluator_invocations_expected"] == (0 if policy == "ADOPT" else 1)


def test_explicit_case_and_resources_are_recorded_without_changing_task(harness):
    harness.options.case = "bp"
    assert smoke.main([*harness.argv, "--case", "bp", "--cpus", "8", "--memory-gb", "32",
                       "--min-gpu-free-mb", "4096"]) == 0
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    assert contract["case_selection"] == "explicit-cli-override"
    assert contract["case"] == "bp"
    assert contract["resources"]["cpus"] == 8
    assert contract["resources"]["memory_gb"] == 32
    assert contract["resources"]["min_gpu_free_mb"] == 4096
    assert contract["task"]["budget"]["global_rounds"] == 5


def test_explicit_audit_and_computation_pins_reach_runtime(harness):
    assert smoke.main([*harness.argv, "--audit-framework-revision", "d" * 40,
                       "--computation-source-revision", "b" * 40]) == 0
    options = harness.calls.runtimes[0].kwargs
    assert options["audit_framework_revision"] == "d" * 40
    assert options["computation_source_revision"] == "b" * 40


@pytest.mark.parametrize("extra", [
    ["--audit-framework-revision", "d" * 40],
    ["--computation-source-revision", "b" * 40],
    ["--audit-framework-revision", "d" * 40, "--computation-source-revision", "b" * 40,
     "--reference-policy", "FRESH"],
])
def test_split_pins_missing_or_fresh_refused_before_runtime(harness, extra):
    with pytest.raises(ValueError):
        smoke.main([*harness.argv, *extra])
    assert not harness.calls.runtimes and not harness.output.exists()


@pytest.mark.parametrize("mode", ["best", "best-per-round"])
def test_retention_is_recorded_and_propagated_without_changing_budget(harness, mode):
    assert smoke.main([*harness.argv, "--retention", mode]) == 0
    assert harness.calls.runtimes[0].retention_mode == mode
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    assert contract["retention"] == mode
    assert contract["candidate_final_budget"] == 5


def test_invalid_retention_is_rejected_before_model_or_physical_execution(harness):
    with pytest.raises(SystemExit):
        smoke.main([*harness.argv, "--retention", "delete-all"])
    assert not harness.calls.drivers and not harness.calls.runtimes


@pytest.mark.parametrize("objective", ["timing", "thermal", "routed_thermal"])
def test_agentic3d_method_uses_real_multi_role_driver(harness, monkeypatch, objective):
    from scripts import run_workspace_agentic3d
    called = []
    def agentic_driver(*args):
        called.append(args)
        return smoke.run_workspace_experiment(*args)
    monkeypatch.setattr(run_workspace_agentic3d, "run_workspace_agentic3d_experiment", agentic_driver)
    harness.argv[harness.argv.index("--objective") + 1] = objective
    assert smoke.main([*harness.argv, "--method", "agentic3d"]) == 0
    assert len(called) == 1
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    assert contract["method"] == "agentic3d"
    assert contract["evaluation_endpoint"]["opt_passes"] == 1


@pytest.mark.parametrize("changes", [
    {"qualification": "api-ping"}, {"image": "sha256:" + "c" * 64},
    {"model": "another-model"}, {"success": False}, {"checks": {}},
    {"eda_executed": True},
])
def test_unqualified_receipt_stops_before_runtime_or_driver(harness, changes):
    value = json.loads(harness.receipt.read_text())
    harness.receipt.write_text(json.dumps({**value, **changes}))
    with pytest.raises(ValueError):
        smoke.main(harness.argv)
    assert not harness.calls.runtimes and not harness.calls.drivers and not harness.output.exists()


@pytest.mark.parametrize("extra", [["--rounds", "6"], ["--rounds", "0"], ["--model", "other"]])
def test_cli_cannot_override_model_or_five_round_budget(harness, extra):
    with pytest.raises(SystemExit):
        smoke.main([*harness.argv, *extra])
    assert not harness.calls.runtimes and not harness.calls.drivers


def test_changed_task_budget_is_rejected_not_rewritten(harness, monkeypatch):
    task = smoke.load_global_task(smoke.ROOT / "global_tasks" / smoke.TASKS["timing"])
    task = replace(task, budget=replace(task.budget, global_rounds=6, expensive_evaluations=6))
    monkeypatch.setattr(smoke, "load_global_task", lambda path: task)
    with pytest.raises(ValueError, match="already define exactly 5"):
        smoke.main(harness.argv)
    assert not harness.calls.runtimes and not harness.calls.drivers


@pytest.mark.parametrize("method", ["codex", "agentic3d"])
def test_explicit_ten_round_timing_pins_one_static_task_for_both_drivers(harness, monkeypatch, method):
    from scripts import run_workspace_agentic3d
    from scripts.workspace_task_selection import select_driver_task, TIMING_TEN_TASK
    if method == "agentic3d":
        monkeypatch.setattr(run_workspace_agentic3d, "run_workspace_agentic3d_experiment",
                            smoke.run_workspace_experiment)
    assert smoke.main([*harness.argv, "--method", method, "--rounds", "10"]) == 0
    runtime = harness.calls.runtimes[0]
    contract = json.loads((harness.output / "smoke_contract.json").read_text())
    selected = select_driver_task(runtime, "timing", framework=smoke.ROOT)
    assert selected.task_id == contract["task"]["task_id"] == TIMING_TEN_TASK
    assert runtime.workspace_task_digest == selected.digest() == contract["task_digest"]
    assert selected.budget.global_rounds == selected.budget.expensive_evaluations == 10
    assert contract["candidate_final_budget"] == runtime.workspace_rounds == 10
    assert contract["evaluation_endpoint"]["task_id"] == TIMING_TEN_TASK
    assert contract["evaluation_endpoint"]["profile"] == "tp_gnn_placement_one_pass"
    assert contract["reference_qualification_in_candidate_budget"] is False


@pytest.mark.parametrize("objective", ["thermal", "routed_thermal"])
def test_ten_round_budget_does_not_expand_thermal_tasks(harness, objective):
    harness.argv[harness.argv.index("--objective") + 1] = objective
    with pytest.raises(ValueError, match="ten-round timing"):
        smoke.main([*harness.argv, "--rounds", "10"])
    assert not harness.calls.runtimes and not harness.calls.drivers
    assert not harness.output.exists()


def test_wrong_case_bundle_is_rejected_before_any_driver_call(harness):
    harness.options.case = "bp"
    with pytest.raises(ValueError, match="profile does not match"):
        smoke.main(harness.argv)
    assert not harness.calls.drivers and not harness.output.exists()


@pytest.mark.parametrize("path", ["outside", "batch", "bundle", "inside-bundle"])
def test_output_cannot_escape_shared_root_or_touch_frozen_input(harness, path):
    output = {"outside": harness.batch.parent / "outside", "batch": harness.batch,
              "bundle": harness.bundle, "inside-bundle": harness.bundle / "experiment"}[path]
    harness.argv[harness.argv.index("--output") + 1] = str(output)
    with pytest.raises(ValueError):
        smoke.main(harness.argv)
    assert not harness.calls.drivers
    assert (harness.bundle / "benchmark_input.json").read_text() == '{"frozen": true}\n'


def test_completed_attempt_is_not_silently_replayed(harness):
    smoke.main(harness.argv)
    before = (harness.output / "smoke_contract.json").read_bytes()
    with pytest.raises(FileExistsError, match="no implicit replay"):
        smoke.main(harness.argv)
    assert len(harness.calls.drivers) == 1
    assert (harness.output / "smoke_contract.json").read_bytes() == before


@pytest.mark.parametrize("changes", [
    {"state": "needs-attention"}, {"case": "bp"}, {"objective": "thermal"},
    {"final_evaluations": 4}, {"reference_evaluations": 1}, {"reference_evaluations": True},
    {"reference_adoptions": 0}, {"reference_adoptions": True},
])
def test_summary_must_match_endpoint_budget_and_explicit_reference_policy(harness, changes):
    harness.options.summary_changes = changes
    with pytest.raises(ValueError, match="pinned task/budget/reference"):
        smoke.main(harness.argv)
    assert json.loads((harness.output / "smoke_status.json").read_text())["state"] == "needs-attention"


def test_driver_failure_is_preserved_and_never_retried(harness, capsys):
    harness.options.driver_error = RuntimeError("Reference routing failed")
    with pytest.raises(RuntimeError, match="Reference routing failed"):
        smoke.main(harness.argv)
    state = json.loads((harness.output / "smoke_status.json").read_text())
    assert state["state"] == "needs-attention"
    assert state["error"] == "RuntimeError: Reference routing failed"
    assert len(harness.calls.drivers) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "global-smoke-needs-attention"


def test_noncompleted_driver_status_is_not_a_success(harness):
    harness.options.status = "needs-attention"
    with pytest.raises(RuntimeError, match="Workspace driver needs inspection"):
        smoke.main(harness.argv)
