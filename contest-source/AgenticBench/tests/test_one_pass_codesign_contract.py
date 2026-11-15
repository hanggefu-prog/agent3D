"""Versioned reference-aligned endpoints; fixtures only, no EDA/model calls."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d import codesign_endpoint as roles
from global_bench.model import load_global_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from scripts.workspace_codesign_evaluator import ENDPOINTS, LEGACY_ENDPOINTS, WorkspaceEvaluator, endpoint_for_task, reference_mounts
from scripts.run_workspace_codesign import TASKS
from test_cts_thermal import cts_run, native_cts_identity
from test_sta_codesign import _write_run
from test_thermal_baseline import thermal_run, write

ROOT = Path(__file__).resolve().parents[1]


def native_one_pass(root):
    path = root / "60_opt/metrics.json"
    value = json.loads(path.read_text())
    value.update(passes_run=1, iterations_run=1, die_passes={"bottom": 1, "top": 1})
    value.pop("pass_records", None)
    value["pass_record"] = {
        "pass": 1, "die_order": ["bottom", "top"],
        "constraints": {"bottom": {}, "top": {}},
        "die_metrics": {"bottom": {"wns_setup": -1.0}, "top": {"wns_setup": -1.0}},
        "global_before": value["initial_global"], "global_after": value["final_global"],
    }
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def physical_gates(monkeypatch):
    # Independent artifact/seed gate suites retain ownership of those checks.
    monkeypatch.setattr(verifier, "_validate_chain", lambda *args, **kwargs: None)
    monkeypatch.setattr(verifier, "_validate_seed_contract", lambda *args, **kwargs: None)


@pytest.mark.parametrize("objective", ["timing", "thermal"])
def test_new_tasks_select_exact_one_pass_without_changing_legacy(objective):
    task = load_global_task(ROOT / "global_tasks" / TASKS[objective])
    assert endpoint_for_task(objective, task) == ENDPOINTS[objective]
    assert ENDPOINTS[objective]["opt_passes"] == 1
    assert task.budget.expensive_evaluations == task.budget.global_rounds == 5
    old = load_global_task(ROOT / "global_tasks" / LEGACY_ENDPOINTS[objective]["task_id"])
    assert endpoint_for_task(objective, old)["opt_passes"] == 2
    assert task.digest() != old.digest()


def test_single_pass_timing_score_and_cross_contract_rejection(tmp_path, physical_gates):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    _write_run(candidate, -0.5, -5.)
    _write_run(reference, -1., -10.)
    native_one_pass(candidate)
    native_one_pass(reference)
    score = verifier._score_timing(candidate, reference, profile="tp_gnn_placement_one_pass")
    assert score["combined_score"] == pytest.approx(2.)
    assert score["opt_passes"] == 1
    with pytest.raises(ValueError, match="exactly 2 OPT"):
        verifier._score_timing(candidate, reference)
    legacy = tmp_path / "legacy"
    _write_run(legacy, -1., -10.)
    for pair in ((candidate, legacy), (legacy, reference), (legacy, legacy)):
        with pytest.raises(ValueError, match="exactly 1 OPT"):
            verifier._score_timing(*pair, profile="tp_gnn_placement_one_pass")


@pytest.mark.parametrize("fault", ["missing_record", "list_record", "pass_number", "die_order",
                                    "missing_die", "empty_die", "die_count", "count_bool",
                                    "constraints", "final_mismatch"])
def test_native_one_pass_requires_actual_both_die_audit(tmp_path, fault):
    _write_run(tmp_path, -1., -10.)
    value = native_one_pass(tmp_path)
    record = value["pass_record"]
    if fault == "missing_record":
        value.pop("pass_record")
    elif fault == "list_record":
        value["pass_records"] = [copy.deepcopy(record)]
    elif fault == "pass_number":
        record["pass"] = 2
    elif fault == "die_order":
        record["die_order"] = ["bottom"]
    elif fault == "missing_die":
        record["die_metrics"].pop("top")
    elif fault == "empty_die":
        record["die_metrics"]["top"] = {}
    elif fault == "die_count":
        value["die_passes"]["top"] = 0
    elif fault == "count_bool":
        value["passes_run"] = True
    elif fault == "constraints":
        record["constraints"].pop("top")
    else:
        record["global_after"] = {"wns_setup": 0., "tns_setup": 0.}
    with pytest.raises(ValueError):
        verifier._validate_opt_passes(value, 1)


def test_one_pass_cts_thermal_keeps_power_gates_and_distinct_boundary(tmp_path, physical_gates):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    for root, mean in ((candidate, 321.), (reference, 322.)):
        cts_run(root, mean)
        native_one_pass(root)
        native_cts_identity(root)
    profile = "tp_gnn_placement_cts_thermal_one_pass"
    score = verifier._score_cts_thermal(candidate, reference, profile=profile)
    assert score["combined_score"] == pytest.approx(322. / 321.)
    assert score["thermal_evaluation_boundary"] == "one-opt-cts-cell-power-v1"
    assert score["opt_passes"] == 1
    assert roles.final_score({"metrics": score}, "thermal", profile=profile) == score["combined_score"]
    assert roles.final_score({"metrics": score}, "thermal") is None
    with pytest.raises(ValueError, match="exactly 2 OPT"):
        verifier._score_cts_thermal(candidate, reference)
    metrics_path = candidate / "95_thermal_snapshot/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["power"]["activity_model"] = "unqualified"
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError):
        verifier._score_cts_thermal(candidate, reference, profile=profile)


def test_profile_selection_cannot_mix_modes_or_relax_pass_count():
    with pytest.raises(ValueError, match="match verifier mode"):
        verifier._profile_passes("timing", "tp_gnn_placement_cts_thermal_one_pass")
    assert verifier._profile_passes("timing") == 2
    assert verifier._profile_passes("timing", "tp_gnn_placement_one_pass") == 1


@pytest.mark.parametrize("objective", ["timing", "thermal"])
def test_versioned_role_prompts_keep_legacy_semantics(objective):
    profile = roles.ONE_PASS_PROFILES[objective]
    text = roles.common(objective, 5, profile=profile)
    assert "OPT pass 2" not in text and "exactly two OPT" not in text and "exactly TWO OPT" not in text
    assert "Exactly 5 final evaluations" in text
    old = roles.common(objective, 5)
    assert ("exactly two OPT" if objective == "timing" else "exactly TWO OPT") in old


def test_task_model_rejects_one_pass_profile_with_two_pass_parameter(tmp_path):
    source = ROOT / "global_tasks" / TASKS["timing"]
    (tmp_path / "instruction.md").write_text((source / "instruction.md").read_text())
    text = (source / "task.yaml").read_text().replace("opt_passes: 1", "opt_passes: 2")
    (tmp_path / "task.yaml").write_text(text)
    with pytest.raises(ValueError, match="versioned evaluator profile"):
        load_global_task(tmp_path)


def test_reference_mounts_are_read_only_alias_targets_without_control_overlap():
    archive = Path("/home/example/agentic3d-reference-20260923-v1")
    original = Path("/home/example/agentic3d-unified-20260917")
    receipt = {"reference_root": str(archive), "original_root": str(original)}
    assert reference_mounts(receipt, [Path("/home/example/new-batch")]) == [(archive, archive), (archive, original)]
    assert reference_mounts({**receipt, "original_root": str(archive)}, []) == [(archive, archive)]
    for alias in ("/", "/opt", "/opt/eda", "/tmp/run", "/home/example/new-batch",
                  "/home/example/new-batch/source", "/home", "relative", "/home/example/a:rw"):
        with pytest.raises(ValueError):
            reference_mounts({**receipt, "original_root": alias}, [Path("/home/example/new-batch")])


def test_new_workspace_uses_native_opt_pinned_image_and_fixed_thread_environment(tmp_path):
    from local_bench.reference_baseline import profile_path, IMAGE_IDS
    runtime = SimpleNamespace(source=tmp_path / "source", framework=ROOT, batch=tmp_path / "batch",
                              image=sorted(IMAGE_IDS)[0], cpus=16, memory_gb=48,
                              contained=lambda path: Path(path), profile=profile_path)
    task = load_global_task(ROOT / "global_tasks" / TASKS["timing"])
    evaluator = WorkspaceEvaluator(runtime, "bp_fe", "timing", tmp_path / "bundle", task)
    assert evaluator.patch is None
    assert runtime.reference_baseline == "reference_20260923"
    evaluator.reference_mount = {
        "reference_root": "/home/example/frozen-reference",
        "original_root": "/home/example/original-reference"}
    output = tmp_path / "output"
    output.mkdir()
    command = evaluator._docker(["true"], 1, output, tmp_path / "snapshot", tmp_path / "source-alias",
                                writable=(output,), name="unit-test-no-execution")
    assert "OPEN3D_OPT_PASSES=1" in command
    assert "OMP_NUM_THREADS=4" in command and "MKL_NUM_THREADS=4" in command
    assert command[command.index("--cpus") + 1] == "16"
    assert "/home/example/frozen-reference:/home/example/frozen-reference:ro" in command
    assert "/home/example/frozen-reference:/home/example/original-reference:ro" in command
    runtime.image = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="pinned"):
        WorkspaceEvaluator(runtime, "bp_fe", "timing", tmp_path / "bundle", task)


def endpoint_run(root, objective):
    if objective == "thermal":
        cts_run(root)
        native_cts_identity(root)
        stages = verifier.TIMING_STAGES + ("cts",)
    else:
        _write_run(root, -1., -10.)
        if objective == "routed_thermal":
            thermal_run(root)
            stages = verifier.THERMAL_STAGES
        else:
            stages = verifier.TIMING_STAGES
    native_one_pass(root)
    write(root / "run.json", {"status": "success", "stages": {
        stage: {"status": "success"} for stage in stages}})


def endpoint_score(candidate, reference, objective):
    if objective == "timing":
        return verifier._score_timing(candidate, reference, profile="tp_gnn_placement_one_pass")
    if objective == "thermal":
        return verifier._score_cts_thermal(candidate, reference, profile="tp_gnn_placement_cts_thermal_one_pass")
    return verifier._score_thermal(candidate, reference, profile="tp_gnn_placement_thermal")


@pytest.mark.parametrize("objective", ["timing", "thermal", "routed_thermal"])
@pytest.mark.parametrize("missing_side", ["candidate", "reference"])
def test_new_global_endpoints_require_stage15_on_both_measurements(tmp_path, monkeypatch, objective, missing_side):
    # Keep the real stage-set gate; the independent manifest/hash and seed
    # suites own those checks. Do not mistake imported legacy stage 20 for a
    # freshly recomputed global timing-feature stage.
    monkeypatch.setattr(verifier, "_validate_stage", lambda *args: None)
    monkeypatch.setattr(verifier, "_validate_seed_contract", lambda *args, **kwargs: None)
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    for root in (candidate, reference):
        endpoint_run(root, objective)
    assert endpoint_score(candidate, reference, objective)["opt_passes"] == 1
    ledger_path = tmp_path / missing_side / "run.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["stages"].pop("timing_extract")
    write(ledger_path, ledger)
    with pytest.raises(ValueError, match="requires a native timing_extract"):
        endpoint_score(candidate, reference, objective)


def test_legacy_chain_without_stage15_remains_distinct_from_new_contract(tmp_path, monkeypatch):
    write(tmp_path / "run.json", {"status": "success", "stages": {"partition": {"status": "success"}}})
    monkeypatch.setattr(verifier, "_validate_stage", lambda *args: None)
    verifier._validate_chain(tmp_path, ("timing_extract", "partition"))
    with pytest.raises(ValueError, match="requires a native timing_extract"):
        verifier._validate_chain(tmp_path, ("timing_extract", "partition"), require_timing_stage=True)


@pytest.mark.parametrize("method,accepted", [("mtkahypar", True), ("tp-gnn-flat", False)])
def test_new_stage15_record_can_only_skip_for_verified_non_timing_partitioner(tmp_path, monkeypatch, method, accepted):
    write(tmp_path / "run.json", {"status": "success", "stages": {
        "timing_extract": {"status": "skipped", "reason": "selected partitioner does not consume TP-GNN timing features"},
        "partition": {"status": "success"}}})
    write(tmp_path / "20_partition/partition_manifest.json", {"method": method})
    write(tmp_path / "20_partition/stage_manifest.json", {"method": method})
    monkeypatch.setattr(verifier, "_validate_stage", lambda *args: None)
    if accepted:
        verifier._validate_chain(tmp_path, ("timing_extract", "partition"), require_timing_stage=True)
    else:
        with pytest.raises(ValueError, match="did not complete timing_extract"):
            verifier._validate_chain(tmp_path, ("timing_extract", "partition"), require_timing_stage=True)


@pytest.mark.parametrize("wrong_side", ["candidate", "reference"])
@pytest.mark.parametrize("fault", ["two_pass", "missing_record", "missing_die"])
def test_explicit_routed_profile_rejects_non_native_one_pass_evidence(tmp_path, physical_gates, wrong_side, fault):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    for root in (candidate, reference):
        endpoint_run(root, "routed_thermal")
    assert endpoint_score(candidate, reference, "routed_thermal")["opt_passes"] == 1
    path = tmp_path / wrong_side / "60_opt/metrics.json"
    value = json.loads(path.read_text())
    if fault == "two_pass":
        value.update(passes_run=2, iterations_run=2, die_passes={"bottom": 2, "top": 2})
    elif fault == "missing_record":
        value.pop("pass_record")
    else:
        value["pass_record"]["die_metrics"].pop("top")
    write(path, value)
    with pytest.raises(ValueError, match="OPT"):
        endpoint_score(candidate, reference, "routed_thermal")
    # Existing historical functional callers retain their previous contract;
    # this path must not be silently reinterpreted as the explicit new profile.
    legacy = verifier._score_thermal(candidate, reference)
    assert legacy["combined_score"] == 1. and "opt_passes" not in legacy


def test_standalone_cli_forwards_explicit_routed_profile(tmp_path, monkeypatch):
    evaluated, submission, output = (tmp_path / name for name in ("evaluated", "submission", "output"))
    records = []
    for name, relative in (("run_manifest", "run.json"),
                           ("final_checkpoint", "95_thermal/checkpoint.json"),
                           ("metrics", "95_thermal/metrics.json")):
        write(evaluated / relative, {"fixture": name})
        content = (evaluated / relative).read_bytes()
        write(submission / (name + ".json"), {"fixture": name})
        records.append({"name": name, "submitted_path": name + ".json", "sha256": hashlib.sha256(content).hexdigest()})
    write(submission / "submission.json", {"artifacts": records})
    seen = []
    def score(candidate, reference, *, profile=None):
        seen.append((candidate, reference, profile))
        return {"combined_score": 1., "opt_passes": 1}
    monkeypatch.setattr(verifier, "_score_thermal", score)
    monkeypatch.setattr("sys.argv", ["verifier", "--mode", "thermal", "--profile", "tp_gnn_placement_thermal",
        "--submission", str(submission), "--output", str(output),
        "--evaluated-run", str(evaluated), "--reference-run", str(evaluated)])
    assert verifier.main() == 0
    assert seen == [(evaluated, evaluated, "tp_gnn_placement_thermal")]
    assert json.loads((output / "verification.json").read_text())["passed"] is True
