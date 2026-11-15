"""EXTEND control-plane dispatch tests. Zero EDA, Docker or model execution.

The parent/input qualification and native process boundaries use explicitly
synthetic fixtures. WorkspaceReference.obtain, WorkspaceEvaluator.evaluate,
prefix copying, command receipts, submission structure, mounts and registry
publication retain their real orchestration implementation.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from global_bench.model import load_global_task
from global_bench.verifiers import open3dbench_codesign_verifier as verifier
from local_bench import reference_baseline as baseline, reference_catalog as catalog
from local_bench.reference_evaluated import ENDPOINTS, expected_frozen_inputs, expected_source_contract
from local_bench.reference_store import file_sha
from scripts import workspace_codesign_evaluator as bridge
from scripts import workspace_reference as selection
from scripts import workspace_reference_extension as extension
from test_workspace_codesign_evaluator import ROOT, git, make_runtime, prefix, repository, stage


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _evidence(root):
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha(p), "role": "synthetic-test"}
            for p in sorted(root.rglob("*")) if p.is_file()]


def setup_extension(repository, tmp_path, monkeypatch, *, parent_endpoint="timing", endpoint="cts_thermal",
                    fail_phase=None):
    # Preserve actual clean-Git and source snapshot behavior with the same
    # reference-owned JSON parameter files consumed by the current recipe.
    for relative in (baseline.PARTITION_CONFIG, baseline.PLACEMENT_CONFIG):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT.parent / relative, path)
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "reference parameter fixture")
    runtime = make_runtime(repository, tmp_path)
    runtime.image = sorted(baseline.IMAGE_IDS)[0]
    runtime.reference_baseline = "reference_20260923"
    runtime.profile = baseline.profile_path
    runtime.reference_policy = "EXTEND"
    runtime.reference_store = tmp_path / "registry"
    bundle, frozen, alias = prefix(runtime)
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda p: {"run_dir": str(frozen)})
    monkeypatch.setattr(selection, "validate_benchmark_input", lambda p: {"run_dir": str(frozen)})
    monkeypatch.setattr(baseline, "validate_reference_bundle", lambda *a: None)
    monkeypatch.setattr(baseline, "validate_physical_prefix", lambda *a: {})
    profile, task_id, objective, _ = ENDPOINTS[endpoint]
    task = load_global_task(ROOT / "global_tasks" / task_id)
    contract = {"case": "bp_fe", "endpoint": endpoint, "task_digest": catalog.task_digest(task.root),
                "evaluator_digest": catalog.evaluator_digest(),
                "frozen_inputs": expected_frozen_inputs("bp_fe", file_sha(bundle / "benchmark_input.json")),
                "physical": expected_source_contract("bp_fe", endpoint, profile=profile,
                    source_revision=runtime.source_commit, image=runtime.image)}
    parent_run = runtime.batch / "parent-native/run"
    stages = list(verifier.TIMING_STAGES) + (["cts"] if parent_endpoint == "cts_thermal" else [])
    for name in stages:
        stage(parent_run, name, verifier.STAGE_DIRS[name])
    for name in ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"):
        for directory in ("10_floorplan", "30_placement"):
            (parent_run / directory / name).write_text("fixed physical handoff " + name)
    for die in ("bottom", "top"):
        path = parent_run / "30_placement/die_init" / die / "design.sdc"
        path.parent.mkdir(parents=True)
        shutil.copyfile(parent_run / "10_floorplan/design.sdc", path)
    for directory in ("20_partition", "30_placement"):
        (parent_run / directory / "partition_assignment.csv").write_text("inst_name,die\na,bottom\n")
    bridge._json(parent_run / "run.json", {"status": "success", "stages": {
        name: {"status": "success", "directory": verifier.STAGE_DIRS[name]} for name in stages}})
    parent_record = tmp_path / "parent-registry/records/parent.json"
    bridge._json(parent_record, {"producer": {"kind": "native-evaluation-v1"}})
    runtime.reference_record = parent_record
    parent = {"record": str(parent_record), "record_sha256": file_sha(parent_record),
              "record_value": {"reference_id": "a" * 64,
                "producer": {"kind": "native-evaluation-v1", "run_root": str(parent_run)},
                "qualification": {"baseline": {"source_run": str(parent_run)}, "evidence": _evidence(parent_run)}},
              "target_contract": contract,
              "start_stage": "cts" if parent_endpoint == "timing" else "global_routing", "stages": stages}
    qualifications = []
    def qualify_parent(path, target, *, record_sha256=None):
        qualifications.append((path, copy.deepcopy(target), record_sha256))
        assert Path(path) == parent_record and target == contract
        assert record_sha256 in (None, file_sha(parent_record))
        return copy.deepcopy(parent)
    monkeypatch.setattr(extension, "qualify_parent", qualify_parent)
    selected = selection.WorkspaceReference(runtime, "bp_fe", objective, bundle, task)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp_fe", objective, bundle, task)
    calls = []
    real_run = subprocess.run
    def native_boundary(argv, *args, **kwargs):
        if not argv or argv[0] != "docker":
            return real_run(argv, *args, **kwargs)  # local Git only
        assert argv[:2] == ["docker", "run"], "No retry/cleanup process expected in these non-timeout cases"
        name = argv[argv.index("--name") + 1]
        phase = name.rsplit("-", 1)[-1]
        calls.append((phase, list(argv)))
        assert phase in {"backend", "thermal", "verify"}, "EXTEND must never dispatch a frontend/overlay"
        if phase == fail_phase:
            return subprocess.CompletedProcess(argv, 17)
        output = runtime.batch / "extension"
        run = output / "run"
        if phase == "backend":
            start = argv[argv.index("--from") + 1]
            assert start == parent["start_stage"]
            names = (["cts"] if start == "cts" else [])
            if endpoint == "routed_thermal":
                names += ["global_routing", "detailed_routing", "report", "thermal"]
            status = bridge._read(run / "run.json")
            for name in names:
                directory = verifier.STAGE_DIRS[name]
                stage(run, name, directory)
                bridge._json(run / directory / "checkpoint.json", {"stage": name})
                status["stages"][name] = {"status": "success", "directory": directory}
            status["status"] = "success"
            bridge._json(run / "run.json", status)
        elif phase == "thermal":
            bridge._json(run / "95_thermal_snapshot/metrics.json", {"fixture_temperature_k": 325})
        else:
            bridge._json(output / "verification/verification.json", {"passed": True, "metrics": {"combined_score": 1.0}})
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", native_boundary)
    published = []
    def qualify_new(producer):
        published.append(copy.deepcopy(producer))
        run = Path(producer["run_root"])
        return {"qualified": True, "case": "bp_fe", "endpoint": endpoint,
                "source_contract": contract["physical"], "frozen_inputs": contract["frozen_inputs"],
                "source_identity": {"fixture": "synthetic-native-boundary"},
                "baseline": {"source_run": str(run), "raw_metrics": {"fixture_score": 1.0}},
                "evidence": _evidence(run)}
    monkeypatch.setattr(selection, "_qualify", qualify_new)
    return SimpleNamespace(runtime=runtime, bundle=bundle, alias=alias, parent=parent, selected=selected,
                           evaluator=evaluator, calls=calls, published=published, qualifications=qualifications,
                           output=runtime.batch / "extension", parent_run=parent_run, task=task)


@pytest.mark.parametrize("parent_endpoint,endpoint,start,phases", [
    ("timing", "cts_thermal", "cts", ["backend", "thermal", "verify"]),
    ("timing", "routed_thermal", "cts", ["backend", "verify"]),
    ("cts_thermal", "routed_thermal", "global_routing", ["backend", "verify"]),
])
def test_extend_dispatches_only_missing_suffix_with_readonly_prefix(repository, tmp_path, monkeypatch,
        parent_endpoint, endpoint, start, phases):
    f = setup_extension(repository, tmp_path, monkeypatch, parent_endpoint=parent_endpoint, endpoint=endpoint)
    before = _files(f.parent_run)
    result = f.selected.obtain(f.evaluator, f.output)
    assert result["success"], result
    assert [phase for phase, _ in f.calls] == phases
    backend = dict(f.calls)["backend"]
    assert backend[backend.index("--from") + 1] == start
    assert backend[backend.index("--to") + 1] == ("cts" if endpoint == "cts_thermal" else "thermal")
    for name in f.parent["stages"]:
        directory = verifier.STAGE_DIRS[name]
        path = f.output / "run" / directory
        assert f"{path}:{path}:ro" in backend
        assert (path / "stage_manifest.json").read_bytes() == (f.parent_run / directory / "stage_manifest.json").read_bytes()
    assert _files(f.parent_run) == before
    assert all("OPEN3D_OPT_PASSES=1" in argv for _, argv in f.calls)
    if endpoint == "cts_thermal":
        thermal = dict(f.calls)["thermal"]
        assert thermal[thermal.index("--threads") + 1] == "2"
        assert "OMP_NUM_THREADS=4" in thermal  # physical baseline stays four threads
    contract = bridge._read(f.output / "evaluation-contract.json")
    assert contract["reference_purpose"] == "EXTEND"
    assert [row["stage"] for row in contract["reused_prefix"]] == f.parent["stages"]
    assert result["reference_origin"] == "extend" and result["reference_evaluator_invocations"] == 1
    assert result["execution"]["native_flow_command_dispatches"] == len(phases) - 1
    assert result["execution"]["native_physical_attempts"] is None  # fixtures never claim native execution
    assert f.published and all(p["parent_reference_id"] == "a" * 64 for p in f.published)
    assert len(f.qualifications) == 2  # selection plus just-before-dispatch qualification


@pytest.mark.parametrize("fail_phase,expected", [("backend", ["backend"]),
                                                ("thermal", ["backend", "thermal"]),
                                                ("verify", ["backend", "thermal", "verify"])])
def test_extension_failure_is_retained_and_never_retried_or_published(repository, tmp_path, monkeypatch,
        fail_phase, expected):
    f = setup_extension(repository, tmp_path, monkeypatch, fail_phase=fail_phase)
    result = f.selected.obtain(f.evaluator, f.output)
    assert result["success"] is False and result["metrics"] == {}
    assert result["reference_evaluator_invocations"] == 1
    assert [phase for phase, _ in f.calls] == expected
    assert not f.published and not f.runtime.reference_store.exists()
    before = _files(f.output)
    with pytest.raises(FileExistsError):
        f.selected.obtain(f.evaluator, f.output)
    assert _files(f.output) == before and [phase for phase, _ in f.calls] == expected


@pytest.mark.parametrize("mutation", ["candidate-reference", "candidate-source", "algorithm-change"])
def test_candidate_work_cannot_be_relabelled_as_extension(repository, tmp_path, monkeypatch, mutation):
    f = setup_extension(repository, tmp_path, monkeypatch)
    subject, reference = repository, None
    if mutation == "candidate-reference":
        reference = f.parent_run
    elif mutation == "candidate-source":
        subject = tmp_path / "candidate"
        subject.mkdir()
    else:
        (repository / "algorithm.py").write_text("VALUE = 2\n")
    result = f.evaluator.evaluate(subject, f.output, reference, parent_reference=f.parent)
    assert result["success"] is False and result["metrics"] == {}
    assert ("never a candidate" in result["error"] or "cannot contain algorithm changes" in result["error"])
    assert not f.calls and not f.published


def test_extension_rejects_target_contract_mismatch_before_native_dispatch(repository, tmp_path, monkeypatch):
    f = setup_extension(repository, tmp_path, monkeypatch)
    f.parent["target_contract"] = copy.deepcopy(f.parent["target_contract"])
    f.parent["target_contract"]["physical"]["source_revision"] = "0" * 40
    # The control plane must compare the independently returned parent contract
    # to its actual task/source/runtime, not merely trust a successful callback.
    monkeypatch.setattr(extension, "qualify_parent", lambda *a, **k: f.parent)
    result = f.evaluator.evaluate(repository, f.output, parent_reference=f.parent)
    assert result["success"] is False and "actual task/source/input/runtime" in result["error"]
    assert not f.calls
