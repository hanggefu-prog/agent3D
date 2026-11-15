"""Latest global boundary tests. Tiny artifacts are not native qualification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from local_bench import reference_latest as latest
from local_bench.reference_store import publish_reference, load_reference
from scripts import workspace_latest_reference as authority
from scripts import workspace_codesign_evaluator as evaluator
from scripts.workspace_task_selection import (LATEST_TASKS, LATEST_BASELINE, LATEST_PROFILE,
    workspace_task_id, load_registered_task)
from scripts.workspace_framework_identity import latest_framework_digest
from scripts.workspace_reference import WorkspaceReference
from global_bench.verifiers.open3dbench_codesign_verifier import _timing_row

FRAMEWORK = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def seal(run, stage, previous):
    root = run / latest.STAGE_DIRS[stage]
    files = {str(p.relative_to(root)): {"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
             "sha256": latest.sha256(p)} for p in root.rglob("*") if p.is_file() and p != root / "stage_manifest.json"}
    write(root / "stage_manifest.json", {"schema_version": 1, "stage": stage, "status": "success",
        "signature": {"upstream": "none" if previous is None else latest.sha256(run / latest.STAGE_DIRS[previous] / "stage_manifest.json")},
        "provenance": {"repo_commit": latest.SOURCE_REVISION, "repo_dirty": False,
                       "container_image_id": latest.IMAGE_ID}, "artifacts": files})


def native_fixture(run, *, cts=True):
    stages = list(latest.STAGE_DIRS) if cts else list(latest.STAGE_DIRS)[:-1]
    previous = None
    for stage in stages:
        directory = run / latest.STAGE_DIRS[stage]
        write(directory / "evidence.json", {"fixture": stage})
        if stage == "floorplan":
            write(directory / "io_metrics.json", {"seed": 1})
        if stage == "partition":
            write(directory / "partition_manifest.json", {"seed": 1001})
            write(directory / "partition_metrics.json", {"instance_count": 10, "top_count": 5,
                "bottom_count": 5, "area_balance_error": 0, "cut_net_count": 3})
        if stage == "placement":
            write(directory / "stages/02_global_place/stage_manifest.json", {"initialization_seed": 1000})
            write(directory / "metrics.json", {"dies": {d: {"missing_connection_object_count": 0}
                for d in ("bottom", "top")}, "total_object_origin_hpwl_um": 100.0})
        if stage in {"placement", "pdn", "hbt", "opt", "cts"}:
            checkpoint = {"schema_version": 1, "stage": stage, "pdk": "ng45", "dies": {}, "assembly": {}}
            for die in ("bottom", "top"):
                checkpoint["dies"][die] = {}
                for key in ("odb", "def", "verilog", "sdc"):
                    name = die + "." + key
                    (directory / name).write_text("tiny-fixture")
                    checkpoint["dies"][die][key] = name
            checkpoint["assembly"] = {"verilog": "bottom.verilog", "sdc": "bottom.sdc"}
            if stage != "placement":
                checkpoint["source_checkpoint"] = "../" + latest.STAGE_DIRS[previous] + "/checkpoint.json"
            ready = {"hbt": "ready_for_opt", "opt": "ready_for_cts", "cts": "ready_for_route"}.get(stage)
            if ready:
                checkpoint[ready] = True
            write(directory / "checkpoint.json", checkpoint)
        if stage == "opt":
            final = {"wns_setup": -1.0, "tns_setup": -2.0,
                     "sta_completeness_checks_passed": 1.0, "passive_hbt_rc": 1.0}
            write(directory / "metrics.json", {"status": "success", "algorithm": "openroad-coordinated-passive-cutpoint",
                "passes_run": 1, "iterations_run": 1, "die_passes": {"bottom": 1, "top": 1},
                "pass_record": {"pass": 1, "die_order": ["bottom", "top"],
                    "die_metrics": {"bottom": {"ok": 1}, "top": {"ok": 1}},
                    "constraints": {"bottom": {}, "top": {}}, "global_after": final},
                "final_global": final, "final_boundary_coverage": {"sta_unconstrained_endpoints": 0},
                "timing_validity": {"authoritative_timing_view": "continuous-passive-hbt",
                    "graph_completeness": "passed", "die_internal_parasitics_in_global_sta": True,
                    "hbt_is_passive_rc_in_authoritative_global_sta": True, "remote_die_load_visible_to_source_driver": True,
                    "unclocked_or_unconstrained_endpoints": 0, "missing_liberty_cells": 0, "combinational_loops": 0}})
        seal(run, stage, previous)
        previous = stage
    write(run / "run.json", {"status": "success", "design": "bp_fe", "type": "lol", "pdk": "ng45",
          "stages": {s: {"status": "success", "runtime_seconds": 1.0} for s in stages}})
    return run


@pytest.mark.parametrize("rounds", [5, 10])
def test_explicit_task_contract(rounds):
    task = load_registered_task(LATEST_TASKS[rounds], objective="timing")
    endpoint = evaluator.endpoint_for_task("timing", task)
    assert endpoint["profile"] == LATEST_PROFILE and endpoint["opt_passes"] == 1
    assert endpoint["backend_stop"] == "opt"
    assert task.budget.expensive_evaluations == rounds
    assert "15_timing_extract" not in task.evaluation.stage_ids()
    assert task.environment.image == latest.IMAGE_ID
    assert task.source.revision == latest.SOURCE_REVISION


@pytest.mark.parametrize("objective,rounds", [("thermal", 6), ("routed_thermal", 5), ("timing", 50)])
def test_no_implicit_endpoint_or_budget_widening(objective, rounds):
    with pytest.raises(ValueError):
        workspace_task_id(objective, rounds, baseline=LATEST_BASELINE)


def test_legacy_selection_unchanged():
    assert workspace_task_id("timing", 10) == "lol_partition_placement_timing_one_pass_10"


def test_actual_latest_native_stage_shape(tmp_path):
    run = native_fixture(tmp_path / "run", cts=False)
    row = _timing_row(run, expected_passes=1, latest=True)
    assert row["wns_setup_ns"] == -1.0 and row["opt_passes"] == 1
    with pytest.raises(ValueError):
        _timing_row(run, expected_passes=2, latest=True)
    with pytest.raises(ValueError):
        _timing_row(run, expected_passes=1, latest=True, reference_suffix=True)


@pytest.mark.parametrize("change", ["artifact", "upstream", "checkpoint", "extra-stage", "one-die", "not-ready"])
def test_latest_chain_mutations(tmp_path, change):
    run = native_fixture(tmp_path / "run")
    if change == "artifact":
        (run / "60_opt/bottom.def").write_text("tamper")
    elif change == "upstream":
        path = run / "70_cts/stage_manifest.json"
        data = latest.read_json(path); data["signature"]["upstream"] = "0" * 64; write(path, data)
    elif change == "extra-stage":
        path = run / "run.json"; data = latest.read_json(path)
        data["stages"]["routing"] = {"status": "success"}; write(path, data)
    else:
        path = run / "70_cts/checkpoint.json"; data = latest.read_json(path)
        if change == "checkpoint":
            data["source_checkpoint"] = "../50_hbt/checkpoint.json"
        elif change == "one-die":
            del data["dies"]["top"]
        else:
            data["ready_for_route"] = False
        write(path, data); seal(run, "cts", "opt")
    with pytest.raises(ValueError):
        authority.validate_latest_chain(run, reference_suffix=True)


def test_normalization_matches_all_frozen_launch_configs():
    plan = latest.read_json(FRAMEWORK.parent / "outputs/reference-alignment-20261004/deployment/runtime-plan-czys-r2.json")
    for case, entry in plan["cases"].items():
        actual = copy.deepcopy(entry["normalized_config"])
        expected = authority.normalized_recipe(case)
        for key in ("partition", "placement"):
            actual[key]["config"] = expected[key]["config"]
        assert actual == expected


def test_normalization_matches_external_public_config_boundary():
    # Explicit compatibility test against the checked-out flow, not a
    # production import and not a native execution.
    from open3dbench.config import load_config
    for case in latest.CASES:
        normalized = load_config(latest.profile_path(case) / "config.yaml")
        public = {k: v for k, v in normalized.items() if not k.startswith("_")}
        assert authority.normalized_recipe(case) == public


def test_initial_subject_uses_real_readable_scientific_json(tmp_path):
    runtime = SimpleNamespace(reference_baseline=LATEST_BASELINE, image=latest.IMAGE_ID,
        framework=FRAMEWORK, profile=latest.profile_path, contained=lambda p: Path(p))
    task = load_registered_task(LATEST_TASKS[10], objective="timing")
    bridge = evaluator.WorkspaceEvaluator(runtime, "bp_fe", "timing", tmp_path, task)
    assert "timing_extract" not in bridge.frontend
    bridge.initialize_subject(tmp_path)
    config = yaml.safe_load((tmp_path / evaluator.CANDIDATE_CONFIG).read_text())
    bound, receipt = evaluator.bind_frontend_config_paths(config, tmp_path, latest_framework=FRAMEWORK)
    for domain in ("partition", "placement"):
        assert latest.sha256(bound[domain]["config"]) == latest.sha256(FRAMEWORK / "local_bench/reference_configs" / (domain + ".json"))
        assert receipt[domain]["origin"] == "explicit"


def test_trusted_recipe_path_narrow_exception(tmp_path):
    base = yaml.safe_load((latest.profile_path("bp_fe") / "config.yaml").read_text())
    bound, audit = evaluator.bind_frontend_config_paths(base, tmp_path, latest_framework=FRAMEWORK)
    assert audit["partition"]["origin"] == "trusted-latest-framework"
    for bad in ("../AgenticBench/local_bench/reference_configs/partition.json", "/tmp/partition.json"):
        base["partition"]["config"] = bad
        with pytest.raises(ValueError):
            evaluator.bind_frontend_config_paths(base, tmp_path, latest_framework=FRAMEWORK)


def test_explicit_reference_factory_no_legacy_fallback(tmp_path):
    runtime = SimpleNamespace(reference_baseline=LATEST_BASELINE, reference_policy="FRESH", source_commit=latest.SOURCE_REVISION)
    with patch("scripts.workspace_framework_identity.revalidate_runtime_framework", return_value=None):
        with pytest.raises(ValueError, match="ADOPT-only"):
            WorkspaceReference(runtime, "bp_fe", "timing", tmp_path, None)


def test_framework_digest_detects_content_and_refuses_symlinks(tmp_path):
    root = tmp_path / "AgenticBench"; root.mkdir()
    (root / "control.py").write_text("1")
    before = latest_framework_digest(root)
    (root / "control.py").write_text("2")
    assert before != latest_framework_digest(root)
    (root / "alias").symlink_to("control.py")
    with pytest.raises(ValueError):
        latest_framework_digest(root)


def test_no_flow_imports_in_new_authority():
    assert "from open3dbench" not in Path(authority.__file__).read_text()


def standalone_control_plane(tmp_path, body, *args):
    # -I excludes CWD/PYTHONPATH; reject even an accidental installed native
    # module. This reproduces the independently deployed control plane.
    preamble = """
import importlib.abc, json, sys
from pathlib import Path
from types import SimpleNamespace
class NoPhysicalFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'open3dbench' or fullname.startswith('open3dbench.'):
            raise AssertionError('Native flow import forbidden: ' + fullname)
sys.meta_path.insert(0, NoPhysicalFlow())
sys.path.insert(0, sys.argv[1])
def guard(event, values):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'socket.connect'):
        raise AssertionError('No process/network dispatch in reference consumer: ' + event)
sys.addaudithook(guard)
from scripts.workspace_reference import WorkspaceReference
from local_bench import reference_latest as latest
from scripts.workspace_task_selection import load_registered_task
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", preamble + body,
        str(FRAMEWORK), *(str(a) for a in args)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_standalone_factory_keeps_family_rejections_without_native_import(tmp_path):
    result = standalone_control_plane(tmp_path, """
runtime = SimpleNamespace(reference_baseline=latest.BASELINE_ID, reference_policy='FRESH', source_commit=latest.SOURCE_REVISION)
try:
    WorkspaceReference(runtime, 'bp_fe', 'timing', Path('.'), None)
except ValueError as error:
    assert 'ADOPT-only' in str(error)
else:
    raise AssertionError('Latest FRESH must fail closed')
legacy = SimpleNamespace(reference_baseline='reference_20260923', reference_policy='EXTEND', reference_record=None)
task = load_registered_task('lol_partition_placement_timing_one_pass', objective='timing')
try:
    WorkspaceReference(legacy, 'bp_fe', 'timing', Path('.'), task)
except ValueError as error:
    assert 'EXTEND requires an explicit qualified parent' in str(error)
else:
    raise AssertionError('Legacy EXTEND must retain its missing-parent refusal')
assert 'local_bench.reference_evaluated' not in sys.modules
print(json.dumps({'latest_fresh_rejected': True, 'legacy_extend_rejected': True, 'native_imports': 0}))
""")
    assert result == {"latest_fresh_rejected": True, "legacy_extend_rejected": True, "native_imports": 0}


@pytest.mark.parametrize("proof_present", [True, False])
def test_prepare_requires_and_passes_independent_execution_proof(tmp_path, monkeypatch, proof_present):
    from scripts.codesign_runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.reference_baseline, runtime.image = LATEST_BASELINE, latest.IMAGE_ID
    runtime.source, runtime.source_commit, runtime.framework = tmp_path / "source", latest.SOURCE_REVISION, FRAMEWORK
    runtime.contained = lambda p: Path(p)
    run = tmp_path / "reference-run"; run.mkdir()
    config = authority.normalized_recipe("bp_fe")
    for domain in ("partition", "placement"):
        path = tmp_path / (domain + ".json"); write(path, {"fixture": domain})
        config[domain]["config"] = str(path)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    write(run / "reference-result.json", {"actual_docker_image_id": latest.IMAGE_ID})
    runtime.reference_run = run
    proof = tmp_path / "proof.json"; write(proof, {"fixture": True})
    binding = tmp_path / "binding.json"; write(binding, {"fixture": True})
    runtime.reference_execution_proofs = {"bp_fe": {"path": str(proof), "sha256": latest.sha256(proof)}} if proof_present else {}
    monkeypatch.setattr(latest, "validate_execution_proof", lambda *a: {"bindings": {"receipt": {"path": str(binding)}}})
    monkeypatch.setattr(latest, "validate_reference_bundle", lambda *a, **k: {})
    calls = []
    runtime.execute = lambda *a: calls.append(a) or 0
    if not proof_present:
        with pytest.raises(ValueError, match="execution proof"):
            runtime.prepare("bp_fe", tmp_path / "new-bundle")
        assert calls == []
    else:
        runtime.prepare("bp_fe", tmp_path / "new-bundle")
        command = calls[0][0]
        assert command[command.index("--execution-proof") + 1] == str(proof.resolve())
        assert command[command.index("--execution-proof-sha256") + 1] == latest.sha256(proof)
        assert proof.resolve() in runtime.latest_reference_mounts
        assert binding.resolve() in runtime.latest_reference_mounts
        assert calls[0][3]["native_eda_dispatches"] == calls[0][3]["model_calls"] == 0


@pytest.fixture
def producer_fixture(tmp_path, monkeypatch):
    from test_reference_latest import fixture as grt_fixture
    run = native_fixture(tmp_path / "run")
    cts = (run / "70_cts/checkpoint.json").read_bytes()
    grt_fixture(run)
    (run / "70_cts/checkpoint.json").write_bytes(cts)
    checkpoint_path = run / "80_route/global_route_checkpoint.json"
    checkpoint = latest.read_json(checkpoint_path)
    checkpoint["source_checkpoint_sha256"] = latest.sha256(run / "70_cts/checkpoint.json")
    write(checkpoint_path, checkpoint)
    write(run / "reference-result.json", {"status": "success", "endpoint": "global-route", "fresh_input": True,
        "source_commit": latest.SOURCE_REVISION, "image_id": latest.IMAGE_ID, "drt_dispatches": 0,
        "executed_stages": [*latest.STAGE_DIRS, "global_route"], "model_calls": 0, "candidate_finals": 0,
        "registry_publications": 0, "grt_metrics": latest.read_json(run / "80_route/grt_metrics.json")})
    config = authority.normalized_recipe("bp_fe")
    for domain in ("partition", "placement"):
        config[domain]["config"] = str(FRAMEWORK / "local_bench/reference_configs" / (domain + ".json"))
    config["run_dir"] = str(run)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    bundle = tmp_path / "bundle"; bundle.mkdir()
    receipt = bundle / "receipt.json"
    proof = tmp_path / "execution-proof.json"
    bindings = {}
    for key in ("controller_plan", "dispatch_receipt", "container_result", "runtime_identity"):
        path = tmp_path / (key + ".json"); write(path, {"fixture": key})
        bindings[key] = {"path": str(path), "sha256": latest.sha256(path)}
    write(proof, {"bindings": bindings})
    write(receipt, {"source_run": str(run), "native_config_sha256": latest.sha256(run / "config.yaml"),
                   "runtime_identity": {"execution_proof": str(proof)},
                   "stage_manifest_sha256": {s: latest.sha256(run / d / "stage_manifest.json") for s, d in latest.STAGE_DIRS.items()}})
    write(bundle / "benchmark_input.json", {"latest_reference_receipt": {"path": str(receipt), "sha256": latest.sha256(receipt)}})
    # Bundle schema is tested by the shared importer suite. All production
    # authority stage/metric/config/GRT/registry checks below remain real.
    monkeypatch.setattr(latest, "validate_reference_bundle", lambda *a, **k: {"physical_input_sha256": {"input": "f" * 64}})
    task = load_registered_task(LATEST_TASKS[10], objective="timing")
    producer = {"kind": authority.KIND, "case": "bp_fe", "task_id": task.task_id,
                "benchmark_input": str(bundle), "run_root": str(run)}
    return SimpleNamespace(run=run, bundle=bundle, receipt=receipt, task=task, producer=producer,
                           contract=authority.contract_for("bp_fe", task, bundle))


def test_real_authority_publication_and_read_only_adoption(producer_fixture, tmp_path):
    f = producer_fixture
    qualified = authority.qualify_producer(f.producer)
    assert qualified["baseline"]["raw_metrics"]["opt_passes"] == 1
    assert qualified["grt_qualification"]["status"] == "passed"
    assert qualified["signoff"] is False
    assert len([r for r in qualified["evidence"] if r["role"].startswith("execution-binding:")]) == 4
    path = publish_reference(tmp_path / "store", contract=f.contract, origin="FRESH",
                             producer=f.producer, validate=authority.qualify_producer)
    record = load_reference(path, contract=f.contract, validate=authority.qualify_producer)
    runtime = SimpleNamespace(reference_baseline=LATEST_BASELINE, reference_policy="ADOPT", reference_record=path,
                              source_commit=latest.SOURCE_REVISION, image=latest.IMAGE_ID)
    with patch("scripts.workspace_framework_identity.revalidate_runtime_framework", return_value=None):
        selected = WorkspaceReference(runtime, "bp_fe", "timing", f.bundle, f.task)
        result = selected.obtain(None, tmp_path / "adoption")
    assert result["reference_id"] == record["reference_id"]
    assert result["metrics"]["combined_score"] == 1.0
    assert selected.evaluator_invocations == result["new_reference_native_attempts"] == 0
    assert selected.policy == selected.to_dict()["policy"] == "ADOPT"
    assert (tmp_path / "adoption/reference_adoption.json").is_file()


@pytest.mark.parametrize("method", ["codex", "agentic3d"])
def test_real_latest_reference_reaches_driver_subject_boundary(producer_fixture, tmp_path, monkeypatch, method):
    """Exercise both real drivers and the real reference adapter before any model/EDA."""
    from scripts import run_workspace_codesign, run_workspace_agentic3d
    module = run_workspace_codesign if method == "codex" else run_workspace_agentic3d
    f = producer_fixture
    record = publish_reference(tmp_path / "driver-store", contract=f.contract, origin="FRESH",
                               producer=f.producer, validate=authority.qualify_producer)
    runtime = SimpleNamespace(reference_baseline=LATEST_BASELINE, reference_policy="ADOPT",
        reference_record=record, source_commit=latest.SOURCE_REVISION, image=latest.IMAGE_ID,
        source=tmp_path / "unused-source", cpus=16, memory_gb=48, gpus=[0, 3],
        min_gpu_free_mb=2048, profile=lambda case: latest.profile_path(case),
        contained=lambda path: Path(path))
    monkeypatch.setattr(module, "select_driver_task", lambda *args, **kwargs: f.task)
    monkeypatch.setattr(module, "independent_subject", lambda source, target: target.mkdir())
    monkeypatch.setattr("scripts.workspace_framework_identity.revalidate_runtime_framework", lambda *a: None)

    class BoundaryEvaluator:
        def __init__(self, *args):
            pass
        def evaluate(self, *args, **kwargs):
            raise AssertionError("ADOPT must not run a reference evaluation")
        def initialize_subject(self, subject):
            raise RuntimeError("Reached subject initialization boundary")

    monkeypatch.setattr(evaluator, "WorkspaceEvaluator", BoundaryEvaluator)
    if method == "agentic3d":
        monkeypatch.setattr(module, "WorkspaceEvaluator", BoundaryEvaluator)
    root = tmp_path / "driver-output"
    run = (module.run_workspace_experiment if method == "codex"
           else module.run_workspace_agentic3d_experiment)
    with pytest.raises(RuntimeError, match="Reached subject initialization boundary"):
        run(runtime, "bp_fe", "timing", f.bundle, root, "/unused-model")
    config = json.loads((root / "run_config.json").read_text())
    state = json.loads((root / "status.json").read_text())
    assert config["reference_policy"] == config["reference_selection"]["policy"] == "ADOPT"
    assert state["reference_adoptions"] == 1
    assert state["reference_evaluations"] == state["final_evaluations"] == 0
    assert (root / "results/reference/reference_adoption.json").is_file()


@pytest.mark.parametrize("change", ["native-config", "source", "image", "manifest-binding"])
def test_authority_rejects_relabelled_science_or_provenance(producer_fixture, change):
    f = producer_fixture
    if change == "native-config":
        config = yaml.safe_load((f.run / "config.yaml").read_text())
        config["cts"]["target_skew_ns"] = 0.2
        (f.run / "config.yaml").write_text(yaml.safe_dump(config))
        receipt = latest.read_json(f.receipt); receipt["native_config_sha256"] = latest.sha256(f.run / "config.yaml")
        write(f.receipt, receipt)
    elif change == "manifest-binding":
        receipt = latest.read_json(f.receipt); receipt["stage_manifest_sha256"]["opt"] = "0" * 64; write(f.receipt, receipt)
    else:
        path = f.run / "70_cts/stage_manifest.json"; data = latest.read_json(path)
        data["provenance"]["repo_commit" if change == "source" else "container_image_id"] = "old"
        write(path, data)
        receipt = latest.read_json(f.receipt); receipt["stage_manifest_sha256"]["cts"] = latest.sha256(path); write(f.receipt, receipt)
    with pytest.raises(ValueError):
        authority.qualify_producer(f.producer)


def test_candidate_scores_only_opt_and_preserves_frozen_backend(producer_fixture, tmp_path):
    f = producer_fixture
    path = publish_reference(tmp_path / "store", contract=f.contract, origin="FRESH", producer=f.producer, validate=authority.qualify_producer)
    candidate = native_fixture(tmp_path / "candidate", cts=False)
    config = authority.normalized_recipe("bp_fe")
    (candidate / "config.yaml").write_text(yaml.safe_dump(config))
    contract = tmp_path / "evaluation-contract.json"
    write(contract, {"task_digest": f.task.digest(), "source_revision": latest.SOURCE_REVISION,
        "image": latest.IMAGE_ID, "evaluation_endpoint": {"profile": LATEST_PROFILE}, "case": "bp_fe",
        "benchmark_input": str(f.bundle)})
    submission = {"task_id": f.task.task_id, "task_digest": f.task.digest(), "base_revision": latest.SOURCE_REVISION}
    result = authority.score_with_latest_record(candidate, f.run, path, contract,
        mode="timing", profile=LATEST_PROFILE, submission=submission)
    assert result["combined_score"] == 1.0
    config["opt"]["threads"] = 8
    (candidate / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="frozen physical"):
        authority.score_with_latest_record(candidate, f.run, path, contract,
            mode="timing", profile=LATEST_PROFILE, submission=submission)


def test_real_importer_global_prefix_and_authority_boundary(tmp_path):
    """Real artifact/import/store APIs; tiny fixture is not physical qualification."""
    from test_reference_latest import full_reference, execution_proof
    run = tmp_path / "native"
    full_reference(run)
    donor = native_fixture(tmp_path / "metric-fixture")
    for name in ("10_floorplan/io_metrics.json", "20_partition/partition_metrics.json",
                 "30_placement/stages/02_global_place/stage_manifest.json", "30_placement/metrics.json",
                 "60_opt/metrics.json"):
        target = run / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((donor / name).read_bytes())
    path = run / "20_partition/partition_manifest.json"
    partition = latest.read_json(path); partition["seed"] = 1001; write(path, partition)
    previous = None
    for stage in latest.STAGE_DIRS:
        seal(run, stage, previous)
        if stage == "partition":
            path = run / "20_partition/stage_manifest.json"
            row = latest.read_json(path); row["method"] = "tp-gnn-flat"
            row["artifacts"]["method_manifest"] = row["artifacts"]["partition_manifest.json"]
            write(path, row)
        previous = stage
    completion_path = run / "reference-result.json"
    completion = latest.read_json(completion_path)
    completion.update(model_calls=0, candidate_finals=0, registry_publications=0,
                      grt_metrics=latest.read_json(run / "80_route/grt_metrics.json"))
    write(completion_path, completion)
    proof = execution_proof(tmp_path, run)
    bundle = tmp_path / "bundle"
    with patch("local_bench.reference_local_placement._source_revision", return_value=latest.SOURCE_REVISION):
        latest.adopt_latest_input(run, bundle, "bp_fe", FRAMEWORK.parent, **proof)
    copied = tmp_path / "candidate-prefix"
    evaluator.copy_frozen_prefix(bundle / "run", copied)
    assert set(latest.read_json(copied / "run.json")["stages"]) == {"input", "floorplan"}
    assert latest.sha256(copied / "00_input/stage_manifest.json") == latest.sha256(run / "00_input/stage_manifest.json")
    task = load_registered_task(LATEST_TASKS[10], objective="timing")
    producer = {"kind": authority.KIND, "case": "bp_fe", "task_id": task.task_id,
                "benchmark_input": str(bundle), "run_root": str(run)}
    contract = authority.contract_for("bp_fe", task, bundle)
    record = publish_reference(tmp_path / "store", contract=contract, origin="FRESH",
                               producer=producer, validate=authority.qualify_producer)
    qualified = load_reference(record, contract=contract, validate=authority.qualify_producer)
    assert qualified["qualification"]["baseline"]["raw_metrics"]["opt_passes"] == 1
    result = standalone_control_plane(tmp_path, """
runtime = SimpleNamespace(reference_baseline=latest.BASELINE_ID, reference_policy='ADOPT',
    source_commit=latest.SOURCE_REVISION, image=latest.IMAGE_ID, reference_record=Path(sys.argv[3]))
task = load_registered_task('lol_partition_placement_timing_20261004_10', objective='timing')
consumer = WorkspaceReference(runtime, 'bp_fe', 'timing', Path(sys.argv[2]), task)
assert type(consumer).__name__ == 'LatestWorkspaceReference'
feedback = consumer.obtain(None, Path(sys.argv[4]))
assert 'local_bench.reference_evaluated' not in sys.modules
assert feedback['new_reference_native_attempts'] == consumer.evaluator_invocations == 0
assert feedback['metrics']['combined_score'] == 1.0
print(json.dumps({'consumer': type(consumer).__name__, 'score': feedback['metrics']['combined_score'], 'native_dispatches': 0}))
""", bundle, record, tmp_path / "standalone-adoption")
    assert result == {"consumer": "LatestWorkspaceReference", "score": 1.0, "native_dispatches": 0}
