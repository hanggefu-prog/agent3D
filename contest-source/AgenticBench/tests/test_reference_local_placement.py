"""Offline native evidence fixtures only; no Docker, models or physical tools."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from local_bench import reference_local_placement as local
from local_bench import reference_catalog as catalog
from local_bench.evaluation import _write_resolved_config, _dreamplace_global_placement_command
from local_bench.model import EvaluationResult
from local_bench.reference_store import file_sha, load_reference
from scripts.local_reference import LocalPlacementReference
from scripts import run_local_task_smoke as smoke
from test_reference_placement_evidence import native_params, HEADER, row


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("stage", ["00_input", "10_floorplan"])
def test_latest_path_rebase_checks_every_physical_byte_and_other_parameter(tmp_path, stage):
    from local_bench.workspace import _rebase_copied_design_params
    baseline, run = tmp_path / "baseline", tmp_path / "run"
    relative = stage + "/design_params.json"
    write(baseline / relative, {"def_input": "old/design.def", "verilog_input": "old/design.v", "num_threads": 4})
    for name in ("design.def", "design.v", "design.sdc"):
        (baseline / stage / name).write_text("frozen " + name)
    frozen = {str(p.relative_to(baseline)): file_sha(p) for p in baseline.rglob('*') if p.is_file()}
    shutil.copytree(baseline, run)
    _rebase_copied_design_params(run / stage)
    local._validate_copied_input_params(baseline, run, relative, frozen)
    expected = json.loads((run / relative).read_text())
    for key, value in [('num_threads', 8), ('def_input', '/unbound/elsewhere')]:
        write(run / relative, {**expected, key: value})
        with pytest.raises(ValueError):
            local._validate_copied_input_params(baseline, run, relative, frozen)
    write(run / relative, expected)
    (run / stage / 'design.def').write_text('changed physical design')
    with pytest.raises(ValueError):
        local._validate_copied_input_params(baseline, run, relative, frozen)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    bundle = batch / "bundle"
    baseline = bundle / "run"
    write(bundle / "benchmark_input.json", {"fixture": "frozen-input"})
    params = {"num_threads": 4, "deterministic_flag": 0, "use_bb": 0}
    write(baseline / "10_floorplan/design_params.json", params)
    partition = baseline / "20_partition/partition_assignment.csv"
    partition.parent.mkdir(parents=True)
    partition.write_text("inst_name,die\nu1,bottom\nu2,top\n")
    write(baseline / "20_partition/die_regions.json", {"fixture": "regions"})
    report = {"method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
              "config": {"decoder": "global-2way"}}
    write(baseline / "20_partition/tp_gnn_report.json", report)
    method = baseline / "20_partition/partition_manifest.json"
    write(method, {"method": "tp-gnn-flat", "method_report": report})
    write(baseline / "20_partition/stage_manifest.json", {"method": "tp-gnn-flat", "status": "success",
        "artifacts": {"method_manifest": {"path": method.name, "bytes": method.stat().st_size, "sha256": file_sha(method)}}})
    frozen = {str(path.relative_to(baseline)): file_sha(path) for path in baseline.rglob("*") if path.is_file()}
    profile = local.baseline.profile_path("bp_fe") / "config.yaml"
    input_report = {"config": str(profile), "run_dir": str(baseline)}
    monkeypatch.setattr(local, "validate_benchmark_input", lambda *a, **k: input_report)
    monkeypatch.setattr(smoke, "validate_benchmark_input", lambda *a, **k: input_report)
    monkeypatch.setattr(local.baseline, "validate_reference_bundle", lambda *a, **k: {
        "physical_input_sha256": frozen, "reference_root": str(tmp_path / "archive")})
    monkeypatch.setattr(local, "_source_revision", lambda *a, **k: "1" * 40)
    def contained(path):
        value = Path(path).resolve()
        value.relative_to(batch)
        return value
    runtime = SimpleNamespace(source=local.SOURCE, framework=local.FRAMEWORK, batch=batch,
        source_commit="1" * 40, image=sorted(local.baseline.IMAGE_IDS)[0], cpus=16, memory_gb=48,
        share_gpu=True, min_gpu_free_mb=2048, contained=contained)
    return SimpleNamespace(batch=batch, bundle=bundle, baseline=baseline, runtime=runtime)


def native_fixture(env, task, evaluation):
    source, spec, candidate, recipe, frozen, baseline, manifest = local._context(local.SOURCE, task, "bp_fe", env.bundle)
    selection = LocalPlacementReference(env.runtime, spec, "bp_fe", env.bundle, evaluation.parents[1], selected_policy="FRESH")
    candidate_root = evaluation.parent / "candidate"
    write(candidate_root / "candidate.json", candidate)
    purpose_path = selection.prepare_fresh(evaluation, candidate_root)
    purpose = json.loads(purpose_path.read_text())
    contract = purpose["contract"]
    write(evaluation.parent / "runtime-source.json", {"schema_version": 1, "image": env.runtime.image,
        "runtime_root": "/opt/dreamplace/dreamplace", "reference_purpose_sha256": file_sha(purpose_path),
        "python_sha256": contract["physical"]["runtime_python_sha256"]})
    write(evaluation.parent / (evaluation.name + ".attempt.json"), {"schema_version": 2, "state": "exited", "exit_code": 0,
        "evaluator_process_started": True, "contract": {"kind": "local-task-smoke-evaluation", "task": task,
            "candidate_sha256": file_sha(candidate_root / "candidate.json"), "source_commit": env.runtime.source_commit,
            "image": env.runtime.image, "reference": None, "create_routing_reference": False,
            "reference_purpose": "FRESH", "reference_purpose_sha256": file_sha(purpose_path)}})
    run = evaluation / "run"
    shutil.copytree(baseline, run)
    config = _write_resolved_config(spec, candidate_root, evaluation / "resolved_config.yaml", candidate)
    write(evaluation / "harness_inputs/design_params.json", json.loads((run / "10_floorplan/design_params.json").read_text()))
    workspace = evaluation / "workspace/Open3DBench-V2.0"
    wrapper = Path("Place-3D/src/placement/concurrent_dreamplace.py")
    (workspace / wrapper).parent.mkdir(parents=True)
    shutil.copyfile(source / wrapper, workspace / wrapper)
    stage = run / local.STAGE
    write(stage / "params.json", native_params())
    (stage / "run.log").write_text(HEADER + (row() if task == local.DYNAMIC_TASK else "")
        + "[INFO] iteration 1, wHPWL 1000, Overflow [0.1, 0.2], MaxDensity [0.6, 0.7]\n")
    (stage / "placement.gp.def").write_text("VERSION 5.8 ;\nCOMPONENTS 2 ;\n"
        "- u1 INV + PLACED ( 100 200 ) N ;\n- u2 INV + PLACED ( 300 200 ) N ;\nEND COMPONENTS\nEND DESIGN\n")
    overlay = evaluation / "runtime/dreamplace-install"
    runtime_root = overlay / "dreamplace" if task == local.DYNAMIC_TASK else Path("/opt/dreamplace/dreamplace")
    if task == local.DYNAMIC_TASK:
        patched = local._expected_overlay(source, candidate)
        runtime_root.mkdir(parents=True)
        for name in local.RUNTIME_MODULES:
            relative = str(local.DREAMPLACE / name)
            (runtime_root / name).write_text(patched.get(relative, (source / relative).read_text()))
    write(stage / "stage_manifest.json", {"stage": "02_global_place", "stage_dir": str(stage),
        "dreamplace_root": str(runtime_root), "partition_dir": str(run / "20_partition"),
        "params": str(stage / "params.json"), "log": str(stage / "run.log"), "returncode": 0,
        "target_density": 0.6, "command": ["/usr/bin/python3", str(runtime_root / "Placer.py"), str(stage / "params.json")]})
    command, cwd = _dreamplace_global_placement_command(spec, config, workspace, run,
        "/usr/bin/python3", base_params_override=evaluation / "harness_inputs/design_params.json")
    artifacts = {"task": task, "run_dir": str(run), "workspace": str(workspace),
        "candidate": str(candidate_root), "output_dir": str(evaluation), "exit_code": 0, "command": command,
        "native_flow_execution": [{"command": command, "cwd": str(cwd), "process_started": True,
                                   "returncode": 0, "status": "completed", "wall_seconds": 12.5}]}
    if task == local.DYNAMIC_TASK:
        artifacts["dreamplace_runtime"] = str(overlay)
    write(evaluation / "evaluation_result.json", {"metrics": {"runs_successfully": 1, "combined_score": 999,
        "placement_native_hpwl": 999999}, "artifacts": artifacts})
    producer = {"kind": "native-local-placement-v1", "task_id": task, "case": "bp_fe",
                "source_root": str(source), "evaluation_root": str(evaluation), "benchmark_input": str(env.bundle)}
    return SimpleNamespace(selection=selection, producer=producer, evaluation=evaluation, run=run, spec=spec)


def test_native_fixture_workspace_matches_real_evaluator_dry_run(setup, monkeypatch):
    from local_bench import evaluation as evaluator

    fixture = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "fixture/reference/result")
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", setup.runtime.image)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    output = setup.batch / "dry-run"
    result = evaluator.evaluate_task(fixture.spec.root, fixture.evaluation.parent / "candidate",
        local.SOURCE, output, baseline_run=setup.baseline, dry_run=True)
    assert result.metrics.get("dry_run") == 1, result.artifacts
    relative = Path(result.artifacts["workspace"]).relative_to(output)
    assert relative == Path("workspace/Open3DBench-V2.0")
    recorded = json.loads((fixture.evaluation / "evaluation_result.json").read_text())
    assert Path(recorded["artifacts"]["workspace"]).relative_to(fixture.evaluation) == relative


@pytest.mark.parametrize("task", [local.DYNAMIC_TASK])
def test_native_fresh_register_then_adopt_reads_actual_metrics(setup, task):
    f = native_fixture(setup, task, setup.batch / "first/reference/result")
    payload = catalog.qualify_producer(f.producer)
    assert payload["baseline"]["raw_metrics"]["placement_native_hpwl"] == 1000
    assert payload["executed_native_stages"] == []
    assert payload["source_contract"] == f.selection.contract["physical"]
    f.selection.publish(f.evaluation)
    reused = LocalPlacementReference(setup.runtime, f.spec, "bp_fe", setup.bundle, setup.batch / "second")
    result = reused.obtain_adopted(setup.batch / "second/reference/adoption")
    assert result.metrics["combined_score"] == 1.0
    assert result.artifacts["new_reference_native_attempts"] == 0
    assert result.artifacts["run_dir"] == str(f.run)
    assert result.artifacts["historical_native_runtime"]["seconds"] == 12.5
    (f.run / local.STAGE / "run.log").write_text("passed=true\n")
    with pytest.raises(ValueError):
        load_reference(reused.record_path, contract=reused.contract, validate=catalog.qualify_producer)


@pytest.mark.parametrize("fault", ["candidate", "purpose", "ledger", "audit", "runtime_path", "command", "hook",
                                  "controls", "recipe", "prefix", "def", "nan", "fake_pass"])
def test_local_qualification_rejects_bad_provenance_or_native_evidence(setup, fault):
    f = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "bad/reference/result")
    root = f.evaluation
    parent = root.parent
    if fault == "candidate":
        path = parent / "candidate/candidate.json"
        value = json.loads(path.read_text()); value["notes"] = "candidate, not initial reference"; write(path, value)
    elif fault == "purpose":
        path = parent / "reference-purpose.json"
        value = json.loads(path.read_text()); value["reference_purpose"] = "CANDIDATE"; write(path, value)
    elif fault == "ledger":
        path = parent / "result.attempt.json"
        value = json.loads(path.read_text()); value["evaluator_process_started"] = False; write(path, value)
    elif fault == "audit":
        path = parent / "runtime-source.json"
        value = json.loads(path.read_text()); value["python_sha256"]["BasicPlace.py"] = "0" * 64; write(path, value)
    elif fault == "runtime_path":
        path = f.run / local.STAGE / "stage_manifest.json"
        value = json.loads(path.read_text()); value["dreamplace_root"] = "/opt/dreamplace/dreamplace"; write(path, value)
    elif fault == "command":
        path = root / "evaluation_result.json"
        value = json.loads(path.read_text()); value["artifacts"]["native_flow_execution"][0]["command"][0] = "/usr/local/bin/python"; write(path, value)
    elif fault == "hook":
        (root / "runtime/dreamplace-install/dreamplace/NonLinearPlace.py").write_text("# wrong active source\n")
    elif fault == "controls":
        path = f.run / local.STAGE / "run.log"
        path.write_text(path.read_text().replace("shared=1", "shared=2"))
    elif fault == "recipe":
        path = f.run / local.STAGE / "params.json"
        value = json.loads(path.read_text()); value["num_threads"] = 32; write(path, value)
    elif fault == "prefix":
        (f.run / "20_partition/partition_assignment.csv").write_text("inst_name,die\nu1,top\n")
    elif fault == "def":
        (f.run / local.STAGE / "placement.gp.def").write_text("COMPONENTS 1 ;\n- u1 INV + PLACED ( 0 0 ) N ;\nEND COMPONENTS\n")
    elif fault == "nan":
        path = f.run / local.STAGE / "run.log"
        path.write_text(path.read_text().replace("[0.1, 0.2]", "[NaN, 0.2]"))
    else:
        (f.run / local.STAGE / "run.log").write_text("passed=true\n")
    with pytest.raises((ValueError, RuntimeError)):
        catalog.qualify_producer(f.producer)


@pytest.mark.parametrize("task", [local.DYNAMIC_TASK])
def test_default_adopt_missing_reference_stops_before_transport_or_model(setup, task):
    def forbidden(*a, **k):
        raise AssertionError("No transport/model may be initialized")
    output = setup.batch / "no-reference"
    with pytest.raises(ValueError, match="no automatic FRESH"):
        smoke.run_smoke(setup.runtime, task=task, benchmark_input=setup.bundle, output=output,
                        provider_factory=forbidden, transport_factory=forbidden)
    assert not output.exists()


def test_local_archive_rejected_without_claiming_source_equivalence(setup):
    _, spec, *_ = local._context(local.SOURCE, local.DYNAMIC_TASK, "bp_fe", setup.bundle)
    record = setup.batch / "archive-record.json"
    write(record, {"producer": {"kind": "frozen-20260923-v1"}})
    with pytest.raises(ValueError, match="equal configuration is not algorithm equivalence"):
        LocalPlacementReference(setup.runtime, spec, "bp_fe", setup.bundle, setup.batch / "attempt", record=record)


def test_local_extend_is_explicitly_unsupported(setup):
    _, spec, *_ = local._context(local.SOURCE, local.DYNAMIC_TASK, "bp_fe", setup.bundle)
    with pytest.raises(ValueError, match="no missing suffix"):
        LocalPlacementReference(setup.runtime, spec, "bp_fe", setup.bundle, setup.batch / "attempt", selected_policy="EXTEND")


@pytest.mark.parametrize("mutation", ["whitespace", "publication_time"])
def test_selected_record_bytes_are_pinned_before_adoption(setup, mutation):
    f = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "first/reference/result")
    f.selection.publish(f.evaluation)
    selected = LocalPlacementReference(setup.runtime, f.spec, "bp_fe", setup.bundle, setup.batch / "next")
    path = selected.record_path
    saved_sha = selected.to_dict()["record_sha256"]
    path.chmod(0o644)
    if mutation == "whitespace":
        path.write_text(path.read_text() + "\n")
    else:
        value = json.loads(path.read_text()); value["published_at"] = "changed"; write(path, value)
    assert selected.to_dict()["record_sha256"] == saved_sha != file_sha(path)
    with pytest.raises(ValueError, match="bytes changed"):
        selected.obtain_adopted(setup.batch / "next/reference/adoption")
    assert not (setup.batch / "next").exists()


@pytest.mark.parametrize("mutation", ["record", "native_log", "wrong_reference"])
def test_each_candidate_requalifies_the_pinned_reference_before_dispatch(setup, mutation):
    f = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "first/reference/result")
    f.selection.publish(f.evaluation)
    runtime = setup.runtime
    runtime.local_reference_selection = f.selection
    runtime.adopted_reference_run = f.run
    runtime.execute = lambda *a, **k: pytest.fail("Invalid reference cannot dispatch any evaluator")
    reference = f.run
    if mutation == "record":
        f.selection.record_path.chmod(0o644)
        f.selection.record_path.write_text(f.selection.record_path.read_text() + "\n")
    elif mutation == "native_log":
        (f.run / local.STAGE / "run.log").write_text("passed=true\n")
    else:
        reference = setup.baseline
    transport = smoke.DockerEvaluationTransport(runtime)
    with pytest.raises(ValueError):
        transport(f.spec.root, f.evaluation.parent / "candidate", local.SOURCE,
                  setup.batch / "candidate/result", baseline_run=setup.baseline, reference_run=reference)
    assert transport.attempts == []


@pytest.mark.parametrize("task", [local.DYNAMIC_TASK])
def test_smoke_fresh_then_adopt_preserves_five_finals_and_separate_reference(setup, task):
    env = setup
    class FixtureTransport:
        def __init__(self, runtime):
            self.receipts, self.attempts = [], []
        def __call__(self, task_dir, candidate_dir, source, output, **kwargs):
            if kwargs.get("reference_run") is None:
                f = native_fixture(env, task, output)
                result = EvaluationResult(**json.loads((f.evaluation / "evaluation_result.json").read_text()))
            else:
                env.runtime.local_reference_selection.verify_selected()
                result = EvaluationResult({"combined_score": 1.0, "runs_successfully": 1}, {})
                write(output / "evaluation_result.json", result.to_dict())
            self.attempts.append({"output_dir": str(output), "evaluator_dispatch_attempted": True,
                                  "native_flow_started": True})
            self.receipts.append({"path": str(output / "evaluation_result.json")})
            return result
    _, spec, candidate, *_ = local._context(local.SOURCE, task, "bp_fe", env.bundle)
    initial = local.initial_program_text(spec, candidate)
    class FixtureProvider:
        def __init__(self, *args):
            self.calls = self.completed_calls = 0
        def draw_sample(self, messages):
            self.calls += 1; self.completed_calls += 1
            return initial
    first = smoke.run_smoke(env.runtime, task=task, benchmark_input=env.bundle,
        output=env.batch / "first", reference_policy="FRESH",
        provider_factory=FixtureProvider, transport_factory=FixtureTransport)
    assert first["state"] == "completed", first["error"]
    assert first["reference_evaluations"] == first["reference_physical_attempts"] == 1
    assert first["candidate_finals"] == first["candidate_physical_attempts"] == first["model_requests"] == 5
    qualification = json.loads((env.batch / "first/reference/qualification.json").read_text())
    assert qualification["evaluation"]["metrics"]["combined_score"] == 1.0  # Not the cached 999.
    assert len(list((env.batch / "reference-store/records").glob("*.json"))) == 1
    second = smoke.run_smoke(env.runtime, task=task, benchmark_input=env.bundle,
        output=env.batch / "second", provider_factory=FixtureProvider, transport_factory=FixtureTransport)
    assert second["state"] == "completed", second["error"]
    assert second["reference_policy"] == "ADOPT"
    assert second["reference_evaluations"] == second["reference_physical_attempts"] == 0
    assert second["candidate_finals"] == second["candidate_physical_attempts"] == second["model_requests"] == 5
    assert len(list((env.batch / "reference-store/records").glob("*.json"))) == 1


@pytest.mark.parametrize("fault", [None, "source", "image"])
def test_installed_source_audit_is_measured_and_fails_before_native_launch(setup, monkeypatch, fault):
    f = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "first/reference/result")
    installed = setup.batch / "installed/dreamplace"
    installed.mkdir(parents=True)
    for name in local.RUNTIME_MODULES:
        shutil.copyfile(local.SOURCE / local.DREAMPLACE / name, installed / name)
    if fault == "source":
        (installed / "BasicPlace.py").write_text("# incompatible installed initializer\n")
    regular = local._regular
    def resolve(path):
        path = Path(path)
        if path.parent == Path("/opt/dreamplace/dreamplace"):
            return regular(installed / path.name)
        return regular(path)
    monkeypatch.setattr(local, "_regular", resolve)
    monkeypatch.setenv("OPEN3DBENCH_IMAGE_ID", "sha256:" + "0" * 64 if fault == "image" else setup.runtime.image)
    output = setup.batch / "audit-measurement.json"
    if fault:
        with pytest.raises(ValueError, match="source/hook|image differs"):
            local.audit_installed_runtime(f.evaluation.parent / "reference-purpose.json", output)
        assert not output.exists()
    else:
        local.audit_installed_runtime(f.evaluation.parent / "reference-purpose.json", output)
        measured = json.loads(output.read_text())
        assert measured["python_sha256"] == f.selection.contract["physical"]["runtime_python_sha256"]


def test_fresh_transport_bootstrap_precedes_unchanged_evaluator_command(setup):
    f = native_fixture(setup, local.DYNAMIC_TASK, setup.batch / "first/reference/result")
    runtime = setup.runtime
    runtime.reference_purpose_receipt = f.evaluation.parent / "reference-purpose.json"
    calls = []
    def execute(command, output, log, contract):
        calls.append((command, contract))
        runtime.last_execution = {"evaluator_dispatch_attempted": True, "evaluator_process_started": True}
        return 0
    runtime.execute = execute
    smoke.DockerEvaluationTransport(runtime)(f.spec.root, f.evaluation.parent / "candidate", runtime.source,
        f.evaluation, baseline_run=setup.baseline, base_config=f.spec.base_config)
    command, contract = calls[0]
    assert command[:2] == ["/usr/bin/python3", "-c"]
    assert "audit_installed_runtime" in command[2] and "os.execv" in command[2]
    assert command[5:8] == ["/usr/bin/python3", "-m", "local_bench"]
    assert contract["reference_purpose"] == "FRESH"
    assert contract["reference_purpose_sha256"] == file_sha(runtime.reference_purpose_receipt)
