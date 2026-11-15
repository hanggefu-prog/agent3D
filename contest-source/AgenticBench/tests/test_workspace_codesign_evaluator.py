"""No EDA/model calls: source, isolation and orchestration contract regressions."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from global_bench.model import load_global_task
from scripts import workspace_codesign_evaluator as bridge

ROOT = Path(__file__).resolve().parents[1]


def test_worker_imports_trusted_flow_from_unrelated_job_directory(tmp_path):
    dependency_root = str(Path(yaml.__file__).resolve().parent.parent)
    script = ROOT / "scripts/workspace_codesign_evaluator.py"
    code = ("import runpy; from pathlib import Path; "
            "m=runpy.run_path(" + repr(str(script)) + "); "
            "bound, audit=m['bind_frontend_config_paths']("
            "{'partition': {'algorithm': 'tp-gnn'}, 'placement': {}}, "
            "Path(" + repr(str(ROOT.parent)) + ")); "
            "assert audit['partition']['origin']=='native-default'; "
            "import open3dbench; assert Path(open3dbench.__file__).resolve().parent.parent=="
            "Path(" + repr(str(ROOT.parent)) + ").resolve()")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": dependency_root}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "algorithm.py").write_text("VALUE = 1\n")
    (root / "open3dbench").mkdir()
    (root / "open3dbench/verifier.py").write_text("TRUSTED = True\n")
    config = root / "Place-3D/config/partition/tp_gnn_fast.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"partition": {"tp_gnn_timing_driven": true}}\n')
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixed base")
    return root


def task(objective="timing"):
    suffix = {"timing": "timing", "thermal": "cts_thermal", "routed_thermal": "thermal"}[objective]
    return load_global_task(ROOT / "global_tasks" / ("lol_partition_placement_" + suffix))


def make_runtime(repository, tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    def contained(value):
        path = Path(value).resolve()
        path.relative_to(batch)
        return path
    @contextlib.contextmanager
    def slot():
        yield 1
    return SimpleNamespace(source=repository, source_commit=git(repository, "rev-parse", "HEAD"),
        framework=ROOT, batch=batch, image="sha256:" + "a" * 64, cpus=8, memory_gb=32,
        profile=lambda case: ROOT / "local_bench/benchmark_inputs" / (case + "_lol_ng45_tp_gnn_sta"),
        contained=contained, slot=slot)


def stage(run, name, directory):
    path = run / directory
    path.mkdir(parents=True)
    payload = path / "metrics.json"
    payload.write_text(json.dumps({"stage": name}))
    manifest = {"schema_version": 1, "stage": name, "status": "success", "signature": {},
                "artifacts": {"metrics": {"path": "metrics.json", "bytes": payload.stat().st_size,
                                            "sha256": bridge._sha(payload)}}}
    bridge._json(path / "stage_manifest.json", manifest)


def prefix(runtime):
    bundle = runtime.batch / "input"
    run = bundle / "run"
    for name, directory in bridge.PREFIX.items():
        stage(run, name, directory)
    for name in ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"):
        (run / "10_floorplan" / name).write_text("frozen " + name)
    bridge._json(run / "run.json", {"status": "success", "stages": {
        name: {"status": "success", "runtime_seconds": 123} for name in bridge.PREFIX}})
    alias = runtime.batch / "input-source"
    alias.mkdir()
    bridge._json(bundle / "benchmark_input.json", {
        "environment": {"open3dbench": {"root": str(alias)}}})
    return bundle, run, alias


def test_actual_git_diff_has_no_five_file_or_line_limit(repository):
    revision = git(repository, "rev-parse", "HEAD")
    (repository / "algorithm.py").unlink()
    for i in range(9):
        (repository / ("new module {}.py".format(i))).write_text("x = 1\n" * 1200)
    changes = bridge.source_changes(repository, revision, ("open3dbench",))
    assert len(changes) == 10
    assert next(row for row in changes if row["path"] == "algorithm.py")["sha256"] is None
    assert next(row for row in changes if row["path"] == "new module 0.py")["sha256"]


def test_protected_edits_cannot_be_hidden_by_candidate_claim(repository):
    revision = git(repository, "rev-parse", "HEAD")
    (repository / "open3dbench/verifier.py").write_text("TRUSTED = False\n")
    with pytest.raises(ValueError, match="Protected"):
        bridge.source_changes(repository, revision, ("open3dbench",))


def test_changed_symlink_is_rejected(repository, tmp_path):
    revision = git(repository, "rev-parse", "HEAD")
    (repository / "outside.py").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        bridge.source_changes(repository, revision, ())


@pytest.mark.parametrize("candidate", [
    {"routing": {"max_iterations": 1}}, {"partition": {"seed": 9}},
    {"placement": {"custom": {"random_seed": 1}}}, {"partition": {"timing_input": "/tmp/fake"}},
    {"placement": {"config": "../../bad.json"}}, {"placement": 1},
])
def test_evaluator_owned_configuration_rejected(candidate):
    with pytest.raises(ValueError):
        bridge.resolved_config({"partition": {}, "placement": {}}, candidate, task(), 8)


def test_frontend_settings_merge_without_method_allowlist_and_keep_seeds():
    base = {"threads": 32, "partition": {"algorithm": "tp-gnn"}, "placement": {"algorithm": "concurrent-dreamplace"},
            "cts": {"target_skew_ns": 0.1}}
    result = bridge.resolved_config(base, {"partition": {"algorithm": "mtkahypar"},
                                          "placement": {"target_density": 0.5}}, task(), 8)
    assert result["partition"] == {"algorithm": "mtkahypar", "seed": 1001}
    assert result["placement"]["target_density"] == 0.5
    assert result["placement"]["seed"] == 1000
    assert result["cts"] == base["cts"]
    assert result["threads"] == 32
    assert "threads" not in result["routing"]
    assert "seed" not in base["partition"]


@pytest.mark.parametrize("spelling", ["config/partition/custom.json", "Place-3D/config/partition/custom.json"])
def test_frontend_config_paths_accept_native_and_workspace_spelling(repository, spelling):
    source = repository / "Place-3D/config/partition/custom.json"
    source.write_text('{"partition": {"tp_gnn_timing_driven": true, "tp_gnn_epochs": 7}}')
    original = {"partition": {"algorithm": "tp-gnn", "config": spelling}, "placement": {}}
    resolved, bindings = bridge.bind_frontend_config_paths(original, repository)
    assert resolved["partition"]["config"] == str(source)
    assert original["partition"]["config"] == spelling
    assert bindings["partition"]["sha256"] == bridge._sha(source)


def test_backend_plan_reads_candidate_config_without_candidate_python(repository, tmp_path):
    from open3dbench.config import load_config
    from open3dbench.runner import FlowRunner

    source = repository / "custom-partition.json"
    source.write_text('{"partition": {"tp_gnn_timing_driven": true, "tp_gnn_epochs": 7}}')
    config = yaml.safe_load((ROOT / "local_bench/benchmark_inputs/bp_lol_ng45_tp_gnn_sta/config.yaml").read_text())
    config["partition"]["config"] = "custom-partition.json"
    resolved, bindings = bridge.bind_frontend_config_paths(config, repository)
    config_path = tmp_path / "evaluation-config.yaml"
    config_path.write_text(yaml.safe_dump(resolved))
    pristine = tmp_path / "pristine-backend"
    pristine.mkdir()
    runner = FlowRunner(pristine, load_config(config_path))
    assert not (pristine / "custom-partition.json").exists()
    # The current public contract passes a JSON file to partition. Timing
    # feature extraction is now internal to that stage, not a separate stage
    # or the removed private runner._tp_gnn_partition_values helper.
    selected = Path(runner.config["partition"]["config"])
    assert selected == source and selected.is_absolute()
    values = json.loads(selected.read_text())["partition"]
    assert values["tp_gnn_epochs"] == 7
    assert values["tp_gnn_timing_driven"] is True
    plan = runner.plan()
    assert next(row for row in plan if row["stage"] == "partition")["skip"] is None
    assert "timing_extract" not in {row["stage"] for row in plan}
    assert bindings["partition"]["source_relative"] == "custom-partition.json"


def test_implicit_partition_config_is_bound_to_candidate_snapshot(repository):
    config, bindings = bridge.bind_frontend_config_paths(
        {"partition": {"algorithm": "tp-gnn"}, "placement": {}}, repository)
    assert config["partition"]["config"] == str(repository / "Place-3D/config/partition/tp_gnn_fast.json")
    assert bindings["partition"]["origin"] == "native-default"


def test_frontend_config_rejects_ambiguous_or_symlink_paths(repository, tmp_path):
    (repository / "duplicate.json").write_text("{}")
    (repository / "Place-3D/duplicate.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        bridge.bind_frontend_config_paths({"partition": {"config": "duplicate.json"}, "placement": {}}, repository)
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (repository / "external.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        bridge.bind_frontend_config_paths({"partition": {"config": "external.json"}, "placement": {}}, repository)


def test_macro_halo_cannot_relax_the_physical_spacing_contract():
    base = {"pdk": "ng45", "partition": {}, "placement": {}}
    with pytest.raises(ValueError, match="physical-spacing"):
        bridge.resolved_config(base, {"placement": {"macro_halo_um": 0}}, task(), 8)
    assert bridge.resolved_config(base, {"placement": {"macro_halo_um": 10.0}}, task(), 8)["placement"]["macro_halo_um"] == 10.0


@pytest.mark.parametrize("domain,key", [("partition", "tp_gnn_epochs"), ("placement", "learning_rate")])
def test_method_knobs_are_not_silently_ignored_at_native_top_level(domain, key):
    with pytest.raises(ValueError, match="put method parameters in the JSON"):
        bridge.resolved_config({"partition": {}, "placement": {}}, {domain: {key: 10}}, task(), 8)


def test_copied_clock_contract_cannot_be_replaced(tmp_path):
    floorplan, placement = tmp_path / "10_floorplan", tmp_path / "30_placement"
    floorplan.mkdir(); placement.mkdir()
    for name in ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"):
        (floorplan / name).write_text("original")
        (placement / name).write_text("original")
    for die in ("bottom", "top"):
        path = placement / "die_init" / die / "design.sdc"
        path.parent.mkdir(parents=True)
        path.write_text("original")
    (tmp_path / "20_partition").mkdir()
    for path in (tmp_path / "20_partition/partition_assignment.csv", placement / "partition_assignment.csv"):
        path.write_text("original")
    assert len(bridge.validate_frontend_input_boundary(tmp_path)) == 9
    (placement / "die_init/top/design.sdc").write_text("relaxed clock")
    with pytest.raises(ValueError, match="fixed input/clock"):
        bridge.validate_frontend_input_boundary(tmp_path)


def test_prefix_preserves_original_manifest_hashes_and_runtime(repository, tmp_path):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    before = {str(p.relative_to(baseline)): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    target = runtime.batch / "copied"
    bridge.copy_frozen_prefix(baseline, target)
    for relative, content in before.items():
        if relative != "run.json":
            assert (target / relative).read_bytes() == content
    assert bridge._read(target / "run.json")["stages"]["floorplan"]["runtime_seconds"] == 123
    (baseline / "10_floorplan/metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        bridge.copy_frozen_prefix(baseline, runtime.batch / "invalid")


def test_docker_only_active_outputs_are_writable(repository, tmp_path):
    runtime = make_runtime(repository, tmp_path)
    bundle, _, alias = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    output = runtime.batch / "candidate"
    output.mkdir()
    run = output / "run"
    run.mkdir()
    source = output / "backend-source"
    argv = evaluator._docker(["/usr/bin/python3", "-m", "open3dbench"], 1, output, source, alias,
                              writable=(run,), readonly=(run / "00_input",), name="unit-test-only")
    mounts = [argv[i + 1] for i, value in enumerate(argv) if value == "-v"]
    assert str(runtime.batch) + ":" + str(runtime.batch) + ":ro" in mounts
    assert f"{source}:{alias}:ro" in mounts
    assert f"{runtime.framework}:{runtime.framework}:ro" in mounts
    assert f"{run}:{run}:rw" in mounts
    assert f"{run}/00_input:{run}/00_input:ro" in mounts
    assert "--read-only" in argv and "--cap-drop=ALL" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cpus") + 1] == "8"
    assert "OPEN3D_OPT_PASSES=2" in argv
    assert not any("docker.sock" in value for value in argv)
    assert not any("API_KEY" in value for value in argv)


@pytest.mark.parametrize("exact_source_alias", [True, False])
def test_split_trusted_framework_mounted_once_without_hiding_computation_source(repository, tmp_path, exact_source_alias):
    runtime = make_runtime(repository, tmp_path)
    bundle, _, contained_alias = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    runtime.framework = tmp_path / "separate-audit/AgenticBench"
    runtime.audit_framework_identity = {"fixture": "identity gate is tested separately"}
    output = runtime.batch / "candidate"
    output.mkdir()
    snapshot = output / "source"
    alias = runtime.source if exact_source_alias else contained_alias
    argv = evaluator._docker(["/usr/bin/python3", "-m", "open3dbench"], 1, output, snapshot, alias,
                             writable=(), name="split-fixture-no-container-execution")
    mounts = [argv[i + 1] for i, value in enumerate(argv) if value == "-v"]
    assert mounts.count(f"{runtime.framework.parent}:{runtime.framework.parent}:ro") == 1
    assert mounts.count(f"{snapshot}:{alias}:ro") == 1
    assert len({value.split(":")[1] for value in mounts}) == len(mounts)
    assert (f"{runtime.source}:{runtime.source}:ro" in mounts) != exact_source_alias
    assert any(value.startswith("PYTHONPATH=" + str(runtime.framework) + ":" + str(alias) + ":") for value in argv)


def test_initial_config_is_explicit_but_evaluator_does_not_modify_subject(repository, tmp_path):
    runtime = make_runtime(repository, tmp_path)
    bundle, _, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    path = evaluator.initialize_subject(repository)
    assert set(yaml.safe_load(path.read_text())) == {"partition", "placement"}
    with pytest.raises(FileExistsError):
        evaluator.initialize_subject(repository)


@pytest.mark.parametrize("objective", ["timing", "thermal", "routed_thermal"])
def test_full_boundary_uses_distinct_frontend_backend_and_verifier(repository, tmp_path, monkeypatch, objective):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, alias = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", objective, bundle, task(objective))
    evaluator.initialize_subject(repository)
    (repository / "algorithm.py").write_text("VALUE = 2\n")
    output = runtime.batch / "evaluated"
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda value: {"run_dir": str(baseline)})
    patches = []
    monkeypatch.setattr(bridge, "apply_candidate_patch", lambda *args: patches.append(args))
    observed = []
    def fake_call(phase, argv, root, deadline):
        observed.append((phase, argv))
        run = root / "run"
        if phase in {"frontend", "backend"}:
            mapping = {name: directory for name, directory in bridge.FRONTEND.items() if name not in bridge.PREFIX} if phase == "frontend" else {
                "pdn": "40_pdn", "hbt": "50_hbt", "opt": "60_opt",
                **({"cts": "70_cts"} if objective != "timing" else {}),
                **({"global_routing": "75_global_route", "detailed_routing": "80_route",
                    "report": "90_report", "thermal": "95_thermal"} if objective == "routed_thermal" else {})}
            status = bridge._read(run / "run.json")
            for name, directory in mapping.items():
                stage(run, name, directory)
                status["stages"][name] = {"status": "success"}
                if name in {"opt", "cts", "thermal"}:
                    bridge._json(run / directory / "checkpoint.json", {"stage": name})
            status["status"] = "success"
            bridge._json(run / "run.json", status)
            if phase == "frontend":
                for name in ("design.v", "design.sdc", "floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"):
                    (run / "30_placement" / name).write_bytes((run / "10_floorplan" / name).read_bytes())
                for die in ("bottom", "top"):
                    path = run / "30_placement/die_init" / die / "design.sdc"
                    path.parent.mkdir(parents=True)
                    path.write_bytes((run / "10_floorplan/design.sdc").read_bytes())
                for directory in ("20_partition", "30_placement"):
                    (run / directory / "partition_assignment.csv").write_text("inst_name,die\na,bottom\n")
        elif phase == "thermal":
            assert not (run / "95_thermal_snapshot").exists()
            bridge._json(run / "95_thermal_snapshot/metrics.json", {"temperature": 325})
        elif phase == "verifier":
            bridge._json(root / "verification/verification.json", {"passed": True, "metrics": {"combined_score": 1.0}})
    monkeypatch.setattr(evaluator, "_call", fake_call)
    result = evaluator.evaluate(repository, output)
    assert result["success"], result
    assert result["metrics"] == {"combined_score": 1.0}
    assert [name for name, _ in observed] == (["frontend", "backend", "thermal", "verifier"] if objective == "thermal" else ["frontend", "backend", "verifier"])
    commands = dict(observed)
    expected = {
        "timing": (2, "opt", "timing", "60_opt", "60_opt"),
        "thermal": (2, "cts", "cts-thermal", "70_cts", "95_thermal_snapshot"),
        "routed_thermal": (1, "thermal", "thermal", "95_thermal", "95_thermal"),
    }[objective]
    passes, backend_stop, verifier_mode, checkpoint, metric_dir = expected
    assert commands["backend"][commands["backend"].index("--to") + 1] == backend_stop
    assert commands["verifier"][commands["verifier"].index("--mode") + 1] == verifier_mode
    assert all(f"OPEN3D_OPT_PASSES={passes}" in argv for argv in commands.values())
    assert len(patches) == (0 if objective == "routed_thermal" else 2)
    contract = bridge._read(output / "evaluation-contract.json")
    assert contract["opt_passes"] == passes
    assert bool(contract["trusted_evaluator_patch_sha256"]) == (passes == 2)
    assert contract["evaluation_endpoint"]["task_id"] == task(objective).task_id
    manifest = bridge._read(output / "submission/submission.json")
    artifacts = {row["name"]: row for row in manifest["artifacts"]}
    assert artifacts["final_checkpoint"]["source"] == str(output / "run" / checkpoint / "checkpoint.json")
    assert artifacts["metrics"]["source"] == str(output / "run" / metric_dir / "metrics.json")
    for phase in ("backend", "verifier"):
        assert f"{output}/backend-source:{alias}:ro" in commands[phase]
    assert f"{output}/source:{alias}:ro" in commands["frontend"]
    assert (output / "source/algorithm.py").read_text() == "VALUE = 2\n"
    assert (output / "backend-source/algorithm.py").read_text() == "VALUE = 1\n"
    assert bridge._read(output / "structural-verification.json")["passed"]
    resolved = yaml.safe_load((output / "evaluation-config.yaml").read_text())
    if objective == "routed_thermal":
        assert resolved["thermal"] == {"enabled": True, "grid_size": 10, "ambient_k": 318.15}
        assert not (output / "run/95_thermal_snapshot").exists()
    assert resolved["partition"]["config"] == str(output / "source/Place-3D/config/partition/tp_gnn_fast.json")
    assert bridge._read(output / "frontend-config-bindings.json")["partition"]["sha256"]
    with pytest.raises(FileExistsError):
        evaluator.evaluate(repository, output)


@pytest.mark.parametrize("objective,wrong_task", [
    ("thermal", "routed_thermal"), ("routed_thermal", "thermal"), ("timing", "thermal"),
])
def test_distinct_thermal_task_identities_cannot_be_interchanged(repository, tmp_path, objective, wrong_task):
    runtime = make_runtime(repository, tmp_path)
    bundle, _, _ = prefix(runtime)
    with pytest.raises(ValueError, match="identity mismatch"):
        bridge.WorkspaceEvaluator(runtime, "bp", objective, bundle, task(wrong_task))


@pytest.mark.parametrize("forbidden", ["75_global_route", "80_route", "90_report", "95_thermal"])
def test_frontend_cannot_inject_routed_evaluator_outputs(repository, tmp_path, monkeypatch, forbidden):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "routed_thermal", bundle, task("routed_thermal"))
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda value: {"run_dir": str(baseline)})
    def fake_frontend(phase, argv, output, deadline):
        assert phase == "frontend", "No backend/verifier after an invalid frontend"
        for name, directory in bridge.FRONTEND.items():
            if name not in bridge.PREFIX:
                stage(output / "run", name, directory)
        (output / "run" / forbidden).mkdir()
    monkeypatch.setattr(evaluator, "_call", fake_frontend)
    result = evaluator.evaluate(repository, runtime.batch / "injected")
    assert result["success"] is False and result["metrics"] == {}
    assert "Frontend wrote downstream evaluator directory: " + forbidden in result["error"]


def test_failed_validation_consumes_attempt_without_any_physical_call(repository, tmp_path, monkeypatch):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda value: {"run_dir": str(baseline)})
    monkeypatch.setattr(evaluator, "_call", lambda *args: pytest.fail("No EDA after invalid candidate"))
    (repository / "open3dbench/verifier.py").write_text("CHANGED=True")
    output = runtime.batch / "failed"
    result = evaluator.evaluate(repository, output)
    assert not result["success"] and "Protected" in result["error"]
    assert bridge._read(output.parent / "failed.workspace-attempt.json")["state"] == "failed"
    with pytest.raises(FileExistsError):
        evaluator.evaluate(repository, output)


def test_native_changes_fail_explicitly_not_silently_ignored(repository, tmp_path, monkeypatch):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda value: {"run_dir": str(baseline)})
    (repository / "algorithm.cu").write_text("// new kernel\n")
    result = evaluator.evaluate(repository, runtime.batch / "native")
    assert not result["success"] and "separately built pinned runtime image" in result["error"]


def test_independent_verifier_failure_does_not_report_metrics(repository, tmp_path, monkeypatch):
    runtime = make_runtime(repository, tmp_path)
    bundle, baseline, _ = prefix(runtime)
    evaluator = bridge.WorkspaceEvaluator(runtime, "bp", "timing", bundle, task())
    monkeypatch.setattr(bridge, "validate_benchmark_input", lambda value: {"run_dir": str(baseline)})
    monkeypatch.setattr(bridge, "apply_candidate_patch", lambda *args: None)
    def fail_frontend(*args):
        raise RuntimeError("Native operation failed")
    monkeypatch.setattr(evaluator, "_call", fail_frontend)
    result = evaluator.evaluate(repository, runtime.batch / "native-failure")
    assert not result["success"] and result["metrics"] == {}
    assert "Native operation failed" in result["error"]


from test_workspace_framework_identity import real_split_bundle


def test_real_split_requalification_registry_adoption_and_candidate_boundary(real_split_bundle, monkeypatch):
    """Real validators/registry/profile binding, stopped before any native call.

    Synthetic native files supply the data, not mocked qualification results.
    The source/profile fixture has two clean Git checkouts, an old importer and
    current audit-local profile resolution. No EDA, container or model runs.
    """
    import shutil

    from local_bench import benchmark_input, reference_baseline as baseline
    from local_bench import reference_catalog as catalog, reference_evaluated as evaluated
    from local_bench.reference_store import file_sha, load_reference
    from scripts import codesign_runtime, requalify_campaign_reference as recovery
    from scripts import workspace_reference as references
    from scripts.run_workspace_codesign import independent_subject
    from scripts.workspace_task_selection import TIMING_TEN_TASK
    from test_reference_evaluated import make_evaluation, add_native_opt_log_aliases, write

    pair = real_split_bundle
    monkeypatch.setattr(recovery, "ROOT", pair.framework)
    monkeypatch.setattr(evaluated, "FRAMEWORK", pair.framework)
    monkeypatch.setattr(catalog, "__file__", str(pair.framework / "local_bench/reference_catalog.py"))
    task_spec = load_global_task(pair.framework / "global_tasks" / TIMING_TEN_TASK)
    image = sorted(baseline.IMAGE_IDS)[0]

    # Complete the synthetic root/prefix bookkeeping consumed by the genuine
    # evaluator copy boundary, then seal the manifest before constructing the
    # historical reference contract. No production artifact is changed.
    manifest = bridge._read(pair.manifest)
    stages = {}
    for stage_name, directory in benchmark_input.PROFILE_STAGE_DIRS.items():
        if stage_name not in manifest["stages"]:
            continue
        path = pair.run / directory / "stage_manifest.json"
        value = bridge._read(path)
        value["signature"] = {"fixture": "native-shaped-prefix"}
        write(path, value)
        manifest["stages"][stage_name]["manifest_sha256"] = file_sha(path)
        stages[stage_name] = {"status": "success", "directory": directory}
    write(pair.manifest, manifest)
    write(pair.run / "run.json", {"design": "bp_fe", "type": "lol", "pdk": "ng45",
                                  "status": "success", "stages": stages})
    validated = benchmark_input.validate_benchmark_input(pair.bundle, task_id=task_spec.objective.evaluator_profile)
    assert Path(validated["config"]) == pair.profile / "config.yaml"
    assert Path(validated["config"]) != pair.physical_profile / "config.yaml"
    assert manifest["current_importer_source"]["root"] == str(pair.source)

    old = pair.batch / "v2/references/bp_fe/timing"
    # This helper only constructs schema-complete tiny files. Its prefix
    # shortcut is confined to construction and removed before any validation.
    with pytest.MonkeyPatch.context() as construction:
        factory = make_evaluation.__wrapped__(pair.batch, construction)
        evaluation, native_run, profile = factory(name="v2/references/bp_fe/timing/reference/result", rounds=10)
    for relative in pair.physical:
        target = native_run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pair.run / relative, target)
    native_contract = bridge._read(evaluation / "evaluation-contract.json")
    native_contract.update(source_revision=pair.source_revision, input_manifest_sha256=file_sha(pair.manifest))
    write(evaluation / "evaluation-contract.json", native_contract)
    for directory in bridge.FRONTEND.values():
        path = native_run / directory / "stage_manifest.json"
        value = bridge._read(path)
        value["provenance"]["repo_commit"] = pair.source_revision
        write(path, value)
    for directory in ("40_pdn", "50_hbt", "60_opt"):
        path = native_run / directory / "stage_manifest.json"
        value = bridge._read(path)
        value["provenance"]["repo_commit"] = pair.source_revision
        write(path, value)
    add_native_opt_log_aliases(native_run)  # Seals real targets before aliases.
    write(evaluation / "feedback.json", {"success": True})
    old_contract = {"case": "bp_fe", "endpoint": "timing",
        "task_digest": catalog.task_digest(task_spec.root), "evaluator_digest": "0" * 64,
        "frozen_inputs": evaluated.expected_frozen_inputs("bp_fe", file_sha(pair.manifest)),
        "physical": evaluated.expected_source_contract("bp_fe", "timing", profile=profile,
            source_revision=pair.source_revision, image=image)}
    write(old / "run_config.json", {"kind": "campaign-reference-only-v1", "case": "bp_fe", "endpoint": "timing",
        "task_id": TIMING_TEN_TASK, "reference_policy": "FRESH", "source_commit": pair.source_revision,
        "image": image, "benchmark_input": str(pair.bundle), "search_dispatch": False,
        "candidate_finals": 0, "candidate_budget_consumed": 0, "model_calls": 0,
        "reference_selection": {"contract": old_contract}})
    write(old / "summary.json", {"state": "failed", "error": recovery.REGISTRATION_ERROR,
        "case": "bp_fe", "endpoint": "timing", "accounting": {"registry_publications": 0}})
    preserved = {p: p.read_bytes() for root in (old, pair.bundle, pair.archive)
                 for p in root.rglob("*") if p.is_file()}
    store, output = pair.batch / "v4-reference-store", pair.batch / "v4/requalifications/bp_fe/timing"
    result = recovery.requalify(source=pair.source, source_revision=pair.source_revision,
        audit_framework_revision=pair.audit_revision, case="bp_fe", image=image,
        benchmark_input=pair.bundle, existing_evaluation=evaluation, old_summary=old / "summary.json",
        reference_store=store, output=output)
    assert result["state"] == "qualified", result
    assert result["accounting"]["registry_publications"] == 1
    assert result["accounting"]["native_physical_attempts"] == result["accounting"]["model_calls"] == 0
    record_path = Path(result["record_path"])
    record = load_reference(record_path, contract=result["contract"], validate=catalog.qualify_producer)
    assert len(record["qualification"]["native_log_aliases"]) == 17
    assert record["qualification"]["source_identity"]["source_revision"] == pair.source_revision

    runtime = codesign_runtime.Runtime(pair.source, pair.batch, image, "1", cpus=16, memory_gb=48,
        audit_framework_revision=pair.audit_revision, computation_source_revision=pair.source_revision)
    runtime.reference_baseline = baseline.BASELINE_ID
    runtime.reference_policy, runtime.reference_record, runtime.reference_store = "ADOPT", record_path, store
    selected = references.WorkspaceReference(runtime, "bp_fe", "timing", pair.bundle, task_spec)
    assert selected.contract == result["contract"]
    assert selected.evaluator_invocations == 0
    class NoReferenceExecution:
        def evaluate(self, *args, **kwargs):
            pytest.fail("ADOPT cannot run an evaluator")
    adoption = selected.obtain(NoReferenceExecution(), pair.batch / "v4/search/results/reference")
    assert adoption["success"] and adoption["metrics"]["combined_score"] == pytest.approx(1.0)
    assert adoption["new_reference_native_attempts"] == adoption["reference_evaluator_invocations"] == 0

    evaluator = bridge.WorkspaceEvaluator(runtime, "bp_fe", "timing", pair.bundle, task_spec)
    subject = pair.batch / "v4/search/subject"
    independent_subject(pair.source, subject)
    evaluator.initialize_subject(subject)
    @contextlib.contextmanager
    def offline_slot():
        yield 1
    monkeypatch.setattr(runtime, "slot", offline_slot)
    observed = []
    def stop_before_native(phase, argv, root, deadline):
        observed.append((phase, argv))
        raise RuntimeError("Offline fixture: stop at the first native dispatch boundary")
    monkeypatch.setattr(evaluator, "_call", stop_before_native)
    candidate_output = pair.batch / "v4/search/results/round-01"
    candidate = evaluator.evaluate(subject, candidate_output, Path(adoption["run"]))
    assert len(observed) == 1 and observed[0][0] == "frontend", candidate
    assert "stop at the first native dispatch boundary" in candidate["error"]
    assert not candidate["success"] and candidate["execution"]["command_dispatches"] == 0
    emitted = bridge._read(candidate_output / "evaluation-contract.json")
    assert emitted["source_revision"] == pair.source_revision
    assert emitted["source_alias"] == str(pair.source)
    assert emitted["audit_framework_identity"] == pair.receipt
    assert emitted["task_digest"] == task_spec.digest() and emitted["opt_passes"] == 1
    assert emitted["reference_purpose"] == "CANDIDATE"
    argv = observed[0][1]
    mounts = [argv[i + 1].split(":") for i, arg in enumerate(argv) if arg == "-v"]
    mapping = {target: (source, mode) for source, target, mode in mounts}
    assert len(mapping) == len(mounts)
    assert mapping[str(pair.source)] == (str(candidate_output / "source"), "ro")
    assert mapping[str(pair.framework.parent)] == (str(pair.framework.parent), "ro")
    assert mapping[str(evaluation)] == (str(evaluation), "ro")
    assert mapping[str(record_path)] == (str(record_path), "ro")
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
