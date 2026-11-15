"""Audit/computation split fixtures: Git and JSON only; no EDA/model/container."""
import os
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import codesign_runtime
from scripts import workspace_framework_identity as identity


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
        stderr=subprocess.DEVNULL, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).decode().strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def pair(tmp_path):
    source, audit = tmp_path / "source", tmp_path / "audit"
    source.mkdir()
    git(source, "init", "-q")
    files = {
        "open3dbench/runner.py": "physical = 1\n",
        "AgenticBench/global_tasks/timing/task.yaml": "rounds: 10\n",
        "AgenticBench/global_bench/verifiers/open3dbench_codesign_verifier.py": "score = 1\n",
        "AgenticBench/local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923/config.yaml": "threads: 4\n",
        "AgenticBench/local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923/profile.yaml": "schema_version: 1\n",
        "AgenticBench/local_bench/reference_evaluated.py": "audit = 1\n",
        "AgenticBench/scripts/codesign_runtime.py": "runtime = 1\n",
    }
    for name, text in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    source_revision = commit(source)
    git(tmp_path, "clone", "-q", "--no-hardlinks", str(source), str(audit))
    (audit / "AgenticBench/local_bench/reference_evaluated.py").write_text("audit = 2\n")
    audit_revision = commit(audit)
    return SimpleNamespace(source=source, framework=audit / "AgenticBench", source_revision=source_revision,
                           audit_revision=audit_revision, batch=tmp_path / "runs")


def validate(pair):
    return identity.validate_audit_framework(pair.source, pair.framework,
        source_revision=pair.source_revision, framework_revision=pair.audit_revision)


def test_explicit_split_records_real_physical_source_and_exact_changed_tree(pair):
    receipt = validate(pair)
    assert receipt["computation_source_revision"] == receipt["physical_producer_revision"] == pair.source_revision
    assert receipt["trusted_framework_revision"] == pair.audit_revision
    assert set(receipt["reviewed_audit_changes"]) == {"AgenticBench/local_bench/reference_evaluated.py"}
    assert receipt["unchanged_tree_entries"] == 5
    assert receipt["native_dispatches"] == receipt["model_calls"] == 0
    assert receipt == validate(pair)


@pytest.mark.parametrize("name", ["open3dbench/runner.py", "Place-3D/algorithm.py",
    "AgenticBench/global_tasks/timing/task.yaml",
    "AgenticBench/global_bench/verifiers/open3dbench_codesign_verifier.py",
    "AgenticBench/local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923/config.yaml",
    "AgenticBench/scripts/unreviewed.py"])
def test_computation_task_scoring_config_and_unreviewed_changes_rejected(pair, name):
    path = pair.framework.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed\n")
    pair.audit_revision = commit(pair.framework.parent)
    with pytest.raises(ValueError, match="unapproved"):
        validate(pair)


@pytest.mark.parametrize("mutation", ["source-dirty", "audit-dirty", "untracked", "wrong-source",
    "wrong-audit", "missing-source", "missing-audit", "symlink", "mode", "deleted"])
def test_dirty_unpinned_or_nonregular_audit_changes_fail_closed(pair, mutation):
    audit_path = pair.framework / "local_bench/reference_evaluated.py"
    if mutation == "source-dirty":
        (pair.source / "open3dbench/runner.py").write_text("changed\n")
    elif mutation == "audit-dirty":
        audit_path.write_text("changed\n")
    elif mutation == "untracked":
        (pair.framework / "extra.py").write_text("untracked\n")
    elif mutation in {"wrong-source", "missing-source"}:
        pair.source_revision = "0" * 40 if mutation == "wrong-source" else None
    elif mutation in {"wrong-audit", "missing-audit"}:
        pair.audit_revision = "0" * 40 if mutation == "wrong-audit" else None
    else:
        if mutation == "symlink":
            audit_path.unlink(); audit_path.symlink_to("../scripts/codesign_runtime.py")
        elif mutation == "mode":
            audit_path.chmod(0o755)
        else:
            audit_path.unlink()
        pair.audit_revision = commit(pair.framework.parent)
    with pytest.raises(ValueError):
        validate(pair)


def test_runtime_requires_opt_in_and_resolves_audit_profile_without_relabeling_physics(pair, monkeypatch):
    from local_bench import reference_baseline as baseline
    monkeypatch.setattr(codesign_runtime, "__file__", str(pair.framework / "scripts/codesign_runtime.py"))
    monkeypatch.setattr(baseline, "__file__", str(pair.framework / "local_bench/reference_baseline.py"))
    monkeypatch.setattr(baseline, "PROFILE_ROOT", pair.framework / "local_bench/benchmark_inputs")
    options = dict(source=pair.source, batch=pair.batch, image="sha256:" + "a" * 64,
                   gpus="1", cpus=16, memory_gb=48)
    with pytest.raises(ValueError, match="explicitly pinned"):
        codesign_runtime.Runtime(**options)
    assert not pair.batch.exists()
    runtime = codesign_runtime.Runtime(**options, audit_framework_revision=pair.audit_revision,
                                      computation_source_revision=pair.source_revision)
    runtime.reference_baseline = "reference_20260923"
    assert runtime.source == pair.source and runtime.source_commit == pair.source_revision
    assert runtime.framework == pair.framework
    assert runtime.profile("bp_fe") == pair.framework / "local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923"
    assert identity.revalidate_runtime_framework(runtime) == runtime.audit_framework_identity
    (pair.framework / "untracked.py").write_text("changed\n")
    with pytest.raises(ValueError, match="clean"):
        identity.revalidate_runtime_framework(runtime)


def test_runtime_rechecks_sealed_compatibility_receipt(pair):
    runtime = SimpleNamespace(source=pair.source, source_commit=pair.source_revision,
        framework=pair.framework, audit_framework_identity=validate(pair))
    runtime.audit_framework_identity["unchanged_tree_entries"] += 1
    with pytest.raises(ValueError, match="after admission"):
        identity.revalidate_runtime_framework(runtime)


def test_runtime_rejects_writable_batch_overlapping_either_checkout(pair, monkeypatch):
    monkeypatch.setattr(codesign_runtime, "__file__", str(pair.framework / "scripts/codesign_runtime.py"))
    for batch in (pair.source / "runs", pair.framework.parent / "runs"):
        with pytest.raises(ValueError, match="must be disjoint"):
            codesign_runtime.Runtime(pair.source, batch, "sha256:" + "a" * 64, "1",
                cpus=16, memory_gb=48, audit_framework_revision=pair.audit_revision,
                computation_source_revision=pair.source_revision)
        assert not batch.exists()


def test_runtime_split_does_not_resolve_away_source_symlink(pair, tmp_path, monkeypatch):
    monkeypatch.setattr(codesign_runtime, "__file__", str(pair.framework / "scripts/codesign_runtime.py"))
    link = tmp_path / "source-alias"
    link.symlink_to(pair.source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        codesign_runtime.Runtime(link, pair.batch, "sha256:" + "a" * 64, "1",
            cpus=16, memory_gb=48, audit_framework_revision=pair.audit_revision,
            computation_source_revision=pair.source_revision)
    assert not pair.batch.exists()


@pytest.fixture
def real_split_bundle(tmp_path, monkeypatch):
    """Two Git checkouts, actual versioned profile, native-shaped sealed bundle.

    Only fixture locations are rebound. Input/profile, artifact, partition,
    archive, source-tree and receipt validation all execute their real code.
    """
    from local_bench import benchmark_input
    from local_bench import reference_baseline as baseline
    from local_bench import reference_evaluated
    repo = Path(__file__).resolve().parents[2]
    source, audit = tmp_path / "source", tmp_path / "audit"
    batch, archive = tmp_path / "runs", tmp_path / "archive"
    bundle = batch / "v2/inputs/bp_fe/bundle"
    run = bundle / "run"
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    def record(path):
        return {"path": path.name, "bytes": path.stat().st_size, "sha256": baseline.sha256(path)}
    stages = {}
    for stage, directory, filename in (("input", "00_input", "design.v"),
            ("floorplan", "10_floorplan", "design.def"), ("partition", "20_partition", "partition_assignment.csv")):
        folder = run / directory
        folder.mkdir(parents=True)
        path = folder / filename
        path.write_text("inst_name,die\na,bottom\n" if stage == "partition" else "frozen " + stage + "\n")
        artifacts = {"payload": record(path)}
        if stage == "partition":
            report = {"method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
                      "config": {"decoder": "global-2way"}}
            write(folder / "tp_gnn_report.json", report)
            write(folder / "partition_manifest.json", {"method": "tp-gnn-flat", "method_report": report})
            artifacts["method_manifest"] = record(folder / "partition_manifest.json")
            artifacts["report"] = record(folder / "tp_gnn_report.json")
        manifest = folder / "stage_manifest.json"
        write(manifest, {"schema_version": 1, "stage": stage, "status": "success",
                         "method": "tp-gnn-flat" if stage == "partition" else "fixture",
                         "artifacts": artifacts})
        stages[stage] = {"directory": directory, "manifest_sha256": baseline.sha256(manifest)}
    shutil.copytree(run, archive / "runs/bp_fe")
    write(archive / "reference.json", {"reference_id": "fixture-reference", "original_root": str(archive)})
    physical = {name: {"bytes": (run / name).stat().st_size, "sha256": baseline.sha256(run / name)}
                for name in ("00_input/design.v", "10_floorplan/design.def")}
    inventory = {"reference_id": "fixture-reference", "reference_json_sha256": baseline.sha256(archive / "reference.json"),
        "cases": {"bp_fe": {"run": "runs/bp_fe", "physical_prefix": physical,
                   "partition_assignment_sha256": baseline.sha256(run / "20_partition/partition_assignment.csv")}}}
    profile_relative = Path("AgenticBench/local_bench/benchmark_inputs/bp_fe_lol_ng45_reference_20260923")
    shutil.copytree(repo / profile_relative, source / profile_relative)
    task_relative = Path("AgenticBench/global_tasks/lol_partition_placement_timing_one_pass_10")
    shutil.copytree(repo / task_relative, source / task_relative)
    # Match make_evaluation's synthetic producer, not deleted legacy recipes.
    # These files exist only in this test's temporary source checkout.
    for relative in (baseline.PARTITION_CONFIG, baseline.PLACEMENT_CONFIG):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, {"fixture": "reference-contract-test", "path": relative})
    # Keep actual trusted evaluator files available for consumers that use the
    # real catalog digest instead of stubbing registry/source contracts.
    for relative in ("local_bench/reference_store.py", "local_bench/reference_catalog.py",
            "local_bench/reference_archive.py", "local_bench/reference_evaluated.py",
            "local_bench/reference_baseline.py", "local_bench/reference_placement_evidence.py",
            "local_bench/metrics.py", "local_bench/reference_local_placement.py",
            "local_bench/reference_local_partition.py", "local_bench/partition_assignment.py",
            "local_bench/post_placement_timing.py", "local_bench/placement_components.py",
            "local_bench/scoped_reference.py", "scripts/workspace_task_selection.py",
            "global_bench/verifiers/open3dbench_codesign_verifier.py"):
        path = source / "AgenticBench" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "AgenticBench" / relative, path)
    write(source / "AgenticBench/local_bench/reference_20260923_identity.json", inventory)
    write(source / "AgenticBench/scripts/codesign_runtime.py", {"runtime": 1})
    git(source, "init", "-q")
    source_revision = commit(source)
    git(tmp_path, "clone", "-q", "--no-hardlinks", str(source), str(audit))
    audit_file = audit / "AgenticBench/local_bench/reference_evaluated.py"
    audit_file.write_text(audit_file.read_text() + "\n# Fixture audit-only revision.\n")
    audit_revision = commit(audit)
    framework = audit / "AgenticBench"
    manifest_path = bundle / "benchmark_input.json"
    profile = source / profile_relative
    importer = {"root": str(source), "revision": source_revision, "dirty": False}
    write(manifest_path, {"schema_version": 1, "benchmark_id": profile.name,
        "profile_sha256": baseline.sha256(profile / "profile.yaml"), "config_sha256": baseline.sha256(profile / "config.yaml"),
        "run_dir": "run", "stages": stages, "reference_root": str(archive), "baseline_id": baseline.BASELINE_ID,
        "import_kind": "frozen_reference_adoption", "environment": {"open3dbench": importer},
        "current_importer_source": importer})
    monkeypatch.setattr(benchmark_input, "PROFILE_ROOT", framework / "local_bench/benchmark_inputs")
    monkeypatch.setattr(baseline, "PROFILE_ROOT", framework / "local_bench/benchmark_inputs")
    monkeypatch.setattr(baseline, "IDENTITY_FILE", framework / "local_bench/reference_20260923_identity.json")
    monkeypatch.setattr(baseline, "__file__", str(framework / "local_bench/reference_baseline.py"))
    monkeypatch.setattr(codesign_runtime, "__file__", str(framework / "scripts/codesign_runtime.py"))
    monkeypatch.setattr(reference_evaluated, "REPO", audit)
    pair = SimpleNamespace(source=source, framework=framework, source_revision=source_revision,
        audit_revision=audit_revision, batch=batch, bundle=bundle, run=run, archive=archive,
        profile=framework / "local_bench/benchmark_inputs" / profile.name, physical_profile=profile,
        manifest=manifest_path, physical=physical)
    pair.receipt = validate(pair)
    return pair


def test_real_bundle_resolves_current_audit_profile_with_unchanged_physical_source(real_split_bundle):
    from local_bench import benchmark_input, reference_baseline as baseline
    from scripts.workspace_codesign_evaluator import input_source_alias
    pair = real_split_bundle
    before = {p: p.read_bytes() for root in (pair.bundle, pair.archive) for p in root.rglob("*") if p.is_file()}
    validated = benchmark_input.validate_benchmark_input(pair.bundle, task_id="tp_gnn_placement_one_pass")
    assert validated["config"] == str(pair.profile / "config.yaml")
    assert validated["config"] != str(pair.physical_profile / "config.yaml")
    receipt = baseline.validate_reference_bundle(pair.bundle, "bp_fe", audit_framework_identity=pair.receipt)
    assert receipt["physical_input_sha256"] == {p: v["sha256"] for p, v in pair.physical.items()}
    runtime = codesign_runtime.Runtime(pair.source, pair.batch, "sha256:" + "a" * 64, "1", cpus=16, memory_gb=48,
        audit_framework_revision=pair.audit_revision, computation_source_revision=pair.source_revision)
    runtime.reference_baseline = baseline.BASELINE_ID
    assert runtime.profile("bp_fe") / "config.yaml" == Path(validated["config"])
    assert runtime.source_commit == pair.source_revision
    assert input_source_alias(runtime, json.loads(pair.manifest.read_text()), pair.bundle,
        pair.batch / "v4/final", baseline_validation=receipt) == pair.source
    assert all(path.read_bytes() == data for path, data in before.items())


def test_real_validator_resolving_other_checkout_is_not_silently_aliased(real_split_bundle, monkeypatch):
    from local_bench import benchmark_input, reference_baseline as baseline
    pair = real_split_bundle
    monkeypatch.setattr(benchmark_input, "PROFILE_ROOT", pair.physical_profile.parent)
    # Equal bytes permit the portable input validator to read either installed
    # profile, but a split consumer must use its own pinned trusted framework.
    actual = benchmark_input.validate_benchmark_input(pair.bundle)
    assert actual["config"] == str(pair.physical_profile / "config.yaml")
    with pytest.raises(ValueError, match="versioned reference-aligned"):
        baseline.validate_reference_bundle(pair.bundle, "bp_fe", audit_framework_identity=pair.receipt)


@pytest.mark.parametrize("mutation", ["profile-hash", "config-hash", "artifact", "method-report", "archive", "importer"])
def test_real_split_bundle_rejects_bad_identity_without_mocking_validator(real_split_bundle, mutation):
    from local_bench import reference_baseline as baseline
    from scripts.workspace_codesign_evaluator import input_source_alias
    pair = real_split_bundle
    data = json.loads(pair.manifest.read_text())
    if mutation.endswith("-hash"):
        data[mutation.replace("-hash", "_sha256")] = "0" * 64
    elif mutation == "artifact":
        (pair.run / "10_floorplan/design.def").write_text("changed")
    elif mutation == "method-report":
        path = pair.run / "20_partition/tp_gnn_report.json"
        report = json.loads(path.read_text())
        report["hierarchy"]["mode"] = "historical"
        path.write_text(json.dumps(report))
        stage_path = path.parent / "stage_manifest.json"
        stage = json.loads(stage_path.read_text())
        stage["artifacts"]["report"].update(bytes=path.stat().st_size, sha256=baseline.sha256(path))
        stage_path.write_text(json.dumps(stage))
        data["stages"]["partition"]["manifest_sha256"] = baseline.sha256(stage_path)
    elif mutation == "archive":
        (pair.archive / "reference.json").write_text("{}")
    else:
        data["current_importer_source"]["revision"] = "0" * 40
    pair.manifest.write_text(json.dumps(data))
    if mutation == "importer":
        receipt = baseline.validate_reference_bundle(pair.bundle, "bp_fe", audit_framework_identity=pair.receipt)
        runtime = SimpleNamespace(source=pair.source, source_commit=pair.source_revision)
        with pytest.raises(ValueError, match="importer identity"):
            input_source_alias(runtime, data, pair.bundle, pair.batch / "v4/final", baseline_validation=receipt)
    else:
        with pytest.raises(ValueError):
            baseline.validate_reference_bundle(pair.bundle, "bp_fe", audit_framework_identity=pair.receipt)


@pytest.mark.parametrize("name", ["profile.yaml", "config.yaml"])
def test_real_split_rejects_committed_audit_profile_drift_even_with_resealed_bundle(real_split_bundle, name):
    from local_bench import benchmark_input, reference_baseline as baseline
    pair = real_split_bundle
    path = pair.profile / name
    path.write_text(path.read_text() + "\n# changed trusted bytes\n")
    pair.audit_revision = commit(pair.framework.parent)
    data = json.loads(pair.manifest.read_text())
    data["profile_sha256" if name == "profile.yaml" else "config_sha256"] = baseline.sha256(path)
    pair.manifest.write_text(json.dumps(data))
    assert benchmark_input.validate_benchmark_input(pair.bundle)["status"] == "ok"
    receipt = {**pair.receipt, "trusted_framework_revision": pair.audit_revision}
    with pytest.raises(ValueError, match="unapproved"):
        baseline.validate_reference_bundle(pair.bundle, "bp_fe", audit_framework_identity=receipt)
