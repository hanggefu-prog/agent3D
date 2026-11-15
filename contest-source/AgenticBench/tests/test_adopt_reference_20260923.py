"""No-EDA adoption regressions with real native/hash validation throughout."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
import subprocess

import pytest

from local_bench import reference_baseline as baseline
from local_bench.benchmark_input import validate_benchmark_input
from scripts import adopt_reference_20260923 as adopter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def refresh(stage):
    path = stage / "stage_manifest.json"
    manifest = json.loads(path.read_text())
    for record in manifest["artifacts"].values():
        artifact = stage / record["path"]
        record.update(bytes=artifact.stat().st_size, sha256=digest(artifact))
    write(path, manifest)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Mock only the pinned fingerprint data, not any of the validators."""
    case = "bp_fe"
    archive, source = tmp_path / "archive", tmp_path / "isolated-source"
    source.mkdir()
    (source / "open3dbench").mkdir()
    (source / "open3dbench/__main__.py").write_text("# Isolated source fixture; never execute.\n")
    for command in (["init", "-q"], ["add", "."],
                    ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", "-C", str(source), *command], check=True, capture_output=True)
    run_relative = "user_runs/original/bp_fe/run"
    run = archive / run_relative
    physical = {}
    root_records = {}
    for index, (stage, directory) in enumerate(adopter.STAGES.items()):
        current = run / directory
        current.mkdir(parents=True)
        artifacts = {}
        if stage != "partition":
            names = ["design.def", "design.v", "design.sdc"]
            if stage == "floorplan":
                names.extend(["floorplan.json", "io_plan.csv", "preplan_manifest.json", "pdn_plan.json"])
            for name in names:
                path = current / name
                path.write_text("frozen physical " + directory + "/" + name + "\n")
                record = {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
                physical[directory + "/" + name] = {key: record[key] for key in ("bytes", "sha256")}
                artifacts[name] = record
            write(current / "design_params.json", {
                "def_input": str(current / "design.def"),
                "verilog_input": str(current / "design.v"),
                "sdc_input": str(current / "design.sdc"),
                "lef_input": ["/original/source/platform.lef"], "random_seed": 1000,
            })
            artifacts["design_params"] = {"path": "design_params.json"}
        else:
            (current / "partition_assignment.csv").write_text("inst_name,die\nu1,bottom\nu2,top\n")
            native = {"method": "tp-gnn-flat", "hierarchy": {"mode": "flat", "consumed_as_feature": False},
                      "config": {"decoder": "global-2way", "seed": 1001},
                      "timing_input": str(current / "timing_input")}
            write(current / "tp_gnn_report.json", native)
            write(current / "partition_manifest.json", {
                "method": "tp-gnn-flat", "method_report": native,
                "original_producer": "/original/source",
            })
            write(current / "timing_input/timing_manifest.json", {"producer": "original_legacy_stage20"})
            artifacts = {
                "method_manifest": {"path": "partition_manifest.json"},
                "assignment": {"path": "partition_assignment.csv"},
                "timing_manifest": {"path": "timing_input/timing_manifest.json"},
            }
        manifest = {
            "schema_version": 1, "stage": stage, "status": "success", "method": "tp-gnn-flat",
            "signature": {"original": "unchanged"},
            "runtime_seconds": 11.5 + index,
            "provenance": {"source": {"revision": "original-native-revision", "dirty": False}},
            "artifacts": artifacts,
        }
        write(current / "stage_manifest.json", manifest)
        refresh(current)
        root_records[stage] = {"status": "success", "directory": directory,
                               "runtime_seconds": 11.5 + index, "started_at": "2026-09-23T00:00:00Z"}
    root_records["routing"] = {"status": "failed", "directory": "80_route"}
    write(run / "run.json", {"design": case, "type": "lol", "pdk": "ng45", "status": "failed",
                              "error": "historical downstream failure", "stages": root_records})
    (run / "config.yaml").write_text("design: bp_fe\noriginal_recipe: unchanged\n")
    write(archive / "reference.json", {"reference_id": "fixture-frozen-reference",
                                        "original_root": "/original/source"})
    registry = {"schema_version": 1, "reference_id": "fixture-frozen-reference",
                "reference_json_sha256": digest(archive / "reference.json"),
                "cases": {case: {"run": run_relative, "physical_prefix": physical,
                                 "partition_assignment_sha256": digest(run / "20_partition/partition_assignment.csv")}}}
    identity = tmp_path / "identity.json"
    write(identity, registry)
    monkeypatch.setattr(baseline, "IDENTITY_FILE", identity)
    return {"archive": archive, "source": source, "run": run, "case": case,
            "output": tmp_path / "fresh-bundle", "registry": registry}


def adopt(frozen, **overrides):
    return adopter.adopt_reference_input(reference_root=frozen["archive"], case=frozen["case"],
                                          source=frozen["source"],
                                          output=overrides.pop("output", frozen["output"]), **overrides)


def test_import_preserves_physics_native_identity_and_separates_import_provenance(frozen, monkeypatch):
    # Production archives are sealed read-only. Exercise real copy/rebase and
    # manifest refresh, not only the permissive temporary-directory case.
    archive_paths = [frozen["archive"], *frozen["archive"].rglob("*")]
    for path in archive_paths:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    sealed_identity = {path: (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
                       for path in archive_paths}
    before = snapshot(frozen["archive"])
    original_call = subprocess.run
    commands = []
    def only_git(command, *args, **kwargs):
        assert command[0] == "git", "Importer must not invoke Python, EDA or any other tool"
        commands.append(command)
        return original_call(command, *args, **kwargs)
    monkeypatch.setattr(adopter.subprocess, "run", only_git)
    result = adopt(frozen)
    assert result["status"] == "ok"
    assert result["reference_validation"]["fixed_partition"] is True
    assert result["reference_validation"]["scoring_reference_qualified"] is False
    assert len(commands) == 3
    assert snapshot(frozen["archive"]) == before
    assert {path: (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
            for path in archive_paths} == sealed_identity
    bundle = frozen["output"]
    copied = bundle / "run"
    assert not (copied / "15_timing_extract").exists()
    assert not (copied / "80_route").exists()
    data = json.loads((bundle / "benchmark_input.json").read_text())
    assert set(data["stages"]) == set(adopter.STAGES)
    assert "preparation_plan" not in data
    assert data["command"] == []
    assert data["import_kind"] == "frozen_reference_adoption"
    assert data["current_importer_source"]["dirty"] is False
    assert data["environment"]["open3dbench"]["root"] == str(frozen["source"])
    assert data["reference_adoption"]["physical_stages_executed"] == []
    assert data["reference_adoption"]["source_run_status"] == "failed"
    assert (copied / "config.yaml").read_bytes() == (frozen["run"] / "config.yaml").read_bytes()
    ledger = json.loads((copied / "run.json").read_text())
    original = json.loads((frozen["run"] / "run.json").read_text())
    assert ledger["status"] == "prepared"
    assert "error" not in ledger
    assert ledger["stages"] == {key: original["stages"][key] for key in adopter.STAGES}
    for name in ("partition_assignment.csv", "partition_manifest.json", "tp_gnn_report.json",
                 "timing_input/timing_manifest.json"):
        assert (copied / "20_partition" / name).read_bytes() == (frozen["run"] / "20_partition" / name).read_bytes()
    for stage, directory in adopter.STAGES.items():
        assert (copied / directory).stat().st_mode & stat.S_IWUSR
        assert (copied / directory / "stage_manifest.json").stat().st_mode & stat.S_IWUSR
        manifest = json.loads((copied / directory / "stage_manifest.json").read_text())
        assert manifest["provenance"]["source"]["revision"] == "original-native-revision"
        assert manifest["signature"] == {"original": "unchanged"}
        if stage == "partition":
            continue
        params = json.loads((copied / directory / "design_params.json").read_text())
        assert params["def_input"] == str(copied / directory / "design.def")
        assert params["lef_input"] == ["/original/source/platform.lef"]
    validated = validate_benchmark_input(bundle, task_id="placement_dynamic_parameters_reference_20260923")
    assert validated["partition_method"] == "tp-gnn-flat"
    # Exercise the real workspace consumer rather than a mock of its ledger contract.
    from scripts.workspace_codesign_evaluator import copy_frozen_prefix
    target = bundle.parent / "candidate-run"
    copy_frozen_prefix(copied, target)
    assert json.loads((target / "run.json").read_text())["stages"]["floorplan"] == original["stages"]["floorplan"]


@pytest.mark.parametrize("existing", ["empty-directory", "file", "symlink"])
def test_existing_output_is_never_reused(frozen, existing):
    path = frozen["output"]
    if existing == "empty-directory":
        path.mkdir()
    elif existing == "file":
        path.write_text("existing")
    else:
        path.symlink_to(path.parent / "missing-target")
    with pytest.raises(FileExistsError, match="already exists"):
        adopt(frozen)


@pytest.mark.parametrize("parent", ["archive", "source"])
def test_output_must_be_independent_from_archive_and_source(frozen, parent):
    with pytest.raises(ValueError, match="disjoint"):
        adopt(frozen, output=frozen[parent] / "new-input")


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_dirty_source_is_rejected_without_output(frozen, change):
    path = frozen["source"] / ("open3dbench/__main__.py" if change == "tracked" else "candidate.py")
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="must be clean"):
        adopt(frozen)
    assert not frozen["output"].exists()


def test_hardlink_copy_mode_is_rejected(frozen, monkeypatch):
    monkeypatch.setenv("OPEN3D_LINK_FROZEN_UPSTREAM", "1")
    with pytest.raises(ValueError, match="hard-linked"):
        adopt(frozen)


@pytest.mark.parametrize("target,match", [
    ("reference.json", "index identity"),
    ("run/00_input/design.def", "Physical input differs"),
    ("run/20_partition/partition_assignment.csv", "partition differs"),
    ("run/20_partition/timing_input/timing_manifest.json", "artifact (size|hash) changed"),
])
def test_native_damage_is_not_repaired_by_manifest_refresh(frozen, target, match):
    path = (frozen["run"] / target[4:]) if target.startswith("run/") else frozen["archive"] / target
    path.write_text("corrupted\n")
    before = snapshot(frozen["archive"])
    with pytest.raises(ValueError, match=match):
        adopt(frozen)
    assert snapshot(frozen["archive"]) == before
    assert not frozen["output"].exists()


@pytest.mark.parametrize("change,match", [
    ("unbound-report", "hash-bound"),
    ("wrong-method", "identities disagree"),
    ("hierarchy", "flat-hierarchy"),
    ("decoder", "global-2way"),
])
def test_no_native_identity_relabel_or_unbound_report_admission(frozen, change, match):
    stage = frozen["run"] / "20_partition"
    report = json.loads((stage / "tp_gnn_report.json").read_text())
    if change == "decoder":
        report["config"]["decoder"] = "ta3d-hierarchical"
    elif change == "hierarchy":
        report["hierarchy"]["consumed_as_feature"] = True
    else:
        report["method"] = "tp-gnn"
    write(stage / "tp_gnn_report.json", report)
    if change != "unbound-report":
        method = json.loads((stage / "partition_manifest.json").read_text())
        method["method_report"] = report
        write(stage / "partition_manifest.json", method)
        refresh(stage)
    with pytest.raises(ValueError, match=match):
        adopt(frozen)
    assert not frozen["output"].exists()


@pytest.mark.parametrize("change", ["failed", "missing", "wrong-directory", "wrong-case"])
def test_original_root_ledger_is_required_and_never_invented(frozen, change):
    path = frozen["run"] / "run.json"
    ledger = json.loads(path.read_text())
    if change == "failed":
        ledger["stages"]["floorplan"]["status"] = "failed"
    elif change == "missing":
        ledger["stages"].pop("floorplan")
    elif change == "wrong-directory":
        ledger["stages"]["floorplan"]["directory"] = "fake"
    else:
        ledger["design"] = "ariane133"
    write(path, ledger)
    with pytest.raises(ValueError, match="Frozen"):
        adopt(frozen)
    assert not frozen["output"].exists()


@pytest.mark.parametrize("filename", ["design_params.json", "stage_manifest.json"])
def test_copied_writable_metadata_cannot_link_back_to_archive(frozen, filename):
    path = frozen["run"] / "00_input" / filename
    original = path.read_bytes()
    path.unlink()
    actual = path.with_name("linked-original.json")
    actual.write_bytes(original)
    path.symlink_to(actual.name)
    with pytest.raises(ValueError, match="writable metadata"):
        adopt(frozen)
    assert actual.read_bytes() == original


def test_copy_corruption_is_detected_against_original_fingerprints(frozen, monkeypatch):
    original_copy = adopter.copy_upstream_stages
    def corrupt(source, target, stages):
        original_copy(source, target, stages)
        (target / "00_input/design.def").write_text("wrong copied physics")
    monkeypatch.setattr(adopter, "copy_upstream_stages", corrupt)
    with pytest.raises(ValueError, match="Physical input differs"):
        adopt(frozen)
    assert frozen["output"].exists(), "Preserve failed import for diagnosis; no retry in place"
    with pytest.raises(FileExistsError):
        adopt(frozen)


@pytest.mark.parametrize("change", ["missing", "empty", "file-symlink", "directory-symlink"])
def test_legacy_timing_manifest_must_exist_without_symlink_substitution(frozen, change):
    stage = frozen["run"] / "20_partition"
    manifest = stage / "timing_input/timing_manifest.json"
    # Keep the ordinary native artifact checker valid where possible so this
    # regression also exercises the explicit historical-layout requirement.
    stage_record = json.loads((stage / "stage_manifest.json").read_text())
    stage_record["artifacts"].pop("timing_manifest")
    write(stage / "stage_manifest.json", stage_record)
    if change == "missing":
        manifest.unlink()
    elif change == "empty":
        manifest.write_text("")
    elif change == "file-symlink":
        target = manifest.with_name("original.json")
        manifest.rename(target)
        manifest.symlink_to(target.name)
    else:
        target = stage / "original-timing"
        manifest.parent.rename(target)
        (stage / "timing_input").symlink_to(target.name, target_is_directory=True)
    before = snapshot(frozen["archive"])
    with pytest.raises(ValueError, match="Legacy stage-20 timing manifest"):
        adopt(frozen)
    assert not frozen["output"].exists()
    assert snapshot(frozen["archive"]) == before


def test_cli_has_only_explicit_import_arguments(frozen, capsys):
    result = adopter.main(["--reference-root", str(frozen["archive"]), "--case", frozen["case"],
                           "--source", str(frozen["source"]), "--output", str(frozen["output"])])
    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["reference_validation"]["scoring_reference_qualified"] is False
