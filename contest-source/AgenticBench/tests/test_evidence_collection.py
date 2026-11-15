import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d.evidence import collect_evidence
from agentic3d.knowledge import digest


def local_driver():
    # Match OpenSSH's joined command semantics, including quoted run paths.
    return SimpleNamespace(_remote=lambda *args: subprocess.run(
        ["/bin/sh", "-c", " ".join(args[2:])], capture_output=True, text=True))


def artifact(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_collects_current_and_legacy_stages_with_stable_citations(tmp_path):
    run = tmp_path / "run with spaces"
    names = ["run.json", "00_input/stage_manifest.json", "10_floorplan/metrics.json",
             "15_timing_extract/timing_extract.log", "75_global_route/metrics.json",
             "80_route/top/route_status.tsv", "80_routing/bottom/route.log"]
    for name in names:
        artifact(run, name, "evidence for " + name)
    output = tmp_path / "agent-workspace/evidence/reference"
    report = collect_evidence(local_driver(), str(run), output)
    assert report["error"] is None
    assert {item["path"] for item in report["files"]} == set(names)
    assert json.loads((output / "index.json").read_text()) == report
    for item in report["files"]:
        assert digest(Path(item["local_file"]).read_text()) == item["sha256"]


def test_prioritizes_old_manifest_and_failure_over_newer_noise(tmp_path):
    run = tmp_path / "run"
    required = ["30_placement/stage_manifest.json", "30_placement/metrics.json",
                "30_placement/die_init/failure.json", "30_placement/run.log"]
    for name in required:
        os.utime(artifact(run, name, "important failure evidence"), (1, 1))
    for number in range(40):
        artifact(run, f"30_placement/noise-{number:03d}.json", "not essential")
    report = collect_evidence(local_driver(), str(run), tmp_path / "evidence")
    assert report["error"] is None
    assert len(report["files"]) == 12
    assert set(required) <= {item["path"] for item in report["files"]}
    assert report["warnings"]


def test_bounded_head_first_error_tail_and_secret_redaction(tmp_path):
    run = tmp_path / "run"
    body = ("HEADER\n" + "ok\n" * 4000 + "[ERROR CTS-0082] no valid clock\n"
            + "continue\n" * 10000 + '\n{"api_key": "sensitive-value"}\n'
            + "Authorization: Bearer hidden-bearer\npassword=hidden-password\nFINAL\n")
    artifact(run, "70_cts/run.log", body)
    report = collect_evidence(local_driver(), str(run), tmp_path / "evidence")
    item = report["files"][0]
    text = Path(item["local_file"]).read_text()
    assert item["truncated"] and len(text) < 10000
    assert "HEADER" in text and "CTS-0082" in text and "FINAL" in text
    assert "sensitive-value" not in text and "hidden-bearer" not in text
    assert "hidden-password" not in text


def test_refuses_symlink_files_directories_and_private_names(tmp_path):
    run = tmp_path / "run"
    artifact(run, "30_placement/run.log", "allowed")
    outside = artifact(tmp_path, "external/run.log", "outside must not leak")
    (run / "30_placement/escape.log").symlink_to(outside)
    (run / "30_placement/escape-dir").symlink_to(outside.parent, target_is_directory=True)
    (run / "80_route").symlink_to(outside.parent, target_is_directory=True)
    for name in ("credentials.json", "api_key.txt", "auth.json", "environment.json",
                 "secrets/nested.json", ".hidden.json"):
        artifact(run, "30_placement/" + name, "private must not be read")
    report = collect_evidence(local_driver(), str(run), tmp_path / "evidence")
    assert report["error"] is None
    assert [item["path"] for item in report["files"]] == ["30_placement/run.log"]


@pytest.mark.parametrize("exists", [True, False])
def test_missing_or_empty_run_reports_evidence_failure(tmp_path, exists):
    run = tmp_path / "run"
    if exists:
        run.mkdir()
    report = collect_evidence(local_driver(), str(run), tmp_path / "evidence")
    assert report["error"] and report["files"] == []


def test_remote_failure_does_not_publish_error_output_as_evidence(tmp_path):
    fake = SimpleNamespace(_remote=lambda *args: SimpleNamespace(
        returncode=1, stdout="private remote error detail"))
    report = collect_evidence(fake, "/run", tmp_path / "evidence")
    assert report["error"] == "bounded evidence read exit 1"
    assert report["files"] == []
    assert "private" not in (tmp_path / "evidence/index.json").read_text()


def test_destination_ancestor_symlink_cannot_write_outside_workspace(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence").symlink_to(outside, target_is_directory=True)
    fake = SimpleNamespace(_remote=lambda *args: pytest.fail("unsafe destination must reject before reading"))
    report = collect_evidence(fake, "/run", workspace / "evidence/reference")
    assert report["error"] and report["files"] == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", ["00.txt", "index.json", "index.json.tmp"])
def test_destination_leaf_symlink_cannot_overwrite_existing_file(tmp_path, name):
    run = tmp_path / "run"
    artifact(run, "30_placement/run.log", "safe log")
    destination = tmp_path / "evidence"
    destination.mkdir()
    outside = artifact(tmp_path, "outside.txt", "original")
    (destination / name).symlink_to(outside)
    report = collect_evidence(local_driver(), str(run), destination)
    assert report["error"] and report["files"] == []
    assert outside.read_text() == "original"


@pytest.mark.parametrize("snapshot", [
    [], None, {}, {"files": None, "errors": [], "warnings": []},
    {"files": [], "errors": [], "warnings": []},
    {"files": [], "errors": 1, "warnings": []},
    {"files": [None], "errors": [], "warnings": []},
    {"files": [{"path": "../private", "size": 1, "truncated": False,
                "excerpt": "complete", "text": "bad"}], "errors": [], "warnings": []},
    {"files": [{"path": "run.json", "size": 1, "truncated": False,
                "excerpt": "complete", "text": "x" * 10001}], "errors": [], "warnings": []},
])
def test_malformed_snapshot_fails_before_publishing_text(tmp_path, snapshot):
    fake = SimpleNamespace(_remote=lambda *args: SimpleNamespace(
        returncode=0, stdout=json.dumps(snapshot)))
    destination = tmp_path / "evidence"
    report = collect_evidence(fake, "/run", destination)
    assert report["error"] and report["files"] == []
    assert list(destination.glob("*.txt")) == []


def test_compact_evaluator_retention_is_readable_without_run_directory(tmp_path):
    from local_bench.evaluation import _discard_heavy_evaluation_artifacts

    output = tmp_path / "evaluation"
    run = output / "run"
    artifact(run, "30_placement/metrics.json", '{"hpwl": 123}')
    artifact(run, "run.json", '{"status": "success"}')
    artifact(output, "stage.log", "evaluation completed")
    artifact(output, "plan.log", "legal plan")
    artifacts = {"output_dir": str(output), "run_dir": str(run)}
    _discard_heavy_evaluation_artifacts(output, artifacts)
    artifact(output, "evaluation_result.json", json.dumps({
        "metrics": {"runs_successfully": 1, "combined_score": 1.05},
        "artifacts": artifacts,
    }))
    assert not run.exists()
    report = collect_evidence(local_driver(), str(run), tmp_path / "evidence")
    assert report["error"] is None
    assert {item["path"] for item in report["files"]} == {
        "evaluator/evaluation_result.json", "evaluator/stage.log", "evaluator/plan.log",
        "evaluator/retained_artifacts/placement_metrics.json", "evaluator/retained_artifacts/run.json",
    }


def test_failed_evaluation_before_run_creation_retains_diagnostic_feedback(tmp_path):
    output = tmp_path / "evaluation"
    artifact(output, "openroad-build.log", "[ERROR] candidate compilation failed")
    artifact(output, "evaluation_result.json", json.dumps({
        "metrics": {"runs_successfully": 0, "combined_score": 0},
        "artifacts": {"output_dir": str(output), "error_type": "ValueError",
                      "error_message": "candidate configuration rejected"},
    }))
    report = collect_evidence(local_driver(), str(output / "run"), tmp_path / "evidence")
    assert report["error"] is None
    assert {item["path"] for item in report["files"]} == {
        "evaluator/evaluation_result.json", "evaluator/openroad-build.log"}
    text = Path(report["files"][0]["local_file"]).read_text()
    assert "candidate configuration rejected" in text


@pytest.mark.parametrize("mismatch", ["output_dir", "run_dir", "evaluation_result"])
def test_evaluator_envelope_must_identify_exact_requested_run(tmp_path, mismatch):
    output = tmp_path / "evaluation"
    artifacts = {"output_dir": str(output), "run_dir": str(output / "run"),
                 "evaluation_result": str(output / "evaluation_result.json"),
                 "error_type": "ValueError"}
    artifacts[mismatch] = str(tmp_path / "unrelated")
    artifact(output, "evaluation_result.json", json.dumps({
        "metrics": {"runs_successfully": 0}, "artifacts": artifacts,
    }))
    report = collect_evidence(local_driver(), str(output / "run"), tmp_path / "evidence")
    assert report["error"] and report["files"] == []


def test_evaluator_fallback_uses_fixed_allowlist_not_artifact_paths(tmp_path):
    output = tmp_path / "evaluation"
    external = artifact(tmp_path, "other_case/metrics.json", "another case must not leak")
    artifact(output, "unrelated.json", "not an allowed artifact")
    retained = output / "retained_artifacts"
    retained.mkdir()
    (retained / "placement_metrics.json").symlink_to(external)
    artifact(output, "evaluation_result.json", json.dumps({
        "metrics": {"runs_successfully": 0},
        "artifacts": {"output_dir": str(output), "error_type": "ValueError",
                      "artifact_retention": {"retained_evidence": {"evil": str(external)}}},
    }))
    report = collect_evidence(local_driver(), str(output / "run"), tmp_path / "evidence")
    assert report["error"] is None
    assert [item["path"] for item in report["files"]] == ["evaluator/evaluation_result.json"]
    assert "another case must not leak" not in Path(report["files"][0]["local_file"]).read_text()
