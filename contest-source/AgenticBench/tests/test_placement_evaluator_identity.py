"""Exercise the real placement evaluator gate; stop before native planning/EDA."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from local_bench import evaluation
from local_bench.model import load_candidate, load_task
from local_bench.program_bridge import initial_program_text, materialize_program


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("placement_preconditioner", "placement_initialization")


def frozen_partition(root, method="tp-gnn-flat", *, mutation=None):
    stage = root / "20_partition"
    stage.mkdir(parents=True)
    report = {"method": method, "config": {"decoder": "global-2way"}}
    if method == "tp-gnn-flat":
        report["hierarchy"] = {"mode": "flat", "consumed_as_feature": False}
    if mutation == "hierarchy":
        report["hierarchy"]["consumed_as_feature"] = True
    if mutation == "decoder":
        report["config"]["decoder"] = "ta3d-hierarchical"
    artifact = stage / "partition_manifest.json"
    artifact.write_text(json.dumps({"method": method, "method_report": report}))
    record = {"path": artifact.name, "bytes": artifact.stat().st_size,
              "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    if mutation == "hash":
        record["sha256"] = "0" * 64
    if mutation == "bytes":
        record["bytes"] += 1
    (stage / "stage_manifest.json").write_text(json.dumps({
        "status": "failed" if mutation == "status" else "success",
        "method": "tp-gnn" if mutation == "identity" else method,
        "artifacts": {"other" if mutation == "unbound" else "method_manifest": record},
    }))
    if mutation == "report":
        report["edited_without_manifest"] = True
    (stage / "tp_gnn_report.json").write_text(json.dumps(report))
    return root


def invoke(tmp_path, monkeypatch, task, expected, actual, *, mutation=None,
           override=None, benchmark_id=None, use_default=False):
    spec = load_task(ROOT / "local_tasks" / task)
    config = yaml.safe_load(spec.base_config.read_text())
    config["partition"]["algorithm"] = expected
    base = tmp_path / "trusted-config.yaml"
    base.write_text(yaml.safe_dump(config))
    candidate = materialize_program(
        spec, initial_program_text(spec, load_candidate(spec.initial_candidate)), ROOT.parent)
    if override:
        # Current component tasks reject attempts to select their own producer.
        candidate.setdefault("config_overrides", {}).setdefault("partition", {})["algorithm"] = override
    if benchmark_id is not None:
        spec = replace(spec, benchmark_input_id=benchmark_id)
    if use_default:
        spec = replace(spec, base_config=base)
    monkeypatch.setattr(evaluation, "load_task", lambda _: spec)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "candidate.json").write_text(json.dumps(candidate))
    run = frozen_partition(tmp_path / "frozen", actual, mutation=mutation)
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    calls = []

    def stop_at_plan(command, **kwargs):
        assert command[1:3] == ["-m", "open3dbench"] and "plan" in command
        calls.append(command)
        raise RuntimeError("test stopped at native plan; no EDA launched")

    monkeypatch.setattr(evaluation.subprocess, "run", stop_at_plan)
    result = evaluation.evaluate_task(
        spec.root, candidate_dir, ROOT.parent, tmp_path / "result",
        baseline_run=run, base_config=None if use_default else base, dry_run=False,
    )
    assert before == {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    if not override:
        assert result.artifacts["cascade"][0] == {"name": "policy", "status": "passed"}
    return result, calls


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", ["tp-gnn", "tp-gnn-flat"])
def test_real_evaluator_accepts_selected_trusted_identity_before_native_plan(tmp_path, monkeypatch, task, method):
    result, calls = invoke(tmp_path, monkeypatch, task, method, method)
    assert len(calls) == 1
    assert "test stopped at native plan" in result.artifacts["error_message"]
    check = result.artifacts["placement_input_validation"]
    assert check["status"] == "passed" and check["partition_method"] == method
    assert check["hash_bound_report_required"] == (method == "tp-gnn-flat")
    assert check["base_config_sha256"] == hashlib.sha256((tmp_path / "trusted-config.yaml").read_bytes()).hexdigest()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("expected,actual", [("tp-gnn", "tp-gnn-flat"), ("tp-gnn-flat", "tp-gnn")])
def test_real_evaluator_rejects_identity_mismatch_in_both_directions(tmp_path, monkeypatch, task, expected, actual):
    result, calls = invoke(tmp_path, monkeypatch, task, expected, actual)
    assert calls == []
    assert "qualify a new profile" in result.artifacts["error_message"]


@pytest.mark.parametrize("expected,actual", [("tp-gnn", "tp-gnn-flat"), ("tp-gnn-flat", "tp-gnn")])
def test_candidate_override_cannot_change_reference_identity(tmp_path, monkeypatch, expected, actual):
    result, calls = invoke(tmp_path, monkeypatch, TASKS[0], expected, actual, override=actual)
    assert calls == []
    assert result.metrics["runs_successfully"] == 0
    assert "partition" in result.artifacts["error_message"]
    assert not (tmp_path / "result/resolved_config.yaml").exists()


@pytest.mark.parametrize("mutation", ["hash", "bytes", "report", "unbound", "hierarchy", "decoder", "identity", "status"])
def test_real_flat_gate_keeps_native_provenance_and_decoder_checks(tmp_path, monkeypatch, mutation):
    result, calls = invoke(tmp_path, monkeypatch, TASKS[0], "tp-gnn-flat", "tp-gnn-flat", mutation=mutation)
    assert calls == []
    assert result.metrics["runs_successfully"] == 0
    assert result.artifacts["error_type"] == "ValueError"


@pytest.mark.parametrize("method", ["mtkahypar", "tp-gnn-custom", None])
def test_unknown_trusted_identity_does_not_disable_the_guard(tmp_path, monkeypatch, method):
    result, calls = invoke(tmp_path, monkeypatch, TASKS[0], method, "tp-gnn-flat")
    assert calls == []
    assert "explicit trusted TP-GNN producer" in result.artifacts["error_message"]


def test_guard_does_not_depend_on_old_profile_suffix(tmp_path, monkeypatch):
    result, calls = invoke(tmp_path, monkeypatch, TASKS[0], "tp-gnn-flat", "tp-gnn-flat",
                           benchmark_id="explicit-flat-input-v2", use_default=True)
    assert len(calls) == 1
    assert result.artifacts["placement_input_validation"]["partition_method"] == "tp-gnn-flat"
