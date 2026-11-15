"""Explicit flat producer qualification; never relabel historical bundles."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_bench import benchmark_input as inputs
from local_bench.model import EvaluationResult
from scripts import run_local_task_smoke as smoke
from scripts import prepare_task_smoke_input as preparation


PROFILE = "bp_fe_lol_ng45_tp_gnn_flat_placement"


def report(method="tp-gnn-flat", decoder="global-2way"):
    value = {"method": method, "config": {"decoder": decoder}}
    if method == "tp-gnn-flat":
        value["hierarchy"] = {"mode": "flat", "consumed_as_feature": False}
    return value


def bundle(root, profile_id=PROFILE, native_report=None, stage_method=None, bound=True):
    profile = inputs.load_benchmark_input_profile(profile_id)
    native_report = report() if native_report is None else native_report
    method = native_report["method"]
    stages = {}
    for stage, directory in profile.required_stages.items():
        current = root / "run" / directory
        current.mkdir(parents=True)
        artifact = current / ("partition_manifest.json" if stage == "partition" else "payload.json")
        artifact.write_text(json.dumps({"method": method, "method_report": native_report}))
        record = {"path": artifact.name, "bytes": artifact.stat().st_size,
                  "sha256": inputs._sha256(artifact)}
        manifest = {"schema_version": 1, "stage": stage, "status": "success",
                    "method": stage_method or method,
                    "artifacts": {"method_manifest" if bound else "other": record}}
        path = current / "stage_manifest.json"
        path.write_text(json.dumps(manifest))
        stages[stage] = {"directory": directory, "manifest_sha256": inputs._sha256(path)}
        if stage == "partition":
            (current / "tp_gnn_report.json").write_text(json.dumps(native_report))
    data = {"schema_version": 1, "benchmark_id": profile_id, "run_dir": "run",
            "profile_sha256": inputs._sha256(profile.root / "profile.yaml"),
            "config_sha256": inputs._sha256(profile.config), "stages": stages}
    (root / "benchmark_input.json").write_text(json.dumps(data))
    return root


def test_new_profile_preserves_all_physical_settings_and_legacy_identity():
    old = inputs.load_benchmark_input_profile("bp_fe_lol_ng45_tp_gnn_placement")
    new = inputs.load_benchmark_input_profile(PROFILE)
    before = yaml.safe_load(old.config.read_text())
    after = yaml.safe_load(new.config.read_text())
    assert before["partition"]["algorithm"] == "tp-gnn"
    assert after["partition"].pop("algorithm") == "tp-gnn-flat"
    before["partition"].pop("algorithm")
    assert before == after
    assert old.partition_method is None
    assert new.partition_method == "tp-gnn-flat"
    assert new.compatible_tasks == old.compatible_tasks
    assert list(new.required_stages) == ["input", "floorplan", "timing_extract", "partition"]


def test_hash_bound_flat_producer_qualifies_without_changing_any_artifact(tmp_path):
    root = bundle(tmp_path / "new")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for task in ["placement_dynamic_parameters", "placement_preconditioner_source"]:
        result = inputs.validate_benchmark_input(root, task_id=task)
        assert result["partition_method"] == "tp-gnn-flat"
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("method", ["tp-gnn", "mtkahypar", "tp-gnn-custom"])
def test_new_profile_rejects_different_native_producer(tmp_path, method):
    root = bundle(tmp_path / "new", native_report=report(method))
    with pytest.raises(ValueError, match="TP-GNN"):
        inputs.validate_benchmark_input(root)


@pytest.mark.parametrize("change,match", [
    ("missing_hierarchy", "native flat-hierarchy"),
    ("hierarchy_feature", "native flat-hierarchy"),
    ("different_decoder", "global-2way"),
    ("mixed_identity", "identities disagree"),
    ("unbound", "hash-bound"),
])
def test_new_profile_requires_exact_native_evidence(tmp_path, change, match):
    value = report()
    if change == "missing_hierarchy":
        value.pop("hierarchy")
    elif change == "hierarchy_feature":
        value["hierarchy"]["consumed_as_feature"] = True
    elif change == "different_decoder":
        value["config"]["decoder"] = "ta3d-hierarchical"
    root = bundle(tmp_path / "new", native_report=value, bound=change != "unbound",
                  stage_method="tp-gnn" if change == "mixed_identity" else None)
    with pytest.raises(ValueError, match=match):
        inputs.validate_benchmark_input(root)


def test_unhashed_report_edits_cannot_override_published_evidence(tmp_path):
    root = bundle(tmp_path / "new")
    (root / "run/20_partition/tp_gnn_report.json").write_text(json.dumps(report(decoder="other")))
    with pytest.raises(ValueError, match="hash-bound"):
        inputs.validate_benchmark_input(root)


def test_historical_placement_profile_still_rejects_flat_result(tmp_path):
    root = bundle(tmp_path / "old", "bp_fe_lol_ng45_tp_gnn_placement")
    with pytest.raises(ValueError, match="qualify a new profile"):
        inputs.validate_benchmark_input(root)


def test_historical_native_identity_still_qualifies_unchanged(tmp_path):
    root = bundle(tmp_path / "old", "bp_fe_lol_ng45_tp_gnn_placement",
                  native_report=report("tp-gnn"))
    assert inputs.validate_benchmark_input(root)["partition_method"] == "tp-gnn"


def test_flat_profile_requires_explicit_contract_and_real_timing_stage(tmp_path):
    source = inputs.load_benchmark_input_profile(PROFILE)
    data = yaml.safe_load((source.root / "profile.yaml").read_text())
    data.pop("partition_contract")
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    (tmp_path / "config.yaml").write_bytes(source.config.read_bytes())
    with pytest.raises(ValueError, match="explicit partition contract"):
        inputs.load_benchmark_input_profile(path)
    legacy_plan = [{"stage": name, "directory": inputs.PROFILE_STAGE_DIRS[name], "skip": None}
                   for name in ["input", "floorplan", "partition"]]
    with pytest.raises(ValueError, match="required profile stage: timing_extract"):
        inputs._preparation_stages(source, legacy_plan)


def test_preparation_wrapper_passes_explicit_profile_without_changing_physical_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(preparation, "isolated_clone", lambda source, output: output.mkdir())
    def prepare(profile, **kwargs):
        calls.append((profile, kwargs))
        return {"status": "ok"}
    monkeypatch.setattr(preparation, "prepare_benchmark_input", prepare)
    output = tmp_path / "bundle"
    assert preparation.prepare_inside(smoke.ROOT.parent, PROFILE, output) == {"status": "ok"}
    assert calls == [(PROFILE, {
        "open3dbench_root": tmp_path / "input-source", "output_dir": output,
        "python_executable": Path("/usr/bin/python3"),
        "dreamplace_install": Path("/opt/dreamplace"), "timeout_seconds": 21600,
    })]
    contract = json.loads((tmp_path / "preparation_contract.json").read_text())
    assert contract["profile"] == PROFILE
    assert contract["trusted_patch"] is None


@pytest.mark.parametrize("contract,algorithm", [
    ({"method": "tp-gnn-custom", "decoder": "global-2way"}, "tp-gnn-flat"),
    ({"method": "tp-gnn-flat", "decoder": "ta3d-hierarchical"}, "tp-gnn-flat"),
    ({"method": "tp-gnn-flat", "decoder": "global-2way"}, "tp-gnn"),
])
def test_explicit_profile_cannot_hide_a_config_alias(tmp_path, contract, algorithm):
    source = inputs.load_benchmark_input_profile(PROFILE)
    profile = yaml.safe_load((source.root / "profile.yaml").read_text())
    config = yaml.safe_load(source.config.read_text())
    profile["partition_contract"] = contract
    config["partition"]["algorithm"] = algorithm
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="(partition contract|configure tp-gnn-flat)"):
        inputs.load_benchmark_input_profile(path)


@pytest.mark.parametrize("task", ["placement_parameters_reference_20260923", "placement_initialization"])
def test_current_tasks_reject_historical_bundle_before_execution(tmp_path, task):
    root = bundle(tmp_path / "input")
    def contained(path):
        resolved = Path(path).resolve()
        resolved.relative_to(tmp_path)
        return resolved
    runtime = SimpleNamespace(source=smoke.ROOT.parent, framework=smoke.ROOT, batch=tmp_path,
                              source_commit="test", image="sha256:" + "a" * 64, cpus=16,
                              memory_gb=48, share_gpu=True, min_gpu_free_mb=2048, contained=contained)
    class Transport:
        physical_calls = 0
        receipts = []
        def __call__(self, *args, **kwargs):
            self.physical_calls += 1
            assert kwargs["base_config"] == inputs.load_benchmark_input_profile(PROFILE).config
            assert kwargs["baseline_run"] == root / "run"
            # Stop the fixture at reference qualification, before any model call.
            return EvaluationResult({"runs_successfully": 0}, {})
    with pytest.raises(ValueError, match="not compatible with task"):
        smoke.run_smoke(runtime, task=task, output=tmp_path / "out", benchmark_input=root,
                        transport_factory=lambda _: pytest.fail("No physical work may be called"),
                        provider_factory=lambda *args: pytest.fail("No model may be called"))
