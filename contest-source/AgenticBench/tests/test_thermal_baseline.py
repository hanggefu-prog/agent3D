"""The thermal reference must not silently use the general LoL defaults."""
import json
from pathlib import Path

import pytest
import yaml

from global_bench.verifiers import open3dbench_codesign_verifier as trusted
from local_bench.evaluation import _write_resolved_config
from local_bench.metrics import _validate_thermal_frontend_methods, tp_gnn_placement_thermal_metrics
from local_bench.model import load_candidate, load_task
from local_bench.verifier import verify_candidate


PROFILE = Path(__file__).resolve().parents[1] / "evaluation_profiles/tp_gnn_placement_thermal"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def frontend(root):
    for stage, directory, method in (
        ("partition", "20_partition", "tp-gnn"),
        ("placement", "30_placement", "concurrent-dreamplace"),
    ):
        write(root / directory / "stage_manifest.json", dict(stage=stage, status="success", method=method))
    write(root / "20_partition/partition_manifest.json", {"method": "tp-gnn"})
    write(root / "20_partition/tp_gnn_report.json", {"method": "tp-gnn", "config": {"decoder": "global-2way"}})


def thermal_run(root, average=322.):
    frontend(root)
    write(root / "run.json", {"status": "success", "stages": {
        name: {"status": "success"} for name in trusted.THERMAL_STAGES}})
    write(root / "20_partition/partition_metrics.json", dict(instance_count=10, top_count=5,
          bottom_count=5, area_balance_error=0., cut_net_count=2))
    write(root / "30_placement/metrics.json", {
        "dies": {die: {"missing_connection_object_count": 0} for die in ("bottom", "top")},
        "total_object_origin_hpwl_um": 100.})
    write(root / "95_thermal/metrics.json", dict(stage="thermal", status="success",
          algorithm="hotspot-steady-grid", warning_count=0, grid_size=10, ambient_k=318.15,
          temperature={"bottom": {"average_k": average, "tile_count": 100},
                       "top": {"average_k": average, "tile_count": 100},
                       "global_average_k": average, "global_maximum_c": average - 273.15},
          power={"total_w": 1., "conservation_error_w": 0.}, stack={"model_status": "provisional"}))


def test_initial_thermal_candidate_explicitly_selects_both_frontend_methods(tmp_path):
    spec = load_task(PROFILE)
    candidate = load_candidate(spec.initial_candidate)
    verify_candidate(spec, candidate)
    assert candidate["source_patch"] == ""
    assert candidate["config_overrides"] == {
        "partition": {"algorithm": "tp-gnn"}, "placement": {"algorithm": "concurrent-dreamplace"}}
    resolved = _write_resolved_config(spec, spec.initial_candidate, tmp_path / "resolved.yaml", candidate)
    for stage, method in (("partition", "tp-gnn"), ("placement", "concurrent-dreamplace")):
        assert resolved[stage]["algorithm"] == method
    assert resolved["thermal"] == {"enabled": True, "grid_size": 10, "ambient_k": 318.15}
    assert resolved["partition"]["seed"] == 1001 and resolved["placement"]["seed"] == 1000
    assert resolved["opt"] == {"algorithm": "openroad"}  # no new two-pass override


@pytest.mark.parametrize("stage,method", [("partition", "mtkahypar"), ("placement", "movable-hbt")])
def test_focused_adapter_rejects_other_selectors(stage, method):
    spec = load_task(PROFILE)
    candidate = load_candidate(spec.initial_candidate)
    candidate["config_overrides"][stage]["algorithm"] = method
    with pytest.raises(ValueError, match="not in"):
        verify_candidate(spec, candidate)


@pytest.mark.parametrize("placement", [None, "concurrent-dreamplace", "movable-hbt"])
def test_older_candidate_without_placement_selector_is_resolved_explicitly(tmp_path, placement):
    spec = load_task(PROFILE)
    candidate = load_candidate(spec.initial_candidate)
    del candidate["config_overrides"]["placement"]
    verify_candidate(spec, candidate)
    base = {"type": "lol"}
    if placement:
        base["placement"] = {"algorithm": placement}
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    if placement == "movable-hbt":
        with pytest.raises(ValueError, match="placement.algorithm"):
            _write_resolved_config(spec, spec.initial_candidate, tmp_path / "out.yaml", candidate, base_path)
    else:
        resolved = _write_resolved_config(spec, spec.initial_candidate, tmp_path / "out.yaml", candidate, base_path)
        assert resolved["placement"]["algorithm"] == "concurrent-dreamplace"


@pytest.mark.parametrize("validate", [_validate_thermal_frontend_methods, trusted._validate_thermal_reference_frontend])
@pytest.mark.parametrize("target", ["partition", "placement", "artifact", "missing"])
def test_reference_method_checks_actual_stage_and_partition_provenance(tmp_path, validate, target):
    frontend(tmp_path)
    validate(tmp_path)
    if target == "missing":
        (tmp_path / "30_placement/stage_manifest.json").unlink()
    elif target == "artifact":
        write(tmp_path / "20_partition/partition_manifest.json", {"method": "mtkahypar"})
    else:
        directory = "20_partition" if target == "partition" else "30_placement"
        write(tmp_path / directory / "stage_manifest.json", dict(stage=target, status="success", method="other"))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(tmp_path)


def test_thermal_scoring_keeps_temperature_objective_and_rejects_wrong_reference(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    thermal_run(candidate, 321.)
    thermal_run(reference, 322.)
    assert tp_gnn_placement_thermal_metrics(reference, None)["combined_score"] == 1.
    result = tp_gnn_placement_thermal_metrics(candidate, reference)
    assert result["combined_score"] == pytest.approx(322. / 321.)
    assert result["global_average_temperature_k"] == 321.
    write(reference / "20_partition/partition_manifest.json", {"method": "mtkahypar"})
    with pytest.raises(ValueError, match="not TP-GNN"):
        tp_gnn_placement_thermal_metrics(candidate, reference)


def test_independent_verifier_checks_reference_without_limiting_candidate_algorithm(tmp_path, monkeypatch):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    frontend(reference)
    seen = []
    def row(root):
        seen.append(root)
        return {"stack": {}, "global_average_temperature_k": 321. if root == candidate else 322.}
    monkeypatch.setattr(trusted, "_thermal_row", row)
    assert trusted._score_thermal(candidate, reference)["combined_score"] == pytest.approx(322. / 321.)
    assert seen == [candidate, reference]
    write(reference / "20_partition/partition_manifest.json", {"method": "mtkahypar"})
    with pytest.raises(ValueError, match="not TP-GNN"):
        trusted._score_thermal(candidate, reference)


@pytest.mark.parametrize("decoder", ["ta3d-hierarchical", None])
def test_reference_decoder_cannot_be_ta3d_or_unknown(tmp_path, decoder):
    thermal_run(tmp_path)
    write(tmp_path / "20_partition/tp_gnn_report.json", {"method": "tp-gnn", "config": {"decoder": decoder}})
    with pytest.raises(ValueError, match="global-2way"):
        trusted._validate_thermal_reference_frontend(tmp_path)
    with pytest.raises(ValueError, match="global-2way"):
        tp_gnn_placement_thermal_metrics(tmp_path, None)


def test_candidate_decoder_may_evolve_against_unchanged_reference(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    thermal_run(candidate, 321.)
    thermal_run(reference, 322.)
    write(candidate / "20_partition/tp_gnn_report.json", {"method": "tp-gnn", "config": {"decoder": "new-method"}})
    assert tp_gnn_placement_thermal_metrics(candidate, reference)["combined_score"] > 1.


@pytest.mark.parametrize("mode,consumed,accepted", [("flat", False, True),
    ("flat", True, False), ("unknown", False, False), (None, False, False)])
def test_standalone_reference_accepts_only_native_flat_provenance(tmp_path, mode, consumed, accepted):
    frontend(tmp_path)
    write(tmp_path / "20_partition/stage_manifest.json", {
        "stage": "partition", "status": "success", "method": "tp-gnn-flat"})
    write(tmp_path / "20_partition/partition_manifest.json", {"method": "tp-gnn-flat"})
    write(tmp_path / "20_partition/tp_gnn_report.json", {"method": "tp-gnn-flat",
        "config": {"decoder": "global-2way"},
        "hierarchy": {"mode": mode, "consumed_as_feature": consumed}})
    if accepted:
        trusted._validate_thermal_reference_frontend(tmp_path)
    else:
        with pytest.raises(ValueError, match="provenance"):
            trusted._validate_thermal_reference_frontend(tmp_path)


def test_standalone_reference_rejects_relabelled_flat(tmp_path):
    frontend(tmp_path)
    write(tmp_path / "20_partition/tp_gnn_report.json", {"method": "tp-gnn",
        "config": {"decoder": "global-2way"},
        "hierarchy": {"mode": "flat", "consumed_as_feature": False}})
    with pytest.raises(ValueError, match="relabel"):
        trusted._validate_thermal_reference_frontend(tmp_path)


@pytest.mark.parametrize("method,reason,accepted", [
    ("mtkahypar", "selected partitioner does not consume TP-GNN timing features", True),
    ("tp-gnn-flat", "selected partitioner does not consume TP-GNN timing features", False),
    ("mtkahypar", "failed timing extraction", False),
])
def test_chain_accepts_only_inapplicable_timing_features(tmp_path, monkeypatch, method, reason, accepted):
    frontend(tmp_path)
    write(tmp_path / "20_partition/partition_manifest.json", {"method": method})
    write(tmp_path / "20_partition/stage_manifest.json", {"method": method})
    write(tmp_path / "run.json", {"status": "success", "stages": {
        "timing_extract": {"status": "skipped", "reason": reason},
        "partition": {"status": "success"}}})
    seen = []
    monkeypatch.setattr(trusted, "_validate_stage", lambda root, stage: seen.append(stage))
    if accepted:
        trusted._validate_chain(tmp_path, ("timing_extract", "partition"))
        assert seen == ["partition"]
    else:
        with pytest.raises(ValueError, match="did not complete"):
            trusted._validate_chain(tmp_path, ("timing_extract", "partition"))
