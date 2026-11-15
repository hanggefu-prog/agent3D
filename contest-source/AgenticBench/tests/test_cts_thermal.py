import json
import hashlib
from pathlib import Path

import pytest

from local_bench.metrics import tp_gnn_placement_cts_thermal_metrics as score
from local_bench.evaluation import EVALUATOR_SOURCE_PATCHES, STA_TWO_PASS_TASK_KINDS
from local_bench.cts_thermal import snapshot_command


def test_reference_power_threads_are_explicit_and_legacy_unchanged():
    args = (Path("/repo"), Path("/run"), {"pdk": "ng45"}, "python3", {})
    legacy = snapshot_command(*args)
    aligned = snapshot_command(*args, power_threads=2)
    assert legacy[legacy.index("--threads") + 1] == "32"
    assert aligned[aligned.index("--threads") + 1] == "2"
    for value in (False, 0, -1, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            snapshot_command(*args, power_threads=value)
from local_bench.model import load_task
from agentic3d.codesign_endpoint import final_score, PROFILES
from test_sta_codesign import _write_run
from test_thermal_baseline import thermal_run, write
from global_bench.verifiers import open3dbench_codesign_verifier as trusted
from local_bench.workspace import copy_upstream_stages
from local_bench.metrics import extract_metrics


def cts_run(root, average=322., passes=2):
    _write_run(root, -1., -10., passes=passes)
    thermal_run(root, average)
    write(root / "run.json", {"status": "success", "stages": {
        key: {"status": "success"} for key in
        ("input", "floorplan", "partition", "placement", "pdn", "hbt", "opt", "cts")}})
    write(root / "70_cts/metrics.json", {"status": "success", "algorithm": "openroad-coordinated-insertion-delay"})
    write(root / "70_cts/checkpoint.json", {"stage": "cts", "ready_for_route": True})
    metrics = json.loads((root / "95_thermal/metrics.json").read_text())
    metrics.update(stage="thermal_snapshot", source_stage="cts",
                   power_extraction={d: {"instance_count": 10, "warning_count": 0} for d in ("bottom", "top")})
    metrics["power"].update(activity_model="vectorless-primary-input-0.1-placement-estimated-rc",
                            provenance={"model": "die-local-openroad-placement-estimated",
                                        "checkpoint": str(root / "70_cts/checkpoint.json")})
    write(root / "95_thermal_snapshot/metrics.json", metrics)
    write(root / "95_thermal_snapshot/snapshot_manifest.json", {
        "schema_version": 1, "status": "success", "source_stage": "cts",
        "source_checkpoint": str(root / "70_cts/checkpoint.json"),
        "metrics": "metrics.json", "grid_size": 10, "ambient_k": 318.15})


def test_new_boundary_shares_exact_timing_patch_and_has_no_routing():
    spec = load_task(Path(__file__).resolve().parents[1] / "evaluation_profiles" / PROFILES["thermal"])
    assert spec.kind in STA_TWO_PASS_TASK_KINDS
    assert spec.start_stage == "partition" and spec.stop_stage == "cts"
    assert EVALUATOR_SOURCE_PATCHES[spec.kind] == EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_sta"]
    command = snapshot_command(Path("/repo"), Path("/run"), {"pdk": "ng45"}, "python3", {})
    assert "/run/70_cts/checkpoint.json" in command
    assert "routing" not in " ".join(command)


def test_cts_thermal_score_and_legacy_isolation(tmp_path):
    c, r = tmp_path / "c", tmp_path / "r"
    cts_run(c, 321.); cts_run(r, 322.)
    metrics = score(c, r)
    assert metrics["combined_score"] == pytest.approx(322. / 321.)
    assert final_score({"metrics": metrics}, "thermal") == metrics["combined_score"]
    thermal_run(tmp_path / "old")
    with pytest.raises(ValueError, match="stage set"):
        score(c, tmp_path / "old")


def test_independent_verifier_agrees_with_profile(tmp_path, monkeypatch):
    # Stage hashing and seeds have dedicated tests; do not bypass any numeric,
    # method, two-pass, CTS, power or temperature gate here.
    monkeypatch.setattr(trusted, "_validate_chain", lambda *args: None)
    monkeypatch.setattr(trusted, "_validate_seed_contract", lambda *args, **kwargs: None)
    c, r = tmp_path / "c", tmp_path / "r"
    cts_run(c, 321.); cts_run(r, 322.)
    actual = trusted._score_cts_thermal(c, r)
    assert actual["combined_score"] == score(c, r)["combined_score"]
    write(c / "70_cts/checkpoint.json", {"stage": "opt"})
    with pytest.raises(ValueError, match="completed CTS"):
        trusted._score_cts_thermal(c, r)


def native_cts_identity(root):
    """Use the current CTS producer's metrics/checkpoint identity verbatim."""
    write(root / "70_cts/metrics.json", {
        "status": "success", "algorithm": "openroad-balanced"})
    write(root / "70_cts/checkpoint.json", {
        "stage": "cts", "ready_for_route": True,
        "source_checkpoint": "../60_opt/checkpoint.json",
        "cts": {"algorithm": "openroad-balanced", "metrics": "metrics.json"}})


def test_current_native_cts_identity_is_supported_without_relabeling(tmp_path, monkeypatch):
    monkeypatch.setattr(trusted, "_validate_chain", lambda *args: None)
    monkeypatch.setattr(trusted, "_validate_seed_contract", lambda *args, **kwargs: None)
    c, r = tmp_path / "candidate", tmp_path / "reference"
    for root, temperature in ((c, 321.), (r, 322.)):
        cts_run(root, temperature)
        native_cts_identity(root)
    before = (c / "70_cts/metrics.json").read_bytes()
    assert score(c, r)["combined_score"] == pytest.approx(322. / 321.)
    result = trusted._score_cts_thermal(c, r)
    assert result["combined_score"] == pytest.approx(322. / 321.)
    assert result["cts_algorithm"] == "openroad-balanced"
    assert (c / "70_cts/metrics.json").read_bytes() == before


@pytest.mark.parametrize("scorer", [score, trusted._score_cts_thermal])
def test_different_native_and_legacy_cts_references_are_not_mixed(tmp_path, monkeypatch, scorer):
    monkeypatch.setattr(trusted, "_validate_chain", lambda *args: None)
    monkeypatch.setattr(trusted, "_validate_seed_contract", lambda *args, **kwargs: None)
    c, r = tmp_path / "candidate", tmp_path / "reference"
    cts_run(c); cts_run(r)
    native_cts_identity(c)
    with pytest.raises(ValueError, match="CTS"):
        scorer(c, r)


@pytest.mark.parametrize("scorer", [score, trusted._score_cts_thermal])
@pytest.mark.parametrize("fault", [
    "algorithm", "checkpoint_identity", "checkpoint_metrics", "snapshot_checkpoint",
    "power_checkpoint", "snapshot_schema", "snapshot_metrics", "snapshot_grid", "snapshot_ambient",
])
def test_native_cts_snapshot_contract_rejects_mismatched_evidence(tmp_path, monkeypatch, scorer, fault):
    monkeypatch.setattr(trusted, "_validate_chain", lambda *args: None)
    monkeypatch.setattr(trusted, "_validate_seed_contract", lambda *args, **kwargs: None)
    c, r = tmp_path / "candidate", tmp_path / "reference"
    cts_run(c); cts_run(r)
    native_cts_identity(c); native_cts_identity(r)
    if fault == "algorithm":
        path = c / "70_cts/metrics.json"
        data = json.loads(path.read_text()); data["algorithm"] = "unregistered-cts"
    elif fault.startswith("checkpoint_"):
        path = c / "70_cts/checkpoint.json"
        data = json.loads(path.read_text())
        data["cts"]["algorithm" if fault == "checkpoint_identity" else "metrics"] = "wrong"
    elif fault == "power_checkpoint":
        path = c / "95_thermal_snapshot/metrics.json"
        data = json.loads(path.read_text())
        data["power"]["provenance"]["checkpoint"] = str(r / "70_cts/checkpoint.json")
    else:
        path = c / "95_thermal_snapshot/snapshot_manifest.json"
        data = json.loads(path.read_text())
        key, value = {
            "snapshot_checkpoint": ("source_checkpoint", str(r / "70_cts/checkpoint.json")),
            "snapshot_schema": ("schema_version", 2),
            "snapshot_metrics": ("metrics", "../old/metrics.json"),
            "snapshot_grid": ("grid_size", 8),
            "snapshot_ambient": ("ambient_k", 300.),
        }[fault]
        data[key] = value
    write(path, data)
    with pytest.raises(ValueError):
        scorer(c, r)


def test_snapshot_command_binds_current_native_cli_and_function_signature(tmp_path):
    import inspect
    from open3dbench.backend import thermal_snapshot

    command = snapshot_command(tmp_path / "repo", tmp_path / "run", {"pdk": "ng45"},
                               "/usr/bin/python3", {
                                   "OPENROAD_EXE": "/usr/local/bin/openroad",
                                   "OPEN3DBENCH_HOTSPOT_ROOT": "/opt/open3dbench-assets/hotspot"})
    assert command[:3] == ["/usr/bin/python3", "-m", "open3dbench.backend.thermal_snapshot"]
    args = thermal_snapshot.build_parser().parse_args(command[3:])
    bound = inspect.signature(thermal_snapshot.run_stage_snapshot).bind(
        args.checkpoint, args.platform_dir, args.hotspot_root, args.output_dir,
        openroad=args.openroad, grid_size=args.grid_size, ambient_k=args.ambient_k,
        threads=args.threads)
    assert Path(bound.arguments["checkpoint_path"]) == tmp_path / "run/70_cts/checkpoint.json"
    assert Path(bound.arguments["output_dir"]) == tmp_path / "run/95_thermal_snapshot"
    assert (args.grid_size, args.ambient_k, args.threads) == (10, 318.15, 32)


@pytest.mark.parametrize("fault", ["one_pass", "cts", "activity", "warnings", "tiles", "method", "checkpoint"])
def test_reject_bad_cts_thermal_evidence(tmp_path, fault):
    cts_run(tmp_path, passes=1 if fault == "one_pass" else 2)
    path = tmp_path / "95_thermal_snapshot/metrics.json"
    data = json.loads(path.read_text())
    if fault == "cts":
        write(tmp_path / "70_cts/checkpoint.json", {"stage": "opt"})
    elif fault == "activity":
        data["power"]["activity_model"] = "other"
    elif fault == "warnings":
        data["power_extraction"]["top"]["warning_count"] = 1
    elif fault == "tiles":
        data["temperature"]["bottom"]["tile_count"] = 99
    elif fault == "method":
        write(tmp_path / "20_partition/partition_manifest.json", {"method": "mtkahypar"})
    elif fault == "checkpoint":
        write(tmp_path / "95_thermal_snapshot/snapshot_manifest.json", {
            "status": "success", "source_stage": "cts", "source_checkpoint": "/another/run"})
    write(path, data)
    with pytest.raises(ValueError):
        score(tmp_path, None)


def frozen_input(root):
    for stage, directory in (("input", "00_input"), ("floorplan", "10_floorplan")):
        folder = root / directory
        for filename in ("design.def", "design.v", "design.sdc"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_text("frozen " + filename)
        write(folder / "design_params.json", {
            "def_input": str(folder / "design.def"),
            "verilog_input": str(folder / "design.v"),
            "sdc_input": str(folder / "design.sdc"), "num_threads": 32})
        artifacts = {p.name: {"path": p.name, "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in folder.iterdir()}
        write(folder / "stage_manifest.json", {
            "schema_version": 1, "stage": stage, "status": "success", "signature": {},
            "design": "bp_fe", "pdk": "ng45", "integration": "lol", "artifacts": artifacts})


def replay(root, source, average=322.):
    cts_run(root, average)
    status = json.loads((root / "run.json").read_text())
    status.update(design="bp_fe", pdk="ng45", type="lol")
    del status["stages"]["input"]
    del status["stages"]["floorplan"]
    write(root / "run.json", status)
    copy_upstream_stages(source, root, ["input", "floorplan"])


def test_frozen_prefix_replay_candidate_and_reference(tmp_path):
    source, c, r = (tmp_path / name for name in ("input", "c", "r"))
    frozen_input(source)
    replay(c, source, 321.); replay(r, source, 322.)
    before = (c / "run.json").read_bytes()
    assert extract_metrics("tp_gnn_placement_cts_thermal", c, r,
                           frozen_input_run=source)["combined_score"] == pytest.approx(322. / 321.)
    assert score(r, None, frozen_input_run=source)["combined_score"] == 1.
    assert (c / "run.json").read_bytes() == before  # never fabricate executed prefix stages
    with pytest.raises(ValueError, match="trusted frozen input"):
        score(c, r)
    with pytest.raises(ValueError, match="itself"):
        score(c, None, frozen_input_run=c)


@pytest.mark.parametrize("fault", ["missing_prefix", "manifest", "artifact", "source_artifact",
    "params", "wrong_path", "case", "missing_cts", "skipped_pdn", "failed_placement", "extra_route",
    "partial_prefix", "reference_tamper", "escape", "one_pass", "power_warning"])
def test_replay_does_not_relax_integrity_or_physical_stages(tmp_path, fault):
    source, c, r = (tmp_path / name for name in ("input", "c", "r"))
    frozen_input(source); replay(c, source); replay(r, source)
    prefix = c / "00_input"
    if fault == "missing_prefix":
        (prefix / "stage_manifest.json").unlink()
    elif fault == "manifest":
        write(prefix / "stage_manifest.json", {})
    elif fault in ("artifact", "source_artifact", "reference_tamper"):
        base = {"artifact": c, "source_artifact": source, "reference_tamper": r}[fault]
        (base / "00_input/design.def").write_text("tampered")
    elif fault in ("params", "wrong_path"):
        params = json.loads((prefix / "design_params.json").read_text())
        params["num_threads" if fault == "params" else "def_input"] = 99 if fault == "params" else "/other/design.def"
        write(prefix / "design_params.json", params)
    elif fault == "escape":
        (prefix / "design.def").unlink()
        (prefix / "design.def").symlink_to(source / "00_input/design.def")
    elif fault in ("one_pass", "power_warning"):
        path = c / ("60_opt/metrics.json" if fault == "one_pass" else "95_thermal_snapshot/metrics.json")
        data = json.loads(path.read_text())
        if fault == "one_pass":
            data["passes_run"] = 1
        else:
            data["power_extraction"]["top"]["warning_count"] = 1
        write(path, data)
    else:
        status = json.loads((c / "run.json").read_text())
        if fault == "case":
            status["design"] = "another_case"
        elif fault == "missing_cts":
            del status["stages"]["cts"]
        elif fault == "partial_prefix":
            status["stages"]["input"] = {"status": "success"}
        elif fault == "extra_route":
            status["stages"]["routing"] = {"status": "success"}
        else:
            stage = "pdn" if fault == "skipped_pdn" else "placement"
            status["stages"][stage]["status"] = "skipped" if fault == "skipped_pdn" else "failed"
        write(c / "run.json", status)
    with pytest.raises((ValueError, FileNotFoundError)):
        score(c, r, frozen_input_run=source)
