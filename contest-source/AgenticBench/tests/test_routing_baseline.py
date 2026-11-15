import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from local_bench.benchmark_input import load_benchmark_input_profile
from local_bench.model import load_task
from local_bench.routing_baseline import (
    FRAMEWORK, TASKS, main, routing_profile, validate_initial_replay, validate_routing_baseline,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, stage):
    return {"path": str(path.relative_to(stage)), "bytes": path.stat().st_size, "sha256": sha(path)}


@pytest.fixture
def baseline(tmp_path):
    profile = routing_profile("bp_fe")
    bundle = tmp_path / "input/benchmark_input.json"
    run = bundle.parent / "run"
    stages = {}
    for stage, directory in profile.required_stages.items():
        target = run / directory
        files = {"payload": ("payload.txt", stage)}
        method = "openroad"
        if stage in ("input", "floorplan"):
            files.update(params=("design_params.json", {"def_input": "/old/design.def", "other": 123}),
                         design=("design.def", "frozen DEF"))
        if stage == "partition":
            method = "tp-gnn"
            files.update(method_manifest=("partition_manifest.json", {
                "method": "tp-gnn", "method_report": {"method": "tp-gnn", "config": {
                    "decoder": "global-2way", "timing_driven": True,
                    "feature_regime": "partition-last", "timing_mode": "partition-last"}}}),
                timing_manifest=("timing_input/timing_manifest.json", {"timing_status": "full_coverage_partition_last"}))
        if stage == "placement":
            method = "concurrent-dreamplace"
        if stage == "cts":
            files["checkpoint"] = ("checkpoint.json", {"stage": "cts", "ready_for_route": True,
                                                          "dies": {"bottom": {}, "top": {}}})
        artifacts = {}
        for key, (filename, value) in files.items():
            write(target / filename, value)
            artifacts[key] = record(target / filename, target)
        manifest = target / "stage_manifest.json"
        write(manifest, {"schema_version": 1, "stage": stage, "status": "success", "method": method,
                         "design": "bp_fe", "integration": "lol", "pdk": "ng45", "artifacts": artifacts})
        stages[stage] = {"directory": directory, "manifest_sha256": sha(manifest)}
    write(bundle, {"schema_version": 1, "benchmark_id": profile.profile_id, "run_dir": "run",
                   "profile_sha256": sha(profile.root / "profile.yaml"), "config_sha256": sha(profile.config), "stages": stages})
    reference = tmp_path / "reference/run"
    shutil.copytree(run, reference)
    row = {"status": "success", "grt_wirelength_um": 100, "grt_total_overflow": 5,
           "grt_congestion_bins": 1, "ndr_policy": "preserve", "layer_window": ["metal2", "metal10"]}
    write(reference / "80_route/metrics.json", {"status": "success", "stage": "route-grt",
        "algorithm": "openroad-die-local-net-order", "dies": {"bottom": row, "top": copy.deepcopy(row)},
        "configuration": {"grt_only": True, "congestion_iterations": 4, "resource_reduction": 0.0, "threads": 32},
        "ready_for_extract": False, "unexpected_warnings": 0})
    write(reference / "80_route/bottom/route.log", "OPEN3D_NET_ORDER die=bottom phase=0 nets=10 overflow_nets=2 "
          "changed_ratio=0 overflow_changed_ratio=0 top10_overlap=1 score_min=0 score_max=1 baseline_hash=1 learned_hash=1\n")
    write(reference / "80_route/stage_manifest.json", {"schema_version": 1, "stage": "routing", "status": "success",
        "artifacts": {"metrics": record(reference / "80_route/metrics.json", reference / "80_route")}})
    return bundle, run, reference


def mutate(baseline, stage, change, artifact=None):
    bundle, run, reference = baseline
    meta = json.loads(bundle.read_text())
    directory = routing_profile("bp_fe").required_stages[stage]
    target = run / directory
    manifest = target / "stage_manifest.json"
    data = json.loads(manifest.read_text())
    if artifact:
        path = target / data["artifacts"][artifact]["path"]
        value = json.loads(path.read_text())
        change(value)
        write(path, value)
        data["artifacts"][artifact] = record(path, target)
        shutil.copyfile(path, reference / directory / data["artifacts"][artifact]["path"])
    else:
        change(data)
    write(manifest, data)
    shutil.copyfile(manifest, reference / directory / "stage_manifest.json")
    meta["stages"][stage]["manifest_sha256"] = sha(manifest)
    write(bundle, meta)


@pytest.mark.parametrize("design", ["bp_fe", "bp", "bp_multi", "ariane133", "swerv_wrapper"])
def test_profiles(design):
    profile = routing_profile(design)
    config = yaml.safe_load(profile.config.read_text())
    assert profile.stop_stage == "cts" and len(profile.required_stages) == 8
    assert config["design"] == design
    assert config["partition"]["algorithm"] == "tp-gnn"
    assert config["placement"]["algorithm"] == "concurrent-dreamplace"
    assert config["routing"]["congestion_iterations"] == 4
    assert config["placement"]["target_density"] == 0.6
    assert config["routing"]["seed"] == -1


@pytest.mark.parametrize("task", TASKS)
def test_task_default_is_new_profile(task):
    spec = load_task(FRAMEWORK / "local_tasks" / task)
    assert spec.benchmark_input_id == "ariane133_lol_ng45_tp_gnn_cts"
    assert spec.base_config == routing_profile("ariane133").config


def test_real_artifact_contract_passes_without_external_calls(baseline, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("read-only check invoked subprocess"))
    bundle, run, reference = baseline
    assert validate_routing_baseline(bundle, reference, "bp_fe")["status"] == "passed"
    assert main(["--case", "bp_fe", "--benchmark-input", str(bundle), "--reference-run", str(reference)]) == 0
    assert '"status": "passed"' in capsys.readouterr().out


@pytest.mark.parametrize("stage,method", [("partition", "mtkahypar"), ("placement", "mol-tiling")])
def test_relabelled_profile_does_not_hide_actual_method(baseline, stage, method):
    mutate(baseline, stage, lambda data: data.update(method=method))
    with pytest.raises(ValueError, match="recorded .* method"):
        validate_routing_baseline(baseline[0], baseline[2], "bp_fe")


@pytest.mark.parametrize("key,value", [("decoder", "ta3d-hierarchical"), ("decoder", None),
                                      ("timing_driven", False), ("feature_regime", "physical-only")])
def test_actual_decoder_and_timing_must_match(baseline, key, value):
    mutate(baseline, "partition", lambda data: data["method_report"]["config"].update({key: value}), "method_manifest")
    with pytest.raises(ValueError, match="global-2way"):
        validate_routing_baseline(baseline[0], baseline[2], "bp_fe")


def test_wrong_case_in_stage_rejected(baseline):
    mutate(baseline, "cts", lambda data: data.update(design="bp"))
    with pytest.raises(ValueError, match="another case/platform"):
        validate_routing_baseline(baseline[0], baseline[2], "bp_fe")


def test_cts_not_ready_rejected(baseline):
    mutate(baseline, "cts", lambda data: data.update(ready_for_route=False), "checkpoint")
    with pytest.raises(ValueError, match="not ready"):
        validate_routing_baseline(baseline[0], baseline[2], "bp_fe")


def test_incomplete_timing_rejected(baseline):
    mutate(baseline, "partition", lambda data: data.update(timing_status="partial"), "timing_manifest")
    with pytest.raises(ValueError, match="full-coverage"):
        validate_routing_baseline(baseline[0], baseline[2], "bp_fe")


def test_historical_profile_is_not_relabelled(baseline):
    bundle, run, reference = baseline
    profile = load_benchmark_input_profile("bp_fe_lol_ng45_mtkahypar_cts")
    data = json.loads(bundle.read_text())
    data.update(benchmark_id=profile.profile_id, profile_sha256=sha(profile.root / "profile.yaml"), config_sha256=sha(profile.config))
    write(bundle, data)
    with pytest.raises(ValueError, match="regenerate, do not relabel"):
        validate_routing_baseline(bundle, reference, "bp_fe")


def test_reference_artifact_tampering_rejected_even_with_identical_manifests(baseline):
    bundle, run, reference = baseline
    write(reference / "70_cts/payload.txt", "tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_routing_baseline(bundle, reference, "bp_fe")


def test_reference_route_metrics_are_hash_checked(baseline):
    bundle, run, reference = baseline
    path = reference / "80_route/metrics.json"
    data = json.loads(path.read_text())
    data["dies"]["bottom"]["grt_wirelength_um"] = 999
    write(path, data)
    with pytest.raises(ValueError, match="artifact (size|hash) changed"):
        validate_routing_baseline(bundle, reference, "bp_fe")


def test_only_expected_path_rebasing_is_allowed(baseline):
    bundle, run, reference = baseline
    path = reference / "00_input/design_params.json"
    write(path, {"def_input": str((reference / "00_input/design.def").resolve()), "other": 123})
    validate_routing_baseline(bundle, reference, "bp_fe")
    write(path, {"def_input": str((reference / "00_input/design.def").resolve()), "other": 456})
    with pytest.raises(ValueError, match="changed frozen design parameters"):
        validate_routing_baseline(bundle, reference, "bp_fe")


@pytest.mark.parametrize("field,value", [("congestion_iterations", 2), ("threads", 8)])
def test_routing_budget_mismatch_rejected(baseline, field, value):
    bundle, run, reference = baseline
    path = reference / "80_route/metrics.json"
    data = json.loads(path.read_text())
    data["configuration"][field] = value
    write(path, data)
    with pytest.raises(ValueError, match="budget mismatch"):
        validate_routing_baseline(bundle, reference, "bp_fe")


@pytest.mark.parametrize("field,value", [("status", "failed"), ("grt_wirelength_um", float("nan")), ("ndr_policy", "strip")])
def test_invalid_die_metrics_rejected(baseline, field, value):
    bundle, run, reference = baseline
    path = reference / "80_route/metrics.json"
    data = json.loads(path.read_text())
    data["dies"]["top"][field] = value
    write(path, data)
    with pytest.raises(ValueError):
        validate_routing_baseline(bundle, reference, "bp_fe")


def test_changed_native_policy_rejected(baseline):
    bundle, run, reference = baseline
    path = reference / "80_route/bottom/route.log"
    path.write_text(path.read_text().replace("learned_hash=1", "learned_hash=2"))
    with pytest.raises(ValueError, match="behavior-preserving"):
        validate_routing_baseline(bundle, reference, "bp_fe")


@pytest.mark.parametrize("field", ["runs_successfully", "routing_complete", "combined_score", "grt_total_overflow", "grt_wirelength_um"])
def test_initial_replay_requires_exact_reference(baseline, field, tmp_path):
    result = tmp_path / "identity.json"
    metrics = {"runs_successfully": 1, "routing_complete": 1, "combined_score": 1,
               "grt_total_overflow": 10, "grt_wirelength_um": 200}
    write(result, {"metrics": metrics})
    validate_initial_replay(result, baseline[2])
    metrics[field] = 0
    write(result, {"metrics": metrics})
    with pytest.raises(ValueError, match=field):
        validate_initial_replay(result, baseline[2])


def test_launcher_check_only_does_not_mount_credentials_or_call_model(baseline, tmp_path):
    bundle, run, reference = baseline
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    docker = fakebin / "docker"
    docker.write_text("#!" + sys.executable + "\nimport json,sys\nprint('sha256:test' if sys.argv[1]=='image' else json.dumps(sys.argv[1:]))\n")
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
               OPENAGENTIC3D_FRAMEWORK=str(FRAMEWORK), OPEN3DBENCH_SOURCE=str(FRAMEWORK.parent),
               OPENAGENTIC3D_BENCHMARK_INPUT=str(bundle), OPEN3DBENCH_REFERENCE_RUN=str(reference))
    script = FRAMEWORK / "scripts/run_routing_heuristic_evolution_case.sh"
    proc = subprocess.run(["bash", str(script), str(tmp_path), "bp_fe", "ripup", "--check-only"],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    args = json.loads(proc.stdout)
    assert args[args.index("--network") + 1] == "none"
    assert not any(".secrets" in arg for arg in args)
    inner = args[args.index("-c") + 1]
    assert inner.index("local_bench.routing_baseline") < inner.index("source \"$secret\"") < inner.index("local_bench.evolve_cli")
    assert inner.index("--identity-result") < inner.index("source \"$secret\"")
    assert "--check-only ]] || exit 0" in inner
    assert not (tmp_path / "evolution").exists()


def test_launcher_syntax():
    subprocess.run(["bash", "-n", str(FRAMEWORK / "scripts/run_routing_heuristic_evolution_case.sh")], check=True)
