"""Synthetic control-plane tests; no model, Docker, compiler, or EDA calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qualify_archive_grt as q
from local_bench import metrics
from open3dbench.backend import three_d_route


def settings():
    return ({"type": "lol", "pdk": "ng45", "routing": {"threads": 4,
        "congestion_iterations": 4, "resource_reduction": 0.0, "max_iterations": 5,
        "die_timeout_seconds": 10800, "seed": -1}},
        {"threads": 4, "congestion_iterations": 4, "resource_reduction": 0.0,
         "ndr_policy": {"bottom": "preserve", "top": "preserve"}})


def test_command_is_only_native_global_route_at_exact_frozen_budget(tmp_path):
    command = q.command_for(tmp_path / "source", tmp_path / "run", "/installed/openroad", *settings())
    assert command[1:3] == ["-m", "open3dbench.backend.three_d_route"]
    assert command[command.index("--mode") + 1] == "global"
    assert command[command.index("--threads") + 1] == "4"
    assert command[command.index("--congestion-iterations") + 1] == "4"
    assert command[command.index("--die-timeout-seconds") + 1] == "10800"
    assert command[command.index("--checkpoint") + 1].endswith("70_cts/checkpoint.json")
    assert command[command.index("--output-dir") + 1].endswith("75_global_route")
    assert not {"--resume", "--from", "--to", "run", "partition", "placement", "opt"}.intersection(command)
    assert q.TIMEOUT_SECONDS == 21600


@pytest.mark.parametrize("field,value", [("threads", 32), ("congestion_iterations", 2),
    ("resource_reduction", .1), ("die_timeout_seconds", 21600), ("seed", 1)])
def test_changed_physical_recipe_is_rejected(tmp_path, field, value):
    config, grt = settings()
    config["routing"][field] = value
    with pytest.raises(ValueError, match="physical contract"):
        q.command_for(tmp_path, tmp_path, "/installed/openroad", config, grt)


def test_copier_keeps_authenticated_manifests_and_does_not_invent_stages(tmp_path, monkeypatch):
    monkeypatch.delenv("OPEN3D_LINK_FROZEN_UPSTREAM", raising=False)
    source = tmp_path / "archive" / "run"
    rows = []
    for stage, directory in q.STAGES.items():
        path = source / directory / "stage_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"stage": stage, "historical": True}))
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": q.file_sha(path)})
    payload = {"baseline": {"source_run": str(source)}, "evidence": rows}
    target = tmp_path / "copied"
    assert len(q.copy_prefix(payload, target)) == 8
    assert not (target / "15_timing_extract").exists()
    assert not (target / "75_global_route").exists()
    for row in rows:
        assert q.file_sha(target / Path(row["path"]).relative_to(source)) == row["sha256"]
    q._verify_evidence(rows)


def test_shared_hardlink_prefix_mode_is_not_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN3D_LINK_FROZEN_UPSTREAM", "1")
    with pytest.raises(ValueError, match="independent copies"):
        q.copy_prefix({}, tmp_path)


def test_sealed_readonly_prefix_is_copied_and_rebased_without_source_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("OPEN3D_LINK_FROZEN_UPSTREAM", raising=False)
    original = tmp_path / "sealed" / "run"
    for stage, directory in q.STAGES.items():
        folder = original / directory
        folder.mkdir(parents=True)
        (folder / "stage_manifest.json").write_text(json.dumps({"stage": stage}))
    input_dir = original / "00_input"
    (input_dir / "design.def").write_text("sealed DEF")
    (input_dir / "design_params.json").write_text(json.dumps({"def_input": "/old/run/00_input/design.def"}))
    files = [p for p in original.rglob("*") if p.is_file()]
    for p in files:
        p.chmod(0o444)
    for p in (p for p in original.rglob("*") if p.is_dir()):
        p.chmod(0o555)
    original.chmod(0o555)
    rows = [{"path": str(p), "bytes": p.stat().st_size, "sha256": q.file_sha(p)} for p in files]
    modes = {str(p): p.stat().st_mode for p in [original, *original.rglob("*")]}
    target = tmp_path / "new" / "run"
    try:
        q.copy_prefix({"baseline": {"source_run": str(original)}, "evidence": rows}, target)
        copied = json.loads((target / "00_input/design_params.json").read_text())
        assert copied["def_input"] == str(target / "00_input/design.def")
        q._verify_evidence(rows)
        assert modes == {str(p): p.stat().st_mode for p in [original, *original.rglob("*")]}
    finally:
        # Restore only this synthetic test fixture so pytest can remove it.
        original.chmod(0o755)
        for p in original.rglob("*"):
            if p.is_dir():
                p.chmod(0o755)
        if target.exists():
            for p in target.rglob("*"):
                if p.is_dir():
                    p.chmod(0o755)


def test_existing_attempt_and_external_source_destinations_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        q._fresh(source / "new", source_root=str(source))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    with pytest.raises(FileExistsError, match="cannot be retried"):
        q._fresh(attempt, source_root=str(source))


@pytest.fixture
def measurement(tmp_path, monkeypatch):
    run = tmp_path / "run"
    projected = {}
    for die in ("bottom", "top"):
        projected[die] = {}
        directory = run / "75_global_route" / die
        directory.mkdir(parents=True)
        for name in ("design.sdc", "design.sdc.ports.tsv"):
            path = directory / name
            path.write_text("frozen " + die + " " + name)
            projected[die][name] = q.file_sha(path)
    called = []
    monkeypatch.setattr(three_d_route, "_validate_global_route_publication", lambda path: called.append(path))
    observed = {"grt_total_overflow": 0., "grt_wirelength_um": 1604911.,
        "net_order_diagnostics_present": 1., "net_order_calls": 2., "net_order_changed_calls": 0.,
        "combined_score": .999, "overflow_score": 1., "wirelength_score": .997}
    monkeypatch.setattr(metrics, "routing_net_ordering_metrics", lambda *args: observed)
    baseline = {"grt_total_overflow": 0., "grt_wire_length_um": 1604911.}
    return run, projected, observed, baseline, called


def test_identity_requires_reproduction_and_suppresses_bootstrap_score(measurement):
    run, projected, observed, baseline, called = measurement
    report = q.validate_measurement(run, baseline, "identity", projected)
    assert "combined_score" not in report and "overflow_score" not in report
    assert report["grt_wirelength_um"] == baseline["grt_wire_length_um"]
    assert called == [run / "75_global_route/checkpoint.json"]
    observed["grt_wirelength_um"] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        q.validate_measurement(run, baseline, "identity", projected)


def test_reversed_policy_requires_reachable_changed_native_order(measurement):
    run, projected, observed, baseline, _ = measurement
    with pytest.raises(ValueError, match="did not change"):
        q.validate_measurement(run, baseline, "reversed", projected)
    observed["net_order_changed_calls"] = 1.
    observed["grt_wirelength_um"] += 10
    assert q.validate_measurement(run, baseline, "reversed", projected)["net_order_changed_calls"] == 1.
    observed["net_order_diagnostics_present"] = 0.
    with pytest.raises(ValueError, match="not reached"):
        q.validate_measurement(run, baseline, "reversed", projected)


def test_changed_projected_sdc_is_not_reference_compatible(measurement):
    run, projected, _, baseline, _ = measurement
    (run / "75_global_route/bottom/design.sdc").write_text("changed")
    with pytest.raises(ValueError, match="SDC/ports differ"):
        q.validate_measurement(run, baseline, "identity", projected)


@pytest.mark.parametrize("task", [key for key in q.TASKS if key != q.TASK])
def test_remaining_tasks_require_identity_equivalence_and_changed_probe(measurement, monkeypatch, task):
    run, projected, observed, baseline, _ = measurement
    observed.update(policy_diagnostics_present=1., policy_calls=3., policy_changed=0.)
    monkeypatch.setattr(metrics, q.TASKS[task]["metric"], lambda *_: observed)
    assert q.validate_measurement(run, baseline, "identity", projected, task)["policy_calls"] == 3.
    with pytest.raises(ValueError, match="did not change"):
        q.validate_measurement(run, baseline, "probe", projected, task)
    observed["policy_changed"] = 1.
    assert q.validate_measurement(run, baseline, "probe", projected, task)["policy_changed"] == 1.
    with pytest.raises(ValueError, match="Identity hook changed"):
        q.validate_measurement(run, baseline, "identity", projected, task)
    observed["policy_calls"] = 0.
    with pytest.raises(ValueError, match="not reached"):
        q.validate_measurement(run, baseline, "probe", projected, task)


@pytest.mark.parametrize("task", [key for key in q.TASKS if key != q.TASK])
def test_new_programs_are_real_verified_native_only_patches(task):
    source = Path(__file__).resolve().parents[2]
    if not (source / q.TASKS[task]["source"]).is_file():
        pytest.skip("The pinned native submodule must be populated for patch integration checks")
    identity = q.expected_candidate(source, "identity", task)
    probe = q.expected_candidate(source, "probe", task)
    assert identity != probe
    for candidate in (identity, probe):
        assert candidate["config_overrides"] == {}
        assert candidate["candidate_files"] == {}
        assert q.TASKS[task]["source"] in candidate["source_patch"]
        assert "open3dbench/backend/three_d_route.py" not in candidate["source_patch"]
        assert "open3dbench/runner.py" not in candidate["source_patch"]


def test_task_and_policy_selection_fail_closed():
    with pytest.raises(ValueError, match="Unsupported qualification task"):
        q.task_contract("global_routing_parameters")
    with pytest.raises(ValueError, match="Unsupported qualification policy"):
        q.task_contract(q.TASK, "probe")
    for task in q.TASKS:
        if task != q.TASK:
            with pytest.raises(ValueError, match="Unsupported qualification policy"):
                q.task_contract(task, "reversed")


def iteration_logs(root, *, height=9, scale=1, iteration=1, search=20, reached=True):
    for die in ("bottom", "top"):
        path = root / "75_global_route" / die / "route.log"
        path.parent.mkdir(parents=True)
        # Top may have zero initial overflow and never reach this hook.
        path.write_text("OPEN3D_ITERATION_CONTROL die=bottom calls={} changed={} "
                        "cost_scale={} cost_height={} search_window={} ripup_threshold=-1\n".format(
                            iteration, int(scale != 1), scale, height, search)
                        if reached and die == "bottom" else "no extra congestion iterations\n")
    return root


def test_iteration_effect_compares_actual_first_integer_control_and_hashes(tmp_path):
    identity = iteration_logs(tmp_path / "identity")
    probe = iteration_logs(tmp_path / "probe", height=18, scale=2)
    proof = q.validate_iteration_probe_effect(identity, probe)
    assert proof["status"] == "passed"
    assert proof["comparisons"] == {"bottom": {"iteration": 1, "identity_cost_height": 9,
                                               "probe_cost_height": 18}}
    assert len(proof["identity_logs"]) == len(proof["probe_logs"]) == 2
    q._verify_evidence(proof["identity_logs"] + proof["probe_logs"])


@pytest.mark.parametrize("override", [
    {"height": 9}, {"height": 19}, {"height": float("nan")},
    {"scale": 1.1}, {"iteration": 2}, {"search": 21}, {"reached": False},
])
def test_iteration_effect_rejects_noops_wrong_phase_and_unmatched_controls(tmp_path, override):
    identity = iteration_logs(tmp_path / "identity")
    args = {"height": 18, "scale": 2, **override}
    probe = iteration_logs(tmp_path / "probe", **args)
    with pytest.raises(ValueError):
        q.validate_iteration_probe_effect(identity, probe)


def test_adapter_rejects_wrapper_patch(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "load_task", lambda _: SimpleNamespace(task_id=q.TASK, kind="routing_net_ordering_source"))
    monkeypatch.setattr(q, "materialize_program", lambda *args: {})
    monkeypatch.setattr(q, "verify_candidate", lambda *args: {"patch": {"paths": [q.UTILITY, "open3dbench/runner.py"]}})
    with pytest.raises(ValueError, match="wrapper patches"):
        q.expected_candidate(tmp_path, "identity")


def test_pre_native_validation_failure_is_retained_without_retry(tmp_path, monkeypatch):
    source, archive, candidate = (tmp_path / name for name in ("source", "archive", "candidate"))
    for path in (source, archive, candidate):
        path.mkdir()
    monkeypatch.setattr(q, "load_candidate", lambda _: {"schema_version": 1})
    monkeypatch.setattr(q, "expected_candidate", lambda *args: {"schema_version": 1})
    monkeypatch.delenv(q.PREBUILT_OPENROAD_ENV, raising=False)
    calls = []
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    output = tmp_path / "attempt"
    result = q.qualify(source, archive, "bp_fe", "identity", candidate, output)
    assert result["status"] == "failed"
    assert result["candidate_finals"] == result["native_physical_attempts"] == 0
    assert "separately built" in result["error"]
    assert (output / "qualification-result.json").is_file()
    with pytest.raises(FileExistsError):
        q.qualify(source, archive, "bp_fe", "identity", candidate, output)
    assert calls == []


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "launch"])
def test_actual_qualification_control_dispatches_native_only_once(tmp_path, monkeypatch, failure):
    """Archive admission and native process are mocked, not the orchestration."""
    source, archive, candidate = (tmp_path / name for name in ("source", "archive", "candidate"))
    for path in (source, archive, candidate):
        path.mkdir()
    original = archive / "physical" / "run"
    for directory in q.STAGES.values():
        path = original / directory / "stage_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    config, grt = settings()
    (original / "config.yaml").write_text(json.dumps(config))
    package = "patch"
    for root in (source, archive / package):
        (root / "open3dbench/backend").mkdir(parents=True)
        (root / "open3dbench/backend/sdc_handoff.py").write_text("same projection")
        (root / "open3dbench/backend/three_d_route.py").write_text("wrapper")
    old = archive / "old-grt"
    old.mkdir()
    (old / "checkpoint.json").write_text("{}")
    for die in ("bottom", "top"):
        (old / die).mkdir()
        for name in ("design.sdc", "design.sdc.ports.tsv"):
            (old / die / name).write_text(name)
    index = {"sdc_patch": {"path": package},
             "cases": {"bp_fe": {"qualified_grt_checkpoint": "old-grt/checkpoint.json"}}}
    payload = {"baseline": {"source_run": str(original)}, "evidence": [], "source_identity": {"source_tree": str(source)},
               "source_contract": {"grt_configuration": grt}}
    candidate_value = {"source_patch": "controlled hook"}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"task": q.TASK, "candidate_sha256": q._digest(candidate_value)}))
    monkeypatch.setenv(q.PREBUILT_OPENROAD_ENV, str(receipt))
    monkeypatch.delenv("OPEN3D_LINK_FROZEN_UPSTREAM", raising=False)
    monkeypatch.setattr(q, "load_candidate", lambda _: candidate_value)
    monkeypatch.setattr(q, "expected_candidate", lambda *args: candidate_value)
    monkeypatch.setattr(q, "prepare_archive_reference", lambda *args: payload)
    monkeypatch.setattr(q, "qualify_platform", lambda *args: {})
    monkeypatch.setattr(q, "_Archive", lambda root: SimpleNamespace(evidence={},
        read=lambda _: index, file=lambda relative: root / relative))
    monkeypatch.setattr(q, "isolated_clone", lambda src, dst, **kwargs: dst.mkdir())
    monkeypatch.setattr(q, "apply_candidate_patch", lambda *args: None)
    monkeypatch.setattr(q, "prepare_openroad_runtime", lambda *args: Path("/installed/openroad"))
    calls = []
    def native(command, **kwargs):
        calls.append((command, kwargs))
        if failure == "timeout":
            raise q.subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "launch":
            raise FileNotFoundError("Native executable launch failed")
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(q.subprocess, "run", native)
    output = tmp_path / "attempt"
    report = q.qualify(source, archive, "bp_fe", "identity", candidate, output)
    assert report["status"] == "failed"
    assert report["native_dispatch_attempts"] == len(calls) == 1
    assert report["native_physical_attempts"] == (0 if failure == "launch" else 1)
    assert report["executed_stages"] == ["global_routing"]
    assert calls[0][1]["timeout"] == 21600
    assert report["model_calls"] == report["candidate_finals"] == report["successful_evaluations"] == 0
    assert (output / "native-grt.log").is_file()
    with pytest.raises(FileExistsError):
        q.qualify(source, archive, "bp_fe", "identity", candidate, output)
    assert len(calls) == 1
