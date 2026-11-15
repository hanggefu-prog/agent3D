"""Bounded interface driver checks using files only; no Docker/EDA/model I/O."""
import json
from contextlib import contextmanager
import fcntl
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench.model import EvaluationResult
from local_bench import reference_local_placement as local
from local_bench.metrics import placement_metrics, placement_nesterov_metrics
from scripts import run_placement_interface_qualification as runner
from scripts.local_reference import LocalPlacementReference
from test_reference_local_placement import setup as native_environment, native_fixture, write
from test_reference_placement_evidence import HEADER, row


def forbidden(*args, **kwargs):
    raise AssertionError("No model, Docker, physical tool, or registry publication is permitted")


@pytest.fixture
def environment(native_environment, monkeypatch):
    env = native_environment
    monkeypatch.setattr(runner, "validate_benchmark_input", local.validate_benchmark_input)
    monkeypatch.setattr(LocalPlacementReference, "publish", forbidden)
    monkeypatch.setattr(runner.LocalSmokeRuntime, "execute", forbidden)
    env.calls, env.fault = [], None

    class FixtureTransport:
        def __init__(self, runtime):
            self.attempts, self.receipts = [], []

        def __call__(self, task_dir, candidate_dir, source, output, **kwargs):
            role = "identity" if kwargs.get("reference_run") is None else "nontrivial"
            env.calls.append((role, candidate_dir, output, kwargs))
            attempt = {"output_dir": str(output), "evaluator_dispatch_attempted": True, "native_flow_started": True}
            self.attempts.append(attempt)
            if env.fault == role + "_uncertain":
                attempt["native_flow_started"] = None
                raise RuntimeError("Uncertain fixture attempt must never be retried")
            materialized = json.loads((candidate_dir / "candidate.json").read_text())
            fixture = native_fixture(env, local.load_task(task_dir).task_id, output)
            result_path = output / "evaluation_result.json"
            result = json.loads(result_path.read_text())
            if role == "nontrivial":
                # Retain the real driver-materialized probe, not fixture identity.
                write(candidate_dir / "candidate.json", materialized)
                overlay = output / "runtime/dreamplace-install"
                (overlay / "dreamplace").mkdir(parents=True, exist_ok=True)
                patched = local._expected_overlay(source, materialized)
                for name in local.RUNTIME_MODULES:
                    relative = str(local.DREAMPLACE / name)
                    (overlay / "dreamplace" / name).write_text(patched.get(relative, (source / relative).read_text()))
                stage = fixture.run / local.STAGE
                manifest_path = stage / "stage_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.update(dreamplace_root=str(overlay / "dreamplace"),
                    command=["/usr/bin/python3", str(overlay / "dreamplace/Placer.py"), str(stage / "params.json")])
                write(manifest_path, manifest)
                result["artifacts"]["dreamplace_runtime"] = str(overlay)
                if fixture.spec.task_id == local.DYNAMIC_TASK:
                    (stage / "run.log").write_text(HEADER + row(shared="1.05", bottom="1.02", top="0.980392156863") +
                        "[INFO] iteration 1, wHPWL 1000, Overflow [0.1, 0.2], MaxDensity [0.6, 0.7]\n")
                scorer = placement_nesterov_metrics if fixture.spec.task_id == local.DYNAMIC_TASK else placement_metrics
                result["metrics"] = {**scorer(fixture.run, kwargs["reference_run"]), "runs_successfully": 1.0}
            if env.fault == role + "_evaluation":
                result["metrics"]["runs_successfully"] = 0
            if env.fault == role + "_evidence":
                (fixture.run / local.STAGE / "run.log").write_text("corrupt fixture evidence\n")
            if env.fault == "nontrivial_overlay" and role == "nontrivial":
                (output / "runtime/dreamplace-install/dreamplace/BasicPlace.py").write_text("unexpected runtime\n")
            if env.fault == "nontrivial_reference" and role == "nontrivial":
                (kwargs["reference_run"] / local.STAGE / "run.log").write_text("changed authority\n")
            write(result_path, result)
            self.receipts.append({"path": str(result_path)})
            return EvaluationResult(**result)
    env.transport = FixtureTransport
    return env


@pytest.mark.parametrize("task", [local.DYNAMIC_TASK])
def test_two_prescribed_attempts_keep_accounting_and_never_publish(environment, task):
    env = environment
    output = env.batch / "qualification"
    result = runner.run_qualification(env.runtime, task=task, benchmark_input=env.bundle,
        output=output, transport_factory=env.transport)
    assert result["state"] == "completed", result["error"]
    assert [call[0] for call in env.calls] == ["identity", "nontrivial"]
    assert result["evaluator_calls_requested"] == result["evaluator_dispatches"] == result["native_physical_attempts"] == 2
    assert result["identity_attempts_requested"] == result["nontrivial_attempts_requested"] == 1
    assert result["identity_successful_evaluations"] == result["nontrivial_successful_evaluations"] == 1
    assert result["model_calls"] == result["generated_model_candidates"] == result["candidate_finals"] == 0
    assert result["reference_store_records_published"] == 0 and not (env.batch / "reference-store").exists()
    assert not result["historical_equivalence_qualified"] and not result["improvement_qualified"]
    assert result["nonidentity_control_observed"] is (True if task == local.DYNAMIC_TASK else None)
    assert result["identity_raw_metrics"]["placement_native_hpwl"] == 1000  # Cached fixture score 999 was ignored.
    config = json.loads((output / "run_config.json").read_text())
    assert config["timeout_seconds"] == 21600 and config["physical_threads"] == 4
    assert config["admission"] == {"ram_available_gib": 64, "data_free_gib": 64,
        "root_free_gib": 10, "gpu_free_mib": 2048, "locks": str(env.batch / "locks")}
    assert (output / ".qualification.lock").is_file()
    assert all(call[3]["timeout_seconds"] == 21600 for call in env.calls)
    assert env.calls[1][3]["reference_run"] == output / "reference/result/run"
    assert env.calls[0][3]["baseline_run"] == env.calls[1][3]["baseline_run"] == env.baseline
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="no retries"):
        runner.run_qualification(env.runtime, task=task, benchmark_input=env.bundle,
            output=output, transport_factory=env.transport)
    assert len(env.calls) == 2 and (output / "summary.json").read_bytes() == before


@pytest.mark.parametrize("fault,attempts", [
    ("identity_evaluation", 1), ("identity_evidence", 1), ("identity_uncertain", 1),
    ("nontrivial_evaluation", 2), ("nontrivial_evidence", 2), ("nontrivial_uncertain", 2),
    ("nontrivial_overlay", 2), ("nontrivial_reference", 2),
])
def test_failure_stops_and_retains_attempt_without_retry(environment, fault, attempts):
    env = environment
    env.fault = fault
    output = env.batch / "failure"
    result = runner.run_qualification(env.runtime, task=local.DYNAMIC_TASK, benchmark_input=env.bundle,
        output=output, transport_factory=env.transport)
    assert result["state"] == "failed" and result["error"]
    assert len(env.calls) == result["evaluator_calls_requested"] == attempts
    assert result["candidate_finals"] == result["model_calls"] == 0
    assert not result["interface_probe_qualified"]
    if "uncertain" in fault:
        assert result["native_physical_attempts"] is None
        assert result["native_physical_attempts_unknown"] == 1
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        runner.run_qualification(env.runtime, task=local.DYNAMIC_TASK, benchmark_input=env.bundle,
            output=output, transport_factory=env.transport)
    assert len(env.calls) == attempts


@pytest.mark.parametrize("fault", ["existing", "input_overlap", "outside", "symlink", "resources"])
def test_unsafe_paths_or_caps_stop_before_dispatch(environment, tmp_path, fault):
    env = environment
    output = env.batch / "new"
    if fault == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("retain me")
    elif fault == "input_overlap":
        output = env.bundle / "new-output"
    elif fault == "outside":
        output = tmp_path / "outside"
    elif fault == "symlink":
        alias = env.batch / "alias"
        alias.symlink_to(env.bundle, target_is_directory=True)
        output = alias / "new-output"
    else:
        env.runtime.cpus = 4
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_qualification(env.runtime, task=local.DYNAMIC_TASK, benchmark_input=env.bundle,
            output=output, transport_factory=env.transport)
    assert not env.calls
    if fault == "existing":
        assert (output / "keep.txt").read_text() == "retain me"


def test_cli_has_no_search_budget_or_timeout_override(monkeypatch, tmp_path):
    args = ["--source", str(tmp_path / "source"), "--batch-root", str(tmp_path / "batch"),
        "--benchmark-input", str(tmp_path / "batch/input"), "--output", str(tmp_path / "batch/out"),
        "--task", local.DYNAMIC_TASK, "--image", "sha256:" + "0" * 64, "--gpu", "1", "--share-gpu"]
    parsed = runner.build_parser().parse_args(args)
    assert parsed.cpus == 16 and parsed.memory_gb == 48 and parsed.share_gpu
    for extra in (["--rounds", "1"], ["--timeout", "1"], ["--cpus", "4"], ["--min-gpu-free-mb", "1"]):
        with pytest.raises(SystemExit):
            runner.build_parser().parse_args(args + extra)
    monkeypatch.setattr(runner, "QualificationRuntime", lambda *a: SimpleNamespace())
    monkeypatch.setattr(runner, "run_qualification", lambda *a, **k: {"state": "completed"})
    assert runner.main(args) == 0
    monkeypatch.setattr(runner, "run_qualification", lambda *a, **k: {"state": "failed"})
    assert runner.main(args) == 1


def test_stronger_disk_guard_releases_shared_gpu_lock_before_wait(monkeypatch, tmp_path):
    state = {"locked": False, "calls": 0, "waits": 0}
    @contextmanager
    def parent_slot(self):
        assert not state["locked"]
        state["locked"] = True
        state["calls"] += 1
        try:
            yield 2
        finally:
            state["locked"] = False
    def disk_usage(path):
        assert state["locked"] and path == tmp_path
        return SimpleNamespace(free=runner.MIN_DATA_FREE_BYTES - (1 if state["calls"] == 1 else 0))
    def wait(seconds):
        assert seconds == 30 and not state["locked"]
        state["waits"] += 1
    monkeypatch.setattr(runner.LocalSmokeRuntime, "slot", parent_slot)
    monkeypatch.setattr(runner.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(runner.time, "sleep", wait)
    runtime = object.__new__(runner.QualificationRuntime)
    runtime.batch = tmp_path
    with runtime.slot() as gpu:
        assert gpu == 2 and state["locked"]
    assert state == {"locked": False, "calls": 2, "waits": 1}


def test_output_lock_is_held_during_dispatch(environment):
    env = environment
    output = env.batch / "locked-output"
    factory = env.transport
    class LockedTransport(factory):
        def __call__(self, *args, **kwargs):
            with (output / ".qualification.lock").open() as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return super().__call__(*args, **kwargs)
    result = runner.run_qualification(env.runtime, task=local.DYNAMIC_TASK, benchmark_input=env.bundle,
        output=output, transport_factory=LockedTransport)
    assert result["state"] == "completed", result["error"]
