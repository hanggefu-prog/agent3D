"""Host qualification lifecycle with synthetic process/build boundaries only."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from scripts import run_archive_grt_qualification as host


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "release"
    source, archive, runs = (root / name for name in ("source", "archive", "runs"))
    for path in (source, archive, runs):
        path.mkdir(parents=True)
    monkeypatch.setattr(host, "FRAMEWORK", source / "AgenticBench")
    order, calls = [], []
    state = {"identity": "passed", "reversed": "passed", "probe": "passed", "build_failure": False}
    @contextmanager
    def slot(root, output):
        order.append("lock-acquired")
        yield str(root / "runs/locks/cpu-grt-qualification.lock")
        order.append("lock-released")
    monkeypatch.setattr(host, "slot", slot)
    monkeypatch.setattr(host, "qualify_inputs", lambda *args: {"native_physical_attempts": 0})
    monkeypatch.setattr(host, "admit", lambda root, output, phase: order.append("admit-" + phase))
    def materialize(source, policy, output, task):
        output.mkdir()
        (output / "candidate.json").write_text("{}")
        return output
    monkeypatch.setattr(host, "materialize", materialize)
    def builder(task, candidate, source, base_image, output, threads, retention, **kwargs):
        policy = output.name[:-len("-build")]
        order.append(policy + "-build")
        assert threads == 8 and retention == "retain"
        assert kwargs["compiler_cache_gb"] == 8
        assert kwargs["compiler_cache_dir"].parent == root / "cache"
        if state["build_failure"]:
            raise RuntimeError("build failed")
        output.mkdir()
        receipt = output / "receipt.json"
        image = "sha256:" + "a" * 64
        receipt.write_text(json.dumps({"image_id": image, "task": task.name + "_source"}))
        return {"image": image, "receipt": str(receipt)}
    monkeypatch.setattr(host, "build_candidate_image", builder)
    def process(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="" if "status" in command else "1" * 40, returncode=0)
        calls.append(command)
        policy = command[command.index("--policy") + 1]
        order.append(policy + "-run")
        destination = host.Path(command[command.index("--output") + 1])
        destination.mkdir()
        result = {"task": command[command.index("--task") + 1], "case": "bp_fe", "policy": policy,
            "status": state[policy], "native_physical_attempts": 1}
        if result["task"] == "global_routing_iteration_controller_source":
            for die in ("bottom", "top"):
                path = destination / "run/75_global_route" / die / "route.log"
                path.parent.mkdir(parents=True)
                scale = 1 if policy == "identity" else 2
                height = 9 if policy == "identity" else state.get("probe_cost_height", 18)
                path.write_text("OPEN3D_ITERATION_CONTROL die={} calls=1 changed={} "
                                "cost_scale={} cost_height={} search_window=20 ripup_threshold=-1\n".format(
                                    die, int(policy != "identity"), scale, height))
        (destination / "qualification-result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0 if state[policy] == "passed" else 1)
    monkeypatch.setattr(host.subprocess, "run", process)
    return root, source, archive, runs / "grt-q01", order, calls, state


def test_pair_is_serial_under_shared_lock_and_cpu_only_bounded_containers(runtime):
    root, source, archive, output, order, calls, _ = runtime
    result = host.run_pair(root, source, archive, "bp_fe", output)
    assert result["state"] == "passed"
    assert result["native_physical_attempts"] == 2
    assert order == ["lock-acquired", "admit-identity-build", "identity-build", "admit-identity-evaluate",
                     "identity-run", "admit-reversed-build", "reversed-build", "admit-reversed-evaluate",
                     "reversed-run", "lock-released"]
    for command in calls:
        assert "--gpus" not in command
        assert command[command.index("--cpus") + 1] == "16"
        assert command[command.index("--memory") + 1] == "48g"
        assert "NVIDIA_VISIBLE_DEVICES=none" in command
        assert str(source) + ":" + str(source) + ":ro" in command
        assert str(archive) + ":" + str(archive) + ":ro" in command
        assert "--network=none" in command
        assert "--read-only" in command
    with pytest.raises(FileExistsError):
        host.run_pair(root, source, archive, "bp_fe", output)
    assert len(calls) == 2


def test_failed_identity_does_not_build_or_dispatch_reversed(runtime):
    root, source, archive, output, order, calls, state = runtime
    state["identity"] = "failed"
    report = host.run_pair(root, source, archive, "bp_fe", output)
    assert report["state"] == "failed"
    assert len(calls) == report["native_physical_attempts"] == 1
    assert not any("reversed" in step for step in order)
    assert report["candidate_finals"] == report["generated_model_candidates"] == report["model_calls"] == 0


@pytest.mark.parametrize("task", [key for key in host.TASKS if key != host.TASK])
def test_remaining_tasks_dispatch_only_selected_identity_and_probe(runtime, task):
    root, source, archive, output, order, calls, _ = runtime
    report = host.run_pair(root, source, archive, "bp_fe", output, task)
    assert report["state"] == "passed"
    assert report["task"] == task
    assert report["native_physical_attempts"] == 2
    assert [row["policy"] for row in report["attempts"]] == ["identity", "probe"]
    assert all(command[command.index("--task") + 1] == task for command in calls)
    assert not any("reversed" in step for step in order)


@pytest.mark.parametrize("task", [key for key in host.TASKS if key != host.TASK])
def test_remaining_task_failed_identity_blocks_probe(runtime, task):
    root, source, archive, output, order, calls, state = runtime
    state["identity"] = "failed"
    report = host.run_pair(root, source, archive, "bp_fe", output, task)
    assert report["state"] == "failed"
    assert len(calls) == report["native_physical_attempts"] == 1
    assert not any("probe" in step for step in order)


def test_iteration_pair_refuses_scale_flag_without_effective_integer_change(runtime):
    root, source, archive, output, _, calls, state = runtime
    state["probe_cost_height"] = 9
    report = host.run_pair(root, source, archive, "bp_fe", output,
                           "global_routing_iteration_controller_source")
    assert report["state"] == "failed"
    assert "did not double effective native cost_height" in report["error"]
    assert report["attempts"][-1]["state"] == "effective-probe-rejected"
    assert len(calls) == report["native_physical_attempts"] == 2
    assert report["qualification_passes"] == report["qualification_rejections"] == 1


def test_failed_build_does_not_dispatch_native_or_retry(runtime):
    root, source, archive, output, order, calls, state = runtime
    state["build_failure"] = True
    report = host.run_pair(root, source, archive, "bp_fe", output)
    assert report["state"] == "failed"
    assert calls == []
    assert report["native_physical_attempts"] == 0
    assert order.count("identity-build") == 1
    assert "reversed-build" not in order


def test_launcher_cannot_mix_controller_and_candidate_source(runtime, monkeypatch):
    root, source, archive, output, _, calls, _ = runtime
    monkeypatch.setattr(host, "FRAMEWORK", root / "other-source/AgenticBench")
    with pytest.raises(ValueError, match="same frozen source"):
        host.run_pair(root, source, archive, "bp_fe", output)
    assert not output.exists() and calls == []


def test_readonly_input_refusal_prevents_first_build(runtime, monkeypatch):
    root, source, archive, output, order, calls, _ = runtime
    def refused(*_args):
        raise ValueError("Frozen Liberty bytes differ")
    monkeypatch.setattr(host, "qualify_inputs", refused)
    report = host.run_pair(root, source, archive, "bp_fe", output)
    assert report["state"] == "failed" and "Liberty" in report["error"]
    assert report["attempts"] == [] and report["native_physical_attempts"] == 0
    assert calls == [] and "identity-build" not in order


def test_missing_native_receipt_stays_unknown_and_blocks_second_policy(runtime, monkeypatch):
    root, source, archive, output, order, calls, _ = runtime
    previous = host.subprocess.run
    def uncertain(command, **kwargs):
        if command[0] != "docker":
            return previous(command, **kwargs)
        calls.append(command)
        return SimpleNamespace(returncode=125)
    monkeypatch.setattr(host.subprocess, "run", uncertain)
    report = host.run_pair(root, source, archive, "bp_fe", output)
    assert report["state"] == "failed"
    assert report["native_physical_attempts"] is None
    assert report["native_physical_attempts_unknown"] == 1
    assert len(calls) == 1
    assert "reversed-build" not in order


@pytest.mark.parametrize("resource", ["memory", "data", "root"])
def test_all_three_live_admission_guards_are_enforced(tmp_path, monkeypatch, resource):
    # Read-only OS probes are replaced; no wait or worker is launched.
    monkeypatch.setattr(host.Path, "read_text", lambda *_args, **_kw:
        "MemAvailable: {} kB\n".format((63 if resource == "memory" else 64) * 1024**2))
    def usage(path):
        if str(path) == "/":
            return SimpleNamespace(free=(9 if resource == "root" else 10) * host.GIB)
        return SimpleNamespace(free=(63 if resource == "data" else 64) * host.GIB)
    monkeypatch.setattr(host.shutil, "disk_usage", usage)
    assert host.guard(tmp_path)["passed"] is False
