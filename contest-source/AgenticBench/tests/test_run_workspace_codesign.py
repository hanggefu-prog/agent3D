"""Control-plane tests only: no model, Docker daemon, GPU or EDA is invoked."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from global_bench.model import load_global_task
from scripts import run_workspace_codesign as driver
from scripts import workspace_codesign_evaluator as evaluator_module


THREAD = "01234567-89ab-cdef-0123-456789abcdef"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo, message):
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def repository(path):
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "algorithm.py").write_text("value = 1\n")
    git(path, "add", "algorithm.py")
    commit(path, "initial")
    return path


def test_independent_subject_pins_detached_source_without_shared_objects(tmp_path):
    source = repository(tmp_path / "source")
    (source / "algorithm.py").write_text("value = 2\n")
    git(source, "add", "algorithm.py")
    revision = commit(source, "second")
    git(source, "checkout", "--detach", revision)
    (source / "untracked.log").write_text("Not part of the frozen source\n")
    subject = tmp_path / "subject"
    driver.independent_subject(source, subject)
    assert git(subject, "rev-parse", "HEAD") == revision
    assert (subject / ".git").is_dir()
    assert not (subject / ".git/objects/info/alternates").exists()
    assert not (subject / "untracked.log").exists()
    original = source / ".git/objects" / revision[:2] / revision[2:]
    copied = subject / ".git/objects" / revision[:2] / revision[2:]
    assert original.read_bytes() == copied.read_bytes()
    assert original.stat().st_ino != copied.stat().st_ino
    (subject / "algorithm.py").write_text("value = 3\n")
    assert (source / "algorithm.py").read_text() == "value = 2\n"
    assert git(source, "status", "--porcelain", "--untracked-files=no") == ""


def test_independent_subject_copies_exact_pinned_openroad_dependency(tmp_path):
    source = repository(tmp_path / "source")
    dependency = repository(source / "OpenROAD-3D/openroad")
    pin = git(dependency, "rev-parse", "HEAD")
    git(source, "update-index", "--add", "--cacheinfo", f"160000,{pin},OpenROAD-3D/openroad")
    commit(source, "pin OpenROAD")
    subject = tmp_path / "subject"
    driver.independent_subject(source, subject)
    target = subject / "OpenROAD-3D/openroad"
    assert git(target, "rev-parse", "HEAD") == pin
    assert (target / ".git").is_dir()
    assert not (target / ".git/objects/info/alternates").exists()
    assert git(subject, "status", "--porcelain") == ""


def test_independent_subject_rejects_dependency_that_drifted_from_root_pin(tmp_path):
    source = repository(tmp_path / "source")
    dependency = repository(source / "OpenROAD-3D/openroad")
    pin = git(dependency, "rev-parse", "HEAD")
    git(source, "update-index", "--add", "--cacheinfo", f"160000,{pin},OpenROAD-3D/openroad")
    commit(source, "pin OpenROAD")
    (dependency / "algorithm.py").write_text("changed = True\n")
    git(dependency, "add", "algorithm.py")
    commit(dependency, "different dependency")
    with pytest.raises(ValueError, match="differs from its Git pin"):
        driver.independent_subject(source, tmp_path / "subject")


@pytest.mark.parametrize("symlink", [False, True])
def test_independent_subject_never_reuses_an_existing_destination(tmp_path, symlink):
    source = repository(tmp_path / "source")
    subject = tmp_path / "subject"
    if symlink:
        subject.symlink_to(tmp_path / "missing-directory")
    else:
        subject.mkdir()
    with pytest.raises(FileExistsError):
        driver.independent_subject(source, subject)


@pytest.mark.parametrize("objective", ["timing", "thermal", "routed_thermal"])
def test_round_prompt_exposes_real_config_workspace_and_task_endpoint(objective):
    task = load_global_task(driver.ROOT / "global_tasks" / driver.TASKS[objective])
    prompt = driver.round_prompt(task, "bp", 2, 5, {"best": {"score": 1.02}, "history": []})
    assert task.instruction in prompt
    for expected in ("/workspace/agentic_candidate.yaml", "partition and placement mappings",
                     "/scratch/plan-round-02.md", "Round 2 of 5", "file-count limit",
                     "Submit file edits", "changes carry over",
                     "separate reference qualification", "5-final budget", "Native compiled code",
                     "partition.config", "placement.config", "30-minute turn", '"score": 1.02'):
        assert expected in prompt
    assert "five-final budget" not in driver.round_prompt(task, "bp", 1, 2, {})
    assert "exactly-two-OPT final budget" not in prompt


def test_workspace_endpoints_cover_distinct_instantiated_global_tasks():
    assert driver.TASKS == {key: value["task_id"] for key, value in evaluator_module.ENDPOINTS.items()}
    assert "lol_full_flow_ppa" not in driver.TASKS.values()
    assert driver.TASKS["thermal"] != driver.TASKS["routed_thermal"]


@pytest.mark.parametrize("result, expected", [
    ({"success": True, "metrics": {"combined_score": 1.25}}, 1.25),
    ({"success": False, "metrics": {"combined_score": 9}}, None),
    ({"success": 1, "metrics": {"combined_score": 9}}, None),
    ({"success": True, "metrics": {"combined_score": True}}, None),
    ({"success": True, "metrics": {"combined_score": float("nan")}}, None),
    ({"success": True, "metrics": {"combined_score": float("inf")}}, None),
    ({"success": True, "metrics": None}, None),
    ({"success": True}, None), (None, None),
])
def test_score_accepts_only_successful_finite_numeric_feedback(result, expected):
    assert driver._score(result) == expected


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # These tests isolate the driver lifecycle, not physical qualification.
    class Reference:
        policy = "FRESH"
        evaluator_invocations = 1
        def __init__(self, runtime, *args):
            self.runtime = runtime
        def to_dict(self):
            return {"policy": self.policy, "fixture": True}
        def obtain(self, evaluator, output):
            return evaluator.evaluate(self.runtime.source, output, None)
    monkeypatch.setattr(driver, "WorkspaceReference", Reference)
    source = tmp_path / "source"
    source.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    config = "design: bp\nthreads: 32\npartition:\n  algorithm: tp-gnn\nplacement:\n  algorithm: concurrent-dreamplace\n"
    (profile / "config.yaml").write_text(config)
    batch = tmp_path / "batch"
    batch.mkdir()
    bundle = batch / "input"
    bundle.mkdir()
    calls = SimpleNamespace(evaluations=[], turns=[], docker=[], events=[], initialize=[], clones=[])
    options = SimpleNamespace(scores=[1.1, 1.2, 1.15], reference_ok=True,
                              turn_ok=True, plan=True, symlink_plan=False,
                              evaluator_error=None, constructor_error=None,
                              start_error=False, stop_error=False)
    task = load_global_task(driver.ROOT / "global_tasks" / driver.TASKS["timing"])
    task = replace(task, budget=replace(task.budget, global_rounds=3, expensive_evaluations=3))
    monkeypatch.setattr(driver, "load_global_task", lambda path: task)

    @contextmanager
    def slot():
        calls.events.append("slot-enter")
        yield 2
        calls.events.append("slot-exit")

    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch)
        return path

    runtime = SimpleNamespace(source=source, source_commit="1" * 40,
                              image="sha256:" + "2" * 64, cpus=16, memory_gb=48,
                              gpus=[2], min_gpu_free_mb=2048, profile=lambda case: profile,
                              contained=contained, slot=slot)

    class Evaluator:
        def __init__(self, *args):
            if options.constructor_error:
                raise options.constructor_error
        def initialize_subject(self, subject):
            calls.initialize.append(subject)
            (subject / "agentic_candidate.yaml").write_text("partition: {}\nplacement: {}\n")
        def evaluate(self, subject, output, reference):
            calls.evaluations.append((subject, output, reference))
            calls.events.append("evaluate-" + output.name)
            output.mkdir(parents=True)
            (output / "run").mkdir()
            if output.name == "reference":
                return {"success": options.reference_ok, "metrics": {"combined_score": 1.0},
                        "run": str(output / "run"), "error": None}
            if options.evaluator_error:
                raise options.evaluator_error
            value = options.scores[int(output.name.split("-")[-1]) - 1]
            return {"success": value is not None,
                    "metrics": {"combined_score": value} if value is not None else {},
                    "run": str(output / "run"), "error": None if value is not None else "Invalid placement"}

    def clone(source, subject):
        calls.clones.append((source, subject))
        subject.mkdir()
        (subject / ".git").mkdir()

    def container_command(**kwargs):
        calls.events.append("container-command")
        calls.container = kwargs
        return ["docker", "run", "--name", kwargs["name"]]

    def subprocess_run(command, **kwargs):
        calls.docker.append(command)
        action = command[1]
        calls.events.append("docker-" + action)
        if (options.start_error and action == "run") or (options.stop_error and action == "stop"):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    def turn(**kwargs):
        calls.turns.append(kwargs)
        calls.events.append("turn")
        number = len(calls.turns)
        scratch = calls.container["scratch"]
        path = scratch / f"plan-round-{number:02d}.md"
        if options.symlink_plan:
            path.symlink_to(profile / "config.yaml")
        elif options.plan:
            path.write_text("Diagnose timing, edit the algorithm, and validate.\n")
        return {"success": options.turn_ok, "thread_id": THREAD if options.turn_ok else None}

    monkeypatch.setattr(evaluator_module, "WorkspaceEvaluator", Evaluator)
    monkeypatch.setattr(driver, "independent_subject", clone)
    monkeypatch.setattr(driver, "agent_container_command", container_command)
    monkeypatch.setattr(driver.subprocess, "run", subprocess_run)
    monkeypatch.setattr(driver, "run_workspace_turn", turn)
    return SimpleNamespace(runtime=runtime, root=batch / "experiment", bundle=bundle,
                           task=task, calls=calls, options=options, config=config)


def run(harness):
    return driver.run_workspace_experiment(harness.runtime, "bp", "timing", harness.bundle,
                                           harness.root, "/trusted/codex")


def read(harness, name):
    return json.loads((harness.root / name).read_text())


def test_full_driver_preserves_context_records_resources_and_counts_failed_finals(harness):
    harness.options.scores = [None, 1.2, 1.1]
    summary = run(harness)
    assert summary["state"] == "completed"
    assert summary["final_evaluations"] == 3
    assert summary["reference_evaluations"] == 1
    assert summary["successful_final_evaluations"] == 2
    assert summary["best"]["round"] == 2 and summary["best"]["score"] == 1.2
    assert len(harness.calls.evaluations) == 4
    assert [row["thread_id"] for row in harness.calls.turns] == [None, THREAD, THREAD]
    assert "Invalid placement" in harness.calls.turns[1]["prompt"]


def test_per_round_retention_is_inside_driver_lock_after_evaluation(harness, monkeypatch):
    from scripts import experiment_retention as retention
    harness.runtime.retention_mode = "best-per-round"
    seen = []
    def hook(root, mode, number, *, driver_lock, final=False):
        assert not driver_lock.closed
        if final and number is None:
            number = read(harness, "status.json")["final_evaluations"]
        summary, status, config = (read(harness, name) for name in
                                   ("summary.json", "status.json", "run_config.json"))
        retention._round_snapshot(root, summary, status, config, (number, final))
        assert len(harness.calls.evaluations) == number + 1
        assert len(harness.calls.turns) == number
        seen.append((number, final))
        return {"state": "applied"}
    monkeypatch.setattr(retention, "finish_round_retention", hook)
    result = run(harness)
    assert result["state"] == "completed"
    assert seen == [(1, False), (2, False), (3, False), (3, True)]


def test_ten_round_driver_loads_static_task_and_keeps_one_reference(harness, monkeypatch):
    from scripts.workspace_task_selection import load_registered_task, TIMING_TEN_TASK
    task = load_registered_task(TIMING_TEN_TASK, objective="timing")
    monkeypatch.setattr(driver, "load_global_task", load_global_task)
    harness.runtime.workspace_rounds = 10
    harness.runtime.workspace_task_digest = task.digest()
    harness.options.scores = [1.1] * 9 + [None]
    summary = run(harness)
    assert summary["state"] == "completed"
    assert summary["final_evaluations"] == 10
    assert summary["reference_evaluations"] == 1
    assert summary["successful_final_evaluations"] == 9
    assert len(harness.calls.evaluations) == 11
    assert len(harness.calls.turns) == 10
    config = read(harness, "run_config.json")
    assert config["task"]["task_id"] == config["evaluation_endpoint"]["task_id"] == TIMING_TEN_TASK
    assert config["task_digest"] == task.digest()
    assert config["rounds"] == config["final_budget"] == 10
    assert all(turn["timeout_seconds"] == 1800 for turn in harness.calls.turns)


def test_adopted_reference_is_not_a_native_evaluator_invocation(harness, monkeypatch):
    class Adopt:
        policy, evaluator_invocations = "ADOPT", 0
        def __init__(self, *a): pass
        def to_dict(self): return {"policy": "ADOPT", "fixture": True}
        def obtain(self, evaluator, output):
            output.mkdir(parents=True)
            run = output / "run"
            run.mkdir()
            return {"success": True, "metrics": {"combined_score": 1.}, "run": str(run)}
    monkeypatch.setattr(driver, "WorkspaceReference", Adopt)
    summary = run(harness)
    assert summary["reference_evaluations"] == 0
    assert summary["reference_adoptions"] == 1
    assert summary["final_evaluations"] == len(harness.calls.evaluations) == 3
    assert read(harness, "run_config.json")["reference_policy"] == "ADOPT"
    assert len(harness.calls.clones) == len(harness.calls.initialize) == 1
    assert len([c for c in harness.calls.docker if c[1] == "stop"]) == 3
    assert read(harness, "status.json")["final_evaluations"] == 3
    config = read(harness, "run_config.json")
    assert config["model"] == "gpt-5.6-sol"
    assert config["task"]["environment"]["cpu"] == 32
    assert config["resources"]["cpus"] == 16
    assert config["resources"]["evaluation_threads"] == 32
    assert config["resources"]["memory_gb"] == 48
    assert config["reference_qualification_in_candidate_budget"] is False
    assert config["case_binding"] == {
        "policy": "explicit caller selection; task template has no pinned case", "selected_case": "bp"}
    assert len(config["profile_config_sha256"]) == 64
    assert (harness.root / "control/base_config.yaml").read_text() == harness.config
    assert read(harness, "control/run_config.json") == config
    events = harness.calls.events
    assert events.index("docker-stop") < events.index("evaluate-round-01")
    assert all((harness.root / "rounds" / f"{i:02d}" / "plan.md").is_file() for i in range(1, 4))


def test_failed_reference_has_summary_and_never_starts_agent(harness):
    harness.options.reference_ok = False
    with pytest.raises(RuntimeError, match="reference failed"):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["state"] == "needs-attention"
    assert summary["failed_phase"] == "qualifying-reference"
    assert summary["final_evaluations"] == 0 and summary["reference_evaluations"] == 1
    assert summary["best"] is None and len(summary["history"]) == 1
    assert not harness.calls.turns and not harness.calls.clones


def test_routed_thermal_driver_uses_original_five_round_task_and_counts_failed_finals(harness, monkeypatch):
    task = load_global_task(driver.ROOT / "global_tasks" / driver.TASKS["routed_thermal"])
    selected = []
    def load(path):
        selected.append(path.name)
        return task
    monkeypatch.setattr(driver, "load_global_task", load)
    harness.options.scores = [1.01, None, 1.02, 1.0, 1.015]
    summary = driver.run_workspace_experiment(harness.runtime, "bp", "routed_thermal", harness.bundle,
                                              harness.root, "/trusted/codex")
    assert selected == ["lol_partition_placement_thermal"]
    assert summary["final_evaluations"] == 5
    assert summary["successful_final_evaluations"] == 4
    assert summary["reference_evaluations"] == 1
    assert summary["best"]["round"] == 3
    assert len(harness.calls.evaluations) == 6
    assert [row["thread_id"] for row in harness.calls.turns] == [None] + [THREAD] * 4
    config = read(harness, "run_config.json")
    assert config["evaluation_endpoint"]["opt_passes"] == 1
    assert config["evaluation_endpoint"]["backend_stop"] == "thermal"
    assert config["task"]["task_id"] == "lol_partition_placement_thermal"
    assert config["task_digest"] == task.digest()
    assert all("maintained full Open3DBench backend" in row["prompt"] for row in harness.calls.turns)


@pytest.mark.parametrize("option", ["turn_ok", "plan", "symlink_plan"])
def test_invalid_agent_turn_cannot_spend_a_physical_final(harness, option):
    setattr(harness.options, option, option == "symlink_plan")
    with pytest.raises(RuntimeError):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["state"] == "needs-attention"
    assert summary["final_evaluations"] == 0
    assert summary["best"]["round"] == 0
    assert len(harness.calls.evaluations) == 1
    assert harness.calls.docker[-1][1] == "stop"
    assert read(harness, "best.json")["round"] == 0


def test_unexpected_evaluator_error_consumes_attempt_and_prevents_replay(harness):
    harness.options.evaluator_error = RuntimeError("Verifier crashed")
    with pytest.raises(RuntimeError, match="Verifier crashed"):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["final_evaluations"] == 1 and summary["failed_phase"] == "final-evaluation"
    assert summary["history"][-1]["round"] == 1
    assert summary["history"][-1]["success"] is False
    previous = (harness.root / "run_config.json").read_bytes()
    with pytest.raises(RuntimeError, match="no implicit replay"):
        run(harness)
    assert (harness.root / "run_config.json").read_bytes() == previous
    assert len(harness.calls.evaluations) == 2


@pytest.mark.parametrize("option", ["start_error", "stop_error"])
def test_container_failure_never_starts_candidate_evaluation(harness, option):
    setattr(harness.options, option, True)
    with pytest.raises(subprocess.CalledProcessError):
        run(harness)
    assert read(harness, "summary.json")["final_evaluations"] == 0
    assert len(harness.calls.evaluations) == 1


def test_evaluator_setup_failure_is_not_left_running(harness):
    harness.options.constructor_error = ValueError("Profile/case identity mismatch")
    with pytest.raises(ValueError, match="Profile/case"):
        run(harness)
    assert read(harness, "status.json")["state"] == "needs-attention"
    assert read(harness, "summary.json")["history"] == []


@pytest.mark.parametrize("budget", [0, True, 3.0, "3", 2])
def test_budget_must_be_positive_integer_and_one_final_per_round(harness, monkeypatch, budget):
    task = replace(harness.task, budget=replace(harness.task.budget, expensive_evaluations=budget))
    monkeypatch.setattr(driver, "load_global_task", lambda path: task)
    with pytest.raises(ValueError, match="one final per round"):
        run(harness)
    assert not harness.root.exists()
    assert not harness.calls.evaluations
