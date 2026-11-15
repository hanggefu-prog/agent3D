"""Mocked role/lifecycle tests: no model, Docker daemon or physical EDA run."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_workspace_agentic3d as driver


SESSIONS = {
    "global-analyzer": "10000000-0000-0000-0000-000000000001",
    "local-analyzer": "20000000-0000-0000-0000-000000000002",
    "task-agent": "30000000-0000-0000-0000-000000000003",
}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # Physical reference gates have separate real-store/control tests.
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
    git(source, "init", "-q")
    (source / "algorithm.py").write_text("value = 1\n")
    git(source, "add", ".")
    git(source, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "initial")
    revision = git(source, "rev-parse", "HEAD")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("design: bp_fe\nthreads: 32\npartition: {}\nplacement: {}\n")
    batch = tmp_path / "batch"
    batch.mkdir()
    bundle = batch / "input"
    bundle.mkdir()
    calls = SimpleNamespace(events=[], evaluations=[], turns=[], containers=[], docker=[])
    options = SimpleNamespace(scores=[1.05, None, 1.03, 1.1, 1.06], fail_phase=None,
                              malformed_phase=None, evidence=None, mutate_phase=None,
                              plan=True, symlink_plan=False, reference_ok=True,
                              evaluator_error=None, infrastructure_error=None,
                              post_decision=None, global_decision=None,
                              tool_calls=1, duplicate_session=False, switch_session=False,
                              start_error=False, stop_error=False, isolation_ok=True,
                              stale_post_evidence=False, fail_round=None,
                              evaluator_error_round=None)

    @contextmanager
    def slot():
        yield 2

    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch)
        return path

    runtime = SimpleNamespace(source=source, source_commit=revision,
                              image="sha256:" + "2" * 64, cpus=16, memory_gb=48,
                              gpus=[2], min_gpu_free_mb=2048, profile=lambda case: profile,
                              contained=contained, slot=slot)

    class Evaluator:
        def __init__(self, runtime, case, objective, bundle, task):
            calls.endpoint = objective
        def initialize_subject(self, subject):
            (subject / "agentic_candidate.yaml").write_text("partition: {}\nplacement: {}\n")
        def evaluate(self, subject, output, reference):
            calls.evaluations.append((subject, output, reference))
            calls.events.append("evaluate-" + output.name)
            output.mkdir(parents=True)
            (output / "run").mkdir()
            if output.name == "reference":
                score = 1.0 if options.reference_ok else None
            else:
                if options.evaluator_error and (options.evaluator_error_round is None
                        or options.evaluator_error_round == int(output.name.split("-")[-1])):
                    raise options.evaluator_error
                score = options.scores[int(output.name.split("-")[-1]) - 1]
            result = {"success": score is not None,
                      "metrics": {"combined_score": score} if score is not None else {},
                      "run": str(output / "run"), "error": None if score is not None else "Invalid placement"}
            if output.name != "reference" and options.infrastructure_error:
                result.update(success=False, metrics={}, error=options.infrastructure_error)
            (output / "feedback.json").write_text(json.dumps(result))
            return result

    def container_command(**kwargs):
        calls.containers.append(kwargs)
        calls.container = kwargs
        return ["docker", "run", "--name", kwargs["name"]]

    original_run = subprocess.run
    def subprocess_run(command, **kwargs):
        if command[0] != "docker":
            return original_run(command, **kwargs)
        calls.docker.append(command)
        if command[1] == "exec":
            proof = {"target": "/workspace/agentic_candidate.yaml",
                     "read_only": options.isolation_ok, "scratch_writable": True, "errno": 30}
            return SimpleNamespace(returncode=0 if options.isolation_ok else 1,
                                   stdout=json.dumps(proof), stderr="")
        if ((options.start_error and command[1] == "run")
                or (options.stop_error and command[1] == "stop")):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    def turn(**kwargs):
        calls.turns.append(kwargs)
        context = json.loads(kwargs["prompt"].split("---BEGIN TRUSTED CONTEXT---\n", 1)[1]
                             .split("\n---END TRUSTED CONTEXT---", 1)[0])
        role, phase = context["role"], context["phase"]
        number = context["candidate_round"]
        calls.events.append(phase)
        output = kwargs["output"]
        output.mkdir(parents=True)
        (output / "prompt.md").write_text(kwargs["prompt"])
        if options.mutate_phase == phase:
            (calls.container["subject"] / "algorithm.py").write_text("tampered = True\n")
        if role == "task-agent":
            (calls.container["subject"] / "algorithm.py").write_text(f"value = {number + 1}\n")
            plan = calls.container["scratch"] / f"plan-round-{number:02d}.md"
            if options.symlink_plan:
                plan.symlink_to(source / "algorithm.py")
            elif options.plan:
                plan.write_text("Diagnose; change algorithm; test; preserve the evaluator.\n")
            response = "Implemented and inspected the candidate."
        else:
            decision = {"global-plan": "dispatch", "local-pre": "recommend",
                        "local-post": "accept" if context.get("eligible_for_acceptance") else "reject",
                        "global-adjudication": "accept" if context.get("eligible_for_acceptance") else "reject"}[phase]
            if phase == "local-post" and options.post_decision:
                decision = options.post_decision
            if phase == "global-adjudication" and options.global_decision:
                decision = options.global_decision
            response = {"schema_version": 1, "role": role, "phase": phase,
                        "candidate_round": number, "decision": decision,
                        "reason": "Read trusted evidence and retain the fixed objective.",
                        "evidence_paths": [options.evidence or "/results/reference/feedback.json"]}
            if phase in {"local-post", "global-adjudication"} and not options.stale_post_evidence:
                response["evidence_paths"] = [f"/results/round-{number:02d}/feedback.json"]
            if phase in {"global-plan", "local-pre"}:
                response.update(hypothesis="Frontend balance may improve the endpoint.",
                                target_stages=["20_partition", "30_placement"],
                                guidance="Inspect the frontend and make one coherent change.")
            response = json.dumps(response)
        if options.malformed_phase == phase:
            response = "not valid JSON"
        session = SESSIONS[role]
        if options.duplicate_session:
            session = SESSIONS["global-analyzer"]
        if options.switch_session and kwargs["thread_id"]:
            session = "40000000-0000-0000-0000-000000000004"
        failed = options.fail_phase == phase and options.fail_round in (None, number)
        return {"success": not failed, "thread_id": session,
                "session_identity_valid": True, "terminal_tool_calls": options.tool_calls,
                "final_message": response}

    monkeypatch.setattr(driver, "WorkspaceEvaluator", Evaluator)
    monkeypatch.setattr(driver, "agent_container_command", container_command)
    monkeypatch.setattr(driver.subprocess, "run", subprocess_run)
    monkeypatch.setattr(driver, "run_workspace_turn", turn)
    return SimpleNamespace(runtime=runtime, root=batch / "experiment", bundle=bundle,
                           calls=calls, options=options, source=source)


def run(harness, objective="timing"):
    return driver.run_workspace_agentic3d_experiment(harness.runtime, "bp_fe", objective,
        harness.bundle, harness.root, "/trusted/codex")


def read(harness, name):
    return json.loads((harness.root / name).read_text())


def test_per_round_retention_waits_for_all_five_roles(harness, monkeypatch):
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
        assert len(harness.calls.turns) == number * 5
        assert read(harness, "history.json")[-1]["adjudication"] == "completed"
        seen.append((number, final))
        return {"state": "applied"}
    monkeypatch.setattr(retention, "finish_round_retention", hook)
    assert run(harness)["state"] == "completed"
    assert seen == [(n, False) for n in range(1, 6)] + [(5, True)]


def test_analysis_failure_prevents_round_cleanup(harness, monkeypatch):
    from scripts import experiment_retention as retention
    harness.runtime.retention_mode = "best-per-round"
    harness.options.fail_phase = "global-adjudication"
    harness.options.fail_round = 1
    def forbidden(*args, **kwargs):
        pytest.fail("Cleanup must not run before successful adjudication")
    monkeypatch.setattr(retention, "finish_round_retention", forbidden)
    with pytest.raises(RuntimeError):
        run(harness)
    assert read(harness, "status.json")["state"] == "needs-attention"


def pin_budget(monkeypatch, *, rounds, finals):
    if rounds == finals == 10:
        from scripts.workspace_task_selection import load_registered_task, TIMING_TEN_TASK
        return load_registered_task(TIMING_TEN_TASK, objective="timing")
    task = driver.load_global_task(driver.ROOT / "global_tasks" / driver.TASKS["timing"])
    task = replace(task, budget=replace(task.budget, global_rounds=rounds,
                                       expensive_evaluations=finals))
    monkeypatch.setattr(driver, "load_global_task", lambda path: task)
    return task


@pytest.mark.parametrize("rounds, finals", [
    (5, 10), (10, 5), (4, 4), (6, 6), (0, 0), (50, 50),
    (True, 5), (5, True), (5.0, 5), (5, 5.0), ("10", 10), (10, "10"),
])
def test_budget_rejects_mismatch_unsupported_or_noninteger_before_any_work(harness, monkeypatch,
                                                                       rounds, finals):
    pin_budget(monkeypatch, rounds=rounds, finals=finals)
    with pytest.raises(ValueError, match="matching rounds.*exactly 5 or 10"):
        run(harness)
    assert not harness.root.exists()
    assert not harness.calls.evaluations and not harness.calls.turns and not harness.calls.docker


@pytest.mark.parametrize("scores", [
    [1.05] * 10,
    [1.05, None, 1.03, 1.1, 1.06, None, 1.2, 1.1, 1.15, 1.25],
    [None] * 10,
])
def test_ten_finals_preserve_role_order_sessions_prompts_and_separate_accounting(harness,
                                                                             monkeypatch, scores):
    task = pin_budget(monkeypatch, rounds=10, finals=10)
    harness.runtime.workspace_rounds = 10
    harness.options.scores = scores
    summary = run(harness)
    assert summary["state"] == "completed"
    assert summary["rounds"] == summary["final_budget"] == summary["final_evaluations"] == 10
    assert summary["generated_candidates"] == 10
    assert summary["reference_evaluations"] == 1
    assert summary["successful_final_evaluations"] == sum(score is not None for score in scores)
    assert summary["failed_final_evaluations"] == sum(score is None for score in scores)
    assert summary["candidate_execution"]["evaluator_invocations"] == 10
    assert summary["candidate_execution"]["native_physical_attempts"] is None  # Mock has no native receipts.
    assert summary["reference_execution"]["evaluator_invocations"] == 1
    assert summary["role_turn_accounting"]["attempted"] == 50
    assert summary["role_turn_accounting"]["completed"] == 50
    assert summary["role_turn_accounting"]["failed"] == 0
    assert summary["role_sessions"] == SESSIONS
    expected = ["evaluate-reference"]
    for number in range(1, 11):
        expected += ["global-plan", "local-pre", "task", f"evaluate-round-{number:02d}",
                     "local-post", "global-adjudication"]
        for phase in driver.PHASE_ROLES:
            context = read(harness, f"rounds/{number:02d}/{phase}/context.json")
            assert context["final_budget"] == 10
            assert context["candidate_round"] == number
            prompt = (harness.root / f"rounds/{number:02d}/{phase}/agent/prompt.md").read_text()
            if phase == "task":
                assert f"Round {number} of 10" in prompt
                assert "10-final budget" in prompt
            else:
                assert f"Round {number} of 10" in prompt
            assert "Five candidate final attempts" not in prompt
        assert (harness.root / f"rounds/{number:02d}/plan.md").is_file()
    assert harness.calls.events == expected
    assert len(harness.calls.evaluations) == 11  # Reference remains outside the ten finals.
    seen = {role: 0 for role in SESSIONS}
    for call, container in zip(harness.calls.turns, harness.calls.containers):
        role = container["scratch"].name
        assert container["subject_read_only"] is (role != "task-agent")
        assert call["thread_id"] == (SESSIONS[role] if seen[role] else None)
        assert call["timeout_seconds"] == 1800
        seen[role] += 1
    assert seen == {"global-analyzer": 20, "local-analyzer": 20, "task-agent": 10}
    config = read(harness, "run_config.json")
    assert config["rounds"] == config["final_budget"] == 10
    assert config["task_digest"] == task.digest()
    assert config["task"]["budget"]["global_rounds"] == 10
    assert config["task"]["budget"]["expensive_evaluations"] == 10
    assert config["reference_qualification_in_candidate_budget"] is False
    status = read(harness, "status.json")
    assert status["rounds"] == status["final_budget"] == status["final_evaluations"] == 10
    assert status["round"] == 10
    best = 1.0
    improvements = 0
    for score in scores:
        if score is not None and score > best:
            best, improvements = score, improvements + 1
    assert summary["best"]["score"] == best
    assert summary["trusted_measured_improvements"] == improvements
    assert summary["role_accepted_improvements"] == improvements


@pytest.mark.parametrize("failure", ["task-turn", "evaluator"])
def test_ten_round_failure_after_five_finals_stops_once_and_preserves_budget(harness, monkeypatch,
                                                                         failure):
    pin_budget(monkeypatch, rounds=10, finals=10)
    harness.runtime.workspace_rounds = 10
    harness.options.scores = [1.05] * 10
    if failure == "task-turn":
        harness.options.fail_phase, harness.options.fail_round = "task", 6
        expected_finals, expected_generated, exception = 5, 5, RuntimeError
    else:
        harness.options.evaluator_error = OSError("No space left on device")
        harness.options.evaluator_error_round = 6
        expected_finals, expected_generated, exception = 6, 6, OSError
    with pytest.raises(exception):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["state"] == "needs-attention"
    assert summary["rounds"] == summary["final_budget"] == 10
    assert summary["final_evaluations"] == expected_finals
    assert summary["generated_candidates"] == expected_generated
    assert summary["reference_evaluations"] == 1
    assert summary["successful_final_evaluations"] == 5
    assert summary["failed_final_evaluations"] == int(failure == "evaluator")
    assert summary["role_turn_accounting"]["attempted"] == 28
    assert summary["role_turn_accounting"]["failed"] == int(failure == "task-turn")
    assert len(harness.calls.evaluations) == expected_finals + 1
    assert not (harness.root / "rounds/07").exists()
    before = len(harness.calls.turns), len(harness.calls.evaluations)
    with pytest.raises(RuntimeError, match="no implicit replay"):
        run(harness)
    assert (len(harness.calls.turns), len(harness.calls.evaluations)) == before


def test_adoption_keeps_five_finals_and_zero_reference_native_accounting(harness, monkeypatch):
    class Adopt:
        policy, evaluator_invocations = "ADOPT", 0
        def __init__(self, *a): pass
        def to_dict(self): return {"policy": "ADOPT", "fixture": True}
        def obtain(self, evaluator, output):
            output.mkdir(parents=True)
            reference = output / "run"
            reference.mkdir()
            result = {"success": True, "metrics": {"combined_score": 1.}, "run": str(reference)}
            (output / "feedback.json").write_text(json.dumps(result))
            return result
    monkeypatch.setattr(driver, "WorkspaceReference", Adopt)
    summary = run(harness)
    assert summary["reference_evaluations"] == 0
    assert summary["reference_adoptions"] == 1
    assert summary["reference_execution"]["native_physical_attempts"] == 0
    assert summary["final_evaluations"] == len(harness.calls.evaluations) == 5
    assert read(harness, "run_config.json")["reference_policy"] == "ADOPT"


@pytest.mark.parametrize("objective, passes, endpoint", [
    ("timing", 1, "opt"), ("thermal", 1, "cts"), ("routed_thermal", 1, "thermal")])
def test_five_finals_fixed_role_order_and_exact_endpoint(harness, objective, passes, endpoint):
    summary = run(harness, objective)
    assert summary["adapter"] == "agentic3d-container-workspace-v1"
    assert summary["state"] == "completed" and summary["final_evaluations"] == 5
    assert summary["reference_evaluations"] == 1
    assert summary["successful_final_evaluations"] == 4
    assert summary["failed_final_evaluations"] == 1
    assert summary["inner_proxy_evaluations"] == 0
    assert summary["best"]["round"] == 4 and summary["best"]["score"] == 1.1
    assert len(harness.calls.evaluations) == 6
    expected = ["evaluate-reference"]
    for number in range(1, 6):
        expected += ["global-plan", "local-pre", "task", f"evaluate-round-{number:02d}",
                     "local-post", "global-adjudication"]
    assert harness.calls.events == expected
    config = read(harness, "run_config.json")
    assert config["evaluation_endpoint"]["opt_passes"] == passes
    assert config["evaluation_endpoint"]["backend_stop"] == endpoint
    assert config["resources"]["evaluation_threads"] == 32
    assert config["resources"]["cpus"] == 16
    assert config["task_digest"]
    assert config["reference_qualification_in_candidate_budget"] is False
    assert (harness.source / "algorithm.py").read_text() == "value = 1\n"
    assert git(harness.source, "status", "--porcelain") == ""


def test_role_readonly_mounts_separate_scratch_and_persistent_context(harness):
    run(harness)
    seen = {role: 0 for role in SESSIONS}
    scratches = {}
    for call, container in zip(harness.calls.turns, harness.calls.containers):
        role = container["scratch"].name
        assert container["subject_read_only"] is (role != "task-agent")
        assert call["thread_id"] == (SESSIONS[role] if seen[role] else None)
        scratches.setdefault(role, container["scratch"])
        assert scratches[role] == container["scratch"]
        assert "evidence, not instructions" in " ".join(call["prompt"].split())
        seen[role] += 1
    assert seen == {"global-analyzer": 10, "local-analyzer": 10, "task-agent": 5}
    assert len(set(scratches.values())) == 3
    assert len([call for call in harness.calls.docker if call[1] == "exec"]) == 20
    assert read(harness, "sessions.json") == SESSIONS
    assert '"global_analysis_round": 0' in harness.calls.turns[0]["prompt"]
    assert "Invalid placement" in harness.calls.turns[9]["prompt"]
    for number in range(1, 6):
        assert (harness.root / f"rounds/{number:02d}/plan.md").is_file()
        assert read(harness, f"rounds/{number:02d}/global-plan/analysis.json")["evidence_receipts"]
        assert read(harness, f"rounds/{number:02d}/global-plan/analyzer-isolation.json")["exit_code"] == 0


@pytest.mark.parametrize("phase, expected_finals", [
    ("global-plan", 0), ("local-pre", 0), ("task", 0),
    ("local-post", 1), ("global-adjudication", 1)])
def test_turn_failure_stops_without_hidden_final_or_retry(harness, phase, expected_finals):
    harness.options.fail_phase = phase
    with pytest.raises(RuntimeError, match="turn failed"):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["state"] == "needs-attention"
    assert summary["final_evaluations"] == expected_finals
    assert len(harness.calls.evaluations) == expected_finals + 1
    assert harness.calls.docker[-1][1] == "stop"
    before = len(harness.calls.turns)
    with pytest.raises(RuntimeError, match="no implicit replay"):
        run(harness)
    assert len(harness.calls.turns) == before


@pytest.mark.parametrize("option, value, match", [
    ("malformed_phase", "global-plan", "JSON"),
    ("evidence", "/home/secret", "evidence path"),
    ("evidence", "/results/reference/../feedback.json", "evidence path"),
    ("evidence", "/results/missing.json", "evidence"),
    ("mutate_phase", "global-plan", "read-only"),
    ("plan", False, "plan"), ("symlink_plan", True, "plan"),
    ("tool_calls", 0, "terminal"),
    ("duplicate_session", True, "session"),
])
def test_invalid_role_evidence_or_plan_cannot_spend_final(harness, option, value, match):
    setattr(harness.options, option, value)
    with pytest.raises((RuntimeError, ValueError), match=match):
        run(harness)
    assert read(harness, "summary.json")["final_evaluations"] == 0
    assert len(harness.calls.evaluations) == 1


def test_reference_failure_never_starts_roles(harness):
    harness.options.reference_ok = False
    with pytest.raises(RuntimeError, match="reference"):
        run(harness)
    assert not harness.calls.turns
    assert read(harness, "summary.json")["final_evaluations"] == 0


def test_mount_probe_failure_stops_before_any_analyzer_model_call(harness):
    harness.options.isolation_ok = False
    with pytest.raises(RuntimeError, match="isolation qualification"):
        run(harness)
    assert not harness.calls.turns
    assert len(harness.calls.evaluations) == 1
    assert harness.calls.docker[-1][1] == "stop"
    assert read(harness, "summary.json")["final_evaluations"] == 0


def test_post_audit_cannot_use_only_baseline_evidence(harness):
    harness.options.stale_post_evidence = True
    with pytest.raises(ValueError, match="current-round"):
        run(harness)
    assert read(harness, "summary.json")["final_evaluations"] == 1


def test_evaluator_exception_is_charged_once_then_stops(harness):
    harness.options.evaluator_error = OSError("No space left on device")
    with pytest.raises(OSError):
        run(harness)
    summary = read(harness, "summary.json")
    assert summary["final_evaluations"] == 1
    assert summary["failed_final_evaluations"] == 1
    assert summary["history"][-1]["success"] is False
    assert len(harness.calls.turns) == 3


def test_explicit_infrastructure_failure_is_charged_once_then_stops(harness):
    harness.options.infrastructure_error = "OSError: [Errno 28] No space left on device"
    with pytest.raises(RuntimeError, match="infrastructure"):
        run(harness)
    assert read(harness, "summary.json")["final_evaluations"] == 1
    assert len(harness.calls.turns) == 3


@pytest.mark.parametrize("message", [
    "FileNotFoundError: candidate-config.json",
    "PermissionError: candidate algorithm attempted to write a protected file",
    "OSError: candidate-selected path does not exist",
])
def test_candidate_file_exceptions_are_not_automatically_infrastructure(harness, message):
    harness.options.infrastructure_error = message
    summary = run(harness)
    assert summary["final_evaluations"] == 5
    assert summary["failed_final_evaluations"] == 5
    assert summary["state"] == "completed"


def test_role_claims_cannot_accept_failed_or_worse_physical_result(harness):
    harness.options.scores = [None, .9, 1., .8, .7]
    harness.options.post_decision = harness.options.global_decision = "accept"
    summary = run(harness)
    assert summary["best"]["round"] == summary["incumbent"]["round"] == 0
    assert not any(row.get("accepted_as_incumbent") for row in summary["history"])
    assert all(not row["acceptance"]["trusted_improvement"] for row in summary["history"][1:])


def test_verified_best_is_distinct_from_role_rejected_incumbent(harness):
    harness.options.post_decision = "reject"
    summary = run(harness)
    assert summary["best"]["score"] == 1.1
    assert summary["incumbent"]["round"] == 0
    assert not any(row.get("accepted_as_incumbent") for row in summary["history"])


def test_post_roles_receive_unambiguous_prior_best_and_neutral_examples(harness):
    summary = run(harness)
    assert summary["history"][1]["prior_best_verified"]["score"] == 1.0
    for phase in ("local-post", "global-adjudication"):
        context = read(harness, f"rounds/01/{phase}/context.json")
        assert context["prior_best_verified"]["score"] == 1.0
        assert context["best_verified"]["score"] == 1.05
        assert context["eligible_for_acceptance"] is True
        prompt = (harness.root / f"rounds/01/{phase}/agent/prompt.md").read_text()
        assert "context.prior_best_verified" in prompt
        assert '"decision": "reject"' in prompt.split("---BEGIN TRUSTED CONTEXT---")[0]
    for phase, decision in (("global-plan", "dispatch"), ("local-pre", "recommend")):
        prompt = (harness.root / f"rounds/01/{phase}/agent/prompt.md").read_text()
        assert f'"decision": "{decision}"' in prompt.split("---BEGIN TRUSTED CONTEXT---")[0]


@pytest.mark.parametrize("option", ["start_error", "stop_error"])
def test_container_failure_never_launches_final(harness, option):
    setattr(harness.options, option, True)
    with pytest.raises(subprocess.CalledProcessError):
        run(harness)
    assert read(harness, "summary.json")["final_evaluations"] == 0
