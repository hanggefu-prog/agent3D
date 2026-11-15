"""No model, GPU, Docker daemon or EDA is invoked by these contract tests."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from local_bench.model import EvaluationResult, load_candidate, load_task
from local_bench.program_bridge import initial_program_text
from local_bench.search_api import CandidateProposal, ProgramEvaluationHarness, TaskEvaluationOptions
from scripts import run_local_task_smoke as smoke


TASK = smoke.ROOT / "local_tasks/placement_initialization"
SPEC = load_task(TASK)
INITIAL = initial_program_text(SPEC, load_candidate(SPEC.initial_candidate))
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def runtime(tmp_path):
    batch = tmp_path / "runs"
    batch.mkdir()
    def contained(path):
        resolved = Path(path).resolve()
        resolved.relative_to(batch)
        return resolved
    return SimpleNamespace(source=smoke.ROOT.parent, framework=smoke.ROOT, batch=batch,
                           source_commit="frozen", image=IMAGE, cpus=16, memory_gb=48,
                           share_gpu=True, min_gpu_free_mb=2048, contained=contained)


class FakeProvider:
    responses = [INITIAL] * 5
    def __init__(self, root, binary, timeout):
        self.calls = 0
        self.completed_calls = 0
        self.messages = []
    def draw_sample(self, messages):
        self.messages.append(messages)
        value = self.responses[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        self.completed_calls += 1
        return value


class FakeTransport:
    failures = {}
    def __init__(self, runtime):
        self.physical_calls = 0
        self.receipts = []
        self.calls = []
        self.attempts = []
    def __call__(self, task, candidate, source, output, **kwargs):
        self.physical_calls += 1
        self.calls.append((task, candidate, source, output, kwargs))
        self.attempts.append({"output_dir": str(output), "evaluator_dispatch_attempted": True,
                              "native_flow_started": True})
        failure = self.failures.get(self.physical_calls)
        if isinstance(failure, Exception):
            raise failure
        output.mkdir(parents=True, exist_ok=False)
        (output / "run").mkdir()
        result = EvaluationResult(
            {"runs_successfully": 0 if failure else 1, "combined_score": 0.0 if failure else 1.0},
            {"run_dir": str(output / "run")})
        smoke.write_json(output / "evaluation_result.json", result.to_dict())
        self.receipts.append({"path": str(output / "evaluation_result.json")})
        return result


def invoke(runtime, **kwargs):
    baseline = runtime.batch / "baseline"
    baseline.mkdir(exist_ok=True)
    return smoke.run_smoke(runtime, task=TASK, output=runtime.batch / "smoke",
                           baseline_run=baseline, provider_factory=FakeProvider,
                           transport_factory=FakeTransport, **kwargs)


def test_bundle_profile_and_case_reach_actual_model_prompt(runtime, monkeypatch):
    bundle = runtime.batch / "input"
    bundle.mkdir()
    run = bundle / "run"
    run.mkdir()
    config = bundle / "config.yaml"
    config.write_text("design: swerv_wrapper\ntype: lol\npdk: ng45\nthreads: 4\n")
    profile_id = "fixture_swerv_selected_profile"
    monkeypatch.setattr(smoke, "validate_benchmark_input", lambda *a, **kw: {
        "run_dir": str(run), "config": str(config), "benchmark_id": profile_id})
    summary = smoke.run_smoke(runtime, task=TASK, output=runtime.batch / "out",
                              benchmark_input=bundle, transport_factory=FakeTransport,
                              provider_factory=FakeProvider)
    assert summary["state"] == "completed"
    first = json.loads((runtime.batch / "out/samples.jsonl").read_text().splitlines()[0])
    messages = "\n".join(row["content"] for row in first["messages"])
    assert "Design: swerv_wrapper;" in messages
    assert profile_id not in messages and str(config) not in messages
    contract = json.loads((runtime.batch / "out/run_config.json").read_text())
    assert contract["input_validation"]["benchmark_id"] == profile_id
    assert contract["base_config"] == str(config)


@pytest.mark.parametrize("rounds", [2, 5, 50])
def test_reference_is_separate_and_duplicate_outputs_get_real_evaluations(runtime, monkeypatch, rounds):
    monkeypatch.setattr(FakeProvider, "responses", [INITIAL] * rounds)
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "completed"
    assert summary["reference_evaluations"] == 1
    assert summary["generated_candidate_budget"] == rounds
    assert summary["reference_excluded_from_budget"] is True
    assert summary["model_requests"] == summary["generated_candidates"] == summary["candidate_finals"] == rounds
    assert summary["candidate_physical_attempts"] == summary["candidate_evaluation_results"] == rounds
    assert summary["successful_candidate_evaluations"] == rounds
    assert summary["cache_hits"] == 0
    assert summary["duplicate_candidate_outputs"] == rounds - 1
    result = json.loads((runtime.batch / "smoke/search_result.json").read_text())
    assert len(result["history"]) == rounds
    assert all(row["status"] == "accepted" for row in result["history"])
    assert (runtime.batch / "smoke/reference/qualification.json").is_file()
    rows = [json.loads(row) for row in (runtime.batch / "smoke/samples.jsonl").read_text().splitlines()]
    assert len(rows) == rounds
    assert all(row["provider_attempts"] == 1 for row in rows)
    assert "Qualified initial reference" in rows[0]["messages"][1]["content"]
    assert "Generate candidate {} of {}".format(rounds, rounds) in json.dumps(rows[-1]["messages"])
    contract = json.loads((runtime.batch / "smoke/run_config.json").read_text())
    assert contract["generated_candidate_budget"] == rounds
    assert contract["reference_excluded_from_budget"] is True
    assert contract["model"] == "gpt-5.6-sol"
    assert len(summary["physical_receipts"]) == rounds + 1


def test_best_retention_hook_runs_only_after_terminal_accounting(runtime, monkeypatch):
    calls = []
    def finish(root, mode):
        status = json.loads((root / "status.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        calls.append((mode, status["state"], summary["candidate_finals"]))
    monkeypatch.setattr(smoke, "finish_retention", finish)
    summary = invoke(runtime, retention="best")
    assert calls == [("best", "completed", 5)]
    assert summary["successful_candidate_evaluations"] == 5
    assert json.loads((runtime.batch / "smoke/run_config.json").read_text())["retention"] == "best"


def test_invalid_retention_refuses_before_any_output(runtime):
    with pytest.raises(ValueError, match="Retention mode"):
        invoke(runtime, retention="delete-all")
    assert not (runtime.batch / "smoke").exists()


@pytest.mark.parametrize("rounds", [2, 5, 50])
def test_failed_reference_prevents_all_model_calls(runtime, monkeypatch, rounds):
    monkeypatch.setattr(FakeTransport, "failures", {1: True})
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "reference_failed"
    assert summary["generated_candidates"] == summary["candidate_finals"] == 0
    assert summary["reference_evaluations"] == 1
    assert summary["candidate_physical_attempts"] == summary["candidate_evaluation_results"] == 0
    assert len(summary["physical_receipts"]) == 1
    assert summary["physical_receipts"][0]["path"].endswith("/reference/result/evaluation_result.json")
    assert not (runtime.batch / "smoke/model").exists()


def test_reference_transport_failure_without_receipt_is_not_a_candidate(runtime, monkeypatch):
    monkeypatch.setattr(FakeTransport, "failures", {
        1: smoke.EvaluationInfrastructureError("Reference did not produce a result")})
    summary = invoke(runtime)
    assert summary["state"] == "reference_failed"
    assert summary["model_requests"] == summary["candidate_finals"] == 0
    assert summary["candidate_physical_attempts"] == summary["candidate_evaluation_results"] == 0
    assert summary["physical_receipts"] == []


@pytest.mark.parametrize("rounds", [2, 5, 50])
def test_rejected_program_consumes_one_round_without_repair_retry(runtime, monkeypatch, rounds):
    monkeypatch.setattr(FakeProvider, "responses", ["import os\n"] + [INITIAL] * (rounds - 1))
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "completed"
    assert summary["model_requests"] == summary["candidate_finals"] == rounds
    assert summary["rejected_candidates"] == 1
    assert summary["candidate_physical_attempts"] == rounds - 1
    assert summary["successful_candidate_evaluations"] == rounds - 1


@pytest.mark.parametrize("rounds", [2, 5, 50])
@pytest.mark.parametrize("failed_candidate", ["first", "last"])
def test_candidate_native_failure_remains_failure_feedback_and_counts(runtime, monkeypatch, rounds, failed_candidate):
    monkeypatch.setattr(FakeProvider, "responses", [INITIAL] * rounds)
    monkeypatch.setattr(FakeTransport, "failures", {2 if failed_candidate == "first" else rounds + 1: True})
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "completed"
    assert summary["failed_candidate_evaluations"] == 1
    assert summary["successful_candidate_evaluations"] == rounds - 1
    assert summary["candidate_finals"] == summary["candidate_physical_attempts"] == rounds
    assert summary["model_requests"] == rounds
    assert summary["cache_hits"] == 0


@pytest.mark.parametrize("rounds", [2, 5, 50])
def test_transport_failure_stops_before_next_model_request(runtime, monkeypatch, rounds):
    monkeypatch.setattr(FakeTransport, "failures", {3: smoke.EvaluationInfrastructureError("No Docker result")})
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "needs_attention"
    assert summary["generated_candidates"] == summary["candidate_finals"] == 2
    assert summary["successful_candidate_evaluations"] == 1
    assert summary["model_requests"] == summary["candidate_evaluator_dispatches"] == 2
    assert summary["error"]["type"] == "EvaluationInfrastructureError"


@pytest.mark.parametrize("rounds", [2, 5, 50])
def test_model_failure_is_not_a_generated_candidate(runtime, monkeypatch, rounds):
    monkeypatch.setattr(FakeProvider, "responses", [smoke.EvaluationInfrastructureError("Auth unavailable")])
    summary = invoke(runtime, rounds=rounds)
    assert summary["state"] == "needs_attention"
    assert summary["model_requests"] == 1
    assert summary["generated_candidates"] == summary["candidate_finals"] == 0
    assert summary["reference_qualified"] is True
    assert summary["candidate_evaluation_results"] == 0


def test_existing_output_is_never_replayed(runtime):
    invoke(runtime)
    before = (runtime.batch / "smoke/summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(runtime)
    assert (runtime.batch / "smoke/summary.json").read_bytes() == before


def test_qualification_only_never_constructs_provider_or_consumes_budget(runtime):
    baseline = runtime.batch / "baseline"
    baseline.mkdir()
    provider = mock.Mock(side_effect=AssertionError("No model during qualification"))
    summary = smoke.run_smoke(runtime, task=TASK, output=runtime.batch / "qualification",
                              baseline_run=baseline, rounds=2, qualification_only=True,
                              provider_factory=provider, transport_factory=FakeTransport)
    provider.assert_not_called()
    assert summary["state"] == "qualified"
    assert summary["reference_evaluations"] == 1
    assert summary["candidate_finals"] == summary["generated_candidates"] == 0
    assert summary["candidate_budget_consumed"] == summary["api_request_count"] == 0
    assert summary["generated_candidate_budget"] == 2
    assert not (runtime.batch / "qualification/evolution").exists()


@pytest.mark.parametrize("task", ["global_routing_ripup_policy",
                                  "global_routing_maze_cost", "global_routing_iteration_controller"])
@pytest.mark.parametrize("reachable", [None, 0, 1])
def test_routing_reference_requires_actual_policy_reachability(task, reachable):
    spec = load_task(smoke.ROOT / "local_tasks" / task)
    result = EvaluationResult({"runs_successfully": 1, "combined_score": 1,
                               "grt_complete": reachable}, {})
    assert smoke._qualified_reference(spec, result) is (reachable == 1)


def test_output_cannot_be_inside_frozen_input(runtime):
    baseline = runtime.batch / "baseline"
    baseline.mkdir()
    with pytest.raises(ValueError, match="separate"):
        smoke.run_smoke(runtime, task=TASK, output=baseline / "new-smoke", baseline_run=baseline)
    assert not (baseline / "new-smoke").exists()


@pytest.mark.parametrize("options", [{"rounds": value} for value in (0, 4, 6, 10, 49, 51, 100, 5.0, 50.0, True)]
                         + [{"model": "other"}, {"timeout": 0}])
def test_smoke_budget_model_and_positive_timeout_are_explicit(runtime, options):
    with pytest.raises(ValueError):
        invoke(runtime, **options)
    assert not (runtime.batch / "smoke").exists()


def test_fifty_rounds_preserve_model_and_physical_timeouts(runtime):
    timeouts = []
    class Provider(FakeProvider):
        responses = [INITIAL] * 50
        def __init__(self, root, binary, timeout):
            assert timeout == 1800
            super().__init__(root, binary, timeout)
    class Transport(FakeTransport):
        def __call__(self, *args, **kwargs):
            timeouts.append(kwargs["timeout_seconds"])
            return super().__call__(*args, **kwargs)
    baseline = runtime.batch / "baseline"
    baseline.mkdir()
    summary = smoke.run_smoke(runtime, task=TASK, output=runtime.batch / "smoke",
                              baseline_run=baseline, rounds=50,
                              provider_factory=Provider, transport_factory=Transport)
    assert summary["state"] == "completed"
    assert timeouts == [21600] * 51  # One separate reference, then 50 candidates.


def cli_arguments(runtime):
    return ["--source", str(runtime.source), "--batch-root", str(runtime.batch),
            "--output", str(runtime.batch / "cli-run"), "--task", str(TASK),
            "--image", IMAGE, "--gpu", "1"]


@pytest.mark.parametrize("requested", [None, 2, 5, 50])
def test_cli_keeps_five_default_and_explicitly_allows_two_or_fifty(runtime, monkeypatch, requested):
    make_runtime = mock.Mock(return_value=runtime)
    run = mock.Mock(return_value={"state": "completed"})
    monkeypatch.setattr(smoke, "LocalSmokeRuntime", make_runtime)
    monkeypatch.setattr(smoke, "run_smoke", run)
    args = cli_arguments(runtime)
    if requested is not None:
        args += ["--rounds", str(requested)]
    assert smoke.main(args) == 0
    make_runtime.assert_called_once()
    run.assert_called_once()
    assert run.call_args.args == (runtime,)
    assert run.call_args.kwargs["rounds"] == (5 if requested is None else requested)
    assert run.call_args.kwargs["model"] == "gpt-5.6-sol"
    assert run.call_args.kwargs["codex_timeout"] == 1800
    assert run.call_args.kwargs["timeout"] == 21600


@pytest.mark.parametrize("requested", ["0", "4", "6", "10", "49", "51", "100", "50.0"])
def test_cli_rejects_unapproved_round_counts_before_runtime(runtime, monkeypatch, requested):
    make_runtime = mock.Mock(side_effect=AssertionError("No runtime may be created"))
    monkeypatch.setattr(smoke, "LocalSmokeRuntime", make_runtime)
    with pytest.raises(SystemExit) as error:
        smoke.main(cli_arguments(runtime) + ["--rounds", requested])
    assert error.value.code == 2
    make_runtime.assert_not_called()


def test_full_harness_callback_preserves_verifier_and_disable_cache_is_opt_in(tmp_path):
    callback = mock.Mock(return_value=EvaluationResult({"runs_successfully": 1, "combined_score": 1.2}))
    options = TaskEvaluationOptions(TASK, tmp_path / "harness", open3dbench_root=smoke.ROOT.parent)
    harness = ProgramEvaluationHarness(options, evaluation_transport=callback, cache_results=False)
    one = harness.evaluate_proposal(CandidateProposal("one", INITIAL))
    two = harness.evaluate_proposal(CandidateProposal("two", INITIAL))
    bad = harness.evaluate_proposal(CandidateProposal("bad", "import os\n"))
    assert one.score == two.score == 1.2
    assert one.status == two.status == "accepted"
    assert bad.status == "rejected"
    assert callback.call_count == 2
    assert callback.call_args.kwargs["dry_run"] is False
    default = ProgramEvaluationHarness(TaskEvaluationOptions(
        TASK, tmp_path / "default", evaluation_mode="verify", open3dbench_root=smoke.ROOT.parent))
    default.evaluate_proposal(CandidateProposal("one", INITIAL))
    assert default.evaluate_proposal(CandidateProposal("two", INITIAL)).status == "cached"


def test_text_only_command_ignores_user_config_and_has_no_terminal_or_native_tools(tmp_path):
    command = smoke.blackbox_command("codex", tmp_path, tmp_path / "final.txt")
    for flag in ("--ignore-user-config", "--ignore-rules", "--json"):
        assert flag in command
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    config = [command[i + 1] for i, value in enumerate(command) if value == "--config"]
    assert "features.shell_tool=false" in config
    assert "features.unified_exec=false" in config
    assert "features.code_mode_host=false" in config
    assert "tools.view_image=false" in config
    assert not any(value.startswith("mcp_servers.") for value in config)
    assert "permissions.agentic_container_only.network.enabled=false" in config
    assert not any("dangerously" in value for value in command)


def test_local_docker_mounts_other_results_readonly_and_only_current_parent_writable(runtime, tmp_path):
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    local.write_root = runtime.batch / "task/round-01"
    local.write_root.mkdir(parents=True)
    local.image_preparation = None
    local.gpus = [0]
    command = local.docker(["python3", "-m", "local_bench"], 0, local.write_root / "scratch")
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "-v"]
    assert f"{runtime.batch}:{runtime.batch}:ro" in mounts
    assert f"{runtime.batch}:{runtime.batch}" not in mounts
    assert f"{runtime.source}:{runtime.source}:ro" in mounts
    assert f"{local.write_root}:{local.write_root}" in mounts
    # A managed checkout can live under ~/.codex/worktrees. Inspect exactly
    # what is mounted; the parent name is not an exposed credentials directory.
    sources = [Path(value.split(":", 1)[0]).resolve() for value in mounts]
    forbidden = [Path.home() / ".codex", Path.home() / ".ssh", Path("/var/run/docker.sock")]
    assert all(not (src == secret or src in secret.parents or secret in src.parents
                   and "worktrees" not in src.parts) for src in sources for secret in forbidden)
    assert not any("docker.sock" in value for value in command)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cpus") + 1] == "16"
    assert command[command.index("--memory") + 1] == "48g"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "OPEN3DBENCH_IMAGE_ID=" + IMAGE in command


def test_candidate_image_receipt_is_pinned_and_passed_inside_private_output(runtime):
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    local.write_root = runtime.batch / "task/round-01"
    local.write_root.mkdir(parents=True)
    receipt = local.write_root / "receipt.json"
    receipt.write_text("{}")
    candidate_image = "sha256:" + "b" * 64
    prepare = mock.Mock(return_value=(candidate_image, receipt))
    local.image_preparation = prepare
    command = local.docker(["python3"], 0, local.write_root / "scratch")
    prepare.assert_called_once_with(0)
    assert candidate_image in command
    assert IMAGE not in command
    assert "OPEN3DBENCH_IMAGE_ID=" + candidate_image in command
    assert "OPEN3D_LOCAL_OPENROAD_PREBUILT_RECEIPT=" + str(receipt) in command


def test_transport_invokes_task_cli_and_returns_failed_result_without_host_eda(runtime):
    root = runtime.batch / "eval"
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    def execute(command, output, log, contract):
        assert command[:4] == ["/usr/bin/python3", "-m", "local_bench", "evaluate"]
        assert "--discard-runtime" not in command
        assert "--dry-run" not in command
        assert runtime.write_root == root
        output.mkdir()
        smoke.write_json(output / "evaluation_result.json", {
            "metrics": {"runs_successfully": 0, "combined_score": 0},
            "artifacts": {"error_type": "RuntimeError", "error_message": "Candidate did not converge"}})
        return 1
    runtime.execute = execute
    transport = smoke.DockerEvaluationTransport(runtime)
    result = transport(TASK, candidate, runtime.source, root / "result")
    assert result.metrics["runs_successfully"] == 0
    assert len(transport.receipts) == 1
    assert runtime.write_root is None


@pytest.mark.parametrize("code, payload", [
    (137, None), (0, {"metrics": {"runs_successfully": 0}, "artifacts": {}}),
    (1, {"metrics": {"runs_successfully": 0}, "artifacts": {"error_type": "FileNotFoundError"}}),
    (0, {"metrics": {"runs_successfully": 1, "combined_score": float("nan")}, "artifacts": {}}),
])
def test_transport_fails_closed_on_missing_or_invalid_or_environment_result(runtime, code, payload):
    candidate = runtime.batch / "eval/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    def execute(command, output, log, contract):
        if payload is not None:
            output.mkdir()
            smoke.write_json(output / "evaluation_result.json", payload)
        return code
    runtime.execute = execute
    with pytest.raises(smoke.EvaluationInfrastructureError):
        smoke.DockerEvaluationTransport(runtime)(TASK, candidate, runtime.source, candidate.parent / "result")
    assert runtime.write_root is None


def test_transport_does_not_overwrite_runtime_failure_receipt(runtime):
    candidate = runtime.batch / "eval/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    ledger = candidate.parent / "result.attempt.json"
    def execute(command, output, log, contract):
        smoke.write_json(ledger, {"state": "infrastructure_failed", "contract": contract, "gpu": 0,
                                 "error": "Image compilation unavailable", "queued_at": 123})
        raise RuntimeError("Image compilation unavailable")
    runtime.execute = execute
    with pytest.raises(smoke.EvaluationInfrastructureError):
        smoke.DockerEvaluationTransport(runtime)(TASK, candidate, runtime.source, candidate.parent / "result")
    record = json.loads(ledger.read_text())
    assert record["state"] == "infrastructure_failed"
    assert record["gpu"] == 0
    assert record["contract"]["task"] == SPEC.task_id
    assert "Image compilation unavailable" in record["error"]
    assert record["queued_at"] == 123
    assert "finished_at" not in record


def test_real_text_provider_requires_completed_turn_and_no_host_tool_calls(tmp_path, monkeypatch):
    def launch(command, **kwargs):
        final = Path(command[command.index("--output-last-message") + 1])
        def communicate(prompt, timeout):
            kwargs["stdout"].write('{"type":"turn.completed"}\n')
            final.write_text(INITIAL)
        return SimpleNamespace(communicate=communicate, returncode=0)
    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    provider = smoke.TextOnlyCodex(tmp_path / "model", "codex", 30)
    assert provider.draw_sample("prompt") == INITIAL
    assert provider.calls == provider.completed_calls == 1
    assert json.loads((tmp_path / "model/sample-01/receipt.json").read_text())["success"]


@pytest.mark.parametrize("event, text", [
    ('{"type":"turn.started"}\n', INITIAL),
    ('{"type":"turn.completed"}\n', ""),
    ('{"type":"turn.completed"}\n{"type":"item.completed","item":{"type":"command_execution"}}\n', INITIAL),
])
def test_text_provider_rejects_incomplete_empty_or_tool_using_turn(tmp_path, monkeypatch, event, text):
    def launch(command, **kwargs):
        final = Path(command[command.index("--output-last-message") + 1])
        def communicate(prompt, timeout):
            kwargs["stdout"].write(event)
            final.write_text(text)
        return SimpleNamespace(communicate=communicate, returncode=0)
    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    provider = smoke.TextOnlyCodex(tmp_path / "model", "codex", 30)
    with pytest.raises(smoke.EvaluationInfrastructureError):
        provider.draw_sample("prompt")
    assert provider.calls == 1
    assert provider.completed_calls == 0


def test_text_provider_timeout_terminates_only_its_own_process_group(tmp_path, monkeypatch):
    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        def communicate(prompt, timeout):
            raise subprocess.TimeoutExpired(command, timeout)
        return SimpleNamespace(communicate=communicate, wait=lambda timeout: None,
                               returncode=-15, pid=918273)
    kill = mock.Mock()
    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    monkeypatch.setattr(smoke.os, "killpg", kill)
    provider = smoke.TextOnlyCodex(tmp_path / "model", "codex", 30)
    with pytest.raises(smoke.EvaluationInfrastructureError):
        provider.draw_sample("prompt")
    kill.assert_called_once_with(918273, smoke.signal.SIGTERM)
    assert provider.completed_calls == 0


def test_routing_reference_bootstrap_is_explicit_and_not_in_generated_budget(runtime, monkeypatch):
    routing = load_task(smoke.ROOT / "local_tasks/global_routing_ripup_policy")
    # Keep the native routing kind but avoid reading its C++ source in this
    # orchestration fixture; real adapters have separate source-policy tests.
    monkeypatch.setattr(smoke, "load_task", lambda path: routing)
    monkeypatch.setattr(smoke, "materialize_program", lambda *args: load_candidate(SPEC.initial_candidate))
    seen = []
    class RefOnly(FakeTransport):
        def __call__(self, *args, **kwargs):
            seen.append(kwargs)
            result = super().__call__(*args, **kwargs)
            result.metrics["grt_complete"] = 1
            return result
    monkeypatch.setattr(smoke, "build_problem_description", lambda *args: "fixture")
    monkeypatch.setattr(smoke, "ProgramEvaluationHarness", mock.Mock(side_effect=RuntimeError("Stop after reference")))
    baseline = runtime.batch / "baseline"
    baseline.mkdir()
    summary = smoke.run_smoke(runtime, task=routing.root, output=runtime.batch / "out", baseline_run=baseline,
                              transport_factory=RefOnly, provider_factory=FakeProvider)
    assert seen[0]["create_routing_reference"] is True
    assert summary["reference_evaluations"] == 1
    assert summary["model_requests"] == summary["candidate_finals"] == 0
    assert summary["reference_qualified"] is True
    assert summary["state"] == "needs_attention"


@pytest.mark.parametrize('classic_binding', [False, True])
def test_routing_image_build_and_evaluation_share_one_runtime_slot(runtime, monkeypatch, classic_binding):
    from scripts import build_routing_candidate_image as builder
    local = object.__new__(smoke.LocalSmokeRuntime)
    local.__dict__.update(runtime.__dict__)
    local.gpus = [0]
    routing = load_task(smoke.ROOT / "local_tasks/global_routing_ripup_policy")
    monkeypatch.setattr(smoke, "load_task", lambda path: routing)
    candidate = runtime.batch / "reference/candidate"
    candidate.mkdir(parents=True)
    smoke.write_json(candidate / "candidate.json", load_candidate(SPEC.initial_candidate))
    monkeypatch.setattr(smoke, "verify_candidate", lambda *args: {"patch": {"paths": ["OpenROAD-3D/openroad/policy.cpp"]}})
    events = []
    held = False
    @contextmanager
    def slot():
        nonlocal held
        assert not held
        held = True
        events.append("lock")
        try:
            yield 0
        finally:
            held = False
            events.append("unlock")
    local.slot = slot
    built_image = "sha256:" + "c" * 64
    base_image = "sha256:" + "d" * 64 if classic_binding else IMAGE
    binding = {'path': '/trusted/proof.json', 'sha256': 'e' * 64}
    if classic_binding:
        local.routing_build_identity = lambda: (base_image, binding)
    def build(**kwargs):
        assert held
        assert kwargs["base_image"] == base_image
        assert kwargs.get('base_runtime_equivalence') == (binding if classic_binding else None)
        assert kwargs["candidate_dir"] == candidate
        assert kwargs["threads"] == 8
        output = kwargs["output_dir"]
        output.mkdir()
        receipt = output / "receipt.json"
        receipt.write_text("{}")
        events.append("build-image")
        return {"image": built_image, "receipt": str(receipt)}
    monkeypatch.setattr(builder, "build_candidate_image", build)
    def launch(command, **kwargs):
        assert held
        assert "--create-routing-reference" in command
        assert built_image in command
        assert "OPEN3DBENCH_IMAGE_ID=" + built_image in command
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        smoke.write_json(output / "evaluation_result.json", {
            "metrics": {"runs_successfully": 1, "combined_score": 1}, "artifacts": {}})
        events.append("evaluate")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(smoke.subprocess, "run", launch)
    transport = smoke.DockerEvaluationTransport(local)
    result = transport(routing.root, candidate, runtime.source, candidate.parent / "result",
                       create_routing_reference=True)
    assert result.metrics["runs_successfully"] == 1
    assert events == ["lock", "build-image", "evaluate", "unlock"]
    assert (candidate.parent / "result.attempt.json").is_file()


def test_reference_compilation_failure_stops_without_model_budget(runtime, monkeypatch):
    monkeypatch.setattr(FakeTransport, "failures", {
        1: smoke.EvaluationInfrastructureError("Candidate image compilation failed")})
    summary = invoke(runtime)
    assert summary["state"] == "reference_failed"
    assert summary["reference_evaluations"] == 1
    assert not summary["reference_qualified"]
    assert summary["model_requests"] == summary["candidate_finals"] == 0
