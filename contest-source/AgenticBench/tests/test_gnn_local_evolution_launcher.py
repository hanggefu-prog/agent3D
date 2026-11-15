"""Launcher-only checks: no model request, Docker process or native EDA."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_bench.model import EvaluationResult
from local_bench.search_api import SearchContext, SearchProblem
from scripts import run_gnn_local_evolution as launch


def token_file(path):
    path.write_text("launcher-unit-test-secret-token\n")
    path.chmod(0o600)
    return path


def test_provider_config_uses_single_streamed_high_reasoning_request():
    config = launch._provider_config("http://127.0.0.1:39123/v1")
    assert config.model == "deepseek-flash"
    assert config.stream and config.max_attempts == 1
    assert config.reasoning_effort == "high" and config.max_tokens == 32768
    assert config.temperature is config.top_p is None
    for url in ("http://example.com/v1", "https://token@example.com/v1", "https://example.com/v1?token=abc"):
        with pytest.raises(ValueError):
            launch._provider_config(url)


def test_credential_mode_and_no_secret_in_five_audit_records(tmp_path, monkeypatch):
    key = token_file(tmp_path / "credential")
    key.chmod(0o644)
    with pytest.raises(ValueError, match="mode-600"):
        launch._credential_file(key)
    key.chmod(0o600)
    previous = "preexisting-environment-value"
    monkeypatch.setenv(launch.TOKEN_ENV, previous)
    calls = []
    class Client:
        def __init__(self, config):
            assert os.environ[config.api_key_env] == key.read_text().strip()
        def draw_sample(self, messages):
            calls.append(messages)
            return "def replacement():\n    return 1\n"
        def close(self):
            pass
    provider = launch.AuditedProvider(tmp_path / "model", launch._provider_config("https://example.com/v1"),
                                     key, lambda *args, **kwargs: None, factory=Client)
    try:
        for _ in range(5):
            provider.generate("task contract", "parent program")
        with pytest.raises(RuntimeError, match="five-request"):
            provider.generate("task", "extra request")
        assert len(calls) == provider.calls == provider.completed == 5
        for path in (tmp_path / "model").rglob("*"):
            if path.is_file():
                assert key.read_text().strip() not in path.read_text()
        assert json.loads((tmp_path / "model/call-005/receipt.json").read_text())["state"] == "completed"
    finally:
        provider.close()
    assert os.environ[launch.TOKEN_ENV] == previous


def test_api_failure_consumes_one_slot_without_retry_and_redacts_error(tmp_path):
    key = token_file(tmp_path / "credential")
    calls = []
    class Client:
        def __init__(self, config):
            pass
        def draw_sample(self, messages):
            calls.append(messages)
            raise RuntimeError("simulated failure " + key.read_text().strip())
        def close(self):
            pass
    provider = launch.AuditedProvider(tmp_path / "model", launch._provider_config("https://example.com/v1"),
                                     key, lambda *args, **kwargs: None, factory=Client)
    try:
        with pytest.raises(launch.EvaluationInfrastructureError, match="slot remains consumed") as error:
            provider.generate("task", "parent")
        assert len(calls) == provider.calls == 1 and provider.completed == 0
        assert key.read_text().strip() not in str(error.value)
        receipt = json.loads((tmp_path / "model/call-001/receipt.json").read_text())
        assert receipt["state"] == "failed" and "[REDACTED]" in receipt["error_message"]
    finally:
        provider.close()


def test_native_openevolve_starts_with_real_separate_reference_metrics():
    pytest.importorskip("openevolve")
    requests = []
    backend = launch.QualifiedOpenEvolve(launch.search_config(), {"combined_score": 1., "runs_successfully": 1.},
        generate=lambda system, user: requests.append((system, user)) or "```python\ndef loss():\n    return 1\n```" )
    problem = SearchProblem("test", "fixed task", "def loss():\n    return 0\n")
    try:
        proposal = backend.ask(SearchContext(problem, 0, 5, (), None))
        assert len(proposal) == len(requests) == 1
        initial = backend.database.programs["qualified-initial"]
        assert initial.metrics["combined_score"] == 1.
        assert initial.metadata["origin"] == "fresh_reference"
        assert initial.metadata["candidate_budget_consumed"] == 0
        assert initial.metadata["island"] == 0
        assert "no measured score" not in requests[0][1]
    finally:
        backend.close()


def test_failed_fresh_reference_never_constructs_provider(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    batch = tmp_path / "batch"
    batch.mkdir()
    baseline = batch / "prepare/run"
    baseline.mkdir(parents=True)
    key = token_file(tmp_path / "host-only-credential")
    runtime = SimpleNamespace(source=launch.FRAMEWORK.parent, framework=launch.FRAMEWORK,
                              batch=batch, source_commit="fixed", image="sha256:" + "1" * 64,
                              cpus=8, memory_gb=32, gpus=[0])
    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch.resolve())
        return path
    runtime.contained = contained
    monkeypatch.setattr(launch, "_verify_preparation", lambda *args: baseline)
    monkeypatch.setattr(launch, "AuditedProvider", lambda *args, **kwargs: pytest.fail("provider constructed"))
    class Transport:
        attempts = []
        def __call__(self, *args, **kwargs):
            return EvaluationResult({"runs_successfully": 0.}, {})
    monkeypatch.setattr(launch, "DockerEvaluationTransport", lambda _: Transport())
    with pytest.raises(launch.EvaluationInfrastructureError, match="did not qualify"):
        launch.run(runtime, task=launch.TASKS[0], output=batch / "search", baseline_run=baseline,
                   key_file=key, base_url="http://127.0.0.1:39123/v1")
    status = json.loads((batch / "search/status.json").read_text())
    assert status["state"] == "failed" and status["model_requests"] == 0
    assert status["candidate_budget_consumed"] == 0 and not status["reference_qualified"]
    with pytest.raises(FileExistsError):
        launch._new_output(runtime, batch / "search")


def test_only_five_slots_are_accepted_by_cli():
    parser = launch.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--rounds", "6"])


def test_prepare_dispatch_uses_an_isolated_writable_compute_checkout(tmp_path, monkeypatch):
    """Stop at dispatch; do not invent a native pipeline result or artifacts."""
    from local_bench import workspace as workspaces
    batch = tmp_path / "batch"
    batch.mkdir()
    output = batch / "prepare"
    clones, dispatched = [], []
    class CapturedDispatch(Exception):
        pass
    def clone(source, destination):
        clones.append((source, destination))
        destination.mkdir()
        return destination
    runtime = SimpleNamespace(source=launch.FRAMEWORK.parent, framework=launch.FRAMEWORK,
                              batch=batch, source_commit="fixed", image="sha256:" + "1" * 64)
    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch.resolve())
        return path
    runtime.contained = contained
    def execute(command, *args):
        dispatched.append(command)
        assert runtime.write_root == output
        raise CapturedDispatch()
    runtime.execute = execute
    monkeypatch.setattr(workspaces, "isolated_clone", clone)
    with pytest.raises(CapturedDispatch):
        launch.prepare(runtime, output=output)
    assert clones == [(runtime.source, output / "workspace")]
    command = dispatched[0]
    assert Path(command[command.index("--repo") + 1]) == output / "workspace"
    assert Path(command[command.index("--repo") + 1]).is_relative_to(output)
    assert command[-2:] == ["--to", "partition"]
    assert not (output / "prepare.json").exists()


def test_five_round_full_harness_keeps_reference_separate_and_disables_cache(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    batch = tmp_path / "batch"
    batch.mkdir()
    baseline = batch / "prepare/run"
    baseline.mkdir(parents=True)
    key = token_file(tmp_path / "host-only-credential")
    runtime = SimpleNamespace(source=launch.FRAMEWORK.parent, framework=launch.FRAMEWORK,
                              batch=batch, source_commit="fixed", image="sha256:" + "1" * 64,
                              cpus=8, memory_gb=32, gpus=[0])
    def contained(path):
        path = Path(path).resolve()
        path.relative_to(batch.resolve())
        return path
    runtime.contained = contained
    monkeypatch.setattr(launch, "_verify_preparation", lambda *args: baseline)
    monkeypatch.setattr(launch, "_assert_source", lambda *args: None)
    spec = launch.load_task(launch.FRAMEWORK / "local_tasks" / launch.TASKS[0])
    original_provider = launch.AuditedProvider
    class Client:
        def __init__(self, config):
            pass
        def draw_sample(self, messages):
            return "```python\n" + spec.program_template.read_text() + "\n```"
        def close(self):
            pass
    monkeypatch.setattr(launch, "AuditedProvider", lambda *args: original_provider(*args, factory=Client))
    calls = []
    class Transport:
        attempts = []
        def __call__(self, task, candidate, source, output, **kwargs):
            calls.append(kwargs)
            output.mkdir(parents=True)
            run = output / "run"
            run.mkdir()
            result = EvaluationResult({"runs_successfully": 1., "combined_score": 1.}, {"run_dir": str(run)})
            (output / "evaluation_result.json").write_text(json.dumps(result.to_dict()))
            return result
    monkeypatch.setattr(launch, "DockerEvaluationTransport", lambda _: Transport())
    result = launch.run(runtime, task=launch.TASKS[0], output=batch / "search", baseline_run=baseline,
                        key_file=key, base_url="http://127.0.0.1:39123/v1")
    assert result["state"] == "completed"
    assert result["model_requests"] == result["candidate_budget_consumed"] == result["candidate_finals"] == 5
    assert len(calls) == 6  # One FRESH evaluation plus five, even for identical model output.
    assert calls[0].get("reference_run") is None
    assert all(call["reference_run"] == batch / "search/reference/result/run" for call in calls[1:])
    assert len(list((batch / "search/evolution").glob("feedback-*.json"))) == 5


def test_preparation_manifest_checks_use_framework_artifacts_without_checkout_imports(tmp_path):
    import ast
    import hashlib
    source = Path(launch.__file__).read_text()
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(name == 'open3dbench' or name.startswith('open3dbench.') for name in imports if name)
    artifact = tmp_path / 'artifact.json'
    artifact.write_text('{}')
    manifest = tmp_path / 'stage_manifest.json'
    value = {'schema_version': 1, 'stage': 'input', 'status': 'success', 'signature': {},
             'artifacts': {'physical': {'path': artifact.name, 'bytes': artifact.stat().st_size,
                                      'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
    manifest.write_text(json.dumps(value))
    assert launch._validate_preparation_stage(manifest, 'input') == value
    del value['signature']
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='signature'):
        launch._validate_preparation_stage(manifest, 'input')
    value['signature'] = {}
    manifest.write_text(json.dumps(value))
    artifact.write_text('changed')
    with pytest.raises(ValueError, match='size changed|hash changed'):
        launch._validate_preparation_stage(manifest, 'input')
