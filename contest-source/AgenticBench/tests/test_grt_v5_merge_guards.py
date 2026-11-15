"""Production-v5 merge invariants, using local synthetic boundaries only.

These are controller/client regressions, never native EDA qualification. The
Claude subprocess is replaced by a local fake; the queue/worker/round state
remain real for the uncertain-termination case.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grt_bench.common import read, write
from grt_bench.contract import OBJECTIVES, validate
from grt_bench.runtime import UncertainExecution
from grt_bench.queue import Queue
from global_bench import grt_campaign as campaign
from global_bench import workspace_claude as claude
from global_bench.grt_development import export_submission
from global_bench.grt_round_tools import RoundRuntime
import test_grt_optimization as fixtures


SESSION = "00000000-0000-0000-0000-000000000001"
MODEL = "claude-fable-5"


@pytest.fixture
def host(monkeypatch):
    f = fixtures.OptimizationTests()
    f.setUp()
    try:
        f.bootstrap()
        framework = Path(__file__).resolve().parents[1]
        config = {
            "root": str(f.root.resolve()),
            "task": str(f.root / "control/tasks" / (validate(f.task) + ".json")),
            "framework": str(framework), "host": "fixture",
            "capacity": {"gpus": ["0", "1"], "gpu_slots_per_device": 3,
                         "cpu": 64, "memory_gb": 160, "storage_gb": 320, "build_slots": 1},
            "resources": f.resources, "worker_count": 1, "round_limit": 2,
            "evaluation_limit": 2, "codex_binary": "synthetic-codex",
            "model": "gpt-6-astra", "agent_cpu": 2, "agent_memory_gb": 4,
            "agent_timeout_seconds": 60,
            "environment": {"root": str(f.root.resolve()), "framework": str(framework),
                            "cases": {}, "base_image": "sha256:" + "a" * 64,
                            "phase_resources": {"backend": dict(f.resources, gpu_count=0)}},
        }
        path = f.root / "campaign-config.json"
        write(path, config)
        c = campaign.Campaign(path)
        monkeypatch.setattr(c, "preflight_codex", lambda: None)
        yield SimpleNamespace(f=f, c=c, config=config)
    finally:
        f.doCleanups()


def proposal(host):
    return {"parent": host.f.baseline, "reason": "Use the verified parent",
            "hypothesis": "Improve the partition policy",
            "evidence": ["Paired parent measurements"],
            "files_to_modify": [fixtures.PARTITION],
            "expected_tradeoffs": dict.fromkeys((*OBJECTIVES, "power_w"), "uncertain")}


def candidate_ready(host):
    child = host.f.child()
    row = host.c.prepare()
    state = read(host.c.state_path)
    row = state["rounds"][-1]
    row.update(status="candidate_ready", candidate=child, parent=host.f.baseline,
               hypothesis="A synthetic legal source change")
    host.c.save(state)
    return child, state, row


def test_round_invocation_approves_only_already_bound_gpu_check(host, monkeypatch):
    row = host.c.prepare()
    calls = []
    monkeypatch.setattr(RoundRuntime, "start", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(RoundRuntime, "stop", lambda self: None)
    def turn(**kwargs):
        calls.append(kwargs)
        return {"success": False, "failure_reason": "Synthetic client stop"}
    monkeypatch.setattr(campaign, "run_workspace_turn", turn)
    host.c.run(row["prompt_sha256"])
    assert len(calls) == 1
    extra = calls[0]["extra_overrides"]
    assert 'mcp_servers.terminal.enabled_tools=["exec","checkout_parent","gpu_check"]' in extra
    assert 'mcp_servers.terminal.tools.checkout_parent.approval_mode="approve"' in extra
    assert 'mcp_servers.terminal.tools.gpu_check.approval_mode="approve"' in extra
    assert calls[0]["model"] == host.config["model"]
    assert read(host.c.state_path)["evaluations_used"] == 0


def synthetic_transport_event():
    return {"type": "assistant", "session_id": SESSION,
            "is_api_error_message": True, "error": "server_error",
            "message": {"role": "assistant", "model": "<synthetic>",
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "content": [{"type": "text", "text":
                            "API Error: Connection lost mid-response. The response above may be incomplete."}]}}


def real_message(model=MODEL):
    return {"type": "assistant", "session_id": SESSION,
            "message": {"role": "assistant", "model": model,
                        "content": [{"type": "text", "text": "Synthetic test response"}]}}


def success_event(model=MODEL):
    result = {"type": "result", "session_id": SESSION, "subtype": "success",
              "is_error": False, "structured_output": {"status": "ready"}}
    if model is not None:
        result["modelUsage"] = {model: {"inputTokens": 1, "outputTokens": 1}}
    return result


@pytest.mark.parametrize("scenario,expected", [
    ("exact-client-notification", True),
    ("nonzero-synthetic-usage", False),
    ("unexpected-real-model", False),
    ("missing-real-model", False),
    ("pending-tool", False),
    ("wrong-session", False),
])
def test_claude_identity_and_pending_operation_guards(tmp_path, monkeypatch, scenario, expected):
    events = [synthetic_transport_event(), real_message(), success_event()]
    if scenario == "nonzero-synthetic-usage":
        events[0]["message"]["usage"]["output_tokens"] = 1
    elif scenario == "unexpected-real-model":
        events[1] = real_message("claude-other-model")
    elif scenario == "missing-real-model":
        events = [synthetic_transport_event(), success_event(None)]
    elif scenario == "pending-tool":
        events[1]["message"]["content"] = [{"type": "tool_use", "id": "unconfirmed",
                                                "name": "mcp__terminal__exec", "input": {}}]
    elif scenario == "wrong-session":
        for event in events:
            event["session_id"] = "00000000-0000-0000-0000-000000000002"
    calls = []
    class FakeProcess:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.stdout = kwargs["stdout"]
        def communicate(self, prompt, timeout):
            for event in events:
                self.stdout.write(json.dumps(event) + "\n")
            self.stdout.flush()
    monkeypatch.setattr(claude.subprocess, "Popen", FakeProcess)
    control = tmp_path / "control"
    control.mkdir()
    provider = tmp_path / "synthetic-provider.json"
    write(provider, {"ANTHROPIC_AUTH_TOKEN": "synthetic-unused-token",
                     "ANTHROPIC_BASE_URL": "https://invalid.example"})
    provider.chmod(0o600)
    schema = tmp_path / "schema.json"
    write(schema, {"type": "object"})
    result = claude.run_claude_turn(
        binary="synthetic-claude", model=MODEL, control=control, output=tmp_path / "out",
        framework=Path(__file__).resolve().parents[1], tool_config=tmp_path / "tools.json",
        output_schema=schema, provider_config=provider, config_directory=tmp_path / "config",
        python="synthetic-python", prompt="Synthetic test", thread_id=SESSION,
        max_attempts=2, backoff_seconds=0)
    assert result["success"] is expected
    assert len(calls) == 1  # No uncertain tool or identity failure is replayed.
    assert "--resume" in calls[0] and calls[0][-1] == SESSION
    assert result["client_attempts"][0]["client_transport_errors"] == (
        0 if scenario == "nonzero-synthetic-usage" else 1)
    assert "--strict-mcp-config" in calls[0] and "--restricted" in calls[0]


def test_selected_parent_development_is_free_but_export_remains_strict(host):
    row = host.c.prepare()
    runtime = RoundRuntime(read(Path(row["control"]) / "tool-config.json"))
    receipt = runtime.handler.checkout(proposal(host))
    parent_file = host.f.root / "candidates" / receipt["parent"] / "tree" / fixtures.PROTECTED
    original = parent_file.read_bytes()
    # The selected-parent copy can contain discarded experiments/protected edits.
    (runtime.workspace / fixtures.PROTECTED).write_text("development_only = True\n")
    (runtime.workspace / "local_experiment.py").write_text("result = 123\n")
    (runtime.workspace / fixtures.PARTITION).write_text("value = 2\n")
    child = export_submission(host.f.archive, runtime.workspace, receipt,
                              [fixtures.PARTITION], Path(row["control"]) / "submission")
    tree = host.f.root / "candidates" / child / "tree"
    assert (tree / fixtures.PARTITION).read_text() == "value = 2\n"
    assert (tree / fixtures.PROTECTED).read_bytes() == original
    assert not (tree / "local_experiment.py").exists()
    assert parent_file.read_bytes() == original
    assert host.f.archive.admit(child)["status"] == "legal"
    with pytest.raises(ValueError, match="task-allowed"):
        export_submission(host.f.archive, runtime.workspace, receipt,
                          [fixtures.PROTECTED], Path(row["control"]) / "illegal")
    bundle = read(Path(row["control"]) / "prompt/bundle.json")
    assert bundle["context"]["local_check_limits"]["source_write_mode"] == "development-copy"
    assert "held-out" not in {case["id"] for case in bundle["context"]["training_cases"]}


def test_pre_admission_build_failure_spends_no_evaluation(host, monkeypatch):
    child, state, row = candidate_ready(host)
    def fail_build(config, candidate):
        assert candidate == child
        raise RuntimeError("Synthetic compiler dependency failure before admission")
    monkeypatch.setattr(campaign, "prepare_runtime", fail_build)
    with pytest.raises(RuntimeError, match="before admission"):
        host.c.complete_evaluation(state, row)
    state = read(host.c.state_path)
    assert state["evaluations_used"] == 0
    assert state["rounds"][-1]["matrices"] == 0
    assert state["rounds"][-1]["status"] == "reconciliation_required"
    assert not any(json.loads(r["payload"])["candidate"] == child for r in host.f.queue.rows())
    assert not (Path(row["control"]) / "matrix/admission.json").exists()


def test_real_uncertain_worker_keeps_matrix_reservation_and_live_queue_lease(host, monkeypatch):
    child, state, row = candidate_ready(host)
    runtime = {"candidate": child, "image": host.config["environment"]["base_image"], "native": False}
    monkeypatch.setattr(campaign, "prepare_runtime", lambda *args: runtime)
    # This synthetic task lacks physical parent checkpoints. Keep the real
    # candidate admission, matrix reservation, broker and worker lifecycle.
    monkeypatch.setattr("grt_bench.incremental.prepare_matrix", lambda *args, **kwargs: {})
    def unconfirmed(*args, **kwargs):
        raise UncertainExecution("Synthetic container termination unconfirmed; retain lease")
    monkeypatch.setattr("grt_bench.worker.evaluate", unconfirmed)
    with pytest.raises(UncertainExecution, match="retain lease"):
        host.c.complete_evaluation(state, row)
    saved = read(host.c.state_path)
    pending = saved["rounds"][-1]
    assert saved["evaluations_used"] == 1 and pending["matrices"] == 1
    assert pending["status"] == "evaluating"
    assert "retain lease" in pending["reconciliation_required"]
    rows = [r for r in host.f.queue.rows() if json.loads(r["payload"])["candidate"] == child]
    assert len(rows) == 6 and sum(r["state"] == "running" for r in rows) == 1
    assert not any(r["state"] in {"failed", "succeeded"} for r in rows)
    assert not host.c.undispatched(pending)
    before = copy.deepcopy(rows)
    def wait_for_reconciliation(*args):
        raise campaign.ReconciliationRequired("The existing lease still needs host evidence")
    monkeypatch.setattr(campaign, "evaluate_matrix", wait_for_reconciliation)
    # Resume reaches the reconciliation boundary, rather than rejecting a round
    # that was incorrectly finalized as a failed candidate.
    with pytest.raises(campaign.ReconciliationRequired, match="host evidence"):
        host.c.resume_evaluation()
    assert read(host.c.state_path)["evaluations_used"] == 1
    assert [r for r in host.f.queue.rows() if json.loads(r["payload"])["candidate"] == child] == before


def test_full_flow_frontend_never_releases_gpu_before_confirmed_termination(tmp_path, monkeypatch):
    from grt_bench import worker
    resources = {"cpu": 1, "memory_gb": 1, "storage_gb": 1, "gpu_count": 1, "build_slots": 0}
    lease = {"id": "a" * 64, "resources": resources, "gpus": ["0"]}
    calls = []
    monkeypatch.setattr(worker, "run_container", lambda *args, **kwargs: {"exit_code": 0})
    def unconfirmed(*args):
        raise UncertainExecution("Synthetic owned frontend still unconfirmed")
    monkeypatch.setattr(worker, "stop_and_confirm", unconfirmed)
    with pytest.raises(UncertainExecution, match="unconfirmed"):
        worker.run_phased(lease, "sha256:" + "b" * 64, [], tmp_path,
                          dict(resources, gpu_count=0), lambda: None,
                          lambda *args: calls.append(args))
    assert calls == []


def test_uncertain_and_cancelling_leases_still_consume_gpu_slots(tmp_path):
    queue = Queue(tmp_path / "queue.sqlite")
    queue.register_host("fixture", {"cpu": 8, "memory_gb": 8, "storage_gb": 8,
                                   "build_slots": 1, "gpus": ["0"], "gpu_slots_per_device": 3})
    resources = {"cpu": 1, "memory_gb": 1, "storage_gb": 1, "gpu_count": 1, "build_slots": 0}
    jobs = [queue.submit({"synthetic_cell": i}, resources) for i in range(4)]
    leases = [queue.claim("fixture") for _ in range(3)]
    assert {lease["id"] for lease in leases} == set(jobs[:3])
    lease = leases[0]
    with queue.transaction() as db:
        db.execute("UPDATE jobs SET expires=0 WHERE id=?", (lease["id"],))
    assert next(r for r in queue.rows() if r["id"] == lease["id"])["state"] == "uncertain"
    evidence = {"terminated": True, "container": "grt-" + lease["id"][:32] + "-frontend",
                "checkpoint_sha256": "c" * 64}
    with pytest.raises(ValueError, match="Stale/cancelled"):
        queue.release(lease["id"], lease["token"], dict(resources, gpu_count=0), evidence)
    with pytest.raises(ValueError, match="Stale result"):
        queue.finish(lease["id"], lease["token"], {"status": "failed"}, False)
    assert queue.claim("fixture") is None
    queue.cancel(lease["id"])
    assert next(r for r in queue.rows() if r["id"] == lease["id"])["state"] == "cancelling"
    assert queue.claim("fixture") is None
    # Only an explicit host termination receipt makes that slot available.
    queue.reconcile(lease["id"], {"job": lease["id"], "terminated": True, "evidence_sha256": "d" * 64})
    assert queue.claim("fixture")["id"] == jobs[3]
