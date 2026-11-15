"""No real model requests: test the simple transport and file-access gate."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d import research_io as io
from agentic3d import sta_knowledge_runner as runner
from agentic3d.knowledge import write_json, digest
from scripts.run_sta_codesign_comparison import CodexExecConfig


@pytest.mark.parametrize("role", [runner.ANALYZER_SCHEMA, runner.LOCAL_SCHEMA,
                                  runner.REFLECTION_SCHEMA, runner.REVIEW_SCHEMA])
def test_each_role_can_report_blocked_without_a_dummy_payload(role):
    schema = io.response_schema(role)
    assert schema["properties"]["response"]["anyOf"][0] == role
    with pytest.raises(io.ResearchBlocked, match="reader unavailable"):
        io.unwrap_response({"blocked": "reader unavailable", "response": None})
    assert io.unwrap_response({"blocked": None, "response": {"normal": "role payload"}}) == {"normal": "role payload"}


@pytest.mark.parametrize("payload", [{}, {"blocked": "", "response": None},
    {"blocked": False, "response": None}, {"blocked": "failed", "response": {}},
    {"blocked": None, "response": None}])
def test_transport_does_not_accept_ambiguous_pause_task(payload):
    with pytest.raises(ValueError):
        io.unwrap_response(payload)


@pytest.mark.parametrize("mode", ["ok", "wrong_hash", "blocked", "process_failure", "bad_json", "launch_error"])
def test_real_access_probe_contract_in_same_readonly_environment(tmp_path, mode):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source, evidence = workspace / "source.py", workspace / "evidence.txt"
    source.write_text("source fixture")
    evidence.write_text("evidence fixture")
    knowledge = workspace / "knowledge/seed/tools.json"
    knowledge.parent.mkdir(parents=True)
    knowledge.write_text("{}")
    calls = []
    def model(prompt, config):
        calls.append(config)
        if mode == "launch_error":
            raise OSError("model binary cannot execute")
        assert config.workspace == workspace and config.ephemeral and config.sandbox == "read-only"
        assert config.resume_thread_id is None
        nonce = json.loads((workspace / "research-access.json").read_text())["nonce"]
        assert nonce not in prompt
        payload = {"nonce": nonce, **{key: hashlib.sha256(path.read_bytes()).hexdigest()
                    for key, path in (("source_sha256", source), ("evidence_sha256", evidence),
                                      ("knowledge_sha256", knowledge))}}
        assert payload["source_sha256"] not in prompt
        if mode == "wrong_hash":
            payload["source_sha256"] = "invented"
        answer = {"blocked": None, "response": payload}
        if mode == "blocked":
            answer = {"blocked": "file tool unavailable", "response": None}
        return SimpleNamespace(success=mode != "process_failure", final_message=(
            "not json" if mode == "bad_json" else json.dumps(answer)))
    args = SimpleNamespace(model="test", codex_bin="no-real-api")
    driver = SimpleNamespace(run_codex=model, CodexExecConfig=CodexExecConfig)
    root = tmp_path / "preflight"
    if mode == "ok":
        io.check_research_access(args, root, workspace, driver, source, evidence)
        io.check_research_access(args, root, workspace, driver, source, evidence)
        assert calls[0].run_dir != calls[1].run_dir  # fresh on explicit resume
    else:
        with pytest.raises(io.ResearchBlocked):
            io.check_research_access(args, root, workspace, driver, source, evidence)
        assert not json.loads((root / "calls/001/access_check.json").read_text())["passed"]


def test_evidence_hash_and_workspace_boundary_are_not_relaxed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "evidence.txt"
    file.write_text("real")
    report = {"error": None, "files": [{"local_file": str(file), "sha256": digest("real")}]}
    assert runner.require_evidence(report, workspace) == report
    file.write_text("tampered")
    with pytest.raises(io.ResearchBlocked):
        runner.require_evidence(report, workspace)
    file = tmp_path / "private.txt"
    file.write_text("real")
    report["files"][0]["local_file"] = str(file)
    with pytest.raises(io.ResearchBlocked):
        runner.require_evidence(report, workspace)


@pytest.mark.parametrize("report", [None, [], {"files": [{}]}, {"files": [None]},
    {"files": [{"local_file": "relative/path", "sha256": "bad"}]}])
def test_malformed_evidence_is_blocked_not_a_candidate_failure(tmp_path, report):
    with pytest.raises(io.ResearchBlocked):
        runner.require_evidence(report, tmp_path)
