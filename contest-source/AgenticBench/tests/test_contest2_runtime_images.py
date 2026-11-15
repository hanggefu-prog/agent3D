"""Actual CLI adapter behavior, including Docker 29's manifest-style IDs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from contest2 import runtime_images as runtime
from contest2 import agent_container


def inspection(record, address):
    return {"Id": address, "RootFS": {"Type": "layers", "Layers": record["rootfs_layers"]},
            "Os": "linux", "Architecture": "amd64"}


@pytest.mark.parametrize("kind", ["config", "docker_manifest", "oci_manifest"])
def test_frozen_contents_accept_all_audited_addresses(kind):
    record = next(iter(runtime.records().values()))
    addresses = [record["config_id"], *record["manifest_ids"]]
    assert runtime.validate_inspection(inspection(record, addresses[
        ["config", "docker_manifest", "oci_manifest"].index(kind)]), record) == record["config_id"]


@pytest.mark.parametrize("tamper", ["id", "layers", "platform"])
def test_same_tag_or_id_does_not_hide_changed_contents(tamper):
    record = next(iter(runtime.records().values()))
    value = inspection(record, record["manifest_ids"][0])
    if tamper == "id":
        value["Id"] = "sha256:" + "c" * 64
    elif tamper == "layers":
        value["RootFS"]["Layers"] = ["sha256:" + "c" * 64]
    else:
        value["Architecture"] = "arm64"
    with pytest.raises(ValueError, match="contents differ"):
        runtime.validate_inspection(value, record)


def test_installed_adapter_handles_run_inspect_and_native_builder_tag(tmp_path, monkeypatch):
    record = next(iter(runtime.records().values()))
    value = inspection(record, record["manifest_ids"][0])
    executable = tmp_path / "real-docker"
    executable.write_text("#!" + sys.executable + "\nimport sys,json\n"
        + "value=" + repr(value) + "\n"
        + "if sys.argv[1:3]==['image','inspect']:\n print(json.dumps([value]))\n"
        + "else:\n print(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o700)
    monkeypatch.setenv("CONTEST_DOCKER_REAL", str(executable))
    monkeypatch.setenv("PATH", os.environ["PATH"])
    binding = runtime.install(record["config_id"], tmp_path / "work")
    assert binding["docker_address"] == record["manifest_ids"][0]
    identifier = subprocess.check_output(["docker", "image", "inspect", record["config_id"],
                                          "--format", "{{.Id}}"], text=True).strip()
    assert identifier == record["config_id"]
    values = json.loads(subprocess.check_output(["docker", "image", "inspect", "base:local"], text=True))
    assert values[0]["Id"] == record["config_id"]
    command = json.loads(subprocess.check_output(["docker", "run", "-e",
        "OPEN3DBENCH_IMAGE_ID=" + record["config_id"], record["config_id"], "-version"], text=True))
    assert record["manifest_ids"][0] in command
    assert "OPEN3DBENCH_IMAGE_ID=" + record["config_id"] in command
    from grt_bench.builder import local_base_tag
    assert local_base_tag(record["config_id"]).startswith("agentic3d-grt-base:")


def test_agent_has_no_default_session_deadline_and_is_stopped(tmp_path, monkeypatch):
    (tmp_path / "agent-result.json").write_text(json.dumps({"candidate_id": "a" * 64}))
    calls, stopped = [], []
    def run(argv, **options):
        calls.append(options)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(agent_container.subprocess, "run", run)
    monkeypatch.setattr(agent_container, "stop_and_confirm", lambda *args: stopped.append(args))
    assert agent_container.run_agent(startup_command=["docker", "run"], name="owned", scratch=tmp_path) == "a" * 64
    assert calls[0].get("timeout") is None
    assert calls[1]["timeout"] is None
    assert len(stopped) == 1


def test_gpu_failure_is_diagnosed_before_agent_and_teardown_is_confirmed(tmp_path, monkeypatch):
    record = next(iter(runtime.records().values()))
    monkeypatch.setattr(runtime, "resolve", lambda image: {"docker_address": image})
    monkeypatch.setattr(runtime, "real_docker", lambda: "/trusted/docker")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: type("Result", (), {
        "returncode": 1, "stdout": "", "stderr": "no kernel image is available"})())
    stopped = []
    with patch("grt_bench.runtime.stop_and_confirm", side_effect=lambda *a: stopped.append(a)):
        with pytest.raises(ValueError, match="official CUDA runtime"):
            runtime.gpu_preflight(record["config_id"], tmp_path, ["0"])
    evidence = json.loads(next(tmp_path.rglob("execution.json")).read_text())
    assert "no kernel image" in evidence["stderr"]
    assert len(stopped) == 1


@pytest.mark.parametrize('retain', [False, True, 'false'])
def test_runtime_retention_boolean(tmp_path, monkeypatch, retain):
    from contest2 import runner
    path = tmp_path / 'runtime.json'
    config = {'queue': 'queue.sqlite', 'host': 'host', 'image': 'sha256:' + 'a' * 64,
              'capacity': {}, 'retain_physical_files': retain}
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, 'resolve', lambda *a: {})
    monkeypatch.setattr(runner, 'install', lambda *a: {})
    monkeypatch.setattr(runner, 'Queue', lambda *a: type('Queue', (), {'register_host': lambda *a: None})())
    monkeypatch.setattr(runner, 'DockerMatrixTransport', lambda **k: None)
    if isinstance(retain, bool):
        assert runner._runtime(path)[1]['retain_physical_files'] is retain
    else:
        with pytest.raises(ValueError, match='boolean'):
            runner._runtime(path)
