"""Mocked cold/cached packaging contract; never runs Docker or native code."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from local_bench import workspace
from scripts import build_routing_candidate_image as builder
from scripts import openroad_compiler_cache as cache

BASE = "sha256:" + "a" * 64
IMAGE = "sha256:" + "b" * 64


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "attempt"
    calls = []
    state = {"binary_mismatch": False, "compile_error": False}
    candidate = {"source_patch": "verified test patch", "schema_version": 1}
    monkeypatch.setattr(builder, "load_task", lambda _: SimpleNamespace(
        kind="routing_net_ordering_source", task_id="routing-test"))
    monkeypatch.setattr(builder, "load_candidate", lambda _: candidate)
    monkeypatch.setattr(builder, "verify_candidate", lambda *args: {"patch": {
        "paths": [workspace.OPENROAD_SUBMODULE + "/src/grt/test.cpp"]}})
    def clone(_source, destination, **kwargs):
        tree = destination / workspace.OPENROAD_SUBMODULE
        tree.mkdir(parents=True)
        (tree / "CMakeLists.txt").write_text("fixture")
    monkeypatch.setattr(builder, "isolated_clone", clone)
    monkeypatch.setattr(builder, "apply_candidate_patch", lambda *args: None)
    monkeypatch.setattr(builder, "_local_image_id", lambda value: value)
    def run(command, **kwargs):
        calls.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="d" * 40 + "\n")
        if command[:2] == ["docker", "build"]:
            assert kwargs["env"]["DOCKER_BUILDKIT"] == "0"
            assert command[-1] == str(output / "package-context")
            assert command[command.index("--cpu-quota") + 1] == "1600000"
            assert command[command.index("--memory") + 1] == "48g"
            dockerfile = (output / "package-context/Dockerfile").read_text()
            assert "cmake" not in dockerfile
            assert " -version" not in dockerfile  # cannot mutate an image layer through candidate execution
            assert "COPY openroad " + str(workspace.PREBUILT_OPENROAD_BINARY) in dockerfile
            (output / "image.id").write_text(IMAGE)
        return subprocess.CompletedProcess(command, 0, stdout="")
    monkeypatch.setattr(builder.subprocess, "run", run)
    def compile(**kwargs):
        state["compile_kwargs"] = kwargs
        target = kwargs["output"]
        target.mkdir()
        (target / "diagnostic.txt").write_text("preserve on failure or success")
        if state["compile_error"]:
            raise cache.CompilerCacheError("fixture compilation failed")
        for name in ("build", "build-source", "artifacts", "logs"):
            (target / name).mkdir()
        binary = target / "artifacts/openroad"
        binary.write_bytes(b"fixture binary")
        binary.chmod(0o755)
        return {"binary_path": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_fingerprint": workspace.openroad_source_fingerprint(kwargs["source_tree"]),
                "build_source_fingerprint": {"fixture": "after generation"},
                "source_generation": {"unexpected_paths": [], "generated_files": []},
                "cache_key": "c" * 64, "cache_identity": {"base_image_id": BASE}}
    monkeypatch.setattr(cache, "compile_cached", compile)
    def read_image(command):
        assert "--network=none" in command
        assert "--read-only" in command
        manifest = json.loads((output / "package-context/runtime.json").read_text())
        if "python3" in command:
            return {"binary_path": str(workspace.PREBUILT_OPENROAD_BINARY),
                    "resolved_command": str(workspace.PREBUILT_OPENROAD_BINARY),
                    "binary_sha256": "e" * 64 if state["binary_mismatch"] else manifest["binary_sha256"],
                    "dependencies": "fixture dependencies"}
        return manifest
    monkeypatch.setattr(builder, "_run_json", read_image)
    def invoke(**kwargs):
        return builder.build_candidate_image(tmp_path / "task", tmp_path / "candidate", source,
            BASE, output, compiler_cache_dir=tmp_path / "cache", **kwargs)
    return SimpleNamespace(invoke=invoke, output=output, state=state, calls=calls, source=source)


def test_cached_build_is_packaged_and_reverified_before_private_cleanup(prepared):
    result = prepared.invoke()
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["compilation_mode"] == "shared-ccache"
    assert receipt["compiler_cache_key"] == "c" * 64
    assert receipt["runtime_verification"]["binary_sha256"] == receipt["binary_sha256"]
    assert not (prepared.output / "compiler-cache-attempt/build").exists()
    assert not (prepared.output / "compiler-cache-attempt/build-source").exists()
    assert (prepared.output / "compiler-cache-attempt/artifacts/openroad").is_file()
    assert (prepared.output / "compiler-cache-attempt/diagnostic.txt").is_file()
    assert prepared.state["compile_kwargs"]["threads"] == 8
    events = [json.loads(row) for row in (prepared.output / "image-build-events.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in events if row["state"] == "completed"] == [
        "source_clone", "cached_compilation", "image_packaging", "installed_runtime_verification"]
    assert all(row["duration_seconds"] >= 0 for row in events if row["state"] == "completed")


def test_retained_build_still_uses_external_cache(prepared):
    prepared.invoke(build_cache="retain")
    assert (prepared.output / "compiler-cache-attempt/build").is_dir()
    assert not (prepared.output / "cache-retention.json").exists()
    assert prepared.state["compile_kwargs"]["cache_root"].name == "cache"


def test_installed_binary_mismatch_never_seals_or_cleans_attempt(prepared):
    prepared.state["binary_mismatch"] = True
    with pytest.raises(workspace.WorkspaceError, match="binary verification"):
        prepared.invoke()
    assert not (prepared.output / "receipt.json").exists()
    assert (prepared.output / "compiler-cache-attempt/build").is_dir()
    assert (prepared.output / "workspace").is_dir()


def test_compile_failure_is_preserved_without_packaging_or_retry(prepared):
    prepared.state["compile_error"] = True
    with pytest.raises(cache.CompilerCacheError, match="compilation failed"):
        prepared.invoke()
    assert not any(row[:2] == ["docker", "build"] for row in prepared.calls)
    assert (prepared.output / "compiler-cache-attempt/diagnostic.txt").is_file()
    assert not (prepared.output / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        prepared.invoke()


@pytest.mark.parametrize("size", [0, 65, True])
def test_invalid_cache_limit_fails_before_any_output(prepared, size):
    with pytest.raises(ValueError, match="compiler_cache_gb"):
        prepared.invoke(compiler_cache_gb=size)
    assert not prepared.output.exists()


@pytest.mark.parametrize("relative", ["candidate", "candidate/ccache", "task", "task/ccache"])
def test_standalone_cache_cannot_mutate_candidate_or_task_archive(prepared, relative):
    root = prepared.source.parent
    with pytest.raises(ValueError, match="separate"):
        builder.build_candidate_image(root / "task", root / "candidate", prepared.source,
            BASE, prepared.output, compiler_cache_dir=root / relative)
    assert not prepared.output.exists()
