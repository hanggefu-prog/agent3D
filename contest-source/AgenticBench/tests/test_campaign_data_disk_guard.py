"""New campaign admission remains 64 GiB for every native dispatch, not once."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import codesign_runtime as module
from scripts import run_local_task_smoke as local


@pytest.mark.parametrize("minimum,available,should_wait", [(40, 50, False), (64, 50, True), (64, 64, False)])
def test_data_guard_preserves_legacy_default_and_enforces_campaign_threshold(tmp_path, monkeypatch,
                                                                          minimum, available, should_wait):
    runtime = object.__new__(module.Runtime)
    runtime.batch = tmp_path
    runtime.gpus, runtime.memory_gb = [0], 48
    runtime.share_gpu, runtime.min_gpu_free_mb = True, 2048
    if minimum != 40:
        runtime.min_data_free_gb = minimum
    (tmp_path / "locks").mkdir()
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k:
        "MemAvailable: 70000000 kB\n" if str(path) == "/proc/meminfo" else original(path, *a, **k))
    monkeypatch.setattr(module.subprocess, "check_output", lambda argv, **kwargs:
        "4096\n" if "--query-gpu=memory.free" in argv else "")
    state = {"free": available}
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path:
        SimpleNamespace(free=(20 if str(path) == "/" else state["free"]) * 1024**3))
    waits = []
    def wait(seconds):
        waits.append(seconds)
        state["free"] = 64
    monkeypatch.setattr(module.time, "sleep", wait)
    # Both calls perform fresh admission; consumption between candidates cannot
    # rely on the scheduler's earlier successful 64-GiB sample.
    for _ in range(2):
        state["free"] = available
        with runtime.slot() as gpu:
            assert gpu == 0
    assert waits == ([30, 30] if should_wait else [])


@pytest.mark.parametrize("value", [0, -1, True, None, "64", 64.0])
def test_invalid_data_guard_rejected_before_source_or_native_actions(value):
    with pytest.raises(ValueError, match="min_data_free_gb"):
        module.Runtime("/not-used", "/not-used", "sha256:" + "a" * 64, "0", min_data_free_gb=value)


def test_local_campaign_cli_sets_and_records_64_gib_guard(monkeypatch, tmp_path):
    seen = []
    def runtime(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(local, "LocalSmokeRuntime", runtime)
    monkeypatch.setattr(local, "run_smoke", lambda *a, **kw: {"state": "completed"})
    assert local.main(["--source", str(tmp_path / "source"), "--batch-root", str(tmp_path / "runs"),
        "--output", str(tmp_path / "output"), "--task", "placement_parameters_reference_20260923",
        "--image", "sha256:" + "a" * 64, "--gpu", "0"]) == 0
    assert seen[0]["min_data_free_gb"] == 64
