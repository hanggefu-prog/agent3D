import json
import subprocess
from pathlib import Path

import pytest

from scripts import openevolve_docker as launcher


def arguments(tmp_path):
    data = tmp_path / "data"
    reference = data / "reference/run"
    reference.mkdir(parents=True)
    return [
        "global_routing_ripup_policy", "--profile", "bp_lol_ng45_reference_20261004",
        "--data-root", str(data), "--benchmark-input", str(data / "input"),
        "--reference-run", str(reference), "--output-dir", str(data / "experiment"),
        "--open3dbench-root", str(launcher.FRAMEWORK.parent), "--iterations", "20",
    ]


def test_plan_exports_real_task_without_docker_or_model_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "validate_benchmark_input", lambda *a, **k: {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("USE_LOCAL_OPENROAD", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only made an external process call")
    original_check_output = subprocess.check_output
    def git_metadata_only(command, **kwargs):
        # Linked worktrees need their read-only administrative mounts even in
        # the exported plan. No Docker, provider or evaluator may be launched.
        assert len(command) == 5 and command[:2] == ["git", "-C"]
        assert command[3] == "rev-parse"
        assert command[4] in {"--git-dir", "--git-common-dir"}
        return original_check_output(command, **kwargs)
    monkeypatch.setattr(subprocess, "call", forbidden)
    monkeypatch.setattr(subprocess, "check_output", git_metadata_only)
    assert launcher.main(arguments(tmp_path) + ["--plan-only"]) == 0
    plan = json.loads((tmp_path / "data/experiment/launch_plan.json").read_text())
    assert plan["iterations"] == 20
    assert "DEEPSEEK_API_KEY" not in plan["container_environment"]
    assert (tmp_path / "data/experiment/bundle/evaluator.py").is_file()
    assert (tmp_path / "data/experiment/bundle/initial_program.py").is_file()


def test_launcher_rejects_input_outside_mounted_data_root(tmp_path):
    args = arguments(tmp_path)
    args[args.index("--benchmark-input") + 1] = str(tmp_path / "unmounted-input")
    with pytest.raises(SystemExit) as exc:
        launcher.main(args + ["--plan-only"])
    assert exc.value.code == 2


def test_launcher_refuses_to_overwrite_experiment(tmp_path):
    args = arguments(tmp_path)
    (tmp_path / "data/experiment").mkdir()
    with pytest.raises(SystemExit) as exc:
        launcher.main(args + ["--plan-only"])
    assert exc.value.code == 2
