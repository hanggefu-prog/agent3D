import json
from pathlib import Path

import pytest

from local_bench import cli as task_cli
from local_bench import profile_cli
from local_bench import sta_cli
from local_bench.model import load_task


ROOT = Path(__file__).resolve().parents[1]


def test_validation_profiles_are_not_search_tasks(capsys):
    assert not (ROOT / "local_tasks" / "placement_two_pass_sta" / "task.yaml").exists()
    assert not (ROOT / "local_tasks" / "tp_gnn_ta3d_sta_validation" / "task.yaml").exists()
    assert not (ROOT / "local_tasks" / "tp_gnn_placement_sta" / "task.yaml").exists()
    assert not (ROOT / "local_tasks" / "tp_gnn_sta_inner" / "task.yaml").exists()
    assert not (ROOT / "local_tasks" / "placement_sta_inner" / "task.yaml").exists()

    assert task_cli.main(["list"]) == 0
    task_rows = json.loads(capsys.readouterr().out)
    task_ids = {row["id"] for row in task_rows}
    assert "placement_two_pass_sta" not in task_ids
    assert "tp_gnn_ta3d_sta_validation" not in task_ids
    assert "tp_gnn_placement_sta" not in task_ids
    assert "tp_gnn_sta_inner" not in task_ids
    assert "placement_sta_inner" not in task_ids


def test_sta_cli_lists_and_loads_profiles(capsys):
    assert sta_cli.main(["list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["profile"] for row in rows} == {
        "placement_two_pass",
        "tp_gnn_placement_two_pass",
        "tp_gnn_ta3d_two_pass",
    }
    spec = load_task(ROOT / "evaluation_profiles" / "placement_two_pass")
    assert spec.task_id == "placement_two_pass_sta"
    assert spec.start_stage == "placement"
    assert spec.stop_stage == "opt"
    codesign = load_task(ROOT / "evaluation_profiles" / "tp_gnn_placement_two_pass")
    assert codesign.task_id == "tp_gnn_placement_sta"
    assert codesign.start_stage == "partition"
    assert codesign.stop_stage == "opt"


def test_generic_profile_cli_includes_inner_proxies(capsys):
    assert profile_cli.main(["list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["profile"] for row in rows} == {
        "placement_proxy",
        "placement_two_pass",
        "tp_gnn_partition_proxy",
        "tp_gnn_placement_thermal",
        "tp_gnn_placement_cts_thermal",
        "tp_gnn_placement_two_pass",
        "tp_gnn_ta3d_two_pass",
    }
    with pytest.raises(FileNotFoundError, match="specialized CLI"):
        sta_cli.main(["inspect", "placement_proxy"])


def test_compatibility_entry_points_delegate_to_skill():
    aliases = {
        "run_sta_codesign_remote_eval.sh": "run_remote_sta.sh",
        "run_placement_best_sta_case.sh": "promote_placement_case.sh",
        "run_tp_gnn_ta3d_sta_validation.sh": "validate_ta3d_objective.sh",
    }
    for alias, implementation in aliases.items():
        text = (ROOT / "scripts" / alias).read_text(encoding="utf-8")
        assert "skills/open3dbench-sta-evaluation/scripts/" + implementation in text


def test_consolidated_docker_launcher_can_dispatch_sta_profiles():
    launcher = (ROOT / "scripts" / "benchmark_docker.sh").read_text(encoding="utf-8")
    assert "module=local_bench.profile_cli" in launcher
    assert "module=local_bench.sta_cli" in launcher
    assert 'exec python3 -m "$module" "$@"' in launcher
