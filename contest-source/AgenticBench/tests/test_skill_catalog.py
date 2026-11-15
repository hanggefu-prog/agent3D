import json
from pathlib import Path

from agentic3d.orchestrator import Agentic3DOrchestrator
from local_bench import skill_cli
from local_bench.skill_catalog import (
    load_skill_catalog,
    required_skills,
    skill_by_name,
)


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_covers_every_skill_directory():
    records = load_skill_catalog(ROOT / "skills")
    assert {record.directory for record in records} == {
        path.name
        for path in (ROOT / "skills").iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    assert {record.category for record in records} == {
        "analyzer",
        "task_agent",
        "infrastructure",
    }


def test_layout_task_receives_common_infrastructure_dependencies():
    records = load_skill_catalog(ROOT / "skills")
    dependencies = {
        record.name
        for record in required_skills(records, "optimize-mol-layout")
    }
    assert dependencies == {
        "open3dbench-environment-check",
        "open3dbench-runtime-build",
        "open3dbench-benchmark-input",
        "open3dbench-candidate-evaluation",
        "open3dbench-results-analysis",
    }


def test_orchestrator_freezes_declared_supporting_skills(tmp_path):
    orchestrator = object.__new__(Agentic3DOrchestrator)
    orchestrator.framework_root = ROOT
    task_dir = tmp_path / "task-agent"
    primary = orchestrator._snapshot_stage_skill("layout", task_dir)
    assert primary == task_dir / "skill" / "SKILL.md"
    records = load_skill_catalog(ROOT / "skills")
    layout = skill_by_name(records, "optimize-mol-layout")
    expected = {record.directory for record in required_skills(records, layout.name)}
    actual = {
        path.parent.name
        for path in (task_dir / "supporting-skills").glob("*/SKILL.md")
    }
    assert actual == expected


def test_skill_cli_lists_and_resolves_dependencies(capsys):
    assert skill_cli.main(["list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(row["name"] == "open3dbench-sta-evaluation" for row in rows)
    assert skill_cli.main(["inspect", "optimize-mol-partition"]) == 0
    record = json.loads(capsys.readouterr().out)
    resolved = {row["name"] for row in record["resolved_requirements"]}
    assert "open3dbench-candidate-evaluation" in resolved
    assert "open3dbench-results-analysis" in resolved
