"""No model/network/physical calls: exercise the actual enhanced-task boundary."""
import json
from types import SimpleNamespace

import pytest

from agentic3d.enhanced_task import (
    compile_enhanced_task, knowledge_catalog, local_context, compact_history, MAX_TASK_CHARS,
)
from agentic3d.knowledge import KnowledgeStore, digest
from agentic3d.sta_protocol import PARTITION_PATH, ANALYZER, LOCAL
from agentic3d import sta_knowledge_runner as runner


SPEC = SimpleNamespace(allowed_source_paths=[PARTITION_PATH])
SEED = {"skills/example/SKILL.md": "---\nname: example\ndescription: Seed method\n---\nSEED_BODY\n",
        "skills/example/scripts/helper.py": "# seed helper\n"}


def skill():
    return {"id": "learned-method", "description": "Reusable diagnosis",
            "files": {"SKILL.md": "---\nname: learned-method\ndescription: Reusable diagnosis\n---\nMETHOD_BODY\n",
                      "scripts/check.py": "# optional long diagnostic\n"}}


def plan():
    return {"hypothesis": "Focus on the diagnosed tail", "evidence": "Measured proxy mismatch",
            "evidence_paths": ["evidence/index.json"], "specialist_instruction": "Check both dies",
            "max_inner_attempts": 3, "knowledge_refs": [], "skills": [], "tasks": [],
            "local_task": {"id": "focused-task", "description": "Tail diagnosis",
                           "stage": "partition", "source_scope": [PARTITION_PATH],
                           "evaluation": "partition", "objective": [
                               {"metric": "timing_weighted_cut", "weight": 1., "scale_floor": 1.}],
                           "procedure": "Integrated relevant method; inspect path-tail evidence first.",
                           "validation": "Check measured cut/balance and final STA separately."}}


def setup_knowledge(tmp_path):
    store = KnowledgeStore.initialize(tmp_path / "knowledge", seed_files=SEED, case="bp")
    workspace = tmp_path / "workspace"
    for name, text in SEED.items():
        dest = workspace / "knowledge/seed" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    store.save("round-01", [skill()], [plan()["local_task"]], {"validation": "hypothesis only"})
    return store, workspace, knowledge_catalog(store, workspace)


def choose(row, attachments=()):
    return {"ref": row["ref"], "reason": "Relevant to the diagnosed mechanism", "attachments": list(attachments)}


def test_catalog_has_descriptors_not_bodies_and_distinct_kinds(tmp_path):
    store, workspace, catalog = setup_knowledge(tmp_path)
    assert {r["kind"] for r in catalog} == {"skill", "task"}
    assert "METHOD_BODY" not in json.dumps(catalog) and "SEED_BODY" not in json.dumps(catalog)
    assert len({r["ref"] for r in catalog}) == len(catalog)
    for row in catalog:
        for resource in row["resources"]:
            assert digest((workspace / resource["path"]).read_text()) == resource["sha256"]
    assert any(r["assessment"] == {"validation": "hypothesis only"} for r in catalog)


def test_selected_resources_and_local_context_are_compact(tmp_path):
    _, workspace, catalog = setup_knowledge(tmp_path)
    row = next(r for r in catalog if r["id"] == "learned-method")
    attachment = next(r["path"] for r in row["resources"] if r["path"].endswith("check.py"))
    p = plan()
    p["knowledge_refs"] = [choose(row, [attachment])]
    packet = compile_enhanced_task(p, catalog, SPEC, workspace)
    assert packet["task"] == p["local_task"]
    assert packet["optional_attachments"][0]["path"] == attachment
    assert packet["sha256"] == digest({k: v for k, v in packet.items() if k != "sha256"})
    state = {k: {} for k in ("case", "phase", "budget", "reference_metrics", "incumbent_metrics",
                             "incumbent_score", "evidence_index")}
    state.update(knowledge_catalog=catalog, knowledge=["DO_NOT_PASS_LIBRARY"],
                 incumbent_candidate="DO_NOT_PASS_FULL_CANDIDATE",
                 history=[{"round": 1, "global_plan": {"skills": ["DO_NOT_PASS_PRIOR_SKILL"]}}])
    payload = local_context(state, packet, {}, 1, [], {})
    serialized = json.dumps(payload)
    assert "DO_NOT_PASS" not in serialized and "METHOD_BODY" not in serialized
    assert "active_local_task" not in payload and "plan" not in payload
    assert serialized.count("Integrated relevant method") == 1
    assert "do not scan knowledge/seed" in LOCAL
    assert "self-contained" in ANALYZER


@pytest.mark.parametrize("mode", ["unknown", "duplicate", "escape", "wrong-source", "tamper", "symlink"])
def test_invalid_references_and_attachments_rejected(tmp_path, mode):
    _, workspace, catalog = setup_knowledge(tmp_path)
    row = next(r for r in catalog if r["id"] == "learned-method")
    path = row["resources"][0]["path"]
    p = plan()
    p["knowledge_refs"] = [choose(row, [path])]
    if mode == "unknown":
        p["knowledge_refs"][0]["ref"] = "test-proposals:excluded"
    elif mode == "duplicate":
        p["knowledge_refs"] *= 2
    elif mode == "escape":
        p["knowledge_refs"][0]["attachments"] = ["../private.py"]
    elif mode == "wrong-source":
        p["knowledge_refs"][0]["attachments"] = [catalog[0]["resources"][0]["path"]]
    elif mode == "tamper":
        (workspace / path).write_text("tampered")
    else:
        (workspace / path).unlink()
        (workspace / path).symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        compile_enhanced_task(p, catalog, SPEC, workspace)


def test_compactness_and_original_hard_constraints(tmp_path):
    _, workspace, catalog = setup_knowledge(tmp_path)
    for mutate in (
        lambda p: p["local_task"].update(procedure="x" * MAX_TASK_CHARS),
        lambda p: p["local_task"].update(source_scope=["open3dbench/runner.py"]),
        lambda p: p["local_task"]["objective"][0].update(weight=2.),
        lambda p: p.update(knowledge_refs=[choose(catalog[0])] * 5),
        lambda p: p.update(max_inner_attempts=4),
    ):
        p = plan()
        mutate(p)
        with pytest.raises(ValueError):
            compile_enhanced_task(p, catalog, SPEC, workspace)
    p = plan()
    # Eight-field tasks remain reusable; enhancement is an execution envelope.
    assert len(compile_enhanced_task(p, catalog, SPEC, workspace)["task"]) == 8


def test_history_retains_failed_hypotheses_without_library_bodies():
    p = plan()
    p["skills"] = [skill()]
    history = [{"round": 1, "accepted_as_incumbent": False, "global_plan": p}]
    compact = compact_history(history)
    assert compact_history(compact) == compact
    assert compact[0]["hypothesis"] == p["hypothesis"]
    assert compact[0]["task_id"] == "focused-task"
    assert "METHOD_BODY" not in json.dumps(compact)


def test_frozen_test_cannot_select_proposals_but_reuses_training(tmp_path):
    split = {"train": ["bp"], "test": ["ariane133"]}
    train = KnowledgeStore.initialize(tmp_path / "train", seed_files=SEED, case="bp",
                                     phase="train", split=split)
    train.save("round-01", [skill()], [], {"validation": "observed"})
    package = tmp_path / "package.json"
    train.export(package)
    original = package.read_bytes()
    test = KnowledgeStore.initialize(tmp_path / "test", seed_files=SEED, case="ariane133",
                                    phase="test", split=split, imported=package)
    workspace = tmp_path / "workspace"
    before = knowledge_catalog(test, workspace)
    changed = skill()
    changed["files"]["SKILL.md"] += "EXCLUDED_TEST_LESSON"
    test.save("round-01", [changed], [], {})
    after = knowledge_catalog(test, workspace)
    assert before == after and package.read_bytes() == original
    selected = next(r for r in after if r["id"] == "learned-method")
    p = plan()
    p["knowledge_refs"] = [choose(selected, [selected["resources"][0]["path"]])]
    assert compile_enhanced_task(p, after, SPEC, workspace)["knowledge_sources"]
    excluded = json.loads((test.root / "test-proposals/round-01/record.json").read_text())["record"]
    p["knowledge_refs"][0]["ref"] = f"learned:skill/learned-method@{digest(excluded)}"
    with pytest.raises(ValueError, match="excluded"):
        compile_enhanced_task(p, after, SPEC, workspace)
    assert not any("EXCLUDED_TEST_LESSON" in f.read_text() for f in workspace.rglob("SKILL.md"))


def test_implementation_contract_includes_enhanced_module(tmp_path):
    for name in ("knowledge.py", "sta_protocol.py", "sta_knowledge_runner.py", "enhanced_task.py",
                 "research_feedback.py", "research_io.py", "evidence.py"):
        path = tmp_path / "agentic3d" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("original")
    path = tmp_path / "scripts/run_sta_codesign_comparison.py"
    path.parent.mkdir()
    path.write_text("driver")
    before = runner.implementation_digest(tmp_path)
    (tmp_path / "agentic3d/enhanced_task.py").write_text("changed")
    assert runner.implementation_digest(tmp_path) != before
