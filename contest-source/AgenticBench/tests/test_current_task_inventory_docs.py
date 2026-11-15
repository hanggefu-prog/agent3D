"""The documentation consistency checker passes on the repo and catches drift."""
import copy
import importlib.util
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT = ROOT / "scripts" / "check_doc_consistency.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_doc_consistency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cdc = _load()


@pytest.fixture(scope="module")
def facts():
    return cdc.collect_code_facts()


@pytest.fixture
def docs_copy(tmp_path):
    for relative, _ in cdc.CHECKS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    return tmp_path


def _edit(root, relative, old, new):
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert old in text, (relative, old)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_repository_docs_match_code(facts):
    assert cdc.check(REPO, facts=facts) == []
    assert cdc.main(["--repo-root", str(REPO)]) == 0


def test_facts_come_from_code(facts):
    from local_bench import metrics
    from local_bench.task_inventory import CORE_TASK_DIRECTORIES

    assert [task["directory"] for task in facts["tasks"]] == list(CORE_TASK_DIRECTORIES)
    assert facts["sta_wns_weight"] == metrics.STA_WNS_WEIGHT
    assert facts["hbt_wns_weight"] == pytest.approx(metrics.STA_WNS_WEIGHT)
    assert facts["post_grt"]["wns"] + facts["post_grt"]["tns"] == pytest.approx(1.0)
    assert facts["matrix_profile_qualified"] is False


def test_unmodified_copy_passes_without_links(docs_copy, facts):
    assert cdc.check(docs_copy, facts=facts, check_links=False) == []


@pytest.mark.parametrize(
    "relative, old, new, expected",
    [
        (cdc.ROOT_README,
         "| Local placement | `placement_preconditioner` | Fixed-partition global placement |\n", "",
         "README.md: local task table"),
        (cdc.ROOT_README, "0.6 WNS / 0.4 TNS", "0.5 WNS / 0.5 TNS", "local HBT timing weights"),
        (cdc.TASKS_DOC, "0.6 WNS / 0.4 TNS", "0.7 WNS / 0.3 TNS", "local timing weights"),
        (cdc.TASKS_DOC, "| `hbt_assignment_policy` |", "| `hbt_policy` |", "task table lists"),
        (cdc.TASK_REPORT, "| 1 / 360 / 12 |", "| 1 / 220 / 12 |",
         "tp_gnn_post_placement_timing limits"),
        (cdc.TASK_REPORT, "| `routing_maze_cost_source` |", "| `routing_maze_cost` |",
         "global_routing_maze_cost kind"),
        (cdc.TASK_REPORT, "| hbt (input, floorplan, partition, placement, pdn) |",
         "| hbt (input, floorplan, partition, placement) |", "hbt_assignment_policy upstream"),
        (cdc.TASK_REPORT, "0.9 * hpwl_score + 0.1 * overflow_score",
         "0.8 * hpwl_score + 0.2 * overflow_score", "placement formula"),
        (cdc.TASK_REPORT, "0.1 * overflow_cost + 0.9 * wirelength_cost",
         "0.2 * overflow_cost + 0.8 * wirelength_cost", "routing v1 formula"),
        (cdc.HARNESS_REPORT, "TNS 0.3", "TNS 0.4", "post-GRT timing weights"),
        (cdc.MATRIX_DOC, "local_default_bp_fe_ariane133_3seed", "local_default", "matrix profile id"),
    ],
)
def test_synthetic_doc_mismatch_fails(docs_copy, facts, relative, old, new, expected):
    _edit(docs_copy, relative, old, new)
    errors = cdc.check(docs_copy, facts=facts, check_links=False)
    assert errors, "mismatch was not detected"
    assert any(expected in error for error in errors), errors


def test_code_change_without_doc_change_fails(docs_copy, facts):
    changed = copy.deepcopy(facts)
    changed["post_grt"]["wns"], changed["post_grt"]["tns"] = 0.8, 0.2
    errors = cdc.check(docs_copy, facts=changed, check_links=False)
    flagged = {error.split(":", 1)[0] for error in errors}
    assert {cdc.ROOT_README, cdc.TASKS_DOC, cdc.TASK_REPORT, cdc.HARNESS_REPORT} <= flagged

    changed = copy.deepcopy(facts)
    changed["tasks"][0]["limits"] = (1, 361, 12)
    errors = cdc.check(docs_copy, facts=changed, check_links=False)
    assert errors == [
        "{}: tp_gnn_post_placement_timing limits (files / added / deleted) are 1 / 361 / 12 "
        "in code, doc says 1 / 360 / 12".format(cdc.TASK_REPORT)]


def test_code_self_inconsistency_is_reported(docs_copy, facts):
    changed = dict(facts, hbt_wns_weight=0.7)
    errors = cdc.check(docs_copy, facts=changed, check_links=False)
    assert any("hbt_policy.score_timing" in error for error in errors)


def test_broken_relative_link_is_reported(tmp_path):
    (tmp_path / "present.md").write_text("x\n")
    doc = tmp_path / "doc.md"
    doc.write_text("[ok](present.md#a) [web](https://example.org) [anchor](#x)\n"
                   "```\n[ignored](missing-in-code.md)\n```\n[bad](missing.md)\n")
    assert cdc.broken_links(doc) == ["doc.md:5: broken link missing.md"]


def test_missing_doc_is_reported(docs_copy, facts):
    (docs_copy / cdc.MATRIX_DOC).unlink()
    assert "{}: file is missing".format(cdc.MATRIX_DOC) in cdc.check(
        docs_copy, facts=facts, check_links=False)


def test_markdown_section_ignores_code_comments():
    text = "## A\nx\n```sh\n# not a heading\n```\ny\n## B\nz\n"
    assert cdc.markdown_section(text, "A") == "x\n```sh\n# not a heading\n```\ny"
    assert cdc.markdown_section(text, "C") is None
