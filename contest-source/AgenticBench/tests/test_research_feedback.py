import copy

import pytest

from agentic3d.research_feedback import index_suggestions, validate_feedback, validate_review
from agentic3d.sta_protocol import LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA
from test_sta_knowledge import task, skill, driver


def suggestion():
    return dict(target="task", change="Use final timing when the proxy disagrees",
                rationale="The two stages can trade off", evidence_paths=[],
                uncertainty="Needs cross-case verification")


def feedback(role="local"):
    head = dict(hypothesis="test", program="candidate source") if role == "local" else dict(assessment="test")
    return dict(**head, observations=[dict(finding="Proxy mismatch", evidence_paths=["evidence/test.json"],
                                          uncertainty="Causal mechanism unproven")], suggestions=[suggestion()])


def advice():
    return index_suggestions([{"attempt": 1, "feedback": feedback()}], feedback("reflector"))


def review():
    raw = skill()
    raw["files"] = [{"path": k, "content": v} for k, v in raw["files"].items()]
    return dict(assessment="Reviewed globally", skills=[raw], tasks=[task()], decisions=[
        dict(suggestion_id=s["suggestion_id"], decision="adopt", reason="Consolidated evidence",
             skill_ids=[raw["id"]], task_ids=[task()["id"]]) for s in advice()])


SPEC = driver.load_task(driver.STA_PROFILE)


@pytest.mark.parametrize("role", ["local", "reflector"])
@pytest.mark.parametrize("field", ["tasks", "skills", "local_task", "decisions"])
def test_non_analyzer_cannot_return_formal_knowledge(role, field):
    value = feedback(role)
    value[field] = []
    with pytest.raises(ValueError, match="Analyzer-only"):
        validate_feedback(value, role, lambda _: None)
    schema = LOCAL_SCHEMA if role == "local" else REFLECTION_SCHEMA
    assert field not in schema["properties"] and not schema["additionalProperties"]


def test_feedback_validates_citations_and_keeps_uncertainty():
    seen = []
    value = validate_feedback(feedback(), "local", seen.append)
    assert seen == ["evidence/test.json"] and "program" not in value
    assert value["suggestions"][0]["uncertainty"]
    def reject(_):
        raise ValueError("missing citation")
    with pytest.raises(ValueError, match="missing citation"):
        validate_feedback(feedback("reflector"), "reflector", reject)


@pytest.mark.parametrize("mutation", ["too-many", "long", "no-uncertainty", "formal-object", "bad-target"])
def test_advice_is_bounded_not_a_formal_definition(mutation):
    value = feedback()
    if mutation == "too-many":
        value["suggestions"] *= 7
    elif mutation == "long":
        value["suggestions"][0]["change"] = "x" * 2001
    elif mutation == "no-uncertainty":
        value["suggestions"][0]["uncertainty"] = ""
    elif mutation == "formal-object":
        value["suggestions"][0]["task"] = task()
    else:
        value["suggestions"][0]["target"] = "judge"
    with pytest.raises(ValueError):
        validate_feedback(value, "local", lambda _: None)


def test_driver_issues_stable_distinct_advice_ids():
    actual = advice()
    assert actual == advice()
    assert [s["suggestion_id"] for s in actual] == ["local-01-01", "reflector-01"]
    assert actual[0]["sha256"] == actual[1]["sha256"]  # identical advice, distinct origins


def test_analyzer_can_consolidate_multiple_suggestions_into_task_and_skill():
    skills, tasks = validate_review(review(), advice(), SPEC)
    assert skills[0]["id"] == skill()["id"] and tasks == [task()]
    assert REVIEW_SCHEMA["additionalProperties"] is False


@pytest.mark.parametrize("action", ["reject", "defer"])
def test_rejected_or_deferred_advice_does_not_publish(action):
    value = review()
    value["skills"], value["tasks"] = [], []
    for decision in value["decisions"]:
        decision.update(decision=action, skill_ids=[], task_ids=[])
    assert validate_review(value, advice(), SPEC) == ([], [])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "unlinked", "empty-adopt",
                                      "nonexistent", "reject-publishes", "duplicate-update", "bad-scope"])
def test_review_requires_complete_traceable_safe_publication(mutation):
    value = review()
    if mutation == "missing":
        value["decisions"].pop()
    elif mutation == "duplicate":
        value["decisions"].append(copy.deepcopy(value["decisions"][0]))
    elif mutation == "unknown":
        value["decisions"][0]["suggestion_id"] = "invented"
    elif mutation == "unlinked":
        for d in value["decisions"]:
            d["task_ids"] = []
    elif mutation == "empty-adopt":
        value["decisions"][0].update(skill_ids=[], task_ids=[])
    elif mutation == "nonexistent":
        value["decisions"][0]["task_ids"] = ["invented"]
    elif mutation == "reject-publishes":
        value["decisions"][0]["decision"] = "reject"
    elif mutation == "duplicate-update":
        value["tasks"].append(copy.deepcopy(value["tasks"][0]))
    else:
        value["tasks"][0]["source_scope"] = ["OpenROAD/STA.cpp"]
    with pytest.raises(ValueError):
        validate_review(value, advice(), SPEC)


def test_empty_feedback_and_review_are_valid():
    value = dict(assessment="No reusable lesson", observations=[], suggestions=[])
    assert validate_feedback(value, "reflector", lambda _: None) == value
    assert validate_review(dict(assessment="No updates", decisions=[], tasks=[], skills=[]), [], SPEC) == ([], [])
