"""Non-authoritative advice and Analyzer-only publication receipts.

Raw feedback lives with round artifacts, never in the reusable knowledge store.
Validation also runs on cached responses, not just model-enforced JSON schemas.
"""
from __future__ import annotations

from .knowledge import canonical, digest
from .sta_protocol import LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA, decode_updates

KNOWLEDGE_AUTHORITY = "global-analyzer-only-v1"


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("invalid role response fields; formal knowledge is Analyzer-only")


def _text(value, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("feedback text must be nonempty and bounded")


def validate_feedback(response, role, citation):
    schema = {"local": LOCAL_SCHEMA, "reflector": REFLECTION_SCHEMA}[role]
    _fields(response, schema["properties"])
    _text(response["hypothesis" if role == "local" else "assessment"], 8000)
    if role == "local" and (not isinstance(response["program"], str) or not response["program"].strip()):
        raise ValueError("local candidate program missing")
    for name in ("observations", "suggestions"):
        values = response[name]
        if not isinstance(values, list) or len(values) > 6:
            raise ValueError("at most six feedback items per kind")
        fields = schema["properties"][name]["items"]["properties"]
        for value in values:
            _fields(value, fields)
            for key in fields:
                if key not in {"evidence_paths", "target"}:
                    _text(value[key])
            if name == "suggestions" and value["target"] not in {"task", "skill"}:
                raise ValueError("invalid suggestion target")
            paths = value["evidence_paths"]
            if not isinstance(paths, list) or len(paths) > 8:
                raise ValueError("invalid feedback evidence paths")
            for path in paths:
                _text(path)
                citation(path)
    feedback = {k: response[k] for k in response if k != "program"}
    if len(canonical(feedback)) > 24000:
        raise ValueError("feedback exceeds 24000 characters")
    return feedback


def index_suggestions(local_feedback, reflection):
    """Driver-issued stable IDs; agents cannot impersonate another advice source."""
    indexed = []
    sources = [(f"local-{item['attempt']:02d}", item["feedback"]) for item in local_feedback]
    sources.append(("reflector", reflection))
    for source, feedback in sources:
        for i, suggestion in enumerate(feedback["suggestions"], 1):
            indexed.append({"suggestion_id": f"{source}-{i:02d}", "source": source,
                            "sha256": digest(suggestion), **suggestion})
    return indexed


def validate_review(response, suggestions, spec):
    _fields(response, REVIEW_SCHEMA["properties"])
    _text(response["assessment"], 8000)
    for kind in ("skills", "tasks"):
        if not isinstance(response[kind], list) or len(response[kind]) > 3:
            raise ValueError("at most three formal updates per kind")
    skills, tasks = decode_updates(response, spec)
    published = {"skill_ids": {s["id"] for s in skills}, "task_ids": {t["id"] for t in tasks}}
    if len(published["skill_ids"]) != len(skills) or len(published["task_ids"]) != len(tasks):
        raise ValueError("duplicate formal update id")
    expected = {s["suggestion_id"] for s in suggestions}
    decisions = response["decisions"]
    if not isinstance(decisions, list) or len(decisions) > 24:
        raise ValueError("invalid review decisions")
    seen, linked = set(), {"skill_ids": set(), "task_ids": set()}
    fields = REVIEW_SCHEMA["properties"]["decisions"]["items"]["properties"]
    for decision in decisions:
        _fields(decision, fields)
        name = decision["suggestion_id"]
        if not isinstance(name, str) or name not in expected or name in seen:
            raise ValueError("unknown or duplicate suggestion review")
        seen.add(name)
        _text(decision["reason"])
        action = decision["decision"]
        if action not in {"adopt", "reject", "defer"}:
            raise ValueError("invalid review decision")
        for kind in linked:
            ids = decision[kind]
            if (not isinstance(ids, list) or len(ids) > 3 or
                    any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids))):
                raise ValueError("invalid publication links")
            if not set(ids) <= published[kind]:
                raise ValueError("review references unpublished knowledge")
            if action != "adopt" and ids:
                raise ValueError("rejected/deferred advice cannot publish knowledge")
            linked[kind].update(ids)
        if action == "adopt" and not decision["skill_ids"] and not decision["task_ids"]:
            raise ValueError("adoption must link a formal update")
    if seen != expected:
        raise ValueError("every suggestion requires an explicit review")
    if linked != published:
        raise ValueError("formal updates require adopted advice provenance")
    return skills, tasks
