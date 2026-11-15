"""Compact local handoffs backed by an explicit, phase-filtered method library.

The analyzer integrates relevant methods into the eight-field task. This module
validates its citations and optional attachments, not the scientific quality of
that synthesis. Knowledge is data, never additional evaluator authority.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .knowledge import canonical, digest, safe_path, write_json
from .sta_protocol import validate_task


CONTEXT_MODE = "enhanced-task-v1"
MAX_SOURCES = 4
MAX_ATTACHMENTS = 8
MAX_TASK_CHARS = 16000
MAX_HANDOFF_CHARS = 24000


def _resource(path, content):
    return {"path": safe_path(path), "sha256": digest(content)}


def knowledge_catalog(store, workspace: Path):
    """Index only seed and KnowledgeStore.records(), never test-proposals.

    Bodies stay in separately stored, inspectable files. References contain a
    full content digest so different revisions cannot alias each other.
    """
    catalog = []
    seeds = json.loads((store.root / "seed/snapshot.json").read_text())
    records = store.records()  # verifies seed/import hashes and phase policy
    groups = {}
    for name, content in sorted(seeds.items()):
        parts = safe_path(name).split("/")
        if len(parts) >= 3 and parts[0] in {"skills", "evaluation_profiles"}:
            groups.setdefault((parts[0], parts[1]), {})["/".join(parts[2:])] = content
    for (area, name), files in sorted(groups.items()):
        main = "SKILL.md" if area == "skills" else "profile.yaml"
        if main not in files:
            continue
        text = files[main]
        metadata = {}
        if main.endswith("yaml"):
            metadata = yaml.safe_load(text) or {}
        elif text.startswith("---\n") and "\n---\n" in text[4:]:
            metadata = yaml.safe_load(text.split("---", 2)[1]) or {}
        description = metadata.get("description", name) if isinstance(metadata, dict) else name
        catalog.append({
            "ref": f"seed:{area}/{name}@{digest(files)}", "id": name,
            "kind": "skill" if area == "skills" else "profile", "origin": "seed",
            "description": str(description)[:800], "assessment": "pre-given, check active regime",
            "resources": [_resource(f"knowledge/seed/{area}/{name}/{p}", c)
                          for p, c in sorted(files.items())],
        })
    for record in records:
        revision = digest(record)
        provenance = record["provenance"]
        assessment = {k: (provenance[k][:800] if isinstance(provenance[k], str) else provenance[k]) for k in
                      ("validation", "assessment", "outer_score", "accepted") if k in provenance}
        for kind, items in (("skill", record["skills"]), ("task", record["tasks"])):
            for item in items:
                name = item["id"]
                files = item["files"] if kind == "skill" else {"task.json": canonical(item)}
                resources = []
                for path, content in sorted(files.items()):
                    # Separate kind directories avoid a task/skill ID collision.
                    relative = f"knowledge/learned/{revision[:16]}/{kind}s/{name}/{safe_path(path)}"
                    destination = workspace / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.is_symlink():
                        raise ValueError("knowledge resource is a symlink")
                    destination.write_text(content, encoding="utf-8")
                    resources.append(_resource(relative, content))
                catalog.append({
                    "ref": f"learned:{kind}/{name}@{revision}", "id": name,
                    "kind": kind, "origin": "permitted-knowledge",
                    "description": item["description"][:800],
                    "assessment": assessment, "resources": resources,
                })
    write_json(workspace / "knowledge/catalog.json", catalog)
    return catalog


def compile_enhanced_task(plan, catalog, spec, workspace: Path):
    """Freeze the local problem and resolve a small number of optional resources."""
    task = validate_task(plan["local_task"], spec)
    attempts = plan["max_inner_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError("invalid inner budget")
    if task["evaluation"] == "final" and attempts != 1:
        raise ValueError("final-only task has one candidate")
    if len(canonical(task)) > MAX_TASK_CHARS:
        raise ValueError("enhanced task exceeds compact task budget")
    choices = plan["knowledge_refs"]
    if not isinstance(choices, list) or len(choices) > MAX_SOURCES:
        raise ValueError("enhanced task accepts at most four knowledge references")
    by_ref = {row["ref"]: row for row in catalog}
    selected, attachments, seen = [], [], set()
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) != {"ref", "reason", "attachments"}:
            raise ValueError("knowledge reference requires ref, reason, attachments")
        ref = choice["ref"]
        if not isinstance(ref, str) or ref not in by_ref or ref in seen:
            raise ValueError("unknown, excluded, or duplicate knowledge reference")
        seen.add(ref)
        reason = choice["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("knowledge selection needs a concise relevance reason")
        row = by_ref[ref]
        resources = {r["path"]: r for r in row["resources"]}
        paths = choice["attachments"]
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("attachments must be a list of catalog paths")
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate attachment")
        for path in paths:
            safe_path(path)
            if path not in resources:
                raise ValueError("attachment is not in selected knowledge reference")
            actual = workspace / path
            if (any(p.is_symlink() for p in (actual, *actual.parents))
                    or not actual.resolve().is_relative_to(workspace.resolve())
                    or not actual.is_file()):
                raise ValueError("unsafe or missing knowledge attachment")
            if digest(actual.read_text(encoding="utf-8")) != resources[path]["sha256"]:
                raise ValueError("knowledge attachment checksum mismatch")
            attachments.append({**resources[path], "source_ref": ref})
        selected.append({k: row[k] for k in ("ref", "id", "kind", "origin", "assessment")}
                        | {"reason": reason})
    if len(attachments) > MAX_ATTACHMENTS:
        raise ValueError("too many enhanced task attachments")
    payload = {
        "context_mode": CONTEXT_MODE,
        "task": task,
        "guidance": {"hypothesis": plan["hypothesis"], "evidence": plan["evidence"],
                     "evidence_paths": plan["evidence_paths"],
                     "specialist_instruction": plan["specialist_instruction"]},
        "knowledge_sources": selected, "optional_attachments": attachments,
        "max_inner_attempts": plan["max_inner_attempts"],
    }
    if len(canonical(payload)) > MAX_HANDOFF_CHARS:
        raise ValueError("enhanced task exceeds compact handoff budget")
    return {**payload, "sha256": digest(payload)}


def compact_history(history):
    """Keep measured feedback, not previous task/skill bodies or source patches."""
    result = []
    for row in history:
        item = {k: row[k] for k in ("round", "target", "inner_attempts", "inner_trials",
                                   "outer_score", "accepted_as_incumbent", "outer_result",
                                   "final_evaluated", "research_id") if k in row}
        plan = row.get("global_plan", {})
        # Preserve the tested hypothesis, including rejected ones, without
        # forwarding the complete task/knowledge response. Idempotent when the
        # analyzer's already compact history is used for the local handoff.
        for key, value in (("hypothesis", plan.get("hypothesis")),
                           ("task_id", plan.get("local_task", {}).get("id")),
                           ("evaluation", plan.get("local_task", {}).get("evaluation"))):
            if value is not None or key in row:
                item[key] = row.get(key, value)
        result.append(item)
    return result


def read_enhanced_task(path: Path):
    packet = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in packet.items() if k != "sha256"}
    if packet.get("context_mode") != CONTEXT_MODE or packet.get("sha256") != digest(payload):
        raise ValueError("enhanced task checksum/context mismatch")
    return packet


def local_context(state, enhanced_task, local_incumbent, attempt, trials, limits):
    """Whitelist local context; never forward the analyzer's complete state."""
    return {
        "enhanced_task": enhanced_task,
        "state": {k: state[k] for k in ("case", "phase", "budget", "reference_metrics",
                                       "incumbent_metrics", "incumbent_score", "evidence_index")},
        "recent_outcomes": compact_history(state["history"])[-5:],
        "local_incumbent": local_incumbent, "attempt": attempt,
        "prior_attempts": trials, "effective_patch_limits": limits,
    }
